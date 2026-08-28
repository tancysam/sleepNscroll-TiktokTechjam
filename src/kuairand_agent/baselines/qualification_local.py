"""Concrete local backend for the atomic official-baseline qualification.

This module is the trusted adapter between the high-level qualification transaction and the
hash-pinned organizer/data/model primitives.  It deliberately has no campaign or candidate-code
surface: public-validation labels enter only the protected scorer, while final inference receives
only canonical inputs and alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.artifacts import (
    PredictionVector,
    StarterFMCheckpoint,
    file_sha256,
    load_checkpoint,
    load_predictions,
    save_checkpoint,
    save_predictions,
)
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.organizer import OrganizerModules, load_verified_organizer
from kuairand_agent.baselines.qualification import (
    FinalPredictionEvidence,
    FMReplayEvidence,
    FMTrainingEvidence,
    QualificationError,
    QualificationMetrics,
    QualificationRequest,
    QualificationSnapshot,
    ResourceUsage,
    RungEvaluationEvidence,
    RungSummaryEvidence,
)
from kuairand_agent.baselines.rungs import (
    RungEvaluation,
    RungSummary,
    ValidationScoringContext,
    build_validation_scoring_context,
    evaluate_popularity_validation,
    evaluate_random_rungs,
    require_reference_parity,
)
from kuairand_agent.baselines.starter_fm import StarterFMAdapter, StarterFMConfig
from kuairand_agent.contract import STARTER_FILE_SHA256, SplitName, verify_starter_kit
from kuairand_agent.data.audit import DataAuditReport, audit_dataset
from kuairand_agent.data.canonical import (
    OUTCOME_FIELDS,
    CanonicalDataset,
    CanonicalInputs,
    ProtectedTargets,
    TrainingTargets,
    load_canonical_dataset,
)
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity
from kuairand_agent.scoring.submission import AlignmentRow

_GOLDEN_USERS: Final = ("10", "20", "30", "10", "40", "20", "30", "10", "10")
_GOLDEN_VIDEOS: Final = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
_GOLDEN_LABELS: Final = (1, 0, 1, 0, 1, 0, 1, 1, 0)
_GOLDEN_SCORES: Final = (0.9, 0.7, 0.4, 0.8, 0.5, 0.6, 0.3, 0.1, 0.2)
_GOLDEN_METRICS: Final = {
    "GAUC": 0.5,
    "nDCG@5": 0.7193038288345124,
    "primary": 0.6096519144172562,
}
_FINAL_ZERO_COUNTERS: Final = (
    "outcome_cells_materialized",
    "outcome_cells_decoded",
    "outcome_cells_converted",
    "outcome_cells_validated",
    "outcome_cells_aggregated",
    "outcome_cells_logged",
    "outcome_cells_scored",
)
_CLEAN_WORKER_FILENAME: Final = "clean-worker-evidence.json"
_CLEAN_WORKER_TIMEOUT_SECONDS: Final = 30 * 60

type Float64Vector = npt.NDArray[np.float64]


class _MetricTriplet(Protocol):
    @property
    def gauc(self) -> float: ...

    @property
    def ndcg_at_5(self) -> float: ...

    @property
    def primary(self) -> float: ...


def _json_digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value).rstrip(b"\n")).hexdigest()


def _json_bytes(value: object) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return encoded + b"\n"


def _metrics(
    value: _MetricTriplet,
    *,
    encoded_float32_labels: bool = False,
) -> QualificationMetrics:
    return QualificationMetrics(
        gauc=float(value.gauc),
        ndcg_at_5=float(value.ndcg_at_5),
        primary=float(value.primary),
        label_protocol=(
            "encoded_labels_float32" if encoded_float32_labels else "integer_labels_float64"
        ),
    )


def _mapping_metrics(
    value: Mapping[str, object],
    location: str,
    *,
    encoded_float32_labels: bool = False,
) -> QualificationMetrics:
    converted: dict[str, float] = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        raw = value.get(name)
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            raise QualificationError(f"{location} metric {name} is not numeric")
        converted[name] = float(raw)
    return QualificationMetrics(
        converted["GAUC"],
        converted["nDCG@5"],
        converted["primary"],
        label_protocol=(
            "encoded_labels_float32" if encoded_float32_labels else "integer_labels_float64"
        ),
    )


def _peak_rss_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


def _usage(
    *,
    wall_started: float,
    cpu_started: float,
    peak_rss_bytes: int | None = None,
) -> ResourceUsage:
    return ResourceUsage(
        wall_seconds=float(time.perf_counter() - wall_started),
        cpu_seconds=float(time.process_time() - cpu_started),
        peak_rss_bytes=max(_peak_rss_bytes(), peak_rss_bytes or 0),
        device="cpu",
    )


@dataclass(frozen=True, slots=True)
class _PrimaryTargets:
    """Narrow primary-only view accepted by the trusted starter adapter."""

    primary: npt.NDArray[np.int8]
    digest: str
    training_inputs_digest: str

    @classmethod
    def from_canonical(
        cls,
        targets: TrainingTargets,
        *,
        training_inputs_digest: str,
    ) -> _PrimaryTargets:
        labels = np.ascontiguousarray(targets.long_view, dtype=np.int8)
        labels.setflags(write=False)
        return cls(
            primary=labels,
            digest=targets.digest,
            training_inputs_digest=training_inputs_digest,
        )

    @property
    def row_count(self) -> int:
        return int(self.primary.size)


@dataclass(frozen=True, slots=True)
class _LocalPayload:
    dataset: CanonicalDataset
    audit: DataAuditReport
    encoding: StarterEncoding
    scoring: ValidationScoringContext
    fm_scorer: _EncodedLabelScorer
    starter_root: Path
    starter_manifest_digest: str
    trusted_fixture_manifest: Mapping[str, object]


def _payload(snapshot: QualificationSnapshot) -> _LocalPayload:
    payload = snapshot.payload
    if not isinstance(payload, _LocalPayload):
        raise QualificationError("local qualification backend received a foreign snapshot")
    if payload.dataset.digest != snapshot.canonical_digest:
        raise QualificationError("local snapshot payload differs from its canonical identity")
    if payload.audit.digest != snapshot.audit_digest:
        raise QualificationError("local snapshot payload differs from its audit identity")
    return payload


def _alignment_rows(dataset: CanonicalDataset, split: SplitName) -> tuple[AlignmentRow, ...]:
    alignment = dataset.split(split).alignment
    return tuple(
        AlignmentRow(row_id, user_id, video_id)
        for row_id, user_id, video_id in zip(
            alignment.row_id,
            alignment.user_id,
            alignment.video_id,
            strict=True,
        )
    )


def _validate_data_identity(report: DataAuditReport, dataset: CanonicalDataset) -> None:
    audit_splits = {str(item["split"]): item for item in report.splits}
    for split_name in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST):
        canonical = dataset.split(split_name)
        try:
            audited = audit_splits[split_name.value]
        except KeyError as exc:
            raise QualificationError(f"audit omitted the {split_name.value} split") from exc
        if audited.get("row_count") != canonical.row_count:
            raise QualificationError(f"audit/canonical {split_name.value} row counts differ")

    trace = report.final_outcome_trace.manifest()
    if trace.get("row_count") != dataset.final.row_count:
        raise QualificationError("audit final skip trace differs from canonical final rows")
    if any(trace.get(name) != 0 for name in _FINAL_ZERO_COUNTERS):
        raise QualificationError("audit reports interpretation of final-period outcomes")
    audit_skipped = trace.get("skipped_fields")
    if (
        not isinstance(audit_skipped, list)
        or len(audit_skipped) != len(OUTCOME_FIELDS)
        or set(audit_skipped) != set(OUTCOME_FIELDS)
    ):
        raise QualificationError("audit did not byte-skip every final-period outcome field")
    if trace.get("skipped_values_recorded") is not False:
        raise QualificationError("audit recorded skipped final-period values")

    canonical_trace = dataset.final.outcome_trace
    if dataset.final.targets is not None:
        raise QualificationError("canonical final split unexpectedly exposes targets")
    if canonical_trace.parsed_fields:
        raise QualificationError("canonical final loader parsed an outcome field")
    if canonical_trace.skipped_fields != OUTCOME_FIELDS:
        raise QualificationError("canonical final loader did not skip every outcome field")
    if canonical_trace.skipped_cell_count != dataset.final.row_count * len(OUTCOME_FIELDS):
        raise QualificationError("canonical final skip-cell accounting is inconsistent")


def _fixture_inputs(prefix: str, rows: int, *, start_time: int = 0) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u{index % 5}" for index in range(rows)),
        video_id=tuple(f"{prefix}-v{index % 7}" for index in range(rows)),
        date=tuple(20220408 for _ in range(rows)),
        duration_ms=tuple(float(800 + index * 101) for index in range(rows)),
        tab=tuple(str(index % 3) for index in range(rows)),
        author_id=tuple(f"a{index % 4}" for index in range(rows)),
        time_ms=tuple(start_time + index for index in range(rows)),
    )


@dataclass(frozen=True, slots=True)
class _EncodedLabelScorer:
    """Bind float32 organizer-label semantics to one exact validation input identity."""

    scorer: ProtectedScorer
    alignment: Alignment
    split: SplitIdentity
    labels: Sequence[int]
    validation_inputs_digest: str

    def __call__(self, scores: Float64Vector, /) -> object:
        return self.scorer.score_with_encoded_labels(
            alignment=self.alignment,
            split=self.split,
            labels=self.labels,
            scores=scores,
            expected_count=self.split.expected_count,
        )


def _raw_rows(
    inputs: CanonicalInputs,
    labels: Sequence[int],
) -> list[list[object]]:
    return [
        [
            inputs.date[index],
            inputs.user_id[index],
            inputs.video_id[index],
            inputs.author_id[index],
            inputs.tab[index],
            inputs.duration_ms[index],
            labels[index],
        ]
        for index in range(len(inputs))
    ]


def _evaluator_golden(starter_dir: Path) -> dict[str, object]:
    split = SplitIdentity(
        name="outer_valid",
        token="qualification-evaluator-golden-v1",
        expected_count=len(_GOLDEN_LABELS),
    )
    alignment = Alignment.from_ids(
        split=split,
        user_ids=_GOLDEN_USERS,
        video_ids=_GOLDEN_VIDEOS,
    )
    scorer = ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment)
    observed = scorer.score(
        alignment=alignment,
        split=split,
        labels=_GOLDEN_LABELS,
        scores=_GOLDEN_SCORES,
        expected_count=len(_GOLDEN_LABELS),
    )
    observed_metrics = _metrics(observed)
    for name, expected in _GOLDEN_METRICS.items():
        if not math.isclose(
            observed_metrics.manifest()[name], expected, rel_tol=0.0, abs_tol=1e-15
        ):
            raise QualificationError(f"pinned evaluator missed hard-coded golden {name}")
    if observed.users != 4 or observed.rows != len(_GOLDEN_LABELS):
        raise QualificationError("pinned evaluator golden support counts changed")
    return {
        "fixture": "mixed-and-degenerate-interleaved-users-v1",
        "expected_metrics": dict(_GOLDEN_METRICS),
        "observed_metrics": observed_metrics.manifest(),
        "expected_users": 4,
        "observed_users": observed.users,
        "rows": observed.rows,
        "scorer_digest": observed.scorer_digest,
        "passed": True,
    }


def _starter_fm_fixture_parity(
    starter_dir: Path,
    organizer: OrganizerModules,
) -> dict[str, object]:
    train = _fixture_inputs("train", 12)
    valid = _fixture_inputs("valid", 6, start_time=100)
    train_labels = tuple(int(index % 3 == 0) for index in range(len(train)))
    valid_labels = (1, 0, 1, 0, 0, 1)
    targets = _PrimaryTargets(
        primary=np.asarray(train_labels, dtype=np.int8),
        digest=hashlib.sha256(b"qualification-starter-fixture-targets-v1").hexdigest(),
        training_inputs_digest=train.digest,
    )
    targets.primary.setflags(write=False)
    encoding = StarterEncoding.fit(train)
    split = SplitIdentity(
        name="inner_valid",
        token="qualification-starter-fm-fixture-v1",
        expected_count=len(valid),
    )
    alignment = Alignment.from_ids(split=split, user_ids=valid.user_id, video_ids=valid.video_id)
    scorer = ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment)

    score = _EncodedLabelScorer(
        scorer=scorer,
        alignment=alignment,
        split=split,
        labels=valid_labels,
        validation_inputs_digest=valid.digest,
    )

    seed = 0
    adapted = StarterFMAdapter(
        starter_dir=starter_dir,
        config=StarterFMConfig(seed=seed),
    ).fit(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=valid,
        validation_scorer=score,
    )
    organizer_fixture = {
        "train": _raw_rows(train, train_labels),
        "valid": _raw_rows(valid, valid_labels),
        "test": [[20220429, "placeholder-user", "placeholder-video", "UNK", "0", 1000.0, 0]],
    }
    fixture_json = json.dumps(organizer_fixture, separators=(",", ":"), ensure_ascii=True)
    code = (
        "import hashlib,json\n"
        "from pathlib import Path\n"
        "import baseline\n"
        "root=Path.cwd().resolve()\n"
        "source=Path(baseline.__file__).resolve()\n"
        f"splits=json.loads({fixture_json!r})\n"
        "result=baseline.run_fm(splits,seed=0,verbose=False)['valid']\n"
        "payload={'baseline_path':str(source),'baseline_sha256':"
        "hashlib.sha256(source.read_bytes()).hexdigest(),'valid':{name:float(result[name]) "
        "for name in ('GAUC','nDCG@5','primary')}}\n"
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))\n"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(starter_dir)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=starter_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise QualificationError(
            "untouched organizer fixture subprocess failed: " + completed.stderr[-2000:]
        )
    try:
        raw_result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError("untouched organizer fixture emitted invalid JSON") from exc
    expected_source = (starter_dir / "baseline.py").resolve()
    if (
        not isinstance(raw_result, dict)
        or raw_result.get("baseline_path") != str(expected_source)
        or raw_result.get("baseline_sha256") != STARTER_FILE_SHA256["baseline.py"]
        or not isinstance(raw_result.get("valid"), dict)
    ):
        raise QualificationError("untouched organizer fixture imported an unexpected baseline")
    direct = _mapping_metrics(
        cast(Mapping[str, object], raw_result["valid"]),
        "organizer",
        encoded_float32_labels=True,
    )
    adapted_metrics = _metrics(adapted.validation_metrics, encoded_float32_labels=True)
    if direct != adapted_metrics:
        raise QualificationError("starter FM adapter differs from untouched organizer fixture")
    return {
        "fixture": "nonempty-placeholder-test-v1",
        "seed": seed,
        "organizer_metrics": direct.manifest(),
        "adapter_metrics": adapted_metrics.manifest(),
        "adapter_logical_digest": adapted.logical_digest,
        "adapter_checkpoint_digest": adapted.checkpoint.digest,
        "adapter_prediction_digest": adapted.validation_predictions.digest,
        "organizer_baseline_sha256": STARTER_FILE_SHA256["baseline.py"],
        "organizer_data_sha256": STARTER_FILE_SHA256["data.py"],
        "organizer_evaluator_sha256": STARTER_FILE_SHA256["evaluate.py"],
        "organizer_manifest_sha256": organizer.manifest_sha256,
        "subprocess_baseline_path": str(expected_source),
        "subprocess_cwd": str(starter_dir),
        "passed": True,
    }


def _trusted_fixture_manifest(
    starter_dir: Path,
    organizer: OrganizerModules,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "evaluator_golden": _evaluator_golden(starter_dir),
        "starter_fm_untouched_fixture_parity": _starter_fm_fixture_parity(starter_dir, organizer),
    }
    manifest["digest"] = _json_digest(manifest)
    return manifest


def _rung_evaluation(value: RungEvaluation) -> RungEvaluationEvidence:
    return RungEvaluationEvidence(
        seed=value.seed,
        metrics=_metrics(value.metrics),
        users=value.users,
        rows=value.rows,
        scorer_digest=value.scorer_digest,
        prediction_digest=value.prediction_digest,
        split_digest=value.split_digest,
        runtime_seconds=float(value.runtime_seconds),
    )


def _rung_summary(value: RungSummary) -> RungSummaryEvidence:
    if value.reference_metrics is None:
        raise QualificationError(f"{value.name.value} has no published reference")
    require_reference_parity(value)
    return RungSummaryEvidence(
        name=value.name.value,
        evaluations=tuple(_rung_evaluation(item) for item in value.evaluations),
        reference_metrics=_metrics(value.reference_metrics),
        reference_passed=value.reference_passed,
    )


def _require_starter(payload: _LocalPayload) -> None:
    current = verify_starter_kit(payload.starter_root)
    if current.manifest_sha256 != payload.starter_manifest_digest:
        raise QualificationError("organizer starter changed during qualification")


def _require_empty_artifact_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise QualificationError("FM artifact destination must be a regular directory")
    if any(path.iterdir()):
        raise QualificationError("FM artifact destination must be empty and non-overwriting")


def _artifact_path(training: FMTrainingEvidence, name: str) -> Path:
    matches = tuple(path for path in training.artifact_paths if path.name == name)
    if len(matches) != 1:
        raise QualificationError(f"FM evidence does not contain exactly one {name}")
    return matches[0]


def _restore(
    training: FMTrainingEvidence,
) -> tuple[StarterEncoding, PredictionVector, StarterFMCheckpoint]:
    encoding_path = _artifact_path(training, "encoding.npz")
    prediction_path = _artifact_path(training, "validation-predictions.npy")
    if file_sha256(encoding_path) != training.artifact_sha256[encoding_path.name]:
        raise QualificationError("encoding artifact file SHA-256 mismatch")
    encoding = StarterEncoding.load(encoding_path)
    if encoding.digest != training.encoding_digest:
        raise QualificationError("replayed encoding differs from training evidence")
    checkpoint = load_checkpoint(
        training.checkpoint_path,
        expected_file_sha256=training.artifact_sha256[training.checkpoint_path.name],
        expected_checkpoint_digest=training.checkpoint_digest,
        expected_encoding_digest=encoding.digest,
        expected_starter_manifest_digest=training.starter_manifest_digest,
        expected_config_digest=training.config_digest,
        expected_seed=training.seed,
    )
    predictions = load_predictions(
        prediction_path,
        expected_file_sha256=training.artifact_sha256[prediction_path.name],
        expected_prediction_digest=PredictionVector(training.validation_scores).digest,
    )
    return encoding, predictions, checkpoint


def _write_exclusive_json(path: Path, value: object) -> str:
    payload = _json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise QualificationError(f"clean worker evidence already exists: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def _clean_worker(
    *,
    data_dir: Path,
    starter_dir: Path,
    artifact_dir: Path,
    expected_audit_digest: str,
    expected_canonical_digest: str,
    expected_starter_digest: str,
    expected_encoding_digest: str,
    expected_config_digest: str,
) -> str:
    """Execute the sixth charged seed-0 source retrain in a fresh interpreter."""

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    _require_empty_artifact_dir(artifact_dir)
    starter = verify_starter_kit(starter_dir)
    if starter.manifest_sha256 != expected_starter_digest:
        raise QualificationError("clean worker organizer identity differs from parent snapshot")
    report = audit_dataset(data_dir)
    if report.digest != expected_audit_digest:
        raise QualificationError("clean worker audit identity differs from parent snapshot")
    dataset = load_canonical_dataset(data_dir)
    if dataset.digest != expected_canonical_digest:
        raise QualificationError("clean worker canonical identity differs from parent snapshot")
    _validate_data_identity(report, dataset)
    targets = dataset.train.targets
    valid_targets = dataset.valid.targets
    if not isinstance(targets, TrainingTargets) or not isinstance(valid_targets, ProtectedTargets):
        raise QualificationError("clean worker canonical target capabilities are invalid")
    encoding = StarterEncoding.fit(dataset.train.inputs)
    if encoding.digest != expected_encoding_digest:
        raise QualificationError("clean worker encoding identity differs from parent snapshot")
    context = build_validation_scoring_context(dataset, starter.root)
    scorer = _EncodedLabelScorer(
        scorer=context.scorer,
        alignment=context.alignment,
        split=context.split,
        labels=valid_targets.reveal_for_scorer(),
        validation_inputs_digest=dataset.valid.inputs.digest,
    )
    config = StarterFMConfig(seed=0)
    if config.digest != expected_config_digest:
        raise QualificationError("clean worker config identity differs from parent request")
    run = StarterFMAdapter(starter_dir=starter.root, config=config).fit(
        encoding=encoding,
        train_inputs=dataset.train.inputs,
        train_targets=_PrimaryTargets.from_canonical(
            targets,
            training_inputs_digest=dataset.train.inputs.digest,
        ),
        validation_inputs=dataset.valid.inputs,
        validation_scorer=scorer,
    )
    encoding_artifact = encoding.save(artifact_dir / "encoding.npz")
    checkpoint_artifact = save_checkpoint(artifact_dir / "checkpoint.npz", run.checkpoint)
    prediction_artifact = save_predictions(
        artifact_dir / "validation-predictions.npy", run.validation_predictions
    )
    resource_usage = _usage(
        wall_started=wall_started,
        cpu_started=cpu_started,
        peak_rss_bytes=run.resources.max_observed_rss_bytes,
    )
    base: dict[str, object] = {
        "schema_version": 1,
        "kind": "clean_seed_zero_source_retrain_worker",
        "seed": 0,
        "audit_digest": report.digest,
        "canonical_digest": dataset.digest,
        "starter_manifest_digest": run.starter_manifest_digest,
        "train_inputs_digest": dataset.train.inputs.digest,
        "validation_inputs_digest": dataset.valid.inputs.digest,
        "encoding_digest": encoding.digest,
        "config_digest": run.config_digest,
        "checkpoint_digest": run.checkpoint.digest,
        "validation_prediction_digest": run.validation_predictions.digest,
        "validation_metrics": run.validation_metrics.manifest(),
        "logical_run": run.logical_manifest(),
        "logical_run_digest": run.logical_digest,
        "artifact_sha256": {
            encoding_artifact.path.name: encoding_artifact.file_sha256,
            checkpoint_artifact.path.name: checkpoint_artifact.file_sha256,
            prediction_artifact.path.name: prediction_artifact.file_sha256,
        },
        "resources": resource_usage.manifest(),
        "final_target_capability": None,
        "final_outcomes_accessed": False,
    }
    base["digest"] = _json_digest(base)
    evidence_path = artifact_dir / _CLEAN_WORKER_FILENAME
    _write_exclusive_json(evidence_path, base)
    print(cast(str, base["digest"]), flush=True)
    return cast(str, base["digest"])


def _read_clean_worker_evidence(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError("cannot read clean worker evidence") from exc
    if not isinstance(decoded, dict) or _json_bytes(decoded) != payload:
        raise QualificationError("clean worker evidence is not canonical JSON")
    manifest = cast(dict[str, object], decoded)
    digest = manifest.get("digest")
    without_digest = {key: value for key, value in manifest.items() if key != "digest"}
    if digest != _json_digest(without_digest):
        raise QualificationError("clean worker evidence digest mismatch")
    return manifest


def _clean_seed_zero_subprocess(
    *,
    snapshot: QualificationSnapshot,
    payload: _LocalPayload,
    artifact_dir: Path,
) -> FMTrainingEvidence:
    config = StarterFMConfig(seed=0)
    repo_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(repo_root / "src")
    command = [
        sys.executable,
        "-m",
        "kuairand_agent.baselines.qualification_local",
        "clean-worker",
        "--data-dir",
        str(payload.audit.data_dir),
        "--starter-dir",
        str(payload.starter_root),
        "--artifact-dir",
        str(artifact_dir),
        "--audit-digest",
        snapshot.audit_digest,
        "--canonical-digest",
        snapshot.canonical_digest,
        "--starter-digest",
        snapshot.starter_manifest_digest,
        "--encoding-digest",
        payload.encoding.digest,
        "--config-digest",
        config.digest,
    ]
    wall_started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=_CLEAN_WORKER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise QualificationError("clean seed-0 worker exceeded its timeout") from exc
    if completed.returncode != 0:
        raise QualificationError("clean seed-0 worker failed: " + completed.stderr[-4000:])
    evidence_path = artifact_dir / _CLEAN_WORKER_FILENAME
    evidence = _read_clean_worker_evidence(evidence_path)
    if completed.stdout.strip() != evidence.get("digest"):
        raise QualificationError("clean worker stdout identity differs from its evidence")

    expected_identity = {
        "audit_digest": snapshot.audit_digest,
        "canonical_digest": snapshot.canonical_digest,
        "starter_manifest_digest": snapshot.starter_manifest_digest,
        "encoding_digest": payload.encoding.digest,
        "config_digest": config.digest,
        "seed": 0,
        "final_target_capability": None,
        "final_outcomes_accessed": False,
    }
    if any(evidence.get(name) != value for name, value in expected_identity.items()):
        raise QualificationError("clean worker source/config/data identity differs from parent")
    artifact_hashes_raw = evidence.get("artifact_sha256")
    if not isinstance(artifact_hashes_raw, dict):
        raise QualificationError("clean worker omitted artifact SHA-256 evidence")
    artifact_hashes = cast(dict[str, str], artifact_hashes_raw)
    core_names = {"encoding.npz", "checkpoint.npz", "validation-predictions.npy"}
    if set(artifact_hashes) != core_names:
        raise QualificationError("clean worker artifact set differs from the exact contract")
    for name, expected_hash in artifact_hashes.items():
        if file_sha256(artifact_dir / name) != expected_hash:
            raise QualificationError(f"clean worker artifact SHA-256 mismatch: {name}")

    encoding = StarterEncoding.load(artifact_dir / "encoding.npz")
    if encoding.digest != payload.encoding.digest:
        raise QualificationError("clean worker persisted another encoding identity")
    checkpoint_digest = evidence.get("checkpoint_digest")
    if not isinstance(checkpoint_digest, str):
        raise QualificationError("clean worker omitted checkpoint identity")
    checkpoint = load_checkpoint(
        artifact_dir / "checkpoint.npz",
        expected_file_sha256=artifact_hashes["checkpoint.npz"],
        expected_checkpoint_digest=checkpoint_digest,
        expected_encoding_digest=encoding.digest,
        expected_starter_manifest_digest=snapshot.starter_manifest_digest,
        expected_config_digest=config.digest,
        expected_seed=0,
    )
    prediction_digest = evidence.get("validation_prediction_digest")
    if not isinstance(prediction_digest, str):
        raise QualificationError("clean worker omitted prediction identity")
    predictions = load_predictions(
        artifact_dir / "validation-predictions.npy",
        expected_file_sha256=artifact_hashes["validation-predictions.npy"],
        expected_prediction_digest=prediction_digest,
        expected_row_count=snapshot.validation_count,
    )
    metrics_raw = evidence.get("validation_metrics")
    if not isinstance(metrics_raw, dict):
        raise QualificationError("clean worker omitted validation metrics")
    metrics = _mapping_metrics(
        cast(Mapping[str, object], metrics_raw),
        "clean worker",
        encoded_float32_labels=True,
    )
    logical_run = evidence.get("logical_run")
    if not isinstance(logical_run, dict):
        raise QualificationError("clean worker omitted logical run evidence")
    resource_raw = evidence.get("resources")
    if not isinstance(resource_raw, dict):
        raise QualificationError("clean worker omitted resource evidence")
    try:
        cpu_seconds = float(resource_raw["cpu_seconds"])
        peak_rss_bytes = int(resource_raw["peak_rss_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("clean worker resource evidence is invalid") from exc
    evidence_hash = file_sha256(evidence_path)
    trace = dict(cast(Mapping[str, object], logical_run))
    trace["clean_subprocess"] = {
        "evidence_digest": evidence["digest"],
        "evidence_file_sha256": evidence_hash,
        "fresh_interpreter": True,
        "source_retrain": True,
        "identity_verified_by_parent": True,
    }
    trace["trusted_fixture_proof"] = dict(payload.trusted_fixture_manifest)
    return FMTrainingEvidence(
        seed=0,
        validation_scores=predictions.scores,
        validation_metrics=metrics,
        checkpoint_path=artifact_dir / "checkpoint.npz",
        checkpoint_digest=checkpoint.digest,
        encoding_digest=encoding.digest,
        config_digest=config.digest,
        starter_manifest_digest=snapshot.starter_manifest_digest,
        artifact_paths=(
            artifact_dir / "encoding.npz",
            artifact_dir / "checkpoint.npz",
            artifact_dir / "validation-predictions.npy",
            evidence_path,
        ),
        artifact_sha256={**artifact_hashes, evidence_path.name: evidence_hash},
        training_trace=trace,
        resources=ResourceUsage(
            wall_seconds=float(time.perf_counter() - wall_started),
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=peak_rss_bytes,
            device="cpu",
        ),
        organizer_parity_passed=True,
    )


class LocalQualificationBackend:
    """Run the complete trusted qualification on local CPU without external services."""

    def snapshot(self, request: QualificationRequest) -> QualificationSnapshot:
        organizer = load_verified_organizer(request.starter_dir)
        report = audit_dataset(request.data_dir)
        dataset = load_canonical_dataset(request.data_dir)
        _validate_data_identity(report, dataset)
        if not isinstance(dataset.train.targets, TrainingTargets):
            raise QualificationError("canonical train split lacks training targets")
        if not isinstance(dataset.valid.targets, ProtectedTargets):
            raise QualificationError("canonical validation split lacks protected targets")
        encoding = StarterEncoding.fit(dataset.train.inputs)
        scoring = build_validation_scoring_context(dataset, organizer.root)
        fm_scorer = _EncodedLabelScorer(
            scorer=scoring.scorer,
            alignment=scoring.alignment,
            split=scoring.split,
            labels=dataset.valid.targets.reveal_for_scorer(),
            validation_inputs_digest=dataset.valid.inputs.digest,
        )
        fixtures = _trusted_fixture_manifest(organizer.root, organizer)
        verification_after = verify_starter_kit(organizer.root)
        if verification_after.manifest_sha256 != organizer.manifest_sha256:
            raise QualificationError("organizer starter changed while building qualification input")
        payload = _LocalPayload(
            dataset=dataset,
            audit=report,
            encoding=encoding,
            scoring=scoring,
            fm_scorer=fm_scorer,
            starter_root=organizer.root,
            starter_manifest_digest=organizer.manifest_sha256,
            trusted_fixture_manifest=fixtures,
        )
        return QualificationSnapshot(
            starter_manifest_digest=organizer.manifest_sha256,
            audit_digest=report.digest,
            audit_manifest=report.manifest(),
            canonical_digest=dataset.digest,
            canonical_manifest=dataset.manifest(),
            evaluator_golden_digest=cast(str, fixtures["digest"]),
            evaluator_golden_passed=True,
            validation_alignment=_alignment_rows(dataset, SplitName.VALID),
            validation_labels=dataset.valid.targets.reveal_for_scorer(),
            final_alignment=_alignment_rows(dataset, SplitName.TEST),
            payload=payload,
        )

    def random_rungs(self, snapshot: QualificationSnapshot) -> RungSummaryEvidence:
        payload = _payload(snapshot)
        _require_starter(payload)
        return _rung_summary(evaluate_random_rungs(payload.scoring))

    def popularity_rung(self, snapshot: QualificationSnapshot) -> RungSummaryEvidence:
        payload = _payload(snapshot)
        _require_starter(payload)
        return _rung_summary(evaluate_popularity_validation(payload.dataset, payload.scoring))

    def train_fm(
        self,
        snapshot: QualificationSnapshot,
        seed: int,
        artifact_dir: Path,
    ) -> FMTrainingEvidence:
        payload = _payload(snapshot)
        _require_starter(payload)
        _require_empty_artifact_dir(artifact_dir)
        if seed == 0 and artifact_dir.name == "clean-retrain-seed-0":
            return _clean_seed_zero_subprocess(
                snapshot=snapshot,
                payload=payload,
                artifact_dir=artifact_dir,
            )
        targets = payload.dataset.train.targets
        if not isinstance(targets, TrainingTargets):
            raise QualificationError("local FM training requires canonical training targets")
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        config = StarterFMConfig(seed=seed)
        adapter = StarterFMAdapter(starter_dir=payload.starter_root, config=config)
        run = adapter.fit(
            encoding=payload.encoding,
            train_inputs=payload.dataset.train.inputs,
            train_targets=_PrimaryTargets.from_canonical(
                targets,
                training_inputs_digest=payload.dataset.train.inputs.digest,
            ),
            validation_inputs=payload.dataset.valid.inputs,
            validation_scorer=payload.fm_scorer,
        )
        encoding_artifact = payload.encoding.save(artifact_dir / "encoding.npz")
        checkpoint_artifact = save_checkpoint(artifact_dir / "checkpoint.npz", run.checkpoint)
        prediction_artifact = save_predictions(
            artifact_dir / "validation-predictions.npy", run.validation_predictions
        )
        _require_starter(payload)
        trace = run.logical_manifest()
        trace["logical_digest"] = run.logical_digest
        trace["trusted_fixture_proof"] = dict(payload.trusted_fixture_manifest)
        trace["full_data_reference_gate"] = "five-seed aggregate enforced by coordinator"
        return FMTrainingEvidence(
            seed=seed,
            validation_scores=run.validation_predictions.scores,
            validation_metrics=_metrics(run.validation_metrics, encoded_float32_labels=True),
            checkpoint_path=checkpoint_artifact.path,
            checkpoint_digest=checkpoint_artifact.checkpoint_digest,
            encoding_digest=encoding_artifact.digest,
            config_digest=run.config_digest,
            starter_manifest_digest=run.starter_manifest_digest,
            artifact_paths=(
                encoding_artifact.path,
                checkpoint_artifact.path,
                prediction_artifact.path,
            ),
            artifact_sha256={
                encoding_artifact.path.name: encoding_artifact.file_sha256,
                checkpoint_artifact.path.name: checkpoint_artifact.file_sha256,
                prediction_artifact.path.name: prediction_artifact.file_sha256,
            },
            training_trace=trace,
            resources=_usage(
                wall_started=wall_started,
                cpu_started=cpu_started,
                peak_rss_bytes=run.resources.max_observed_rss_bytes,
            ),
            organizer_parity_passed=True,
        )

    def replay_fm(
        self,
        snapshot: QualificationSnapshot,
        training: FMTrainingEvidence,
    ) -> FMReplayEvidence:
        payload = _payload(snapshot)
        _require_starter(payload)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        encoding, persisted_predictions, checkpoint = _restore(training)
        if persisted_predictions.row_count != snapshot.validation_count:
            raise QualificationError("persisted validation predictions have the wrong row count")
        replay = StarterFMAdapter(
            starter_dir=payload.starter_root,
            config=StarterFMConfig(seed=training.seed),
        ).predict(
            checkpoint=checkpoint,
            encoding=encoding,
            inputs=payload.dataset.valid.inputs,
            expected_prediction_digest=persisted_predictions.digest,
        )
        if replay.scores.tobytes(order="C") != persisted_predictions.scores.tobytes(order="C"):
            raise QualificationError("checkpoint replay differs from persisted predictions")
        metrics = self.score_validation(snapshot, replay.scores)
        return FMReplayEvidence(
            seed=training.seed,
            validation_scores=replay.scores,
            validation_metrics=metrics,
            checkpoint_digest=checkpoint.digest,
            resources=_usage(wall_started=wall_started, cpu_started=cpu_started),
        )

    def score_validation(
        self,
        snapshot: QualificationSnapshot,
        scores: Float64Vector,
    ) -> QualificationMetrics:
        payload = _payload(snapshot)
        return _metrics(
            cast(_MetricTriplet, payload.fm_scorer(scores)),
            encoded_float32_labels=True,
        )

    def predict_final(
        self,
        snapshot: QualificationSnapshot,
        training: FMTrainingEvidence,
    ) -> FinalPredictionEvidence:
        payload = _payload(snapshot)
        _require_starter(payload)
        if payload.dataset.final.targets is not None:
            raise QualificationError("final inference refuses a split with target capability")
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        encoding, _, checkpoint = _restore(training)
        predictions = StarterFMAdapter(
            starter_dir=payload.starter_root,
            config=StarterFMConfig(seed=training.seed),
        ).predict(
            checkpoint=checkpoint,
            encoding=encoding,
            inputs=payload.dataset.final.inputs,
        )
        if predictions.row_count != snapshot.final_count:
            raise QualificationError("final inference changed canonical row count")
        _require_starter(payload)
        return FinalPredictionEvidence(
            scores=predictions.scores,
            checkpoint_digest=checkpoint.digest,
            resources=_usage(wall_started=wall_started, cpu_started=cpu_started),
        )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trusted KuaiRand qualification worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    clean = subparsers.add_parser("clean-worker")
    clean.add_argument("--data-dir", required=True)
    clean.add_argument("--starter-dir", required=True)
    clean.add_argument("--artifact-dir", required=True)
    clean.add_argument("--audit-digest", required=True)
    clean.add_argument("--canonical-digest", required=True)
    clean.add_argument("--starter-digest", required=True)
    clean.add_argument("--encoding-digest", required=True)
    clean.add_argument("--config-digest", required=True)
    parsed = parser.parse_args(argv)
    if parsed.command != "clean-worker":  # pragma: no cover - argparse restricts the command.
        parser.error("unknown qualification worker command")
    _clean_worker(
        data_dir=Path(cast(str, parsed.data_dir)),
        starter_dir=Path(cast(str, parsed.starter_dir)),
        artifact_dir=Path(cast(str, parsed.artifact_dir)),
        expected_audit_digest=cast(str, parsed.audit_digest),
        expected_canonical_digest=cast(str, parsed.canonical_digest),
        expected_starter_digest=cast(str, parsed.starter_digest),
        expected_encoding_digest=cast(str, parsed.encoding_digest),
        expected_config_digest=cast(str, parsed.config_digest),
    )
    return 0


__all__ = ["LocalQualificationBackend"]


if __name__ == "__main__":  # pragma: no cover - covered through the parent integration path.
    raise SystemExit(_main())
