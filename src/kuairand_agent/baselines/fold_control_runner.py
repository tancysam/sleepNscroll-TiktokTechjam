"""Journal-aware fresh-process execution for trusted train-derived FM controls.

The public module is deliberately narrow: callers submit one frozen fold-A or fold-B control
request and receive a typed, replayable result.  All workspace construction, launch ordering,
process supervision, output validation, content-addressed persistence, and terminal rehydration
remain behind :class:`SupervisedFoldFMRunner`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.artifacts import (
    file_sha256,
    load_checkpoint,
    load_predictions,
    save_checkpoint,
    save_predictions,
)
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.fold_controls import (
    FoldFMControlRun,
    PrimaryTrainingTargets,
    build_fold_scoring_context,
    run_fold_fm_control,
)
from kuairand_agent.baselines.starter_fm import (
    AggregateMetrics,
    EpochTrace,
    StarterFMRun,
    TrainingResources,
)
from kuairand_agent.campaign.budgets import LaunchCategory
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateExecutionPendingError,
)
from kuairand_agent.contract import STARTER_FILE_SHA256, verify_starter_kit
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.folds import FOLD_A_SPEC, FOLD_B_SPEC, TemporalFoldSpec
from kuairand_agent.execution.artifacts import (
    ArtifactError,
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
    DirectoryEntryRef,
)
from kuairand_agent.execution.candidate_executor import (
    NO_CHECKPOINT_DIGEST,
    CandidateAction,
    CandidateExecutionArtifacts,
    LocalCandidateLimits,
)
from kuairand_agent.execution.policy import (
    ApprovedInput,
    CandidateInputRole,
    DeclaredOutput,
    OutputDeclaration,
    SplitRole,
)
from kuairand_agent.execution.runner import ExecutionResult, ExecutionSpec, Runner
from kuairand_agent.execution.workspace import (
    CandidateWorkspace,
    WorkspaceMaterializer,
    WorkspaceSpec,
)

type FoldName = Literal["A", "B"]
type LabelInput = Sequence[object] | npt.NDArray[np.generic] | None

_EXECUTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_VERSION: Final = 1
_FOLD_SPECS: Final[Mapping[str, TemporalFoldSpec]] = {
    FOLD_A_SPEC.name: FOLD_A_SPEC,
    FOLD_B_SPEC.name: FOLD_B_SPEC,
}
_CANONICAL_FIELDS: Final = (
    "user_id",
    "video_id",
    "date",
    "duration_ms",
    "tab",
    "author_id",
    "time_ms",
)
_WORKER_FILENAME: Final = "fold_control_worker.py"
_RESULT_FILENAME: Final = "result.json"
_CHECKPOINT_FILENAME: Final = "checkpoint.npz"
_ENCODING_FILENAME: Final = "encoding.npz"
_PREDICTION_FILENAME: Final = "predictions.npy"


class SupervisedFoldFMError(RuntimeError):
    """A supervised fold-FM execution cannot safely proceed or be rehydrated."""


class SupervisedFoldFMExecutionError(SupervisedFoldFMError):
    """The reserved supervised child failed or returned invalid output."""

    def __init__(
        self,
        message: str,
        *,
        result: ExecutionResult | None = None,
        artifacts: CandidateExecutionArtifacts | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.artifacts = artifacts


@dataclass(frozen=True, slots=True)
class FoldFMControlExecutionRequest:
    """One immutable training request; construction never inspects protected label values."""

    execution_id: str
    fold_name: FoldName
    fold_token: str
    seed: int
    prefix_inputs: CanonicalInputs
    prefix_labels: LabelInput = field(repr=False)
    query_inputs: CanonicalInputs
    query_labels: LabelInput = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.execution_id) is not str or _EXECUTION_ID.fullmatch(self.execution_id) is None:
            raise SupervisedFoldFMError(
                "execution_id must use 1-64 ASCII letters, digits, '_' or '-'"
            )
        if self.fold_name not in {"A", "B"}:
            raise SupervisedFoldFMError("fold_name must be 'A' or 'B'")
        if type(self.fold_token) is not str or _SHA256.fullmatch(self.fold_token) is None:
            raise SupervisedFoldFMError("fold_token must be a lowercase SHA-256 digest")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise SupervisedFoldFMError("seed must be a uint32-compatible integer")
        if not isinstance(self.prefix_inputs, CanonicalInputs):
            raise SupervisedFoldFMError("prefix_inputs must be CanonicalInputs")
        if not isinstance(self.query_inputs, CanonicalInputs):
            raise SupervisedFoldFMError("query_inputs must be CanonicalInputs")


@dataclass(frozen=True, slots=True)
class FoldFMControlEvidence:
    """Content-addressed closure needed to verify and rehydrate one exact control."""

    execution_id: str
    source_digest: str
    config_digest: str
    data_digest: str
    checkpoint: ArtifactRef
    encoding: ArtifactRef
    predictions: ArtifactRef
    worker_result: ArtifactRef
    result_manifest: ArtifactRef
    journal_artifacts: CandidateExecutionArtifacts
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if _EXECUTION_ID.fullmatch(self.execution_id) is None:
            raise SupervisedFoldFMError("evidence execution_id is invalid")
        for name in ("source_digest", "config_digest", "data_digest"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise SupervisedFoldFMError(f"evidence {name} is invalid")
        expected_kinds = {
            "checkpoint": ArtifactKind.CHECKPOINT,
            "encoding": ArtifactKind.OTHER,
            "predictions": ArtifactKind.PREDICTION,
            "worker_result": ArtifactKind.MANIFEST,
            "result_manifest": ArtifactKind.MANIFEST,
        }
        for name, kind in expected_kinds.items():
            reference = getattr(self, name)
            if not isinstance(reference, ArtifactRef) or reference.kind is not kind:
                raise SupervisedFoldFMError(f"evidence {name} has the wrong artifact kind")
        if not isinstance(self.journal_artifacts, CandidateExecutionArtifacts):
            raise SupervisedFoldFMError("evidence journal_artifacts are invalid")
        material = {
            "schema_version": _SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "data_digest": self.data_digest,
            "checkpoint": self.checkpoint.manifest(),
            "encoding": self.encoding.manifest(),
            "predictions": self.predictions.manifest(),
            "worker_result": self.worker_result.manifest(),
            "result_manifest": self.result_manifest.manifest(),
            "journal_closure_digest": self.journal_artifacts.closure_digest,
        }
        object.__setattr__(
            self,
            "digest",
            hashlib.sha256(
                b"kuairand-supervised-fold-fm-evidence-v1\0" + _canonical_json(material)
            ).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class SupervisedFoldFMRun:
    """Typed fold control plus exact local-process and immutable-artifact evidence."""

    control: FoldFMControlRun
    evidence: FoldFMControlEvidence
    execution: ExecutionResult | None
    resumed: bool
    worker_pid: int

    def __post_init__(self) -> None:
        if not isinstance(self.control, FoldFMControlRun):
            raise SupervisedFoldFMError("control must be FoldFMControlRun")
        if not isinstance(self.evidence, FoldFMControlEvidence):
            raise SupervisedFoldFMError("evidence must be FoldFMControlEvidence")
        if self.execution is not None and not isinstance(self.execution, ExecutionResult):
            raise SupervisedFoldFMError("execution must be ExecutionResult or absent on resume")
        if type(self.resumed) is not bool:
            raise SupervisedFoldFMError("resumed must be boolean")
        if type(self.worker_pid) is not int or self.worker_pid <= 0:
            raise SupervisedFoldFMError("worker_pid must be positive")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _manifest_digest(namespace: bytes, value: object) -> str:
    return hashlib.sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _artifact_ref(value: object, *, kind: ArtifactKind, name: str) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise SupervisedFoldFMError(f"{name} artifact reference is not an object")
    try:
        reference = ArtifactRef.from_manifest(value)
    except (TypeError, ValueError) as exc:
        raise SupervisedFoldFMError(f"{name} artifact reference is invalid") from exc
    if reference.kind is not kind:
        raise SupervisedFoldFMError(f"{name} artifact reference has the wrong kind")
    return reference


def _strict_mapping(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise SupervisedFoldFMError(f"{name} manifest fields are not exact")
    return cast(Mapping[str, object], value)


def _normalize_labels(values: LabelInput, *, count: int, role: str) -> npt.NDArray[np.int8]:
    if values is None:
        raise SupervisedFoldFMError(f"{role} labels are absent")
    try:
        snapshot = tuple(values)
    except TypeError as exc:
        raise SupervisedFoldFMError(f"{role} labels are not iterable") from exc
    if len(snapshot) != count:
        raise SupervisedFoldFMError(f"{role} labels must contain exactly {count} rows")
    normalized: list[int] = []
    for index, value in enumerate(snapshot):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) not in {0, 1}
        ):
            raise SupervisedFoldFMError(f"{role} labels[{index}] must be integer 0 or 1")
        normalized.append(int(value))
    result = np.asarray(normalized, dtype=np.int8)
    result.setflags(write=False)
    return result


def _query_target_digest(inputs_digest: str, labels: npt.NDArray[np.int8]) -> str:
    digest = hashlib.sha256(b"kuairand-fold-query-targets-v1\0")
    digest.update(inputs_digest.encode("ascii"))
    digest.update(labels.astype("<i1", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _validate_roles_before_labels(request: FoldFMControlExecutionRequest) -> TemporalFoldSpec:
    spec = _FOLD_SPECS[request.fold_name]
    if len(request.prefix_inputs) <= 0:
        raise SupervisedFoldFMError("prefix_inputs cannot be empty")
    if len(request.query_inputs) <= 0:
        raise SupervisedFoldFMError("query_inputs cannot be empty")
    if any(not spec.train_start <= date <= spec.train_end for date in request.prefix_inputs.date):
        raise SupervisedFoldFMError(
            f"fold {request.fold_name} prefix dates are outside its train-derived window"
        )
    if any(not spec.valid_start <= date <= spec.valid_end for date in request.query_inputs.date):
        raise SupervisedFoldFMError(
            f"fold {request.fold_name} query dates are outside its train-derived window"
        )
    return spec


def _put_array(
    artifact_store: ArtifactStore,
    values: Sequence[object] | npt.NDArray[np.generic],
) -> ArtifactRef:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iufU":
        raise SupervisedFoldFMError("fold capability must be a non-empty scalar NPY vector")
    if array.dtype.kind in "iuf" and not bool(np.isfinite(array).all()):
        raise SupervisedFoldFMError("fold capability contains a non-finite value")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="fold-fm-capability-",
        suffix=".npy",
        dir=artifact_store.staging_root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        return artifact_store.put_file(temporary, kind=ArtifactKind.INPUT)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _source_snapshot(artifact_store: ArtifactStore, source: bytes) -> DirectoryArtifactRef:
    source_ref = artifact_store.put_bytes(source, kind=ArtifactKind.SOURCE)
    entry = DirectoryEntryRef(_WORKER_FILENAME, source_ref)
    empty_manifest_ref = ArtifactRef("0" * 64, 0, ArtifactKind.MANIFEST)
    provisional = DirectoryArtifactRef(
        kind=ArtifactKind.SOURCE,
        manifest_artifact=empty_manifest_ref,
        entries=(entry,),
        total_size_bytes=source_ref.size_bytes,
    )
    manifest_ref = artifact_store.put_bytes(
        _canonical_json(provisional.manifest()), kind=ArtifactKind.MANIFEST
    )
    result = DirectoryArtifactRef(
        kind=ArtifactKind.SOURCE,
        manifest_artifact=manifest_ref,
        entries=(entry,),
        total_size_bytes=source_ref.size_bytes,
    )
    artifact_store.verify_directory(result)
    return result


def _worker_source(source_root: Path) -> bytes:
    payload = (
        "from __future__ import annotations\n"
        "import sys\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from kuairand_agent.baselines.fold_control_runner import _worker_entrypoint\n"
        "raise SystemExit(_worker_entrypoint())\n"
    )
    return payload.encode("utf-8")


def _source_digest(worker_source: bytes, starter_manifest_digest: str) -> str:
    module_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "baselines/artifacts.py",
        "baselines/encoding.py",
        "baselines/fold_control_runner.py",
        "baselines/fold_controls.py",
        "baselines/organizer.py",
        "baselines/starter_fm.py",
        "contract.py",
        "scoring/protected.py",
    )
    entries = []
    for relative in relative_paths:
        path = module_root / relative
        entries.append((relative, file_sha256(path)))
    return _manifest_digest(
        b"kuairand-supervised-fold-fm-source-v1",
        {
            "worker_sha256": hashlib.sha256(worker_source).hexdigest(),
            "implementation": entries,
            "starter_manifest_digest": starter_manifest_digest,
        },
    )


def _limits_manifest(limits: LocalCandidateLimits) -> dict[str, object]:
    return {
        "timeout_seconds": float(limits.timeout_seconds),
        "memory_limit_bytes": limits.memory_limit_bytes,
        "workspace_disk_limit_bytes": limits.workspace_disk_limit_bytes,
        "output_limit_bytes": limits.output_limit_bytes,
        "temp_limit_bytes": limits.temp_limit_bytes,
        "threads": limits.threads,
        "stdout_limit_bytes": limits.stdout_limit_bytes,
        "stderr_limit_bytes": limits.stderr_limit_bytes,
        "process_limit": limits.process_limit,
        "device": limits.device,
    }


def _base_artifacts(
    artifact_store: ArtifactStore, result: ExecutionResult
) -> list[tuple[str, ArtifactRef]]:
    return [
        (
            "execution_manifest",
            artifact_store.put_bytes(
                _canonical_json(result.manifest()), kind=ArtifactKind.MANIFEST
            ),
        ),
        ("stderr", artifact_store.put_file(result.stderr.path, kind=ArtifactKind.LOG)),
        ("stdout", artifact_store.put_file(result.stdout.path, kind=ArtifactKind.LOG)),
    ]


def _load_array(path: Path, *, kind: str) -> npt.NDArray[np.generic]:
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise SupervisedFoldFMError(f"worker input {kind} is not a safe NPY vector") from exc
    if not isinstance(raw, np.ndarray) or raw.ndim != 1 or raw.size == 0:
        raise SupervisedFoldFMError(f"worker input {kind} is not a non-empty NPY vector")
    return raw


def _worker_input_paths(request: Mapping[str, object], workspace: Path) -> dict[str, Path]:
    approved = request.get("approved_inputs")
    if not isinstance(approved, list):
        raise SupervisedFoldFMError("worker request approved_inputs is invalid")
    result: dict[str, Path] = {}
    for item in approved:
        manifest = _strict_mapping(
            item,
            {"name", "role", "workspace_path", "artifact"},
            "worker approved input",
        )
        name = manifest["name"]
        relative = manifest["workspace_path"]
        artifact = manifest["artifact"]
        if type(name) is not str or type(relative) is not str or not isinstance(artifact, Mapping):
            raise SupervisedFoldFMError("worker approved input values are invalid")
        path = workspace / relative
        if path.parent != workspace / "inputs" or not path.is_file() or path.is_symlink():
            raise SupervisedFoldFMError("worker approved input path is outside inputs")
        if file_sha256(path) != artifact.get("sha256"):
            raise SupervisedFoldFMError("worker approved input file digest mismatch")
        result[name] = path
    return result


def _load_worker_inputs(handles: Mapping[str, Path], prefix: str) -> CanonicalInputs:
    columns: dict[str, Sequence[object]] = {}
    for field_name in _CANONICAL_FIELDS:
        handle = f"{prefix}_{field_name}"
        try:
            path = handles[handle]
        except KeyError as exc:
            raise SupervisedFoldFMError(f"worker request omitted {handle}") from exc
        array = _load_array(path, kind=handle)
        expected_kind = "U" if field_name in {"user_id", "video_id", "tab", "author_id"} else None
        if expected_kind is not None and array.dtype.kind != expected_kind:
            raise SupervisedFoldFMError(f"worker input {handle} has the wrong string dtype")
        if field_name in {"date", "time_ms"} and array.dtype.kind not in "iu":
            raise SupervisedFoldFMError(f"worker input {handle} has the wrong integer dtype")
        if field_name == "duration_ms" and array.dtype != np.dtype("float64"):
            raise SupervisedFoldFMError(f"worker input {handle} has the wrong float dtype")
        columns[field_name] = cast(Sequence[object], array.tolist())
    return CanonicalInputs(**columns)  # type: ignore[arg-type]


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _worker_entrypoint(argv: Sequence[str] | None = None) -> int:
    """Trusted fresh-process entry point; not part of the caller-facing interface."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    workspace = Path.cwd().resolve(strict=True)
    request_path = (workspace / arguments.request).resolve(strict=True)
    output = (workspace / arguments.output).resolve(strict=True)
    if request_path != workspace / "request.json" or output != workspace / "output":
        raise SupervisedFoldFMError("worker paths differ from the frozen workspace layout")
    raw_bytes = request_path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SupervisedFoldFMError("worker request is not JSON") from exc
    if not isinstance(raw, Mapping) or raw_bytes != _canonical_json(raw):
        raise SupervisedFoldFMError("worker request is not canonical JSON")
    request = raw.get("request")
    if not isinstance(request, Mapping):
        raise SupervisedFoldFMError("worker request payload is absent")
    handles = _worker_input_paths(raw, workspace)
    prefix_inputs = _load_worker_inputs(handles, "prefix")
    query_inputs = _load_worker_inputs(handles, "query")
    prefix_labels_raw = _load_array(handles["prefix_long_view"], kind="prefix_long_view")
    query_labels_raw = _load_array(handles["query_long_view"], kind="query_long_view")
    if prefix_labels_raw.dtype != np.dtype("int8") or query_labels_raw.dtype != np.dtype("int8"):
        raise SupervisedFoldFMError("worker labels must use exact int8 storage")

    starter_members = request.get("starter_members")
    if not isinstance(starter_members, list):
        raise SupervisedFoldFMError("worker starter member inventory is absent")
    starter_dir = workspace / "tmp" / "starter"
    starter_dir.mkdir(mode=0o700)
    observed_names: list[str] = []
    for item in starter_members:
        member = _strict_mapping(item, {"name", "handle", "sha256"}, "starter member")
        name = member["name"]
        handle = member["handle"]
        expected_sha = member["sha256"]
        if type(name) is not str or type(handle) is not str or type(expected_sha) is not str:
            raise SupervisedFoldFMError("worker starter member values are invalid")
        if name not in STARTER_FILE_SHA256 or expected_sha != STARTER_FILE_SHA256[name]:
            raise SupervisedFoldFMError("worker starter member identity is not pinned")
        source = handles[handle]
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise SupervisedFoldFMError("worker starter member content differs from its pin")
        _write_exclusive(starter_dir / name, payload)
        observed_names.append(name)
    if tuple(observed_names) != tuple(sorted(STARTER_FILE_SHA256)):
        raise SupervisedFoldFMError("worker starter inventory is not exact and sorted")
    starter_identity = verify_starter_kit(starter_dir)

    fold_name = request.get("fold_name")
    fold_token = request.get("fold_token")
    seed = request.get("seed")
    if fold_name not in {"A", "B"} or type(fold_token) is not str or type(seed) is not int:
        raise SupervisedFoldFMError("worker fold identity is invalid")
    control = run_fold_fm_control(
        prefix_inputs,
        prefix_labels_raw,
        query_inputs,
        query_labels_raw,
        starter_dir,
        seed=seed,
        fold_name=fold_name,
        fold_token=fold_token,
    )
    replay = control.replay_predictions(starter_dir=starter_dir, query_inputs=query_inputs)
    if (
        replay.digest != control.predictions.digest
        or replay.scores.tobytes() != control.predictions.scores.tobytes()
    ):
        raise SupervisedFoldFMError("worker checkpoint replay differs from fitted predictions")

    checkpoint_artifact = save_checkpoint(output / _CHECKPOINT_FILENAME, control.checkpoint)
    encoding_artifact = control.encoding.save(output / _ENCODING_FILENAME)
    prediction_artifact = save_predictions(output / _PREDICTION_FILENAME, control.predictions)
    worker_result = {
        "schema_version": _SCHEMA_VERSION,
        "execution_id": raw.get("execution_id"),
        "worker_pid": os.getpid(),
        "source_digest": request.get("source_digest"),
        "config_digest": request.get("config_digest"),
        "data_digest": request.get("data_digest"),
        "starter_manifest_digest": starter_identity.manifest_sha256,
        "fold_control": control.manifest(),
        "checkpoint_file_sha256": checkpoint_artifact.file_sha256,
        "encoding_file_sha256": encoding_artifact.file_sha256,
        "prediction_file_sha256": prediction_artifact.file_sha256,
        "replay_prediction_digest": replay.digest,
        "replay_exact": True,
    }
    _write_exclusive(output / _RESULT_FILENAME, _canonical_json(worker_result))
    return 0


def _metrics(value: object) -> AggregateMetrics:
    manifest = _strict_mapping(value, {"GAUC", "nDCG@5", "primary"}, "metrics")
    return AggregateMetrics(
        gauc=cast(float, manifest["GAUC"]),
        ndcg_at_5=cast(float, manifest["nDCG@5"]),
        primary=cast(float, manifest["primary"]),
    )


def _resources(value: object) -> TrainingResources:
    keys = {
        "wall_seconds",
        "rss_before_bytes",
        "rss_after_bytes",
        "max_observed_rss_bytes",
        "train_rows",
        "validation_rows",
        "total_dim",
        "epochs_completed",
        "optimizer_steps",
        "device",
        "precision",
    }
    manifest = _strict_mapping(value, keys, "resources")
    return TrainingResources(**cast(Any, dict(manifest)))


def _restore_control(
    manifest_value: object,
    *,
    encoding: StarterEncoding,
    checkpoint_path: Path,
    checkpoint_file_sha256: str,
    predictions_path: Path,
    prediction_file_sha256: str,
) -> FoldFMControlRun:
    control_manifest = _strict_mapping(
        manifest_value,
        {
            "schema_version",
            "fold_name",
            "fold_token",
            "prefix_inputs_digest",
            "query_inputs_digest",
            "query_alignment_digest",
            "encoding_digest",
            "training",
            "digest",
            "resources",
        },
        "fold control",
    )
    training = _strict_mapping(
        control_manifest["training"],
        {
            "schema_version",
            "train_inputs_digest",
            "training_targets_digest",
            "validation_inputs_digest",
            "encoding_digest",
            "config_digest",
            "starter_manifest_digest",
            "checkpoint",
            "validation_predictions",
            "validation_metrics",
            "trace",
        },
        "starter FM training",
    )
    checkpoint_manifest = _strict_mapping(
        training["checkpoint"],
        {
            "schema_version",
            "model",
            "V_shape",
            "W_shape",
            "dtype",
            "encoding_digest",
            "config_digest",
            "starter_manifest_digest",
            "seed",
            "best_epoch",
            "epochs_completed",
            "optimizer_steps",
            "checkpoint_digest",
        },
        "checkpoint",
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_file_sha256=checkpoint_file_sha256,
        expected_checkpoint_digest=cast(str, checkpoint_manifest["checkpoint_digest"]),
        expected_encoding_digest=encoding.digest,
        expected_starter_manifest_digest=cast(str, training["starter_manifest_digest"]),
        expected_config_digest=cast(str, training["config_digest"]),
        expected_seed=cast(int, checkpoint_manifest["seed"]),
    )
    if checkpoint.manifest() != checkpoint_manifest:
        raise SupervisedFoldFMError("restored checkpoint manifest differs from worker evidence")
    prediction_manifest = _strict_mapping(
        training["validation_predictions"],
        {"schema_version", "row_count", "dtype", "prediction_digest"},
        "predictions",
    )
    predictions = load_predictions(
        predictions_path,
        expected_file_sha256=prediction_file_sha256,
        expected_prediction_digest=cast(str, prediction_manifest["prediction_digest"]),
        expected_row_count=cast(int, prediction_manifest["row_count"]),
    )
    trace_values = training["trace"]
    if not isinstance(trace_values, list):
        raise SupervisedFoldFMError("training trace must be an array")
    trace: list[EpochTrace] = []
    for value in trace_values:
        trace_manifest = _strict_mapping(
            value,
            {
                "epoch",
                "batch_count",
                "optimizer_steps",
                "mean_loss",
                "metrics",
                "prediction_digest",
                "improved",
                "bad_epochs",
            },
            "epoch trace",
        )
        trace.append(
            EpochTrace(
                epoch=cast(int, trace_manifest["epoch"]),
                batch_count=cast(int, trace_manifest["batch_count"]),
                optimizer_steps=cast(int, trace_manifest["optimizer_steps"]),
                mean_loss=cast(float, trace_manifest["mean_loss"]),
                metrics=_metrics(trace_manifest["metrics"]),
                prediction_digest=cast(str, trace_manifest["prediction_digest"]),
                improved=cast(bool, trace_manifest["improved"]),
                bad_epochs=cast(int, trace_manifest["bad_epochs"]),
            )
        )
    starter_run = StarterFMRun(
        checkpoint=checkpoint,
        validation_predictions=predictions,
        validation_metrics=_metrics(training["validation_metrics"]),
        trace=tuple(trace),
        resources=_resources(control_manifest["resources"]),
        train_inputs_digest=cast(str, training["train_inputs_digest"]),
        training_targets_digest=cast(str, training["training_targets_digest"]),
        validation_inputs_digest=cast(str, training["validation_inputs_digest"]),
        encoding_digest=cast(str, training["encoding_digest"]),
        config_digest=cast(str, training["config_digest"]),
        starter_manifest_digest=cast(str, training["starter_manifest_digest"]),
    )
    control = FoldFMControlRun(
        fold_name=cast(str, control_manifest["fold_name"]),
        fold_token=cast(str, control_manifest["fold_token"]),
        prefix_inputs_digest=cast(str, control_manifest["prefix_inputs_digest"]),
        query_inputs_digest=cast(str, control_manifest["query_inputs_digest"]),
        query_alignment_digest=cast(str, control_manifest["query_alignment_digest"]),
        encoding=encoding,
        training=starter_run,
    )
    if control.manifest() != control_manifest:
        raise SupervisedFoldFMError("rehydrated fold control differs from exact worker manifest")
    return control


class SupervisedFoldFMRunner:
    """Deep module for one charged, supervised, content-addressed official-FM control."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        workspace_materializer: WorkspaceMaterializer,
        control_root: Path,
        interpreter: Path,
        starter_dir: Path,
        limits: LocalCandidateLimits,
        runner: Runner | None = None,
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise SupervisedFoldFMError("artifact_store must be ArtifactStore")
        if not isinstance(workspace_materializer, WorkspaceMaterializer):
            raise SupervisedFoldFMError("workspace_materializer must be WorkspaceMaterializer")
        if workspace_materializer.artifact_store is not artifact_store:
            raise SupervisedFoldFMError("workspace and fold runner artifact stores differ")
        if (
            not isinstance(control_root, Path)
            or not control_root.is_dir()
            or control_root.is_symlink()
        ):
            raise SupervisedFoldFMError("control_root must be an existing real directory")
        if not isinstance(interpreter, Path):
            raise SupervisedFoldFMError("interpreter must be pathlib.Path")
        if not isinstance(starter_dir, Path):
            raise SupervisedFoldFMError("starter_dir must be pathlib.Path")
        if not isinstance(limits, LocalCandidateLimits):
            raise SupervisedFoldFMError("limits must be LocalCandidateLimits")
        if limits.device != "cpu":
            raise SupervisedFoldFMError("official fold FM controls require the CPU reference path")
        self.artifact_store = artifact_store
        self.workspace_materializer = workspace_materializer
        self.control_root = control_root.resolve(strict=True)
        self.interpreter = interpreter
        self.starter_dir = starter_dir.resolve(strict=True)
        self.limits = limits
        self.runner = Runner() if runner is None else runner

    def _request_identities(
        self,
        request: FoldFMControlExecutionRequest,
    ) -> tuple[
        npt.NDArray[np.int8],
        npt.NDArray[np.int8],
        str,
        str,
        str,
        bytes,
    ]:
        _validate_roles_before_labels(request)
        starter = verify_starter_kit(self.starter_dir)
        prefix_labels = _normalize_labels(
            request.prefix_labels, count=len(request.prefix_inputs), role="prefix"
        )
        query_labels = _normalize_labels(
            request.query_labels, count=len(request.query_inputs), role="query"
        )
        prefix_targets = PrimaryTrainingTargets.bind(request.prefix_inputs, prefix_labels)
        scoring = build_fold_scoring_context(
            self.starter_dir,
            request.fold_name,
            request.fold_token,
            request.query_inputs,
            query_labels,
        )
        query_targets_digest = _query_target_digest(request.query_inputs.digest, query_labels)
        data_digest = _manifest_digest(
            b"kuairand-supervised-fold-fm-data-v1",
            {
                "fold_name": request.fold_name,
                "fold_token": request.fold_token,
                "prefix_inputs_digest": request.prefix_inputs.digest,
                "prefix_targets_digest": prefix_targets.digest,
                "query_inputs_digest": request.query_inputs.digest,
                "query_targets_digest": query_targets_digest,
                "query_alignment_digest": scoring.query_alignment_digest,
            },
        )
        source_root = Path(__file__).resolve().parents[2]
        worker_source = _worker_source(source_root)
        source_digest = _source_digest(worker_source, starter.manifest_sha256)
        config_digest = _manifest_digest(
            b"kuairand-supervised-fold-fm-config-v1",
            {
                "schema_version": _SCHEMA_VERSION,
                "fold_name": request.fold_name,
                "fold_token": request.fold_token,
                "seed": request.seed,
                "starter_manifest_digest": starter.manifest_sha256,
                "limits": _limits_manifest(self.limits),
            },
        )
        return (
            prefix_labels,
            query_labels,
            source_digest,
            config_digest,
            data_digest,
            worker_source,
        )

    def _approved_inputs(
        self,
        request: FoldFMControlExecutionRequest,
        prefix_labels: npt.NDArray[np.int8],
        query_labels: npt.NDArray[np.int8],
    ) -> tuple[tuple[ApprovedInput, ...], list[dict[str, str]]]:
        approved: list[ApprovedInput] = []
        for prefix, inputs in (("prefix", request.prefix_inputs), ("query", request.query_inputs)):
            for field_name in _CANONICAL_FIELDS:
                approved.append(
                    ApprovedInput(
                        f"{prefix}_{field_name}",
                        CandidateInputRole.TRAIN_INPUTS,
                        _put_array(self.artifact_store, inputs.column(field_name)),
                    )
                )
        approved.extend(
            (
                ApprovedInput(
                    "prefix_long_view",
                    CandidateInputRole.TRAIN_TARGETS,
                    _put_array(self.artifact_store, prefix_labels),
                ),
                ApprovedInput(
                    "query_long_view",
                    CandidateInputRole.TRAIN_TARGETS,
                    _put_array(self.artifact_store, query_labels),
                ),
            )
        )
        starter_members: list[dict[str, str]] = []
        for index, name in enumerate(sorted(STARTER_FILE_SHA256)):
            handle = f"starter_{index}"
            reference = self.artifact_store.put_file(
                self.starter_dir / name, kind=ArtifactKind.INPUT
            )
            if reference.sha256 != STARTER_FILE_SHA256[name]:
                raise SupervisedFoldFMError("starter input differs from its published pin")
            approved.append(ApprovedInput(handle, CandidateInputRole.TRAIN_INPUTS, reference))
            starter_members.append({"name": name, "handle": handle, "sha256": reference.sha256})
        return tuple(approved), starter_members

    def _workspace_and_spec(
        self,
        request: FoldFMControlExecutionRequest,
        *,
        prefix_labels: npt.NDArray[np.int8],
        query_labels: npt.NDArray[np.int8],
        source_digest: str,
        config_digest: str,
        data_digest: str,
        worker_source: bytes,
    ) -> tuple[CandidateWorkspace, ExecutionSpec]:
        approved, starter_members = self._approved_inputs(request, prefix_labels, query_labels)
        source_snapshot = _source_snapshot(self.artifact_store, worker_source)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "source_digest": source_digest,
            "config_digest": config_digest,
            "data_digest": data_digest,
            "fold_name": request.fold_name,
            "fold_token": request.fold_token,
            "seed": request.seed,
            "prefix_inputs_digest": request.prefix_inputs.digest,
            "query_inputs_digest": request.query_inputs.digest,
            "starter_members": starter_members,
        }
        workspace = self.workspace_materializer.materialize(
            WorkspaceSpec(
                execution_id=request.execution_id,
                split_role=SplitRole.INNER_TRAIN,
                source_snapshot=source_snapshot,
                approved_inputs=approved,
                request_payload=payload,
                output_limit_bytes=self.limits.output_limit_bytes,
                temp_limit_bytes=self.limits.temp_limit_bytes,
            )
        )
        nonce = hashlib.sha256(
            b"kuairand-supervised-fold-fm-execution-v1\0"
            + request.execution_id.encode("ascii")
            + source_digest.encode("ascii")
            + config_digest.encode("ascii")
            + data_digest.encode("ascii")
        ).hexdigest()[:32]
        spec = ExecutionSpec(
            execution_id=request.execution_id,
            nonce=nonce,
            interpreter=self.interpreter,
            arguments=(
                f"source/{_WORKER_FILENAME}",
                "--request",
                "request.json",
                "--output",
                "output",
            ),
            workspace=workspace.root,
            control_dir=self.control_root / request.execution_id,
            timeout_seconds=self.limits.timeout_seconds,
            memory_limit_bytes=self.limits.memory_limit_bytes,
            workspace_disk_limit_bytes=self.limits.workspace_disk_limit_bytes,
            stdout_limit_bytes=self.limits.stdout_limit_bytes,
            stderr_limit_bytes=self.limits.stderr_limit_bytes,
            threads=self.limits.threads,
            source_digest=source_digest,
            config_digest=config_digest,
            data_digest=data_digest,
            checkpoint_digest=NO_CHECKPOINT_DIGEST,
            device="cpu",
            process_limit=self.limits.process_limit,
            python_hash_seed=request.seed,
            extra_environment=(
                ("KUAIRAND_MODE", "train"),
                ("KUAIRAND_SEED", str(request.seed)),
                ("KUAIRAND_SPLIT_ROLE", SplitRole.INNER_TRAIN.value),
            ),
        )
        return workspace, spec

    def _cleanup_workspace(
        self,
        workspace: CandidateWorkspace,
    ) -> tuple[ArtifactRef, str | None]:
        error_type: str | None = None
        try:
            self.workspace_materializer.cleanup(workspace)
        except Exception as exc:
            error_type = type(exc).__name__
        receipt = self.artifact_store.put_bytes(
            _canonical_json(
                {
                    "schema_version": 1,
                    "execution_id": workspace.execution_id,
                    "workspace_removed": error_type is None,
                    "error_type": error_type,
                }
            ),
            kind=ArtifactKind.MANIFEST,
        )
        return receipt, error_type

    def _finish_failed(
        self,
        *,
        result: ExecutionResult,
        journal: CampaignStoreCandidateJournal,
        workspace: CandidateWorkspace,
        diagnostic: str,
        cleanup: tuple[ArtifactRef, str | None] | None = None,
    ) -> SupervisedFoldFMExecutionError:
        cleanup_receipt, cleanup_error = (
            self._cleanup_workspace(workspace) if cleanup is None else cleanup
        )
        if cleanup_error is not None:
            diagnostic = f"{diagnostic}; trusted workspace cleanup failed: {cleanup_error}"
        bounded_diagnostic = diagnostic[:4096]
        entries = list(_base_artifacts(self.artifact_store, result))
        entries.extend(
            (
                ("workspace_cleanup", cleanup_receipt),
                (
                    "failure_diagnostic",
                    self.artifact_store.put_bytes(
                        bounded_diagnostic.encode("utf-8"),
                        kind=ArtifactKind.LOG,
                    ),
                ),
            )
        )
        artifacts = CandidateExecutionArtifacts(
            tuple(entries),
            output_validated=False,
            diagnostic=bounded_diagnostic,
        )
        journal.finish(action=CandidateAction.TRAIN, result=result, artifacts=artifacts)
        return SupervisedFoldFMExecutionError(
            bounded_diagnostic,
            result=result,
            artifacts=artifacts,
        )

    def _output_ceilings(self, request: FoldFMControlExecutionRequest) -> dict[str, int]:
        return {
            _RESULT_FILENAME: 2 * 1024**2,
            _CHECKPOINT_FILENAME: max(1024**2, len(request.prefix_inputs) * 256),
            _ENCODING_FILENAME: max(1024**2, len(request.prefix_inputs) * 128),
            _PREDICTION_FILENAME: max(1024**2, len(request.query_inputs) * 16 + 4096),
        }

    def _restore_from_artifacts(
        self,
        request: FoldFMControlExecutionRequest,
        *,
        journal_artifacts: CandidateExecutionArtifacts,
        source_digest: str,
        config_digest: str,
        data_digest: str,
        execution: ExecutionResult | None,
        resumed: bool,
    ) -> SupervisedFoldFMRun:
        try:
            result_ref = journal_artifacts.artifact("candidate_result")
            checkpoint_ref = journal_artifacts.artifact("checkpoint")
        except KeyError as exc:
            raise SupervisedFoldFMError("terminal fold FM evidence is incomplete") from exc
        result_bytes = self.artifact_store.read_bytes(result_ref, max_bytes=result_ref.size_bytes)
        try:
            raw = json.loads(result_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SupervisedFoldFMError("fold FM evidence manifest is not JSON") from exc
        if not isinstance(raw, Mapping) or result_bytes != _canonical_json(raw):
            raise SupervisedFoldFMError("fold FM evidence manifest is not canonical JSON")
        evidence_manifest = _strict_mapping(
            raw,
            {
                "schema_version",
                "execution_id",
                "source_digest",
                "config_digest",
                "data_digest",
                "worker_result",
                "checkpoint",
                "encoding",
                "predictions",
                "fold_control_digest",
            },
            "fold FM evidence",
        )
        exact = {
            "schema_version": (evidence_manifest["schema_version"], _SCHEMA_VERSION),
            "execution_id": (evidence_manifest["execution_id"], request.execution_id),
            "source_digest": (evidence_manifest["source_digest"], source_digest),
            "config_digest": (evidence_manifest["config_digest"], config_digest),
            "data_digest": (evidence_manifest["data_digest"], data_digest),
        }
        mismatches = [name for name, values in exact.items() if values[0] != values[1]]
        if mismatches:
            raise SupervisedFoldFMError(
                "terminal fold FM evidence differs from request: " + ", ".join(mismatches)
            )
        worker_ref = _artifact_ref(
            evidence_manifest["worker_result"], kind=ArtifactKind.MANIFEST, name="worker_result"
        )
        encoded_checkpoint_ref = _artifact_ref(
            evidence_manifest["checkpoint"], kind=ArtifactKind.CHECKPOINT, name="checkpoint"
        )
        if encoded_checkpoint_ref != checkpoint_ref:
            raise SupervisedFoldFMError("journal checkpoint differs from fold FM evidence")
        encoding_ref = _artifact_ref(
            evidence_manifest["encoding"], kind=ArtifactKind.OTHER, name="encoding"
        )
        predictions_ref = _artifact_ref(
            evidence_manifest["predictions"],
            kind=ArtifactKind.PREDICTION,
            name="predictions",
        )
        worker_bytes = self.artifact_store.read_bytes(worker_ref, max_bytes=worker_ref.size_bytes)
        try:
            worker_raw = json.loads(worker_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SupervisedFoldFMError("worker result artifact is not JSON") from exc
        worker = _strict_mapping(
            worker_raw,
            {
                "schema_version",
                "execution_id",
                "worker_pid",
                "source_digest",
                "config_digest",
                "data_digest",
                "starter_manifest_digest",
                "fold_control",
                "checkpoint_file_sha256",
                "encoding_file_sha256",
                "prediction_file_sha256",
                "replay_prediction_digest",
                "replay_exact",
            },
            "worker result",
        )
        if worker_bytes != _canonical_json(worker):
            raise SupervisedFoldFMError("worker result artifact is not canonical JSON")
        worker_exact = {
            "schema_version": (worker["schema_version"], _SCHEMA_VERSION),
            "execution_id": (worker["execution_id"], request.execution_id),
            "source_digest": (worker["source_digest"], source_digest),
            "config_digest": (worker["config_digest"], config_digest),
            "data_digest": (worker["data_digest"], data_digest),
            "replay_exact": (worker["replay_exact"], True),
        }
        worker_mismatches = [
            name for name, values in worker_exact.items() if values[0] != values[1]
        ]
        if worker_mismatches:
            raise SupervisedFoldFMError(
                "worker result identity differs: " + ", ".join(worker_mismatches)
            )
        encoding_path = self.artifact_store.verify(encoding_ref)
        encoding = StarterEncoding.load(
            encoding_path,
            expected_file_sha256=cast(str, worker["encoding_file_sha256"]),
        )
        control = _restore_control(
            worker["fold_control"],
            encoding=encoding,
            checkpoint_path=self.artifact_store.verify(checkpoint_ref),
            checkpoint_file_sha256=cast(str, worker["checkpoint_file_sha256"]),
            predictions_path=self.artifact_store.verify(predictions_ref),
            prediction_file_sha256=cast(str, worker["prediction_file_sha256"]),
        )
        if control.digest != evidence_manifest["fold_control_digest"]:
            raise SupervisedFoldFMError("fold control digest differs from evidence manifest")
        if (
            control.fold_name != request.fold_name
            or control.fold_token != request.fold_token
            or control.seed != request.seed
            or control.prefix_inputs_digest != request.prefix_inputs.digest
            or control.query_inputs_digest != request.query_inputs.digest
            or control.predictions.digest != worker["replay_prediction_digest"]
        ):
            raise SupervisedFoldFMError("rehydrated fold control differs from exact request")
        prefix_targets = PrimaryTrainingTargets.bind(request.prefix_inputs, request.prefix_labels)
        scoring = build_fold_scoring_context(
            self.starter_dir,
            request.fold_name,
            request.fold_token,
            request.query_inputs,
            request.query_labels,
        )
        if control.training_targets_digest != prefix_targets.digest:
            raise SupervisedFoldFMError("rehydrated prefix target identity differs")
        rescored = scoring.score_with_encoded_labels(control.predictions.scores)
        if (
            rescored.gauc != control.metrics.gauc
            or rescored.ndcg_at_5 != control.metrics.ndcg_at_5
            or rescored.primary != control.metrics.primary
        ):
            raise SupervisedFoldFMError("rehydrated predictions do not reproduce protected metrics")
        replay = control.replay_predictions(
            starter_dir=self.starter_dir,
            query_inputs=request.query_inputs,
        )
        if (
            replay.digest != control.predictions.digest
            or replay.scores.tobytes() != control.predictions.scores.tobytes()
        ):
            raise SupervisedFoldFMError("rehydrated checkpoint does not replay exactly")
        evidence = FoldFMControlEvidence(
            execution_id=request.execution_id,
            source_digest=source_digest,
            config_digest=config_digest,
            data_digest=data_digest,
            checkpoint=checkpoint_ref,
            encoding=encoding_ref,
            predictions=predictions_ref,
            worker_result=worker_ref,
            result_manifest=result_ref,
            journal_artifacts=journal_artifacts,
        )
        return SupervisedFoldFMRun(
            control=control,
            evidence=evidence,
            execution=execution,
            resumed=resumed,
            worker_pid=cast(int, worker["worker_pid"]),
        )

    def run(
        self,
        request: FoldFMControlExecutionRequest,
        *,
        journal: CampaignStoreCandidateJournal,
        cancel_event: threading.Event | None = None,
    ) -> SupervisedFoldFMRun:
        """Reserve, supervise, validate, persist, or exactly rehydrate one fold FM control."""

        if not isinstance(request, FoldFMControlExecutionRequest):
            raise SupervisedFoldFMError("request must be FoldFMControlExecutionRequest")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise SupervisedFoldFMError("cancel_event must be threading.Event or None")
        if not isinstance(journal, CampaignStoreCandidateJournal):
            raise SupervisedFoldFMError("journal must be CampaignStoreCandidateJournal")
        if journal.artifact_store is not self.artifact_store:
            raise SupervisedFoldFMError("journal and fold runner artifact stores differ")
        expected_category = (
            LaunchCategory.DIVERSE_INNER_SCREEN
            if request.fold_name == "B"
            else LaunchCategory.TEMPORAL_FOLD_CONFIRMATION
        )
        if journal.policy.category is not expected_category:
            raise SupervisedFoldFMError(
                f"fold {request.fold_name} requires launch category {expected_category.value}"
            )
        (
            prefix_labels,
            query_labels,
            source_digest,
            config_digest,
            data_digest,
            worker_source,
        ) = self._request_identities(request)

        existing = journal.store.execution(request.execution_id)
        if existing is not None:
            if existing.status not in {"SUCCEEDED", "FAILED"}:
                raise CandidateExecutionPendingError(
                    f"fold FM execution {request.execution_id!r} is already {existing.status}"
                )
            try:
                terminal = journal.rehydrate_terminal(request.execution_id)
            except ArtifactError as exc:
                raise SupervisedFoldFMError(
                    "terminal fold FM journal artifact integrity verification failed"
                ) from exc
            if existing.status != "SUCCEEDED" or not terminal.artifacts.output_validated:
                raise SupervisedFoldFMExecutionError(
                    f"fold FM execution {request.execution_id!r} is terminal FAILED",
                    artifacts=terminal.artifacts,
                )
            try:
                return self._restore_from_artifacts(
                    request,
                    journal_artifacts=terminal.artifacts,
                    source_digest=source_digest,
                    config_digest=config_digest,
                    data_digest=data_digest,
                    execution=None,
                    resumed=True,
                )
            except SupervisedFoldFMError:
                raise
            except ArtifactError as exc:
                raise SupervisedFoldFMError(
                    "terminal fold FM indirect artifact integrity verification failed"
                ) from exc

        workspace, spec = self._workspace_and_spec(
            request,
            prefix_labels=prefix_labels,
            query_labels=query_labels,
            source_digest=source_digest,
            config_digest=config_digest,
            data_digest=data_digest,
            worker_source=worker_source,
        )
        try:
            journal.prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)
        except Exception:
            self.workspace_materializer.cleanup(workspace)
            raise
        result = self.runner.run(
            spec,
            commit_launch=journal.commit,
            cancel_event=cancel_event,
        )
        if not result.succeeded:
            raise self._finish_failed(
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic=f"supervised fold FM child failed: {result.outcome.value}",
            )
        ceilings = self._output_ceilings(request)
        try:
            self.workspace_materializer.policy.validate_outputs(
                workspace,
                OutputDeclaration(
                    tuple(
                        DeclaredOutput(name, ceiling) for name, ceiling in sorted(ceilings.items())
                    )
                ),
            )
            worker_output = workspace.output_dir / _RESULT_FILENAME
            worker_ref = self.artifact_store.put_file(worker_output, kind=ArtifactKind.MANIFEST)
            checkpoint_ref = self.artifact_store.put_file(
                workspace.output_dir / _CHECKPOINT_FILENAME,
                kind=ArtifactKind.CHECKPOINT,
            )
            encoding_ref = self.artifact_store.put_file(
                workspace.output_dir / _ENCODING_FILENAME,
                kind=ArtifactKind.OTHER,
            )
            predictions_ref = self.artifact_store.put_file(
                workspace.output_dir / _PREDICTION_FILENAME,
                kind=ArtifactKind.PREDICTION,
            )
            worker_bytes = self.artifact_store.read_bytes(
                worker_ref, max_bytes=worker_ref.size_bytes
            )
            worker_raw = json.loads(worker_bytes)
            if not isinstance(worker_raw, Mapping) or worker_bytes != _canonical_json(worker_raw):
                raise SupervisedFoldFMError("worker result is not canonical JSON")
            fold_control = worker_raw.get("fold_control")
            if not isinstance(fold_control, Mapping) or type(fold_control.get("digest")) is not str:
                raise SupervisedFoldFMError("worker result omitted the fold control digest")
            evidence_payload = {
                "schema_version": _SCHEMA_VERSION,
                "execution_id": request.execution_id,
                "source_digest": source_digest,
                "config_digest": config_digest,
                "data_digest": data_digest,
                "worker_result": worker_ref.manifest(),
                "checkpoint": checkpoint_ref.manifest(),
                "encoding": encoding_ref.manifest(),
                "predictions": predictions_ref.manifest(),
                "fold_control_digest": fold_control["digest"],
            }
            result_ref = self.artifact_store.put_bytes(
                _canonical_json(evidence_payload), kind=ArtifactKind.MANIFEST
            )
            cleanup = self._cleanup_workspace(workspace)
            if cleanup[1] is not None:
                raise self._finish_failed(
                    result=result,
                    journal=journal,
                    workspace=workspace,
                    diagnostic="supervised fold FM workspace cleanup failed",
                    cleanup=cleanup,
                )
            journal_artifacts = CandidateExecutionArtifacts(
                tuple(
                    [
                        *_base_artifacts(self.artifact_store, result),
                        ("candidate_result", result_ref),
                        ("checkpoint", checkpoint_ref),
                        ("workspace_cleanup", cleanup[0]),
                    ]
                ),
                output_validated=True,
            )
            restored = self._restore_from_artifacts(
                request,
                journal_artifacts=journal_artifacts,
                source_digest=source_digest,
                config_digest=config_digest,
                data_digest=data_digest,
                execution=result,
                resumed=False,
            )
        except Exception as exc:
            if isinstance(exc, SupervisedFoldFMExecutionError):
                raise
            raise self._finish_failed(
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic=f"supervised fold FM output validation failed: {type(exc).__name__}",
            ) from exc
        journal.finish(
            action=CandidateAction.TRAIN,
            result=result,
            artifacts=journal_artifacts,
        )
        return restored


__all__ = [
    "FoldFMControlEvidence",
    "FoldFMControlExecutionRequest",
    "FoldName",
    "SupervisedFoldFMError",
    "SupervisedFoldFMExecutionError",
    "SupervisedFoldFMRun",
    "SupervisedFoldFMRunner",
]
