from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from kuairand_agent.baselines.artifacts import (
    PredictionVector,
    StarterFMCheckpoint,
    save_checkpoint,
    save_predictions,
)
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.campaign.qualification_evidence import (
    QualificationEvidenceError,
    QualificationExpectations,
    load_official_fm_qualification,
)
from kuairand_agent.contract import (
    BENCHMARK_CONTRACT,
    DATASET_ARCHIVE_MD5,
    DATASET_ARCHIVE_SHA256,
    STARTER_FILE_SHA256,
    SplitName,
)
from kuairand_agent.data.canonical import (
    OUTCOME_FIELDS,
    CanonicalAlignment,
    CanonicalFinalSplit,
    CanonicalInputs,
    OutcomeAccessTrace,
)

_STARTER = "1" * 64
_CANONICAL = "2" * 64
_AUDIT = "3" * 64
_TRAIN_INPUTS = "4" * 64
_VALID_INPUTS = "5" * 64
_TRAIN_TARGETS = "6" * 64
_VALID_TARGETS = "7" * 64
_VALIDATION_LABELS = "8" * 64
_FIXTURE_DIGEST = "9" * 64
_SCORER = STARTER_FILE_SHA256["evaluate.py"]
_VALIDATION_ROWS = 4
_FINAL_ROWS = 3
_METRICS = (
    (0.6671333909034729, 0.5358057022094727, 0.6014695167541504),
    (0.667394757270813, 0.536126971244812, 0.6017608642578125),
    (0.6670642495155334, 0.53511643409729, 0.6010903120040894),
    (0.667461097240448, 0.5355450510978699, 0.6015030741691589),
    (0.6679478287696838, 0.5361263751983643, 0.6020370721817017),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _signed(body: dict[str, object]) -> dict[str, object]:
    return body | {"digest": _digest(body)}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value) + b"\n")


def _file_record(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _tree_digest(root: Path) -> str:
    records = [_file_record(path, root) for path in sorted(root.rglob("*")) if path.is_file()]
    return _digest(records)


def _published_fm_metrics() -> dict[str, float]:
    for rung in BENCHMARK_CONTRACT.reference_rungs:
        if rung.name == "fm_official":
            return rung.validation.manifest()
    raise AssertionError("frozen contract omitted official FM")


def _canonical_final_manifest() -> dict[str, object]:
    inputs = CanonicalInputs(
        user_id=("final-user-0", "final-user-1", "final-user-2"),
        video_id=("final-video-0", "final-video-1", "final-video-2"),
        date=(20220429, 20220429, 20220429),
        duration_ms=(1_000.0, 2_000.0, 3_000.0),
        tab=("0", "1", "2"),
        author_id=("author-0", "author-1", "author-2"),
        time_ms=(1, 2, 3),
    )
    return CanonicalFinalSplit(
        name=SplitName.TEST,
        inputs=inputs,
        alignment=CanonicalAlignment(
            split=SplitName.TEST,
            row_id=(0, 1, 2),
            user_id=inputs.user_id,
            video_id=inputs.video_id,
        ),
        outcome_trace=OutcomeAccessTrace(
            split=SplitName.TEST,
            row_count=3,
            parsed_fields=(),
            skipped_fields=OUTCOME_FIELDS,
        ),
    ).manifest()


def _snapshot() -> dict[str, object]:
    splits = [
        {
            "name": "train",
            "row_count": 10,
            "inputs_digest": _TRAIN_INPUTS,
            "target_access": "training",
            "target_digest": _TRAIN_TARGETS,
            "outcome_access": {"parsed_cell_count": 110, "skipped_values_recorded": False},
        },
        {
            "name": "valid",
            "row_count": _VALIDATION_ROWS,
            "inputs_digest": _VALID_INPUTS,
            "target_access": "protected_scorer_only",
            "target_digest": _VALID_TARGETS,
            "outcome_access": {"parsed_cell_count": 4, "skipped_values_recorded": False},
        },
        _canonical_final_manifest(),
    ]
    final_trace = {
        "row_count": _FINAL_ROWS,
        "outcome_cells_materialized": 0,
        "outcome_cells_decoded": 0,
        "outcome_cells_converted": 0,
        "outcome_cells_validated": 0,
        "outcome_cells_aggregated": 0,
        "outcome_cells_logged": 0,
        "outcome_cells_scored": 0,
        "skipped_values_recorded": False,
    }
    return {
        "audit_digest": _AUDIT,
        "audit_manifest": {
            "digest": _AUDIT,
            "archive_identity": {
                "md5": DATASET_ARCHIVE_MD5,
                "sha256": DATASET_ARCHIVE_SHA256,
            },
            "final_outcome_trace": final_trace,
        },
        "benchmark_digest": BENCHMARK_CONTRACT.digest,
        "canonical_digest": _CANONICAL,
        "canonical_manifest": {"digest": _CANONICAL, "splits": splits},
        "evaluator_golden_digest": _FIXTURE_DIGEST,
        "evaluator_golden_passed": True,
        "final_alignment_count": _FINAL_ROWS,
        "final_target_capability": None,
        "starter_manifest_digest": _STARTER,
        "validation_alignment_count": _VALIDATION_ROWS,
        "validation_label_digest": _VALIDATION_LABELS,
    }


def _build_qualification(root: Path) -> Path:
    root.mkdir()
    snapshot = _snapshot()
    _write_json(root / "verification" / "snapshot-first.json", snapshot)
    _write_json(root / "verification" / "snapshot-second.json", snapshot)

    encoding = StarterEncoding(
        edges=tuple(float(index) for index in range(9)),
        vocabs=(("u",), ("v",), ("a",), ("t",), ("0",)),
        training_inputs_digest=_TRAIN_INPUTS,
    )
    runs: list[dict[str, object]] = []
    for seed, (gauc, ndcg, primary) in enumerate(_METRICS):
        seed_dir = root / "fm" / f"seed-{seed}"
        seed_dir.mkdir(parents=True)
        encoding_artifact = encoding.save(seed_dir / "encoding.npz")
        config_digest = hashlib.sha256(f"config-{seed}".encode()).hexdigest()
        checkpoint = StarterFMCheckpoint(
            V=np.zeros((encoding.total_dim, 16), dtype=np.float32),
            W=np.zeros(encoding.total_dim, dtype=np.float32),
            b=np.float32(seed / 100),
            encoding_digest=encoding.digest,
            config_digest=config_digest,
            starter_manifest_digest=_STARTER,
            seed=seed,
            best_epoch=1,
            epochs_completed=1,
            optimizer_steps=1,
        )
        checkpoint_artifact = save_checkpoint(seed_dir / "checkpoint.npz", checkpoint)
        predictions = PredictionVector(np.asarray([0.1, 0.2, 0.3, 0.4]) + seed / 100)
        prediction_artifact = save_predictions(seed_dir / "validation-predictions.npy", predictions)
        artifacts = [
            _file_record(seed_dir / "encoding.npz", root),
            _file_record(seed_dir / "checkpoint.npz", root),
            _file_record(seed_dir / "validation-predictions.npy", root),
        ]
        metrics = {"GAUC": gauc, "nDCG@5": ndcg, "primary": primary}
        trusted_proof = {
            "digest": _FIXTURE_DIGEST,
            "evaluator_golden": {"passed": True, "scorer_digest": _SCORER},
            "starter_fm_untouched_fixture_parity": {
                "passed": True,
                "organizer_evaluator_sha256": _SCORER,
                "organizer_manifest_sha256": _STARTER,
            },
        }
        trace: dict[str, object] = {
            "checkpoint": checkpoint.manifest(),
            "config_digest": config_digest,
            "encoding_digest": encoding.digest,
            "schema_version": 1,
            "starter_manifest_digest": _STARTER,
            "train_inputs_digest": _TRAIN_INPUTS,
            "training_targets_digest": _TRAIN_TARGETS,
            "trusted_fixture_proof": trusted_proof,
            "validation_inputs_digest": _VALID_INPUTS,
            "validation_metrics": metrics,
            "validation_predictions": predictions.manifest(),
        }
        run: dict[str, object] = {
            "artifact_file_sha256": {
                "encoding.npz": encoding_artifact.file_sha256,
                "checkpoint.npz": checkpoint_artifact.file_sha256,
                "validation-predictions.npy": prediction_artifact.file_sha256,
            },
            "artifacts": artifacts,
            "checkpoint_digest": checkpoint.digest,
            "config_digest": config_digest,
            "encoding_digest": encoding.digest,
            "organizer_parity_passed": True,
            "resources": {
                "cpu_seconds": 1.0,
                "device": "cpu",
                "peak_rss_bytes": 1,
                "wall_seconds": 1.0,
            },
            "seed": seed,
            "starter_manifest_digest": _STARTER,
            "training_trace": trace,
            "validation_metrics": metrics,
            "validation_prediction_digest": predictions.digest,
        }
        _write_json(seed_dir / "run.json", run)
        runs.append(run)

    replay_manifests: list[dict[str, object]] = []
    for seed, run in enumerate(runs):
        replay = {
            "seed": seed,
            "checkpoint_digest": run["checkpoint_digest"],
            "prediction_digest": run["validation_prediction_digest"],
            "prediction_identity": True,
            "metrics": run["validation_metrics"],
            "charged_launch": False,
        }
        _write_json(root / "replays" / f"seed-{seed}.json", replay)
        replay_manifests.append(replay)

    clean_dir = root / "clean-retrain-seed-0"
    shutil.copytree(root / "fm" / "seed-0", clean_dir)
    clean = cast(dict[str, object], json.loads((clean_dir / "run.json").read_text()))
    clean_trace = cast(dict[str, object], clean["training_trace"])
    logical_run = {
        key: value for key, value in clean_trace.items() if key != "trusted_fixture_proof"
    }
    clean_worker = _signed(
        {
            "schema_version": 1,
            "kind": "clean_seed_zero_source_retrain_worker",
            "seed": 0,
            "starter_manifest_digest": _STARTER,
            "canonical_digest": _CANONICAL,
            "audit_digest": _AUDIT,
            "train_inputs_digest": _TRAIN_INPUTS,
            "validation_inputs_digest": _VALID_INPUTS,
            "checkpoint_digest": clean["checkpoint_digest"],
            "encoding_digest": clean["encoding_digest"],
            "config_digest": clean["config_digest"],
            "validation_prediction_digest": clean["validation_prediction_digest"],
            "validation_metrics": clean["validation_metrics"],
            "final_target_capability": None,
            "final_outcomes_accessed": False,
            "artifact_sha256": dict(cast(dict[str, object], clean["artifact_file_sha256"])),
            "logical_run": logical_run,
            "logical_run_digest": _digest(logical_run),
            "resources": clean["resources"],
        }
    )
    _write_json(clean_dir / "clean-worker-evidence.json", clean_worker)
    clean_hashes = cast(dict[str, object], clean["artifact_file_sha256"])
    clean_hashes["clean-worker-evidence.json"] = hashlib.sha256(
        (clean_dir / "clean-worker-evidence.json").read_bytes()
    ).hexdigest()
    clean["artifacts"] = [
        _file_record(clean_dir / "encoding.npz", root),
        _file_record(clean_dir / "checkpoint.npz", root),
        _file_record(clean_dir / "validation-predictions.npy", root),
        _file_record(clean_dir / "clean-worker-evidence.json", root),
    ]
    clean_trace["clean_subprocess"] = {
        "evidence_digest": clean_worker["digest"],
        "evidence_file_sha256": clean_hashes["clean-worker-evidence.json"],
        "fresh_interpreter": True,
        "identity_verified_by_parent": True,
        "source_retrain": True,
    }
    clean.update(
        {
            "source_seed": 0,
            "prediction_identity": True,
            "within_user_order_identity": True,
            "absolute_tolerance": 0.0,
        }
    )
    _write_json(clean_dir / "run.json", clean)

    validation_bytes = b"row_id,user_id,video_id,score\n"
    final_bytes = b"row_id,user_id,video_id,score\n"
    (root / "validation").mkdir()
    (root / "validation" / "submission.csv").write_bytes(validation_bytes)
    (root / "final").mkdir()
    (root / "final" / "submission.csv").write_bytes(final_bytes)

    fallback_model = root / "fallback" / "model"
    shutil.copytree(root / "fm" / "seed-4", fallback_model)
    source_tree_digest = _tree_digest(root / "fm" / "seed-4")
    fallback_tree_digest = _tree_digest(fallback_model)
    seed_four = runs[4]
    fallback = _signed(
        {
            "schema_version": 1,
            "kind": "immutable_official_fm_fallback",
            "seed": 4,
            "validation_metrics": seed_four["validation_metrics"],
            "checkpoint_digest": seed_four["checkpoint_digest"],
            "encoding_digest": seed_four["encoding_digest"],
            "config_digest": seed_four["config_digest"],
            "validation_prediction_digest": seed_four["validation_prediction_digest"],
            "validation_submission": {
                "path": "validation/submission.csv",
                "sha256": hashlib.sha256(validation_bytes).hexdigest(),
                "prediction_digest": seed_four["validation_prediction_digest"],
                "round_trip_identity": True,
                "protected_metrics_preserved": True,
            },
            "final_submission": {
                "path": "final/submission.csv",
                "sha256": hashlib.sha256(final_bytes).hexdigest(),
                "prediction_digest": "b" * 64,
                "round_trip_identity": True,
                "final_outcomes_accessed": False,
            },
            "source_model_tree_digest": source_tree_digest,
            "fallback_model_tree_digest": fallback_tree_digest,
            "replay_verified": True,
            "clean_seed_zero_retrain_verified": True,
        }
    )
    _write_json(root / "fallback" / "manifest.json", fallback)

    mean = {
        name: sum(cast(dict[str, float], run["validation_metrics"])[name] for run in runs)
        / len(runs)
        for name in ("GAUC", "nDCG@5", "primary")
    }
    scorer_evaluation = {
        "scorer_digest": _SCORER,
        "rows": _VALIDATION_ROWS,
        "users": 2,
    }
    root_body: dict[str, object] = {
        "schema_version": 1,
        "status": "baseline_reproduced",
        "benchmark_digest": BENCHMARK_CONTRACT.digest,
        "qualification_input_digest": _digest(snapshot),
        "double_build_identity": True,
        "rungs": {
            "random": {
                "evaluations": [scorer_evaluation],
                "reference_passed": True,
            },
            "item_popularity": {
                "evaluations": [scorer_evaluation],
                "reference_passed": True,
            },
        },
        "fm": {
            "seeds": [0, 1, 2, 3, 4],
            "runs": runs,
            "five_seed_mean": mean,
            "published_reference": _published_fm_metrics(),
            "reference_passed": True,
            "checkpoint_replays": replay_manifests,
            "clean_seed_zero": clean,
        },
        "launch_accounting": {
            "charged_launches": 6,
            "expected_launches": 6,
            "records": [
                *(
                    {
                        "launch_number": seed + 1,
                        "kind": "official_fm_training",
                        "seed": seed,
                        "charged": True,
                    }
                    for seed in range(5)
                ),
                {
                    "launch_number": 6,
                    "kind": "clean_source_retrain",
                    "seed": 0,
                    "charged": True,
                },
            ],
            "random_rungs_charged": False,
            "popularity_rung_charged": False,
            "checkpoint_replays_charged": False,
        },
        "fallback": fallback,
        "resource_usage": {},
        "final_period": {
            "input_rows": _FINAL_ROWS,
            "target_capability": None,
            "outcomes_accessed": False,
            "outcomes_scored": False,
        },
    }
    root_body["artifacts"] = [
        _file_record(path, root) for path in sorted(root.rglob("*")) if path.is_file()
    ]
    _write_json(root / "manifest.json", _signed(root_body))
    return root


def _expectations() -> QualificationExpectations:
    return QualificationExpectations(
        canonical_digest=_CANONICAL,
        starter_manifest_digest=_STARTER,
        scorer_digest=_SCORER,
        validation_row_count=_VALIDATION_ROWS,
        final_row_count=_FINAL_ROWS,
    )


def _resign_root_artifact_index(root: Path) -> None:
    manifest = cast(dict[str, object], json.loads((root / "manifest.json").read_text()))
    body = {key: value for key, value in manifest.items() if key not in {"artifacts", "digest"}}
    body["artifacts"] = [
        _file_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    ]
    _write_json(root / "manifest.json", _signed(body))


def _rewrite_snapshots(
    root: Path,
    transform: Callable[[dict[str, object]], None],
) -> None:
    snapshot_path = root / "verification" / "snapshot-first.json"
    snapshot = cast(dict[str, object], json.loads(snapshot_path.read_text(encoding="ascii")))
    transform(snapshot)
    _write_json(snapshot_path, snapshot)
    _write_json(root / "verification" / "snapshot-second.json", snapshot)
    manifest = cast(dict[str, object], json.loads((root / "manifest.json").read_text()))
    manifest["qualification_input_digest"] = _digest(snapshot)
    _write_json(root / "manifest.json", manifest)
    _resign_root_artifact_index(root)


def _final_snapshot_split(snapshot: dict[str, object]) -> dict[str, object]:
    canonical = cast(dict[str, object], snapshot["canonical_manifest"])
    splits = cast(list[object], canonical["splits"])
    matches = [
        cast(dict[str, object], item)
        for item in splits
        if isinstance(item, dict) and item.get("name") == "test"
    ]
    assert len(matches) == 1
    return matches[0]


def test_accepts_real_canonical_final_split_without_target_shape(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")
    snapshot = cast(
        dict[str, object],
        json.loads((root / "verification" / "snapshot-first.json").read_text()),
    )
    final = _final_snapshot_split(snapshot)

    assert "target_access" not in final
    assert "target_digest" not in final
    load_official_fm_qualification(root, expectations=_expectations())


def test_accepts_explicit_null_final_target_shape(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")

    def add_null_target_shape(snapshot: dict[str, object]) -> None:
        final = _final_snapshot_split(snapshot)
        final["target_access"] = None
        final["target_digest"] = None

    _rewrite_snapshots(root, add_null_target_shape)

    load_official_fm_qualification(root, expectations=_expectations())


def test_accepts_legacy_none_final_target_access(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")

    def add_legacy_target_sentinel(snapshot: dict[str, object]) -> None:
        final = _final_snapshot_split(snapshot)
        final["target_access"] = "none"
        final["target_digest"] = None

    _rewrite_snapshots(root, add_legacy_target_sentinel)

    load_official_fm_qualification(root, expectations=_expectations())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_access", "training"),
        ("target_access", "protected_scorer_only"),
        ("target_access", "NONE"),
        ("target_access", []),
        ("target_access", {}),
        ("target_access", False),
        ("target_access", 0),
        ("target_digest", "f" * 64),
    ),
)
def test_rejects_non_null_final_target_shape(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = _build_qualification(tmp_path / "qualification")

    def add_target_claim(snapshot: dict[str, object]) -> None:
        _final_snapshot_split(snapshot)[field] = value

    _rewrite_snapshots(root, add_target_claim)

    with pytest.raises(QualificationEvidenceError, match="final split exposed target capability"):
        load_official_fm_qualification(root, expectations=_expectations())


def test_rejects_parsed_final_outcomes_with_target_free_shape(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")

    def add_parsed_final_outcome(snapshot: dict[str, object]) -> None:
        final = _final_snapshot_split(snapshot)
        outcome = cast(dict[str, object], final["outcome_access"])
        outcome["parsed_cell_count"] = 1

    _rewrite_snapshots(root, add_parsed_final_outcome)

    with pytest.raises(QualificationEvidenceError, match="final outcomes were materialized"):
        load_official_fm_qualification(root, expectations=_expectations())


def test_loads_exact_matching_seed_public_evidence_and_fallback(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")

    evidence = load_official_fm_qualification(root, expectations=_expectations())

    assert tuple(run.seed for run in evidence.outer_runs) == (0, 1, 2)
    assert evidence.outer_seed(1).metrics.primary == 0.6017608642578125
    assert (
        evidence.outer_seed(1).validation_predictions_path
        == (root / "fm" / "seed-1" / "validation-predictions.npy").resolve()
    )
    assert evidence.outer_seed(1).checkpoint_path.name == "checkpoint.npz"
    assert evidence.canonical_digest == _CANONICAL
    assert evidence.scorer_digest == _SCORER
    assert evidence.fallback.seed == 4
    assert evidence.fallback.metrics.primary == 0.6020370721817017
    assert evidence.fallback.model_dir == (root / "fallback" / "model").resolve()


def test_rejects_tampered_indexed_model_artifact(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")
    predictions = root / "fm" / "seed-1" / "validation-predictions.npy"
    predictions.write_bytes(predictions.read_bytes() + b"tamper")

    with pytest.raises(QualificationEvidenceError, match="artifact changed"):
        load_official_fm_qualification(root, expectations=_expectations())


def test_rejects_missing_matching_seed_artifact(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")
    (root / "fm" / "seed-2" / "checkpoint.npz").unlink()

    with pytest.raises(QualificationEvidenceError, match="artifact closure mismatch"):
        load_official_fm_qualification(root, expectations=_expectations())


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("canonical_digest", "canonical dataset identity mismatch"),
        ("starter_manifest_digest", "starter-kit identity mismatch"),
        ("scorer_digest", "used another protected scorer"),
    ),
)
def test_rejects_current_identity_mismatch(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    root = _build_qualification(tmp_path / "qualification")
    baseline = _expectations()
    expectations = QualificationExpectations(
        canonical_digest=("f" * 64 if field == "canonical_digest" else baseline.canonical_digest),
        starter_manifest_digest=(
            "f" * 64 if field == "starter_manifest_digest" else baseline.starter_manifest_digest
        ),
        scorer_digest=("f" * 64 if field == "scorer_digest" else baseline.scorer_digest),
        validation_row_count=baseline.validation_row_count,
        final_row_count=baseline.final_row_count,
    )

    with pytest.raises(QualificationEvidenceError, match=message):
        load_official_fm_qualification(root, expectations=expectations)


def test_rejects_resigned_but_divergent_fallback_model_tree(tmp_path: Path) -> None:
    root = _build_qualification(tmp_path / "qualification")
    fallback_predictions = root / "fallback" / "model" / "validation-predictions.npy"
    fallback_predictions.write_bytes(fallback_predictions.read_bytes() + b"different-copy")
    _resign_root_artifact_index(root)

    with pytest.raises(QualificationEvidenceError, match="fallback model tree differs"):
        load_official_fm_qualification(root, expectations=_expectations())
