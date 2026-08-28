from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from kuairand_agent.candidate_api.protocol import (
    CandidateProtocolError,
    PredictionExpectation,
    TrainExpectation,
    validate_prediction_outputs,
    validate_train_outputs,
)
from kuairand_agent.research.materialize import (
    materialize_candidate,
    require_material_executable_change,
)
from kuairand_agent.research.schemas import (
    GeneratedFile,
    GeneratedPackage,
    ParentSnapshot,
    ParentSourceFile,
)

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "candidate_templates" / "pairwise_fm"
SOURCE_DIGEST = "1" * 64
TRAIN_DATA_DIGEST = "2" * 64
PREDICT_DATA_DIGEST = "3" * 64
SNAPSHOT_DIGEST = "4" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npy(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)


def _approved_input(name: str, role: str, path: Path, request_root: Path) -> dict[str, object]:
    return {
        "name": name,
        "role": role,
        "workspace_path": path.relative_to(request_root).as_posix(),
        "artifact": {
            "schema_version": 1,
            "algorithm": "sha256",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "kind": "input",
        },
    }


def _write_request(
    path: Path,
    *,
    split_role: str,
    capabilities: tuple[tuple[str, str, Path], ...],
    request: dict[str, object],
) -> None:
    value = {
        "schema_version": 1,
        "execution_id": f"pairwise-fm-{split_role}",
        "split_role": split_role,
        "source_snapshot_sha256": SNAPSHOT_DIGEST,
        "approved_inputs": [
            _approved_input(name, role, capability_path, path.parent)
            for name, role, capability_path in capabilities
        ],
        "budgets": {
            "output_limit_bytes": 8 * 1024 * 1024,
            "temp_limit_bytes": 8 * 1024 * 1024,
        },
        "request": request,
    }
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="ascii",
    )


def _run(*arguments: str, cwd: Path, success: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, str(TEMPLATE / "candidate.py"), *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if success:
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == ""
        assert completed.stderr == ""
    else:
        assert completed.returncode != 0
        assert completed.stdout == ""
        assert "CandidateInputError" in completed.stderr
    return completed


def _training_payload(config_digest: str, *, seed: int = 7) -> dict[str, object]:
    return {
        "protocol_schema_version": 1,
        "source_digest": SOURCE_DIGEST,
        "config_digest": config_digest,
        "data_digest": TRAIN_DATA_DIGEST,
        "split_token": "fold-b-train",
        "seed": seed,
        "features_handle": "features",
        "targets_handle": "targets",
        "user_groups_handle": "user_groups",
    }


def _prediction_payload(
    config_digest: str,
    checkpoint_digest: str,
    *,
    expected_count: int,
) -> dict[str, object]:
    return {
        "protocol_schema_version": 1,
        "source_digest": SOURCE_DIGEST,
        "config_digest": config_digest,
        "data_digest": PREDICT_DATA_DIGEST,
        "split_token": "fold-b-valid",
        "features_handle": "features",
        "expected_count": expected_count,
        "checkpoint_digest": checkpoint_digest,
    }


def _tiny_capabilities(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    user_groups = np.repeat(np.arange(12, dtype=np.int64), 6)
    within_user_signal = np.tile(
        np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float64),
        12,
    )
    targets = (within_user_signal > 0.0).astype(np.int8)
    features = np.column_stack(
        (
            within_user_signal,
            np.sin(within_user_signal),
            (user_groups % 3).astype(np.float64),
        )
    ).astype("<f8")
    features_path = inputs / "features.npy"
    targets_path = inputs / "targets.npy"
    groups_path = inputs / "user-groups.npy"
    for path, values in (
        (features_path, features),
        (targets_path, targets),
        (groups_path, user_groups),
    ):
        if not path.exists():
            _write_npy(path, values)
    return features_path, targets_path, groups_path, features, targets


def test_template_is_a_material_generated_pairwise_candidate(tmp_path: Path) -> None:
    source = (TEMPLATE / "candidate.py").read_text(encoding="utf-8")
    config = (TEMPLATE / "config.json").read_text(encoding="utf-8")
    parent = ParentSnapshot(
        candidate_id="minimal-pairwise-parent",
        files=(
            ParentSourceFile.create(
                "candidate.py",
                "def train_model(features, targets, user_groups, config, seed):\n"
                "    return None\n\n"
                "def predict_scores(features, checkpoint):\n"
                "    return []\n",
            ),
            ParentSourceFile.create("config.json", config),
        ),
    )
    package = GeneratedPackage(
        request_id="generated-pairwise-materiality",
        response_id="generated-pairwise-source",
        files=(
            GeneratedFile("candidate.py", source),
            GeneratedFile("config.json", config),
        ),
        material_change_summary=(
            "Own positive-weighted logged same-user pair sampling and pairwise FM training."
        ),
        material_symbols=("GAUCPairSampler", "train_model", "predict_scores"),
    )

    child = materialize_candidate(parent, package, tmp_path / "materialized-pairwise")
    evidence = require_material_executable_change(parent, child)

    assert evidence.changed_symbols == (
        "candidate.py:GAUCPairSampler",
        "candidate.py:predict_scores",
        "candidate.py:train_model",
    )
    assert evidence.reachable_python_files == ("candidate.py",)


def test_train_predict_commands_are_protocol_valid_and_exactly_replayable(
    tmp_path: Path,
) -> None:
    features_path, targets_path, groups_path, features, targets = _tiny_capabilities(tmp_path)
    config_digest = _sha256(TEMPLATE / "config.json")
    train_request = tmp_path / "train-request.json"
    _write_request(
        train_request,
        split_role="inner_train",
        capabilities=(
            ("features", "train_inputs", features_path),
            ("targets", "train_targets", targets_path),
            ("user_groups", "train_inputs", groups_path),
        ),
        request=_training_payload(config_digest),
    )

    first_train = tmp_path / "train-a"
    second_train = tmp_path / "train-b"
    for output in (first_train, second_train):
        _run("train", "--request", str(train_request), "--output", str(output), cwd=tmp_path)

    expected_train = TrainExpectation(
        source_digest=SOURCE_DIGEST,
        config_digest=config_digest,
        data_digest=TRAIN_DATA_DIGEST,
        split_token="fold-b-train",
        checkpoint_path="checkpoint/model.txt",
    )
    first = validate_train_outputs(first_train.resolve(), expected_train)
    second = validate_train_outputs(second_train.resolve(), expected_train)
    assert first.checkpoint_digest == second.checkpoint_digest
    assert first.checkpoint_path.read_bytes() == second.checkpoint_path.read_bytes()
    assert (first_train / "candidate_result.json").read_bytes() == (
        second_train / "candidate_result.json"
    ).read_bytes()
    assert first.manifest.diagnostics["sampling_policy"] == ("positive_weighted_logged_same_user")
    assert first.manifest.diagnostics["stored_row_index_count"] == features.shape[0]
    assert first.manifest.diagnostics["sampled_pairs"] == 8 * 4096

    predict_request = tmp_path / "predict-request.json"
    _write_request(
        predict_request,
        split_role="inner_valid",
        capabilities=(("features", "inner_valid_inputs", features_path),),
        request=_prediction_payload(
            config_digest,
            first.checkpoint_digest,
            expected_count=features.shape[0],
        ),
    )
    first_prediction = tmp_path / "prediction-a"
    second_prediction = tmp_path / "prediction-b"
    for output in (first_prediction, second_prediction):
        _run(
            "predict",
            "--request",
            str(predict_request),
            "--checkpoint",
            str(first.checkpoint_path),
            "--output",
            str(output),
            cwd=tmp_path,
        )

    expected_prediction = PredictionExpectation(
        source_digest=SOURCE_DIGEST,
        config_digest=config_digest,
        data_digest=PREDICT_DATA_DIGEST,
        split_token="fold-b-valid",
        checkpoint_digest=first.checkpoint_digest,
        expected_count=features.shape[0],
    )
    predicted = validate_prediction_outputs(first_prediction.resolve(), expected_prediction)
    replayed = validate_prediction_outputs(second_prediction.resolve(), expected_prediction)
    assert predicted.scores.dtype.str == "<f8"
    assert predicted.scores.tobytes() == replayed.scores.tobytes()
    assert predicted.scores_sha256 == replayed.scores_sha256
    assert float(predicted.scores[targets == 1].mean()) > float(
        predicted.scores[targets == 0].mean()
    )
    assert (first_prediction / "prediction_result.json").read_bytes() == (
        second_prediction / "prediction_result.json"
    ).read_bytes()

    # Even if an untrusted candidate rewrites its file digest declaration, the trusted
    # protocol rejects a non-finite output vector.
    bad_scores = np.asarray(predicted.scores).copy()
    bad_scores[0] = np.nan
    scores_path = first_prediction / "scores.npy"
    scores_path.unlink()
    _write_npy(scores_path, bad_scores)
    manifest_path = first_prediction / "prediction_result.json"
    manifest = cast(
        dict[str, object],
        json.loads(manifest_path.read_text(encoding="ascii")),
    )
    manifest["scores_sha256"] = _sha256(scores_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="ascii",
    )
    with pytest.raises(CandidateProtocolError, match="finite"):
        validate_prediction_outputs(first_prediction.resolve(), expected_prediction)


@pytest.mark.parametrize(
    ("command", "split_role"),
    (("train", "final"), ("predict", "train")),
)
def test_commands_reject_wrong_phase_without_creating_output(
    tmp_path: Path,
    command: str,
    split_role: str,
) -> None:
    features_path, targets_path, groups_path, features, _ = _tiny_capabilities(tmp_path)
    config_digest = _sha256(TEMPLATE / "config.json")
    request_path = tmp_path / f"wrong-{command}-phase.json"
    output = tmp_path / f"wrong-{command}-phase-output"
    arguments: tuple[str, ...]
    if command == "train":
        _write_request(
            request_path,
            split_role=split_role,
            capabilities=(
                ("features", "train_inputs", features_path),
                ("targets", "train_targets", targets_path),
                ("user_groups", "train_inputs", groups_path),
            ),
            request=_training_payload(config_digest),
        )
        arguments = ("train", "--request", str(request_path), "--output", str(output))
    else:
        _write_request(
            request_path,
            split_role=split_role,
            capabilities=(("features", "train_inputs", features_path),),
            request=_prediction_payload(config_digest, "f" * 64, expected_count=features.shape[0]),
        )
        arguments = (
            "predict",
            "--request",
            str(request_path),
            "--checkpoint",
            str(features_path),
            "--output",
            str(output),
        )

    _run(*arguments, cwd=tmp_path, success=False)
    assert not output.exists()


def test_prediction_rejects_any_label_capability_without_reading_or_writing_output(
    tmp_path: Path,
) -> None:
    features_path, targets_path, _, features, _ = _tiny_capabilities(tmp_path)
    config_digest = _sha256(TEMPLATE / "config.json")
    request_path = tmp_path / "label-leak-request.json"
    _write_request(
        request_path,
        split_role="outer_valid",
        capabilities=(
            ("features", "outer_valid_inputs", features_path),
            ("targets", "train_targets", targets_path),
        ),
        request=_prediction_payload(config_digest, "f" * 64, expected_count=features.shape[0]),
    )
    # The request manifest is complete, but remove the forbidden label file. A phase-first
    # implementation rejects its role without opening or hashing those label bytes.
    targets_path.unlink()
    output = tmp_path / "label-leak-output"

    completed = _run(
        "predict",
        "--request",
        str(request_path),
        "--checkpoint",
        str(features_path),
        "--output",
        str(output),
        cwd=tmp_path,
        success=False,
    )

    assert "role is not allowed for the workspace phase" in completed.stderr
    assert "unavailable" not in completed.stderr
    assert not output.exists()


def test_nonfinite_training_capability_is_rejected_before_output(
    tmp_path: Path,
) -> None:
    features_path, targets_path, groups_path, _, _ = _tiny_capabilities(tmp_path)
    features_path.unlink()
    _write_npy(features_path, np.array([[1.0, np.nan, 3.0]], dtype="<f8"))
    targets_path.unlink()
    _write_npy(targets_path, np.array([1], dtype=np.int8))
    groups_path.unlink()
    _write_npy(groups_path, np.array([1], dtype=np.int64))
    config_digest = _sha256(TEMPLATE / "config.json")
    request_path = tmp_path / "nonfinite.json"
    _write_request(
        request_path,
        split_role="inner_train",
        capabilities=(
            ("features", "train_inputs", features_path),
            ("targets", "train_targets", targets_path),
            ("user_groups", "train_inputs", groups_path),
        ),
        request=_training_payload(config_digest),
    )
    output = tmp_path / "nonfinite-output"

    _run(
        "train",
        "--request",
        str(request_path),
        "--output",
        str(output),
        cwd=tmp_path,
        success=False,
    )
    assert not output.exists()


def test_finite_checkpoint_that_would_emit_infinite_scores_is_rejected_before_output(
    tmp_path: Path,
) -> None:
    features_path, _, _, features, _ = _tiny_capabilities(tmp_path)
    checkpoint_path = tmp_path / "explosive-model.txt"

    def scalar(value: int) -> np.ndarray:
        return np.asarray(value, dtype="<i8")

    with checkpoint_path.open("xb") as handle:
        np.savez(
            handle,
            checkpoint_schema_version=scalar(1),
            eligible_positive_count=scalar(1),
            eligible_user_count=scalar(1),
            epochs=scalar(1),
            factor_dim=scalar(1),
            factors=np.zeros((features.shape[1], 1), dtype="<f8"),
            feature_mean=np.zeros(features.shape[1], dtype="<f8"),
            feature_scale=np.ones(features.shape[1], dtype="<f8"),
            final_data_loss=np.asarray(0.0, dtype="<f8"),
            linear=np.full(features.shape[1], 1e308, dtype="<f8"),
            optimizer_steps=scalar(1),
            pair_space_size=scalar(1),
            sampled_pairs=scalar(1),
            seed=scalar(0),
            stored_row_index_count=scalar(2),
        )
    config_digest = _sha256(TEMPLATE / "config.json")
    request_path = tmp_path / "explosive-predict.json"
    _write_request(
        request_path,
        split_role="final",
        capabilities=(("features", "final_inputs", features_path),),
        request=_prediction_payload(
            config_digest,
            _sha256(checkpoint_path),
            expected_count=features.shape[0],
        ),
    )
    output = tmp_path / "explosive-output"

    _run(
        "predict",
        "--request",
        str(request_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(output),
        cwd=tmp_path,
        success=False,
    )
    assert not output.exists()
