from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.candidate_api.protocol import (
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
TEMPLATE = ROOT / "candidate_templates" / "lambdarank"
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
        "execution_id": f"lambdarank-{split_role}",
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
        timeout=60,
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


def _tiny_capabilities(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    steps = np.repeat(np.arange(8, dtype=np.float64), 6)
    user_groups = np.tile(np.array([20, 10, 30, 40, 50, 60], dtype=np.int64), 8)
    targets = (steps >= 4).astype(np.int8)
    features = np.column_stack(
        (
            steps,
            np.sin(steps),
            (user_groups % 7).astype(np.float64),
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
    return features_path, targets_path, groups_path, features


def test_template_is_a_material_generated_source_candidate(tmp_path: Path) -> None:
    source = (TEMPLATE / "candidate.py").read_text(encoding="utf-8")
    config = (TEMPLATE / "config.json").read_text(encoding="utf-8")
    parent = ParentSnapshot(
        candidate_id="minimal-tree-parent",
        files=(
            ParentSourceFile.create(
                "candidate.py",
                "def train_model(features, targets, user_groups, config, seed):\n"
                "    return None\n\n"
                "def predict_scores(features, checkpoint_text):\n"
                "    return []\n",
            ),
            ParentSourceFile.create("config.json", config),
        ),
    )
    package = GeneratedPackage(
        request_id="generated-tree-materiality",
        response_id="generated-tree-source",
        files=(
            GeneratedFile("candidate.py", source),
            GeneratedFile("config.json", config),
        ),
        material_change_summary="Own stable-grouped deterministic LambdaRank training.",
        material_symbols=("train_model", "predict_scores"),
    )

    child = materialize_candidate(parent, package, tmp_path / "materialized-tree")
    evidence = require_material_executable_change(parent, child)

    assert evidence.changed_symbols == (
        "candidate.py:predict_scores",
        "candidate.py:train_model",
    )
    assert evidence.reachable_python_files == ("candidate.py",)


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="the optional pinned research-tree dependency is not installed",
)
def test_literal_train_predict_commands_are_protocol_valid_and_exactly_replayable(
    tmp_path: Path,
) -> None:
    features_path, targets_path, groups_path, features = _tiny_capabilities(tmp_path)
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
    assert first.manifest.diagnostics["seed"] == 7
    assert first.manifest.diagnostics["tree_count_policy"] == "frozen_train_derived"

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
    assert float(predicted.scores[24:].mean()) > float(predicted.scores[:24].mean())
    assert (first_prediction / "prediction_result.json").read_bytes() == (
        second_prediction / "prediction_result.json"
    ).read_bytes()

    wrong_checkpoint_request = tmp_path / "wrong-checkpoint-request.json"
    _write_request(
        wrong_checkpoint_request,
        split_role="inner_valid",
        capabilities=(("features", "inner_valid_inputs", features_path),),
        request=_prediction_payload(
            config_digest,
            "f" * 64,
            expected_count=features.shape[0],
        ),
    )
    _run(
        "predict",
        "--request",
        str(wrong_checkpoint_request),
        "--checkpoint",
        str(first.checkpoint_path),
        "--output",
        str(tmp_path / "wrong-checkpoint-output"),
        cwd=tmp_path,
        success=False,
    )


@pytest.mark.parametrize(
    ("case", "split_role", "payload_mutation"),
    (
        ("extra-request-key", "inner_train", {"unexpected": True}),
        ("final-training", "final", {}),
        ("non-uint32-seed", "inner_train", {"seed": -1}),
    ),
)
def test_train_command_rejects_malformed_and_final_like_requests(
    tmp_path: Path,
    case: str,
    split_role: str,
    payload_mutation: dict[str, object],
) -> None:
    features_path, targets_path, groups_path, _ = _tiny_capabilities(tmp_path)
    config_digest = _sha256(TEMPLATE / "config.json")
    payload = _training_payload(config_digest)
    payload.update(payload_mutation)
    request_path = tmp_path / f"{case}.json"
    _write_request(
        request_path,
        split_role=split_role,
        capabilities=(
            ("features", "train_inputs", features_path),
            ("targets", "train_targets", targets_path),
            ("user_groups", "train_inputs", groups_path),
        ),
        request=payload,
    )

    output = tmp_path / f"{case}-output"
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


@pytest.mark.parametrize("bad_groups", (np.array([1, 2]), np.array([1.0, np.nan] * 24)))
def test_train_command_rejects_misaligned_or_nonfinite_groups(
    tmp_path: Path,
    bad_groups: np.ndarray,
) -> None:
    features_path, targets_path, groups_path, _ = _tiny_capabilities(tmp_path)
    groups_path.unlink()
    _write_npy(groups_path, bad_groups)
    config_digest = _sha256(TEMPLATE / "config.json")
    request_path = tmp_path / "bad-groups.json"
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

    output = tmp_path / "bad-groups-output"
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


def test_predict_request_cannot_smuggle_target_or_group_handles(tmp_path: Path) -> None:
    features_path, _, _, features = _tiny_capabilities(tmp_path)
    config_digest = _sha256(TEMPLATE / "config.json")
    payload = _prediction_payload(config_digest, "a" * 64, expected_count=features.shape[0])
    payload["targets_handle"] = "features"
    request_path = tmp_path / "prediction-with-target.json"
    _write_request(
        request_path,
        split_role="outer_valid",
        capabilities=(("features", "outer_valid_inputs", features_path),),
        request=payload,
    )

    output = tmp_path / "prediction-with-target-output"
    _run(
        "predict",
        "--request",
        str(request_path),
        "--checkpoint",
        str(tmp_path / "unread-checkpoint.txt"),
        "--output",
        str(output),
        cwd=tmp_path,
        success=False,
    )
    assert not output.exists()
