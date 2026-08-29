from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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
SEED = ROOT / "candidate_seed"
SOURCE = "1" * 64
TRAIN_DATA = "2" * 64
PREDICT_DATA = "3" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npy(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)


def _write_request(
    path: Path,
    *,
    approved: tuple[tuple[str, str], ...],
    request: dict[str, object],
    split_role: str,
) -> None:
    value = {
        "schema_version": 1,
        "execution_id": f"seed-{split_role}",
        "split_role": split_role,
        "source_snapshot_sha256": "4" * 64,
        "approved_inputs": [
            {
                "name": name,
                "role": "train_targets" if name == "targets" else f"{split_role}_inputs",
                "workspace_path": relative,
                "artifact": {"sha256": "5" * 64, "size_bytes": 1, "kind": "input"},
            }
            for name, relative in approved
        ],
        "budgets": {"output_limit_bytes": 1024 * 1024, "temp_limit_bytes": 1024},
        "request": request,
    }
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="ascii",
    )


def _run(*arguments: str, cwd: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SEED / "candidate.py"), *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_seed_is_a_material_generated_source_candidate(tmp_path: Path) -> None:
    source = (SEED / "candidate.py").read_text(encoding="utf-8")
    model_source = (SEED / "model_impl.py").read_text(encoding="utf-8")
    config = (SEED / "config.json").read_text(encoding="utf-8")
    parent = ParentSnapshot(
        candidate_id="minimal-parent",
        files=(
            ParentSourceFile.create(
                "candidate.py",
                "def train_model(features, targets, config):\n    return None\n\n"
                "def predict_scores(features, checkpoint):\n    return []\n",
            ),
            ParentSourceFile.create("config.json", config),
        ),
    )
    package = GeneratedPackage(
        request_id="seed-materiality",
        response_id="seed-source",
        files=(
            GeneratedFile("candidate.py", source),
            GeneratedFile("model_impl.py", model_source),
            GeneratedFile("config.json", config),
        ),
        material_change_summary="Own deterministic logistic training and prediction mechanics.",
        material_symbols=("train_model", "predict_scores"),
    )

    child = materialize_candidate(parent, package, tmp_path / "materialized-seed")
    evidence = require_material_executable_change(parent, child)

    assert evidence.changed_symbols == (
        "model_impl.py:predict_scores",
        "model_impl.py:train_model",
    )
    assert evidence.reachable_python_files == ("candidate.py", "model_impl.py")


def test_seed_keeps_mutable_model_code_out_of_stable_protocol_entrypoint() -> None:
    entrypoint = (SEED / "candidate.py").read_text(encoding="utf-8")
    model = (SEED / "model_impl.py").read_text(encoding="utf-8")

    assert "from model_impl import" in entrypoint
    assert "def train_model(" not in entrypoint
    assert "def predict_scores(" not in entrypoint
    assert "def train_model(" in model
    assert "def predict_scores(" in model
    assert len(model.encode("utf-8")) < len(entrypoint.encode("utf-8"))


def test_seed_train_predict_commands_are_deterministic_and_protocol_valid(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    train_features = np.array([[-2.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [1.0, 1.0]], dtype="<f8")
    train_targets = np.array([0.0, 1.0, 0.0, 1.0], dtype="<f8")
    _write_npy(inputs / "train-features.npy", train_features)
    _write_npy(inputs / "train-targets.npy", train_targets)
    config_digest = _sha256(SEED / "config.json")
    train_request = tmp_path / "train-request.json"
    _write_request(
        train_request,
        approved=(
            ("features", "inputs/train-features.npy"),
            ("targets", "inputs/train-targets.npy"),
        ),
        split_role="train",
        request={
            "protocol_schema_version": 1,
            "source_digest": SOURCE,
            "config_digest": config_digest,
            "data_digest": TRAIN_DATA,
            "split_token": "train-seed-fold",
            "features_handle": "features",
            "targets_handle": "targets",
        },
    )

    first_train = tmp_path / "train-a"
    second_train = tmp_path / "train-b"
    _run("train", "--request", str(train_request), "--output", str(first_train), cwd=tmp_path)
    _run("train", "--request", str(train_request), "--output", str(second_train), cwd=tmp_path)

    expected_train = TrainExpectation(
        source_digest=SOURCE,
        config_digest=config_digest,
        data_digest=TRAIN_DATA,
        split_token="train-seed-fold",
        checkpoint_path="checkpoint/model.npz",
    )
    first = validate_train_outputs(first_train, expected_train)
    second = validate_train_outputs(second_train, expected_train)
    assert first.checkpoint_digest == second.checkpoint_digest
    assert first.checkpoint_path.read_bytes() == second.checkpoint_path.read_bytes()
    assert (first_train / "candidate_result.json").read_bytes() == (
        second_train / "candidate_result.json"
    ).read_bytes()

    predict_request = tmp_path / "predict-request.json"
    _write_request(
        predict_request,
        approved=(("features", "inputs/train-features.npy"),),
        split_role="inner_valid",
        request={
            "protocol_schema_version": 1,
            "source_digest": SOURCE,
            "config_digest": config_digest,
            "data_digest": PREDICT_DATA,
            "split_token": "inner-valid-seed-fold",
            "features_handle": "features",
            "expected_count": 4,
            "checkpoint_digest": first.checkpoint_digest,
        },
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
        source_digest=SOURCE,
        config_digest=config_digest,
        data_digest=PREDICT_DATA,
        split_token="inner-valid-seed-fold",
        checkpoint_digest=first.checkpoint_digest,
        expected_count=4,
    )
    predicted = validate_prediction_outputs(first_prediction, expected_prediction)
    replayed = validate_prediction_outputs(second_prediction, expected_prediction)
    assert predicted.scores.tobytes() == replayed.scores.tobytes()
    assert predicted.scores_sha256 == replayed.scores_sha256
    assert predicted.scores[1] > predicted.scores[3] > predicted.scores[2] > predicted.scores[0]
    assert (first_prediction / "prediction_result.json").read_bytes() == (
        second_prediction / "prediction_result.json"
    ).read_bytes()
