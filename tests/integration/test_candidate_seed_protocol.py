from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
SEED = ROOT / "candidate_seed"
SOURCE = "1" * 64
TRAIN_DATA = "2" * 64
PREDICT_DATA = "3" * 64


def _load_seed_model_impl() -> Any:
    """Import the seed model module directly, the way a materialized candidate would."""

    spec = importlib.util.spec_from_file_location("seed_model_impl", SEED / "model_impl.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                "def train_model(features, targets, user_groups, config, seed):\n"
                "    return None\n\n"
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
    train_user_groups = np.array([10, 10, 20, 20], dtype="<i8")
    _write_npy(inputs / "train-features.npy", train_features)
    _write_npy(inputs / "train-targets.npy", train_targets)
    _write_npy(inputs / "train-user-groups.npy", train_user_groups)
    config_digest = _sha256(SEED / "config.json")
    train_request = tmp_path / "train-request.json"
    _write_request(
        train_request,
        approved=(
            ("features", "inputs/train-features.npy"),
            ("targets", "inputs/train-targets.npy"),
            ("user_groups", "inputs/train-user-groups.npy"),
        ),
        split_role="train",
        request={
            "protocol_schema_version": 1,
            "source_digest": SOURCE,
            "config_digest": config_digest,
            "data_digest": TRAIN_DATA,
            "split_token": "train-seed-fold",
            "seed": 7,
            "features_handle": "features",
            "targets_handle": "targets",
            "user_groups_handle": "user_groups",
        },
    )

    first_train = tmp_path / "train-a"
    second_train = tmp_path / "train-b"
    first_train.mkdir()
    _run("train", "--request", str(train_request), "--output", str(first_train), cwd=tmp_path)
    _run("train", "--request", str(train_request), "--output", str(second_train), cwd=tmp_path)

    expected_train = TrainExpectation(
        source_digest=SOURCE,
        config_digest=config_digest,
        data_digest=TRAIN_DATA,
        split_token="train-seed-fold",
        checkpoint_path="checkpoint/model.txt",
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
    first_prediction.mkdir()
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


def test_failing_training_diagnostics_does_not_discard_a_successful_training_run(
    tmp_path: Path,
) -> None:
    """Diagnostics are informational, so they must never destroy real evaluation evidence.

    In overnight-11 two of three candidates trained successfully and were then thrown away by an
    exception inside their own ``training_diagnostics``: one stored a non scalar under a scalar
    key, the other rejected its own checkpoint. Nothing downstream scores diagnostics.
    """

    work = tmp_path / "candidate"
    work.mkdir()
    (work / "config.json").write_bytes((SEED / "config.json").read_bytes())
    (work / "candidate.py").write_bytes((SEED / "candidate.py").read_bytes())
    model = (SEED / "model_impl.py").read_text(encoding="utf-8")
    # Reproduce the overnight-11 failure shape: training works, diagnostics raise.
    model += (
        "\n\ndef training_diagnostics(config, checkpoint):\n"
        "    raise TypeError('only 0-dimensional arrays can be converted to Python scalars')\n"
    )
    (work / "model_impl.py").write_text(model, encoding="utf-8")

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_npy(
        inputs / "train-features.npy",
        np.array([[-2.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [1.0, 1.0]], dtype="<f8"),
    )
    _write_npy(inputs / "train-targets.npy", np.array([0.0, 1.0, 0.0, 1.0], dtype="<f8"))
    _write_npy(inputs / "train-user-groups.npy", np.array([10, 10, 20, 20], dtype="<i8"))
    config_digest = _sha256(work / "config.json")
    request = tmp_path / "train-request.json"
    _write_request(
        request,
        approved=(
            ("features", "inputs/train-features.npy"),
            ("targets", "inputs/train-targets.npy"),
            ("user_groups", "inputs/train-user-groups.npy"),
        ),
        split_role="train",
        request={
            "protocol_schema_version": 1,
            "source_digest": SOURCE,
            "config_digest": config_digest,
            "data_digest": TRAIN_DATA,
            "split_token": "train-seed-fold",
            "seed": 7,
            "features_handle": "features",
            "targets_handle": "targets",
            "user_groups_handle": "user_groups",
        },
    )

    output = tmp_path / "train-out"
    completed = subprocess.run(
        [
            sys.executable,
            str(work / "candidate.py"),
            "train",
            "--request",
            str(request),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    validated = validate_train_outputs(
        output,
        TrainExpectation(
            source_digest=SOURCE,
            config_digest=config_digest,
            data_digest=TRAIN_DATA,
            split_token="train-seed-fold",
            checkpoint_path="checkpoint/model.txt",
        ),
    )
    assert validated.checkpoint_path.exists()
    result = json.loads((output / "candidate_result.json").read_text(encoding="utf-8"))
    diagnostics = result["diagnostics"]
    # The failure is reported in schema, and the model authored keys are absent.
    assert diagnostics["diagnostics_failed"] == 1.0
    assert "epochs" not in diagnostics
    assert "final_objective" not in diagnostics
    # Controller owned shape facts still describe the run that did happen.
    assert diagnostics["row_count"] == 4


def _identity_fixture(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A realistic 37 column matrix: 33 continuous columns then four identity code columns."""

    rng = np.random.default_rng(seed)
    per_user = 5
    columns = [np.full(rows, 0.3366)]
    for _ in range(9):
        exposure = rng.integers(0, 400, size=rows).astype(np.float64)
        positive = np.minimum(exposure, rng.integers(0, 200, size=rows)).astype(np.float64)
        columns.extend([exposure, positive, (positive + 10.0) / (exposure + 20.0)])
    duration = rng.gamma(2.0, 40.0, size=rows)
    columns.extend(
        [
            duration,
            np.log1p(duration),
            (duration >= 18.0).astype(np.float64),
            rng.integers(0, 14, size=rows).astype(np.float64),
            rng.integers(0, 15, size=rows).astype(np.float64),
        ]
    )
    for cardinality in (7538, 6482, 15, 6):
        columns.append(rng.integers(0, cardinality, size=rows).astype(np.float64))
    features = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)

    user_groups = np.repeat(np.arange(rows // per_user), per_user).astype(np.int64)
    targets = np.empty(rows, dtype=np.float64)
    for user in range(rows // per_user):
        block = slice(user * per_user, (user + 1) * per_user)
        labels = rng.integers(0, 2, size=per_user).astype(np.float64)
        if labels.sum() in (0.0, float(per_user)):
            labels[0] = 1.0 - labels[0]
        targets[block] = labels
    return features, targets, user_groups


def test_seed_fm_helper_keeps_its_two_accumulators_at_different_shapes() -> None:
    """Mixing the (N, rank) and (N,) accumulators is this project's most frequent defect.

    Three generated candidates across runs 11 and 12 raised a broadcast error on exactly this,
    so the provided helper is pinned rather than left to prose in the briefing.
    """

    seed_module = _load_seed_model_impl()
    features, _, _ = _identity_fixture(400, 3)
    codes = seed_module.categorical_codes(features, 4)

    assert len(codes) == 4
    assert all(code.dtype == np.int64 for code in codes)
    tables = [
        np.linspace(0.0, 1.0, seed_module.embedding_table_size(code) * 6).reshape(-1, 6)
        for code in codes
    ]
    scores = seed_module.fm_interaction_scores(tables, codes)

    assert scores.shape == (features.shape[0],)
    assert bool(np.isfinite(scores).all())
    assert float(scores.std()) > 0.0


def test_seed_fm_helper_clamps_identities_absent_from_training() -> None:
    seed_module = _load_seed_model_impl()
    features, _, _ = _identity_fixture(200, 5)
    codes = seed_module.categorical_codes(features, 4)
    tables = [np.ones((seed_module.embedding_table_size(code), 4)) for code in codes]
    # A validation identity beyond every training code must land on the spare row, not raise.
    unseen = [code.copy() for code in codes]
    unseen[0][:] = tables[0].shape[0] + 50

    scores = seed_module.fm_interaction_scores(tables, unseen)

    assert bool(np.isfinite(scores).all())


def test_seed_pair_sampler_stays_inside_each_user_and_respects_labels() -> None:
    """Within-user pair sampling crashed three of three candidates that attempted it."""

    seed_module = _load_seed_model_impl()
    generator = np.random.default_rng(11)
    for trial in range(8):
        _, targets, user_groups = _identity_fixture(1000, trial)
        positives, negatives = seed_module.within_user_pairs(targets, user_groups, generator, 2048)

        assert positives.shape == negatives.shape == (2048,)
        assert float(targets[positives].min()) == 1.0
        assert float(targets[negatives].max()) == 0.0
        assert bool((user_groups[positives] == user_groups[negatives]).all())


def test_seed_pair_sampler_rejects_a_split_with_no_eligible_user() -> None:
    seed_module = _load_seed_model_impl()
    targets = np.zeros(10, dtype=np.float64)
    user_groups = np.repeat(np.arange(2), 5).astype(np.int64)

    with pytest.raises(Exception, match="GAUC eligible"):
        seed_module.within_user_pairs(targets, user_groups, np.random.default_rng(0), 4)
