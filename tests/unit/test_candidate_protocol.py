from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.candidate_api.protocol import (
    CandidateProtocolError,
    PredictionExpectation,
    TrainExpectation,
    parse_prediction_result_json,
    parse_train_result_json,
    validate_prediction_outputs,
    validate_train_outputs,
)

SOURCE = "1" * 64
CONFIG = "2" * 64
TRAIN_DATA = "3" * 64
PREDICT_DATA = "4" * 64
CHECKPOINT = "5" * 64
SCORES = "6" * 64


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _train_manifest(
    *,
    checkpoint_digest: str = CHECKPOINT,
    diagnostics: object | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "train",
        "source_digest": SOURCE,
        "config_digest": CONFIG,
        "data_digest": TRAIN_DATA,
        "split_token": "inner-train-a",
        "checkpoint_digest": checkpoint_digest,
        "artifacts": [
            {
                "path": "checkpoint/model.npz",
                "sha256": checkpoint_digest,
                "size_bytes": 17,
            }
        ],
        "diagnostics": {} if diagnostics is None else diagnostics,
    }


def _prediction_manifest(
    *,
    expected_count: int = 3,
    dtype: str = "<f8",
    scores_sha256: str = SCORES,
    diagnostics: object | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "prediction",
        "source_digest": SOURCE,
        "config_digest": CONFIG,
        "data_digest": PREDICT_DATA,
        "split_token": "inner-valid-a",
        "checkpoint_digest": CHECKPOINT,
        "expected_count": expected_count,
        "dtype": dtype,
        "scores_path": "scores.npy",
        "scores_sha256": scores_sha256,
        "diagnostics": {} if diagnostics is None else diagnostics,
    }


def test_train_parser_returns_a_strict_typed_manifest() -> None:
    manifest = parse_train_result_json(_json(_train_manifest()))

    assert manifest.source_digest == SOURCE
    assert manifest.checkpoint_digest == CHECKPOINT
    assert manifest.artifacts[0].path == "checkpoint/model.npz"
    assert manifest.to_wire() == _train_manifest()


@pytest.mark.parametrize(
    "payload,match",
    (
        (b'{"schema_version":1,"schema_version":1}', "duplicate"),
        (_json({**_train_manifest(), "unexpected": True}), "exact keys"),
        (_json({key: value for key, value in _train_manifest().items() if key != "kind"}), "keys"),
        (_json({**_train_manifest(), "schema_version": True}), "schema_version"),
        (_json({**_train_manifest(), "source_digest": "A" * 64}), "source_digest"),
        (_json({**_train_manifest(), "split_token": ""}), "split_token"),
        (_json(_train_manifest(diagnostics={"nested": {"GAUC": 0.99}})), "official metric"),
        (_json(_train_manifest(diagnostics={"ndcg@5": 0.99})), "official metric"),
        (_json(_train_manifest(diagnostics={"primary": 0.99})), "official metric"),
        (_json(_train_manifest(diagnostics={"metrics": {"loss": 0.1}})), "official metric"),
    ),
)
def test_train_parser_rejects_ambiguous_or_metric_bearing_json(payload: bytes, match: str) -> None:
    with pytest.raises(CandidateProtocolError, match=match):
        parse_train_result_json(payload)


def test_parser_rejects_nonfinite_json_constants() -> None:
    payload = _json(_train_manifest()).replace(b'"diagnostics":{}', b'"diagnostics":{"loss":NaN}')

    with pytest.raises(CandidateProtocolError, match="non-finite"):
        parse_train_result_json(payload)


def test_train_output_validation_binds_manifest_to_exact_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train-output"
    checkpoint = output / "checkpoint" / "model.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = _train_manifest(checkpoint_digest=digest)
    manifest["artifacts"] = [{"path": "checkpoint/model.npz", "sha256": digest, "size_bytes": 18}]
    (output / "candidate_result.json").write_bytes(_json(manifest))

    result = validate_train_outputs(
        output,
        TrainExpectation(
            source_digest=SOURCE,
            config_digest=CONFIG,
            data_digest=TRAIN_DATA,
            split_token="inner-train-a",
            checkpoint_path="checkpoint/model.npz",
        ),
    )

    assert result.checkpoint_path == checkpoint
    assert result.checkpoint_digest == digest
    assert result.checkpoint_size_bytes == 18
    assert result.manifest.to_wire() == manifest


@pytest.mark.parametrize(
    "field,bad_value,match",
    (
        ("source_digest", "a" * 64, "source_digest"),
        ("config_digest", "a" * 64, "config_digest"),
        ("data_digest", "a" * 64, "data_digest"),
        ("split_token", "wrong-split", "split_token"),
    ),
)
def test_train_output_validation_rejects_identity_mismatches(
    tmp_path: Path, field: str, bad_value: object, match: str
) -> None:
    output = tmp_path / "train-output"
    checkpoint = output / "checkpoint" / "model.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = _train_manifest(checkpoint_digest=digest)
    manifest["artifacts"] = [{"path": "checkpoint/model.npz", "sha256": digest, "size_bytes": 18}]
    manifest[field] = bad_value
    (output / "candidate_result.json").write_bytes(_json(manifest))

    with pytest.raises(CandidateProtocolError, match=match):
        validate_train_outputs(
            output,
            TrainExpectation(
                source_digest=SOURCE,
                config_digest=CONFIG,
                data_digest=TRAIN_DATA,
                split_token="inner-train-a",
                checkpoint_path="checkpoint/model.npz",
            ),
        )


def test_train_output_validation_rejects_wrong_or_extra_output_paths(tmp_path: Path) -> None:
    output = tmp_path / "train-output"
    checkpoint = output / "checkpoint" / "model.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = _train_manifest(checkpoint_digest=digest)
    manifest["artifacts"] = [{"path": "checkpoint/other.npz", "sha256": digest, "size_bytes": 18}]
    (output / "candidate_result.json").write_bytes(_json(manifest))

    expectation = TrainExpectation(
        source_digest=SOURCE,
        config_digest=CONFIG,
        data_digest=TRAIN_DATA,
        split_token="inner-train-a",
        checkpoint_path="checkpoint/model.npz",
    )
    with pytest.raises(CandidateProtocolError, match="artifact paths"):
        validate_train_outputs(output, expectation)

    manifest["artifacts"] = [{"path": "checkpoint/model.npz", "sha256": digest, "size_bytes": 18}]
    (output / "candidate_result.json").write_bytes(_json(manifest))
    (output / "undeclared.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(CandidateProtocolError, match="inventory"):
        validate_train_outputs(output, expectation)


def test_prediction_parser_returns_a_strict_typed_manifest() -> None:
    manifest = parse_prediction_result_json(_json(_prediction_manifest()))

    assert manifest.expected_count == 3
    assert manifest.dtype == "<f8"
    assert manifest.split_token == "inner-valid-a"
    assert manifest.to_wire() == _prediction_manifest()


def _write_prediction_output(
    output: Path,
    scores: np.ndarray,
    **overrides: object,
) -> dict[str, object]:
    output.mkdir()
    with (output / "scores.npy").open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)
    digest = hashlib.sha256((output / "scores.npy").read_bytes()).hexdigest()
    manifest = _prediction_manifest(scores_sha256=digest)
    manifest.update(overrides)
    (output / "prediction_result.json").write_bytes(_json(manifest))
    return manifest


def _prediction_expectation() -> PredictionExpectation:
    return PredictionExpectation(
        source_digest=SOURCE,
        config_digest=CONFIG,
        data_digest=PREDICT_DATA,
        split_token="inner-valid-a",
        checkpoint_digest=CHECKPOINT,
        expected_count=3,
        dtype="<f8",
        scores_path="scores.npy",
    )


def test_prediction_output_validation_returns_immutable_exact_float64_scores(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prediction-output"
    manifest = _write_prediction_output(output, np.array([0.125, -0.0, 2.5], dtype="<f8"))

    result = validate_prediction_outputs(output, _prediction_expectation())

    assert result.scores_path == output / "scores.npy"
    assert result.scores.dtype.str == "<f8"
    assert result.scores.shape == (3,)
    assert result.scores.tobytes() == np.array([0.125, -0.0, 2.5], dtype="<f8").tobytes()
    assert result.scores.flags.writeable is False
    assert result.manifest.to_wire() == manifest


@pytest.mark.parametrize(
    "scores,overrides,match",
    (
        (np.array([0.1, 0.2], dtype="<f8"), {}, "shape"),
        (np.array([0.1, np.nan, 0.2], dtype="<f8"), {}, "finite"),
        (np.array([0.1, 0.2, 0.3], dtype="<f4"), {"dtype": "<f4"}, "dtype"),
        (np.array([[0.1, 0.2, 0.3]], dtype="<f8"), {}, "shape"),
        (np.array([0.1, 0.2, 0.3], dtype="<f8"), {"expected_count": 2}, "expected_count"),
        (np.array([0.1, 0.2, 0.3], dtype="<f8"), {"split_token": "wrong"}, "split_token"),
        (
            np.array([0.1, 0.2, 0.3], dtype="<f8"),
            {"checkpoint_digest": "a" * 64},
            "checkpoint_digest",
        ),
        (
            np.array([0.1, 0.2, 0.3], dtype="<f8"),
            {"scores_path": "predictions.npy"},
            "scores_path",
        ),
    ),
)
def test_prediction_output_validation_rejects_shape_dtype_finiteness_and_identity_errors(
    tmp_path: Path,
    scores: np.ndarray,
    overrides: dict[str, object],
    match: str,
) -> None:
    output = tmp_path / "prediction-output"
    _write_prediction_output(output, scores, **overrides)

    with pytest.raises(CandidateProtocolError, match=match):
        validate_prediction_outputs(output, _prediction_expectation())


def test_prediction_output_validation_rejects_metric_claims_and_digest_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prediction-output"
    _write_prediction_output(
        output,
        np.array([0.1, 0.2, 0.3], dtype="<f8"),
        diagnostics={"validation_metric": 0.999},
    )

    with pytest.raises(CandidateProtocolError, match="official metric"):
        validate_prediction_outputs(output, _prediction_expectation())

    (output / "prediction_result.json").write_bytes(
        _json(_prediction_manifest(scores_sha256="a" * 64))
    )
    with pytest.raises(CandidateProtocolError, match="scores_sha256"):
        validate_prediction_outputs(output, _prediction_expectation())


def test_prediction_output_validation_rejects_linked_scores(tmp_path: Path) -> None:
    output = tmp_path / "prediction-output"
    output.mkdir()
    external = tmp_path / "outside.npy"
    with external.open("xb") as handle:
        np.save(handle, np.array([0.1, 0.2, 0.3], dtype="<f8"), allow_pickle=False)
    (output / "scores.npy").symlink_to(external)
    digest = hashlib.sha256(external.read_bytes()).hexdigest()
    (output / "prediction_result.json").write_bytes(
        _json(_prediction_manifest(scores_sha256=digest))
    )

    with pytest.raises(CandidateProtocolError, match="symlink"):
        validate_prediction_outputs(output, _prediction_expectation())
