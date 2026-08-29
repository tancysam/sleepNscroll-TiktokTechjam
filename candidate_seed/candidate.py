"""Deterministic, self-contained generated-candidate seed.

This file intentionally owns its feature normalization, logistic objective, fixed-step optimizer,
checkpoint representation, and inference mechanics.  It does not import trusted controller or
scorer code and it never computes organizer metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np

SCHEMA_VERSION = 1
SCORES_DTYPE = "<f8"
MAX_JSON_BYTES = 256 * 1024
CONFIG_KEYS = {
    "candidate_family",
    "epochs",
    "l2",
    "learning_rate",
    "logit_clip",
    "schema_version",
}
TRAIN_KEYS = {
    "config_digest",
    "data_digest",
    "features_handle",
    "protocol_schema_version",
    "seed",
    "source_digest",
    "split_token",
    "targets_handle",
    "user_groups_handle",
}
PREDICT_KEYS = {
    "checkpoint_digest",
    "config_digest",
    "data_digest",
    "expected_count",
    "features_handle",
    "protocol_schema_version",
    "source_digest",
    "split_token",
}


class CandidateInputError(ValueError):
    """The candidate-visible request or capability is malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise CandidateInputError(f"JSON input exceeds {MAX_JSON_BYTES} bytes")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateInputError("JSON input must contain one object")
    return value


def _require_exact_keys(value: dict[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise CandidateInputError(f"{name} keys do not match the protocol")


def _require_digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CandidateInputError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_seed(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise CandidateInputError("seed must be an unsigned 32-bit integer")
    return value


def _require_token(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > 256
        or not value.isascii()
        or any(character.isspace() for character in value)
    ):
        raise CandidateInputError("split_token is invalid")
    return value


def _relative_path(value: object) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise CandidateInputError("capability path must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CandidateInputError("capability path must be a canonical relative POSIX path")
    return path


def _workspace_request(path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    document = _read_json(path)
    request = document.get("request")
    approved = document.get("approved_inputs")
    if not isinstance(request, dict) or not isinstance(approved, list):
        raise CandidateInputError("request.json is not a candidate workspace request")
    capabilities: dict[str, Path] = {}
    for item in approved:
        if not isinstance(item, dict):
            raise CandidateInputError("approved input declaration must be an object")
        name = item.get("name")
        if type(name) is not str or not name or name in capabilities:
            raise CandidateInputError("approved input names must be unique non-empty strings")
        relative = _relative_path(item.get("workspace_path"))
        capabilities[name] = path.parent.joinpath(*relative.parts)
    return request, capabilities


def _load_numeric_array(path: Path, name: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        if hasattr(value, "close"):
            value.close()
        raise CandidateInputError(f"{name} must contain one NumPy array")
    if value.dtype.kind not in "biuf" or not bool(np.isfinite(value).all()):
        raise CandidateInputError(f"{name} must contain finite numeric values")
    return np.ascontiguousarray(value, dtype=np.dtype(SCORES_DTYPE))


def _load_config(expected_digest: str) -> dict[str, object]:
    path = Path(__file__).with_name("config.json")
    if _sha256(path) != expected_digest:
        raise CandidateInputError("config_digest does not identify candidate config.json")
    config = _read_json(path)
    _require_exact_keys(config, CONFIG_KEYS, "config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("config schema_version must be 1")
    if config["candidate_family"] != "deterministic_logistic_seed":
        raise CandidateInputError("candidate_family is invalid")
    _config_epochs(config)
    for name in ("l2", "learning_rate", "logit_clip"):
        _config_float(config, name)
    return config


def _config_epochs(config: dict[str, object]) -> int:
    value = config["epochs"]
    if type(value) is not int or not 1 <= value <= 10_000:
        raise CandidateInputError("epochs is invalid")
    return value


def _config_float(config: dict[str, object], name: str) -> float:
    value = config[name]
    if type(value) not in {int, float}:
        raise CandidateInputError(f"{name} is invalid")
    numeric = float(cast(int | float, value))
    if not 0 < numeric < 1_000:
        raise CandidateInputError(f"{name} is invalid")
    return numeric


def _sigmoid(logits: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(logits, -clip, clip)
    result = np.reciprocal(1.0 + np.exp(-bounded), dtype=np.float64)
    return cast(np.ndarray, result)


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit a fixed-step standardized logistic model with deterministic full-batch updates.

    ``user_groups`` is the trusted per-row user identity for this split. This pointwise
    objective does not consume it, but the benchmark ranks within a user, so any ranking
    objective must group by this array rather than by a feature column. ``seed`` is supplied
    per request by the controller; this objective is deterministic and records it for replay
    identity.
    """

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise CandidateInputError("training features must have non-empty shape (N, D)")
    if targets.shape != (features.shape[0],):
        raise CandidateInputError("training targets must have shape (N,)")
    if user_groups.shape != (features.shape[0],):
        raise CandidateInputError("training user_groups must have shape (N,)")
    if not bool(np.logical_or(targets == 0.0, targets == 1.0).all()):
        raise CandidateInputError("training targets must be binary")
    _require_seed(seed)
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    normalized = np.ascontiguousarray((features - mean) / scale, dtype=np.float64)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = np.float64(0.0)
    epochs = _config_epochs(config)
    learning_rate = np.float64(_config_float(config, "learning_rate"))
    l2 = np.float64(_config_float(config, "l2"))
    clip = _config_float(config, "logit_clip")
    row_count = np.float64(features.shape[0])
    for _ in range(epochs):
        probabilities = _sigmoid(normalized @ weights + bias, clip)
        error = probabilities - targets
        weights -= learning_rate * ((normalized.T @ error) / row_count + l2 * weights)
        bias -= learning_rate * np.mean(error, dtype=np.float64)
    probabilities = _sigmoid(normalized @ weights + bias, clip)
    epsilon = np.float64(1e-12)
    objective = -np.mean(
        targets * np.log(np.clip(probabilities, epsilon, 1.0))
        + (1.0 - targets) * np.log(np.clip(1.0 - probabilities, epsilon, 1.0)),
        dtype=np.float64,
    ) + np.float64(0.5) * l2 * np.dot(weights, weights)
    if not bool(np.isfinite(objective)):
        raise CandidateInputError("training objective became non-finite")
    return {
        "bias": np.asarray(bias, dtype=np.float64),
        "feature_mean": np.ascontiguousarray(mean, dtype=np.float64),
        "feature_scale": np.ascontiguousarray(scale, dtype=np.float64),
        "final_objective": np.asarray(objective, dtype=np.float64),
        "seed": np.asarray(float(seed), dtype=np.float64),
        "weights": np.ascontiguousarray(weights, dtype=np.float64),
    }


def predict_scores(features: np.ndarray, checkpoint: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the owned normalization and logistic interaction from a verified checkpoint."""

    expected = {
        "bias",
        "feature_mean",
        "feature_scale",
        "final_objective",
        "seed",
        "weights",
    }
    if set(checkpoint) != expected:
        raise CandidateInputError("checkpoint inventory is invalid")
    mean = checkpoint["feature_mean"]
    scale = checkpoint["feature_scale"]
    weights = checkpoint["weights"]
    bias = checkpoint["bias"]
    if features.ndim != 2 or features.shape[1:] != weights.shape:
        raise CandidateInputError("prediction feature shape does not match the checkpoint")
    if mean.shape != weights.shape or scale.shape != weights.shape or bias.shape != ():
        raise CandidateInputError("checkpoint array shapes are invalid")
    if not all(array.dtype == np.dtype(SCORES_DTYPE) for array in checkpoint.values()):
        raise CandidateInputError("checkpoint arrays must use float64")
    if not all(bool(np.isfinite(array).all()) for array in checkpoint.values()):
        raise CandidateInputError("checkpoint arrays must be finite")
    logits = ((features - mean) / scale) @ weights + bias
    return np.ascontiguousarray(_sigmoid(logits, 40.0), dtype=np.dtype(SCORES_DTYPE))


def _write_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    with path.open("xb") as handle:
        handle.write(payload)


def _train(request_path: Path, output: Path) -> None:
    request, capabilities = _workspace_request(request_path)
    _require_exact_keys(request, TRAIN_KEYS, "training request")
    if request["protocol_schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("protocol_schema_version must be 1")
    source_digest = _require_digest(request["source_digest"], "source_digest")
    config_digest = _require_digest(request["config_digest"], "config_digest")
    data_digest = _require_digest(request["data_digest"], "data_digest")
    split_token = _require_token(request["split_token"])
    seed = _require_seed(request["seed"])
    features_handle = request["features_handle"]
    targets_handle = request["targets_handle"]
    user_groups_handle = request["user_groups_handle"]
    if (
        type(features_handle) is not str
        or type(targets_handle) is not str
        or type(user_groups_handle) is not str
    ):
        raise CandidateInputError("training capability handles must be strings")
    try:
        feature_path = capabilities[features_handle]
        target_path = capabilities[targets_handle]
        user_groups_path = capabilities[user_groups_handle]
    except KeyError as exc:
        raise CandidateInputError("training capability handle is not approved") from exc
    features = _load_numeric_array(feature_path, "training features")
    targets = _load_numeric_array(target_path, "training targets")
    user_groups = _load_numeric_array(user_groups_path, "training user groups")
    config = _load_config(config_digest)
    checkpoint = train_model(features, targets, user_groups, config, seed)

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoint"
    checkpoint_dir.mkdir()
    # The trusted protocol pins this exact path
    # (GeneratedCandidateIdentity.checkpoint_path); it is not a free choice.
    # The bytes remain a NumPy archive.
    checkpoint_path = checkpoint_dir / "model.txt"
    with checkpoint_path.open("xb") as handle:
        np.savez(
            handle,
            bias=checkpoint["bias"],
            feature_mean=checkpoint["feature_mean"],
            feature_scale=checkpoint["feature_scale"],
            final_objective=checkpoint["final_objective"],
            seed=checkpoint["seed"],
            weights=checkpoint["weights"],
        )
    checkpoint_digest = _sha256(checkpoint_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "train",
        "source_digest": source_digest,
        "config_digest": config_digest,
        "data_digest": data_digest,
        "split_token": split_token,
        "checkpoint_digest": checkpoint_digest,
        "artifacts": [
            {
                "path": "checkpoint/model.txt",
                "sha256": checkpoint_digest,
                "size_bytes": checkpoint_path.stat().st_size,
            }
        ],
        "diagnostics": {
            "epochs": _config_epochs(config),
            "feature_count": int(features.shape[1]),
            "final_objective": float(checkpoint["final_objective"]),
            "row_count": int(features.shape[0]),
            "seed": seed,
            "user_count": int(np.unique(user_groups).size),
        },
    }
    _write_json(output / "candidate_result.json", result)


def _load_checkpoint(path: Path, expected_digest: str) -> dict[str, np.ndarray]:
    if _sha256(path) != expected_digest:
        raise CandidateInputError("checkpoint_digest does not identify checkpoint bytes")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _predict(request_path: Path, checkpoint_path: Path, output: Path) -> None:
    request, capabilities = _workspace_request(request_path)
    _require_exact_keys(request, PREDICT_KEYS, "prediction request")
    if request["protocol_schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("protocol_schema_version must be 1")
    source_digest = _require_digest(request["source_digest"], "source_digest")
    config_digest = _require_digest(request["config_digest"], "config_digest")
    _load_config(config_digest)
    data_digest = _require_digest(request["data_digest"], "data_digest")
    split_token = _require_token(request["split_token"])
    checkpoint_digest = _require_digest(request["checkpoint_digest"], "checkpoint_digest")
    expected_count = request["expected_count"]
    if type(expected_count) is not int or expected_count <= 0:
        raise CandidateInputError("expected_count must be a positive integer")
    features_handle = request["features_handle"]
    if type(features_handle) is not str or features_handle not in capabilities:
        raise CandidateInputError("prediction feature handle is not approved")
    features = _load_numeric_array(capabilities[features_handle], "prediction features")
    if features.ndim != 2 or features.shape[0] != expected_count:
        raise CandidateInputError("prediction features do not match expected_count")
    checkpoint = _load_checkpoint(checkpoint_path, checkpoint_digest)
    scores = predict_scores(features, checkpoint)

    output.mkdir(parents=True, exist_ok=False)
    scores_path = output / "scores.npy"
    with scores_path.open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)
    scores_digest = _sha256(scores_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "prediction",
        "source_digest": source_digest,
        "config_digest": config_digest,
        "data_digest": data_digest,
        "split_token": split_token,
        "checkpoint_digest": checkpoint_digest,
        "expected_count": expected_count,
        "dtype": SCORES_DTYPE,
        "scores_path": "scores.npy",
        "scores_sha256": scores_digest,
        "diagnostics": {
            "feature_count": int(features.shape[1]),
            "row_count": int(features.shape[0]),
        },
    }
    _write_json(output / "prediction_result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic KuaiRand candidate seed")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--request", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--request", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "train":
        _train(arguments.request, arguments.output)
    else:
        _predict(arguments.request, arguments.checkpoint, arguments.output)


if __name__ == "__main__":
    main()
