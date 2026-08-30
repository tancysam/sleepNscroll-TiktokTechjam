"""Deterministic, self-contained generated-candidate seed.

This file intentionally owns its feature normalization, logistic objective, fixed-step optimizer,
checkpoint representation, and inference mechanics.  It does not import trusted controller or
scorer code and it never computes organizer metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import numpy as np
from model_impl import predict_scores, train_model, training_diagnostics, validate_config

SCHEMA_VERSION = 1
SCORES_DTYPE = "<f8"
MAX_JSON_BYTES = 256 * 1024
MAX_CHECKPOINT_ARRAYS = 64
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
CHECKPOINT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
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


def _require_uint32(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise CandidateInputError(f"{name} must be a uint32-compatible integer")
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


def _load_user_groups(path: Path, expected: int) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise CandidateInputError("training user groups must be one safe NumPy array") from exc
    if not isinstance(value, np.ndarray):
        if hasattr(value, "close"):
            value.close()
        raise CandidateInputError("training user groups must contain one NumPy array")
    if value.shape != (expected,):
        raise CandidateInputError(f"training user groups must have shape ({expected},)")
    if value.dtype.kind not in "iuf":
        raise CandidateInputError("training user groups must use a non-boolean numeric dtype")
    try:
        finite = bool(np.isfinite(value).all())
    except TypeError as exc:
        raise CandidateInputError(
            "training user groups must contain finite numeric values"
        ) from exc
    if not finite:
        raise CandidateInputError("training user groups must contain finite numeric values")
    return np.ascontiguousarray(value)


def _load_config(expected_digest: str) -> dict[str, object]:
    path = Path(__file__).with_name("config.json")
    if _sha256(path) != expected_digest:
        raise CandidateInputError("config_digest does not identify candidate config.json")
    config = _read_json(path)
    validate_config(config)
    return config


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


def _prepare_output(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise CandidateInputError("output must be a real directory")
        if tuple(path.iterdir()):
            raise CandidateInputError("output directory must be empty")
        return
    path.mkdir(parents=True, exist_ok=False)


def _validate_checkpoint(value: object) -> dict[str, np.ndarray]:
    if type(value) is not dict or not 1 <= len(value) <= MAX_CHECKPOINT_ARRAYS:
        raise CandidateInputError(
            f"model checkpoint must be a dict containing 1..{MAX_CHECKPOINT_ARRAYS} arrays"
        )
    result: dict[str, np.ndarray] = {}
    total_bytes = 0
    for name, raw_array in value.items():
        if type(name) is not str or CHECKPOINT_NAME.fullmatch(name) is None:
            raise CandidateInputError("checkpoint names must be bare ASCII Python identifiers")
        if not isinstance(raw_array, np.ndarray):
            raise CandidateInputError(f"checkpoint entry {name!r} must be a NumPy array")
        array = np.array(raw_array, copy=True, order="C", subok=False)
        if array.dtype.kind not in "biuf":
            raise CandidateInputError(f"checkpoint entry {name!r} must use a safe numeric dtype")
        try:
            finite = bool(np.isfinite(array).all())
        except TypeError as exc:
            raise CandidateInputError(
                f"checkpoint entry {name!r} must contain finite numeric values"
            ) from exc
        if not finite:
            raise CandidateInputError(
                f"checkpoint entry {name!r} must contain finite numeric values"
            )
        total_bytes += int(array.nbytes)
        if total_bytes > MAX_CHECKPOINT_BYTES:
            raise CandidateInputError(
                f"model checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte limit"
            )
        result[name] = array
    return result


def _validate_diagnostics(value: object) -> dict[str, int | float]:
    if type(value) is not dict or len(value) > 64:
        raise CandidateInputError("training diagnostics must be a dict with at most 64 entries")
    result: dict[str, int | float] = {}
    for name, raw in value.items():
        if type(name) is not str or CHECKPOINT_NAME.fullmatch(name) is None:
            raise CandidateInputError("training diagnostic names must be ASCII identifiers")
        if type(raw) not in {int, float} or not math.isfinite(float(raw)):
            raise CandidateInputError("training diagnostic values must be finite numbers")
        result[name] = raw
    return result


def _validate_scores(value: object, expected_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != (expected_count,):
        raise CandidateInputError(f"model scores must have shape ({expected_count},)")
    if value.dtype.kind not in "biuf" or not bool(np.isfinite(value).all()):
        raise CandidateInputError("model scores must contain finite numeric values")
    return np.ascontiguousarray(value, dtype=np.dtype(SCORES_DTYPE))


def _train(request_path: Path, output: Path) -> None:
    request, capabilities = _workspace_request(request_path)
    _require_exact_keys(request, TRAIN_KEYS, "training request")
    if request["protocol_schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("protocol_schema_version must be 1")
    source_digest = _require_digest(request["source_digest"], "source_digest")
    config_digest = _require_digest(request["config_digest"], "config_digest")
    data_digest = _require_digest(request["data_digest"], "data_digest")
    split_token = _require_token(request["split_token"])
    seed = _require_uint32(request["seed"], "seed")
    features_handle = request["features_handle"]
    targets_handle = request["targets_handle"]
    user_groups_handle = request["user_groups_handle"]
    if (
        type(features_handle) is not str
        or type(targets_handle) is not str
        or type(user_groups_handle) is not str
    ):
        raise CandidateInputError("training capability handles must be strings")
    if len({features_handle, targets_handle, user_groups_handle}) != 3:
        raise CandidateInputError("training capability handles must be distinct")
    try:
        feature_path = capabilities[features_handle]
        target_path = capabilities[targets_handle]
        user_group_path = capabilities[user_groups_handle]
    except KeyError as exc:
        raise CandidateInputError("training capability handle is not approved") from exc
    features = _load_numeric_array(feature_path, "training features")
    targets = _load_numeric_array(target_path, "training targets")
    user_groups = _load_user_groups(user_group_path, features.shape[0])
    config = _load_config(config_digest)
    checkpoint = _validate_checkpoint(train_model(features, targets, user_groups, config, seed))
    # Diagnostics are informational only: nothing downstream scores them, and the checkpoint
    # above is already validated.  Letting them abort a training run that has actually succeeded
    # throws away real evaluation evidence, which is exactly what happened to two of three
    # candidates in overnight-11, one of which had trained for over two minutes.
    try:
        diagnostics = _validate_diagnostics(training_diagnostics(config, checkpoint))
    except Exception:  # Model authored code may raise anything.
        # Diagnostics accept only finite numbers, so the failure is reported as a flag rather
        # than as text.  It is retained in candidate_result.json for the campaign record.
        diagnostics = {"diagnostics_failed": 1.0}

    _prepare_output(output)
    checkpoint_dir = output / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.txt"
    with checkpoint_path.open("xb") as handle:
        np.savez(handle, **checkpoint)
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
            **diagnostics,
            "feature_count": int(features.shape[1]),
            "row_count": int(features.shape[0]),
        },
    }
    _write_json(output / "candidate_result.json", result)


def _load_checkpoint(path: Path, expected_digest: str) -> dict[str, np.ndarray]:
    if _sha256(path) != expected_digest:
        raise CandidateInputError("checkpoint_digest does not identify checkpoint bytes")
    with np.load(path, allow_pickle=False) as archive:
        checkpoint = {name: np.array(archive[name], copy=True) for name in archive.files}
    return _validate_checkpoint(checkpoint)


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
    scores = _validate_scores(predict_scores(features, checkpoint), expected_count)

    _prepare_output(output)
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
