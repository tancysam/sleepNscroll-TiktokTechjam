"""Self-contained deterministic LightGBM LambdaRank generated candidate.

Training consumes only controller-approved numeric feature, binary-target, and numeric user-group
capabilities.  A private stable grouped view is built for LightGBM; prediction remains in the
canonical capability order and has no target or user-group input.  This generated-plane program
does not import the trusted controller or scorer and does not compute organizer metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple, cast

import numpy as np

SCHEMA_VERSION = 1
PINNED_LIGHTGBM_VERSION = "4.7.0"
SCORES_DTYPE = "<f8"
CHECKPOINT_PATH = "checkpoint/model.txt"
MAX_JSON_BYTES = 256 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024

WORKSPACE_KEYS = {
    "approved_inputs",
    "budgets",
    "execution_id",
    "request",
    "schema_version",
    "source_snapshot_sha256",
    "split_role",
}
APPROVED_INPUT_KEYS = {"artifact", "name", "role", "workspace_path"}
ARTIFACT_KEYS = {"algorithm", "kind", "schema_version", "sha256", "size_bytes"}
BUDGET_KEYS = {"output_limit_bytes", "temp_limit_bytes"}
CONFIG_KEYS = {
    "candidate_family",
    "learning_rate",
    "min_data_in_leaf",
    "num_boost_round",
    "num_leaves",
    "num_threads",
    "schema_version",
    "seed_policy",
    "tree_count_policy",
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
TRAIN_SPLIT_ROLES = {"train", "inner_train"}
PREDICT_INPUT_ROLES = {
    "inner_valid": "inner_valid_inputs",
    "outer_valid": "outer_valid_inputs",
    "final": "final_inputs",
}
HANDLE_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class CandidateInputError(ValueError):
    """The candidate-visible request, capability, config, or checkpoint is malformed."""


class Capability(NamedTuple):
    """One verified candidate-visible capability path and phase-specific role."""

    path: Path
    role: str


def _sha256(path: Path, *, maximum: int | None = None) -> tuple[str, int]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CandidateInputError("capability or checkpoint must be a regular file")
    if maximum is not None and metadata.st_size > maximum:
        raise CandidateInputError("capability or checkpoint exceeds its byte limit")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if maximum is not None and size > maximum:
                raise CandidateInputError("capability or checkpoint exceeds its byte limit")
            digest.update(chunk)
    after = path.lstat()
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise CandidateInputError("capability or checkpoint changed while being read")
    return digest.hexdigest(), size


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateInputError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CandidateInputError(f"JSON contains non-finite constant {value}")


def _read_json(path: Path) -> dict[str, object]:
    _, size = _sha256(path, maximum=MAX_JSON_BYTES)
    if size > MAX_JSON_BYTES:
        raise CandidateInputError(f"JSON input exceeds {MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except CandidateInputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError) as exc:
        raise CandidateInputError("JSON input is malformed") from exc
    if not isinstance(value, dict):
        raise CandidateInputError("JSON input must contain one object")
    return value


def _require_exact_keys(value: dict[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise CandidateInputError(f"{name} keys do not match the exact schema")


def _require_digest(value: object, name: str) -> str:
    if type(value) is not str or DIGEST_RE.fullmatch(value) is None:
        raise CandidateInputError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise CandidateInputError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _require_uint32(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise CandidateInputError(f"{name} must be a uint32-compatible integer")
    return value


def _require_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if type(value) is not float or not math.isfinite(value) or not minimum <= value <= maximum:
        raise CandidateInputError(f"{name} must be a finite float in [{minimum}, {maximum}]")
    return value


def _require_token(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > 256
        or not value.isascii()
        or not value.isprintable()
        or any(character.isspace() for character in value)
    ):
        raise CandidateInputError("split_token is invalid")
    return value


def _relative_input_path(value: object) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise CandidateInputError("capability path must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or part.startswith(".") for part in path.parts)
        or len(path.parts) != 2
        or path.parts[0] != "inputs"
    ):
        raise CandidateInputError("capability path must name one canonical inputs/ file")
    return path


def _validate_artifact(value: object, *, path: Path) -> None:
    if not isinstance(value, dict):
        raise CandidateInputError("approved input artifact must be an object")
    artifact = cast(dict[str, object], value)
    _require_exact_keys(artifact, ARTIFACT_KEYS, "approved input artifact")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("approved input artifact schema_version must be 1")
    if artifact["algorithm"] != "sha256" or artifact["kind"] != "input":
        raise CandidateInputError("approved input artifact identity is invalid")
    expected_digest = _require_digest(artifact["sha256"], "approved input artifact sha256")
    expected_size = artifact["size_bytes"]
    if type(expected_size) is not int or expected_size < 0:
        raise CandidateInputError("approved input artifact size_bytes is invalid")
    actual_digest, actual_size = _sha256(path)
    if (actual_digest, actual_size) != (expected_digest, expected_size):
        raise CandidateInputError("approved input artifact bytes do not match the declaration")


def _validate_budgets(value: object) -> None:
    if not isinstance(value, dict):
        raise CandidateInputError("workspace budgets must be an object")
    budgets = cast(dict[str, object], value)
    _require_exact_keys(budgets, BUDGET_KEYS, "workspace budgets")
    for key in sorted(BUDGET_KEYS):
        _require_positive_int(budgets[key], f"workspace budgets.{key}", 2**63 - 1)


def _workspace_request(path: Path) -> tuple[dict[str, object], str, dict[str, Capability]]:
    document = _read_json(path)
    _require_exact_keys(document, WORKSPACE_KEYS, "workspace request")
    if document["schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("workspace schema_version must be 1")
    _require_digest(document["source_snapshot_sha256"], "source_snapshot_sha256")
    execution_id = document["execution_id"]
    split_role = document["split_role"]
    if type(execution_id) is not str or not execution_id or len(execution_id) > 64:
        raise CandidateInputError("execution_id is invalid")
    if type(split_role) is not str:
        raise CandidateInputError("split_role must be a string")
    _validate_budgets(document["budgets"])
    request = document["request"]
    approved = document["approved_inputs"]
    if not isinstance(request, dict) or not isinstance(approved, list) or len(approved) > 32:
        raise CandidateInputError("workspace request payload or approved inputs are invalid")
    capabilities: dict[str, Capability] = {}
    used_paths: set[PurePosixPath] = set()
    for item in approved:
        if not isinstance(item, dict):
            raise CandidateInputError("approved input declaration must be an object")
        declaration = cast(dict[str, object], item)
        _require_exact_keys(declaration, APPROVED_INPUT_KEYS, "approved input declaration")
        name = declaration["name"]
        role = declaration["role"]
        if (
            type(name) is not str
            or HANDLE_RE.fullmatch(name) is None
            or name in capabilities
            or type(role) is not str
        ):
            raise CandidateInputError("approved input name or role is invalid")
        relative = _relative_input_path(declaration["workspace_path"])
        if relative in used_paths:
            raise CandidateInputError("approved input paths must be unique")
        capability_path = path.parent.joinpath(*relative.parts)
        _validate_artifact(declaration["artifact"], path=capability_path)
        capabilities[name] = Capability(capability_path, role)
        used_paths.add(relative)
    return cast(dict[str, object], request), split_role, capabilities


def _load_numeric_array(path: Path, name: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise CandidateInputError(f"{name} must be one safe NumPy array") from exc
    if not isinstance(value, np.ndarray):
        if hasattr(value, "close"):
            value.close()
        raise CandidateInputError(f"{name} must contain one NumPy array")
    if value.dtype.kind not in "biuf":
        raise CandidateInputError(f"{name} must contain numeric values")
    try:
        finite = bool(np.isfinite(value).all())
    except TypeError as exc:
        raise CandidateInputError(f"{name} must contain numeric values") from exc
    if not finite:
        raise CandidateInputError(f"{name} must contain finite values")
    return value


def _load_features(path: Path, name: str) -> np.ndarray:
    value = _load_numeric_array(path, name)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise CandidateInputError(f"{name} must have non-empty shape (N, D)")
    try:
        return np.ascontiguousarray(value, dtype=np.dtype(SCORES_DTYPE))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandidateInputError(f"{name} must be representable as float64") from exc


def _load_targets(path: Path, expected: int) -> np.ndarray:
    value = _load_numeric_array(path, "training targets")
    if value.shape != (expected,):
        raise CandidateInputError(f"training targets must have shape ({expected},)")
    numeric = np.asarray(value, dtype=np.float64)
    if not bool(np.isin(numeric, (0.0, 1.0)).all()):
        raise CandidateInputError("training targets must contain only binary 0 and 1 values")
    return np.ascontiguousarray(numeric, dtype=np.int8)


def _load_user_groups(path: Path, expected: int) -> np.ndarray:
    value = _load_numeric_array(path, "training user groups")
    if value.shape != (expected,):
        raise CandidateInputError(f"training user groups must have shape ({expected},)")
    if value.dtype.kind not in "iuf":
        raise CandidateInputError("training user groups must use a non-boolean numeric dtype")
    return np.ascontiguousarray(value)


def _stable_user_grouping(user_groups: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return first-seen-group, within-group-stable row order and exact query sizes."""

    if user_groups.ndim != 1 or user_groups.size == 0 or user_groups.dtype.kind not in "iuf":
        raise CandidateInputError("training user groups must be a non-empty numeric vector")
    if not bool(np.isfinite(user_groups).all()):
        raise CandidateInputError("training user groups must contain finite values")
    _, first_indices, sorted_codes = np.unique(
        user_groups,
        return_index=True,
        return_inverse=True,
    )
    unique_by_first_seen = np.argsort(first_indices, kind="stable")
    first_seen_codes = np.empty(unique_by_first_seen.size, dtype=np.int64)
    first_seen_codes[unique_by_first_seen] = np.arange(unique_by_first_seen.size, dtype=np.int64)
    row_codes = first_seen_codes[sorted_codes]
    permutation = np.argsort(row_codes, kind="stable").astype(np.int64, copy=False)
    sizes = np.bincount(row_codes, minlength=unique_by_first_seen.size)
    group_sizes = tuple(int(value) for value in sizes)
    if (
        not group_sizes
        or sum(group_sizes) != user_groups.size
        or any(size <= 0 for size in group_sizes)
    ):
        raise CandidateInputError("training user grouping is internally inconsistent")
    return np.ascontiguousarray(permutation), group_sizes


def _validate_config(config: dict[str, object]) -> dict[str, object]:
    _require_exact_keys(config, CONFIG_KEYS, "config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("config schema_version must be 1")
    if config["candidate_family"] != "deterministic_lambdarank":
        raise CandidateInputError("candidate_family is invalid")
    if config["seed_policy"] != "controller_request_uint32":
        raise CandidateInputError("seed_policy must be controller_request_uint32")
    if config["tree_count_policy"] != "frozen_train_derived":
        raise CandidateInputError("tree_count_policy must be frozen_train_derived")
    _require_positive_int(config["num_threads"], "num_threads", 64)
    _require_positive_int(config["num_boost_round"], "num_boost_round", 10_000)
    _require_positive_int(config["num_leaves"], "num_leaves", 4096)
    _require_positive_int(config["min_data_in_leaf"], "min_data_in_leaf", 10_000_000)
    _require_float(config["learning_rate"], "learning_rate", 0.000001, 1.0)
    return config


def _load_config(expected_digest: str) -> dict[str, object]:
    path = Path(__file__).with_name("config.json")
    actual_digest, _ = _sha256(path, maximum=MAX_JSON_BYTES)
    if actual_digest != expected_digest:
        raise CandidateInputError("config_digest does not identify candidate config.json")
    return _validate_config(_read_json(path))


def _lightgbm() -> Any:
    try:
        import lightgbm
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise CandidateInputError(
            "LightGBM and its native runtime are required from the pinned research-tree group"
        ) from exc
    if getattr(lightgbm, "__version__", None) != PINNED_LIGHTGBM_VERSION:
        raise CandidateInputError(
            f"LightGBM {PINNED_LIGHTGBM_VERSION} is required, "
            f"found {getattr(lightgbm, '__version__', None)!r}"
        )
    return lightgbm


def _parameters(config: dict[str, object], seed: int) -> dict[str, object]:
    return {
        "objective": "lambdarank",
        "metric": "None",
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": cast(int, config["num_threads"]),
        "seed": seed,
        "data_random_seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "extra_seed": seed,
        "label_gain": [0, 1],
        "lambdarank_norm": True,
        "lambdarank_truncation_level": 8,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "extra_trees": False,
        "learning_rate": cast(float, config["learning_rate"]),
        "num_leaves": cast(int, config["num_leaves"]),
        "min_data_in_leaf": cast(int, config["min_data_in_leaf"]),
        "lambda_l2": 1.0,
        "verbosity": -1,
    }


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> tuple[str, tuple[int, ...]]:
    """Fit a fixed-tree-count deterministic CPU LambdaRank model on a stable grouped view."""

    validated_config = _validate_config(config)
    validated_seed = _require_uint32(seed, "seed")
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise CandidateInputError("training features must have non-empty shape (N, D)")
    if features.dtype != np.dtype(SCORES_DTYPE) or not bool(np.isfinite(features).all()):
        raise CandidateInputError("training features must be finite little-endian float64")
    if targets.shape != (features.shape[0],) or targets.dtype != np.dtype(np.int8):
        raise CandidateInputError("training targets must be an int8 vector aligned to features")
    if not bool(np.isin(targets, (0, 1)).all()):
        raise CandidateInputError("training targets must be binary")
    if user_groups.shape != (features.shape[0],):
        raise CandidateInputError("training user groups must align to features")
    permutation, group_sizes = _stable_user_grouping(user_groups)
    grouped_features = np.ascontiguousarray(features[permutation], dtype=np.float64)
    grouped_targets = np.ascontiguousarray(targets[permutation], dtype=np.int8)
    lightgbm = _lightgbm()
    dataset = lightgbm.Dataset(
        grouped_features,
        label=grouped_targets,
        group=list(group_sizes),
        free_raw_data=False,
    )
    booster = lightgbm.train(
        _parameters(validated_config, validated_seed),
        dataset,
        num_boost_round=cast(int, validated_config["num_boost_round"]),
    )
    model_text = cast(str, booster.model_to_string())
    if not model_text or "\x00" in model_text:
        raise CandidateInputError("LightGBM returned an invalid model checkpoint")
    if len(model_text.encode("utf-8")) > MAX_CHECKPOINT_BYTES:
        raise CandidateInputError("LightGBM model checkpoint exceeds its byte limit")
    return model_text, group_sizes


def predict_scores(features: np.ndarray, checkpoint_text: str) -> np.ndarray:
    """Predict directly in canonical feature order; no grouping or inverse scatter is needed."""

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise CandidateInputError("prediction features must have non-empty shape (N, D)")
    if features.dtype != np.dtype(SCORES_DTYPE) or not bool(np.isfinite(features).all()):
        raise CandidateInputError("prediction features must be finite little-endian float64")
    if type(checkpoint_text) is not str or not checkpoint_text or "\x00" in checkpoint_text:
        raise CandidateInputError("checkpoint text is invalid")
    lightgbm = _lightgbm()
    try:
        booster = lightgbm.Booster(model_str=checkpoint_text)
        raw = booster.predict(features, num_iteration=booster.num_trees())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CandidateInputError("verified checkpoint cannot produce predictions") from exc
    scores = np.asarray(raw)
    if scores.shape != (features.shape[0],) or scores.dtype.kind not in "iuf":
        raise CandidateInputError("LightGBM prediction shape or dtype is invalid")
    result = np.ascontiguousarray(scores, dtype=np.dtype(SCORES_DTYPE))
    if not bool(np.isfinite(result).all()):
        raise CandidateInputError("LightGBM predictions must be finite")
    return result


def _prepare_output(path: Path) -> None:
    if path.exists():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CandidateInputError("output must be a real directory")
        if tuple(path.iterdir()):
            raise CandidateInputError("output directory must be empty")
        return
    path.mkdir(parents=True, exist_ok=False)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def _write_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(payload) > MAX_JSON_BYTES:
        raise CandidateInputError("result JSON exceeds its byte limit")
    _write_bytes(path, payload)


def _require_capability(
    capabilities: dict[str, Capability],
    handle: object,
    *,
    role: str,
    name: str,
) -> Capability:
    if type(handle) is not str or HANDLE_RE.fullmatch(handle) is None:
        raise CandidateInputError(f"{name} must be an approved capability handle")
    try:
        capability = capabilities[handle]
    except KeyError as exc:
        raise CandidateInputError(f"{name} is not approved") from exc
    if capability.role != role:
        raise CandidateInputError(f"{name} has the wrong capability role")
    return capability


def _train(request_path: Path, output: Path) -> None:
    request, split_role, capabilities = _workspace_request(request_path)
    _require_exact_keys(request, TRAIN_KEYS, "training request")
    if split_role not in TRAIN_SPLIT_ROLES:
        raise CandidateInputError("training is allowed only for train or inner_train workspaces")
    if request["protocol_schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("protocol_schema_version must be 1")
    source_digest = _require_digest(request["source_digest"], "source_digest")
    config_digest = _require_digest(request["config_digest"], "config_digest")
    data_digest = _require_digest(request["data_digest"], "data_digest")
    split_token = _require_token(request["split_token"])
    seed = _require_uint32(request["seed"], "seed")
    feature_capability = _require_capability(
        capabilities,
        request["features_handle"],
        role="train_inputs",
        name="features_handle",
    )
    target_capability = _require_capability(
        capabilities,
        request["targets_handle"],
        role="train_targets",
        name="targets_handle",
    )
    group_capability = _require_capability(
        capabilities,
        request["user_groups_handle"],
        role="train_inputs",
        name="user_groups_handle",
    )
    requested_handles = {
        cast(str, request["features_handle"]),
        cast(str, request["targets_handle"]),
        cast(str, request["user_groups_handle"]),
    }
    if len(requested_handles) != 3 or set(capabilities) != requested_handles:
        raise CandidateInputError("training requires exactly three distinct approved capabilities")
    config = _load_config(config_digest)
    features = _load_features(feature_capability.path, "training features")
    targets = _load_targets(target_capability.path, features.shape[0])
    user_groups = _load_user_groups(group_capability.path, features.shape[0])
    model_text, group_sizes = train_model(features, targets, user_groups, config, seed)

    _prepare_output(output)
    checkpoint_dir = output / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = output / CHECKPOINT_PATH
    model_bytes = model_text.encode("utf-8")
    _write_bytes(checkpoint_path, model_bytes)
    checkpoint_digest = hashlib.sha256(model_bytes).hexdigest()
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
                "path": CHECKPOINT_PATH,
                "sha256": checkpoint_digest,
                "size_bytes": len(model_bytes),
            }
        ],
        "diagnostics": {
            "backend_version": PINNED_LIGHTGBM_VERSION,
            "feature_count": int(features.shape[1]),
            "group_count": len(group_sizes),
            "largest_group_size": max(group_sizes),
            "num_boost_round": cast(int, config["num_boost_round"]),
            "row_count": int(features.shape[0]),
            "seed": seed,
            "tree_count_policy": "frozen_train_derived",
        },
    }
    _write_json(output / "candidate_result.json", result)


def _load_checkpoint(path: Path, expected_digest: str) -> str:
    actual_digest, _ = _sha256(path, maximum=MAX_CHECKPOINT_BYTES)
    if actual_digest != expected_digest:
        raise CandidateInputError("checkpoint_digest does not identify checkpoint bytes")
    try:
        model_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CandidateInputError("checkpoint must be valid UTF-8 model text") from exc
    if not model_text or "\x00" in model_text:
        raise CandidateInputError("checkpoint model text is invalid")
    return model_text


def _predict(request_path: Path, checkpoint_path: Path, output: Path) -> None:
    request, split_role, capabilities = _workspace_request(request_path)
    _require_exact_keys(request, PREDICT_KEYS, "prediction request")
    expected_input_role = PREDICT_INPUT_ROLES.get(split_role)
    if expected_input_role is None:
        raise CandidateInputError(
            "prediction requires inner_valid, outer_valid, or final workspace"
        )
    if request["protocol_schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("protocol_schema_version must be 1")
    source_digest = _require_digest(request["source_digest"], "source_digest")
    config_digest = _require_digest(request["config_digest"], "config_digest")
    _load_config(config_digest)
    data_digest = _require_digest(request["data_digest"], "data_digest")
    split_token = _require_token(request["split_token"])
    checkpoint_digest = _require_digest(request["checkpoint_digest"], "checkpoint_digest")
    expected_count = _require_positive_int(request["expected_count"], "expected_count", 2**63 - 1)
    feature_capability = _require_capability(
        capabilities,
        request["features_handle"],
        role=expected_input_role,
        name="features_handle",
    )
    if set(capabilities) != {cast(str, request["features_handle"])}:
        raise CandidateInputError("prediction requires exactly one approved feature capability")
    features = _load_features(feature_capability.path, "prediction features")
    if features.shape[0] != expected_count:
        raise CandidateInputError("prediction features do not match expected_count")
    checkpoint_text = _load_checkpoint(checkpoint_path, checkpoint_digest)
    scores = predict_scores(features, checkpoint_text)

    _prepare_output(output)
    scores_path = output / "scores.npy"
    with scores_path.open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)
        handle.flush()
    scores_digest, _ = _sha256(scores_path)
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
            "backend_version": PINNED_LIGHTGBM_VERSION,
            "feature_count": int(features.shape[1]),
            "row_count": int(features.shape[0]),
        },
    }
    _write_json(output / "prediction_result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic generated LambdaRank candidate")
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
