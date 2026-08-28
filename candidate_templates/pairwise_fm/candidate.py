"""Self-contained deterministic GAUC-aligned pairwise factorization machine.

The generated source sees only controller-approved numeric capabilities. Its sampler draws a
positive ticket uniformly across eligible logged positives, which selects a user in proportion
to the user's positive count, then draws one of that user's logged negatives uniformly. No
catalog or unlogged negative source exists in this package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path, PurePosixPath
from typing import NamedTuple, cast

import numpy as np

SCHEMA_VERSION = 1
SCORES_DTYPE = "<f8"
CHECKPOINT_PATH = "checkpoint/model.txt"
MAX_JSON_BYTES = 256 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_SAMPLED_PAIRS = 1_000_000

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
    "epochs",
    "factor_dim",
    "l2",
    "learning_rate",
    "pair_batch_size",
    "pairs_per_epoch",
    "schema_version",
    "seed_policy",
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


class PairBatch(NamedTuple):
    """One deterministic batch of logged same-user positive and negative rows."""

    positive_indices: np.ndarray
    negative_indices: np.ndarray
    user_group_indices: np.ndarray


class GAUCPairSampler:
    """Linear-memory positive-weighted sampler over logged same-user impressions."""

    def __init__(self, user_groups: np.ndarray, targets: np.ndarray) -> None:
        groups = np.asarray(user_groups)
        labels = np.asarray(targets)
        if (
            groups.ndim != 1
            or groups.size == 0
            or groups.dtype.kind not in "iuf"
            or not bool(np.isfinite(groups).all())
        ):
            raise CandidateInputError("user groups must be a non-empty finite numeric vector")
        if labels.shape != groups.shape or labels.dtype.kind not in "biuf":
            raise CandidateInputError("targets must be a numeric vector aligned to user groups")
        try:
            numeric_labels = np.asarray(labels, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CandidateInputError("targets must contain only binary 0 and 1 values") from exc
        if not bool(np.isfinite(numeric_labels).all()) or not bool(
            np.isin(numeric_labels, (0.0, 1.0)).all()
        ):
            raise CandidateInputError("targets must contain only binary 0 and 1 values")

        _, first_indices, sorted_codes = np.unique(
            groups,
            return_index=True,
            return_inverse=True,
        )
        unique_by_first_seen = np.argsort(first_indices, kind="stable")
        first_seen_codes = np.empty(unique_by_first_seen.size, dtype=np.int64)
        first_seen_codes[unique_by_first_seen] = np.arange(
            unique_by_first_seen.size,
            dtype=np.int64,
        )
        row_codes = first_seen_codes[sorted_codes]
        binary = np.ascontiguousarray(numeric_labels, dtype=np.int8)
        positive_counts = np.bincount(row_codes[binary == 1], minlength=unique_by_first_seen.size)
        negative_counts = np.bincount(row_codes[binary == 0], minlength=unique_by_first_seen.size)
        eligible = np.logical_and(positive_counts > 0, negative_counts > 0)
        if not bool(eligible.any()):
            raise CandidateInputError("pair sampling requires at least one mixed-label user")
        eligible_codes = np.flatnonzero(eligible)
        compact_code = np.full(unique_by_first_seen.size, -1, dtype=np.int64)
        compact_code[eligible_codes] = np.arange(eligible_codes.size, dtype=np.int64)
        eligible_rows = eligible[row_codes]
        positive_rows = np.flatnonzero(np.logical_and(binary == 1, eligible_rows))
        negative_rows = np.flatnonzero(np.logical_and(binary == 0, eligible_rows))
        positive_rows = positive_rows[
            np.argsort(compact_code[row_codes[positive_rows]], kind="stable")
        ].astype(np.int64, copy=False)
        negative_rows = negative_rows[
            np.argsort(compact_code[row_codes[negative_rows]], kind="stable")
        ].astype(np.int64, copy=False)
        eligible_positive_counts = positive_counts[eligible].astype(np.int64, copy=False)
        eligible_negative_counts = negative_counts[eligible].astype(np.int64, copy=False)
        cumulative_positive_counts = np.cumsum(eligible_positive_counts, dtype=np.int64)
        negative_offsets = np.empty(eligible_codes.size, dtype=np.int64)
        negative_offsets[0] = 0
        if eligible_codes.size > 1:
            np.cumsum(eligible_negative_counts[:-1], out=negative_offsets[1:])

        self._positive_indices = np.ascontiguousarray(positive_rows)
        self._negative_indices = np.ascontiguousarray(negative_rows)
        self._cumulative_positive_counts = np.ascontiguousarray(cumulative_positive_counts)
        self._negative_offsets = np.ascontiguousarray(negative_offsets)
        self._negative_counts = np.ascontiguousarray(eligible_negative_counts)
        for array in (
            self._positive_indices,
            self._negative_indices,
            self._cumulative_positive_counts,
            self._negative_offsets,
            self._negative_counts,
        ):
            array.setflags(write=False)
        self.eligible_user_count = int(eligible_codes.size)
        self.eligible_positive_count = int(cumulative_positive_counts[-1])
        self.stored_row_index_count = int(positive_rows.size + negative_rows.size)
        self.pair_space_size = sum(
            int(positive) * int(negative)
            for positive, negative in zip(
                eligible_positive_counts,
                eligible_negative_counts,
                strict=True,
            )
        )

    def sample(self, pair_count: int, *, seed: int) -> PairBatch:
        """Draw without enumerating any user's positive-negative Cartesian product."""

        if type(pair_count) is not int or not 1 <= pair_count <= MAX_SAMPLED_PAIRS:
            raise CandidateInputError(f"pair_count must be an integer in [1, {MAX_SAMPLED_PAIRS}]")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise CandidateInputError("seed must be a uint32-compatible integer")
        rng = np.random.default_rng(seed)
        positive_tickets = rng.integers(
            0,
            self.eligible_positive_count,
            size=pair_count,
            dtype=np.int64,
        )
        group_indices = np.searchsorted(
            self._cumulative_positive_counts,
            positive_tickets,
            side="right",
        ).astype(np.int64, copy=False)
        positive_rows = self._positive_indices[positive_tickets]
        negative_positions = rng.integers(
            0,
            self._negative_counts[group_indices],
            dtype=np.int64,
        )
        negative_positions += self._negative_offsets[group_indices]
        negative_rows = self._negative_indices[negative_positions]
        for array in (positive_rows, negative_rows, group_indices):
            array.setflags(write=False)
        return PairBatch(positive_rows, negative_rows, group_indices)


def _sha256(path: Path, *, maximum: int | None = None) -> tuple[str, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CandidateInputError("capability or checkpoint is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CandidateInputError("capability or checkpoint must be a regular file")
    if maximum is not None and metadata.st_size > maximum:
        raise CandidateInputError("capability or checkpoint exceeds its byte limit")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                if maximum is not None and size > maximum:
                    raise CandidateInputError("capability or checkpoint exceeds its byte limit")
                digest.update(chunk)
        after = path.lstat()
    except CandidateInputError:
        raise
    except OSError as exc:
        raise CandidateInputError("capability or checkpoint could not be read") from exc
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
    if split_role in TRAIN_SPLIT_ROLES:
        allowed_roles = {"train_inputs", "train_targets"}
    elif split_role in PREDICT_INPUT_ROLES:
        allowed_roles = {PREDICT_INPUT_ROLES[split_role]}
    else:
        allowed_roles = set()
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
        if role not in allowed_roles:
            raise CandidateInputError("approved input role is not allowed for the workspace phase")
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


def _validate_config(config: dict[str, object]) -> dict[str, object]:
    _require_exact_keys(config, CONFIG_KEYS, "config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise CandidateInputError("config schema_version must be 1")
    if config["candidate_family"] != "deterministic_pairwise_fm":
        raise CandidateInputError("candidate_family is invalid")
    if config["seed_policy"] != "controller_request_uint32":
        raise CandidateInputError("seed_policy must be controller_request_uint32")
    _require_positive_int(config["epochs"], "epochs", 1_000)
    _require_positive_int(config["factor_dim"], "factor_dim", 64)
    _require_positive_int(config["pair_batch_size"], "pair_batch_size", MAX_SAMPLED_PAIRS)
    _require_positive_int(config["pairs_per_epoch"], "pairs_per_epoch", MAX_SAMPLED_PAIRS)
    _require_float(config["learning_rate"], "learning_rate", 0.000001, 1.0)
    _require_float(config["l2"], "l2", 0.0, 1.0)
    return config


def _load_config(expected_digest: str) -> dict[str, object]:
    path = Path(__file__).with_name("config.json")
    actual_digest, _ = _sha256(path, maximum=MAX_JSON_BYTES)
    if actual_digest != expected_digest:
        raise CandidateInputError("config_digest does not identify candidate config.json")
    return _validate_config(_read_json(path))


def _fm_scores(features: np.ndarray, linear: np.ndarray, factors: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        projected = features @ factors
        interactions = np.float64(0.5) * (
            projected * projected - (features * features) @ (factors * factors)
        ).sum(axis=1, dtype=np.float64)
        result = features @ linear + interactions
    return np.ascontiguousarray(result, dtype=np.float64)


def _pairwise_gradients(
    positive: np.ndarray,
    negative: np.ndarray,
    linear: np.ndarray,
    factors: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    positive_projection = positive @ factors
    negative_projection = negative @ factors
    positive_scores = _fm_scores(positive, linear, factors)
    negative_scores = _fm_scores(negative, linear, factors)
    margins = positive_scores - negative_scores
    losses = np.logaddexp(0.0, -margins)
    score_gradient = -np.exp(-np.logaddexp(0.0, margins))
    batch_size = np.float64(positive.shape[0])
    linear_gradient = (positive - negative).T @ score_gradient / batch_size
    linear_gradient += np.float64(l2) * linear
    projected_gradient = (
        positive.T @ (score_gradient[:, None] * positive_projection)
        - negative.T @ (score_gradient[:, None] * negative_projection)
    ) / batch_size
    square_coefficient = (positive * positive - negative * negative).T @ score_gradient / batch_size
    factor_gradient = projected_gradient - square_coefficient[:, None] * factors
    factor_gradient += np.float64(l2) * factors
    return (
        float(np.mean(losses, dtype=np.float64)),
        np.ascontiguousarray(linear_gradient),
        np.ascontiguousarray(factor_gradient),
    )


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the package-owned pairwise factorization machine."""

    validated = _validate_config(config)
    normalized_seed = _require_uint32(seed, "seed")
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] == 0
        or features.dtype != np.dtype(SCORES_DTYPE)
        or not bool(np.isfinite(features).all())
    ):
        raise CandidateInputError("training features must be finite little-endian float64 (N, D)")
    if (
        targets.shape != (features.shape[0],)
        or targets.dtype != np.dtype(np.int8)
        or not bool(np.isin(targets, (0, 1)).all())
    ):
        raise CandidateInputError("training targets must be an aligned binary int8 vector")
    if user_groups.shape != (features.shape[0],):
        raise CandidateInputError("training user groups must align to features")
    sampler = GAUCPairSampler(user_groups, targets)
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    normalized = np.ascontiguousarray((features - mean) / scale, dtype=np.float64)
    factor_dim = cast(int, validated["factor_dim"])
    rng = np.random.default_rng(normalized_seed)
    factors = np.ascontiguousarray(
        rng.normal(0.0, 0.01, size=(features.shape[1], factor_dim)),
        dtype=np.float64,
    )
    linear = np.zeros(features.shape[1], dtype=np.float64)
    first_linear = np.zeros_like(linear)
    second_linear = np.zeros_like(linear)
    first_factors = np.zeros_like(factors)
    second_factors = np.zeros_like(factors)
    beta_one = np.float64(0.9)
    beta_two = np.float64(0.999)
    epsilon = np.float64(1e-8)
    learning_rate = np.float64(cast(float, validated["learning_rate"]))
    l2 = cast(float, validated["l2"])
    epochs = cast(int, validated["epochs"])
    pairs_per_epoch = cast(int, validated["pairs_per_epoch"])
    pair_batch_size = cast(int, validated["pair_batch_size"])
    optimizer_step = 0
    sampled_pairs = 0
    final_data_loss = 0.0
    for epoch in range(epochs):
        sample_seed = (normalized_seed + (epoch + 1) * 0x9E3779B1) & 0xFFFFFFFF
        sampled = sampler.sample(pairs_per_epoch, seed=sample_seed)
        epoch_loss_sum = 0.0
        for offset in range(0, pairs_per_epoch, pair_batch_size):
            limit = min(offset + pair_batch_size, pairs_per_epoch)
            positive = normalized[sampled.positive_indices[offset:limit]]
            negative = normalized[sampled.negative_indices[offset:limit]]
            data_loss, linear_gradient, factor_gradient = _pairwise_gradients(
                positive,
                negative,
                linear,
                factors,
                l2,
            )
            linear_gradient = np.clip(linear_gradient, -10.0, 10.0)
            factor_gradient = np.clip(factor_gradient, -10.0, 10.0)
            optimizer_step += 1
            first_linear *= beta_one
            first_linear += (1.0 - beta_one) * linear_gradient
            second_linear *= beta_two
            second_linear += (1.0 - beta_two) * (linear_gradient * linear_gradient)
            first_factors *= beta_one
            first_factors += (1.0 - beta_one) * factor_gradient
            second_factors *= beta_two
            second_factors += (1.0 - beta_two) * (factor_gradient * factor_gradient)
            first_correction = 1.0 - beta_one**optimizer_step
            second_correction = 1.0 - beta_two**optimizer_step
            linear -= (
                learning_rate
                * (first_linear / first_correction)
                / (np.sqrt(second_linear / second_correction) + epsilon)
            )
            factors -= (
                learning_rate
                * (first_factors / first_correction)
                / (np.sqrt(second_factors / second_correction) + epsilon)
            )
            if not bool(np.isfinite(linear).all()) or not bool(np.isfinite(factors).all()):
                raise CandidateInputError("pairwise FM optimizer produced non-finite state")
            batch_count = limit - offset
            epoch_loss_sum += data_loss * batch_count
            sampled_pairs += batch_count
        final_data_loss = epoch_loss_sum / pairs_per_epoch
    if not math.isfinite(final_data_loss):
        raise CandidateInputError("pairwise FM objective produced non-finite diagnostics")
    return {
        "checkpoint_schema_version": np.asarray(SCHEMA_VERSION, dtype="<i8"),
        "eligible_positive_count": np.asarray(sampler.eligible_positive_count, dtype="<i8"),
        "eligible_user_count": np.asarray(sampler.eligible_user_count, dtype="<i8"),
        "epochs": np.asarray(epochs, dtype="<i8"),
        "factor_dim": np.asarray(factor_dim, dtype="<i8"),
        "factors": np.ascontiguousarray(factors, dtype=np.float64),
        "feature_mean": np.ascontiguousarray(mean, dtype=np.float64),
        "feature_scale": np.ascontiguousarray(scale, dtype=np.float64),
        "final_data_loss": np.asarray(final_data_loss, dtype="<f8"),
        "linear": np.ascontiguousarray(linear, dtype=np.float64),
        "optimizer_steps": np.asarray(optimizer_step, dtype="<i8"),
        "pair_space_size": np.asarray(sampler.pair_space_size, dtype="<i8"),
        "sampled_pairs": np.asarray(sampled_pairs, dtype="<i8"),
        "seed": np.asarray(normalized_seed, dtype="<i8"),
        "stored_row_index_count": np.asarray(sampler.stored_row_index_count, dtype="<i8"),
    }


CHECKPOINT_KEYS = {
    "checkpoint_schema_version",
    "eligible_positive_count",
    "eligible_user_count",
    "epochs",
    "factor_dim",
    "factors",
    "feature_mean",
    "feature_scale",
    "final_data_loss",
    "linear",
    "optimizer_steps",
    "pair_space_size",
    "sampled_pairs",
    "seed",
    "stored_row_index_count",
}


def _checkpoint_integer(checkpoint: dict[str, np.ndarray], name: str, *, positive: bool) -> int:
    value = checkpoint[name]
    if value.shape != () or value.dtype != np.dtype("<i8"):
        raise CandidateInputError(f"checkpoint {name} must be a little-endian int64 scalar")
    result = int(value)
    if result < (1 if positive else 0):
        raise CandidateInputError(f"checkpoint {name} is outside its valid range")
    return result


def _validate_checkpoint(checkpoint: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(checkpoint) != CHECKPOINT_KEYS:
        raise CandidateInputError("checkpoint inventory is invalid")
    for name in (
        "checkpoint_schema_version",
        "eligible_positive_count",
        "eligible_user_count",
        "epochs",
        "factor_dim",
        "optimizer_steps",
        "pair_space_size",
        "sampled_pairs",
        "seed",
        "stored_row_index_count",
    ):
        _checkpoint_integer(checkpoint, name, positive=name != "seed")
    checkpoint_schema = _checkpoint_integer(
        checkpoint,
        "checkpoint_schema_version",
        positive=True,
    )
    if checkpoint_schema != SCHEMA_VERSION:
        raise CandidateInputError("checkpoint schema version is unsupported")
    factor_dim = _checkpoint_integer(checkpoint, "factor_dim", positive=True)
    mean = checkpoint["feature_mean"]
    scale = checkpoint["feature_scale"]
    linear = checkpoint["linear"]
    factors = checkpoint["factors"]
    final_data_loss = checkpoint["final_data_loss"]
    if (
        mean.ndim != 1
        or mean.size == 0
        or mean.dtype != np.dtype(SCORES_DTYPE)
        or scale.shape != mean.shape
        or scale.dtype != np.dtype(SCORES_DTYPE)
        or linear.shape != mean.shape
        or linear.dtype != np.dtype(SCORES_DTYPE)
        or factors.shape != (mean.size, factor_dim)
        or factors.dtype != np.dtype(SCORES_DTYPE)
        or final_data_loss.shape != ()
        or final_data_loss.dtype != np.dtype(SCORES_DTYPE)
    ):
        raise CandidateInputError("checkpoint model array shapes or dtypes are invalid")
    if (
        not all(
            bool(np.isfinite(array).all())
            for array in (mean, scale, linear, factors, final_data_loss)
        )
        or not bool(np.all(scale > 0.0))
        or float(final_data_loss) < 0.0
    ):
        raise CandidateInputError("checkpoint model arrays must be finite and well-formed")
    return checkpoint


def predict_scores(features: np.ndarray, checkpoint: dict[str, np.ndarray]) -> np.ndarray:
    """Replay the package-owned factorization machine."""

    validated = _validate_checkpoint(checkpoint)
    mean = validated["feature_mean"]
    scale = validated["feature_scale"]
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1:] != mean.shape
        or features.dtype != np.dtype(SCORES_DTYPE)
        or not bool(np.isfinite(features).all())
    ):
        raise CandidateInputError("prediction features do not match the finite checkpoint schema")
    normalized = np.ascontiguousarray((features - mean) / scale, dtype=np.float64)
    scores = _fm_scores(normalized, validated["linear"], validated["factors"])
    if scores.shape != (features.shape[0],) or not bool(np.isfinite(scores).all()):
        raise CandidateInputError("pairwise FM predictions must be a finite vector")
    return np.ascontiguousarray(scores, dtype=np.dtype(SCORES_DTYPE))


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
    checkpoint = train_model(features, targets, user_groups, config, seed)

    _prepare_output(output)
    checkpoint_dir = output / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = output / CHECKPOINT_PATH
    with checkpoint_path.open("xb") as handle:
        np.savez(handle, **checkpoint)
        handle.flush()
    checkpoint_digest, checkpoint_size = _sha256(
        checkpoint_path,
        maximum=MAX_CHECKPOINT_BYTES,
    )
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
                "size_bytes": checkpoint_size,
            }
        ],
        "diagnostics": {
            "backend": "numpy",
            "eligible_positive_count": int(checkpoint["eligible_positive_count"]),
            "eligible_user_count": int(checkpoint["eligible_user_count"]),
            "epochs": int(checkpoint["epochs"]),
            "factor_dim": int(checkpoint["factor_dim"]),
            "feature_count": int(features.shape[1]),
            "final_data_loss": float(checkpoint["final_data_loss"]),
            "optimizer_steps": int(checkpoint["optimizer_steps"]),
            "pair_space_size": int(checkpoint["pair_space_size"]),
            "row_count": int(features.shape[0]),
            "sampled_pairs": int(checkpoint["sampled_pairs"]),
            "sampling_policy": "positive_weighted_logged_same_user",
            "seed": seed,
            "stored_row_index_count": int(checkpoint["stored_row_index_count"]),
            "training_objective": "same_user_pairwise_logistic",
        },
    }
    _write_json(output / "candidate_result.json", result)


def _load_checkpoint(path: Path, expected_digest: str) -> dict[str, np.ndarray]:
    actual_digest, _ = _sha256(path, maximum=MAX_CHECKPOINT_BYTES)
    if actual_digest != expected_digest:
        raise CandidateInputError("checkpoint_digest does not identify checkpoint bytes")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise CandidateInputError("checkpoint must be a safe NumPy archive") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        if hasattr(loaded, "close"):
            loaded.close()
        raise CandidateInputError("checkpoint must be one NumPy archive")
    try:
        if set(loaded.files) != CHECKPOINT_KEYS:
            raise CandidateInputError("checkpoint inventory is invalid")
        checkpoint = {name: loaded[name] for name in sorted(CHECKPOINT_KEYS)}
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise CandidateInputError("checkpoint arrays could not be loaded safely") from exc
    finally:
        loaded.close()
    return _validate_checkpoint(checkpoint)


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
    checkpoint = _load_checkpoint(checkpoint_path, checkpoint_digest)
    scores = predict_scores(features, checkpoint)

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
            "backend": "numpy",
            "factor_dim": int(checkpoint["factor_dim"]),
            "feature_count": int(features.shape[1]),
            "row_count": int(features.shape[0]),
        },
    }
    _write_json(output / "prediction_result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic generated pairwise FM candidate")
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
