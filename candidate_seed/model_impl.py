"""Small mutable model surface for autonomous candidate generation.

The stable ``candidate.py`` entrypoint owns capabilities, protocol validation, checkpoint I/O,
and result writing. Research implementations should normally replace only this module and
``config.json``.
"""

from __future__ import annotations

from typing import cast

import numpy as np

SCORES_DTYPE = "<f8"
CONFIG_KEYS = {
    "candidate_family",
    "epochs",
    "l2",
    "learning_rate",
    "logit_clip",
    "schema_version",
}


class CandidateModelError(ValueError):
    """The candidate-owned model configuration or numeric state is invalid."""


def _config_epochs(config: dict[str, object]) -> int:
    value = config["epochs"]
    if type(value) is not int or not 1 <= value <= 10_000:
        raise CandidateModelError("epochs is invalid")
    return value


def _config_float(config: dict[str, object], name: str) -> float:
    value = config[name]
    if type(value) not in {int, float}:
        raise CandidateModelError(f"{name} is invalid")
    numeric = float(cast(int | float, value))
    if not 0 < numeric < 1_000:
        raise CandidateModelError(f"{name} is invalid")
    return numeric


def validate_config(config: dict[str, object]) -> None:
    """Validate the model-owned configuration before training or prediction."""

    if set(config) != CONFIG_KEYS:
        raise CandidateModelError("model config keys do not match the implementation")
    if config["schema_version"] != 1:
        raise CandidateModelError("config schema_version must be 1")
    if config["candidate_family"] != "deterministic_logistic_seed":
        raise CandidateModelError("candidate_family is invalid")
    _config_epochs(config)
    for name in ("l2", "learning_rate", "logit_clip"):
        _config_float(config, name)


def _sigmoid(logits: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(logits, -clip, clip)
    result = np.reciprocal(1.0 + np.exp(-bounded), dtype=np.float64)
    return cast(np.ndarray, result)


# ---------------------------------------------------------------------------------------------
# Provided, verified building blocks.
#
# The controller appends categorical identity codes to the feature matrix, and writing the
# interaction maths and the within-user pair sampler from scratch has been the dominant source of
# candidate failure in this project. Both helpers below are tested; prefer calling them over
# reimplementing them. ``train_model`` here does not use them, so a candidate that stays pointwise
# and linear is unaffected.
# ---------------------------------------------------------------------------------------------


def categorical_codes(features: np.ndarray, category_count: int) -> list[np.ndarray]:
    """Return the trailing ``category_count`` identity code columns as int64 vectors."""

    if features.ndim != 2 or features.shape[1] <= category_count:
        raise CandidateModelError("features must have more columns than categorical codes")
    codes: list[np.ndarray] = []
    for column in range(features.shape[1] - category_count, features.shape[1]):
        raw = features[:, column]
        rounded = np.rint(raw)
        if not bool(np.equal(raw, rounded).all()) or float(rounded.min()) < 0.0:
            raise CandidateModelError("categorical code columns must be nonnegative integers")
        codes.append(np.ascontiguousarray(rounded, dtype=np.int64))
    return codes


def embedding_table_size(code: np.ndarray) -> int:
    """Rows needed for ``code``, with one spare row for identities unseen during training.

    Each fold fits its own vocabulary, so size tables from the training matrix you are handed and
    never from a constant.
    """

    return int(code.max()) + 2


def fm_interaction_scores(embeddings: list[np.ndarray], codes: list[np.ndarray]) -> np.ndarray:
    """Second-order factorization-machine interaction term, shape ``(N,)``.

    The two accumulators deliberately have DIFFERENT shapes: ``pair_sum`` is ``(N, rank)`` and
    ``square_sum`` is ``(N,)``. Mixing them raises a broadcast error, which is the most frequent
    defect in this codebase's history. Codes at or beyond a table's last row are clamped onto the
    spare row, so an identity unseen in training scores rather than raising.
    """

    if not embeddings or len(embeddings) != len(codes):
        raise CandidateModelError("one embedding table is required per code column")
    rows = int(codes[0].shape[0])
    rank = int(embeddings[0].shape[1])
    pair_sum = np.zeros((rows, rank), dtype=np.float64)
    square_sum = np.zeros(rows, dtype=np.float64)
    for table, code in zip(embeddings, codes, strict=True):
        if table.ndim != 2 or table.shape[1] != rank:
            raise CandidateModelError("embedding tables must share one rank")
        if code.shape != (rows,):
            raise CandidateModelError("code columns must align with the feature rows")
        part = table[np.minimum(code, table.shape[0] - 1)]
        pair_sum += part
        square_sum += np.sum(part * part, axis=1)
    return 0.5 * (np.sum(pair_sum * pair_sum, axis=1) - square_sum)


def within_user_pairs(
    targets: np.ndarray,
    user_groups: np.ndarray,
    generator: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample GAUC-matched ``(positive row, negative row)`` pairs from mixed-label users.

    Drawing the positive uniformly from the pooled positives weights each user by its positive
    count, which is what GAUC does; the negative is then uniform inside that same user's rows.

    Every lookup below uses the COMPACT position of a group in the eligible layout, never a raw
    ``user_groups`` value, and every offset is drawn against that group's own negative count.
    Confusing either is what raises IndexError.
    """

    if targets.shape != user_groups.shape or targets.ndim != 1:
        raise CandidateModelError("targets and user_groups must be aligned one-dimensional")
    if type(count) is not int or count <= 0:
        raise CandidateModelError("pair count must be a positive integer")

    order = np.argsort(user_groups, kind="stable")
    sorted_groups = user_groups[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), boundaries.astype(np.int64)))
    ends = np.concatenate((boundaries.astype(np.int64), np.asarray([sorted_groups.size])))

    positive_blocks: list[np.ndarray] = []
    negative_blocks: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        rows = order[start:end]
        labels = targets[rows]
        positives, negatives = rows[labels == 1.0], rows[labels == 0.0]
        if positives.size and negatives.size:
            positive_blocks.append(positives)
            negative_blocks.append(negatives)
    if not positive_blocks:
        raise CandidateModelError("no GAUC eligible user has both a positive and a negative")

    positive_rows = np.concatenate(positive_blocks)
    negative_rows = np.concatenate(negative_blocks)
    positive_counts = np.asarray([block.size for block in positive_blocks], dtype=np.int64)
    negative_counts = np.asarray([block.size for block in negative_blocks], dtype=np.int64)
    group_of_positive = np.repeat(np.arange(positive_counts.size), positive_counts)
    negative_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(negative_counts)[:-1]))

    picked = generator.integers(0, positive_rows.size, size=count)
    groups = group_of_positive[picked]
    offsets = generator.integers(0, negative_counts[groups])
    return positive_rows[picked], negative_rows[negative_starts[groups] + offsets]


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit a fixed-step standardized logistic model with deterministic full-batch updates."""

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise CandidateModelError("training features must have non-empty shape (N, D)")
    if targets.shape != (features.shape[0],):
        raise CandidateModelError("training targets must have shape (N,)")
    if user_groups.shape != (features.shape[0],):
        raise CandidateModelError("training user groups must have shape (N,)")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise CandidateModelError("seed must be a uint32-compatible integer")
    if not bool(np.logical_or(targets == 0.0, targets == 1.0).all()):
        raise CandidateModelError("training targets must be binary")
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
        raise CandidateModelError("training objective became non-finite")
    return {
        "bias": np.asarray(bias, dtype=np.float64),
        "feature_mean": np.ascontiguousarray(mean, dtype=np.float64),
        "feature_scale": np.ascontiguousarray(scale, dtype=np.float64),
        "final_objective": np.asarray(objective, dtype=np.float64),
        "weights": np.ascontiguousarray(weights, dtype=np.float64),
    }


def predict_scores(features: np.ndarray, checkpoint: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the owned normalization and logistic interaction from a verified checkpoint."""

    expected = {"bias", "feature_mean", "feature_scale", "final_objective", "weights"}
    if set(checkpoint) != expected:
        raise CandidateModelError("checkpoint inventory is invalid")
    mean = checkpoint["feature_mean"]
    scale = checkpoint["feature_scale"]
    weights = checkpoint["weights"]
    bias = checkpoint["bias"]
    if features.ndim != 2 or features.shape[1:] != weights.shape:
        raise CandidateModelError("prediction feature shape does not match the checkpoint")
    if mean.shape != weights.shape or scale.shape != weights.shape or bias.shape != ():
        raise CandidateModelError("checkpoint array shapes are invalid")
    if not all(array.dtype == np.dtype(SCORES_DTYPE) for array in checkpoint.values()):
        raise CandidateModelError("checkpoint arrays must use float64")
    if not all(bool(np.isfinite(array).all()) for array in checkpoint.values()):
        raise CandidateModelError("checkpoint arrays must be finite")
    logits = ((features - mean) / scale) @ weights + bias
    return np.ascontiguousarray(_sigmoid(logits, 40.0), dtype=np.dtype(SCORES_DTYPE))


def training_diagnostics(
    config: dict[str, object], checkpoint: dict[str, np.ndarray]
) -> dict[str, int | float]:
    """Return bounded JSON diagnostics owned by the model implementation."""

    return {
        "epochs": _config_epochs(config),
        "final_objective": float(checkpoint["final_objective"]),
    }
