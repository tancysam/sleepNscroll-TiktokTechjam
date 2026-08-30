"""Protected query-balanced pointwise composition over the categorical specialist.

Autonomous candidate code may import and compose this module, but cannot replace it. The exact
recipe was selected on train-derived Fold B in Attempt 14 and is retained as a deterministic
specialist for the reviewed ``video_type_code`` portfolio experiment. It consumes only the frozen
first 82 approved features, targets, and user groups and has no scorer, metric, filesystem,
network, or controller dependency.
"""

from __future__ import annotations

import math

import numpy as np
from reference_categorical_ranker import (
    _fit_reference_categorical_ranker,
    reference_categorical_ranker_diagnostics,
    reference_categorical_ranker_scores,
)

POINTWISE_FEATURE_COUNT = 82
POINTWISE_INPUT_COUNT = 83
POINTWISE_EPOCHS = 3
POINTWISE_BATCH_SIZE = 4096
POINTWISE_LEARNING_RATE = 0.001
POINTWISE_WEIGHT_DECAY = 0.00001
POINTWISE_GRADIENT_CLIP = 5.0
POINTWISE_RESIDUAL_SCALE = 0.5
POINTWISE_LOGIT_CLIP = 30.0
POINTWISE_HIDDEN_ONE = 256
POINTWISE_HIDDEN_TWO = 64
POINTWISE_SEED = 0

_BASE_KEYS = {
    "categorical_rank_code_offsets",
    "categorical_rank_feature_count",
    "categorical_rank_model_utf8",
    "categorical_rank_residual_shrinkage",
    "categorical_rank_schema_version",
    "categorical_rank_tree_count",
    "categorical_rank_seed",
    "reference_factors",
    "reference_feature_positions",
    "reference_final_pairwise_loss",
    "reference_linear",
    "reference_sampled_pairs",
    "reference_schema_version",
    "reference_total_dim",
    "reference_seed",
}
_POINTWISE_KEYS = {
    "pointwise_bias_one",
    "pointwise_bias_out",
    "pointwise_bias_two",
    "pointwise_center",
    "pointwise_final_loss",
    "pointwise_residual_scale",
    "pointwise_scale",
    "pointwise_schema_version",
    "pointwise_weight_one",
    "pointwise_weight_out",
    "pointwise_weight_two",
    "pointwise_seed",
}
_STATE_KEYS = _BASE_KEYS | _POINTWISE_KEYS


class ReferencePointwiseRankerError(ValueError):
    """The protected pointwise-ranker input or state violates its frozen contract."""


def _legacy_features(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] < POINTWISE_FEATURE_COUNT
        or matrix.dtype != np.dtype("<f8")
        or not bool(np.isfinite(matrix).all())
    ):
        raise ReferencePointwiseRankerError(
            "features must be finite little-endian float64 with shape (N, D), D >= 82"
        )
    return np.ascontiguousarray(matrix[:, :POINTWISE_FEATURE_COUNT])


def _base_state(checkpoint: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: checkpoint[name] for name in _BASE_KEYS}


def _specialist_features(matrix: np.ndarray) -> np.ndarray:
    """Keep the legacy Pointwise 82-column surface while satisfying the 83-column specialist."""
    return np.pad(matrix, ((0, 0), (0, 1)), constant_values=0.0)


def _gelu(value: np.ndarray) -> np.ndarray:
    coefficient = np.float32(0.7978845608028654)
    cubic = np.float32(0.044715)
    inner = coefficient * (value + cubic * value * value * value)
    return np.float32(0.5) * value * (np.float32(1.0) + np.tanh(inner))


def _gelu_gradient(value: np.ndarray) -> np.ndarray:
    coefficient = np.float32(0.7978845608028654)
    cubic = np.float32(0.044715)
    inner = coefficient * (value + cubic * value * value * value)
    tangent = np.tanh(inner)
    slope = coefficient * (np.float32(1.0) + np.float32(3.0) * cubic * value * value)
    return np.float32(0.5) * (np.float32(1.0) + tangent) + np.float32(0.5) * value * (
        np.float32(1.0) - tangent * tangent
    ) * slope


def _network(
    matrix: np.ndarray,
    parameters: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    weight_one, bias_one, weight_two, bias_two, weight_out, bias_out = parameters
    pre_one = matrix @ weight_one + bias_one
    hidden_one = _gelu(pre_one)
    pre_two = hidden_one @ weight_two + bias_two
    hidden_two = _gelu(pre_two)
    raw = (hidden_two @ weight_out + bias_out).reshape(-1)
    return raw, (matrix, pre_one, hidden_one, pre_two, hidden_two)


def _robust_location_scale(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(inputs, axis=0)
    lower, upper = np.percentile(inputs, (25.0, 75.0), axis=0)
    scale = upper - lower
    fallback = np.std(inputs, axis=0, dtype=np.float64)
    scale = np.where(scale > 1.0e-8, scale, fallback)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return (
        np.ascontiguousarray(location, dtype=np.float64),
        np.ascontiguousarray(scale, dtype=np.float64),
    )


def _query_weights(user_groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(user_groups, return_inverse=True, return_counts=True)
    weights = np.reciprocal(counts[inverse].astype(np.float64))
    weights *= np.float64(weights.size) / np.sum(weights, dtype=np.float64)
    return np.ascontiguousarray(weights, dtype=np.float32)


def _fit_pointwise_state(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    specialist_scores: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    matrix = _legacy_features(features)
    labels = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(user_groups)
    if labels.shape != (matrix.shape[0],) or groups.shape != labels.shape:
        raise ReferencePointwiseRankerError("targets and user_groups must align to features")
    if not bool(np.isfinite(labels).all()) or not bool(np.isin(labels, (0.0, 1.0)).all()):
        raise ReferencePointwiseRankerError("targets must contain only binary 0 and 1")
    if specialist_scores.shape != labels.shape or not bool(np.isfinite(specialist_scores).all()):
        raise ReferencePointwiseRankerError("specialist scores must align to features")
    if type(epochs) is not int or not 1 <= epochs <= POINTWISE_EPOCHS:
        raise ReferencePointwiseRankerError("pointwise epochs are outside the protected bound")

    combined = np.empty((matrix.shape[0], POINTWISE_INPUT_COUNT), dtype=np.float64)
    combined[:, :POINTWISE_FEATURE_COUNT] = matrix
    combined[:, POINTWISE_FEATURE_COUNT] = specialist_scores
    center, scale = _robust_location_scale(combined)
    row_weights = _query_weights(groups)
    rng = np.random.default_rng(seed)
    parameters = (
        rng.normal(
            0.0,
            math.sqrt(2.0 / POINTWISE_INPUT_COUNT),
            (POINTWISE_INPUT_COUNT, POINTWISE_HIDDEN_ONE),
        ).astype(np.float32),
        np.zeros(POINTWISE_HIDDEN_ONE, dtype=np.float32),
        rng.normal(
            0.0,
            math.sqrt(2.0 / POINTWISE_HIDDEN_ONE),
            (POINTWISE_HIDDEN_ONE, POINTWISE_HIDDEN_TWO),
        ).astype(np.float32),
        np.zeros(POINTWISE_HIDDEN_TWO, dtype=np.float32),
        rng.normal(0.0, 0.01, (POINTWISE_HIDDEN_TWO, 1)).astype(np.float32),
        np.zeros(1, dtype=np.float32),
    )
    first_moments = tuple(np.zeros_like(value) for value in parameters)
    second_moments = tuple(np.zeros_like(value) for value in parameters)
    order = np.arange(matrix.shape[0], dtype=np.int64)
    step = 0
    final_loss = math.nan

    for _ in range(epochs):
        rng.shuffle(order)
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, matrix.shape[0], POINTWISE_BATCH_SIZE):
            indices = order[start : start + POINTWISE_BATCH_SIZE]
            inputs = np.asarray((combined[indices] - center) / scale, dtype=np.float32)
            raw, cache = _network(inputs, parameters)
            tangent = np.tanh(raw)
            correction = np.float32(POINTWISE_RESIDUAL_SCALE) * tangent
            logits = np.clip(
                specialist_scores[indices].astype(np.float32) + correction,
                -POINTWISE_LOGIT_CLIP,
                POINTWISE_LOGIT_CLIP,
            )
            batch_labels = labels[indices].astype(np.float32)
            sample_weights = row_weights[indices]
            denominator = np.sum(sample_weights, dtype=np.float32)
            probabilities = np.reciprocal(np.float32(1.0) + np.exp(-logits))
            gradient_raw = (
                sample_weights
                * (probabilities - batch_labels)
                / denominator
                * np.float32(POINTWISE_RESIDUAL_SCALE)
                * (np.float32(1.0) - tangent * tangent)
            )[:, None]
            batch_inputs, pre_one, hidden_one, pre_two, hidden_two = cache
            weight_one, _, weight_two, _, weight_out, _ = parameters
            gradients = (
                batch_inputs.T @ (
                    (gradient_raw @ weight_out.T * _gelu_gradient(pre_two)) @ weight_two.T
                    * _gelu_gradient(pre_one)
                )
                + np.float32(POINTWISE_WEIGHT_DECAY) * weight_one,
                np.sum(
                    (gradient_raw @ weight_out.T * _gelu_gradient(pre_two)) @ weight_two.T
                    * _gelu_gradient(pre_one),
                    axis=0,
                ),
                hidden_one.T @ (gradient_raw @ weight_out.T * _gelu_gradient(pre_two))
                + np.float32(POINTWISE_WEIGHT_DECAY) * weight_two,
                np.sum(gradient_raw @ weight_out.T * _gelu_gradient(pre_two), axis=0),
                hidden_two.T @ gradient_raw
                + np.float32(POINTWISE_WEIGHT_DECAY) * weight_out,
                np.sum(gradient_raw, axis=0),
            )
            norm_squared = sum(
                float(np.sum(gradient * gradient, dtype=np.float64)) for gradient in gradients
            )
            multiplier = min(
                1.0,
                POINTWISE_GRADIENT_CLIP / max(math.sqrt(norm_squared), 1.0e-12),
            )
            step += 1
            for parameter, gradient, first, second in zip(
                parameters, gradients, first_moments, second_moments, strict=True
            ):
                gradient *= np.float32(multiplier)
                first *= np.float32(0.9)
                first += np.float32(0.1) * gradient
                second *= np.float32(0.999)
                second += np.float32(0.001) * gradient * gradient
                parameter -= np.float32(POINTWISE_LEARNING_RATE) * (
                    first / np.float32(1.0 - 0.9**step)
                ) / (np.sqrt(second / np.float32(1.0 - 0.999**step)) + np.float32(1.0e-8))
            losses = np.logaddexp(np.float32(0.0), logits) - batch_labels * logits
            weighted_loss_sum += float(np.sum(sample_weights * losses, dtype=np.float64))
            weight_sum += float(np.sum(sample_weights, dtype=np.float64))
        final_loss = weighted_loss_sum / weight_sum

    if not math.isfinite(final_loss) or not all(
        bool(np.isfinite(value).all()) for value in parameters
    ):
        raise ReferencePointwiseRankerError("pointwise optimizer produced non-finite state")
    names = ("weight_one", "bias_one", "weight_two", "bias_two", "weight_out", "bias_out")
    state = {
        f"pointwise_{name}": np.ascontiguousarray(value)
        for name, value in zip(names, parameters, strict=True)
    }
    state.update(
        {
            "pointwise_center": center,
            "pointwise_scale": scale,
            "pointwise_final_loss": np.asarray(final_loss, dtype=np.float64),
            "pointwise_residual_scale": np.asarray(
                POINTWISE_RESIDUAL_SCALE, dtype=np.float64
            ),
            "pointwise_schema_version": np.asarray(1, dtype=np.int64),
            "pointwise_seed": np.asarray(seed, dtype=np.uint64),
        }
    )
    return state


def _fit_reference_pointwise_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    pairs_per_epoch: int,
    pairwise_epochs: int,
    tree_count: int,
    pointwise_epochs: int,
    seed: int = POINTWISE_SEED,
) -> dict[str, np.ndarray]:
    """Fit a bounded-size form of the frozen composition for deterministic tests."""

    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ReferencePointwiseRankerError("fit seed must fit uint32")
    matrix = _legacy_features(features)
    specialist_features = _specialist_features(matrix)
    specialist = _fit_reference_categorical_ranker(
        specialist_features,
        targets,
        user_groups,
        pairs_per_epoch=pairs_per_epoch,
        pairwise_epochs=pairwise_epochs,
        tree_count=tree_count,
        seed=seed,
    )
    specialist_scores = reference_categorical_ranker_scores(specialist_features, specialist)
    checkpoint = dict(specialist)
    checkpoint.update(
        _fit_pointwise_state(
            matrix,
            targets,
            user_groups,
            specialist_scores,
            epochs=pointwise_epochs,
            seed=seed,
        )
    )
    reference_pointwise_ranker_diagnostics(checkpoint)
    return checkpoint


def train_reference_pointwise_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    seed: int = POINTWISE_SEED,
) -> dict[str, np.ndarray]:
    """Fit the frozen categorical specialist plus query-balanced pointwise correction."""

    return _fit_reference_pointwise_ranker(
        features,
        targets,
        user_groups,
        pairs_per_epoch=250000,
        pairwise_epochs=5,
        tree_count=300,
        pointwise_epochs=POINTWISE_EPOCHS,
        seed=seed,
    )


def reference_pointwise_ranker_scores(
    features: np.ndarray,
    checkpoint: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the frozen categorical-specialist plus pointwise-correction score."""

    matrix = _legacy_features(features)
    state = {name: checkpoint[name] for name in _STATE_KEYS if name in checkpoint}
    if set(state) != _STATE_KEYS:
        raise ReferencePointwiseRankerError("pointwise checkpoint inventory is invalid")
    specialist_scores = reference_categorical_ranker_scores(
        _specialist_features(matrix), _base_state(state)
    )
    parameters = tuple(
        state[name]
        for name in (
            "pointwise_weight_one",
            "pointwise_bias_one",
            "pointwise_weight_two",
            "pointwise_bias_two",
            "pointwise_weight_out",
            "pointwise_bias_out",
        )
    )
    expected_shapes = (
        (POINTWISE_INPUT_COUNT, POINTWISE_HIDDEN_ONE),
        (POINTWISE_HIDDEN_ONE,),
        (POINTWISE_HIDDEN_ONE, POINTWISE_HIDDEN_TWO),
        (POINTWISE_HIDDEN_TWO,),
        (POINTWISE_HIDDEN_TWO, 1),
        (1,),
    )
    if tuple(value.shape for value in parameters) != expected_shapes or not all(
        value.dtype == np.dtype("<f4") and bool(np.isfinite(value).all())
        for value in parameters
    ):
        raise ReferencePointwiseRankerError("pointwise network state is invalid")
    center = state["pointwise_center"]
    scale = state["pointwise_scale"]
    if (
        center.shape != (POINTWISE_INPUT_COUNT,)
        or scale.shape != center.shape
        or not bool(np.isfinite(center).all())
        or not bool(np.isfinite(scale).all())
        or bool((scale <= 0.0).any())
    ):
        raise ReferencePointwiseRankerError("pointwise normalization state is invalid")
    combined = np.empty((matrix.shape[0], POINTWISE_INPUT_COUNT), dtype=np.float32)
    combined[:, :POINTWISE_FEATURE_COUNT] = np.asarray(
        (matrix - center[:-1]) / scale[:-1], dtype=np.float32
    )
    combined[:, POINTWISE_FEATURE_COUNT] = np.asarray(
        (specialist_scores - center[-1]) / scale[-1], dtype=np.float32
    )
    raw, _ = _network(combined, parameters)
    score_scale = state["pointwise_residual_scale"]
    if score_scale.shape != () or float(score_scale.item()) != POINTWISE_RESIDUAL_SCALE:
        raise ReferencePointwiseRankerError("pointwise score scale is invalid")
    scores = specialist_scores + float(score_scale.item()) * np.tanh(raw.astype(np.float64))
    if scores.shape != (matrix.shape[0],) or not bool(np.isfinite(scores).all()):
        raise ReferencePointwiseRankerError("pointwise composed scores are invalid")
    return np.ascontiguousarray(scores, dtype=np.float64)


def reference_pointwise_ranker_diagnostics(
    checkpoint: dict[str, np.ndarray],
) -> dict[str, int | float]:
    """Return bounded non-metric diagnostics for composition and replay audit."""

    state = {name: checkpoint[name] for name in _STATE_KEYS if name in checkpoint}
    if set(state) != _STATE_KEYS:
        raise ReferencePointwiseRankerError("pointwise checkpoint inventory is invalid")
    schema = state["pointwise_schema_version"]
    loss = state["pointwise_final_loss"]
    score_scale = state["pointwise_residual_scale"]
    if any(value.shape != () for value in (schema, loss, score_scale)) or (
        int(schema.item()) != 1
        or not math.isfinite(float(loss.item()))
        or float(score_scale.item()) != POINTWISE_RESIDUAL_SCALE
    ):
        raise ReferencePointwiseRankerError("pointwise diagnostics metadata is invalid")
    return {
        **reference_categorical_ranker_diagnostics(_base_state(state)),
        "pointwise_batch_size": POINTWISE_BATCH_SIZE,
        "pointwise_epochs": POINTWISE_EPOCHS,
        "pointwise_final_loss": float(loss.item()),
        "pointwise_hidden_one": POINTWISE_HIDDEN_ONE,
        "pointwise_hidden_two": POINTWISE_HIDDEN_TWO,
        "pointwise_residual_scale": float(score_scale.item()),
    }
