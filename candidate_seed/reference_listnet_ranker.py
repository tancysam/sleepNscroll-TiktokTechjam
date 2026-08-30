"""Protected user-balanced ListNet composition over the categorical specialist.

Autonomous candidate code may import and compose this module, but cannot replace it. The exact
recipe was selected on train-derived Fold B and confirmed once with frozen parameters on Fold A
during Attempt 14. It consumes only approved numeric feature, target, and user-group arrays and
has no scorer, metric, filesystem, network, or controller dependency.
"""

from __future__ import annotations

import math

import numpy as np
from reference_categorical_ranker import (
    _fit_reference_categorical_ranker,
    reference_categorical_ranker_diagnostics,
    reference_categorical_ranker_scores,
)

LISTNET_FEATURE_COUNT = 82
LISTNET_INPUT_COUNT = 83
LISTNET_HIDDEN_ONE = 192
LISTNET_HIDDEN_TWO = 64
LISTNET_EPOCHS = 8
LISTNET_LEARNING_RATE = 0.001
LISTNET_WEIGHT_DECAY = 0.00001
LISTNET_GRADIENT_CLIP = 5.0
LISTNET_QUERY_BATCH_SIZE = 128
LISTNET_SCORE_SCALE = 0.15
LISTNET_LOGIT_CLIP = 20.0
LISTNET_SEED = 0

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
_LISTNET_KEYS = {
    "listwise_bias_one",
    "listwise_bias_out",
    "listwise_bias_two",
    "listwise_center",
    "listwise_eligible_queries",
    "listwise_final_loss",
    "listwise_residual_scale",
    "listwise_scale",
    "listwise_schema_version",
    "listwise_weight_one",
    "listwise_weight_out",
    "listwise_weight_two",
    "listwise_seed",
}
_STATE_KEYS = _BASE_KEYS | _LISTNET_KEYS


class ReferenceListNetRankerError(ValueError):
    """The protected ListNet-ranker input or state violates its frozen contract."""


def _legacy_features(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] < LISTNET_FEATURE_COUNT
        or matrix.dtype != np.dtype("<f8")
        or not bool(np.isfinite(matrix).all())
    ):
        raise ReferenceListNetRankerError(
            "features must be finite little-endian float64 with shape (N, D), D >= 82"
        )
    return np.ascontiguousarray(matrix[:, :LISTNET_FEATURE_COUNT])


def _base_state(checkpoint: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: checkpoint[name] for name in _BASE_KEYS}


def _specialist_features(matrix: np.ndarray) -> np.ndarray:
    """Keep the legacy ListNet 82-column surface while satisfying the 83-column specialist."""
    return np.pad(matrix, ((0, 0), (0, 1)), constant_values=0.0)


def _gelu(value: np.ndarray) -> np.ndarray:
    coefficient = np.float64(math.sqrt(2.0 / math.pi))
    inner = coefficient * (value + np.float64(0.044715) * value**3)
    return np.float64(0.5) * value * (np.float64(1.0) + np.tanh(inner))


def _gelu_gradient(value: np.ndarray) -> np.ndarray:
    coefficient = np.float64(math.sqrt(2.0 / math.pi))
    inner = coefficient * (value + np.float64(0.044715) * value**3)
    tangent = np.tanh(inner)
    inner_gradient = coefficient * (np.float64(1.0) + np.float64(0.134145) * value**2)
    return (
        np.float64(0.5) * (np.float64(1.0) + tangent)
        + np.float64(0.5) * value * (np.float64(1.0) - tangent**2) * inner_gradient
    )


def _network(
    inputs: np.ndarray,
    parameters: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    weight_one, bias_one, weight_two, bias_two, weight_out, bias_out = parameters
    pre_one = inputs @ weight_one + bias_one
    hidden_one = _gelu(pre_one)
    pre_two = hidden_one @ weight_two + bias_two
    hidden_two = _gelu(pre_two)
    raw = (hidden_two @ weight_out + bias_out).reshape(-1)
    bounded = np.tanh(raw)
    return bounded, (inputs, pre_one, hidden_one, pre_two, hidden_two, raw)


def _standardized_inputs(
    features: np.ndarray,
    specialist_scores: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    matrix = _legacy_features(features)
    if specialist_scores.shape != (matrix.shape[0],) or not bool(
        np.isfinite(specialist_scores).all()
    ):
        raise ReferenceListNetRankerError("specialist scores do not align to features")
    if (
        center.shape != (LISTNET_INPUT_COUNT,)
        or scale.shape != (LISTNET_INPUT_COUNT,)
        or not bool(np.isfinite(center).all())
        or not bool(np.isfinite(scale).all())
        or bool((scale <= 0.0).any())
    ):
        raise ReferenceListNetRankerError("ListNet normalization state is invalid")
    combined = np.empty((matrix.shape[0], LISTNET_INPUT_COUNT), dtype=np.float64)
    combined[:, :LISTNET_FEATURE_COUNT] = matrix
    combined[:, LISTNET_FEATURE_COUNT] = specialist_scores
    standardized = (combined - center) / scale
    return np.ascontiguousarray(np.clip(standardized, -8.0, 8.0), dtype=np.float64)


def _eligible_queries(
    user_groups: np.ndarray,
    targets: np.ndarray,
) -> list[np.ndarray]:
    _, first, inverse = np.unique(user_groups, return_index=True, return_inverse=True)
    first_order = np.argsort(first, kind="stable")
    remap = np.empty(first_order.size, dtype=np.int64)
    remap[first_order] = np.arange(first_order.size, dtype=np.int64)
    codes = remap[inverse]
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    return [
        rows
        for rows in np.split(order, boundaries)
        if bool((targets[rows] == 1.0).any()) and bool((targets[rows] == 0.0).any())
    ]


def _fit_listnet_state(
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
        raise ReferenceListNetRankerError("targets and user_groups must align to features")
    if not bool(np.isfinite(labels).all()) or not bool(np.isin(labels, (0.0, 1.0)).all()):
        raise ReferenceListNetRankerError("targets must contain only binary 0 and 1")
    if (
        groups.dtype.kind not in "iuf"
        or groups.dtype.kind == "b"
        or not bool(np.isfinite(groups).all())
    ):
        raise ReferenceListNetRankerError("user_groups must be finite numeric values")
    if type(epochs) is not int or not 1 <= epochs <= LISTNET_EPOCHS:
        raise ReferenceListNetRankerError("ListNet epochs are outside the protected bound")

    combined = np.empty((matrix.shape[0], LISTNET_INPUT_COUNT), dtype=np.float64)
    combined[:, :LISTNET_FEATURE_COUNT] = matrix
    combined[:, LISTNET_FEATURE_COUNT] = specialist_scores
    center = np.median(combined, axis=0)
    lower = np.quantile(combined, 0.25, axis=0)
    upper = np.quantile(combined, 0.75, axis=0)
    scale = np.where(upper - lower > 1e-12, upper - lower, 1.0)
    inputs = _standardized_inputs(matrix, specialist_scores, center, scale)
    eligible = _eligible_queries(groups, labels)
    if not eligible:
        raise ReferenceListNetRankerError("ListNet requires a mixed-label user")

    rng = np.random.default_rng(seed)
    parameters = (
        rng.normal(
            0.0,
            math.sqrt(2.0 / LISTNET_INPUT_COUNT),
            (LISTNET_INPUT_COUNT, LISTNET_HIDDEN_ONE),
        ),
        np.zeros(LISTNET_HIDDEN_ONE, dtype=np.float64),
        rng.normal(
            0.0,
            math.sqrt(2.0 / LISTNET_HIDDEN_ONE),
            (LISTNET_HIDDEN_ONE, LISTNET_HIDDEN_TWO),
        ),
        np.zeros(LISTNET_HIDDEN_TWO, dtype=np.float64),
        rng.normal(0.0, math.sqrt(2.0 / LISTNET_HIDDEN_TWO), (LISTNET_HIDDEN_TWO, 1)),
        np.zeros(1, dtype=np.float64),
    )
    parameters = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in parameters)
    first_moments = tuple(np.zeros_like(value) for value in parameters)
    second_moments = tuple(np.zeros_like(value) for value in parameters)
    best_parameters = tuple(value.copy() for value in parameters)
    best_loss = math.inf
    step = 0

    for _ in range(epochs):
        permutation = rng.permutation(len(eligible))
        epoch_loss = 0.0
        for batch_start in range(0, len(eligible), LISTNET_QUERY_BATCH_SIZE):
            selected = permutation[batch_start : batch_start + LISTNET_QUERY_BATCH_SIZE]
            row_blocks = [eligible[int(index)] for index in selected]
            rows = np.concatenate(row_blocks)
            bounded, cache = _network(inputs[rows], parameters)
            score_gradient = np.zeros(rows.size, dtype=np.float64)
            offset = 0
            batch_loss = 0.0
            for query_rows in row_blocks:
                size = query_rows.size
                local = np.clip(
                    specialist_scores[query_rows]
                    + LISTNET_SCORE_SCALE * bounded[offset : offset + size],
                    -LISTNET_LOGIT_CLIP,
                    LISTNET_LOGIT_CLIP,
                )
                shifted = local - np.max(local)
                probability = np.exp(shifted)
                probability /= np.sum(probability)
                positive_count = np.sum(labels[query_rows], dtype=np.float64)
                target_distribution = labels[query_rows] / positive_count
                batch_loss -= float(
                    np.dot(target_distribution, np.log(np.maximum(probability, 1e-12)))
                )
                score_gradient[offset : offset + size] = probability - target_distribution
                offset += size
            query_count = len(row_blocks)
            epoch_loss += batch_loss
            score_gradient *= LISTNET_SCORE_SCALE * (1.0 - bounded**2) / float(query_count)

            batch_inputs, pre_one, hidden_one, pre_two, hidden_two, _ = cache
            _, _, weight_two, _, weight_out, _ = parameters
            output_gradient = score_gradient[:, None]
            gradient_out = hidden_two.T @ output_gradient
            gradient_bias_out = np.sum(output_gradient, axis=0)
            gradient_hidden_two = output_gradient @ weight_out.T
            gradient_pre_two = gradient_hidden_two * _gelu_gradient(pre_two)
            gradient_weight_two = hidden_one.T @ gradient_pre_two
            gradient_bias_two = np.sum(gradient_pre_two, axis=0)
            gradient_hidden_one = gradient_pre_two @ weight_two.T
            gradient_pre_one = gradient_hidden_one * _gelu_gradient(pre_one)
            gradient_weight_one = batch_inputs.T @ gradient_pre_one
            gradient_bias_one = np.sum(gradient_pre_one, axis=0)
            gradients = (
                gradient_weight_one,
                gradient_bias_one,
                gradient_weight_two,
                gradient_bias_two,
                gradient_out,
                gradient_bias_out,
            )
            gradients = tuple(
                gradient + LISTNET_WEIGHT_DECAY * parameter
                if gradient.ndim == 2
                else gradient
                for gradient, parameter in zip(gradients, parameters, strict=True)
            )
            norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
            if norm > LISTNET_GRADIENT_CLIP:
                multiplier = LISTNET_GRADIENT_CLIP / norm
                gradients = tuple(gradient * multiplier for gradient in gradients)
            step += 1
            for parameter, gradient, first_moment, second_moment in zip(
                parameters,
                gradients,
                first_moments,
                second_moments,
                strict=True,
            ):
                first_moment *= 0.9
                first_moment += 0.1 * gradient
                second_moment *= 0.999
                second_moment += 0.001 * gradient**2
                parameter -= LISTNET_LEARNING_RATE * (
                    first_moment / (1.0 - 0.9**step)
                ) / (np.sqrt(second_moment / (1.0 - 0.999**step)) + 1e-8)
        epoch_loss /= float(len(eligible))
        if not math.isfinite(epoch_loss) or not all(
            bool(np.isfinite(parameter).all()) for parameter in parameters
        ):
            raise ReferenceListNetRankerError("ListNet optimizer produced non-finite state")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_parameters = tuple(value.copy() for value in parameters)

    names = ("weight_one", "bias_one", "weight_two", "bias_two", "weight_out", "bias_out")
    state = {
        f"listwise_{name}": np.ascontiguousarray(value, dtype=np.float64)
        for name, value in zip(names, best_parameters, strict=True)
    }
    state.update(
        {
            "listwise_center": np.ascontiguousarray(center, dtype=np.float64),
            "listwise_scale": np.ascontiguousarray(scale, dtype=np.float64),
            "listwise_final_loss": np.asarray(best_loss, dtype=np.float64),
            "listwise_eligible_queries": np.asarray(len(eligible), dtype=np.int64),
            "listwise_residual_scale": np.asarray(LISTNET_SCORE_SCALE, dtype=np.float64),
            "listwise_schema_version": np.asarray(1, dtype=np.int64),
            "listwise_seed": np.asarray(seed, dtype=np.uint64),
        }
    )
    return state


def _fit_reference_listnet_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    pairs_per_epoch: int,
    pairwise_epochs: int,
    tree_count: int,
    listnet_epochs: int,
    seed: int = LISTNET_SEED,
) -> dict[str, np.ndarray]:
    """Fit a bounded-size form of the frozen composition for deterministic tests."""

    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ReferenceListNetRankerError("fit seed must fit uint32")
    specialist_features = _specialist_features(_legacy_features(features))
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
        _fit_listnet_state(
            features,
            targets,
            user_groups,
            specialist_scores,
            epochs=listnet_epochs,
            seed=seed,
        )
    )
    reference_listnet_ranker_diagnostics(checkpoint)
    return checkpoint


def train_reference_listnet_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    seed: int = LISTNET_SEED,
) -> dict[str, np.ndarray]:
    """Fit the frozen categorical specialist plus user-balanced ListNet composition."""

    return _fit_reference_listnet_ranker(
        features,
        targets,
        user_groups,
        pairs_per_epoch=250000,
        pairwise_epochs=5,
        tree_count=300,
        listnet_epochs=LISTNET_EPOCHS,
        seed=seed,
    )


def reference_listnet_ranker_scores(
    features: np.ndarray,
    checkpoint: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the frozen pairwise-FM, categorical-tree, and ListNet composed score."""

    state = {name: checkpoint[name] for name in _STATE_KEYS if name in checkpoint}
    if set(state) != _STATE_KEYS:
        raise ReferenceListNetRankerError("ListNet checkpoint inventory is invalid")
    specialist_scores = reference_categorical_ranker_scores(
        _specialist_features(_legacy_features(features)), _base_state(state)
    )
    parameters = tuple(
        state[name]
        for name in (
            "listwise_weight_one",
            "listwise_bias_one",
            "listwise_weight_two",
            "listwise_bias_two",
            "listwise_weight_out",
            "listwise_bias_out",
        )
    )
    expected_shapes = (
        (LISTNET_INPUT_COUNT, LISTNET_HIDDEN_ONE),
        (LISTNET_HIDDEN_ONE,),
        (LISTNET_HIDDEN_ONE, LISTNET_HIDDEN_TWO),
        (LISTNET_HIDDEN_TWO,),
        (LISTNET_HIDDEN_TWO, 1),
        (1,),
    )
    if tuple(value.shape for value in parameters) != expected_shapes or not all(
        value.dtype == np.dtype("<f8") and bool(np.isfinite(value).all())
        for value in parameters
    ):
        raise ReferenceListNetRankerError("ListNet network state is invalid")
    inputs = _standardized_inputs(
        features,
        specialist_scores,
        state["listwise_center"],
        state["listwise_scale"],
    )
    bounded, _ = _network(inputs, parameters)
    scale = state["listwise_residual_scale"]
    if scale.shape != () or float(scale.item()) != LISTNET_SCORE_SCALE:
        raise ReferenceListNetRankerError("ListNet score scale is invalid")
    scores = specialist_scores + float(scale.item()) * bounded
    if scores.shape != (inputs.shape[0],) or not bool(np.isfinite(scores).all()):
        raise ReferenceListNetRankerError("ListNet composed scores are invalid")
    return np.ascontiguousarray(scores, dtype=np.float64)


def reference_listnet_ranker_diagnostics(
    checkpoint: dict[str, np.ndarray],
) -> dict[str, int | float]:
    """Return bounded non-metric diagnostics for composition and replay audit."""

    state = {name: checkpoint[name] for name in _STATE_KEYS if name in checkpoint}
    if set(state) != _STATE_KEYS:
        raise ReferenceListNetRankerError("ListNet checkpoint inventory is invalid")
    schema = state["listwise_schema_version"]
    loss = state["listwise_final_loss"]
    eligible = state["listwise_eligible_queries"]
    score_scale = state["listwise_residual_scale"]
    if any(value.shape != () for value in (schema, loss, eligible, score_scale)) or (
        int(schema.item()) != 1
        or int(eligible.item()) <= 0
        or not math.isfinite(float(loss.item()))
        or float(score_scale.item()) != LISTNET_SCORE_SCALE
    ):
        raise ReferenceListNetRankerError("ListNet diagnostics metadata is invalid")
    # Validate the complete network state without requiring an inference matrix.
    parameters = tuple(state[name] for name in sorted(_LISTNET_KEYS) if "weight" in name)
    if not parameters or not all(bool(np.isfinite(value).all()) for value in parameters):
        raise ReferenceListNetRankerError("ListNet diagnostics state is non-finite")
    return {
        **reference_categorical_ranker_diagnostics(_base_state(state)),
        "listnet_eligible_queries": int(eligible.item()),
        "listnet_epochs": LISTNET_EPOCHS,
        "listnet_final_loss": float(loss.item()),
        "listnet_hidden_one": LISTNET_HIDDEN_ONE,
        "listnet_hidden_two": LISTNET_HIDDEN_TWO,
        "listnet_score_scale": float(score_scale.item()),
    }
