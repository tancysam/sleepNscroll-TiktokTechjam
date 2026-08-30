"""Protected native-categorical LambdaRank specialist around the verified pairwise FM.

Autonomous candidate code may import and compose this module, but cannot replace it.  The
specialist was selected on train-derived Fold B and confirmed once with frozen parameters on
Fold A.  It consumes only the approved numeric feature, target, and user-group arrays and has no
scorer, metric, filesystem, network, or controller dependency.
"""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
from reference_pairwise_fm import (
    _fit_reference_pairwise_fm,
    reference_pairwise_fm_diagnostics,
    reference_pairwise_fm_scores,
)

CATEGORICAL_RANK_FEATURE_COUNT = 83
CATEGORICAL_RANK_FEATURE_POSITIONS = (51, 52, 53, 54, 55, 82)
CATEGORICAL_RANK_LEARNING_RATE = 0.05
CATEGORICAL_RANK_NUM_LEAVES = 63
CATEGORICAL_RANK_MIN_DATA_IN_LEAF = 200
CATEGORICAL_RANK_TREE_COUNT = 300
CATEGORICAL_RANK_TRUNCATION_LEVEL = 8
CATEGORICAL_RANK_CAT_SMOOTH = 20.0
CATEGORICAL_RANK_CAT_L2 = 10.0
CATEGORICAL_RANK_RESIDUAL_SHRINKAGE = 0.2
CATEGORICAL_RANK_SEED = 20260830

_REFERENCE_KEYS = {
    "reference_factors",
    "reference_feature_positions",
    "reference_final_pairwise_loss",
    "reference_linear",
    "reference_sampled_pairs",
    "reference_schema_version",
    "reference_total_dim",
    "reference_seed",
}
_CATEGORICAL_RANK_KEYS = {
    "categorical_rank_code_offsets",
    "categorical_rank_feature_count",
    "categorical_rank_model_utf8",
    "categorical_rank_residual_shrinkage",
    "categorical_rank_schema_version",
    "categorical_rank_tree_count",
    "categorical_rank_seed",
}
_STATE_KEYS = _REFERENCE_KEYS | _CATEGORICAL_RANK_KEYS


class ReferenceCategoricalRankerError(ValueError):
    """The protected categorical-ranker input or state violates its frozen contract."""


def _features(value: np.ndarray) -> np.ndarray:
    features = np.asarray(value)
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] < CATEGORICAL_RANK_FEATURE_COUNT
        or features.dtype != np.dtype("<f8")
        or not bool(np.isfinite(features).all())
    ):
        raise ReferenceCategoricalRankerError(
        "features must be finite little-endian float64 with shape (N, D), D >= 83"
        )
    return np.ascontiguousarray(features[:, :CATEGORICAL_RANK_FEATURE_COUNT])


def _local_categorical_features(
    features: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    values = _features(features)
    if (
        offsets.shape != (len(CATEGORICAL_RANK_FEATURE_POSITIONS),)
        or offsets.dtype.kind not in "iu"
    ):
        raise ReferenceCategoricalRankerError("categorical-rank code offsets are invalid")
    selected = values[:, CATEGORICAL_RANK_FEATURE_POSITIONS]
    if not bool(np.equal(selected, np.floor(selected)).all()):
        raise ReferenceCategoricalRankerError("categorical-rank codes must be integers")
    local = selected - offsets.astype(np.float64, copy=False)
    if bool((local < 0.0).any()) or float(np.max(local)) > float(np.iinfo(np.int32).max - 1):
        raise ReferenceCategoricalRankerError(
            "categorical-rank code is outside its prefix-fitted local domain"
        )
    transformed = np.array(values, dtype=np.float64, order="C", copy=True)
    transformed[:, CATEGORICAL_RANK_FEATURE_POSITIONS] = local
    return transformed


def _reference_state(checkpoint: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: checkpoint[name] for name in _REFERENCE_KEYS}


def _booster(checkpoint: dict[str, np.ndarray]) -> lgb.Booster:
    encoded = checkpoint["categorical_rank_model_utf8"]
    if encoded.ndim != 1 or encoded.dtype != np.dtype(np.uint8) or encoded.size == 0:
        raise ReferenceCategoricalRankerError("categorical-rank model encoding is invalid")
    try:
        model_text = encoded.tobytes().decode("utf-8")
        return lgb.Booster(model_str=model_text)
    except (UnicodeDecodeError, lgb.basic.LightGBMError) as exc:
        raise ReferenceCategoricalRankerError(
            "categorical-rank model cannot be restored"
        ) from exc


def _fit_reference_categorical_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    pairs_per_epoch: int,
    pairwise_epochs: int,
    tree_count: int,
    seed: int = CATEGORICAL_RANK_SEED,
) -> dict[str, np.ndarray]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ReferenceCategoricalRankerError("fit seed must fit uint32")
    matrix = _features(features)
    labels = np.asarray(targets)
    groups = np.asarray(user_groups)
    if labels.shape != (matrix.shape[0],) or groups.shape != (matrix.shape[0],):
        raise ReferenceCategoricalRankerError("targets and user_groups must align to features")
    numeric_labels = np.asarray(labels, dtype=np.float64)
    if not bool(np.isfinite(numeric_labels).all()) or not bool(
        np.isin(numeric_labels, (0.0, 1.0)).all()
    ):
        raise ReferenceCategoricalRankerError("targets must contain only binary 0 and 1")
    if (
        groups.dtype.kind not in "iuf"
        or groups.dtype.kind == "b"
        or not bool(np.isfinite(groups).all())
    ):
        raise ReferenceCategoricalRankerError("user_groups must be finite numeric values")
    if type(tree_count) is not int or not 1 <= tree_count <= CATEGORICAL_RANK_TREE_COUNT:
        raise ReferenceCategoricalRankerError("tree_count is outside the protected bound")

    reference = _fit_reference_pairwise_fm(
        matrix,
        numeric_labels,
        groups,
        pairs_per_epoch=pairs_per_epoch,
        epochs=pairwise_epochs,
        seed=seed,
    )
    reference_scores = reference_pairwise_fm_scores(matrix, reference)
    raw_categories = matrix[:, CATEGORICAL_RANK_FEATURE_POSITIONS]
    offsets = np.min(raw_categories, axis=0).astype(np.int64)
    transformed = _local_categorical_features(matrix, offsets)

    _, group_codes = np.unique(groups, return_inverse=True)
    order = np.argsort(group_codes, kind="stable")
    sorted_codes = group_codes[order]
    query_sizes = np.bincount(sorted_codes).astype(np.int32, copy=False)
    if query_sizes.size < 2 or int(query_sizes.sum()) != matrix.shape[0]:
        raise ReferenceCategoricalRankerError(
            "categorical LambdaRank requires at least two non-empty user queries"
        )
    training = lgb.Dataset(
        np.ascontiguousarray(transformed[order], dtype=np.float64),
        label=np.ascontiguousarray(numeric_labels[order], dtype=np.float64),
        group=np.ascontiguousarray(query_sizes),
        init_score=np.ascontiguousarray(reference_scores[order], dtype=np.float64),
        categorical_feature=list(CATEGORICAL_RANK_FEATURE_POSITIONS),
        free_raw_data=True,
    )
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "label_gain": [0, 1],
        "lambdarank_truncation_level": CATEGORICAL_RANK_TRUNCATION_LEVEL,
        "lambdarank_norm": True,
        "learning_rate": CATEGORICAL_RANK_LEARNING_RATE,
        "num_leaves": CATEGORICAL_RANK_NUM_LEAVES,
        "min_data_in_leaf": CATEGORICAL_RANK_MIN_DATA_IN_LEAF,
        "feature_pre_filter": False,
        "cat_smooth": CATEGORICAL_RANK_CAT_SMOOTH,
        "cat_l2": CATEGORICAL_RANK_CAT_L2,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 4,
        "verbosity": -1,
        "seed": seed,
        "data_random_seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
    }
    booster = lgb.train(params, training, num_boost_round=tree_count)
    trained_tree_count = booster.num_trees()
    model_bytes = booster.model_to_string(num_iteration=trained_tree_count).encode("utf-8")
    checkpoint = dict(reference)
    checkpoint.update(
        {
            "categorical_rank_code_offsets": np.ascontiguousarray(offsets),
            "categorical_rank_feature_count": np.asarray(
                CATEGORICAL_RANK_FEATURE_COUNT, dtype=np.int64
            ),
            "categorical_rank_model_utf8": np.frombuffer(model_bytes, dtype=np.uint8).copy(),
            "categorical_rank_residual_shrinkage": np.asarray(
                CATEGORICAL_RANK_RESIDUAL_SHRINKAGE, dtype=np.float64
            ),
            "categorical_rank_schema_version": np.asarray(1, dtype=np.int64),
            "categorical_rank_tree_count": np.asarray(trained_tree_count, dtype=np.int64),
            "categorical_rank_seed": np.asarray(seed, dtype=np.uint64),
        }
    )
    reference_categorical_ranker_diagnostics(checkpoint)
    return checkpoint


def train_reference_categorical_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    seed: int = CATEGORICAL_RANK_SEED,
) -> dict[str, np.ndarray]:
    """Fit the frozen native-categorical LambdaRank and exact pairwise-FM composition."""

    return _fit_reference_categorical_ranker(
        features,
        targets,
        user_groups,
        pairs_per_epoch=250000,
        pairwise_epochs=5,
        tree_count=CATEGORICAL_RANK_TREE_COUNT,
        seed=seed,
    )


def reference_categorical_ranker_scores(
    features: np.ndarray,
    checkpoint: dict[str, np.ndarray],
) -> np.ndarray:
    """Return exact pairwise-FM score plus the frozen native-categorical tree residual."""

    state = {name: checkpoint[name] for name in _STATE_KEYS if name in checkpoint}
    if set(state) != _STATE_KEYS:
        raise ReferenceCategoricalRankerError("categorical-rank checkpoint inventory is invalid")
    feature_count = state["categorical_rank_feature_count"]
    shrinkage = state["categorical_rank_residual_shrinkage"]
    tree_count = state["categorical_rank_tree_count"]
    schema_version = state["categorical_rank_schema_version"]
    if any(value.shape != () for value in (feature_count, shrinkage, tree_count, schema_version)):
        raise ReferenceCategoricalRankerError("categorical-rank scalar metadata is invalid")
    if (
        int(feature_count.item()) != CATEGORICAL_RANK_FEATURE_COUNT
        or int(schema_version.item()) != 1
        or int(tree_count.item()) <= 0
        or not math.isfinite(float(shrinkage.item()))
        or float(shrinkage.item()) != CATEGORICAL_RANK_RESIDUAL_SHRINKAGE
    ):
        raise ReferenceCategoricalRankerError("categorical-rank metadata is invalid")
    matrix = _local_categorical_features(
        features,
        state["categorical_rank_code_offsets"],
    )
    reference_scores = reference_pairwise_fm_scores(features, _reference_state(state))
    residual = np.asarray(
        _booster(state).predict(matrix, raw_score=True, num_iteration=int(tree_count.item())),
        dtype=np.float64,
    )
    scores = reference_scores + float(shrinkage.item()) * residual
    if scores.shape != (matrix.shape[0],) or not bool(np.isfinite(scores).all()):
        raise ReferenceCategoricalRankerError("categorical-rank scores are invalid")
    return np.ascontiguousarray(scores, dtype=np.float64)


def reference_categorical_ranker_diagnostics(
    checkpoint: dict[str, np.ndarray],
) -> dict[str, int | float]:
    """Return bounded non-metric diagnostics for composition and replay audit."""

    state = {name: checkpoint[name] for name in _STATE_KEYS if name in checkpoint}
    if set(state) != _STATE_KEYS:
        raise ReferenceCategoricalRankerError("categorical-rank checkpoint inventory is invalid")
    feature_count = state["categorical_rank_feature_count"]
    shrinkage = state["categorical_rank_residual_shrinkage"]
    tree_count = state["categorical_rank_tree_count"]
    schema_version = state["categorical_rank_schema_version"]
    offsets = state["categorical_rank_code_offsets"]
    if (
        any(value.shape != () for value in (feature_count, shrinkage, tree_count, schema_version))
        or offsets.shape != (len(CATEGORICAL_RANK_FEATURE_POSITIONS),)
        or offsets.dtype.kind not in "iu"
        or bool((offsets < 0).any())
    ):
        raise ReferenceCategoricalRankerError("categorical-rank diagnostics metadata is invalid")
    booster = _booster(state)
    seed = state["categorical_rank_seed"]
    expected_trees = int(tree_count.item())
    if (
        int(feature_count.item()) != CATEGORICAL_RANK_FEATURE_COUNT
        or int(schema_version.item()) != 1
        or expected_trees <= 0
        or booster.num_feature() != CATEGORICAL_RANK_FEATURE_COUNT
        or booster.num_trees() != expected_trees
        or float(shrinkage.item()) != CATEGORICAL_RANK_RESIDUAL_SHRINKAGE
        or seed.shape != ()
        or seed.dtype.kind not in "iu"
        or not 0 <= int(seed.item()) <= 2**32 - 1
    ):
        raise ReferenceCategoricalRankerError("categorical-rank diagnostics are inconsistent")
    return {
        **reference_pairwise_fm_diagnostics(_reference_state(state)),
        "categorical_rank_feature_count": int(feature_count.item()),
        "categorical_rank_residual_shrinkage": float(shrinkage.item()),
        "categorical_rank_tree_count": expected_trees,
    }
