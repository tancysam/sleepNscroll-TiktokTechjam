"""Protected candidate-side implementation of the verified categorical pairwise FM.

This module is part of the immutable candidate runtime surface.  Autonomous model code may
import and compose it, but cannot replace it in an overlay.  It consumes only the numeric feature,
target, and user-group arrays already approved for candidate training; it has no scorer, metric,
filesystem, network, or controller dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

REFERENCE_FEATURE_POSITIONS = (51, 52, 53, 54, 55)
REFERENCE_FACTOR_DIM = 16
REFERENCE_SEED = 20260830
REFERENCE_LEARNING_RATE = 0.001
REFERENCE_L2 = 0.000001
REFERENCE_BATCH_SIZE = 8192
REFERENCE_PAIRS_PER_EPOCH = 250000
REFERENCE_EPOCHS = 5
_MAX_SAMPLED_PAIRS = 1_000_000
_STATE_KEYS = {
    "reference_factors",
    "reference_feature_positions",
    "reference_final_pairwise_loss",
    "reference_linear",
    "reference_sampled_pairs",
    "reference_schema_version",
    "reference_total_dim",
    "reference_seed",
}


class ReferencePairwiseFMError(ValueError):
    """The protected pairwise-FM input or state violates its frozen numeric contract."""


@dataclass(frozen=True, slots=True)
class _PairBatch:
    positive_indices: np.ndarray
    negative_indices: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.positive_indices.size)


class _GAUCPairSampler:
    """Positive-ticket sampler matching organizer GAUC's positive-count user weighting."""

    def __init__(self, user_groups: np.ndarray, targets: np.ndarray) -> None:
        groups = np.asarray(user_groups)
        labels = np.asarray(targets)
        if (
            groups.ndim != 1
            or groups.size == 0
            or groups.dtype.kind not in "iuf"
            or groups.dtype.kind == "b"
            or not bool(np.isfinite(groups).all())
        ):
            raise ReferencePairwiseFMError("user_groups must be a non-empty finite numeric vector")
        if labels.shape != groups.shape or labels.dtype.kind not in "biuf":
            raise ReferencePairwiseFMError("targets must be numeric and aligned to user_groups")
        numeric_labels = np.asarray(labels, dtype=np.float64)
        if not bool(np.isfinite(numeric_labels).all()) or not bool(
            np.isin(numeric_labels, (0.0, 1.0)).all()
        ):
            raise ReferencePairwiseFMError("targets must contain only binary 0 and 1")

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
        group_count = unique_by_first_seen.size
        positive_counts = np.bincount(row_codes[binary == 1], minlength=group_count)
        negative_counts = np.bincount(row_codes[binary == 0], minlength=group_count)
        eligible = np.logical_and(positive_counts > 0, negative_counts > 0)
        if not bool(eligible.any()):
            raise ReferencePairwiseFMError("pairwise FM requires at least one mixed-label user")
        eligible_codes = np.flatnonzero(eligible)
        compact_code = np.full(group_count, -1, dtype=np.int64)
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
        self.eligible_positive_count = int(cumulative_positive_counts[-1])

    def sample(self, pair_count: int, *, seed: int) -> _PairBatch:
        if type(pair_count) is not int or not 1 <= pair_count <= _MAX_SAMPLED_PAIRS:
            raise ReferencePairwiseFMError("pair_count is outside the protected bound")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ReferencePairwiseFMError("sample seed must fit uint32")
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
        return _PairBatch(
            positive_indices=np.ascontiguousarray(positive_rows),
            negative_indices=np.ascontiguousarray(negative_rows),
        )


def sample_reference_logged_pairs(
    user_groups: np.ndarray,
    targets: np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic same-user positive/negative row indices.

    This public candidate-side seam lets new pairwise objectives reuse the exact positive-ticket
    sampler verified by the protected reference FM instead of recreating group offsets or index
    maps in generated code.
    """

    sampled = _GAUCPairSampler(user_groups, targets).sample(pair_count, seed=seed)
    return sampled.positive_indices, sampled.negative_indices


def _encoded_codes(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] <= REFERENCE_FEATURE_POSITIONS[-1]
        or values.dtype != np.dtype("<f8")
        or not bool(np.isfinite(values).all())
    ):
        raise ReferencePairwiseFMError(
            "features must be finite little-endian float64 with organizer-code columns"
        )
    raw = values[:, REFERENCE_FEATURE_POSITIONS]
    if bool((raw < 0.0).any()) or not bool(np.equal(raw, np.floor(raw)).all()):
        raise ReferencePairwiseFMError("organizer categorical codes must be non-negative integers")
    if float(np.max(raw)) > float(np.iinfo(np.int32).max - 2):
        raise ReferencePairwiseFMError("organizer categorical codes exceed int32")
    return np.ascontiguousarray(raw, dtype=np.int32)


def _fm_scores(
    encoded: np.ndarray,
    factors: np.ndarray,
    linear: np.ndarray,
) -> np.ndarray:
    embeddings = factors[encoded]
    summed = embeddings.sum(axis=1, dtype=np.float32)
    interactions = np.float32(0.5) * (
        (summed**2).sum(axis=1, dtype=np.float32)
        - (embeddings**2).sum(axis=(1, 2), dtype=np.float32)
    )
    scores = linear[encoded].sum(axis=1, dtype=np.float32) + interactions
    return np.ascontiguousarray(scores, dtype=np.float32)


def _batch_gradients(
    positive: np.ndarray,
    negative: np.ndarray,
    factors: np.ndarray,
    linear: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    positive_scores = np.asarray(_fm_scores(positive, factors, linear), dtype=np.float64)
    negative_scores = np.asarray(_fm_scores(negative, factors, linear), dtype=np.float64)
    margins = positive_scores - negative_scores
    losses = np.logaddexp(0.0, -margins)
    negative_sigmoid = np.exp(-np.logaddexp(0.0, margins))
    positive_gradient = np.asarray(
        -(negative_sigmoid / float(positive.shape[0])),
        dtype=np.float32,
    )
    negative_gradient = -positive_gradient

    factor_gradient = np.multiply(
        factors,
        np.float32(REFERENCE_L2),
        dtype=np.float32,
    )
    linear_gradient = np.multiply(
        linear,
        np.float32(REFERENCE_L2),
        dtype=np.float32,
    )
    for matrix, score_gradient in (
        (positive, positive_gradient),
        (negative, negative_gradient),
    ):
        embeddings = factors[matrix]
        summed = embeddings.sum(axis=1, dtype=np.float32)
        row_gradients = score_gradient[:, None, None] * (summed[:, None, :] - embeddings)
        np.add.at(factor_gradient, matrix, row_gradients)
        np.add.at(linear_gradient, matrix, score_gradient[:, None])
    loss = float(np.mean(losses, dtype=np.float64))
    if (
        not math.isfinite(loss)
        or not bool(np.isfinite(factor_gradient).all())
        or not bool(np.isfinite(linear_gradient).all())
    ):
        raise ReferencePairwiseFMError("pairwise FM loss or gradient became non-finite")
    return loss, factor_gradient, linear_gradient


def _fit_reference_pairwise_fm(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    pairs_per_epoch: int,
    epochs: int,
    seed: int = REFERENCE_SEED,
) -> dict[str, np.ndarray]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ReferencePairwiseFMError("fit seed must fit uint32")
    encoded = _encoded_codes(features)
    labels = np.asarray(targets)
    groups = np.asarray(user_groups)
    if labels.shape != (encoded.shape[0],) or groups.shape != (encoded.shape[0],):
        raise ReferencePairwiseFMError("targets and user_groups must align to features")
    sampler = _GAUCPairSampler(groups, labels)
    total_dim = int(np.max(encoded)) + 2
    rng = np.random.default_rng(seed)
    factors = rng.normal(
        0.0,
        0.01,
        size=(total_dim, REFERENCE_FACTOR_DIM),
    ).astype(np.float32)
    linear = np.zeros(total_dim, dtype=np.float32)
    factor_first_moment = np.zeros_like(factors)
    factor_second_moment = np.zeros_like(factors)
    linear_first_moment = np.zeros_like(linear)
    linear_second_moment = np.zeros_like(linear)
    sampling_rng = np.random.default_rng(seed)
    optimizer_steps = 0
    final_pairwise_loss = math.nan

    for _ in range(epochs):
        sample_seed = int(sampling_rng.integers(0, 2**32, dtype=np.uint64))
        sampled = sampler.sample(pairs_per_epoch, seed=sample_seed)
        weighted_loss = 0.0
        for start in range(0, sampled.pair_count, REFERENCE_BATCH_SIZE):
            stop = min(start + REFERENCE_BATCH_SIZE, sampled.pair_count)
            positive = encoded[sampled.positive_indices[start:stop]]
            negative = encoded[sampled.negative_indices[start:stop]]
            loss, factor_gradient, linear_gradient = _batch_gradients(
                positive,
                negative,
                factors,
                linear,
            )
            batch_size = stop - start
            weighted_loss += loss * batch_size
            optimizer_steps += 1
            for parameter, gradient, first_moment, second_moment in (
                (
                    factors,
                    factor_gradient,
                    factor_first_moment,
                    factor_second_moment,
                ),
                (
                    linear,
                    linear_gradient,
                    linear_first_moment,
                    linear_second_moment,
                ),
            ):
                first_moment *= np.float32(0.9)
                first_moment += np.float32(0.1) * gradient
                second_moment *= np.float32(0.999)
                second_moment += np.float32(0.001) * (gradient * gradient)
                parameter -= (
                    REFERENCE_LEARNING_RATE
                    * (first_moment / (1.0 - 0.9**optimizer_steps))
                    / (np.sqrt(second_moment / (1.0 - 0.999**optimizer_steps)) + 1e-8)
                )
            if not bool(np.isfinite(factors).all()) or not bool(np.isfinite(linear).all()):
                raise ReferencePairwiseFMError("pairwise FM optimizer produced non-finite state")
        final_pairwise_loss = weighted_loss / sampled.pair_count

    return {
        "reference_factors": np.ascontiguousarray(factors, dtype=np.float32),
        "reference_feature_positions": np.asarray(
            REFERENCE_FEATURE_POSITIONS,
            dtype=np.int64,
        ),
        "reference_final_pairwise_loss": np.asarray(
            final_pairwise_loss,
            dtype=np.float64,
        ),
        "reference_linear": np.ascontiguousarray(linear, dtype=np.float32),
        "reference_sampled_pairs": np.asarray(pairs_per_epoch * epochs, dtype=np.int64),
        "reference_schema_version": np.asarray(1, dtype=np.int64),
        "reference_total_dim": np.asarray(total_dim, dtype=np.int64),
        "reference_seed": np.asarray(seed, dtype=np.uint64),
    }


def train_reference_pairwise_fm(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    seed: int = REFERENCE_SEED,
) -> dict[str, np.ndarray]:
    """Fit the exact inner-fold-verified pairwise FM with no exposed tunables."""

    return _fit_reference_pairwise_fm(
        features,
        targets,
        user_groups,
        pairs_per_epoch=REFERENCE_PAIRS_PER_EPOCH,
        epochs=REFERENCE_EPOCHS,
        seed=seed,
    )


def reference_pairwise_fm_scores(
    features: np.ndarray,
    checkpoint: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the frozen reference FM score from numeric inference features and state."""

    if set(checkpoint) != _STATE_KEYS:
        raise ReferencePairwiseFMError("reference checkpoint inventory is invalid")
    positions = checkpoint["reference_feature_positions"]
    total_dim_raw = checkpoint["reference_total_dim"]
    seed_raw = checkpoint["reference_seed"]
    factors = checkpoint["reference_factors"]
    linear = checkpoint["reference_linear"]
    if (
        positions.shape != (5,)
        or positions.dtype.kind not in "iu"
        or tuple(int(value) for value in positions) != REFERENCE_FEATURE_POSITIONS
        or total_dim_raw.shape != ()
        or total_dim_raw.dtype.kind not in "iu"
        or seed_raw.shape != ()
        or seed_raw.dtype.kind not in "iu"
        or not 0 <= int(seed_raw.item()) <= 2**32 - 1
    ):
        raise ReferencePairwiseFMError("reference checkpoint metadata is invalid")
    total_dim = int(total_dim_raw)
    if (
        total_dim <= 0
        or factors.dtype != np.dtype("<f4")
        or factors.shape != (total_dim, REFERENCE_FACTOR_DIM)
        or linear.dtype != np.dtype("<f4")
        or linear.shape != (total_dim,)
        or not bool(np.isfinite(factors).all())
        or not bool(np.isfinite(linear).all())
    ):
        raise ReferencePairwiseFMError("reference checkpoint parameters are invalid")
    encoded = _encoded_codes(features)
    if int(np.max(encoded)) >= total_dim:
        raise ReferencePairwiseFMError("inference organizer code exceeds fitted reference state")
    return np.ascontiguousarray(
        _fm_scores(encoded, factors, linear),
        dtype=np.float64,
    )


def reference_pairwise_fm_diagnostics(
    checkpoint: dict[str, np.ndarray],
) -> dict[str, int | float]:
    """Expose bounded non-metric diagnostics for autonomous composition and audit."""

    # Reuse prediction-state validation without requiring an inference matrix.
    if set(checkpoint) != _STATE_KEYS:
        raise ReferencePairwiseFMError("reference checkpoint inventory is invalid")
    loss = checkpoint["reference_final_pairwise_loss"]
    sampled = checkpoint["reference_sampled_pairs"]
    total_dim = checkpoint["reference_total_dim"]
    if loss.shape != () or sampled.shape != () or total_dim.shape != ():
        raise ReferencePairwiseFMError("reference checkpoint diagnostics are invalid")
    final_loss = float(loss)
    if not math.isfinite(final_loss):
        raise ReferencePairwiseFMError("reference final pairwise loss is non-finite")
    return {
        "reference_epochs": REFERENCE_EPOCHS,
        "reference_factor_dim": REFERENCE_FACTOR_DIM,
        "reference_final_pairwise_loss": final_loss,
        "reference_pairs_per_epoch": REFERENCE_PAIRS_PER_EPOCH,
        "reference_sampled_pairs": int(sampled),
        "reference_total_dim": int(total_dim),
    }
