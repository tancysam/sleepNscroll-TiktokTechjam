"""Deterministic GAUC-aligned pairwise sampling and objective primitives.

The organizer GAUC weights a user's AUC by that user's positive count.  Sampling an
eligible user in proportion to its positive count, then a positive and negative uniformly
within that user, therefore assigns each positive-negative pair probability
``1 / (sum_eligible(P_u) * N_u)``.  The production sampler below stores only row indices
partitioned by user; Cartesian pair enumeration exists solely in the explicitly bounded
fixture oracle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.data.capabilities import DataPhase

type VectorInput = Sequence[object] | npt.NDArray[np.generic]
type UserId = int | str
type Int64Vector = npt.NDArray[np.int64]
type Float64Vector = npt.NDArray[np.float64]

DEFAULT_MAX_ORACLE_PAIRS: Final = 100_000
MAX_SAMPLED_PAIRS: Final = 1_000_000


class PairwisePrimitiveError(ValueError):
    """Raised when pairwise inputs violate the train-only scientific contract."""


def _require_training_phase(phase: DataPhase) -> None:
    # Check this before touching label-bearing input.  In particular, callers cannot use this
    # primitive to validate, convert, or summarize an outer-validation or final label vector.
    if not isinstance(phase, DataPhase):
        raise PairwisePrimitiveError("phase must be a DataPhase")
    if phase not in {DataPhase.TRAIN, DataPhase.INNER_TRAIN}:
        raise PairwisePrimitiveError("pairwise labels are allowed only for train or inner_train")


def _vector(value: VectorInput, name: str) -> npt.NDArray[np.generic]:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PairwisePrimitiveError(f"{name} must be a one-dimensional vector") from exc
    if result.ndim != 1:
        raise PairwisePrimitiveError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise PairwisePrimitiveError(f"{name} cannot be empty")
    return result


def _user_id(value: object, location: str) -> UserId:
    if type(value) is bool:
        raise PairwisePrimitiveError(f"{location} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value and "\x00" not in value:
        return value
    raise PairwisePrimitiveError(f"{location} must be an integer or non-empty string")


def _training_rows(
    user_ids: VectorInput,
    labels: VectorInput,
    *,
    phase: DataPhase,
) -> tuple[tuple[UserId, ...], npt.NDArray[np.int8]]:
    _require_training_phase(phase)
    users_raw = _vector(user_ids, "user_ids")
    labels_raw = _vector(labels, "labels")
    if users_raw.size != labels_raw.size:
        raise PairwisePrimitiveError(
            f"user_ids and labels must have equal lengths; got {users_raw.size} and "
            f"{labels_raw.size}"
        )
    if labels_raw.dtype.kind not in "biuf":
        raise PairwisePrimitiveError("labels must be numeric binary values")
    try:
        labels_float = np.asarray(labels_raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PairwisePrimitiveError("labels must be numeric binary values") from exc
    if not np.isfinite(labels_float).all() or not np.isin(labels_float, (0.0, 1.0)).all():
        raise PairwisePrimitiveError("labels must contain only binary 0 and 1 values")
    normalized_users = tuple(
        _user_id(value, f"user_ids[{index}]") for index, value in enumerate(users_raw.tolist())
    )
    normalized_labels = np.ascontiguousarray(labels_float, dtype=np.int8)
    normalized_labels.setflags(write=False)
    return normalized_users, normalized_labels


@dataclass(frozen=True, slots=True)
class PairDistribution:
    """Exact tiny-fixture pair distribution returned by the bounded oracle."""

    positive_indices: Int64Vector = field(repr=False)
    negative_indices: Int64Vector = field(repr=False)
    probabilities: Float64Vector = field(repr=False)

    @property
    def pair_count(self) -> int:
        return int(self.probabilities.size)


@dataclass(frozen=True, slots=True)
class _EligibleGroup:
    positive_indices: Int64Vector = field(repr=False)
    negative_indices: Int64Vector = field(repr=False)


def _eligible_groups(
    users: tuple[UserId, ...], targets: npt.NDArray[np.int8]
) -> tuple[_EligibleGroup, ...]:
    grouped: dict[UserId, tuple[list[int], list[int]]] = {}
    for row_index, (user, target) in enumerate(zip(users, targets, strict=True)):
        positives, negatives = grouped.setdefault(user, ([], []))
        (positives if int(target) == 1 else negatives).append(row_index)
    result: list[_EligibleGroup] = []
    for positive, negative in grouped.values():
        if not positive or not negative:
            continue
        positive_array = np.asarray(positive, dtype=np.int64)
        negative_array = np.asarray(negative, dtype=np.int64)
        positive_array.setflags(write=False)
        negative_array.setflags(write=False)
        result.append(
            _EligibleGroup(
                positive_indices=cast(Int64Vector, positive_array),
                negative_indices=cast(Int64Vector, negative_array),
            )
        )
    if not result:
        raise PairwisePrimitiveError("pairwise sampling requires at least one mixed-label user")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PairBatch:
    """One seeded batch of same-user positive and negative row positions."""

    positive_indices: Int64Vector = field(repr=False)
    negative_indices: Int64Vector = field(repr=False)
    user_group_indices: Int64Vector = field(repr=False)

    @property
    def pair_count(self) -> int:
        return int(self.positive_indices.size)


@dataclass(frozen=True, slots=True)
class PairwiseLossResult:
    """Weighted mean logistic loss and gradients with respect to both scores."""

    loss: float
    positive_gradient: Float64Vector = field(repr=False)
    negative_gradient: Float64Vector = field(repr=False)


def _finite_float_vector(value: VectorInput, name: str) -> Float64Vector:
    raw = _vector(value, name)
    if raw.dtype.kind not in "iuf":
        raise PairwisePrimitiveError(f"{name} must contain finite real numbers")
    try:
        result = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PairwisePrimitiveError(f"{name} must contain finite real numbers") from exc
    if not np.isfinite(result).all():
        raise PairwisePrimitiveError(f"{name} must contain finite real numbers")
    return result


def pairwise_logistic_loss_and_gradient(
    positive_scores: VectorInput,
    negative_scores: VectorInput,
    *,
    weights: VectorInput | None = None,
) -> PairwiseLossResult:
    """Return stable weighted ``softplus(-(positive - negative))`` and score gradients.

    Weights are normalized to sum to one, so omitted weights yield the ordinary batch mean.
    Gradients are with respect to each positive and negative score, not merely the margin.
    """

    positive = _finite_float_vector(positive_scores, "positive_scores")
    negative = _finite_float_vector(negative_scores, "negative_scores")
    if positive.size != negative.size:
        raise PairwisePrimitiveError(
            "positive_scores and negative_scores must have equal lengths; "
            f"got {positive.size} and {negative.size}"
        )
    if weights is None:
        normalized_weights = np.full(positive.size, 1.0 / positive.size, dtype=np.float64)
    else:
        raw_weights = _finite_float_vector(weights, "weights")
        if raw_weights.size != positive.size:
            raise PairwisePrimitiveError(
                f"weights must have length {positive.size}, got {raw_weights.size}"
            )
        if np.any(raw_weights < 0.0) or not np.any(raw_weights > 0.0):
            raise PairwisePrimitiveError("weights must be non-negative with a positive sum")
        scaled_weights = raw_weights / float(raw_weights.max())
        normalized_weights = scaled_weights / scaled_weights.sum(dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        margins = positive - negative
    if not np.isfinite(margins).all():
        raise PairwisePrimitiveError("score margins must be representable as finite float64")
    losses = np.logaddexp(0.0, -margins)
    negative_sigmoid = np.exp(-np.logaddexp(0.0, margins))
    positive_gradient = -normalized_weights * negative_sigmoid
    negative_gradient = -positive_gradient
    loss = float(np.dot(normalized_weights, losses))
    if not np.isfinite(loss):
        raise PairwisePrimitiveError("weighted pairwise loss is not finite")
    for array in (positive_gradient, negative_gradient):
        array.setflags(write=False)
    return PairwiseLossResult(
        loss=loss,
        positive_gradient=positive_gradient,
        negative_gradient=negative_gradient,
    )


@dataclass(frozen=True, slots=True, init=False)
class GAUCPairSampler:
    """Bounded-memory sampler implementing the organizer-GAUC pair distribution.

    Construction stores each eligible row index exactly once.  :meth:`sample` allocates only
    arrays proportional to the requested batch size, regardless of the Cartesian pair-space
    size.
    """

    _positive_indices: Int64Vector = field(repr=False)
    _negative_indices: Int64Vector = field(repr=False)
    _cumulative_positive_counts: Int64Vector = field(repr=False)
    _negative_offsets: Int64Vector = field(repr=False)
    _negative_counts: Int64Vector = field(repr=False)
    eligible_user_count: int
    eligible_positive_count: int
    stored_row_index_count: int
    pair_space_size: int

    def __init__(
        self,
        user_ids: VectorInput,
        labels: VectorInput,
        *,
        phase: DataPhase,
    ) -> None:
        users, targets = _training_rows(user_ids, labels, phase=phase)
        groups = _eligible_groups(users, targets)
        positive_counts = np.fromiter(
            (group.positive_indices.size for group in groups), dtype=np.int64, count=len(groups)
        )
        negative_counts = np.fromiter(
            (group.negative_indices.size for group in groups), dtype=np.int64, count=len(groups)
        )
        positive_indices = np.concatenate(tuple(group.positive_indices for group in groups))
        negative_indices = np.concatenate(tuple(group.negative_indices for group in groups))
        cumulative = np.cumsum(positive_counts, dtype=np.int64)
        negative_offsets = np.empty(len(groups), dtype=np.int64)
        negative_offsets[0] = 0
        if len(groups) > 1:
            np.cumsum(negative_counts[:-1], dtype=np.int64, out=negative_offsets[1:])
        for array in (
            positive_indices,
            negative_indices,
            cumulative,
            negative_offsets,
            negative_counts,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "_positive_indices", positive_indices)
        object.__setattr__(self, "_negative_indices", negative_indices)
        object.__setattr__(self, "_cumulative_positive_counts", cast(Int64Vector, cumulative))
        object.__setattr__(self, "_negative_offsets", cast(Int64Vector, negative_offsets))
        object.__setattr__(self, "_negative_counts", cast(Int64Vector, negative_counts))
        object.__setattr__(self, "eligible_user_count", len(groups))
        object.__setattr__(self, "eligible_positive_count", int(cumulative[-1]))
        object.__setattr__(
            self,
            "stored_row_index_count",
            int(positive_indices.size + negative_indices.size),
        )
        object.__setattr__(
            self,
            "pair_space_size",
            sum(
                int(group.positive_indices.size) * int(group.negative_indices.size)
                for group in groups
            ),
        )

    def sample(self, pair_count: int, *, seed: int) -> PairBatch:
        """Draw one reproducible batch without materializing a user's Cartesian pairs."""

        if type(pair_count) is not int or not 1 <= pair_count <= MAX_SAMPLED_PAIRS:
            raise PairwisePrimitiveError(
                f"pair_count must be an integer in [1, {MAX_SAMPLED_PAIRS}]"
            )
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise PairwisePrimitiveError("seed must be an unsigned 32-bit integer")

        rng = np.random.default_rng(seed)
        positive_tickets = rng.integers(
            0, self.eligible_positive_count, size=pair_count, dtype=np.int64
        )
        group_indices = np.searchsorted(
            self._cumulative_positive_counts, positive_tickets, side="right"
        ).astype(np.int64, copy=False)
        # A positive ticket already selects both its user (with weight P_u) and one of that
        # user's positives uniformly.  Reusing it avoids a second random draw without changing
        # the target law: every positive-negative pair still has probability 1/(sum(P) * N_u).
        positive_rows = self._positive_indices[positive_tickets]
        del positive_tickets

        negative_high = self._negative_counts[group_indices]
        negative_positions = rng.integers(0, negative_high, dtype=np.int64)
        del negative_high
        negative_positions += self._negative_offsets[group_indices]
        negative_rows = self._negative_indices[negative_positions]
        for array in (group_indices, positive_rows, negative_rows):
            array.setflags(write=False)
        return PairBatch(
            positive_indices=positive_rows,
            negative_indices=cast(Int64Vector, negative_rows),
            user_group_indices=group_indices,
        )


def brute_force_pair_distribution(
    user_ids: VectorInput,
    labels: VectorInput,
    *,
    phase: DataPhase,
    max_pairs: int = DEFAULT_MAX_ORACLE_PAIRS,
) -> PairDistribution:
    """Enumerate the exact GAUC-aligned distribution for bounded tests and diagnostics.

    Production training must use :class:`GAUCPairSampler` instead.  This independent oracle is
    intentionally capped before allocating Cartesian pairs.
    """

    if type(max_pairs) is not int or max_pairs <= 0:
        raise PairwisePrimitiveError("max_pairs must be a positive integer")
    users, targets = _training_rows(user_ids, labels, phase=phase)

    groups = _eligible_groups(users, targets)
    pair_count = sum(
        int(group.positive_indices.size) * int(group.negative_indices.size) for group in groups
    )
    if pair_count > max_pairs:
        raise PairwisePrimitiveError(
            f"brute-force oracle would enumerate {pair_count} pairs, exceeding {max_pairs}"
        )
    total_positive = sum(int(group.positive_indices.size) for group in groups)
    positive_rows = np.empty(pair_count, dtype=np.int64)
    negative_rows = np.empty(pair_count, dtype=np.int64)
    probabilities = np.empty(pair_count, dtype=np.float64)
    cursor = 0
    for group in groups:
        probability = 1.0 / (total_positive * int(group.negative_indices.size))
        for positive_index in group.positive_indices:
            for negative_index in group.negative_indices:
                positive_rows[cursor] = positive_index
                negative_rows[cursor] = negative_index
                probabilities[cursor] = probability
                cursor += 1
    probabilities /= probabilities.sum(dtype=np.float64)
    for array in (positive_rows, negative_rows, probabilities):
        array.setflags(write=False)
    return PairDistribution(
        positive_indices=cast(Int64Vector, positive_rows),
        negative_indices=cast(Int64Vector, negative_rows),
        probabilities=cast(Float64Vector, probabilities),
    )


__all__ = [
    "DEFAULT_MAX_ORACLE_PAIRS",
    "MAX_SAMPLED_PAIRS",
    "GAUCPairSampler",
    "PairBatch",
    "PairDistribution",
    "PairwiseLossResult",
    "PairwisePrimitiveError",
    "brute_force_pair_distribution",
    "pairwise_logistic_loss_and_gradient",
]
