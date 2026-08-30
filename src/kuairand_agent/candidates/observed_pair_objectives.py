"""Train-only observed-pair interventions with an exact uniform control.

The public ablation builder keeps the verified GAUC-aligned sampler as its control and changes
exactly half of the trained comparisons to logged same-user, same-duration-bucket pairs.  It is a
training primitive only: no scorer, validation label, catalog negative, or prediction capability
is accepted here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.candidates.pairwise import (
    MAX_SAMPLED_PAIRS,
    GAUCPairSampler,
    PairwisePrimitiveError,
)
from kuairand_agent.data.capabilities import DataPhase

type VectorInput = Sequence[object] | npt.NDArray[np.generic]
type Int64Vector = npt.NDArray[np.int64]
type BoolVector = npt.NDArray[np.bool_]

DURATION_SECONDS_POSITION: Final = 46
DURATION_BUCKET_EDGES_SECONDS: Final = (5.0, 10.0, 18.0, 30.0, 60.0)
_DURATION_BUCKET_COUNT: Final = len(DURATION_BUCKET_EDGES_SECONDS) + 1
_TREATMENT_SEED_DOMAIN: Final = b"kuairand-duration-conditioned-pairs-v1\0"


class ObservedPairObjectiveError(ValueError):
    """An observed-pair ablation input violates the train-only frozen contract."""


def _freeze(value: npt.NDArray[np.generic], dtype: np.dtype[np.generic]) -> npt.NDArray[np.generic]:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _training_phase(phase: DataPhase) -> None:
    if not isinstance(phase, DataPhase):
        raise ObservedPairObjectiveError("phase must be a DataPhase")
    if phase not in {DataPhase.TRAIN, DataPhase.INNER_TRAIN}:
        raise ObservedPairObjectiveError(
            "observed-pair labels are allowed only for train or inner_train"
        )


def _uint32(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ObservedPairObjectiveError(f"{name} must be an unsigned 32-bit integer")
    return value


def _pair_count(value: object) -> int:
    if type(value) is not int or not 2 <= value <= MAX_SAMPLED_PAIRS or value % 2 != 0:
        raise ObservedPairObjectiveError(
            f"pair_count must be an even integer in [2, {MAX_SAMPLED_PAIRS}]"
        )
    return value


def _features(value: object) -> npt.NDArray[np.float64]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f8")
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] <= DURATION_SECONDS_POSITION
        or not value.flags.c_contiguous
        or not bool(np.isfinite(value).all())
    ):
        raise ObservedPairObjectiveError(
            "features must be finite C-contiguous little-endian float64 with duration column 46"
        )
    durations = value[:, DURATION_SECONDS_POSITION]
    if bool((durations < 0.0).any()):
        raise ObservedPairObjectiveError("duration seconds must be non-negative")
    return cast(npt.NDArray[np.float64], value)


def _groups_and_targets(
    user_groups: VectorInput,
    targets: VectorInput,
    *,
    expected: int,
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.int8]]:
    groups = np.asarray(user_groups)
    labels = np.asarray(targets)
    if (
        groups.shape != (expected,)
        or groups.dtype.kind not in "iuf"
        or groups.dtype.kind == "b"
        or not bool(np.isfinite(groups).all())
    ):
        raise ObservedPairObjectiveError(
            "user_groups must be a finite non-boolean numeric vector aligned to features"
        )
    if labels.shape != (expected,) or labels.dtype.kind not in "biuf":
        raise ObservedPairObjectiveError("targets must be a numeric vector aligned to features")
    numeric = np.asarray(labels, dtype=np.float64)
    if not bool(np.isfinite(numeric).all()) or not bool(np.isin(numeric, (0.0, 1.0)).all()):
        raise ObservedPairObjectiveError("targets must contain only binary 0 and 1")
    return groups, np.ascontiguousarray(numeric, dtype=np.int8)


def _child_seed(seed: int) -> int:
    digest = hashlib.sha256(_TREATMENT_SEED_DOMAIN)
    digest.update(seed.to_bytes(4, "little", signed=False))
    return int.from_bytes(digest.digest()[:4], "little", signed=False)


def _duration_codes(features: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    return np.searchsorted(
        np.asarray(DURATION_BUCKET_EDGES_SECONDS, dtype=np.float64),
        features[:, DURATION_SECONDS_POSITION],
        side="right",
    ).astype(np.int64, copy=False)


def _conditioned_pairs(
    groups: npt.NDArray[np.generic],
    targets: npt.NDArray[np.int8],
    duration_codes: npt.NDArray[np.int64],
    *,
    pair_count: int,
    seed: int,
) -> tuple[Int64Vector, Int64Vector, int, int]:
    _, first_indices, sorted_codes = np.unique(groups, return_index=True, return_inverse=True)
    first_seen = np.argsort(first_indices, kind="stable")
    remap = np.empty(first_seen.size, dtype=np.int64)
    remap[first_seen] = np.arange(first_seen.size, dtype=np.int64)
    user_codes = remap[sorted_codes]
    pair_codes = user_codes * _DURATION_BUCKET_COUNT + duration_codes
    pair_group_count = first_seen.size * _DURATION_BUCKET_COUNT
    positive_counts = np.bincount(pair_codes[targets == 1], minlength=pair_group_count)
    negative_counts = np.bincount(pair_codes[targets == 0], minlength=pair_group_count)
    eligible = np.logical_and(positive_counts > 0, negative_counts > 0)
    if not bool(eligible.any()):
        raise ObservedPairObjectiveError(
            "duration-conditioned sampling requires a same-user bucket with both labels"
        )
    eligible_codes = np.flatnonzero(eligible)
    compact = np.full(pair_group_count, -1, dtype=np.int64)
    compact[eligible_codes] = np.arange(eligible_codes.size, dtype=np.int64)
    eligible_rows = eligible[pair_codes]
    positive_rows = np.flatnonzero(np.logical_and(targets == 1, eligible_rows))
    negative_rows = np.flatnonzero(np.logical_and(targets == 0, eligible_rows))
    positive_rows = positive_rows[
        np.argsort(compact[pair_codes[positive_rows]], kind="stable")
    ].astype(np.int64, copy=False)
    negative_rows = negative_rows[
        np.argsort(compact[pair_codes[negative_rows]], kind="stable")
    ].astype(np.int64, copy=False)
    eligible_positive_counts = positive_counts[eligible].astype(np.int64, copy=False)
    eligible_negative_counts = negative_counts[eligible].astype(np.int64, copy=False)
    cumulative_positive = np.cumsum(eligible_positive_counts, dtype=np.int64)
    negative_offsets = np.empty(eligible_codes.size, dtype=np.int64)
    negative_offsets[0] = 0
    if eligible_codes.size > 1:
        np.cumsum(eligible_negative_counts[:-1], dtype=np.int64, out=negative_offsets[1:])

    rng = np.random.default_rng(seed)
    tickets = rng.integers(0, int(cumulative_positive[-1]), size=pair_count, dtype=np.int64)
    sampled_groups = np.searchsorted(cumulative_positive, tickets, side="right")
    sampled_positive = positive_rows[tickets]
    negative_positions = rng.integers(
        0,
        eligible_negative_counts[sampled_groups],
        dtype=np.int64,
    )
    negative_positions += negative_offsets[sampled_groups]
    sampled_negative = negative_rows[negative_positions]
    return (
        np.ascontiguousarray(sampled_positive),
        np.ascontiguousarray(sampled_negative),
        int(eligible_codes.size),
        int(cumulative_positive[-1]),
    )


@dataclass(frozen=True, slots=True, init=False)
class DurationPairAblation:
    """Exact uniform control and equal-budget 50/50 duration-conditioned treatment."""

    control_positive_indices: Int64Vector = field(repr=False)
    control_negative_indices: Int64Vector = field(repr=False)
    treatment_positive_indices: Int64Vector = field(repr=False)
    treatment_negative_indices: Int64Vector = field(repr=False)
    intervention_mask: BoolVector = field(repr=False)
    seed: int
    treatment_seed: int
    conditioned_eligible_group_count: int
    conditioned_eligible_positive_count: int

    def __init__(
        self,
        *,
        control_positive_indices: npt.NDArray[np.generic],
        control_negative_indices: npt.NDArray[np.generic],
        treatment_positive_indices: npt.NDArray[np.generic],
        treatment_negative_indices: npt.NDArray[np.generic],
        intervention_mask: npt.NDArray[np.generic],
        seed: int,
        treatment_seed: int,
        conditioned_eligible_group_count: int,
        conditioned_eligible_positive_count: int,
    ) -> None:
        arrays = (
            _freeze(control_positive_indices, np.dtype("<i8")),
            _freeze(control_negative_indices, np.dtype("<i8")),
            _freeze(treatment_positive_indices, np.dtype("<i8")),
            _freeze(treatment_negative_indices, np.dtype("<i8")),
        )
        mask = _freeze(intervention_mask, np.dtype("?"))
        if (
            any(array.shape != arrays[0].shape for array in arrays[1:])
            or mask.shape != arrays[0].shape
        ):
            raise ObservedPairObjectiveError("pair ablation arrays must have identical shapes")
        object.__setattr__(self, "control_positive_indices", cast(Int64Vector, arrays[0]))
        object.__setattr__(self, "control_negative_indices", cast(Int64Vector, arrays[1]))
        object.__setattr__(self, "treatment_positive_indices", cast(Int64Vector, arrays[2]))
        object.__setattr__(self, "treatment_negative_indices", cast(Int64Vector, arrays[3]))
        object.__setattr__(self, "intervention_mask", cast(BoolVector, mask))
        object.__setattr__(self, "seed", _uint32(seed, "seed"))
        object.__setattr__(self, "treatment_seed", _uint32(treatment_seed, "treatment_seed"))
        object.__setattr__(
            self,
            "conditioned_eligible_group_count",
            conditioned_eligible_group_count,
        )
        object.__setattr__(
            self,
            "conditioned_eligible_positive_count",
            conditioned_eligible_positive_count,
        )

    @property
    def pair_count(self) -> int:
        return int(self.intervention_mask.size)

    @property
    def intervention_pair_count(self) -> int:
        return int(np.count_nonzero(self.intervention_mask))


def prepare_duration_pair_ablation(
    features: object,
    user_groups: VectorInput,
    targets: VectorInput,
    *,
    pair_count: int,
    seed: int,
    phase: DataPhase,
) -> DurationPairAblation:
    """Build matched control/treatment pairs without changing total training examples."""

    _training_phase(phase)
    count = _pair_count(pair_count)
    root_seed = _uint32(seed, "seed")
    matrix = _features(features)
    groups, labels = _groups_and_targets(user_groups, targets, expected=matrix.shape[0])
    try:
        control = GAUCPairSampler(groups, labels, phase=phase).sample(count, seed=root_seed)
    except PairwisePrimitiveError as exc:
        raise ObservedPairObjectiveError(str(exc)) from exc
    treatment_seed = _child_seed(root_seed)
    conditioned_positive, conditioned_negative, eligible_groups, eligible_positives = (
        _conditioned_pairs(
            groups,
            labels,
            _duration_codes(matrix),
            pair_count=count // 2,
            seed=treatment_seed,
        )
    )
    mask = np.arange(count, dtype=np.int64) % 2 == 1
    treatment_positive = np.array(control.positive_indices, dtype=np.int64, copy=True)
    treatment_negative = np.array(control.negative_indices, dtype=np.int64, copy=True)
    treatment_positive[mask] = conditioned_positive
    treatment_negative[mask] = conditioned_negative
    return DurationPairAblation(
        control_positive_indices=control.positive_indices,
        control_negative_indices=control.negative_indices,
        treatment_positive_indices=treatment_positive,
        treatment_negative_indices=treatment_negative,
        intervention_mask=mask,
        seed=root_seed,
        treatment_seed=treatment_seed,
        conditioned_eligible_group_count=eligible_groups,
        conditioned_eligible_positive_count=eligible_positives,
    )


__all__ = [
    "DURATION_BUCKET_EDGES_SECONDS",
    "DURATION_SECONDS_POSITION",
    "DurationPairAblation",
    "ObservedPairObjectiveError",
    "prepare_duration_pair_ablation",
]
