"""Protected candidate-side 50/50 duration-conditioned observed-pair ablation.

The uniform half is produced by the verified reference sampler.  The intervention half uses only
logged training rows from the same user and frozen duration bucket.  This module has no scorer,
metric, filesystem, network, or controller dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from reference_pairwise_fm import sample_reference_logged_pairs  # type: ignore[import-not-found]

REFERENCE_DURATION_SECONDS_POSITION = 46
REFERENCE_DURATION_BUCKET_EDGES_SECONDS = (5.0, 10.0, 18.0, 30.0, 60.0)
REFERENCE_MAX_SAMPLED_PAIRS = 1_000_000
_DURATION_BUCKET_COUNT = len(REFERENCE_DURATION_BUCKET_EDGES_SECONDS) + 1
_TREATMENT_SEED_DOMAIN = b"kuairand-duration-conditioned-pairs-v1\0"


class ReferenceObservedPairObjectiveError(ValueError):
    """A protected observed-pair input violates its frozen contract."""


def _freeze(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _uint32(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ReferenceObservedPairObjectiveError(f"{name} must be an unsigned 32-bit integer")
    return value


def _pair_count(value: object) -> int:
    if type(value) is not int or not 2 <= value <= REFERENCE_MAX_SAMPLED_PAIRS or value % 2 != 0:
        raise ReferenceObservedPairObjectiveError(
            "pair_count must be an even integer in [2, 1000000]"
        )
    return value


def _features(value: object) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f8")
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] <= REFERENCE_DURATION_SECONDS_POSITION
        or not value.flags.c_contiguous
        or not bool(np.isfinite(value).all())
    ):
        raise ReferenceObservedPairObjectiveError(
            "features must be finite C-contiguous little-endian float64 with duration column 46"
        )
    if bool((value[:, REFERENCE_DURATION_SECONDS_POSITION] < 0.0).any()):
        raise ReferenceObservedPairObjectiveError("duration seconds must be non-negative")
    return value


def _groups_and_targets(
    user_groups: np.ndarray,
    targets: np.ndarray,
    *,
    expected: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(user_groups)
    labels = np.asarray(targets)
    if (
        groups.shape != (expected,)
        or groups.dtype.kind not in "iuf"
        or groups.dtype.kind == "b"
        or not bool(np.isfinite(groups).all())
    ):
        raise ReferenceObservedPairObjectiveError(
            "user_groups must be a finite non-boolean numeric vector aligned to features"
        )
    if labels.shape != (expected,) or labels.dtype.kind not in "biuf":
        raise ReferenceObservedPairObjectiveError(
            "targets must be a numeric vector aligned to features"
        )
    numeric = np.asarray(labels, dtype=np.float64)
    if not bool(np.isfinite(numeric).all()) or not bool(np.isin(numeric, (0.0, 1.0)).all()):
        raise ReferenceObservedPairObjectiveError("targets must contain only binary 0 and 1")
    return groups, np.ascontiguousarray(numeric, dtype=np.int8)


def _child_seed(seed: int) -> int:
    digest = hashlib.sha256(_TREATMENT_SEED_DOMAIN)
    digest.update(seed.to_bytes(4, "little", signed=False))
    return int.from_bytes(digest.digest()[:4], "little", signed=False)


def _duration_codes(features: np.ndarray) -> np.ndarray:
    return np.searchsorted(
        np.asarray(REFERENCE_DURATION_BUCKET_EDGES_SECONDS, dtype=np.float64),
        features[:, REFERENCE_DURATION_SECONDS_POSITION],
        side="right",
    ).astype(np.int64, copy=False)


def _conditioned_pairs(
    groups: np.ndarray,
    targets: np.ndarray,
    duration_codes: np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
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
        raise ReferenceObservedPairObjectiveError(
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


@dataclass(frozen=True, slots=True)
class ReferenceDurationPairAblation:
    """Immutable exact control and equal-size duration-conditioned treatment arrays."""

    control_positive_indices: np.ndarray
    control_negative_indices: np.ndarray
    treatment_positive_indices: np.ndarray
    treatment_negative_indices: np.ndarray
    intervention_mask: np.ndarray
    seed: int
    treatment_seed: int
    conditioned_eligible_group_count: int
    conditioned_eligible_positive_count: int

    @property
    def pair_count(self) -> int:
        return int(self.intervention_mask.size)

    @property
    def intervention_pair_count(self) -> int:
        return int(np.count_nonzero(self.intervention_mask))


def prepare_reference_duration_pair_ablation(
    features: np.ndarray,
    user_groups: np.ndarray,
    targets: np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> ReferenceDurationPairAblation:
    """Return exact protected-uniform control and a deterministic 50/50 treatment."""

    count = _pair_count(pair_count)
    root_seed = _uint32(seed, "seed")
    matrix = _features(features)
    groups, labels = _groups_and_targets(user_groups, targets, expected=matrix.shape[0])
    try:
        control_positive, control_negative = sample_reference_logged_pairs(
            groups,
            labels,
            pair_count=count,
            seed=root_seed,
        )
    except ValueError as exc:
        raise ReferenceObservedPairObjectiveError(str(exc)) from exc
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
    treatment_positive = np.array(control_positive, dtype=np.int64, copy=True)
    treatment_negative = np.array(control_negative, dtype=np.int64, copy=True)
    treatment_positive[mask] = conditioned_positive
    treatment_negative[mask] = conditioned_negative
    return ReferenceDurationPairAblation(
        control_positive_indices=_freeze(control_positive, np.dtype("<i8")),
        control_negative_indices=_freeze(control_negative, np.dtype("<i8")),
        treatment_positive_indices=_freeze(treatment_positive, np.dtype("<i8")),
        treatment_negative_indices=_freeze(treatment_negative, np.dtype("<i8")),
        intervention_mask=_freeze(mask, np.dtype("?")),
        seed=root_seed,
        treatment_seed=treatment_seed,
        conditioned_eligible_group_count=eligible_groups,
        conditioned_eligible_positive_count=eligible_positives,
    )


__all__ = [
    "REFERENCE_DURATION_BUCKET_EDGES_SECONDS",
    "REFERENCE_DURATION_SECONDS_POSITION",
    "REFERENCE_MAX_SAMPLED_PAIRS",
    "ReferenceDurationPairAblation",
    "ReferenceObservedPairObjectiveError",
    "prepare_reference_duration_pair_ablation",
]
