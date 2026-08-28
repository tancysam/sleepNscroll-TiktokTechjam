"""Stable user-contiguous ranking views with exact canonical scatter-back.

The adapter groups logged impressions only for train-derived LightGBM-style ranking views.
Users retain first-appearance order, rows within a user retain canonical order, and repeated
``(user_id, video_id)`` impressions remain distinct positional rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.data.capabilities import DataPhase

type Identity = int | str
type IdentityInput = Sequence[object] | npt.NDArray[np.generic]
type RowArray = Sequence[object] | npt.NDArray[np.generic]
type Int64Vector = npt.NDArray[np.int64]

GROUPING_SCHEMA_VERSION: Final = 1
_ALLOWED_GROUPING_PHASES: Final = frozenset(
    {DataPhase.TRAIN, DataPhase.INNER_TRAIN, DataPhase.INNER_VALID}
)


class GroupingError(ValueError):
    """Raised when an aligned ranking view cannot be grouped without identity loss."""


def _identity(value: object, location: str) -> Identity:
    if type(value) is bool:
        raise GroupingError(f"{location} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value and "\x00" not in value:
        return value
    raise GroupingError(f"{location} must be an integer or non-empty string")


def _identities(value: IdentityInput, name: str) -> tuple[Identity, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise GroupingError(f"{name} must be one-dimensional")
        raw = value.tolist()
    else:
        if isinstance(value, (str, bytes)):
            raise GroupingError(f"{name} must be a one-dimensional identity sequence")
        try:
            raw = list(value)
        except TypeError as exc:
            raise GroupingError(f"{name} must be a one-dimensional identity sequence") from exc
    if not raw:
        raise GroupingError(f"{name} cannot be empty")
    return tuple(_identity(item, f"{name}[{index}]") for index, item in enumerate(raw))


def _identity_wire(value: Identity) -> tuple[str, Identity]:
    return ("i", value) if type(value) is int else ("s", value)


def _row_array(value: RowArray, *, expected: int, name: str) -> npt.NDArray[np.generic]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise GroupingError(f"{name} must be an array with rows on axis zero") from exc
    if array.ndim == 0:
        raise GroupingError(f"{name} must have rows on axis zero")
    if array.shape[0] != expected:
        raise GroupingError(
            f"{name} row count must equal grouping row_count {expected}, got {array.shape[0]}"
        )
    return array


@dataclass(frozen=True, slots=True)
class UserGrouping:
    """Immutable permutation from canonical rows to stable user-contiguous rows."""

    phase: DataPhase
    row_count: int
    group_sizes: tuple[int, ...]
    grouped_to_canonical: Int64Vector = field(repr=False)
    canonical_to_grouped: Int64Vector = field(repr=False)
    digest: str

    def to_grouped(self, canonical_rows: RowArray) -> npt.NDArray[np.generic]:
        """Return a read-only user-contiguous copy of canonical row-aligned values."""

        values = _row_array(canonical_rows, expected=self.row_count, name="canonical_rows")
        grouped = np.ascontiguousarray(np.take(values, self.grouped_to_canonical, axis=0))
        grouped.setflags(write=False)
        return grouped

    def to_canonical(self, grouped_rows: RowArray) -> npt.NDArray[np.generic]:
        """Scatter a complete grouped result back to exact canonical physical order."""

        values = _row_array(grouped_rows, expected=self.row_count, name="grouped_rows")
        canonical = np.ascontiguousarray(np.take(values, self.canonical_to_grouped, axis=0))
        canonical.setflags(write=False)
        return canonical


def build_user_grouping(
    user_ids: IdentityInput,
    video_ids: IdentityInput,
    *,
    phase: DataPhase,
) -> UserGrouping:
    """Build a deterministic train-derived ranking permutation without labels or row IDs."""

    if not isinstance(phase, DataPhase):
        raise GroupingError("phase must be a DataPhase")
    if phase not in _ALLOWED_GROUPING_PHASES:
        raise GroupingError(
            "LightGBM-style grouping is allowed only for train, inner_train, or inner_valid"
        )
    users = _identities(user_ids, "user_ids")
    videos = _identities(video_ids, "video_ids")
    if len(users) != len(videos):
        raise GroupingError(
            f"user_ids and video_ids must have equal lengths; got {len(users)} and {len(videos)}"
        )

    positions: dict[Identity, list[int]] = {}
    for canonical_index, user in enumerate(users):
        positions.setdefault(user, []).append(canonical_index)
    permutation = np.fromiter(
        (index for group in positions.values() for index in group),
        dtype=np.int64,
        count=len(users),
    )
    inverse = np.empty(len(users), dtype=np.int64)
    inverse[permutation] = np.arange(len(users), dtype=np.int64)
    if not np.array_equal(np.sort(permutation), np.arange(len(users), dtype=np.int64)):
        raise GroupingError("grouping permutation must contain every canonical row exactly once")
    group_sizes = tuple(len(group) for group in positions.values())

    manifest = {
        "schema_version": GROUPING_SCHEMA_VERSION,
        "phase": phase.value,
        "row_count": len(users),
        "group_sizes": list(group_sizes),
        "grouped_to_canonical": permutation.tolist(),
        "canonical_to_grouped": inverse.tolist(),
        "alignment": [
            [_identity_wire(user), _identity_wire(video)]
            for user, video in zip(users, videos, strict=True)
        ],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    permutation.setflags(write=False)
    inverse.setflags(write=False)
    return UserGrouping(
        phase=phase,
        row_count=len(users),
        group_sizes=group_sizes,
        grouped_to_canonical=cast(Int64Vector, permutation),
        canonical_to_grouped=cast(Int64Vector, inverse),
        digest=hashlib.sha256(payload).hexdigest(),
    )


__all__ = ["GROUPING_SCHEMA_VERSION", "GroupingError", "UserGrouping", "build_user_grouping"]
