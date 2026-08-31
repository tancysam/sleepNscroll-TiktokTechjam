"""Types that may cross the quarantined protected-evaluation seam.

Protected labels and aggregate protected results are deliberately owned by evaluation.  Data
loading may construct :class:`ProtectedLabels`, while proposal, training, and search code must not
import this module.  The label container exposes no iterator or general-purpose column accessor;
only trusted evaluation adapters should call :meth:`ProtectedLabels.reveal_for_evaluator`.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable


class ProtectedEvidenceError(ValueError):
    """Raised when protected evidence violates its quarantined schema."""


class ProtectedAccess(StrEnum):
    """The sole authorized consumer of protected labels."""

    EVALUATOR_ONLY = "protected_scorer_only"


def _sha256_text(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtectedEvidenceError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ProtectedLabels:
    """Opaque binary validation labels available only to trusted evaluation.

    The values are snapshotted into a private immutable tuple.  There is intentionally no
    ``column``, ``values``, or iterator interface that candidate-facing code could consume.
    """

    _long_view: Sequence[int] = field(repr=False)
    digest: str = field(init=False)
    access: ProtectedAccess = field(init=False, default=ProtectedAccess.EVALUATOR_ONLY)

    def __post_init__(self) -> None:
        normalized: list[int] = []
        for index, value in enumerate(self._long_view):
            if type(value) is not int or value not in (0, 1):
                raise ProtectedEvidenceError(f"protected long_view[{index}] must be binary")
            normalized.append(value)
        values = tuple(normalized)
        digest = hashlib.sha256(b"kuairand-protected-outer-targets-v1\0")
        encoded_name = b"long_view"
        digest.update(struct.pack("<H", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(values)))
        for value in values:
            digest.update(b"i")
            digest.update(struct.pack("<q", value))
        object.__setattr__(self, "_long_view", values)
        object.__setattr__(self, "digest", digest.hexdigest())

    def __len__(self) -> int:
        return len(self._long_view)

    def reveal_for_evaluator(self) -> tuple[int, ...]:
        """Reveal labels after the caller has entered trusted evaluation."""

        return cast(tuple[int, ...], self._long_view)

    def reveal_for_scorer(self) -> tuple[int, ...]:
        """Compatibility alias for legacy trusted scoring adapters."""

        return self.reveal_for_evaluator()


@runtime_checkable
class AggregateScoreLike(Protocol):
    """Structural adapter input for converting a legacy aggregate scorer result."""

    gauc: float
    ndcg_at_5: float
    primary: float
    users: int
    rows: int
    scorer_digest: str
    prediction_digest: str
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class ProtectedResult:
    """Quarantined aggregate result from one authorized protected evaluation.

    Only aggregate metrics and content identities cross this seam.  Row-level labels are never a
    member of a result and therefore cannot leak into proposal, training, or search schemas.
    """

    gauc: float
    ndcg_at_5: float
    primary: float
    users: int
    rows: int
    scorer_digest: str
    prediction_digest: str
    runtime_seconds: float

    def __post_init__(self) -> None:
        metrics = {
            "gauc": self.gauc,
            "ndcg_at_5": self.ndcg_at_5,
            "primary": self.primary,
        }
        for name, value in metrics.items():
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ProtectedEvidenceError(f"{name} must be a finite float in [0, 1]")
        expected_float64 = (self.gauc + self.ndcg_at_5) / 2.0
        if not math.isclose(self.primary, expected_float64, rel_tol=0.0, abs_tol=1e-7):
            raise ProtectedEvidenceError("primary must be the mean of gauc and ndcg_at_5")
        if type(self.rows) is not int or self.rows <= 0:
            raise ProtectedEvidenceError("rows must be a positive integer")
        if type(self.users) is not int or not 1 <= self.users <= self.rows:
            raise ProtectedEvidenceError("users must be an integer in [1, rows]")
        _sha256_text(self.scorer_digest, "scorer_digest")
        _sha256_text(self.prediction_digest, "prediction_digest")
        if (
            type(self.runtime_seconds) is not float
            or not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0.0
        ):
            raise ProtectedEvidenceError("runtime_seconds must be a finite non-negative float")

    @classmethod
    def from_aggregate(cls, result: AggregateScoreLike) -> ProtectedResult:
        """Copy a legacy trusted scorer aggregate into the protected result type."""

        if not isinstance(result, AggregateScoreLike):
            raise ProtectedEvidenceError("protected evaluator returned an invalid aggregate")
        return cls(
            gauc=result.gauc,
            ndcg_at_5=result.ndcg_at_5,
            primary=result.primary,
            users=result.users,
            rows=result.rows,
            scorer_digest=result.scorer_digest,
            prediction_digest=result.prediction_digest,
            runtime_seconds=result.runtime_seconds,
        )

    @property
    def ndcg5(self) -> float:
        return self.ndcg_at_5

    def as_dict(self) -> dict[str, float | int | str]:
        """Return the stable aggregate evidence representation."""

        return {
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
            "users": self.users,
            "rows": self.rows,
            "scorer_digest": self.scorer_digest,
            "prediction_digest": self.prediction_digest,
            "runtime_seconds": self.runtime_seconds,
        }


__all__ = [
    "AggregateScoreLike",
    "ProtectedAccess",
    "ProtectedEvidenceError",
    "ProtectedLabels",
    "ProtectedResult",
]
