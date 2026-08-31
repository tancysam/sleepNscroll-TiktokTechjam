"""Label-free deterministic within-user rank normalization and fusion.

Utilities in this module consume aligned prediction vectors only.  They have no label argument,
perform no metric evaluation, and expose no weight-selection operation; grid selection belongs
to train-derived inner-fold policy in the trusted controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.submission import prediction_digest

type Identity = int | str
type IdentityInput = Sequence[object] | npt.NDArray[np.generic]
type ScoreInput = Sequence[object] | npt.NDArray[np.generic]
type Float64Vector = npt.NDArray[np.float64]

FUSION_SCHEMA_VERSION: Final = 1
TIE_POLICY: Final = "descending-average-rank; singleton=0.5"
FUSION_WEIGHT_GRID: Final = (
    (1.0, 0.0),
    (0.75, 0.25),
    (0.5, 0.5),
    (0.25, 0.75),
    (0.0, 1.0),
)
_ALLOWED_PREDICTION_PHASES: Final = frozenset(
    {DataPhase.INNER_VALID, DataPhase.OUTER_VALID, DataPhase.FINAL}
)


class FusionError(ValueError):
    """Raised when label-free aligned prediction fusion cannot be performed safely."""


def _require_prediction_phase(phase: DataPhase) -> None:
    # Validate before touching prediction content.  Training labels are never an input here, and
    # outer/final phases remain strictly label-free.
    if not isinstance(phase, DataPhase):
        raise FusionError("phase must be a DataPhase")
    if phase not in _ALLOWED_PREDICTION_PHASES:
        raise FusionError("rank fusion is allowed only for inner_valid, outer_valid, or final")


def _identity(value: object, location: str) -> Identity:
    if type(value) is bool:
        raise FusionError(f"{location} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value and "\x00" not in value:
        return value
    raise FusionError(f"{location} must be an integer or non-empty string")


def _identities(value: IdentityInput, name: str) -> tuple[Identity, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise FusionError(f"{name} must be one-dimensional")
        raw = value.tolist()
    else:
        if isinstance(value, (str, bytes)):
            raise FusionError(f"{name} must be a one-dimensional identity sequence")
        try:
            raw = list(value)
        except TypeError as exc:
            raise FusionError(f"{name} must be a one-dimensional identity sequence") from exc
    if not raw:
        raise FusionError(f"{name} cannot be empty")
    return tuple(_identity(item, f"{name}[{index}]") for index, item in enumerate(raw))


def _scores(value: ScoreInput, *, expected: int, name: str) -> Float64Vector:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise FusionError(f"{name} must be a one-dimensional finite numeric vector") from exc
    if raw.ndim != 1:
        raise FusionError(f"{name} must be one-dimensional")
    if raw.size != expected:
        raise FusionError(f"{name} must have length {expected}, got {raw.size}")
    if raw.dtype.kind not in "iuf":
        raise FusionError(f"{name} must have a real numeric dtype")
    try:
        values = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FusionError(f"{name} must be representable as float64") from exc
    if not np.isfinite(values).all():
        raise FusionError(f"{name} must contain only finite values")
    return values


def _wire_identity(value: Identity) -> tuple[str, Identity]:
    return ("i", value) if type(value) is int else ("s", value)


def _alignment_digest(
    users: tuple[Identity, ...], videos: tuple[Identity, ...], phase: DataPhase
) -> str:
    manifest = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "phase": phase.value,
        "rows": [
            [_wire_identity(user), _wire_identity(video)]
            for user, video in zip(users, videos, strict=True)
        ],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RankNormalizedPrediction:
    """One immutable canonical-order vector of within-user descending percentiles."""

    scores: Float64Vector = field(repr=False)
    prediction_digest: str
    alignment_digest: str
    phase: DataPhase
    tie_policy: str = TIE_POLICY


@dataclass(frozen=True, slots=True)
class FusionResult:
    """One immutable two-member rank blend with complete prediction provenance."""

    scores: Float64Vector = field(repr=False)
    weights: tuple[float, float]
    member_prediction_digests: tuple[str, str]
    normalized_prediction_digests: tuple[str, str]
    prediction_digest: str
    alignment_digest: str
    fusion_digest: str
    phase: DataPhase
    #: Pearson correlation of the two members' within-user percentile vectors, which is Spearman
    #: on their raw scores because percentiles ARE the ranks.  It is the one number that separates
    #: "you rebuilt the control, worse" (near 1) from "you are genuinely different but weak"
    #: (low), and no train-only analysis request can reach it because it needs both members'
    #: validation predictions.  ``None`` for a degenerate vector with no variance.
    member_rank_correlation: float | None = None


def _rank_correlation(first: Float64Vector, second: Float64Vector) -> float | None:
    """Pearson on percentile vectors. ``None`` when either side is constant."""

    first_centred = first - first.mean()
    second_centred = second - second.mean()
    denominator = float(np.linalg.norm(first_centred) * np.linalg.norm(second_centred))
    if not math.isfinite(denominator) or denominator == 0.0:
        return None
    value = float(np.dot(first_centred, second_centred) / denominator)
    if not math.isfinite(value):
        return None
    # Floating point can put an exactly-identical pair a few ulp outside [-1, 1].
    return max(-1.0, min(1.0, value))


def normalize_within_user_percentiles(
    user_ids: IdentityInput,
    video_ids: IdentityInput,
    scores: ScoreInput,
    *,
    phase: DataPhase,
) -> RankNormalizedPrediction:
    """Normalize scores to ``[0, 1]`` by equal midrank within each logged user slate.

    Highest scores receive one and lowest scores receive zero.  Exact ties receive the same
    average rank, so canonical row order is never converted into a learned tie-breaking signal.
    A singleton slate receives the neutral value ``0.5``.
    """

    _require_prediction_phase(phase)
    users = _identities(user_ids, "user_ids")
    videos = _identities(video_ids, "video_ids")
    if len(users) != len(videos):
        raise FusionError(
            f"user_ids and video_ids must have equal lengths; got {len(users)} and {len(videos)}"
        )
    values = _scores(scores, expected=len(users), name="scores")

    positions: dict[Identity, list[int]] = {}
    for canonical_index, user in enumerate(users):
        positions.setdefault(user, []).append(canonical_index)
    normalized = np.empty(len(users), dtype=np.float64)
    for canonical_positions in positions.values():
        count = len(canonical_positions)
        if count == 1:
            normalized[canonical_positions[0]] = 0.5
            continue
        group_positions = np.asarray(canonical_positions, dtype=np.int64)
        group_scores = values[group_positions]
        order = np.argsort(-group_scores, kind="stable")
        rank_start = 0
        while rank_start < count:
            rank_end = rank_start + 1
            tied_score = group_scores[order[rank_start]]
            while rank_end < count and group_scores[order[rank_end]] == tied_score:
                rank_end += 1
            average_rank = (rank_start + rank_end - 1) / 2.0
            # Subtract ranks before division so worked rational values such as 5/6 have the same
            # float64 bytes as direct percentile construction and therefore stable digests.
            percentile = (count - 1 - average_rank) / (count - 1)
            normalized[group_positions[order[rank_start:rank_end]]] = percentile
            rank_start = rank_end

    normalized.setflags(write=False)
    return RankNormalizedPrediction(
        scores=cast(Float64Vector, normalized),
        prediction_digest=prediction_digest(normalized),
        alignment_digest=_alignment_digest(users, videos, phase),
        phase=phase,
    )


def fuse_ranked_predictions(
    user_ids: IdentityInput,
    video_ids: IdentityInput,
    first_scores: ScoreInput,
    second_scores: ScoreInput,
    *,
    weights: tuple[float, float],
    phase: DataPhase,
) -> FusionResult:
    """Blend two label-free percentile vectors using one predeclared grid point.

    This function deliberately cannot inspect targets or select a grid point.  The trusted
    controller selects one weight on train-derived inner folds and supplies it explicitly.
    """

    _require_prediction_phase(phase)
    if (
        type(weights) is not tuple
        or len(weights) != 2
        or any(type(value) is not float for value in weights)
        or weights not in FUSION_WEIGHT_GRID
    ):
        raise FusionError(f"weights must be one exact member of {FUSION_WEIGHT_GRID}")

    first = normalize_within_user_percentiles(user_ids, video_ids, first_scores, phase=phase)
    second = normalize_within_user_percentiles(user_ids, video_ids, second_scores, phase=phase)
    if first.alignment_digest != second.alignment_digest:
        raise FusionError("rank-fusion members must use identical canonical alignment")
    raw_first = _scores(first_scores, expected=first.scores.size, name="first_scores")
    raw_second = _scores(second_scores, expected=first.scores.size, name="second_scores")
    member_digests = (prediction_digest(raw_first), prediction_digest(raw_second))
    normalized_digests = (first.prediction_digest, second.prediction_digest)
    fused = np.ascontiguousarray(
        weights[0] * first.scores + weights[1] * second.scores, dtype=np.float64
    )
    if not np.isfinite(fused).all():  # defensive; normalized members and grid weights are finite
        raise FusionError("rank-fusion output must contain only finite values")
    output_digest = prediction_digest(fused)
    manifest = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "phase": phase.value,
        "normalization": "within_user_descending_midrank_percentile_v1",
        "tie_policy": TIE_POLICY,
        "weights": list(weights),
        "weight_grid": [list(point) for point in FUSION_WEIGHT_GRID],
        "alignment_digest": first.alignment_digest,
        "member_prediction_digests": list(member_digests),
        "normalized_prediction_digests": list(normalized_digests),
        "prediction_digest": output_digest,
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    fused.setflags(write=False)
    return FusionResult(
        scores=fused,
        weights=weights,
        member_prediction_digests=member_digests,
        normalized_prediction_digests=normalized_digests,
        prediction_digest=output_digest,
        alignment_digest=first.alignment_digest,
        fusion_digest=hashlib.sha256(payload).hexdigest(),
        phase=phase,
        member_rank_correlation=_rank_correlation(first.scores, second.scores),
    )


__all__ = [
    "FUSION_SCHEMA_VERSION",
    "FUSION_WEIGHT_GRID",
    "TIE_POLICY",
    "FusionError",
    "FusionResult",
    "RankNormalizedPrediction",
    "fuse_ranked_predictions",
    "normalize_within_user_percentiles",
]
