"""Exact paired user-cluster uncertainty for frozen promotion decisions.

This module is intentionally below the protected-evaluation boundary.  It consumes only
already-computed per-user metric contributions and canonical row identity.  It cannot read data,
score predictions, reserve protected queries, or choose a finalist.

The version-one procedure is fixed: resample whole users 10,000 times with seed ``20260831`` and
report a one-sided 95 percent percentile lower confidence bound.  Missing, duplicated, reordered,
or otherwise non-paired clusters are evidence errors rather than denominator adjustments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Final

import numpy as np
import numpy.typing as npt

from kuairand_agent.domain.identity import canonical_json_sha256
from kuairand_agent.evaluation.promotion import PROMOTION_POLICY_V1

type ClusterId = int | str
type Float64Vector = npt.NDArray[np.float64]

BOOTSTRAP_RESAMPLES: Final = PROMOTION_POLICY_V1.resample_count
BOOTSTRAP_SEED: Final = PROMOTION_POLICY_V1.bootstrap_seed
CONFIDENCE_LEVEL: Final = PROMOTION_POLICY_V1.confidence_level
FAMILY_WISE_ALPHA: Final = PROMOTION_POLICY_V1.family_wise_alpha
MAX_FINALISTS: Final = PROMOTION_POLICY_V1.max_frozen_finalists
MATERIAL_PRIMARY_MARGIN: Final = PROMOTION_POLICY_V1.scientific_lower_bound_strictly_greater_than
_QUANTILE_METHOD: Final = "linear"


class PromotionEvidenceError(ValueError):
    """Raised when exact paired promotion evidence cannot be established."""


def _cluster_id(value: object, location: str) -> ClusterId:
    if type(value) is bool:
        raise PromotionEvidenceError(f"{location} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value and "\x00" not in value:
        return value
    raise PromotionEvidenceError(f"{location} must be an integer or non-empty string")


def _finite(value: object, location: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise PromotionEvidenceError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PromotionEvidenceError(f"{location} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class UserClusterMetric:
    """One side of an organizer-user metric contribution on canonical physical rows.

    ``gauc_numerator`` is the user's AUC multiplied by the organizer GAUC weight (the user's
    positive-label count).  Keeping the numerator and denominator separate makes bootstrap
    aggregation exact instead of averaging per-user AUCs incorrectly.
    """

    cluster_id: ClusterId
    row_ids: tuple[int, ...]
    gauc_numerator: float
    gauc_denominator: int
    ndcg_at_5: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_id", _cluster_id(self.cluster_id, "cluster_id"))
        if type(self.row_ids) is not tuple or not self.row_ids:
            raise PromotionEvidenceError("row_ids must be a non-empty tuple")
        if any(type(row_id) is not int or row_id < 0 for row_id in self.row_ids):
            raise PromotionEvidenceError("row_ids must contain non-negative built-in integers")
        if tuple(sorted(self.row_ids)) != self.row_ids or len(set(self.row_ids)) != len(
            self.row_ids
        ):
            raise PromotionEvidenceError("cluster row_ids must be unique and strictly ordered")
        if type(self.gauc_denominator) is not int or self.gauc_denominator <= 0:
            raise PromotionEvidenceError(
                "gauc_denominator must be positive for every eligible promotion cluster"
            )
        numerator = _finite(self.gauc_numerator, "gauc_numerator")
        ndcg = _finite(self.ndcg_at_5, "ndcg_at_5")
        if not 0.0 <= numerator <= self.gauc_denominator:
            raise PromotionEvidenceError("gauc_numerator must be in [0, gauc_denominator]")
        if not 0.0 <= ndcg <= 1.0:
            raise PromotionEvidenceError("ndcg_at_5 must be in [0, 1]")
        object.__setattr__(self, "gauc_numerator", numerator)
        object.__setattr__(self, "ndcg_at_5", ndcg)


@dataclass(frozen=True, slots=True)
class OneSidedLowerBound:
    """Point delta and deterministic one-sided percentile lower confidence bound."""

    point_delta: float
    lower_bound: float
    confidence_level: float = CONFIDENCE_LEVEL
    method: str = "user-cluster-percentile-linear"

    def __post_init__(self) -> None:
        point = _finite(self.point_delta, "point_delta")
        lower = _finite(self.lower_bound, "lower_bound")
        if type(self.confidence_level) is not float or self.confidence_level != CONFIDENCE_LEVEL:
            raise PromotionEvidenceError("confidence_level must be the frozen one-sided 0.95")
        if self.method != "user-cluster-percentile-linear":
            raise PromotionEvidenceError("unsupported lower-bound method")
        object.__setattr__(self, "point_delta", point)
        object.__setattr__(self, "lower_bound", lower)


@dataclass(frozen=True, slots=True)
class PairedUserClusterBootstrap:
    """Frozen paired component/primary deltas and their bootstrap distribution."""

    gauc: OneSidedLowerBound
    ndcg_at_5: OneSidedLowerBound
    primary: OneSidedLowerBound
    clusters: int
    rows: int
    resamples: int
    seed: int
    alignment_sha256: str
    primary_replicates: tuple[float, ...]

    def __post_init__(self) -> None:
        metrics = (self.gauc, self.ndcg_at_5, self.primary)
        if not all(isinstance(item, OneSidedLowerBound) for item in metrics):
            raise PromotionEvidenceError("bootstrap metrics must be OneSidedLowerBound values")
        if type(self.clusters) is not int or self.clusters <= 0:
            raise PromotionEvidenceError("clusters must be a positive integer")
        if type(self.rows) is not int or self.rows < self.clusters:
            raise PromotionEvidenceError("rows must be an integer no smaller than clusters")
        if self.resamples != BOOTSTRAP_RESAMPLES or self.seed != BOOTSTRAP_SEED:
            raise PromotionEvidenceError("bootstrap resample count or seed differs from policy v1")
        _sha256(self.alignment_sha256, "alignment_sha256")
        if (
            type(self.primary_replicates) is not tuple
            or len(self.primary_replicates) != self.resamples
        ):
            raise PromotionEvidenceError("primary_replicates must contain every frozen replicate")
        if any(not math.isfinite(value) for value in self.primary_replicates):
            raise PromotionEvidenceError("primary_replicates must be finite")

    def lower_bound_at_alpha(self, alpha: float) -> float:
        """Return a lower percentile bound used by the fixed Holm step-down procedure."""

        value = _finite(alpha, "alpha")
        if not 0.0 < value < 1.0:
            raise PromotionEvidenceError("alpha must be strictly between zero and one")
        return float(np.quantile(self.primary_replicates, value, method=_QUANTILE_METHOD))

    def tail_probability(self, margin: float) -> float:
        """Return the conservative empirical mass at or below a frozen null margin."""

        threshold = _finite(margin, "margin")
        count = sum(value <= threshold for value in self.primary_replicates)
        # The plus-one correction prevents a finite Monte Carlo run from reporting p=0.
        return (count + 1.0) / (self.resamples + 1.0)


def _validate_exact_pairing(
    candidate: Sequence[UserClusterMetric], fallback: Sequence[UserClusterMetric]
) -> tuple[tuple[UserClusterMetric, ...], tuple[UserClusterMetric, ...], int]:
    candidate_clusters = tuple(candidate)
    fallback_clusters = tuple(fallback)
    if not candidate_clusters or not fallback_clusters:
        raise PromotionEvidenceError("candidate and fallback cluster evidence cannot be empty")
    if len(candidate_clusters) != len(fallback_clusters):
        raise PromotionEvidenceError("candidate and fallback must contain the same cluster count")
    if any(not isinstance(item, UserClusterMetric) for item in candidate_clusters):
        raise PromotionEvidenceError("candidate evidence must contain UserClusterMetric values")
    if any(not isinstance(item, UserClusterMetric) for item in fallback_clusters):
        raise PromotionEvidenceError("fallback evidence must contain UserClusterMetric values")

    candidate_ids = tuple(item.cluster_id for item in candidate_clusters)
    fallback_ids = tuple(item.cluster_id for item in fallback_clusters)
    if candidate_ids != fallback_ids:
        raise PromotionEvidenceError(
            "candidate and fallback clusters must have identical ordered cluster identity"
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise PromotionEvidenceError("cluster identity cannot be duplicated")

    all_rows: list[int] = []
    previous_first_row = -1
    for index, (candidate_item, fallback_item) in enumerate(
        zip(candidate_clusters, fallback_clusters, strict=True)
    ):
        if candidate_item.row_ids != fallback_item.row_ids:
            raise PromotionEvidenceError(f"cluster {index} has mismatched ordered row identity")
        if candidate_item.gauc_denominator != fallback_item.gauc_denominator:
            raise PromotionEvidenceError(f"cluster {index} has mismatched GAUC eligibility/weight")
        first_row = candidate_item.row_ids[0]
        if first_row <= previous_first_row:
            raise PromotionEvidenceError("clusters must be ordered by first canonical row")
        previous_first_row = first_row
        all_rows.extend(candidate_item.row_ids)

    row_count = len(all_rows)
    if len(set(all_rows)) != row_count:
        raise PromotionEvidenceError("cluster evidence contains duplicated row identity")
    if set(all_rows) != set(range(row_count)):
        raise PromotionEvidenceError(
            "cluster evidence must cover every zero-based canonical row exactly once"
        )
    return candidate_clusters, fallback_clusters, row_count


def _sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PromotionEvidenceError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _alignment_sha256(clusters: tuple[UserClusterMetric, ...]) -> str:
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "cluster_unit": PROMOTION_POLICY_V1.cluster_unit,
            "clusters": [
                {
                    "cluster_id": [
                        "i" if type(item.cluster_id) is int else "s",
                        item.cluster_id,
                    ],
                    "row_ids": list(item.row_ids),
                    "gauc_denominator": item.gauc_denominator,
                }
                for item in clusters
            ],
        }
    )


def _lower_bound(point: float, replicates: Float64Vector) -> OneSidedLowerBound:
    alpha = 1.0 - CONFIDENCE_LEVEL
    return OneSidedLowerBound(
        point_delta=point,
        lower_bound=float(np.quantile(replicates, alpha, method=_QUANTILE_METHOD)),
    )


def paired_user_cluster_bootstrap(
    candidate: Sequence[UserClusterMetric],
    fallback: Sequence[UserClusterMetric],
) -> PairedUserClusterBootstrap:
    """Compute the frozen 10,000-replicate paired user-cluster bootstrap.

    The two inputs must independently enumerate the same eligible organizer users and exact
    canonical rows in the same order.  No inner join, silent drop, reorder, or imputation occurs.
    """

    candidate_clusters, fallback_clusters, row_count = _validate_exact_pairing(candidate, fallback)
    cluster_count = len(candidate_clusters)

    candidate_gauc_numerators = np.asarray(
        [item.gauc_numerator for item in candidate_clusters], dtype=np.float64
    )
    fallback_gauc_numerators = np.asarray(
        [item.gauc_numerator for item in fallback_clusters], dtype=np.float64
    )
    denominators = np.asarray(
        [item.gauc_denominator for item in candidate_clusters], dtype=np.float64
    )
    ndcg_deltas = np.asarray(
        [
            candidate_item.ndcg_at_5 - fallback_item.ndcg_at_5
            for candidate_item, fallback_item in zip(
                candidate_clusters, fallback_clusters, strict=True
            )
        ],
        dtype=np.float64,
    )

    gauc_point = float(
        (math.fsum(candidate_gauc_numerators) - math.fsum(fallback_gauc_numerators))
        / math.fsum(denominators)
    )
    ndcg_point = float(math.fsum(ndcg_deltas) / cluster_count)
    primary_point = (gauc_point + ndcg_point) / 2.0

    gauc_replicates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    ndcg_replicates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Bound temporary memory without changing the deterministic random stream.
    batch_size = max(1, min(256, 2_000_000 // cluster_count))
    offset = 0
    while offset < BOOTSTRAP_RESAMPLES:
        count = min(batch_size, BOOTSTRAP_RESAMPLES - offset)
        sampled = rng.integers(0, cluster_count, size=(count, cluster_count))
        sampled_denominators = denominators[sampled].sum(axis=1)
        if np.any(sampled_denominators <= 0.0):  # guarded by positive per-cluster denominators
            raise PromotionEvidenceError("a bootstrap replicate has no eligible GAUC denominator")
        gauc_replicates[offset : offset + count] = (
            candidate_gauc_numerators[sampled].sum(axis=1)
            - fallback_gauc_numerators[sampled].sum(axis=1)
        ) / sampled_denominators
        ndcg_replicates[offset : offset + count] = ndcg_deltas[sampled].mean(axis=1)
        offset += count

    primary_replicates = np.ascontiguousarray(
        (gauc_replicates + ndcg_replicates) / 2.0, dtype=np.float64
    )
    return PairedUserClusterBootstrap(
        gauc=_lower_bound(gauc_point, gauc_replicates),
        ndcg_at_5=_lower_bound(ndcg_point, ndcg_replicates),
        primary=_lower_bound(primary_point, primary_replicates),
        clusters=cluster_count,
        rows=row_count,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        alignment_sha256=_alignment_sha256(candidate_clusters),
        primary_replicates=tuple(float(value) for value in primary_replicates),
    )


@dataclass(frozen=True, slots=True)
class HolmHypothesis:
    """One ordered hypothesis in a Holm family."""

    finalist_id: str
    rank: int
    raw_p_value: float
    adjusted_p_value: float
    alpha_threshold: float
    rejected: bool

    def __post_init__(self) -> None:
        if type(self.finalist_id) is not str or not self.finalist_id or "\x00" in self.finalist_id:
            raise PromotionEvidenceError("Holm finalist_id must be a non-empty string")
        if type(self.rank) is not int or not 1 <= self.rank <= MAX_FINALISTS:
            raise PromotionEvidenceError("Holm rank must be one or two")
        for name in ("raw_p_value", "adjusted_p_value"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise PromotionEvidenceError(f"{name} must be in [0, 1]")
        threshold = _finite(self.alpha_threshold, "alpha_threshold")
        if not 0.0 < threshold <= FAMILY_WISE_ALPHA:
            raise PromotionEvidenceError("alpha_threshold must be in (0, family-wise alpha]")
        if type(self.rejected) is not bool:
            raise PromotionEvidenceError("Holm rejected must be boolean")


def holm_step_down(p_values: Mapping[str, float]) -> tuple[HolmHypothesis, ...]:
    """Apply fixed-alpha Holm correction to a frozen family of at most two finalists."""

    if not isinstance(p_values, Mapping) or not p_values:
        raise PromotionEvidenceError("Holm correction requires one or two finalist p-values")
    if len(p_values) > MAX_FINALISTS:
        raise PromotionEvidenceError("Holm family cannot contain more than two finalists")
    validated: list[tuple[str, float]] = []
    for finalist_id, p_value in p_values.items():
        if type(finalist_id) is not str or not finalist_id or "\x00" in finalist_id:
            raise PromotionEvidenceError("Holm finalist IDs must be non-empty strings")
        value = _finite(p_value, f"p_values[{finalist_id!r}]")
        if not 0.0 <= value <= 1.0:
            raise PromotionEvidenceError("Holm p-values must be in [0, 1]")
        validated.append((finalist_id, value))
    ordered = sorted(validated, key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    previous_adjusted = 0.0
    continue_rejecting = True
    decisions: list[HolmHypothesis] = []
    for zero_rank, (finalist_id, p_value) in enumerate(ordered):
        multiplier = family_size - zero_rank
        adjusted = min(1.0, max(previous_adjusted, multiplier * p_value))
        threshold = FAMILY_WISE_ALPHA / multiplier
        rejected = continue_rejecting and p_value <= threshold
        if not rejected:
            continue_rejecting = False
        decisions.append(
            HolmHypothesis(
                finalist_id=finalist_id,
                rank=zero_rank + 1,
                raw_p_value=p_value,
                adjusted_p_value=adjusted,
                alpha_threshold=threshold,
                rejected=rejected,
            )
        )
        previous_adjusted = adjusted
    return tuple(decisions)


@dataclass(frozen=True, slots=True)
class HolmBootstrapDecision:
    """Holm result bound to one finalist's adjusted bootstrap lower confidence bound."""

    finalist_id: str
    rank: int
    raw_p_value: float
    adjusted_p_value: float
    alpha_threshold: float
    adjusted_lower_bound: float
    materially_confirmed: bool

    def __post_init__(self) -> None:
        # Reuse the closed scalar validation owned by the generic Holm result.
        HolmHypothesis(
            finalist_id=self.finalist_id,
            rank=self.rank,
            raw_p_value=self.raw_p_value,
            adjusted_p_value=self.adjusted_p_value,
            alpha_threshold=self.alpha_threshold,
            rejected=self.materially_confirmed,
        )
        lower = _finite(self.adjusted_lower_bound, "adjusted_lower_bound")
        if type(self.materially_confirmed) is not bool:
            raise PromotionEvidenceError("materially_confirmed must be boolean")
        if self.materially_confirmed and (
            lower <= MATERIAL_PRIMARY_MARGIN or self.adjusted_p_value > FAMILY_WISE_ALPHA
        ):
            raise PromotionEvidenceError(
                "material confirmation requires a strict material lower bound and adjusted alpha"
            )


def holm_correct_bootstrap(
    finalists: Mapping[str, PairedUserClusterBootstrap],
) -> tuple[HolmBootstrapDecision, ...]:
    """Apply frozen Holm correction to one or two exact finalist bootstrap results."""

    if not isinstance(finalists, Mapping) or not finalists:
        raise PromotionEvidenceError("Holm correction requires one or two finalist results")
    if len(finalists) > MAX_FINALISTS:
        raise PromotionEvidenceError("Holm family cannot contain more than two finalists")
    for finalist_id, result in finalists.items():
        if not isinstance(result, PairedUserClusterBootstrap):
            raise PromotionEvidenceError(
                f"finalists[{finalist_id!r}] must be PairedUserClusterBootstrap"
            )
    first_result = next(iter(finalists.values()))
    if any(
        result.alignment_sha256 != first_result.alignment_sha256
        or result.clusters != first_result.clusters
        or result.rows != first_result.rows
        for result in finalists.values()
    ):
        raise PromotionEvidenceError(
            "Holm finalists must use the same exact cluster and row alignment"
        )
    hypotheses = holm_step_down(
        {
            finalist_id: result.tail_probability(MATERIAL_PRIMARY_MARGIN)
            for finalist_id, result in finalists.items()
        }
    )
    decisions: list[HolmBootstrapDecision] = []
    for hypothesis in hypotheses:
        result = finalists[hypothesis.finalist_id]
        adjusted_lower = result.lower_bound_at_alpha(hypothesis.alpha_threshold)
        decisions.append(
            HolmBootstrapDecision(
                finalist_id=hypothesis.finalist_id,
                rank=hypothesis.rank,
                raw_p_value=hypothesis.raw_p_value,
                adjusted_p_value=hypothesis.adjusted_p_value,
                alpha_threshold=hypothesis.alpha_threshold,
                adjusted_lower_bound=adjusted_lower,
                materially_confirmed=(
                    hypothesis.rejected and adjusted_lower > MATERIAL_PRIMARY_MARGIN
                ),
            )
        )
    return tuple(decisions)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONFIDENCE_LEVEL",
    "FAMILY_WISE_ALPHA",
    "MATERIAL_PRIMARY_MARGIN",
    "MAX_FINALISTS",
    "HolmBootstrapDecision",
    "HolmHypothesis",
    "OneSidedLowerBound",
    "PairedUserClusterBootstrap",
    "PromotionEvidenceError",
    "UserClusterMetric",
    "holm_correct_bootstrap",
    "holm_step_down",
    "paired_user_cluster_bootstrap",
]
