"""Paired user-cluster bootstrap diagnostics for organizer ranking metrics.

This module is deliberately a diagnostic, not a scorer or promotion gate.  It accepts only
already-aligned in-memory validation vectors, derives one contribution vector per user using the
organizer's GAUC and nDCG@5 conventions, and resamples whole users.  It never reads a dataset or
artifact and rejects the final phase before inspecting any supplied vector.

Standalone point estimates reproduce the organizer aggregation mathematically.  Production
callers can instead bind the displayed points to two fresh protected ``ScoreResult`` objects;
the percentile intervals remain independently reconstructed in either mode.  Official metric
claims must still come from :mod:`kuairand_agent.scoring.protected`; these intervals only describe
paired uncertainty around candidate-minus-control deltas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Integral
from typing import Final, Literal, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.submission import prediction_digest

type VectorInput = Sequence[object] | npt.NDArray[np.generic]
type UserId = int | str
type Float64Vector = npt.NDArray[np.float64]
type Int64Vector = npt.NDArray[np.int64]

DEFAULT_BOOTSTRAP_RESAMPLES: Final = 2_000
MAX_BOOTSTRAP_RESAMPLES: Final = 100_000
DEFAULT_CONFIDENCE_LEVEL: Final = 0.95
_ALLOWED_PHASES: Final = frozenset({DataPhase.INNER_VALID, DataPhase.OUTER_VALID})
_PROTECTED_SCORER_DIGEST: Final = STARTER_FILE_SHA256["evaluate.py"]

type PointEstimateSource = Literal[
    "independent_diagnostic_reconstruction", "protected_organizer_scorer"
]


@runtime_checkable
class _AggregateScore(Protocol):
    """Legacy aggregate adapter shape without importing quarantined evaluation types."""

    @property
    def gauc(self) -> float: ...

    @property
    def ndcg_at_5(self) -> float: ...

    @property
    def primary(self) -> float: ...

    @property
    def users(self) -> int: ...

    @property
    def rows(self) -> int: ...

    @property
    def scorer_digest(self) -> str: ...

    @property
    def prediction_digest(self) -> str: ...

    @property
    def runtime_seconds(self) -> float: ...


class BootstrapDiagnosticError(ValueError):
    """Raised when paired bootstrap inputs violate the trusted diagnostic contract."""


@dataclass(frozen=True, slots=True)
class PercentileConfidenceInterval:
    """A deterministic equal-tailed percentile interval over paired bootstrap deltas."""

    lower: float
    upper: float
    confidence_level: float
    method: Literal["percentile-linear"] = field(default="percentile-linear", init=False)

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "confidence_level"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise BootstrapDiagnosticError(f"interval {name} must be a finite float")
        if not 0.0 < self.confidence_level < 1.0:
            raise BootstrapDiagnosticError(
                "interval confidence_level must be strictly between 0 and 1"
            )
        if self.lower > self.upper:
            raise BootstrapDiagnosticError("interval lower bound cannot exceed its upper bound")

    def as_dict(self) -> dict[str, float | str]:
        """Return a stable JSON-compatible representation."""

        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class PairedMetricDiagnostic:
    """Point values and paired candidate-minus-control uncertainty for one metric."""

    candidate: float
    control: float
    delta: float
    interval: PercentileConfidenceInterval

    def __post_init__(self) -> None:
        for name in ("candidate", "control", "delta"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise BootstrapDiagnosticError(f"metric {name} must be a finite float")
        if not 0.0 <= self.candidate <= 1.0 or not 0.0 <= self.control <= 1.0:
            raise BootstrapDiagnosticError("candidate and control metric values must be in [0, 1]")
        expected = self.candidate - self.control
        if not math.isclose(self.delta, expected, rel_tol=0.0, abs_tol=1e-15):
            raise BootstrapDiagnosticError("metric delta must equal candidate minus control")
        if not isinstance(self.interval, PercentileConfidenceInterval):
            raise BootstrapDiagnosticError("metric interval must be a PercentileConfidenceInterval")

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "candidate": self.candidate,
            "control": self.control,
            "delta": self.delta,
            "confidence_interval": self.interval.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProtectedPointEstimateProvenance:
    """Exact protected-scorer identity behind displayed bootstrap point estimates."""

    scorer_digest: str
    candidate_prediction_digest: str
    control_prediction_digest: str
    rows: int
    users: int
    primary_aggregation: Literal["selector_decimal_mean_of_protected_gauc_and_ndcg_at_5"] = field(
        default="selector_decimal_mean_of_protected_gauc_and_ndcg_at_5",
        init=False,
    )

    def __post_init__(self) -> None:
        if self.scorer_digest != _PROTECTED_SCORER_DIGEST:
            raise BootstrapDiagnosticError("protected point scorer identity is not pinned")
        for name in ("candidate_prediction_digest", "control_prediction_digest"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise BootstrapDiagnosticError(f"protected point {name} is not a SHA-256 digest")
        if type(self.rows) is not int or self.rows <= 0:
            raise BootstrapDiagnosticError("protected point rows must be a positive integer")
        if type(self.users) is not int or not 1 <= self.users <= self.rows:
            raise BootstrapDiagnosticError("protected point users must be in [1, rows]")

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible provenance binding."""

        return {
            "scorer_digest": self.scorer_digest,
            "candidate_prediction_digest": self.candidate_prediction_digest,
            "control_prediction_digest": self.control_prediction_digest,
            "rows": self.rows,
            "users": self.users,
            "primary_aggregation": self.primary_aggregation,
        }


@dataclass(frozen=True, slots=True)
class UserClusterBootstrapDiagnostic:
    """Paired organizer-metric uncertainty diagnostic over validation users.

    ``decision_use`` is intentionally fixed.  A caller must not reinterpret this interval as an
    incumbent-selection or promotion gate; the campaign's predeclared matched-seed policy owns
    those decisions.
    """

    gauc: PairedMetricDiagnostic
    ndcg_at_5: PairedMetricDiagnostic
    primary: PairedMetricDiagnostic
    rows: int
    users: int
    gauc_eligible_users: int
    resamples: int
    seed: int
    phase: DataPhase
    point_estimate_source: PointEstimateSource = "independent_diagnostic_reconstruction"
    point_estimate_provenance: ProtectedPointEstimateProvenance | None = None
    decision_use: Literal["diagnostic_only"] = field(default="diagnostic_only", init=False)

    def __post_init__(self) -> None:
        if self.point_estimate_source == "independent_diagnostic_reconstruction":
            if self.point_estimate_provenance is not None:
                raise BootstrapDiagnosticError(
                    "independent point estimates cannot declare protected provenance"
                )
        elif self.point_estimate_source == "protected_organizer_scorer":
            if not isinstance(self.point_estimate_provenance, ProtectedPointEstimateProvenance):
                raise BootstrapDiagnosticError(
                    "protected point estimates require protected scorer provenance"
                )
            if (
                self.point_estimate_provenance.rows != self.rows
                or self.point_estimate_provenance.users != self.users
            ):
                raise BootstrapDiagnosticError(
                    "protected point provenance row/user identity changed"
                )
        else:  # pragma: no cover - statically excluded, retained for hostile construction.
            raise BootstrapDiagnosticError("unsupported point-estimate source")

    @property
    def ndcg5(self) -> PairedMetricDiagnostic:
        """Concise alias for the organizer's ``nDCG@5`` spelling."""

        return self.ndcg_at_5

    @property
    def gating_eligible(self) -> Literal[False]:
        """Make the non-gating contract machine-readable."""

        return False

    def as_dict(self) -> dict[str, object]:
        """Return the stable evidence-manifest representation."""

        return {
            "schema_version": 2,
            "decision_use": self.decision_use,
            "gating_eligible": self.gating_eligible,
            "phase": self.phase.value,
            "rows": self.rows,
            "users": self.users,
            "gauc_eligible_users": self.gauc_eligible_users,
            "resamples": self.resamples,
            "seed": self.seed,
            "point_estimate_source": self.point_estimate_source,
            "point_estimate_provenance": (
                None
                if self.point_estimate_provenance is None
                else self.point_estimate_provenance.as_dict()
            ),
            "metrics": {
                "GAUC": self.gauc.as_dict(),
                "nDCG@5": self.ndcg_at_5.as_dict(),
                "primary": self.primary.as_dict(),
            },
        }


@dataclass(frozen=True, slots=True)
class _UserContribution:
    candidate_gauc_numerator: float
    control_gauc_numerator: float
    gauc_denominator: int
    candidate_ndcg: float
    control_ndcg: float

    def order_key(self) -> tuple[int, float, float, float, float]:
        """Order by metric content so relabeling users cannot perturb seeded resamples."""

        return (
            self.gauc_denominator,
            self.candidate_gauc_numerator,
            self.control_gauc_numerator,
            self.candidate_ndcg,
            self.control_ndcg,
        )


def _require_phase(phase: DataPhase) -> None:
    # This check intentionally precedes all vector conversion.  In particular, a caller cannot
    # use the function to validate, summarize, or otherwise touch an outcome vector for final.
    if not isinstance(phase, DataPhase):
        raise BootstrapDiagnosticError("phase must be a DataPhase")
    if phase not in _ALLOWED_PHASES:
        allowed = ", ".join(sorted(item.value for item in _ALLOWED_PHASES))
        raise BootstrapDiagnosticError(
            f"bootstrap diagnostics are allowed only for {allowed}; got {phase.value}"
        )


def _vector(
    value: VectorInput, name: str, *, object_dtype: bool = False
) -> npt.NDArray[np.generic]:
    try:
        result = np.asarray(value, dtype=object if object_dtype else None)
    except (TypeError, ValueError) as exc:
        raise BootstrapDiagnosticError(f"{name} must be a one-dimensional vector") from exc
    if result.ndim != 1:
        raise BootstrapDiagnosticError(f"{name} must be one-dimensional; got shape {result.shape}")
    if result.size == 0:
        raise BootstrapDiagnosticError(f"{name} cannot be empty")
    return result


def _user_id(value: object, location: str) -> UserId:
    if isinstance(value, (bool, np.bool_)):
        raise BootstrapDiagnosticError(f"{location} must be an integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value and "\x00" not in value:
        return value
    raise BootstrapDiagnosticError(f"{location} must be an integer or non-empty string")


def _labels(value: VectorInput) -> npt.NDArray[np.int8]:
    raw = _vector(value, "labels")
    if raw.dtype.kind not in "biuf":
        raise BootstrapDiagnosticError("labels must contain numeric binary values")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BootstrapDiagnosticError("labels must contain numeric binary values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise BootstrapDiagnosticError("labels must contain only binary 0 and 1 values")
    return np.ascontiguousarray(numeric, dtype=np.int8)


def _scores(value: VectorInput, name: str) -> Float64Vector:
    raw = _vector(value, name)
    if raw.dtype.kind not in "iuf":
        raise BootstrapDiagnosticError(f"{name} must contain finite real numbers")
    try:
        result = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BootstrapDiagnosticError(f"{name} must contain finite real numbers") from exc
    if not np.isfinite(result).all():
        raise BootstrapDiagnosticError(f"{name} must contain finite real numbers")
    return result


def _organizer_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Match the immutable organizer's Mann-Whitney U implementation, including ties."""

    pairs = sorted(zip(scores, labels, strict=True))
    ranks = [0.0] * len(pairs)
    left = 0
    while left < len(pairs):
        right = left
        while right + 1 < len(pairs) and pairs[right + 1][0] == pairs[left][0]:
            right += 1
        average_rank = (left + right) / 2.0 + 1.0
        for index in range(left, right + 1):
            ranks[index] = average_rank
        left = right + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, pairs, strict=True) if label == 1
    )
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _organizer_ndcg_at_5(ordered_labels: Sequence[int]) -> float:
    """Match organizer binary-gain nDCG@5, including zero-positive users as zero."""

    discounts = [math.log2(index + 2) for index in range(5)]
    dcg = sum(((2**label) - 1) / discounts[index] for index, label in enumerate(ordered_labels[:5]))
    ideal = sorted(ordered_labels, reverse=True)[:5]
    ideal_dcg = sum(((2**label) - 1) / discounts[index] for index, label in enumerate(ideal))
    return 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg


def _ranked_metrics(rows: Sequence[tuple[int, float]]) -> tuple[float, float, int]:
    ranked = sorted(rows, key=lambda item: -item[1])
    labels = [label for label, _ in ranked]
    scores = [score for _, score in ranked]
    positives = sum(labels)
    denominator = positives if 0 < positives < len(labels) else 0
    numerator = denominator * _organizer_auc(labels, scores) if denominator else 0.0
    return numerator, _organizer_ndcg_at_5(labels), denominator


def _derive_contributions(
    users: tuple[UserId, ...],
    labels: npt.NDArray[np.int8],
    candidate: Float64Vector,
    control: Float64Vector,
) -> tuple[_UserContribution, ...]:
    grouped: dict[UserId, list[tuple[int, float, float]]] = {}
    for user, label, candidate_score, control_score in zip(
        users, labels, candidate, control, strict=True
    ):
        grouped.setdefault(user, []).append(
            (int(label), float(candidate_score), float(control_score))
        )

    contributions: list[_UserContribution] = []
    for rows in grouped.values():
        candidate_rows = [(label, candidate_score) for label, candidate_score, _ in rows]
        control_rows = [(label, control_score) for label, _, control_score in rows]
        candidate_numerator, candidate_ndcg, candidate_denominator = _ranked_metrics(candidate_rows)
        control_numerator, control_ndcg, control_denominator = _ranked_metrics(control_rows)
        if candidate_denominator != control_denominator:  # pragma: no cover - labels are shared.
            raise AssertionError("paired GAUC denominators unexpectedly differ")
        contributions.append(
            _UserContribution(
                candidate_gauc_numerator=candidate_numerator,
                control_gauc_numerator=control_numerator,
                gauc_denominator=candidate_denominator,
                candidate_ndcg=candidate_ndcg,
                control_ndcg=control_ndcg,
            )
        )

    # Seeded bootstrap draws address positions.  Ordering by contribution content makes the
    # finite Monte Carlo result invariant to a bijective user-ID relabeling and to row permutations
    # that preserve the organizer's within-user ordering.  Equal keys have identical effects.
    contributions.sort(key=_UserContribution.order_key)
    return tuple(contributions)


def _point_metrics(
    candidate_gauc_numerator: Float64Vector,
    control_gauc_numerator: Float64Vector,
    gauc_denominator: Int64Vector,
    candidate_ndcg: Float64Vector,
    control_ndcg: Float64Vector,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    denominator = int(sum(int(value) for value in gauc_denominator))
    if denominator:
        candidate_gauc = math.fsum(float(value) for value in candidate_gauc_numerator) / denominator
        control_gauc = math.fsum(float(value) for value in control_gauc_numerator) / denominator
    else:
        candidate_gauc = control_gauc = 0.5
    users = int(candidate_ndcg.size)
    candidate_ndcg_value = math.fsum(float(value) for value in candidate_ndcg) / users
    control_ndcg_value = math.fsum(float(value) for value in control_ndcg) / users
    candidate_primary = (candidate_gauc + candidate_ndcg_value) / 2.0
    control_primary = (control_gauc + control_ndcg_value) / 2.0
    return (
        (candidate_gauc, candidate_ndcg_value, candidate_primary),
        (control_gauc, control_ndcg_value, control_primary),
    )


def _interval(deltas: Float64Vector, confidence_level: float) -> PercentileConfidenceInterval:
    tail = (1.0 - confidence_level) / 2.0
    bounds = np.quantile(deltas, (tail, 1.0 - tail), method="linear")
    return PercentileConfidenceInterval(
        lower=float(bounds[0]),
        upper=float(bounds[1]),
        confidence_level=confidence_level,
    )


def _protected_metrics(
    result: _AggregateScore,
    *,
    location: str,
    expected_rows: int,
    expected_users: int,
    expected_prediction_digest: str,
) -> tuple[float, float, float]:
    """Validate one fresh protected aggregate and return its organizer metric triplet."""

    if not isinstance(result, _AggregateScore):
        raise BootstrapDiagnosticError(f"{location} protected result must be a ScoreResult")
    if result.rows != expected_rows or type(result.rows) is not int:
        raise BootstrapDiagnosticError(f"{location} protected result rows changed")
    if result.users != expected_users or type(result.users) is not int:
        raise BootstrapDiagnosticError(f"{location} protected result users changed")
    if result.prediction_digest != expected_prediction_digest:
        raise BootstrapDiagnosticError(f"{location} protected result prediction digest changed")
    if result.scorer_digest != _PROTECTED_SCORER_DIGEST:
        raise BootstrapDiagnosticError(f"{location} protected result scorer identity changed")
    if (
        type(result.runtime_seconds) is not float
        or not math.isfinite(result.runtime_seconds)
        or result.runtime_seconds < 0.0
    ):
        raise BootstrapDiagnosticError(f"{location} protected scorer runtime is invalid")

    metrics = (result.gauc, result.ndcg_at_5, result.primary)
    if any(type(value) is not float or not math.isfinite(value) for value in metrics):
        raise BootstrapDiagnosticError(f"{location} protected organizer metrics are invalid")
    if any(not 0.0 <= value <= 1.0 for value in metrics):
        raise BootstrapDiagnosticError(f"{location} protected organizer metrics are outside [0, 1]")
    gauc, ndcg, primary = metrics
    exact_float64_primary = (gauc + ndcg) / 2.0
    exact_encoded_float32_primary = float((np.float32(gauc) + np.float32(ndcg)) / 2.0)
    if primary not in {exact_float64_primary, exact_encoded_float32_primary}:
        raise BootstrapDiagnosticError(f"{location} protected organizer primary is malformed")
    # Campaign selection intentionally normalizes primary from the two protected component
    # metrics using decimal-string arithmetic.  The raw ScoreResult primary above remains
    # validated because it proves the untouched evaluator's scalar convention; the displayed
    # bootstrap point uses the selector convention so it is exactly comparable to durable
    # OrganizerMetrics without weakening either contract.
    normalized_primary = float((Decimal(str(gauc)) + Decimal(str(ndcg))) / 2)
    return (gauc, ndcg, normalized_primary)


def paired_user_cluster_bootstrap(
    user_ids: VectorInput,
    labels: VectorInput,
    candidate_scores: VectorInput,
    control_scores: VectorInput,
    *,
    phase: DataPhase,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = 0,
    candidate_protected_result: _AggregateScore | None = None,
    control_protected_result: _AggregateScore | None = None,
) -> UserClusterBootstrapDiagnostic:
    """Return paired user-cluster percentile intervals for organizer metric deltas.

    Every replicate samples ``U`` users with replacement from the ``U`` observed users.  Repeated
    draws are treated as independent copies of the entire user cluster.  GAUC is reconstructed
    from each selected user's organizer numerator and positive-count denominator; nDCG@5 averages
    all selected users, including all-negative users at zero.  The primary delta is recomputed per
    replicate as ``(GAUC delta + nDCG@5 delta) / 2``.

    The result is paired because each cluster draw is shared by candidate and control.  Supplying
    both protected results changes only the displayed point values and their deltas; every
    replicate and interval remains owned by this independent diagnostic reconstruction.  It is a
    non-gating uncertainty diagnostic: official scores and campaign decisions remain owned by the
    protected scorer and predeclared scientific policy.
    """

    _require_phase(phase)
    if (candidate_protected_result is None) != (control_protected_result is None):
        raise BootstrapDiagnosticError(
            "candidate and control protected results must be supplied together"
        )
    if type(resamples) is not int or not 1 <= resamples <= MAX_BOOTSTRAP_RESAMPLES:
        raise BootstrapDiagnosticError(
            f"resamples must be an integer in [1, {MAX_BOOTSTRAP_RESAMPLES}]"
        )
    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float, np.integer, np.floating)
    ):
        raise BootstrapDiagnosticError("confidence_level must be a finite number in (0, 1)")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise BootstrapDiagnosticError("confidence_level must be a finite number in (0, 1)")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise BootstrapDiagnosticError("seed must be an unsigned 32-bit integer")

    raw_users = _vector(user_ids, "user_ids", object_dtype=True)
    normalized_labels = _labels(labels)
    candidate = _scores(candidate_scores, "candidate_scores")
    control = _scores(control_scores, "control_scores")
    row_count = int(raw_users.size)
    lengths: Mapping[str, int] = {
        "labels": int(normalized_labels.size),
        "candidate_scores": int(candidate.size),
        "control_scores": int(control.size),
    }
    for name, length in lengths.items():
        if length != row_count:
            raise BootstrapDiagnosticError(
                f"all vectors must have equal lengths; user_ids has {row_count} rows but "
                f"{name} has {length}"
            )
    users = tuple(
        _user_id(value, f"user_ids[{index}]") for index, value in enumerate(raw_users.tolist())
    )

    contributions = _derive_contributions(users, normalized_labels, candidate, control)
    user_count = len(contributions)
    candidate_gauc_numerator = np.fromiter(
        (item.candidate_gauc_numerator for item in contributions),
        dtype=np.float64,
        count=user_count,
    )
    control_gauc_numerator = np.fromiter(
        (item.control_gauc_numerator for item in contributions),
        dtype=np.float64,
        count=user_count,
    )
    gauc_denominator = np.fromiter(
        (item.gauc_denominator for item in contributions), dtype=np.int64, count=user_count
    )
    candidate_ndcg = np.fromiter(
        (item.candidate_ndcg for item in contributions), dtype=np.float64, count=user_count
    )
    control_ndcg = np.fromiter(
        (item.control_ndcg for item in contributions), dtype=np.float64, count=user_count
    )

    independent_candidate_point, independent_control_point = _point_metrics(
        cast(Float64Vector, candidate_gauc_numerator),
        cast(Float64Vector, control_gauc_numerator),
        cast(Int64Vector, gauc_denominator),
        cast(Float64Vector, candidate_ndcg),
        cast(Float64Vector, control_ndcg),
    )
    provenance: ProtectedPointEstimateProvenance | None = None
    point_source: PointEstimateSource = "independent_diagnostic_reconstruction"
    if candidate_protected_result is None:
        candidate_point = independent_candidate_point
        control_point = independent_control_point
    else:
        if control_protected_result is None:  # pragma: no cover - pairedness checked above.
            raise AssertionError("paired protected result unexpectedly absent")
        candidate_digest = prediction_digest(candidate)
        control_digest = prediction_digest(control)
        candidate_point = _protected_metrics(
            candidate_protected_result,
            location="candidate",
            expected_rows=row_count,
            expected_users=user_count,
            expected_prediction_digest=candidate_digest,
        )
        control_point = _protected_metrics(
            control_protected_result,
            location="control",
            expected_rows=row_count,
            expected_users=user_count,
            expected_prediction_digest=control_digest,
        )
        if candidate_protected_result.scorer_digest != control_protected_result.scorer_digest:
            raise BootstrapDiagnosticError("protected result scorer identities differ")
        provenance = ProtectedPointEstimateProvenance(
            scorer_digest=candidate_protected_result.scorer_digest,
            candidate_prediction_digest=candidate_digest,
            control_prediction_digest=control_digest,
            rows=row_count,
            users=user_count,
        )
        point_source = "protected_organizer_scorer"

    gauc_deltas = np.empty(resamples, dtype=np.float64)
    ndcg_deltas = np.empty(resamples, dtype=np.float64)
    primary_deltas = np.empty(resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(resamples):
        drawn = rng.integers(0, user_count, size=user_count, dtype=np.int64)
        denominator = int(gauc_denominator[drawn].sum(dtype=np.int64))
        if denominator:
            candidate_gauc = float(
                candidate_gauc_numerator[drawn].sum(dtype=np.float64) / denominator
            )
            control_gauc = float(control_gauc_numerator[drawn].sum(dtype=np.float64) / denominator)
            gauc_delta = candidate_gauc - control_gauc
        else:
            # Organizer fallback is 0.5 for both members of the paired comparison.
            gauc_delta = 0.0
        ndcg_delta = float(
            (
                candidate_ndcg[drawn].sum(dtype=np.float64)
                - control_ndcg[drawn].sum(dtype=np.float64)
            )
            / user_count
        )
        gauc_deltas[replicate] = gauc_delta
        ndcg_deltas[replicate] = ndcg_delta
        primary_deltas[replicate] = (gauc_delta + ndcg_delta) / 2.0

    point_deltas = tuple(
        candidate_value - control_value
        for candidate_value, control_value in zip(candidate_point, control_point, strict=True)
    )
    metric_points = tuple(zip(candidate_point, control_point, point_deltas, strict=True))
    intervals = (
        _interval(cast(Float64Vector, gauc_deltas), confidence),
        _interval(cast(Float64Vector, ndcg_deltas), confidence),
        _interval(cast(Float64Vector, primary_deltas), confidence),
    )
    metrics = tuple(
        PairedMetricDiagnostic(
            candidate=float(candidate_value),
            control=float(control_value),
            delta=float(delta),
            interval=interval,
        )
        for (candidate_value, control_value, delta), interval in zip(
            metric_points, intervals, strict=True
        )
    )
    return UserClusterBootstrapDiagnostic(
        gauc=metrics[0],
        ndcg_at_5=metrics[1],
        primary=metrics[2],
        rows=row_count,
        users=user_count,
        gauc_eligible_users=int(np.count_nonzero(gauc_denominator)),
        resamples=resamples,
        seed=seed,
        phase=phase,
        point_estimate_source=point_source,
        point_estimate_provenance=provenance,
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "MAX_BOOTSTRAP_RESAMPLES",
    "BootstrapDiagnosticError",
    "PairedMetricDiagnostic",
    "PercentileConfidenceInterval",
    "ProtectedPointEstimateProvenance",
    "UserClusterBootstrapDiagnostic",
    "paired_user_cluster_bootstrap",
]
