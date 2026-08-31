"""Frozen final prediction selection with separate submission and scientific decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final

from kuairand_agent.domain.decisions import (
    ReplayGrade,
    ScientificDisposition,
    SubmissionDisposition,
)
from kuairand_agent.domain.identity import PredictionId
from kuairand_agent.evaluation.promotion import PROMOTION_POLICY_V1, PromotionPolicy
from kuairand_agent.evaluation.resampling import HolmBootstrapDecision


class FinalSelectionError(ValueError):
    """Raised when finalization inputs cannot produce a safe fallback decision."""


def _finite_metric(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalSelectionError(f"{location} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FinalSelectionError(f"{location} must be a finite number in [0, 1]")
    return result


def _finite_delta(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalSelectionError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FinalSelectionError(f"{location} must be a finite number")
    return result


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class MetricVector:
    """Full-precision protected GAUC and nDCG@5 with derived primary."""

    gauc: float
    ndcg_at_5: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "gauc", _finite_metric(self.gauc, "gauc"))
        object.__setattr__(self, "ndcg_at_5", _finite_metric(self.ndcg_at_5, "ndcg_at_5"))

    @property
    def primary_decimal(self) -> Decimal:
        return (_decimal(self.gauc) + _decimal(self.ndcg_at_5)) / Decimal(2)

    @property
    def primary(self) -> float:
        return float(self.primary_decimal)


@dataclass(frozen=True, slots=True)
class InnerPromotionEvidence:
    """Frozen inner evidence required before protected-batch eligibility."""

    completed_temporal_folds: int
    confirmation_seeds: tuple[int, ...]
    mean_primary_delta: float
    worst_temporal_fold_primary_delta: float
    mean_gauc_delta: float
    mean_ndcg_at_5_delta: float

    def __post_init__(self) -> None:
        if type(self.completed_temporal_folds) is not int or self.completed_temporal_folds < 0:
            raise FinalSelectionError("completed_temporal_folds must be a non-negative integer")
        if type(self.confirmation_seeds) is not tuple:
            raise FinalSelectionError("confirmation_seeds must be a tuple")
        if len(set(self.confirmation_seeds)) != len(self.confirmation_seeds) or any(
            type(seed) is not int or seed < 0 for seed in self.confirmation_seeds
        ):
            raise FinalSelectionError(
                "confirmation_seeds must contain unique non-negative integers"
            )
        for name in (
            "mean_primary_delta",
            "worst_temporal_fold_primary_delta",
            "mean_gauc_delta",
            "mean_ndcg_at_5_delta",
        ):
            object.__setattr__(self, name, _finite_delta(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class FallbackEvidence:
    """Already-qualified official FM vector that wins ties and evidence failures."""

    prediction_id: PredictionId
    protected_metrics: MetricVector
    qualified_official_fm: bool = True
    resource_receipts_valid: bool = True
    replay_grades: frozenset[ReplayGrade] = frozenset(
        {ReplayGrade.SCORING_EXACT, ReplayGrade.EXPERIMENT_SAME_BACKEND}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_id, PredictionId):
            raise FinalSelectionError("fallback prediction_id must be a PredictionId")
        if not isinstance(self.protected_metrics, MetricVector):
            raise FinalSelectionError("fallback protected_metrics must be a MetricVector")
        if (
            type(self.qualified_official_fm) is not bool
            or type(self.resource_receipts_valid) is not bool
        ):
            raise FinalSelectionError("fallback qualification/resource flags must be boolean")
        if not isinstance(self.replay_grades, frozenset) or any(
            not isinstance(grade, ReplayGrade) for grade in self.replay_grades
        ):
            raise FinalSelectionError("fallback replay_grades must contain ReplayGrade values")


@dataclass(frozen=True, slots=True)
class FinalistEvidence:
    """One exact protected finalist and all evidence needed for frozen selection."""

    prediction_id: PredictionId
    protected_metrics: MetricVector
    inner: InnerPromotionEvidence
    holm: HolmBootstrapDecision | None
    resource_receipts_valid: bool
    replay_grades: frozenset[ReplayGrade]
    evidence_valid: bool = True
    preference_rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_id, PredictionId):
            raise FinalSelectionError("finalist prediction_id must be a PredictionId")
        if not isinstance(self.protected_metrics, MetricVector):
            raise FinalSelectionError("finalist protected_metrics must be a MetricVector")
        if not isinstance(self.inner, InnerPromotionEvidence):
            raise FinalSelectionError("finalist inner must be InnerPromotionEvidence")
        if self.holm is not None and not isinstance(self.holm, HolmBootstrapDecision):
            raise FinalSelectionError("finalist holm must be HolmBootstrapDecision or None")
        if type(self.resource_receipts_valid) is not bool or type(self.evidence_valid) is not bool:
            raise FinalSelectionError("finalist evidence/resource flags must be boolean")
        if not isinstance(self.replay_grades, frozenset) or any(
            not isinstance(grade, ReplayGrade) for grade in self.replay_grades
        ):
            raise FinalSelectionError("finalist replay_grades must contain ReplayGrade values")
        if type(self.preference_rank) is not int or self.preference_rank < 0:
            raise FinalSelectionError("preference_rank must be a non-negative integer")


class SelectionFailure(StrEnum):
    """Machine-readable gates for one finalist assessment."""

    EXPLICIT_INVALID_EVIDENCE = "EXPLICIT_INVALID_EVIDENCE"
    INNER_FOLDS_INCOMPLETE = "INNER_FOLDS_INCOMPLETE"
    INNER_SEEDS_INCOMPLETE = "INNER_SEEDS_INCOMPLETE"
    INNER_PRIMARY_SUBMATERIAL = "INNER_PRIMARY_SUBMATERIAL"
    INNER_WORST_FOLD_REGRESSION = "INNER_WORST_FOLD_REGRESSION"
    INNER_COMPONENT_REGRESSION = "INNER_COMPONENT_REGRESSION"
    RESOURCE_RECEIPT_INVALID = "RESOURCE_RECEIPT_INVALID"
    REPLAY_GRADE_MISSING = "REPLAY_GRADE_MISSING"
    PROTECTED_PRIMARY_NOT_STRICTLY_BETTER = "PROTECTED_PRIMARY_NOT_STRICTLY_BETTER"
    PROTECTED_COMPONENT_REGRESSION = "PROTECTED_COMPONENT_REGRESSION"
    HOLM_EVIDENCE_MISSING = "HOLM_EVIDENCE_MISSING"
    HOLM_IDENTITY_MISMATCH = "HOLM_IDENTITY_MISMATCH"
    HOLM_LOWER_BOUND_NOT_MATERIAL = "HOLM_LOWER_BOUND_NOT_MATERIAL"
    PROTECTED_POINT_DELTA_NOT_MATERIAL = "PROTECTED_POINT_DELTA_NOT_MATERIAL"


_INVALID_EVIDENCE_FAILURES: Final = frozenset(
    {
        SelectionFailure.EXPLICIT_INVALID_EVIDENCE,
        SelectionFailure.INNER_FOLDS_INCOMPLETE,
        SelectionFailure.INNER_SEEDS_INCOMPLETE,
        SelectionFailure.INNER_PRIMARY_SUBMATERIAL,
        SelectionFailure.INNER_WORST_FOLD_REGRESSION,
        SelectionFailure.INNER_COMPONENT_REGRESSION,
        SelectionFailure.RESOURCE_RECEIPT_INVALID,
        SelectionFailure.REPLAY_GRADE_MISSING,
        SelectionFailure.HOLM_EVIDENCE_MISSING,
        SelectionFailure.HOLM_IDENTITY_MISMATCH,
    }
)


@dataclass(frozen=True, slots=True)
class FinalistAssessment:
    """Complete gate result for one finalist without changing campaign state."""

    prediction_id: PredictionId
    gauc_delta: float
    ndcg_at_5_delta: float
    primary_delta: float
    inner_eligible: bool
    evidence_valid_for_submission: bool
    evidence_valid_for_science: bool
    submission_eligible: bool
    materially_confirmed: bool
    failures: tuple[SelectionFailure, ...]


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Separate terminal submission and scientific dispositions for one exact vector."""

    selected_prediction_id: PredictionId
    fallback_prediction_id: PredictionId
    submission_disposition: SubmissionDisposition
    scientific_disposition: ScientificDisposition
    assessments: tuple[FinalistAssessment, ...]


def _inner_failures(
    evidence: InnerPromotionEvidence, policy: PromotionPolicy
) -> tuple[SelectionFailure, ...]:
    failures: list[SelectionFailure] = []
    if evidence.completed_temporal_folds < policy.minimum_completed_temporal_folds:
        failures.append(SelectionFailure.INNER_FOLDS_INCOMPLETE)
    if evidence.confirmation_seeds != policy.confirmation_seeds:
        failures.append(SelectionFailure.INNER_SEEDS_INCOMPLETE)
    if _decimal(evidence.mean_primary_delta) < _decimal(policy.inner_mean_primary_delta_minimum):
        failures.append(SelectionFailure.INNER_PRIMARY_SUBMATERIAL)
    if _decimal(evidence.worst_temporal_fold_primary_delta) < _decimal(
        policy.inner_worst_fold_primary_delta_minimum
    ):
        failures.append(SelectionFailure.INNER_WORST_FOLD_REGRESSION)
    component_minimum = _decimal(policy.inner_mean_component_delta_minimum)
    if (
        _decimal(evidence.mean_gauc_delta) < component_minimum
        or _decimal(evidence.mean_ndcg_at_5_delta) < component_minimum
    ):
        failures.append(SelectionFailure.INNER_COMPONENT_REGRESSION)
    return tuple(failures)


def _assess(
    fallback: FallbackEvidence,
    finalist: FinalistEvidence,
    policy: PromotionPolicy,
) -> FinalistAssessment:
    gauc_delta_decimal = _decimal(finalist.protected_metrics.gauc) - _decimal(
        fallback.protected_metrics.gauc
    )
    ndcg_delta_decimal = _decimal(finalist.protected_metrics.ndcg_at_5) - _decimal(
        fallback.protected_metrics.ndcg_at_5
    )
    primary_delta_decimal = (
        finalist.protected_metrics.primary_decimal - fallback.protected_metrics.primary_decimal
    )
    failures: list[SelectionFailure] = []
    if not finalist.evidence_valid:
        failures.append(SelectionFailure.EXPLICIT_INVALID_EVIDENCE)
    inner_failures = _inner_failures(finalist.inner, policy)
    failures.extend(inner_failures)
    if policy.submission_requires_valid_resource_receipts and not finalist.resource_receipts_valid:
        failures.append(SelectionFailure.RESOURCE_RECEIPT_INVALID)
    if not set(policy.submission_required_replay_grades).issubset(finalist.replay_grades):
        failures.append(SelectionFailure.REPLAY_GRADE_MISSING)

    submission_base_valid = not any(failure in _INVALID_EVIDENCE_FAILURES for failure in failures)
    if primary_delta_decimal <= _decimal(policy.submission_primary_delta_strictly_greater_than):
        failures.append(SelectionFailure.PROTECTED_PRIMARY_NOT_STRICTLY_BETTER)
    component_minimum = _decimal(policy.submission_component_delta_minimum)
    if gauc_delta_decimal < component_minimum or ndcg_delta_decimal < component_minimum:
        failures.append(SelectionFailure.PROTECTED_COMPONENT_REGRESSION)
    submission_eligible = submission_base_valid and not any(
        failure
        in {
            SelectionFailure.PROTECTED_PRIMARY_NOT_STRICTLY_BETTER,
            SelectionFailure.PROTECTED_COMPONENT_REGRESSION,
        }
        for failure in failures
    )

    science_base_valid = submission_base_valid
    if finalist.holm is None:
        failures.append(SelectionFailure.HOLM_EVIDENCE_MISSING)
        science_base_valid = False
    elif finalist.holm.finalist_id != finalist.prediction_id.value:
        failures.append(SelectionFailure.HOLM_IDENTITY_MISMATCH)
        science_base_valid = False
    elif not finalist.holm.materially_confirmed or _decimal(
        finalist.holm.adjusted_lower_bound
    ) <= _decimal(policy.scientific_lower_bound_strictly_greater_than):
        failures.append(SelectionFailure.HOLM_LOWER_BOUND_NOT_MATERIAL)
    if primary_delta_decimal < _decimal(policy.scientific_point_delta_minimum):
        failures.append(SelectionFailure.PROTECTED_POINT_DELTA_NOT_MATERIAL)
    scientific_component_minimum = _decimal(policy.scientific_component_delta_minimum)
    # A component failure may already be present under the identical v1 threshold.
    if (
        gauc_delta_decimal < scientific_component_minimum
        or ndcg_delta_decimal < scientific_component_minimum
    ) and SelectionFailure.PROTECTED_COMPONENT_REGRESSION not in failures:
        failures.append(SelectionFailure.PROTECTED_COMPONENT_REGRESSION)
    materially_confirmed = science_base_valid and not any(
        failure
        in {
            SelectionFailure.HOLM_LOWER_BOUND_NOT_MATERIAL,
            SelectionFailure.PROTECTED_POINT_DELTA_NOT_MATERIAL,
            SelectionFailure.PROTECTED_COMPONENT_REGRESSION,
        }
        for failure in failures
    )
    return FinalistAssessment(
        prediction_id=finalist.prediction_id,
        gauc_delta=float(gauc_delta_decimal),
        ndcg_at_5_delta=float(ndcg_delta_decimal),
        primary_delta=float(primary_delta_decimal),
        inner_eligible=not inner_failures,
        evidence_valid_for_submission=submission_base_valid,
        evidence_valid_for_science=science_base_valid,
        submission_eligible=submission_eligible,
        materially_confirmed=materially_confirmed,
        failures=tuple(failures),
    )


def select_final_prediction(
    fallback: FallbackEvidence,
    finalists: tuple[FinalistEvidence, ...],
    *,
    policy: PromotionPolicy = PROMOTION_POLICY_V1,
) -> SelectionDecision:
    """Select an exact vector while preserving independent scientific truth.

    Invalid evidence, an exact protected tie, a component regression, or a missing operational
    guard can never displace the qualified official FM fallback.
    """

    if not isinstance(fallback, FallbackEvidence):
        raise FinalSelectionError("fallback must be FallbackEvidence")
    if not isinstance(policy, PromotionPolicy):
        raise FinalSelectionError("policy must be PromotionPolicy")
    policy.validate()
    if type(finalists) is not tuple:
        raise FinalSelectionError("finalists must be a tuple")
    if len(finalists) > policy.max_frozen_finalists:
        raise FinalSelectionError("protected finalist family exceeds the frozen maximum of two")
    if any(not isinstance(finalist, FinalistEvidence) for finalist in finalists):
        raise FinalSelectionError("finalists must contain FinalistEvidence values")
    finalist_ids = tuple(finalist.prediction_id for finalist in finalists)
    if len(set(finalist_ids)) != len(finalist_ids):
        raise FinalSelectionError("finalist prediction IDs must be unique")
    if fallback.prediction_id in finalist_ids:
        raise FinalSelectionError("fallback cannot also appear as a challenger finalist")
    required_grades = set(policy.submission_required_replay_grades)
    if (
        not fallback.qualified_official_fm
        or not fallback.resource_receipts_valid
        or not required_grades.issubset(fallback.replay_grades)
    ):
        raise FinalSelectionError("official FM fallback is not safely qualified")

    assessments = tuple(_assess(fallback, finalist, policy) for finalist in finalists)
    assessment_by_id = {assessment.prediction_id: assessment for assessment in assessments}
    eligible = [
        finalist
        for finalist in finalists
        if assessment_by_id[finalist.prediction_id].submission_eligible
    ]
    if not eligible:
        scientific = (
            ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE
            if not finalists
            or not any(assessment.evidence_valid_for_science for assessment in assessments)
            else ScientificDisposition.NOT_CONFIRMED
        )
        return SelectionDecision(
            selected_prediction_id=fallback.prediction_id,
            fallback_prediction_id=fallback.prediction_id,
            submission_disposition=SubmissionDisposition.OFFICIAL_FM_RETAINED,
            scientific_disposition=scientific,
            assessments=assessments,
        )

    # Point score owns submission choice; predeclared preference rank and PredictionId only make
    # a challenger/challenger point tie deterministic.  The official fallback already won its own
    # tie because challenger eligibility is strictly greater than zero.
    selected = min(
        eligible,
        key=lambda item: (
            -item.protected_metrics.primary_decimal,
            item.preference_rank,
            item.prediction_id.value,
        ),
    )
    selected_assessment = assessment_by_id[selected.prediction_id]
    if selected_assessment.materially_confirmed:
        scientific = ScientificDisposition.MATERIALLY_CONFIRMED
    elif selected_assessment.evidence_valid_for_science:
        scientific = ScientificDisposition.NOT_CONFIRMED
    else:
        scientific = ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE
    return SelectionDecision(
        selected_prediction_id=selected.prediction_id,
        fallback_prediction_id=fallback.prediction_id,
        submission_disposition=SubmissionDisposition.CHALLENGER_SELECTED,
        scientific_disposition=scientific,
        assessments=assessments,
    )


__all__ = [
    "FallbackEvidence",
    "FinalSelectionError",
    "FinalistAssessment",
    "FinalistEvidence",
    "InnerPromotionEvidence",
    "MetricVector",
    "SelectionDecision",
    "SelectionFailure",
    "select_final_prediction",
]
