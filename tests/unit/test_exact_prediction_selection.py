from __future__ import annotations

from dataclasses import replace

from kuairand_agent.domain.decisions import (
    ReplayGrade,
    ScientificDisposition,
    SubmissionDisposition,
)
from kuairand_agent.domain.identity import PredictionId
from kuairand_agent.evaluation.resampling import HolmBootstrapDecision
from kuairand_agent.finalization.selection import (
    FallbackEvidence,
    FinalistEvidence,
    InnerPromotionEvidence,
    MetricVector,
    SelectionFailure,
    select_final_prediction,
)

_GRADES = frozenset({ReplayGrade.SCORING_EXACT, ReplayGrade.EXPERIMENT_SAME_BACKEND})


def _id(character: str) -> PredictionId:
    return PredictionId(character * 64)


def _inner() -> InnerPromotionEvidence:
    return InnerPromotionEvidence(
        completed_temporal_folds=2,
        confirmation_seeds=(0, 1, 2),
        mean_primary_delta=0.003,
        worst_temporal_fold_primary_delta=-0.002,
        mean_gauc_delta=-0.001,
        mean_ndcg_at_5_delta=0.004,
    )


def _holm(prediction_id: PredictionId, lower: float, confirmed: bool) -> HolmBootstrapDecision:
    return HolmBootstrapDecision(
        finalist_id=prediction_id.value,
        rank=1,
        raw_p_value=0.001,
        adjusted_p_value=0.001,
        alpha_threshold=0.05,
        adjusted_lower_bound=lower,
        materially_confirmed=confirmed,
    )


def _fallback() -> FallbackEvidence:
    return FallbackEvidence(
        prediction_id=_id("f"),
        protected_metrics=MetricVector(0.600, 0.600),
        replay_grades=_GRADES,
    )


def _finalist(
    character: str = "a",
    *,
    metrics: MetricVector | None = None,
    lower: float = 0.003,
    confirmed: bool = True,
) -> FinalistEvidence:
    prediction_id = _id(character)
    return FinalistEvidence(
        prediction_id=prediction_id,
        protected_metrics=metrics or MetricVector(0.603, 0.603),
        inner=_inner(),
        holm=_holm(prediction_id, lower, confirmed),
        resource_receipts_valid=True,
        replay_grades=_GRADES,
    )


def test_exact_challenger_vector_can_be_selected_and_materially_confirmed() -> None:
    fallback = _fallback()
    finalist = _finalist()

    decision = select_final_prediction(fallback, (finalist,))

    assert decision.selected_prediction_id == finalist.prediction_id
    assert decision.fallback_prediction_id == fallback.prediction_id
    assert decision.submission_disposition is SubmissionDisposition.CHALLENGER_SELECTED
    assert decision.scientific_disposition is ScientificDisposition.MATERIALLY_CONFIRMED
    assert decision.assessments[0].primary_delta == 0.003
    assert decision.assessments[0].submission_eligible
    assert decision.assessments[0].materially_confirmed


def test_strictly_better_vector_can_be_selected_without_overclaiming_science() -> None:
    finalist = _finalist(
        metrics=MetricVector(0.601, 0.601),
        lower=0.001,
        confirmed=False,
    )

    decision = select_final_prediction(_fallback(), (finalist,))

    assert decision.selected_prediction_id == finalist.prediction_id
    assert decision.submission_disposition is SubmissionDisposition.CHALLENGER_SELECTED
    assert decision.scientific_disposition is ScientificDisposition.NOT_CONFIRMED
    assert SelectionFailure.HOLM_LOWER_BOUND_NOT_MATERIAL in decision.assessments[0].failures


def test_component_regression_blocks_submission_despite_primary_gain() -> None:
    finalist = _finalist(metrics=MetricVector(0.598, 0.610), lower=0.003)

    decision = select_final_prediction(_fallback(), (finalist,))

    assert decision.selected_prediction_id == decision.fallback_prediction_id
    assert decision.submission_disposition is SubmissionDisposition.OFFICIAL_FM_RETAINED
    assert decision.scientific_disposition is ScientificDisposition.NOT_CONFIRMED
    assert SelectionFailure.PROTECTED_COMPONENT_REGRESSION in decision.assessments[0].failures


def test_exact_tie_retains_qualified_official_fm() -> None:
    finalist = _finalist(metrics=MetricVector(0.600, 0.600), lower=0.003)

    decision = select_final_prediction(_fallback(), (finalist,))

    assert decision.selected_prediction_id == decision.fallback_prediction_id
    assert decision.submission_disposition is SubmissionDisposition.OFFICIAL_FM_RETAINED
    assert (
        SelectionFailure.PROTECTED_PRIMARY_NOT_STRICTLY_BETTER in decision.assessments[0].failures
    )


def test_invalid_or_missing_evidence_retains_fallback_and_is_reported_separately() -> None:
    invalid = replace(_finalist(), evidence_valid=False)
    decision = select_final_prediction(_fallback(), (invalid,))

    assert decision.submission_disposition is SubmissionDisposition.OFFICIAL_FM_RETAINED
    assert decision.scientific_disposition is ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE

    missing_uncertainty = replace(_finalist(metrics=MetricVector(0.601, 0.601)), holm=None)
    decision = select_final_prediction(_fallback(), (missing_uncertainty,))
    assert decision.submission_disposition is SubmissionDisposition.CHALLENGER_SELECTED
    assert decision.scientific_disposition is ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE


def test_exact_selected_prediction_is_highest_eligible_finalist() -> None:
    lower = _finalist("a", metrics=MetricVector(0.603, 0.603))
    higher = _finalist("b", metrics=MetricVector(0.604, 0.604))

    decision = select_final_prediction(_fallback(), (lower, higher))

    assert decision.selected_prediction_id == higher.prediction_id
    assert len(decision.assessments) == 2
