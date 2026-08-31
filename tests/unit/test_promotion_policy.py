from __future__ import annotations

import dataclasses

import pytest

from kuairand_agent.domain.decisions import (
    EvidenceStage,
    ReplayGrade,
    ScientificDisposition,
    SubmissionDisposition,
)
from kuairand_agent.domain.identity import canonical_json_sha256
from kuairand_agent.evaluation.promotion import (
    PROMOTION_POLICY_V1,
    PROMOTION_POLICY_V1_DIGEST,
    PromotionPolicyError,
)


def test_promotion_policy_v1_pins_exact_resampling_and_inner_thresholds() -> None:
    policy = PROMOTION_POLICY_V1

    assert policy.cluster_unit == "organizer_user_identity"
    assert policy.resample_count == 10_000
    assert policy.bootstrap_seed == 20_260_831
    assert policy.confidence_level == 0.95
    assert policy.comparison == "one_sided_95_percent_lower_confidence_bound"
    assert policy.max_frozen_finalists == 2
    assert policy.multiplicity_procedure == "holm"
    assert policy.family_wise_alpha == 0.05
    assert policy.minimum_eligible_cluster_fraction == 1.0
    assert policy.minimum_completed_temporal_folds == 2
    assert policy.confirmation_seeds == (0, 1, 2)
    assert policy.inner_mean_primary_delta_minimum == 0.002
    assert policy.inner_worst_fold_primary_delta_minimum == -0.002
    assert policy.inner_mean_component_delta_minimum == -0.001


def test_submission_and_material_confirmation_gates_remain_distinct() -> None:
    policy = PROMOTION_POLICY_V1

    assert policy.submission_primary_delta_strictly_greater_than == 0.0
    assert policy.submission_component_delta_minimum == -0.001
    assert policy.submission_requires_valid_resource_receipts is True
    assert policy.submission_required_replay_grades == (
        ReplayGrade.SCORING_EXACT,
        ReplayGrade.EXPERIMENT_SAME_BACKEND,
    )
    assert policy.scientific_lower_bound_strictly_greater_than == 0.002
    assert policy.scientific_point_delta_minimum == 0.002
    assert policy.scientific_component_delta_minimum == -0.001
    assert policy.tie_breaking_policy == "retain_simpler_cheaper_already_qualified_fallback"


def test_promotion_policy_manifest_has_one_pinned_canonical_digest() -> None:
    assert PROMOTION_POLICY_V1_DIGEST == (
        "b4d1f8c6cd7157519d403847bcce9e6b470f62072160d5515d09540c45fa149e"
    )
    assert canonical_json_sha256(PROMOTION_POLICY_V1.manifest()) == PROMOTION_POLICY_V1_DIGEST


def test_policy_validation_rejects_post_outcome_tuning() -> None:
    tuned = dataclasses.replace(PROMOTION_POLICY_V1, resample_count=9999)

    with pytest.raises(PromotionPolicyError, match="differs from frozen"):
        tuned.validate()


def test_dispositions_replay_grades_and_evidence_stages_are_separate_closed_types() -> None:
    assert set(SubmissionDisposition) == {
        SubmissionDisposition.CHALLENGER_SELECTED,
        SubmissionDisposition.OFFICIAL_FM_RETAINED,
    }
    assert set(ScientificDisposition) == {
        ScientificDisposition.MATERIALLY_CONFIRMED,
        ScientificDisposition.NOT_CONFIRMED,
        ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE,
    }
    assert set(ReplayGrade) == {
        ReplayGrade.SCORING_EXACT,
        ReplayGrade.EXPERIMENT_SAME_BACKEND,
        ReplayGrade.EXPERIMENT_TOLERANT,
        ReplayGrade.CROSS_BACKEND_PORTABILITY,
        ReplayGrade.BUNDLE_EXACT,
    }
    assert tuple(EvidenceStage) == (
        EvidenceStage.INNER,
        EvidenceStage.OUTER,
        EvidenceStage.FINAL,
    )
