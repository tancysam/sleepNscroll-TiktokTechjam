"""Frozen promotion policy shared by every execution profile and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import canonical_json_sha256


class PromotionPolicyError(ValueError):
    """Raised when a promotion policy differs from frozen version-1 semantics."""


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Executable, profile-independent thresholds for promotion and scientific claims."""

    schema_version: int
    primary_metric_formula: str
    gauc_non_regression_margin: float
    ndcg_at_5_non_regression_margin: float
    practical_primary_improvement_margin: float
    cluster_unit: str
    resample_count: int
    bootstrap_seed: int
    confidence_level: float
    comparison: str
    simultaneous_hypothesis_family: str
    max_frozen_finalists: int
    multiplicity_procedure: str
    family_wise_alpha: float
    tie_breaking_policy: str
    invalid_cluster_handling: str
    minimum_eligible_cluster_fraction: float
    required_temporal_folds: str
    minimum_completed_temporal_folds: int
    confirmation_seeds: tuple[int, ...]
    protected_query_eligibility_rule: str
    inner_mean_primary_delta_minimum: float
    inner_worst_fold_primary_delta_minimum: float
    inner_mean_component_delta_minimum: float
    submission_primary_delta_strictly_greater_than: float
    submission_component_delta_minimum: float
    submission_requires_valid_resource_receipts: bool
    submission_required_replay_grades: tuple[ReplayGrade, ...]
    scientific_lower_bound_strictly_greater_than: float
    scientific_point_delta_minimum: float
    scientific_component_delta_minimum: float
    arithmetic: str
    published_decimal_places: int

    def validate(self) -> Self:
        """Reject post-outcome tuning or a partial reproduction of policy version 1."""

        if self.manifest() != _PROMOTION_POLICY_V1_MANIFEST:
            raise PromotionPolicyError("promotion policy differs from frozen version 1")
        return self

    def manifest(self) -> dict[str, object]:
        """Return a complete value-only manifest suitable for durable policy receipts."""

        return {
            "schema_version": self.schema_version,
            "primary_metric_formula": self.primary_metric_formula,
            "component_non_regression_margins": {
                "GAUC": self.gauc_non_regression_margin,
                "nDCG@5": self.ndcg_at_5_non_regression_margin,
            },
            "practical_primary_improvement_margin": self.practical_primary_improvement_margin,
            "resampling": {
                "cluster_unit": self.cluster_unit,
                "resample_count": self.resample_count,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "comparison": self.comparison,
                "invalid_cluster_handling": self.invalid_cluster_handling,
                "minimum_eligible_cluster_fraction": self.minimum_eligible_cluster_fraction,
            },
            "simultaneous_testing": {
                "hypothesis_family": self.simultaneous_hypothesis_family,
                "max_frozen_finalists": self.max_frozen_finalists,
                "multiplicity_procedure": self.multiplicity_procedure,
                "family_wise_alpha": self.family_wise_alpha,
            },
            "required_inner_evidence": {
                "temporal_folds": self.required_temporal_folds,
                "minimum_completed_temporal_folds": self.minimum_completed_temporal_folds,
                "confirmation_seeds": list(self.confirmation_seeds),
            },
            "protected_query_eligibility": {
                "rule": self.protected_query_eligibility_rule,
                "mean_primary_delta_minimum": self.inner_mean_primary_delta_minimum,
                "worst_temporal_fold_primary_delta_minimum": (
                    self.inner_worst_fold_primary_delta_minimum
                ),
                "each_mean_component_delta_minimum": self.inner_mean_component_delta_minimum,
            },
            "submission_challenger_eligibility": {
                "protected_primary_delta_strictly_greater_than": (
                    self.submission_primary_delta_strictly_greater_than
                ),
                "each_protected_component_delta_minimum": (self.submission_component_delta_minimum),
                "requires_valid_resource_receipts": (
                    self.submission_requires_valid_resource_receipts
                ),
                "required_replay_grades": [
                    grade.value for grade in self.submission_required_replay_grades
                ],
            },
            "material_scientific_confirmation": {
                "holm_adjusted_primary_lower_bound_strictly_greater_than": (
                    self.scientific_lower_bound_strictly_greater_than
                ),
                "protected_primary_point_delta_minimum": self.scientific_point_delta_minimum,
                "each_protected_component_delta_minimum": (self.scientific_component_delta_minimum),
            },
            "tie_breaking_policy": self.tie_breaking_policy,
            "arithmetic": self.arithmetic,
            "published_decimal_places": self.published_decimal_places,
        }

    @property
    def digest(self) -> str:
        """SHA-256 of the exact canonical value-only policy manifest."""

        return canonical_json_sha256(self.manifest())


_PROMOTION_POLICY_V1_MANIFEST: Final[dict[str, object]] = {
    "schema_version": 1,
    "primary_metric_formula": "(GAUC + nDCG@5) / 2",
    "component_non_regression_margins": {"GAUC": -0.001, "nDCG@5": -0.001},
    "practical_primary_improvement_margin": 0.002,
    "resampling": {
        "cluster_unit": "organizer_user_identity",
        "resample_count": 10_000,
        "bootstrap_seed": 20_260_831,
        "confidence_level": 0.95,
        "comparison": "one_sided_95_percent_lower_confidence_bound",
        "invalid_cluster_handling": "hard_evidence_failure_no_denominator_adjustment",
        "minimum_eligible_cluster_fraction": 1.0,
    },
    "simultaneous_testing": {
        "hypothesis_family": "all_frozen_finalists_in_protected_batch",
        "max_frozen_finalists": 2,
        "multiplicity_procedure": "holm",
        "family_wise_alpha": 0.05,
    },
    "required_inner_evidence": {
        "temporal_folds": "all_configured_temporal_folds",
        "minimum_completed_temporal_folds": 2,
        "confirmation_seeds": [0, 1, 2],
    },
    "protected_query_eligibility": {
        "rule": "no_protected_query_for_submaterial_inner_delta",
        "mean_primary_delta_minimum": 0.002,
        "worst_temporal_fold_primary_delta_minimum": -0.002,
        "each_mean_component_delta_minimum": -0.001,
    },
    "submission_challenger_eligibility": {
        "protected_primary_delta_strictly_greater_than": 0.0,
        "each_protected_component_delta_minimum": -0.001,
        "requires_valid_resource_receipts": True,
        "required_replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        ],
    },
    "material_scientific_confirmation": {
        "holm_adjusted_primary_lower_bound_strictly_greater_than": 0.002,
        "protected_primary_point_delta_minimum": 0.002,
        "each_protected_component_delta_minimum": -0.001,
    },
    "tie_breaking_policy": "retain_simpler_cheaper_already_qualified_fallback",
    "arithmetic": "full_precision_internal_metrics",
    "published_decimal_places": 4,
}

PROMOTION_POLICY_V1: Final = PromotionPolicy(
    schema_version=1,
    primary_metric_formula="(GAUC + nDCG@5) / 2",
    gauc_non_regression_margin=-0.001,
    ndcg_at_5_non_regression_margin=-0.001,
    practical_primary_improvement_margin=0.002,
    cluster_unit="organizer_user_identity",
    resample_count=10_000,
    bootstrap_seed=20_260_831,
    confidence_level=0.95,
    comparison="one_sided_95_percent_lower_confidence_bound",
    simultaneous_hypothesis_family="all_frozen_finalists_in_protected_batch",
    max_frozen_finalists=2,
    multiplicity_procedure="holm",
    family_wise_alpha=0.05,
    tie_breaking_policy="retain_simpler_cheaper_already_qualified_fallback",
    invalid_cluster_handling="hard_evidence_failure_no_denominator_adjustment",
    minimum_eligible_cluster_fraction=1.0,
    required_temporal_folds="all_configured_temporal_folds",
    minimum_completed_temporal_folds=2,
    confirmation_seeds=(0, 1, 2),
    protected_query_eligibility_rule="no_protected_query_for_submaterial_inner_delta",
    inner_mean_primary_delta_minimum=0.002,
    inner_worst_fold_primary_delta_minimum=-0.002,
    inner_mean_component_delta_minimum=-0.001,
    submission_primary_delta_strictly_greater_than=0.0,
    submission_component_delta_minimum=-0.001,
    submission_requires_valid_resource_receipts=True,
    submission_required_replay_grades=(
        ReplayGrade.SCORING_EXACT,
        ReplayGrade.EXPERIMENT_SAME_BACKEND,
    ),
    scientific_lower_bound_strictly_greater_than=0.002,
    scientific_point_delta_minimum=0.002,
    scientific_component_delta_minimum=-0.001,
    arithmetic="full_precision_internal_metrics",
    published_decimal_places=4,
).validate()

# This literal is the durable policy identity recorded by campaigns and decisions.  Import fails
# closed if a source edit changes policy values without an explicit versioned digest update.
PROMOTION_POLICY_V1_DIGEST: Final = (
    "b4d1f8c6cd7157519d403847bcce9e6b470f62072160d5515d09540c45fa149e"
)
if PROMOTION_POLICY_V1.digest != PROMOTION_POLICY_V1_DIGEST:  # pragma: no cover - import guard
    raise RuntimeError("frozen PromotionPolicy v1 digest differs from its canonical manifest")


__all__ = [
    "PROMOTION_POLICY_V1",
    "PROMOTION_POLICY_V1_DIGEST",
    "PromotionPolicy",
    "PromotionPolicyError",
]
