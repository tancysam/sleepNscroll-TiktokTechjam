"""Mandatory offline scientific ladder for a valid provider-free campaign."""

from __future__ import annotations

from dataclasses import dataclass

from kuairand_agent.domain.experiment import (
    ExperimentSpec,
    FeatureViewId,
    FoldProtocol,
    GroupingUnit,
    MechanismMetadata,
    ModelFamily,
    Objective,
    RankFusionMember,
    RankFusionRecipe,
    RequiredAblation,
    ResourceClass,
    ScreeningFidelity,
    StrictPastPolicy,
    TemporalEntity,
    TrainingTarget,
)

_DAY_MS = 24 * 60 * 60 * 1000


def _metadata(
    mechanism: str,
    hypothesis: str,
    expected: str,
    rejection: str,
) -> MechanismMetadata:
    return MechanismMetadata(
        mechanism=mechanism,
        falsifiable_hypothesis=hypothesis,
        expected_metric_effect=expected,
        leakage_argument=(
            "Only organizer-approved inputs and train-derived strict-past histories are used; "
            "public-validation and final-period outcomes are absent."
        ),
        rejection_criterion=rejection,
        attributions=("authoritative implementation plan sections 9 and 13",),
    )


@dataclass(frozen=True, slots=True)
class DeterministicProposalLadder:
    """Build the complete, ordered provider-independent experiment sequence."""

    fallback_prediction_ref: str = "official-fm-qualified"
    promotion_policy_version: str = "promotion-v1"

    def proposals(self) -> tuple[ExperimentSpec, ...]:
        official = ExperimentSpec(
            proposal_key="official-fm",
            metadata=_metadata(
                "Qualified sparse user/item factorization machine fallback.",
                "The organizer FM establishes a reproducible non-neural ranking anchor.",
                "Preserve the qualified fallback primary score and exact prediction vector.",
                "Reject qualification on any organizer parity or exact replay mismatch.",
            ),
            parent_experiment_refs=(),
            fallback_prediction_ref=self.fallback_prediction_ref,
            feature_view_ids=(FeatureViewId.OFFICIAL_FM_SPARSE,),
            strict_past=StrictPastPolicy(),
            training_target=TrainingTarget.LONG_VIEW,
            auxiliary_targets=(),
            model_family=ModelFamily.OFFICIAL_FM,
            hyperparameters={
                "k": 16,
                "max_epochs": 40,
                "learning_rate": 0.001,
                "l2": 0.000001,
                "batch_size": 8192,
                "patience": 4,
                "improvement_threshold": 0.00001,
                "predict_batch_size": 200000,
            },
            objective=Objective.OFFICIAL_POINTWISE_LOGLOSS,
            grouping_unit=GroupingUnit.IMPRESSION,
            fold_protocol=FoldProtocol.TEMPORAL_AB,
            seeds=(0,),
            screening_fidelity=ScreeningFidelity.FOLD_B,
            required_ablations=(),
            resource_class=ResourceClass.MEDIUM,
            promotion_policy_version=self.promotion_policy_version,
        )
        lambdarank = ExperimentSpec(
            proposal_key="lambdarank-base",
            metadata=_metadata(
                "Grouped gradient-boosted trees optimized for top-five ranking.",
                "A user-grouped LambdaRank objective extracts nonlinear context interactions "
                "which complement the sparse FM.",
                "Increase nDCG@5 without a hidden GAUC regression.",
                "Close the family when matched folds fail to improve over the official FM.",
            ),
            parent_experiment_refs=(official.proposal_key,),
            fallback_prediction_ref=self.fallback_prediction_ref,
            feature_view_ids=(FeatureViewId.LEAK_SAFE_BASE,),
            strict_past=StrictPastPolicy(),
            training_target=TrainingTarget.LONG_VIEW,
            auxiliary_targets=(),
            model_family=ModelFamily.LIGHTGBM_LAMBDARANK,
            hyperparameters={
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "lambda_l2": 1.0,
                "lambdarank_truncation_level": 8,
            },
            objective=Objective.LAMBDARANK_NDCG5,
            grouping_unit=GroupingUnit.USER,
            fold_protocol=FoldProtocol.TEMPORAL_AB,
            seeds=(0,),
            screening_fidelity=ScreeningFidelity.REDUCED,
            required_ablations=(RequiredAblation.MATCHED_PARENT,),
            resource_class=ResourceClass.MEDIUM,
            promotion_policy_version=self.promotion_policy_version,
        )
        pointwise = ExperimentSpec(
            proposal_key="pointwise-base-control",
            metadata=_metadata(
                "Pointwise tree control using the same leak-safe feature view and capacity.",
                "Holding data and capacity fixed isolates whether the grouped ranking objective "
                "causes the LambdaRank gain.",
                "Falsify objective attribution if pointwise performs equivalently or better.",
                "Treat a non-improvement as valid matched-control evidence, not "
                "infrastructure failure.",
            ),
            parent_experiment_refs=(lambdarank.proposal_key,),
            fallback_prediction_ref=self.fallback_prediction_ref,
            feature_view_ids=(FeatureViewId.LEAK_SAFE_BASE,),
            strict_past=StrictPastPolicy(),
            training_target=TrainingTarget.LONG_VIEW,
            auxiliary_targets=(),
            model_family=ModelFamily.LIGHTGBM_POINTWISE,
            hyperparameters={
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "lambda_l2": 1.0,
            },
            objective=Objective.POINTWISE_BINARY_LOGLOSS,
            grouping_unit=GroupingUnit.IMPRESSION,
            fold_protocol=FoldProtocol.TEMPORAL_AB,
            seeds=(0,),
            screening_fidelity=ScreeningFidelity.REDUCED,
            required_ablations=(RequiredAblation.MATCHED_PARENT,),
            resource_class=ResourceClass.MEDIUM,
            promotion_policy_version=self.promotion_policy_version,
        )
        strict_past_policy = StrictPastPolicy(
            entities=(TemporalEntity.USER, TemporalEntity.VIDEO, TemporalEntity.USER_AUTHOR),
            lookback_windows_ms=(_DAY_MS, 3 * _DAY_MS, 7 * _DAY_MS),
        )
        strict_past = ExperimentSpec(
            proposal_key="lambdarank-strict-past",
            metadata=_metadata(
                "Candidate-aware strict-past recency and frequency companions.",
                "Pre-impression user, video, and user-author histories add time-local signal which "
                "the sparse FM and static tree view cannot represent.",
                "Improve both temporal-fold primary deltas and portfolio complementarity.",
                "Reject if the matched no-history ablation is not worse on the predeclared folds.",
            ),
            parent_experiment_refs=(lambdarank.proposal_key,),
            fallback_prediction_ref=self.fallback_prediction_ref,
            feature_view_ids=(
                FeatureViewId.LEAK_SAFE_BASE,
                FeatureViewId.STRICT_PAST_RECENCY,
                FeatureViewId.STRICT_PAST_FREQUENCY,
            ),
            strict_past=strict_past_policy,
            training_target=TrainingTarget.LONG_VIEW,
            auxiliary_targets=(),
            model_family=ModelFamily.LIGHTGBM_LAMBDARANK,
            hyperparameters=dict(lambdarank.hyperparameters),
            objective=Objective.LAMBDARANK_NDCG5,
            grouping_unit=GroupingUnit.USER,
            fold_protocol=FoldProtocol.TEMPORAL_AB,
            seeds=(0,),
            screening_fidelity=ScreeningFidelity.FOLD_B,
            required_ablations=(
                RequiredAblation.MATCHED_PARENT,
                RequiredAblation.NO_STRICT_PAST_FEATURES,
            ),
            resource_class=ResourceClass.LARGE,
            promotion_policy_version=self.promotion_policy_version,
        )
        strict_past_pointwise = ExperimentSpec(
            proposal_key="pointwise-strict-past-control",
            metadata=_metadata(
                "Pointwise strict-past matched control.",
                "Holding the new histories fixed tests whether their value depends on a ranking "
                "objective rather than generic pointwise discrimination.",
                "Separate feature contribution from objective contribution.",
                "Reject the objective mechanism if the control matches LambdaRank within "
                "tolerance.",
            ),
            parent_experiment_refs=(strict_past.proposal_key,),
            fallback_prediction_ref=self.fallback_prediction_ref,
            feature_view_ids=strict_past.feature_view_ids,
            strict_past=strict_past_policy,
            training_target=TrainingTarget.LONG_VIEW,
            auxiliary_targets=(),
            model_family=ModelFamily.LIGHTGBM_POINTWISE,
            hyperparameters=dict(pointwise.hyperparameters),
            objective=Objective.POINTWISE_BINARY_LOGLOSS,
            grouping_unit=GroupingUnit.IMPRESSION,
            fold_protocol=FoldProtocol.TEMPORAL_AB,
            seeds=(0,),
            screening_fidelity=ScreeningFidelity.FOLD_B,
            required_ablations=(
                RequiredAblation.MATCHED_PARENT,
                RequiredAblation.POINTWISE_OBJECTIVE,
            ),
            resource_class=ResourceClass.LARGE,
            promotion_policy_version=self.promotion_policy_version,
        )
        fusion = ExperimentSpec(
            proposal_key="exact-rank-fusion",
            metadata=_metadata(
                "Exact user-level percentile-rank fusion of frozen diverse predictions.",
                "Combining the sparse FM, base tree ranker, and strict-past specialist reduces "
                "family-specific error when each member has nonzero complementary contribution.",
                "Increase the primary score as one persisted exact vector.",
                "Reject when any member has zero contribution or the exact vector is submaterial.",
            ),
            parent_experiment_refs=(
                official.proposal_key,
                lambdarank.proposal_key,
                strict_past.proposal_key,
            ),
            fallback_prediction_ref=self.fallback_prediction_ref,
            feature_view_ids=(FeatureViewId.FROZEN_PREDICTIONS,),
            strict_past=StrictPastPolicy(),
            training_target=TrainingTarget.LONG_VIEW,
            auxiliary_targets=(),
            model_family=ModelFamily.EXACT_RANK_FUSION,
            hyperparameters={},
            objective=Objective.EXACT_WEIGHTED_RANK,
            grouping_unit=GroupingUnit.USER,
            fold_protocol=FoldProtocol.TEMPORAL_AB,
            seeds=(0,),
            screening_fidelity=ScreeningFidelity.PORTFOLIO,
            required_ablations=(RequiredAblation.SINGLE_MEMBER,),
            resource_class=ResourceClass.SMALL,
            promotion_policy_version=self.promotion_policy_version,
            rank_fusion=RankFusionRecipe(
                members=(
                    RankFusionMember(prediction_ref="official-fm-prediction", weight=0.25),
                    RankFusionMember(prediction_ref="lambdarank-base-prediction", weight=0.35),
                    RankFusionMember(prediction_ref="strict-past-prediction", weight=0.40),
                )
            ),
        )
        return (
            official,
            lambdarank,
            pointwise,
            strict_past,
            strict_past_pointwise,
            fusion,
        )

    def catalog(self) -> dict[str, ExperimentSpec]:
        return {proposal.proposal_key: proposal for proposal in self.proposals()}


def deterministic_proposals(
    *,
    fallback_prediction_ref: str = "official-fm-qualified",
    promotion_policy_version: str = "promotion-v1",
) -> tuple[ExperimentSpec, ...]:
    """Convenience constructor for the mandatory ladder."""

    return DeterministicProposalLadder(
        fallback_prediction_ref=fallback_prediction_ref,
        promotion_policy_version=promotion_policy_version,
    ).proposals()
