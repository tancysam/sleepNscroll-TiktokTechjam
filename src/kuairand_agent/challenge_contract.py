"""Executable challenge rules layered on the pinned organizer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

from kuairand_agent.domain.identity import canonical_json_sha256


class ChallengeContractError(ValueError):
    """Raised when competition policy drifts from the operative organizer contract."""


@dataclass(frozen=True, slots=True)
class ChallengeContract:
    """Every challenge rule required before campaign admission."""

    schema_version: int
    dataset: str
    target: str
    ranking_unit: str
    metric_names: tuple[str, ...]
    primary_formula: str
    ndcg_cutoff: int
    convergence_epsilon: float
    convergence_patience: int
    convergence_comparison: str
    max_scientific_iterations: int
    wall_clock_seconds: int
    final_selection_policy: str
    max_protected_evaluations_per_contract_lineage: int
    external_training_data_allowed: bool
    hidden_test_outcomes_representable: bool
    prediction_context: str

    def validate(self) -> Self:
        """Reject any departure from the frozen six-hour competition rules."""

        # Local import prevents a module cycle when ``contract`` composes the full manifest.
        from kuairand_agent.contract import BENCHMARK_CONTRACT

        benchmark = BENCHMARK_CONTRACT.validate()
        expected_metrics = (benchmark.metrics.gauc_name, benchmark.metrics.ndcg_name)
        checks = (
            (self.schema_version == 1, "challenge schema_version must be 1"),
            (self.dataset == benchmark.task.dataset, "challenge dataset must be KuaiRand-Pure"),
            (self.target == benchmark.task.target, "challenge target must be native long_view"),
            (
                self.ranking_unit == benchmark.task.ranking_unit,
                "challenge ranking unit differs from the organizer contract",
            ),
            (
                self.metric_names == expected_metrics,
                "challenge metrics must be exactly GAUC and nDCG@5",
            ),
            (
                self.primary_formula == benchmark.metrics.primary_formula,
                "challenge primary must be mean(GAUC, nDCG@5)",
            ),
            (self.ndcg_cutoff == 5, "challenge nDCG cutoff must be 5"),
            (
                self.convergence_epsilon == benchmark.convergence.epsilon,
                "challenge convergence epsilon must be 0.002",
            ),
            (
                self.convergence_patience == benchmark.convergence.patience,
                "challenge convergence patience must be 3",
            ),
            (
                self.convergence_comparison == benchmark.convergence.comparison,
                "challenge convergence comparison differs from the executable benchmark",
            ),
            (
                self.max_scientific_iterations == 50,
                "challenge permits at most 50 scientific iterations",
            ),
            (self.wall_clock_seconds == 21_600, "challenge wall clock must be six hours"),
            (
                self.final_selection_policy
                == "validation_best_subject_to_promotion_and_fallback_gates",
                "final selection policy changed",
            ),
            (
                self.max_protected_evaluations_per_contract_lineage == 6,
                "protected evaluation lineage budget must be 6",
            ),
            (not self.external_training_data_allowed, "external training data must be forbidden"),
            (
                not self.hidden_test_outcomes_representable,
                "hidden test outcomes must be unrepresentable",
            ),
            (
                self.prediction_context == "strict_past_only",
                "prediction context must be strict past only",
            ),
        )
        for accepted, message in checks:
            if not accepted:
                raise ChallengeContractError(message)
        return self

    def manifest(self) -> dict[str, object]:
        """Return the complete path- and profile-independent challenge manifest."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "target": self.target,
            "ranking_unit": self.ranking_unit,
            "metrics": {
                "names": list(self.metric_names),
                "primary_formula": self.primary_formula,
                "ndcg_cutoff": self.ndcg_cutoff,
            },
            "scientific_iterations": {"maximum": self.max_scientific_iterations},
            "wall_clock_seconds": self.wall_clock_seconds,
            "convergence": {
                "epsilon": self.convergence_epsilon,
                "patience": self.convergence_patience,
                "comparison": self.convergence_comparison,
            },
            "selection": self.final_selection_policy,
            "protected_evaluations": {
                "maximum": self.max_protected_evaluations_per_contract_lineage,
                "lineage": "ContractId",
            },
            "external_training_data_allowed": self.external_training_data_allowed,
            "hidden_test_outcomes_representable": self.hidden_test_outcomes_representable,
            "prediction_context": self.prediction_context,
        }

    @property
    def digest(self) -> str:
        """Canonical identity of the challenge-policy layer."""

        return canonical_json_sha256(self.manifest())


KUAI_PURE_CHALLENGE: Final = ChallengeContract(
    schema_version=1,
    dataset="KuaiRand-Pure",
    target="long_view",
    ranking_unit="within-user ranking over logged impressions",
    metric_names=("GAUC", "nDCG@5"),
    primary_formula="(GAUC + nDCG@5) / 2",
    ndcg_cutoff=5,
    convergence_epsilon=0.002,
    convergence_patience=3,
    convergence_comparison="eligible outer primary delta strictly greater than epsilon",
    max_scientific_iterations=50,
    wall_clock_seconds=21_600,
    final_selection_policy="validation_best_subject_to_promotion_and_fallback_gates",
    max_protected_evaluations_per_contract_lineage=6,
    external_training_data_allowed=False,
    hidden_test_outcomes_representable=False,
    prediction_context="strict_past_only",
).validate()


__all__ = ["KUAI_PURE_CHALLENGE", "ChallengeContract", "ChallengeContractError"]
