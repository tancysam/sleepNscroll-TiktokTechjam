"""Pure, deterministic inner/outer promotion and incumbent policy."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final

MATERIAL_PRIMARY_DELTA: Final = Decimal("0.002")
WORST_FOLD_DEGRADATION_LIMIT: Final = Decimal("0.002")
HARD_OUTER_CANDIDATE_LIMIT: Final = 6
REQUIRED_FOLDS: Final = frozenset({"A", "B"})


class SelectionPolicyError(ValueError):
    """Raised when trusted selection evidence is malformed."""


def _identifier(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise SelectionPolicyError(f"{location} must be a non-empty string without NUL bytes")
    return value


def _digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionPolicyError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _metric(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionPolicyError(f"{location} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SelectionPolicyError(f"{location} must be a finite number in [0, 1]")
    return result


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class OrganizerMetrics:
    """Organizer GAUC and nDCG@5; primary is their exact arithmetic mean."""

    gauc: float
    ndcg_at_5: float

    def __post_init__(self) -> None:
        _metric(self.gauc, "gauc")
        _metric(self.ndcg_at_5, "ndcg_at_5")

    @property
    def primary(self) -> float:
        return float(self.primary_decimal)

    @property
    def primary_decimal(self) -> Decimal:
        return (_decimal(self.gauc) + _decimal(self.ndcg_at_5)) / 2


@dataclass(frozen=True, slots=True)
class GateEvidence:
    """All non-negotiable trusted gates required before incumbent replacement."""

    policy: bool = True
    imports: bool = True
    smoke: bool = True
    source_identity: bool = True
    data_identity: bool = True
    output_contract: bool = True
    resource_envelope: bool = True
    scorer: bool = True
    replay: bool = True
    serialization_alignment_clean: bool = True

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not bool:
                raise SelectionPolicyError(f"gate {name} must be boolean")

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name in self.__dataclass_fields__ if not getattr(self, name))


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    """Candidate score against both its parent and relevant promotion reference."""

    fold_id: str
    candidate: OrganizerMetrics
    parent: OrganizerMetrics
    reference: OrganizerMetrics

    def __post_init__(self) -> None:
        if self.fold_id not in REQUIRED_FOLDS:
            raise SelectionPolicyError("fold_id must be 'A' or 'B'")
        if not all(
            isinstance(score, OrganizerMetrics)
            for score in (self.candidate, self.parent, self.reference)
        ):
            raise SelectionPolicyError("fold scores must be OrganizerMetrics")

    @property
    def delta_to_parent(self) -> Decimal:
        return self.candidate.primary_decimal - self.parent.primary_decimal

    @property
    def delta_to_reference(self) -> Decimal:
        return self.candidate.primary_decimal - self.reference.primary_decimal


@dataclass(frozen=True, slots=True)
class SeedMetrics:
    seed: int
    metrics: OrganizerMetrics

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise SelectionPolicyError("seed must be an unsigned 32-bit integer")
        if not isinstance(self.metrics, OrganizerMetrics):
            raise SelectionPolicyError("seed metrics must be OrganizerMetrics")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    parent_id: str
    gates: GateEvidence
    folds: tuple[FoldEvidence, ...] = ()
    outer_by_seed: tuple[SeedMetrics, ...] = ()
    diversity_root: bool = False
    metric_specialist_for_blending: bool = False

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate_id")
        _identifier(self.parent_id, "parent_id")
        if self.candidate_id == self.parent_id:
            raise SelectionPolicyError("candidate_id must differ from parent_id")
        if not isinstance(self.gates, GateEvidence):
            raise SelectionPolicyError("gates must be GateEvidence")
        if (
            type(self.diversity_root) is not bool
            or type(self.metric_specialist_for_blending) is not bool
        ):
            raise SelectionPolicyError("candidate flags must be boolean")
        if any(not isinstance(fold, FoldEvidence) for fold in self.folds):
            raise SelectionPolicyError("candidate folds must contain FoldEvidence values")
        if any(not isinstance(item, SeedMetrics) for item in self.outer_by_seed):
            raise SelectionPolicyError("candidate outer evidence must contain SeedMetrics")
        fold_ids = [fold.fold_id for fold in self.folds]
        if len(fold_ids) != len(set(fold_ids)):
            raise SelectionPolicyError("candidate fold evidence contains duplicate fold ids")
        seeds = [item.seed for item in self.outer_by_seed]
        if len(seeds) != len(set(seeds)):
            raise SelectionPolicyError("candidate outer evidence contains duplicate seeds")


@dataclass(frozen=True, slots=True)
class IncumbentEvidence:
    candidate_id: str
    inner_by_fold: tuple[tuple[str, OrganizerMetrics], ...]
    outer_by_seed: tuple[SeedMetrics, ...]
    evidence_receipt_digest: str
    replayable: bool = True
    eligible: bool = True
    official_fm: bool = False

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "incumbent candidate_id")
        _digest(self.evidence_receipt_digest, "incumbent evidence_receipt_digest")
        fold_ids: list[str] = []
        for item in self.inner_by_fold:
            if not isinstance(item, tuple) or len(item) != 2:
                raise SelectionPolicyError("incumbent inner evidence must contain fold pairs")
            fold_id, metrics = item
            if fold_id not in REQUIRED_FOLDS:
                raise SelectionPolicyError("incumbent fold id must be 'A' or 'B'")
            if not isinstance(metrics, OrganizerMetrics):
                raise SelectionPolicyError("incumbent fold scores must be OrganizerMetrics")
            fold_ids.append(fold_id)
        if len(fold_ids) != len(set(fold_ids)):
            raise SelectionPolicyError("incumbent evidence contains duplicate fold ids")
        if any(not isinstance(item, SeedMetrics) for item in self.outer_by_seed):
            raise SelectionPolicyError("incumbent outer evidence must contain SeedMetrics")
        seeds = [item.seed for item in self.outer_by_seed]
        if len(seeds) != len(set(seeds)):
            raise SelectionPolicyError("incumbent evidence contains duplicate seeds")
        if any(
            type(flag) is not bool for flag in (self.replayable, self.eligible, self.official_fm)
        ):
            raise SelectionPolicyError("incumbent flags must be boolean")


@dataclass(frozen=True, slots=True)
class SelectionContext:
    configured_seeds: tuple[int, ...] = (0, 1, 2)
    outer_candidate_ids: frozenset[str] = frozenset()
    outer_promotion_limit: int = HARD_OUTER_CANDIDATE_LIMIT
    screen_margin: float = 0.0
    sufficient_finalization_time: bool = True

    def __post_init__(self) -> None:
        if not self.configured_seeds:
            raise SelectionPolicyError("configured_seeds cannot be empty")
        if len(self.configured_seeds) != len(set(self.configured_seeds)):
            raise SelectionPolicyError("configured_seeds must be unique")
        if any(
            type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in self.configured_seeds
        ):
            raise SelectionPolicyError("configured_seeds must contain unsigned 32-bit integers")
        if type(self.outer_promotion_limit) is not int or not (
            0 <= self.outer_promotion_limit <= HARD_OUTER_CANDIDATE_LIMIT
        ):
            raise SelectionPolicyError("outer_promotion_limit must be an integer in [0, 6]")
        _metric(self.screen_margin, "screen_margin")
        if type(self.sufficient_finalization_time) is not bool:
            raise SelectionPolicyError("sufficient_finalization_time must be boolean")
        for candidate_id in self.outer_candidate_ids:
            _identifier(candidate_id, "outer candidate id")
        if len(self.outer_candidate_ids) > self.outer_promotion_limit:
            raise SelectionPolicyError("outer_candidate_ids cannot exceed outer_promotion_limit")


class SelectionReason(StrEnum):
    STRUCTURAL_GATE_FAILED = "structural_gate_failed"
    INNER_FOLDS_INCOMPLETE = "inner_folds_incomplete"
    FOLD_B_SCREEN_FAILED = "fold_b_screen_failed"
    INNER_MEAN_NOT_POSITIVE = "inner_mean_not_positive"
    WORST_FOLD_GUARD_FAILED = "worst_fold_guard_failed"
    OUTER_CANDIDATE_LIMIT = "outer_candidate_limit"
    INSUFFICIENT_FINALIZATION_TIME = "insufficient_finalization_time"
    OUTER_ELIGIBLE = "outer_eligible"
    MATCHED_SEEDS_REQUIRED = "matched_seeds_required"
    INCUMBENT_EVIDENCE_INCOMPLETE = "incumbent_evidence_incomplete"
    SPECIALIST_ONLY = "specialist_only"
    OUTER_NOT_BETTER = "outer_not_better"
    PROMOTED_MATERIAL = "promoted_material"
    PROMOTED_UNCONFIRMED = "promoted_unconfirmed"


class SelectionOutcome(StrEnum):
    OUTER_REJECTED = "outer_rejected"
    OUTER_ELIGIBLE = "outer_eligible"
    CONFIRMATION_REQUIRED = "confirmation_required"
    RETAIN_INCUMBENT = "retain_incumbent"
    PROMOTE_CONFIRMED = "promote_confirmed"
    PROMOTE_UNCONFIRMED = "promote_unconfirmed"


@dataclass(frozen=True, slots=True)
class OuterEligibilityDecision:
    eligible: bool
    reasons: tuple[SelectionReason, ...]
    failed_gates: tuple[str, ...]
    consumes_outer_slot: bool


@dataclass(frozen=True, slots=True)
class ConfirmationStats:
    seeds: tuple[int, ...]
    paired_primary_deltas: tuple[float, ...]
    mean_primary_delta: float
    median_primary_delta: float
    minimum_primary_delta: float
    population_std_primary_delta: float
    mean_inner_primary_delta: float
    worst_inner_primary_delta: float


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    outcome: SelectionOutcome
    reason: SelectionReason
    selected_candidate_id: str
    challenger_candidate_id: str
    outer: OuterEligibilityDecision
    confirmation: ConfirmationStats | None = None


class Selector:
    """Frozen trusted selector; generated code and the research model cannot override it."""

    def assess_outer_eligibility(
        self, challenger: CandidateEvidence, context: SelectionContext
    ) -> OuterEligibilityDecision:
        if not isinstance(challenger, CandidateEvidence) or not isinstance(
            context, SelectionContext
        ):
            raise SelectionPolicyError("invalid outer eligibility inputs")
        reasons: list[SelectionReason] = []
        failed_gates = challenger.gates.failures
        if failed_gates:
            reasons.append(SelectionReason.STRUCTURAL_GATE_FAILED)

        folds = {fold.fold_id: fold for fold in challenger.folds}
        if set(folds) != REQUIRED_FOLDS:
            reasons.append(SelectionReason.INNER_FOLDS_INCOMPLETE)
        else:
            fold_b = folds["B"]
            screen_delta = fold_b.delta_to_parent
            if not challenger.diversity_root and screen_delta <= Decimal(
                str(context.screen_margin)
            ):
                reasons.append(SelectionReason.FOLD_B_SCREEN_FAILED)
            mean_reference_delta = sum(
                (fold.delta_to_reference for fold in challenger.folds), Decimal(0)
            ) / len(challenger.folds)
            if mean_reference_delta <= Decimal(0):
                reasons.append(SelectionReason.INNER_MEAN_NOT_POSITIVE)
            worst_parent_delta = min(fold.delta_to_parent for fold in challenger.folds)
            if (
                worst_parent_delta < -WORST_FOLD_DEGRADATION_LIMIT
                and not challenger.metric_specialist_for_blending
            ):
                reasons.append(SelectionReason.WORST_FOLD_GUARD_FAILED)

        is_existing_outer_candidate = challenger.candidate_id in context.outer_candidate_ids
        if (
            not is_existing_outer_candidate
            and len(context.outer_candidate_ids) >= context.outer_promotion_limit
        ):
            reasons.append(SelectionReason.OUTER_CANDIDATE_LIMIT)
        if not context.sufficient_finalization_time:
            reasons.append(SelectionReason.INSUFFICIENT_FINALIZATION_TIME)
        eligible = not reasons
        return OuterEligibilityDecision(
            eligible=eligible,
            reasons=(SelectionReason.OUTER_ELIGIBLE,) if eligible else tuple(reasons),
            failed_gates=failed_gates,
            consumes_outer_slot=eligible and not is_existing_outer_candidate,
        )

    def decide(
        self,
        incumbent: IncumbentEvidence,
        challenger: CandidateEvidence,
        context: SelectionContext,
    ) -> SelectionDecision:
        """Apply outer eligibility, matched seeds, and deterministic incumbent replacement."""

        if not isinstance(incumbent, IncumbentEvidence):
            raise SelectionPolicyError("incumbent must be IncumbentEvidence")
        if not incumbent.eligible or not incumbent.replayable:
            raise SelectionPolicyError(
                "incumbent must be eligible and replayable; resolve fallback first"
            )
        outer = self.assess_outer_eligibility(challenger, context)
        if not outer.eligible:
            return SelectionDecision(
                outcome=SelectionOutcome.OUTER_REJECTED,
                reason=outer.reasons[0],
                selected_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                outer=outer,
            )

        challenger_seeds = {item.seed: item.metrics for item in challenger.outer_by_seed}
        incumbent_seeds = {item.seed: item.metrics for item in incumbent.outer_by_seed}
        required = context.configured_seeds
        if not challenger_seeds:
            return SelectionDecision(
                outcome=SelectionOutcome.OUTER_ELIGIBLE,
                reason=SelectionReason.OUTER_ELIGIBLE,
                selected_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                outer=outer,
            )
        if any(seed not in challenger_seeds for seed in required):
            return SelectionDecision(
                outcome=SelectionOutcome.CONFIRMATION_REQUIRED,
                reason=SelectionReason.MATCHED_SEEDS_REQUIRED,
                selected_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                outer=outer,
            )
        incumbent_folds = dict(incumbent.inner_by_fold)
        challenger_folds = {fold.fold_id: fold.candidate for fold in challenger.folds}
        if any(seed not in incumbent_seeds for seed in required) or (
            set(incumbent_folds) != REQUIRED_FOLDS or set(challenger_folds) != REQUIRED_FOLDS
        ):
            return SelectionDecision(
                outcome=SelectionOutcome.CONFIRMATION_REQUIRED,
                reason=SelectionReason.INCUMBENT_EVIDENCE_INCOMPLETE,
                selected_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                outer=outer,
            )

        paired_decimal = tuple(
            challenger_seeds[seed].primary_decimal - incumbent_seeds[seed].primary_decimal
            for seed in required
        )
        inner_paired_decimal = tuple(
            challenger_folds[fold_id].primary_decimal - incumbent_folds[fold_id].primary_decimal
            for fold_id in sorted(REQUIRED_FOLDS)
        )
        paired = tuple(float(value) for value in paired_decimal)
        mean_paired = sum(paired_decimal, Decimal(0)) / len(paired_decimal)
        mean_inner = sum(inner_paired_decimal, Decimal(0)) / len(inner_paired_decimal)
        worst_inner = min(inner_paired_decimal)
        stats = ConfirmationStats(
            seeds=required,
            paired_primary_deltas=paired,
            mean_primary_delta=float(mean_paired),
            median_primary_delta=float(statistics.median(paired)),
            minimum_primary_delta=min(paired),
            population_std_primary_delta=float(statistics.pstdev(paired)),
            mean_inner_primary_delta=float(mean_inner),
            worst_inner_primary_delta=float(worst_inner),
        )

        if challenger.metric_specialist_for_blending and any(
            fold.delta_to_parent < -WORST_FOLD_DEGRADATION_LIMIT for fold in challenger.folds
        ):
            return SelectionDecision(
                outcome=SelectionOutcome.RETAIN_INCUMBENT,
                reason=SelectionReason.SPECIALIST_ONLY,
                selected_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                outer=outer,
                confirmation=stats,
            )
        if mean_inner <= Decimal(0) or worst_inner < -WORST_FOLD_DEGRADATION_LIMIT:
            return SelectionDecision(
                outcome=SelectionOutcome.RETAIN_INCUMBENT,
                reason=(
                    SelectionReason.INNER_MEAN_NOT_POSITIVE
                    if mean_inner <= Decimal(0)
                    else SelectionReason.WORST_FOLD_GUARD_FAILED
                ),
                selected_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                outer=outer,
                confirmation=stats,
            )
        if mean_paired <= Decimal(0):
            return SelectionDecision(
                outcome=SelectionOutcome.RETAIN_INCUMBENT,
                reason=SelectionReason.OUTER_NOT_BETTER,
                selected_candidate_id=incumbent.candidate_id,
                challenger_candidate_id=challenger.candidate_id,
                outer=outer,
                confirmation=stats,
            )
        if mean_paired > MATERIAL_PRIMARY_DELTA:
            outcome = SelectionOutcome.PROMOTE_CONFIRMED
            reason = SelectionReason.PROMOTED_MATERIAL
        else:
            outcome = SelectionOutcome.PROMOTE_UNCONFIRMED
            reason = SelectionReason.PROMOTED_UNCONFIRMED
        return SelectionDecision(
            outcome=outcome,
            reason=reason,
            selected_candidate_id=challenger.candidate_id,
            challenger_candidate_id=challenger.candidate_id,
            outer=outer,
            confirmation=stats,
        )

    def replay_fallback(self, newest_to_oldest: tuple[IncumbentEvidence, ...]) -> IncumbentEvidence:
        """Return the newest replayable eligible ancestor, ultimately the official FM."""

        if not newest_to_oldest:
            raise SelectionPolicyError("incumbent lineage cannot be empty")
        official = [item for item in newest_to_oldest if item.official_fm]
        if len(official) != 1 or not official[0].replayable or not official[0].eligible:
            raise SelectionPolicyError("lineage must contain one replayable eligible official FM")
        for item in newest_to_oldest:
            if item.replayable and item.eligible:
                return item
        raise AssertionError("validated official FM must have been selected")  # pragma: no cover
