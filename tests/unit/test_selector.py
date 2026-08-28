from __future__ import annotations

import math
from dataclasses import replace

import pytest

from kuairand_agent.campaign.selector import (
    CandidateEvidence,
    FoldEvidence,
    GateEvidence,
    IncumbentEvidence,
    OrganizerMetrics,
    SeedMetrics,
    SelectionContext,
    SelectionOutcome,
    SelectionPolicyError,
    SelectionReason,
    Selector,
)


def _metrics(primary: float) -> OrganizerMetrics:
    return OrganizerMetrics(gauc=primary, ndcg_at_5=primary)


def _folds(
    candidate_a: float = 0.603,
    candidate_b: float = 0.603,
    *,
    parent_a: float = 0.600,
    parent_b: float = 0.600,
    reference_a: float = 0.600,
    reference_b: float = 0.600,
) -> tuple[FoldEvidence, ...]:
    return (
        FoldEvidence("A", _metrics(candidate_a), _metrics(parent_a), _metrics(reference_a)),
        FoldEvidence("B", _metrics(candidate_b), _metrics(parent_b), _metrics(reference_b)),
    )


def _seeds(primary: float, seeds: tuple[int, ...] = (0, 1, 2)) -> tuple[SeedMetrics, ...]:
    return tuple(SeedMetrics(seed, _metrics(primary)) for seed in seeds)


def _challenger(
    *,
    gates: GateEvidence | None = None,
    folds: tuple[FoldEvidence, ...] | None = None,
    outer: tuple[SeedMetrics, ...] = (),
    diversity_root: bool = False,
    specialist: bool = False,
    candidate_id: str = "challenger",
) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=candidate_id,
        parent_id="parent",
        gates=gates or GateEvidence(),
        folds=_folds() if folds is None else folds,
        outer_by_seed=outer,
        diversity_root=diversity_root,
        metric_specialist_for_blending=specialist,
    )


def _incumbent(
    outer: float = 0.600,
    *,
    inner_a: float = 0.600,
    inner_b: float = 0.600,
    candidate_id: str = "incumbent",
    replayable: bool = True,
    eligible: bool = True,
    official_fm: bool = False,
) -> IncumbentEvidence:
    return IncumbentEvidence(
        candidate_id=candidate_id,
        inner_by_fold=(("A", _metrics(inner_a)), ("B", _metrics(inner_b))),
        outer_by_seed=_seeds(outer),
        evidence_receipt_digest="f" * 64,
        replayable=replayable,
        eligible=eligible,
        official_fm=official_fm,
    )


@pytest.mark.parametrize(
    "gate",
    [
        "policy",
        "imports",
        "smoke",
        "source_identity",
        "data_identity",
        "output_contract",
        "resource_envelope",
        "scorer",
        "replay",
        "serialization_alignment_clean",
    ],
)
def test_every_structural_gate_fails_closed_before_outer_scoring(gate: str) -> None:
    selector = Selector()
    gates = replace(GateEvidence(), **{gate: False})
    decision = selector.decide(_incumbent(), _challenger(gates=gates), SelectionContext())
    assert decision.outcome is SelectionOutcome.OUTER_REJECTED
    assert decision.reason is SelectionReason.STRUCTURAL_GATE_FAILED
    assert decision.selected_candidate_id == "incumbent"
    assert decision.outer.failed_gates == (gate,)


def test_fold_b_screen_is_strict_and_diversity_root_is_preallocated_exception() -> None:
    selector = Selector()
    exact = _challenger(folds=_folds(candidate_a=0.604, candidate_b=0.601))
    context = SelectionContext(screen_margin=0.001)
    rejected = selector.assess_outer_eligibility(exact, context)
    assert SelectionReason.FOLD_B_SCREEN_FAILED in rejected.reasons

    above = _challenger(
        folds=_folds(candidate_a=0.604, candidate_b=math.nextafter(0.601, math.inf))
    )
    assert selector.assess_outer_eligibility(above, context).eligible

    diversity = _challenger(folds=_folds(candidate_a=0.603, candidate_b=0.599), diversity_root=True)
    assert selector.assess_outer_eligibility(diversity, SelectionContext()).eligible


def test_inner_mean_must_be_positive_and_worst_fold_exact_minus_point_002_passes() -> None:
    selector = Selector()
    zero_mean = _challenger(folds=_folds(candidate_a=0.599, candidate_b=0.601))
    zero = selector.assess_outer_eligibility(zero_mean, SelectionContext())
    assert SelectionReason.INNER_MEAN_NOT_POSITIVE in zero.reasons

    exact_guard = _challenger(folds=_folds(candidate_a=0.598, candidate_b=0.603))
    assert selector.assess_outer_eligibility(exact_guard, SelectionContext()).eligible

    below_guard = _challenger(folds=_folds(candidate_a=0.597_999, candidate_b=0.603))
    rejected = selector.assess_outer_eligibility(below_guard, SelectionContext())
    assert SelectionReason.WORST_FOLD_GUARD_FAILED in rejected.reasons


def test_sixth_distinct_outer_candidate_is_allowed_seventh_is_rejected() -> None:
    selector = Selector()
    five = frozenset(f"candidate-{index}" for index in range(5))
    sixth = selector.assess_outer_eligibility(
        _challenger(), SelectionContext(outer_candidate_ids=five)
    )
    assert sixth.eligible
    assert sixth.consumes_outer_slot

    six = frozenset((*five, "candidate-5"))
    seventh = selector.assess_outer_eligibility(
        _challenger(), SelectionContext(outer_candidate_ids=six)
    )
    assert not seventh.eligible
    assert SelectionReason.OUTER_CANDIDATE_LIMIT in seventh.reasons

    resumed_same = selector.assess_outer_eligibility(
        _challenger(candidate_id="candidate-5"),
        SelectionContext(outer_candidate_ids=six),
    )
    assert resumed_same.eligible
    assert not resumed_same.consumes_outer_slot


def test_outer_promotion_requires_sufficient_finalization_time() -> None:
    decision = Selector().assess_outer_eligibility(
        _challenger(), SelectionContext(sufficient_finalization_time=False)
    )
    assert not decision.eligible
    assert SelectionReason.INSUFFICIENT_FINALIZATION_TIME in decision.reasons


def test_no_outer_score_is_eligible_but_partial_matched_seeds_cannot_promote() -> None:
    selector = Selector()
    eligible = selector.decide(_incumbent(), _challenger(), SelectionContext())
    assert eligible.outcome is SelectionOutcome.OUTER_ELIGIBLE
    assert eligible.selected_candidate_id == "incumbent"

    partial = selector.decide(
        _incumbent(),
        _challenger(outer=_seeds(0.604, seeds=(0, 1))),
        SelectionContext(),
    )
    assert partial.outcome is SelectionOutcome.CONFIRMATION_REQUIRED
    assert partial.reason is SelectionReason.MATCHED_SEEDS_REQUIRED


def test_exact_public_delta_point_002_is_unconfirmed_and_just_above_is_material() -> None:
    selector = Selector()
    exact = selector.decide(
        _incumbent(0.600),
        _challenger(outer=_seeds(0.602)),
        SelectionContext(),
    )
    assert exact.outcome is SelectionOutcome.PROMOTE_UNCONFIRMED
    assert exact.reason is SelectionReason.PROMOTED_UNCONFIRMED
    assert exact.confirmation is not None
    assert exact.confirmation.mean_primary_delta == 0.002

    above_value = math.nextafter(0.602, math.inf)
    above = selector.decide(
        _incumbent(0.600),
        _challenger(outer=_seeds(above_value)),
        SelectionContext(),
    )
    assert above.outcome is SelectionOutcome.PROMOTE_CONFIRMED
    assert above.reason is SelectionReason.PROMOTED_MATERIAL
    assert above.selected_candidate_id == "challenger"


def test_exact_public_tie_and_regression_retain_incumbent() -> None:
    selector = Selector()
    for outer in (0.600, 0.599):
        decision = selector.decide(
            _incumbent(0.600),
            _challenger(outer=_seeds(outer)),
            SelectionContext(),
        )
        assert decision.outcome is SelectionOutcome.RETAIN_INCUMBENT
        assert decision.reason is SelectionReason.OUTER_NOT_BETTER
        assert decision.selected_candidate_id == "incumbent"


def test_metric_specialist_may_consume_outer_slot_but_cannot_replace_fallback() -> None:
    selector = Selector()
    specialist = _challenger(
        folds=_folds(candidate_a=0.596, candidate_b=0.605),
        outer=_seeds(0.610),
        specialist=True,
    )
    outer = selector.assess_outer_eligibility(specialist, SelectionContext())
    assert outer.eligible
    decision = selector.decide(_incumbent(), specialist, SelectionContext())
    assert decision.outcome is SelectionOutcome.RETAIN_INCUMBENT
    assert decision.reason is SelectionReason.SPECIALIST_ONLY


def test_confirmation_statistics_use_matched_seeds_and_inner_guard() -> None:
    challenger_seeds = (
        SeedMetrics(0, _metrics(0.604)),
        SeedMetrics(1, _metrics(0.603)),
        SeedMetrics(2, _metrics(0.605)),
    )
    decision = Selector().decide(
        _incumbent(0.600),
        _challenger(outer=challenger_seeds),
        SelectionContext(),
    )
    assert decision.outcome is SelectionOutcome.PROMOTE_CONFIRMED
    assert decision.confirmation is not None
    assert decision.confirmation.seeds == (0, 1, 2)
    assert decision.confirmation.minimum_primary_delta == pytest.approx(0.003)
    assert decision.confirmation.mean_primary_delta == pytest.approx(0.004)
    assert decision.confirmation.mean_inner_primary_delta == pytest.approx(0.003)
    assert decision.confirmation.worst_inner_primary_delta == pytest.approx(0.003)


def test_replay_failure_cannot_promote_and_falls_back_through_immutable_lineage() -> None:
    selector = Selector()
    invalid = selector.decide(
        _incumbent(),
        _challenger(gates=GateEvidence(replay=False), outer=_seeds(0.700)),
        SelectionContext(),
    )
    assert invalid.outcome is SelectionOutcome.OUTER_REJECTED
    assert invalid.selected_candidate_id == "incumbent"

    current = _incumbent(0.620, candidate_id="current", replayable=False)
    previous = _incumbent(0.610, candidate_id="previous")
    official = _incumbent(0.600, candidate_id="official-fm", official_fm=True)
    assert selector.replay_fallback((current, previous, official)) == previous
    assert selector.replay_fallback((replace(current, eligible=False), official)) == official


def test_lineage_must_retain_one_replayable_official_fm() -> None:
    with pytest.raises(SelectionPolicyError, match="official FM"):
        Selector().replay_fallback((_incumbent(candidate_id="candidate"),))
