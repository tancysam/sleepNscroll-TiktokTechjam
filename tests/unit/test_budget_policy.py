from __future__ import annotations

import json
from typing import cast

import pytest

from kuairand_agent.campaign.budgets import (
    CATEGORY_CEILINGS,
    MAX_TRAINING_LAUNCHES,
    AdmissionReason,
    BudgetLedger,
    BudgetPolicyError,
    BudgetReallocation,
    LaunchCategory,
    LaunchRequest,
    RuntimeHistory,
    WorkKind,
    WorkPhase,
)


def _request(
    execution_id: str,
    category: LaunchCategory,
    *,
    phase: WorkPhase = WorkPhase.RESEARCH,
    p95: float = 1.0,
    cleanup: float = 0.0,
) -> LaunchRequest:
    repair = category is LaunchCategory.RECOVERY_RESERVE
    return LaunchRequest(
        execution_id=execution_id,
        family="test-family",
        kind=WorkKind.FULL_TRAIN_EVALUATE,
        phase=phase,
        p95_runtime_seconds=p95,
        cleanup_seconds=cleanup,
        category=category,
        original_category=(LaunchCategory.DIVERSE_INNER_SCREEN if repair else category),
        repair_child=repair,
    )


def _admit_and_charge(
    ledger: BudgetLedger, request: LaunchRequest, *, remaining: float = 21_600.0
) -> BudgetLedger:
    admission = ledger.admit(
        request,
        remaining_seconds=remaining,
        finalization_reserve_seconds=3_600,
    )
    assert admission.allowed, admission.reason
    return ledger.charge_started_launch(admission)


def test_frozen_category_ceilings_total_exactly_fifty() -> None:
    assert dict(CATEGORY_CEILINGS) == {
        LaunchCategory.BASELINE_QUALIFICATION_REPLAY: 6,
        LaunchCategory.DIVERSE_INNER_SCREEN: 20,
        LaunchCategory.TEMPORAL_FOLD_CONFIRMATION: 6,
        LaunchCategory.DISTINCT_OUTER_PROMOTION: 6,
        LaunchCategory.MATCHED_SEED_CONFIRMATION: 5,
        LaunchCategory.BLEND_FUSION: 3,
        LaunchCategory.FINAL_TRAINING_REPLAY: 2,
        LaunchCategory.RECOVERY_RESERVE: 2,
    }
    assert sum(CATEGORY_CEILINGS.values()) == MAX_TRAINING_LAUNCHES == 50


def test_six_qualification_launches_are_imported_before_research() -> None:
    ledger = BudgetLedger.after_qualification()
    assert ledger.training_launches.value == 6
    assert ledger.scientific_iterations.value == 0
    assert ledger.used(LaunchCategory.BASELINE_QUALIFICATION_REPLAY) == 6
    assert all(charge.imported_from_qualification for charge in ledger.charges)


def test_exactly_forty_four_campaign_launches_reach_cap_and_forty_fifth_is_rejected() -> None:
    ledger = BudgetLedger.after_qualification()
    sequence = 0
    for category, ceiling in CATEGORY_CEILINGS.items():
        already_used = ledger.used(category)
        for _ in range(ceiling - already_used):
            sequence += 1
            ledger = _admit_and_charge(ledger, _request(f"campaign-{sequence}", category))
    assert sequence == 44
    assert ledger.training_launches.value == 50
    rejected = ledger.admit(
        _request("campaign-45", LaunchCategory.DIVERSE_INNER_SCREEN),
        remaining_seconds=21_600,
        finalization_reserve_seconds=3_600,
    )
    assert not rejected.allowed
    assert rejected.reason is AdmissionReason.HARD_LAUNCH_CAP
    assert ledger.training_launches.value == 50


def test_category_ceiling_and_explicit_reallocation_preserve_total() -> None:
    ledger = BudgetLedger.after_qualification()
    for index in range(20):
        ledger = _admit_and_charge(
            ledger,
            _request(f"inner-{index}", LaunchCategory.DIVERSE_INNER_SCREEN),
        )
    blocked = ledger.admit(
        _request("inner-over-cap", LaunchCategory.DIVERSE_INNER_SCREEN),
        remaining_seconds=21_600,
        finalization_reserve_seconds=3_600,
    )
    assert blocked.reason is AdmissionReason.CATEGORY_CAP

    transfer = BudgetReallocation(
        source=LaunchCategory.BLEND_FUSION,
        target=LaunchCategory.DIVERSE_INNER_SCREEN,
        amount=1,
        reason="unused blend capacity approved for one additional inner screen",
    )
    reallocated = ledger.approve_reallocation(transfer)
    assert sum(reallocated.effective_ceilings.values()) == 50
    assert reallocated.effective_ceilings[LaunchCategory.BLEND_FUSION] == 2
    assert reallocated.effective_ceilings[LaunchCategory.DIVERSE_INNER_SCREEN] == 21
    charged = _admit_and_charge(reallocated, _request("inner-reallocated", transfer.target))
    assert charged.charges[-1].category is transfer.target
    assert charged.reallocations[-1] == transfer


def test_training_repairs_use_recovery_reserve_and_record_original_category() -> None:
    with pytest.raises(BudgetPolicyError, match="must use recovery_reserve"):
        LaunchRequest(
            execution_id="bad-repair",
            family="fm",
            kind=WorkKind.FULL_TRAIN_EVALUATE,
            phase=WorkPhase.RESEARCH,
            p95_runtime_seconds=1,
            category=LaunchCategory.DIVERSE_INNER_SCREEN,
            original_category=LaunchCategory.DIVERSE_INNER_SCREEN,
            repair_child=True,
        )
    ledger = _admit_and_charge(
        BudgetLedger.after_qualification(),
        _request("repair-1", LaunchCategory.RECOVERY_RESERVE),
    )
    charge = ledger.charges[-1]
    assert charge.repair_child
    assert charge.category is LaunchCategory.RECOVERY_RESERVE
    assert charge.original_category is LaunchCategory.DIVERSE_INNER_SCREEN


@pytest.mark.parametrize(
    "kind",
    [
        WorkKind.CHECKPOINT_INFERENCE_REPLAY,
        WorkKind.STATIC_CHECK,
        WorkKind.SYNTHETIC_SMOKE,
        WorkKind.PROVIDER_ACTION,
    ],
)
def test_static_smoke_provider_and_checkpoint_replay_are_uncharged(kind: WorkKind) -> None:
    ledger = BudgetLedger.after_qualification()
    request = LaunchRequest(
        execution_id=f"uncharged-{kind.value}",
        family="fm",
        kind=kind,
        phase=WorkPhase.RESEARCH,
        p95_runtime_seconds=1,
    )
    admitted = ledger.admit(
        request,
        remaining_seconds=21_600,
        finalization_reserve_seconds=3_600,
    )
    assert admitted.allowed
    assert admitted.projected_launch_number is None
    assert ledger.charge_started_launch(admitted) == ledger


def test_insufficient_time_rejection_does_not_charge() -> None:
    ledger = BudgetLedger.after_qualification()
    rejected = ledger.admit(
        _request("too-slow", LaunchCategory.DIVERSE_INNER_SCREEN, p95=101),
        remaining_seconds=3_700,
        finalization_reserve_seconds=3_600,
    )
    assert rejected.reason is AdmissionReason.FINALIZATION_RESERVE
    assert ledger.training_launches.value == 6
    with pytest.raises(BudgetPolicyError, match="rejected"):
        ledger.charge_started_launch(rejected)


def test_exact_reserve_edge_and_required_completion_admission() -> None:
    ledger = BudgetLedger.after_qualification()
    exact = ledger.admit(
        _request("exact", LaunchCategory.DIVERSE_INNER_SCREEN, p95=100),
        remaining_seconds=3_700,
        finalization_reserve_seconds=3_600,
    )
    assert exact.allowed
    assert exact.required_seconds == 3_700

    reserve_only = ledger.admit(
        _request("reserve-only", LaunchCategory.DIVERSE_INNER_SCREEN, p95=0),
        remaining_seconds=3_600,
        finalization_reserve_seconds=3_600,
    )
    assert reserve_only.reason is AdmissionReason.FINALIZATION_RESERVE

    final = ledger.admit(
        _request(
            "final-replay",
            LaunchCategory.FINAL_TRAINING_REPLAY,
            phase=WorkPhase.FINALIZATION,
            p95=3_600,
        ),
        remaining_seconds=3_600,
        finalization_reserve_seconds=3_600,
    )
    assert final.allowed


@pytest.mark.parametrize("outcome", ["failed", "timeout", "out_of_memory"])
def test_started_failed_timeout_and_oom_launches_remain_charged(outcome: str) -> None:
    ledger = BudgetLedger.after_qualification()
    charged = _admit_and_charge(
        ledger,
        _request(f"started-{outcome}", LaunchCategory.DIVERSE_INNER_SCREEN),
    )
    # Outcome is persisted by the execution store.  The budget ledger has no refund operation.
    assert charged.training_launches.value == 7
    assert charged.charges[-1].execution_id == f"started-{outcome}"


def test_scientific_iteration_counter_is_not_launch_counter() -> None:
    ledger = BudgetLedger.after_qualification(max_scientific_iterations=2)
    ledger = _admit_and_charge(ledger, _request("screen", LaunchCategory.DIVERSE_INNER_SCREEN))
    ledger = _admit_and_charge(
        ledger, _request("fold-a", LaunchCategory.TEMPORAL_FOLD_CONFIRMATION)
    )
    assert ledger.training_launches.value == 8
    assert ledger.scientific_iterations.value == 0
    closed = ledger.complete_scientific_iteration()
    assert closed.training_launches.value == 8
    assert closed.scientific_iterations.value == 1


def test_budget_manifest_resume_keeps_exact_counters_and_rejects_corruption() -> None:
    original = BudgetLedger.after_qualification().complete_scientific_iteration()
    original = _admit_and_charge(
        original, _request("resume-me", LaunchCategory.DIVERSE_INNER_SCREEN)
    )
    restored = BudgetLedger.from_manifest(
        json.loads(json.dumps(original.manifest(), sort_keys=True))
    )
    assert restored == original
    assert restored.training_launches.value == 7
    assert restored.scientific_iterations.value == 1

    corrupt = original.manifest()
    assert isinstance(corrupt["charges"], list)
    charges = cast(list[dict[str, object]], corrupt["charges"])
    charges[0]["imported_from_qualification"] = False
    with pytest.raises(BudgetPolicyError, match="six qualification launches"):
        BudgetLedger.from_manifest(corrupt)


def test_runtime_history_has_deterministic_rolling_p50_and_conservative_p95() -> None:
    history = RuntimeHistory(window_size=3)
    for elapsed in (1.0, 100.0, 2.0, 3.0):
        history = history.record("pairwise-fm", elapsed)
    estimate = history.estimate("pairwise-fm", fallback_seconds=900)
    assert estimate.sample_count == 3
    assert estimate.p50_seconds == 3.0
    assert estimate.p95_seconds == 100.0

    other = history.record("tree", 7.0)
    assert other.estimate("tree", fallback_seconds=900).p95_seconds == 7.0
    fallback = other.estimate("neural", fallback_seconds=1_800)
    assert fallback.sample_count == 0
    assert fallback.p50_seconds == fallback.p95_seconds == 1_800
