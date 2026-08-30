from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from kuairand_agent.campaign import full_campaign_runtime as runtime
from kuairand_agent.campaign.scientific import CandidateOutcome, ResourceEvidence
from kuairand_agent.campaign.selector import GateEvidence, OrganizerMetrics


def _run(
    *,
    primary: float | None = 0.61,
    gates: GateEvidence | None = None,
    replay_verified: bool = True,
    wall_seconds: float = 2.5,
) -> SimpleNamespace:
    metrics = None if primary is None else OrganizerMetrics(primary, primary)
    return SimpleNamespace(
        metrics=metrics,
        gates=GateEvidence() if gates is None else gates,
        replay_verified=replay_verified,
        resources=ResourceEvidence(
            wall_seconds=wall_seconds,
            peak_rss_bytes=2 * 1024**2,
            disk_bytes=0,
        ),
    )


def _result(
    outcome: CandidateOutcome,
    *,
    runs: tuple[SimpleNamespace, ...] = (),
    candidate_gates: GateEvidence | None = None,
    incumbent_id: str = "official-fm",
) -> Any:
    candidate = SimpleNamespace(
        candidate_id="challenger",
        gates=GateEvidence() if candidate_gates is None else candidate_gates,
    )
    candidate_result = SimpleNamespace(candidate=candidate, outcome=outcome, runs=runs)
    return cast(
        Any,
        SimpleNamespace(
            candidates=(candidate_result,),
            incumbent=SimpleNamespace(candidate_id=incumbent_id),
        ),
    )


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    (
        (CandidateOutcome.BUDGET_REJECTED, "budget_blocked"),
        (CandidateOutcome.CALLBACK_FAILED, "execution_failed"),
        (CandidateOutcome.OUTER_FAILED, "execution_failed"),
    ),
)
def test_reflection_summary_never_substitutes_incumbent_metrics_for_unscored_results(
    outcome: CandidateOutcome,
    expected_status: str,
) -> None:
    summary = runtime._scientific_reflection_summary(
        _result(outcome),
        candidate_id="challenger",
    )

    assert summary.status == expected_status
    assert summary.gauc is None
    assert summary.ndcg_at_5 is None
    assert summary.primary is None
    assert summary.runtime_seconds is None
    assert summary.peak_memory_mb is None


def test_reflection_summary_reports_only_a_completed_scientific_rejection() -> None:
    summary = runtime._scientific_reflection_summary(
        _result(CandidateOutcome.SCREEN_REJECTED, runs=(_run(primary=0.607),)),
        candidate_id="challenger",
    )

    assert summary.tier == "inner"
    assert summary.status == "rejected"
    assert summary.gauc == pytest.approx(0.607)
    assert summary.ndcg_at_5 == pytest.approx(0.607)
    assert summary.primary == pytest.approx(0.607)
    assert summary.runtime_seconds == pytest.approx(2.5)
    assert summary.peak_memory_mb == pytest.approx(2.0)


def test_reflection_summary_fails_closed_when_any_attempted_run_is_invalid() -> None:
    failed = _run(
        primary=None,
        gates=GateEvidence(output_contract=False),
        replay_verified=False,
        wall_seconds=1.75,
    )
    summary = runtime._scientific_reflection_summary(
        _result(
            CandidateOutcome.CALLBACK_FAILED,
            runs=(_run(primary=0.609), failed),
        ),
        candidate_id="challenger",
    )

    assert summary.status == "execution_failed"
    assert summary.primary is None
    assert summary.runtime_seconds == pytest.approx(1.75)


def test_reflection_summary_requires_promotion_outcome_and_incumbent_agreement() -> None:
    promoted = runtime._scientific_reflection_summary(
        _result(
            CandidateOutcome.PROMOTED_CONFIRMED,
            runs=(_run(primary=0.612),),
            incumbent_id="challenger",
        ),
        candidate_id="challenger",
    )
    inconsistent = runtime._scientific_reflection_summary(
        _result(
            CandidateOutcome.PROMOTED_CONFIRMED,
            runs=(_run(primary=0.612),),
            incumbent_id="official-fm",
        ),
        candidate_id="challenger",
    )

    assert promoted.tier == "outer"
    assert promoted.status == "promoted"
    assert promoted.primary == pytest.approx(0.612)
    assert inconsistent.status == "execution_failed"
    assert inconsistent.primary is None


def test_family_blocking_accepts_only_valid_completed_scientific_rejections() -> None:
    valid = _result(CandidateOutcome.INNER_REJECTED, runs=(_run(), _run()))
    budget = _result(CandidateOutcome.BUDGET_REJECTED)
    failed = _result(
        CandidateOutcome.CALLBACK_FAILED,
        runs=(_run(primary=None, gates=GateEvidence(smoke=False)),),
    )

    assert runtime._is_completed_scientific_rejection(valid.candidates[0]) is True
    assert runtime._is_completed_scientific_rejection(budget.candidates[0]) is False
    assert runtime._is_completed_scientific_rejection(failed.candidates[0]) is False


def test_exhausted_outer_budget_allows_only_exact_receipt_attempts() -> None:
    assert runtime._may_attempt_outer_evaluation(
        outer_queries_remaining=1,
        reusable_receipt_count=0,
    )
    assert runtime._may_attempt_outer_evaluation(
        outer_queries_remaining=0,
        reusable_receipt_count=3,
    )
    assert not runtime._may_attempt_outer_evaluation(
        outer_queries_remaining=0,
        reusable_receipt_count=0,
    )
