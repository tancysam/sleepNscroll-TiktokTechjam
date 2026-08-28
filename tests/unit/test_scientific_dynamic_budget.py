from __future__ import annotations

from dataclasses import replace

import pytest

from kuairand_agent.campaign.scientific import (
    CampaignStopReason,
    CandidateOutcome,
    ScientificCampaignCancelled,
    ScientificRunEvidence,
    ScientificRunRequest,
    run_scientific_campaign,
)
from tests.unit.test_scientific_campaign import (
    _candidate,
    _config,
    _evidence,
    _fallback,
    _Ledger,
)


def test_nondefault_request_wall_and_reserve_drive_scientific_admission() -> None:
    config = replace(
        _config(),
        wall_clock_seconds=4_200,
        finalization_reserve_seconds=3_600,
        elapsed_seconds_at_start=600.0,
    )
    calls: list[ScientificRunRequest] = []

    def must_not_run(request: ScientificRunRequest) -> ScientificRunEvidence:
        calls.append(request)
        raise AssertionError("reserve admission must reject before runner invocation")

    result = run_scientific_campaign(
        config=config,
        fallback=_fallback(),
        candidates=(_candidate(),),
        runner=must_not_run,
        outer_ledger=_Ledger(),
    )

    assert config.manifest()["wall_clock_seconds"] == 4_200
    assert config.manifest()["finalization_reserve_seconds"] == 3_600
    assert result.stop_reason is CampaignStopReason.FINALIZATION_RESERVE
    assert result.candidates[0].outcome is CandidateOutcome.BUDGET_REJECTED
    assert result.launches_used == config.launches_already_used
    assert calls == []


def test_trusted_cancellation_crosses_candidate_failure_containment() -> None:
    ledger = _Ledger()

    def cancel(_request: ScientificRunRequest) -> ScientificRunEvidence:
        raise ScientificCampaignCancelled("synthetic trusted cancellation")

    with pytest.raises(ScientificCampaignCancelled, match="trusted cancellation"):
        run_scientific_campaign(
            config=_config(),
            fallback=_fallback(),
            candidates=(_candidate(),),
            runner=cancel,
            outer_ledger=ledger,
        )

    assert ledger.reservations == []
    assert ledger.completions == []


def test_durable_cursors_continue_across_one_candidate_driver_calls() -> None:
    config = _config()
    fallback = _fallback()
    ledger = _Ledger()

    def reject_screen(request: ScientificRunRequest) -> ScientificRunEvidence:
        return _evidence(request, 0.1)

    first = run_scientific_campaign(
        config=config,
        fallback=fallback,
        candidates=(_candidate("first"),),
        runner=reject_screen,
        outer_ledger=ledger,
    )
    second = run_scientific_campaign(
        config=config,
        fallback=fallback,
        candidates=(_candidate("second"),),
        runner=reject_screen,
        outer_ledger=ledger,
        initial_incumbent=first.incumbent,
        initial_convergence=first.convergence,
        initial_launches_used=first.launches_used,
        initial_elapsed_seconds=first.elapsed_seconds,
    )

    assert first.convergence.completed_iterations == 1
    assert second.convergence.completed_iterations == 2
    assert second.convergence.non_material_streak == 2
    assert second.launches_used == first.launches_used + 1
    assert second.elapsed_seconds >= first.elapsed_seconds
