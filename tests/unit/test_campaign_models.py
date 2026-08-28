from __future__ import annotations

import json

import pytest

from kuairand_agent.campaign.models import (
    CampaignLifecycle,
    CampaignModelError,
    CampaignState,
    ExperimentLifecycle,
    ExperimentState,
    ScientificIterationCount,
    TrainingLaunchCount,
)

EXPERIMENT_TRANSITIONS = {
    ExperimentState.PROPOSED: {
        ExperimentState.POLICY_REJECTED,
        ExperimentState.MATERIALIZED,
    },
    ExperimentState.POLICY_REJECTED: set(),
    ExperimentState.MATERIALIZED: {
        ExperimentState.STATIC_REJECTED,
        ExperimentState.SMOKE_REJECTED,
        ExperimentState.INNER_RUNNING,
    },
    ExperimentState.STATIC_REJECTED: {ExperimentState.REPAIRING, ExperimentState.CLOSED},
    ExperimentState.SMOKE_REJECTED: {ExperimentState.REPAIRING, ExperimentState.CLOSED},
    ExperimentState.REPAIRING: {ExperimentState.MATERIALIZED, ExperimentState.CLOSED},
    ExperimentState.INNER_RUNNING: {ExperimentState.FAILED, ExperimentState.INNER_SCORED},
    ExperimentState.FAILED: {ExperimentState.REPAIRING, ExperimentState.CLOSED},
    ExperimentState.INNER_SCORED: {
        ExperimentState.INNER_REJECTED,
        ExperimentState.OUTER_ELIGIBLE,
    },
    ExperimentState.INNER_REJECTED: set(),
    ExperimentState.OUTER_ELIGIBLE: {ExperimentState.OUTER_SCORED},
    ExperimentState.OUTER_SCORED: {ExperimentState.PROMOTED, ExperimentState.REJECTED},
    ExperimentState.PROMOTED: {ExperimentState.REFLECTED},
    ExperimentState.REJECTED: {ExperimentState.REFLECTED},
    ExperimentState.REFLECTED: set(),
    ExperimentState.CLOSED: set(),
}


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        (current, next_state)
        for current, allowed in EXPERIMENT_TRANSITIONS.items()
        for next_state in allowed
    ],
)
def test_every_legal_experiment_transition(
    current: ExperimentState, next_state: ExperimentState
) -> None:
    lifecycle = ExperimentLifecycle("exp-1", scientific_iteration=1, state=current, revision=7)
    transitioned = lifecycle.transition(next_state)
    assert transitioned.state is next_state
    assert transitioned.revision == 8
    assert lifecycle.state is current


@pytest.mark.parametrize(
    ("current", "next_state"),
    [
        (current, next_state)
        for current, allowed in EXPERIMENT_TRANSITIONS.items()
        for next_state in ExperimentState
        if next_state not in allowed
    ],
)
def test_every_forbidden_experiment_transition(
    current: ExperimentState, next_state: ExperimentState
) -> None:
    lifecycle = ExperimentLifecycle("exp-1", scientific_iteration=1, state=current)
    with pytest.raises(CampaignModelError, match="forbidden experiment transition"):
        lifecycle.transition(next_state)


def test_terminal_experiment_states_are_immutable() -> None:
    terminal = {state for state, allowed in EXPERIMENT_TRANSITIONS.items() if not allowed}
    for state in terminal:
        lifecycle = ExperimentLifecycle("exp-terminal", 1, state=state)
        assert lifecycle.terminal
        with pytest.raises(CampaignModelError, match="forbidden"):
            lifecycle.transition(ExperimentState.MATERIALIZED)


def test_experiment_manifest_round_trip_is_strict() -> None:
    original = (
        ExperimentLifecycle("exp-restart", 3)
        .transition(ExperimentState.MATERIALIZED)
        .transition(ExperimentState.INNER_RUNNING)
    )
    restored = ExperimentLifecycle.from_manifest(
        json.loads(json.dumps(original.manifest(), sort_keys=True))
    )
    assert restored == original

    unknown = original.manifest()
    unknown["surprise"] = True
    with pytest.raises(CampaignModelError, match="unknown"):
        ExperimentLifecycle.from_manifest(unknown)

    missing = original.manifest()
    del missing["state"]
    with pytest.raises(CampaignModelError, match="missing"):
        ExperimentLifecycle.from_manifest(missing)

    bad_integer = original.manifest()
    bad_integer["revision"] = True
    with pytest.raises(CampaignModelError, match="revision must be an integer"):
        ExperimentLifecycle.from_manifest(bad_integer)


def test_scientific_iterations_and_training_launches_are_distinct_types() -> None:
    iterations = ScientificIterationCount(2).increment()
    launches = TrainingLaunchCount(2).increment()
    assert iterations.value == launches.value == 3
    assert type(iterations) is ScientificIterationCount
    assert type(launches) is TrainingLaunchCount

    with pytest.raises(CampaignModelError, match="non-negative integer"):
        ScientificIterationCount(True)
    with pytest.raises(CampaignModelError, match="non-negative integer"):
        TrainingLaunchCount(-1)


def test_campaign_resume_and_terminal_transitions_are_explicit() -> None:
    campaign = CampaignLifecycle("campaign-1").transition(CampaignState.RUNNING)
    paused = campaign.transition(CampaignState.PAUSED)
    resumed = paused.transition(CampaignState.RUNNING)
    finalizing = resumed.transition(CampaignState.FINALIZATION_REQUIRED).transition(
        CampaignState.FINALIZING
    )
    completed = finalizing.transition(CampaignState.COMPLETED)
    assert completed.terminal
    assert completed.revision == 6
    with pytest.raises(CampaignModelError, match="forbidden campaign transition"):
        completed.transition(CampaignState.RUNNING)
