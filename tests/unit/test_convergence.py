from __future__ import annotations

import json
import math

import pytest

from kuairand_agent.campaign.convergence import (
    ConvergenceState,
    ConvergenceStateError,
    update_convergence,
)


def test_exact_epsilon_is_non_material_but_advances_best() -> None:
    state = ConvergenceState.initial(0.6000)
    state = update_convergence(state, 0.6020)
    assert state.best_primary == 0.6020
    assert state.non_material_streak == 1
    assert state.completed_iterations == 1


def test_just_above_epsilon_strictly_resets_streak() -> None:
    state = ConvergenceState.initial(0.6000).update_after_iteration(None)
    state = state.update_after_iteration(math.nextafter(0.6020, math.inf))
    assert state.non_material_streak == 0
    assert state.completed_iterations == 2


def test_regression_failure_and_nonpromotion_each_increment_once() -> None:
    state = ConvergenceState.initial(0.6016)
    state = state.update_after_iteration(0.5900)  # eligible outer regression
    state = state.update_after_iteration(None)  # failed iteration
    state = state.update_after_iteration(None)  # policy-valid but not promoted
    assert state.best_primary == 0.6016
    assert state.non_material_streak == 3
    assert state.converged
    assert state.should_stop
    assert not state.may_launch_scientific_iteration
    with pytest.raises(ConvergenceStateError, match="after convergence"):
        state.update_after_iteration(0.9)


def test_repair_does_not_count_separately_from_scientific_iteration() -> None:
    state = ConvergenceState.initial(0.6016)
    before_repair = state.manifest()
    # A repair execution is deliberately not passed to update_after_iteration.
    assert state.manifest() == before_repair
    repaired_iteration = state.update_after_iteration(0.6010)
    assert repaired_iteration.completed_iterations == 1
    assert repaired_iteration.non_material_streak == 1


def test_reserved_confirmation_can_finish_after_convergence() -> None:
    state = ConvergenceState.initial(0.6016).reserve_required_completion()
    for _ in range(3):
        state = state.update_after_iteration(None)
    assert state.converged
    assert not state.may_launch_scientific_iteration
    assert not state.should_stop

    finished = state.finish_required_completion()
    assert finished.completed_iterations == 3
    assert finished.should_stop
    with pytest.raises(ConvergenceStateError, match="reserve new work"):
        finished.reserve_required_completion()


def test_restart_round_trip_preserves_exact_convergence_decision() -> None:
    original = ConvergenceState.initial(0.6016).update_after_iteration(0.6020)
    persisted = json.loads(json.dumps(original.manifest(), sort_keys=True))
    resumed = ConvergenceState.from_manifest(persisted)
    assert resumed == original
    assert resumed.update_after_iteration(0.6040).non_material_streak == 2
    assert resumed.update_after_iteration(0.6040000001).non_material_streak == 0


def test_restart_rejects_unknown_or_invalid_state() -> None:
    manifest = ConvergenceState.initial(0.6016).manifest()
    manifest["surprise"] = True
    with pytest.raises(ConvergenceStateError, match="unknown"):
        ConvergenceState.from_manifest(manifest)

    invalid = ConvergenceState.initial(0.6016).manifest()
    invalid["best_primary"] = float("nan")
    with pytest.raises(ConvergenceStateError, match="finite"):
        ConvergenceState.from_manifest(invalid)
