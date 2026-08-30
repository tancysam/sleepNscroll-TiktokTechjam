from __future__ import annotations

import json
import math

import pytest

from kuairand_agent.campaign.convergence import (
    CompletedScientificIteration,
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


def test_cumulative_sub_epsilon_improvements_reset_patience_window() -> None:
    state = ConvergenceState.initial(0.6000)
    state = state.update_after_iteration(0.6007)
    state = state.update_after_iteration(0.6014)
    state = state.update_after_iteration(0.6021)

    assert state.best_primary == 0.6021
    assert state.non_material_streak == 0
    assert not state.converged


def test_cumulative_improvement_exactly_at_epsilon_does_not_reset_patience_window() -> None:
    state = ConvergenceState.initial(0.6000)
    state = state.update_after_iteration(0.6010)
    state = state.update_after_iteration(0.6020)

    assert state.best_primary == 0.6020
    assert state.patience_window_start_primary == 0.6000
    assert state.non_material_streak == 2


def test_typed_completed_iteration_is_the_observation_seam() -> None:
    state = ConvergenceState.initial(0.6000)

    observed = state.observe(CompletedScientificIteration(eligible_outer_primary=0.6020))
    nonpromoted = observed.observe(CompletedScientificIteration(eligible_outer_primary=None))

    assert observed.manifest() == {
        "schema_version": 2,
        "best_primary": 0.6020,
        "patience_window_start_primary": 0.6000,
        "non_material_streak": 1,
        "completed_iterations": 1,
        "required_completion_pending": False,
    }
    assert nonpromoted.completed_iterations == 2
    assert nonpromoted.non_material_streak == 2
    with pytest.raises(ConvergenceStateError, match="CompletedScientificIteration"):
        state.observe(None)  # type: ignore[arg-type]
    assert state.completed_iterations == 0


def test_just_above_epsilon_strictly_resets_streak() -> None:
    state = ConvergenceState.initial(0.6000).update_after_iteration(None)
    state = state.update_after_iteration(math.nextafter(0.6020, math.inf))
    assert state.non_material_streak == 0
    assert state.completed_iterations == 2


def test_regression_failure_and_nonpromotion_each_increment_once() -> None:
    state = ConvergenceState.initial(0.6016)
    state = state.update_after_iteration(0.5900)  # eligible outer regression
    state = state.update_after_iteration(None)  # completed iteration with no eligible result
    state = state.update_after_iteration(None)  # another valid, non-promoted iteration
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


def test_restart_round_trip_preserves_patience_window_anchor() -> None:
    original = ConvergenceState.initial(0.6000).update_after_iteration(0.6007)
    original = original.update_after_iteration(0.6014)
    persisted = json.loads(json.dumps(original.manifest(), sort_keys=True))
    resumed = ConvergenceState.from_manifest(persisted)
    assert resumed == original
    reset = resumed.update_after_iteration(0.6021)
    assert reset.non_material_streak == 0
    assert reset.patience_window_start_primary == 0.6021


def test_schema_v1_manifest_migrates_to_a_fresh_v2_patience_window() -> None:
    legacy = {
        "schema_version": 1,
        "best_primary": 0.6014,
        "non_material_streak": 2,
        "completed_iterations": 2,
        "required_completion_pending": False,
    }

    migrated = ConvergenceState.from_manifest(legacy)

    assert migrated.schema_version == 2
    assert migrated.best_primary == 0.6014
    assert migrated.patience_window_start_primary == 0.6014
    assert migrated.non_material_streak == 2
    assert migrated.manifest() == {
        "schema_version": 2,
        "best_primary": 0.6014,
        "patience_window_start_primary": 0.6014,
        "non_material_streak": 2,
        "completed_iterations": 2,
        "required_completion_pending": False,
    }


def test_restart_rejects_unknown_or_invalid_state() -> None:
    manifest = ConvergenceState.initial(0.6016).manifest()
    manifest["surprise"] = True
    with pytest.raises(ConvergenceStateError, match="unknown"):
        ConvergenceState.from_manifest(manifest)

    invalid = ConvergenceState.initial(0.6016).manifest()
    invalid["best_primary"] = float("nan")
    with pytest.raises(ConvergenceStateError, match="finite"):
        ConvergenceState.from_manifest(invalid)
