"""Durable implementation of the frozen scientific-iteration convergence rule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Final, Self

from kuairand_agent.contract import BENCHMARK_CONTRACT

STATE_SCHEMA_VERSION: Final = 1
EPSILON: Final = Decimal(str(BENCHMARK_CONTRACT.convergence.epsilon))
PATIENCE: Final = BENCHMARK_CONTRACT.convergence.patience


class ConvergenceStateError(ValueError):
    """Raised when convergence state is invalid or cannot be resumed safely."""


def _primary(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConvergenceStateError(f"{location} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ConvergenceStateError(f"{location} must be finite in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class ConvergenceState:
    """Restart-safe state updated exactly once per completed scientific iteration.

    Repair executions and reserved seed-confirmation/finalization work are not scientific
    iterations and therefore do not call :meth:`update_after_iteration`.
    """

    best_primary: float
    non_material_streak: int = 0
    completed_iterations: int = 0
    required_completion_pending: bool = False
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _primary(self.best_primary, "best_primary")
        if type(self.non_material_streak) is not int or self.non_material_streak < 0:
            raise ConvergenceStateError("non_material_streak must be a non-negative integer")
        if type(self.completed_iterations) is not int or self.completed_iterations < 0:
            raise ConvergenceStateError("completed_iterations must be a non-negative integer")
        if self.non_material_streak > self.completed_iterations:
            raise ConvergenceStateError("non_material_streak cannot exceed completed_iterations")
        if type(self.required_completion_pending) is not bool:
            raise ConvergenceStateError("required_completion_pending must be boolean")
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ConvergenceStateError(
                f"unsupported convergence schema_version {self.schema_version}"
            )

    @classmethod
    def initial(cls, best_primary: float) -> Self:
        """Start from the qualified eligible incumbent, normally the official FM."""

        return cls(best_primary=_primary(best_primary, "best_primary"))

    @property
    def converged(self) -> bool:
        """Whether the non-material streak reached the frozen patience."""

        return self.non_material_streak >= PATIENCE

    @property
    def may_launch_scientific_iteration(self) -> bool:
        """Prevent additional research once convergence has been observed."""

        return not self.converged

    @property
    def should_stop(self) -> bool:
        """Stop fully after any already-reserved required completion has finished."""

        return self.converged and not self.required_completion_pending

    def update_after_iteration(
        self,
        eligible_outer_primary: float | None,
        *,
        required_completion_pending: bool | None = None,
        produced_measurement: bool = True,
    ) -> Self:
        """Apply one completed scientific iteration against the *previous* best.

        ``None`` represents failure, policy rejection, or a non-promoted iteration. A valid
        non-material outer improvement still advances ``best_primary`` while incrementing the
        streak; subsequent iterations compare against that new best.

        ``produced_measurement=False`` marks an iteration whose code raised before any evaluation.
        The organizers' convergence rule is three consecutive rounds whose *validation primary*
        fails to improve by more than epsilon; a branch that crashed produced no validation primary
        at all, so it is not a round that failed to improve -- it is a round that did not happen.
        Counting it spent a scientific slot on an engineering defect, which is how a campaign could
        converge having tested one hypothesis instead of three. The iteration is still recorded in
        ``completed_iterations``, so the hard iteration cap, the wall clock and the repeated
        pre-admission failure rule all continue to bound a campaign that only crashes.
        """

        if self.converged:
            raise ConvergenceStateError("cannot launch a scientific iteration after convergence")
        if required_completion_pending is None:
            pending = self.required_completion_pending
        elif type(required_completion_pending) is bool:
            pending = required_completion_pending
        else:
            raise ConvergenceStateError("required_completion_pending must be boolean")

        observed = (
            None
            if eligible_outer_primary is None
            else _primary(eligible_outer_primary, "eligible_outer_primary")
        )
        if type(produced_measurement) is not bool:
            raise ConvergenceStateError("produced_measurement must be boolean")
        material = observed is not None and (
            Decimal(str(observed)) - Decimal(str(self.best_primary)) > EPSILON
        )
        next_best = self.best_primary if observed is None else max(self.best_primary, observed)
        if material:
            streak = 0
        elif produced_measurement:
            streak = self.non_material_streak + 1
        else:
            # No validation primary was produced, so this round cannot be one of the three the
            # organizers' rule counts. The iteration still advances completed_iterations.
            streak = self.non_material_streak
        return type(self)(
            best_primary=next_best,
            non_material_streak=streak,
            completed_iterations=self.completed_iterations + 1,
            required_completion_pending=pending,
        )

    def reserve_required_completion(self) -> Self:
        """Mark confirmation/finalization already reserved before convergence."""

        if self.converged:
            raise ConvergenceStateError("cannot reserve new work after convergence")
        if self.required_completion_pending:
            return self
        return replace(self, required_completion_pending=True)

    def finish_required_completion(self) -> Self:
        """Clear reserved work without consuming another scientific iteration."""

        if not self.required_completion_pending:
            raise ConvergenceStateError("no required completion is pending")
        return replace(self, required_completion_pending=False)

    def manifest(self) -> dict[str, object]:
        """Return the complete durable state for JSON persistence."""

        return {
            "schema_version": self.schema_version,
            "best_primary": self.best_primary,
            "non_material_streak": self.non_material_streak,
            "completed_iterations": self.completed_iterations,
            "required_completion_pending": self.required_completion_pending,
        }

    @classmethod
    def from_manifest(cls, raw: Mapping[str, object]) -> Self:
        """Strictly restore state without silently defaulting missing or unknown fields."""

        expected = {
            "schema_version",
            "best_primary",
            "non_material_streak",
            "completed_iterations",
            "required_completion_pending",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown:
            raise ConvergenceStateError(
                f"unknown convergence state field(s): {', '.join(sorted(unknown))}"
            )
        if missing:
            raise ConvergenceStateError(
                f"missing convergence state field(s): {', '.join(sorted(missing))}"
            )
        schema_version = raw["schema_version"]
        streak = raw["non_material_streak"]
        iterations = raw["completed_iterations"]
        pending = raw["required_completion_pending"]
        if type(schema_version) is not int:
            raise ConvergenceStateError("schema_version must be an integer")
        if type(streak) is not int:
            raise ConvergenceStateError("non_material_streak must be an integer")
        if type(iterations) is not int:
            raise ConvergenceStateError("completed_iterations must be an integer")
        if type(pending) is not bool:
            raise ConvergenceStateError("required_completion_pending must be boolean")
        return cls(
            schema_version=schema_version,
            best_primary=_primary(raw["best_primary"], "best_primary"),
            non_material_streak=streak,
            completed_iterations=iterations,
            required_completion_pending=pending,
        )


def update_convergence(
    state: ConvergenceState,
    eligible_outer_primary: float | None,
    *,
    required_completion_pending: bool | None = None,
) -> ConvergenceState:
    """Functional form of :meth:`ConvergenceState.update_after_iteration`."""

    return state.update_after_iteration(
        eligible_outer_primary,
        required_completion_pending=required_completion_pending,
    )
