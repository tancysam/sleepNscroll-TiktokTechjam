"""Durable implementation of the frozen scientific-iteration convergence rule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Final, Self

from kuairand_agent.contract import BENCHMARK_CONTRACT

STATE_SCHEMA_VERSION: Final = 2
EPSILON: Final = Decimal(str(BENCHMARK_CONTRACT.convergence.epsilon))
PATIENCE: Final = BENCHMARK_CONTRACT.convergence.patience

# An iteration that produced no eligible outer primary is not an observation of the benchmark's
# comparison, which BENCHMARK_CONTRACT.convergence states as "eligible outer primary delta strictly
# greater than epsilon".  It used to increment non_material_streak anyway, so three consecutive
# REJECTIONS reported stop_reason "converged": runs 09 to 17 all stopped at iteration 3, and runs 16
# and 17 did so with no candidate ever reaching outer validation at all, having spent about 4% of a
# six-hour budget.  Rejections now accumulate separately and stop the campaign under their own
# truthful reason.  Epsilon, patience and the meaning of "converged" are unchanged and still frozen.
MAX_UNMEASURED_STREAK: Final = 6


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
    unmeasured_streak: int = 0
    completed_iterations: int = 0
    required_completion_pending: bool = False
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _primary(self.best_primary, "best_primary")
        if type(self.non_material_streak) is not int or self.non_material_streak < 0:
            raise ConvergenceStateError("non_material_streak must be a non-negative integer")
        if type(self.unmeasured_streak) is not int or self.unmeasured_streak < 0:
            raise ConvergenceStateError("unmeasured_streak must be a non-negative integer")
        if type(self.completed_iterations) is not int or self.completed_iterations < 0:
            raise ConvergenceStateError("completed_iterations must be a non-negative integer")
        if self.non_material_streak > self.completed_iterations:
            raise ConvergenceStateError("non_material_streak cannot exceed completed_iterations")
        if self.unmeasured_streak > self.completed_iterations:
            raise ConvergenceStateError("unmeasured_streak cannot exceed completed_iterations")
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
        """Whether the non-material streak reached the frozen patience.

        Strictly the frozen rule: only iterations that produced an eligible outer primary can
        contribute, so this stays a claim about measurements and never about rejections.
        """

        return self.non_material_streak >= PATIENCE

    @property
    def unmeasured_exhausted(self) -> bool:
        """Whether consecutive iterations produced no eligible outer primary for too long."""

        return self.unmeasured_streak >= MAX_UNMEASURED_STREAK

    @property
    def terminal(self) -> bool:
        """Whether research must end, for either reason."""

        return self.converged or self.unmeasured_exhausted

    @property
    def may_launch_scientific_iteration(self) -> bool:
        """Prevent additional research once a terminal condition has been observed."""

        return not self.terminal

    @property
    def should_stop(self) -> bool:
        """Stop fully after any already-reserved required completion has finished."""

        return self.terminal and not self.required_completion_pending

    def update_after_iteration(
        self,
        eligible_outer_primary: float | None,
        *,
        required_completion_pending: bool | None = None,
    ) -> Self:
        """Apply one completed scientific iteration against the *previous* best.

        ``None`` represents failure, policy rejection, or a non-promoted iteration. It produced no
        eligible outer primary, so there is no delta to compare against epsilon: it advances
        ``unmeasured_streak`` and leaves ``non_material_streak`` alone. A valid non-material outer
        improvement still advances ``best_primary`` while incrementing the non-material streak;
        subsequent iterations compare against that new best. Either streak reaching its limit ends
        research, but only the measured one is ever reported as convergence.
        """

        if self.terminal:
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
        material = observed is not None and (
            Decimal(str(observed)) - Decimal(str(self.best_primary)) > EPSILON
        )
        next_best = self.best_primary if observed is None else max(self.best_primary, observed)
        if observed is None:
            non_material_streak = self.non_material_streak
            unmeasured_streak = self.unmeasured_streak + 1
        else:
            non_material_streak = 0 if material else self.non_material_streak + 1
            unmeasured_streak = 0
        return type(self)(
            best_primary=next_best,
            non_material_streak=non_material_streak,
            unmeasured_streak=unmeasured_streak,
            completed_iterations=self.completed_iterations + 1,
            required_completion_pending=pending,
        )

    def reserve_required_completion(self) -> Self:
        """Mark confirmation/finalization already reserved before convergence."""

        if self.terminal:
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
            "unmeasured_streak": self.unmeasured_streak,
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
            "unmeasured_streak",
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
        unmeasured = raw["unmeasured_streak"]
        iterations = raw["completed_iterations"]
        pending = raw["required_completion_pending"]
        if type(schema_version) is not int:
            raise ConvergenceStateError("schema_version must be an integer")
        if type(streak) is not int:
            raise ConvergenceStateError("non_material_streak must be an integer")
        if type(unmeasured) is not int:
            raise ConvergenceStateError("unmeasured_streak must be an integer")
        if type(iterations) is not int:
            raise ConvergenceStateError("completed_iterations must be an integer")
        if type(pending) is not bool:
            raise ConvergenceStateError("required_completion_pending must be boolean")
        return cls(
            schema_version=schema_version,
            best_primary=_primary(raw["best_primary"], "best_primary"),
            non_material_streak=streak,
            unmeasured_streak=unmeasured,
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
