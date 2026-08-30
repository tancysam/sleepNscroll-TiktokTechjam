"""Durable implementation of the frozen scientific-iteration convergence rule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Final, Self

from kuairand_agent.contract import BENCHMARK_CONTRACT

STATE_SCHEMA_VERSION: Final = 2
LEGACY_STATE_SCHEMA_VERSION: Final = 1
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
class CompletedScientificIteration:
    """One scientifically interpretable observation presented to convergence.

    ``None`` means that the completed iteration was scientifically valid but produced no eligible
    outer-primary observation. Execution, policy, serialization, and budget failures are not
    completed scientific iterations and must not be represented by this type.
    """

    eligible_outer_primary: float | None

    def __post_init__(self) -> None:
        if self.eligible_outer_primary is not None:
            object.__setattr__(
                self,
                "eligible_outer_primary",
                _primary(self.eligible_outer_primary, "eligible_outer_primary"),
            )


@dataclass(frozen=True, slots=True)
class ConvergenceState:
    """Restart-safe state updated exactly once per completed scientific iteration.

    Repair executions and reserved seed-confirmation/finalization work are not scientific
    iterations and therefore do not call :meth:`observe`.
    """

    best_primary: float
    patience_window_start_primary: float
    non_material_streak: int = 0
    completed_iterations: int = 0
    required_completion_pending: bool = False
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _primary(self.best_primary, "best_primary")
        _primary(self.patience_window_start_primary, "patience_window_start_primary")
        if Decimal(str(self.patience_window_start_primary)) > Decimal(str(self.best_primary)):
            raise ConvergenceStateError(
                "patience_window_start_primary cannot exceed best_primary"
            )
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

        primary = _primary(best_primary, "best_primary")
        return cls(best_primary=primary, patience_window_start_primary=primary)

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

    def observe(
        self,
        observation: CompletedScientificIteration,
        *,
        required_completion_pending: bool | None = None,
    ) -> Self:
        """Apply one completed scientific iteration against the *previous* best.

        A valid non-material outer improvement still advances ``best_primary`` while incrementing
        the streak; subsequent iterations compare against that new best.
        """

        if not isinstance(observation, CompletedScientificIteration):
            raise ConvergenceStateError("observation must be CompletedScientificIteration")
        if self.converged:
            raise ConvergenceStateError("cannot launch a scientific iteration after convergence")
        if required_completion_pending is None:
            pending = self.required_completion_pending
        elif type(required_completion_pending) is bool:
            pending = required_completion_pending
        else:
            raise ConvergenceStateError("required_completion_pending must be boolean")

        observed = observation.eligible_outer_primary
        next_best = self.best_primary if observed is None else max(self.best_primary, observed)
        material = (
            Decimal(str(next_best)) - Decimal(str(self.patience_window_start_primary)) > EPSILON
        )
        return type(self)(
            best_primary=next_best,
            patience_window_start_primary=(
                next_best if material else self.patience_window_start_primary
            ),
            non_material_streak=0 if material else self.non_material_streak + 1,
            completed_iterations=self.completed_iterations + 1,
            required_completion_pending=pending,
        )

    def update_after_iteration(
        self,
        eligible_outer_primary: float | None,
        *,
        required_completion_pending: bool | None = None,
    ) -> Self:
        """Compatibility adapter for callers that still pass the legacy scalar observation."""

        return self.observe(
            CompletedScientificIteration(eligible_outer_primary),
            required_completion_pending=required_completion_pending,
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
            "patience_window_start_primary": self.patience_window_start_primary,
            "non_material_streak": self.non_material_streak,
            "completed_iterations": self.completed_iterations,
            "required_completion_pending": self.required_completion_pending,
        }

    @classmethod
    def from_manifest(cls, raw: Mapping[str, object]) -> Self:
        """Strictly restore state without silently defaulting missing or unknown fields."""

        legacy_expected = {
            "schema_version",
            "best_primary",
            "non_material_streak",
            "completed_iterations",
            "required_completion_pending",
        }
        current_expected = legacy_expected | {"patience_window_start_primary"}
        schema_version = raw.get("schema_version")
        if type(schema_version) is not int:
            raise ConvergenceStateError("schema_version must be an integer")
        if schema_version == LEGACY_STATE_SCHEMA_VERSION:
            expected = legacy_expected
        elif schema_version == STATE_SCHEMA_VERSION:
            expected = current_expected
        else:
            raise ConvergenceStateError(f"unsupported convergence schema_version {schema_version}")
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
        streak = raw["non_material_streak"]
        iterations = raw["completed_iterations"]
        pending = raw["required_completion_pending"]
        if type(streak) is not int:
            raise ConvergenceStateError("non_material_streak must be an integer")
        if type(iterations) is not int:
            raise ConvergenceStateError("completed_iterations must be an integer")
        if type(pending) is not bool:
            raise ConvergenceStateError("required_completion_pending must be boolean")
        best_primary = _primary(raw["best_primary"], "best_primary")
        window_start = (
            best_primary
            if schema_version == LEGACY_STATE_SCHEMA_VERSION
            else _primary(raw["patience_window_start_primary"], "patience_window_start_primary")
        )
        return cls(
            best_primary=best_primary,
            patience_window_start_primary=window_start,
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
