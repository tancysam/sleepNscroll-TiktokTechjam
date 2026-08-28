"""Pure campaign domain types.

The campaign has two deliberately different units of work:

``scientific iteration``
    One closed hypothesis in the autonomous research loop.  A repaired hypothesis still
    closes exactly one scientific iteration.

``training launch``
    One started full train/evaluate execution.  Failed, timed-out, out-of-memory, repaired,
    confirmation, and final replay executions are all training launches.

The distinct value objects below make it difficult for persistence and controller code to
silently use one counter in place of the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, Self

DOMAIN_SCHEMA_VERSION: Final = 1


class CampaignModelError(ValueError):
    """Raised when a campaign value or lifecycle transition is invalid."""


def _non_negative_integer(value: object, location: str) -> int:
    if type(value) is not int or value < 0:
        raise CampaignModelError(f"{location} must be a non-negative integer")
    return value


def _identifier(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise CampaignModelError(f"{location} must be a non-empty string without NUL bytes")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ScientificIterationCount:
    """Number of completed scientific hypotheses, not executions."""

    value: int = 0

    def __post_init__(self) -> None:
        _non_negative_integer(self.value, "scientific iteration count")

    def increment(self) -> Self:
        return type(self)(self.value + 1)


@dataclass(frozen=True, slots=True, order=True)
class TrainingLaunchCount:
    """Number of full train/evaluate processes that actually started."""

    value: int = 0

    def __post_init__(self) -> None:
        _non_negative_integer(self.value, "training launch count")

    def increment(self) -> Self:
        return type(self)(self.value + 1)


class ExperimentState(StrEnum):
    """Durable states in the frozen scientific-iteration state machine."""

    PROPOSED = "PROPOSED"
    POLICY_REJECTED = "POLICY_REJECTED"
    MATERIALIZED = "MATERIALIZED"
    STATIC_REJECTED = "STATIC_REJECTED"
    SMOKE_REJECTED = "SMOKE_REJECTED"
    REPAIRING = "REPAIRING"
    INNER_RUNNING = "INNER_RUNNING"
    FAILED = "FAILED"
    INNER_SCORED = "INNER_SCORED"
    INNER_REJECTED = "INNER_REJECTED"
    OUTER_ELIGIBLE = "OUTER_ELIGIBLE"
    OUTER_SCORED = "OUTER_SCORED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    REFLECTED = "REFLECTED"
    CLOSED = "CLOSED"


_EXPERIMENT_TRANSITIONS: Final[dict[ExperimentState, frozenset[ExperimentState]]] = {
    ExperimentState.PROPOSED: frozenset(
        {ExperimentState.POLICY_REJECTED, ExperimentState.MATERIALIZED}
    ),
    ExperimentState.POLICY_REJECTED: frozenset(),
    ExperimentState.MATERIALIZED: frozenset(
        {
            ExperimentState.STATIC_REJECTED,
            ExperimentState.SMOKE_REJECTED,
            ExperimentState.INNER_RUNNING,
        }
    ),
    ExperimentState.STATIC_REJECTED: frozenset({ExperimentState.REPAIRING, ExperimentState.CLOSED}),
    ExperimentState.SMOKE_REJECTED: frozenset({ExperimentState.REPAIRING, ExperimentState.CLOSED}),
    ExperimentState.REPAIRING: frozenset({ExperimentState.MATERIALIZED, ExperimentState.CLOSED}),
    ExperimentState.INNER_RUNNING: frozenset(
        {ExperimentState.FAILED, ExperimentState.INNER_SCORED}
    ),
    ExperimentState.FAILED: frozenset({ExperimentState.REPAIRING, ExperimentState.CLOSED}),
    ExperimentState.INNER_SCORED: frozenset(
        {ExperimentState.INNER_REJECTED, ExperimentState.OUTER_ELIGIBLE}
    ),
    ExperimentState.INNER_REJECTED: frozenset(),
    ExperimentState.OUTER_ELIGIBLE: frozenset({ExperimentState.OUTER_SCORED}),
    ExperimentState.OUTER_SCORED: frozenset({ExperimentState.PROMOTED, ExperimentState.REJECTED}),
    ExperimentState.PROMOTED: frozenset({ExperimentState.REFLECTED}),
    ExperimentState.REJECTED: frozenset({ExperimentState.REFLECTED}),
    ExperimentState.REFLECTED: frozenset(),
    ExperimentState.CLOSED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ExperimentLifecycle:
    """Restart-safe lifecycle projection for one scientific experiment.

    The append-only transition log remains authoritative.  ``revision`` is the number of
    accepted transitions and is suitable for optimistic persistence checks.
    """

    experiment_id: str
    scientific_iteration: int
    state: ExperimentState = ExperimentState.PROPOSED
    revision: int = 0
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        if type(self.scientific_iteration) is not int or self.scientific_iteration <= 0:
            raise CampaignModelError("scientific_iteration must be a positive integer")
        _non_negative_integer(self.revision, "revision")
        if not isinstance(self.state, ExperimentState):
            raise CampaignModelError("state must be an ExperimentState")
        if self.schema_version != DOMAIN_SCHEMA_VERSION:
            raise CampaignModelError(f"unsupported domain schema_version {self.schema_version}")

    @property
    def terminal(self) -> bool:
        return not _EXPERIMENT_TRANSITIONS[self.state]

    @property
    def allowed_next_states(self) -> frozenset[ExperimentState]:
        return _EXPERIMENT_TRANSITIONS[self.state]

    def transition(self, next_state: ExperimentState) -> Self:
        if not isinstance(next_state, ExperimentState):
            raise CampaignModelError("next_state must be an ExperimentState")
        if next_state not in self.allowed_next_states:
            raise CampaignModelError(
                f"forbidden experiment transition {self.state.value} -> {next_state.value}"
            )
        return replace(self, state=next_state, revision=self.revision + 1)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "scientific_iteration": self.scientific_iteration,
            "state": self.state.value,
            "revision": self.revision,
        }

    @classmethod
    def from_manifest(cls, raw: Mapping[str, object]) -> Self:
        expected = {
            "schema_version",
            "experiment_id",
            "scientific_iteration",
            "state",
            "revision",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown:
            raise CampaignModelError(
                f"unknown experiment lifecycle field(s): {', '.join(sorted(unknown))}"
            )
        if missing:
            raise CampaignModelError(
                f"missing experiment lifecycle field(s): {', '.join(sorted(missing))}"
            )
        schema_version = raw["schema_version"]
        scientific_iteration = raw["scientific_iteration"]
        revision = raw["revision"]
        state = raw["state"]
        if type(schema_version) is not int:
            raise CampaignModelError("schema_version must be an integer")
        if type(scientific_iteration) is not int:
            raise CampaignModelError("scientific_iteration must be an integer")
        if type(revision) is not int:
            raise CampaignModelError("revision must be an integer")
        if type(state) is not str:
            raise CampaignModelError("state must be a string")
        try:
            restored_state = ExperimentState(state)
        except ValueError as exc:
            raise CampaignModelError(f"unknown experiment state {state!r}") from exc
        return cls(
            schema_version=schema_version,
            experiment_id=_identifier(raw["experiment_id"], "experiment_id"),
            scientific_iteration=scientific_iteration,
            state=restored_state,
            revision=revision,
        )


class CampaignState(StrEnum):
    """Controller lifecycle; paused campaigns may resume, terminal campaigns may not."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINALIZATION_REQUIRED = "FINALIZATION_REQUIRED"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    INCOMPLETE = "INCOMPLETE"


_CAMPAIGN_TRANSITIONS: Final[dict[CampaignState, frozenset[CampaignState]]] = {
    CampaignState.CREATED: frozenset(
        {CampaignState.RUNNING, CampaignState.FINALIZATION_REQUIRED, CampaignState.INCOMPLETE}
    ),
    CampaignState.RUNNING: frozenset(
        {CampaignState.PAUSED, CampaignState.FINALIZATION_REQUIRED, CampaignState.INCOMPLETE}
    ),
    CampaignState.PAUSED: frozenset(
        {CampaignState.RUNNING, CampaignState.FINALIZATION_REQUIRED, CampaignState.INCOMPLETE}
    ),
    CampaignState.FINALIZATION_REQUIRED: frozenset(
        {CampaignState.FINALIZING, CampaignState.INCOMPLETE}
    ),
    CampaignState.FINALIZING: frozenset({CampaignState.COMPLETED, CampaignState.INCOMPLETE}),
    CampaignState.COMPLETED: frozenset(),
    CampaignState.INCOMPLETE: frozenset(),
}


@dataclass(frozen=True, slots=True)
class CampaignLifecycle:
    """Small pure campaign-state projection used by the durable store."""

    campaign_id: str
    state: CampaignState = CampaignState.CREATED
    revision: int = 0

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        _non_negative_integer(self.revision, "revision")
        if not isinstance(self.state, CampaignState):
            raise CampaignModelError("state must be a CampaignState")

    @property
    def terminal(self) -> bool:
        return not _CAMPAIGN_TRANSITIONS[self.state]

    def transition(self, next_state: CampaignState) -> Self:
        if not isinstance(next_state, CampaignState):
            raise CampaignModelError("next_state must be a CampaignState")
        if next_state not in _CAMPAIGN_TRANSITIONS[self.state]:
            raise CampaignModelError(
                f"forbidden campaign transition {self.state.value} -> {next_state.value}"
            )
        return replace(self, state=next_state, revision=self.revision + 1)
