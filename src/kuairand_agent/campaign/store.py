"""Durable SQLite persistence for one autonomous research campaign.

The controller is intentionally the only writer.  Evidence is append-only, campaign mutations
use optimistic revisions, and every launch or public-validation reservation is charged before the
external action can start.  Large artifacts live in a content-addressed filesystem store; this
module commits their immutable metadata and owner links atomically with state transitions.

Two databases are used:

* :class:`CampaignStore` is private to one campaign/run directory.
* :class:`OuterQueryLedger` is project-wide and prevents a new run directory from erasing prior
  public-validation queries.

The project ledger is reserved first.  SQLite cannot make two independent WAL databases one
atomic commit, so a crash between the two reservations deliberately consumes the project slot.
That conservative failure mode protects the six-query contract.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, Self, cast

SCHEMA_VERSION: Final = 1
APPLICATION_ID: Final = 0x4B524143  # ``KRAC``
DEFAULT_MAX_LAUNCHES: Final = 50
DEFAULT_OUTER_QUERY_LIMIT: Final = 6
QUALIFICATION_LAUNCH_COUNT: Final = 6

type LaunchState = Literal["RESERVED", "STARTED", "FINISHED", "NOT_STARTED", "START_UNCERTAIN"]
type QueryState = Literal["RESERVED", "COMPLETED", "START_UNCERTAIN"]
type EntityType = Literal["campaign", "experiment", "execution"]

_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_NONCE_RE: Final = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
UNFINISHED_EXECUTION_STATES: Final = frozenset({"STARTING", "RUNNING"})
_QUALIFICATION_PATTERN: Final = (
    (1, "official_fm_training", 0),
    (2, "official_fm_training", 1),
    (3, "official_fm_training", 2),
    (4, "official_fm_training", 3),
    (5, "official_fm_training", 4),
    (6, "clean_source_retrain", 0),
)
_LAUNCH_TRANSITIONS: Final[Mapping[str, frozenset[str]]] = {
    "RESERVED": frozenset({"STARTED", "NOT_STARTED", "START_UNCERTAIN"}),
    "STARTED": frozenset({"FINISHED", "START_UNCERTAIN"}),
    "START_UNCERTAIN": frozenset({"FINISHED"}),
    "FINISHED": frozenset(),
    "NOT_STARTED": frozenset(),
}


class StoreError(RuntimeError):
    """Base error for campaign persistence failures."""


class CampaignExistsError(StoreError):
    """Raised when creation would overwrite an existing campaign database."""


class CampaignNotFoundError(StoreError):
    """Raised when a requested database or campaign identity does not exist."""


class StoreVersionError(StoreError):
    """Raised when schema identity or SQLite safety settings do not match."""


class StoreInvariantError(StoreError):
    """Raised when a durable-state invariant would be violated."""


class RevisionConflictError(StoreError):
    """Raised when a caller attempts to mutate a stale campaign snapshot."""


class LaunchLimitError(StoreError):
    """Raised before a launch reservation would exceed its hard ceiling."""


class OuterQueryLimitError(StoreError):
    """Raised before a public-validation reservation would exceed its ceiling."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Immutable metadata for an already validated content-addressed artifact."""

    digest: str
    kind: str
    relative_path: str
    size_bytes: int
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        _digest(self.digest, "artifact digest")
        _text(self.kind, "artifact kind")
        _relative_path(self.relative_path)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise StoreInvariantError("artifact size_bytes must be a non-negative integer")
        _json_object(self.metadata, "artifact metadata")


@dataclass(frozen=True, slots=True)
class LaunchRecord:
    """Latest immutable event for one logical launch."""

    launch_id: str
    launch_number: int
    reservation_key: str
    category: str
    original_category: str
    purpose: str
    state: str
    charged: bool
    event_seq: int
    experiment_id: str | None
    scientific_iteration: int | None
    seed: int | None
    start_receipt_digest: str | None


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Immutable restart projection of one generated research experiment."""

    experiment_id: str
    iteration_number: int
    parent_experiment_id: str | None
    hypothesis: str
    mechanism: str
    method_attribution: str
    status: str
    metadata: Mapping[str, object]
    created_at: str


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """Immutable restart projection of one complete research-model exchange."""

    proposal_id: str
    experiment_id: str
    request_digest: str
    response_digest: str
    provider: str
    metadata: Mapping[str, object]
    created_at: str


@dataclass(frozen=True, slots=True)
class SourceSnapshotRecord:
    """Immutable restart projection of one generated-source lineage snapshot."""

    snapshot_id: str
    experiment_id: str
    source_digest: str
    parent_source_digest: str | None
    diff_digest: str | None
    metadata: Mapping[str, object]
    created_at: str


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Immutable, restart-safe projection of one trusted execution identity.

    ``STARTING`` means the runner has not durably supplied a process receipt. ``RUNNING`` always
    carries a strictly decoded receipt.  The process mapping and metadata are recursively frozen.
    """

    execution_id: str
    experiment_id: str | None
    launch_id: str | None
    launch_number: int | None
    launch_category: str | None
    original_launch_category: str | None
    kind: str
    tier: str
    seed: int | None
    command: tuple[str, ...]
    status: str
    nonce: str | None
    source_digest: str | None
    config_digest: str | None
    capability_digest: str | None
    environment_digest: str | None
    data_digest: str | None
    checkpoint_digest: str | None
    process_record_digest: str | None
    process_record: Mapping[str, object] | None
    process_id: int | None
    process_create_time: float | None
    process_group_id: int | None
    process_command_digest: str | None
    process_environment_digest: str | None
    result_digest: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class IncumbentRecord:
    """One immutable incumbent/fallback history entry."""

    incumbent_id: str
    experiment_id: str | None
    eligibility: str
    source_digest: str
    checkpoint_digest: str
    artifact_closure_digest: str
    replay_verified: bool
    outer_primary_mean: float | None
    is_fallback: bool
    revision: int


@dataclass(frozen=True, slots=True)
class ValidationQueryRecord:
    """Latest event for one public-validation query."""

    query_id: str
    candidate_fingerprint: str
    state: str
    event_seq: int
    result_digest: str | None
    metrics: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """Immutable trusted-score projection used to make durable retries exact."""

    metric_id: str
    split_role: str
    seed: int | None
    gauc: float
    ndcg_at_5: float
    primary: float
    scorer_digest: str
    prediction_digest: str
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """Immutable failure projection used to distinguish retries from conflicts."""

    failure_id: str
    category: str
    fingerprint: str
    retry_ordinal: int
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ReallocationRecord:
    """Immutable category-budget transfer used to rebuild launch admission state."""

    reallocation_id: str
    from_category: str
    to_category: str
    launch_count: int
    reason: str
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CampaignSnapshot:
    """Stable read projection created inside one SQLite read transaction."""

    campaign_id: str
    revision: int
    status: str
    phase: str
    max_launches: int
    launches_used: int
    outer_query_limit: int
    outer_queries_used: int
    qualification_digest: str | None
    convergence_state: Mapping[str, object]
    incumbent: IncumbentRecord | None

    @property
    def launches_remaining(self) -> int:
        return self.max_launches - self.launches_used

    @property
    def outer_queries_remaining(self) -> int:
        return self.outer_query_limit - self.outer_queries_used


@dataclass(frozen=True, slots=True)
class CampaignIdentityRecord:
    """Immutable create-time identities that every resume request must match exactly."""

    campaign_id: str
    config_digest: str
    benchmark_digest: str
    starter_manifest_digest: str
    dataset_manifest_digest: str
    source_digest: str | None
    environment_digest: str
    hard_deadline_utc: str
    max_launches: int
    outer_query_limit: int
    created_at: str


@dataclass(frozen=True, slots=True)
class StoreHealth:
    """Read-only evidence that the required SQLite safety settings are active."""

    journal_mode: str
    foreign_keys: bool
    synchronous: int
    user_version: int
    quick_check: str
    schema_digest: str
    catalog_digest: str


@dataclass(frozen=True, slots=True)
class OuterLedgerSnapshot:
    """Stable project-ledger summary for one benchmark identity."""

    revision: int
    max_queries: int
    queries_used: int

    @property
    def queries_remaining(self) -> int:
        return self.max_queries - self.queries_used


@dataclass(frozen=True, slots=True)
class OuterQueryProjectionRecord:
    """One project-wide reservation with its exact append revision and latest state."""

    query_id: str
    campaign_id: str
    benchmark_digest: str
    dataset_digest: str
    scorer_digest: str
    candidate_fingerprint: str
    reservation_revision: int
    state: str
    event_seq: int
    result_digest: str | None
    reservation_metadata: Mapping[str, object]
    latest_metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class OuterQueryLedgerProjection:
    """Exact, immutable public view of the append-only project query history."""

    revision: int
    max_queries: int
    queries: tuple[OuterQueryProjectionRecord, ...]

    @property
    def candidate_fingerprints(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.candidate_fingerprint for item in self.queries))


def _append_only_statements(tables: Sequence[str]) -> tuple[str, ...]:
    statements: list[str] = []
    for table in tables:
        statements.extend(
            (
                f"""CREATE TRIGGER {table}_reject_update
                BEFORE UPDATE ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END""",
                f"""CREATE TRIGGER {table}_reject_delete
                BEFORE DELETE ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END""",
            )
        )
    return tuple(statements)


_CAMPAIGN_SCHEMA_STATEMENTS: Final = (
    """CREATE TABLE schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE campaigns (
        campaign_id TEXT PRIMARY KEY CHECK(length(campaign_id) > 0),
        revision INTEGER NOT NULL CHECK(revision >= 0),
        status TEXT NOT NULL CHECK(length(status) > 0),
        phase TEXT NOT NULL CHECK(length(phase) > 0),
        config_digest TEXT NOT NULL CHECK(length(config_digest) = 64),
        benchmark_digest TEXT NOT NULL CHECK(length(benchmark_digest) = 64),
        starter_digest TEXT NOT NULL CHECK(length(starter_digest) = 64),
        dataset_digest TEXT NOT NULL CHECK(length(dataset_digest) = 64),
        environment_digest TEXT NOT NULL CHECK(length(environment_digest) = 64),
        source_digest TEXT CHECK(source_digest IS NULL OR length(source_digest) = 64),
        hard_deadline_utc TEXT NOT NULL CHECK(length(hard_deadline_utc) > 0),
        max_launches INTEGER NOT NULL CHECK(max_launches >= 6 AND max_launches <= 50),
        outer_query_limit INTEGER NOT NULL CHECK(outer_query_limit >= 0 AND outer_query_limit <= 6),
        qualification_digest TEXT CHECK(
            qualification_digest IS NULL OR length(qualification_digest) = 64
        ),
        convergence_json TEXT NOT NULL CHECK(json_valid(convergence_json)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE experiments (
        experiment_id TEXT PRIMARY KEY CHECK(length(experiment_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        iteration_number INTEGER NOT NULL CHECK(iteration_number >= 0),
        parent_experiment_id TEXT REFERENCES experiments(experiment_id),
        hypothesis TEXT NOT NULL,
        mechanism TEXT NOT NULL,
        method_attribution TEXT NOT NULL,
        status TEXT NOT NULL CHECK(length(status) > 0),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        UNIQUE(campaign_id, iteration_number)
    ) STRICT""",
    """CREATE TABLE executions (
        execution_id TEXT PRIMARY KEY CHECK(length(execution_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_id TEXT REFERENCES experiments(experiment_id),
        launch_id TEXT,
        kind TEXT NOT NULL CHECK(length(kind) > 0),
        tier TEXT NOT NULL CHECK(length(tier) > 0),
        seed INTEGER CHECK(seed IS NULL OR seed >= 0),
        command_json TEXT NOT NULL CHECK(json_valid(command_json)),
        status TEXT NOT NULL CHECK(length(status) > 0),
        nonce TEXT CHECK(nonce IS NULL OR (length(nonce) >= 16 AND length(nonce) <= 128)),
        source_digest TEXT CHECK(source_digest IS NULL OR length(source_digest) = 64),
        config_digest TEXT CHECK(config_digest IS NULL OR length(config_digest) = 64),
        capability_digest TEXT CHECK(capability_digest IS NULL OR length(capability_digest) = 64),
        environment_digest TEXT CHECK(
            environment_digest IS NULL OR length(environment_digest) = 64
        ),
        data_digest TEXT CHECK(data_digest IS NULL OR length(data_digest) = 64),
        checkpoint_digest TEXT CHECK(checkpoint_digest IS NULL OR length(checkpoint_digest) = 64),
        process_record_digest TEXT CHECK(
            process_record_digest IS NULL OR length(process_record_digest) = 64
        ),
        process_record_json TEXT CHECK(
            process_record_json IS NULL OR json_valid(process_record_json)
        ),
        result_digest TEXT CHECK(result_digest IS NULL OR length(result_digest) = 64),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        CHECK((process_record_digest IS NULL) = (process_record_json IS NULL)),
        CHECK(status != 'RUNNING' OR process_record_digest IS NOT NULL)
    ) STRICT""",
    """CREATE TABLE proposals (
        proposal_id TEXT PRIMARY KEY CHECK(length(proposal_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
        request_digest TEXT NOT NULL CHECK(length(request_digest) = 64),
        response_digest TEXT NOT NULL CHECK(length(response_digest) = 64),
        provider TEXT NOT NULL CHECK(length(provider) > 0),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE source_snapshots (
        snapshot_id TEXT PRIMARY KEY CHECK(length(snapshot_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
        source_digest TEXT NOT NULL CHECK(length(source_digest) = 64),
        parent_source_digest TEXT CHECK(
            parent_source_digest IS NULL OR length(parent_source_digest) = 64
        ),
        diff_digest TEXT CHECK(diff_digest IS NULL OR length(diff_digest) = 64),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE metrics (
        metric_id TEXT PRIMARY KEY CHECK(length(metric_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_id TEXT REFERENCES experiments(experiment_id),
        execution_id TEXT REFERENCES executions(execution_id),
        split_role TEXT NOT NULL CHECK(length(split_role) > 0),
        seed INTEGER CHECK(seed IS NULL OR seed >= 0),
        gauc REAL NOT NULL CHECK(gauc >= 0.0 AND gauc <= 1.0),
        ndcg_at_5 REAL NOT NULL CHECK(ndcg_at_5 >= 0.0 AND ndcg_at_5 <= 1.0),
        primary_value REAL NOT NULL CHECK(primary_value >= 0.0 AND primary_value <= 1.0),
        primary_delta REAL,
        scorer_digest TEXT NOT NULL CHECK(length(scorer_digest) = 64),
        prediction_digest TEXT NOT NULL CHECK(length(prediction_digest) = 64),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE artifacts (
        digest TEXT PRIMARY KEY CHECK(length(digest) = 64),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        kind TEXT NOT NULL CHECK(length(kind) > 0),
        relative_path TEXT NOT NULL CHECK(length(relative_path) > 0),
        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        UNIQUE(campaign_id, relative_path)
    ) STRICT""",
    """CREATE TABLE failures (
        failure_id TEXT PRIMARY KEY CHECK(length(failure_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_id TEXT REFERENCES experiments(experiment_id),
        execution_id TEXT REFERENCES executions(execution_id),
        category TEXT NOT NULL CHECK(length(category) > 0),
        fingerprint TEXT NOT NULL CHECK(length(fingerprint) > 0),
        traceback_digest TEXT CHECK(traceback_digest IS NULL OR length(traceback_digest) = 64),
        repair_action TEXT,
        recovery_outcome TEXT,
        retry_ordinal INTEGER NOT NULL CHECK(retry_ordinal >= 0),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE interventions (
        intervention_id TEXT PRIMARY KEY CHECK(length(intervention_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        category TEXT NOT NULL CHECK(length(category) > 0),
        description TEXT NOT NULL,
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE transitions (
        transition_id INTEGER PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        revision_before INTEGER NOT NULL CHECK(revision_before >= 0),
        revision_after INTEGER NOT NULL CHECK(revision_after = revision_before + 1),
        entity_type TEXT NOT NULL CHECK(length(entity_type) > 0),
        entity_id TEXT NOT NULL CHECK(length(entity_id) > 0),
        from_state TEXT,
        to_state TEXT NOT NULL CHECK(length(to_state) > 0),
        reason TEXT NOT NULL,
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        UNIQUE(campaign_id, revision_after)
    ) STRICT""",
    """CREATE TABLE launches (
        launch_event_id INTEGER PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        launch_id TEXT NOT NULL CHECK(length(launch_id) > 0),
        event_seq INTEGER NOT NULL CHECK(event_seq >= 1),
        launch_number INTEGER NOT NULL CHECK(launch_number >= 1),
        reservation_key TEXT NOT NULL CHECK(length(reservation_key) > 0),
        category TEXT NOT NULL CHECK(length(category) > 0),
        original_category TEXT NOT NULL CHECK(length(original_category) > 0),
        purpose TEXT NOT NULL CHECK(length(purpose) > 0),
        experiment_id TEXT REFERENCES experiments(experiment_id),
        scientific_iteration INTEGER CHECK(
            scientific_iteration IS NULL OR scientific_iteration >= 0
        ),
        seed INTEGER CHECK(seed IS NULL OR seed >= 0),
        state TEXT NOT NULL CHECK(
            state IN ('RESERVED','STARTED','FINISHED','NOT_STARTED','START_UNCERTAIN')
        ),
        charged INTEGER NOT NULL CHECK(charged IN (0, 1)),
        start_receipt_digest TEXT CHECK(
            start_receipt_digest IS NULL OR length(start_receipt_digest) = 64
        ),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        UNIQUE(campaign_id, launch_id, event_seq)
    ) STRICT""",
    """CREATE UNIQUE INDEX launches_reservation_key
        ON launches(campaign_id, reservation_key) WHERE event_seq = 1""",
    """CREATE UNIQUE INDEX launches_number
        ON launches(campaign_id, launch_number) WHERE event_seq = 1""",
    """CREATE TABLE reallocations (
        reallocation_id TEXT PRIMARY KEY CHECK(length(reallocation_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        from_category TEXT NOT NULL CHECK(length(from_category) > 0),
        to_category TEXT NOT NULL CHECK(length(to_category) > 0),
        launch_count INTEGER NOT NULL CHECK(launch_count >= 1),
        reason TEXT NOT NULL,
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE artifact_links (
        artifact_link_id INTEGER PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        artifact_digest TEXT NOT NULL REFERENCES artifacts(digest),
        owner_type TEXT NOT NULL CHECK(length(owner_type) > 0),
        owner_id TEXT NOT NULL CHECK(length(owner_id) > 0),
        role TEXT NOT NULL CHECK(length(role) > 0),
        created_at TEXT NOT NULL,
        UNIQUE(campaign_id, owner_type, owner_id, role)
    ) STRICT""",
    """CREATE TABLE incumbent_history (
        history_id INTEGER PRIMARY KEY,
        incumbent_id TEXT NOT NULL UNIQUE CHECK(length(incumbent_id) > 0),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        revision INTEGER NOT NULL CHECK(revision >= 1),
        experiment_id TEXT REFERENCES experiments(experiment_id),
        eligibility TEXT NOT NULL CHECK(length(eligibility) > 0),
        source_digest TEXT NOT NULL CHECK(length(source_digest) = 64),
        checkpoint_digest TEXT NOT NULL CHECK(length(checkpoint_digest) = 64),
        artifact_closure_digest TEXT NOT NULL CHECK(length(artifact_closure_digest) = 64),
        replay_verified INTEGER NOT NULL CHECK(replay_verified IN (0, 1)),
        outer_primary_mean REAL CHECK(
            outer_primary_mean IS NULL OR (outer_primary_mean >= 0.0 AND outer_primary_mean <= 1.0)
        ),
        is_fallback INTEGER NOT NULL CHECK(is_fallback IN (0, 1)),
        reason TEXT NOT NULL,
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE UNIQUE INDEX one_fallback_per_campaign
        ON incumbent_history(campaign_id) WHERE is_fallback = 1""",
    """CREATE TABLE runtime_samples (
        runtime_sample_id INTEGER PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        execution_id TEXT REFERENCES executions(execution_id),
        family TEXT NOT NULL CHECK(length(family) > 0),
        elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds >= 0.0),
        peak_rss_bytes INTEGER NOT NULL CHECK(peak_rss_bytes >= 0),
        disk_bytes INTEGER NOT NULL CHECK(disk_bytes >= 0),
        outcome TEXT NOT NULL CHECK(length(outcome) > 0),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE validation_queries (
        query_event_id INTEGER PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        query_id TEXT NOT NULL CHECK(length(query_id) > 0),
        event_seq INTEGER NOT NULL CHECK(event_seq >= 1),
        experiment_id TEXT REFERENCES experiments(experiment_id),
        candidate_fingerprint TEXT NOT NULL CHECK(length(candidate_fingerprint) = 64),
        benchmark_digest TEXT NOT NULL CHECK(length(benchmark_digest) = 64),
        dataset_digest TEXT NOT NULL CHECK(length(dataset_digest) = 64),
        scorer_digest TEXT NOT NULL CHECK(length(scorer_digest) = 64),
        state TEXT NOT NULL CHECK(state IN ('RESERVED','COMPLETED','START_UNCERTAIN')),
        result_digest TEXT CHECK(result_digest IS NULL OR length(result_digest) = 64),
        metrics_json TEXT CHECK(metrics_json IS NULL OR json_valid(metrics_json)),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        UNIQUE(campaign_id, query_id, event_seq)
    ) STRICT""",
    """CREATE UNIQUE INDEX validation_query_candidate
        ON validation_queries(campaign_id, candidate_fingerprint) WHERE event_seq = 1""",
    *_append_only_statements(
        (
            "proposals",
            "source_snapshots",
            "metrics",
            "artifacts",
            "failures",
            "interventions",
            "transitions",
            "launches",
            "reallocations",
            "artifact_links",
            "incumbent_history",
            "runtime_samples",
            "validation_queries",
        )
    ),
)

_CAMPAIGN_TABLES: Final = frozenset(
    {
        "schema_meta",
        "campaigns",
        "experiments",
        "executions",
        "proposals",
        "source_snapshots",
        "metrics",
        "artifacts",
        "failures",
        "interventions",
        "transitions",
        "launches",
        "reallocations",
        "artifact_links",
        "incumbent_history",
        "runtime_samples",
        "validation_queries",
    }
)

_LEDGER_SCHEMA_STATEMENTS: Final = (
    """CREATE TABLE ledger_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE ledger_state (
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        revision INTEGER NOT NULL CHECK(revision >= 0),
        updated_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE outer_queries (
        query_event_id INTEGER PRIMARY KEY,
        query_id TEXT NOT NULL CHECK(length(query_id) > 0),
        event_seq INTEGER NOT NULL CHECK(event_seq >= 1),
        campaign_id TEXT NOT NULL CHECK(length(campaign_id) > 0),
        benchmark_digest TEXT NOT NULL CHECK(length(benchmark_digest) = 64),
        dataset_digest TEXT NOT NULL CHECK(length(dataset_digest) = 64),
        scorer_digest TEXT NOT NULL CHECK(length(scorer_digest) = 64),
        candidate_fingerprint TEXT NOT NULL CHECK(length(candidate_fingerprint) = 64),
        state TEXT NOT NULL CHECK(state IN ('RESERVED','COMPLETED','START_UNCERTAIN')),
        result_digest TEXT CHECK(result_digest IS NULL OR length(result_digest) = 64),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        UNIQUE(query_id, event_seq)
    ) STRICT""",
    """CREATE UNIQUE INDEX outer_query_candidate
        ON outer_queries(benchmark_digest, dataset_digest, scorer_digest, candidate_fingerprint)
        WHERE event_seq = 1""",
    *_append_only_statements(("outer_queries",)),
)

_LEDGER_TABLES: Final = frozenset({"ledger_meta", "ledger_state", "outer_queries"})

_LINEAGE_SCHEMA_STATEMENTS: Final = (
    """CREATE TABLE ledger_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE ledger_state (
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        revision INTEGER NOT NULL CHECK(revision >= 0),
        updated_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE lineage_events (
        event_id INTEGER PRIMARY KEY,
        campaign_id TEXT NOT NULL CHECK(length(campaign_id) > 0),
        benchmark_digest TEXT NOT NULL CHECK(length(benchmark_digest) = 64),
        starter_digest TEXT NOT NULL CHECK(length(starter_digest) = 64),
        source_digest TEXT NOT NULL CHECK(length(source_digest) = 64),
        outcome TEXT NOT NULL CHECK(outcome IN ('rejected', 'admitted')),
        candidate_id TEXT NOT NULL CHECK(length(candidate_id) > 0),
        proposal_family TEXT NOT NULL CHECK(length(proposal_family) > 0),
        proposal_signature TEXT CHECK(
            proposal_signature IS NULL OR length(proposal_signature) = 64
        ),
        repairs_attempted INTEGER CHECK(repairs_attempted IS NULL OR repairs_attempted >= 0),
        root_failure_fingerprint TEXT CHECK(
            root_failure_fingerprint IS NULL OR length(root_failure_fingerprint) = 64
        ),
        root_failure_category TEXT,
        root_failure_code TEXT,
        root_failure_subject TEXT,
        terminal_failure_fingerprint TEXT CHECK(
            terminal_failure_fingerprint IS NULL OR length(terminal_failure_fingerprint) = 64
        ),
        terminal_failure_category TEXT,
        terminal_failure_code TEXT,
        terminal_failure_subject TEXT,
        diagnostic TEXT,
        inner_fold_a_gauc REAL CHECK(
            inner_fold_a_gauc IS NULL OR (inner_fold_a_gauc BETWEEN 0.0 AND 1.0)
        ),
        inner_fold_a_ndcg_at_5 REAL CHECK(
            inner_fold_a_ndcg_at_5 IS NULL OR (inner_fold_a_ndcg_at_5 BETWEEN 0.0 AND 1.0)
        ),
        inner_fold_a_primary REAL CHECK(
            inner_fold_a_primary IS NULL OR (inner_fold_a_primary BETWEEN 0.0 AND 1.0)
        ),
        inner_fold_b_gauc REAL CHECK(
            inner_fold_b_gauc IS NULL OR (inner_fold_b_gauc BETWEEN 0.0 AND 1.0)
        ),
        inner_fold_b_ndcg_at_5 REAL CHECK(
            inner_fold_b_ndcg_at_5 IS NULL OR (inner_fold_b_ndcg_at_5 BETWEEN 0.0 AND 1.0)
        ),
        inner_fold_b_primary REAL CHECK(
            inner_fold_b_primary IS NULL OR (inner_fold_b_primary BETWEEN 0.0 AND 1.0)
        ),
        parent_fold_a_primary REAL CHECK(
            parent_fold_a_primary IS NULL OR (parent_fold_a_primary BETWEEN 0.0 AND 1.0)
        ),
        parent_fold_b_primary REAL CHECK(
            parent_fold_b_primary IS NULL OR (parent_fold_b_primary BETWEEN 0.0 AND 1.0)
        ),
        promoted INTEGER CHECK(promoted IS NULL OR promoted IN (0, 1)),
        metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
        created_at TEXT NOT NULL,
        CHECK(
            (outcome = 'rejected'
                AND repairs_attempted IS NOT NULL
                AND root_failure_fingerprint IS NOT NULL
                AND root_failure_category IS NOT NULL
                AND root_failure_code IS NOT NULL
                AND root_failure_subject IS NOT NULL
                AND terminal_failure_fingerprint IS NOT NULL
                AND terminal_failure_category IS NOT NULL
                AND terminal_failure_code IS NOT NULL
                AND terminal_failure_subject IS NOT NULL
                AND diagnostic IS NOT NULL
                AND inner_fold_a_gauc IS NULL AND inner_fold_a_ndcg_at_5 IS NULL
                AND inner_fold_a_primary IS NULL AND inner_fold_b_gauc IS NULL
                AND inner_fold_b_ndcg_at_5 IS NULL AND inner_fold_b_primary IS NULL
                AND parent_fold_a_primary IS NULL AND parent_fold_b_primary IS NULL
                AND promoted IS NULL)
            OR
            (outcome = 'admitted'
                AND repairs_attempted IS NULL
                AND root_failure_fingerprint IS NULL
                AND root_failure_category IS NULL
                AND root_failure_code IS NULL
                AND root_failure_subject IS NULL
                AND terminal_failure_fingerprint IS NULL
                AND terminal_failure_category IS NULL
                AND terminal_failure_code IS NULL
                AND terminal_failure_subject IS NULL
                AND diagnostic IS NULL
                -- Fold metrics are optional even for 'admitted': training can still fail (e.g.
                -- CALLBACK_FAILED) after materialization succeeds, before any metric exists. A
                -- candidate that fails the Fold B screen never reaches Fold A, so each fold's own
                -- (inner metrics, parent reference) pair travels together, but Fold A and Fold B
                -- are independent of each other. Promotion requires both folds to exist.
                AND (
                    (inner_fold_a_gauc IS NULL AND inner_fold_a_ndcg_at_5 IS NULL
                        AND inner_fold_a_primary IS NULL AND parent_fold_a_primary IS NULL)
                    OR
                    (inner_fold_a_gauc IS NOT NULL AND inner_fold_a_ndcg_at_5 IS NOT NULL
                        AND inner_fold_a_primary IS NOT NULL AND parent_fold_a_primary IS NOT NULL)
                )
                AND (
                    (inner_fold_b_gauc IS NULL AND inner_fold_b_ndcg_at_5 IS NULL
                        AND inner_fold_b_primary IS NULL AND parent_fold_b_primary IS NULL)
                    OR
                    (inner_fold_b_gauc IS NOT NULL AND inner_fold_b_ndcg_at_5 IS NOT NULL
                        AND inner_fold_b_primary IS NOT NULL AND parent_fold_b_primary IS NOT NULL)
                )
                AND (
                    promoted IS NULL
                    OR (inner_fold_a_primary IS NOT NULL AND inner_fold_b_primary IS NOT NULL)
                ))
        )
    ) STRICT""",
    """CREATE INDEX lineage_events_scope
        ON lineage_events(benchmark_digest, starter_digest, source_digest, event_id)""",
    *_append_only_statements(("lineage_events",)),
)

_LINEAGE_TABLES: Final = frozenset({"ledger_meta", "ledger_state", "lineage_events"})


def _schema_code_digest(statements: Sequence[str]) -> str:
    payload = "\n-- statement --\n".join(statement.strip() for statement in statements)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CAMPAIGN_SCHEMA_DIGEST: Final = _schema_code_digest(_CAMPAIGN_SCHEMA_STATEMENTS)
_LEDGER_SCHEMA_DIGEST: Final = _schema_code_digest(_LEDGER_SCHEMA_STATEMENTS)
_LINEAGE_SCHEMA_DIGEST: Final = _schema_code_digest(_LINEAGE_SCHEMA_STATEMENTS)


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise StoreInvariantError(f"{location} must be a non-empty string without NUL bytes")
    return value


def _optional_text(value: object, location: str) -> str | None:
    return None if value is None else _text(value, location)


def _digest(value: object, location: str) -> str:
    text = _text(value, location)
    if _DIGEST_RE.fullmatch(text) is None:
        raise StoreInvariantError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _optional_digest(value: object, location: str) -> str | None:
    return None if value is None else _digest(value, location)


def _optional_nonce(value: object, location: str) -> str | None:
    if value is None:
        return None
    nonce = _text(value, location)
    if _NONCE_RE.fullmatch(nonce) is None:
        raise StoreInvariantError(
            f"{location} must use 16-128 ASCII letters, digits, underscores, or hyphens"
        )
    return nonce


def _relative_path(value: object) -> str:
    text = _text(value, "artifact relative_path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise StoreInvariantError("artifact relative_path must be normalized and relative")
    return text


def _json(value: object, location: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StoreInvariantError(f"{location} must be finite JSON: {exc}") from exc


def _json_object(value: object, location: str) -> str:
    if not isinstance(value, Mapping):
        raise StoreInvariantError(f"{location} must be a mapping")
    for key in value:
        if type(key) is not str:
            raise StoreInvariantError(f"{location} keys must be strings")
    # ``MappingProxyType({})`` gives public APIs an immutable empty default but is not directly
    # understood by ``json``.  Materialize only the top-level object after validating its keys.
    return _json(dict(value), location)


def _load_json_object(value: object, location: str) -> Mapping[str, object]:
    text = _text(value, location)
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StoreVersionError(f"{location} contains invalid JSON") from exc
    if not isinstance(decoded, dict) or any(type(key) is not str for key in decoded):
        raise StoreVersionError(f"{location} must contain a JSON object")
    if _json_object(decoded, location) != text:
        raise StoreVersionError(f"{location} must contain canonical JSON")
    frozen = _freeze_json(decoded, location)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by decoded type above.
        raise StoreVersionError(f"{location} must contain a JSON object")
    return cast(Mapping[str, object], frozen)


def _freeze_json(value: object, location: str) -> object:
    """Recursively freeze already-decoded JSON for immutable public projections."""

    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise StoreVersionError(f"{location} contains a non-string object key")
        return MappingProxyType(
            {cast(str, key): _freeze_json(item, location) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item, location) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise StoreVersionError(f"{location} contains a non-JSON value")


def _load_json_string_tuple(value: object, location: str) -> tuple[str, ...]:
    text = _text(value, location)
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StoreVersionError(f"{location} contains invalid JSON") from exc
    if (
        not isinstance(decoded, list)
        or not decoded
        or any(type(item) is not str or not item for item in decoded)
    ):
        raise StoreVersionError(f"{location} must contain a non-empty JSON string array")
    if _json(decoded, location) != text:
        raise StoreVersionError(f"{location} must contain canonical JSON")
    return tuple(cast(list[str], decoded))


def _convergence_json(value: Mapping[str, object]) -> str:
    expected = {
        "schema_version",
        "best_primary",
        "non_material_streak",
        "completed_iterations",
        "required_completion_pending",
    }
    if set(value) != expected:
        raise StoreInvariantError("convergence state must contain the exact version-1 fields")
    if value["schema_version"] != 1:
        raise StoreInvariantError("convergence schema_version must be 1")
    best = value["best_primary"]
    if isinstance(best, bool) or not isinstance(best, (int, float)):
        raise StoreInvariantError("convergence best_primary must be numeric")
    if not math.isfinite(float(best)) or not 0.0 <= float(best) <= 1.0:
        raise StoreInvariantError("convergence best_primary must be finite in [0, 1]")
    for key in ("non_material_streak", "completed_iterations"):
        number = value[key]
        if type(number) is not int or number < 0:
            raise StoreInvariantError(f"convergence {key} must be a non-negative integer")
    if cast(int, value["non_material_streak"]) > cast(int, value["completed_iterations"]):
        raise StoreInvariantError(
            "convergence non_material_streak cannot exceed completed_iterations"
        )
    if type(value["required_completion_pending"]) is not bool:
        raise StoreInvariantError("convergence required_completion_pending must be boolean")
    return _json_object(value, "convergence state")


def _timestamp(value: object, location: str) -> str:
    text = _text(value, location)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise StoreInvariantError(f"{location} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise StoreInvariantError(f"{location} must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


_PROCESS_RECORD_KEYS: Final = frozenset(
    {
        "schema_version",
        "execution_id",
        "nonce",
        "pid",
        "process_create_time",
        "process_group_id",
        "command",
        "command_digest",
        "environment_digest",
        "interpreter_real_path",
        "workspace",
        "control_dir",
        "source_digest",
        "config_digest",
        "data_digest",
        "checkpoint_digest",
        "started_at_utc",
        "launcher_sha256",
    }
)


def _runner_command_digest(command: Sequence[str]) -> str:
    digest = hashlib.sha256(b"kuairand-runner-command-v1\0")
    digest.update(_json(list(command), "runner command").encode("ascii"))
    return digest.hexdigest()


def _runner_process_record_digest(record: Mapping[str, object]) -> str:
    digest = hashlib.sha256(b"kuairand-runner-process-record-v2\0")
    digest.update(_json(dict(record), "runner process record").encode("ascii"))
    return digest.hexdigest()


def _validated_process_record(
    raw: Mapping[str, object],
    *,
    execution_id: str,
    nonce: str,
    command: tuple[str, ...],
    source_digest: str | None,
    config_digest: str | None,
    data_digest: str | None,
    checkpoint_digest: str | None,
) -> tuple[str, Mapping[str, object], str]:
    """Validate the exact trusted runner-v2 process receipt and return frozen canonical JSON."""

    if set(raw) != _PROCESS_RECORD_KEYS:
        raise StoreInvariantError("process record must contain the exact runner-v2 fields")
    if raw["schema_version"] != 2:
        raise StoreInvariantError("process record schema_version must be 2")
    if raw["execution_id"] != execution_id:
        raise StoreInvariantError("process record execution_id does not match the execution")
    if raw["nonce"] != nonce:
        raise StoreInvariantError("process record nonce does not match the execution")
    pid = raw["pid"]
    process_group_id = raw["process_group_id"]
    if type(pid) is not int or pid <= 0:
        raise StoreInvariantError("process record pid must be a positive integer")
    if type(process_group_id) is not int or process_group_id <= 0:
        raise StoreInvariantError("process record process_group_id must be positive")
    create_time = raw["process_create_time"]
    if (
        isinstance(create_time, bool)
        or not isinstance(create_time, (int, float))
        or not math.isfinite(float(create_time))
        or float(create_time) <= 0.0
    ):
        raise StoreInvariantError("process record create time must be finite and positive")
    raw_command = raw["command"]
    if (
        not isinstance(raw_command, (list, tuple))
        or not raw_command
        or any(type(part) is not str or not part for part in raw_command)
    ):
        raise StoreInvariantError("process record command must be a non-empty string array")
    observed_command = tuple(cast(Sequence[str], raw_command))
    if observed_command != command:
        raise StoreInvariantError("process record command does not match the execution")
    command_digest = _digest(raw["command_digest"], "process record command_digest")
    if command_digest != _runner_command_digest(command):
        raise StoreInvariantError("process record command_digest is invalid")
    # This is the runner's sanitized-runtime identity.  It is deliberately distinct from the
    # execution's locked scientific environment identity because the runtime root is allocated
    # only during launch, so its exact digest cannot be known when STARTING is persisted.
    _digest(raw["environment_digest"], "process record environment_digest")
    _digest(raw["launcher_sha256"], "process record launcher_sha256")
    interpreter = _text(raw["interpreter_real_path"], "process record interpreter_real_path")
    workspace = _text(raw["workspace"], "process record workspace")
    control_dir = _text(raw["control_dir"], "process record control_dir")
    if (
        not Path(interpreter).is_absolute()
        or not Path(workspace).is_absolute()
        or not Path(control_dir).is_absolute()
    ):
        raise StoreInvariantError(
            "process record interpreter, workspace, and control_dir must be absolute"
        )
    if Path(control_dir).is_relative_to(Path(workspace)) or Path(workspace).is_relative_to(
        Path(control_dir)
    ):
        raise StoreInvariantError("process record workspace and control_dir must be disjoint")
    for key, expected in (
        ("source_digest", source_digest),
        ("config_digest", config_digest),
        ("data_digest", data_digest),
        ("checkpoint_digest", checkpoint_digest),
    ):
        observed = _digest(raw[key], f"process record {key}")
        if expected is None or observed != expected:
            raise StoreInvariantError(f"process record {key} does not match the execution")
    _timestamp(raw["started_at_utc"], "process record started_at_utc")
    encoded = _json_object(raw, "process record")
    decoded = _load_json_object(encoded, "process record")
    return encoded, decoded, _runner_process_record_digest(raw)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _claim_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError as exc:
        raise CampaignExistsError(f"campaign database already exists: {path}") from exc
    os.close(descriptor)


def _cleanup_claimed_database(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, isolation_level=None)
    else:
        connection = sqlite3.connect(path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA foreign_keys = ON")
    if read_only:
        connection.execute("PRAGMA query_only = ON")
    else:
        mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
        if mode != "wal":
            connection.close()
            raise StoreVersionError(f"SQLite refused WAL mode, got {mode!r}")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA wal_autocheckpoint = 1000")
    return connection


def _catalog_digest(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        """SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name"""
    ).fetchall()
    payload: list[list[str | None]] = []
    for row in rows:
        payload.append(
            [
                cast(str, row["type"]),
                cast(str, row["name"]),
                cast(str, row["tbl_name"]),
                cast(str | None, row["sql"]),
            ]
        )
    return hashlib.sha256(_json(payload, "schema catalog").encode("ascii")).hexdigest()


def _initialize_schema(
    connection: sqlite3.Connection,
    *,
    statements: Sequence[str],
    meta_table: str,
    kind: str,
    schema_digest: str,
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in statements:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        catalog = _catalog_digest(connection)
        connection.executemany(
            f"INSERT INTO {meta_table}(key, value) VALUES (?, ?)",
            (
                ("kind", kind),
                ("schema_version", str(SCHEMA_VERSION)),
                ("schema_digest", schema_digest),
                ("catalog_digest", catalog),
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _meta(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    try:
        rows = connection.execute(f"SELECT key, value FROM {table}").fetchall()
    except sqlite3.DatabaseError as exc:
        raise StoreVersionError(f"missing or unreadable {table}") from exc
    result: dict[str, str] = {}
    for row in rows:
        result[cast(str, row["key"])] = cast(str, row["value"])
    return result


def _verify_schema(
    connection: sqlite3.Connection,
    *,
    meta_table: str,
    kind: str,
    schema_digest: str,
    expected_tables: frozenset[str],
) -> None:
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    if user_version != SCHEMA_VERSION or application_id != APPLICATION_ID:
        message = (
            f"unsupported database identity user_version={user_version}, "
            f"application_id={application_id}"
        )
        raise StoreVersionError(message)
    values = _meta(connection, meta_table)
    expected_meta = {"kind", "schema_version", "schema_digest", "catalog_digest"}
    if set(values) < expected_meta:
        raise StoreVersionError(f"{meta_table} is missing required schema identity fields")
    if values["kind"] != kind or values["schema_version"] != str(SCHEMA_VERSION):
        raise StoreVersionError("database kind or schema version does not match this store")
    if values["schema_digest"] != schema_digest:
        raise StoreVersionError("database was created by a different schema definition")
    if values["catalog_digest"] != _catalog_digest(connection):
        raise StoreVersionError("database schema catalog was modified")
    actual_tables = frozenset(
        cast(str, row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )
    if actual_tables != expected_tables:
        raise StoreVersionError("database table set does not match the strict schema")
    table_rows = connection.execute("PRAGMA table_list").fetchall()
    strict_by_name = {cast(str, row[1]): int(row[5]) for row in table_rows}
    if any(strict_by_name.get(table) != 1 for table in expected_tables):
        raise StoreVersionError("every application table must use SQLite STRICT mode")
    if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise StoreVersionError("SQLite foreign key enforcement is disabled")
    if str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
        raise StoreVersionError("campaign databases must remain in WAL mode")
    if int(connection.execute("PRAGMA synchronous").fetchone()[0]) != 2:
        raise StoreVersionError("campaign databases must use synchronous FULL")
    if cast(str, connection.execute("PRAGMA quick_check").fetchone()[0]) != "ok":
        raise StoreVersionError("SQLite quick_check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise StoreVersionError("SQLite foreign_key_check failed")


def _row_text(row: sqlite3.Row, key: str) -> str:
    return cast(str, row[key])


def _row_optional_text(row: sqlite3.Row, key: str) -> str | None:
    return cast(str | None, row[key])


def _row_optional_int(row: sqlite3.Row, key: str) -> int | None:
    return cast(int | None, row[key])


def _stored_digest(value: object, location: str) -> str:
    try:
        return _digest(value, location)
    except StoreInvariantError as exc:
        raise StoreVersionError(str(exc)) from exc


def _stored_text(value: object, location: str) -> str:
    try:
        return _text(value, location)
    except StoreInvariantError as exc:
        raise StoreVersionError(str(exc)) from exc


def _stored_optional_text(value: object, location: str) -> str | None:
    return None if value is None else _stored_text(value, location)


def _stored_optional_digest(value: object, location: str) -> str | None:
    return None if value is None else _stored_digest(value, location)


def _stored_timestamp(value: object, location: str) -> str:
    try:
        normalized = _timestamp(value, location)
    except StoreInvariantError as exc:
        raise StoreVersionError(str(exc)) from exc
    if normalized != value:
        raise StoreVersionError(f"{location} must use canonical UTC microsecond form")
    return normalized


def _stored_optional_timestamp(value: object, location: str) -> str | None:
    return None if value is None else _stored_timestamp(value, location)


class CampaignStore:
    """Strict single-campaign store with optimistic, append-only evidence APIs."""

    def __init__(self, path: Path, connection: sqlite3.Connection, *, read_only: bool) -> None:
        self.path = path
        self._connection = connection
        self._read_only = read_only
        row = connection.execute("SELECT campaign_id FROM campaigns").fetchall()
        if len(row) != 1:
            raise StoreVersionError("a campaign database must contain exactly one campaign")
        self.campaign_id = _row_text(row[0], "campaign_id")

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        campaign_id: str,
        config_digest: str,
        benchmark_digest: str,
        starter_digest: str,
        dataset_digest: str,
        environment_digest: str,
        hard_deadline_utc: str,
        initial_convergence: Mapping[str, object],
        source_digest: str | None = None,
        max_launches: int = DEFAULT_MAX_LAUNCHES,
        outer_query_limit: int = DEFAULT_OUTER_QUERY_LIMIT,
    ) -> Self:
        """Create a new database with ``O_EXCL`` semantics; never overwrite an existing run."""

        database = Path(path).absolute()
        identity = _text(campaign_id, "campaign_id")
        config = _digest(config_digest, "config_digest")
        benchmark = _digest(benchmark_digest, "benchmark_digest")
        starter = _digest(starter_digest, "starter_digest")
        dataset = _digest(dataset_digest, "dataset_digest")
        environment = _digest(environment_digest, "environment_digest")
        source = _optional_digest(source_digest, "source_digest")
        deadline = _timestamp(hard_deadline_utc, "hard_deadline_utc")
        convergence = _convergence_json(initial_convergence)
        if type(max_launches) is not int or not 6 <= max_launches <= DEFAULT_MAX_LAUNCHES:
            raise StoreInvariantError("max_launches must be an integer in [6, 50]")
        if type(outer_query_limit) is not int or not 0 <= outer_query_limit <= 6:
            raise StoreInvariantError("outer_query_limit must be an integer in [0, 6]")

        _claim_database(database)
        connection: sqlite3.Connection | None = None
        try:
            connection = _connect(database, read_only=False)
            _initialize_schema(
                connection,
                statements=_CAMPAIGN_SCHEMA_STATEMENTS,
                meta_table="schema_meta",
                kind="kuairand-campaign-store",
                schema_digest=_CAMPAIGN_SCHEMA_DIGEST,
            )
            now = _now()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """INSERT INTO campaigns(
                        campaign_id, revision, status, phase, config_digest, benchmark_digest,
                        starter_digest, dataset_digest, environment_digest, source_digest,
                        hard_deadline_utc, max_launches, outer_query_limit, qualification_digest,
                        convergence_json, created_at, updated_at
                    ) VALUES (
                        ?, 0, 'CREATED', 'qualifying', ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?
                    )""",
                    (
                        identity,
                        config,
                        benchmark,
                        starter,
                        dataset,
                        environment,
                        source,
                        deadline,
                        max_launches,
                        outer_query_limit,
                        convergence,
                        now,
                        now,
                    ),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            os.chmod(database, 0o600)
            return cls(database, connection, read_only=False)
        except BaseException:
            if connection is not None:
                connection.close()
            _cleanup_claimed_database(database)
            raise

    @classmethod
    def open(
        cls, path: str | Path, *, read_only: bool = False, campaign_id: str | None = None
    ) -> Self:
        """Open and fully verify an existing campaign database."""

        database = Path(path).absolute()
        if database.is_symlink() or not database.is_file():
            raise CampaignNotFoundError(f"campaign database does not exist: {database}")
        connection = _connect(database, read_only=read_only)
        try:
            _verify_schema(
                connection,
                meta_table="schema_meta",
                kind="kuairand-campaign-store",
                schema_digest=_CAMPAIGN_SCHEMA_DIGEST,
                expected_tables=_CAMPAIGN_TABLES,
            )
            store = cls(database, connection, read_only=read_only)
            if campaign_id is not None and store.campaign_id != _text(campaign_id, "campaign_id"):
                raise CampaignNotFoundError(
                    f"database contains campaign {store.campaign_id!r}, not {campaign_id!r}"
                )
            return store
        except BaseException:
            connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        if write and self._read_only:
            raise StoreInvariantError("read-only campaign store cannot be mutated")
        self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield self._connection
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def _campaign_row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM campaigns WHERE campaign_id = ?", (self.campaign_id,)
        ).fetchone()
        if row is None:
            raise CampaignNotFoundError(f"campaign disappeared: {self.campaign_id}")
        return cast(sqlite3.Row, row)

    def _require_revision(
        self, connection: sqlite3.Connection, expected_revision: int
    ) -> sqlite3.Row:
        if type(expected_revision) is not int or expected_revision < 0:
            raise StoreInvariantError("expected_revision must be a non-negative integer")
        row = self._campaign_row(connection)
        actual = int(row["revision"])
        if actual != expected_revision:
            raise RevisionConflictError(
                f"campaign revision changed: expected {expected_revision}, found {actual}"
            )
        return row

    def _advance_revision(
        self,
        connection: sqlite3.Connection,
        *,
        expected_revision: int,
        entity_type: str,
        entity_id: str,
        from_state: str | None,
        to_state: str,
        reason: str,
        metadata: Mapping[str, object],
    ) -> int:
        next_revision = expected_revision + 1
        now = _now()
        changed = connection.execute(
            """UPDATE campaigns SET revision = ?, updated_at = ?
            WHERE campaign_id = ? AND revision = ?""",
            (next_revision, now, self.campaign_id, expected_revision),
        ).rowcount
        if changed != 1:
            raise RevisionConflictError("campaign revision changed during transaction")
        connection.execute(
            """INSERT INTO transitions(
                campaign_id, revision_before, revision_after, entity_type, entity_id,
                from_state, to_state, reason, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.campaign_id,
                expected_revision,
                next_revision,
                _text(entity_type, "transition entity_type"),
                _text(entity_id, "transition entity_id"),
                from_state,
                _text(to_state, "transition to_state"),
                _text(reason, "transition reason"),
                _json_object(metadata, "transition metadata"),
                now,
            ),
        )
        return next_revision

    def health(self) -> StoreHealth:
        values = _meta(self._connection, "schema_meta")
        return StoreHealth(
            journal_mode=str(self._connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
            foreign_keys=bool(self._connection.execute("PRAGMA foreign_keys").fetchone()[0]),
            synchronous=int(self._connection.execute("PRAGMA synchronous").fetchone()[0]),
            user_version=int(self._connection.execute("PRAGMA user_version").fetchone()[0]),
            quick_check=cast(str, self._connection.execute("PRAGMA quick_check").fetchone()[0]),
            schema_digest=values["schema_digest"],
            catalog_digest=values["catalog_digest"],
        )

    def identity(self) -> CampaignIdentityRecord:
        """Return the immutable campaign identity from one stable read transaction."""

        with self._transaction(write=False) as connection:
            row = self._campaign_row(connection)
            return CampaignIdentityRecord(
                campaign_id=self.campaign_id,
                config_digest=_stored_digest(row["config_digest"], "campaign config_digest"),
                benchmark_digest=_stored_digest(
                    row["benchmark_digest"], "campaign benchmark_digest"
                ),
                starter_manifest_digest=_stored_digest(
                    row["starter_digest"], "campaign starter_digest"
                ),
                dataset_manifest_digest=_stored_digest(
                    row["dataset_digest"], "campaign dataset_digest"
                ),
                source_digest=_stored_optional_digest(
                    row["source_digest"], "campaign source_digest"
                ),
                environment_digest=_stored_digest(
                    row["environment_digest"], "campaign environment_digest"
                ),
                hard_deadline_utc=_stored_timestamp(
                    row["hard_deadline_utc"], "campaign hard_deadline_utc"
                ),
                max_launches=int(row["max_launches"]),
                outer_query_limit=int(row["outer_query_limit"]),
                created_at=_stored_timestamp(row["created_at"], "campaign created_at"),
            )

    def snapshot(self) -> CampaignSnapshot:
        """Read all summary fields from one stable SQLite snapshot."""

        with self._transaction(write=False) as connection:
            row = self._campaign_row(connection)
            used = int(
                connection.execute(
                    """WITH latest AS (
                        SELECT launch_id, MAX(event_seq) AS event_seq
                        FROM launches WHERE campaign_id = ? GROUP BY launch_id
                    )
                    SELECT COUNT(*) FROM launches AS item
                    JOIN latest USING(launch_id, event_seq)
                    WHERE item.campaign_id = ? AND item.charged = 1""",
                    (self.campaign_id, self.campaign_id),
                ).fetchone()[0]
            )
            outer_used = int(
                connection.execute(
                    """SELECT COUNT(*) FROM validation_queries
                    WHERE campaign_id = ? AND event_seq = 1""",
                    (self.campaign_id,),
                ).fetchone()[0]
            )
            incumbent_row = connection.execute(
                """SELECT * FROM incumbent_history
                WHERE campaign_id = ? ORDER BY history_id DESC LIMIT 1""",
                (self.campaign_id,),
            ).fetchone()
            convergence = _load_json_object(row["convergence_json"], "campaign convergence_json")
            return CampaignSnapshot(
                campaign_id=self.campaign_id,
                revision=int(row["revision"]),
                status=_row_text(row, "status"),
                phase=_row_text(row, "phase"),
                max_launches=int(row["max_launches"]),
                launches_used=used,
                outer_query_limit=int(row["outer_query_limit"]),
                outer_queries_used=outer_used,
                qualification_digest=_row_optional_text(row, "qualification_digest"),
                convergence_state=convergence,
                incumbent=(
                    None if incumbent_row is None else self._incumbent_from_row(incumbent_row)
                ),
            )

    def _latest_launch_row(
        self, connection: sqlite3.Connection, launch_id: str
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """SELECT * FROM launches WHERE campaign_id = ? AND launch_id = ?
            ORDER BY event_seq DESC LIMIT 1""",
            (self.campaign_id, launch_id),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    @staticmethod
    def _launch_from_row(row: sqlite3.Row) -> LaunchRecord:
        return LaunchRecord(
            launch_id=_row_text(row, "launch_id"),
            launch_number=int(row["launch_number"]),
            reservation_key=_row_text(row, "reservation_key"),
            category=_row_text(row, "category"),
            original_category=_row_text(row, "original_category"),
            purpose=_row_text(row, "purpose"),
            state=_row_text(row, "state"),
            charged=bool(row["charged"]),
            event_seq=int(row["event_seq"]),
            experiment_id=_row_optional_text(row, "experiment_id"),
            scientific_iteration=_row_optional_int(row, "scientific_iteration"),
            seed=_row_optional_int(row, "seed"),
            start_receipt_digest=_row_optional_text(row, "start_receipt_digest"),
        )

    def launches(self) -> tuple[LaunchRecord, ...]:
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """WITH latest AS (
                    SELECT launch_id, MAX(event_seq) AS event_seq
                    FROM launches WHERE campaign_id = ? GROUP BY launch_id
                )
                SELECT item.* FROM launches AS item
                JOIN latest USING(launch_id, event_seq)
                WHERE item.campaign_id = ? ORDER BY item.launch_number""",
                (self.campaign_id, self.campaign_id),
            ).fetchall()
            return tuple(self._launch_from_row(row) for row in rows)

    def import_qualification_launches(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        manifest_digest: str,
        expected_revision: int,
    ) -> tuple[LaunchRecord, ...]:
        """Atomically charge the exact five FM seeds plus clean seed-zero retrain."""

        manifest = _digest(manifest_digest, "qualification manifest_digest")
        if len(records) != QUALIFICATION_LAUNCH_COUNT:
            raise StoreInvariantError(
                "qualification import must contain exactly six launch records"
            )
        normalized: list[dict[str, object]] = []
        expected_keys = {"launch_number", "kind", "seed", "charged"}
        for record, (number, kind, seed) in zip(records, _QUALIFICATION_PATTERN, strict=True):
            if set(record) != expected_keys:
                raise StoreInvariantError("qualification launch records must have the exact fields")
            if (
                record["launch_number"] != number
                or record["kind"] != kind
                or record["seed"] != seed
                or record["charged"] is not True
            ):
                raise StoreInvariantError(
                    "qualification launch records do not match the frozen six"
                )
            normalized.append(dict(record))

        with self._transaction(write=True) as connection:
            campaign = self._require_revision(connection, expected_revision)
            if campaign["qualification_digest"] is not None:
                raise StoreInvariantError("qualification launches were already imported")
            if (
                connection.execute(
                    "SELECT 1 FROM launches WHERE campaign_id = ? LIMIT 1", (self.campaign_id,)
                ).fetchone()
                is not None
            ):
                raise StoreInvariantError("qualification must be imported before any other launch")
            now = _now()
            for record in normalized:
                number = cast(int, record["launch_number"])
                kind = cast(str, record["kind"])
                seed = cast(int, record["seed"])
                connection.execute(
                    """INSERT INTO launches(
                        campaign_id, launch_id, event_seq, launch_number, reservation_key,
                        category, original_category, purpose, experiment_id,
                        scientific_iteration, seed, state, charged, start_receipt_digest,
                        metadata_json, created_at
                    ) VALUES (?, ?, 1, ?, ?, 'baseline_qualification',
                        'baseline_qualification', ?, NULL, NULL, ?, 'FINISHED', 1, NULL, ?, ?)""",
                    (
                        self.campaign_id,
                        f"qualification-{number:02d}",
                        number,
                        f"qualification:{manifest}:{number}",
                        kind,
                        seed,
                        _json_object(record, "qualification launch record"),
                        now,
                    ),
                )
            connection.execute(
                "UPDATE campaigns SET qualification_digest = ? WHERE campaign_id = ?",
                (manifest, self.campaign_id),
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="qualification",
                entity_id=manifest,
                from_state="UNIMPORTED",
                to_state="IMPORTED",
                reason="import exact six qualification launches",
                metadata={"launch_count": QUALIFICATION_LAUNCH_COUNT},
            )
        return self.launches()

    def reserve_launch(
        self,
        *,
        launch_id: str,
        reservation_key: str,
        category: str,
        purpose: str,
        expected_revision: int,
        experiment_id: str | None = None,
        scientific_iteration: int | None = None,
        seed: int | None = None,
        original_category: str | None = None,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> LaunchRecord:
        """Reserve and charge one launch before any process can start.

        An exact retry by ``reservation_key`` is idempotent even if the caller retained the old
        revision after an uncertain response.
        """

        identifier = _text(launch_id, "launch_id")
        key = _text(reservation_key, "reservation_key")
        launch_category = _text(category, "launch category")
        original = (
            launch_category
            if original_category is None
            else _text(original_category, "original_category")
        )
        launch_purpose = _text(purpose, "launch purpose")
        experiment = _optional_text(experiment_id, "experiment_id")
        if scientific_iteration is not None and (
            type(scientific_iteration) is not int or scientific_iteration < 0
        ):
            raise StoreInvariantError("scientific_iteration must be a non-negative integer")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise StoreInvariantError("seed must be a non-negative integer")
        metadata_json = _json_object(metadata, "launch metadata")

        with self._transaction(write=True) as connection:
            existing = connection.execute(
                """SELECT * FROM launches
                WHERE campaign_id = ? AND reservation_key = ? AND event_seq = 1""",
                (self.campaign_id, key),
            ).fetchone()
            if existing is not None:
                if (
                    _row_text(existing, "launch_id") != identifier
                    or _row_text(existing, "category") != launch_category
                    or _row_text(existing, "original_category") != original
                    or _row_text(existing, "purpose") != launch_purpose
                    or _row_optional_text(existing, "experiment_id") != experiment
                    or _row_optional_int(existing, "scientific_iteration") != scientific_iteration
                    or _row_optional_int(existing, "seed") != seed
                    or _row_text(existing, "metadata_json") != metadata_json
                ):
                    raise StoreInvariantError("reservation_key already names a different launch")
                latest = self._latest_launch_row(connection, identifier)
                if latest is None:
                    raise StoreInvariantError("launch reservation lost its latest event")
                return self._launch_from_row(latest)

            campaign = self._require_revision(connection, expected_revision)
            used = int(
                connection.execute(
                    """WITH latest AS (
                        SELECT launch_id, MAX(event_seq) AS event_seq
                        FROM launches WHERE campaign_id = ? GROUP BY launch_id
                    ) SELECT COUNT(*) FROM launches AS item
                    JOIN latest USING(launch_id, event_seq)
                    WHERE item.campaign_id = ? AND item.charged = 1""",
                    (self.campaign_id, self.campaign_id),
                ).fetchone()[0]
            )
            if used >= int(campaign["max_launches"]):
                raise LaunchLimitError("launch reservation would exceed the frozen campaign limit")
            next_number = int(
                connection.execute(
                    """SELECT COALESCE(MAX(launch_number), 0) + 1 FROM launches
                    WHERE campaign_id = ? AND event_seq = 1""",
                    (self.campaign_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO launches(
                    campaign_id, launch_id, event_seq, launch_number, reservation_key,
                    category, original_category, purpose, experiment_id, scientific_iteration,
                    seed, state, charged, start_receipt_digest, metadata_json, created_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, 'RESERVED', 1, NULL, ?, ?)""",
                (
                    self.campaign_id,
                    identifier,
                    next_number,
                    key,
                    launch_category,
                    original,
                    launch_purpose,
                    experiment,
                    scientific_iteration,
                    seed,
                    metadata_json,
                    _now(),
                ),
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="launch",
                entity_id=identifier,
                from_state=None,
                to_state="RESERVED",
                reason="reserve launch before external execution",
                metadata={"launch_number": next_number, "category": launch_category},
            )
            latest = self._latest_launch_row(connection, identifier)
            if latest is None:
                raise StoreInvariantError("reserved launch could not be reloaded")
            return self._launch_from_row(latest)

    def transition_launch(
        self,
        launch_id: str,
        *,
        to_state: LaunchState,
        expected_revision: int,
        start_receipt_digest: str | None = None,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> LaunchRecord:
        """Append a launch event; only ``NOT_STARTED`` releases its conservative charge."""

        identifier = _text(launch_id, "launch_id")
        desired = _text(to_state, "launch state")
        receipt = _optional_digest(start_receipt_digest, "start receipt digest")
        metadata_json = _json_object(metadata, "launch transition metadata")
        with self._transaction(write=True) as connection:
            latest = self._latest_launch_row(connection, identifier)
            if latest is None:
                raise StoreInvariantError(f"unknown launch_id {identifier!r}")
            current = _row_text(latest, "state")
            if current == desired:
                current_receipt = _row_optional_text(latest, "start_receipt_digest")
                if receipt is not None and current_receipt != receipt:
                    raise StoreInvariantError(
                        "launch transition retry supplied a different start receipt"
                    )
                if _row_text(latest, "metadata_json") != metadata_json:
                    raise StoreInvariantError(
                        "launch transition retry supplied different immutable metadata"
                    )
                return self._launch_from_row(latest)
            allowed = _LAUNCH_TRANSITIONS.get(current, frozenset())
            if desired not in allowed:
                raise StoreInvariantError(f"invalid launch transition {current} -> {desired}")
            self._require_revision(connection, expected_revision)
            previous_receipt = _row_optional_text(latest, "start_receipt_digest")
            effective_receipt = receipt if receipt is not None else previous_receipt
            if desired == "STARTED" and effective_receipt is None:
                raise StoreInvariantError("STARTED launch requires a durable start receipt digest")
            charged = 0 if desired == "NOT_STARTED" else 1
            connection.execute(
                """INSERT INTO launches(
                    campaign_id, launch_id, event_seq, launch_number, reservation_key,
                    category, original_category, purpose, experiment_id, scientific_iteration,
                    seed, state, charged, start_receipt_digest, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.campaign_id,
                    identifier,
                    int(latest["event_seq"]) + 1,
                    int(latest["launch_number"]),
                    _row_text(latest, "reservation_key"),
                    _row_text(latest, "category"),
                    _row_text(latest, "original_category"),
                    _row_text(latest, "purpose"),
                    _row_optional_text(latest, "experiment_id"),
                    _row_optional_int(latest, "scientific_iteration"),
                    _row_optional_int(latest, "seed"),
                    desired,
                    charged,
                    effective_receipt,
                    metadata_json,
                    _now(),
                ),
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="launch",
                entity_id=identifier,
                from_state=current,
                to_state=desired,
                reason="append launch reconciliation event",
                metadata=metadata,
            )
            updated = self._latest_launch_row(connection, identifier)
            if updated is None:
                raise StoreInvariantError("launch transition could not be reloaded")
            return self._launch_from_row(updated)

    def set_convergence_state(
        self,
        state: Mapping[str, object],
        *,
        expected_revision: int,
        reason: str = "persist convergence state",
    ) -> CampaignSnapshot:
        encoded = _convergence_json(state)
        with self._transaction(write=True) as connection:
            campaign = self._require_revision(connection, expected_revision)
            if campaign["convergence_json"] != encoded:
                connection.execute(
                    "UPDATE campaigns SET convergence_json = ? WHERE campaign_id = ?",
                    (encoded, self.campaign_id),
                )
                self._advance_revision(
                    connection,
                    expected_revision=expected_revision,
                    entity_type="campaign",
                    entity_id=self.campaign_id,
                    from_state=_row_text(campaign, "status"),
                    to_state=_row_text(campaign, "status"),
                    reason=reason,
                    metadata={"convergence_state": state},
                )
        return self.snapshot()

    def set_campaign_phase(
        self,
        *,
        phase: str,
        status: str,
        expected_revision: int,
        reason: str,
    ) -> CampaignSnapshot:
        next_phase = _text(phase, "campaign phase")
        next_status = _text(status, "campaign status")
        with self._transaction(write=True) as connection:
            current = self._require_revision(connection, expected_revision)
            connection.execute(
                "UPDATE campaigns SET phase = ?, status = ? WHERE campaign_id = ?",
                (next_phase, next_status, self.campaign_id),
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="campaign",
                entity_id=self.campaign_id,
                from_state=_row_text(current, "status"),
                to_state=next_status,
                reason=reason,
                metadata={"from_phase": _row_text(current, "phase"), "to_phase": next_phase},
            )
        return self.snapshot()

    def create_experiment(
        self,
        *,
        experiment_id: str,
        iteration_number: int,
        hypothesis: str,
        mechanism: str,
        method_attribution: str,
        expected_revision: int,
        parent_experiment_id: str | None = None,
        status: str = "PLANNED",
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        identifier = _text(experiment_id, "experiment_id")
        if type(iteration_number) is not int or iteration_number < 0:
            raise StoreInvariantError("iteration_number must be a non-negative integer")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO experiments(
                    experiment_id, campaign_id, iteration_number, parent_experiment_id,
                    hypothesis, mechanism, method_attribution, status, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    iteration_number,
                    _optional_text(parent_experiment_id, "parent_experiment_id"),
                    _text(hypothesis, "hypothesis"),
                    _text(mechanism, "mechanism"),
                    _text(method_attribution, "method_attribution"),
                    _text(status, "experiment status"),
                    _json_object(metadata, "experiment metadata"),
                    _now(),
                ),
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="experiment",
                entity_id=identifier,
                from_state=None,
                to_state=status,
                reason="create experiment",
                metadata={"iteration_number": iteration_number},
            )

    @staticmethod
    def _experiment_from_row(row: sqlite3.Row) -> ExperimentRecord:
        iteration_number = row["iteration_number"]
        if type(iteration_number) is not int or iteration_number < 0:
            raise StoreVersionError(
                "stored experiment iteration_number must be a non-negative integer"
            )
        return ExperimentRecord(
            experiment_id=_stored_text(row["experiment_id"], "experiment experiment_id"),
            iteration_number=iteration_number,
            parent_experiment_id=_stored_optional_text(
                row["parent_experiment_id"], "experiment parent_experiment_id"
            ),
            hypothesis=_stored_text(row["hypothesis"], "experiment hypothesis"),
            mechanism=_stored_text(row["mechanism"], "experiment mechanism"),
            method_attribution=_stored_text(
                row["method_attribution"], "experiment method_attribution"
            ),
            status=_stored_text(row["status"], "experiment status"),
            metadata=_load_json_object(row["metadata_json"], "experiment metadata_json"),
            created_at=_stored_timestamp(row["created_at"], "experiment created_at"),
        )

    def experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Return one exact experiment projection, or ``None`` when it is absent."""

        identifier = _text(experiment_id, "experiment_id")
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """SELECT * FROM experiments
                WHERE campaign_id = ? AND experiment_id = ?""",
                (self.campaign_id, identifier),
            ).fetchone()
            return None if row is None else self._experiment_from_row(row)

    def create_execution(
        self,
        *,
        execution_id: str,
        kind: str,
        tier: str,
        command: Sequence[str],
        expected_revision: int,
        experiment_id: str | None = None,
        launch_id: str | None = None,
        seed: int | None = None,
        status: str = "PLANNED",
        source_digest: str | None = None,
        config_digest: str | None = None,
        capability_digest: str | None = None,
        environment_digest: str | None = None,
        data_digest: str | None = None,
        checkpoint_digest: str | None = None,
        nonce: str | None = None,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        identifier = _text(execution_id, "execution_id")
        if not command or any(type(part) is not str or not part for part in command):
            raise StoreInvariantError("execution command must be a non-empty string sequence")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise StoreInvariantError("execution seed must be a non-negative integer")
        state = _text(status, "execution status")
        execution_nonce = _optional_nonce(nonce, "execution nonce")
        source_identity = _optional_digest(source_digest, "execution source_digest")
        config_identity = _optional_digest(config_digest, "execution config_digest")
        capability_identity = _optional_digest(capability_digest, "execution capability_digest")
        environment_identity = _optional_digest(environment_digest, "execution environment_digest")
        data_identity = _optional_digest(data_digest, "execution data_digest")
        checkpoint_identity = _optional_digest(checkpoint_digest, "execution checkpoint_digest")
        if state == "STARTING" and execution_nonce is None:
            raise StoreInvariantError("STARTING execution requires its immutable runner nonce")
        if state == "STARTING" and any(
            identity is None
            for identity in (
                source_identity,
                config_identity,
                capability_identity,
                environment_identity,
                data_identity,
                checkpoint_identity,
            )
        ):
            raise StoreInvariantError(
                "STARTING execution requires all immutable source, config, capability, "
                "environment, data, and checkpoint identities"
            )
        if state == "RUNNING":
            raise StoreInvariantError(
                "RUNNING requires transition_execution with a process receipt"
            )
        now = _now()
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO executions(
                    execution_id, campaign_id, experiment_id, launch_id, kind, tier, seed,
                    command_json, status, nonce, source_digest, config_digest, capability_digest,
                    environment_digest, data_digest, checkpoint_digest, process_record_digest,
                    process_record_json, result_digest, metadata_json, created_at, updated_at,
                    started_at, finished_at
                ) VALUES (
                    :execution_id, :campaign_id, :experiment_id, :launch_id, :kind, :tier, :seed,
                    :command_json, :status, :nonce, :source_digest, :config_digest,
                    :capability_digest, :environment_digest, :data_digest, :checkpoint_digest,
                    NULL, NULL, NULL, :metadata_json, :created_at, :updated_at, NULL, NULL
                )""",
                {
                    "execution_id": identifier,
                    "campaign_id": self.campaign_id,
                    "experiment_id": _optional_text(experiment_id, "experiment_id"),
                    "launch_id": _optional_text(launch_id, "launch_id"),
                    "kind": _text(kind, "execution kind"),
                    "tier": _text(tier, "execution tier"),
                    "seed": seed,
                    "command_json": _json(list(command), "execution command"),
                    "status": state,
                    "nonce": execution_nonce,
                    "source_digest": source_identity,
                    "config_digest": config_identity,
                    "capability_digest": capability_identity,
                    "environment_digest": environment_identity,
                    "data_digest": data_identity,
                    "checkpoint_digest": checkpoint_identity,
                    "metadata_json": _json_object(metadata, "execution metadata"),
                    "created_at": now,
                    "updated_at": now,
                },
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="execution",
                entity_id=identifier,
                from_state=None,
                to_state=state,
                reason="create execution",
                metadata={"kind": kind, "tier": tier},
            )

    def _execution_row(
        self, connection: sqlite3.Connection, execution_id: str
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """SELECT item.*,
                launch.launch_number AS projected_launch_number,
                launch.category AS projected_launch_category,
                launch.original_category AS projected_original_launch_category
            FROM executions AS item
            LEFT JOIN launches AS launch
              ON launch.campaign_id = item.campaign_id
             AND launch.launch_id = item.launch_id
             AND launch.event_seq = 1
            WHERE item.campaign_id = ? AND item.execution_id = ?""",
            (self.campaign_id, execution_id),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    @staticmethod
    def _execution_from_row(row: sqlite3.Row) -> ExecutionRecord:
        command = _load_json_string_tuple(row["command_json"], "execution command_json")
        metadata = _load_json_object(row["metadata_json"], "execution metadata_json")
        nonce = _row_optional_text(row, "nonce")
        if nonce is not None:
            try:
                nonce = _optional_nonce(nonce, "stored execution nonce")
            except StoreInvariantError as exc:
                raise StoreVersionError(str(exc)) from exc

        process_digest_raw = _row_optional_text(row, "process_record_digest")
        process_json = _row_optional_text(row, "process_record_json")
        if (process_digest_raw is None) != (process_json is None):
            raise StoreVersionError("execution process record identity is only partially stored")
        process_digest: str | None = None
        process_record: Mapping[str, object] | None = None
        if process_json is not None and process_digest_raw is not None:
            if nonce is None:
                raise StoreVersionError("execution process record has no stored nonce")
            try:
                process_digest = _digest(process_digest_raw, "stored process record digest")
                decoded = _load_json_object(process_json, "execution process_record_json")
                canonical, process_record, computed_process_digest = _validated_process_record(
                    decoded,
                    execution_id=_row_text(row, "execution_id"),
                    nonce=nonce,
                    command=command,
                    source_digest=_row_optional_text(row, "source_digest"),
                    config_digest=_row_optional_text(row, "config_digest"),
                    data_digest=_row_optional_text(row, "data_digest"),
                    checkpoint_digest=_row_optional_text(row, "checkpoint_digest"),
                )
                if canonical != process_json:
                    raise StoreVersionError("execution process record is not canonical JSON")
                if computed_process_digest != process_digest:
                    raise StoreVersionError(
                        "execution process record digest does not match its exact receipt"
                    )
            except StoreInvariantError as exc:
                raise StoreVersionError(f"invalid stored process record: {exc}") from exc

        status = _row_text(row, "status")
        if status == "STARTING" and process_record is not None:
            raise StoreVersionError("STARTING execution cannot already contain a process receipt")
        if status == "RUNNING" and process_record is None:
            raise StoreVersionError("RUNNING execution is missing its process receipt")

        created_at = _stored_timestamp(row["created_at"], "execution created_at")
        updated_at = _stored_timestamp(row["updated_at"], "execution updated_at")
        started_at = _stored_optional_timestamp(row["started_at"], "execution started_at")
        finished_at = _stored_optional_timestamp(row["finished_at"], "execution finished_at")
        if process_record is not None:
            process_started = _timestamp(
                process_record["started_at_utc"], "stored process started_at_utc"
            )
            if started_at != process_started:
                raise StoreVersionError(
                    "execution started_at does not match its process receipt timestamp"
                )
        if status in UNFINISHED_EXECUTION_STATES and finished_at is not None:
            raise StoreVersionError("unfinished execution cannot have a finished_at timestamp")

        process_id = None if process_record is None else cast(int, process_record["pid"])
        process_create_time = (
            None
            if process_record is None
            else float(cast(int | float, process_record["process_create_time"]))
        )
        process_group_id = (
            None if process_record is None else cast(int, process_record["process_group_id"])
        )
        process_command_digest = (
            None if process_record is None else cast(str, process_record["command_digest"])
        )
        process_environment_digest = (
            None if process_record is None else cast(str, process_record["environment_digest"])
        )
        return ExecutionRecord(
            execution_id=_row_text(row, "execution_id"),
            experiment_id=_row_optional_text(row, "experiment_id"),
            launch_id=_row_optional_text(row, "launch_id"),
            launch_number=_row_optional_int(row, "projected_launch_number"),
            launch_category=_row_optional_text(row, "projected_launch_category"),
            original_launch_category=_row_optional_text(row, "projected_original_launch_category"),
            kind=_row_text(row, "kind"),
            tier=_row_text(row, "tier"),
            seed=_row_optional_int(row, "seed"),
            command=command,
            status=status,
            nonce=nonce,
            source_digest=_stored_optional_digest(row["source_digest"], "execution source_digest"),
            config_digest=_stored_optional_digest(row["config_digest"], "execution config_digest"),
            capability_digest=_stored_optional_digest(
                row["capability_digest"], "execution capability_digest"
            ),
            environment_digest=_stored_optional_digest(
                row["environment_digest"], "execution environment_digest"
            ),
            data_digest=_stored_optional_digest(row["data_digest"], "execution data_digest"),
            checkpoint_digest=_stored_optional_digest(
                row["checkpoint_digest"], "execution checkpoint_digest"
            ),
            process_record_digest=process_digest,
            process_record=process_record,
            process_id=process_id,
            process_create_time=process_create_time,
            process_group_id=process_group_id,
            process_command_digest=process_command_digest,
            process_environment_digest=process_environment_digest,
            result_digest=_stored_optional_digest(row["result_digest"], "execution result_digest"),
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            finished_at=finished_at,
            metadata=metadata,
        )

    def executions(self) -> tuple[ExecutionRecord, ...]:
        """Return all executions from one stable snapshot in deterministic creation order."""

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT item.*,
                    launch.launch_number AS projected_launch_number,
                    launch.category AS projected_launch_category,
                    launch.original_category AS projected_original_launch_category
                FROM executions AS item
                LEFT JOIN launches AS launch
                  ON launch.campaign_id = item.campaign_id
                 AND launch.launch_id = item.launch_id
                 AND launch.event_seq = 1
                WHERE item.campaign_id = ?
                ORDER BY item.created_at, item.execution_id""",
                (self.campaign_id,),
            ).fetchall()
            return tuple(self._execution_from_row(row) for row in rows)

    def execution(self, execution_id: str) -> ExecutionRecord | None:
        """Return one exact execution projection, or ``None`` when it is absent."""

        identifier = _text(execution_id, "execution_id")
        with self._transaction(write=False) as connection:
            row = self._execution_row(connection, identifier)
            return None if row is None else self._execution_from_row(row)

    def artifacts_for(
        self, *, owner_type: str, owner_id: str
    ) -> tuple[tuple[str, ArtifactSpec], ...]:
        """Return immutable artifact links for one owner in deterministic role order."""

        owner_kind = _text(owner_type, "artifact owner_type")
        identifier = _text(owner_id, "artifact owner_id")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT link.role, artifact.*
                FROM artifact_links AS link
                JOIN artifacts AS artifact ON artifact.digest = link.artifact_digest
                WHERE link.campaign_id = ? AND link.owner_type = ? AND link.owner_id = ?
                ORDER BY link.role""",
                (self.campaign_id, owner_kind, identifier),
            ).fetchall()
            return tuple(
                (
                    _row_text(row, "role"),
                    ArtifactSpec(
                        digest=_stored_digest(row["digest"], "artifact digest"),
                        kind=_row_text(row, "kind"),
                        relative_path=_row_text(row, "relative_path"),
                        size_bytes=int(row["size_bytes"]),
                        metadata=_load_json_object(row["metadata_json"], "artifact metadata"),
                    ),
                )
                for row in rows
            )

    def unfinished_executions(self) -> tuple[ExecutionRecord, ...]:
        """Return only ``STARTING`` and ``RUNNING`` executions for restart reconciliation."""

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT item.*,
                    launch.launch_number AS projected_launch_number,
                    launch.category AS projected_launch_category,
                    launch.original_category AS projected_original_launch_category
                FROM executions AS item
                LEFT JOIN launches AS launch
                  ON launch.campaign_id = item.campaign_id
                 AND launch.launch_id = item.launch_id
                 AND launch.event_seq = 1
                WHERE item.campaign_id = ? AND item.status IN ('STARTING', 'RUNNING')
                ORDER BY item.created_at, item.execution_id""",
                (self.campaign_id,),
            ).fetchall()
            return tuple(self._execution_from_row(row) for row in rows)

    def transition_execution(
        self,
        execution_id: str,
        *,
        from_state: str,
        to_state: str,
        expected_revision: int,
        reason: str,
        process_record_digest: str | None = None,
        process_record: Mapping[str, object] | None = None,
        result_digest: str | None = None,
        finished_at: str | None = None,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> ExecutionRecord:
        """Atomically persist a runner receipt or terminal result and append its transition.

        The supported active lifecycle is ``STARTING -> RUNNING``. Terminal outcomes may follow
        either state, covering failures before candidate release as well as completed processes.
        Exact retries are idempotent; conflicting receipt or result identities fail closed.
        """

        identifier = _text(execution_id, "execution_id")
        expected_state = _text(from_state, "execution from_state")
        desired_state = _text(to_state, "execution to_state")
        transition_reason = _text(reason, "execution transition reason")
        process_digest = _optional_digest(process_record_digest, "process_record_digest")
        result = _optional_digest(result_digest, "execution result_digest")
        finished = None if finished_at is None else _timestamp(finished_at, "execution finished_at")
        if (process_digest is None) != (process_record is None):
            raise StoreInvariantError(
                "process_record and process_record_digest must be supplied together"
            )
        if expected_state not in UNFINISHED_EXECUTION_STATES:
            raise StoreInvariantError(
                "transition_execution may only reconcile STARTING or RUNNING executions"
            )
        if desired_state == "STARTING":
            raise StoreInvariantError("execution creation is the only transition into STARTING")

        with self._transaction(write=True) as connection:
            row = self._execution_row(connection, identifier)
            if row is None:
                raise StoreInvariantError(f"unknown execution {identifier!r}")
            current = _row_text(row, "status")
            command = _load_json_string_tuple(row["command_json"], "execution command_json")
            nonce = _row_optional_text(row, "nonce")

            encoded_process: str | None = None
            if desired_state == "RUNNING":
                if expected_state != "STARTING":
                    raise StoreInvariantError("RUNNING execution must transition from STARTING")
                if process_record is None or process_digest is None or nonce is None:
                    raise StoreInvariantError(
                        "RUNNING execution requires nonce, process record, and record digest"
                    )
                encoded_process, _, computed_process_digest = _validated_process_record(
                    process_record,
                    execution_id=identifier,
                    nonce=nonce,
                    command=command,
                    source_digest=_row_optional_text(row, "source_digest"),
                    config_digest=_row_optional_text(row, "config_digest"),
                    data_digest=_row_optional_text(row, "data_digest"),
                    checkpoint_digest=_row_optional_text(row, "checkpoint_digest"),
                )
                if computed_process_digest != process_digest:
                    raise StoreInvariantError(
                        "process_record_digest does not match the exact runner-v2 receipt"
                    )
                if result is not None or finished is not None:
                    raise StoreInvariantError("RUNNING transition cannot contain terminal evidence")
            elif process_record is not None or process_digest is not None:
                raise StoreInvariantError(
                    "process receipt may only be installed by the RUNNING transition"
                )
            elif finished is None:
                raise StoreInvariantError("terminal execution transition requires finished_at")

            if current == desired_state:
                if desired_state == "RUNNING":
                    if (
                        _row_optional_text(row, "process_record_digest") != process_digest
                        or _row_optional_text(row, "process_record_json") != encoded_process
                    ):
                        raise StoreInvariantError(
                            "RUNNING retry supplied different immutable process evidence"
                        )
                elif (
                    _row_optional_text(row, "result_digest") != result
                    or _row_optional_text(row, "finished_at") != finished
                ):
                    raise StoreInvariantError(
                        "terminal retry supplied different immutable result evidence"
                    )
                return self._execution_from_row(row)

            if current != expected_state:
                raise StoreInvariantError(
                    f"execution state changed: expected {expected_state}, found {current}"
                )
            self._require_revision(connection, expected_revision)
            now = _now()
            if desired_state == "RUNNING":
                assert encoded_process is not None and process_record is not None
                started = _timestamp(
                    process_record["started_at_utc"], "process record started_at_utc"
                )
                connection.execute(
                    """UPDATE executions SET status = ?, process_record_digest = ?,
                        process_record_json = ?, started_at = ?, updated_at = ?
                    WHERE campaign_id = ? AND execution_id = ?""",
                    (
                        desired_state,
                        process_digest,
                        encoded_process,
                        started,
                        now,
                        self.campaign_id,
                        identifier,
                    ),
                )
            else:
                connection.execute(
                    """UPDATE executions SET status = ?, result_digest = ?, finished_at = ?,
                        updated_at = ? WHERE campaign_id = ? AND execution_id = ?""",
                    (
                        desired_state,
                        result,
                        finished,
                        now,
                        self.campaign_id,
                        identifier,
                    ),
                )
            self._link_artifacts(
                connection,
                owner_type="execution",
                owner_id=identifier,
                artifacts=artifacts,
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="execution",
                entity_id=identifier,
                from_state=expected_state,
                to_state=desired_state,
                reason=transition_reason,
                metadata=metadata,
            )
            updated = self._execution_row(connection, identifier)
            if updated is None:
                raise StoreInvariantError("execution transition could not be reloaded")
            return self._execution_from_row(updated)

    def record_proposal(
        self,
        *,
        proposal_id: str,
        experiment_id: str,
        request_digest: str,
        response_digest: str,
        provider: str,
        expected_revision: int,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        """Append one complete proposer exchange and its content-addressed evidence."""

        identifier = _text(proposal_id, "proposal_id")
        experiment = _text(experiment_id, "proposal experiment_id")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO proposals(
                    proposal_id, campaign_id, experiment_id, request_digest, response_digest,
                    provider, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    experiment,
                    _digest(request_digest, "proposal request_digest"),
                    _digest(response_digest, "proposal response_digest"),
                    _text(provider, "proposal provider"),
                    _json_object(metadata, "proposal metadata"),
                    _now(),
                ),
            )
            self._link_artifacts(
                connection,
                owner_type="proposal",
                owner_id=identifier,
                artifacts=artifacts,
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="proposal",
                entity_id=identifier,
                from_state=None,
                to_state="RECORDED",
                reason="append complete proposer request and response evidence",
                metadata={"experiment_id": experiment, "provider": provider},
            )

    @staticmethod
    def _proposal_from_row(row: sqlite3.Row) -> ProposalRecord:
        return ProposalRecord(
            proposal_id=_stored_text(row["proposal_id"], "proposal proposal_id"),
            experiment_id=_stored_text(row["experiment_id"], "proposal experiment_id"),
            request_digest=_stored_digest(row["request_digest"], "proposal request_digest"),
            response_digest=_stored_digest(row["response_digest"], "proposal response_digest"),
            provider=_stored_text(row["provider"], "proposal provider"),
            metadata=_load_json_object(row["metadata_json"], "proposal metadata_json"),
            created_at=_stored_timestamp(row["created_at"], "proposal created_at"),
        )

    def proposal(self, proposal_id: str) -> ProposalRecord | None:
        """Return one exact proposal exchange, or ``None`` when it is absent."""

        identifier = _text(proposal_id, "proposal_id")
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """SELECT * FROM proposals
                WHERE campaign_id = ? AND proposal_id = ?""",
                (self.campaign_id, identifier),
            ).fetchone()
            return None if row is None else self._proposal_from_row(row)

    def record_source_snapshot(
        self,
        *,
        snapshot_id: str,
        experiment_id: str,
        source_digest: str,
        expected_revision: int,
        parent_source_digest: str | None = None,
        diff_digest: str | None = None,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        """Append generated-source lineage after policy validation and hashing."""

        identifier = _text(snapshot_id, "snapshot_id")
        experiment = _text(experiment_id, "source snapshot experiment_id")
        source = _digest(source_digest, "source snapshot source_digest")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO source_snapshots(
                    snapshot_id, campaign_id, experiment_id, source_digest,
                    parent_source_digest, diff_digest, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    experiment,
                    source,
                    _optional_digest(parent_source_digest, "parent_source_digest"),
                    _optional_digest(diff_digest, "diff_digest"),
                    _json_object(metadata, "source snapshot metadata"),
                    _now(),
                ),
            )
            self._link_artifacts(
                connection,
                owner_type="source_snapshot",
                owner_id=identifier,
                artifacts=artifacts,
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="source_snapshot",
                entity_id=identifier,
                from_state=None,
                to_state="RECORDED",
                reason="append generated-source snapshot and diff identity",
                metadata={"experiment_id": experiment, "source_digest": source},
            )

    @staticmethod
    def _source_snapshot_from_row(row: sqlite3.Row) -> SourceSnapshotRecord:
        return SourceSnapshotRecord(
            snapshot_id=_stored_text(row["snapshot_id"], "source snapshot snapshot_id"),
            experiment_id=_stored_text(row["experiment_id"], "source snapshot experiment_id"),
            source_digest=_stored_digest(row["source_digest"], "source snapshot source_digest"),
            parent_source_digest=_stored_optional_digest(
                row["parent_source_digest"], "source snapshot parent_source_digest"
            ),
            diff_digest=_stored_optional_digest(row["diff_digest"], "source snapshot diff_digest"),
            metadata=_load_json_object(row["metadata_json"], "source snapshot metadata_json"),
            created_at=_stored_timestamp(row["created_at"], "source snapshot created_at"),
        )

    def source_snapshot(self, snapshot_id: str) -> SourceSnapshotRecord | None:
        """Return one exact generated-source snapshot, or ``None`` when it is absent."""

        identifier = _text(snapshot_id, "snapshot_id")
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """SELECT * FROM source_snapshots
                WHERE campaign_id = ? AND snapshot_id = ?""",
                (self.campaign_id, identifier),
            ).fetchone()
            return None if row is None else self._source_snapshot_from_row(row)

    def record_metric(
        self,
        *,
        metric_id: str,
        split_role: str,
        gauc: float,
        ndcg_at_5: float,
        primary: float,
        scorer_digest: str,
        prediction_digest: str,
        expected_revision: int,
        experiment_id: str | None = None,
        execution_id: str | None = None,
        seed: int | None = None,
        primary_delta: float | None = None,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        """Append one trusted-score record; generated metrics never enter through this API."""

        identifier = _text(metric_id, "metric_id")
        values = _metrics(gauc, ndcg_at_5, primary)
        if seed is not None and (type(seed) is not int or seed < 0):
            raise StoreInvariantError("metric seed must be a non-negative integer")
        delta: float | None
        if primary_delta is None:
            delta = None
        elif isinstance(primary_delta, bool) or not isinstance(primary_delta, (int, float)):
            raise StoreInvariantError("primary_delta must be numeric")
        else:
            delta = float(primary_delta)
            if not math.isfinite(delta) or not -1.0 <= delta <= 1.0:
                raise StoreInvariantError("primary_delta must be finite in [-1, 1]")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO metrics(
                    metric_id, campaign_id, experiment_id, execution_id, split_role, seed,
                    gauc, ndcg_at_5, primary_value, primary_delta, scorer_digest,
                    prediction_digest, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    _optional_text(experiment_id, "metric experiment_id"),
                    _optional_text(execution_id, "metric execution_id"),
                    _text(split_role, "metric split_role"),
                    seed,
                    values["GAUC"],
                    values["nDCG@5"],
                    values["primary"],
                    delta,
                    _digest(scorer_digest, "metric scorer_digest"),
                    _digest(prediction_digest, "metric prediction_digest"),
                    _json_object(metadata, "metric metadata"),
                    _now(),
                ),
            )
            self._link_artifacts(
                connection,
                owner_type="metric",
                owner_id=identifier,
                artifacts=artifacts,
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="metric",
                entity_id=identifier,
                from_state=None,
                to_state="RECORDED",
                reason="append trusted scorer metrics",
                metadata={"split_role": split_role, "primary": values["primary"]},
            )

    def metric(self, metric_id: str) -> MetricRecord | None:
        """Return one trusted metric without exposing the writable connection."""

        identifier = _text(metric_id, "metric_id")
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """SELECT * FROM metrics WHERE campaign_id = ? AND metric_id = ?""",
                (self.campaign_id, identifier),
            ).fetchone()
            if row is None:
                return None
            values = _metrics(row["gauc"], row["ndcg_at_5"], row["primary_value"])
            seed = _row_optional_int(row, "seed")
            if seed is not None and seed < 0:  # pragma: no cover - strict schema owns this.
                raise StoreVersionError("stored metric seed must be non-negative")
            return MetricRecord(
                metric_id=_row_text(row, "metric_id"),
                split_role=_row_text(row, "split_role"),
                seed=seed,
                gauc=values["GAUC"],
                ndcg_at_5=values["nDCG@5"],
                primary=values["primary"],
                scorer_digest=_stored_digest(row["scorer_digest"], "metric scorer_digest"),
                prediction_digest=_stored_digest(
                    row["prediction_digest"], "metric prediction_digest"
                ),
                metadata=_load_json_object(row["metadata_json"], "metric metadata"),
            )

    def record_intervention(
        self,
        *,
        intervention_id: str,
        category: str,
        description: str,
        expected_revision: int,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        """Append one human intervention for the judge-readable audit trail."""

        identifier = _text(intervention_id, "intervention_id")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO interventions(
                    intervention_id, campaign_id, category, description, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    _text(category, "intervention category"),
                    _text(description, "intervention description"),
                    _json_object(metadata, "intervention metadata"),
                    _now(),
                ),
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="intervention",
                entity_id=identifier,
                from_state=None,
                to_state="RECORDED",
                reason="append human intervention evidence",
                metadata={"category": category},
            )

    def _ensure_artifact(self, connection: sqlite3.Connection, artifact: ArtifactSpec) -> None:
        existing = connection.execute(
            "SELECT * FROM artifacts WHERE digest = ?", (artifact.digest,)
        ).fetchone()
        encoded = _json_object(artifact.metadata, "artifact metadata")
        if existing is not None:
            observed = (
                _row_text(existing, "kind"),
                _row_text(existing, "relative_path"),
                int(existing["size_bytes"]),
                _row_text(existing, "metadata_json"),
            )
            expected = (artifact.kind, artifact.relative_path, artifact.size_bytes, encoded)
            if observed != expected:
                raise StoreInvariantError(
                    "artifact digest already has different immutable metadata"
                )
            return
        connection.execute(
            """INSERT INTO artifacts(
                digest, campaign_id, kind, relative_path, size_bytes, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact.digest,
                self.campaign_id,
                artifact.kind,
                artifact.relative_path,
                artifact.size_bytes,
                encoded,
                _now(),
            ),
        )

    def _link_artifacts(
        self,
        connection: sqlite3.Connection,
        *,
        owner_type: str,
        owner_id: str,
        artifacts: Sequence[tuple[str, ArtifactSpec]],
    ) -> None:
        for role, artifact in artifacts:
            self._ensure_artifact(connection, artifact)
            connection.execute(
                """INSERT INTO artifact_links(
                    campaign_id, artifact_digest, owner_type, owner_id, role, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    self.campaign_id,
                    artifact.digest,
                    _text(owner_type, "artifact owner_type"),
                    _text(owner_id, "artifact owner_id"),
                    _text(role, "artifact role"),
                    _now(),
                ),
            )

    def transition_entity(
        self,
        entity_type: EntityType,
        entity_id: str,
        *,
        from_state: str,
        to_state: str,
        expected_revision: int,
        reason: str,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        """Atomically update status, register artifacts/links, and append a transition."""

        identifier = _text(entity_id, "entity_id")
        expected_state = _text(from_state, "from_state")
        desired_state = _text(to_state, "to_state")
        table_map = {
            "campaign": ("campaigns", "campaign_id"),
            "experiment": ("experiments", "experiment_id"),
            "execution": ("executions", "execution_id"),
        }
        if entity_type not in table_map:
            raise StoreInvariantError(f"unsupported transition entity_type {entity_type!r}")
        if entity_type == "execution" and (
            expected_state in UNFINISHED_EXECUTION_STATES
            or desired_state in UNFINISHED_EXECUTION_STATES
        ):
            raise StoreInvariantError(
                "active execution lifecycle must use transition_execution "
                "to preserve resume evidence"
            )
        table, id_column = table_map[entity_type]
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            row = connection.execute(
                f"SELECT status FROM {table} WHERE {id_column} = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise StoreInvariantError(f"unknown {entity_type} {identifier!r}")
            actual_state = _row_text(row, "status")
            if actual_state != expected_state:
                raise StoreInvariantError(
                    f"{entity_type} state changed: expected {expected_state}, found {actual_state}"
                )
            connection.execute(
                f"UPDATE {table} SET status = ? WHERE {id_column} = ?",
                (desired_state, identifier),
            )
            self._link_artifacts(
                connection,
                owner_type=entity_type,
                owner_id=identifier,
                artifacts=artifacts,
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type=entity_type,
                entity_id=identifier,
                from_state=expected_state,
                to_state=desired_state,
                reason=reason,
                metadata=metadata,
            )

    def add_artifact_reference(
        self,
        artifact: ArtifactSpec,
        *,
        owner_type: str,
        owner_id: str,
        role: str,
        expected_revision: int,
    ) -> int:
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            self._link_artifacts(
                connection,
                owner_type=owner_type,
                owner_id=owner_id,
                artifacts=((role, artifact),),
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="artifact_link",
                entity_id=artifact.digest,
                from_state=None,
                to_state="LINKED",
                reason="link validated content-addressed artifact",
                metadata={"owner_type": owner_type, "owner_id": owner_id, "role": role},
            )

    def record_failure(
        self,
        *,
        failure_id: str,
        category: str,
        fingerprint: str,
        retry_ordinal: int,
        expected_revision: int,
        experiment_id: str | None = None,
        execution_id: str | None = None,
        traceback_digest: str | None = None,
        repair_action: str | None = None,
        recovery_outcome: str | None = None,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        if type(retry_ordinal) is not int or retry_ordinal < 0:
            raise StoreInvariantError("retry_ordinal must be a non-negative integer")
        identifier = _text(failure_id, "failure_id")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO failures(
                    failure_id, campaign_id, experiment_id, execution_id, category, fingerprint,
                    traceback_digest, repair_action, recovery_outcome, retry_ordinal,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    _optional_text(experiment_id, "failure experiment_id"),
                    _optional_text(execution_id, "failure execution_id"),
                    _text(category, "failure category"),
                    _text(fingerprint, "failure fingerprint"),
                    _optional_digest(traceback_digest, "traceback_digest"),
                    _optional_text(repair_action, "repair_action"),
                    _optional_text(recovery_outcome, "recovery_outcome"),
                    retry_ordinal,
                    _json_object(metadata, "failure metadata"),
                    _now(),
                ),
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="failure",
                entity_id=identifier,
                from_state=None,
                to_state="RECORDED",
                reason="persist failure before recovery decision",
                metadata={"category": category, "fingerprint": fingerprint},
            )

    def failure(self, failure_id: str) -> FailureRecord | None:
        """Return one recorded failure without exposing private SQL to adapters."""

        identifier = _text(failure_id, "failure_id")
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """SELECT * FROM failures WHERE campaign_id = ? AND failure_id = ?""",
                (self.campaign_id, identifier),
            ).fetchone()
            if row is None:
                return None
            retry_ordinal = int(row["retry_ordinal"])
            if retry_ordinal < 0:  # pragma: no cover - strict schema owns this.
                raise StoreVersionError("stored failure retry_ordinal must be non-negative")
            return FailureRecord(
                failure_id=_row_text(row, "failure_id"),
                category=_row_text(row, "category"),
                fingerprint=_row_text(row, "fingerprint"),
                retry_ordinal=retry_ordinal,
                metadata=_load_json_object(row["metadata_json"], "failure metadata"),
            )

    @staticmethod
    def _incumbent_from_row(row: sqlite3.Row) -> IncumbentRecord:
        return IncumbentRecord(
            incumbent_id=_row_text(row, "incumbent_id"),
            experiment_id=_row_optional_text(row, "experiment_id"),
            eligibility=_row_text(row, "eligibility"),
            source_digest=_row_text(row, "source_digest"),
            checkpoint_digest=_row_text(row, "checkpoint_digest"),
            artifact_closure_digest=_row_text(row, "artifact_closure_digest"),
            replay_verified=bool(row["replay_verified"]),
            outer_primary_mean=cast(float | None, row["outer_primary_mean"]),
            is_fallback=bool(row["is_fallback"]),
            revision=int(row["revision"]),
        )

    def current_incumbent(self) -> IncumbentRecord | None:
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """SELECT * FROM incumbent_history WHERE campaign_id = ?
                ORDER BY history_id DESC LIMIT 1""",
                (self.campaign_id,),
            ).fetchone()
            return None if row is None else self._incumbent_from_row(row)

    def record_incumbent(
        self,
        *,
        incumbent_id: str,
        eligibility: str,
        source_digest: str,
        checkpoint_digest: str,
        artifact_closure_digest: str,
        replay_verified: bool,
        is_fallback: bool,
        expected_revision: int,
        reason: str,
        experiment_id: str | None = None,
        outer_primary_mean: float | None = None,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> IncumbentRecord:
        identifier = _text(incumbent_id, "incumbent_id")
        if type(replay_verified) is not bool or not replay_verified:
            raise StoreInvariantError("an incumbent must already be replay verified")
        if type(is_fallback) is not bool:
            raise StoreInvariantError("is_fallback must be boolean")
        if outer_primary_mean is not None and (
            isinstance(outer_primary_mean, bool)
            or not isinstance(outer_primary_mean, (int, float))
            or not math.isfinite(float(outer_primary_mean))
            or not 0.0 <= float(outer_primary_mean) <= 1.0
        ):
            raise StoreInvariantError("outer_primary_mean must be finite in [0, 1]")
        source = _digest(source_digest, "incumbent source_digest")
        checkpoint = _digest(checkpoint_digest, "incumbent checkpoint_digest")
        closure = _digest(artifact_closure_digest, "incumbent artifact_closure_digest")
        with self._transaction(write=True) as connection:
            existing = connection.execute(
                "SELECT * FROM incumbent_history WHERE incumbent_id = ?", (identifier,)
            ).fetchone()
            if existing is not None:
                record = self._incumbent_from_row(existing)
                if (
                    record.eligibility != eligibility
                    or record.source_digest != source
                    or record.checkpoint_digest != checkpoint
                    or record.artifact_closure_digest != closure
                    or record.is_fallback != is_fallback
                ):
                    raise StoreInvariantError(
                        "incumbent_id already names different immutable evidence"
                    )
                return record
            self._require_revision(connection, expected_revision)
            previous = connection.execute(
                "SELECT COUNT(*) FROM incumbent_history WHERE campaign_id = ?",
                (self.campaign_id,),
            ).fetchone()[0]
            if is_fallback and int(previous) != 0:
                raise StoreInvariantError("the immutable fallback must be the first incumbent")
            if not is_fallback and int(previous) == 0:
                raise StoreInvariantError("a qualified fallback must exist before any challenger")
            next_revision = expected_revision + 1
            connection.execute(
                """INSERT INTO incumbent_history(
                    incumbent_id, campaign_id, revision, experiment_id, eligibility,
                    source_digest, checkpoint_digest, artifact_closure_digest, replay_verified,
                    outer_primary_mean, is_fallback, reason, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    next_revision,
                    _optional_text(experiment_id, "incumbent experiment_id"),
                    _text(eligibility, "incumbent eligibility"),
                    source,
                    checkpoint,
                    closure,
                    None if outer_primary_mean is None else float(outer_primary_mean),
                    int(is_fallback),
                    _text(reason, "incumbent reason"),
                    _json_object(metadata, "incumbent metadata"),
                    _now(),
                ),
            )
            self._link_artifacts(
                connection,
                owner_type="incumbent",
                owner_id=identifier,
                artifacts=artifacts,
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="incumbent",
                entity_id=identifier,
                from_state=None,
                to_state=eligibility,
                reason=reason,
                metadata={"is_fallback": is_fallback, "replay_verified": True},
            )
            row = connection.execute(
                "SELECT * FROM incumbent_history WHERE incumbent_id = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise StoreInvariantError("incumbent history insert disappeared")
            return self._incumbent_from_row(row)

    def record_runtime_sample(
        self,
        *,
        family: str,
        elapsed_seconds: float,
        peak_rss_bytes: int,
        disk_bytes: int,
        outcome: str,
        expected_revision: int,
        execution_id: str | None = None,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)):
            raise StoreInvariantError("elapsed_seconds must be numeric")
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise StoreInvariantError("elapsed_seconds must be finite and non-negative")
        for name, value in (("peak_rss_bytes", peak_rss_bytes), ("disk_bytes", disk_bytes)):
            if type(value) is not int or value < 0:
                raise StoreInvariantError(f"{name} must be a non-negative integer")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            cursor = connection.execute(
                """INSERT INTO runtime_samples(
                    campaign_id, execution_id, family, elapsed_seconds, peak_rss_bytes,
                    disk_bytes, outcome, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.campaign_id,
                    _optional_text(execution_id, "runtime execution_id"),
                    _text(family, "runtime family"),
                    elapsed,
                    peak_rss_bytes,
                    disk_bytes,
                    _text(outcome, "runtime outcome"),
                    _json_object(metadata, "runtime metadata"),
                    _now(),
                ),
            )
            if cursor.lastrowid is None:
                raise StoreInvariantError("runtime sample insert did not return an identity")
            sample_id = cursor.lastrowid
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="runtime_sample",
                entity_id=str(sample_id),
                from_state=None,
                to_state="RECORDED",
                reason="append runtime sample",
                metadata={"family": family, "outcome": outcome},
            )
            return sample_id

    def record_reallocation(
        self,
        *,
        reallocation_id: str,
        from_category: str,
        to_category: str,
        launch_count: int,
        reason: str,
        expected_revision: int,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> int:
        if type(launch_count) is not int or launch_count < 1:
            raise StoreInvariantError("reallocation launch_count must be a positive integer")
        identifier = _text(reallocation_id, "reallocation_id")
        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            connection.execute(
                """INSERT INTO reallocations(
                    reallocation_id, campaign_id, from_category, to_category, launch_count,
                    reason, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    self.campaign_id,
                    _text(from_category, "from_category"),
                    _text(to_category, "to_category"),
                    launch_count,
                    _text(reason, "reallocation reason"),
                    _json_object(metadata, "reallocation metadata"),
                    _now(),
                ),
            )
            return self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="reallocation",
                entity_id=identifier,
                from_state=from_category,
                to_state=to_category,
                reason=reason,
                metadata={"launch_count": launch_count},
            )

    def reallocations(self) -> tuple[ReallocationRecord, ...]:
        """Return the append-only category transfers in their durable insertion order."""

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT * FROM reallocations WHERE campaign_id = ?
                ORDER BY rowid""",
                (self.campaign_id,),
            ).fetchall()
            return tuple(
                ReallocationRecord(
                    reallocation_id=_row_text(row, "reallocation_id"),
                    from_category=_row_text(row, "from_category"),
                    to_category=_row_text(row, "to_category"),
                    launch_count=int(row["launch_count"]),
                    reason=_row_text(row, "reason"),
                    metadata=_load_json_object(row["metadata_json"], "reallocation metadata"),
                )
                for row in rows
            )

    def _local_query_by_candidate(
        self, connection: sqlite3.Connection, candidate_fingerprint: str
    ) -> sqlite3.Row | None:
        reservation = connection.execute(
            """SELECT query_id FROM validation_queries
            WHERE campaign_id = ? AND candidate_fingerprint = ? AND event_seq = 1""",
            (self.campaign_id, candidate_fingerprint),
        ).fetchone()
        if reservation is None:
            return None
        row = connection.execute(
            """SELECT * FROM validation_queries WHERE campaign_id = ? AND query_id = ?
            ORDER BY event_seq DESC LIMIT 1""",
            (self.campaign_id, _row_text(reservation, "query_id")),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    def _local_query_reservation_by_id(
        self, connection: sqlite3.Connection, query_id: str
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """SELECT * FROM validation_queries
            WHERE campaign_id = ? AND query_id = ? AND event_seq = 1""",
            (self.campaign_id, query_id),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    def _local_query_latest_by_id(
        self, connection: sqlite3.Connection, query_id: str
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """SELECT * FROM validation_queries
            WHERE campaign_id = ? AND query_id = ? ORDER BY event_seq DESC LIMIT 1""",
            (self.campaign_id, query_id),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    @staticmethod
    def _require_query_identity(
        reservation: sqlite3.Row,
        *,
        query_id: str,
        candidate_fingerprint: str,
        scorer_digest: str,
        experiment_id: str | None,
        metadata_json: str | None = None,
    ) -> None:
        mismatch = (
            _row_text(reservation, "query_id") != query_id
            or _row_text(reservation, "candidate_fingerprint") != candidate_fingerprint
            or _row_text(reservation, "scorer_digest") != scorer_digest
            or _row_optional_text(reservation, "experiment_id") != experiment_id
        )
        if metadata_json is not None:
            mismatch = mismatch or _row_text(reservation, "metadata_json") != metadata_json
        if mismatch:
            raise StoreInvariantError("query identity already names different immutable evidence")

    @staticmethod
    def _query_from_row(row: sqlite3.Row) -> ValidationQueryRecord:
        metrics_raw = row["metrics_json"]
        metrics = None if metrics_raw is None else _load_json_object(metrics_raw, "query metrics")
        return ValidationQueryRecord(
            query_id=_row_text(row, "query_id"),
            candidate_fingerprint=_row_text(row, "candidate_fingerprint"),
            state=_row_text(row, "state"),
            event_seq=int(row["event_seq"]),
            result_digest=_row_optional_text(row, "result_digest"),
            metrics=metrics,
        )

    def reserve_public_query(
        self,
        ledger: OuterQueryLedger,
        *,
        query_id: str,
        candidate_fingerprint: str,
        scorer_digest: str,
        expected_revision: int,
        experiment_id: str | None = None,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> ValidationQueryRecord:
        """Reserve project-wide first, then append the campaign-local query event."""

        identifier = _text(query_id, "query_id")
        candidate = _digest(candidate_fingerprint, "candidate_fingerprint")
        scorer = _digest(scorer_digest, "scorer_digest")
        experiment = _optional_text(experiment_id, "query experiment_id")
        metadata_json = _json_object(metadata, "query metadata")

        # Fail obvious local conflicts before consuming a project-wide slot.  A concurrent local
        # change after this read can still leave a conservative project reservation, by design.
        with self._transaction(write=False) as connection:
            campaign = self._require_revision(connection, expected_revision)
            existing_id = self._local_query_reservation_by_id(connection, identifier)
            if existing_id is not None:
                self._require_query_identity(
                    existing_id,
                    query_id=identifier,
                    candidate_fingerprint=candidate,
                    scorer_digest=scorer,
                    experiment_id=experiment,
                    metadata_json=metadata_json,
                )
            existing = self._local_query_by_candidate(connection, candidate)
            if existing is not None:
                reservation = self._local_query_reservation_by_id(connection, identifier)
                if reservation is None:
                    raise StoreInvariantError("local validation reservation disappeared")
                self._require_query_identity(
                    reservation,
                    query_id=identifier,
                    candidate_fingerprint=candidate,
                    scorer_digest=scorer,
                    experiment_id=experiment,
                    metadata_json=metadata_json,
                )
                return self._query_from_row(existing)
            used = int(
                connection.execute(
                    """SELECT COUNT(*) FROM validation_queries
                    WHERE campaign_id = ? AND event_seq = 1""",
                    (self.campaign_id,),
                ).fetchone()[0]
            )
            if used >= int(campaign["outer_query_limit"]):
                raise OuterQueryLimitError("campaign public-validation limit is exhausted")
            benchmark = _row_text(campaign, "benchmark_digest")
            dataset = _row_text(campaign, "dataset_digest")

        ledger.reserve(
            query_id=identifier,
            campaign_id=self.campaign_id,
            benchmark_digest=benchmark,
            dataset_digest=dataset,
            scorer_digest=scorer,
            candidate_fingerprint=candidate,
            metadata=metadata,
        )

        with self._transaction(write=True) as connection:
            campaign = self._require_revision(connection, expected_revision)
            existing_id = self._local_query_reservation_by_id(connection, identifier)
            if existing_id is not None:
                self._require_query_identity(
                    existing_id,
                    query_id=identifier,
                    candidate_fingerprint=candidate,
                    scorer_digest=scorer,
                    experiment_id=experiment,
                    metadata_json=metadata_json,
                )
            existing = self._local_query_by_candidate(connection, candidate)
            if existing is not None:
                reservation = self._local_query_reservation_by_id(connection, identifier)
                if reservation is None:
                    raise StoreInvariantError("local validation reservation disappeared")
                self._require_query_identity(
                    reservation,
                    query_id=identifier,
                    candidate_fingerprint=candidate,
                    scorer_digest=scorer,
                    experiment_id=experiment,
                    metadata_json=metadata_json,
                )
                return self._query_from_row(existing)
            used = int(
                connection.execute(
                    """SELECT COUNT(*) FROM validation_queries
                    WHERE campaign_id = ? AND event_seq = 1""",
                    (self.campaign_id,),
                ).fetchone()[0]
            )
            if used >= int(campaign["outer_query_limit"]):
                raise OuterQueryLimitError("campaign public-validation limit is exhausted")
            connection.execute(
                """INSERT INTO validation_queries(
                    campaign_id, query_id, event_seq, experiment_id, candidate_fingerprint,
                    benchmark_digest, dataset_digest, scorer_digest, state, result_digest,
                    metrics_json, metadata_json, created_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, 'RESERVED', NULL, NULL, ?, ?)""",
                (
                    self.campaign_id,
                    identifier,
                    experiment,
                    candidate,
                    benchmark,
                    dataset,
                    scorer,
                    metadata_json,
                    _now(),
                ),
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="validation_query",
                entity_id=identifier,
                from_state=None,
                to_state="RESERVED",
                reason="reserve public-validation query before scoring",
                metadata={"candidate_fingerprint": candidate},
            )
            latest = self._local_query_by_candidate(connection, candidate)
            if latest is None:
                raise StoreInvariantError("local validation reservation disappeared")
            return self._query_from_row(latest)

    def complete_public_query(
        self,
        ledger: OuterQueryLedger,
        *,
        query_id: str,
        result_digest: str,
        gauc: float,
        ndcg_at_5: float,
        primary: float,
        prediction_digest: str,
        scorer_digest: str,
        expected_revision: int,
        metric_id: str | None = None,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> ValidationQueryRecord:
        identifier = _text(query_id, "query_id")
        result = _digest(result_digest, "query result_digest")
        prediction = _digest(prediction_digest, "query prediction_digest")
        scorer = _digest(scorer_digest, "query scorer_digest")
        metric_values = _metrics(gauc, ndcg_at_5, primary)
        metrics_json = _json_object(metric_values, "query metrics")
        completion_metadata_json = _json_object(metadata, "query completion metadata")
        resolved_metric_id = (
            f"outer:{identifier}" if metric_id is None else _text(metric_id, "metric_id")
        )

        completed_record: ValidationQueryRecord | None = None
        with self._transaction(write=False) as connection:
            self._require_revision(connection, expected_revision)
            reservation = self._local_query_reservation_by_id(connection, identifier)
            if reservation is None:
                raise StoreInvariantError("cannot complete an unreserved local validation query")
            if _row_text(reservation, "scorer_digest") != scorer:
                raise StoreInvariantError("query completion scorer differs from its reservation")
            latest = self._local_query_latest_by_id(connection, identifier)
            if latest is None:
                raise StoreInvariantError("validation query lost its latest event")
            if _row_text(latest, "state") == "COMPLETED":
                if (
                    _row_optional_text(latest, "result_digest") != result
                    or _row_optional_text(latest, "metrics_json") != metrics_json
                    or _row_text(latest, "metadata_json") != completion_metadata_json
                ):
                    raise StoreInvariantError("query was already completed with another result")
                metric = connection.execute(
                    """SELECT scorer_digest, prediction_digest FROM metrics
                    WHERE metric_id = ? AND campaign_id = ?""",
                    (resolved_metric_id, self.campaign_id),
                ).fetchone()
                if metric is None or (
                    _row_text(metric, "scorer_digest") != scorer
                    or _row_text(metric, "prediction_digest") != prediction
                ):
                    raise StoreInvariantError(
                        "query completion retry differs from its trusted metric evidence"
                    )
                completed_record = self._query_from_row(latest)

        # Complete the project ledger before committing local metrics.  A crash after this call
        # retains the conservative global charge and a retry can finish the local transaction.
        ledger.complete(identifier, result_digest=result, metadata=metadata)
        if completed_record is not None:
            return completed_record

        with self._transaction(write=True) as connection:
            self._require_revision(connection, expected_revision)
            reservation = self._local_query_reservation_by_id(connection, identifier)
            if reservation is None:
                raise StoreInvariantError("cannot complete an unreserved local validation query")
            if _row_text(reservation, "scorer_digest") != scorer:
                raise StoreInvariantError("query completion scorer differs from its reservation")
            latest = self._local_query_latest_by_id(connection, identifier)
            if latest is None:
                raise StoreInvariantError("validation query lost its latest event")
            if _row_text(latest, "state") == "COMPLETED":
                if (
                    _row_optional_text(latest, "result_digest") != result
                    or _row_optional_text(latest, "metrics_json") != metrics_json
                ):
                    raise StoreInvariantError("query was already completed with another result")
                return self._query_from_row(latest)
            connection.execute(
                """INSERT INTO validation_queries(
                    campaign_id, query_id, event_seq, experiment_id, candidate_fingerprint,
                    benchmark_digest, dataset_digest, scorer_digest, state, result_digest,
                    metrics_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', ?, ?, ?, ?)""",
                (
                    self.campaign_id,
                    identifier,
                    int(latest["event_seq"]) + 1,
                    _row_optional_text(reservation, "experiment_id"),
                    _row_text(reservation, "candidate_fingerprint"),
                    _row_text(reservation, "benchmark_digest"),
                    _row_text(reservation, "dataset_digest"),
                    scorer,
                    result,
                    metrics_json,
                    completion_metadata_json,
                    _now(),
                ),
            )
            connection.execute(
                """INSERT INTO metrics(
                    metric_id, campaign_id, experiment_id, execution_id, split_role, seed,
                    gauc, ndcg_at_5, primary_value, primary_delta, scorer_digest,
                    prediction_digest, metadata_json, created_at
                ) VALUES (?, ?, ?, NULL, 'outer_valid', NULL, ?, ?, ?, NULL, ?, ?, ?, ?)""",
                (
                    resolved_metric_id,
                    self.campaign_id,
                    _row_optional_text(reservation, "experiment_id"),
                    metric_values["GAUC"],
                    metric_values["nDCG@5"],
                    metric_values["primary"],
                    scorer,
                    prediction,
                    _json_object(metadata, "outer metric metadata"),
                    _now(),
                ),
            )
            self._advance_revision(
                connection,
                expected_revision=expected_revision,
                entity_type="validation_query",
                entity_id=identifier,
                from_state=_row_text(latest, "state"),
                to_state="COMPLETED",
                reason="persist exact protected public-validation result",
                metadata={"result_digest": result, "metric_id": resolved_metric_id},
            )
            updated = connection.execute(
                """SELECT * FROM validation_queries WHERE campaign_id = ? AND query_id = ?
                ORDER BY event_seq DESC LIMIT 1""",
                (self.campaign_id, identifier),
            ).fetchone()
            if updated is None:
                raise StoreInvariantError("completed validation query disappeared")
            return self._query_from_row(updated)


def _metrics(gauc: object, ndcg_at_5: object, primary: object) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, raw in (("GAUC", gauc), ("nDCG@5", ndcg_at_5), ("primary", primary)):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise StoreInvariantError(f"{name} must be numeric")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise StoreInvariantError(f"{name} must be finite in [0, 1]")
        output[name] = value
    expected = (output["GAUC"] + output["nDCG@5"]) / 2.0
    if not math.isclose(output["primary"], expected, rel_tol=0.0, abs_tol=1e-12):
        raise StoreInvariantError("primary must equal the mean of GAUC and nDCG@5")
    return output


class OuterQueryLedger:
    """Project-wide append-only public-validation reservation ledger."""

    def __init__(self, path: Path, connection: sqlite3.Connection, *, read_only: bool) -> None:
        self.path = path
        self._connection = connection
        self._read_only = read_only
        values = _meta(connection, "ledger_meta")
        try:
            self.max_queries = int(values["max_queries"])
        except (KeyError, ValueError) as exc:
            raise StoreVersionError("ledger max_queries metadata is invalid") from exc
        if not 1 <= self.max_queries <= DEFAULT_OUTER_QUERY_LIMIT:
            raise StoreVersionError("ledger max_queries is outside the supported range")

    @classmethod
    def create(cls, path: str | Path, *, max_queries: int = DEFAULT_OUTER_QUERY_LIMIT) -> Self:
        if type(max_queries) is not int or not 1 <= max_queries <= DEFAULT_OUTER_QUERY_LIMIT:
            raise StoreInvariantError("project ledger max_queries must be an integer in [1, 6]")
        database = Path(path).absolute()
        _claim_database(database)
        connection: sqlite3.Connection | None = None
        try:
            connection = _connect(database, read_only=False)
            _initialize_schema(
                connection,
                statements=_LEDGER_SCHEMA_STATEMENTS,
                meta_table="ledger_meta",
                kind="kuairand-outer-query-ledger",
                schema_digest=_LEDGER_SCHEMA_DIGEST,
            )
            now = _now()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO ledger_meta(key, value) VALUES ('max_queries', ?)",
                    (str(max_queries),),
                )
                connection.execute(
                    "INSERT INTO ledger_state(singleton, revision, updated_at) VALUES (1, 0, ?)",
                    (now,),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            os.chmod(database, 0o600)
            return cls(database, connection, read_only=False)
        except BaseException:
            if connection is not None:
                connection.close()
            _cleanup_claimed_database(database)
            raise

    @classmethod
    def open(cls, path: str | Path, *, read_only: bool = False) -> Self:
        database = Path(path).absolute()
        if database.is_symlink() or not database.is_file():
            raise CampaignNotFoundError(f"project outer-query ledger does not exist: {database}")
        connection = _connect(database, read_only=read_only)
        try:
            _verify_schema(
                connection,
                meta_table="ledger_meta",
                kind="kuairand-outer-query-ledger",
                schema_digest=_LEDGER_SCHEMA_DIGEST,
                expected_tables=_LEDGER_TABLES,
            )
            return cls(database, connection, read_only=read_only)
        except BaseException:
            connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        if write and self._read_only:
            raise StoreInvariantError("read-only outer-query ledger cannot be mutated")
        self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield self._connection
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def _revision(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT revision FROM ledger_state WHERE singleton = 1").fetchone()
        if row is None:
            raise StoreVersionError("outer-query ledger state is missing")
        return int(row["revision"])

    def _advance(self, connection: sqlite3.Connection) -> int:
        revision = self._revision(connection) + 1
        connection.execute(
            "UPDATE ledger_state SET revision = ?, updated_at = ? WHERE singleton = 1",
            (revision, _now()),
        )
        return revision

    def reserve(
        self,
        *,
        query_id: str,
        campaign_id: str,
        benchmark_digest: str,
        dataset_digest: str,
        scorer_digest: str,
        candidate_fingerprint: str,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> ValidationQueryRecord:
        identifier = _text(query_id, "query_id")
        campaign = _text(campaign_id, "campaign_id")
        benchmark = _digest(benchmark_digest, "benchmark_digest")
        dataset = _digest(dataset_digest, "dataset_digest")
        scorer = _digest(scorer_digest, "scorer_digest")
        candidate = _digest(candidate_fingerprint, "candidate_fingerprint")
        metadata_json = _json_object(metadata, "outer-query metadata")
        with self._transaction(write=True) as connection:
            reservation_by_id = connection.execute(
                "SELECT * FROM outer_queries WHERE query_id = ? AND event_seq = 1",
                (identifier,),
            ).fetchone()
            if reservation_by_id is not None and (
                _row_text(reservation_by_id, "campaign_id") != campaign
                or _row_text(reservation_by_id, "benchmark_digest") != benchmark
                or _row_text(reservation_by_id, "dataset_digest") != dataset
                or _row_text(reservation_by_id, "scorer_digest") != scorer
                or _row_text(reservation_by_id, "candidate_fingerprint") != candidate
                or _row_text(reservation_by_id, "metadata_json") != metadata_json
            ):
                raise StoreInvariantError(
                    "query_id already names different project-wide immutable evidence"
                )
            reservation = connection.execute(
                """SELECT * FROM outer_queries
                WHERE benchmark_digest = ? AND dataset_digest = ? AND scorer_digest = ?
                    AND candidate_fingerprint = ? AND event_seq = 1""",
                (benchmark, dataset, scorer, candidate),
            ).fetchone()
            if reservation is not None:
                if (
                    _row_text(reservation, "query_id") != identifier
                    or _row_text(reservation, "campaign_id") != campaign
                    or _row_text(reservation, "metadata_json") != metadata_json
                ):
                    raise StoreInvariantError(
                        "candidate fingerprint already has a project-wide query reservation"
                    )
                latest = connection.execute(
                    """SELECT * FROM outer_queries
                    WHERE query_id = ? ORDER BY event_seq DESC LIMIT 1""",
                    (identifier,),
                ).fetchone()
                if latest is None:
                    raise StoreInvariantError("outer-query reservation lost its latest event")
                return self._record_from_row(latest)
            used = int(
                connection.execute(
                    "SELECT COUNT(*) FROM outer_queries WHERE event_seq = 1"
                ).fetchone()[0]
            )
            if used >= self.max_queries:
                raise OuterQueryLimitError("project-wide public-validation limit is exhausted")
            connection.execute(
                """INSERT INTO outer_queries(
                    query_id, event_seq, campaign_id, benchmark_digest, dataset_digest,
                    scorer_digest, candidate_fingerprint, state, result_digest,
                    metadata_json, created_at
                ) VALUES (?, 1, ?, ?, ?, ?, ?, 'RESERVED', NULL, ?, ?)""",
                (
                    identifier,
                    campaign,
                    benchmark,
                    dataset,
                    scorer,
                    candidate,
                    metadata_json,
                    _now(),
                ),
            )
            self._advance(connection)
            row = connection.execute(
                "SELECT * FROM outer_queries WHERE query_id = ? AND event_seq = 1", (identifier,)
            ).fetchone()
            if row is None:
                raise StoreInvariantError("outer-query reservation disappeared")
            return self._record_from_row(row)

    def complete(
        self,
        query_id: str,
        *,
        result_digest: str,
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> ValidationQueryRecord:
        identifier = _text(query_id, "query_id")
        result = _digest(result_digest, "outer-query result_digest")
        metadata_json = _json_object(metadata, "outer-query completion metadata")
        with self._transaction(write=True) as connection:
            reservation = connection.execute(
                "SELECT * FROM outer_queries WHERE query_id = ? AND event_seq = 1", (identifier,)
            ).fetchone()
            if reservation is None:
                raise StoreInvariantError("cannot complete an unreserved project query")
            latest = connection.execute(
                "SELECT * FROM outer_queries WHERE query_id = ? ORDER BY event_seq DESC LIMIT 1",
                (identifier,),
            ).fetchone()
            if latest is None:
                raise StoreInvariantError("project query lost its latest event")
            if _row_text(latest, "state") == "COMPLETED":
                if _row_optional_text(latest, "result_digest") != result:
                    raise StoreInvariantError("project query already completed with another result")
                return self._record_from_row(latest)
            connection.execute(
                """INSERT INTO outer_queries(
                    query_id, event_seq, campaign_id, benchmark_digest, dataset_digest,
                    scorer_digest, candidate_fingerprint, state, result_digest,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'COMPLETED', ?, ?, ?)""",
                (
                    identifier,
                    int(latest["event_seq"]) + 1,
                    _row_text(reservation, "campaign_id"),
                    _row_text(reservation, "benchmark_digest"),
                    _row_text(reservation, "dataset_digest"),
                    _row_text(reservation, "scorer_digest"),
                    _row_text(reservation, "candidate_fingerprint"),
                    result,
                    metadata_json,
                    _now(),
                ),
            )
            self._advance(connection)
            updated = connection.execute(
                "SELECT * FROM outer_queries WHERE query_id = ? ORDER BY event_seq DESC LIMIT 1",
                (identifier,),
            ).fetchone()
            if updated is None:
                raise StoreInvariantError("project query completion disappeared")
            return self._record_from_row(updated)

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> ValidationQueryRecord:
        return ValidationQueryRecord(
            query_id=_row_text(row, "query_id"),
            candidate_fingerprint=_row_text(row, "candidate_fingerprint"),
            state=_row_text(row, "state"),
            event_seq=int(row["event_seq"]),
            result_digest=_row_optional_text(row, "result_digest"),
            metrics=None,
        )

    def snapshot(
        self, *, benchmark_digest: str, dataset_digest: str, scorer_digest: str
    ) -> OuterLedgerSnapshot:
        _digest(benchmark_digest, "benchmark_digest")
        _digest(dataset_digest, "dataset_digest")
        _digest(scorer_digest, "scorer_digest")
        with self._transaction(write=False) as connection:
            used = int(
                connection.execute(
                    "SELECT COUNT(*) FROM outer_queries WHERE event_seq = 1"
                ).fetchone()[0]
            )
            return OuterLedgerSnapshot(
                revision=self._revision(connection),
                max_queries=self.max_queries,
                queries_used=used,
            )

    def projection(self) -> OuterQueryLedgerProjection:
        """Return exact reservation revisions and latest events in append order.

        Every project-ledger revision appends exactly one ``outer_queries`` event.  The strict
        equality check below therefore also detects any history/state drift before exposing the
        projection to a retrying orchestration adapter.
        """

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT * FROM outer_queries ORDER BY query_event_id"
            ).fetchall()
            revision = self._revision(connection)
            if len(rows) != revision:
                raise StoreVersionError(
                    "outer-query event count differs from the durable ledger revision"
                )
            reservations: dict[str, tuple[int, sqlite3.Row]] = {}
            latest: dict[str, sqlite3.Row] = {}
            order: list[str] = []
            for exact_revision, row in enumerate(rows, start=1):
                identifier = _row_text(row, "query_id")
                event_seq = int(row["event_seq"])
                if event_seq == 1:
                    if identifier in reservations:
                        raise StoreVersionError("project query contains duplicate reservations")
                    reservations[identifier] = (exact_revision, row)
                    order.append(identifier)
                elif identifier not in reservations:
                    raise StoreVersionError("project query event precedes its reservation")
                previous = latest.get(identifier)
                if previous is not None and event_seq != int(previous["event_seq"]) + 1:
                    raise StoreVersionError("project query event sequence is not contiguous")
                latest[identifier] = row

            projected: list[OuterQueryProjectionRecord] = []
            for identifier in order:
                reservation_revision, reservation = reservations[identifier]
                current = latest[identifier]
                projected.append(
                    OuterQueryProjectionRecord(
                        query_id=identifier,
                        campaign_id=_row_text(reservation, "campaign_id"),
                        benchmark_digest=_stored_digest(
                            reservation["benchmark_digest"],
                            "outer-query benchmark_digest",
                        ),
                        dataset_digest=_stored_digest(
                            reservation["dataset_digest"],
                            "outer-query dataset_digest",
                        ),
                        scorer_digest=_stored_digest(
                            reservation["scorer_digest"],
                            "outer-query scorer_digest",
                        ),
                        candidate_fingerprint=_stored_digest(
                            reservation["candidate_fingerprint"],
                            "outer-query candidate_fingerprint",
                        ),
                        reservation_revision=reservation_revision,
                        state=_row_text(current, "state"),
                        event_seq=int(current["event_seq"]),
                        result_digest=_stored_optional_digest(
                            current["result_digest"], "outer-query result_digest"
                        ),
                        reservation_metadata=_load_json_object(
                            reservation["metadata_json"],
                            "outer-query reservation metadata",
                        ),
                        latest_metadata=_load_json_object(
                            current["metadata_json"],
                            "outer-query latest metadata",
                        ),
                    )
                )
            return OuterQueryLedgerProjection(
                revision=revision,
                max_queries=self.max_queries,
                queries=tuple(projected),
            )


@dataclass(frozen=True, slots=True)
class LineageFoldMetrics:
    """One immutable (GAUC, nDCG@5, primary) triple for a single inner fold."""

    gauc: float
    ndcg_at_5: float
    primary: float

    def __post_init__(self) -> None:
        for name in ("gauc", "ndcg_at_5", "primary"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise StoreInvariantError(f"lineage fold metric {name} must be a real number")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise StoreInvariantError(f"lineage fold metric {name} must be finite in [0, 1]")


@dataclass(frozen=True, slots=True)
class LineageEventRecord:
    """One immutable project-wide research-lineage event."""

    event_id: int
    campaign_id: str
    outcome: Literal["rejected", "admitted"]
    candidate_id: str
    proposal_family: str
    proposal_signature: str | None
    repairs_attempted: int | None
    root_failure_fingerprint: str | None
    root_failure_category: str | None
    root_failure_code: str | None
    root_failure_subject: str | None
    terminal_failure_fingerprint: str | None
    terminal_failure_category: str | None
    terminal_failure_code: str | None
    terminal_failure_subject: str | None
    diagnostic: str | None
    inner_fold_a: LineageFoldMetrics | None
    inner_fold_b: LineageFoldMetrics | None
    parent_fold_a_primary: float | None
    parent_fold_b_primary: float | None
    promoted: bool | None
    created_at: str


@dataclass(frozen=True, slots=True)
class ResearchLineageSummary:
    """Bounded, typed cross-run evidence for one exact benchmark/starter/source identity.

    Scoping by all three digests together means a change to the organizer benchmark contract, the
    starter kit, or (most importantly) this repository's own trusted controller source
    automatically starts a clean slate: stale evidence from a since-fixed bug can never keep
    blocking a corrected agent.
    """

    root_failure_totals: Mapping[str, int]
    proposal_family_rejection_totals: Mapping[str, int]
    proposal_family_admission_totals: Mapping[str, int]
    recent_events: tuple[LineageEventRecord, ...]


def _lineage_event_record(row: sqlite3.Row) -> LineageEventRecord:
    """Build one typed lineage event from its stored row."""

    return LineageEventRecord(
        event_id=int(row["event_id"]),
        campaign_id=_row_text(row, "campaign_id"),
        outcome=cast(Literal["rejected", "admitted"], _row_text(row, "outcome")),
        candidate_id=_row_text(row, "candidate_id"),
        proposal_family=_row_text(row, "proposal_family"),
        proposal_signature=row["proposal_signature"],
        repairs_attempted=row["repairs_attempted"],
        root_failure_fingerprint=row["root_failure_fingerprint"],
        root_failure_category=row["root_failure_category"],
        root_failure_code=row["root_failure_code"],
        root_failure_subject=row["root_failure_subject"],
        terminal_failure_fingerprint=row["terminal_failure_fingerprint"],
        terminal_failure_category=row["terminal_failure_category"],
        terminal_failure_code=row["terminal_failure_code"],
        terminal_failure_subject=row["terminal_failure_subject"],
        diagnostic=row["diagnostic"],
        inner_fold_a=(
            None
            if row["inner_fold_a_primary"] is None
            else LineageFoldMetrics(
                gauc=row["inner_fold_a_gauc"],
                ndcg_at_5=row["inner_fold_a_ndcg_at_5"],
                primary=row["inner_fold_a_primary"],
            )
        ),
        inner_fold_b=(
            None
            if row["inner_fold_b_primary"] is None
            else LineageFoldMetrics(
                gauc=row["inner_fold_b_gauc"],
                ndcg_at_5=row["inner_fold_b_ndcg_at_5"],
                primary=row["inner_fold_b_primary"],
            )
        ),
        parent_fold_a_primary=row["parent_fold_a_primary"],
        parent_fold_b_primary=row["parent_fold_b_primary"],
        promoted=(None if row["promoted"] is None else bool(row["promoted"])),
        created_at=_row_text(row, "created_at"),
    )


class ResearchLineageLedger:
    """Project-wide append-only ledger of generated-candidate proposal outcomes.

    Unlike :class:`OuterQueryLedger`, this ledger is advisory rather than a hard safety limit: it
    lets a later campaign start with typed knowledge of which proposal families and root failures
    an earlier campaign already exhausted against the exact same benchmark, starter kit, and
    trusted controller source identity, instead of rediscovering the same failure from scratch.
    """

    def __init__(self, path: Path, connection: sqlite3.Connection, *, read_only: bool) -> None:
        self.path = path
        self._connection = connection
        self._read_only = read_only

    @classmethod
    def create(cls, path: str | Path) -> Self:
        database = Path(path).absolute()
        _claim_database(database)
        connection: sqlite3.Connection | None = None
        try:
            connection = _connect(database, read_only=False)
            _initialize_schema(
                connection,
                statements=_LINEAGE_SCHEMA_STATEMENTS,
                meta_table="ledger_meta",
                kind="kuairand-research-lineage-ledger",
                schema_digest=_LINEAGE_SCHEMA_DIGEST,
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO ledger_state(singleton, revision, updated_at) VALUES (1, 0, ?)",
                    (_now(),),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            os.chmod(database, 0o600)
            return cls(database, connection, read_only=False)
        except BaseException:
            if connection is not None:
                connection.close()
            _cleanup_claimed_database(database)
            raise

    @classmethod
    def open(cls, path: str | Path, *, read_only: bool = False) -> Self:
        database = Path(path).absolute()
        if database.is_symlink() or not database.is_file():
            raise CampaignNotFoundError(
                f"project research-lineage ledger does not exist: {database}"
            )
        connection = _connect(database, read_only=read_only)
        try:
            _verify_schema(
                connection,
                meta_table="ledger_meta",
                kind="kuairand-research-lineage-ledger",
                schema_digest=_LINEAGE_SCHEMA_DIGEST,
                expected_tables=_LINEAGE_TABLES,
            )
            return cls(database, connection, read_only=read_only)
        except BaseException:
            connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        if write and self._read_only:
            raise StoreInvariantError("read-only research-lineage ledger cannot be mutated")
        self._connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield self._connection
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def _advance(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE ledger_state SET revision = revision + 1, updated_at = ? WHERE singleton = 1",
            (_now(),),
        )

    def record_rejection(
        self,
        *,
        campaign_id: str,
        benchmark_digest: str,
        starter_digest: str,
        source_digest: str,
        candidate_id: str,
        proposal_family: str,
        proposal_signature: str | None,
        repairs_attempted: int,
        root_failure_fingerprint: str,
        root_failure_category: str,
        root_failure_code: str,
        root_failure_subject: str,
        terminal_failure_fingerprint: str,
        terminal_failure_category: str,
        terminal_failure_code: str,
        terminal_failure_subject: str,
        diagnostic: str,
    ) -> None:
        """Append one pre-admission rejection observed by a live research campaign."""

        campaign = _text(campaign_id, "campaign_id")
        benchmark = _digest(benchmark_digest, "benchmark_digest")
        starter = _digest(starter_digest, "starter_digest")
        source = _digest(source_digest, "source_digest")
        candidate = _text(candidate_id, "candidate_id")
        family = _text(proposal_family, "proposal_family")
        signature = _optional_digest(proposal_signature, "proposal_signature")
        if type(repairs_attempted) is not int or repairs_attempted < 0:
            raise StoreInvariantError("repairs_attempted must be a non-negative integer")
        root_fingerprint = _digest(root_failure_fingerprint, "root_failure_fingerprint")
        terminal_fingerprint = _digest(terminal_failure_fingerprint, "terminal_failure_fingerprint")
        bounded_diagnostic = _text(diagnostic, "diagnostic")[:2000]
        metadata_json = _json_object({}, "lineage-event metadata")
        with self._transaction(write=True) as connection:
            connection.execute(
                """INSERT INTO lineage_events(
                    campaign_id, benchmark_digest, starter_digest, source_digest, outcome,
                    candidate_id, proposal_family, proposal_signature, repairs_attempted,
                    root_failure_fingerprint, root_failure_category, root_failure_code,
                    root_failure_subject, terminal_failure_fingerprint, terminal_failure_category,
                    terminal_failure_code, terminal_failure_subject, diagnostic, metadata_json,
                    created_at
                ) VALUES (?, ?, ?, ?, 'rejected', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    campaign,
                    benchmark,
                    starter,
                    source,
                    candidate,
                    family,
                    signature,
                    repairs_attempted,
                    root_fingerprint,
                    _text(root_failure_category, "root_failure_category"),
                    _text(root_failure_code, "root_failure_code"),
                    _text(root_failure_subject, "root_failure_subject"),
                    terminal_fingerprint,
                    _text(terminal_failure_category, "terminal_failure_category"),
                    _text(terminal_failure_code, "terminal_failure_code"),
                    _text(terminal_failure_subject, "terminal_failure_subject"),
                    bounded_diagnostic,
                    metadata_json,
                    _now(),
                ),
            )
            self._advance(connection)

    def record_admission(
        self,
        *,
        campaign_id: str,
        benchmark_digest: str,
        starter_digest: str,
        source_digest: str,
        candidate_id: str,
        proposal_family: str,
        proposal_signature: str | None,
        inner_fold_a: LineageFoldMetrics | None = None,
        inner_fold_b: LineageFoldMetrics | None = None,
        parent_fold_a_primary: float | None = None,
        parent_fold_b_primary: float | None = None,
        promoted: bool | None = None,
    ) -> None:
        """Append one successful admission observed by a live research campaign.

        Fold evidence is optional and each fold travels independently: a candidate that fails the
        Fold B screen never reaches Fold A, but its real Fold B result is still worth a future
        campaign knowing, so pass ``inner_fold_b``/``parent_fold_b_primary`` alone in that case.
        ``promoted`` requires both folds, since promotion is impossible without a Fold A
        confirmation. Call with no evidence at all right after materialization succeeds if
        training has not run yet; a corrected caller may still choose to record a second, richer
        event once evidence exists rather than mutate this one, since the table is append-only.
        """

        campaign = _text(campaign_id, "campaign_id")
        benchmark = _digest(benchmark_digest, "benchmark_digest")
        starter = _digest(starter_digest, "starter_digest")
        source = _digest(source_digest, "source_digest")
        candidate = _text(candidate_id, "candidate_id")
        family = _text(proposal_family, "proposal_family")
        signature = _optional_digest(proposal_signature, "proposal_signature")
        if (inner_fold_a is None) != (parent_fold_a_primary is None):
            raise StoreInvariantError("inner_fold_a and parent_fold_a_primary travel together")
        if (inner_fold_b is None) != (parent_fold_b_primary is None):
            raise StoreInvariantError("inner_fold_b and parent_fold_b_primary travel together")
        if promoted is not None and (inner_fold_a is None or inner_fold_b is None):
            raise StoreInvariantError("promoted requires both Fold A and Fold B evidence")
        for name, value in (
            ("parent_fold_a_primary", parent_fold_a_primary),
            ("parent_fold_b_primary", parent_fold_b_primary),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise StoreInvariantError(f"{name} must be finite in [0, 1]")
        if promoted is not None and type(promoted) is not bool:
            raise StoreInvariantError("promoted must be boolean")
        metadata_json = _json_object({}, "lineage-event metadata")
        with self._transaction(write=True) as connection:
            connection.execute(
                """INSERT INTO lineage_events(
                    campaign_id, benchmark_digest, starter_digest, source_digest, outcome,
                    candidate_id, proposal_family, proposal_signature,
                    inner_fold_a_gauc, inner_fold_a_ndcg_at_5, inner_fold_a_primary,
                    inner_fold_b_gauc, inner_fold_b_ndcg_at_5, inner_fold_b_primary,
                    parent_fold_a_primary, parent_fold_b_primary, promoted,
                    metadata_json, created_at
                ) VALUES (
                    ?, ?, ?, ?, 'admitted', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    campaign,
                    benchmark,
                    starter,
                    source,
                    candidate,
                    family,
                    signature,
                    None if inner_fold_a is None else inner_fold_a.gauc,
                    None if inner_fold_a is None else inner_fold_a.ndcg_at_5,
                    None if inner_fold_a is None else inner_fold_a.primary,
                    None if inner_fold_b is None else inner_fold_b.gauc,
                    None if inner_fold_b is None else inner_fold_b.ndcg_at_5,
                    None if inner_fold_b is None else inner_fold_b.primary,
                    parent_fold_a_primary,
                    parent_fold_b_primary,
                    None if promoted is None else int(promoted),
                    metadata_json,
                    _now(),
                ),
            )
            self._advance(connection)

    def summary(
        self,
        *,
        benchmark_digest: str,
        starter_digest: str,
        source_digest: str,
        limit: int = 20,
    ) -> ResearchLineageSummary:
        """Return bounded typed evidence for one exact benchmark/starter/source identity."""

        benchmark = _digest(benchmark_digest, "benchmark_digest")
        starter = _digest(starter_digest, "starter_digest")
        source = _digest(source_digest, "source_digest")
        if type(limit) is not int or not 1 <= limit <= 200:
            raise StoreInvariantError(
                "research-lineage summary limit must be an integer in [1, 200]"
            )
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """SELECT * FROM lineage_events
                WHERE benchmark_digest = ? AND starter_digest = ? AND source_digest = ?
                ORDER BY event_id ASC""",
                (benchmark, starter, source),
            ).fetchall()
        root_failure_totals: dict[str, int] = {}
        family_rejections: dict[str, int] = {}
        family_admissions: dict[str, int] = {}
        events: list[LineageEventRecord] = []
        for row in rows:
            outcome = _row_text(row, "outcome")
            family = _row_text(row, "proposal_family")
            if outcome == "rejected":
                fingerprint = _row_text(row, "root_failure_fingerprint")
                root_failure_totals[fingerprint] = root_failure_totals.get(fingerprint, 0) + 1
                family_rejections[family] = family_rejections.get(family, 0) + 1
            else:
                family_admissions[family] = family_admissions.get(family, 0) + 1
            events.append(_lineage_event_record(row))
        return ResearchLineageSummary(
            root_failure_totals=MappingProxyType(root_failure_totals),
            proposal_family_rejection_totals=MappingProxyType(family_rejections),
            proposal_family_admission_totals=MappingProxyType(family_admissions),
            recent_events=tuple(events[-limit:]),
        )

    def events_for_campaign(self, campaign_id: str) -> tuple[LineageEventRecord, ...]:
        """Return one campaign's own events in durable order, across every identity scope.

        Reporting reads a single completed campaign, so unlike :meth:`summary` this is scoped by
        campaign rather than by benchmark/starter/source identity: a run's own log must stay
        readable after a later code change moves the active lineage scope.
        """

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT * FROM lineage_events WHERE campaign_id = ? ORDER BY event_id ASC",
                (_text(campaign_id, "campaign_id"),),
            ).fetchall()
        return tuple(_lineage_event_record(row) for row in rows)
