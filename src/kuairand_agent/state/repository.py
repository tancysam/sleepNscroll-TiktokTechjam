"""The sole durable write interface for autonomous laboratory truth.

``StateRepository`` is intentionally a deep module: callers submit typed records and state
transitions while SQLite details, sequence allocation, idempotency, fencing, and event journaling
remain local to this implementation.  The legacy campaign stores are deliberately not imported or
dual-written.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Protocol, Self, cast, runtime_checkable

from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import (
    BundleId,
    CampaignId,
    ContractId,
    DecisionId,
    FamilyId,
    PredictionId,
    Sha256Id,
    canonical_json_bytes,
    canonical_json_sha256,
)
from kuairand_agent.finalization.organizer_check import (
    OrganizerCheckError,
    validate_organizer_check_manifest,
)
from kuairand_agent.finalization.replay_grades import (
    BundleRegenerationEvidence,
    ReplayGradeError,
    ReplayGradeReceipt,
    combine_replay_grade_receipts,
    validate_bundle_regeneration_evidence_manifest,
    validate_replay_grade_receipt_manifest,
)
from kuairand_agent.observability.receipts import ReceiptError, ScriptedReplayReceipt
from kuairand_agent.state.schema import configure_connection, migrate

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_STATEFUL_TABLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "campaign": "campaigns",
        "trial": "trials",
        "attempt": "attempts",
        "replay": "replays",
        "bundle": "bundles",
        "provider_operation": "provider_operations",
    }
)
_LEASE_RESOURCE_TABLES: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "family": ("families", "family_id"),
        "experiment": ("campaign_experiments", "experiment_id"),
        "trial": ("trials", "trial_id"),
        "attempt": ("attempts", "attempt_id"),
        "prediction": ("predictions", "prediction_id"),
        "replay": ("replays", "replay_id"),
        "bundle": ("bundles", "bundle_id"),
        "provider_operation": ("provider_operations", "operation_id"),
        "protected_query_reservation": (
            "protected_query_reservations",
            "reservation_id",
        ),
    }
)
_ATOMIC_FINAL_STATES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "campaign": frozenset({"COMPLETED", "COMPLETED_OFFLINE_FIXTURE"}),
        "replay": frozenset({"VERIFIED"}),
        "bundle": frozenset({"SEALED"}),
    }
)
_TERMINAL_PROJECTION_SCHEMA_VERSION: Final = 1
_TERMINAL_REDACTION_POLICY_VERSION: Final = 1
_SELF_REFERENCE_MARKER: Final = "excluded-self-reference"
_PUBLICATION_FIELDS: Final = frozenset(
    {
        "bundle_id",
        "bundle_sha256",
        "manifest_sha256",
        "inventory_sha256",
        "bundle_path",
        "published_path",
        "evidence_manifest_path",
        "publication",
    }
)
_BUNDLE_EVIDENCE_MEMBERS: Final = frozenset(
    {
        "contract-manifest.json",
        "campaign-manifest.json",
        "campaign-state-snapshot.sqlite3",
        "event-export.jsonl",
        "selection-evidence.json",
        "scientific-decision.json",
        "submission-decision.json",
        "replay-receipt.json",
        "resource-receipts.jsonl",
        "protected-query-accounting.json",
        "provider-accounting.json",
        "failure-summary.json",
        "submission.csv",
        "report.md",
    }
)
_BUNDLE_MEMBERS: Final = _BUNDLE_EVIDENCE_MEMBERS | frozenset(
    {"bundle-manifest.json", "bundle.sha256"}
)
_BUNDLE_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_version",
        "contract_id",
        "campaign_id",
        "selected_prediction_id",
        "terminal_projection",
        "identity",
        "submission_sha256",
        "replay_receipt_sha256",
        "required_paths",
        "evidence",
    }
)
_FROZEN_FILE_RECEIPT_DOMAIN: Final = b"kuairand-frozen-bundle-file-v1\0"
_OFFLINE_FIXTURE_TERMINAL_STATE: Final = "COMPLETED_OFFLINE_FIXTURE"
_OFFLINE_FIXTURE_REPLAY_INPUT_FIELDS: Final = frozenset(
    {
        "contract_id",
        "campaign_id",
        "prediction_id",
        "replay_grades",
        "scripted_replay_receipt",
    }
)
_OFFLINE_FIXTURE_REPLAY_JOURNAL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "verifier",
        "grade",
        "qualification_scope",
        "scripted_replay_receipt",
    }
)
_SCRIPTED_REPLAY_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "contract_id",
        "campaign_id",
        "prediction_id",
        "grade",
        "qualification_scope",
        "first_prediction_sha256",
        "replay_prediction_sha256",
        "first_result_sha256",
        "replay_result_sha256",
        "exact_prediction_bytes",
        "exact_metrics_recomputed",
        "protected_metrics_evaluated",
        "official_fm_qualified",
        "full_data_qualified",
        "receipt_id",
    }
)
_OFFLINE_FIXTURE_BUNDLE_CLAIM_FIELDS: Final = frozenset(
    {
        "resource_receipt_id",
        "replay_grade",
        "replay_grades",
        "submission_disposition",
        "scientific_disposition",
        "campaign_kind",
        "qualification_scope",
        "protected_query_count",
        "exact_metrics",
    }
)
_OFFLINE_FIXTURE_REPLAY_GRADES: Final = (
    ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
    ReplayGrade.BUNDLE_EXACT.value,
)
_OFFLINE_FIXTURE_RESOURCE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "contract_id",
        "campaign_id",
        "prediction_id",
        "attempt_id",
        "campaign_kind",
        "qualification_scope",
        "declared_resource_profile",
        "actual_trainer_backend",
        "actual_trainer_device",
        "observed_resources",
        "timing",
        "preferred_backend_qualified",
        "official_fm_qualified",
        "full_data_qualified",
    }
)
_SCRIPTED_RESULT_RESOURCE_FIELDS: Final = frozenset(
    {
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "peak_disk_bytes",
        "peak_process_count",
        "threads",
        "device",
        "cpu_seconds_measured",
    }
)
_SCRIPTED_TIMING_FIELDS: Final = frozenset(
    {"started_monotonic_ns", "ended_monotonic_ns", "wall_seconds"}
)
_PRODUCTION_TERMINAL_STATE: Final = "COMPLETED"
_PRODUCTION_REPLAY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "contract_id",
        "campaign_id",
        "prediction_id",
        "qualification_manifest_digest",
        "qualification_fallback_digest",
        "original_prediction_sha256",
        "replay_prediction_sha256",
        "row_alignment_sha256",
        "submission_sha256",
        "organizer_check_sha256",
        "organizer_check",
        "clean_replay_evidence_sha256",
        "clean_replay_evidence",
        "scoring_exact_receipt",
        "same_backend_receipt",
        "achieved_replay_grades",
        "required_terminal_replay_grades",
        "qualification_scope",
        "official_fm_qualified",
        "full_data_qualified",
        "final_period_outcomes_accessed",
    }
)
_PRODUCTION_BUNDLE_CLAIM_FIELDS: Final = frozenset(
    {
        "schema_version",
        "resource_receipt_id",
        "prepublication_replay_grades",
        "required_replay_grades",
        "bundle_exact_status",
        "submission_disposition",
        "scientific_disposition",
        "campaign_kind",
        "qualification_scope",
        "protected_query_count",
        "provider_operation_count",
        "exact_metrics",
        "official_fm_qualified",
        "full_data_qualified",
        "final_period_outcomes_accessed",
        "qualification_manifest_digest",
    }
)
_PRODUCTION_PREPUBLICATION_REPLAY_GRADES: Final = (
    ReplayGrade.SCORING_EXACT.value,
    ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
)
_PRODUCTION_REQUIRED_REPLAY_GRADES: Final = (
    ReplayGrade.SCORING_EXACT.value,
    ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
    ReplayGrade.BUNDLE_EXACT.value,
)
_POSTPUBLICATION_RESOURCE_SCOPE: Final = "THROUGH_BUNDLE_PUBLICATION_EXCLUDING_TERMINAL_COMMIT"
_POSTPUBLICATION_RESOURCE_FIELDS: Final = frozenset(
    {
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "disk_bytes",
        "device",
    }
)
_PRODUCTION_RESOURCE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "contract_id",
        "campaign_id",
        "prediction_id",
        "campaign_kind",
        "qualification_scope",
        "qualification_manifest_digest",
        "qualification_fallback_digest",
        "performance_profile_digest",
        "declared_resource_profile",
        "actual_trainer_backend",
        "actual_trainer_device",
        "qualified_training_resources",
        "controller_resources",
        "controller_resource_scope",
        "preferred_backend_qualified",
        "official_fm_qualified",
        "full_data_qualified",
        "final_period_outcomes_accessed",
    }
)
_PRODUCTION_QUALIFIED_RESOURCE_FIELDS: Final = frozenset(
    {"wall_seconds", "cpu_seconds", "peak_rss_bytes", "disk_bytes", "device"}
)
_PRODUCTION_CONTROLLER_RESOURCE_FIELDS: Final = frozenset(
    {"wall_seconds", "cpu_seconds", "peak_rss_bytes", "disk_bytes", "device"}
)


@runtime_checkable
class ValueIdentifier(Protocol):
    """Structural support for domain IDs backed by a string ``value``."""

    @property
    def value(self) -> str: ...


type IdInput = str | Sha256Id | ValueIdentifier
type CampaignIdInput = str | CampaignId | ValueIdentifier
type ContractIdInput = str | ContractId | ValueIdentifier
type FamilyIdInput = str | FamilyId | ValueIdentifier
type PredictionIdInput = str | PredictionId | ValueIdentifier
type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
type Clock = Callable[[], datetime]


class StateError(RuntimeError):
    """Base error for the single state authority."""


class StateInvariantError(StateError):
    """Raised before an operation could violate durable state."""


class StateConflictError(StateError):
    """Raised when compare-and-swap observes stale caller state."""


class StateNotFoundError(StateError):
    """Raised when a requested authoritative record does not exist."""


class ProtectedBudgetExhaustedError(StateError):
    """Raised before a protected query could exceed its contract lineage budget."""


class ProtectedOutcomeTerminalError(StateError):
    """Raised when code attempts to replace a RESULT or UNKNOWN protected outcome."""


class LeaseConflictError(StateError):
    """Raised when another non-stale fenced worker owns a resource."""


class TerminalPreparationConflictError(StateConflictError):
    """Raised when one source horizon already binds a different terminal intent."""

    code = "TERMINAL_PREPARATION_CONFLICT"


class PreparedSourceStaleError(StateConflictError):
    """Raised when authority events advanced after a terminal projection was prepared."""

    code = "PREPARED_SOURCE_STALE"


class PublishedBundleVerificationError(StateInvariantError):
    """Raised when a purported atomically published bundle fails exact verification."""

    code = "PUBLISHED_BUNDLE_INVALID"


class PublishedBundleConflictError(StateConflictError):
    """Raised when a preparation was consumed by a different published bundle."""

    code = "PUBLISHED_BUNDLE_CONFLICT"


class PreparedFinalizationRequiredError(StateInvariantError):
    """Raised by the removed caller-supplied terminal finalization compatibility seam."""

    code = "PREPARED_FINALIZATION_REQUIRED"


class RecordKind(StrEnum):
    FAMILY = "family"
    EXPERIMENT = "experiment"
    TRIAL = "trial"
    ATTEMPT = "attempt"
    ARTIFACT = "artifact"
    PREDICTION = "prediction"
    INNER_EVALUATION = "inner_evaluation"
    PROMOTION_DECISION = "promotion_decision"
    RANK_GRAPH = "rank_graph"
    REPLAY = "replay"
    BUNDLE = "bundle"
    RESOURCE_RECEIPT = "resource_receipt"
    PROVIDER_OPERATION = "provider_operation"
    FAILURE = "failure"


_RESEARCH_RECORD_KINDS: Final = frozenset(
    {
        RecordKind.FAMILY,
        RecordKind.EXPERIMENT,
        RecordKind.TRIAL,
        RecordKind.ATTEMPT,
        RecordKind.PREDICTION,
        RecordKind.INNER_EVALUATION,
        RecordKind.RANK_GRAPH,
        RecordKind.PROVIDER_OPERATION,
    }
)


class ProtectedOutcomeKind(StrEnum):
    RESULT = "RESULT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class CampaignHandle:
    campaign_id: str
    contract_id: str
    state: str
    revision: int
    created: bool


@dataclass(frozen=True, slots=True)
class FamilyEvidenceReceipt:
    fingerprint: str
    created: bool


@dataclass(frozen=True, slots=True)
class DurableRecord:
    """One record accepted by :meth:`StateRepository.register`.

    ``references`` and ``attributes`` are checked against an exact schema for ``kind``.  This keeps
    one registration method without surrendering referential or type validation to callers.
    """

    kind: RecordKind
    record_id: IdInput
    campaign_id: IdInput
    contract_id: IdInput
    references: Mapping[str, IdInput] = field(default_factory=dict)
    attributes: Mapping[str, object] = field(default_factory=dict)
    payload: Mapping[str, object] = field(default_factory=dict)
    state: str | None = None
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class Transition:
    entity_kind: str
    entity_id: str
    prior_state: str
    new_state: str
    revision: int
    event_seq: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class ProtectedReservation:
    reservation_id: str
    campaign_id: str
    contract_id: str
    family_id: str
    prediction_id: str
    query_ordinal: int
    idempotency_key: str
    state: str
    created: bool


@dataclass(frozen=True, slots=True)
class ProtectedOutcome:
    evaluation_id: str
    reservation_id: str
    prediction_id: str
    outcome: ProtectedOutcomeKind
    metrics: Mapping[str, object] | None
    unknown_reason: str | None
    created: bool


@dataclass(frozen=True, slots=True)
class Lease:
    resource_kind: str
    resource_id: str
    owner_id: str
    fence_token: int
    expires_at: str
    completed: bool


@dataclass(frozen=True, slots=True)
class CampaignSourceRevision:
    campaign_revision: int
    last_event_seq: int


@dataclass(frozen=True, slots=True)
class TerminalPreparation:
    """Bundle-independent terminal intent used to break the content-addressing cycle."""

    decision_id: str | DecisionId | ValueIdentifier
    replay_id: IdInput
    selected_prediction_id: PredictionIdInput
    fallback_prediction_id: PredictionIdInput
    terminal_state: str
    decision_payload: Mapping[str, object] = field(default_factory=dict)
    replay_payload: Mapping[str, object] = field(default_factory=dict)
    bundle_claims: Mapping[str, object] = field(default_factory=dict)
    scoring_exact_receipt: ReplayGradeReceipt | None = None
    same_backend_receipt: ReplayGradeReceipt | None = None


@dataclass(frozen=True, slots=True)
class PreparedTerminalProjection:
    preparation_id: str
    campaign_id: str
    contract_id: str
    source: CampaignSourceRevision
    terminal_state: str
    projection_schema_version: int
    redaction_policy_version: int
    projection_sha256: str
    projection: Mapping[str, object]
    created: bool


@dataclass(frozen=True, slots=True)
class PreparedProjectionArtifacts:
    preparation_id: str
    snapshot_path: Path
    snapshot_sha256: str
    snapshot_size_bytes: int
    event_export_path: Path
    event_export_sha256: str
    event_export_size_bytes: int


@dataclass(frozen=True, slots=True)
class PublishedBundleReceipt:
    root: Path
    bundle_id: str | BundleId | ValueIdentifier
    manifest_sha256: str
    inventory_sha256: str
    submission_sha256: str
    file_count: int
    total_size_bytes: int
    regeneration_evidence: BundleRegenerationEvidence | None = None
    bundle_exact_receipt: ReplayGradeReceipt | None = None
    postpublication_resource_receipt: PostpublicationResourceReceipt | None = None


@dataclass(frozen=True, slots=True, init=False)
class PostpublicationResourceReceipt:
    """Authority-only resource evidence measured through successful bundle publication.

    The receipt is intentionally excluded from the already content-addressed bundle.  Its identity
    binds the exact prepublication authority receipt and is persisted atomically with terminal
    campaign state.
    """

    receipt_id: str
    contract_id: str
    campaign_id: str
    prediction_id: str
    prepublication_resource_receipt_id: str
    performance_profile_digest: str
    scope: str
    measurements: Mapping[str, object]

    @classmethod
    def from_measurement(
        cls,
        *,
        contract_id: ContractIdInput,
        campaign_id: CampaignIdInput,
        prediction_id: PredictionIdInput,
        prepublication_resource_receipt_id: str,
        performance_profile_digest: str,
        measurements: Mapping[str, object],
    ) -> PostpublicationResourceReceipt:
        """Authenticate one finite CPU process-tree measurement at the fixed honest scope."""

        if not isinstance(measurements, Mapping):
            raise StateInvariantError("postpublication measurements must be a mapping")
        normalized_measurements = _validate_production_resource_measurements(
            measurements,
            fields=_POSTPUBLICATION_RESOURCE_FIELDS,
            location="postpublication controller resources",
            peak_rss_positive=True,
        )
        contract = _identifier(contract_id, "postpublication contract_id")
        campaign = _identifier(campaign_id, "postpublication campaign_id")
        prediction = _identifier(prediction_id, "postpublication prediction_id")
        prepublication = _sha256(
            prepublication_resource_receipt_id,
            "prepublication_resource_receipt_id",
        )
        profile = _sha256(performance_profile_digest, "performance_profile_digest")
        body: dict[str, object] = {
            "schema_version": 1,
            "contract_id": contract,
            "campaign_id": campaign,
            "prediction_id": prediction,
            "prepublication_resource_receipt_id": prepublication,
            "performance_profile_digest": profile,
            "scope": _POSTPUBLICATION_RESOURCE_SCOPE,
            "measurements": normalized_measurements,
        }
        result = object.__new__(cls)
        object.__setattr__(result, "receipt_id", canonical_json_sha256(body))
        object.__setattr__(result, "contract_id", contract)
        object.__setattr__(result, "campaign_id", campaign)
        object.__setattr__(result, "prediction_id", prediction)
        object.__setattr__(result, "prepublication_resource_receipt_id", prepublication)
        object.__setattr__(result, "performance_profile_digest", profile)
        object.__setattr__(result, "scope", _POSTPUBLICATION_RESOURCE_SCOPE)
        object.__setattr__(result, "measurements", MappingProxyType(normalized_measurements))
        return result

    def manifest(self) -> dict[str, object]:
        """Return the exact immutable authority payload, including its content identity."""

        return {
            "schema_version": 1,
            "receipt_id": self.receipt_id,
            "contract_id": self.contract_id,
            "campaign_id": self.campaign_id,
            "prediction_id": self.prediction_id,
            "prepublication_resource_receipt_id": self.prepublication_resource_receipt_id,
            "performance_profile_digest": self.performance_profile_digest,
            "scope": self.scope,
            "measurements": dict(self.measurements),
        }


@dataclass(frozen=True, slots=True)
class Reconciliation:
    interrupted_attempt_ids: tuple[str, ...]


class StateRepository:
    """SQLite adapter at the sole durable-state seam.

    Every public mutation opens one ``BEGIN IMMEDIATE`` transaction.  The class owns no persistent
    connection, so it is safe to construct before forking and straightforward to use from recovery
    tools.  SQLite serializes writers and WAL permits simultaneous read-only inspection.
    """

    def __init__(self, database_path: Path, *, clock: Clock | None = None) -> None:
        self._database_path = database_path
        self._clock = clock or _utc_now

    @classmethod
    def open(cls, state_root: Path, *, clock: Clock | None = None) -> Self:
        """Open or explicitly migrate ``<state-root>/authority.sqlite3``."""

        if not isinstance(state_root, Path):
            raise StateInvariantError("state_root must be a pathlib.Path")
        state_root.mkdir(parents=True, exist_ok=True)
        database_path = state_root / "authority.sqlite3"
        repository = cls(database_path, clock=clock)
        with repository._connect() as connection:
            migrate(connection, created_at=repository._timestamp())
        return repository

    @property
    def database_path(self) -> Path:
        return self._database_path

    def create_campaign(
        self,
        *,
        campaign_id: CampaignIdInput,
        contract_id: ContractIdInput,
        contract_manifest: Mapping[str, object],
        config: Mapping[str, object],
        idempotency_key: str,
        protected_query_limit: int,
        initial_state: str = "READY",
    ) -> CampaignHandle:
        """Create a campaign and its first event, or return its exact idempotent replay."""

        campaign = _identifier(campaign_id, "campaign_id")
        contract = _identifier(contract_id, "contract_id")
        key = _text(idempotency_key, "idempotency_key")
        state = _text(initial_state, "initial_state")
        if state in _ATOMIC_FINAL_STATES["campaign"]:
            raise StateInvariantError("campaign cannot be created in an atomic final state")
        if type(protected_query_limit) is not int or protected_query_limit < 0:
            raise StateInvariantError("protected_query_limit must be a non-negative integer")
        manifest_json = _canonical_json(contract_manifest, "contract_manifest")
        config_json = _canonical_json(config, "config")
        with self._transaction() as connection:
            now = self._timestamp()
            existing_contract = connection.execute(
                "SELECT manifest_json, protected_query_limit FROM contracts WHERE contract_id = ?",
                (contract,),
            ).fetchone()
            if existing_contract is None:
                connection.execute(
                    """
                    INSERT INTO contracts(
                        contract_id, manifest_json, protected_query_limit, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (contract, manifest_json, protected_query_limit, now),
                )
            elif (str(existing_contract[0]), int(existing_contract[1])) != (
                manifest_json,
                protected_query_limit,
            ):
                raise StateInvariantError("ContractId already binds different immutable content")
            existing_rows = connection.execute(
                """
                SELECT campaign_id, contract_id, idempotency_key, config_json, state, revision,
                       protected_query_limit
                FROM campaigns WHERE idempotency_key = ? OR campaign_id = ?
                """,
                (key, campaign),
            ).fetchall()
            if len(existing_rows) > 1:
                raise StateInvariantError(
                    "campaign idempotency key and CampaignId identify different campaigns"
                )
            if existing_rows:
                existing = existing_rows[0]
                if (
                    str(existing[0]),
                    str(existing[1]),
                    str(existing[2]),
                    str(existing[3]),
                    int(existing[6]),
                ) != (campaign, contract, key, config_json, protected_query_limit):
                    raise StateInvariantError(
                        "campaign idempotency key or CampaignId already binds different content"
                    )
                creation = connection.execute(
                    """
                    SELECT new_state FROM campaign_events
                    WHERE campaign_id = ? AND event_seq = 1
                          AND event_type = 'campaign_created'
                    """,
                    (campaign,),
                ).fetchone()
                if creation is None or str(creation[0]) != state:
                    raise StateInvariantError(
                        "campaign creation state differs from idempotent replay"
                    )
                return CampaignHandle(
                    campaign_id=campaign,
                    contract_id=contract,
                    state=str(existing[4]),
                    revision=int(existing[5]),
                    created=False,
                )
            connection.execute(
                """
                INSERT INTO campaigns(
                    campaign_id, contract_id, idempotency_key, config_json, state, revision,
                    next_event_seq, protected_query_limit, terminal, started_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, 1, ?, 0, ?, ?)
                """,
                (campaign, contract, key, config_json, state, protected_query_limit, now, now),
            )
            event_seq = self._append_event(
                connection,
                campaign_id=campaign,
                event_type="campaign_created",
                entity_kind="campaign",
                entity_id=campaign,
                prior_state=None,
                new_state=state,
                revision=0,
                payload_json=config_json,
                now=now,
            )
            if event_seq != 1:
                raise StateInvariantError("new campaign did not allocate event sequence one")
            return CampaignHandle(campaign, contract, state, 0, True)

    def register(self, record: DurableRecord, *, lease: Lease | None = None) -> bool:
        """Register one immutable or initial stateful record.

        Returns ``True`` for the first insert and ``False`` for an exact idempotent replay.  A
        repeated identity with different content is rejected.
        """

        if not isinstance(record, DurableRecord):
            raise StateInvariantError("record must be DurableRecord")
        normalized = _normalize_record(record)
        with self._transaction() as connection:
            now = self._timestamp()
            resource_kind, resource_id = _registration_fence_scope(normalized)
            campaign_row = self._assert_campaign_lineage(
                connection,
                campaign_id=normalized.campaign_id,
                contract_id=normalized.contract_id,
                writable=False,
            )
            existing_record = self._record_exists(connection, normalized)
            if bool(campaign_row["terminal"]):
                if not existing_record:
                    raise StateInvariantError("terminal campaign is immutable")
                inserted = self._insert_record(connection, normalized, now=now)
                if inserted:  # pragma: no cover - existence was established above
                    raise StateInvariantError("terminal campaign accepted a new record")
                return False
            if bool(campaign_row["research_frozen"]) and normalized.kind in _RESEARCH_RECORD_KINDS:
                if not existing_record:
                    raise StateInvariantError(
                        "research is permanently frozen after the first protected outcome"
                    )
                inserted = self._insert_record(connection, normalized, now=now)
                if inserted:  # pragma: no cover - existence was established above
                    raise StateInvariantError("research freeze accepted a new record")
                return False
            self._assert_write_fence(
                connection,
                campaign_id=normalized.campaign_id,
                lease=lease,
                now=now,
                resource_kind=resource_kind,
                resource_id=resource_id,
            )
            self._assert_record_references(connection, normalized)
            if normalized.kind is RecordKind.FAMILY:
                self._ensure_family_ledger(connection, normalized, now=now)
            elif normalized.kind is RecordKind.EXPERIMENT:
                self._ensure_experiment_ledger(connection, normalized, now=now)
            inserted = self._insert_record(connection, normalized, now=now)
            if inserted:
                revision = 0
                self._append_event(
                    connection,
                    campaign_id=normalized.campaign_id,
                    event_type=f"{normalized.kind.value}_registered",
                    entity_kind=normalized.kind.value,
                    entity_id=normalized.record_id,
                    prior_state=None,
                    new_state=normalized.state,
                    revision=revision,
                    payload_json=_registration_payload_json(normalized),
                    now=now,
                )
            return inserted

    def record_family_evidence(
        self,
        *,
        contract_id: ContractIdInput,
        campaign_id: CampaignIdInput,
        family_id: FamilyIdInput,
        representation: str,
        model_family: str,
        objective: str,
        temporal_policy: str,
        fusion_member: str,
        result: str,
        lease: Lease | None = None,
    ) -> FamilyEvidenceReceipt:
        """Append one replay-safe scientific branch result to the cross-campaign ledger."""

        contract = _identifier(contract_id, "contract_id")
        campaign = _identifier(campaign_id, "campaign_id")
        family = _identifier(family_id, "family_id")
        parts = tuple(
            _text(value, name)
            for value, name in (
                (representation, "representation"),
                (model_family, "model_family"),
                (objective, "objective"),
                (temporal_policy, "temporal_policy"),
                (fusion_member, "fusion_member"),
                (result, "result"),
            )
        )
        if result not in {
            "improved",
            "no_improvement",
            "unsupported",
            "infrastructure_failure",
        }:
            raise StateInvariantError("family evidence result is not a closed BranchResult")
        fingerprint = hashlib.sha256(" | ".join(parts).encode()).hexdigest()
        with self._transaction() as connection:
            now = self._timestamp()
            existing = connection.execute(
                """
                SELECT representation, model_family, objective, temporal_policy,
                       fusion_member, result
                FROM family_evidence
                WHERE contract_id = ? AND family_id = ? AND fingerprint = ?
                """,
                (contract, family, fingerprint),
            ).fetchone()
            if existing is not None:
                if tuple(map(str, existing)) != parts:
                    raise StateInvariantError(
                        "family evidence fingerprint already binds different content"
                    )
                return FamilyEvidenceReceipt(fingerprint, False)
            campaign_row = self._assert_campaign_lineage(
                connection, campaign_id=campaign, contract_id=contract, writable=True
            )
            if bool(campaign_row["research_frozen"]):
                raise StateInvariantError(
                    "research is permanently frozen after the first protected outcome"
                )
            association = connection.execute(
                """
                SELECT 1 FROM families
                WHERE campaign_id = ? AND contract_id = ? AND family_id = ?
                """,
                (campaign, contract, family),
            ).fetchone()
            if association is None:
                raise StateInvariantError("family evidence is outside campaign lineage")
            self._assert_write_fence(
                connection,
                campaign_id=campaign,
                lease=lease,
                now=now,
                resource_kind="family",
                resource_id=family,
            )
            connection.execute(
                """
                INSERT INTO family_evidence(
                    contract_id, family_id, fingerprint, origin_campaign_id, representation,
                    model_family, objective, temporal_policy, fusion_member, result, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (contract, family, fingerprint, campaign, *parts, now),
            )
            self._append_event(
                connection,
                campaign_id=campaign,
                event_type="family_evidence_recorded",
                entity_kind="family_evidence",
                entity_id=fingerprint,
                prior_state=None,
                new_state=result,
                revision=0,
                payload_json=_canonical_json(
                    {
                        "family_id": family,
                        "fingerprint": fingerprint,
                        "fusion_member": fusion_member,
                        "model_family": model_family,
                        "objective": objective,
                        "representation": representation,
                        "result": result,
                        "temporal_policy": temporal_policy,
                    },
                    "family evidence event",
                ),
                now=now,
            )
            return FamilyEvidenceReceipt(fingerprint, True)

    def transition(
        self,
        *,
        campaign_id: IdInput,
        entity_kind: str,
        entity_id: IdInput,
        expected_state: str,
        expected_revision: int,
        new_state: str,
        event_type: str,
        payload: Mapping[str, object] | None = None,
        process_identity: Mapping[str, object] | None = None,
        terminal: bool = False,
        lease: Lease | None = None,
    ) -> Transition:
        """Compare-and-swap one state and append its immutable event in the same transaction."""

        campaign = _identifier(campaign_id, "campaign_id")
        kind = _text(entity_kind, "entity_kind")
        identity = _identifier(entity_id, "entity_id")
        prior = _text(expected_state, "expected_state")
        target = _text(new_state, "new_state")
        event = _text(event_type, "event_type")
        if type(expected_revision) is not int or expected_revision < 0:
            raise StateInvariantError("expected_revision must be a non-negative integer")
        if prior == target:
            raise StateInvariantError("state transitions must change state")
        table = _STATEFUL_TABLES.get(kind)
        if table is None:
            raise StateInvariantError(f"unsupported stateful entity_kind {kind!r}")
        if kind in _ATOMIC_FINAL_STATES and (terminal or target in _ATOMIC_FINAL_STATES[kind]):
            raise StateInvariantError(
                "campaign/replay/bundle completion must use atomic finalize_campaign"
            )
        payload_json = _canonical_json(payload or {}, "payload")
        if process_identity is not None and (kind != "attempt" or target != "RUNNING"):
            raise StateInvariantError(
                "process_identity is accepted only when an attempt enters RUNNING"
            )
        process_identity_json = (
            _canonical_json(process_identity, "process_identity")
            if process_identity is not None
            else None
        )
        with self._transaction() as connection:
            now = self._timestamp()
            self._assert_write_fence(
                connection,
                campaign_id=campaign,
                lease=lease,
                now=now,
                resource_kind=kind,
                resource_id=identity,
            )
            campaign_status = connection.execute(
                "SELECT research_frozen FROM campaigns WHERE campaign_id = ?", (campaign,)
            ).fetchone()
            if campaign_status is None:
                raise StateNotFoundError("campaign does not exist")
            if (
                bool(campaign_status[0])
                and kind in {"trial", "attempt", "provider_operation"}
                and not terminal
            ):
                raise StateInvariantError(
                    "research is permanently frozen; only terminal cleanup transitions remain"
                )
            reserved_event_seq = None
            if table == "campaigns":
                updated = connection.execute(
                    """
                    UPDATE campaigns
                    SET state = ?, revision = revision + 1, terminal = ?, updated_at = ?,
                        completed_at = CASE WHEN ? = 1 THEN ? ELSE completed_at END
                    WHERE campaign_id = ? AND campaign_id = ? AND state = ?
                          AND revision = ? AND terminal = 0
                    """,
                    (
                        target,
                        int(terminal),
                        now,
                        int(terminal),
                        now,
                        identity,
                        campaign,
                        prior,
                        expected_revision,
                    ),
                )
            elif table == "attempts":
                updated = connection.execute(
                    """
                    UPDATE attempts
                    SET state = ?, revision = revision + 1, terminal = ?, updated_at = ?,
                        process_identity_json = COALESCE(?, process_identity_json)
                    WHERE attempt_id = ? AND campaign_id = ? AND state = ?
                          AND revision = ? AND terminal = 0
                    """,
                    (
                        target,
                        int(terminal),
                        now,
                        process_identity_json,
                        identity,
                        campaign,
                        prior,
                        expected_revision,
                    ),
                )
            else:
                updated = connection.execute(
                    f"""
                UPDATE {table}
                SET state = ?, revision = revision + 1, terminal = ?, updated_at = ?
                WHERE {_primary_key(table)} = ? AND campaign_id = ? AND state = ?
                      AND revision = ? AND terminal = 0
                """,
                    (
                        target,
                        int(terminal),
                        now,
                        identity,
                        campaign,
                        prior,
                        expected_revision,
                    ),
                )
            if updated.rowcount != 1:
                self._raise_transition_conflict(
                    connection,
                    table=table,
                    entity_id=identity,
                    campaign_id=campaign,
                )
            revision = expected_revision + 1
            event_seq = self._append_event(
                connection,
                campaign_id=campaign,
                event_type=event,
                entity_kind=kind,
                entity_id=identity,
                prior_state=prior,
                new_state=target,
                revision=revision,
                payload_json=payload_json,
                now=now,
                event_seq=reserved_event_seq,
            )
            return Transition(kind, identity, prior, target, revision, event_seq, terminal)

    def reserve_protected_query(
        self,
        *,
        reservation_id: IdInput,
        campaign_id: CampaignIdInput,
        contract_id: ContractIdInput,
        family_id: FamilyIdInput,
        prediction_id: PredictionIdInput,
        idempotency_key: str,
        expected_campaign_revision: int,
        lease: Lease | None = None,
    ) -> ProtectedReservation:
        """Atomically allocate a contract ordinal against one exact ``PredictionId``.

        The idempotency key is also the scorer call key.  An exact replay returns the original
        reservation without charging another ordinal.
        """

        reservation = _identifier(reservation_id, "reservation_id")
        campaign = _identifier(campaign_id, "campaign_id")
        contract = _identifier(contract_id, "contract_id")
        family = _identifier(family_id, "family_id")
        prediction = _identifier(prediction_id, "prediction_id")
        key = _text(idempotency_key, "idempotency_key")
        if type(expected_campaign_revision) is not int or expected_campaign_revision < 0:
            raise StateInvariantError("expected_campaign_revision must be non-negative")
        with self._transaction() as connection:
            now = self._timestamp()
            existing_rows = connection.execute(
                """
                SELECT reservation_id, campaign_id, contract_id, family_id, prediction_id,
                       query_ordinal, idempotency_key, state
                FROM protected_query_reservations
                WHERE (contract_id = ? AND idempotency_key = ?) OR reservation_id = ?
                """,
                (contract, key, reservation),
            ).fetchall()
            if len(existing_rows) > 1:
                raise StateInvariantError(
                    "reservation id and idempotency key identify different protected queries"
                )
            if existing_rows:
                existing = existing_rows[0]
                expected = (reservation, campaign, contract, family, prediction, key)
                actual = (
                    str(existing[0]),
                    str(existing[1]),
                    str(existing[2]),
                    str(existing[3]),
                    str(existing[4]),
                    str(existing[6]),
                )
                if actual != expected:
                    raise StateInvariantError(
                        "protected reservation idempotency identity binds different content"
                    )
                return ProtectedReservation(
                    reservation,
                    campaign,
                    contract,
                    family,
                    prediction,
                    int(existing[5]),
                    key,
                    str(existing[7]),
                    False,
                )
            self._assert_write_fence(connection, campaign_id=campaign, lease=lease, now=now)
            campaign_row = self._assert_campaign_lineage(
                connection, campaign_id=campaign, contract_id=contract, writable=True
            )
            if int(campaign_row["revision"]) != expected_campaign_revision:
                raise StateConflictError("stale campaign revision for protected reservation")
            eligible = connection.execute(
                """
                SELECT 1 FROM families
                WHERE family_id = ? AND campaign_id = ? AND contract_id = ?
                      AND protected_eligible = 1
                """,
                (family, campaign, contract),
            ).fetchone()
            if eligible is None:
                raise StateInvariantError("family is not protected-query eligible")
            prediction_row = connection.execute(
                """
                SELECT p.trial_id, p.rank_graph_id, COALESCE(e.family_id, r.family_id)
                FROM predictions AS p
                LEFT JOIN trials AS t ON t.trial_id = p.trial_id
                LEFT JOIN campaign_experiments AS e
                  ON e.campaign_id = t.campaign_id AND e.experiment_id = t.experiment_id
                LEFT JOIN rank_graphs AS r ON r.rank_graph_id = p.rank_graph_id
                WHERE p.prediction_id = ? AND p.campaign_id = ? AND p.contract_id = ?
                """,
                (prediction, campaign, contract),
            ).fetchone()
            if prediction_row is None:
                raise StateInvariantError("protected PredictionId is outside campaign lineage")
            if prediction_row[2] is None or str(prediction_row[2]) != family:
                raise StateInvariantError("protected PredictionId differs from eligible FamilyId")
            prior_prediction = connection.execute(
                """
                SELECT state FROM protected_query_reservations
                WHERE contract_id = ? AND prediction_id = ?
                """,
                (contract, prediction),
            ).fetchone()
            if prior_prediction is not None:
                raise ProtectedOutcomeTerminalError(
                    "exact PredictionId already has a protected reservation; do not retry it"
                )
            contract_limit = int(
                connection.execute(
                    "SELECT protected_query_limit FROM contracts WHERE contract_id = ?", (contract,)
                ).fetchone()[0]
            )
            used = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM protected_query_reservations WHERE contract_id = ?
                    """,
                    (contract,),
                ).fetchone()[0]
            )
            if used >= contract_limit:
                raise ProtectedBudgetExhaustedError(
                    f"ContractId protected-query limit {contract_limit} is exhausted"
                )
            ordinal = used + 1
            updated = connection.execute(
                """
                UPDATE campaigns
                SET revision = revision + 1, updated_at = ?
                WHERE campaign_id = ? AND revision = ? AND terminal = 0
                """,
                (now, campaign, expected_campaign_revision),
            )
            if updated.rowcount != 1:
                raise StateConflictError("campaign changed while reserving protected query")
            try:
                connection.execute(
                    """
                    INSERT INTO protected_query_reservations(
                        reservation_id, contract_id, campaign_id, family_id, prediction_id,
                        query_ordinal, idempotency_key, state, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'RESERVED', ?)
                    """,
                    (reservation, contract, campaign, family, prediction, ordinal, key, now),
                )
            except sqlite3.IntegrityError as exc:
                if "prediction" in str(exc).lower():
                    raise ProtectedOutcomeTerminalError(
                        "exact PredictionId has already consumed a protected reservation"
                    ) from exc
                raise
            self._append_event(
                connection,
                campaign_id=campaign,
                event_type="protected_query_reserved",
                entity_kind="protected_query_reservation",
                entity_id=reservation,
                prior_state=None,
                new_state="RESERVED",
                revision=expected_campaign_revision + 1,
                payload_json=_canonical_json(
                    {"prediction_id": prediction, "query_ordinal": ordinal}, "reservation event"
                ),
                now=now,
            )
            return ProtectedReservation(
                reservation,
                campaign,
                contract,
                family,
                prediction,
                ordinal,
                key,
                "RESERVED",
                True,
            )

    def complete_protected_query(
        self,
        *,
        evaluation_id: IdInput,
        reservation_id: IdInput,
        outcome: ProtectedOutcomeKind,
        metrics: Mapping[str, object] | None = None,
        unknown_reason: str | None = None,
        lease: Lease | None = None,
    ) -> ProtectedOutcome:
        """Journal exactly one RESULT or UNKNOWN outcome; a terminal outcome cannot be replaced."""

        evaluation = _identifier(evaluation_id, "evaluation_id")
        reservation = _identifier(reservation_id, "reservation_id")
        if not isinstance(outcome, ProtectedOutcomeKind):
            raise StateInvariantError("outcome must be ProtectedOutcomeKind")
        if outcome is ProtectedOutcomeKind.RESULT:
            if metrics is None:
                raise StateInvariantError("RESULT requires metrics")
            metrics_json = _canonical_json(metrics, "metrics")
            reason = None
        else:
            if metrics is not None:
                raise StateInvariantError("UNKNOWN cannot carry metrics")
            reason = _text(unknown_reason, "unknown_reason")
            metrics_json = None
        with self._transaction() as connection:
            now = self._timestamp()
            reservation_row = connection.execute(
                """
                SELECT contract_id, campaign_id, prediction_id, state
                FROM protected_query_reservations WHERE reservation_id = ?
                """,
                (reservation,),
            ).fetchone()
            if reservation_row is None:
                raise StateNotFoundError("protected reservation does not exist")
            contract, campaign, prediction, state = map(str, reservation_row)
            existing = connection.execute(
                """
                SELECT evaluation_id, outcome, metrics_json, unknown_reason
                FROM protected_evaluations WHERE reservation_id = ?
                """,
                (reservation,),
            ).fetchone()
            if existing is not None:
                exact = (
                    str(existing[0]) == evaluation
                    and str(existing[1]) == outcome.value
                    and cast(str | None, existing[2]) == metrics_json
                    and cast(str | None, existing[3]) == reason
                )
                if not exact:
                    raise ProtectedOutcomeTerminalError(
                        "protected RESULT/UNKNOWN is terminal and cannot be retried or replaced"
                    )
                return ProtectedOutcome(
                    evaluation,
                    reservation,
                    prediction,
                    outcome,
                    _json_mapping(metrics_json) if metrics_json is not None else None,
                    reason,
                    False,
                )
            self._assert_write_fence(
                connection,
                campaign_id=campaign,
                lease=lease,
                now=now,
                resource_kind="protected_query_reservation",
                resource_id=reservation,
            )
            if state != "RESERVED":
                raise ProtectedOutcomeTerminalError(
                    "protected reservation is terminal without a replaceable outcome"
                )
            updated = connection.execute(
                """
                UPDATE protected_query_reservations
                SET state = ?, completed_at = ?
                WHERE reservation_id = ? AND state = 'RESERVED'
                """,
                (outcome.value, now, reservation),
            )
            if updated.rowcount != 1:
                raise ProtectedOutcomeTerminalError("protected reservation completed concurrently")
            connection.execute(
                """
                INSERT INTO protected_evaluations(
                    evaluation_id, reservation_id, contract_id, campaign_id, prediction_id,
                    outcome, metrics_json, unknown_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation,
                    reservation,
                    contract,
                    campaign,
                    prediction,
                    outcome.value,
                    metrics_json,
                    reason,
                    now,
                ),
            )
            revision = self._freeze_research(connection, campaign_id=campaign, now=now)
            self._append_event(
                connection,
                campaign_id=campaign,
                event_type="protected_query_completed",
                entity_kind="protected_evaluation",
                entity_id=evaluation,
                prior_state="RESERVED",
                new_state=outcome.value,
                revision=revision,
                payload_json=metrics_json
                or _canonical_json({"unknown_reason": reason}, "unknown outcome event"),
                now=now,
            )
            return ProtectedOutcome(
                evaluation,
                reservation,
                prediction,
                outcome,
                _json_mapping(metrics_json) if metrics_json is not None else None,
                reason,
                True,
            )

    def claim_lease(
        self,
        *,
        resource_kind: str,
        resource_id: IdInput,
        owner_id: str,
        ttl: timedelta,
        campaign_id: CampaignIdInput | None = None,
    ) -> Lease:
        """Claim an absent or expired lease and allocate a monotonic fencing token."""

        kind = _text(resource_kind, "resource_kind")
        requested_identity = _identifier(resource_id, "resource_id")
        owner = _text(owner_id, "owner_id")
        seconds = _ttl_seconds(ttl)
        if kind != "campaign" and kind not in _LEASE_RESOURCE_TABLES:
            raise StateInvariantError(f"unsupported lease resource_kind {kind!r}")
        with self._transaction() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            expires = _timestamp(now_dt + timedelta(seconds=seconds))
            if kind in {"family", "experiment"}:
                if campaign_id is None:
                    raise StateInvariantError(f"{kind} lease requires campaign_id scope")
                scope_campaign = _identifier(campaign_id, "campaign_id")
                table, primary_key = _LEASE_RESOURCE_TABLES[kind]
                association = connection.execute(
                    f"SELECT 1 FROM {table} WHERE campaign_id = ? AND {primary_key} = ?",
                    (scope_campaign, requested_identity),
                ).fetchone()
                if association is None:
                    raise StateNotFoundError(f"campaign-{kind} lease target does not exist")
                identity = _association_lease_resource_id(kind, scope_campaign, requested_identity)
            else:
                if campaign_id is not None:
                    raise StateInvariantError(
                        "campaign_id scope is accepted only for association leases"
                    )
                identity = requested_identity
            existing = connection.execute(
                """
                SELECT owner_id, fence_token, expires_at, completed FROM leases
                WHERE resource_kind = ? AND resource_id = ?
                """,
                (kind, identity),
            ).fetchone()
            if existing is not None:
                existing_owner, token, existing_expiry, completed = (
                    str(existing[0]),
                    int(existing[1]),
                    str(existing[2]),
                    bool(existing[3]),
                )
                if completed:
                    raise LeaseConflictError("completed lease cannot be acquired")
                if existing_expiry > now:
                    if existing_owner == owner:
                        return Lease(kind, identity, owner, token, existing_expiry, False)
                    raise LeaseConflictError("resource has a live lease owned by another worker")
            self._assert_lease_scope_available(
                connection,
                resource_kind=kind,
                resource_id=identity,
                now=now,
            )
            counter = connection.execute(
                """
                SELECT next_token FROM fencing_counters
                WHERE resource_kind = ? AND resource_id = ?
                """,
                (kind, identity),
            ).fetchone()
            token = 1 if counter is None else int(counter[0])
            if counter is None:
                connection.execute(
                    """
                    INSERT INTO fencing_counters(resource_kind, resource_id, next_token)
                    VALUES (?, ?, 2)
                    """,
                    (kind, identity),
                )
            else:
                connection.execute(
                    """
                    UPDATE fencing_counters SET next_token = next_token + 1
                    WHERE resource_kind = ? AND resource_id = ?
                    """,
                    (kind, identity),
                )
            connection.execute(
                """
                INSERT INTO leases(
                    resource_kind, resource_id, owner_id, fence_token, expires_at, completed,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(resource_kind, resource_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    fence_token = excluded.fence_token,
                    expires_at = excluded.expires_at,
                    completed = 0,
                    updated_at = excluded.updated_at
                """,
                (kind, identity, owner, token, expires, now),
            )
            return Lease(kind, identity, owner, token, expires, False)

    def renew_lease(
        self,
        *,
        resource_kind: str,
        resource_id: IdInput,
        owner_id: str,
        fence_token: int,
        ttl: timedelta,
    ) -> Lease:
        """Renew only the exact live fenced lease."""

        kind = _text(resource_kind, "resource_kind")
        identity = _identifier(resource_id, "resource_id")
        owner = _text(owner_id, "owner_id")
        token = _positive_int(fence_token, "fence_token")
        seconds = _ttl_seconds(ttl)
        with self._transaction() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            expires = _timestamp(now_dt + timedelta(seconds=seconds))
            updated = connection.execute(
                """
                UPDATE leases SET expires_at = ?, updated_at = ?
                WHERE resource_kind = ? AND resource_id = ? AND owner_id = ?
                      AND fence_token = ? AND completed = 0 AND expires_at > ?
                """,
                (expires, now, kind, identity, owner, token, now),
            )
            if updated.rowcount != 1:
                raise LeaseConflictError("lease renewal rejected by owner, fencing, or expiry")
            return Lease(kind, identity, owner, token, expires, False)

    def release_lease(
        self,
        *,
        resource_kind: str,
        resource_id: IdInput,
        owner_id: str,
        fence_token: int,
        complete: bool,
    ) -> Lease:
        """Release an exact fenced lease; completion permanently closes the resource lease."""

        kind = _text(resource_kind, "resource_kind")
        identity = _identifier(resource_id, "resource_id")
        owner = _text(owner_id, "owner_id")
        token = _positive_int(fence_token, "fence_token")
        with self._transaction() as connection:
            now = self._timestamp()
            if complete:
                updated = connection.execute(
                    """
                    UPDATE leases SET expires_at = ?, completed = 1, updated_at = ?
                    WHERE resource_kind = ? AND resource_id = ? AND owner_id = ?
                          AND fence_token = ? AND completed = 0 AND expires_at > ?
                    """,
                    (now, now, kind, identity, owner, token, now),
                )
            else:
                updated = connection.execute(
                    """
                    UPDATE leases SET expires_at = ?, updated_at = ?
                    WHERE resource_kind = ? AND resource_id = ? AND owner_id = ?
                          AND fence_token = ? AND completed = 0 AND expires_at > ?
                    """,
                    (now, now, kind, identity, owner, token, now),
                )
            if updated.rowcount != 1:
                raise LeaseConflictError("lease release rejected by owner or fencing token")
            return Lease(kind, identity, owner, token, now, complete)

    def prepare_terminal_projection(
        self,
        *,
        campaign_id: CampaignIdInput,
        contract_id: ContractIdInput,
        expected_state: str,
        expected_revision: int,
        preparation: TerminalPreparation,
        lease: Lease | None = None,
    ) -> PreparedTerminalProjection:
        """Persist one bundle-independent terminal intent at an exact event horizon."""

        campaign = _identifier(campaign_id, "campaign_id")
        contract = _identifier(contract_id, "contract_id")
        source_state = _text(expected_state, "expected_state")
        if type(expected_revision) is not int or expected_revision < 0:
            raise StateInvariantError("expected_revision must be non-negative")
        if not isinstance(preparation, TerminalPreparation):
            raise StateInvariantError("preparation must be TerminalPreparation")
        decision = _identifier(preparation.decision_id, "decision_id")
        replay = _identifier(preparation.replay_id, "replay_id")
        selected = _identifier(preparation.selected_prediction_id, "selected_prediction_id")
        fallback = _identifier(preparation.fallback_prediction_id, "fallback_prediction_id")
        terminal_state = _text(preparation.terminal_state, "terminal_state")
        if terminal_state not in _ATOMIC_FINAL_STATES["campaign"]:
            raise StateInvariantError("final campaign state is not an allowed terminal state")
        _reject_publication_fields(preparation.bundle_claims)
        if terminal_state == _PRODUCTION_TERMINAL_STATE:
            if not isinstance(preparation.same_backend_receipt, ReplayGradeReceipt):
                raise StateInvariantError(
                    "production preparation requires a typed same-backend receipt"
                )
            validated_replay, validated_claims = _normalize_production_evidence(
                contract_id=contract,
                campaign_id=campaign,
                selected_prediction_id=selected,
                fallback_prediction_id=fallback,
                replay_payload=preparation.replay_payload,
                bundle_claims=preparation.bundle_claims,
                scoring_exact_receipt=preparation.scoring_exact_receipt,
                same_backend_receipt=preparation.same_backend_receipt,
            )
        else:
            if (
                preparation.scoring_exact_receipt is not None
                or preparation.same_backend_receipt is not None
            ):
                raise StateInvariantError(
                    "offline terminal preparation cannot attach production replay receipts"
                )
            validated_replay, validated_claims = _normalize_offline_fixture_evidence(
                contract_id=contract,
                campaign_id=campaign,
                selected_prediction_id=selected,
                replay_payload=preparation.replay_payload,
                bundle_claims=preparation.bundle_claims,
            )
        decision_json = _canonical_json(preparation.decision_payload, "decision_payload")
        replay_json = (
            _canonical_json_preserve_numbers(validated_replay, "validated replay evidence")
            if terminal_state == _PRODUCTION_TERMINAL_STATE
            else _canonical_json(validated_replay, "validated replay evidence")
        )
        claims_json = _canonical_json(validated_claims, "validated bundle claims")
        intent = (
            contract,
            source_state,
            expected_revision,
            terminal_state,
            decision,
            replay,
            selected,
            fallback,
            decision_json,
            replay_json,
            claims_json,
        )
        with self._transaction() as connection:
            self._assert_campaign_lineage(
                connection, campaign_id=campaign, contract_id=contract, writable=False
            )
            _validated_resource_receipt_payload(
                connection,
                receipt_id=cast(str, validated_claims["resource_receipt_id"]),
                campaign_id=campaign,
                contract_id=contract,
                selected_prediction_id=selected,
                terminal_state=terminal_state,
                expected_qualification_manifest_digest=cast(
                    str | None, validated_replay.get("qualification_manifest_digest")
                ),
                expected_qualification_fallback_digest=cast(
                    str | None, validated_replay.get("qualification_fallback_digest")
                ),
                expected_prediction_sha256=cast(
                    str | None, validated_replay.get("original_prediction_sha256")
                ),
                expected_row_alignment_sha256=cast(
                    str | None, validated_replay.get("row_alignment_sha256")
                ),
            )
            if terminal_state == _PRODUCTION_TERMINAL_STATE:
                _validate_production_authority_counts(connection, campaign_id=campaign)
            current = connection.execute(
                """
                SELECT state, revision, next_event_seq, terminal
                FROM campaigns WHERE campaign_id = ?
                """,
                (campaign,),
            ).fetchone()
            if current is None:  # pragma: no cover - lineage check above
                raise StateNotFoundError("campaign does not exist")
            current_last_event_seq = int(current[2]) - 1
            existing = connection.execute(
                """
                SELECT preparation_id, contract_id, source_state, source_campaign_revision,
                       source_last_event_seq, terminal_state, decision_id, replay_id,
                       selected_prediction_id, fallback_prediction_id, decision_payload_json,
                       replay_payload_json, bundle_claims_json, projection_schema_version,
                       redaction_policy_version, prepared_projection_sha256,
                       prepared_projection_json
                FROM terminal_preparations
                WHERE campaign_id = ? AND source_campaign_revision = ?
                  AND (
                    source_last_event_seq = ?
                    OR EXISTS (
                        SELECT 1 FROM bundle_publications AS bp
                        WHERE bp.preparation_id = terminal_preparations.preparation_id
                    )
                  )
                ORDER BY source_last_event_seq
                """,
                (campaign, expected_revision, current_last_event_seq),
            ).fetchone()
            if existing is not None:
                observed = (
                    str(existing[1]),
                    str(existing[2]),
                    int(existing[3]),
                    str(existing[5]),
                    str(existing[6]),
                    str(existing[7]),
                    str(existing[8]),
                    str(existing[9]),
                    str(existing[10]),
                    str(existing[11]),
                    str(existing[12]),
                )
                if observed != intent:
                    raise TerminalPreparationConflictError(
                        "campaign/source horizon already binds a different terminal intent"
                    )
                return _prepared_terminal_projection(existing, campaign_id=campaign, created=False)

            now = self._timestamp()
            self._assert_campaign_lineage(
                connection, campaign_id=campaign, contract_id=contract, writable=True
            )
            if (str(current[0]), int(current[1])) != (source_state, expected_revision):
                raise StateConflictError("campaign changed before terminal preparation")
            self._assert_write_fence(connection, campaign_id=campaign, lease=lease, now=now)
            self._assert_finalization_quiescent(
                connection, campaign_id=campaign, replay_id=replay, bundle_id="", now=now
            )
            predictions = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM predictions
                    WHERE prediction_id IN (?, ?) AND campaign_id = ? AND contract_id = ?
                    """,
                    (selected, fallback, campaign, contract),
                ).fetchone()[0]
            )
            if predictions != (1 if selected == fallback else 2):
                raise StateInvariantError("final predictions are outside exact campaign lineage")
            if (
                connection.execute(
                    "SELECT 1 FROM selection_decisions WHERE decision_id = ?", (decision,)
                ).fetchone()
                is not None
            ):
                raise StateInvariantError("DecisionId already exists before terminal preparation")
            replay_row = connection.execute(
                """
                SELECT contract_id, campaign_id, prediction_id, terminal
                FROM replays WHERE replay_id = ?
                """,
                (replay,),
            ).fetchone()
            if replay_row is not None and (
                tuple(replay_row[:3]) != (contract, campaign, selected) or bool(replay_row[3])
            ):
                raise StateInvariantError("ReplayId already binds incompatible or terminal content")
            last_event_seq = int(current[2]) - 1
            projection = _build_terminal_projection(
                connection,
                campaign_id=campaign,
                contract_id=contract,
                source_state=source_state,
                source_revision=expected_revision,
                source_last_event_seq=last_event_seq,
                terminal_state=terminal_state,
                decision_id=decision,
                replay_id=replay,
                selected_prediction_id=selected,
                fallback_prediction_id=fallback,
                decision_json=decision_json,
                replay_json=replay_json,
                bundle_claims_json=claims_json,
                prepared_at=now,
            )
            projection_json = (
                _canonical_json_preserve_numbers(projection, "prepared terminal projection")
                if terminal_state == _PRODUCTION_TERMINAL_STATE
                else _canonical_json(projection, "prepared terminal projection")
            )
            projection_sha = hashlib.sha256(projection_json.encode()).hexdigest()
            preparation_id = hashlib.sha256(
                b"kuairand-terminal-preparation-v1\0"
                + canonical_json_bytes(
                    {
                        "campaign_id": campaign,
                        "contract_id": contract,
                        "source_campaign_revision": expected_revision,
                        "source_last_event_seq": last_event_seq,
                        "projection_schema_version": _TERMINAL_PROJECTION_SCHEMA_VERSION,
                        "redaction_policy_version": _TERMINAL_REDACTION_POLICY_VERSION,
                        "projection_sha256": projection_sha,
                    }
                )
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO terminal_preparations(
                    preparation_id, campaign_id, contract_id, source_state,
                    source_campaign_revision, source_last_event_seq, terminal_state,
                    decision_id, replay_id, selected_prediction_id, fallback_prediction_id,
                    decision_payload_json, replay_payload_json, bundle_claims_json,
                    projection_schema_version, redaction_policy_version,
                    prepared_projection_sha256, prepared_projection_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preparation_id,
                    campaign,
                    contract,
                    source_state,
                    expected_revision,
                    last_event_seq,
                    terminal_state,
                    decision,
                    replay,
                    selected,
                    fallback,
                    decision_json,
                    replay_json,
                    claims_json,
                    _TERMINAL_PROJECTION_SCHEMA_VERSION,
                    _TERMINAL_REDACTION_POLICY_VERSION,
                    projection_sha,
                    projection_json,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT preparation_id, contract_id, source_state, source_campaign_revision,
                       source_last_event_seq, terminal_state, decision_id, replay_id,
                       selected_prediction_id, fallback_prediction_id, decision_payload_json,
                       replay_payload_json, bundle_claims_json, projection_schema_version,
                       redaction_policy_version, prepared_projection_sha256,
                       prepared_projection_json
                FROM terminal_preparations WHERE preparation_id = ?
                """,
                (preparation_id,),
            ).fetchone()
            if row is None:  # pragma: no cover - insert above
                raise StateInvariantError("terminal preparation was not persisted")
            return _prepared_terminal_projection(row, campaign_id=campaign, created=True)

    def materialize_prepared_terminal_projection(
        self,
        *,
        preparation_id: IdInput,
        snapshot_destination: Path,
        event_export_destination: Path,
    ) -> PreparedProjectionArtifacts:
        """Atomically materialize the immutable terminal projection as SQLite and JSONL."""

        identity = _identifier(preparation_id, "preparation_id")
        if not isinstance(snapshot_destination, Path) or not isinstance(
            event_export_destination, Path
        ):
            raise StateInvariantError("projection destinations must be pathlib.Path values")
        if snapshot_destination.resolve(strict=False) == event_export_destination.resolve(
            strict=False
        ):
            raise StateInvariantError("prepared projection destinations must differ")
        with self._readonly() as connection:
            row = connection.execute(
                """
                SELECT campaign_id, contract_id, source_campaign_revision,
                       source_last_event_seq, terminal_state, projection_schema_version,
                       redaction_policy_version, prepared_projection_sha256,
                       prepared_projection_json
                FROM terminal_preparations WHERE preparation_id = ?
                """,
                (identity,),
            ).fetchone()
        if row is None:
            raise StateNotFoundError("terminal preparation does not exist")
        projection_json = str(row[8])
        if hashlib.sha256(projection_json.encode()).hexdigest() != str(row[7]):
            raise StateInvariantError("stored terminal projection digest is invalid")
        projection = _json_mapping(projection_json)
        snapshot_sha, snapshot_size = _write_terminal_projection_sqlite(
            snapshot_destination,
            preparation_id=identity,
            projection_sha256=str(row[7]),
            projection=projection,
            preserve_numbers=str(row[4]) == _PRODUCTION_TERMINAL_STATE,
        )
        events = projection.get("events")
        if not isinstance(events, list):
            raise StateInvariantError("prepared terminal projection has no event list")
        event_payload = b"".join(
            (
                _canonical_json_preserve_numbers(event, "terminal event").encode("ascii")
                if str(row[4]) == _PRODUCTION_TERMINAL_STATE
                else canonical_json_bytes(event)
            )
            + b"\n"
            for event in events
        )
        event_sha, event_size = _atomic_publish_bytes(event_export_destination, event_payload)
        return PreparedProjectionArtifacts(
            identity,
            snapshot_destination.resolve(),
            snapshot_sha,
            snapshot_size,
            event_export_destination.resolve(),
            event_sha,
            event_size,
        )

    def finalize_prepared_campaign(
        self,
        *,
        preparation_id: IdInput,
        publication: PublishedBundleReceipt,
        lease: Lease | None = None,
    ) -> Transition:
        """Consume a verified publication and atomically commit its prepared terminal intent."""

        identity = _identifier(preparation_id, "preparation_id")
        if not isinstance(publication, PublishedBundleReceipt):
            raise StateInvariantError("publication must be PublishedBundleReceipt")
        bundle = _identifier(publication.bundle_id, "bundle_id")
        manifest_sha = _sha256(publication.manifest_sha256, "manifest_sha256")
        inventory_sha = _sha256(publication.inventory_sha256, "inventory_sha256")
        submission_sha = _sha256(publication.submission_sha256, "submission_sha256")
        file_count = _positive_int(publication.file_count, "file_count")
        total_size = _positive_int(publication.total_size_bytes, "total_size_bytes")
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT campaign_id, contract_id, source_state, source_campaign_revision,
                       source_last_event_seq, terminal_state, decision_id, replay_id,
                       selected_prediction_id, fallback_prediction_id, decision_payload_json,
                       replay_payload_json, bundle_claims_json, prepared_projection_sha256,
                       projection_schema_version, redaction_policy_version
                FROM terminal_preparations WHERE preparation_id = ?
                """,
                (identity,),
            ).fetchone()
            if row is None:
                raise StateNotFoundError("terminal preparation does not exist")
            campaign = str(row[0])
            contract = str(row[1])
            source_state = str(row[2])
            source_revision = int(row[3])
            source_last_event_seq = int(row[4])
            terminal_state = str(row[5])
            decision = str(row[6])
            replay = str(row[7])
            selected = str(row[8])
            fallback = str(row[9])
            decision_json = str(row[10])
            replay_json = str(row[11])
            claims = dict(_json_mapping(str(row[12])))
            projection_sha = str(row[13])
            validated_replay, validated_claims = _validate_stored_terminal_evidence(
                terminal_state=terminal_state,
                contract_id=contract,
                campaign_id=campaign,
                selected_prediction_id=selected,
                fallback_prediction_id=fallback,
                replay_json=replay_json,
                bundle_claims=claims,
            )
            resource_receipt_id = cast(str, validated_claims["resource_receipt_id"])
            resource_receipt_payload = _validated_resource_receipt_payload(
                connection,
                receipt_id=resource_receipt_id,
                campaign_id=campaign,
                contract_id=contract,
                selected_prediction_id=selected,
                terminal_state=terminal_state,
                expected_qualification_manifest_digest=cast(
                    str | None, validated_replay.get("qualification_manifest_digest")
                ),
                expected_qualification_fallback_digest=cast(
                    str | None, validated_replay.get("qualification_fallback_digest")
                ),
                expected_prediction_sha256=cast(
                    str | None, validated_replay.get("original_prediction_sha256")
                ),
                expected_row_alignment_sha256=cast(
                    str | None, validated_replay.get("row_alignment_sha256")
                ),
            )
            if terminal_state == _PRODUCTION_TERMINAL_STATE:
                _validate_production_authority_counts(connection, campaign_id=campaign)
                publication_proof = _validated_production_publication_proof(
                    publication,
                    contract_id=contract,
                    prediction_id=selected,
                    bundle_id=bundle,
                    submission_sha256=submission_sha,
                    inventory_sha256=inventory_sha,
                    scoring_exact_receipt_manifest=validated_replay.get("scoring_exact_receipt"),
                    same_backend_receipt_manifest=validated_replay.get("same_backend_receipt"),
                )
            else:
                if (
                    publication.regeneration_evidence is not None
                    or publication.bundle_exact_receipt is not None
                ):
                    raise StateInvariantError(
                        "offline publication cannot attach production bundle replay proofs"
                    )
                publication_proof = None
            postpublication_resource = _validated_postpublication_resource_receipt(
                publication.postpublication_resource_receipt,
                terminal_state=terminal_state,
                campaign_id=campaign,
                contract_id=contract,
                selected_prediction_id=selected,
                prepublication_resource_receipt_id=resource_receipt_id,
                prepublication_resource_payload=resource_receipt_payload,
            )
            root = _verify_published_bundle(
                publication,
                preparation_id=identity,
                projection_sha256=projection_sha,
                projection_schema_version=int(row[14]),
                redaction_policy_version=int(row[15]),
                campaign_id=campaign,
                contract_id=contract,
                selected_prediction_id=selected,
                prepared_replay_json=replay_json,
                bundle_id=bundle,
                manifest_sha256=manifest_sha,
                inventory_sha256=inventory_sha,
                submission_sha256=submission_sha,
                file_count=file_count,
                total_size_bytes=total_size,
                resource_receipt_id=resource_receipt_id,
                resource_receipt_payload=resource_receipt_payload,
            )
            final_claims = claims
            if publication_proof is not None:
                final_claims = claims | {
                    "bundle_exact_status": "VERIFIED",
                    "replay_grade": ReplayGrade.BUNDLE_EXACT.value,
                    "replay_grades": list(_PRODUCTION_REQUIRED_REPLAY_GRADES),
                    "publication_proof": publication_proof,
                    "postpublication_resource_receipt": postpublication_resource,
                }
            bundle_payload = final_claims | {
                "bundle_path": str(root),
                "bundle_sha256": bundle,
                "evidence_manifest_path": str(root / "bundle-manifest.json"),
                "inventory_sha256": inventory_sha,
                "manifest_sha256": manifest_sha,
                "submission_sha256": submission_sha,
                "preparation_id": identity,
                "prepared_projection_sha256": projection_sha,
            }
            bundle_json = (
                _canonical_json_preserve_numbers(bundle_payload, "published bundle payload")
                if terminal_state == _PRODUCTION_TERMINAL_STATE
                else _canonical_json(bundle_payload, "published bundle payload")
            )
            campaign_row = self._assert_campaign_lineage(
                connection, campaign_id=campaign, contract_id=contract, writable=False
            )
            if bool(campaign_row["terminal"]):
                transition = self._idempotent_finalization(
                    connection,
                    campaign_id=campaign,
                    contract_id=contract,
                    expected_state=source_state,
                    expected_revision=source_revision,
                    terminal_state=terminal_state,
                    decision_id=decision,
                    replay_id=replay,
                    bundle_id=bundle,
                    selected_prediction_id=selected,
                    fallback_prediction_id=fallback,
                    manifest_sha256=manifest_sha,
                    decision_json=decision_json,
                    replay_json=replay_json,
                    bundle_json=bundle_json,
                )
                publication_row = connection.execute(
                    """
                    SELECT preparation_id, campaign_id, contract_id, published_path,
                           manifest_sha256, inventory_sha256, submission_sha256,
                           file_count, total_size_bytes, final_event_seq
                    FROM bundle_publications WHERE bundle_id = ?
                    """,
                    (bundle,),
                ).fetchone()
                expected_publication = (
                    identity,
                    campaign,
                    contract,
                    str(root),
                    manifest_sha,
                    inventory_sha,
                    submission_sha,
                    file_count,
                    total_size,
                    transition.event_seq,
                )
                if publication_row is None or tuple(publication_row) != expected_publication:
                    raise PublishedBundleConflictError(
                        "terminal campaign binds a different bundle publication"
                    )
                return transition

            now = self._timestamp()
            self._assert_write_fence(connection, campaign_id=campaign, lease=lease, now=now)
            current = connection.execute(
                "SELECT state, revision, next_event_seq FROM campaigns WHERE campaign_id = ?",
                (campaign,),
            ).fetchone()
            if current is None:  # pragma: no cover - lineage check above
                raise StateNotFoundError("campaign does not exist")
            observed_source = (str(current[0]), int(current[1]), int(current[2]) - 1)
            expected_source = (source_state, source_revision, source_last_event_seq)
            if observed_source != expected_source:
                raise PreparedSourceStaleError(
                    "prepared source is stale: "
                    f"expected state/revision/event={expected_source!r}, "
                    f"observed={observed_source!r}"
                )
            self._assert_finalization_quiescent(
                connection,
                campaign_id=campaign,
                replay_id=replay,
                bundle_id=bundle,
                now=now,
            )
            predictions = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM predictions
                    WHERE prediction_id IN (?, ?) AND campaign_id = ? AND contract_id = ?
                    """,
                    (selected, fallback, campaign, contract),
                ).fetchone()[0]
            )
            if predictions != (1 if selected == fallback else 2):
                raise StateInvariantError("final predictions are outside exact campaign lineage")
            connection.execute(
                """
                INSERT INTO selection_decisions(
                    decision_id, contract_id, campaign_id, selected_prediction_id,
                    fallback_prediction_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (decision, contract, campaign, selected, fallback, decision_json, now),
            )
            replay_row = connection.execute(
                """
                SELECT contract_id, campaign_id, prediction_id, state, revision, terminal
                FROM replays WHERE replay_id = ?
                """,
                (replay,),
            ).fetchone()
            if replay_row is None:
                connection.execute(
                    """
                    INSERT INTO replays(
                        replay_id, contract_id, campaign_id, prediction_id, state, revision,
                        terminal, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'VERIFIED', 0, 1, ?, ?, ?)
                    """,
                    (replay, contract, campaign, selected, replay_json, now, now),
                )
                replay_prior_state = None
                replay_revision = 0
            elif tuple(replay_row[:3]) != (contract, campaign, selected) or bool(replay_row[5]):
                raise StateInvariantError("ReplayId already binds incompatible or terminal content")
            else:
                connection.execute(
                    """
                    UPDATE replays SET state = 'VERIFIED', revision = revision + 1,
                        terminal = 1, payload_json = ?, updated_at = ?
                    WHERE replay_id = ? AND terminal = 0
                    """,
                    (replay_json, now, replay),
                )
                replay_prior_state = str(replay_row[3])
                replay_revision = int(replay_row[4]) + 1
            self._append_event(
                connection,
                campaign_id=campaign,
                event_type="replay_verified",
                entity_kind="replay",
                entity_id=replay,
                prior_state=replay_prior_state,
                new_state="VERIFIED",
                revision=replay_revision,
                payload_json=replay_json,
                now=now,
            )
            connection.execute(
                """
                INSERT INTO bundles(
                    bundle_id, contract_id, campaign_id, replay_id, selected_prediction_id,
                    state, revision, terminal, manifest_sha256, payload_json, created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, 'SEALED', 0, 1, ?, ?, ?, ?)
                """,
                (
                    bundle,
                    contract,
                    campaign,
                    replay,
                    selected,
                    manifest_sha,
                    bundle_json,
                    now,
                    now,
                ),
            )
            self._append_event(
                connection,
                campaign_id=campaign,
                event_type="bundle_sealed",
                entity_kind="bundle",
                entity_id=bundle,
                prior_state=None,
                new_state="SEALED",
                revision=0,
                payload_json=bundle_json,
                now=now,
            )
            event_seq = self._allocate_event_seq(connection, campaign_id=campaign)
            updated = connection.execute(
                """
                UPDATE campaigns
                SET state = ?, revision = revision + 1, terminal = 1,
                    selected_prediction_id = ?, fallback_prediction_id = ?,
                    updated_at = ?, completed_at = ?
                WHERE campaign_id = ? AND contract_id = ? AND state = ?
                      AND revision = ? AND next_event_seq = ? AND terminal = 0
                """,
                (
                    terminal_state,
                    selected,
                    fallback,
                    now,
                    now,
                    campaign,
                    contract,
                    source_state,
                    source_revision,
                    source_last_event_seq + 4,
                ),
            )
            if updated.rowcount != 1:
                raise PreparedSourceStaleError("campaign changed before prepared finalization CAS")
            final_revision = source_revision + 1
            self._append_event(
                connection,
                campaign_id=campaign,
                event_type="campaign_finalized",
                entity_kind="campaign",
                entity_id=campaign,
                prior_state=source_state,
                new_state=terminal_state,
                revision=final_revision,
                payload_json=_canonical_json(
                    {
                        "bundle_id": bundle,
                        "decision_id": decision,
                        "fallback_prediction_id": fallback,
                        "preparation_id": identity,
                        "replay_id": replay,
                        "selected_prediction_id": selected,
                    },
                    "prepared finalization event",
                ),
                now=now,
                event_seq=event_seq,
            )
            connection.execute(
                """
                INSERT INTO bundle_publications(
                    bundle_id, preparation_id, campaign_id, contract_id, published_path,
                    manifest_sha256, inventory_sha256, submission_sha256, file_count,
                    total_size_bytes, final_event_seq, committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bundle,
                    identity,
                    campaign,
                    contract,
                    str(root),
                    manifest_sha,
                    inventory_sha,
                    submission_sha,
                    file_count,
                    total_size,
                    event_seq,
                    now,
                ),
            )
            return Transition(
                "campaign",
                campaign,
                source_state,
                terminal_state,
                final_revision,
                event_seq,
                True,
            )

    def finalize_campaign(
        self,
        *,
        campaign_id: CampaignIdInput,
        contract_id: ContractIdInput,
        expected_state: str,
        expected_revision: int,
        finalization: object,
        lease: Lease | None = None,
    ) -> Transition:
        """Fail closed: terminal bundle commits require the prepared publication protocol."""

        del campaign_id, contract_id, expected_state, expected_revision, finalization, lease
        raise PreparedFinalizationRequiredError(
            "finalize_campaign is disabled; use prepare_terminal_projection, "
            "materialize_prepared_terminal_projection, and finalize_prepared_campaign"
        )

    def _idempotent_finalization(
        self,
        connection: sqlite3.Connection,
        *,
        campaign_id: str,
        contract_id: str,
        expected_state: str,
        expected_revision: int,
        terminal_state: str,
        decision_id: str,
        replay_id: str,
        bundle_id: str,
        selected_prediction_id: str,
        fallback_prediction_id: str,
        manifest_sha256: str,
        decision_json: str,
        replay_json: str,
        bundle_json: str,
    ) -> Transition:
        rows = connection.execute(
            """
            SELECT c.state, c.revision, c.selected_prediction_id, c.fallback_prediction_id,
                   d.contract_id, d.selected_prediction_id, d.fallback_prediction_id,
                   d.payload_json, r.contract_id, r.prediction_id, r.state, r.terminal,
                   r.payload_json, b.contract_id, b.replay_id, b.selected_prediction_id,
                   b.state, b.terminal, b.manifest_sha256, b.payload_json,
                   e.event_seq, e.prior_state, e.new_state, e.revision
            FROM campaigns AS c
            JOIN selection_decisions AS d
              ON d.campaign_id = c.campaign_id AND d.decision_id = ?
            JOIN replays AS r ON r.campaign_id = c.campaign_id AND r.replay_id = ?
            JOIN bundles AS b ON b.campaign_id = c.campaign_id AND b.bundle_id = ?
            JOIN campaign_events AS e
              ON e.campaign_id = c.campaign_id AND e.event_type = 'campaign_finalized'
            WHERE c.campaign_id = ? AND c.contract_id = ?
            """,
            (decision_id, replay_id, bundle_id, campaign_id, contract_id),
        ).fetchone()
        expected = (
            terminal_state,
            expected_revision + 1,
            selected_prediction_id,
            fallback_prediction_id,
            contract_id,
            selected_prediction_id,
            fallback_prediction_id,
            decision_json,
            contract_id,
            selected_prediction_id,
            "VERIFIED",
            1,
            replay_json,
            contract_id,
            replay_id,
            selected_prediction_id,
            "SEALED",
            1,
            manifest_sha256,
            bundle_json,
        )
        if rows is None or tuple(rows[:20]) != expected:
            raise StateConflictError("terminal campaign differs from requested finalization")
        if (str(rows[21]), str(rows[22]), int(rows[23])) != (
            expected_state,
            terminal_state,
            expected_revision + 1,
        ):
            raise StateConflictError("terminal campaign event differs from requested finalization")
        return Transition(
            "campaign",
            campaign_id,
            expected_state,
            terminal_state,
            expected_revision + 1,
            int(rows[20]),
            True,
        )

    def reconcile_missing_attempts(
        self,
        *,
        process_exists: Callable[[Mapping[str, object]], bool],
    ) -> Reconciliation:
        """Close RUNNING attempts whose durable process identity is absent or no longer live."""

        if not callable(process_exists):
            raise StateInvariantError("process_exists must be callable")
        interrupted: list[str] = []
        with self._readonly() as connection:
            rows = connection.execute(
                """
                SELECT attempt_id, campaign_id, contract_id, revision, process_identity_json
                FROM attempts WHERE state = 'RUNNING' AND terminal = 0 ORDER BY attempt_id
                """
            ).fetchall()
        for row in rows:
            identity_json = cast(str | None, row[4])
            identity = _json_mapping(identity_json) if identity_json is not None else None
            if identity is not None and process_exists(identity):
                continue
            attempt = str(row[0])
            campaign = str(row[1])
            contract = str(row[2])
            expected_revision = int(row[3])
            with self._transaction() as connection:
                now = self._timestamp()
                active_lease = connection.execute(
                    """
                    SELECT 1 FROM leases
                    WHERE completed = 0 AND expires_at > ? AND (
                        (resource_kind = 'campaign' AND resource_id = ?)
                        OR (resource_kind = 'attempt' AND resource_id = ?)
                    )
                    """,
                    (now, campaign, attempt),
                ).fetchone()
                if active_lease is not None:
                    continue
                updated = connection.execute(
                    """
                    UPDATE attempts SET state = 'INTERRUPTED', revision = revision + 1,
                        terminal = 1, updated_at = ?
                    WHERE attempt_id = ? AND campaign_id = ? AND contract_id = ?
                          AND state = 'RUNNING' AND revision = ? AND terminal = 0
                    """,
                    (now, attempt, campaign, contract, expected_revision),
                )
                if updated.rowcount != 1:
                    continue
                failure_id = hashlib.sha256(
                    f"interrupted:{contract}:{campaign}:{attempt}:{expected_revision + 1}".encode()
                ).hexdigest()
                payload_json = _canonical_json(
                    {"reason": "durable process identity is absent"}, "interruption payload"
                )
                connection.execute(
                    """
                    INSERT INTO failures(
                        failure_id, contract_id, campaign_id, entity_kind, entity_id,
                        failure_kind, payload_json, created_at
                    ) VALUES (?, ?, ?, 'attempt', ?, 'INTERRUPTED', ?, ?)
                    """,
                    (failure_id, contract, campaign, attempt, payload_json, now),
                )
                self._append_event(
                    connection,
                    campaign_id=campaign,
                    event_type="attempt_interrupted_during_recovery",
                    entity_kind="attempt",
                    entity_id=attempt,
                    prior_state="RUNNING",
                    new_state="INTERRUPTED",
                    revision=expected_revision + 1,
                    payload_json=payload_json,
                    now=now,
                )
                interrupted.append(attempt)
        return Reconciliation(tuple(interrupted))

    def inspect(self, *, campaign_id: IdInput) -> Mapping[str, object]:
        """Return a read-only projection assembled solely from the authority database."""

        from kuairand_agent.state.projections import inspect_campaign

        campaign = _identifier(campaign_id, "campaign_id")
        with self._readonly() as connection:
            return inspect_campaign(connection, campaign_id=campaign)

    def verify_artifact(self, *, artifact_id: IdInput, path: Path) -> None:
        """Verify a committed artifact's exact bytes before a caller reuses it."""

        identity = _identifier(artifact_id, "artifact_id")
        with self._readonly() as connection:
            row = connection.execute(
                "SELECT sha256, size_bytes FROM artifacts WHERE artifact_id = ?", (identity,)
            ).fetchone()
        if row is None:
            raise StateNotFoundError("artifact does not exist")
        _verify_artifact(path, expected_sha256=str(row[0]), expected_size=int(row[1]))

    def rebuild_projection(self, *, campaign_id: IdInput, destination: Path) -> Path:
        """Atomically rebuild a deletable JSON projection from SQLite truth."""

        from kuairand_agent.state.projections import write_campaign_projection

        if not isinstance(destination, Path):
            raise StateInvariantError("destination must be a pathlib.Path")
        authority_targets = (
            self._database_path,
            Path(f"{self._database_path}-wal"),
            Path(f"{self._database_path}-shm"),
        )
        resolved_destination = destination.resolve(strict=False)
        for authority_target in authority_targets:
            if resolved_destination == authority_target.resolve(strict=False) or (
                destination.exists()
                and authority_target.exists()
                and os.path.samefile(destination, authority_target)
            ):
                raise StateInvariantError("projection destination cannot replace authority files")
        snapshot = self.inspect(campaign_id=campaign_id)
        return write_campaign_projection(snapshot, destination=destination)

    def _assert_record_references(
        self, connection: sqlite3.Connection, record: _NormalizedRecord
    ) -> None:
        checks: list[tuple[str, str, str]] = []
        refs = record.references
        kind = record.kind
        if kind is RecordKind.EXPERIMENT:
            checks.append(("families", "family_id", refs["family_id"]))
            if "parent_experiment_id" in refs:
                checks.append(
                    ("campaign_experiments", "experiment_id", refs["parent_experiment_id"])
                )
        elif kind is RecordKind.TRIAL:
            checks.append(("campaign_experiments", "experiment_id", refs["experiment_id"]))
        elif kind is RecordKind.ATTEMPT:
            checks.append(("trials", "trial_id", refs["trial_id"]))
        elif kind is RecordKind.ARTIFACT and "attempt_id" in refs:
            checks.append(("attempts", "attempt_id", refs["attempt_id"]))
        elif kind is RecordKind.PREDICTION:
            checks.append(("artifacts", "artifact_id", refs["artifact_id"]))
            if "trial_id" in refs:
                checks.append(("trials", "trial_id", refs["trial_id"]))
            else:
                checks.append(("rank_graphs", "rank_graph_id", refs["rank_graph_id"]))
        elif kind is RecordKind.RANK_GRAPH:
            checks.append(("families", "family_id", refs["family_id"]))
        elif kind in {
            RecordKind.INNER_EVALUATION,
            RecordKind.PROMOTION_DECISION,
            RecordKind.REPLAY,
        }:
            checks.append(("predictions", "prediction_id", refs["prediction_id"]))
        elif kind is RecordKind.BUNDLE:
            checks.extend(
                (
                    ("replays", "replay_id", refs["replay_id"]),
                    (
                        "predictions",
                        "prediction_id",
                        refs["selected_prediction_id"],
                    ),
                )
            )
        elif kind is RecordKind.RESOURCE_RECEIPT and "attempt_id" in refs:
            checks.append(("attempts", "attempt_id", refs["attempt_id"]))
        elif kind is RecordKind.FAILURE:
            entity_kind = cast(str, record.attributes["entity_kind"])
            target = _FAILURE_ENTITY_TARGETS.get(entity_kind)
            if target is None:
                raise StateInvariantError("failure entity_kind is not a durable entity kind")
            table, key = target
            checks.append((table, key, refs["entity_id"]))
        for table, key, identity in checks:
            row = connection.execute(
                f"""
                SELECT 1 FROM {table}
                WHERE {key} = ? AND campaign_id = ? AND contract_id = ?
                """,
                (identity, record.campaign_id, record.contract_id),
            ).fetchone()
            if row is None:
                raise StateInvariantError(
                    f"{record.kind.value} reference {key} is outside campaign lineage"
                )

    def _assert_finalization_quiescent(
        self,
        connection: sqlite3.Connection,
        *,
        campaign_id: str,
        replay_id: str,
        bundle_id: str,
        now: str,
    ) -> None:
        open_work = int(
            connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM trials
                     WHERE campaign_id = ? AND terminal = 0)
                  + (SELECT COUNT(*) FROM attempts
                     WHERE campaign_id = ? AND terminal = 0)
                  + (SELECT COUNT(*) FROM provider_operations
                     WHERE campaign_id = ? AND terminal = 0)
                  + (SELECT COUNT(*) FROM protected_query_reservations
                     WHERE campaign_id = ? AND state = 'RESERVED')
                  + (SELECT COUNT(*) FROM replays
                     WHERE campaign_id = ? AND terminal = 0 AND replay_id <> ?)
                  + (SELECT COUNT(*) FROM bundles
                     WHERE campaign_id = ? AND terminal = 0 AND bundle_id <> ?)
                """,
                (
                    campaign_id,
                    campaign_id,
                    campaign_id,
                    campaign_id,
                    campaign_id,
                    replay_id,
                    campaign_id,
                    bundle_id,
                ),
            ).fetchone()[0]
        )
        live_resource_leases = sum(
            campaign_id
            in self._lease_campaign_ids(
                connection, resource_kind=str(row[0]), resource_id=str(row[1])
            )
            for row in connection.execute(
                """
                SELECT resource_kind, resource_id FROM leases
                WHERE completed = 0 AND expires_at > ?
                  AND NOT (resource_kind = 'campaign' AND resource_id = ?)
                """,
                (now, campaign_id),
            ).fetchall()
        )
        if open_work or live_resource_leases:
            raise StateInvariantError(
                "campaign finalization requires all other work and resource leases to be terminal"
            )

    def _assert_write_fence(
        self,
        connection: sqlite3.Connection,
        *,
        campaign_id: str,
        lease: Lease | None,
        now: str,
        resource_kind: str | None = None,
        resource_id: str | None = None,
    ) -> None:
        scoped_resource_id = (
            _association_lease_resource_id(resource_kind, campaign_id, resource_id)
            if resource_kind in {"family", "experiment"} and resource_id is not None
            else resource_id
        )
        rows = connection.execute(
            """
            SELECT resource_kind, resource_id, owner_id, fence_token, expires_at, completed
            FROM leases
            WHERE (resource_kind = 'campaign' AND resource_id = ?)
               OR (resource_kind = ? AND resource_id = ?)
            """,
            (campaign_id, resource_kind, scoped_resource_id),
        ).fetchall()
        if not rows:
            if lease is not None:
                raise LeaseConflictError("supplied write fence does not exist")
            return
        live_rows = [row for row in rows if not bool(row[5]) and str(row[4]) > now]
        if len(live_rows) > 1:
            raise LeaseConflictError("overlapping campaign and resource write fences")
        if not live_rows:
            raise LeaseConflictError("mutation requires a newly claimed durable write fence")
        row = live_rows[0]
        if lease is None:
            raise LeaseConflictError("mutation requires the current durable write fence")
        expected = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            int(row[3]),
        )
        supplied = (
            lease.resource_kind,
            lease.resource_id,
            lease.owner_id,
            lease.fence_token,
        )
        if supplied != expected or bool(row[5]) or str(row[4]) <= now or lease.completed:
            raise LeaseConflictError("stale or completed write fence rejected mutation")

    def _assert_lease_scope_available(
        self,
        connection: sqlite3.Connection,
        *,
        resource_kind: str,
        resource_id: str,
        now: str,
    ) -> None:
        if resource_kind == "campaign":
            campaign_exists = connection.execute(
                "SELECT 1 FROM campaigns WHERE campaign_id = ?", (resource_id,)
            ).fetchone()
            if campaign_exists is None:
                raise StateNotFoundError("campaign lease target does not exist")
            live_descendants = connection.execute(
                """
                SELECT resource_kind, resource_id FROM leases
                WHERE resource_kind <> 'campaign' AND completed = 0 AND expires_at > ?
                """,
                (now,),
            ).fetchall()
            for descendant in live_descendants:
                descendant_campaigns = self._lease_campaign_ids(
                    connection,
                    resource_kind=str(descendant[0]),
                    resource_id=str(descendant[1]),
                )
                if resource_id in descendant_campaigns:
                    raise LeaseConflictError(
                        "campaign has a live descendant lease owned by another scope"
                    )
            return
        campaign_ids = self._lease_campaign_ids(
            connection,
            resource_kind=resource_kind,
            resource_id=resource_id,
        )
        if not campaign_ids:
            if resource_kind in _LEASE_RESOURCE_TABLES:
                raise StateNotFoundError(f"{resource_kind} lease target does not exist")
            return
        live_campaign = connection.execute(
            f"""
            SELECT 1 FROM leases
            WHERE resource_kind = 'campaign'
                  AND resource_id IN ({", ".join("?" for _ in campaign_ids)})
                  AND completed = 0 AND expires_at > ?
            """,
            (*campaign_ids, now),
        ).fetchone()
        if live_campaign is not None:
            raise LeaseConflictError("resource belongs to a campaign with a live lease")

    def _lease_campaign_ids(
        self,
        connection: sqlite3.Connection,
        *,
        resource_kind: str,
        resource_id: str,
    ) -> tuple[str, ...]:
        target = _LEASE_RESOURCE_TABLES.get(resource_kind)
        if target is None:
            return ()
        if resource_kind in {"family", "experiment"}:
            table, primary_key = target
            associations = connection.execute(
                f"SELECT campaign_id, {primary_key} FROM {table}"
            ).fetchall()
            return tuple(
                str(row[0])
                for row in associations
                if _association_lease_resource_id(resource_kind, str(row[0]), str(row[1]))
                == resource_id
            )
        table, primary_key = target
        rows = connection.execute(
            f"SELECT campaign_id FROM {table} WHERE {primary_key} = ?", (resource_id,)
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _insert_record(
        self, connection: sqlite3.Connection, record: _NormalizedRecord, *, now: str
    ) -> bool:
        if record.kind is RecordKind.EXPERIMENT:
            return self._insert_experiment_association(connection, record, now=now)
        values = _record_insert(record, now=now)
        primary_key = _primary_key(record.table)
        if record.kind is RecordKind.FAMILY:
            existing = connection.execute(
                "SELECT * FROM families WHERE campaign_id = ? AND family_id = ?",
                (record.campaign_id, record.record_id),
            ).fetchone()
        else:
            existing = connection.execute(
                f"SELECT * FROM {record.table} WHERE {primary_key} = ?", (record.record_id,)
            ).fetchone()
        if existing is not None:
            cursor = connection.execute(f"SELECT * FROM {record.table} LIMIT 0")
            if cursor.description is None:  # pragma: no cover - SELECT always describes columns
                raise StateInvariantError("state table has no column description")
            existing_columns = [description[0] for description in cursor.description]
            existing_map = dict(zip(existing_columns, existing, strict=True))
            ignored = {"created_at", "updated_at"}
            if record.kind in _STATEFUL_RECORD_KINDS:
                ignored.update({"state", "revision", "terminal", "payload_json"})
            if record.kind is RecordKind.ATTEMPT:
                ignored.add("process_identity_json")
            for key, value in values.items():
                if key in ignored:
                    continue
                if existing_map[key] != value:
                    raise StateInvariantError(
                        f"{record.kind.value} identity already binds different immutable content"
                    )
            if record.kind in _STATEFUL_RECORD_KINDS:
                registration = connection.execute(
                    """
                    SELECT new_state, payload_json FROM campaign_events
                    WHERE campaign_id = ? AND entity_kind = ? AND entity_id = ?
                          AND event_type = ?
                    ORDER BY event_seq LIMIT 1
                    """,
                    (
                        record.campaign_id,
                        record.kind.value,
                        record.record_id,
                        f"{record.kind.value}_registered",
                    ),
                ).fetchone()
                if registration is None or (str(registration[0]), str(registration[1])) != (
                    record.state,
                    _registration_payload_json(record),
                ):
                    raise StateInvariantError(
                        f"{record.kind.value} initial registration differs from idempotent replay"
                    )
            return False
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        connection.execute(
            f"INSERT INTO {record.table}({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        return True

    def _record_exists(self, connection: sqlite3.Connection, record: _NormalizedRecord) -> bool:
        if record.kind is RecordKind.FAMILY:
            row = connection.execute(
                "SELECT 1 FROM families WHERE campaign_id = ? AND family_id = ?",
                (record.campaign_id, record.record_id),
            ).fetchone()
        elif record.kind is RecordKind.EXPERIMENT:
            row = connection.execute(
                """
                SELECT 1 FROM campaign_experiments
                WHERE campaign_id = ? AND experiment_id = ?
                """,
                (record.campaign_id, record.record_id),
            ).fetchone()
        else:
            row = connection.execute(
                f"SELECT 1 FROM {record.table} WHERE {_primary_key(record.table)} = ?",
                (record.record_id,),
            ).fetchone()
        return row is not None

    def _insert_experiment_association(
        self,
        connection: sqlite3.Connection,
        record: _NormalizedRecord,
        *,
        now: str,
    ) -> bool:
        references = record.references
        association = connection.execute(
            """
            SELECT contract_id, family_id, parent_experiment_id
            FROM campaign_experiments
            WHERE campaign_id = ? AND experiment_id = ?
            """,
            (record.campaign_id, record.record_id),
        ).fetchone()
        expected_association = (
            record.contract_id,
            references["family_id"],
            references.get("parent_experiment_id"),
        )
        if association is not None:
            if tuple(association) != expected_association:
                raise StateInvariantError(
                    "campaign experiment association already binds different lineage"
                )
            return False
        connection.execute(
            """
            INSERT INTO campaign_experiments(
                campaign_id, contract_id, experiment_id, family_id,
                parent_experiment_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.campaign_id,
                record.contract_id,
                record.record_id,
                references["family_id"],
                references.get("parent_experiment_id"),
                now,
            ),
        )
        return True

    def _ensure_family_ledger(
        self,
        connection: sqlite3.Connection,
        record: _NormalizedRecord,
        *,
        now: str,
    ) -> None:
        protected_eligible = int(cast(bool, record.attributes["protected_eligible"]))
        existing = connection.execute(
            """
            SELECT protected_eligible, payload_json FROM family_ledger
            WHERE contract_id = ? AND family_id = ?
            """,
            (record.contract_id, record.record_id),
        ).fetchone()
        if existing is not None:
            if (int(existing[0]), str(existing[1])) != (
                protected_eligible,
                record.payload_json,
            ):
                raise StateInvariantError(
                    "(ContractId, FamilyId) ledger already binds different immutable content"
                )
            return
        connection.execute(
            """
            INSERT INTO family_ledger(
                contract_id, family_id, protected_eligible, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.contract_id,
                record.record_id,
                protected_eligible,
                record.payload_json,
                now,
            ),
        )

    def _ensure_experiment_ledger(
        self,
        connection: sqlite3.Connection,
        record: _NormalizedRecord,
        *,
        now: str,
    ) -> None:
        references = record.references
        expected = (
            references["family_id"],
            references.get("parent_experiment_id"),
            record.payload_json,
        )
        existing = connection.execute(
            """
            SELECT family_id, parent_experiment_id, payload_json
            FROM experiment_ledger
            WHERE contract_id = ? AND experiment_id = ?
            """,
            (record.contract_id, record.record_id),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != expected:
                raise StateInvariantError(
                    "(ContractId, ExperimentId) ledger already binds different immutable semantics"
                )
            return
        connection.execute(
            """
            INSERT INTO experiment_ledger(
                contract_id, experiment_id, family_id, parent_experiment_id,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.contract_id,
                record.record_id,
                references["family_id"],
                references.get("parent_experiment_id"),
                record.payload_json,
                now,
            ),
        )

    def _append_event(
        self,
        connection: sqlite3.Connection,
        *,
        campaign_id: str,
        event_type: str,
        entity_kind: str,
        entity_id: str,
        prior_state: str | None,
        new_state: str | None,
        revision: int,
        payload_json: str,
        now: str,
        event_seq: int | None = None,
    ) -> int:
        if event_seq is None:
            event_seq = self._allocate_event_seq(connection, campaign_id=campaign_id)
        else:
            existing = connection.execute(
                """
                SELECT 1 FROM campaign_events WHERE campaign_id = ? AND event_seq = ?
                """,
                (campaign_id, event_seq),
            ).fetchone()
            if existing is not None:
                raise StateConflictError("preallocated campaign event sequence is already used")
        connection.execute(
            """
            INSERT INTO campaign_events(
                campaign_id, event_seq, event_type, entity_kind, entity_id, prior_state,
                new_state, revision, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                event_seq,
                event_type,
                entity_kind,
                entity_id,
                prior_state,
                new_state,
                revision,
                payload_json,
                now,
            ),
        )
        return event_seq

    def _allocate_event_seq(self, connection: sqlite3.Connection, *, campaign_id: str) -> int:
        row = connection.execute(
            "SELECT next_event_seq FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise StateNotFoundError("campaign does not exist for event")
        event_seq = int(row[0])
        updated = connection.execute(
            """
            UPDATE campaigns SET next_event_seq = next_event_seq + 1
            WHERE campaign_id = ? AND next_event_seq = ?
            """,
            (campaign_id, event_seq),
        )
        if updated.rowcount != 1:
            raise StateConflictError("campaign event sequence allocation conflicted")
        return event_seq

    def _freeze_research(
        self, connection: sqlite3.Connection, *, campaign_id: str, now: str
    ) -> int:
        updated = connection.execute(
            """
            UPDATE campaigns
            SET revision = revision + 1, research_frozen = 1, updated_at = ?
            WHERE campaign_id = ? AND terminal = 0
            """,
            (now, campaign_id),
        )
        if updated.rowcount != 1:
            raise StateConflictError("campaign is missing or terminal")
        return int(
            connection.execute(
                "SELECT revision FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()[0]
        )

    def _assert_campaign_lineage(
        self,
        connection: sqlite3.Connection,
        *,
        campaign_id: str,
        contract_id: str,
        writable: bool,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT campaign_id, contract_id, revision, research_frozen, terminal
            FROM campaigns WHERE campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()
        if row is None:
            raise StateNotFoundError("campaign does not exist")
        if str(row["contract_id"]) != contract_id:
            raise StateInvariantError("record ContractId differs from campaign ContractId")
        if writable and bool(row["terminal"]):
            raise StateInvariantError("terminal campaign is immutable")
        return cast(sqlite3.Row, row)

    def _raise_transition_conflict(
        self,
        connection: sqlite3.Connection,
        *,
        table: str,
        entity_id: str,
        campaign_id: str,
    ) -> None:
        row = connection.execute(
            f"""
            SELECT state, revision, terminal FROM {table}
            WHERE {_primary_key(table)} = ? AND campaign_id = ?
            """,
            (entity_id, campaign_id),
        ).fetchone()
        if row is None:
            raise StateNotFoundError(f"{table} record does not exist")
        raise StateConflictError(
            "compare-and-swap conflict: "
            f"state={row[0]!r}, revision={row[1]}, terminal={bool(row[2])}"
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise StateInvariantError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _timestamp(self) -> str:
        return _timestamp(self._now())

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            configure_connection(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise StateInvariantError("authority constraint rejected mutation") from exc
            except BaseException:
                connection.rollback()
                raise

    @contextlib.contextmanager
    def _readonly(self) -> Iterator[sqlite3.Connection]:
        uri = f"file:{self._database_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN")
        try:
            yield connection
        finally:
            connection.rollback()
            connection.close()


def _prepared_terminal_projection(
    row: sqlite3.Row, *, campaign_id: str, created: bool
) -> PreparedTerminalProjection:
    projection = _json_mapping(str(row[16]))
    return PreparedTerminalProjection(
        preparation_id=str(row[0]),
        campaign_id=campaign_id,
        contract_id=str(row[1]),
        source=CampaignSourceRevision(int(row[3]), int(row[4])),
        terminal_state=str(row[5]),
        projection_schema_version=int(row[13]),
        redaction_policy_version=int(row[14]),
        projection_sha256=str(row[15]),
        projection=projection,
        created=created,
    )


def _build_terminal_projection(
    connection: sqlite3.Connection,
    *,
    campaign_id: str,
    contract_id: str,
    source_state: str,
    source_revision: int,
    source_last_event_seq: int,
    terminal_state: str,
    decision_id: str,
    replay_id: str,
    selected_prediction_id: str,
    fallback_prediction_id: str,
    decision_json: str,
    replay_json: str,
    bundle_claims_json: str,
    prepared_at: str,
) -> Mapping[str, object]:
    from kuairand_agent.state.projections import inspect_campaign

    snapshot = json.loads(
        canonical_json_bytes(inspect_campaign(connection, campaign_id=campaign_id)).decode()
    )
    if not isinstance(snapshot, dict):  # pragma: no cover - projection is always a mapping
        raise StateInvariantError("campaign projection is not a JSON object")
    campaign = cast(dict[str, object], snapshot["campaign"])
    campaign.update(
        {
            "state": terminal_state,
            "revision": source_revision + 1,
            "next_event_seq": source_last_event_seq + 4,
            "terminal": True,
            "selected_prediction_id": selected_prediction_id,
            "fallback_prediction_id": fallback_prediction_id,
            "updated_at": prepared_at,
            "completed_at": prepared_at,
        }
    )
    entities = cast(dict[str, object], snapshot["entities"])
    decisions = cast(list[object], entities["selection_decisions"])
    decisions.append(
        {
            "decision_id": decision_id,
            "contract_id": contract_id,
            "campaign_id": campaign_id,
            "selected_prediction_id": selected_prediction_id,
            "fallback_prediction_id": fallback_prediction_id,
            "payload": json.loads(decision_json),
            "created_at": prepared_at,
        }
    )
    replays = cast(list[object], entities["replays"])
    existing_replay = next(
        (
            cast(dict[str, object], candidate)
            for candidate in replays
            if isinstance(candidate, dict) and candidate.get("replay_id") == replay_id
        ),
        None,
    )
    if existing_replay is None:
        replay_prior_state: str | None = None
        replay_revision = 0
        replays.append(
            {
                "replay_id": replay_id,
                "contract_id": contract_id,
                "campaign_id": campaign_id,
                "prediction_id": selected_prediction_id,
                "state": "VERIFIED",
                "revision": replay_revision,
                "terminal": True,
                "payload": json.loads(replay_json),
                "created_at": prepared_at,
                "updated_at": prepared_at,
            }
        )
    else:
        replay_prior_state = cast(str, existing_replay["state"])
        replay_revision = cast(int, existing_replay["revision"]) + 1
        existing_replay.update(
            {
                "state": "VERIFIED",
                "revision": replay_revision,
                "terminal": True,
                "payload": json.loads(replay_json),
                "updated_at": prepared_at,
            }
        )
    bundles = cast(list[object], entities["bundles"])
    bundles.append(
        {
            "availability": _SELF_REFERENCE_MARKER,
            "contract_id": contract_id,
            "campaign_id": campaign_id,
            "replay_id": replay_id,
            "selected_prediction_id": selected_prediction_id,
            "state": "SEALED",
            "revision": 0,
            "terminal": True,
            "claims": json.loads(bundle_claims_json),
        }
    )
    events = cast(list[object], snapshot["events"])
    events.extend(
        (
            {
                "campaign_id": campaign_id,
                "event_seq": source_last_event_seq + 1,
                "event_type": "replay_verified",
                "entity_kind": "replay",
                "entity_id": replay_id,
                "prior_state": replay_prior_state,
                "new_state": "VERIFIED",
                "revision": replay_revision,
                "payload": json.loads(replay_json),
                "created_at": prepared_at,
            },
            {
                "campaign_id": campaign_id,
                "event_seq": source_last_event_seq + 2,
                "event_type": "bundle_sealed",
                "entity_kind": "bundle",
                "entity_id": _SELF_REFERENCE_MARKER,
                "prior_state": None,
                "new_state": "SEALED",
                "revision": 0,
                "payload": {
                    "availability": _SELF_REFERENCE_MARKER,
                    "claims": json.loads(bundle_claims_json),
                    "replay_id": replay_id,
                    "selected_prediction_id": selected_prediction_id,
                },
                "created_at": prepared_at,
            },
            {
                "campaign_id": campaign_id,
                "event_seq": source_last_event_seq + 3,
                "event_type": "campaign_finalized",
                "entity_kind": "campaign",
                "entity_id": campaign_id,
                "prior_state": source_state,
                "new_state": terminal_state,
                "revision": source_revision + 1,
                "payload": {
                    "bundle_availability": _SELF_REFERENCE_MARKER,
                    "decision_id": decision_id,
                    "fallback_prediction_id": fallback_prediction_id,
                    "replay_id": replay_id,
                    "selected_prediction_id": selected_prediction_id,
                },
                "created_at": prepared_at,
            },
        )
    )
    snapshot["terminal_projection"] = {
        "schema_version": _TERMINAL_PROJECTION_SCHEMA_VERSION,
        "source_revision": {
            "campaign_revision": source_revision,
            "last_event_seq": source_last_event_seq,
        },
        "redaction_policy": {
            "version": _TERMINAL_REDACTION_POLICY_VERSION,
            "reason": "exclude content-addressed bundle self-reference",
            "excluded_fields": [
                "bundle.bundle_id",
                "bundle.manifest_sha256",
                "bundle.publication",
                "events.bundle_sealed.entity_id",
                "events.campaign_finalized.payload.bundle_id",
                "actual_terminal_timestamps",
            ],
        },
    }
    return MappingProxyType(snapshot)


def _reject_publication_fields(value: object, *, path: str = "bundle_claims") -> None:
    if path == "bundle_claims" and not isinstance(value, Mapping):
        raise StateInvariantError("bundle_claims must be a mapping")
    if isinstance(value, Mapping):
        for key, candidate in value.items():
            if not isinstance(key, str):
                raise StateInvariantError("bundle_claims keys must be strings")
            normalized = key.lower().replace("-", "_")
            if normalized in _PUBLICATION_FIELDS or (
                ("bundle" in normalized or "manifest" in normalized or "publication" in normalized)
                and any(token in normalized for token in ("id", "sha", "hash", "path", "uri"))
            ):
                raise StateInvariantError(f"{path}.{key} is a future publication self-reference")
            _reject_publication_fields(candidate, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, candidate in enumerate(value):
            _reject_publication_fields(candidate, path=f"{path}[{index}]")


def _write_terminal_projection_sqlite(
    destination: Path,
    *,
    preparation_id: str,
    projection_sha256: str,
    projection: Mapping[str, object],
    preserve_numbers: bool = False,
) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        projection_json = (
            _canonical_json_preserve_numbers(dict(projection), "terminal projection")
            if preserve_numbers
            else canonical_json_bytes(dict(projection)).decode()
        )
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode = DELETE;
                PRAGMA synchronous = FULL;
                CREATE TABLE projection_metadata(
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    preparation_id TEXT NOT NULL,
                    projection_sha256 TEXT NOT NULL,
                    projection_schema_version INTEGER NOT NULL,
                    redaction_policy_version INTEGER NOT NULL,
                    projection_json TEXT NOT NULL
                ) STRICT;
                CREATE TABLE campaigns(
                    campaign_id TEXT PRIMARY KEY,
                    contract_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    next_event_seq INTEGER NOT NULL,
                    terminal INTEGER NOT NULL,
                    selected_prediction_id TEXT,
                    fallback_prediction_id TEXT,
                    completed_at TEXT,
                    projection_json TEXT NOT NULL
                ) STRICT;
                CREATE TABLE campaign_events(
                    campaign_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    entity_kind TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    prior_state TEXT,
                    new_state TEXT,
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(campaign_id, event_seq)
                ) STRICT;
                CREATE TABLE entity_records(
                    entity_kind TEXT NOT NULL,
                    record_ordinal INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(entity_kind, record_ordinal)
                ) STRICT;
                """
            )
            metadata = cast(Mapping[str, object], projection["terminal_projection"])
            redaction = cast(Mapping[str, object], metadata["redaction_policy"])
            connection.execute(
                """
                INSERT INTO projection_metadata VALUES (1, ?, ?, ?, ?, ?)
                """,
                (
                    preparation_id,
                    projection_sha256,
                    int(cast(int, metadata["schema_version"])),
                    int(cast(int, redaction["version"])),
                    projection_json,
                ),
            )
            campaign = cast(Mapping[str, object], projection["campaign"])
            connection.execute(
                """
                INSERT INTO campaigns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign["campaign_id"],
                    campaign["contract_id"],
                    campaign["state"],
                    campaign["revision"],
                    campaign["next_event_seq"],
                    int(cast(bool, campaign["terminal"])),
                    campaign.get("selected_prediction_id"),
                    campaign.get("fallback_prediction_id"),
                    campaign.get("completed_at"),
                    canonical_json_bytes(dict(campaign)).decode(),
                ),
            )
            for event in cast(list[Mapping[str, object]], projection["events"]):
                connection.execute(
                    """
                    INSERT INTO campaign_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["campaign_id"],
                        event["event_seq"],
                        event["event_type"],
                        event["entity_kind"],
                        event["entity_id"],
                        event.get("prior_state"),
                        event.get("new_state"),
                        event["revision"],
                        canonical_json_bytes(event.get("payload", {})).decode(),
                        event["created_at"],
                    ),
                )
            entities = cast(Mapping[str, object], projection["entities"])
            for kind in sorted(entities):
                for ordinal, record in enumerate(cast(list[object], entities[kind])):
                    connection.execute(
                        "INSERT INTO entity_records VALUES (?, ?, ?)",
                        (kind, ordinal, canonical_json_bytes(record).decode()),
                    )
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise StateInvariantError("prepared SQLite projection failed integrity check")
        finally:
            connection.close()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        return _publish_existing_temp(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_publish_bytes(destination: Path, payload: bytes) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return _publish_existing_temp(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_existing_temp(temporary: Path, destination: Path) -> tuple[str, int]:
    candidate_sha, candidate_size = _hash_regular_file(temporary)
    if destination.exists():
        existing_sha, existing_size = _hash_regular_file(destination)
        if (existing_sha, existing_size) != (candidate_sha, candidate_size):
            raise StateConflictError("existing prepared projection artifact differs")
        os.chmod(destination, 0o444, follow_symlinks=False)
        temporary.unlink()
        return existing_sha, existing_size
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError:
        existing_sha, existing_size = _hash_regular_file(destination)
        if (existing_sha, existing_size) != (candidate_sha, candidate_size):
            raise StateConflictError("concurrent prepared projection artifact differs") from None
    os.chmod(destination, 0o444, follow_symlinks=False)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    temporary.unlink()
    published_sha, published_size = _hash_regular_file(destination)
    if (published_sha, published_size) != (candidate_sha, candidate_size):
        raise StateConflictError("prepared projection changed during atomic publication")
    return published_sha, published_size


def _hash_regular_file(path: Path) -> tuple[str, int]:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PublishedBundleVerificationError("published member is not a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise PublishedBundleVerificationError("published member is not stably readable") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PublishedBundleVerificationError("published member changed during verification")
    return digest.hexdigest(), after.st_size


def _prepared_replay_binds_receipt(
    prepared_replay_json: str, replay_receipt: Mapping[str, object]
) -> bool:
    try:
        prepared = json.loads(prepared_replay_json)
    except json.JSONDecodeError as exc:  # pragma: no cover - repository stores canonical JSON
        raise StateInvariantError("stored prepared replay payload is malformed") from exc
    if prepared == replay_receipt:
        return True
    if not isinstance(prepared, dict):  # pragma: no cover - canonical mapping invariant
        return False
    return any(
        key.endswith("replay_receipt") and candidate == replay_receipt
        for key, candidate in prepared.items()
    )


def _validate_scripted_replay_receipt(
    value: object,
    *,
    contract_id: str,
    campaign_id: str,
    selected_prediction_id: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != _SCRIPTED_REPLAY_RECEIPT_FIELDS:
        raise StateInvariantError("scripted replay receipt does not match the exact schema")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise StateInvariantError("scripted replay receipt schema_version must be integer 1")
    if type(value.get("grade")) is not str:
        raise StateInvariantError("scripted replay receipt grade must be text")
    if type(value.get("qualification_scope")) is not str:
        raise StateInvariantError("scripted replay receipt qualification scope must be text")
    if value.get("exact_prediction_bytes") is not True or any(
        value.get(field) is not False
        for field in (
            "exact_metrics_recomputed",
            "protected_metrics_evaluated",
            "official_fm_qualified",
            "full_data_qualified",
        )
    ):
        raise StateInvariantError("scripted replay receipt overstates verified evidence")
    try:
        grade = ReplayGrade(cast(str, value["grade"]))
        receipt = ScriptedReplayReceipt(
            contract_id=_sha256(value.get("contract_id"), "scripted contract_id"),
            campaign_id=_sha256(value.get("campaign_id"), "scripted campaign_id"),
            prediction_id=_sha256(value.get("prediction_id"), "scripted prediction_id"),
            first_prediction_sha256=_sha256(
                value.get("first_prediction_sha256"), "scripted first prediction digest"
            ),
            replay_prediction_sha256=_sha256(
                value.get("replay_prediction_sha256"), "scripted replay prediction digest"
            ),
            first_result_sha256=_sha256(
                value.get("first_result_sha256"), "scripted first result digest"
            ),
            replay_result_sha256=_sha256(
                value.get("replay_result_sha256"), "scripted replay result digest"
            ),
            grade=grade,
            qualification_scope=cast(str, value["qualification_scope"]),
            schema_version=cast(int, value["schema_version"]),
        )
    except (KeyError, ReceiptError, ValueError) as exc:
        raise StateInvariantError("scripted replay receipt evidence is invalid") from exc
    normalized = receipt.manifest()
    if canonical_json_bytes(dict(value)) != canonical_json_bytes(normalized):
        raise StateInvariantError("scripted replay receipt identity or fields are forged")
    if (
        receipt.contract_id,
        receipt.campaign_id,
        receipt.prediction_id,
    ) != (contract_id, campaign_id, selected_prediction_id):
        raise StateInvariantError("scripted replay receipt is outside terminal lineage")
    return normalized


def _normalize_offline_fixture_bundle_claims(
    value: Mapping[str, object],
) -> dict[str, object]:
    if frozenset(value) != _OFFLINE_FIXTURE_BUNDLE_CLAIM_FIELDS:
        raise StateInvariantError("offline fixture bundle claims do not match the exact schema")
    resource_receipt_id = _sha256(value.get("resource_receipt_id"), "resource_receipt_id")
    if (
        type(value.get("replay_grades")) is not list
        or tuple(cast(list[object], value["replay_grades"])) != _OFFLINE_FIXTURE_REPLAY_GRADES
    ):
        raise StateInvariantError("offline fixture bundle claims contain unsupported replay grades")
    if type(value.get("protected_query_count")) is not int:
        raise StateInvariantError("offline fixture protected query count must be an integer")
    normalized: dict[str, object] = {
        "resource_receipt_id": resource_receipt_id,
        "replay_grade": ReplayGrade.BUNDLE_EXACT.value,
        "replay_grades": list(_OFFLINE_FIXTURE_REPLAY_GRADES),
        "submission_disposition": "SCRIPTED_FALLBACK_RETAINED",
        "scientific_disposition": "INSUFFICIENT_VALID_EVIDENCE",
        "campaign_kind": "OFFLINE_SCRIPTED_FIXTURE",
        "qualification_scope": "SCRIPTED_FIXTURE_ONLY",
        "protected_query_count": 0,
        "exact_metrics": None,
    }
    if canonical_json_bytes(dict(value)) != canonical_json_bytes(normalized):
        raise StateInvariantError("offline fixture bundle claims overstate verified evidence")
    return normalized


def _normalize_offline_fixture_evidence(
    *,
    contract_id: str,
    campaign_id: str,
    selected_prediction_id: str,
    replay_payload: Mapping[str, object],
    bundle_claims: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if frozenset(replay_payload) != _OFFLINE_FIXTURE_REPLAY_INPUT_FIELDS:
        raise StateInvariantError("offline fixture replay payload does not match the exact schema")
    if (
        replay_payload.get("contract_id"),
        replay_payload.get("campaign_id"),
        replay_payload.get("prediction_id"),
    ) != (contract_id, campaign_id, selected_prediction_id):
        raise StateInvariantError("offline fixture replay payload is outside terminal lineage")
    grades = replay_payload.get("replay_grades")
    if type(grades) is not list or tuple(cast(list[object], grades)) != (
        _OFFLINE_FIXTURE_REPLAY_GRADES
    ):
        raise StateInvariantError("offline fixture replay payload contains unsupported grades")
    receipt = _validate_scripted_replay_receipt(
        replay_payload.get("scripted_replay_receipt"),
        contract_id=contract_id,
        campaign_id=campaign_id,
        selected_prediction_id=selected_prediction_id,
    )
    journaled_replay = {
        "schema_version": 1,
        "verifier": "SCRIPTED_REPLAY_RECEIPT",
        "grade": receipt["grade"],
        "qualification_scope": receipt["qualification_scope"],
        "scripted_replay_receipt": receipt,
    }
    return journaled_replay, _normalize_offline_fixture_bundle_claims(bundle_claims)


def _production_scoring_receipt(
    value: object,
    *,
    clean_replay: object,
    contract_id: str,
    prediction_id: str,
    original_prediction_sha256: str,
    replay_prediction_sha256: str,
    submission_sha256: str,
) -> ReplayGradeReceipt:
    if not isinstance(clean_replay, Mapping) or frozenset(clean_replay) != {
        "schema_version",
        "candidate_id",
        "identity",
        "equality",
        "training_replay",
        "validation",
        "final",
        "capabilities",
        "workspace",
    }:
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: clean replay has an unexpected schema"
        )
    if (
        clean_replay.get("schema_version") != 1
        or type(clean_replay.get("candidate_id")) is not str
        or not cast(str, clean_replay.get("candidate_id"))
        or type(clean_replay.get("training_replay")) is not str
        or not cast(str, clean_replay.get("training_replay"))
    ):
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: clean replay header is invalid"
        )
    identity = clean_replay.get("identity")
    if not isinstance(identity, Mapping) or frozenset(identity) != {
        "source_sha256",
        "config_sha256",
        "features_sha256",
        "checkpoint_sha256",
        "validation_prediction_artifact_sha256",
        "validation_prediction_digest",
        "data_sha256",
        "environment_sha256",
    }:
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: frozen replay identity schema differs"
        )
    for name, candidate in identity.items():
        _sha256(candidate, f"clean replay identity {name}")
    equality = clean_replay.get("equality")
    if (
        not isinstance(equality, Mapping)
        or frozenset(equality) != {"policy", "absolute_tolerance"}
        or equality.get("policy") != "exact_same_host_bytes"
        or equality.get("absolute_tolerance") != 0.0
    ):
        raise StateInvariantError("PRODUCTION_SCORING_RECEIPT_INVALID: clean replay is not exact")
    validation = clean_replay.get("validation")
    if not isinstance(validation, Mapping) or frozenset(validation) != {
        "row_count",
        "reference_prediction_digest",
        "replay_prediction_digest",
        "replay_prediction_file_sha256",
        "exact_prediction_bytes",
        "maximum_absolute_difference",
        "top5_order_identical",
        "protected_metrics_identical",
        "metrics",
        "public_submission_sha256",
        "public_submission_prediction_digest",
        "csv_serialization",
    }:
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: validation replay schema differs"
        )
    validation_reference = _sha256(
        validation.get("reference_prediction_digest"),
        "clean replay validation reference prediction",
    )
    validation_replay = _sha256(
        validation.get("replay_prediction_digest"),
        "clean replay validation replay prediction",
    )
    _sha256(
        validation.get("replay_prediction_file_sha256"),
        "clean replay validation prediction file",
    )
    _sha256(
        validation.get("public_submission_sha256"),
        "clean replay validation submission",
    )
    _sha256(
        validation.get("public_submission_prediction_digest"),
        "clean replay validation submission prediction",
    )
    csv = validation.get("csv_serialization")
    metrics = validation.get("metrics")
    if not isinstance(metrics, Mapping) or frozenset(metrics) != {"GAUC", "nDCG@5", "primary"}:
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: validation metrics schema differs"
        )
    metric_values: dict[str, float] = {}
    for name, candidate in metrics.items():
        if type(candidate) not in {int, float} or not math.isfinite(cast(int | float, candidate)):
            raise StateInvariantError(
                f"PRODUCTION_SCORING_RECEIPT_INVALID: validation metric {name} is invalid"
            )
        metric_values[name] = float(cast(int | float, candidate))
    if (
        type(validation.get("row_count")) is not int
        or cast(int, validation["row_count"]) <= 0
        or validation_reference != validation_replay
        or validation.get("exact_prediction_bytes") is not True
        or validation.get("maximum_absolute_difference") != 0.0
        or validation.get("top5_order_identical") is not True
        or validation.get("protected_metrics_identical") is not True
        or validation.get("public_submission_prediction_digest") != validation_replay
        or not math.isclose(
            metric_values["primary"],
            (metric_values["GAUC"] + metric_values["nDCG@5"]) / 2.0,
            rel_tol=0.0,
            abs_tol=2e-7,
        )
        or not isinstance(csv, Mapping)
        or frozenset(csv)
        != {
            "float64_round_trip_identity",
            "within_user_order_preserved",
            "top5_preserved",
            "protected_metrics_preserved",
        }
        or any(value is not True for value in csv.values())
        or identity.get("validation_prediction_digest") != validation_reference
    ):
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: validation replay is not scoring exact"
        )
    final = clean_replay.get("final")
    if not isinstance(final, Mapping) or frozenset(final) != {
        "row_count",
        "prediction_digest",
        "prediction_file_sha256",
        "submission_sha256",
        "submission_prediction_digest",
        "finite_scores",
        "csv_round_trip_identity",
        "outcome_access",
        "final_outcomes_accessed",
        "final_outcomes_scored",
    }:
        raise StateInvariantError("PRODUCTION_SCORING_RECEIPT_INVALID: final replay schema differs")
    final_prediction = _sha256(final.get("prediction_digest"), "final replay prediction")
    final_submission_prediction = _sha256(
        final.get("submission_prediction_digest"), "final replay submission prediction"
    )
    final_submission = _sha256(final.get("submission_sha256"), "final replay submission")
    _sha256(final.get("prediction_file_sha256"), "final replay prediction file")
    if (
        type(final.get("row_count")) is not int
        or cast(int, final["row_count"]) <= 0
        or final_prediction != final_submission_prediction
        or final_prediction not in {original_prediction_sha256, replay_prediction_sha256}
        or original_prediction_sha256 != replay_prediction_sha256
        or final_submission != submission_sha256
        or final.get("finite_scores") is not True
        or final.get("csv_round_trip_identity") is not True
        or final.get("outcome_access") != "none"
        or final.get("final_outcomes_accessed") is not False
        or final.get("final_outcomes_scored") is not False
    ):
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: final replay is not target-free exact evidence"
        )
    capabilities = clean_replay.get("capabilities")
    if not isinstance(capabilities, Mapping) or frozenset(capabilities) != {
        "validation_digest",
        "final_digest",
        "validation_phase",
        "final_phase",
        "labels_exposed_to_backend",
        "raw_data_path_exposed_to_backend",
    }:
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: replay capability schema differs"
        )
    validation_capability = _sha256(
        capabilities.get("validation_digest"), "validation capability digest"
    )
    final_capability = _sha256(capabilities.get("final_digest"), "final capability digest")
    if (
        capabilities.get("validation_phase") != "outer_valid"
        or capabilities.get("final_phase") != "final"
        or capabilities.get("labels_exposed_to_backend") is not False
        or capabilities.get("raw_data_path_exposed_to_backend") is not False
    ):
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: replay capability boundary is unsafe"
        )
    workspace = clean_replay.get("workspace")
    if not isinstance(workspace, Mapping) or dict(workspace) != {
        "fresh_materialization": True,
        "artifact_identities_reverified_after_inference": True,
        "clean_workspace_removed": True,
    }:
        raise StateInvariantError(
            "PRODUCTION_SCORING_RECEIPT_INVALID: clean workspace evidence differs"
        )
    clean_replay_digest = hashlib.sha256(
        json.dumps(
            dict(clean_replay),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    expected_receipt_evidence = {
        "kind": "clean_replay",
        "clean_replay_sha256": clean_replay_digest,
        "candidate_id": clean_replay.get("candidate_id"),
        "frozen_identity": dict(identity),
        "equality": equality.get("policy"),
        "absolute_tolerance": equality.get("absolute_tolerance"),
        "training_replay": clean_replay.get("training_replay"),
        "validation": dict(validation),
        "final": dict(final),
        "capabilities": {
            "validation_digest": validation_capability,
            "final_digest": final_capability,
        },
        "clean_workspace_removed": True,
        "conclusion": "stored and replayed prediction bytes have identical exact metrics",
    }
    try:
        return validate_replay_grade_receipt_manifest(
            value,
            expected_grade=ReplayGrade.SCORING_EXACT,
            expected_contract_id=contract_id,
            expected_prediction_id=prediction_id,
            expected_evidence=expected_receipt_evidence,
        )
    except ReplayGradeError as exc:
        raise StateInvariantError(f"PRODUCTION_SCORING_RECEIPT_INVALID: {exc}") from exc


def _production_same_backend_receipt(
    value: object,
    *,
    scoring_receipt: ReplayGradeReceipt,
    contract_id: str,
    prediction_id: str,
) -> ReplayGradeReceipt:
    """Authenticate same-backend proof against the exact clean replay used for scoring."""

    expected_evidence = dict(scoring_receipt.evidence)
    expected_evidence["conclusion"] = (
        "frozen trial artifacts in the frozen environment reproduced exact predictions"
    )
    try:
        return validate_replay_grade_receipt_manifest(
            value,
            expected_grade=ReplayGrade.EXPERIMENT_SAME_BACKEND,
            expected_contract_id=contract_id,
            expected_prediction_id=prediction_id,
            expected_evidence=expected_evidence,
        )
    except ReplayGradeError as exc:
        raise StateInvariantError(f"PRODUCTION_SAME_BACKEND_RECEIPT_INVALID: {exc}") from exc


def _normalize_production_replay(
    value: Mapping[str, object],
    *,
    contract_id: str,
    campaign_id: str,
    selected_prediction_id: str,
    scoring_exact_receipt: ReplayGradeReceipt | None = None,
    same_backend_receipt: ReplayGradeReceipt | None = None,
) -> dict[str, object]:
    if frozenset(value) != _PRODUCTION_REPLAY_FIELDS:
        raise StateInvariantError("production replay evidence does not match the exact schema")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 3:
        raise StateInvariantError("production replay schema_version must be integer 3")
    lineage = (
        _sha256(value.get("contract_id"), "production replay contract_id"),
        _sha256(value.get("campaign_id"), "production replay campaign_id"),
        _sha256(value.get("prediction_id"), "production replay prediction_id"),
    )
    if lineage != (contract_id, campaign_id, selected_prediction_id):
        raise StateInvariantError("production replay evidence is outside terminal lineage")
    achieved = value.get("achieved_replay_grades")
    required = value.get("required_terminal_replay_grades")
    if (
        type(achieved) is not list
        or tuple(cast(list[object], achieved)) != _PRODUCTION_PREPUBLICATION_REPLAY_GRADES
        or type(required) is not list
        or tuple(cast(list[object], required)) != _PRODUCTION_REQUIRED_REPLAY_GRADES
    ):
        raise StateInvariantError(
            "production replay evidence confuses achieved and required grades"
        )
    if value.get("qualification_scope") != "FULL_DATA_CPU":
        raise StateInvariantError("production replay qualification scope is unsupported")
    if (
        value.get("official_fm_qualified") is not True
        or value.get("full_data_qualified") is not True
        or value.get("final_period_outcomes_accessed") is not False
    ):
        raise StateInvariantError("production replay qualification or outcome evidence is unsafe")
    digests = {
        field_name: _sha256(value.get(field_name), f"production replay {field_name}")
        for field_name in (
            "qualification_manifest_digest",
            "qualification_fallback_digest",
            "original_prediction_sha256",
            "replay_prediction_sha256",
            "row_alignment_sha256",
            "submission_sha256",
            "organizer_check_sha256",
        )
    }
    if digests["original_prediction_sha256"] != digests["replay_prediction_sha256"]:
        raise StateInvariantError("production replay prediction bytes are not exact")
    clean_replay = value.get("clean_replay_evidence")
    if not isinstance(clean_replay, Mapping):
        raise StateInvariantError("production replay lacks clean replay evidence")
    clean_digest = _sha256(
        value.get("clean_replay_evidence_sha256"), "clean_replay_evidence_sha256"
    )
    if canonical_json_sha256(dict(clean_replay)) != clean_digest:
        raise StateInvariantError("production clean replay digest is invalid")
    scoring = _production_scoring_receipt(
        value.get("scoring_exact_receipt"),
        clean_replay=clean_replay,
        contract_id=contract_id,
        prediction_id=selected_prediction_id,
        original_prediction_sha256=digests["original_prediction_sha256"],
        replay_prediction_sha256=digests["replay_prediction_sha256"],
        submission_sha256=digests["submission_sha256"],
    )
    same_backend = _production_same_backend_receipt(
        value.get("same_backend_receipt"),
        scoring_receipt=scoring,
        contract_id=contract_id,
        prediction_id=selected_prediction_id,
    )
    if scoring_exact_receipt is not None:
        if not isinstance(scoring_exact_receipt, ReplayGradeReceipt):
            raise StateInvariantError("production preparation scoring receipt must be typed")
        try:
            typed = validate_replay_grade_receipt_manifest(
                scoring_exact_receipt.manifest(),
                expected_grade=ReplayGrade.SCORING_EXACT,
                expected_contract_id=contract_id,
                expected_prediction_id=selected_prediction_id,
                expected_evidence=scoring.evidence,
            )
        except ReplayGradeError as exc:
            raise StateInvariantError(f"PRODUCTION_SCORING_RECEIPT_INVALID: {exc}") from exc
        if typed.manifest() != scoring.manifest():
            raise StateInvariantError(
                "production typed scoring receipt differs from replay payload"
            )
    if same_backend_receipt is not None:
        if not isinstance(same_backend_receipt, ReplayGradeReceipt):
            raise StateInvariantError("production preparation same-backend receipt must be typed")
        try:
            typed_same_backend = validate_replay_grade_receipt_manifest(
                same_backend_receipt.manifest(),
                expected_grade=ReplayGrade.EXPERIMENT_SAME_BACKEND,
                expected_contract_id=contract_id,
                expected_prediction_id=selected_prediction_id,
                expected_evidence=same_backend.evidence,
            )
        except ReplayGradeError as exc:
            raise StateInvariantError(f"PRODUCTION_SAME_BACKEND_RECEIPT_INVALID: {exc}") from exc
        if typed_same_backend.manifest() != same_backend.manifest():
            raise StateInvariantError(
                "production typed same-backend receipt differs from replay payload"
            )
    organizer = value.get("organizer_check")
    if not isinstance(organizer, Mapping):
        raise StateInvariantError("production replay lacks organizer check evidence")
    if canonical_json_sha256(dict(organizer)) != digests["organizer_check_sha256"]:
        raise StateInvariantError("PRODUCTION_ORGANIZER_CHECK_INVALID: digest differs")
    clean_final = cast(Mapping[str, object], clean_replay["final"])
    organizer_starter = _sha256(
        organizer.get("starter_manifest_sha256"), "organizer starter manifest"
    )
    try:
        normalized_organizer = validate_organizer_check_manifest(
            organizer,
            expected_submission_sha256=digests["submission_sha256"],
            expected_submission_size_bytes=None,
            expected_starter_manifest_sha256=organizer_starter,
            expected_final_rows=cast(int, clean_final["row_count"]),
        )
    except OrganizerCheckError as exc:
        raise StateInvariantError(f"PRODUCTION_ORGANIZER_CHECK_INVALID: {exc}") from exc
    normalized: dict[str, object] = {
        "schema_version": 3,
        "contract_id": contract_id,
        "campaign_id": campaign_id,
        "prediction_id": selected_prediction_id,
        **digests,
        "organizer_check": normalized_organizer,
        "clean_replay_evidence_sha256": clean_digest,
        "clean_replay_evidence": dict(clean_replay),
        "scoring_exact_receipt": scoring.manifest(),
        "same_backend_receipt": same_backend.manifest(),
        "achieved_replay_grades": list(_PRODUCTION_PREPUBLICATION_REPLAY_GRADES),
        "required_terminal_replay_grades": list(_PRODUCTION_REQUIRED_REPLAY_GRADES),
        "qualification_scope": "FULL_DATA_CPU",
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
    }
    if canonical_json_bytes(dict(value)) != canonical_json_bytes(normalized):
        raise StateInvariantError("production replay evidence contains forged fields")
    return normalized


def _normalize_production_bundle_claims(
    value: Mapping[str, object], *, qualification_manifest_digest: str
) -> dict[str, object]:
    if frozenset(value) != _PRODUCTION_BUNDLE_CLAIM_FIELDS:
        raise StateInvariantError("production bundle claims do not match the exact schema")
    resource_receipt_id = _sha256(value.get("resource_receipt_id"), "resource_receipt_id")
    claimed_qualification = _sha256(
        value.get("qualification_manifest_digest"), "qualification_manifest_digest"
    )
    prepublication_grades = value.get("prepublication_replay_grades")
    required_grades = value.get("required_replay_grades")
    if (
        value.get("schema_version") != 2
        or type(prepublication_grades) is not list
        or tuple(cast(list[object], prepublication_grades))
        != _PRODUCTION_PREPUBLICATION_REPLAY_GRADES
        or type(required_grades) is not list
        or tuple(cast(list[object], required_grades)) != _PRODUCTION_REQUIRED_REPLAY_GRADES
        or value.get("bundle_exact_status") != "PENDING_PUBLICATION_PROOF"
    ):
        raise StateInvariantError(
            "production bundle claims confuse prepublication and required replay grades"
        )
    if (
        type(value.get("protected_query_count")) is not int
        or value.get("protected_query_count") != 0
        or type(value.get("provider_operation_count")) is not int
        or value.get("provider_operation_count") != 0
    ):
        raise StateInvariantError("production terminal claims require zero protected/provider use")
    normalized: dict[str, object] = {
        "schema_version": 2,
        "resource_receipt_id": resource_receipt_id,
        "prepublication_replay_grades": list(_PRODUCTION_PREPUBLICATION_REPLAY_GRADES),
        "required_replay_grades": list(_PRODUCTION_REQUIRED_REPLAY_GRADES),
        "bundle_exact_status": "PENDING_PUBLICATION_PROOF",
        "submission_disposition": "FALLBACK_RETAINED",
        "scientific_disposition": "INSUFFICIENT_VALID_EVIDENCE",
        "campaign_kind": "PRODUCTION_FULL_DATA",
        "qualification_scope": "FULL_DATA_CPU",
        "protected_query_count": 0,
        "provider_operation_count": 0,
        "exact_metrics": None,
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
        "qualification_manifest_digest": claimed_qualification,
    }
    if claimed_qualification != qualification_manifest_digest:
        raise StateInvariantError("production bundle qualification identity differs from replay")
    if canonical_json_bytes(dict(value)) != canonical_json_bytes(normalized):
        raise StateInvariantError("production bundle claims overstate verified evidence")
    return normalized


def _normalize_production_evidence(
    *,
    contract_id: str,
    campaign_id: str,
    selected_prediction_id: str,
    fallback_prediction_id: str,
    replay_payload: Mapping[str, object],
    bundle_claims: Mapping[str, object],
    scoring_exact_receipt: ReplayGradeReceipt | None = None,
    same_backend_receipt: ReplayGradeReceipt | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if selected_prediction_id != fallback_prediction_id:
        raise StateInvariantError("production fallback-retained completion must select fallback")
    replay = _normalize_production_replay(
        replay_payload,
        contract_id=contract_id,
        campaign_id=campaign_id,
        selected_prediction_id=selected_prediction_id,
        scoring_exact_receipt=scoring_exact_receipt,
        same_backend_receipt=same_backend_receipt,
    )
    claims = _normalize_production_bundle_claims(
        bundle_claims,
        qualification_manifest_digest=cast(str, replay["qualification_manifest_digest"]),
    )
    return replay, claims


def _validate_stored_terminal_evidence(
    *,
    terminal_state: str,
    contract_id: str,
    campaign_id: str,
    selected_prediction_id: str,
    fallback_prediction_id: str,
    replay_json: str,
    bundle_claims: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    replay = _json_mapping(replay_json)
    if terminal_state == _PRODUCTION_TERMINAL_STATE:
        return _normalize_production_evidence(
            contract_id=contract_id,
            campaign_id=campaign_id,
            selected_prediction_id=selected_prediction_id,
            fallback_prediction_id=fallback_prediction_id,
            replay_payload=replay,
            bundle_claims=bundle_claims,
        )
    if terminal_state != _OFFLINE_FIXTURE_TERMINAL_STATE:
        raise StateInvariantError("stored terminal state has no replay evidence verifier")
    if frozenset(replay) != _OFFLINE_FIXTURE_REPLAY_JOURNAL_FIELDS:
        raise StateInvariantError("stored replay evidence does not match the exact journal schema")
    if (
        type(replay.get("schema_version")) is not int
        or replay.get("schema_version") != 1
        or replay.get("verifier") != "SCRIPTED_REPLAY_RECEIPT"
    ):
        raise StateInvariantError("stored replay evidence names an unsupported verifier")
    receipt = _validate_scripted_replay_receipt(
        replay.get("scripted_replay_receipt"),
        contract_id=contract_id,
        campaign_id=campaign_id,
        selected_prediction_id=selected_prediction_id,
    )
    if (
        replay.get("grade") != receipt["grade"]
        or replay.get("qualification_scope") != receipt["qualification_scope"]
    ):
        raise StateInvariantError("journaled replay grade is not derived from verified evidence")
    claims = _normalize_offline_fixture_bundle_claims(bundle_claims)
    return dict(replay), claims


def _validated_production_publication_proof(
    publication: PublishedBundleReceipt,
    *,
    contract_id: str,
    prediction_id: str,
    bundle_id: str,
    submission_sha256: str,
    inventory_sha256: str,
    scoring_exact_receipt_manifest: object,
    same_backend_receipt_manifest: object,
) -> dict[str, object]:
    if not isinstance(publication.regeneration_evidence, BundleRegenerationEvidence):
        raise StateInvariantError(
            "PRODUCTION_BUNDLE_REGENERATION_INVALID: typed evidence is required"
        )
    if not isinstance(publication.bundle_exact_receipt, ReplayGradeReceipt):
        raise StateInvariantError("PRODUCTION_BUNDLE_RECEIPT_INVALID: typed receipt is required")
    try:
        regeneration = validate_bundle_regeneration_evidence_manifest(
            publication.regeneration_evidence.manifest()
        )
    except ReplayGradeError as exc:
        raise StateInvariantError(f"PRODUCTION_BUNDLE_REGENERATION_INVALID: {exc}") from exc
    if (
        regeneration.contract_id != contract_id
        or regeneration.prediction_id != prediction_id
        or regeneration.first_bundle_id != bundle_id
        or regeneration.regenerated_bundle_id != bundle_id
        or regeneration.first_submission_sha256 != submission_sha256
        or regeneration.regenerated_submission_sha256 != submission_sha256
        or regeneration.first_inventory_sha256 != inventory_sha256
        or regeneration.regenerated_inventory_sha256 != inventory_sha256
    ):
        raise StateInvariantError(
            "PRODUCTION_BUNDLE_REGENERATION_INVALID: proof differs from publication"
        )
    try:
        bundle_receipt = validate_replay_grade_receipt_manifest(
            publication.bundle_exact_receipt.manifest(),
            expected_grade=ReplayGrade.BUNDLE_EXACT,
            expected_contract_id=contract_id,
            expected_prediction_id=prediction_id,
            expected_evidence=regeneration.manifest(),
        )
        scoring_receipt = validate_replay_grade_receipt_manifest(
            scoring_exact_receipt_manifest,
            expected_grade=ReplayGrade.SCORING_EXACT,
            expected_contract_id=contract_id,
            expected_prediction_id=prediction_id,
        )
        same_backend_receipt = validate_replay_grade_receipt_manifest(
            same_backend_receipt_manifest,
            expected_grade=ReplayGrade.EXPERIMENT_SAME_BACKEND,
            expected_contract_id=contract_id,
            expected_prediction_id=prediction_id,
        )
        scoring_evidence = dict(scoring_receipt.evidence)
        same_backend_evidence = dict(same_backend_receipt.evidence)
        scoring_evidence.pop("conclusion", None)
        same_backend_evidence.pop("conclusion", None)
        if same_backend_evidence != scoring_evidence:
            raise ReplayGradeError(
                "same-backend and scoring receipts do not bind the same clean replay"
            )
        report = combine_replay_grade_receipts(
            (scoring_receipt, same_backend_receipt, bundle_receipt)
        )
    except ReplayGradeError as exc:
        raise StateInvariantError(f"PRODUCTION_BUNDLE_RECEIPT_INVALID: {exc}") from exc
    if report.achieved_grades != {
        ReplayGrade.SCORING_EXACT,
        ReplayGrade.EXPERIMENT_SAME_BACKEND,
        ReplayGrade.BUNDLE_EXACT,
    }:
        raise StateInvariantError(
            "PRODUCTION_REPLAY_GRADE_INCOMPLETE: terminal proof lacks required grades"
        )
    return {
        "schema_version": 1,
        "regeneration_evidence": regeneration.manifest(),
        "bundle_exact_receipt": bundle_receipt.manifest(),
        "replay_grade_receipts": {
            ReplayGrade.SCORING_EXACT.value: scoring_receipt.manifest(),
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value: same_backend_receipt.manifest(),
            ReplayGrade.BUNDLE_EXACT.value: bundle_receipt.manifest(),
        },
        "replay_grade_report": report.manifest(),
    }


def _validated_postpublication_resource_receipt(
    value: PostpublicationResourceReceipt | None,
    *,
    terminal_state: str,
    campaign_id: str,
    contract_id: str,
    selected_prediction_id: str,
    prepublication_resource_receipt_id: str,
    prepublication_resource_payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Authenticate and cap the strongest authority-bound resource observation."""

    if terminal_state != _PRODUCTION_TERMINAL_STATE:
        if value is not None:
            raise StateInvariantError(
                "offline publication cannot attach a production postpublication receipt"
            )
        return None
    if not isinstance(value, PostpublicationResourceReceipt):
        raise StateInvariantError(
            "production publication requires a typed postpublication resource receipt"
        )
    expected_profile_digest = _sha256(
        prepublication_resource_payload.get("performance_profile_digest"),
        "authoritative performance_profile_digest",
    )
    if (
        value.contract_id,
        value.campaign_id,
        value.prediction_id,
        value.prepublication_resource_receipt_id,
        value.performance_profile_digest,
        value.scope,
    ) != (
        contract_id,
        campaign_id,
        selected_prediction_id,
        prepublication_resource_receipt_id,
        expected_profile_digest,
        _POSTPUBLICATION_RESOURCE_SCOPE,
    ):
        raise StateInvariantError(
            "postpublication resource receipt is outside terminal authority lineage"
        )
    reconstructed = PostpublicationResourceReceipt.from_measurement(
        contract_id=contract_id,
        campaign_id=campaign_id,
        prediction_id=selected_prediction_id,
        prepublication_resource_receipt_id=prepublication_resource_receipt_id,
        performance_profile_digest=expected_profile_digest,
        measurements=value.measurements,
    )
    if reconstructed.manifest() != value.manifest():
        raise StateInvariantError("postpublication resource receipt identity is invalid")

    declared = prepublication_resource_payload.get("declared_resource_profile")
    qualified = prepublication_resource_payload.get("qualified_training_resources")
    prepublication = prepublication_resource_payload.get("controller_resources")
    if not all(isinstance(item, Mapping) for item in (declared, qualified, prepublication)):
        raise StateInvariantError("authoritative production resource evidence is malformed")
    declared_mapping = cast(Mapping[str, object], declared)
    measurement_sets = (
        cast(Mapping[str, object], qualified),
        cast(Mapping[str, object], prepublication),
        value.measurements,
    )
    wall_cap = _positive_int(
        declared_mapping.get("wall_clock_seconds"), "declared wall_clock_seconds"
    )
    rss_cap = (
        _positive_int(
            declared_mapping.get("process_tree_rss_hard_cap_mb"),
            "declared process_tree_rss_hard_cap_mb",
        )
        * 1024
        * 1024
    )
    disk_cap = (
        _positive_int(
            declared_mapping.get("candidate_disk_hard_cap_mb"),
            "declared candidate_disk_hard_cap_mb",
        )
        * 1024
        * 1024
    )
    strongest_wall = max(cast(float, item["wall_seconds"]) for item in measurement_sets)
    strongest_rss = max(cast(int, item["peak_rss_bytes"]) for item in measurement_sets)
    strongest_disk = max(cast(int, item["disk_bytes"]) for item in measurement_sets)
    if strongest_wall > wall_cap or strongest_rss > rss_cap or strongest_disk > disk_cap:
        raise StateInvariantError(
            "strongest authority-bound production resources exceed declared caps"
        )
    return value.manifest()


def _validated_resource_receipt_payload(
    connection: sqlite3.Connection,
    *,
    receipt_id: str,
    campaign_id: str,
    contract_id: str,
    selected_prediction_id: str,
    terminal_state: str,
    expected_qualification_manifest_digest: str | None,
    expected_qualification_fallback_digest: str | None,
    expected_prediction_sha256: str | None,
    expected_row_alignment_sha256: str | None,
) -> dict[str, object]:
    if terminal_state == _PRODUCTION_TERMINAL_STATE:
        if None in {
            expected_qualification_manifest_digest,
            expected_qualification_fallback_digest,
            expected_prediction_sha256,
            expected_row_alignment_sha256,
        }:
            raise StateInvariantError("production resource validation lacks replay identities")
        return _validated_production_resource_receipt_payload(
            connection,
            receipt_id=receipt_id,
            campaign_id=campaign_id,
            contract_id=contract_id,
            selected_prediction_id=selected_prediction_id,
            expected_qualification_manifest_digest=cast(
                str, expected_qualification_manifest_digest
            ),
            expected_qualification_fallback_digest=cast(
                str, expected_qualification_fallback_digest
            ),
            expected_prediction_sha256=cast(str, expected_prediction_sha256),
            expected_row_alignment_sha256=cast(str, expected_row_alignment_sha256),
        )
    if terminal_state != _OFFLINE_FIXTURE_TERMINAL_STATE:
        raise StateInvariantError("resource receipt has no terminal evidence verifier")
    return _validated_offline_fixture_resource_receipt_payload(
        connection,
        receipt_id=receipt_id,
        campaign_id=campaign_id,
        contract_id=contract_id,
        selected_prediction_id=selected_prediction_id,
    )


def _validate_production_resource_measurements(
    value: object,
    *,
    fields: frozenset[str],
    location: str,
    peak_rss_positive: bool,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise StateInvariantError(f"{location} does not match the exact schema")
    normalized = dict(value)
    for field_name in ("wall_seconds", "cpu_seconds"):
        observed = normalized.get(field_name)
        if (
            type(observed) not in {int, float}
            or not math.isfinite(cast(int | float, observed))
            or cast(int | float, observed) < 0
        ):
            raise StateInvariantError(f"{location} {field_name} is invalid")
    peak_rss = normalized.get("peak_rss_bytes")
    disk = normalized.get("disk_bytes")
    if type(peak_rss) is not int or peak_rss < int(peak_rss_positive):
        raise StateInvariantError(f"{location} peak_rss_bytes is invalid")
    if type(disk) is not int or disk < 0:
        raise StateInvariantError(f"{location} disk_bytes is invalid")
    if normalized.get("device") != "cpu":
        raise StateInvariantError(f"{location} must attest CPU execution")
    return normalized


def _validated_production_resource_receipt_payload(
    connection: sqlite3.Connection,
    *,
    receipt_id: str,
    campaign_id: str,
    contract_id: str,
    selected_prediction_id: str,
    expected_qualification_manifest_digest: str,
    expected_qualification_fallback_digest: str,
    expected_prediction_sha256: str,
    expected_row_alignment_sha256: str,
) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT attempt_id, payload_json FROM resource_receipts
        WHERE receipt_id = ? AND campaign_id = ? AND contract_id = ?
        """,
        (receipt_id, campaign_id, contract_id),
    ).fetchone()
    if row is None:
        raise StateInvariantError(
            "production resource_receipt_id is outside campaign/contract authority"
        )
    if row[0] is not None:
        raise StateInvariantError("production qualification resource receipt must omit attempt_id")
    payload = dict(_json_mapping(str(row[1])))
    if frozenset(payload) != _PRODUCTION_RESOURCE_FIELDS:
        raise StateInvariantError("production resource receipt does not match exact schema")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise StateInvariantError("production resource receipt schema_version must be integer 1")
    lineage = (
        _sha256(payload.get("contract_id"), "production resource contract_id"),
        _sha256(payload.get("campaign_id"), "production resource campaign_id"),
        _sha256(payload.get("prediction_id"), "production resource prediction_id"),
    )
    if lineage != (contract_id, campaign_id, selected_prediction_id):
        raise StateInvariantError("production resource receipt is outside prediction lineage")
    qualification_manifest_digest = _sha256(
        payload.get("qualification_manifest_digest"), "qualification_manifest_digest"
    )
    qualification_fallback_digest = _sha256(
        payload.get("qualification_fallback_digest"), "qualification_fallback_digest"
    )
    performance_profile_digest = _sha256(
        payload.get("performance_profile_digest"), "performance_profile_digest"
    )
    if (
        qualification_manifest_digest != expected_qualification_manifest_digest
        or qualification_fallback_digest != expected_qualification_fallback_digest
    ):
        raise StateInvariantError("production resource qualification identity differs from replay")
    if (
        payload.get("campaign_kind") != "PRODUCTION_FULL_DATA"
        or payload.get("qualification_scope") != "FULL_DATA_CPU"
        or payload.get("actual_trainer_backend") != "organizer-numpy-fm"
        or payload.get("actual_trainer_device") != "cpu"
        or payload.get("controller_resource_scope") != "PREPUBLICATION_SELF_EXCLUDING"
        or payload.get("preferred_backend_qualified") is not False
        or payload.get("official_fm_qualified") is not True
        or payload.get("full_data_qualified") is not True
        or payload.get("final_period_outcomes_accessed") is not False
    ):
        raise StateInvariantError("production resource receipt qualification evidence is unsafe")
    declared_profile = payload.get("declared_resource_profile")
    if not isinstance(declared_profile, Mapping) or not declared_profile:
        raise StateInvariantError("production declared resource profile must be a mapping")
    try:
        canonical_json_bytes(dict(declared_profile))
    except (TypeError, ValueError) as exc:
        raise StateInvariantError("production declared resource profile is not canonical") from exc
    if (
        declared_profile.get("name") != "competition-cpu"
        or declared_profile.get("preferred_backend") != "lightgbm-cpu"
        or declared_profile.get("device") != "cpu"
    ):
        raise StateInvariantError("production declared resource profile differs from CPU policy")
    qualified_resources = _validate_production_resource_measurements(
        payload.get("qualified_training_resources"),
        fields=_PRODUCTION_QUALIFIED_RESOURCE_FIELDS,
        location="qualified training resources",
        peak_rss_positive=False,
    )
    controller_resources = _validate_production_resource_measurements(
        payload.get("controller_resources"),
        fields=_PRODUCTION_CONTROLLER_RESOURCE_FIELDS,
        location="production controller resources",
        peak_rss_positive=True,
    )
    wall_cap = declared_profile.get("wall_clock_seconds")
    rss_cap_mb = declared_profile.get("process_tree_rss_hard_cap_mb")
    disk_cap_mb = declared_profile.get("candidate_disk_hard_cap_mb")
    if (
        type(wall_cap) is not int
        or wall_cap <= 0
        or type(rss_cap_mb) is not int
        or rss_cap_mb <= 0
        or type(disk_cap_mb) is not int
        or disk_cap_mb <= 0
    ):
        raise StateInvariantError("production declared resource caps are invalid")
    for location, measurements in (
        ("qualified training", qualified_resources),
        ("prepublication controller", controller_resources),
    ):
        if (
            cast(float, measurements["wall_seconds"]) > wall_cap
            or cast(int, measurements["peak_rss_bytes"]) > rss_cap_mb * 1024 * 1024
            or cast(int, measurements["disk_bytes"]) > disk_cap_mb * 1024 * 1024
        ):
            raise StateInvariantError(f"production {location} resources exceed declared caps")
    authority = connection.execute(
        """
        SELECT p.prediction_bytes_sha256, p.ordered_rows_sha256, p.payload_json, c.config_json
        FROM predictions AS p
        JOIN campaigns AS c
          ON c.campaign_id = p.campaign_id AND c.contract_id = p.contract_id
        WHERE p.prediction_id = ? AND p.campaign_id = ? AND p.contract_id = ?
        """,
        (selected_prediction_id, campaign_id, contract_id),
    ).fetchone()
    if authority is None:
        raise StateInvariantError("production resource receipt lacks a registered prediction")
    if (
        str(authority[0]) != expected_prediction_sha256
        or str(authority[1]) != expected_row_alignment_sha256
    ):
        raise StateInvariantError("production replay differs from registered prediction bytes")
    prediction_payload = _json_mapping(str(authority[2]))
    campaign_config = _json_mapping(str(authority[3]))
    if (
        prediction_payload.get("qualification_manifest_digest") != qualification_manifest_digest
        or prediction_payload.get("qualification_fallback_digest") != qualification_fallback_digest
        or prediction_payload.get("final_period_outcomes_accessed") is not False
    ):
        raise StateInvariantError("registered fallback prediction lacks qualification lineage")
    if (
        campaign_config.get("qualification_manifest_digest") != qualification_manifest_digest
        or campaign_config.get("performance_profile_digest") != performance_profile_digest
        or campaign_config.get("resource_profile") != dict(declared_profile)
    ):
        raise StateInvariantError("campaign admission identities differ from resource receipt")
    normalized: dict[str, object] = {
        "schema_version": 1,
        "contract_id": contract_id,
        "campaign_id": campaign_id,
        "prediction_id": selected_prediction_id,
        "campaign_kind": "PRODUCTION_FULL_DATA",
        "qualification_scope": "FULL_DATA_CPU",
        "qualification_manifest_digest": qualification_manifest_digest,
        "qualification_fallback_digest": qualification_fallback_digest,
        "performance_profile_digest": performance_profile_digest,
        "declared_resource_profile": dict(declared_profile),
        "actual_trainer_backend": "organizer-numpy-fm",
        "actual_trainer_device": "cpu",
        "qualified_training_resources": qualified_resources,
        "controller_resources": controller_resources,
        "controller_resource_scope": "PREPUBLICATION_SELF_EXCLUDING",
        "preferred_backend_qualified": False,
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
    }
    if canonical_json_bytes(payload) != canonical_json_bytes(normalized):
        raise StateInvariantError("production resource receipt contains forged fields")
    if hashlib.sha256(canonical_json_bytes(normalized)).hexdigest() != receipt_id:
        raise StateInvariantError("authoritative resource receipt identity is invalid")
    return normalized


def _validate_production_authority_counts(
    connection: sqlite3.Connection, *, campaign_id: str
) -> None:
    counts = connection.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM protected_query_reservations WHERE campaign_id = ?),
          (SELECT COUNT(*) FROM protected_evaluations WHERE campaign_id = ?),
          (SELECT COUNT(*) FROM provider_operations WHERE campaign_id = ?)
        """,
        (campaign_id, campaign_id, campaign_id),
    ).fetchone()
    if counts is None or tuple(int(value) for value in counts) != (0, 0, 0):
        raise StateInvariantError(
            "production terminal authority contains protected queries or provider operations"
        )


def _validated_offline_fixture_resource_receipt_payload(
    connection: sqlite3.Connection,
    *,
    receipt_id: str,
    campaign_id: str,
    contract_id: str,
    selected_prediction_id: str,
) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT attempt_id, payload_json FROM resource_receipts
        WHERE receipt_id = ? AND campaign_id = ? AND contract_id = ?
        """,
        (receipt_id, campaign_id, contract_id),
    ).fetchone()
    if row is None:
        raise StateInvariantError(
            "offline fixture resource_receipt_id is outside campaign/contract authority"
        )
    attempt_id = str(row[0]) if row[0] is not None else ""
    payload = dict(_json_mapping(str(row[1])))
    if frozenset(payload) != _OFFLINE_FIXTURE_RESOURCE_FIELDS:
        raise StateInvariantError("offline fixture resource receipt does not match exact schema")
    lineage = connection.execute(
        """
        SELECT a.trial_id, a.terminal, a.payload_json, p.trial_id,
               p.prediction_bytes_sha256, p.payload_json, c.config_json
        FROM attempts AS a
        JOIN predictions AS p
          ON p.campaign_id = a.campaign_id AND p.contract_id = a.contract_id
        JOIN campaigns AS c ON c.campaign_id = a.campaign_id
        WHERE a.attempt_id = ? AND a.campaign_id = ? AND a.contract_id = ?
          AND p.prediction_id = ?
        """,
        (attempt_id, campaign_id, contract_id, selected_prediction_id),
    ).fetchone()
    if lineage is None or str(lineage[0]) != str(lineage[3]) or not bool(lineage[1]):
        raise StateInvariantError("offline fixture resource receipt lacks terminal attempt lineage")
    attempt_payload = _json_mapping(str(lineage[2]))
    prediction_payload = _json_mapping(str(lineage[5]))
    campaign_config = _json_mapping(str(lineage[6]))
    result = prediction_payload.get("trial_result")
    trainer = attempt_payload.get("trainer_identity")
    if not isinstance(result, Mapping) or not isinstance(trainer, Mapping):
        raise StateInvariantError("offline fixture resource receipt lacks trainer result evidence")
    result_trainer = result.get("trainer_identity")
    resources = result.get("resources")
    timing = result.get("timing")
    declared_profile = campaign_config.get("resource_profile")
    if not all(
        isinstance(value, Mapping)
        for value in (result_trainer, resources, timing, declared_profile)
    ):
        raise StateInvariantError("offline fixture resource receipt lacks measured evidence")
    result_trainer = cast(Mapping[str, object], result_trainer)
    resources = cast(Mapping[str, object], resources)
    timing = cast(Mapping[str, object], timing)
    declared_profile = cast(Mapping[str, object], declared_profile)
    if frozenset(resources) != _SCRIPTED_RESULT_RESOURCE_FIELDS:
        raise StateInvariantError("offline fixture result resource evidence schema differs")
    if frozenset(timing) != _SCRIPTED_TIMING_FIELDS:
        raise StateInvariantError("offline fixture timing evidence schema differs")
    for field_name in ("wall_seconds", "cpu_seconds"):
        value = resources.get(field_name)
        if (
            type(value) not in {int, float}
            or not math.isfinite(cast(int | float, value))
            or cast(int | float, value) < 0
        ):
            raise StateInvariantError(f"offline fixture {field_name} measurement is invalid")
    for field_name, minimum in (
        ("peak_rss_bytes", 0),
        ("peak_disk_bytes", 0),
        ("peak_process_count", 1),
        ("threads", 1),
    ):
        value = resources.get(field_name)
        if type(value) is not int or value < minimum:
            raise StateInvariantError(f"offline fixture {field_name} measurement is invalid")
    if type(resources.get("cpu_seconds_measured")) is not bool:
        raise StateInvariantError("offline fixture CPU measurement flag must be bool")
    started = timing.get("started_monotonic_ns")
    ended = timing.get("ended_monotonic_ns")
    timing_wall = timing.get("wall_seconds")
    if (
        type(started) is not int
        or started < 0
        or type(ended) is not int
        or ended < started
        or type(timing_wall) not in {int, float}
        or not math.isfinite(cast(int | float, timing_wall))
        or cast(int | float, timing_wall) < 0
        or not math.isclose(
            cast(int | float, timing_wall),
            (ended - started) / 1_000_000_000,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            cast(int | float, resources["wall_seconds"]),
            cast(int | float, timing_wall),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise StateInvariantError("offline fixture timing measurements are inconsistent")
    if any(
        type(candidate) is not str or not candidate or "\x00" in candidate
        for candidate in (
            result_trainer.get("backend"),
            result_trainer.get("device"),
            resources.get("device"),
        )
    ) or resources.get("device") != result_trainer.get("device"):
        raise StateInvariantError("offline fixture trainer resource identity is invalid")
    expected_result_lineage = (
        str(lineage[0]),
        attempt_id,
        selected_prediction_id,
        str(lineage[4]),
    )
    if (
        tuple(
            result.get(field)
            for field in ("trial_id", "attempt_id", "prediction_id", "prediction_sha256")
        )
        != expected_result_lineage
    ):
        raise StateInvariantError("offline fixture resource receipt prediction lineage differs")
    if (
        trainer.get("backend"),
        trainer.get("device"),
    ) != (result_trainer.get("backend"), result_trainer.get("device")):
        raise StateInvariantError("offline fixture trainer identity differs from terminal attempt")
    expected_resources = {
        "wall_seconds": resources.get("wall_seconds"),
        "wall_seconds_measured": True,
        "cpu_seconds": resources.get("cpu_seconds"),
        "cpu_seconds_measured": resources.get("cpu_seconds_measured"),
        "peak_rss_bytes": resources.get("peak_rss_bytes"),
        "peak_rss_bytes_measured": True,
        "peak_disk_bytes": None,
        "peak_disk_bytes_measured": False,
        "peak_process_count": resources.get("peak_process_count"),
        "peak_process_count_measured": True,
        "threads": None,
        "threads_measured": False,
        "declared_thread_limit": resources.get("threads"),
        "device": resources.get("device"),
    }
    expected_payload: dict[str, object] = {
        "schema_version": 1,
        "contract_id": contract_id,
        "campaign_id": campaign_id,
        "prediction_id": selected_prediction_id,
        "attempt_id": attempt_id,
        "campaign_kind": "OFFLINE_SCRIPTED_FIXTURE",
        "qualification_scope": "SCRIPTED_FIXTURE_ONLY",
        "declared_resource_profile": dict(declared_profile),
        "actual_trainer_backend": result_trainer.get("backend"),
        "actual_trainer_device": result_trainer.get("device"),
        "observed_resources": expected_resources,
        "timing": dict(timing),
        "preferred_backend_qualified": False,
        "official_fm_qualified": False,
        "full_data_qualified": False,
    }
    if canonical_json_bytes(payload) != canonical_json_bytes(expected_payload):
        raise StateInvariantError(
            "offline fixture resource receipt differs from authoritative measured evidence"
        )
    expected_receipt_id = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if expected_receipt_id != receipt_id:
        raise StateInvariantError("authoritative resource receipt identity is invalid")
    return payload


def _verify_published_bundle(
    publication: PublishedBundleReceipt,
    *,
    preparation_id: str,
    projection_sha256: str,
    projection_schema_version: int,
    redaction_policy_version: int,
    campaign_id: str,
    contract_id: str,
    selected_prediction_id: str,
    prepared_replay_json: str,
    bundle_id: str,
    manifest_sha256: str,
    inventory_sha256: str,
    submission_sha256: str,
    file_count: int,
    total_size_bytes: int,
    resource_receipt_id: str,
    resource_receipt_payload: Mapping[str, object],
) -> Path:
    root = publication.root
    if not isinstance(root, Path):
        raise PublishedBundleVerificationError("published bundle root must be pathlib.Path")
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise PublishedBundleVerificationError("published bundle root is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise PublishedBundleVerificationError("published bundle root must be a real directory")
    if root_metadata.st_mode & 0o222:
        raise PublishedBundleVerificationError("published bundle root is not sealed read-only")
    root = root.resolve()
    entries: list[dict[str, object]] = []
    members: dict[str, tuple[str, int]] = {}
    for member in sorted(root.iterdir(), key=lambda path: path.name):
        if member.lstat().st_mode & 0o222:
            raise PublishedBundleVerificationError("published bundle member is not sealed")
        sha256, size = _hash_regular_file(member)
        entries.append({"path": member.name, "sha256": sha256, "size_bytes": size})
        members[member.name] = (sha256, size)
    if frozenset(members) != _BUNDLE_MEMBERS:
        raise PublishedBundleVerificationError(
            "published bundle member set differs from the complete required layout"
        )
    if len(entries) != file_count or sum(size for _, size in members.values()) != total_size_bytes:
        raise PublishedBundleVerificationError("published bundle count or size receipt differs")
    if hashlib.sha256(canonical_json_bytes(entries)).hexdigest() != inventory_sha256:
        raise PublishedBundleVerificationError("published bundle inventory digest differs")
    if members["bundle-manifest.json"][0] != manifest_sha256:
        raise PublishedBundleVerificationError("published manifest digest differs")
    try:
        manifest_value = json.loads((root / "bundle-manifest.json").read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublishedBundleVerificationError("published manifest is unreadable") from exc
    if not isinstance(manifest_value, dict):
        raise PublishedBundleVerificationError("published manifest must be a JSON object")
    schema_version = manifest_value.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise PublishedBundleVerificationError(
            "prepared finalization requires published manifest schema v2"
        )
    if frozenset(manifest_value) != _BUNDLE_MANIFEST_FIELDS:
        raise PublishedBundleVerificationError(
            f"published manifest schema v{schema_version} has unexpected fields"
        )
    identity = manifest_value.get("identity")
    if identity != {
        "algorithm": "sha256",
        "definition": (
            "domain BundleId over selected prediction, replay output, submission, and manifest"
        ),
    }:
        raise PublishedBundleVerificationError("published manifest identity contract differs")
    required_paths = manifest_value.get("required_paths")
    if (
        not isinstance(required_paths, list)
        or any(not isinstance(path, str) for path in required_paths)
        or len(required_paths) != len(_BUNDLE_MEMBERS)
        or frozenset(required_paths) != _BUNDLE_MEMBERS
    ):
        raise PublishedBundleVerificationError("manifest required_paths is not the exact layout")
    terminal = manifest_value.get("terminal_projection")
    expected_terminal = {
        "schema_version": projection_schema_version,
        "preparation_id": preparation_id,
        "projection_sha256": projection_sha256,
        "redaction_policy_version": redaction_policy_version,
    }
    if (
        not isinstance(terminal, dict)
        or frozenset(terminal)
        != {
            "schema_version",
            "preparation_id",
            "projection_sha256",
            "source_revision",
            "redaction_policy_version",
        }
        or type(terminal.get("schema_version")) is not int
        or type(terminal.get("redaction_policy_version")) is not int
        or any(
            terminal.get(key) != expected_terminal[key]
            for key in (
                "schema_version",
                "preparation_id",
                "projection_sha256",
                "redaction_policy_version",
            )
        )
    ):
        raise PublishedBundleVerificationError("manifest does not bind exact terminal preparation")
    source_revision = terminal.get("source_revision")
    if not isinstance(source_revision, dict) or frozenset(source_revision) != {
        "campaign_revision",
        "last_event_seq",
    }:
        raise PublishedBundleVerificationError("manifest source revision schema differs")
    if (
        type(source_revision["campaign_revision"]) is not int
        or source_revision["campaign_revision"] < 0
        or type(source_revision["last_event_seq"]) is not int
        or source_revision["last_event_seq"] <= 0
    ):
        raise PublishedBundleVerificationError("manifest source revision values are invalid")
    if (
        manifest_value.get("contract_id") != contract_id
        or manifest_value.get("campaign_id") != campaign_id
        or manifest_value.get("selected_prediction_id") != selected_prediction_id
        or manifest_value.get("submission_sha256") != submission_sha256
    ):
        raise PublishedBundleVerificationError("manifest lineage differs from terminal preparation")
    replay_receipt_sha = manifest_value.get("replay_receipt_sha256")
    if not isinstance(replay_receipt_sha, str):
        raise PublishedBundleVerificationError("manifest lacks replay receipt digest")
    replay_receipt_sha = _sha256(replay_receipt_sha, "replay digest")
    if members["submission.csv"][0] != submission_sha256:
        raise PublishedBundleVerificationError("submission member differs from its manifest digest")
    if members["replay-receipt.json"][0] != replay_receipt_sha:
        raise PublishedBundleVerificationError("replay receipt differs from its manifest digest")
    try:
        replay_receipt = json.loads((root / "replay-receipt.json").read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublishedBundleVerificationError("replay receipt is unreadable") from exc
    if not isinstance(replay_receipt, dict) or not _prepared_replay_binds_receipt(
        prepared_replay_json, replay_receipt
    ):
        raise PublishedBundleVerificationError(
            "replay receipt is not bound by the prepared replay payload"
        )
    if replay_receipt.get("schema_version") == 3:
        try:
            campaign_evidence = json.loads((root / "campaign-manifest.json").read_text("utf-8"))
            scientific_evidence = json.loads((root / "scientific-decision.json").read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PublishedBundleVerificationError(
                "production organizer evidence wrappers are unreadable"
            ) from exc
        if not isinstance(campaign_evidence, dict) or not isinstance(scientific_evidence, dict):
            raise PublishedBundleVerificationError(
                "production organizer evidence wrappers must be objects"
            )
        admission = campaign_evidence.get("production_admission")
        admission_data = admission.get("data") if isinstance(admission, Mapping) else None
        organizer = replay_receipt.get("organizer_check")
        if (
            campaign_evidence.get("contract_id") != contract_id
            or campaign_evidence.get("campaign_id") != campaign_id
            or not isinstance(admission, Mapping)
            or not isinstance(admission_data, Mapping)
            or not isinstance(organizer, Mapping)
            or scientific_evidence.get("organizer_check") != dict(organizer)
            or scientific_evidence.get("exact_replay_evidence")
            != {
                "clean_replay_evidence_sha256": replay_receipt.get("clean_replay_evidence_sha256"),
                "clean_replay_evidence": replay_receipt.get("clean_replay_evidence"),
                "scoring_exact_receipt": replay_receipt.get("scoring_exact_receipt"),
                "same_backend_receipt": replay_receipt.get("same_backend_receipt"),
            }
        ):
            raise PublishedBundleVerificationError(
                "production decision evidence differs from retained replay proof"
            )
        organizer_digest = replay_receipt.get("organizer_check_sha256")
        if canonical_json_sha256(dict(organizer)) != organizer_digest:
            raise PublishedBundleVerificationError(
                "production organizer evidence digest differs from replay receipt"
            )
        try:
            validate_organizer_check_manifest(
                organizer,
                expected_submission_sha256=submission_sha256,
                expected_submission_size_bytes=members["submission.csv"][1],
                expected_starter_manifest_sha256=_sha256(
                    admission.get("starter_manifest_sha256"),
                    "production admission starter manifest",
                ),
                expected_final_rows=_positive_int(
                    admission_data.get("final_rows"), "production admission final_rows"
                ),
            )
        except (OrganizerCheckError, StateError) as exc:
            raise PublishedBundleVerificationError(
                f"production organizer evidence is invalid: {exc}"
            ) from exc
    derived_bundle = BundleId.derive(
        selected_prediction_id=PredictionId(selected_prediction_id),
        replay_output_sha256={"replay-receipt.json": replay_receipt_sha},
        submission_sha256=submission_sha256,
        manifest_sha256=manifest_sha256,
    ).value
    if derived_bundle != bundle_id:
        raise PublishedBundleVerificationError("BundleId does not derive from published manifest")
    try:
        digest_line = (root / "bundle.sha256").read_text("ascii")
    except (OSError, UnicodeError) as exc:
        raise PublishedBundleVerificationError("bundle digest file is unreadable") from exc
    if digest_line != f"{bundle_id}\n":
        raise PublishedBundleVerificationError("bundle digest file differs from BundleId")
    try:
        snapshot = sqlite3.connect(
            f"file:{(root / 'campaign-state-snapshot.sqlite3').as_posix()}?mode=ro", uri=True
        )
        projection_row = snapshot.execute(
            """
            SELECT preparation_id, projection_sha256, projection_json
            FROM projection_metadata WHERE singleton = 1
            """
        ).fetchone()
        integrity = snapshot.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        raise PublishedBundleVerificationError("terminal SQLite projection is invalid") from exc
    finally:
        if "snapshot" in locals():
            snapshot.close()
    if (
        projection_row is None
        or tuple(projection_row[:2]) != (preparation_id, projection_sha256)
        or integrity != ("ok",)
    ):
        raise PublishedBundleVerificationError("SQLite projection differs from preparation")
    stored_projection_json = str(projection_row[2])
    if hashlib.sha256(stored_projection_json.encode()).hexdigest() != projection_sha256:
        raise PublishedBundleVerificationError("SQLite projection payload digest differs")
    projection_value = json.loads(stored_projection_json)
    projection_terminal = projection_value.get("terminal_projection")
    if not isinstance(projection_terminal, dict) or terminal.get(
        "source_revision"
    ) != projection_terminal.get("source_revision"):
        raise PublishedBundleVerificationError("manifest source revision differs from projection")
    expected_events = b"".join(
        (
            _canonical_json_preserve_numbers(event, "terminal event").encode("ascii")
            if replay_receipt.get("schema_version") == 3
            else canonical_json_bytes(event)
        )
        + b"\n"
        for event in projection_value["events"]
    )
    try:
        actual_events = (root / "event-export.jsonl").read_bytes()
    except OSError as exc:
        raise PublishedBundleVerificationError("terminal event export is unreadable") from exc
    if actual_events != expected_events:
        raise PublishedBundleVerificationError("event export differs from prepared projection")
    expected_resource_receipts = (
        canonical_json_bytes(dict(resource_receipt_payload) | {"receipt_id": resource_receipt_id})
        + b"\n"
    )
    try:
        actual_resource_receipts = (root / "resource-receipts.jsonl").read_bytes()
    except OSError as exc:
        raise PublishedBundleVerificationError("resource receipt evidence is unreadable") from exc
    if actual_resource_receipts != expected_resource_receipts:
        raise PublishedBundleVerificationError(
            "resource receipt evidence differs from authoritative campaign record"
        )
    evidence = manifest_value.get("evidence")
    if not isinstance(evidence, list):
        raise PublishedBundleVerificationError("manifest lacks evidence receipts")
    if len(evidence) != len(_BUNDLE_EVIDENCE_MEMBERS):
        raise PublishedBundleVerificationError("manifest evidence receipt count differs")
    evidence_by_role = {
        item.get("role"): item
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("role"), str)
    }
    if frozenset(evidence_by_role) != _BUNDLE_EVIDENCE_MEMBERS:
        raise PublishedBundleVerificationError("manifest evidence roles differ from exact layout")
    for role in _BUNDLE_EVIDENCE_MEMBERS:
        receipt = evidence_by_role.get(role)
        if (
            not isinstance(receipt, dict)
            or frozenset(receipt)
            != {"schema_version", "receipt_id", "role", "sha256", "size_bytes"}
            or type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != 1
            or (receipt.get("sha256"), receipt.get("size_bytes")) != members[role]
        ):
            raise PublishedBundleVerificationError(f"manifest evidence receipt differs for {role}")
        receipt_body = {
            "schema_version": 1,
            "role": role,
            "sha256": members[role][0],
            "size_bytes": members[role][1],
        }
        expected_receipt_id = hashlib.sha256(
            _FROZEN_FILE_RECEIPT_DOMAIN + canonical_json_bytes(receipt_body)
        ).hexdigest()
        if receipt.get("receipt_id") != expected_receipt_id:
            raise PublishedBundleVerificationError(
                f"manifest evidence receipt identity differs for {role}"
            )
    return root


@dataclass(frozen=True, slots=True)
class _NormalizedRecord:
    kind: RecordKind
    table: str
    record_id: str
    campaign_id: str
    contract_id: str
    references: Mapping[str, str]
    attributes: Mapping[str, object]
    payload_json: str
    state: str | None
    terminal: bool


_RECORD_TABLES: Final[Mapping[RecordKind, str]] = MappingProxyType(
    {
        RecordKind.FAMILY: "families",
        RecordKind.EXPERIMENT: "campaign_experiments",
        RecordKind.TRIAL: "trials",
        RecordKind.ATTEMPT: "attempts",
        RecordKind.ARTIFACT: "artifacts",
        RecordKind.PREDICTION: "predictions",
        RecordKind.INNER_EVALUATION: "inner_evaluations",
        RecordKind.PROMOTION_DECISION: "promotion_decisions",
        RecordKind.RANK_GRAPH: "rank_graphs",
        RecordKind.REPLAY: "replays",
        RecordKind.BUNDLE: "bundles",
        RecordKind.RESOURCE_RECEIPT: "resource_receipts",
        RecordKind.PROVIDER_OPERATION: "provider_operations",
        RecordKind.FAILURE: "failures",
    }
)
_FAILURE_ENTITY_TARGETS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "campaign": ("campaigns", "campaign_id"),
        "family": ("families", "family_id"),
        "experiment": ("campaign_experiments", "experiment_id"),
        "trial": ("trials", "trial_id"),
        "attempt": ("attempts", "attempt_id"),
        "artifact": ("artifacts", "artifact_id"),
        "prediction": ("predictions", "prediction_id"),
        "inner_evaluation": ("inner_evaluations", "evaluation_id"),
        "promotion_decision": ("promotion_decisions", "decision_id"),
        "rank_graph": ("rank_graphs", "rank_graph_id"),
        "replay": ("replays", "replay_id"),
        "bundle": ("bundles", "bundle_id"),
        "resource_receipt": ("resource_receipts", "receipt_id"),
        "provider_operation": ("provider_operations", "operation_id"),
        "protected_query_reservation": (
            "protected_query_reservations",
            "reservation_id",
        ),
        "protected_evaluation": ("protected_evaluations", "evaluation_id"),
        "selection_decision": ("selection_decisions", "decision_id"),
    }
)

_REFERENCE_SCHEMAS: Final[Mapping[RecordKind, tuple[frozenset[str], frozenset[str]]]] = (
    MappingProxyType(
        {
            RecordKind.FAMILY: (frozenset(), frozenset()),
            RecordKind.EXPERIMENT: (
                frozenset({"family_id"}),
                frozenset({"parent_experiment_id"}),
            ),
            RecordKind.TRIAL: (frozenset({"experiment_id"}), frozenset()),
            RecordKind.ATTEMPT: (frozenset({"trial_id"}), frozenset()),
            RecordKind.ARTIFACT: (frozenset(), frozenset({"attempt_id"})),
            RecordKind.PREDICTION: (
                frozenset({"artifact_id"}),
                frozenset({"trial_id", "rank_graph_id"}),
            ),
            RecordKind.INNER_EVALUATION: (frozenset({"prediction_id"}), frozenset()),
            RecordKind.PROMOTION_DECISION: (frozenset({"prediction_id"}), frozenset()),
            RecordKind.RANK_GRAPH: (frozenset({"family_id"}), frozenset()),
            RecordKind.REPLAY: (frozenset({"prediction_id"}), frozenset()),
            RecordKind.BUNDLE: (
                frozenset({"replay_id", "selected_prediction_id"}),
                frozenset(),
            ),
            RecordKind.RESOURCE_RECEIPT: (frozenset(), frozenset({"attempt_id"})),
            RecordKind.PROVIDER_OPERATION: (frozenset(), frozenset()),
            RecordKind.FAILURE: (frozenset({"entity_id"}), frozenset()),
        }
    )
)

_ATTRIBUTE_SCHEMAS: Final[Mapping[RecordKind, tuple[frozenset[str], frozenset[str]]]] = (
    MappingProxyType(
        {
            RecordKind.FAMILY: (frozenset({"protected_eligible"}), frozenset()),
            RecordKind.EXPERIMENT: (frozenset(), frozenset()),
            RecordKind.TRIAL: (frozenset(), frozenset()),
            RecordKind.ATTEMPT: (
                frozenset({"attempt_ordinal"}),
                frozenset({"process_identity"}),
            ),
            RecordKind.ARTIFACT: (
                frozenset({"kind", "relative_path", "sha256", "size_bytes", "verified_path"}),
                frozenset(),
            ),
            RecordKind.PREDICTION: (
                frozenset({"ordered_rows_sha256", "prediction_bytes_sha256"}),
                frozenset(),
            ),
            RecordKind.INNER_EVALUATION: (frozenset(), frozenset()),
            RecordKind.PROMOTION_DECISION: (frozenset(), frozenset()),
            RecordKind.RANK_GRAPH: (frozenset(), frozenset()),
            RecordKind.REPLAY: (frozenset(), frozenset()),
            RecordKind.BUNDLE: (frozenset({"manifest_sha256"}), frozenset()),
            RecordKind.RESOURCE_RECEIPT: (frozenset(), frozenset()),
            RecordKind.PROVIDER_OPERATION: (frozenset(), frozenset()),
            RecordKind.FAILURE: (
                frozenset({"entity_kind", "failure_kind"}),
                frozenset(),
            ),
        }
    )
)

_STATEFUL_RECORD_KINDS: Final = frozenset(
    {
        RecordKind.TRIAL,
        RecordKind.ATTEMPT,
        RecordKind.REPLAY,
        RecordKind.BUNDLE,
        RecordKind.PROVIDER_OPERATION,
    }
)


def _normalize_record(record: DurableRecord) -> _NormalizedRecord:
    kind = record.kind
    if not isinstance(kind, RecordKind):
        raise StateInvariantError("record kind must be RecordKind")
    record_id = _identifier(record.record_id, "record_id")
    campaign_id = _identifier(record.campaign_id, "campaign_id")
    contract_id = _identifier(record.contract_id, "contract_id")
    required_refs, optional_refs = _REFERENCE_SCHEMAS[kind]
    actual_ref_keys = frozenset(record.references)
    if not required_refs <= actual_ref_keys or not actual_ref_keys <= required_refs | optional_refs:
        raise StateInvariantError(
            f"{kind.value} references must include {sorted(required_refs)!r} and only "
            f"optional {sorted(optional_refs)!r}"
        )
    references = {key: _identifier(value, key) for key, value in record.references.items()}
    required_attrs, optional_attrs = _ATTRIBUTE_SCHEMAS[kind]
    actual_attr_keys = frozenset(record.attributes)
    if (
        not required_attrs <= actual_attr_keys
        or not actual_attr_keys <= required_attrs | optional_attrs
    ):
        raise StateInvariantError(
            f"{kind.value} attributes must include {sorted(required_attrs)!r} and only "
            f"optional {sorted(optional_attrs)!r}"
        )
    attributes = dict(record.attributes)
    if kind in _STATEFUL_RECORD_KINDS:
        state = _text(record.state, "state")
        if record.terminal:
            raise StateInvariantError("terminal stateful records must use transition/finalization")
        if kind.value in _ATOMIC_FINAL_STATES and state in _ATOMIC_FINAL_STATES[kind.value]:
            raise StateInvariantError(f"{kind.value} cannot be registered in an atomic final state")
    elif record.state is not None or record.terminal:
        raise StateInvariantError("immutable record cannot declare state or terminal")
    else:
        state = None
    _validate_record_attributes(kind, references, attributes)
    payload_json = _canonical_json(record.payload, "record payload")
    return _NormalizedRecord(
        kind,
        _RECORD_TABLES[kind],
        record_id,
        campaign_id,
        contract_id,
        MappingProxyType(references),
        MappingProxyType(attributes),
        payload_json,
        state,
        False,
    )


def _registration_fence_scope(record: _NormalizedRecord) -> tuple[str | None, str | None]:
    """Return the narrowest already-durable parent whose lease fences registration."""

    references = record.references
    if record.kind in {RecordKind.EXPERIMENT, RecordKind.RANK_GRAPH}:
        return "family", references["family_id"]
    if record.kind is RecordKind.TRIAL:
        return "experiment", references["experiment_id"]
    if "attempt_id" in references:
        return "attempt", references["attempt_id"]
    if record.kind is RecordKind.ATTEMPT:
        return "trial", references["trial_id"]
    if record.kind is RecordKind.PREDICTION and "trial_id" in references:
        return "trial", references["trial_id"]
    if record.kind in {
        RecordKind.INNER_EVALUATION,
        RecordKind.PROMOTION_DECISION,
        RecordKind.REPLAY,
    }:
        return "prediction", references["prediction_id"]
    if record.kind is RecordKind.BUNDLE:
        return "replay", references["replay_id"]
    return None, None


def _association_lease_resource_id(resource_kind: str, campaign_id: str, resource_id: str) -> str:
    return hashlib.sha256(f"{resource_kind}-lease:{campaign_id}:{resource_id}".encode()).hexdigest()


def _registration_payload_json(record: _NormalizedRecord) -> str:
    """Preserve the immutable initial attempt identity in its registration event."""

    if record.kind is not RecordKind.ATTEMPT:
        return record.payload_json
    process_identity = record.attributes.get("process_identity")
    return _canonical_json(
        {
            "initial_process_identity": process_identity,
            "record_payload": json.loads(record.payload_json),
        },
        "attempt registration payload",
    )


def _validate_record_attributes(
    kind: RecordKind, references: Mapping[str, str], attributes: Mapping[str, object]
) -> None:
    if kind is RecordKind.FAMILY and type(attributes["protected_eligible"]) is not bool:
        raise StateInvariantError("protected_eligible must be bool")
    if kind is RecordKind.ATTEMPT:
        _positive_int(attributes["attempt_ordinal"], "attempt_ordinal")
        process_identity = attributes.get("process_identity")
        if process_identity is not None:
            if not isinstance(process_identity, Mapping):
                raise StateInvariantError("process_identity must be a mapping")
            _canonical_json(process_identity, "process_identity")
    if kind is RecordKind.ARTIFACT:
        _text(attributes["kind"], "artifact kind")
        _relative_path(attributes["relative_path"])
        _sha256(attributes["sha256"], "artifact sha256")
        size = attributes["size_bytes"]
        if type(size) is not int or size < 0:
            raise StateInvariantError("artifact size_bytes must be non-negative")
        _verify_artifact(
            attributes["verified_path"],
            expected_sha256=cast(str, attributes["sha256"]),
            expected_size=size,
        )
    if kind is RecordKind.PREDICTION:
        if ("trial_id" in references) == ("rank_graph_id" in references):
            raise StateInvariantError("prediction must have exactly one trial_id or rank_graph_id")
        _sha256(attributes["ordered_rows_sha256"], "ordered_rows_sha256")
        _sha256(attributes["prediction_bytes_sha256"], "prediction_bytes_sha256")
    if kind is RecordKind.BUNDLE:
        _sha256(attributes["manifest_sha256"], "manifest_sha256")
    if kind is RecordKind.FAILURE:
        _text(attributes["entity_kind"], "entity_kind")
        _text(attributes["failure_kind"], "failure_kind")


def _record_insert(record: _NormalizedRecord, *, now: str) -> dict[str, object]:
    common: dict[str, object] = {
        _primary_key(record.table): record.record_id,
        "contract_id": record.contract_id,
        "campaign_id": record.campaign_id,
    }
    kind = record.kind
    refs = record.references
    attrs = record.attributes
    if kind is RecordKind.FAMILY:
        common.update(
            protected_eligible=int(cast(bool, attrs["protected_eligible"])),
            payload_json=record.payload_json,
            created_at=now,
        )
    elif kind is RecordKind.EXPERIMENT:
        common.update(
            family_id=refs["family_id"],
            parent_experiment_id=refs.get("parent_experiment_id"),
            payload_json=record.payload_json,
            created_at=now,
        )
    elif kind in _STATEFUL_RECORD_KINDS:
        if kind is RecordKind.TRIAL:
            common["experiment_id"] = refs["experiment_id"]
        elif kind is RecordKind.ATTEMPT:
            common.update(
                trial_id=refs["trial_id"],
                attempt_ordinal=attrs["attempt_ordinal"],
                process_identity_json=(
                    _canonical_json(attrs["process_identity"], "process_identity")
                    if "process_identity" in attrs
                    else None
                ),
            )
        elif kind is RecordKind.REPLAY:
            common["prediction_id"] = refs["prediction_id"]
        elif kind is RecordKind.BUNDLE:
            common.update(
                replay_id=refs["replay_id"],
                selected_prediction_id=refs["selected_prediction_id"],
                manifest_sha256=attrs["manifest_sha256"],
            )
        common.update(
            state=record.state,
            revision=0,
            terminal=0,
            payload_json=record.payload_json,
            created_at=now,
            updated_at=now,
        )
    elif kind is RecordKind.ARTIFACT:
        common.update(
            attempt_id=refs.get("attempt_id"),
            kind=attrs["kind"],
            relative_path=attrs["relative_path"],
            sha256=attrs["sha256"],
            size_bytes=attrs["size_bytes"],
            payload_json=record.payload_json,
            created_at=now,
        )
    elif kind is RecordKind.PREDICTION:
        common.update(
            trial_id=refs.get("trial_id"),
            rank_graph_id=refs.get("rank_graph_id"),
            artifact_id=refs["artifact_id"],
            ordered_rows_sha256=attrs["ordered_rows_sha256"],
            prediction_bytes_sha256=attrs["prediction_bytes_sha256"],
            payload_json=record.payload_json,
            created_at=now,
        )
    elif kind in {RecordKind.INNER_EVALUATION, RecordKind.PROMOTION_DECISION}:
        common.update(
            prediction_id=refs["prediction_id"],
            payload_json=record.payload_json,
            created_at=now,
        )
    elif kind is RecordKind.RANK_GRAPH:
        common.update(family_id=refs["family_id"], payload_json=record.payload_json, created_at=now)
    elif kind is RecordKind.RESOURCE_RECEIPT:
        common.update(
            attempt_id=refs.get("attempt_id"), payload_json=record.payload_json, created_at=now
        )
    elif kind is RecordKind.FAILURE:
        common.update(
            entity_kind=attrs["entity_kind"],
            entity_id=refs["entity_id"],
            failure_kind=attrs["failure_kind"],
            payload_json=record.payload_json,
            created_at=now,
        )
    else:  # pragma: no cover - exhaustive over RecordKind
        raise StateInvariantError(f"unsupported record kind {kind.value}")
    return common


def _primary_key(table: str) -> str:
    return {
        "campaigns": "campaign_id",
        "families": "family_id",
        "campaign_experiments": "experiment_id",
        "trials": "trial_id",
        "attempts": "attempt_id",
        "artifacts": "artifact_id",
        "predictions": "prediction_id",
        "inner_evaluations": "evaluation_id",
        "promotion_decisions": "decision_id",
        "rank_graphs": "rank_graph_id",
        "replays": "replay_id",
        "bundles": "bundle_id",
        "resource_receipts": "receipt_id",
        "provider_operations": "operation_id",
        "failures": "failure_id",
    }[table]


def _identifier(value: IdInput, name: str) -> str:
    raw = value.value if isinstance(value, ValueIdentifier) else value
    return _text(raw, name)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or "\n" in value:
        raise StateInvariantError(f"{name} must be one non-empty normalized line")
    return value


def _sha256(value: object, name: str) -> str:
    text = _text(value, name)
    if _SHA256_RE.fullmatch(text) is None:
        raise StateInvariantError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _relative_path(value: object) -> str:
    text = _text(value, "relative_path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise StateInvariantError("relative_path must be normalized and remain below its root")
    return text


def _verify_artifact(value: object, *, expected_sha256: str, expected_size: int) -> None:
    if not isinstance(value, Path):
        raise StateInvariantError("verified_path must be a pathlib.Path")
    try:
        before = value.lstat()
        if not stat.S_ISREG(before.st_mode) or value.is_symlink():
            raise StateInvariantError("verified artifact must be a non-symlink regular file")
        digest = hashlib.sha256()
        with value.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = value.lstat()
    except OSError as exc:
        raise StateInvariantError("verified artifact is not stably readable") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise StateInvariantError("verified artifact changed while being hashed")
    if after.st_size != expected_size or digest.hexdigest() != expected_sha256:
        raise StateInvariantError("verified artifact size or SHA-256 does not match registration")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise StateInvariantError(f"{name} must be a positive integer")
    return value


def _ttl_seconds(value: timedelta) -> float:
    if not isinstance(value, timedelta):
        raise StateInvariantError("ttl must be datetime.timedelta")
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise StateInvariantError("ttl must be finite and positive")
    return seconds


def _canonical_json(value: object, name: str) -> str:
    try:
        encoded = canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StateInvariantError(f"{name} must be finite JSON") from exc
    decoded: object = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise StateInvariantError(f"{name} must be a JSON object")
    return encoded


def _canonical_json_preserve_numbers(value: object, name: str) -> str:
    """Encode finite canonical JSON without rewriting receipt-significant float spellings."""

    try:
        normalized = _json_value_preserve_numbers(value, location=name)
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        decoded: object = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StateInvariantError(f"{name} must be finite JSON") from exc
    if not isinstance(decoded, dict):
        raise StateInvariantError(f"{name} must be a JSON object")
    return encoded


def _json_value_preserve_numbers(value: object, *, location: str) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, candidate in value.items():
            if type(key) is not str:
                raise TypeError(f"{location} contains a non-text object key")
            normalized[key] = _json_value_preserve_numbers(candidate, location=f"{location}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value_preserve_numbers(candidate, location=f"{location}[{index}]")
            for index, candidate in enumerate(value)
        ]
    raise TypeError(f"{location} contains unsupported type {type(value).__name__}")


def _json_mapping(value: str) -> Mapping[str, object]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise StateInvariantError("stored JSON is not an object")
    return MappingProxyType(cast(dict[str, object], decoded))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "CampaignHandle",
    "CampaignSourceRevision",
    "DurableRecord",
    "FamilyEvidenceReceipt",
    "IdInput",
    "Lease",
    "LeaseConflictError",
    "PostpublicationResourceReceipt",
    "PreparedFinalizationRequiredError",
    "PreparedProjectionArtifacts",
    "PreparedSourceStaleError",
    "PreparedTerminalProjection",
    "ProtectedBudgetExhaustedError",
    "ProtectedOutcome",
    "ProtectedOutcomeKind",
    "ProtectedOutcomeTerminalError",
    "ProtectedReservation",
    "PublishedBundleConflictError",
    "PublishedBundleReceipt",
    "PublishedBundleVerificationError",
    "Reconciliation",
    "RecordKind",
    "StateConflictError",
    "StateError",
    "StateInvariantError",
    "StateNotFoundError",
    "StateRepository",
    "TerminalPreparation",
    "TerminalPreparationConflictError",
    "Transition",
    "ValueIdentifier",
]
