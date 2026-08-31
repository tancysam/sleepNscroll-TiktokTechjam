from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kuairand_agent.domain.identity import CampaignId, ContractId, FamilyId
from kuairand_agent.state import (
    DurableRecord,
    RecordKind,
    StateConflictError,
    StateInvariantError,
    StateRepository,
)
from kuairand_agent.state import schema as state_schema
from kuairand_agent.state.schema import APPLICATION_ID, SCHEMA_VERSION


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(slots=True)
class _Clock:
    value: datetime = datetime(2026, 8, 31, 1, 2, 3, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def _repository(tmp_path: Path) -> tuple[StateRepository, ContractId, CampaignId]:
    repository = StateRepository.open(tmp_path / "state", clock=_Clock())
    contract = ContractId(_digest("contract"))
    campaign = CampaignId(_digest("campaign"))
    created = repository.create_campaign(
        campaign_id=campaign,
        contract_id=contract,
        contract_manifest={"metric": "primary", "numeric": 1.0},
        config={"profile": "cpu"},
        idempotency_key="campaign-request-1",
        protected_query_limit=2,
    )
    assert created.created
    return repository, contract, campaign


def test_open_enables_required_sqlite_safety_and_migrates_minimum_tables(tmp_path: Path) -> None:
    repository, _, _ = _repository(tmp_path)

    with sqlite3.connect(repository.database_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])

    assert journal_mode == "wal"
    assert application_id == APPLICATION_ID
    assert schema_version == SCHEMA_VERSION
    assert {
        "contracts",
        "campaigns",
        "campaign_events",
        "families",
        "family_ledger",
        "family_evidence",
        "experiment_ledger",
        "campaign_experiments",
        "trials",
        "attempts",
        "artifacts",
        "predictions",
        "inner_evaluations",
        "protected_query_reservations",
        "protected_evaluations",
        "promotion_decisions",
        "rank_graphs",
        "selection_decisions",
        "replays",
        "bundles",
        "resource_receipts",
        "provider_operations",
        "failures",
        "leases",
        "terminal_preparations",
        "bundle_publications",
    } <= tables


def test_v1_migration_backfills_experiment_ledger_and_campaign_association(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    database_path = state_root / "authority.sqlite3"
    contract = _digest("migration-contract")
    campaign = _digest("migration-campaign")
    family = _digest("migration-family")
    experiment = _digest("migration-experiment")
    timestamp = "2026-08-31T00:00:00.000000Z"
    with sqlite3.connect(database_path, isolation_level=None) as connection:
        state_schema.configure_connection(connection)
        state_schema._apply_migration_one(connection, created_at=timestamp)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO contracts VALUES (?, '{}', 1, ?)", (contract, timestamp))
        connection.execute(
            """
            INSERT INTO campaigns(
                campaign_id, contract_id, idempotency_key, config_json, state, revision,
                next_event_seq, protected_query_limit, research_frozen, terminal,
                started_at, updated_at
            ) VALUES (?, ?, 'migration', '{}', 'READY', 0, 1, 1, 0, 0, ?, ?)
            """,
            (campaign, contract, timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO family_ledger VALUES (?, ?, 0, '{}', ?)",
            (contract, family, timestamp),
        )
        connection.execute(
            "INSERT INTO families VALUES (?, ?, ?, 0, '{}', ?)",
            (family, contract, campaign, timestamp),
        )
        connection.execute(
            """
            INSERT INTO experiments(
                experiment_id, contract_id, campaign_id, family_id,
                parent_experiment_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, NULL, '{"spec":"v1"}', ?)
            """,
            (experiment, contract, campaign, family, timestamp),
        )
        connection.execute(
            """
            INSERT INTO trials(
                trial_id, contract_id, campaign_id, experiment_id, state, revision,
                terminal, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'READY', 0, 0, '{}', ?, ?)
            """,
            (_digest("migration-trial"), contract, campaign, experiment, timestamp, timestamp),
        )
        connection.commit()
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 1

    repository = StateRepository.open(state_root)
    with sqlite3.connect(repository.database_path) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
        ledger = connection.execute(
            """
            SELECT family_id, payload_json FROM experiment_ledger
            WHERE contract_id = ? AND experiment_id = ?
            """,
            (contract, experiment),
        ).fetchone()
        association = connection.execute(
            """
            SELECT contract_id, family_id FROM campaign_experiments
            WHERE campaign_id = ? AND experiment_id = ?
            """,
            (campaign, experiment),
        ).fetchone()
        migrated_trial = connection.execute(
            "SELECT campaign_id, experiment_id FROM trials WHERE trial_id = ?",
            (_digest("migration-trial"),),
        ).fetchone()
        legacy_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experiments'"
        ).fetchone()

    assert ledger == (family, '{"spec":"v1"}')
    assert association == (contract, family)
    assert migrated_trial == (campaign, experiment)
    assert legacy_table is None


def test_campaign_creation_is_content_idempotent_and_canonical(tmp_path: Path) -> None:
    repository, contract, campaign = _repository(tmp_path)

    replay = repository.create_campaign(
        campaign_id=campaign,
        contract_id=contract,
        contract_manifest={"numeric": 1, "metric": "primary"},
        config={"profile": "cpu"},
        idempotency_key="campaign-request-1",
        protected_query_limit=2,
    )

    assert not replay.created
    assert replay.revision == 0
    with pytest.raises(StateInvariantError, match="atomic final state"):
        repository.create_campaign(
            campaign_id=CampaignId(_digest("initially-complete-campaign")),
            contract_id=contract,
            contract_manifest={"numeric": 1, "metric": "primary"},
            config={"profile": "cpu"},
            idempotency_key="initially-complete-campaign",
            protected_query_limit=2,
            initial_state="COMPLETED",
        )
    with pytest.raises(StateInvariantError, match="different content"):
        repository.create_campaign(
            campaign_id=campaign,
            contract_id=contract,
            contract_manifest={"metric": "primary", "numeric": 1},
            config={"profile": "gpu"},
            idempotency_key="campaign-request-1",
            protected_query_limit=2,
        )


def test_registration_and_state_change_share_the_immutable_event_path(tmp_path: Path) -> None:
    repository, contract, campaign = _repository(tmp_path)
    family = FamilyId(_digest("family"))
    trial = _digest("trial")
    experiment = _digest("experiment")
    family_record = DurableRecord(
        kind=RecordKind.FAMILY,
        record_id=family,
        campaign_id=campaign,
        contract_id=contract,
        attributes={"protected_eligible": True},
        payload={"mechanism": "official-fm"},
    )
    assert repository.register(family_record)
    assert not repository.register(family_record)
    repository.register(
        DurableRecord(
            kind=RecordKind.EXPERIMENT,
            record_id=experiment,
            campaign_id=campaign,
            contract_id=contract,
            references={"family_id": family},
            payload={"spec": "frozen"},
        )
    )
    repository.register(
        DurableRecord(
            kind=RecordKind.TRIAL,
            record_id=trial,
            campaign_id=campaign,
            contract_id=contract,
            references={"experiment_id": experiment},
            payload={"backend": "cpu"},
            state="PENDING",
        )
    )
    attempt_record = DurableRecord(
        kind=RecordKind.ATTEMPT,
        record_id=_digest("attempt"),
        campaign_id=campaign,
        contract_id=contract,
        references={"trial_id": trial},
        attributes={"attempt_ordinal": 1},
        state="STARTING",
    )
    repository.register(attempt_record)
    repository.transition(
        campaign_id=campaign,
        entity_kind="attempt",
        entity_id=attempt_record.record_id,
        expected_state="STARTING",
        expected_revision=0,
        new_state="RUNNING",
        event_type="attempt_process_attached",
        process_identity={"pid": 42, "start_time": 1.0},
    )
    assert not repository.register(attempt_record)
    with pytest.raises(StateInvariantError, match="initial registration"):
        repository.register(
            DurableRecord(
                kind=RecordKind.ATTEMPT,
                record_id=attempt_record.record_id,
                campaign_id=campaign,
                contract_id=contract,
                references={"trial_id": trial},
                attributes={"attempt_ordinal": 1, "process_identity": {"pid": 99}},
                state="STARTING",
            )
        )

    transition = repository.transition(
        campaign_id=campaign,
        entity_kind="trial",
        entity_id=trial,
        expected_state="PENDING",
        expected_revision=0,
        new_state="RUNNING",
        event_type="trial_started",
        payload={"worker": "worker-a"},
    )

    assert transition.revision == 1
    with pytest.raises(StateInvariantError, match="atomic finalize_campaign"):
        repository.transition(
            campaign_id=campaign,
            entity_kind="campaign",
            entity_id=campaign,
            expected_state="READY",
            expected_revision=0,
            new_state="COMPLETED",
            event_type="unsafe_completion",
            terminal=True,
        )
    with pytest.raises(StateInvariantError, match="atomic finalize_campaign"):
        repository.transition(
            campaign_id=campaign,
            entity_kind="campaign",
            entity_id=campaign,
            expected_state="READY",
            expected_revision=0,
            new_state="COMPLETED",
            event_type="unsafe_nonterminal_completion",
            terminal=False,
        )
    with pytest.raises(StateConflictError, match="compare-and-swap"):
        repository.transition(
            campaign_id=campaign,
            entity_kind="trial",
            entity_id=trial,
            expected_state="PENDING",
            expected_revision=0,
            new_state="RUNNING",
            event_type="duplicate_start",
        )
    with (
        sqlite3.connect(repository.database_path) as connection,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        connection.execute(
            "UPDATE campaign_events SET event_type = 'rewritten' WHERE campaign_id = ?",
            (campaign.value,),
        )

    with pytest.raises(StateInvariantError, match="outside campaign lineage"):
        repository.register(
            DurableRecord(
                RecordKind.FAILURE,
                _digest("orphan-failure"),
                campaign,
                contract,
                references={"entity_id": _digest("missing-trial")},
                attributes={"entity_kind": "trial", "failure_kind": "TIMEOUT"},
            )
        )


def test_registration_rejects_unsafe_artifacts_and_nonfinite_payloads(tmp_path: Path) -> None:
    repository, contract, campaign = _repository(tmp_path)
    verified = tmp_path / "artifact.bin"
    verified.write_bytes(b"data")
    with pytest.raises(StateInvariantError, match="relative_path"):
        repository.register(
            DurableRecord(
                kind=RecordKind.ARTIFACT,
                record_id=_digest("artifact"),
                campaign_id=campaign,
                contract_id=contract,
                attributes={
                    "kind": "prediction",
                    "relative_path": "../escape.bin",
                    "sha256": hashlib.sha256(b"data").hexdigest(),
                    "size_bytes": 4,
                    "verified_path": verified,
                },
            )
        )
    with pytest.raises(StateInvariantError, match="finite JSON"):
        repository.register(
            DurableRecord(
                kind=RecordKind.FAMILY,
                record_id=_digest("bad-family"),
                campaign_id=campaign,
                contract_id=contract,
                attributes={"protected_eligible": False},
                payload={"bad": float("nan")},
            )
        )
