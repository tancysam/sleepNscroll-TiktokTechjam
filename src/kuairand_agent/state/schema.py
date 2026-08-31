"""SQLite schema and migrations for the single laboratory state authority.

The schema deliberately keeps every correctness-critical record in one database.  JSON files and
reports are projections; they never participate in a write decision.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Final

APPLICATION_ID: Final = 0x4B524153  # ``KRAS``
SCHEMA_VERSION: Final = 4
DEFAULT_BUSY_TIMEOUT_MS: Final = 5_000


class SchemaError(RuntimeError):
    """Raised when the authority database cannot be opened safely."""


_MIGRATION_1: Final[tuple[str, ...]] = (
    """
    CREATE TABLE authority_metadata (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        created_at TEXT NOT NULL,
        schema_version INTEGER NOT NULL CHECK (schema_version > 0)
    ) STRICT
    """,
    """
    CREATE TABLE contracts (
        contract_id TEXT PRIMARY KEY,
        manifest_json TEXT NOT NULL,
        protected_query_limit INTEGER NOT NULL CHECK (protected_query_limit >= 0),
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE campaigns (
        campaign_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        idempotency_key TEXT NOT NULL UNIQUE,
        config_json TEXT NOT NULL,
        state TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        next_event_seq INTEGER NOT NULL DEFAULT 1 CHECK (next_event_seq > 0),
        protected_query_limit INTEGER NOT NULL CHECK (protected_query_limit >= 0),
        research_frozen INTEGER NOT NULL DEFAULT 0 CHECK (research_frozen IN (0, 1)),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        selected_prediction_id TEXT,
        fallback_prediction_id TEXT,
        started_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    ) STRICT
    """,
    """
    CREATE TABLE campaign_events (
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        event_seq INTEGER NOT NULL CHECK (event_seq > 0),
        event_type TEXT NOT NULL,
        entity_kind TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        prior_state TEXT,
        new_state TEXT,
        revision INTEGER NOT NULL CHECK (revision >= 0),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (campaign_id, event_seq)
    ) STRICT
    """,
    """
    CREATE TABLE families (
        family_id TEXT NOT NULL,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        protected_eligible INTEGER NOT NULL DEFAULT 0 CHECK (protected_eligible IN (0, 1)),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (campaign_id, family_id),
        UNIQUE (campaign_id, contract_id, family_id)
    ) STRICT
    """,
    """
    CREATE TABLE family_ledger (
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        family_id TEXT NOT NULL,
        protected_eligible INTEGER NOT NULL CHECK (protected_eligible IN (0, 1)),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (contract_id, family_id)
    ) STRICT
    """,
    """
    CREATE TABLE family_evidence (
        contract_id TEXT NOT NULL,
        family_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        origin_campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        representation TEXT NOT NULL,
        model_family TEXT NOT NULL,
        objective TEXT NOT NULL,
        temporal_policy TEXT NOT NULL,
        fusion_member TEXT NOT NULL,
        result TEXT NOT NULL CHECK (
            result IN ('improved', 'no_improvement', 'unsupported', 'infrastructure_failure')
        ),
        created_at TEXT NOT NULL,
        PRIMARY KEY (contract_id, family_id, fingerprint),
        FOREIGN KEY (contract_id, family_id)
            REFERENCES family_ledger(contract_id, family_id)
    ) STRICT
    """,
    """
    CREATE TABLE experiments (
        experiment_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        family_id TEXT NOT NULL,
        parent_experiment_id TEXT REFERENCES experiments(experiment_id),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (campaign_id, contract_id, family_id)
            REFERENCES families(campaign_id, contract_id, family_id)
    ) STRICT
    """,
    """
    CREATE TABLE trials (
        trial_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
        state TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE attempts (
        attempt_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        trial_id TEXT NOT NULL REFERENCES trials(trial_id),
        attempt_ordinal INTEGER NOT NULL CHECK (attempt_ordinal > 0),
        state TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        process_identity_json TEXT,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (trial_id, attempt_ordinal)
    ) STRICT
    """,
    """
    CREATE TABLE artifacts (
        artifact_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        attempt_id TEXT REFERENCES attempts(attempt_id),
        kind TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (campaign_id, relative_path)
    ) STRICT
    """,
    """
    CREATE TABLE predictions (
        prediction_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        trial_id TEXT REFERENCES trials(trial_id),
        rank_graph_id TEXT REFERENCES rank_graphs(rank_graph_id),
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
        ordered_rows_sha256 TEXT NOT NULL,
        prediction_bytes_sha256 TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK ((trial_id IS NULL) <> (rank_graph_id IS NULL))
    ) STRICT
    """,
    """
    CREATE TABLE inner_evaluations (
        evaluation_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE protected_query_reservations (
        reservation_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        family_id TEXT NOT NULL,
        prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        query_ordinal INTEGER NOT NULL CHECK (query_ordinal > 0),
        idempotency_key TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('RESERVED', 'RESULT', 'UNKNOWN')),
        created_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE (contract_id, query_ordinal),
        UNIQUE (contract_id, idempotency_key),
        UNIQUE (contract_id, prediction_id),
        FOREIGN KEY (campaign_id, contract_id, family_id)
            REFERENCES families(campaign_id, contract_id, family_id)
    ) STRICT
    """,
    """
    CREATE TABLE protected_evaluations (
        evaluation_id TEXT PRIMARY KEY,
        reservation_id TEXT NOT NULL UNIQUE REFERENCES protected_query_reservations(reservation_id),
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        outcome TEXT NOT NULL CHECK (outcome IN ('RESULT', 'UNKNOWN')),
        metrics_json TEXT,
        unknown_reason TEXT,
        created_at TEXT NOT NULL,
        CHECK (
            (outcome = 'RESULT' AND metrics_json IS NOT NULL AND unknown_reason IS NULL)
            OR (outcome = 'UNKNOWN' AND metrics_json IS NULL AND unknown_reason IS NOT NULL)
        )
    ) STRICT
    """,
    """
    CREATE TABLE promotion_decisions (
        decision_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE rank_graphs (
        rank_graph_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        family_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (campaign_id, contract_id, family_id)
            REFERENCES families(campaign_id, contract_id, family_id)
    ) STRICT
    """,
    """
    CREATE TABLE selection_decisions (
        decision_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        selected_prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        fallback_prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE replays (
        replay_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        state TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE bundles (
        bundle_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        replay_id TEXT NOT NULL REFERENCES replays(replay_id),
        selected_prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        state TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        manifest_sha256 TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE resource_receipts (
        receipt_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        attempt_id TEXT REFERENCES attempts(attempt_id),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE provider_operations (
        operation_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        state TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE failures (
        failure_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        entity_kind TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        failure_kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE leases (
        resource_kind TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        fence_token INTEGER NOT NULL CHECK (fence_token > 0),
        expires_at TEXT NOT NULL,
        completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
        updated_at TEXT NOT NULL,
        PRIMARY KEY (resource_kind, resource_id)
    ) STRICT
    """,
    """
    CREATE TABLE fencing_counters (
        resource_kind TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        next_token INTEGER NOT NULL CHECK (next_token > 0),
        PRIMARY KEY (resource_kind, resource_id)
    ) STRICT
    """,
    "CREATE INDEX campaign_events_entity ON campaign_events(entity_kind, entity_id)",
    "CREATE INDEX attempts_open ON attempts(campaign_id, state) WHERE terminal = 0",
    "CREATE INDEX protected_contract_state ON protected_query_reservations(contract_id, state)",
    "CREATE INDEX artifacts_campaign ON artifacts(campaign_id, kind)",
    "CREATE INDEX predictions_campaign ON predictions(campaign_id)",
    """
    CREATE TRIGGER campaign_events_no_update
    BEFORE UPDATE ON campaign_events
    BEGIN
        SELECT RAISE(ABORT, 'campaign events are immutable');
    END
    """,
    """
    CREATE TRIGGER campaign_events_no_delete
    BEFORE DELETE ON campaign_events
    BEGIN
        SELECT RAISE(ABORT, 'campaign events are immutable');
    END
    """,
    """
    CREATE TRIGGER protected_query_reservations_no_delete
    BEFORE DELETE ON protected_query_reservations
    BEGIN
        SELECT RAISE(ABORT, 'protected query reservations cannot be deleted');
    END
    """,
    """
    CREATE TRIGGER protected_query_reservations_guard_update
    BEFORE UPDATE ON protected_query_reservations
    WHEN OLD.state <> 'RESERVED'
      OR NEW.state NOT IN ('RESULT', 'UNKNOWN')
      OR NEW.completed_at IS NULL
      OR OLD.reservation_id IS NOT NEW.reservation_id
      OR OLD.contract_id IS NOT NEW.contract_id
      OR OLD.campaign_id IS NOT NEW.campaign_id
      OR OLD.family_id IS NOT NEW.family_id
      OR OLD.prediction_id IS NOT NEW.prediction_id
      OR OLD.query_ordinal IS NOT NEW.query_ordinal
      OR OLD.idempotency_key IS NOT NEW.idempotency_key
      OR OLD.created_at IS NOT NEW.created_at
    BEGIN
        SELECT RAISE(ABORT, 'protected reservation permits one terminal completion only');
    END
    """,
)

_MIGRATION_2: Final[tuple[str, ...]] = (
    """
    CREATE TABLE experiment_ledger (
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        experiment_id TEXT NOT NULL,
        family_id TEXT NOT NULL,
        parent_experiment_id TEXT,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (contract_id, experiment_id),
        FOREIGN KEY (contract_id, family_id)
            REFERENCES family_ledger(contract_id, family_id),
        FOREIGN KEY (contract_id, parent_experiment_id)
            REFERENCES experiment_ledger(contract_id, experiment_id)
    ) STRICT
    """,
    """
    CREATE TABLE campaign_experiments (
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        experiment_id TEXT NOT NULL,
        family_id TEXT NOT NULL,
        parent_experiment_id TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (campaign_id, experiment_id),
        UNIQUE (campaign_id, contract_id, experiment_id),
        FOREIGN KEY (contract_id, experiment_id)
            REFERENCES experiment_ledger(contract_id, experiment_id),
        FOREIGN KEY (campaign_id, contract_id, family_id)
            REFERENCES families(campaign_id, contract_id, family_id),
        FOREIGN KEY (campaign_id, parent_experiment_id)
            REFERENCES campaign_experiments(campaign_id, experiment_id)
    ) STRICT
    """,
)

_MIGRATION_3: Final[tuple[str, ...]] = (
    """
    CREATE TABLE terminal_preparations (
        preparation_id TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        source_state TEXT NOT NULL,
        source_campaign_revision INTEGER NOT NULL CHECK (source_campaign_revision >= 0),
        source_last_event_seq INTEGER NOT NULL CHECK (source_last_event_seq > 0),
        terminal_state TEXT NOT NULL,
        decision_id TEXT NOT NULL,
        replay_id TEXT NOT NULL,
        selected_prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        fallback_prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
        decision_payload_json TEXT NOT NULL,
        replay_payload_json TEXT NOT NULL,
        bundle_claims_json TEXT NOT NULL,
        projection_schema_version INTEGER NOT NULL CHECK (projection_schema_version > 0),
        redaction_policy_version INTEGER NOT NULL CHECK (redaction_policy_version > 0),
        prepared_projection_sha256 TEXT NOT NULL,
        prepared_projection_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (campaign_id, source_campaign_revision, source_last_event_seq)
    ) STRICT
    """,
    """
    CREATE TABLE bundle_publications (
        bundle_id TEXT PRIMARY KEY REFERENCES bundles(bundle_id),
        preparation_id TEXT NOT NULL UNIQUE REFERENCES terminal_preparations(preparation_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        published_path TEXT NOT NULL,
        manifest_sha256 TEXT NOT NULL,
        inventory_sha256 TEXT NOT NULL,
        submission_sha256 TEXT NOT NULL,
        file_count INTEGER NOT NULL CHECK (file_count > 0),
        total_size_bytes INTEGER NOT NULL CHECK (total_size_bytes > 0),
        final_event_seq INTEGER NOT NULL CHECK (final_event_seq > 0),
        committed_at TEXT NOT NULL,
        UNIQUE (campaign_id)
    ) STRICT
    """,
)

_MIGRATION_4: Final[tuple[str, ...]] = (
    """
    CREATE TABLE trials_v4 (
        trial_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
        campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
        experiment_id TEXT NOT NULL,
        state TEXT NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (campaign_id, experiment_id)
            REFERENCES campaign_experiments(campaign_id, experiment_id)
    ) STRICT
    """,
)

_APPEND_ONLY_TABLES: Final = (
    "contracts",
    "families",
    "family_ledger",
    "family_evidence",
    "experiments",
    "artifacts",
    "predictions",
    "inner_evaluations",
    "protected_evaluations",
    "promotion_decisions",
    "rank_graphs",
    "selection_decisions",
    "resource_receipts",
    "failures",
)

_TERMINAL_TABLES: Final = (
    "campaigns",
    "trials",
    "attempts",
    "replays",
    "bundles",
    "provider_operations",
)


def configure_connection(connection: sqlite3.Connection) -> None:
    """Apply mandatory safety settings to every writer connection."""

    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA synchronous = FULL")
    journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
    if journal_mode is None or str(journal_mode[0]).lower() != "wal":
        raise SchemaError("authority database must use WAL journal mode")


def migrate(connection: sqlite3.Connection, *, created_at: str) -> None:
    """Migrate an authority database forward without accepting unknown versions."""

    configure_connection(connection)
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    if application_id not in (0, APPLICATION_ID):
        raise SchemaError("path contains a different SQLite application")
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version > 0 and application_id != APPLICATION_ID:
        raise SchemaError("existing authority database has no valid application identity")
    if version > SCHEMA_VERSION:
        raise SchemaError(
            f"authority schema {version} is newer than supported version {SCHEMA_VERSION}"
        )
    if version == 0:
        _apply_migration_one(connection, created_at=created_at)
        version = 1
    if version == 1:
        _apply_migration_two(connection)
        version = 2
    if version == 2:
        _apply_migration_three(connection)
        version = 3
    if version == 3:
        _apply_migration_four(connection)
        version = 4
    if version != SCHEMA_VERSION:
        raise SchemaError(f"authority schema migration stopped at unexpected version {version}")
    metadata = connection.execute(
        "SELECT schema_version FROM authority_metadata WHERE singleton = 1"
    ).fetchone()
    if metadata is None or int(metadata[0]) != SCHEMA_VERSION:
        raise SchemaError("authority metadata does not match the migrated schema")


def _apply_migration_one(connection: sqlite3.Connection, *, created_at: str) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        concurrent_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if concurrent_version == SCHEMA_VERSION:
            connection.commit()
            return
        if concurrent_version != 0:
            raise SchemaError(
                f"authority schema changed concurrently to unsupported version {concurrent_version}"
            )
        _execute_all(connection, _MIGRATION_1)
        for table in _APPEND_ONLY_TABLES:
            connection.execute(
                f"""
                CREATE TRIGGER {table}_no_update
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} rows are immutable');
                END
                """
            )
            connection.execute(
                f"""
                CREATE TRIGGER {table}_no_delete
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} rows are immutable');
                END
                """
            )
        for table in _TERMINAL_TABLES:
            connection.execute(
                f"""
                CREATE TRIGGER {table}_terminal_no_update
                BEFORE UPDATE ON {table}
                WHEN OLD.terminal = 1
                BEGIN
                    SELECT RAISE(ABORT, '{table} terminal rows are immutable');
                END
                """
            )
            connection.execute(
                f"""
                CREATE TRIGGER {table}_terminal_no_delete
                BEFORE DELETE ON {table}
                WHEN OLD.terminal = 1
                BEGIN
                    SELECT RAISE(ABORT, '{table} terminal rows are immutable');
                END
                """
            )
        connection.execute(
            """
            INSERT INTO authority_metadata(singleton, created_at, schema_version)
            VALUES (1, ?, 1)
            """,
            (created_at,),
        )
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _apply_migration_two(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        concurrent_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if concurrent_version >= 2:
            connection.commit()
            return
        if concurrent_version != 1:
            raise SchemaError(
                f"authority schema changed concurrently to unsupported version {concurrent_version}"
            )
        _execute_all(connection, _MIGRATION_2)
        connection.execute(
            """
            INSERT INTO experiment_ledger(
                contract_id, experiment_id, family_id, parent_experiment_id,
                payload_json, created_at
            )
            SELECT contract_id, experiment_id, family_id, parent_experiment_id,
                   payload_json, created_at
            FROM experiments
            ORDER BY CASE WHEN parent_experiment_id IS NULL THEN 0 ELSE 1 END, experiment_id
            """
        )
        connection.execute(
            """
            INSERT INTO campaign_experiments(
                campaign_id, contract_id, experiment_id, family_id,
                parent_experiment_id, created_at
            )
            SELECT campaign_id, contract_id, experiment_id, family_id,
                   parent_experiment_id, created_at
            FROM experiments
            ORDER BY CASE WHEN parent_experiment_id IS NULL THEN 0 ELSE 1 END, experiment_id
            """
        )
        for table in ("experiment_ledger", "campaign_experiments"):
            connection.execute(
                f"""
                CREATE TRIGGER {table}_no_update
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} rows are immutable');
                END
                """
            )
            connection.execute(
                f"""
                CREATE TRIGGER {table}_no_delete
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} rows are immutable');
                END
                """
            )
        connection.execute("UPDATE authority_metadata SET schema_version = 2 WHERE singleton = 1")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _apply_migration_three(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        concurrent_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if concurrent_version >= 3:
            connection.commit()
            return
        if concurrent_version != 2:
            raise SchemaError(
                f"authority schema changed concurrently to unsupported version {concurrent_version}"
            )
        _execute_all(connection, _MIGRATION_3)
        for table in ("terminal_preparations", "bundle_publications"):
            connection.execute(
                f"""
                CREATE TRIGGER {table}_no_update
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} rows are immutable');
                END
                """
            )
            connection.execute(
                f"""
                CREATE TRIGGER {table}_no_delete
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} rows are immutable');
                END
                """
            )
        connection.execute("UPDATE authority_metadata SET schema_version = 3 WHERE singleton = 1")
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _apply_migration_four(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("BEGIN IMMEDIATE")
    try:
        concurrent_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if concurrent_version == SCHEMA_VERSION:
            connection.commit()
            return
        if concurrent_version != 3:
            raise SchemaError(
                f"authority schema changed concurrently to unsupported version {concurrent_version}"
            )
        _execute_all(connection, _MIGRATION_4)
        connection.execute(
            """
            INSERT INTO trials_v4(
                trial_id, contract_id, campaign_id, experiment_id, state, revision,
                terminal, payload_json, created_at, updated_at
            )
            SELECT trial_id, contract_id, campaign_id, experiment_id, state, revision,
                   terminal, payload_json, created_at, updated_at
            FROM trials
            """
        )
        connection.execute("DROP TABLE trials")
        connection.execute("ALTER TABLE trials_v4 RENAME TO trials")
        connection.execute("DROP TABLE experiments")
        connection.execute(
            """
            CREATE TRIGGER trials_terminal_no_update
            BEFORE UPDATE ON trials
            WHEN OLD.terminal = 1
            BEGIN
                SELECT RAISE(ABORT, 'trials terminal rows are immutable');
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER trials_terminal_no_delete
            BEFORE DELETE ON trials
            WHEN OLD.terminal = 1
            BEGIN
                SELECT RAISE(ABORT, 'trials terminal rows are immutable');
            END
            """
        )
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise SchemaError("authority v4 migration produced invalid foreign-key lineage")
        connection.execute("UPDATE authority_metadata SET schema_version = 4 WHERE singleton = 1")
        connection.execute("PRAGMA user_version = 4")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _execute_all(connection: sqlite3.Connection, statements: Iterable[str]) -> None:
    for statement in statements:
        connection.execute(statement)


__all__ = [
    "APPLICATION_ID",
    "DEFAULT_BUSY_TIMEOUT_MS",
    "SCHEMA_VERSION",
    "SchemaError",
    "configure_connection",
    "migrate",
]
