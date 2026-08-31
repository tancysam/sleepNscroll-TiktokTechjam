"""Deletable, rebuildable views derived from ``authority.sqlite3``."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast


class ProjectionError(RuntimeError):
    """Raised when an authority projection cannot be inspected or written."""


_ENTITY_TABLES: Final[tuple[tuple[str, str], ...]] = (
    ("families", "family_id"),
    ("trials", "trial_id"),
    ("attempts", "attempt_id"),
    ("artifacts", "artifact_id"),
    ("predictions", "prediction_id"),
    ("inner_evaluations", "evaluation_id"),
    ("protected_query_reservations", "query_ordinal"),
    ("protected_evaluations", "evaluation_id"),
    ("promotion_decisions", "decision_id"),
    ("rank_graphs", "rank_graph_id"),
    ("selection_decisions", "decision_id"),
    ("replays", "replay_id"),
    ("bundles", "bundle_id"),
    ("resource_receipts", "receipt_id"),
    ("provider_operations", "operation_id"),
    ("failures", "failure_id"),
    ("terminal_preparations", "preparation_id"),
    ("bundle_publications", "bundle_id"),
)


def inspect_campaign(connection: sqlite3.Connection, *, campaign_id: str) -> Mapping[str, object]:
    """Assemble the authoritative inspection shape without performing any writes."""

    connection.row_factory = sqlite3.Row
    campaign = connection.execute(
        "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
    ).fetchone()
    if campaign is None:
        raise ProjectionError("campaign does not exist")
    contract = connection.execute(
        "SELECT * FROM contracts WHERE contract_id = ?", (campaign["contract_id"],)
    ).fetchone()
    if contract is None:  # pragma: no cover - enforced by foreign keys
        raise ProjectionError("campaign contract is missing")
    entities: dict[str, object] = {}
    experiment_associations = connection.execute(
        """
        SELECT ce.experiment_id, ce.contract_id, ce.campaign_id, ce.family_id,
               ce.parent_experiment_id, el.payload_json, ce.created_at
        FROM campaign_experiments AS ce
        JOIN experiment_ledger AS el
          ON el.contract_id = ce.contract_id AND el.experiment_id = ce.experiment_id
        WHERE ce.campaign_id = ? ORDER BY ce.experiment_id
        """,
        (campaign_id,),
    ).fetchall()
    entities["experiments"] = [_row_to_json(row) for row in experiment_associations]
    for table, order_key in _ENTITY_TABLES:
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE campaign_id = ? ORDER BY {order_key}",
            (campaign_id,),
        ).fetchall()
        entities[table] = [_row_to_json(row) for row in rows]
    events = connection.execute(
        """
        SELECT * FROM campaign_events WHERE campaign_id = ? ORDER BY event_seq
        """,
        (campaign_id,),
    ).fetchall()
    contract_reservations = connection.execute(
        """
        SELECT reservation_id, campaign_id, prediction_id, query_ordinal, state, created_at,
               completed_at
        FROM protected_query_reservations
        WHERE contract_id = ? ORDER BY query_ordinal
        """,
        (campaign["contract_id"],),
    ).fetchall()
    family_ledger = connection.execute(
        """
        SELECT * FROM family_ledger WHERE contract_id = ? ORDER BY family_id
        """,
        (campaign["contract_id"],),
    ).fetchall()
    family_evidence = connection.execute(
        """
        SELECT * FROM family_evidence
        WHERE contract_id = ? ORDER BY family_id, fingerprint
        """,
        (campaign["contract_id"],),
    ).fetchall()
    experiment_ledger = connection.execute(
        """
        SELECT * FROM experiment_ledger
        WHERE contract_id = ? ORDER BY experiment_id
        """,
        (campaign["contract_id"],),
    ).fetchall()
    return {
        "schema_version": 1,
        "contract": _row_to_json(contract),
        "contract_protected_query_reservations": [
            _row_to_json(row) for row in contract_reservations
        ],
        "contract_family_ledger": [_row_to_json(row) for row in family_ledger],
        "contract_family_evidence": [_row_to_json(row) for row in family_evidence],
        "contract_experiment_ledger": [_row_to_json(row) for row in experiment_ledger],
        "campaign": _row_to_json(campaign),
        "events": [_row_to_json(row) for row in events],
        "entities": entities,
    }


def write_campaign_projection(snapshot: Mapping[str, object], *, destination: Path) -> Path:
    """Write canonical JSON through fsync and atomic replacement."""

    if not isinstance(snapshot, Mapping):
        raise ProjectionError("snapshot must be a mapping")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                snapshot,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProjectionError("snapshot must be finite JSON") from exc
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _row_to_json(row: sqlite3.Row) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in row.keys():  # noqa: SIM118 - sqlite3.Row iteration yields values, not keys
        value = row[key]
        if key.endswith("_json") and value is not None:
            try:
                result[key.removesuffix("_json")] = json.loads(cast(str, value))
            except json.JSONDecodeError as exc:  # pragma: no cover - canonical repository writes
                raise ProjectionError(f"authority contains malformed JSON in {key}") from exc
        elif key in {"terminal", "completed", "protected_eligible", "research_frozen"}:
            result[key] = bool(value)
        else:
            result[key] = cast(object, value)
    return result


__all__ = ["ProjectionError", "inspect_campaign", "write_campaign_projection"]
