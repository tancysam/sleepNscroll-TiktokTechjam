from __future__ import annotations

import hashlib
from pathlib import Path

from kuairand_agent.state import (
    DurableRecord,
    RecordKind,
    StateRepository,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _running_attempt(repository: StateRepository) -> tuple[str, str, str, str]:
    contract = _digest("contract")
    campaign = _digest("campaign")
    family = _digest("family")
    experiment = _digest("experiment")
    trial = _digest("trial")
    attempt = _digest("attempt")
    repository.create_campaign(
        campaign_id=campaign,
        contract_id=contract,
        contract_manifest={"frozen": True},
        config={"profile": "cpu"},
        idempotency_key="campaign",
        protected_query_limit=1,
    )
    repository.register(
        DurableRecord(
            RecordKind.FAMILY,
            family,
            campaign,
            contract,
            attributes={"protected_eligible": False},
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.EXPERIMENT,
            experiment,
            campaign,
            contract,
            references={"family_id": family},
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.TRIAL,
            trial,
            campaign,
            contract,
            references={"experiment_id": experiment},
            state="RUNNING",
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.ATTEMPT,
            attempt,
            campaign,
            contract,
            references={"trial_id": trial},
            attributes={"attempt_ordinal": 1, "process_identity": {"pid": 999999}},
            state="RUNNING",
        )
    )
    return contract, campaign, trial, attempt


def test_restart_closes_missing_running_process_as_durable_interruption(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    repository = StateRepository.open(state_root)
    _, campaign, _, attempt = _running_attempt(repository)

    reopened = StateRepository.open(state_root)
    recovery = reopened.reconcile_missing_attempts(process_exists=lambda _identity: False)
    repeated = reopened.reconcile_missing_attempts(process_exists=lambda _identity: False)
    snapshot = reopened.inspect(campaign_id=campaign)

    assert recovery.interrupted_attempt_ids == (attempt,)
    assert repeated.interrupted_attempt_ids == ()
    attempts = snapshot["entities"]
    assert isinstance(attempts, dict)
    attempt_rows = attempts["attempts"]
    failure_rows = attempts["failures"]
    assert isinstance(attempt_rows, list) and attempt_rows[0]["state"] == "INTERRUPTED"
    assert isinstance(failure_rows, list) and failure_rows[0]["failure_kind"] == "INTERRUPTED"
