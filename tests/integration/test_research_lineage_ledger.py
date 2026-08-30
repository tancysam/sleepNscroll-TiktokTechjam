"""Cross-run durable evidence for generated-candidate proposal outcomes.

Unlike :class:`~kuairand_agent.campaign.store.OuterQueryLedger`, this ledger enforces no hard
limit.  It exists so a fresh campaign against the exact same benchmark, starter kit, and trusted
controller source identity does not repeat an already-characterized failure from scratch, while a
change to any of those three digests (in particular a corrective code fix) starts a clean slate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kuairand_agent.campaign.store import (
    CampaignExistsError,
    CampaignNotFoundError,
    ResearchLineageLedger,
    StoreInvariantError,
)

_BENCHMARK = "b" * 64
_STARTER = "c" * 64
_SOURCE = "d" * 64
_OTHER_SOURCE = "e" * 64
_ROOT_FINGERPRINT = "1" * 64
_TERMINAL_FINGERPRINT = "2" * 64
_SIGNATURE = "3" * 64


def _record_rejection(
    ledger: ResearchLineageLedger,
    *,
    campaign_id: str,
    candidate_id: str,
    source_digest: str = _SOURCE,
    proposal_family: str = "pairwise",
    root_failure_fingerprint: str = _ROOT_FINGERPRINT,
) -> None:
    ledger.record_rejection(
        campaign_id=campaign_id,
        benchmark_digest=_BENCHMARK,
        starter_digest=_STARTER,
        source_digest=source_digest,
        candidate_id=candidate_id,
        proposal_family=proposal_family,
        proposal_signature=None,
        repairs_attempted=1,
        root_failure_fingerprint=root_failure_fingerprint,
        root_failure_category="static_policy",
        root_failure_code="reserved_filename",
        root_failure_subject="baseline.py",
        terminal_failure_fingerprint=_TERMINAL_FINGERPRINT,
        terminal_failure_category="static_policy",
        terminal_failure_code="reserved_filename",
        terminal_failure_subject="baseline.py",
        diagnostic="reserved candidate filename is forbidden: 'baseline.py'",
    )


def test_ledger_persists_and_scopes_evidence_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "lineage.sqlite3"
    ledger = ResearchLineageLedger.create(path)
    try:
        _record_rejection(ledger, campaign_id="run-a", candidate_id="cand-1")
        _record_rejection(ledger, campaign_id="run-a", candidate_id="cand-2")
        ledger.record_admission(
            campaign_id="run-a",
            benchmark_digest=_BENCHMARK,
            starter_digest=_STARTER,
            source_digest=_SOURCE,
            candidate_id="cand-3",
            proposal_family="listwise",
            proposal_signature=_SIGNATURE,
        )
    finally:
        ledger.close()

    # A second campaign, opening the ledger fresh (simulating a brand-new run directory), sees
    # the full history of the first.
    reopened = ResearchLineageLedger.open(path)
    try:
        summary = reopened.summary(
            benchmark_digest=_BENCHMARK, starter_digest=_STARTER, source_digest=_SOURCE
        )
        assert dict(summary.root_failure_totals) == {_ROOT_FINGERPRINT: 2}
        assert dict(summary.proposal_family_rejection_totals) == {"pairwise": 2}
        assert dict(summary.proposal_family_admission_totals) == {"listwise": 1}
        assert len(summary.recent_events) == 3
        assert [event.outcome for event in summary.recent_events] == [
            "rejected",
            "rejected",
            "admitted",
        ]

        # A different trusted-source digest (e.g. after a corrective code fix) starts clean: a
        # since-fixed bug can never keep blocking a corrected agent.
        clean = reopened.summary(
            benchmark_digest=_BENCHMARK, starter_digest=_STARTER, source_digest=_OTHER_SOURCE
        )
        assert dict(clean.root_failure_totals) == {}
        assert dict(clean.proposal_family_rejection_totals) == {}
        assert dict(clean.proposal_family_admission_totals) == {}
        assert clean.recent_events == ()
    finally:
        reopened.close()


def test_summary_accumulates_across_independent_campaign_ids(tmp_path: Path) -> None:
    path = tmp_path / "lineage.sqlite3"
    ledger = ResearchLineageLedger.create(path)
    try:
        _record_rejection(ledger, campaign_id="run-a", candidate_id="cand-1")
        _record_rejection(ledger, campaign_id="run-b", candidate_id="cand-1")
        _record_rejection(ledger, campaign_id="run-c", candidate_id="cand-1")
        summary = ledger.summary(
            benchmark_digest=_BENCHMARK, starter_digest=_STARTER, source_digest=_SOURCE
        )
        # Three separate run directories, same exact code identity: the third occurrence of an
        # identical root fingerprint is exactly the signal this project's own circuit breaker
        # treats as `repeated_pre_admission_failure` within one run.
        assert summary.root_failure_totals[_ROOT_FINGERPRINT] == 3
        campaign_ids = {event.campaign_id for event in summary.recent_events}
        assert campaign_ids == {"run-a", "run-b", "run-c"}
    finally:
        ledger.close()


def test_summary_limit_bounds_recent_events_but_not_totals(tmp_path: Path) -> None:
    path = tmp_path / "lineage.sqlite3"
    ledger = ResearchLineageLedger.create(path)
    try:
        for index in range(5):
            _record_rejection(ledger, campaign_id="run-a", candidate_id=f"cand-{index}")
        summary = ledger.summary(
            benchmark_digest=_BENCHMARK,
            starter_digest=_STARTER,
            source_digest=_SOURCE,
            limit=2,
        )
        assert summary.root_failure_totals[_ROOT_FINGERPRINT] == 5
        assert len(summary.recent_events) == 2
        # The most recent events, not an arbitrary two.
        assert [event.candidate_id for event in summary.recent_events] == ["cand-3", "cand-4"]
    finally:
        ledger.close()


def test_read_only_ledger_cannot_be_mutated(tmp_path: Path) -> None:
    path = tmp_path / "lineage.sqlite3"
    ResearchLineageLedger.create(path).close()
    reopened = ResearchLineageLedger.open(path, read_only=True)
    try:
        with pytest.raises(StoreInvariantError, match="read-only"):
            _record_rejection(reopened, campaign_id="run-a", candidate_id="cand-1")
    finally:
        reopened.close()


def test_lineage_events_table_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "lineage.sqlite3"
    ledger = ResearchLineageLedger.create(path)
    try:
        _record_rejection(ledger, campaign_id="run-a", candidate_id="cand-1")
    finally:
        ledger.close()

    raw = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute("UPDATE lineage_events SET candidate_id = 'tampered' WHERE event_id = 1")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute("DELETE FROM lineage_events WHERE event_id = 1")
    finally:
        raw.close()


def test_record_rejection_rejects_invalid_repairs_attempted(tmp_path: Path) -> None:
    ledger = ResearchLineageLedger.create(tmp_path / "lineage.sqlite3")
    try:
        with pytest.raises(StoreInvariantError, match="repairs_attempted"):
            ledger.record_rejection(
                campaign_id="run-a",
                benchmark_digest=_BENCHMARK,
                starter_digest=_STARTER,
                source_digest=_SOURCE,
                candidate_id="cand-1",
                proposal_family="pairwise",
                proposal_signature=None,
                repairs_attempted=-1,
                root_failure_fingerprint=_ROOT_FINGERPRINT,
                root_failure_category="static_policy",
                root_failure_code="reserved_filename",
                root_failure_subject="baseline.py",
                terminal_failure_fingerprint=_TERMINAL_FINGERPRINT,
                terminal_failure_category="static_policy",
                terminal_failure_code="reserved_filename",
                terminal_failure_subject="baseline.py",
                diagnostic="reserved candidate filename is forbidden: 'baseline.py'",
            )
    finally:
        ledger.close()


def test_create_refuses_to_overwrite_existing_ledger(tmp_path: Path) -> None:
    path = tmp_path / "lineage.sqlite3"
    ResearchLineageLedger.create(path).close()
    with pytest.raises(CampaignExistsError):
        ResearchLineageLedger.create(path)


def test_open_missing_ledger_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(CampaignNotFoundError):
        ResearchLineageLedger.open(tmp_path / "missing.sqlite3")
