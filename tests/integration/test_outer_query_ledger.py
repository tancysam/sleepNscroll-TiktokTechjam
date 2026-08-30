from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kuairand_agent.campaign.store import (
    CampaignExistsError,
    CampaignStore,
    OuterQueryLedger,
    OuterQueryLimitError,
    StoreInvariantError,
)

_CONFIG = "a" * 64
_BENCHMARK = "b" * 64
_STARTER = "c" * 64
_DATASET = "d" * 64
_ENVIRONMENT = "e" * 64
_SCORER = "f" * 64


def _create_campaign(path: Path, campaign_id: str) -> CampaignStore:
    return CampaignStore.create(
        path,
        campaign_id=campaign_id,
        config_digest=_CONFIG,
        benchmark_digest=_BENCHMARK,
        starter_digest=_STARTER,
        dataset_digest=_DATASET,
        environment_digest=_ENVIRONMENT,
        hard_deadline_utc="2030-01-01T00:00:00Z",
        initial_convergence={
            "schema_version": 1,
            "best_primary": 0.6016,
            "non_material_streak": 0,
            "completed_iterations": 0,
            "required_completion_pending": False,
        },
    )


def test_project_ledger_is_global_and_query_lifecycle_is_replayable(tmp_path: Path) -> None:
    ledger_path = tmp_path / "outer.sqlite"
    first_path = tmp_path / "campaign-1.sqlite"
    second_path = tmp_path / "campaign-2.sqlite"
    with (
        OuterQueryLedger.create(ledger_path, max_queries=2) as ledger,
        _create_campaign(first_path, "campaign-1") as first,
        _create_campaign(second_path, "campaign-2") as second,
    ):
        reservation = first.reserve_public_query(
            ledger,
            query_id="outer-001",
            candidate_fingerprint="1" * 64,
            scorer_digest=_SCORER,
            expected_revision=0,
            metadata={"promotion_ordinal": 1},
        )
        assert reservation.state == "RESERVED"
        assert reservation.event_seq == 1
        assert first.snapshot().outer_queries_used == 1

        retry = first.reserve_public_query(
            ledger,
            query_id="outer-001",
            candidate_fingerprint="1" * 64,
            scorer_digest=_SCORER,
            expected_revision=1,
            metadata={"promotion_ordinal": 1},
        )
        assert retry == reservation
        assert first.snapshot().revision == 1

        with pytest.raises(StoreInvariantError, match="scorer differs"):
            first.complete_public_query(
                ledger,
                query_id="outer-001",
                result_digest="2" * 64,
                gauc=0.67,
                ndcg_at_5=0.54,
                primary=0.605,
                prediction_digest="3" * 64,
                scorer_digest="0" * 64,
                expected_revision=1,
            )
        assert (
            ledger.snapshot(
                benchmark_digest=_BENCHMARK,
                dataset_digest=_DATASET,
                scorer_digest=_SCORER,
            ).revision
            == 1
        )

        completed = first.complete_public_query(
            ledger,
            query_id="outer-001",
            result_digest="2" * 64,
            gauc=0.67,
            ndcg_at_5=0.54,
            primary=0.605,
            prediction_digest="3" * 64,
            scorer_digest=_SCORER,
            expected_revision=1,
            metadata={"trusted_scorer": True},
        )
        assert completed.state == "COMPLETED"
        assert completed.event_seq == 2
        assert completed.metrics == {"GAUC": 0.67, "nDCG@5": 0.54, "primary": 0.605}

        completion_retry = first.complete_public_query(
            ledger,
            query_id="outer-001",
            result_digest="2" * 64,
            gauc=0.67,
            ndcg_at_5=0.54,
            primary=0.605,
            prediction_digest="3" * 64,
            scorer_digest=_SCORER,
            expected_revision=2,
            metadata={"trusted_scorer": True},
        )
        assert completion_retry == completed
        assert first.snapshot().revision == 2

        second_reservation = second.reserve_public_query(
            ledger,
            query_id="outer-002",
            candidate_fingerprint="4" * 64,
            scorer_digest=_SCORER,
            expected_revision=0,
        )
        assert second_reservation.state == "RESERVED"

        with pytest.raises(StoreInvariantError, match="query_id already names"):
            ledger.reserve(
                query_id="outer-002",
                campaign_id="campaign-2",
                benchmark_digest=_BENCHMARK,
                dataset_digest=_DATASET,
                scorer_digest=_SCORER,
                candidate_fingerprint="5" * 64,
            )

        # The ration is per campaign, so campaign-2 still holds its own second slot even though
        # campaign-1 has already spent one.
        third_reservation = second.reserve_public_query(
            ledger,
            query_id="outer-003",
            candidate_fingerprint="5" * 64,
            scorer_digest=_SCORER,
            expected_revision=1,
        )
        assert third_reservation.state == "RESERVED"
        assert second.snapshot().outer_queries_used == 2

        # Its third is refused: exhaustion is enforced, just scoped to the campaign.
        with pytest.raises(OuterQueryLimitError, match="campaign public-validation limit"):
            second.reserve_public_query(
                ledger,
                query_id="outer-004",
                candidate_fingerprint="6" * 64,
                scorer_digest=_SCORER,
                expected_revision=2,
            )
        assert second.snapshot().outer_queries_used == 2
        assert second.snapshot().revision == 2

        # The log itself stays project-wide: every query ever made is retained for disclosure,
        # which is exactly what plan.md 12.2 asks the append-only project log to preserve.
        ledger_snapshot = ledger.snapshot(
            benchmark_digest="7" * 64,
            dataset_digest="8" * 64,
            scorer_digest="9" * 64,
        )
        assert ledger_snapshot.max_queries == 2
        assert ledger_snapshot.queries_used == 3
        # Four appended events: outer-001 reserve and complete, then outer-002 and outer-003.
        assert ledger_snapshot.revision == 4

        with sqlite3.connect(first_path) as raw:
            metric = raw.execute(
                """SELECT split_role, gauc, ndcg_at_5, primary_value, scorer_digest
                FROM metrics WHERE metric_id = 'outer:outer-001'"""
            ).fetchone()
            assert metric == ("outer_valid", 0.67, 0.54, 0.605, _SCORER)

    with pytest.raises(CampaignExistsError, match="already exists"):
        OuterQueryLedger.create(ledger_path)
    with OuterQueryLedger.open(ledger_path, read_only=True) as reopened:
        assert (
            reopened.snapshot(
                benchmark_digest=_BENCHMARK,
                dataset_digest=_DATASET,
                scorer_digest=_SCORER,
            ).queries_used
            == 3
        )
        with pytest.raises(StoreInvariantError, match="read-only"):
            reopened.reserve(
                query_id="outer-read-only",
                campaign_id="campaign-read-only",
                benchmark_digest=_BENCHMARK,
                dataset_digest=_DATASET,
                scorer_digest=_SCORER,
                candidate_fingerprint="a" * 64,
            )

    raw = sqlite3.connect(ledger_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(
                "UPDATE outer_queries SET state = 'START_UNCERTAIN' WHERE query_id = ?",
                ("outer-002",),
            )
    finally:
        raw.close()


def test_project_reservation_crash_window_is_conservatively_charged(tmp_path: Path) -> None:
    ledger_path = tmp_path / "outer.sqlite"
    campaign_path = tmp_path / "campaign.sqlite"
    with (
        OuterQueryLedger.create(ledger_path, max_queries=1) as ledger,
        _create_campaign(campaign_path, "campaign-local") as campaign,
    ):
        # The orphan belongs to this same campaign: the ration is per campaign, so a crash between
        # the project reservation and the local commit must still be charged to the campaign that
        # made it, rather than being silently reissued.
        orphaned = ledger.reserve(
            query_id="orphaned-before-local-commit",
            campaign_id="campaign-local",
            benchmark_digest=_BENCHMARK,
            dataset_digest=_DATASET,
            scorer_digest=_SCORER,
            candidate_fingerprint="1" * 64,
            metadata={"fault_injection": "crash-after-project-reserve"},
        )
        assert orphaned.state == "RESERVED"

        with pytest.raises(OuterQueryLimitError, match="campaign public-validation limit"):
            campaign.reserve_public_query(
                ledger,
                query_id="must-not-score",
                candidate_fingerprint="2" * 64,
                scorer_digest=_SCORER,
                expected_revision=0,
            )

        local = campaign.snapshot()
        assert local.outer_queries_used == 0
        assert local.revision == 0
        project = ledger.snapshot(
            benchmark_digest=_BENCHMARK,
            dataset_digest=_DATASET,
            scorer_digest=_SCORER,
        )
        assert project.queries_used == 1
        assert project.queries_remaining == 0
        assert project.revision == 1
