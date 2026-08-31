from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from kuairand_agent.campaign.scientific import (
    OuterPromotionCompletion,
    OuterPromotionRequest,
)
from kuairand_agent.campaign.scientific_store import (
    DurableScientificLedgerAdapter,
    ScientificStoreError,
    TrustedOuterSeedEvidence,
)
from kuairand_agent.campaign.selector import OrganizerMetrics, SeedMetrics
from kuairand_agent.campaign.store import (
    CampaignStore,
    OuterQueryLedger,
    StoreInvariantError,
)

_CAMPAIGN = "1" * 64
_BENCHMARK = "2" * 64
_STARTER = "3" * 64
_DATASET = "4" * 64
_ENVIRONMENT = "5" * 64
_SCORER = "6" * 64
_TRAINING_POLICY = "7" * 64
_COMPLETION_EVIDENCE = "8" * 64
_SOURCE = "d" * 64
_PARENT_SOURCE = "e" * 64
_EXECUTABLE_DIFF = "f" * 64
_MATERIAL_CHANGE = "0" * 64
_CONTROLLER_ATTESTATION = "a" * 64


def _create_store(path: Path, campaign_id: str, *, outer_limit: int = 6) -> CampaignStore:
    return CampaignStore.create(
        path,
        campaign_id=campaign_id,
        config_digest=_CAMPAIGN,
        benchmark_digest=_BENCHMARK,
        starter_digest=_STARTER,
        dataset_digest=_DATASET,
        environment_digest=_ENVIRONMENT,
        hard_deadline_utc="2030-01-01T00:00:00Z",
        initial_convergence={
            "schema_version": 2,
            "best_primary": 0.6016,
            "non_material_streak": 0,
            "unmeasured_streak": 0,
            "completed_iterations": 0,
            "required_completion_pending": False,
        },
        outer_query_limit=outer_limit,
    )


def _request(
    index: int = 0, *, overrides: Mapping[str, str] | None = None
) -> OuterPromotionRequest:
    values = {
        "campaign_digest": _CAMPAIGN,
        "candidate_id": f"candidate-{index}",
        "candidate_fingerprint": f"{index + 9:x}" * 64,
        "source_digest": _SOURCE,
        "parent_source_digest": _PARENT_SOURCE,
        "executable_diff_digest": _EXECUTABLE_DIFF,
        "material_change_digest": _MATERIAL_CHANGE,
        "controller_attestation_digest": _CONTROLLER_ATTESTATION,
        "benchmark_digest": _BENCHMARK,
        "dataset_digest": _DATASET,
        "scorer_digest": _SCORER,
        "training_policy_digest": _TRAINING_POLICY,
    }
    if overrides is not None:
        values.update(overrides)
    return OuterPromotionRequest(**values)


def _metrics() -> tuple[SeedMetrics, ...]:
    return (
        SeedMetrics(0, OrganizerMetrics(0.67, 0.53)),
        SeedMetrics(1, OrganizerMetrics(0.68, 0.54)),
        SeedMetrics(2, OrganizerMetrics(0.69, 0.55)),
    )


def _registry(request: OuterPromotionRequest) -> dict[tuple[str, int], TrustedOuterSeedEvidence]:
    return {
        (request.digest, item.seed): TrustedOuterSeedEvidence(
            request_digest=request.digest,
            seed=item.seed,
            metrics=item.metrics,
            prediction_digest=f"{item.seed + 10:x}" * 64,
            score_evidence_digest=f"{item.seed + 13:x}" * 64,
        )
        for item in _metrics()
    }


def _completion(
    request: OuterPromotionRequest,
    reservation_id: str,
    reservation_revision: int,
    *,
    successful: bool = True,
    seeds: tuple[SeedMetrics, ...] | None = None,
) -> OuterPromotionCompletion:
    return OuterPromotionCompletion(
        reservation_id=reservation_id,
        request_digest=request.digest,
        reservation_revision=reservation_revision,
        candidate_fingerprint=request.candidate_fingerprint,
        successful=successful,
        seed_metrics=_metrics() if seeds is None else seeds,
        evidence_digest=_COMPLETION_EVIDENCE,
    )


def test_successful_bundle_is_exact_restart_safe_and_never_uses_bundle_as_prediction(
    tmp_path: Path,
) -> None:
    campaign_path = tmp_path / "campaign.sqlite"
    ledger_path = tmp_path / "outer.sqlite"
    request = _request()
    registry = _registry(request)
    with (
        _create_store(campaign_path, "campaign-one") as store,
        OuterQueryLedger.create(ledger_path) as ledger,
    ):
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry=registry,
        )
        assert adapter.snapshot().revision == 0
        reservation = adapter.reserve(request)
        assert reservation.ledger_revision == 1
        assert reservation.consumes_slot is True
        reservation_request = ledger.projection().queries[0].reservation_metadata["request"]
        assert reservation_request == request.manifest()
        assert reservation_request["source_digest"] == _SOURCE
        assert reservation_request["parent_source_digest"] == _PARENT_SOURCE
        assert reservation_request["executable_diff_digest"] == _EXECUTABLE_DIFF
        assert reservation_request["material_change_digest"] == _MATERIAL_CHANGE
        assert reservation_request["controller_attestation_digest"] == _CONTROLLER_ATTESTATION
        completion = _completion(
            request,
            reservation.reservation_id,
            reservation.ledger_revision,
        )
        adapter.complete(reservation, completion)
        after = adapter.snapshot()
        assert after.revision == 2
        assert after.candidate_fingerprints == (request.candidate_fingerprint,)

        projection = ledger.projection()
        assert projection.revision == 2
        assert projection.queries[0].reservation_revision == 1
        assert projection.queries[0].state == "COMPLETED"
        assert projection.queries[0].latest_metadata["request_digest"] == request.digest
        assert projection.queries[0].latest_metadata["representative_seed"] == 0
        reserved_request = projection.queries[0].reservation_metadata["request"]
        completed_request = projection.queries[0].latest_metadata["request"]
        assert isinstance(reserved_request, Mapping)
        assert isinstance(completed_request, Mapping)
        for name, expected in (
            ("source_digest", _SOURCE),
            ("parent_source_digest", _PARENT_SOURCE),
            ("executable_diff_digest", _EXECUTABLE_DIFF),
            ("material_change_digest", _MATERIAL_CHANGE),
            ("controller_attestation_digest", _CONTROLLER_ATTESTATION),
        ):
            assert projection.queries[0].reservation_metadata[name] == expected
            assert reserved_request[name] == expected
            assert projection.queries[0].latest_metadata[name] == expected
            assert completed_request[name] == expected
        assert projection.queries[0].latest_metadata["request"] == request.manifest()

        with sqlite3.connect(campaign_path) as raw:
            rows = raw.execute(
                """SELECT seed, split_role, prediction_digest, scorer_digest
                FROM metrics ORDER BY seed"""
            ).fetchall()
            query_rows = raw.execute("SELECT COUNT(*) FROM validation_queries").fetchone()[0]
        assert rows == [
            (None, "outer_valid", "a" * 64, _SCORER),
            (1, "outer_valid_matched_seed", "b" * 64, _SCORER),
            (2, "outer_valid_matched_seed", "c" * 64, _SCORER),
        ]
        assert all(row[2] != _COMPLETION_EVIDENCE for row in rows)
        assert query_rows == 2

    with (
        CampaignStore.open(campaign_path) as reopened_store,
        OuterQueryLedger.open(ledger_path) as reopened_ledger,
    ):
        restarted = DurableScientificLedgerAdapter(
            reopened_store,
            reopened_ledger,
            scorer_digest=_SCORER,
            evidence_registry=registry,
        )
        retried_reservation = restarted.reserve(request)
        assert retried_reservation == reservation
        restarted.complete(retried_reservation, completion)
        assert reopened_store.snapshot().revision == 4
        assert reopened_ledger.projection().revision == 2
        with sqlite3.connect(campaign_path) as raw:
            assert raw.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 3
            assert raw.execute("SELECT COUNT(*) FROM validation_queries").fetchone()[0] == 2


def test_missing_or_conflicting_trusted_evidence_fails_before_public_completion(
    tmp_path: Path,
) -> None:
    campaign_path = tmp_path / "campaign.sqlite"
    with (
        _create_store(campaign_path, "campaign-one") as store,
        OuterQueryLedger.create(tmp_path / "outer.sqlite") as ledger,
    ):
        request = _request()
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry={},
        )
        reservation = adapter.reserve(request)
        completion = _completion(request, reservation.reservation_id, reservation.ledger_revision)
        with pytest.raises(ScientificStoreError, match="missing for matched seed 0"):
            adapter.complete(reservation, completion)
        assert ledger.projection().queries[0].state == "RESERVED"
        assert store.snapshot().outer_queries_used == 1
        with sqlite3.connect(campaign_path) as raw:
            assert raw.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0

        registry = _registry(request)
        registry[(request.digest, 1)] = TrustedOuterSeedEvidence(
            request_digest=request.digest,
            seed=1,
            metrics=OrganizerMetrics(0.1, 0.1),
            prediction_digest="b" * 64,
            score_evidence_digest="e" * 64,
        )
        mismatched = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry=registry,
        )
        with pytest.raises(ScientificStoreError, match="differs for matched seed 1"):
            mismatched.complete(reservation, completion)
        with sqlite3.connect(campaign_path) as raw:
            assert raw.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0


def test_failed_bundle_stays_reserved_records_exact_failure_and_is_idempotent(
    tmp_path: Path,
) -> None:
    campaign_path = tmp_path / "campaign.sqlite"
    with (
        _create_store(campaign_path, "campaign-one") as store,
        OuterQueryLedger.create(tmp_path / "outer.sqlite") as ledger,
    ):
        request = _request()
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry={},
        )
        reservation = adapter.reserve(request)
        failed = _completion(
            request,
            reservation.reservation_id,
            reservation.ledger_revision,
            successful=False,
            seeds=_metrics()[:1],
        )
        adapter.complete(reservation, failed)
        first_revision = store.snapshot().revision
        adapter.complete(reservation, failed)
        assert store.snapshot().revision == first_revision
        assert ledger.projection().revision == 1
        assert ledger.projection().queries[0].state == "RESERVED"
        with sqlite3.connect(campaign_path) as raw:
            failure = raw.execute("SELECT fingerprint, metadata_json FROM failures").fetchone()
            assert raw.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 1
        assert failure[0] == _COMPLETION_EVIDENCE
        assert request.digest in failure[1]
        assert '"successful":false' in failure[1]

        registry = _registry(request)
        success_adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry=registry,
        )
        success = _completion(request, reservation.reservation_id, reservation.ledger_revision)
        with pytest.raises(ScientificStoreError, match="cannot later become successful"):
            success_adapter.complete(reservation, success)


def test_request_identity_and_reservation_revision_conflicts_fail_closed(tmp_path: Path) -> None:
    with (
        _create_store(tmp_path / "campaign.sqlite", "campaign-one") as store,
        OuterQueryLedger.create(tmp_path / "outer.sqlite") as ledger,
    ):
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry={},
        )
        with pytest.raises(ScientificStoreError, match="identity differs"):
            adapter.reserve(_request(overrides={"scorer_digest": "0" * 64}))
        assert ledger.projection().revision == 0

        request = _request()
        reservation = adapter.reserve(request)
        with pytest.raises(ScientificStoreError, match="project reservation identity"):
            DurableScientificLedgerAdapter(
                store,
                ledger,
                scorer_digest="0" * 64,
                evidence_registry={},
            )
        corrupt = _completion(
            request,
            reservation.reservation_id,
            reservation.ledger_revision + 1,
        )
        with pytest.raises(ScientificStoreError, match="completion identity differs"):
            adapter.complete(reservation, corrupt)
        assert ledger.projection().revision == 1


@pytest.mark.parametrize(
    ("field", "altered_digest"),
    (
        ("source_digest", "1" * 64),
        ("parent_source_digest", "2" * 64),
        ("executable_diff_digest", "3" * 64),
        ("material_change_digest", "4" * 64),
        ("controller_attestation_digest", "5" * 64),
    ),
)
def test_source_and_materiality_identity_conflicts_cannot_reuse_reserved_candidate(
    tmp_path: Path,
    field: str,
    altered_digest: str,
) -> None:
    with (
        _create_store(tmp_path / "campaign.sqlite", "campaign-one") as store,
        OuterQueryLedger.create(tmp_path / "outer.sqlite") as ledger,
    ):
        request = _request()
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry={},
        )
        reservation = adapter.reserve(request)
        assert adapter.reserve(request) == reservation

        conflicting = _request(overrides={field: altered_digest})
        assert conflicting.digest != request.digest
        with pytest.raises(ScientificStoreError, match="reservation failed closed"):
            adapter.reserve(conflicting)
        assert ledger.projection().revision == 1
        assert store.snapshot().revision == 1
        assert len(ledger.projection().queries) == 1


def test_project_first_fault_retains_global_charge_without_local_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path = tmp_path / "campaign.sqlite"
    with (
        _create_store(campaign_path, "campaign-one") as store,
        OuterQueryLedger.create(tmp_path / "outer.sqlite") as ledger,
    ):
        request = _request()
        original = CampaignStore.reserve_public_query

        def crash_after_project_reserve(
            current: CampaignStore,
            project: OuterQueryLedger,
            **kwargs: object,
        ) -> object:
            project.reserve(
                query_id=str(kwargs["query_id"]),
                campaign_id=current.campaign_id,
                benchmark_digest=_BENCHMARK,
                dataset_digest=_DATASET,
                scorer_digest=str(kwargs["scorer_digest"]),
                candidate_fingerprint=str(kwargs["candidate_fingerprint"]),
                metadata=kwargs["metadata"],  # type: ignore[arg-type]
            )
            raise StoreInvariantError("fault after project reservation")

        monkeypatch.setattr(CampaignStore, "reserve_public_query", crash_after_project_reserve)
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry={},
        )
        with pytest.raises(ScientificStoreError, match="failed closed"):
            adapter.reserve(request)
        assert ledger.projection().revision == 1
        assert ledger.projection().queries[0].state == "RESERVED"
        assert store.snapshot().outer_queries_used == 0
        monkeypatch.setattr(CampaignStore, "reserve_public_query", original)
        repaired = adapter.reserve(request)
        assert repaired.ledger_revision == 1
        assert repaired.consumes_slot is True
        assert ledger.projection().revision == 1
        assert store.snapshot().outer_queries_used == 1


def test_completion_fault_reuses_persisted_seed_metrics_without_double_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_path = tmp_path / "campaign.sqlite"
    with (
        _create_store(campaign_path, "campaign-one") as store,
        OuterQueryLedger.create(tmp_path / "outer.sqlite") as ledger,
    ):
        request = _request()
        adapter = DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry=_registry(request),
        )
        reservation = adapter.reserve(request)
        completion = _completion(request, reservation.reservation_id, reservation.ledger_revision)
        original = CampaignStore.complete_public_query

        def fail_logical_completion(
            _current: CampaignStore,
            _project: OuterQueryLedger,
            **_kwargs: object,
        ) -> object:
            raise StoreInvariantError("fault before logical completion")

        monkeypatch.setattr(CampaignStore, "complete_public_query", fail_logical_completion)
        with pytest.raises(ScientificStoreError, match="completion failed closed"):
            adapter.complete(reservation, completion)
        with sqlite3.connect(campaign_path) as raw:
            assert raw.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 2
        assert ledger.projection().revision == 1
        monkeypatch.setattr(CampaignStore, "complete_public_query", original)

        adapter.complete(reservation, completion)
        with sqlite3.connect(campaign_path) as raw:
            assert raw.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 3
        assert ledger.projection().revision == 2


def test_project_six_slot_limit_survives_distinct_campaign_databases(tmp_path: Path) -> None:
    ledger_path = tmp_path / "outer.sqlite"
    ledger = OuterQueryLedger.create(ledger_path, max_queries=6)
    stores: list[CampaignStore] = []
    try:
        for index in range(6):
            store = _create_store(tmp_path / f"campaign-{index}.sqlite", f"campaign-{index}")
            stores.append(store)
            request = _request(index)
            adapter = DurableScientificLedgerAdapter(
                store,
                ledger,
                scorer_digest=_SCORER,
                evidence_registry={},
            )
            assert adapter.reserve(request).ledger_revision == index + 1
        seventh = _create_store(tmp_path / "campaign-6.sqlite", "campaign-6")
        stores.append(seventh)
        adapter = DurableScientificLedgerAdapter(
            seventh,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry={},
        )
        with pytest.raises(ScientificStoreError, match="failed closed"):
            adapter.reserve(_request(6))
        projection = ledger.projection()
        assert projection.revision == 6
        assert len(projection.candidate_fingerprints) == 6
        assert seventh.snapshot().outer_queries_used == 0
    finally:
        for store in stores:
            store.close()
        ledger.close()


def test_constructor_rejects_campaign_project_limit_mismatch(tmp_path: Path) -> None:
    with (
        _create_store(tmp_path / "campaign.sqlite", "campaign-one", outer_limit=5) as store,
        OuterQueryLedger.create(tmp_path / "outer.sqlite", max_queries=6) as ledger,
        pytest.raises(ScientificStoreError, match="same immutable identity"),
    ):
        DurableScientificLedgerAdapter(
            store,
            ledger,
            scorer_digest=_SCORER,
            evidence_registry={},
        )
