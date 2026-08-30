from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest

from kuairand_agent.campaign.scoring_receipts import (
    ScoringEquivalenceIdentity,
    ScoringReceiptBook,
    ScoringReceiptError,
)
from kuairand_agent.campaign.store import (
    OuterQueryLedgerProjection,
    OuterQueryProjectionRecord,
)

_BENCHMARK = "1" * 64
_DATASET = "2" * 64
_SCORER = "3" * 64


def _metadata(
    *,
    candidate: str = "candidate-a",
    source: str = "4" * 64,
    request: str = "5" * 64,
    evidence: str = "6" * 64,
    prediction: str = "7" * 64,
    score_evidence: str = "8" * 64,
    gauc: float = 0.67,
    ndcg: float = 0.53,
    seed: int = 0,
    successful: bool = True,
    split_role: str = "outer_valid_matched_seed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "scientific_outer_promotion_completion",
        "request_digest": request,
        "request": {
            "schema_version": 1,
            "request_digest": request,
            "candidate_id": candidate,
            "candidate_fingerprint": "9" * 64,
            "source_digest": source,
            "parent_source_digest": "a" * 64,
            "executable_diff_digest": "b" * 64,
            "material_change_digest": "c" * 64,
            "controller_attestation_digest": "d" * 64,
            "benchmark_digest": _BENCHMARK,
            "dataset_digest": _DATASET,
            "scorer_digest": _SCORER,
            "training_policy_digest": "e" * 64,
        },
        "reservation_id": "scientific-outer-" + request,
        "reservation_revision": 4,
        "candidate_id": candidate,
        "candidate_fingerprint": "9" * 64,
        "source_digest": source,
        "successful": successful,
        "evidence_digest": evidence,
        "representative_seed": seed,
        "split_role": split_role,
        "seed_metrics": [
            {"seed": seed, "GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2}
        ],
        "trusted_seed_evidence": [
            {
                "seed": seed,
                "prediction_digest": prediction,
                "score_evidence_digest": score_evidence,
            }
        ],
    }


def _row(
    *,
    query_id: str = "query-a",
    revision: int = 4,
    metadata: Mapping[str, object] | None = None,
    state: str = "COMPLETED",
    result: str = "f" * 64,
    prediction: str = "7" * 64,
) -> OuterQueryProjectionRecord:
    if metadata is not None:
        metadata = {**metadata, "reservation_revision": revision}
    return OuterQueryProjectionRecord(
        query_id=query_id,
        campaign_id="campaign-a",
        benchmark_digest=_BENCHMARK,
        dataset_digest=_DATASET,
        scorer_digest=_SCORER,
        candidate_fingerprint="9" * 64,
        reservation_revision=revision,
        state=state,
        event_seq=2 if state == "COMPLETED" else 1,
        result_digest=result if state == "COMPLETED" else None,
        reservation_metadata={"request_digest": "5" * 64},
        latest_metadata={} if metadata is None else metadata,
    )


def _projection(*rows: OuterQueryProjectionRecord) -> OuterQueryLedgerProjection:
    return OuterQueryLedgerProjection(
        revision=sum(row.event_seq for row in rows), max_queries=6, queries=rows
    )


def test_completed_projection_yields_receipt_with_scoring_identity_and_provenance() -> None:
    book = ScoringReceiptBook.from_projection(_projection(_row(metadata=_metadata())))
    identity = ScoringEquivalenceIdentity(
        benchmark_digest=_BENCHMARK,
        dataset_digest=_DATASET,
        scorer_digest=_SCORER,
        split_role="outer_valid_matched_seed",
        prediction_digest="7" * 64,
    )

    receipt = book.find(identity)

    assert receipt is not None
    assert receipt.identity == identity
    assert receipt.gauc == 0.67
    assert receipt.ndcg_at_5 == 0.53
    assert receipt.primary == 0.6
    assert receipt.origin_query_id == "query-a"
    assert receipt.origin_revision == 4
    assert receipt.origin_result_digest == "f" * 64
    assert receipt.origin_evidence_digest == "6" * 64
    assert receipt.score_evidence_digest == "8" * 64
    assert receipt.seed == 0
    assert receipt.identity.manifest()["schema_version"] == 1
    assert len(receipt.identity.digest) == 64
    assert receipt.digest == receipt.digest
    assert receipt.manifest()["identity"] == receipt.identity.manifest()
    changed_origin = replace(receipt, origin_query_id="query-other", origin_revision=99)
    assert changed_origin.identity.digest == receipt.identity.digest
    assert changed_origin.digest != receipt.digest


def test_execution_and_operational_metadata_do_not_change_equivalence() -> None:
    first = _metadata()
    second = _metadata(
        candidate="candidate-b", source="a" * 64, request="b" * 64, evidence="c" * 64
    )
    second["execution_id"] = "execution-changed"
    second["trusted_seed_evidence"] = [
        {
            "seed": 0,
            "prediction_digest": "7" * 64,
            "score_evidence_digest": "d" * 64,
        }
    ]
    rows = (
        _row(query_id="later", revision=9, metadata=second),
        _row(query_id="earlier", revision=3, metadata=first),
    )

    book = ScoringReceiptBook.from_projection(_projection(*rows))
    identity = ScoringEquivalenceIdentity(
        _BENCHMARK, _DATASET, _SCORER, "outer_valid_matched_seed", "7" * 64
    )

    receipt = book.find(identity)
    assert receipt is not None
    assert receipt.origin_query_id == "earlier"
    assert receipt.origin_revision == 3
    assert len(book) == 1


def test_equivalence_digest_excludes_operational_provenance_but_binds_scorer_inputs() -> None:
    baseline = ScoringEquivalenceIdentity(_BENCHMARK, _DATASET, _SCORER, "outer_valid", "7" * 64)
    same_inputs = ScoringEquivalenceIdentity(
        benchmark_digest=_BENCHMARK,
        dataset_digest=_DATASET,
        scorer_digest=_SCORER,
        split_role="outer_valid",
        prediction_digest="7" * 64,
    )
    changed_prediction = ScoringEquivalenceIdentity(
        _BENCHMARK, _DATASET, _SCORER, "outer_valid", "8" * 64
    )

    assert baseline.digest == same_inputs.digest
    assert baseline.digest != changed_prediction.digest
    assert set(baseline.manifest()) == {
        "schema_version",
        "benchmark_digest",
        "dataset_digest",
        "scorer_digest",
        "split_role",
        "prediction_digest",
    }

    for field in ("benchmark_digest", "dataset_digest", "scorer_digest", "prediction_digest"):
        values = {
            "benchmark_digest": _BENCHMARK,
            "dataset_digest": _DATASET,
            "scorer_digest": _SCORER,
            "prediction_digest": "7" * 64,
        }
        values[field] = "f" * 64
        changed = ScoringEquivalenceIdentity(
            benchmark_digest=values["benchmark_digest"],
            dataset_digest=values["dataset_digest"],
            scorer_digest=values["scorer_digest"],
            split_role="outer_valid",
            prediction_digest=values["prediction_digest"],
        )
        assert changed.digest != baseline.digest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prediction_digest", "a" * 64),
        ("dataset_digest", "b" * 64),
        ("scorer_digest", "c" * 64),
    ),
)
def test_prediction_dataset_and_scorer_changes_do_not_match(field: str, value: str) -> None:
    original_book = ScoringReceiptBook.from_projection(_projection(_row(metadata=_metadata())))
    row = _row(metadata=_metadata())
    if field == "prediction_digest":
        metadata = _metadata(prediction=value)
        row = replace(row, latest_metadata=metadata)
    else:
        metadata = _metadata()
        request = dict(cast(Mapping[str, object], metadata["request"]))
        request[field] = value
        metadata["request"] = request
        if field == "dataset_digest":
            row = replace(row, dataset_digest=value, latest_metadata=metadata)
        else:
            row = replace(row, scorer_digest=value, latest_metadata=metadata)
    changed_book = ScoringReceiptBook.from_projection(_projection(row))

    original = ScoringEquivalenceIdentity(
        _BENCHMARK, _DATASET, _SCORER, "outer_valid_matched_seed", "7" * 64
    )
    changed = ScoringEquivalenceIdentity(
        benchmark_digest=_BENCHMARK,
        dataset_digest=value if field == "dataset_digest" else _DATASET,
        scorer_digest=value if field == "scorer_digest" else _SCORER,
        split_role="outer_valid_matched_seed",
        prediction_digest=value if field == "prediction_digest" else "7" * 64,
    )
    assert original_book.find(changed) is None
    assert changed_book.find(original) is None


def test_conflicting_metrics_for_same_identity_fail_closed() -> None:
    first = _row(query_id="one", revision=2, metadata=_metadata())
    second = _row(query_id="two", revision=5, metadata=_metadata(gauc=0.68))

    with pytest.raises(ScoringReceiptError, match="conflicting metrics"):
        ScoringReceiptBook.from_projection(_projection(first, second))


def test_reserved_and_failed_rows_are_ignored_but_malformed_completed_is_rejected() -> None:
    reserved = _row(state="RESERVED", metadata={})
    failed_metadata = _metadata(successful=False)
    failed = _row(query_id="failed", metadata=failed_metadata)
    assert len(ScoringReceiptBook.from_projection(_projection(reserved, failed))) == 0

    malformed = _metadata()
    del malformed["trusted_seed_evidence"]
    with pytest.raises(ScoringReceiptError, match="trusted_seed_evidence"):
        ScoringReceiptBook.from_projection(_projection(_row(metadata=malformed)))


def test_lookup_is_deterministic_and_from_completed_rows_is_pure_adapter() -> None:
    rows = [
        _row(query_id="z", revision=8, metadata=_metadata()),
        _row(query_id="a", revision=8, metadata=_metadata()),
    ]
    first = ScoringReceiptBook.from_completed_rows(rows)
    second = ScoringReceiptBook.from_completed_rows(reversed(rows))
    identity = ScoringEquivalenceIdentity(
        _BENCHMARK, _DATASET, _SCORER, "outer_valid_matched_seed", "7" * 64
    )

    assert first.find(identity) == second.find(identity)
    assert first.find(identity).origin_query_id == "a"  # type: ignore[union-attr]
    assert rows[0].latest_metadata["evidence_digest"] == "6" * 64
