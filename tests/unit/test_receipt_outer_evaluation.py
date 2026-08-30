from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from kuairand_agent.campaign.receipt_outer_evaluation import (
    ReceiptAwareOuterEvaluationLedger,
)
from kuairand_agent.campaign.scientific import (
    OuterPromotionCompletion,
    OuterPromotionLedgerSnapshot,
    OuterPromotionRequest,
    OuterPromotionReservation,
)
from kuairand_agent.campaign.scoring_receipts import (
    ScoringEquivalenceIdentity,
    ScoringReceiptBook,
    TrustedScoreReceipt,
)
from kuairand_agent.campaign.selector import OrganizerMetrics, SeedMetrics
from kuairand_agent.scoring.protected import ScoreResult
from kuairand_agent.scoring.submission import prediction_digest

_CAMPAIGN = "1" * 64
_BENCHMARK = "2" * 64
_DATASET = "3" * 64
_SCORER = "4" * 64
_FINGERPRINT = "5" * 64


def _request() -> OuterPromotionRequest:
    return OuterPromotionRequest(
        campaign_digest=_CAMPAIGN,
        candidate_id="candidate-new",
        candidate_fingerprint=_FINGERPRINT,
        source_digest="6" * 64,
        parent_source_digest="7" * 64,
        executable_diff_digest="8" * 64,
        material_change_digest="9" * 64,
        controller_attestation_digest="a" * 64,
        benchmark_digest=_BENCHMARK,
        dataset_digest=_DATASET,
        scorer_digest=_SCORER,
        training_policy_digest="b" * 64,
    )


def _receipt(scores: npt.NDArray[np.float64]) -> TrustedScoreReceipt:
    return TrustedScoreReceipt(
        identity=ScoringEquivalenceIdentity(
            benchmark_digest=_BENCHMARK,
            dataset_digest=_DATASET,
            scorer_digest=_SCORER,
            split_role="outer_valid_matched_seed",
            prediction_digest=prediction_digest(scores),
        ),
        metrics=OrganizerMetrics(0.67, 0.53),
        origin_query_id="outer-prior",
        origin_revision=2,
        origin_result_digest="c" * 64,
        origin_evidence_digest="d" * 64,
        seed=0,
        score_evidence_digest="e" * 64,
    )


@dataclass
class _RecordingLedger:
    revision: int = 2
    reserve_calls: list[OuterPromotionRequest] = field(default_factory=list)
    complete_calls: list[tuple[OuterPromotionReservation, OuterPromotionCompletion]] = field(
        default_factory=list
    )

    def snapshot(self) -> OuterPromotionLedgerSnapshot:
        return OuterPromotionLedgerSnapshot(
            revision=self.revision,
            campaign_digest=_CAMPAIGN,
            benchmark_digest=_BENCHMARK,
            dataset_digest=_DATASET,
            scorer_digest=_SCORER,
            max_distinct_candidates=6,
            candidate_fingerprints=(),
        )

    def reserve(self, request: OuterPromotionRequest) -> OuterPromotionReservation:
        self.reserve_calls.append(request)
        self.revision += 1
        return OuterPromotionReservation(
            reservation_id="durable-reservation",
            request_digest=request.digest,
            ledger_revision=self.revision,
            candidate_id=request.candidate_id,
            candidate_fingerprint=request.candidate_fingerprint,
            consumes_slot=True,
        )

    def complete(
        self,
        reservation: OuterPromotionReservation,
        completion: OuterPromotionCompletion,
    ) -> None:
        self.complete_calls.append((reservation, completion))
        self.revision += 1


def _completion(
    request: OuterPromotionRequest,
    reservation: OuterPromotionReservation,
    metrics: OrganizerMetrics,
) -> OuterPromotionCompletion:
    return OuterPromotionCompletion(
        reservation_id=reservation.reservation_id,
        request_digest=request.digest,
        reservation_revision=reservation.ledger_revision,
        candidate_fingerprint=request.candidate_fingerprint,
        successful=True,
        seed_metrics=(SeedMetrics(0, metrics),),
        evidence_digest="f" * 64,
    )


def test_exact_receipt_reuse_happens_after_prediction_identity_without_reservation() -> None:
    scores = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    receipt = _receipt(scores)
    durable = _RecordingLedger()
    ledger = ReceiptAwareOuterEvaluationLedger(durable, ScoringReceiptBook((receipt,)))
    request = _request()

    reservation = ledger.reserve(request)
    assert durable.reserve_calls == []

    callback_called = False

    def protected_callback(_scores: npt.NDArray[np.float64]) -> ScoreResult:
        nonlocal callback_called
        callback_called = True
        raise AssertionError("the protected scorer must not run on an exact receipt hit")

    result = ledger.score(
        request=request,
        seed=0,
        scores=scores,
        users=2,
        rows=3,
        protected_callback=protected_callback,
    )
    assert callback_called is False
    assert result.prediction_digest == prediction_digest(scores)
    assert (result.gauc, result.ndcg_at_5) == (0.67, 0.53)
    assert result.primary == (0.67 + 0.53) / 2.0
    assert ledger.receipt_for(request.digest, 0) == receipt
    assert durable.reserve_calls == []

    ledger.complete(reservation, _completion(request, reservation, receipt.metrics))
    assert durable.reserve_calls == []
    assert durable.complete_calls == []


def test_receipt_miss_reserves_before_protected_scoring_and_completes_normally() -> None:
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
    durable = _RecordingLedger()
    different_prediction = _receipt(np.asarray([0.1, 0.2, 0.3], dtype=np.float64))
    ledger = ReceiptAwareOuterEvaluationLedger(
        durable,
        ScoringReceiptBook((different_prediction,)),
    )
    request = _request()
    provisional = ledger.reserve(request)
    assert durable.reserve_calls == []

    def protected_callback(values: npt.NDArray[np.float64]) -> ScoreResult:
        assert durable.reserve_calls == [request]
        return ScoreResult(
            gauc=0.66,
            ndcg_at_5=0.52,
            primary=0.59,
            users=2,
            rows=3,
            scorer_digest=_SCORER,
            prediction_digest=prediction_digest(values),
            runtime_seconds=0.25,
        )

    result = ledger.score(
        request=request,
        seed=0,
        scores=scores,
        users=2,
        rows=3,
        protected_callback=protected_callback,
    )
    assert result.primary == 0.59
    assert ledger.receipt_for(request.digest, 0) is None
    ledger.complete(
        provisional,
        _completion(request, provisional, OrganizerMetrics(result.gauc, result.ndcg_at_5)),
    )
    assert len(durable.complete_calls) == 1
    durable_reservation, durable_completion = durable.complete_calls[0]
    assert durable_reservation.reservation_id == "durable-reservation"
    assert durable_completion.reservation_id == durable_reservation.reservation_id
    assert durable_completion.reservation_revision == durable_reservation.ledger_revision
