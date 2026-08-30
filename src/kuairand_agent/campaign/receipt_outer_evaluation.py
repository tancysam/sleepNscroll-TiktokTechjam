"""Lazy protected-score admission with exact read-only receipt reuse.

The scientific controller must decide whether an outer candidate is eligible before inference,
while scoring equivalence cannot be known until the exact prediction vector exists.  This module
bridges those two moments.  ``reserve`` returns a provisional in-memory admission.  ``score``
then hashes the canonical float64 predictions and either returns an exact trusted receipt or
durably reserves the project query immediately before invoking the protected scorer.

The wrapped durable ledger remains the authority for misses.  Receipt hits never mutate it and
never weaken candidate/source identity: only the scorer-input identity is reusable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from kuairand_agent.campaign.scientific import (
    OuterPromotionCompletion,
    OuterPromotionLedger,
    OuterPromotionLedgerSnapshot,
    OuterPromotionRequest,
    OuterPromotionReservation,
)
from kuairand_agent.campaign.scoring_receipts import (
    DEFAULT_MATCHED_SEED_SPLIT_ROLE,
    ScoringEquivalenceIdentity,
    ScoringReceiptBook,
    TrustedScoreReceipt,
)
from kuairand_agent.scoring.protected import ScoreResult
from kuairand_agent.scoring.submission import prediction_digest

type ProtectedScoreCallback = Callable[[npt.NDArray[np.float64]], ScoreResult]


class ReceiptAwareOuterEvaluationError(RuntimeError):
    """Raised when provisional, scorer, or receipt identities contradict each other."""


class ReceiptAwareOuterEvaluationLedger:
    """Implement the outer-ledger interface while deferring new query reservation.

    A caller must invoke ``reserve(request)`` before ``score``.  On a receipt miss, the wrapped
    ledger is reserved synchronously before ``protected_callback`` is entered.  On an exact hit,
    ``complete`` is a validated no-op, so neither the project ledger nor campaign query counter
    changes.
    """

    def __init__(self, durable: OuterPromotionLedger, receipts: ScoringReceiptBook) -> None:
        if not isinstance(receipts, ScoringReceiptBook):
            raise ReceiptAwareOuterEvaluationError("receipts must be ScoringReceiptBook")
        snapshot = durable.snapshot()
        if not isinstance(snapshot, OuterPromotionLedgerSnapshot):
            raise ReceiptAwareOuterEvaluationError("durable ledger returned an invalid snapshot")
        self._durable = durable
        self._receipts = receipts
        self._virtual_revision = snapshot.revision
        self._requests: dict[str, OuterPromotionRequest] = {}
        self._provisional: dict[str, OuterPromotionReservation] = {}
        self._durable_reservations: dict[str, OuterPromotionReservation] = {}
        self._score_results: dict[tuple[str, int], ScoreResult] = {}
        self._score_prediction_digests: dict[tuple[str, int], str] = {}
        self._used_receipts: dict[tuple[str, int], TrustedScoreReceipt] = {}

    def snapshot(self) -> OuterPromotionLedgerSnapshot:
        observed = self._durable.snapshot()
        if not isinstance(observed, OuterPromotionLedgerSnapshot):
            raise ReceiptAwareOuterEvaluationError("durable ledger returned an invalid snapshot")
        if observed.revision > self._virtual_revision:
            self._virtual_revision = observed.revision
        return replace(observed, revision=self._virtual_revision)

    def reserve(self, request: OuterPromotionRequest) -> OuterPromotionReservation:
        if not isinstance(request, OuterPromotionRequest):
            raise ReceiptAwareOuterEvaluationError("request must be OuterPromotionRequest")
        prior_request = self._requests.get(request.digest)
        if prior_request is not None:
            if prior_request != request:
                raise ReceiptAwareOuterEvaluationError("promotion request digest is contradictory")
            return self._provisional[request.digest]

        observed = self._durable.snapshot()
        if request.candidate_fingerprint in observed.candidate_fingerprints:
            durable = self._durable.reserve(request)
            if not isinstance(durable, OuterPromotionReservation):
                raise ReceiptAwareOuterEvaluationError(
                    "durable ledger returned an invalid reservation"
                )
            self._requests[request.digest] = request
            self._provisional[request.digest] = durable
            self._durable_reservations[request.digest] = durable
            self._virtual_revision = max(self._virtual_revision, durable.ledger_revision)
            return durable

        self._virtual_revision = max(self._virtual_revision, observed.revision) + 1
        provisional = OuterPromotionReservation(
            reservation_id=f"scientific-receipt-aware-{request.digest}",
            request_digest=request.digest,
            ledger_revision=self._virtual_revision,
            candidate_id=request.candidate_id,
            candidate_fingerprint=request.candidate_fingerprint,
            consumes_slot=True,
        )
        self._requests[request.digest] = request
        self._provisional[request.digest] = provisional
        return provisional

    def _assert_pending(self, request: OuterPromotionRequest) -> None:
        if not isinstance(request, OuterPromotionRequest):
            raise ReceiptAwareOuterEvaluationError("request must be OuterPromotionRequest")
        if self._requests.get(request.digest) != request:
            raise ReceiptAwareOuterEvaluationError(
                "score requires the exact provisionally admitted promotion request"
            )

    def _ensure_durable_reservation(
        self, request: OuterPromotionRequest
    ) -> OuterPromotionReservation:
        persisted = self._durable_reservations.get(request.digest)
        if persisted is not None:
            return persisted
        persisted = self._durable.reserve(request)
        if not isinstance(persisted, OuterPromotionReservation):
            raise ReceiptAwareOuterEvaluationError("durable ledger returned an invalid reservation")
        self._durable_reservations[request.digest] = persisted
        self._virtual_revision = max(self._virtual_revision, persisted.ledger_revision)
        return persisted

    def score(
        self,
        *,
        request: OuterPromotionRequest,
        seed: int,
        scores: npt.NDArray[np.float64],
        users: int,
        rows: int,
        protected_callback: ProtectedScoreCallback,
    ) -> ScoreResult:
        """Reuse exact metrics, or reserve durably before calling the protected scorer."""

        self._assert_pending(request)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ReceiptAwareOuterEvaluationError("seed must be an unsigned 32-bit integer")
        if type(users) is not int or users <= 0:
            raise ReceiptAwareOuterEvaluationError("users must be a positive integer")
        if type(rows) is not int or rows <= 0:
            raise ReceiptAwareOuterEvaluationError("rows must be a positive integer")
        if not callable(protected_callback):
            raise ReceiptAwareOuterEvaluationError("protected_callback must be callable")
        observed_prediction = prediction_digest(scores)
        key = (request.digest, seed)
        prior_prediction = self._score_prediction_digests.get(key)
        if prior_prediction is not None:
            if prior_prediction != observed_prediction:
                raise ReceiptAwareOuterEvaluationError(
                    "one request and seed produced contradictory prediction identities"
                )
            return self._score_results[key]

        identity = ScoringEquivalenceIdentity(
            benchmark_digest=request.benchmark_digest,
            dataset_digest=request.dataset_digest,
            scorer_digest=request.scorer_digest,
            split_role=DEFAULT_MATCHED_SEED_SPLIT_ROLE,
            prediction_digest=observed_prediction,
        )
        receipt = self._receipts.find(identity)
        if receipt is None:
            self._ensure_durable_reservation(request)
            result = protected_callback(scores)
        else:
            result = ScoreResult(
                gauc=receipt.metrics.gauc,
                ndcg_at_5=receipt.metrics.ndcg_at_5,
                primary=(receipt.metrics.gauc + receipt.metrics.ndcg_at_5) / 2.0,
                users=users,
                rows=rows,
                scorer_digest=request.scorer_digest,
                prediction_digest=observed_prediction,
                runtime_seconds=0.0,
            )
            self._used_receipts[key] = receipt
        self._score_prediction_digests[key] = observed_prediction
        self._score_results[key] = result
        return result

    def receipt_for(self, request_digest: str, seed: int) -> TrustedScoreReceipt | None:
        """Return the receipt used for one current request/seed, never a fuzzy match."""

        return self._used_receipts.get((request_digest, seed))

    def complete(
        self,
        reservation: OuterPromotionReservation,
        completion: OuterPromotionCompletion,
    ) -> None:
        if not isinstance(reservation, OuterPromotionReservation):
            raise ReceiptAwareOuterEvaluationError(
                "reservation must be OuterPromotionReservation"
            )
        if not isinstance(completion, OuterPromotionCompletion):
            raise ReceiptAwareOuterEvaluationError("completion must be OuterPromotionCompletion")
        request = self._requests.get(completion.request_digest)
        provisional = self._provisional.get(completion.request_digest)
        if request is None or provisional != reservation:
            raise ReceiptAwareOuterEvaluationError(
                "completion differs from its provisional admission"
            )
        expected = (
            reservation.reservation_id,
            reservation.ledger_revision,
            reservation.candidate_fingerprint,
        )
        observed = (
            completion.reservation_id,
            completion.reservation_revision,
            completion.candidate_fingerprint,
        )
        if observed != expected:
            raise ReceiptAwareOuterEvaluationError(
                "completion identity differs from its provisional admission"
            )

        durable = self._durable_reservations.get(request.digest)
        if durable is not None:
            translated = replace(
                completion,
                reservation_id=durable.reservation_id,
                reservation_revision=durable.ledger_revision,
            )
            self._durable.complete(durable, translated)
            self._virtual_revision = max(self._virtual_revision, self._durable.snapshot().revision)
            return

        if not completion.successful:
            return
        for item in completion.seed_metrics:
            receipt = self._used_receipts.get((request.digest, item.seed))
            if receipt is None or receipt.metrics != item.metrics:
                raise ReceiptAwareOuterEvaluationError(
                    "receipt-only completion lacks exact trusted seed metrics"
                )


__all__ = [
    "ReceiptAwareOuterEvaluationError",
    "ReceiptAwareOuterEvaluationLedger",
]
