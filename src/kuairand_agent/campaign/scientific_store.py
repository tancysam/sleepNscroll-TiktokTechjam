"""Durable bridge from the scientific promotion protocol to campaign persistence.

The scientific controller intentionally carries only aggregate matched-seed metrics.  This
adapter obtains the corresponding *real* prediction identities from a trusted injected registry,
charges the project-wide ledger before the campaign-local reservation, and writes one logical
public query plus three separately attributable seed metrics.  Exact deterministic identifiers
make every write restart-safe without making score queries repeatable for altered evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from kuairand_agent.campaign.scientific import (
    HARD_OUTER_PROMOTION_LIMIT,
    MATCHED_SEEDS,
    OuterPromotionCompletion,
    OuterPromotionLedgerSnapshot,
    OuterPromotionRequest,
    OuterPromotionReservation,
)
from kuairand_agent.campaign.selector import OrganizerMetrics
from kuairand_agent.campaign.store import (
    CampaignStore,
    FailureRecord,
    MetricRecord,
    OuterQueryLedger,
    OuterQueryProjectionRecord,
    StoreError,
)

SCIENTIFIC_STORE_SCHEMA_VERSION: Final = 1
_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_RESERVATION_PREFIX: Final = "scientific-outer-"
_METRIC_PREFIX: Final = "scientific-outer-metric-"
_FAILURE_PREFIX: Final = "scientific-outer-failure-"


class ScientificStoreError(RuntimeError):
    """Raised when durable promotion evidence is absent, stale, or contradictory."""


def _digest(value: object, location: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ScientificStoreError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _plain_json(item: object) -> object:
    if isinstance(item, Mapping):
        return {str(key): _plain_json(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [_plain_json(value) for value in item]
    return item


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScientificStoreError("scientific store metadata must be finite JSON") from exc


def _same_json(left: object, right: object) -> bool:
    return _canonical_json(left) == _canonical_json(right)


@dataclass(frozen=True, slots=True)
class TrustedOuterSeedEvidence:
    """Trusted score/prediction binding registered after protected scoring.

    Registry keys are ``(promotion_request_digest, seed)``. ``score_evidence_digest`` identifies
    the protected scorer receipt or scientific run evidence; it is deliberately distinct from
    both the prediction digest and the matched-seed bundle digest.
    """

    request_digest: str
    seed: int
    metrics: OrganizerMetrics
    prediction_digest: str
    score_evidence_digest: str

    def __post_init__(self) -> None:
        _digest(self.request_digest, "trusted evidence request_digest")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ScientificStoreError("trusted evidence seed must be an unsigned 32-bit integer")
        if not isinstance(self.metrics, OrganizerMetrics):
            raise ScientificStoreError("trusted evidence metrics must be OrganizerMetrics")
        _digest(self.prediction_digest, "trusted evidence prediction_digest")
        _digest(self.score_evidence_digest, "trusted evidence score_evidence_digest")


type TrustedOuterEvidenceRegistry = Mapping[tuple[str, int], TrustedOuterSeedEvidence]


def _query_id(request_digest: str) -> str:
    return f"{_RESERVATION_PREFIX}{request_digest}"


def _metric_id(request_digest: str, seed: int) -> str:
    return f"{_METRIC_PREFIX}{request_digest}-seed-{seed}"


def _failure_id(request_digest: str) -> str:
    return f"{_FAILURE_PREFIX}{request_digest}"


def _request_metadata(request: OuterPromotionRequest, *, consumes_slot: bool) -> dict[str, object]:
    return {
        "schema_version": SCIENTIFIC_STORE_SCHEMA_VERSION,
        "kind": "scientific_outer_promotion_reservation",
        "request_digest": request.digest,
        "request": request.manifest(),
        "candidate_id": request.candidate_id,
        "candidate_fingerprint": request.candidate_fingerprint,
        "source_digest": request.source_digest,
        "parent_source_digest": request.parent_source_digest,
        "executable_diff_digest": request.executable_diff_digest,
        "material_change_digest": request.material_change_digest,
        "controller_attestation_digest": request.controller_attestation_digest,
        "consumes_slot": consumes_slot,
    }


def _seed_metric_manifest(seed: int, metrics: OrganizerMetrics) -> dict[str, object]:
    return {
        "seed": seed,
        "GAUC": metrics.gauc,
        "nDCG@5": metrics.ndcg_at_5,
        "primary": metrics.primary,
    }


class DurableScientificLedgerAdapter:
    """Implement ``OuterPromotionLedger`` over the two durable SQLite stores.

    Args:
        store: The single writable campaign store for this run.
        outer_ledger: The writable project-wide public-query ledger.
        scorer_digest: The immutable protected organizer-scorer identity.
        evidence_registry: Trusted per-request/per-seed prediction evidence. The mapping may be
            populated by the trusted runner after adapter construction, before ``complete``.
        representative_seed: The seed represented by the one logical validation-query completion.
            The other two seeds are persisted via ``CampaignStore.record_metric``.
    """

    def __init__(
        self,
        store: CampaignStore,
        outer_ledger: OuterQueryLedger,
        *,
        scorer_digest: str,
        evidence_registry: TrustedOuterEvidenceRegistry,
        representative_seed: int = MATCHED_SEEDS[0],
    ) -> None:
        if not isinstance(store, CampaignStore):
            raise ScientificStoreError("store must be CampaignStore")
        if not isinstance(outer_ledger, OuterQueryLedger):
            raise ScientificStoreError("outer_ledger must be OuterQueryLedger")
        if not isinstance(evidence_registry, Mapping):
            raise ScientificStoreError("evidence_registry must be a mapping")
        if representative_seed not in MATCHED_SEEDS:
            raise ScientificStoreError("representative_seed must be one of matched seeds 0, 1, 2")
        identity = store.identity()
        scorer = _digest(scorer_digest, "scorer_digest")
        projection = outer_ledger.projection()
        if identity.outer_query_limit != projection.max_queries:
            raise ScientificStoreError(
                "campaign and project outer-query limits must have the same immutable identity"
            )
        if projection.max_queries > HARD_OUTER_PROMOTION_LIMIT:
            raise ScientificStoreError("project outer-query limit exceeds scientific policy")
        self._store = store
        self._outer_ledger = outer_ledger
        self._registry = evidence_registry
        self._representative_seed = representative_seed
        self._campaign_digest = identity.config_digest
        self._campaign_id = identity.campaign_id
        self._benchmark_digest = identity.benchmark_digest
        self._dataset_digest = identity.dataset_manifest_digest
        self._scorer_digest = scorer
        self._max_distinct_candidates = projection.max_queries
        self._assert_project_campaign_identity(projection.queries)

    def _assert_project_campaign_identity(
        self, queries: tuple[OuterQueryProjectionRecord, ...]
    ) -> None:
        expected = (
            self._benchmark_digest,
            self._dataset_digest,
            self._scorer_digest,
        )
        for item in queries:
            if (
                item.campaign_id == self._campaign_id
                and (
                    item.benchmark_digest,
                    item.dataset_digest,
                    item.scorer_digest,
                )
                != expected
            ):
                raise ScientificStoreError(
                    "project reservation identity differs from bound campaign and scorer"
                )

    def _assert_store_identity(self) -> None:
        identity = self._store.identity()
        observed = (
            identity.campaign_id,
            identity.config_digest,
            identity.benchmark_digest,
            identity.dataset_manifest_digest,
            identity.outer_query_limit,
        )
        expected = (
            self._campaign_id,
            self._campaign_digest,
            self._benchmark_digest,
            self._dataset_digest,
            self._max_distinct_candidates,
        )
        if observed != expected:
            raise ScientificStoreError("campaign store identity changed after adapter binding")

    def snapshot(self) -> OuterPromotionLedgerSnapshot:
        self._assert_store_identity()
        projection = self._outer_ledger.projection()
        if projection.max_queries != self._max_distinct_candidates:
            raise ScientificStoreError("project ledger limit changed after adapter binding")
        self._assert_project_campaign_identity(projection.queries)
        # Scoped to this campaign, matching the per-campaign ration the ledger enforces on
        # reservation. The selector compares this set against outer_promotion_limit, so passing
        # every campaign's fingerprints made the *project* history exhaust a *campaign's*
        # allowance: once six candidates had ever been queried, every later candidate was rejected
        # as `outer_candidate_limit` even when its own campaign had spent none of its six.
        own_fingerprints = tuple(
            dict.fromkeys(
                item.candidate_fingerprint
                for item in projection.queries
                if item.campaign_id == self._campaign_id
            )
        )
        return OuterPromotionLedgerSnapshot(
            revision=projection.revision,
            campaign_digest=self._campaign_digest,
            benchmark_digest=self._benchmark_digest,
            dataset_digest=self._dataset_digest,
            scorer_digest=self._scorer_digest,
            max_distinct_candidates=self._max_distinct_candidates,
            candidate_fingerprints=own_fingerprints,
        )

    def _validate_request(self, request: OuterPromotionRequest) -> None:
        if not isinstance(request, OuterPromotionRequest):
            raise ScientificStoreError("request must be OuterPromotionRequest")
        expected = (
            self._campaign_digest,
            self._benchmark_digest,
            self._dataset_digest,
            self._scorer_digest,
        )
        observed = (
            request.campaign_digest,
            request.benchmark_digest,
            request.dataset_digest,
            request.scorer_digest,
        )
        if observed != expected:
            raise ScientificStoreError("outer promotion request identity differs from campaign")

    @staticmethod
    def _find_query(
        queries: tuple[OuterQueryProjectionRecord, ...], query_id: str
    ) -> OuterQueryProjectionRecord | None:
        return next((item for item in queries if item.query_id == query_id), None)

    def reserve(self, request: OuterPromotionRequest) -> OuterPromotionReservation:
        self._assert_store_identity()
        self._validate_request(request)
        before = self._outer_ledger.projection()
        identifier = _query_id(request.digest)
        existing = self._find_query(before.queries, identifier)
        if existing is None:
            consumes_slot = request.candidate_fingerprint not in before.candidate_fingerprints
            metadata = _request_metadata(request, consumes_slot=consumes_slot)
        else:
            consumes_slot_raw = existing.reservation_metadata.get("consumes_slot")
            if type(consumes_slot_raw) is not bool:
                raise ScientificStoreError("stored reservation has invalid consumes_slot")
            consumes_slot = consumes_slot_raw
            metadata = _request_metadata(request, consumes_slot=consumes_slot)
            if not _same_json(existing.reservation_metadata, metadata):
                raise ScientificStoreError("stored reservation differs from promotion request")

        try:
            self._store.reserve_public_query(
                self._outer_ledger,
                query_id=identifier,
                candidate_fingerprint=request.candidate_fingerprint,
                scorer_digest=self._scorer_digest,
                expected_revision=self._store.snapshot().revision,
                metadata=metadata,
            )
        except StoreError as exc:
            raise ScientificStoreError("durable outer reservation failed closed") from exc

        after = self._outer_ledger.projection()
        persisted = self._find_query(after.queries, identifier)
        if persisted is None or not _same_json(persisted.reservation_metadata, metadata):
            raise ScientificStoreError("durable outer reservation disappeared or changed")
        return OuterPromotionReservation(
            reservation_id=identifier,
            request_digest=request.digest,
            ledger_revision=persisted.reservation_revision,
            candidate_id=request.candidate_id,
            candidate_fingerprint=request.candidate_fingerprint,
            consumes_slot=consumes_slot,
        )

    def _request_record(self, reservation: OuterPromotionReservation) -> OuterQueryProjectionRecord:
        projection = self._outer_ledger.projection()
        record = self._find_query(projection.queries, reservation.reservation_id)
        if record is None:
            raise ScientificStoreError("completion names an unknown durable reservation")
        metadata = record.reservation_metadata
        expected = (
            metadata.get("request_digest"),
            metadata.get("candidate_id"),
            metadata.get("candidate_fingerprint"),
            record.reservation_revision,
            record.campaign_id,
            record.benchmark_digest,
            record.dataset_digest,
            record.scorer_digest,
        )
        observed = (
            reservation.request_digest,
            reservation.candidate_id,
            reservation.candidate_fingerprint,
            reservation.ledger_revision,
            self._campaign_id,
            self._benchmark_digest,
            self._dataset_digest,
            self._scorer_digest,
        )
        if observed != expected:
            raise ScientificStoreError("reservation identity differs from durable history")
        consumes_slot = metadata.get("consumes_slot")
        if type(consumes_slot) is not bool or consumes_slot is not reservation.consumes_slot:
            raise ScientificStoreError("reservation slot identity differs from durable history")
        request_manifest = metadata.get("request")
        if not isinstance(request_manifest, Mapping):
            raise ScientificStoreError("durable reservation request manifest is missing")
        for name in (
            "source_digest",
            "parent_source_digest",
            "executable_diff_digest",
            "material_change_digest",
            "controller_attestation_digest",
        ):
            flattened = _digest(metadata.get(name), f"stored reservation {name}")
            nested = _digest(request_manifest.get(name), f"stored request {name}")
            if flattened != nested:
                raise ScientificStoreError(
                    f"stored reservation {name} differs from its request manifest"
                )
        return record

    def _trusted_evidence(
        self, request_digest: str, seed: int, metrics: OrganizerMetrics
    ) -> TrustedOuterSeedEvidence:
        evidence = self._registry.get((request_digest, seed))
        if not isinstance(evidence, TrustedOuterSeedEvidence):
            raise ScientificStoreError(
                f"trusted prediction evidence is missing for matched seed {seed}"
            )
        if (
            evidence.request_digest != request_digest
            or evidence.seed != seed
            or evidence.metrics != metrics
        ):
            raise ScientificStoreError(f"trusted evidence differs for matched seed {seed}")
        return evidence

    @staticmethod
    def _completion_metadata(
        reservation: OuterPromotionReservation,
        completion: OuterPromotionCompletion,
        request_record: OuterQueryProjectionRecord,
        evidence: tuple[TrustedOuterSeedEvidence, ...],
        representative_seed: int,
    ) -> dict[str, object]:
        request_manifest = request_record.reservation_metadata.get("request")
        if not isinstance(request_manifest, Mapping):
            raise ScientificStoreError("durable reservation request manifest is missing")
        source_identities = {
            name: _digest(request_manifest.get(name), f"stored request {name}")
            for name in (
                "source_digest",
                "parent_source_digest",
                "executable_diff_digest",
                "material_change_digest",
                "controller_attestation_digest",
            )
        }
        return {
            "schema_version": SCIENTIFIC_STORE_SCHEMA_VERSION,
            "kind": "scientific_outer_promotion_completion",
            "request_digest": completion.request_digest,
            "request": _plain_json(request_manifest),
            "reservation_id": reservation.reservation_id,
            "reservation_revision": completion.reservation_revision,
            "candidate_id": reservation.candidate_id,
            "candidate_fingerprint": completion.candidate_fingerprint,
            **source_identities,
            "successful": completion.successful,
            "evidence_digest": completion.evidence_digest,
            "representative_seed": representative_seed,
            "seed_metrics": [
                _seed_metric_manifest(item.seed, item.metrics) for item in completion.seed_metrics
            ],
            "trusted_seed_evidence": [
                {
                    "seed": item.seed,
                    "prediction_digest": item.prediction_digest,
                    "score_evidence_digest": item.score_evidence_digest,
                }
                for item in evidence
            ],
        }

    @staticmethod
    def _validate_metric(
        existing: MetricRecord,
        expected: TrustedOuterSeedEvidence,
        metadata: object,
    ) -> None:
        observed = (
            existing.split_role,
            existing.seed,
            existing.gauc,
            existing.ndcg_at_5,
            existing.primary,
            existing.prediction_digest,
        )
        wanted = (
            "outer_valid_matched_seed",
            expected.seed,
            expected.metrics.gauc,
            expected.metrics.ndcg_at_5,
            expected.metrics.primary,
            expected.prediction_digest,
        )
        if observed != wanted or not _same_json(existing.metadata, metadata):
            raise ScientificStoreError("stored matched-seed metric conflicts with completion")

    def _record_nonrepresentative_metrics(
        self,
        request_digest: str,
        evidence: tuple[TrustedOuterSeedEvidence, ...],
        metadata: Mapping[str, object],
    ) -> None:
        for item in evidence:
            if item.seed == self._representative_seed:
                continue
            identifier = _metric_id(request_digest, item.seed)
            existing = self._store.metric(identifier)
            if existing is not None:
                self._validate_metric(existing, item, metadata)
                if existing.scorer_digest != self._scorer_digest:
                    raise ScientificStoreError("stored matched-seed scorer identity differs")
                continue
            try:
                self._store.record_metric(
                    metric_id=identifier,
                    split_role="outer_valid_matched_seed",
                    seed=item.seed,
                    gauc=item.metrics.gauc,
                    ndcg_at_5=item.metrics.ndcg_at_5,
                    primary=item.metrics.primary,
                    scorer_digest=self._scorer_digest,
                    prediction_digest=item.prediction_digest,
                    expected_revision=self._store.snapshot().revision,
                    metadata=metadata,
                )
            except StoreError as exc:
                raise ScientificStoreError("matched-seed metric persistence failed closed") from exc

    def _record_failed_completion(
        self,
        reservation: OuterPromotionReservation,
        completion: OuterPromotionCompletion,
        request_record: OuterQueryProjectionRecord,
    ) -> None:
        metadata = self._completion_metadata(
            reservation,
            completion,
            request_record,
            (),
            self._representative_seed,
        )
        identifier = _failure_id(completion.request_digest)
        existing = self._store.failure(identifier)
        if existing is not None:
            expected = FailureRecord(
                failure_id=identifier,
                category="outer_matched_seed_bundle_incomplete",
                fingerprint=completion.evidence_digest,
                retry_ordinal=0,
                metadata=metadata,
            )
            if (
                existing.failure_id != expected.failure_id
                or existing.category != expected.category
                or existing.fingerprint != expected.fingerprint
                or existing.retry_ordinal != expected.retry_ordinal
                or not _same_json(existing.metadata, expected.metadata)
            ):
                raise ScientificStoreError("failed completion conflicts with durable failure")
            return
        try:
            self._store.record_failure(
                failure_id=identifier,
                category="outer_matched_seed_bundle_incomplete",
                fingerprint=completion.evidence_digest,
                retry_ordinal=0,
                expected_revision=self._store.snapshot().revision,
                recovery_outcome="public reservation retained; incumbent preserved",
                metadata=metadata,
            )
        except StoreError as exc:
            raise ScientificStoreError("failed matched-seed bundle could not be recorded") from exc

    def complete(
        self,
        reservation: OuterPromotionReservation,
        completion: OuterPromotionCompletion,
    ) -> None:
        self._assert_store_identity()
        if not isinstance(reservation, OuterPromotionReservation):
            raise ScientificStoreError("reservation must be OuterPromotionReservation")
        if not isinstance(completion, OuterPromotionCompletion):
            raise ScientificStoreError("completion must be OuterPromotionCompletion")
        if (
            completion.reservation_id != reservation.reservation_id
            or completion.request_digest != reservation.request_digest
            or completion.reservation_revision != reservation.ledger_revision
            or completion.candidate_fingerprint != reservation.candidate_fingerprint
        ):
            raise ScientificStoreError("completion identity differs from reservation")
        request_record = self._request_record(reservation)
        prior_failure = self._store.failure(_failure_id(completion.request_digest))
        if prior_failure is not None and completion.successful:
            raise ScientificStoreError("failed matched-seed bundle cannot later become successful")
        if not completion.successful:
            if request_record.state == "COMPLETED":
                raise ScientificStoreError("completed public query cannot later become failed")
            self._record_failed_completion(reservation, completion, request_record)
            return

        seeds = tuple(item.seed for item in completion.seed_metrics)
        if seeds != MATCHED_SEEDS:
            raise ScientificStoreError("successful completion requires ordered seeds 0, 1, and 2")
        evidence = tuple(
            self._trusted_evidence(completion.request_digest, item.seed, item.metrics)
            for item in completion.seed_metrics
        )
        metadata = self._completion_metadata(
            reservation,
            completion,
            request_record,
            evidence,
            self._representative_seed,
        )
        self._record_nonrepresentative_metrics(
            completion.request_digest,
            evidence,
            metadata,
        )
        representative = next(item for item in evidence if item.seed == self._representative_seed)
        try:
            self._store.complete_public_query(
                self._outer_ledger,
                query_id=reservation.reservation_id,
                result_digest=completion.evidence_digest,
                gauc=representative.metrics.gauc,
                ndcg_at_5=representative.metrics.ndcg_at_5,
                primary=representative.metrics.primary,
                prediction_digest=representative.prediction_digest,
                scorer_digest=self._scorer_digest,
                expected_revision=self._store.snapshot().revision,
                metric_id=_metric_id(completion.request_digest, representative.seed),
                metadata=metadata,
            )
        except StoreError as exc:
            raise ScientificStoreError("logical outer-query completion failed closed") from exc


__all__ = [
    "SCIENTIFIC_STORE_SCHEMA_VERSION",
    "DurableScientificLedgerAdapter",
    "ScientificStoreError",
    "TrustedOuterEvidenceRegistry",
    "TrustedOuterSeedEvidence",
]
