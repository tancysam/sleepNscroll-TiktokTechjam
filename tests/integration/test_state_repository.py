from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from kuairand_agent.data.canonical import OUTCOME_FIELDS
from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import (
    BundleId,
    CampaignId,
    ContractId,
    DecisionId,
    FamilyId,
    PredictionId,
    canonical_json_bytes,
    canonical_json_sha256,
)
from kuairand_agent.finalization.bundle import (
    REQUIRED_EVIDENCE_ROLES,
    BundleFinalizationRequest,
    BundleFinalizer,
    FrozenFileReceipt,
    TerminalProjectionBinding,
)
from kuairand_agent.finalization.organizer_check import REQUIRED_DATA_FILENAMES
from kuairand_agent.finalization.replay import (
    CleanReplayEvidence,
    FinalReplayEvidence,
    FrozenReplayIdentity,
    ReplayEquality,
    ValidationReplayEvidence,
)
from kuairand_agent.finalization.replay_grades import (
    ReplayGradeReceipt,
    derive_clean_replay_grade_receipts,
)
from kuairand_agent.observability.receipts import ScriptedReplayReceipt
from kuairand_agent.state import (
    DurableRecord,
    LeaseConflictError,
    PreparedFinalizationRequiredError,
    PreparedSourceStaleError,
    PreparedTerminalProjection,
    ProtectedBudgetExhaustedError,
    ProtectedOutcomeKind,
    ProtectedOutcomeTerminalError,
    PublishedBundleReceipt,
    PublishedBundleVerificationError,
    RecordKind,
    StateInvariantError,
    StateRepository,
    TerminalPreparation,
)
from kuairand_agent.state import projections as state_projections
from kuairand_agent.state.repository import PostpublicationResourceReceipt


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(slots=True)
class _Clock:
    value: datetime = datetime(2026, 8, 31, 2, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


@dataclass(frozen=True, slots=True)
class _Lineage:
    family_id: FamilyId
    experiment_id: str
    trial_id: str
    attempt_id: str
    artifact_id: str
    prediction_id: PredictionId


def _campaign(
    repository: StateRepository,
    *,
    contract: ContractId,
    campaign: CampaignId,
    key: str,
    limit: int,
    config: Mapping[str, object] | None = None,
) -> None:
    repository.create_campaign(
        campaign_id=campaign,
        contract_id=contract,
        contract_manifest={"contract": "frozen"},
        config=config
        or {
            "campaign": campaign.value,
            "resource_profile": {
                "name": "state-test-scripted-fixture",
                "preferred_backend": "scripted",
                "device": "cpu",
            },
        },
        idempotency_key=key,
        protected_query_limit=limit,
    )


def _lineage(
    repository: StateRepository,
    *,
    contract: ContractId,
    campaign: CampaignId,
    suffix: str,
    prediction_payload: Mapping[str, object] | None = None,
) -> _Lineage:
    family = FamilyId(_digest(f"family-{suffix}"))
    experiment = _digest(f"experiment-{suffix}")
    trial = _digest(f"trial-{suffix}")
    attempt = _digest(f"attempt-{suffix}")
    artifact = _digest(f"artifact-{suffix}")
    prediction = PredictionId(_digest(f"prediction-{suffix}"))
    artifact_bytes = f"prediction-bytes-{suffix}".encode()
    prediction_bytes_sha256 = _digest(f"prediction-bytes-{suffix}")
    trainer_identity = {"backend": "scripted", "device": "cpu"}
    resources = {
        "wall_seconds": 0.1,
        "cpu_seconds": 0.05,
        "cpu_seconds_measured": True,
        "peak_rss_bytes": 1024,
        "peak_disk_bytes": 0,
        "peak_process_count": 1,
        "threads": 1,
        "device": "cpu",
    }
    timing = {
        "started_monotonic_ns": 0,
        "ended_monotonic_ns": 100_000_000,
        "wall_seconds": 0.1,
    }
    verified_path = repository.database_path.parent / f"{artifact}.bin"
    verified_path.write_bytes(artifact_bytes)
    repository.register(
        DurableRecord(
            RecordKind.FAMILY,
            family,
            campaign,
            contract,
            attributes={"protected_eligible": True},
            payload={"family": suffix},
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.EXPERIMENT,
            experiment,
            campaign,
            contract,
            references={"family_id": family},
            payload={"spec": suffix},
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.TRIAL,
            trial,
            campaign,
            contract,
            references={"experiment_id": experiment},
            payload={"backend": "cpu"},
            state="READY",
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.ATTEMPT,
            attempt,
            campaign,
            contract,
            references={"trial_id": trial},
            attributes={"attempt_ordinal": 1, "process_identity": {"pid": 123}},
            payload={"infrastructure_retry": 1, "trainer_identity": trainer_identity},
            state="RUNNING",
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.ARTIFACT,
            artifact,
            campaign,
            contract,
            references={"attempt_id": attempt},
            attributes={
                "kind": "prediction",
                "relative_path": f"artifacts/{suffix}.bin",
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "size_bytes": len(artifact_bytes),
                "verified_path": verified_path,
            },
            payload={"available": True},
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.PREDICTION,
            prediction,
            campaign,
            contract,
            references={"trial_id": trial, "artifact_id": artifact},
            attributes={
                "ordered_rows_sha256": _digest(f"rows-{suffix}"),
                "prediction_bytes_sha256": prediction_bytes_sha256,
            },
            payload={
                "row_count": 2,
                "trial_result": {
                    "trial_id": trial,
                    "attempt_id": attempt,
                    "prediction_id": prediction.value,
                    "prediction_sha256": prediction_bytes_sha256,
                    "trainer_identity": trainer_identity,
                    "resources": resources,
                    "timing": timing,
                },
                **({} if prediction_payload is None else dict(prediction_payload)),
            },
        )
    )
    return _Lineage(family, experiment, trial, attempt, artifact, prediction)


def test_protected_query_is_exact_contract_scoped_and_unknown_is_terminal(tmp_path: Path) -> None:
    repository = StateRepository.open(tmp_path / "state")
    contract = ContractId(_digest("contract"))
    campaign = CampaignId(_digest("campaign-a"))
    _campaign(repository, contract=contract, campaign=campaign, key="campaign-a", limit=1)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="a")
    provider_operation_id = _digest("pre-protected-provider-operation")
    repository.register(
        DurableRecord(
            RecordKind.PROVIDER_OPERATION,
            provider_operation_id,
            campaign,
            contract,
            state="PENDING",
        )
    )

    reservation = repository.reserve_protected_query(
        reservation_id=_digest("reservation-a"),
        campaign_id=campaign,
        contract_id=contract,
        family_id=lineage.family_id,
        prediction_id=lineage.prediction_id,
        idempotency_key="score-once-a",
        expected_campaign_revision=0,
    )
    replay = repository.reserve_protected_query(
        reservation_id=_digest("reservation-a"),
        campaign_id=campaign,
        contract_id=contract,
        family_id=lineage.family_id,
        prediction_id=lineage.prediction_id,
        idempotency_key="score-once-a",
        expected_campaign_revision=0,
    )
    reservation_lease = repository.claim_lease(
        resource_kind="protected_query_reservation",
        resource_id=reservation.reservation_id,
        owner_id="protected-scorer",
        ttl=timedelta(seconds=30),
    )
    with pytest.raises(LeaseConflictError, match="current durable write fence"):
        repository.complete_protected_query(
            evaluation_id=_digest("evaluation-a"),
            reservation_id=reservation.reservation_id,
            outcome=ProtectedOutcomeKind.UNKNOWN,
            unknown_reason="scorer response could not be proven",
        )
    outcome = repository.complete_protected_query(
        evaluation_id=_digest("evaluation-a"),
        reservation_id=reservation.reservation_id,
        outcome=ProtectedOutcomeKind.UNKNOWN,
        unknown_reason="scorer response could not be proven",
        lease=reservation_lease,
    )
    repository.release_lease(
        resource_kind="protected_query_reservation",
        resource_id=reservation.reservation_id,
        owner_id=reservation_lease.owner_id,
        fence_token=reservation_lease.fence_token,
        complete=True,
    )

    assert reservation.created and reservation.query_ordinal == 1
    assert not replay.created and replay.query_ordinal == 1
    assert outcome.outcome is ProtectedOutcomeKind.UNKNOWN
    assert not repository.register(
        DurableRecord(
            RecordKind.FAMILY,
            lineage.family_id,
            campaign,
            contract,
            attributes={"protected_eligible": True},
            payload={"family": "a"},
        )
    )
    with pytest.raises(StateInvariantError, match="research is permanently frozen"):
        repository.register(
            DurableRecord(
                RecordKind.FAMILY,
                FamilyId(_digest("post-protected-family")),
                campaign,
                contract,
                attributes={"protected_eligible": True},
            )
        )
    with pytest.raises(StateInvariantError, match="research is permanently frozen"):
        repository.register(
            DurableRecord(
                RecordKind.PROVIDER_OPERATION,
                _digest("post-protected-provider-operation"),
                campaign,
                contract,
                state="RUNNING",
            )
        )
    with pytest.raises(StateInvariantError, match="only terminal cleanup"):
        repository.transition(
            campaign_id=campaign,
            entity_kind="trial",
            entity_id=lineage.trial_id,
            expected_state="READY",
            expected_revision=0,
            new_state="RUNNING",
            event_type="forbidden_post_protected_trial_start",
        )
    with pytest.raises(StateInvariantError, match="only terminal cleanup"):
        repository.transition(
            campaign_id=campaign,
            entity_kind="provider_operation",
            entity_id=provider_operation_id,
            expected_state="PENDING",
            expected_revision=0,
            new_state="RUNNING",
            event_type="forbidden_post_protected_provider_start",
        )
    provider_cleanup = repository.transition(
        campaign_id=campaign,
        entity_kind="provider_operation",
        entity_id=provider_operation_id,
        expected_state="PENDING",
        expected_revision=0,
        new_state="CANCELLED",
        event_type="post_protected_provider_cleanup",
        terminal=True,
    )
    assert provider_cleanup.terminal
    cleanup = repository.transition(
        campaign_id=campaign,
        entity_kind="attempt",
        entity_id=lineage.attempt_id,
        expected_state="RUNNING",
        expected_revision=0,
        new_state="INTERRUPTED",
        event_type="post_protected_terminal_cleanup",
        terminal=True,
    )
    assert cleanup.terminal
    assert repository.register(
        DurableRecord(
            RecordKind.PROMOTION_DECISION,
            _digest("post-protected-promotion"),
            campaign,
            contract,
            references={"prediction_id": lineage.prediction_id},
            payload={"protected_outcome": "UNKNOWN", "decision": "fallback"},
        )
    )
    with pytest.raises(StateInvariantError, match="research is permanently frozen"):
        repository.record_family_evidence(
            contract_id=contract,
            campaign_id=campaign,
            family_id=lineage.family_id,
            representation="adaptive-after-protected",
            model_family="fm",
            objective="pointwise",
            temporal_policy="none",
            fusion_member="none",
            result="improved",
        )
    assert repository.inspect(campaign_id=campaign)["campaign"][  # type: ignore[index]
        "research_frozen"
    ]
    with pytest.raises(ProtectedOutcomeTerminalError, match="terminal"):
        repository.complete_protected_query(
            evaluation_id=_digest("evaluation-a-retry"),
            reservation_id=reservation.reservation_id,
            outcome=ProtectedOutcomeKind.RESULT,
            metrics={"primary": 0.7},
        )

    second_campaign = CampaignId(_digest("campaign-b"))
    _campaign(
        repository,
        contract=contract,
        campaign=second_campaign,
        key="campaign-b",
        limit=1,
    )
    second_lineage = _lineage(repository, contract=contract, campaign=second_campaign, suffix="b")
    with pytest.raises(ProtectedBudgetExhaustedError, match="exhausted"):
        repository.reserve_protected_query(
            reservation_id=_digest("reservation-b"),
            campaign_id=second_campaign,
            contract_id=contract,
            family_id=second_lineage.family_id,
            prediction_id=second_lineage.prediction_id,
            idempotency_key="score-once-b",
            expected_campaign_revision=0,
        )


def test_rank_graph_protected_query_is_bound_to_its_exact_family(tmp_path: Path) -> None:
    repository = StateRepository.open(tmp_path / "state")
    contract = ContractId(_digest("rank-graph-contract"))
    campaign = CampaignId(_digest("rank-graph-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="rank-graph", limit=1)
    bound = _lineage(repository, contract=contract, campaign=campaign, suffix="rank-bound")
    unrelated = _lineage(repository, contract=contract, campaign=campaign, suffix="rank-unrelated")
    rank_graph_id = _digest("protected-rank-graph")
    repository.register(
        DurableRecord(
            RecordKind.RANK_GRAPH,
            rank_graph_id,
            campaign,
            contract,
            references={"family_id": bound.family_id},
            payload={"recipe": "frozen"},
        )
    )
    prediction_id = PredictionId(_digest("protected-rank-prediction"))
    repository.register(
        DurableRecord(
            RecordKind.PREDICTION,
            prediction_id,
            campaign,
            contract,
            references={"rank_graph_id": rank_graph_id, "artifact_id": bound.artifact_id},
            attributes={
                "ordered_rows_sha256": _digest("protected-rank-rows"),
                "prediction_bytes_sha256": _digest("protected-rank-bytes"),
            },
        )
    )

    with pytest.raises(StateInvariantError, match="differs from eligible FamilyId"):
        repository.reserve_protected_query(
            reservation_id=_digest("wrong-rank-reservation"),
            campaign_id=campaign,
            contract_id=contract,
            family_id=unrelated.family_id,
            prediction_id=prediction_id,
            idempotency_key="wrong-rank-family",
            expected_campaign_revision=0,
        )
    reservation = repository.reserve_protected_query(
        reservation_id=_digest("right-rank-reservation"),
        campaign_id=campaign,
        contract_id=contract,
        family_id=bound.family_id,
        prediction_id=prediction_id,
        idempotency_key="right-rank-family",
        expected_campaign_revision=0,
    )
    assert reservation.family_id == bound.family_id.value


def test_fencing_tokens_prevent_stale_or_duplicate_workers(tmp_path: Path) -> None:
    clock = _Clock()
    repository = StateRepository.open(tmp_path / "state", clock=clock)
    contract = ContractId(_digest("lease-contract"))
    campaign = CampaignId(_digest("lease-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="lease-campaign", limit=1)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="lease")

    first = repository.claim_lease(
        resource_kind="campaign",
        resource_id=campaign,
        owner_id="worker-a",
        ttl=timedelta(seconds=10),
    )
    assert (
        repository.claim_lease(
            resource_kind="campaign",
            resource_id=campaign,
            owner_id="worker-a",
            ttl=timedelta(seconds=20),
        )
        == first
    )
    with pytest.raises(LeaseConflictError, match="another worker"):
        repository.claim_lease(
            resource_kind="campaign",
            resource_id=campaign,
            owner_id="worker-b",
            ttl=timedelta(seconds=10),
        )

    clock.advance(seconds=11)
    with pytest.raises(LeaseConflictError, match="release"):
        repository.release_lease(
            resource_kind="campaign",
            resource_id=campaign,
            owner_id="worker-a",
            fence_token=first.fence_token,
            complete=True,
        )
    second = repository.claim_lease(
        resource_kind="campaign",
        resource_id=campaign,
        owner_id="worker-b",
        ttl=timedelta(seconds=10),
    )
    assert second.fence_token == first.fence_token + 1
    with pytest.raises(LeaseConflictError, match="stale"):
        repository.transition(
            campaign_id=campaign,
            entity_kind="trial",
            entity_id=lineage.trial_id,
            expected_state="READY",
            expected_revision=0,
            new_state="RUNNING",
            event_type="stale_worker_start",
            lease=first,
        )
    repository.transition(
        campaign_id=campaign,
        entity_kind="trial",
        entity_id=lineage.trial_id,
        expected_state="READY",
        expected_revision=0,
        new_state="RUNNING",
        event_type="current_worker_start",
        lease=second,
    )
    with pytest.raises(LeaseConflictError, match="renewal"):
        repository.renew_lease(
            resource_kind="campaign",
            resource_id=campaign,
            owner_id="worker-a",
            fence_token=first.fence_token,
            ttl=timedelta(seconds=10),
        )
    repository.release_lease(
        resource_kind="campaign",
        resource_id=campaign,
        owner_id="worker-b",
        fence_token=second.fence_token,
        complete=True,
    )
    with pytest.raises(LeaseConflictError, match="completed"):
        repository.claim_lease(
            resource_kind="campaign",
            resource_id=campaign,
            owner_id="worker-c",
            ttl=timedelta(seconds=10),
        )


def test_descendant_registration_requires_its_live_attempt_fence(tmp_path: Path) -> None:
    repository = StateRepository.open(tmp_path / "state")
    contract = ContractId(_digest("descendant-contract"))
    campaign = CampaignId(_digest("descendant-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="descendant", limit=1)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="descendant")
    lease = repository.claim_lease(
        resource_kind="attempt",
        resource_id=lineage.attempt_id,
        owner_id="attempt-worker",
        ttl=timedelta(seconds=30),
    )
    artifact_bytes = b"second-artifact"
    artifact_path = tmp_path / "second-artifact.bin"
    artifact_path.write_bytes(artifact_bytes)
    artifact = DurableRecord(
        RecordKind.ARTIFACT,
        _digest("second-artifact"),
        campaign,
        contract,
        references={"attempt_id": lineage.attempt_id},
        attributes={
            "kind": "checkpoint",
            "relative_path": "artifacts/second.bin",
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "size_bytes": len(artifact_bytes),
            "verified_path": artifact_path,
        },
    )

    with pytest.raises(LeaseConflictError, match="current durable write fence"):
        repository.register(artifact)
    assert repository.register(artifact, lease=lease)


def test_registration_honors_family_experiment_and_prediction_fences(tmp_path: Path) -> None:
    repository = StateRepository.open(tmp_path / "state")
    contract = ContractId(_digest("registration-scope-contract"))
    campaign = CampaignId(_digest("registration-scope-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="registration-scope", limit=1)
    lineage = _lineage(
        repository, contract=contract, campaign=campaign, suffix="registration-scope"
    )

    family_lease = repository.claim_lease(
        resource_kind="family",
        resource_id=lineage.family_id,
        owner_id="family-worker",
        ttl=timedelta(seconds=30),
        campaign_id=campaign,
    )
    experiment_id = _digest("leased-experiment")
    experiment = DurableRecord(
        RecordKind.EXPERIMENT,
        experiment_id,
        campaign,
        contract,
        references={"family_id": lineage.family_id},
    )
    with pytest.raises(LeaseConflictError):
        repository.register(experiment)
    assert repository.register(experiment, lease=family_lease)
    repository.release_lease(
        resource_kind="family",
        resource_id=family_lease.resource_id,
        owner_id=family_lease.owner_id,
        fence_token=family_lease.fence_token,
        complete=False,
    )

    experiment_lease = repository.claim_lease(
        resource_kind="experiment",
        resource_id=experiment_id,
        owner_id="experiment-worker",
        ttl=timedelta(seconds=30),
        campaign_id=campaign,
    )
    trial = DurableRecord(
        RecordKind.TRIAL,
        _digest("leased-trial"),
        campaign,
        contract,
        references={"experiment_id": experiment_id},
        state="READY",
    )
    with pytest.raises(LeaseConflictError):
        repository.register(trial)
    assert repository.register(trial, lease=experiment_lease)
    repository.release_lease(
        resource_kind="experiment",
        resource_id=experiment_lease.resource_id,
        owner_id=experiment_lease.owner_id,
        fence_token=experiment_lease.fence_token,
        complete=False,
    )

    prediction_lease = repository.claim_lease(
        resource_kind="prediction",
        resource_id=lineage.prediction_id,
        owner_id="prediction-worker",
        ttl=timedelta(seconds=30),
    )
    dependents = (
        DurableRecord(
            RecordKind.INNER_EVALUATION,
            _digest("leased-inner-evaluation"),
            campaign,
            contract,
            references={"prediction_id": lineage.prediction_id},
        ),
        DurableRecord(
            RecordKind.PROMOTION_DECISION,
            _digest("leased-promotion-decision"),
            campaign,
            contract,
            references={"prediction_id": lineage.prediction_id},
        ),
        DurableRecord(
            RecordKind.REPLAY,
            _digest("leased-replay"),
            campaign,
            contract,
            references={"prediction_id": lineage.prediction_id},
            state="PENDING",
        ),
    )
    for dependent in dependents:
        with pytest.raises(LeaseConflictError):
            repository.register(dependent)
        assert repository.register(dependent, lease=prediction_lease)


def test_campaign_and_descendant_leases_cannot_overlap(tmp_path: Path) -> None:
    repository = StateRepository.open(tmp_path / "state")
    contract = ContractId(_digest("scope-contract"))
    campaign = CampaignId(_digest("scope-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="scope", limit=1)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="scope")
    attempt = repository.claim_lease(
        resource_kind="attempt",
        resource_id=lineage.attempt_id,
        owner_id="attempt-worker",
        ttl=timedelta(seconds=30),
    )
    with pytest.raises(LeaseConflictError, match="descendant lease"):
        repository.claim_lease(
            resource_kind="campaign",
            resource_id=campaign,
            owner_id="campaign-worker",
            ttl=timedelta(seconds=30),
        )
    repository.release_lease(
        resource_kind="attempt",
        resource_id=lineage.attempt_id,
        owner_id=attempt.owner_id,
        fence_token=attempt.fence_token,
        complete=False,
    )
    campaign_lease = repository.claim_lease(
        resource_kind="campaign",
        resource_id=campaign,
        owner_id="campaign-worker",
        ttl=timedelta(seconds=30),
    )
    with pytest.raises(LeaseConflictError, match="campaign with a live lease"):
        repository.claim_lease(
            resource_kind="trial",
            resource_id=lineage.trial_id,
            owner_id="trial-worker",
            ttl=timedelta(seconds=30),
        )
    repository.transition(
        campaign_id=campaign,
        entity_kind="trial",
        entity_id=lineage.trial_id,
        expected_state="READY",
        expected_revision=0,
        new_state="RUNNING",
        event_type="campaign_worker_started_trial",
        lease=campaign_lease,
    )


def test_lease_expiry_is_checked_after_waiting_for_the_write_lock(tmp_path: Path) -> None:
    clock = _Clock()
    repository = StateRepository.open(tmp_path / "state", clock=clock)
    contract = ContractId(_digest("lock-contract"))
    campaign = CampaignId(_digest("lock-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="lock", limit=1)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="lock")
    lease = repository.claim_lease(
        resource_kind="campaign",
        resource_id=campaign,
        owner_id="blocked-worker",
        ttl=timedelta(seconds=10),
    )
    blocker = sqlite3.connect(repository.database_path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    started = threading.Event()
    errors: list[BaseException] = []

    def mutate_after_lock() -> None:
        started.set()
        try:
            repository.transition(
                campaign_id=campaign,
                entity_kind="trial",
                entity_id=lineage.trial_id,
                expected_state="READY",
                expected_revision=0,
                new_state="RUNNING",
                event_type="blocked_trial_start",
                lease=lease,
            )
        except BaseException as exc:  # pragma: no cover - assertion below checks the error
            errors.append(exc)

    thread = threading.Thread(target=mutate_after_lock)
    thread.start()
    assert started.wait(timeout=2)
    threading.Event().wait(0.05)
    clock.advance(seconds=11)
    blocker.commit()
    blocker.close()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], LeaseConflictError)
    inspected = repository.inspect(campaign_id=campaign)
    assert inspected["entities"]["trials"][0]["state"] == "READY"  # type: ignore[index]


def test_legacy_caller_supplied_finalization_fails_closed_without_mutation(
    tmp_path: Path,
) -> None:
    repository = StateRepository.open(tmp_path / "legacy-finalization-state")
    contract = ContractId(_digest("legacy-finalization-contract"))
    campaign = CampaignId(_digest("legacy-finalization-campaign"))
    _campaign(
        repository,
        contract=contract,
        campaign=campaign,
        key="legacy-finalization",
        limit=0,
    )
    before = repository.inspect(campaign_id=campaign)

    with pytest.raises(PreparedFinalizationRequiredError, match="prepare_terminal_projection"):
        repository.finalize_campaign(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            finalization=object(),
        )

    assert repository.inspect(campaign_id=campaign) == before


def test_cross_campaign_references_and_authority_projection_targets_are_rejected(
    tmp_path: Path,
) -> None:
    repository = StateRepository.open(tmp_path / "state")
    contract = ContractId(_digest("cross-contract"))
    first = CampaignId(_digest("cross-first"))
    second = CampaignId(_digest("cross-second"))
    _campaign(repository, contract=contract, campaign=first, key="cross-first", limit=1)
    _campaign(repository, contract=contract, campaign=second, key="cross-second", limit=1)
    family = FamilyId(_digest("cross-family"))
    experiment = _digest("cross-experiment")
    repository.register(
        DurableRecord(
            RecordKind.FAMILY,
            family,
            first,
            contract,
            attributes={"protected_eligible": False},
        )
    )
    repository.register(
        DurableRecord(
            RecordKind.EXPERIMENT,
            experiment,
            first,
            contract,
            references={"family_id": family},
        )
    )
    assert repository.register(
        DurableRecord(
            RecordKind.FAMILY,
            family,
            second,
            contract,
            attributes={"protected_eligible": False},
        )
    )
    with sqlite3.connect(repository.database_path) as connection:
        ledger_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM family_ledger WHERE contract_id = ? AND family_id = ?",
                (contract.value, family.value),
            ).fetchone()[0]
        )
    assert ledger_count == 1
    first_evidence = repository.record_family_evidence(
        contract_id=contract,
        campaign_id=first,
        family_id=family,
        representation="base",
        model_family="fm",
        objective="pointwise",
        temporal_policy="strict-past:none",
        fusion_member="none",
        result="no_improvement",
    )
    replayed_evidence = repository.record_family_evidence(
        contract_id=contract,
        campaign_id=second,
        family_id=family,
        representation="base",
        model_family="fm",
        objective="pointwise",
        temporal_policy="strict-past:none",
        fusion_member="none",
        result="no_improvement",
    )
    second_evidence = repository.record_family_evidence(
        contract_id=contract,
        campaign_id=second,
        family_id=family,
        representation="recency",
        model_family="lambdarank",
        objective="lambdarank",
        temporal_policy="strict-past:30d",
        fusion_member="none",
        result="improved",
    )
    assert first_evidence.created and not replayed_evidence.created
    assert first_evidence.fingerprint == replayed_evidence.fingerprint
    assert second_evidence.created and second_evidence.fingerprint != first_evidence.fingerprint

    with pytest.raises(StateInvariantError, match="outside campaign lineage"):
        repository.register(
            DurableRecord(
                RecordKind.TRIAL,
                _digest("cross-trial"),
                second,
                contract,
                references={"experiment_id": experiment},
                state="READY",
            )
        )
    with pytest.raises(StateInvariantError, match="different immutable semantics"):
        repository.register(
            DurableRecord(
                RecordKind.EXPERIMENT,
                experiment,
                second,
                contract,
                references={"family_id": family},
                payload={"campaign_salt": "forbidden"},
            )
        )
    assert repository.register(
        DurableRecord(
            RecordKind.EXPERIMENT,
            experiment,
            second,
            contract,
            references={"family_id": family},
        )
    )
    assert not repository.register(
        DurableRecord(
            RecordKind.EXPERIMENT,
            experiment,
            second,
            contract,
            references={"family_id": family},
        )
    )
    reused_trial = _digest("reused-experiment-trial")
    assert repository.register(
        DurableRecord(
            RecordKind.TRIAL,
            reused_trial,
            second,
            contract,
            references={"experiment_id": experiment},
            state="READY",
        )
    )
    with sqlite3.connect(repository.database_path) as connection:
        assert (
            int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM experiment_ledger
                    WHERE contract_id = ? AND experiment_id = ?
                    """,
                    (contract.value, experiment),
                ).fetchone()[0]
            )
            == 1
        )
        assert (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM campaign_experiments WHERE experiment_id = ?",
                    (experiment,),
                ).fetchone()[0]
            )
            == 2
        )
    first_projection = repository.inspect(campaign_id=first)
    second_projection = repository.inspect(campaign_id=second)
    assert first_projection["entities"]["experiments"][0]["experiment_id"] == experiment  # type: ignore[index]
    assert second_projection["entities"]["experiments"][0]["experiment_id"] == experiment  # type: ignore[index]
    scoped_experiment_lease = repository.claim_lease(
        resource_kind="experiment",
        resource_id=experiment,
        campaign_id=first,
        owner_id="first-campaign-experiment-worker",
        ttl=timedelta(seconds=30),
    )
    with pytest.raises(LeaseConflictError):
        repository.register(
            DurableRecord(
                RecordKind.TRIAL,
                _digest("first-campaign-blocked-trial"),
                first,
                contract,
                references={"experiment_id": experiment},
                state="READY",
            )
        )
    assert repository.register(
        DurableRecord(
            RecordKind.TRIAL,
            _digest("second-campaign-independent-trial"),
            second,
            contract,
            references={"experiment_id": experiment},
            state="READY",
        )
    )
    repository.release_lease(
        resource_kind="experiment",
        resource_id=scoped_experiment_lease.resource_id,
        owner_id=scoped_experiment_lease.owner_id,
        fence_token=scoped_experiment_lease.fence_token,
        complete=False,
    )
    other_contract = ContractId(_digest("cross-other-contract"))
    third = CampaignId(_digest("cross-third"))
    _campaign(repository, contract=other_contract, campaign=third, key="cross-third", limit=1)
    assert repository.register(
        DurableRecord(
            RecordKind.FAMILY,
            family,
            third,
            other_contract,
            attributes={"protected_eligible": True},
            payload={"new_contract_lineage": True},
        )
    )
    assert repository.register(
        DurableRecord(
            RecordKind.EXPERIMENT,
            experiment,
            third,
            other_contract,
            references={"family_id": family},
        )
    )
    assert repository.register(
        DurableRecord(
            RecordKind.TRIAL,
            _digest("same-experiment-other-contract-trial"),
            third,
            other_contract,
            references={"experiment_id": experiment},
            state="READY",
        )
    )
    scoped_family_lease = repository.claim_lease(
        resource_kind="family",
        resource_id=family,
        campaign_id=first,
        owner_id="first-contract-family-worker",
        ttl=timedelta(seconds=30),
    )
    with pytest.raises(LeaseConflictError):
        repository.register(
            DurableRecord(
                RecordKind.EXPERIMENT,
                _digest("first-campaign-leased-experiment"),
                first,
                contract,
                references={"family_id": family},
            )
        )
    assert repository.register(
        DurableRecord(
            RecordKind.EXPERIMENT,
            _digest("other-contract-independent-experiment"),
            third,
            other_contract,
            references={"family_id": family},
        )
    )
    repository.release_lease(
        resource_kind="family",
        resource_id=scoped_family_lease.resource_id,
        owner_id=scoped_family_lease.owner_id,
        fence_token=scoped_family_lease.fence_token,
        complete=False,
    )
    with sqlite3.connect(repository.database_path) as connection:
        assert (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM family_ledger WHERE family_id = ?",
                    (family.value,),
                ).fetchone()[0]
            )
            == 2
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM experiment_ledger WHERE experiment_id = ?",
            (experiment,),
        ).fetchone() == (2,)
    with pytest.raises(StateInvariantError, match="cannot replace authority"):
        repository.rebuild_projection(campaign_id=first, destination=repository.database_path)
    assert repository.inspect(campaign_id=first)["campaign"] is not None


def test_inspect_uses_one_read_snapshot_during_a_concurrent_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = StateRepository.open(tmp_path / "state")
    contract = ContractId(_digest("snapshot-contract"))
    campaign = CampaignId(_digest("snapshot-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="snapshot", limit=1)
    writer_started = threading.Event()
    writer_finished = threading.Event()
    writer_errors: list[BaseException] = []
    original_inspect = state_projections.inspect_campaign

    def writer() -> None:
        try:
            assert writer_started.wait(timeout=2)
            repository.register(
                DurableRecord(
                    RecordKind.FAMILY,
                    FamilyId(_digest("snapshot-family")),
                    campaign,
                    contract,
                    attributes={"protected_eligible": False},
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion below reports thread errors
            writer_errors.append(exc)
        finally:
            writer_finished.set()

    def paused_inspect(connection: sqlite3.Connection, *, campaign_id: str) -> dict[str, object]:
        campaign_row = connection.execute(
            "SELECT 1 FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        assert campaign_row is not None and campaign_row[0] == 1
        writer_started.set()
        assert writer_finished.wait(timeout=2)
        if writer_errors:
            raise writer_errors[0]
        return dict(original_inspect(connection, campaign_id=campaign_id))

    thread = threading.Thread(target=writer)
    thread.start()
    monkeypatch.setattr(state_projections, "inspect_campaign", paused_inspect)
    snapshot = repository.inspect(campaign_id=campaign)
    thread.join(timeout=2)
    monkeypatch.setattr(state_projections, "inspect_campaign", original_inspect)
    live = repository.inspect(campaign_id=campaign)

    assert not thread.is_alive()
    snapshot_entities = snapshot["entities"]
    live_entities = live["entities"]
    assert isinstance(snapshot_entities, dict)
    assert isinstance(live_entities, dict)
    snapshot_families = snapshot_entities["families"]
    live_families = live_entities["families"]
    assert isinstance(snapshot_families, list) and snapshot_families == []
    assert isinstance(live_families, list) and len(live_families) == 1


def _close_lineage(repository: StateRepository, *, campaign: CampaignId, lineage: _Lineage) -> None:
    repository.transition(
        campaign_id=campaign,
        entity_kind="attempt",
        entity_id=lineage.attempt_id,
        expected_state="RUNNING",
        expected_revision=0,
        new_state="SUCCEEDED",
        event_type="attempt_succeeded_before_preparation",
        terminal=True,
    )
    repository.transition(
        campaign_id=campaign,
        entity_kind="trial",
        entity_id=lineage.trial_id,
        expected_state="READY",
        expected_revision=0,
        new_state="SUCCEEDED",
        event_type="trial_succeeded_before_preparation",
        terminal=True,
    )


def _offline_terminal_preparation(
    repository: StateRepository,
    *,
    contract: ContractId,
    campaign: CampaignId,
    lineage: _Lineage,
    suffix: str,
) -> TerminalPreparation:
    resource_receipt_body = {
        "schema_version": 1,
        "contract_id": contract.value,
        "campaign_id": campaign.value,
        "prediction_id": lineage.prediction_id.value,
        "attempt_id": lineage.attempt_id,
        "campaign_kind": "OFFLINE_SCRIPTED_FIXTURE",
        "qualification_scope": "SCRIPTED_FIXTURE_ONLY",
        "declared_resource_profile": {
            "name": "state-test-scripted-fixture",
            "preferred_backend": "scripted",
            "device": "cpu",
        },
        "actual_trainer_backend": "scripted",
        "actual_trainer_device": "cpu",
        "observed_resources": {
            "wall_seconds": 0.1,
            "wall_seconds_measured": True,
            "cpu_seconds": 0.05,
            "cpu_seconds_measured": True,
            "peak_rss_bytes": 1024,
            "peak_rss_bytes_measured": True,
            "peak_disk_bytes": None,
            "peak_disk_bytes_measured": False,
            "peak_process_count": 1,
            "peak_process_count_measured": True,
            "threads": None,
            "threads_measured": False,
            "declared_thread_limit": 1,
            "device": "cpu",
        },
        "timing": {
            "started_monotonic_ns": 0,
            "ended_monotonic_ns": 100_000_000,
            "wall_seconds": 0.1,
        },
        "preferred_backend_qualified": False,
        "official_fm_qualified": False,
        "full_data_qualified": False,
    }
    resource_receipt_id = hashlib.sha256(canonical_json_bytes(resource_receipt_body)).hexdigest()
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            resource_receipt_id,
            campaign,
            contract,
            references={"attempt_id": lineage.attempt_id},
            payload=resource_receipt_body,
        )
    )
    prediction_sha = _digest(f"scripted-prediction:{suffix}")
    result_sha = _digest(f"scripted-result:{suffix}")
    receipt = ScriptedReplayReceipt(
        contract_id=contract.value,
        campaign_id=campaign.value,
        prediction_id=lineage.prediction_id.value,
        first_prediction_sha256=prediction_sha,
        replay_prediction_sha256=prediction_sha,
        first_result_sha256=result_sha,
        replay_result_sha256=result_sha,
    )
    replay_grades = [
        ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        ReplayGrade.BUNDLE_EXACT.value,
    ]
    return TerminalPreparation(
        decision_id=DecisionId(_digest(f"{suffix}:decision")),
        replay_id=_digest(f"{suffix}:replay"),
        selected_prediction_id=lineage.prediction_id,
        fallback_prediction_id=lineage.prediction_id,
        terminal_state="COMPLETED_OFFLINE_FIXTURE",
        decision_payload={"disposition": "scripted fixture retained"},
        replay_payload={
            "contract_id": contract.value,
            "campaign_id": campaign.value,
            "prediction_id": lineage.prediction_id.value,
            "replay_grades": replay_grades,
            "scripted_replay_receipt": receipt.manifest(),
        },
        bundle_claims={
            "resource_receipt_id": resource_receipt_id,
            "replay_grade": ReplayGrade.BUNDLE_EXACT.value,
            "replay_grades": replay_grades,
            "submission_disposition": "SCRIPTED_FALLBACK_RETAINED",
            "scientific_disposition": "INSUFFICIENT_VALID_EVIDENCE",
            "campaign_kind": "OFFLINE_SCRIPTED_FIXTURE",
            "qualification_scope": "SCRIPTED_FIXTURE_ONLY",
            "protected_query_count": 0,
            "exact_metrics": None,
        },
    )


def _publish_prepared_bundle(
    *,
    root: Path,
    prepared: PreparedTerminalProjection,
    snapshot_path: Path,
    event_path: Path,
    selected_prediction_id: PredictionId,
    omit_role: str | None = None,
    replay_receipt_override: dict[str, object] | None = None,
    manifest_extra: dict[str, object] | None = None,
    declared_submission_sha256: str | None = None,
    manifest_schema_version: int = 2,
    evidence_schema_version: int = 1,
    tamper_receipt_id_for_role: str | None = None,
    resource_receipts_override: bytes | None = None,
    campaign_manifest_override: Mapping[str, object] | None = None,
    scientific_decision_override: Mapping[str, object] | None = None,
    derive_production_proof: bool = False,
) -> PublishedBundleReceipt:
    evidence_roles = (
        "contract-manifest.json",
        "campaign-manifest.json",
        "campaign-state-snapshot.sqlite3",
        "event-export.jsonl",
        "selection-evidence.json",
        "scientific-decision.json",
        "submission-decision.json",
        "replay-receipt.json",
        "resource-receipts.jsonl",
        "protected-query-accounting.json",
        "provider-accounting.json",
        "failure-summary.json",
        "submission.csv",
        "report.md",
    )
    projected_entities = prepared.projection["entities"]
    assert isinstance(projected_entities, dict)
    projected_replays = projected_entities["replays"]
    assert isinstance(projected_replays, list) and len(projected_replays) == 1
    projected_replay = projected_replays[0]
    assert isinstance(projected_replay, dict)
    projected_replay_payload = projected_replay["payload"]
    assert isinstance(projected_replay_payload, dict)
    prepared_receipt = projected_replay_payload.get(
        "scripted_replay_receipt", projected_replay_payload
    )
    assert isinstance(prepared_receipt, dict)
    replay_receipt = replay_receipt_override or prepared_receipt
    projected_resources = projected_entities["resource_receipts"]
    assert isinstance(projected_resources, list) and len(projected_resources) == 1
    projected_resource = projected_resources[0]
    assert isinstance(projected_resource, dict)
    resource_payload = projected_resource["payload"]
    resource_receipt_id = projected_resource["receipt_id"]
    assert isinstance(resource_payload, dict) and isinstance(resource_receipt_id, str)
    payloads: dict[str, bytes] = {
        "contract-manifest.json": canonical_json_bytes({"contract": prepared.contract_id}) + b"\n",
        "campaign-manifest.json": canonical_json_bytes(
            campaign_manifest_override or {"campaign": prepared.campaign_id}
        )
        + b"\n",
        "campaign-state-snapshot.sqlite3": snapshot_path.read_bytes(),
        "event-export.jsonl": event_path.read_bytes(),
        "selection-evidence.json": b"{}\n",
        "scientific-decision.json": canonical_json_bytes(scientific_decision_override or {})
        + b"\n",
        "submission-decision.json": b"{}\n",
        "replay-receipt.json": canonical_json_bytes(replay_receipt) + b"\n",
        "resource-receipts.jsonl": resource_receipts_override
        or canonical_json_bytes(resource_payload | {"receipt_id": resource_receipt_id}) + b"\n",
        "protected-query-accounting.json": b"{}\n",
        "provider-accounting.json": b"{}\n",
        "failure-summary.json": b"{}\n",
        "submission.csv": b"row_id,prediction\n0,0.0\n",
        "report.md": b"# Prepared bundle\n",
    }
    if derive_production_proof:
        if (
            any(
                value is not None
                for value in (
                    omit_role,
                    manifest_extra,
                    declared_submission_sha256,
                    tamper_receipt_id_for_role,
                )
            )
            or manifest_schema_version != 2
            or evidence_schema_version != 1
        ):
            raise AssertionError("typed production proof requires the canonical bundle layout")
        source_root = root.parent / f".{root.name}-frozen-sources"
        source_root.mkdir(parents=True)
        receipts: list[FrozenFileReceipt] = []
        for role in REQUIRED_EVIDENCE_ROLES:
            source = source_root / role.value
            source.write_bytes(payloads[role.value])
            receipts.append(FrozenFileReceipt.capture(role, source))
        result = BundleFinalizer().finalize(
            BundleFinalizationRequest(
                destination=root,
                contract_id=ContractId(prepared.contract_id),
                campaign_id=CampaignId(prepared.campaign_id),
                selected_prediction_id=selected_prediction_id,
                terminal_projection=TerminalProjectionBinding(
                    preparation_id=prepared.preparation_id,
                    projection_sha256=prepared.projection_sha256,
                    campaign_revision=prepared.source.campaign_revision,
                    last_event_seq=prepared.source.last_event_seq,
                ),
                receipts=tuple(receipts),
            )
        )
        return PublishedBundleReceipt(
            root=result.root,
            bundle_id=result.bundle_id,
            manifest_sha256=result.manifest_sha256,
            inventory_sha256=result.inventory_sha256,
            submission_sha256=result.submission_sha256,
            file_count=result.file_count,
            total_size_bytes=result.total_size_bytes,
            regeneration_evidence=result.regeneration_evidence,
            bundle_exact_receipt=result.replay_grade,
        )

    root.mkdir(parents=True)
    evidence = []
    for role_name in evidence_roles:
        if role_name == omit_role:
            continue
        path = root / role_name
        path.write_bytes(payloads[role_name])
        sha256 = hashlib.sha256(payloads[role_name]).hexdigest()
        receipt_body = {
            "schema_version": evidence_schema_version,
            "role": role_name,
            "sha256": sha256,
            "size_bytes": len(payloads[role_name]),
        }
        receipt_id = hashlib.sha256(
            b"kuairand-frozen-bundle-file-v1\0" + canonical_json_bytes(receipt_body)
        ).hexdigest()
        if role_name == tamper_receipt_id_for_role:
            receipt_id = _digest(f"tampered-receipt:{role_name}")
        evidence.append(
            {
                **receipt_body,
                "receipt_id": receipt_id,
            }
        )
    replay_sha = hashlib.sha256(payloads["replay-receipt.json"]).hexdigest()
    actual_submission_sha = hashlib.sha256(payloads["submission.csv"]).hexdigest()
    submission_sha = declared_submission_sha256 or actual_submission_sha
    manifest = {
        "schema_version": manifest_schema_version,
        "contract_id": prepared.contract_id,
        "campaign_id": prepared.campaign_id,
        "selected_prediction_id": selected_prediction_id.value,
        "terminal_projection": {
            "schema_version": 1,
            "preparation_id": prepared.preparation_id,
            "projection_sha256": prepared.projection_sha256,
            "source_revision": {
                "campaign_revision": prepared.source.campaign_revision,
                "last_event_seq": prepared.source.last_event_seq,
            },
            "redaction_policy_version": 1,
        },
        "identity": {
            "algorithm": "sha256",
            "definition": (
                "domain BundleId over selected prediction, replay output, submission, and manifest"
            ),
        },
        "submission_sha256": submission_sha,
        "replay_receipt_sha256": replay_sha,
        "required_paths": [*evidence_roles, "bundle-manifest.json", "bundle.sha256"],
        "evidence": evidence,
    }
    if manifest_extra is not None:
        manifest.update(manifest_extra)
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    manifest_path = root / "bundle-manifest.json"
    manifest_path.write_bytes(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    bundle_id = BundleId.derive(
        selected_prediction_id=selected_prediction_id,
        replay_output_sha256={"replay-receipt.json": replay_sha},
        submission_sha256=submission_sha,
        manifest_sha256=manifest_sha,
    ).value
    (root / "bundle.sha256").write_text(f"{bundle_id}\n", encoding="ascii")
    entries: list[dict[str, object]] = []
    total_size = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        payload = path.read_bytes()
        total_size += len(payload)
        entries.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        path.chmod(0o444)
    root.chmod(0o555)
    return PublishedBundleReceipt(
        root=root,
        bundle_id=bundle_id,
        manifest_sha256=manifest_sha,
        inventory_sha256=hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
        submission_sha256=submission_sha,
        file_count=len(entries),
        total_size_bytes=total_size,
    )


def _production_clean_replay(
    *,
    contract: ContractId,
    prediction: PredictionId,
    prediction_sha256: str,
    submission_sha256: str,
    suffix: str,
) -> tuple[CleanReplayEvidence, ReplayGradeReceipt, ReplayGradeReceipt]:
    validation_prediction_digest = _digest(f"{suffix}:validation-predictions")
    evidence = CleanReplayEvidence(
        candidate_id=f"official-fm-seed-4-{suffix}",
        identity=FrozenReplayIdentity(
            source_sha256=_digest(f"{suffix}:source"),
            config_sha256=_digest(f"{suffix}:config"),
            features_sha256=_digest(f"{suffix}:features"),
            checkpoint_sha256=_digest(f"{suffix}:checkpoint"),
            validation_prediction_artifact_sha256=_digest(
                f"{suffix}:validation-prediction-artifact"
            ),
            validation_prediction_digest=validation_prediction_digest,
            data_sha256=_digest(f"{suffix}:data"),
            environment_sha256=_digest(f"{suffix}:environment"),
        ),
        equality=ReplayEquality.EXACT,
        absolute_tolerance=0.0,
        training_replay="checkpoint_replay",
        validation=ValidationReplayEvidence(
            row_count=2,
            reference_prediction_digest=validation_prediction_digest,
            replay_prediction_digest=validation_prediction_digest,
            replay_prediction_file_sha256=_digest(f"{suffix}:validation-prediction-file"),
            exact_prediction_bytes=True,
            maximum_absolute_difference=0.0,
            top5_order_identical=True,
            protected_metrics_identical=True,
            metrics={"GAUC": 0.6, "nDCG@5": 0.4, "primary": 0.5},
            public_submission_sha256=_digest(f"{suffix}:validation-submission"),
            public_submission_prediction_digest=validation_prediction_digest,
            csv_round_trip_identity=True,
            csv_within_user_order_preserved=True,
            csv_top5_preserved=True,
            csv_protected_metrics_preserved=True,
        ),
        final=FinalReplayEvidence(
            row_count=1,
            prediction_digest=prediction_sha256,
            prediction_file_sha256=_digest(f"{suffix}:final-prediction-file"),
            submission_sha256=submission_sha256,
            submission_prediction_digest=prediction_sha256,
            finite_scores=True,
            csv_round_trip_identity=True,
        ),
        validation_capability_digest=_digest(f"{suffix}:validation-capability"),
        final_capability_digest=_digest(f"{suffix}:final-capability"),
    )
    receipts = derive_clean_replay_grade_receipts(
        contract_id=contract,
        prediction_id=prediction,
        evidence=evidence,
    )
    scoring = tuple(receipt for receipt in receipts if receipt.grade is ReplayGrade.SCORING_EXACT)
    same_backend = tuple(
        receipt for receipt in receipts if receipt.grade is ReplayGrade.EXPERIMENT_SAME_BACKEND
    )
    assert len(scoring) == len(same_backend) == 1
    return evidence, scoring[0], same_backend[0]


def _production_organizer_check(
    *, submission_sha256: str, submission_size_bytes: int, starter_manifest_sha256: str
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for relative_path in sorted(REQUIRED_DATA_FILENAMES):
        is_video = relative_path == "video_features_basic_pure.csv"
        files.append(
            {
                "relative_path": relative_path,
                "sha256": _digest(f"masked:{relative_path}"),
                "size_bytes": 1,
                "data_rows": None if is_video else 1,
                "final_rows_masked": (
                    None if is_video else int(relative_path == "log_standard_4_22_to_5_08_pure.csv")
                ),
            }
        )
    digest_body = {
        "schema_version": 1,
        "files": files,
        "registered_outcome_fields": list(OUTCOME_FIELDS),
        "final_rows_masked": 1,
        "final_outcome_cells_replaced": len(OUTCOME_FIELDS),
    }
    stdout = "submission check passed\n"
    stderr = ""
    return {
        "schema_version": 1,
        "checker": "hash-pinned organizer submit.py",
        "mode": "check_only",
        "split": "test",
        "starter_manifest_sha256": starter_manifest_sha256,
        "submission": {"sha256": submission_sha256, "size_bytes": submission_size_bytes},
        "masked_data_view": {
            "schema_version": 1,
            "files": files,
            "final_outcome_isolation": {
                "registered_fields": list(OUTCOME_FIELDS),
                "final_rows_masked": 1,
                "final_outcome_cells_replaced": len(OUTCOME_FIELDS),
                "outcome_cells_sliced": 0,
                "outcome_cells_decoded": 0,
                "outcome_cells_converted": 0,
                "outcome_cells_validated": 0,
                "outcome_cells_logged": 0,
                "outcome_cells_hashed": 0,
                "outcome_cells_scored": 0,
            },
            "digest": hashlib.sha256(canonical_json_bytes(digest_body)).hexdigest(),
        },
        "command": [
            "python",
            "-B",
            "submit.py",
            "submission.csv",
            "--data_dir",
            "<private-masked-data-view>",
            "--split",
            "test",
            "--check",
        ],
        "returncode": 0,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
    }


def _production_terminal_preparation(
    repository: StateRepository,
    *,
    contract: ContractId,
    campaign: CampaignId,
    lineage: _Lineage,
    suffix: str,
    qualification_manifest_digest: str,
    qualification_fallback_digest: str,
    performance_profile_digest: str,
    resource_profile: Mapping[str, object],
) -> TerminalPreparation:
    submission = b"row_id,prediction\n0,0.0\n"
    submission_sha256 = hashlib.sha256(submission).hexdigest()
    prediction_sha256 = _digest(f"prediction-bytes-{suffix}")
    clean_replay, scoring_receipt, same_backend_receipt = _production_clean_replay(
        contract=contract,
        prediction=lineage.prediction_id,
        prediction_sha256=prediction_sha256,
        submission_sha256=submission_sha256,
        suffix=suffix,
    )
    clean_replay_manifest = clean_replay.manifest()
    organizer_check = _production_organizer_check(
        submission_sha256=submission_sha256,
        submission_size_bytes=len(submission),
        starter_manifest_sha256=_digest(f"{suffix}:starter-manifest"),
    )
    resource_body: dict[str, object] = {
        "schema_version": 1,
        "contract_id": contract.value,
        "campaign_id": campaign.value,
        "prediction_id": lineage.prediction_id.value,
        "campaign_kind": "PRODUCTION_FULL_DATA",
        "qualification_scope": "FULL_DATA_CPU",
        "qualification_manifest_digest": qualification_manifest_digest,
        "qualification_fallback_digest": qualification_fallback_digest,
        "performance_profile_digest": performance_profile_digest,
        "declared_resource_profile": dict(resource_profile),
        "actual_trainer_backend": "organizer-numpy-fm",
        "actual_trainer_device": "cpu",
        "qualified_training_resources": {
            "wall_seconds": 12.5,
            "cpu_seconds": 12.0,
            "peak_rss_bytes": 1_900_000_000,
            "disk_bytes": 4_000_000,
            "device": "cpu",
        },
        "controller_resources": {
            "wall_seconds": 1.5,
            "cpu_seconds": 1.0,
            "peak_rss_bytes": 2_000_000_000,
            "disk_bytes": 5_000_000,
            "device": "cpu",
        },
        "controller_resource_scope": "PREPUBLICATION_SELF_EXCLUDING",
        "preferred_backend_qualified": False,
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
    }
    resource_receipt_id = hashlib.sha256(canonical_json_bytes(resource_body)).hexdigest()
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            resource_receipt_id,
            campaign,
            contract,
            payload=resource_body,
        )
    )
    replay_payload = {
        "schema_version": 3,
        "contract_id": contract.value,
        "campaign_id": campaign.value,
        "prediction_id": lineage.prediction_id.value,
        "qualification_manifest_digest": qualification_manifest_digest,
        "qualification_fallback_digest": qualification_fallback_digest,
        "original_prediction_sha256": prediction_sha256,
        "replay_prediction_sha256": prediction_sha256,
        "row_alignment_sha256": _digest(f"rows-{suffix}"),
        "submission_sha256": submission_sha256,
        "organizer_check_sha256": canonical_json_sha256(organizer_check),
        "organizer_check": organizer_check,
        "clean_replay_evidence_sha256": canonical_json_sha256(clean_replay_manifest),
        "clean_replay_evidence": clean_replay_manifest,
        "scoring_exact_receipt": scoring_receipt.manifest(),
        "same_backend_receipt": same_backend_receipt.manifest(),
        "achieved_replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        ],
        "required_terminal_replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
            ReplayGrade.BUNDLE_EXACT.value,
        ],
        "qualification_scope": "FULL_DATA_CPU",
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
    }
    return TerminalPreparation(
        decision_id=DecisionId(_digest(f"{suffix}:production-decision")),
        replay_id=_digest(f"{suffix}:production-replay"),
        selected_prediction_id=lineage.prediction_id,
        fallback_prediction_id=lineage.prediction_id,
        terminal_state="COMPLETED",
        decision_payload={"disposition": "qualified fallback retained"},
        replay_payload=replay_payload,
        scoring_exact_receipt=scoring_receipt,
        same_backend_receipt=same_backend_receipt,
        bundle_claims={
            "schema_version": 2,
            "resource_receipt_id": resource_receipt_id,
            "prepublication_replay_grades": [
                ReplayGrade.SCORING_EXACT.value,
                ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
            ],
            "required_replay_grades": [
                ReplayGrade.SCORING_EXACT.value,
                ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
                ReplayGrade.BUNDLE_EXACT.value,
            ],
            "bundle_exact_status": "PENDING_PUBLICATION_PROOF",
            "submission_disposition": "FALLBACK_RETAINED",
            "scientific_disposition": "INSUFFICIENT_VALID_EVIDENCE",
            "campaign_kind": "PRODUCTION_FULL_DATA",
            "qualification_scope": "FULL_DATA_CPU",
            "protected_query_count": 0,
            "provider_operation_count": 0,
            "exact_metrics": None,
            "official_fm_qualified": True,
            "full_data_qualified": True,
            "final_period_outcomes_accessed": False,
            "qualification_manifest_digest": qualification_manifest_digest,
        },
    )


def _production_terminal_fixture(
    tmp_path: Path, *, suffix: str
) -> tuple[StateRepository, ContractId, CampaignId, _Lineage, TerminalPreparation]:
    repository = StateRepository.open(tmp_path / f"production-{suffix}-state")
    contract = ContractId(_digest(f"production-{suffix}-contract"))
    campaign = CampaignId(_digest(f"production-{suffix}-campaign"))
    qualification_manifest_digest = _digest(f"production-{suffix}-qualification")
    qualification_fallback_digest = _digest(f"production-{suffix}-fallback")
    performance_profile_digest = _digest(f"production-{suffix}-performance")
    resource_profile = {
        "name": "competition-cpu",
        "preferred_backend": "lightgbm-cpu",
        "device": "cpu",
        "wall_clock_seconds": 300,
        "process_tree_rss_hard_cap_mb": 4096,
        "candidate_disk_hard_cap_mb": 64,
    }
    _campaign(
        repository,
        contract=contract,
        campaign=campaign,
        key=f"production-{suffix}",
        limit=0,
        config={
            "campaign": campaign.value,
            "resource_profile": resource_profile,
            "qualification_manifest_digest": qualification_manifest_digest,
            "performance_profile_digest": performance_profile_digest,
        },
    )
    lineage = _lineage(
        repository,
        contract=contract,
        campaign=campaign,
        suffix=suffix,
        prediction_payload={
            "qualification_manifest_digest": qualification_manifest_digest,
            "qualification_fallback_digest": qualification_fallback_digest,
            "final_period_outcomes_accessed": False,
        },
    )
    _close_lineage(repository, campaign=campaign, lineage=lineage)
    preparation = _production_terminal_preparation(
        repository,
        contract=contract,
        campaign=campaign,
        lineage=lineage,
        suffix=suffix,
        qualification_manifest_digest=qualification_manifest_digest,
        qualification_fallback_digest=qualification_fallback_digest,
        performance_profile_digest=performance_profile_digest,
        resource_profile=resource_profile,
    )
    return repository, contract, campaign, lineage, preparation


def _production_postpublication_receipt(
    *,
    contract: ContractId,
    campaign: CampaignId,
    lineage: _Lineage,
    preparation: TerminalPreparation,
    suffix: str,
    measurements: Mapping[str, object] | None = None,
) -> PostpublicationResourceReceipt:
    return PostpublicationResourceReceipt.from_measurement(
        contract_id=contract,
        campaign_id=campaign,
        prediction_id=lineage.prediction_id,
        prepublication_resource_receipt_id=cast(
            str, preparation.bundle_claims["resource_receipt_id"]
        ),
        performance_profile_digest=_digest(f"production-{suffix}-performance"),
        measurements=measurements
        or {
            "wall_seconds": 2.5,
            "cpu_seconds": 2.0,
            "peak_rss_bytes": 2_100_000_000,
            "disk_bytes": 6_000_000,
            "device": "cpu",
        },
    )


def test_production_fallback_terminal_requires_flat_exact_evidence_and_sealed_bundle(
    tmp_path: Path,
) -> None:
    repository, contract, campaign, lineage, preparation = _production_terminal_fixture(
        tmp_path, suffix="production-success"
    )
    prepared = repository.prepare_terminal_projection(
        campaign_id=campaign,
        contract_id=contract,
        expected_state="READY",
        expected_revision=0,
        preparation=preparation,
    )
    projected_entities = cast(dict[str, object], prepared.projection["entities"])
    projected_replays = cast(list[object], projected_entities["replays"])
    projected_replay = cast(dict[str, object], projected_replays[0])
    assert projected_replay["payload"] == preparation.replay_payload
    artifacts = repository.materialize_prepared_terminal_projection(
        preparation_id=prepared.preparation_id,
        snapshot_destination=tmp_path / "production-evidence" / "campaign-state-snapshot.sqlite3",
        event_export_destination=tmp_path / "production-evidence" / "event-export.jsonl",
    )
    replay_payload = cast(dict[str, object], preparation.replay_payload)
    organizer_check = cast(dict[str, object], replay_payload["organizer_check"])
    campaign_manifest = {
        "contract_id": contract.value,
        "campaign_id": campaign.value,
        "production_admission": {
            "starter_manifest_sha256": organizer_check["starter_manifest_sha256"],
            "data": {"final_rows": 1},
        },
    }
    scientific_decision = {
        "organizer_check": organizer_check,
        "exact_replay_evidence": {
            "clean_replay_evidence_sha256": replay_payload["clean_replay_evidence_sha256"],
            "clean_replay_evidence": replay_payload["clean_replay_evidence"],
            "scoring_exact_receipt": replay_payload["scoring_exact_receipt"],
            "same_backend_receipt": replay_payload["same_backend_receipt"],
        },
    }
    publication = _publish_prepared_bundle(
        root=tmp_path / "production-published-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        campaign_manifest_override=campaign_manifest,
        scientific_decision_override=scientific_decision,
        derive_production_proof=True,
    )
    publication = replace(
        publication,
        postpublication_resource_receipt=_production_postpublication_receipt(
            contract=contract,
            campaign=campaign,
            lineage=lineage,
            preparation=preparation,
            suffix="production-success",
        ),
    )
    unbound = _publish_prepared_bundle(
        root=tmp_path / "production-unbound-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        replay_receipt_override={"forged": True},
        campaign_manifest_override=campaign_manifest,
        scientific_decision_override=scientific_decision,
        derive_production_proof=True,
    )
    unbound = replace(
        unbound,
        postpublication_resource_receipt=publication.postpublication_resource_receipt,
    )
    with pytest.raises(PublishedBundleVerificationError, match="prepared replay payload"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=unbound,
        )
    with pytest.raises(StateInvariantError, match="typed evidence is required"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=replace(
                publication,
                regeneration_evidence=None,
                bundle_exact_receipt=None,
            ),
        )
    with pytest.raises(StateInvariantError, match="proof differs from publication"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=replace(
                publication,
                regeneration_evidence=unbound.regeneration_evidence,
                bundle_exact_receipt=unbound.bundle_exact_receipt,
            ),
        )
    with pytest.raises(StateInvariantError, match="typed postpublication resource receipt"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=replace(publication, postpublication_resource_receipt=None),
        )
    over_cap = _production_postpublication_receipt(
        contract=contract,
        campaign=campaign,
        lineage=lineage,
        preparation=preparation,
        suffix="production-success",
        measurements={
            "wall_seconds": 301.0,
            "cpu_seconds": 2.0,
            "peak_rss_bytes": 2_100_000_000,
            "disk_bytes": 6_000_000,
            "device": "cpu",
        },
    )
    with pytest.raises(StateInvariantError, match="strongest authority-bound"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=replace(publication, postpublication_resource_receipt=over_cap),
        )
    transition = repository.finalize_prepared_campaign(
        preparation_id=prepared.preparation_id,
        publication=publication,
    )
    replayed_transition = repository.finalize_prepared_campaign(
        preparation_id=prepared.preparation_id,
        publication=publication,
    )
    assert replayed_transition == transition
    assert transition.terminal and transition.new_state == "COMPLETED"
    projected_campaign = repository.inspect(campaign_id=campaign)["campaign"]
    assert isinstance(projected_campaign, dict)
    assert projected_campaign["selected_prediction_id"] == lineage.prediction_id.value
    assert projected_campaign["fallback_prediction_id"] == lineage.prediction_id.value
    finalized_entities = cast(
        dict[str, object], repository.inspect(campaign_id=campaign)["entities"]
    )
    bundles = cast(list[dict[str, object]], finalized_entities["bundles"])
    bundle_payload = cast(dict[str, object], bundles[0]["payload"])
    publication_proof = cast(dict[str, object], bundle_payload["publication_proof"])
    assert (
        publication_proof["bundle_exact_receipt"]
        == cast(ReplayGradeReceipt, publication.bundle_exact_receipt).manifest()
    )
    grade_report = cast(dict[str, object], publication_proof["replay_grade_report"])
    assert grade_report["achieved_grades"] == [
        ReplayGrade.BUNDLE_EXACT.value,
        ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        ReplayGrade.SCORING_EXACT.value,
    ]
    assert (
        bundle_payload["postpublication_resource_receipt"]
        == cast(
            PostpublicationResourceReceipt, publication.postpublication_resource_receipt
        ).manifest()
    )


def test_production_fallback_terminal_rejects_forged_replay_claims_and_resources(
    tmp_path: Path,
) -> None:
    repository, contract, campaign, _lineage_record, preparation = _production_terminal_fixture(
        tmp_path, suffix="production-forgery"
    )
    mismatched_replay = dict(preparation.replay_payload)
    mismatched_replay["replay_prediction_sha256"] = _digest("different-replay-bytes")
    with pytest.raises(StateInvariantError, match="prediction bytes are not exact"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, replay_payload=mismatched_replay),
        )
    forged_scoring_replay = dict(preparation.replay_payload)
    forged_scoring_receipt = dict(
        cast(Mapping[str, object], forged_scoring_replay["scoring_exact_receipt"])
    )
    forged_scoring_receipt["receipt_id"] = _digest("forged-scoring-receipt")
    forged_scoring_replay["scoring_exact_receipt"] = forged_scoring_receipt
    with pytest.raises(StateInvariantError, match="PRODUCTION_SCORING_RECEIPT_INVALID"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, replay_payload=forged_scoring_replay),
        )
    with pytest.raises(StateInvariantError, match="typed same-backend receipt"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, same_backend_receipt=None),
        )
    forged_same_backend_replay = dict(preparation.replay_payload)
    forged_same_backend_receipt = dict(
        cast(Mapping[str, object], forged_same_backend_replay["same_backend_receipt"])
    )
    forged_same_backend_receipt["receipt_id"] = _digest("forged-same-backend-receipt")
    forged_same_backend_replay["same_backend_receipt"] = forged_same_backend_receipt
    with pytest.raises(StateInvariantError, match="PRODUCTION_SAME_BACKEND_RECEIPT_INVALID"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, replay_payload=forged_same_backend_replay),
        )
    forged_prediction_replay = dict(preparation.replay_payload)
    forged_prediction_replay["original_prediction_sha256"] = _digest("forged-original-prediction")
    forged_prediction_replay["replay_prediction_sha256"] = _digest("forged-original-prediction")
    with pytest.raises(StateInvariantError, match="final replay is not target-free exact evidence"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, replay_payload=forged_prediction_replay),
        )
    with pytest.raises(StateInvariantError, match="must select fallback"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(
                preparation,
                fallback_prediction_id=PredictionId(_digest("different-production-fallback")),
            ),
        )
    leaked_replay = dict(preparation.replay_payload)
    leaked_replay["final_period_outcomes_accessed"] = True
    with pytest.raises(StateInvariantError, match="qualification or outcome evidence is unsafe"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, replay_payload=leaked_replay),
        )
    provider_claims = dict(preparation.bundle_claims)
    provider_claims["provider_operation_count"] = 1
    with pytest.raises(StateInvariantError, match="zero protected/provider use"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, bundle_claims=provider_claims),
        )
    original_resource_id = cast(str, preparation.bundle_claims["resource_receipt_id"])
    with sqlite3.connect(repository.database_path) as connection:
        resource_payload = json.loads(
            cast(
                str,
                connection.execute(
                    "SELECT payload_json FROM resource_receipts WHERE receipt_id = ?",
                    (original_resource_id,),
                ).fetchone()[0],
            )
        )
    preferred_resource = json.loads(canonical_json_bytes(resource_payload))
    preferred_resource["preferred_backend_qualified"] = True
    preferred_resource_id = hashlib.sha256(canonical_json_bytes(preferred_resource)).hexdigest()
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            preferred_resource_id,
            campaign,
            contract,
            payload=preferred_resource,
        )
    )
    preferred_claims = dict(preparation.bundle_claims)
    preferred_claims["resource_receipt_id"] = preferred_resource_id
    with pytest.raises(StateInvariantError, match="qualification evidence is unsafe"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, bundle_claims=preferred_claims),
        )
    scoped_resource = json.loads(canonical_json_bytes(resource_payload))
    scoped_resource["controller_resource_scope"] = "FULL_PROCESS_TREE"
    scoped_resource_id = hashlib.sha256(canonical_json_bytes(scoped_resource)).hexdigest()
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            scoped_resource_id,
            campaign,
            contract,
            payload=scoped_resource,
        )
    )
    scoped_claims = dict(preparation.bundle_claims)
    scoped_claims["resource_receipt_id"] = scoped_resource_id
    with pytest.raises(StateInvariantError, match="qualification evidence is unsafe"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, bundle_claims=scoped_claims),
        )
    resource_payload["controller_resources"]["disk_bytes"] = -1
    forged_resource_id = hashlib.sha256(canonical_json_bytes(resource_payload)).hexdigest()
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            forged_resource_id,
            campaign,
            contract,
            payload=resource_payload,
        )
    )
    forged_resource_claims = dict(preparation.bundle_claims)
    forged_resource_claims["resource_receipt_id"] = forged_resource_id
    with pytest.raises(StateInvariantError, match="disk_bytes is invalid"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, bundle_claims=forged_resource_claims),
        )
    wrong_identity = _digest("wrong-production-resource-identity")
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            wrong_identity,
            campaign,
            contract,
            payload={
                **resource_payload,
                "controller_resources": {
                    **resource_payload["controller_resources"],
                    "disk_bytes": 1,
                },
            },
        )
    )
    wrong_identity_claims = dict(preparation.bundle_claims)
    wrong_identity_claims["resource_receipt_id"] = wrong_identity
    with pytest.raises(StateInvariantError, match="resource receipt identity is invalid"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(preparation, bundle_claims=wrong_identity_claims),
        )


def test_production_fallback_terminal_rejects_nonzero_durable_provider_usage(
    tmp_path: Path,
) -> None:
    repository, contract, campaign, _lineage_record, preparation = _production_terminal_fixture(
        tmp_path, suffix="production-provider"
    )
    repository.register(
        DurableRecord(
            RecordKind.PROVIDER_OPERATION,
            _digest("production-provider-operation"),
            campaign,
            contract,
            state="PENDING",
        )
    )
    with pytest.raises(StateInvariantError, match="protected queries or provider operations"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=preparation,
        )


def test_terminal_preparation_requires_authenticated_replay_and_resource_evidence(
    tmp_path: Path,
) -> None:
    repository = StateRepository.open(tmp_path / "strict-replay-state")
    contract = ContractId(_digest("strict-replay-contract"))
    campaign = CampaignId(_digest("strict-replay-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="strict-replay", limit=0)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="strict-replay")
    _close_lineage(repository, campaign=campaign, lineage=lineage)
    valid = _offline_terminal_preparation(
        repository,
        contract=contract,
        campaign=campaign,
        lineage=lineage,
        suffix="strict-replay",
    )

    unsupported_payload = dict(valid.replay_payload)
    unsupported_payload["replay_grades"] = [ReplayGrade.SCORING_EXACT.value]
    with pytest.raises(StateInvariantError, match="unsupported grades"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(valid, replay_payload=unsupported_payload),
        )

    forged_grade_payload = dict(valid.replay_payload)
    forged_grade_receipt = dict(
        cast(Mapping[str, object], forged_grade_payload["scripted_replay_receipt"])
    )
    forged_grade_receipt["grade"] = ReplayGrade.SCORING_EXACT.value
    forged_grade_payload["scripted_replay_receipt"] = forged_grade_receipt
    with pytest.raises(StateInvariantError, match="receipt evidence is invalid"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(valid, replay_payload=forged_grade_payload),
        )

    forged_payload = dict(valid.replay_payload)
    forged_receipt = dict(cast(Mapping[str, object], forged_payload["scripted_replay_receipt"]))
    forged_receipt["replay_result_sha256"] = _digest("forged-replay-result")
    forged_payload["scripted_replay_receipt"] = forged_receipt
    with pytest.raises(StateInvariantError, match="receipt evidence is invalid"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(valid, replay_payload=forged_payload),
        )

    unsupported_claims = dict(valid.bundle_claims)
    unsupported_claims["replay_grade"] = ReplayGrade.SCORING_EXACT.value
    with pytest.raises(StateInvariantError, match="overstate verified evidence"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(valid, bundle_claims=unsupported_claims),
        )

    foreign_resource_claims = dict(valid.bundle_claims)
    foreign_resource_claims["resource_receipt_id"] = _digest("unregistered-resource-receipt")
    with pytest.raises(StateInvariantError, match="outside campaign/contract authority"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(valid, bundle_claims=foreign_resource_claims),
        )

    generic_resource_body = {
        "schema_version": 1,
        "attempt_id": lineage.attempt_id,
        "cpu_seconds": 1.0,
    }
    generic_resource_id = hashlib.sha256(canonical_json_bytes(generic_resource_body)).hexdigest()
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            generic_resource_id,
            campaign,
            contract,
            references={"attempt_id": lineage.attempt_id},
            payload=generic_resource_body,
        )
    )
    generic_resource_claims = dict(valid.bundle_claims)
    generic_resource_claims["resource_receipt_id"] = generic_resource_id
    with pytest.raises(StateInvariantError, match="resource receipt does not match exact schema"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=replace(valid, bundle_claims=generic_resource_claims),
        )

    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM terminal_preparations").fetchone() == (0,)


def test_prepared_terminal_projection_materializes_and_finalizes_atomically(
    tmp_path: Path,
) -> None:
    repository = StateRepository.open(tmp_path / "prepared-state")
    contract = ContractId(_digest("prepared-contract"))
    campaign = CampaignId(_digest("prepared-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="prepared", limit=0)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="prepared")
    _close_lineage(repository, campaign=campaign, lineage=lineage)
    with pytest.raises(StateInvariantError, match="typed same-backend receipt"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=TerminalPreparation(
                decision_id=DecisionId(_digest("unverified-production-decision")),
                replay_id=_digest("unverified-production-replay"),
                selected_prediction_id=lineage.prediction_id,
                fallback_prediction_id=lineage.prediction_id,
                terminal_state="COMPLETED",
                replay_payload={"grade": "PREDICTION_EXACT"},
                bundle_claims={"replay_grades": ["PREDICTION_EXACT"]},
            ),
        )
    preparation = _offline_terminal_preparation(
        repository,
        contract=contract,
        campaign=campaign,
        lineage=lineage,
        suffix="prepared",
    )
    before = repository.inspect(campaign_id=campaign)
    with pytest.raises(StateInvariantError, match="future publication self-reference"):
        repository.prepare_terminal_projection(
            campaign_id=campaign,
            contract_id=contract,
            expected_state="READY",
            expected_revision=0,
            preparation=TerminalPreparation(
                decision_id=preparation.decision_id,
                replay_id=preparation.replay_id,
                selected_prediction_id=preparation.selected_prediction_id,
                fallback_prediction_id=preparation.fallback_prediction_id,
                terminal_state=preparation.terminal_state,
                decision_payload=preparation.decision_payload,
                replay_payload=preparation.replay_payload,
                bundle_claims={"metadata": [[{"bundle_id": _digest("deep-future-bundle")}]]},
            ),
        )
    prepared = repository.prepare_terminal_projection(
        campaign_id=campaign,
        contract_id=contract,
        expected_state="READY",
        expected_revision=0,
        preparation=preparation,
    )
    replayed = repository.prepare_terminal_projection(
        campaign_id=campaign,
        contract_id=contract,
        expected_state="READY",
        expected_revision=0,
        preparation=preparation,
    )
    after_prepare = repository.inspect(campaign_id=campaign)
    assert prepared.created and not replayed.created
    assert prepared.preparation_id == replayed.preparation_id
    assert prepared.projection_sha256 == replayed.projection_sha256
    assert after_prepare["campaign"] == before["campaign"]
    projected_campaign = prepared.projection["campaign"]
    projected_entities = prepared.projection["entities"]
    assert isinstance(projected_campaign, dict) and isinstance(projected_entities, dict)
    assert projected_campaign["state"] == "COMPLETED_OFFLINE_FIXTURE"
    assert projected_campaign["terminal"] is True
    assert projected_entities["bundles"][0]["availability"] == "excluded-self-reference"

    artifacts = repository.materialize_prepared_terminal_projection(
        preparation_id=prepared.preparation_id,
        snapshot_destination=tmp_path / "evidence" / "campaign-state-snapshot.sqlite3",
        event_export_destination=tmp_path / "evidence" / "event-export.jsonl",
    )
    regenerated_artifacts = repository.materialize_prepared_terminal_projection(
        preparation_id=prepared.preparation_id,
        snapshot_destination=tmp_path / "evidence" / "campaign-state-snapshot.sqlite3",
        event_export_destination=tmp_path / "evidence" / "event-export.jsonl",
    )
    assert regenerated_artifacts == artifacts
    with sqlite3.connect(artifacts.snapshot_path) as connection:
        terminal_row = connection.execute(
            "SELECT state, revision, terminal, selected_prediction_id FROM campaigns"
        ).fetchone()
        metadata = connection.execute(
            "SELECT preparation_id, projection_sha256 FROM projection_metadata"
        ).fetchone()
    assert terminal_row == ("COMPLETED_OFFLINE_FIXTURE", 1, 1, lineage.prediction_id.value)
    assert metadata == (prepared.preparation_id, prepared.projection_sha256)
    publication = _publish_prepared_bundle(
        root=tmp_path / "published-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
    )
    incomplete_publication = _publish_prepared_bundle(
        root=tmp_path / "incomplete-published-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        omit_role="report.md",
    )
    with pytest.raises(PublishedBundleVerificationError, match="complete required layout"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=incomplete_publication,
        )
    false_submission_publication = _publish_prepared_bundle(
        root=tmp_path / "false-submission-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        declared_submission_sha256=_digest("false-submission-digest"),
    )
    with pytest.raises(PublishedBundleVerificationError, match="submission member differs"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=false_submission_publication,
        )
    unbound_replay_publication = _publish_prepared_bundle(
        root=tmp_path / "unbound-replay-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        replay_receipt_override={"different": "replay"},
    )
    with pytest.raises(PublishedBundleVerificationError, match="prepared replay payload"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=unbound_replay_publication,
        )
    forged_resource_publication = _publish_prepared_bundle(
        root=tmp_path / "forged-resource-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        resource_receipts_override=b'{"forged":true}\n',
    )
    with pytest.raises(PublishedBundleVerificationError, match="authoritative campaign record"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=forged_resource_publication,
        )
    self_referential_manifest = _publish_prepared_bundle(
        root=tmp_path / "self-referential-manifest-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        manifest_extra={"bundle_id": _digest("forbidden-manifest-bundle-id")},
    )
    with pytest.raises(PublishedBundleVerificationError, match="unexpected fields"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=self_referential_manifest,
        )
    legacy_manifest_publication = _publish_prepared_bundle(
        root=tmp_path / "legacy-manifest-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        manifest_schema_version=1,
    )
    with pytest.raises(PublishedBundleVerificationError, match="manifest schema v2"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=legacy_manifest_publication,
        )
    mislabeled_receipt_publication = _publish_prepared_bundle(
        root=tmp_path / "mislabeled-receipt-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        evidence_schema_version=2,
    )
    with pytest.raises(PublishedBundleVerificationError, match="evidence receipt differs"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=mislabeled_receipt_publication,
        )
    tampered_receipt_publication = _publish_prepared_bundle(
        root=tmp_path / "tampered-receipt-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
        tamper_receipt_id_for_role="report.md",
    )
    with pytest.raises(PublishedBundleVerificationError, match="receipt identity differs"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=tampered_receipt_publication,
        )
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER inject_prepared_publication_failure
            BEFORE INSERT ON bundle_publications
            BEGIN
                SELECT RAISE(ABORT, 'injected prepared publication failure');
            END
            """
        )
    with pytest.raises(StateInvariantError, match="authority constraint rejected mutation"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=publication,
        )
    rolled_back = repository.inspect(campaign_id=campaign)
    rolled_back_campaign = rolled_back["campaign"]
    rolled_back_entities = rolled_back["entities"]
    assert isinstance(rolled_back_campaign, dict) and not rolled_back_campaign["terminal"]
    assert isinstance(rolled_back_entities, dict)
    assert rolled_back_entities["selection_decisions"] == []
    assert rolled_back_entities["replays"] == []
    assert rolled_back_entities["bundles"] == []
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute("DROP TRIGGER inject_prepared_publication_failure")
    transition = repository.finalize_prepared_campaign(
        preparation_id=prepared.preparation_id,
        publication=publication,
    )
    replay = repository.finalize_prepared_campaign(
        preparation_id=prepared.preparation_id,
        publication=publication,
    )
    terminal = repository.inspect(campaign_id=campaign)
    assert transition == replay
    assert transition.terminal and transition.event_seq == prepared.source.last_event_seq + 3
    terminal_campaign = terminal["campaign"]
    assert (
        isinstance(terminal_campaign, dict)
        and terminal_campaign["state"] == "COMPLETED_OFFLINE_FIXTURE"
    )
    with sqlite3.connect(repository.database_path) as connection:
        publication_row = connection.execute(
            "SELECT preparation_id, bundle_id FROM bundle_publications"
        ).fetchone()
    assert publication_row == (prepared.preparation_id, publication.bundle_id)


def test_prepared_finalization_rejects_event_horizon_drift_without_partial_commit(
    tmp_path: Path,
) -> None:
    repository = StateRepository.open(tmp_path / "stale-state")
    contract = ContractId(_digest("stale-prepared-contract"))
    campaign = CampaignId(_digest("stale-prepared-campaign"))
    _campaign(repository, contract=contract, campaign=campaign, key="stale-prepared", limit=0)
    lineage = _lineage(repository, contract=contract, campaign=campaign, suffix="stale-prepared")
    _close_lineage(repository, campaign=campaign, lineage=lineage)
    preparation = _offline_terminal_preparation(
        repository,
        contract=contract,
        campaign=campaign,
        lineage=lineage,
        suffix="stale-prepared",
    )
    prepared = repository.prepare_terminal_projection(
        campaign_id=campaign,
        contract_id=contract,
        expected_state="READY",
        expected_revision=0,
        preparation=preparation,
    )
    artifacts = repository.materialize_prepared_terminal_projection(
        preparation_id=prepared.preparation_id,
        snapshot_destination=tmp_path / "stale-evidence" / "campaign-state-snapshot.sqlite3",
        event_export_destination=tmp_path / "stale-evidence" / "event-export.jsonl",
    )
    publication = _publish_prepared_bundle(
        root=tmp_path / "stale-published-bundle",
        prepared=prepared,
        snapshot_path=artifacts.snapshot_path,
        event_path=artifacts.event_export_path,
        selected_prediction_id=lineage.prediction_id,
    )
    repository.register(
        DurableRecord(
            RecordKind.RESOURCE_RECEIPT,
            _digest("post-preparation-receipt"),
            campaign,
            contract,
            payload={"late": True},
        )
    )
    with pytest.raises(PreparedSourceStaleError, match="prepared source is stale"):
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=publication,
        )
    snapshot = repository.inspect(campaign_id=campaign)
    campaign_projection = snapshot["campaign"]
    entities = snapshot["entities"]
    assert isinstance(campaign_projection, dict) and campaign_projection["state"] == "READY"
    assert isinstance(entities, dict)
    assert entities["selection_decisions"] == [] and entities["bundles"] == []
    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM bundle_publications").fetchone() == (0,)


def test_inspect_reads_uncheckpointed_wal_without_changing_durable_authority_bytes(
    tmp_path: Path,
) -> None:
    repository = StateRepository.open(tmp_path / "wal-inspection-state")
    keeper = sqlite3.connect(repository.database_path)
    try:
        keeper.execute("PRAGMA journal_mode = WAL")
        keeper.execute("PRAGMA wal_autocheckpoint = 0")
        contract = ContractId(_digest("wal-inspection-contract"))
        campaign = CampaignId(_digest("wal-inspection-campaign"))
        _campaign(
            repository,
            contract=contract,
            campaign=campaign,
            key="wal-inspection",
            limit=0,
        )
        wal_path = Path(f"{repository.database_path}-wal")
        assert wal_path.exists() and wal_path.stat().st_size > 0
        durable_paths = (repository.database_path, wal_path)
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in durable_paths
        }

        snapshot = repository.inspect(campaign_id=campaign)

        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in durable_paths}
        assert before == after
        campaign_projection = snapshot["campaign"]
        assert isinstance(campaign_projection, dict)
        assert campaign_projection["campaign_id"] == campaign.value
    finally:
        keeper.close()
