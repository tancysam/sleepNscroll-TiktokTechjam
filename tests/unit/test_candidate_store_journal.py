from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kuairand_agent.campaign.budgets import (
    AdmissionReason,
    LaunchCategory,
    WorkPhase,
)
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateAdmissionError,
    CandidateExecutionPendingError,
    CandidateJournalError,
    CandidateJournalPolicy,
    reconstruct_budget_ledger,
)
from kuairand_agent.campaign.clock import DeadlineObservation, DeadlineState
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.execution.artifacts import ArtifactStore
from kuairand_agent.execution.candidate_executor import CandidateAction
from kuairand_agent.execution.policy import SplitRole
from kuairand_agent.execution.runner import ExecutionSpec
from kuairand_agent.execution.workspace import CandidateWorkspace


@dataclass
class _Clock:
    now: datetime = datetime(2030, 1, 1, tzinfo=UTC)
    monotonic: int = 10_000_000_000

    def utc_now(self) -> datetime:
        return self.now

    def monotonic_ns(self) -> int:
        return self.monotonic

    def boot_identity(self) -> str:
        return "test-boot"


def _convergence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "best_primary": 0.6016,
        "non_material_streak": 0,
        "completed_iterations": 0,
        "required_completion_pending": False,
    }


def _qualification() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "launch_number": number,
            "kind": "clean_source_retrain" if number == 6 else "official_fm_training",
            "seed": 0 if number == 6 else number - 1,
            "charged": True,
        }
        for number in range(1, 7)
    )


def _campaign(tmp_path: Path) -> tuple[CampaignStore, DeadlineObservation]:
    clock = _Clock()
    deadline = DeadlineState.start(
        clock,
        wall_clock_seconds=7_200,
        finalization_reserve_seconds=3_600,
    )
    store = CampaignStore.create(
        tmp_path / "campaign.sqlite",
        campaign_id="journal-campaign",
        config_digest="1" * 64,
        benchmark_digest="2" * 64,
        starter_digest="3" * 64,
        dataset_digest="4" * 64,
        environment_digest="5" * 64,
        source_digest="6" * 64,
        hard_deadline_utc=deadline.utc_deadline.isoformat(),
        initial_convergence=_convergence(),
    )
    store.import_qualification_launches(
        _qualification(),
        manifest_digest="7" * 64,
        expected_revision=0,
    )
    return store, deadline.observe(clock)


def _execution(tmp_path: Path, execution_id: str) -> tuple[ExecutionSpec, CandidateWorkspace]:
    workspace_root = tmp_path / f"workspace-{execution_id}"
    workspace_root.mkdir()
    control_root = tmp_path / "controls"
    control_root.mkdir(exist_ok=True)
    workspace = CandidateWorkspace(
        root=workspace_root,
        execution_id=execution_id,
        split_role=SplitRole.INNER_TRAIN,
        source_snapshot_sha256="a" * 64,
        source_files=(),
        input_files=(),
        output_limit_bytes=1024,
        temp_limit_bytes=1024,
        request_sha256="b" * 64,
        manifest_digest="c" * 64,
    )
    spec = ExecutionSpec(
        execution_id=execution_id,
        nonce=f"nonce-{execution_id}-0123456789",
        interpreter=Path(sys.executable),
        arguments=("-c", "pass"),
        workspace=workspace_root,
        control_dir=control_root / execution_id,
        timeout_seconds=30,
        memory_limit_bytes=1024**3,
        workspace_disk_limit_bytes=1024**2,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
        threads=1,
        source_digest="d" * 64,
        config_digest="e" * 64,
        data_digest="f" * 64,
        checkpoint_digest="0" * 64,
        python_hash_seed=11,
    )
    return spec, workspace


def _train_policy() -> CandidateJournalPolicy:
    return CandidateJournalPolicy(
        family="generated_lambdarank",
        phase=WorkPhase.RESEARCH,
        p95_runtime_seconds=60,
        cleanup_seconds=10,
        category=LaunchCategory.DIVERSE_INNER_SCREEN,
    )


def _reservation_key(action: CandidateAction, spec: ExecutionSpec) -> str:
    payload = json.dumps(
        {
            "action": action.value,
            "checkpoint_digest": spec.checkpoint_digest,
            "config_digest": spec.config_digest,
            "data_digest": spec.data_digest,
            "execution_id": spec.execution_id,
            "source_digest": spec.source_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "candidate:" + hashlib.sha256(payload).hexdigest()


def test_reconstructs_qualification_alias_and_ignores_not_started_number_gap(
    tmp_path: Path,
) -> None:
    store, deadline = _campaign(tmp_path)
    try:
        released = store.reserve_launch(
            launch_id="abandoned-before-start",
            reservation_key="test:abandoned-before-start",
            category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
            purpose="pre-spawn fault",
            expected_revision=store.snapshot().revision,
        )
        assert released.launch_number == 7
        store.transition_launch(
            released.launch_id,
            to_state="NOT_STARTED",
            expected_revision=store.snapshot().revision,
            metadata={"candidate_released": False},
        )

        ledger = reconstruct_budget_ledger(store)
        assert ledger.training_launches.value == 6
        assert ledger.used(LaunchCategory.BASELINE_QUALIFICATION_REPLAY) == 6
        assert ledger.remaining_in_category(LaunchCategory.DIVERSE_INNER_SCREEN) == 20

        artifacts = ArtifactStore(tmp_path / "artifacts")
        spec, workspace = _execution(tmp_path, "fold-b-tree")
        journal = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=_train_policy(),
        )
        journal.prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)

        launch = store.launches()[-1]
        execution = store.executions()[-1]
        assert launch.launch_number == 8
        assert launch.state == "RESERVED"
        assert launch.charged is True
        assert execution.launch_id == launch.launch_id
        assert execution.status == "STARTING"
        assert execution.source_digest == spec.source_digest
        assert execution.config_digest == spec.config_digest
        assert execution.capability_digest == workspace.manifest_digest
        assert execution.environment_digest == store.identity().environment_digest
        assert execution.data_digest == spec.data_digest
        assert execution.checkpoint_digest == spec.checkpoint_digest
        assert reconstruct_budget_ledger(store).training_launches.value == 7
    finally:
        store.close()


def test_prediction_is_durable_but_uncharged(tmp_path: Path) -> None:
    store, deadline = _campaign(tmp_path)
    try:
        artifacts = ArtifactStore(tmp_path / "artifacts")
        spec, workspace = _execution(tmp_path, "fold-b-predict")
        journal = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=CandidateJournalPolicy(
                family="generated_lambdarank",
                phase=WorkPhase.RESEARCH,
                p95_runtime_seconds=30,
                cleanup_seconds=5,
            ),
        )
        journal.prepare(action=CandidateAction.PREDICT, spec=spec, workspace=workspace)

        execution = store.executions()[-1]
        assert execution.status == "STARTING"
        assert execution.launch_id is None
        assert len(store.launches()) == 6
        assert store.snapshot().launches_used == 6
    finally:
        store.close()


def test_reconstructs_durable_category_reallocation(tmp_path: Path) -> None:
    store, _deadline = _campaign(tmp_path)
    try:
        store.record_reallocation(
            reallocation_id="blend-to-screen",
            from_category=LaunchCategory.BLEND_FUSION.value,
            to_category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
            launch_count=1,
            reason="no eligible complementary blend remained",
            expected_revision=store.snapshot().revision,
        )
        ledger = reconstruct_budget_ledger(store)
        assert ledger.effective_ceilings[LaunchCategory.BLEND_FUSION] == 2
        assert ledger.effective_ceilings[LaunchCategory.DIVERSE_INNER_SCREEN] == 21
        assert store.reallocations()[0].reallocation_id == "blend-to-screen"
    finally:
        store.close()


def test_restart_completes_exact_orphan_reservation_but_rejects_released_one(
    tmp_path: Path,
) -> None:
    store, deadline = _campaign(tmp_path)
    try:
        artifacts = ArtifactStore(tmp_path / "artifacts")
        spec, workspace = _execution(tmp_path, "orphan-reservation")
        policy = _train_policy()
        store.reserve_launch(
            launch_id=f"{spec.execution_id}-launch",
            reservation_key=_reservation_key(CandidateAction.TRAIN, spec),
            category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
            original_category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
            purpose=policy.family,
            expected_revision=store.snapshot().revision,
            seed=spec.python_hash_seed,
            metadata={
                "action": "train",
                "repair_child": False,
                "source_digest": spec.source_digest,
            },
        )
        journal = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=policy,
        )
        journal.prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)
        assert len(store.launches()) == 7
        prepared = store.execution(spec.execution_id)
        assert prepared is not None
        assert prepared.status == "STARTING"
        with pytest.raises(CandidateExecutionPendingError):
            CampaignStoreCandidateJournal(
                store=store,
                artifact_store=artifacts,
                deadline=deadline,
                policy=policy,
            ).prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)

        released_spec, released_workspace = _execution(tmp_path, "released-reservation")
        released = store.reserve_launch(
            launch_id=f"{released_spec.execution_id}-launch",
            reservation_key=_reservation_key(CandidateAction.TRAIN, released_spec),
            category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
            original_category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
            purpose=policy.family,
            expected_revision=store.snapshot().revision,
            seed=released_spec.python_hash_seed,
            metadata={
                "action": "train",
                "repair_child": False,
                "source_digest": released_spec.source_digest,
            },
        )
        store.transition_launch(
            released.launch_id,
            to_state="NOT_STARTED",
            expected_revision=store.snapshot().revision,
            metadata={"candidate_released": False},
        )
        with pytest.raises(CandidateJournalError, match="resumable RESERVED"):
            CampaignStoreCandidateJournal(
                store=store,
                artifact_store=artifacts,
                deadline=deadline,
                policy=policy,
            ).prepare(
                action=CandidateAction.TRAIN,
                spec=released_spec,
                workspace=released_workspace,
            )
        assert store.execution(released_spec.execution_id) is None
    finally:
        store.close()


def test_category_and_deadline_admission_fail_before_reservation(tmp_path: Path) -> None:
    store, deadline = _campaign(tmp_path)
    try:
        for index in range(20):
            store.reserve_launch(
                launch_id=f"screen-{index:02d}",
                reservation_key=f"screen:{index:02d}",
                category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
                purpose="fill frozen category allocation",
                expected_revision=store.snapshot().revision,
            )
        artifacts = ArtifactStore(tmp_path / "artifacts")
        spec, workspace = _execution(tmp_path, "over-category")
        journal = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=_train_policy(),
        )
        with pytest.raises(CandidateAdmissionError) as rejected:
            journal.prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)
        assert rejected.value.reason is AdmissionReason.CATEGORY_CAP
        assert len(store.launches()) == 26
        assert not any(item.execution_id == spec.execution_id for item in store.executions())

        exhausted = deadline.state.observe(
            _Clock(
                now=datetime(2030, 1, 1, tzinfo=UTC) + timedelta(seconds=3_601),
                monotonic=10_000_000_000 + 3_601_000_000_000,
            )
        )
        prediction_spec, prediction_workspace = _execution(tmp_path, "reserve-active")
        prediction = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=exhausted,
            policy=CandidateJournalPolicy(
                family="generated_lambdarank",
                phase=WorkPhase.RESEARCH,
                p95_runtime_seconds=1,
            ),
        )
        with pytest.raises(CandidateAdmissionError) as deadline_rejected:
            prediction.prepare(
                action=CandidateAction.PREDICT,
                spec=prediction_spec,
                workspace=prediction_workspace,
            )
        assert deadline_rejected.value.reason is AdmissionReason.FINALIZATION_RESERVE
        assert not any(
            item.execution_id == prediction_spec.execution_id for item in store.executions()
        )
    finally:
        store.close()
