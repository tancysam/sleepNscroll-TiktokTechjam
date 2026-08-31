from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kuairand_agent.campaign.budgets import LaunchCategory, WorkPhase
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateJournalPolicy,
)
from kuairand_agent.campaign.clock import DeadlineObservation, DeadlineState
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.execution.candidate_executor import (
    GeneratedCandidateExecutor,
    GeneratedCandidateIdentity,
    GeneratedPredictRequest,
    GeneratedTrainRequest,
    LocalCandidateLimits,
    put_numpy_capability,
)
from kuairand_agent.execution.policy import SplitRole, WorkspacePolicy
from kuairand_agent.execution.workspace import WorkspaceMaterializer

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "candidate_templates" / "lambdarank"


@dataclass
class _Clock:
    def utc_now(self) -> datetime:
        return datetime(2030, 1, 1, tzinfo=UTC)

    def monotonic_ns(self) -> int:
        return 1_000_000_000

    def boot_identity(self) -> str:
        return "integration-boot"


def _campaign(path: Path) -> tuple[CampaignStore, DeadlineObservation]:
    clock = _Clock()
    deadline = DeadlineState.start(
        clock,
        wall_clock_seconds=7_200,
        finalization_reserve_seconds=3_600,
    )
    store = CampaignStore.create(
        path,
        campaign_id="generated-integration",
        config_digest="1" * 64,
        benchmark_digest="2" * 64,
        starter_digest="3" * 64,
        dataset_digest="4" * 64,
        environment_digest="5" * 64,
        source_digest="6" * 64,
        hard_deadline_utc=deadline.utc_deadline.isoformat(),
        initial_convergence={
            "schema_version": 2,
            "best_primary": 0.6016,
            "non_material_streak": 0,
            "unmeasured_streak": 0,
            "completed_iterations": 0,
            "required_completion_pending": False,
        },
    )
    records = tuple(
        {
            "launch_number": number,
            "kind": "clean_source_retrain" if number == 6 else "official_fm_training",
            "seed": 0 if number == 6 else number - 1,
            "charged": True,
        }
        for number in range(1, 7)
    )
    store.import_qualification_launches(
        records,
        manifest_digest="7" * 64,
        expected_revision=0,
    )
    return store, deadline.observe(clock)


def test_real_generated_train_predict_persists_and_rehydrates_exact_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    materializer = WorkspaceMaterializer(
        tmp_path / "workspaces",
        artifact_store=artifacts,
        policy=WorkspacePolicy(max_source_files=8),
    )
    controls = tmp_path / "controls"
    controls.mkdir()
    executor = GeneratedCandidateExecutor(
        artifact_store=artifacts,
        workspace_materializer=materializer,
        control_root=controls,
        interpreter=Path(sys.executable),
        limits=LocalCandidateLimits(
            timeout_seconds=60,
            memory_limit_bytes=2 * 1024**3,
            workspace_disk_limit_bytes=512 * 1024**2,
            output_limit_bytes=128 * 1024**2,
            temp_limit_bytes=128 * 1024**2,
            threads=2,
        ),
    )
    source = artifacts.put_directory(TEMPLATE, kind=ArtifactKind.SOURCE)
    identity = GeneratedCandidateIdentity(
        source_snapshot=source,
        source_digest="8" * 64,
        config_digest=hashlib.sha256((TEMPLATE / "config.json").read_bytes()).hexdigest(),
    )
    steps = np.repeat(np.arange(8, dtype=np.float64), 6)
    groups = np.tile(np.asarray([20, 10, 30, 40, 50, 60], dtype=np.int64), 8)
    targets = (steps >= 4).astype(np.int8)
    features = np.column_stack((steps, np.sin(steps), groups % 7)).astype("<f8")
    feature_ref = put_numpy_capability(artifacts, features)
    target_ref = put_numpy_capability(artifacts, targets)
    group_ref = put_numpy_capability(artifacts, groups)

    store, deadline = _campaign(tmp_path / "campaign.sqlite")
    try:
        train_journal = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=CandidateJournalPolicy(
                family="generated_lambdarank",
                phase=WorkPhase.RESEARCH,
                p95_runtime_seconds=60,
                cleanup_seconds=10,
                category=LaunchCategory.DIVERSE_INNER_SCREEN,
            ),
        )
        trained = executor.train(
            GeneratedTrainRequest(
                execution_id="durable-tree-train",
                identity=identity,
                split_role=SplitRole.INNER_TRAIN,
                data_digest="9" * 64,
                split_token="fold-b-train",
                seed=7,
                features=feature_ref,
                targets=target_ref,
                user_groups=group_ref,
            ),
            journal=train_journal,
        )
        train_journal.finish(
            action=train_journal.rehydrate_terminal("durable-tree-train").action,
            result=trained.execution,
            artifacts=trained.artifacts,
        )
        train_record = store.execution("durable-tree-train")
        assert train_record is not None
        assert train_record.status == "SUCCEEDED"
        assert trained.execution.process is not None
        assert train_record.process_record_digest == trained.execution.process.digest
        assert store.launches()[-1].state == "FINISHED"
        assert store.snapshot().launches_used == 7
        restored_train = train_journal.rehydrate_terminal("durable-tree-train")
        assert restored_train.artifacts.closure_digest == trained.artifacts.closure_digest
        assert restored_train.artifacts.artifact("checkpoint") == trained.checkpoint

        predict_journal = CampaignStoreCandidateJournal(
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
        predicted = executor.predict(
            GeneratedPredictRequest(
                execution_id="durable-tree-predict",
                identity=identity,
                split_role=SplitRole.INNER_VALID,
                data_digest="a" * 64,
                split_token="fold-b-valid",
                expected_count=features.shape[0],
                features=feature_ref,
                checkpoint=trained.checkpoint,
            ),
            journal=predict_journal,
        )
        prediction_record = store.execution("durable-tree-predict")
        assert prediction_record is not None
        assert prediction_record.status == "SUCCEEDED"
        assert prediction_record.launch_id is None
        assert store.snapshot().launches_used == 7
        restored_prediction = predict_journal.rehydrate_terminal("durable-tree-predict")
        assert restored_prediction.artifacts.closure_digest == predicted.artifacts.closure_digest
        assert restored_prediction.artifacts.artifact("prediction") == predicted.prediction
        assert tuple(
            role
            for role, _ in store.artifacts_for(
                owner_type="execution", owner_id="durable-tree-predict"
            )
        ) == (
            "execution_manifest",
            "prediction",
            "prediction_result",
            "stderr",
            "stdout",
            "workspace_cleanup",
        )
    finally:
        store.close()

    with CampaignStore.open(
        tmp_path / "campaign.sqlite", campaign_id="generated-integration"
    ) as reopened:
        restarted = CampaignStoreCandidateJournal(
            store=reopened,
            artifact_store=artifacts,
            deadline=deadline,
            policy=CandidateJournalPolicy(
                family="generated_lambdarank",
                phase=WorkPhase.RESEARCH,
                p95_runtime_seconds=30,
            ),
        )
        restored = restarted.rehydrate_terminal("durable-tree-predict")
        assert restored.artifacts.artifact("prediction") == predicted.prediction
