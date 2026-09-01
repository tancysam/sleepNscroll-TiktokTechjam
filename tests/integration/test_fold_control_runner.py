from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kuairand_agent.baselines.fold_control_runner import (
    FoldFMControlExecutionRequest,
    SupervisedFoldFMRunner,
)
from kuairand_agent.campaign.budgets import LaunchCategory, WorkPhase
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateJournalPolicy,
)
from kuairand_agent.campaign.clock import DeadlineObservation, DeadlineState
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.execution.artifacts import ArtifactStore
from kuairand_agent.execution.candidate_executor import LocalCandidateLimits
from kuairand_agent.execution.workspace import WorkspaceMaterializer

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"


@dataclass
class _Clock:
    def utc_now(self) -> datetime:
        return datetime(2030, 1, 1, tzinfo=UTC)

    def monotonic_ns(self) -> int:
        return 1_000_000_000

    def boot_identity(self) -> str:
        return "fold-control-integration-boot"


def _campaign(path: Path) -> tuple[CampaignStore, DeadlineObservation]:
    clock = _Clock()
    deadline = DeadlineState.start(
        clock,
        wall_clock_seconds=7_200,
        finalization_reserve_seconds=3_600,
    )
    store = CampaignStore.create(
        path,
        campaign_id="fold-control-integration",
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
    store.import_qualification_launches(
        tuple(
            {
                "launch_number": number,
                "kind": "clean_source_retrain" if number == 6 else "official_fm_training",
                "seed": 0 if number == 6 else number - 1,
                "charged": True,
            }
            for number in range(1, 7)
        ),
        manifest_digest="7" * 64,
        expected_revision=0,
    )
    return store, deadline.observe(clock)


def _inputs(prefix: str, dates: tuple[int, ...], *, start_time: int) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u-{index // 2}" for index in range(len(dates))),
        video_id=tuple(f"{prefix}-v-{index % 5}" for index in range(len(dates))),
        date=dates,
        duration_ms=tuple(float(800 + index * 101) for index in range(len(dates))),
        tab=tuple(str(index % 3) for index in range(len(dates))),
        author_id=tuple(f"a-{index % 4}" for index in range(len(dates))),
        time_ms=tuple(start_time + index for index in range(len(dates))),
    )


def test_fresh_supervised_fold_control_charges_once_and_rehydrates_without_child(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    workspaces = WorkspaceMaterializer(tmp_path / "workspaces", artifact_store=artifacts)
    controls = tmp_path / "controls"
    controls.mkdir()
    runner = SupervisedFoldFMRunner(
        artifact_store=artifacts,
        workspace_materializer=workspaces,
        control_root=controls,
        interpreter=Path(sys.executable),
        starter_dir=STARTER,
        limits=LocalCandidateLimits(
            timeout_seconds=60,
            memory_limit_bytes=2 * 1024**3,
            workspace_disk_limit_bytes=128 * 1024**2,
            output_limit_bytes=64 * 1024**2,
            temp_limit_bytes=64 * 1024**2,
            threads=2,
        ),
    )
    prefix = _inputs(
        "prefix",
        tuple(20220408 + index % 8 for index in range(16)),
        start_time=0,
    )
    query = _inputs(
        "query",
        (20220416, 20220416, 20220417, 20220417, 20220418, 20220418),
        start_time=100,
    )
    request = FoldFMControlExecutionRequest(
        execution_id="fold-a-fm-seed-2",
        fold_name="A",
        fold_token="a" * 64,
        seed=2,
        prefix_inputs=prefix,
        prefix_labels=tuple(index % 2 for index in range(len(prefix))),
        query_inputs=query,
        query_labels=(1, 0, 1, 0, 0, 1),
    )
    store, deadline = _campaign(tmp_path / "campaign.sqlite")
    try:
        journal = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=CandidateJournalPolicy(
                family="official_fm_fold_control",
                phase=WorkPhase.REQUIRED_CONFIRMATION,
                p95_runtime_seconds=60,
                cleanup_seconds=5,
                category=LaunchCategory.TEMPORAL_FOLD_CONFIRMATION,
            ),
        )
        first = runner.run(request, journal=journal)
        launch_count = len(store.launches())

        resumed_journal = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=journal.policy,
        )
        resumed = runner.run(request, journal=resumed_journal)

        assert first.execution is not None and first.execution.succeeded
        assert first.worker_pid != os.getpid()
        assert first.control.predictions.scores.tobytes() == (
            resumed.control.predictions.scores.tobytes()
        )
        assert first.control.digest == resumed.control.digest
        assert first.evidence.digest == resumed.evidence.digest
        assert resumed.execution is None
        assert resumed.resumed is True
        assert len(store.launches()) == launch_count == 7
        assert store.launches()[-1].state == "FINISHED"
        assert store.execution(request.execution_id).status == "SUCCEEDED"  # type: ignore[union-attr]
        assert not (tmp_path / "workspaces" / request.execution_id).exists()
    finally:
        store.close()
