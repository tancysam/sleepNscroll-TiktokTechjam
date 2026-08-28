from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.campaign.budgets import LaunchCategory, WorkPhase
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateExecutionTerminalError,
    CandidateJournalPolicy,
)
from kuairand_agent.campaign.clock import DeadlineState, SystemClock
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.execution.candidate_executor import (
    CandidateExecutionError,
    GeneratedCandidateExecutor,
    GeneratedCandidateIdentity,
    GeneratedTrainRequest,
    LocalCandidateLimits,
    put_numpy_capability,
)
from kuairand_agent.execution.policy import SplitRole, WorkspacePolicy
from kuairand_agent.execution.runner import ExecutionOutcome
from kuairand_agent.execution.workspace import WorkspaceMaterializer


def _store(path: Path) -> tuple[CampaignStore, object]:
    clock = SystemClock()
    deadline = DeadlineState.start(
        clock,
        wall_clock_seconds=7_200,
        finalization_reserve_seconds=3_600,
    )
    store = CampaignStore.create(
        path,
        campaign_id="signal-campaign",
        config_digest="1" * 64,
        benchmark_digest="2" * 64,
        starter_digest="3" * 64,
        dataset_digest="4" * 64,
        environment_digest="5" * 64,
        source_digest="6" * 64,
        hard_deadline_utc=deadline.utc_deadline.isoformat(),
        initial_convergence={
            "schema_version": 1,
            "best_primary": 0.6016,
            "non_material_streak": 0,
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


def _journal(
    store: CampaignStore,
    artifacts: ArtifactStore,
    deadline: object,
) -> CampaignStoreCandidateJournal:
    from kuairand_agent.campaign.clock import DeadlineObservation

    assert isinstance(deadline, DeadlineObservation)
    return CampaignStoreCandidateJournal(
        store=store,
        artifact_store=artifacts,
        deadline=deadline,
        policy=CandidateJournalPolicy(
            family="generated_signal_fixture",
            phase=WorkPhase.RESEARCH,
            p95_runtime_seconds=60.0,
            cleanup_seconds=5.0,
            category=LaunchCategory.DIVERSE_INNER_SCREEN,
        ),
    )


def test_released_signal_cancellation_is_charged_once_and_preserves_fallback(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidate.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    (source / "config.json").write_text("{}", encoding="ascii")
    (source / "README.md").write_text("signal fixture\n", encoding="utf-8")
    snapshot = artifacts.put_directory(source, kind=ArtifactKind.SOURCE)
    identity = GeneratedCandidateIdentity(
        source_snapshot=snapshot,
        source_digest=hashlib.sha256(b"generated-signal-source").hexdigest(),
        config_digest=hashlib.sha256(b"{}").hexdigest(),
    )
    workspaces = WorkspaceMaterializer(
        tmp_path / "workspaces",
        artifact_store=artifacts,
        policy=WorkspacePolicy(max_source_files=8),
    )
    controls = tmp_path / "controls"
    controls.mkdir()
    executor = GeneratedCandidateExecutor(
        artifact_store=artifacts,
        workspace_materializer=workspaces,
        control_root=controls,
        interpreter=Path(sys.executable),
        limits=LocalCandidateLimits(
            timeout_seconds=30.0,
            memory_limit_bytes=512 * 1024**2,
            workspace_disk_limit_bytes=64 * 1024**2,
            output_limit_bytes=16 * 1024**2,
            temp_limit_bytes=16 * 1024**2,
            threads=1,
        ),
    )
    features = put_numpy_capability(artifacts, np.ones((4, 2), dtype=np.float64))
    targets = put_numpy_capability(artifacts, np.asarray((0, 1, 0, 1), dtype=np.int8))
    groups = put_numpy_capability(artifacts, np.asarray((1, 1, 2, 2), dtype=np.int64))
    request = GeneratedTrainRequest(
        execution_id="signal-generated-train",
        identity=identity,
        split_role=SplitRole.INNER_TRAIN,
        data_digest="8" * 64,
        split_token="fold-b-signal",
        seed=0,
        features=features,
        targets=targets,
        user_groups=groups,
    )
    store, deadline = _store(tmp_path / "campaign.sqlite")
    cancellation = threading.Event()
    try:
        fallback = store.record_incumbent(
            incumbent_id="official-fm-fallback-seed-4",
            eligibility="qualified_replayable_fallback",
            source_digest="9" * 64,
            checkpoint_digest="a" * 64,
            artifact_closure_digest="b" * 64,
            replay_verified=True,
            is_fallback=True,
            expected_revision=store.snapshot().revision,
            reason="retain immutable official FM before generated research",
            outer_primary_mean=0.6016,
        )

        def cancel_after_release() -> None:
            release = controls / request.execution_id / "release.json"
            limit = time.monotonic() + 5.0
            while not release.is_file() and time.monotonic() < limit:
                time.sleep(0.01)
            if release.is_file():
                cancellation.set()

        trigger = threading.Thread(target=cancel_after_release)
        trigger.start()
        with pytest.raises(CandidateExecutionError) as raised:
            executor.train(
                request,
                journal=_journal(store, artifacts, deadline),
                cancel_event=cancellation,
            )
        trigger.join(timeout=2.0)

        assert not trigger.is_alive()
        assert raised.value.result is not None
        assert raised.value.result.outcome is ExecutionOutcome.CANCELLED
        assert raised.value.result.candidate_released
        assert raised.value.result.cleanup_verified
        assert store.snapshot().launches_used == 7
        assert store.launches()[-1].charged
        assert store.launches()[-1].state == "FINISHED"
        terminal = store.execution(request.execution_id)
        assert terminal is not None and terminal.status == "FAILED"
        assert store.current_incumbent() == fallback
        assert not (tmp_path / "workspaces" / request.execution_id).exists()

        cancellation.clear()
        with pytest.raises(CandidateExecutionTerminalError):
            executor.train(
                request,
                journal=_journal(store, artifacts, deadline),
                cancel_event=cancellation,
            )
        assert store.snapshot().launches_used == 7
        assert len(store.launches()) == 7
        assert store.current_incumbent() == fallback
        assert not (tmp_path / "workspaces" / request.execution_id).exists()
    finally:
        store.close()
