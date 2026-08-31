from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast, overload

import pytest

from kuairand_agent.baselines.fold_control_runner import (
    FoldFMControlExecutionRequest,
    SupervisedFoldFMError,
    SupervisedFoldFMExecutionError,
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
from kuairand_agent.execution.runner import (
    ExecutionOutcome,
    ExecutionResult,
    ExecutionSpec,
    LogEvidence,
    ProcessRecord,
    Runner,
)
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
        return "fold-control-fault-boot"


class _NeverReleasedRunner:
    def run(
        self,
        spec: ExecutionSpec,
        *,
        commit_launch: Callable[[ProcessRecord], None],
        cancel_event: object | None = None,
    ) -> ExecutionResult:
        del commit_launch, cancel_event
        spec.control_dir.mkdir(mode=0o700)
        stdout = spec.control_dir / "stdout.log"
        stderr = spec.control_dir / "stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"synthetic spawn failure\n")
        timestamp = datetime.now(UTC).isoformat(timespec="microseconds")
        return ExecutionResult(
            execution_id=spec.execution_id,
            outcome=ExecutionOutcome.SPAWN_FAILED,
            process=None,
            candidate_released=False,
            exit_code=None,
            terminating_signal=None,
            started_at_utc=timestamp,
            ended_at_utc=timestamp,
            wall_seconds=0.0,
            peak_tree_rss_bytes=0,
            peak_workspace_bytes=0,
            peak_process_count=0,
            stdout=LogEvidence(
                stdout,
                0,
                0,
                False,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            stderr=LogEvidence(
                stderr,
                24,
                24,
                False,
                "e286d9eed43233639b99439467110de721b4a12624d878206acac3164b276d07",
            ),
            cleanup_verified=True,
            device="cpu",
            threads=spec.threads,
            detail="synthetic pre-release failure",
        )


class _ExplodingLabels(Sequence[object]):
    def __len__(self) -> int:
        raise AssertionError("protected labels were inspected")

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        raise AssertionError(f"protected labels[{index}] were inspected")


def _campaign(path: Path) -> tuple[CampaignStore, DeadlineObservation]:
    clock = _Clock()
    deadline = DeadlineState.start(
        clock,
        wall_clock_seconds=7_200,
        finalization_reserve_seconds=3_600,
    )
    store = CampaignStore.create(
        path,
        campaign_id="fold-control-fault",
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


def _request(
    execution_id: str,
    *,
    query_dates: tuple[int, ...] = (
        20220416,
        20220416,
        20220417,
        20220417,
        20220418,
        20220418,
    ),
    prefix_labels: Sequence[object] | None = None,
    query_labels: Sequence[object] | None = None,
) -> FoldFMControlExecutionRequest:
    prefix = _inputs(
        "prefix",
        tuple(20220408 + index % 8 for index in range(16)),
        start_time=0,
    )
    query = _inputs("query", query_dates, start_time=100)
    return FoldFMControlExecutionRequest(
        execution_id=execution_id,
        fold_name="A",
        fold_token="a" * 64,
        seed=2,
        prefix_inputs=prefix,
        prefix_labels=(
            tuple(index % 2 for index in range(len(prefix)))
            if prefix_labels is None
            else prefix_labels
        ),
        query_inputs=query,
        query_labels=(
            tuple(index % 2 for index in range(len(query)))
            if query_labels is None
            else query_labels
        ),
    )


def _runner(
    tmp_path: Path,
    artifacts: ArtifactStore,
    *,
    timeout_seconds: float = 60,
    runner: Runner | None = None,
) -> SupervisedFoldFMRunner:
    workspaces = WorkspaceMaterializer(tmp_path / "workspaces", artifact_store=artifacts)
    controls = tmp_path / "controls"
    controls.mkdir()
    return SupervisedFoldFMRunner(
        artifact_store=artifacts,
        workspace_materializer=workspaces,
        control_root=controls,
        interpreter=Path(sys.executable),
        starter_dir=STARTER,
        limits=LocalCandidateLimits(
            timeout_seconds=timeout_seconds,
            memory_limit_bytes=2 * 1024**3,
            workspace_disk_limit_bytes=128 * 1024**2,
            output_limit_bytes=64 * 1024**2,
            temp_limit_bytes=64 * 1024**2,
            threads=2,
        ),
        runner=runner,
    )


def _journal(
    store: CampaignStore,
    artifacts: ArtifactStore,
    deadline: DeadlineObservation,
) -> CampaignStoreCandidateJournal:
    return CampaignStoreCandidateJournal(
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


def test_pre_release_failure_is_terminal_but_not_charged(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    supervised = _runner(
        tmp_path,
        artifacts,
        runner=cast(Runner, _NeverReleasedRunner()),
    )
    store, deadline = _campaign(tmp_path / "campaign.sqlite")
    try:
        with pytest.raises(SupervisedFoldFMExecutionError) as raised:
            supervised.run(
                _request("fold-a-never-released"),
                journal=_journal(store, artifacts, deadline),
            )

        assert raised.value.result is not None
        assert raised.value.result.candidate_released is False
        assert store.launches()[-1].state == "NOT_STARTED"
        assert store.launches()[-1].charged is False
        assert store.snapshot().launches_used == 6
        assert store.execution("fold-a-never-released").status == "FAILED"  # type: ignore[union-attr]
    finally:
        store.close()


def test_pre_admission_cancellation_is_terminal_resumable_and_not_charged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    supervised = _runner(tmp_path, artifacts)
    store, deadline = _campaign(tmp_path / "campaign.sqlite")
    cancellation = threading.Event()
    cancellation.set()

    def fail_cleanup(_workspace: object) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(supervised.workspace_materializer, "cleanup", fail_cleanup)
    try:
        with pytest.raises(SupervisedFoldFMExecutionError) as raised:
            supervised.run(
                _request("fold-a-cancelled"),
                journal=_journal(store, artifacts, deadline),
                cancel_event=cancellation,
            )

        assert raised.value.result is not None
        assert raised.value.result.outcome is ExecutionOutcome.CANCELLED
        assert raised.value.result.process is None
        assert store.launches()[-1].state == "NOT_STARTED"
        assert store.launches()[-1].charged is False
        assert store.snapshot().launches_used == 6
        terminal = store.execution("fold-a-cancelled")
        assert terminal is not None and terminal.status == "FAILED"
        failure_artifacts = raised.value.artifacts
        assert failure_artifacts is not None
        diagnostic = artifacts.read_bytes(
            failure_artifacts.artifact("failure_diagnostic"),
            max_bytes=4096,
        ).decode("utf-8")
        assert diagnostic == failure_artifacts.diagnostic
        assert "workspace cleanup failed: OSError" in diagnostic
        assert (
            dict(store.artifacts_for(owner_type="execution", owner_id="fold-a-cancelled"))[
                "failure_diagnostic"
            ].digest
            == failure_artifacts.artifact("failure_diagnostic").sha256
        )
        assert (tmp_path / "workspaces" / "fold-a-cancelled").exists()
    finally:
        store.close()


def test_successful_fold_child_with_cleanup_failure_is_durably_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    supervised = _runner(tmp_path, artifacts)
    store, deadline = _campaign(tmp_path / "campaign.sqlite")

    def fail_cleanup(_workspace: object) -> None:
        raise OSError("synthetic success cleanup failure")

    monkeypatch.setattr(supervised.workspace_materializer, "cleanup", fail_cleanup)
    try:
        with pytest.raises(SupervisedFoldFMExecutionError) as raised:
            supervised.run(
                _request("fold-a-success-cleanup-failure"),
                journal=_journal(store, artifacts, deadline),
            )

        assert raised.value.result is not None and raised.value.result.succeeded
        failure_artifacts = raised.value.artifacts
        assert failure_artifacts is not None and not failure_artifacts.output_validated
        receipt = json.loads(
            artifacts.read_bytes(
                failure_artifacts.artifact("workspace_cleanup"),
                max_bytes=4096,
            )
        )
        assert receipt == {
            "error_type": "OSError",
            "execution_id": "fold-a-success-cleanup-failure",
            "schema_version": 1,
            "workspace_removed": False,
        }
        assert "workspace cleanup failed: OSError" in artifacts.read_bytes(
            failure_artifacts.artifact("failure_diagnostic"),
            max_bytes=4096,
        ).decode("utf-8")
        terminal = store.execution("fold-a-success-cleanup-failure")
        assert terminal is not None and terminal.status == "FAILED"
        assert store.launches()[-1].state == "FINISHED"
        assert store.launches()[-1].charged is True
    finally:
        store.close()


def test_timeout_after_release_remains_a_charged_failed_launch(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    supervised = _runner(tmp_path, artifacts, timeout_seconds=0.001)
    store, deadline = _campaign(tmp_path / "campaign.sqlite")
    try:
        with pytest.raises(SupervisedFoldFMExecutionError) as raised:
            supervised.run(
                _request("fold-a-timeout"),
                journal=_journal(store, artifacts, deadline),
            )

        assert raised.value.result is not None
        assert raised.value.result.outcome is ExecutionOutcome.TIMED_OUT
        assert raised.value.result.candidate_released is True
        assert store.launches()[-1].state == "FINISHED"
        assert store.launches()[-1].charged is True
        assert store.snapshot().launches_used == 7
    finally:
        store.close()


def test_resume_rejects_tampered_indirect_prediction_artifact(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    supervised = _runner(tmp_path, artifacts)
    request = _request("fold-a-tamper")
    store, deadline = _campaign(tmp_path / "campaign.sqlite")
    try:
        run = supervised.run(request, journal=_journal(store, artifacts, deadline))
        prediction_path = artifacts.object_path(run.evidence.predictions)
        os.chmod(prediction_path, 0o600)
        prediction_path.write_bytes(b"tampered")
        os.chmod(prediction_path, 0o400)

        with pytest.raises(SupervisedFoldFMError, match=r"artifact|prediction|digest|size"):
            supervised.run(request, journal=_journal(store, artifacts, deadline))
        assert len(store.launches()) == 7
    finally:
        store.close()


def test_outer_or_final_like_dates_fail_before_any_label_access(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    supervised = _runner(tmp_path, artifacts)
    store, deadline = _campaign(tmp_path / "campaign.sqlite")
    try:
        request = _request(
            "fold-a-forbidden-role",
            query_dates=(20220422, 20220422),
            prefix_labels=_ExplodingLabels(),
            query_labels=_ExplodingLabels(),
        )
        with pytest.raises(SupervisedFoldFMError, match="query dates"):
            supervised.run(request, journal=_journal(store, artifacts, deadline))

        assert len(store.launches()) == 6
        assert store.execution(request.execution_id) is None
        assert not (tmp_path / "workspaces" / request.execution_id).exists()
    finally:
        store.close()
