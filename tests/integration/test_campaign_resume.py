from __future__ import annotations

import json
import sys
from pathlib import Path

from kuairand_agent.campaign.controller import CAMPAIGN_DATABASE_NAME, CampaignEngine
from kuairand_agent.campaign.store import CampaignStore
from kuairand_agent.execution.runner import ExecutionOutcome, ExecutionSpec, ProcessRecord, Runner

from .test_campaign_controller import (
    ENVIRONMENT_DIGEST,
    FakeClock,
    build_request,
)

SOURCE = "2" * 64
CONFIG = "3" * 64
CAPABILITY = "4" * 64
DATA = "5" * 64
CHECKPOINT = "6" * 64


def _execution_spec(run_dir: Path) -> ExecutionSpec:
    workspace = run_dir / "workspaces" / "failed-child"
    workspace.mkdir(parents=True)
    (run_dir / "controls").mkdir()
    return ExecutionSpec(
        execution_id="failed-child",
        nonce="failed-child-nonce-0123456789",
        interpreter=Path(sys.executable),
        arguments=("-c", "raise SystemExit(7)"),
        workspace=workspace,
        control_dir=run_dir / "controls" / "failed-child",
        timeout_seconds=5.0,
        memory_limit_bytes=256 * 1024 * 1024,
        workspace_disk_limit_bytes=16 * 1024 * 1024,
        stdout_limit_bytes=64 * 1024,
        stderr_limit_bytes=64 * 1024,
        threads=1,
        source_digest=SOURCE,
        config_digest=CONFIG,
        data_digest=DATA,
        checkpoint_digest=CHECKPOINT,
        poll_interval_seconds=0.01,
        disk_poll_interval_seconds=0.02,
        termination_grace_seconds=0.1,
    )


def test_scripted_failed_child_and_starting_execution_reconcile_without_incumbent_loss(
    tmp_path: Path,
) -> None:
    request = build_request(tmp_path)
    clock = FakeClock()
    engine = CampaignEngine(clock=clock)
    engine.create(request)
    spec = _execution_spec(request.run_dir)

    with CampaignStore.open(request.run_dir / CAMPAIGN_DATABASE_NAME) as store:
        incumbent_before = store.current_incumbent()
        assert incumbent_before is not None
        store.create_execution(
            execution_id="provider-starting",
            kind="provider_action",
            tier="provider",
            command=(str(Path(sys.executable).resolve()), "-c", "pass"),
            expected_revision=store.snapshot().revision,
            status="STARTING",
            nonce="provider-starting-nonce-012345",
            source_digest=SOURCE,
            config_digest=CONFIG,
            capability_digest=CAPABILITY,
            environment_digest=ENVIRONMENT_DIGEST,
            data_digest=DATA,
            checkpoint_digest=CHECKPOINT,
        )
        launch = store.reserve_launch(
            launch_id="failed-child-launch",
            reservation_key="failed-child-reservation",
            category="diverse_inner_screen",
            original_category="diverse_inner_screen",
            purpose="scripted failed child",
            expected_revision=store.snapshot().revision,
            scientific_iteration=1,
            seed=0,
        )
        store.create_execution(
            execution_id=spec.execution_id,
            kind="full_train_evaluate",
            tier="inner_screen",
            command=spec.command,
            expected_revision=store.snapshot().revision,
            launch_id=launch.launch_id,
            seed=0,
            status="STARTING",
            nonce=spec.nonce,
            source_digest=SOURCE,
            config_digest=CONFIG,
            capability_digest=CAPABILITY,
            environment_digest=ENVIRONMENT_DIGEST,
            data_digest=DATA,
            checkpoint_digest=CHECKPOINT,
        )

        def commit_launch(record: ProcessRecord) -> None:
            store.transition_execution(
                spec.execution_id,
                from_state="STARTING",
                to_state="RUNNING",
                expected_revision=store.snapshot().revision,
                reason="persist process receipt before candidate release",
                process_record_digest=record.digest,
                process_record=record.manifest(),
            )
            store.transition_launch(
                launch.launch_id,
                to_state="STARTED",
                expected_revision=store.snapshot().revision,
                start_receipt_digest=record.digest,
                metadata={"execution_id": spec.execution_id},
            )

        result = Runner().run(spec, commit_launch=commit_launch)
        assert result.outcome is ExecutionOutcome.EXIT_NONZERO
        assert result.exit_code == 7
        assert {item.status for item in store.unfinished_executions()} == {
            "STARTING",
            "RUNNING",
        }
        assert store.current_incumbent() == incumbent_before

    resumed = engine.resume(request.run_dir)

    assert resumed.status == "RUNNING"
    assert resumed.launches_used == 7
    assert resumed.unfinished_execution_ids == ()
    assert resumed.reconciliation_count == 2
    assert resumed.incumbent_id == incumbent_before.incumbent_id
    assert resumed.incumbent_is_fallback
    with CampaignStore.open(request.run_dir / CAMPAIGN_DATABASE_NAME, read_only=True) as store:
        executions = {item.execution_id: item for item in store.executions()}
        assert executions["provider-starting"].status == "INTERRUPTED"
        assert executions["failed-child"].status == "INTERRUPTED"
        failed_launch = next(
            item for item in store.launches() if item.launch_id == "failed-child-launch"
        )
        assert failed_launch.state == "FINISHED"
        assert failed_launch.charged
        assert store.current_incumbent() == incumbent_before

    reports = sorted((request.run_dir / "controller" / "reconciliations").iterdir())
    runner_reports = [
        json.loads(path.read_text(encoding="ascii"))["runner_reconciliation"] for path in reports
    ]
    assert any(
        report is not None and report["outcome"] == "already_dead" for report in runner_reports
    )

    resumed_again = engine.resume(request.run_dir)
    assert resumed_again.reconciliation_count == 2
    assert resumed_again.revision == resumed.revision
    assert resumed_again.incumbent_id == incumbent_before.incumbent_id


def test_resume_preserves_original_deadline_and_exposes_finalization_required(
    tmp_path: Path,
) -> None:
    request = build_request(tmp_path)
    clock = FakeClock()
    engine = CampaignEngine(clock=clock)
    engine.create(request)
    deadline_dir = request.run_dir / "controller" / "deadline"
    initial = json.loads((deadline_dir / "checkpoint-00000000.json").read_text(encoding="ascii"))[
        "state"
    ]

    clock.advance(3600)
    reserve = engine.resume(request.run_dir)

    assert reserve.status == "FINALIZATION_REQUIRED"
    assert reserve.phase == "finalization_required"
    assert reserve.finalization_required
    assert reserve.deadline_elapsed_seconds == 3600.0
    assert reserve.deadline_remaining_seconds == 3600.0
    assert reserve.finalization_reserve_seconds == 3600

    clock.advance(3600)
    expired = engine.resume(request.run_dir)
    assert expired.status == "INCOMPLETE"
    assert expired.phase == "deadline_exhausted"
    assert expired.deadline_remaining_seconds == 0.0
    assert expired.incumbent_id == "official-fm-fallback-seed-4"

    checkpoints = sorted(deadline_dir.iterdir())
    assert len(checkpoints) == 4
    for path in checkpoints:
        state = json.loads(path.read_text(encoding="ascii"))["state"]
        for key in (
            "wall_clock_seconds",
            "finalization_reserve_seconds",
            "started_utc",
            "utc_deadline",
            "original_boot_identity",
            "monotonic_started_ns",
            "monotonic_deadline_ns",
        ):
            assert state[key] == initial[key]

    before_status = tuple(deadline_dir.iterdir())
    assert engine.status(request.run_dir).status == "INCOMPLETE"
    assert tuple(deadline_dir.iterdir()) == before_status


def test_resume_finishes_launch_after_terminal_execution_split_commit(tmp_path: Path) -> None:
    request = build_request(tmp_path)
    clock = FakeClock()
    engine = CampaignEngine(clock=clock)
    engine.create(request)

    with CampaignStore.open(request.run_dir / CAMPAIGN_DATABASE_NAME) as store:
        incumbent = store.current_incumbent()
        launch = store.reserve_launch(
            launch_id="split-commit-launch",
            reservation_key="split-commit-reservation",
            category="diverse_inner_screen",
            original_category="diverse_inner_screen",
            purpose="fault injection split commit",
            expected_revision=store.snapshot().revision,
            scientific_iteration=1,
            seed=0,
        )
        store.create_execution(
            execution_id="split-commit-execution",
            kind="full_train_evaluate",
            tier="inner_screen",
            command=(str(Path(sys.executable).resolve()), "-c", "pass"),
            expected_revision=store.snapshot().revision,
            launch_id=launch.launch_id,
            seed=0,
            status="STARTING",
            nonce="split-commit-nonce-0123456789",
            source_digest=SOURCE,
            config_digest=CONFIG,
            capability_digest=CAPABILITY,
            environment_digest=ENVIRONMENT_DIGEST,
            data_digest=DATA,
            checkpoint_digest=CHECKPOINT,
        )
        store.transition_execution(
            "split-commit-execution",
            from_state="STARTING",
            to_state="INTERRUPTED",
            expected_revision=store.snapshot().revision,
            reason="fault injection after terminal commit and before launch commit",
            finished_at=clock.utc_now().isoformat(),
        )
        store.reserve_launch(
            launch_id="orphan-reservation-launch",
            reservation_key="orphan-reservation",
            category="diverse_inner_screen",
            original_category="diverse_inner_screen",
            purpose="fault injection before execution row",
            expected_revision=store.snapshot().revision,
            scientific_iteration=2,
            seed=0,
        )
        assert (
            next(item for item in store.launches() if item.launch_id == launch.launch_id).state
            == "RESERVED"
        )

    resumed = engine.resume(request.run_dir)

    assert resumed.launches_used == 8
    assert resumed.reconciliation_count == 2
    assert resumed.incumbent_id == "official-fm-fallback-seed-4"
    with CampaignStore.open(request.run_dir / CAMPAIGN_DATABASE_NAME, read_only=True) as store:
        repaired = next(
            item for item in store.launches() if item.launch_id == "split-commit-launch"
        )
        assert repaired.state == "FINISHED"
        assert repaired.charged
        orphan = next(
            item for item in store.launches() if item.launch_id == "orphan-reservation-launch"
        )
        assert orphan.state == "FINISHED"
        assert orphan.charged
        assert store.current_incumbent() == incumbent
