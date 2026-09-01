from __future__ import annotations

import errno
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from kuairand_agent.campaign.budgets import LaunchCategory, WorkPhase
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateJournalError,
    CandidateJournalPolicy,
)
from kuairand_agent.campaign.clock import DeadlineObservation, DeadlineState
from kuairand_agent.campaign.store import ArtifactSpec, CampaignStore
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactPersistenceError,
    ArtifactStore,
)
from kuairand_agent.execution.candidate_executor import (
    CandidateAction,
    CandidateExecutionArtifacts,
)
from kuairand_agent.execution.policy import SplitRole
from kuairand_agent.execution.runner import (
    ExecutionOutcome,
    ExecutionResult,
    ExecutionSpec,
    ProcessRecord,
    Runner,
)
from kuairand_agent.execution.workspace import CandidateWorkspace


@dataclass
class _Clock:
    def utc_now(self) -> datetime:
        return datetime(2030, 1, 1, tzinfo=UTC)

    def monotonic_ns(self) -> int:
        return 1_000_000_000

    def boot_identity(self) -> str:
        return "fault-test-boot"


def _campaign(tmp_path: Path) -> tuple[CampaignStore, DeadlineObservation]:
    clock = _Clock()
    deadline = DeadlineState.start(
        clock,
        wall_clock_seconds=7_200,
        finalization_reserve_seconds=3_600,
    )
    store = CampaignStore.create(
        tmp_path / "campaign.sqlite",
        campaign_id="fault-campaign",
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


def _execution(tmp_path: Path, execution_id: str) -> tuple[ExecutionSpec, CandidateWorkspace]:
    root = tmp_path / "workspace"
    root.mkdir()
    controls = tmp_path / "controls"
    controls.mkdir()
    workspace = CandidateWorkspace(
        root=root,
        execution_id=execution_id,
        split_role=SplitRole.INNER_TRAIN,
        source_snapshot_sha256="8" * 64,
        source_files=(),
        input_files=(),
        output_limit_bytes=1024,
        temp_limit_bytes=1024,
        request_sha256="9" * 64,
        manifest_digest="a" * 64,
    )
    return (
        ExecutionSpec(
            execution_id=execution_id,
            nonce="fault-receipt-0123456789",
            interpreter=Path(sys.executable),
            arguments=("-c", "pass"),
            workspace=root,
            control_dir=controls / execution_id,
            timeout_seconds=30,
            memory_limit_bytes=1024**3,
            workspace_disk_limit_bytes=1024**2,
            stdout_limit_bytes=1024,
            stderr_limit_bytes=1024,
            threads=1,
            source_digest="b" * 64,
            config_digest="c" * 64,
            data_digest="d" * 64,
            checkpoint_digest="e" * 64,
        ),
        workspace,
    )


def _train_policy() -> CandidateJournalPolicy:
    return CandidateJournalPolicy(
        family="generated_lambdarank",
        phase=WorkPhase.RESEARCH,
        p95_runtime_seconds=30,
        category=LaunchCategory.DIVERSE_INNER_SCREEN,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _record_fallback(store: CampaignStore) -> None:
    store.record_incumbent(
        incumbent_id="official-fm-fallback-seed-4",
        eligibility="official_fm_qualified",
        source_digest="1" * 64,
        checkpoint_digest="2" * 64,
        artifact_closure_digest="3" * 64,
        replay_verified=True,
        is_fallback=True,
        expected_revision=store.snapshot().revision,
        reason="immutable fault-test fallback",
        outer_primary_mean=0.6016,
    )


def _base_evidence(
    artifacts: ArtifactStore,
    result: ExecutionResult,
) -> CandidateExecutionArtifacts:
    return CandidateExecutionArtifacts(
        (
            (
                "execution_manifest",
                artifacts.put_bytes(
                    _canonical_json(result.manifest()),
                    kind=ArtifactKind.MANIFEST,
                ),
            ),
            ("stderr", artifacts.put_file(result.stderr.path, kind=ArtifactKind.LOG)),
            ("stdout", artifacts.put_file(result.stdout.path, kind=ArtifactKind.LOG)),
        ),
        output_validated=False,
        diagnostic="fault-test terminal evidence",
    )


def test_restart_finishes_partial_process_receipt_without_charging_unreleased_candidate(
    tmp_path: Path,
) -> None:
    store, deadline = _campaign(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    spec, workspace = _execution(tmp_path, "receipt-before-launch-event")
    journal = CampaignStoreCandidateJournal(
        store=store,
        artifact_store=artifacts,
        deadline=deadline,
        policy=_train_policy(),
    )
    try:
        journal.prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)

        def persist_receipt_then_crash(process_record: ProcessRecord) -> None:
            store.transition_execution(
                spec.execution_id,
                from_state="STARTING",
                to_state="RUNNING",
                expected_revision=store.snapshot().revision,
                reason="fault injection: receipt committed before launch event",
                process_record_digest=process_record.digest,
                process_record=process_record.manifest(),
                metadata={"candidate_released": False},
            )
            raise RuntimeError("injected crash before STARTED launch event")

        result = Runner().run(spec, commit_launch=persist_receipt_then_crash)
        assert result.outcome is ExecutionOutcome.LAUNCH_COMMIT_FAILED
        assert result.candidate_released is False
        running = store.execution(spec.execution_id)
        assert running is not None
        assert running.status == "RUNNING"
        assert store.launches()[-1].state == "RESERVED"

        evidence = CandidateExecutionArtifacts(
            (
                (
                    "execution_manifest",
                    artifacts.put_bytes(
                        _canonical_json(result.manifest()), kind=ArtifactKind.MANIFEST
                    ),
                ),
                ("stderr", artifacts.put_file(result.stderr.path, kind=ArtifactKind.LOG)),
                ("stdout", artifacts.put_file(result.stdout.path, kind=ArtifactKind.LOG)),
            ),
            output_validated=False,
            diagnostic="injected launch commit failure",
        )
        artifact_specs = tuple(
            (
                role,
                ArtifactSpec(
                    digest=reference.sha256,
                    kind=reference.kind.value,
                    relative_path=reference.object_relative_path.as_posix(),
                    size_bytes=reference.size_bytes,
                ),
            )
            for role, reference in evidence.entries
        )
        result_digest = hashlib.sha256(
            b"kuairand-candidate-execution-result-v1\0" + _canonical_json(result.manifest())
        ).hexdigest()
        store.transition_execution(
            spec.execution_id,
            from_state="RUNNING",
            to_state="FAILED",
            expected_revision=store.snapshot().revision,
            reason="fault injection: crash after terminal evidence transaction",
            result_digest=result_digest,
            finished_at=result.ended_at_utc,
            artifacts=artifact_specs,
            metadata={"candidate_released": False, "output_validated": False},
        )
        terminal = store.execution(spec.execution_id)
        assert terminal is not None
        assert terminal.status == "FAILED"
        launch = store.launches()[-1]
        assert launch.launch_number == 7
        assert launch.state == "RESERVED"
        assert launch.charged is True

        restarted = CampaignStoreCandidateJournal(
            store=store,
            artifact_store=artifacts,
            deadline=deadline,
            policy=_train_policy(),
        )
        restored = restarted.rehydrate_terminal(spec.execution_id)
        launch = store.launches()[-1]
        assert launch.state == "NOT_STARTED"
        assert launch.charged is False
        assert store.snapshot().launches_used == 6
        assert restored.artifacts.closure_digest == evidence.closure_digest

        restarted.finish(
            action=CandidateAction.TRAIN,
            result=result,
            artifacts=evidence,
        )
        conflicting = CandidateExecutionArtifacts(
            tuple(
                (role, reference)
                if role != "execution_manifest"
                else (
                    role,
                    artifacts.put_bytes(b"different", kind=ArtifactKind.MANIFEST),
                )
                for role, reference in evidence.entries
            ),
            output_validated=False,
            diagnostic="different terminal closure",
        )
        with pytest.raises(CandidateJournalError, match="trusted runner result"):
            restarted.finish(
                action=CandidateAction.TRAIN,
                result=result,
                artifacts=conflicting,
            )
    finally:
        store.close()


def test_terminal_artifacts_survive_database_transition_exception_and_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, deadline = _campaign(tmp_path)
    _record_fallback(store)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    spec, workspace = _execution(tmp_path, "artifact-before-terminal-transition")
    journal = CampaignStoreCandidateJournal(
        store=store,
        artifact_store=artifacts,
        deadline=deadline,
        policy=_train_policy(),
    )
    try:
        journal.prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)
        result = Runner().run(spec, commit_launch=journal.commit)
        assert result.succeeded and result.candidate_released
        evidence = _base_evidence(artifacts, result)
        for _role, reference in evidence.entries:
            artifacts.verify(reference)

        original_transition = store.transition_execution
        injected = False

        def fail_first_terminal_transition(*args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            if kwargs.get("to_state") in {"SUCCEEDED", "FAILED"} and not injected:
                injected = True
                raise RuntimeError("injected crash after artifact commit")
            return original_transition(*args, **kwargs)

        monkeypatch.setattr(store, "transition_execution", fail_first_terminal_transition)
        with pytest.raises(RuntimeError, match="after artifact commit"):
            journal.finish(
                action=CandidateAction.TRAIN,
                result=result,
                artifacts=evidence,
            )

        pending = store.execution(spec.execution_id)
        assert pending is not None and pending.status == "RUNNING"
        assert store.artifacts_for(owner_type="execution", owner_id=spec.execution_id) == ()
        assert store.launches()[-1].state == "STARTED"
        assert store.snapshot().launches_used == 7
        incumbent = store.current_incumbent()
        assert incumbent is not None
        assert incumbent.incumbent_id == "official-fm-fallback-seed-4"

        journal.finish(
            action=CandidateAction.TRAIN,
            result=result,
            artifacts=evidence,
        )
        terminal = journal.rehydrate_terminal(spec.execution_id)
        assert terminal.execution.status == "FAILED"
        assert terminal.artifacts.closure_digest == evidence.closure_digest
        assert store.launches()[-1].state == "FINISHED"
        assert store.snapshot().launches_used == 7

        journal.finish(
            action=CandidateAction.TRAIN,
            result=result,
            artifacts=evidence,
        )
        assert store.snapshot().launches_used == 7
        assert store.current_incumbent() == incumbent
    finally:
        store.close()


def test_enospc_staging_write_is_clean_and_terminal_evidence_can_resume(
    tmp_path: Path,
) -> None:
    store, deadline = _campaign(tmp_path)
    _record_fallback(store)
    fail_next_write = True

    def injected_staging_write(destination: BinaryIO, payload: bytes) -> int:
        nonlocal fail_next_write
        if fail_next_write:
            fail_next_write = False
            raise OSError(errno.ENOSPC, "synthetic artifact volume full")
        written = destination.write(payload)
        assert written is not None
        return written

    artifacts = ArtifactStore(
        tmp_path / "artifacts",
        staging_writer=injected_staging_write,
    )
    spec, workspace = _execution(tmp_path, "enospc-terminal-evidence")
    journal = CampaignStoreCandidateJournal(
        store=store,
        artifact_store=artifacts,
        deadline=deadline,
        policy=_train_policy(),
    )
    try:
        journal.prepare(action=CandidateAction.TRAIN, spec=spec, workspace=workspace)
        result = Runner().run(spec, commit_launch=journal.commit)
        assert result.succeeded and result.candidate_released

        with pytest.raises(ArtifactPersistenceError, match="ENOSPC"):
            _base_evidence(artifacts, result)

        assert tuple(artifacts.staging_root.iterdir()) == ()
        assert tuple(artifacts.objects_root.rglob("[0-9a-f]" * 64)) == ()
        pending = store.execution(spec.execution_id)
        assert pending is not None and pending.status == "RUNNING"
        assert store.launches()[-1].state == "STARTED"
        assert store.snapshot().launches_used == 7
        incumbent = store.current_incumbent()
        assert incumbent is not None
        assert incumbent.incumbent_id == "official-fm-fallback-seed-4"

        evidence = _base_evidence(artifacts, result)
        journal.finish(
            action=CandidateAction.TRAIN,
            result=result,
            artifacts=evidence,
        )
        restored = journal.rehydrate_terminal(spec.execution_id)
        assert restored.execution.status == "FAILED"
        assert restored.artifacts.closure_digest == evidence.closure_digest
        assert store.launches()[-1].state == "FINISHED"
        assert store.snapshot().launches_used == 7
        assert store.current_incumbent() == incumbent
        assert tuple(artifacts.staging_root.iterdir()) == ()
    finally:
        store.close()
