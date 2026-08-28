from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import psutil  # type: ignore[import-untyped]
import pytest

import kuairand_agent.execution.runner as runner_module
from kuairand_agent.execution.runner import (
    ExecutionOutcome,
    ExecutionResult,
    ExecutionSpec,
    ProcessIdentity,
    ProcessRecord,
    ReconciliationExpectation,
    ReconciliationOutcome,
    Runner,
)


def _spec(root: Path, code: str, *, timeout: float = 0.5) -> ExecutionSpec:
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    return ExecutionSpec(
        execution_id=f"tree-{root.name}",
        nonce=f"tree-nonce-{root.name}-0123456789",
        interpreter=Path(sys.executable),
        arguments=("-c", code),
        workspace=workspace,
        control_dir=root / "control",
        timeout_seconds=timeout,
        memory_limit_bytes=512 * 1024 * 1024,
        workspace_disk_limit_bytes=16 * 1024 * 1024,
        stdout_limit_bytes=16 * 1024,
        stderr_limit_bytes=16 * 1024,
        threads=1,
        source_digest="1" * 64,
        config_digest="2" * 64,
        data_digest="3" * 64,
        checkpoint_digest="4" * 64,
        poll_interval_seconds=0.02,
        disk_poll_interval_seconds=0.05,
        termination_grace_seconds=0.15,
    )


def _alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _wait_dead(pid: int) -> None:
    deadline = time.monotonic() + 2.0
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _alive(pid), f"process {pid} survived runner cleanup"


def _expectation(record: ProcessRecord) -> ReconciliationExpectation:
    return ReconciliationExpectation(
        execution_id=record.execution_id,
        nonce=record.nonce,
        command_digest=record.command_digest,
        environment_digest=record.environment_digest,
        interpreter_real_path=Path(record.interpreter_real_path),
        workspace=Path(record.workspace),
        control_dir=Path(record.control_dir),
        source_digest=record.source_digest,
        config_digest=record.config_digest,
        data_digest=record.data_digest,
        checkpoint_digest=record.checkpoint_digest,
    )


def _start_live_execution(
    spec: ExecutionSpec,
) -> tuple[
    threading.Thread,
    threading.Event,
    list[ProcessRecord],
    list[ExecutionResult],
]:
    committed = threading.Event()
    cancellation = threading.Event()
    records: list[ProcessRecord] = []
    results: list[ExecutionResult] = []

    def commit(record: ProcessRecord) -> None:
        records.append(record)
        committed.set()

    def execute() -> None:
        results.append(
            Runner().run(
                spec,
                commit_launch=commit,
                cancel_event=cancellation,
            )
        )

    thread = threading.Thread(target=execute, name=f"run-{spec.execution_id}")
    thread.start()
    assert committed.wait(2.0), "candidate launch never reached its durable commit seam"
    deadline = time.monotonic() + 2.0
    while not (spec.control_dir / "release.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (spec.control_dir / "release.json").is_file(), "candidate was not released"
    return thread, cancellation, records, results


def test_timeout_kills_child_and_grandchild_in_isolated_process_group(tmp_path: Path) -> None:
    grandchild = """
import os, pathlib, time
pathlib.Path('grandchild.pid').write_text(str(os.getpid()), encoding='ascii')
time.sleep(30)
"""
    child = f"""
import os, pathlib, subprocess, sys, time
pathlib.Path('child.pid').write_text(str(os.getpid()), encoding='ascii')
subprocess.Popen([sys.executable, '-c', {grandchild!r}])
time.sleep(30)
"""
    parent = f"""
import subprocess, sys, time
subprocess.Popen([sys.executable, '-c', {child!r}])
time.sleep(30)
"""
    spec = _spec(tmp_path / "nested-tree", parent)

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.TIMED_OUT
    assert result.cleanup_verified
    child_pid = int((spec.workspace / "child.pid").read_text(encoding="ascii"))
    grandchild_pid = int((spec.workspace / "grandchild.pid").read_text(encoding="ascii"))
    _wait_dead(child_pid)
    _wait_dead(grandchild_pid)


def test_timeout_kills_identity_tracked_descendant_that_escaped_process_group(
    tmp_path: Path,
) -> None:
    escaped = """
import os, pathlib, time
os.setsid()
pathlib.Path('escaped.pid').write_text(str(os.getpid()), encoding='ascii')
time.sleep(30)
"""
    parent = f"""
import subprocess, sys, time
subprocess.Popen([sys.executable, '-c', {escaped!r}])
time.sleep(30)
"""
    spec = _spec(tmp_path / "escaped-tree", parent, timeout=0.7)

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.TIMED_OUT
    assert result.cleanup_verified
    escaped_pid = int((spec.workspace / "escaped.pid").read_text(encoding="ascii"))
    _wait_dead(escaped_pid)


def test_cleanup_never_signals_unrelated_process_group_or_reused_bare_pid(tmp_path: Path) -> None:
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    unrelated_identity = psutil.Process(unrelated.pid).create_time()
    try:
        spec = _spec(
            tmp_path / "unrelated",
            "import time; time.sleep(30)",
            timeout=0.3,
        )
        result = Runner().run(spec, commit_launch=lambda _record: None)

        assert result.outcome is ExecutionOutcome.TIMED_OUT
        assert result.cleanup_verified
        assert unrelated.poll() is None
        assert psutil.Process(unrelated.pid).create_time() == unrelated_identity
    finally:
        unrelated.terminate()
        try:
            unrelated.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            unrelated.kill()
            unrelated.wait(timeout=2.0)


def test_resume_reconciliation_interrupts_an_exact_matching_live_tree(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path / "reconcile-live",
        "import time; time.sleep(30)",
        timeout=10.0,
    )
    thread, _cancellation, records, run_results = _start_live_execution(spec)
    record = records[0]

    result = Runner().reconcile(
        record,
        expected=_expectation(record),
        termination_grace_seconds=0.15,
    )

    thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert run_results
    assert result.outcome is ReconciliationOutcome.INTERRUPTED
    assert result.root_identity_matched
    assert result.root_was_live
    assert result.signal_sent == 15
    assert result.cleanup_verified
    assert not result.surviving_identities
    assert record.identity in result.observed_identities
    assert result.manifest()["candidate_adopted"] is False
    assert result.manifest()["candidate_relaunched"] is False


def test_resume_reconciliation_escalates_a_matching_sigterm_resistant_tree(
    tmp_path: Path,
) -> None:
    code = """
import pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path('ready').write_text('ready', encoding='ascii')
time.sleep(30)
"""
    spec = _spec(tmp_path / "reconcile-force", code, timeout=10.0)
    thread, _cancellation, records, _run_results = _start_live_execution(spec)
    deadline = time.monotonic() + 2.0
    while not (spec.workspace / "ready").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (spec.workspace / "ready").is_file()
    record = records[0]

    result = Runner().reconcile(
        record,
        expected=_expectation(record),
        termination_grace_seconds=0.05,
    )

    thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert result.outcome is ReconciliationOutcome.TERMINATED
    assert result.signal_sent == 9
    assert result.cleanup_verified
    assert len(result.digest) == 64


def test_resume_reconciliation_never_signals_on_scientific_or_pid_identity_mismatch(
    tmp_path: Path,
) -> None:
    spec = _spec(
        tmp_path / "reconcile-mismatch",
        "import time; time.sleep(30)",
        timeout=10.0,
    )
    thread, cancellation, records, _run_results = _start_live_execution(spec)
    record = records[0]
    try:
        wrong_science = replace(_expectation(record), source_digest="f" * 64)
        scientific_result = Runner().reconcile(record, expected=wrong_science)

        assert scientific_result.outcome is ReconciliationOutcome.IDENTITY_MISMATCH
        assert scientific_result.signal_sent is None
        assert _alive(record.identity.pid)

        reused_identity = ProcessIdentity(
            pid=record.identity.pid,
            create_time=record.identity.create_time + 10_000.0,
        )
        reused_record = replace(record, identity=reused_identity)
        reused_result = Runner().reconcile(
            reused_record,
            expected=_expectation(reused_record),
        )

        assert reused_result.outcome is ReconciliationOutcome.IDENTITY_MISMATCH
        assert reused_result.signal_sent is None
        assert _alive(record.identity.pid)
    finally:
        cancellation.set()
        thread.join(timeout=3.0)
    assert not thread.is_alive()


def test_resume_reconciliation_reports_an_already_dead_record_without_signaling(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path / "reconcile-dead", "pass", timeout=2.0)
    launch = Runner().run(spec, commit_launch=lambda _record: None)
    assert launch.process is not None

    result = Runner().reconcile(launch.process, expected=_expectation(launch.process))

    assert result.outcome is ReconciliationOutcome.ALREADY_DEAD
    assert result.signal_sent is None
    assert not result.root_was_live
    assert not result.cleanup_verified


def test_resume_reconciliation_reports_inspection_denial_after_safe_identity_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(
        tmp_path / "reconcile-inspection",
        "import time; time.sleep(30)",
        timeout=10.0,
    )
    thread, _cancellation, records, _run_results = _start_live_execution(spec)
    record = records[0]

    def deny_inspection(_process: object) -> tuple[object, ...]:
        raise PermissionError("synthetic narrow process-inspection denial")

    monkeypatch.setattr(runner_module, "_direct_children", deny_inspection)
    result = Runner().reconcile(record, expected=_expectation(record))

    thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert result.outcome is ReconciliationOutcome.INSPECTION_FAILED
    assert result.root_identity_matched
    assert result.signal_sent in {15, 9}
    assert not result.cleanup_verified
    assert "inspection failed" in result.detail
