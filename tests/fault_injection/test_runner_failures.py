from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

import kuairand_agent.execution.runner as runner_module
from kuairand_agent.execution.runner import ExecutionOutcome, ExecutionSpec, Runner


def _spec(
    root: Path,
    code: str,
    *,
    timeout: float = 5.0,
    memory_limit: int = 512 * 1024 * 1024,
    disk_limit: int = 16 * 1024 * 1024,
    process_limit: int = 64,
) -> ExecutionSpec:
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    return ExecutionSpec(
        execution_id=f"fault-{root.name}",
        nonce=f"fault-nonce-{root.name}-0123456789",
        interpreter=Path(sys.executable),
        arguments=("-c", code),
        workspace=workspace,
        control_dir=root / "control",
        timeout_seconds=timeout,
        memory_limit_bytes=memory_limit,
        workspace_disk_limit_bytes=disk_limit,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
        threads=1,
        source_digest="1" * 64,
        config_digest="2" * 64,
        data_digest="3" * 64,
        checkpoint_digest="4" * 64,
        process_limit=process_limit,
        poll_interval_seconds=0.01,
        disk_poll_interval_seconds=0.02,
        termination_grace_seconds=0.1,
    )


def test_memory_limit_terminates_candidate_and_records_peak_tree_rss(tmp_path: Path) -> None:
    code = """
import time
allocation = bytearray(96 * 1024 * 1024)
allocation[0] = 1
time.sleep(30)
"""
    spec = _spec(
        tmp_path / "memory",
        code,
        memory_limit=48 * 1024 * 1024,
    )

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.MEMORY_LIMIT
    assert result.cleanup_verified
    assert result.peak_tree_rss_bytes > spec.memory_limit_bytes
    assert result.candidate_released


def test_workspace_disk_limit_terminates_oversized_output(tmp_path: Path) -> None:
    code = """
from pathlib import Path
import time
Path('oversized.bin').write_bytes(b'x' * (2 * 1024 * 1024))
time.sleep(30)
"""
    spec = _spec(
        tmp_path / "disk",
        code,
        disk_limit=256 * 1024,
    )

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.DISK_LIMIT
    assert result.cleanup_verified
    assert result.peak_workspace_bytes > spec.workspace_disk_limit_bytes


def test_cancellation_terminates_tree_and_returns_durable_resource_evidence(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path / "cancel", "import time; time.sleep(30)")
    cancellation = threading.Event()

    def cancel_soon() -> None:
        time.sleep(0.1)
        cancellation.set()

    trigger = threading.Thread(target=cancel_soon)
    trigger.start()
    result = Runner().run(
        spec,
        commit_launch=lambda _record: None,
        cancel_event=cancellation,
    )
    trigger.join(timeout=1.0)

    assert result.outcome is ExecutionOutcome.CANCELLED
    assert result.cleanup_verified
    assert result.wall_seconds < 3.0
    assert result.process is not None
    assert (spec.control_dir / "result.json").is_file()


def test_cancellation_present_before_admission_never_spawns_or_commits(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "cancel-before-admission", "raise AssertionError('must not run')")
    cancellation = threading.Event()
    cancellation.set()
    commits = 0

    def commit(_record: object) -> None:
        nonlocal commits
        commits += 1

    result = Runner().run(spec, commit_launch=commit, cancel_event=cancellation)

    assert result.outcome is ExecutionOutcome.CANCELLED
    assert result.process is None
    assert result.candidate_released is False
    assert result.cleanup_verified
    assert commits == 0
    assert not (spec.control_dir / "process.json").exists()
    assert (spec.control_dir / "result.json").is_file()


def test_cancellation_set_by_durable_commit_never_releases_candidate_code(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "candidate-ran"
    spec = _spec(
        tmp_path / "cancel-during-commit",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
    )
    cancellation = threading.Event()
    commits = 0

    def commit(_record: object) -> None:
        nonlocal commits
        commits += 1
        cancellation.set()

    result = Runner().run(spec, commit_launch=commit, cancel_event=cancellation)

    assert result.outcome is ExecutionOutcome.CANCELLED
    assert result.process is not None
    assert result.candidate_released is False
    assert result.cleanup_verified
    assert commits == 1
    assert not marker.exists()
    assert not (spec.control_dir / "release.json").exists()
    assert (spec.control_dir / "result.json").is_file()


def test_process_limit_counts_descendants_and_terminates_the_tree(tmp_path: Path) -> None:
    code = """
import subprocess, sys, time
children = [
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    for _index in range(3)
]
time.sleep(30)
"""
    spec = _spec(tmp_path / "process-limit", code, process_limit=2)

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.PROCESS_LIMIT
    assert result.cleanup_verified
    assert result.peak_process_count > spec.process_limit


def test_process_inspection_denial_is_reported_without_an_uncaught_cleanup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path / "inspection-denied", "import time; time.sleep(30)")

    def deny_inspection(_process: object) -> tuple[object, ...]:
        raise PermissionError("synthetic local containment denial")

    monkeypatch.setattr(runner_module, "_direct_children", deny_inspection)

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.INSPECTION_FAILED
    assert not result.cleanup_verified
    assert result.detail is not None and "inspection was denied" in result.detail
    assert result.terminating_signal in {15, 9}
