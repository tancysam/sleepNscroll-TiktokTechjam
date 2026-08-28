from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil  # type: ignore[import-untyped]
import pytest

_HELPER = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from kuairand_agent.execution.runner import ExecutionSpec, Runner
from kuairand_agent.execution.signals import cancellation_on_signals

root = Path(sys.argv[1]).resolve(strict=True)
workspace = root / "workspace"
workspace.mkdir()
spec = ExecutionSpec(
    execution_id="external-signal-candidate",
    nonce="external-signal-nonce-0123456789",
    interpreter=Path(sys.executable),
    arguments=("-c", "import time; time.sleep(30)"),
    workspace=workspace,
    control_dir=root / "control",
    timeout_seconds=30.0,
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
with cancellation_on_signals() as cancellation:
    result = Runner().run(
        spec,
        commit_launch=lambda _record: None,
        cancel_event=cancellation,
    )
(root / "summary.json").write_text(
    json.dumps(result.manifest(), sort_keys=True, separators=(",", ":")),
    encoding="ascii",
)
"""


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10.0
    while not path.is_file() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.is_file(), process.stderr.read() if process.stderr is not None else ""


def _alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


@pytest.mark.parametrize("requested_signal", [signal.SIGINT, signal.SIGTERM])
def test_external_signal_cancels_exact_candidate_tree_and_flushes_evidence(
    tmp_path: Path,
    requested_signal: signal.Signals,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", _HELPER, str(tmp_path)],
        cwd=Path(__file__).parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    identity = psutil.Process(process.pid).create_time()
    try:
        _wait_for(tmp_path / "control" / "release.json", process)
        record = json.loads((tmp_path / "control" / "process.json").read_text(encoding="utf-8"))
        candidate_pid = int(record["pid"])
        assert _alive(candidate_pid)
        assert process.poll() is None
        assert psutil.Process(process.pid).create_time() == identity

        process.send_signal(requested_signal)
        stdout, stderr = process.communicate(timeout=10.0)

        assert process.returncode == 0, (stdout, stderr)
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["outcome"] == "cancelled"
        assert summary["candidate_released"] is True
        assert summary["cleanup_verified"] is True
        assert (tmp_path / "control" / "result.json").is_file()
        assert not _alive(candidate_pid)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
