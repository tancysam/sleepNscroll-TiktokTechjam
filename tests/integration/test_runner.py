from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kuairand_agent.execution.runner import (
    ExecutionOutcome,
    ExecutionSpec,
    ProcessRecord,
    Runner,
    RunnerInputError,
    active_python_interpreter,
)


def _spec(
    root: Path,
    arguments: tuple[str, ...],
    *,
    timeout: float = 3.0,
    stdout_limit: int = 64 * 1024,
    stderr_limit: int = 64 * 1024,
    device: str = "cpu",
    extra_environment: tuple[tuple[str, str], ...] = (),
) -> ExecutionSpec:
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    return ExecutionSpec(
        execution_id=f"exec-{root.name}",
        nonce=f"nonce-{root.name}-0123456789abcdef",
        interpreter=Path(sys.executable),
        arguments=arguments,
        workspace=workspace,
        control_dir=root / "control",
        timeout_seconds=timeout,
        memory_limit_bytes=512 * 1024 * 1024,
        workspace_disk_limit_bytes=16 * 1024 * 1024,
        stdout_limit_bytes=stdout_limit,
        stderr_limit_bytes=stderr_limit,
        threads=2,
        source_digest="1" * 64,
        config_digest="2" * 64,
        data_digest="3" * 64,
        checkpoint_digest="4" * 64,
        device=device,
        poll_interval_seconds=0.02,
        disk_poll_interval_seconds=0.05,
        termination_grace_seconds=0.2,
        extra_environment=extra_environment,
    )


def test_active_python_interpreter_preserves_virtual_environment_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch path must retain pyvenv.cfg discovery while evidence records its real target."""

    interpreter_link = tmp_path / "venv" / "bin" / "python"
    interpreter_link.parent.mkdir(parents=True)
    interpreter_link.symlink_to(Path(sys.executable).resolve(strict=True))
    monkeypatch.setattr(sys, "executable", str(interpreter_link))

    selected = active_python_interpreter()

    assert selected == interpreter_link
    assert selected.resolve(strict=True) != selected


def test_active_virtual_environment_interpreter_reaches_installed_site_packages(
    tmp_path: Path,
) -> None:
    selected = active_python_interpreter()
    if selected.resolve(strict=True) == selected:
        pytest.skip("active interpreter is not a virtual-environment symlink")
    spec = _spec(
        tmp_path / "venv-site-packages",
        ("-c", "import numpy; print(numpy.__version__)"),
    )
    committed: list[ProcessRecord] = []

    result = Runner().run(spec, commit_launch=committed.append)

    assert result.outcome is ExecutionOutcome.SUCCEEDED
    assert result.process is not None
    assert result.process.command[0] == str(selected)
    assert result.process.interpreter_real_path == str(selected.resolve(strict=True))


def test_runner_success_uses_sanitized_private_environment_and_records_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-runner-seam")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/must-not-cross.sock")
    code = """
import json, os, pathlib, sys, time
keys = [
    'HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'KUAIRAND_DEVICE',
    'KUAIRAND_MODE', 'OPENAI_API_KEY', 'SSH_AUTH_SOCK'
]
pathlib.Path('environment.json').write_text(
    json.dumps({key: os.environ.get(key) for key in keys}), encoding='utf-8'
)
print('candidate stdout is diagnostic only; primary=999')
print('candidate stderr', file=sys.stderr)
time.sleep(0.05)
"""
    spec = _spec(
        tmp_path / "success",
        ("-c", code),
        extra_environment=(("KUAIRAND_MODE", "fixture"),),
    )
    committed: list[ProcessRecord] = []

    result = Runner().run(spec, commit_launch=committed.append)

    assert result.outcome is ExecutionOutcome.SUCCEEDED
    assert result.succeeded
    assert result.exit_code == 0
    assert result.terminating_signal is None
    assert result.candidate_released
    assert result.cleanup_verified
    assert result.process is committed[0]
    assert result.process is not None
    assert result.process.identity.pid == result.process.process_group_id
    assert result.process.nonce == spec.nonce
    assert result.process.command == spec.command
    assert result.peak_tree_rss_bytes > 0
    assert result.peak_process_count >= 1
    assert result.device == "cpu"
    assert result.threads == 2
    assert result.candidate_metrics_accepted is False
    assert "metrics" not in result.manifest()
    assert (spec.control_dir / "process.json").is_file()
    assert (spec.control_dir / "release.json").is_file()
    assert (spec.control_dir / "result.json").is_file()
    persisted_process = ProcessRecord.from_manifest(
        json.loads((spec.control_dir / "process.json").read_text(encoding="ascii"))
    )
    assert persisted_process == result.process
    assert persisted_process.digest == result.process.digest

    environment = json.loads((spec.workspace / "environment.json").read_text(encoding="utf-8"))
    assert isinstance(environment, dict)
    for private_name in (
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    ):
        private_path = Path(environment[private_name])
        assert private_path.is_relative_to(spec.workspace)
        assert private_path.is_dir()
    for thread_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert environment[thread_name] == "2"
    assert environment["KUAIRAND_DEVICE"] == "cpu"
    assert environment["KUAIRAND_MODE"] == "fixture"
    assert environment["OPENAI_API_KEY"] is None
    assert environment["SSH_AUTH_SOCK"] is None
    assert "primary=999" in result.stdout.path.read_text(encoding="utf-8")
    assert "candidate stderr" in result.stderr.path.read_text(encoding="utf-8")


def test_runner_continuously_drains_and_bounds_both_logs(tmp_path: Path) -> None:
    byte_count = 512 * 1024
    code = f"""
import sys
sys.stdout.write('x' * {byte_count})
sys.stdout.flush()
sys.stderr.write('y' * {byte_count})
sys.stderr.flush()
"""
    spec = _spec(
        tmp_path / "bounded-logs",
        ("-c", code),
        stdout_limit=4096,
        stderr_limit=2048,
    )

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.SUCCEEDED
    assert result.stdout.retained_bytes == 4096
    assert result.stderr.retained_bytes == 2048
    assert result.stdout.observed_bytes == byte_count
    assert result.stderr.observed_bytes == byte_count
    assert result.stdout.truncated
    assert result.stderr.truncated
    assert result.stdout.path.stat().st_size == 4096
    assert result.stderr.path.stat().st_size == 2048


def test_launch_commit_failure_never_releases_candidate_code(tmp_path: Path) -> None:
    code = "from pathlib import Path; Path('candidate-ran').write_text('unsafe')"
    spec = _spec(tmp_path / "commit-failure", ("-c", code))

    def fail_commit(_record: ProcessRecord) -> None:
        raise RuntimeError("synthetic store failure")

    result = Runner().run(spec, commit_launch=fail_commit)

    assert result.outcome is ExecutionOutcome.LAUNCH_COMMIT_FAILED
    assert not result.candidate_released
    assert result.cleanup_verified
    assert not (spec.workspace / "candidate-ran").exists()
    assert not (spec.control_dir / "release.json").exists()


def test_nonzero_candidate_exit_is_process_evidence_not_a_metric(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path / "nonzero",
        ("-c", "import sys; print('GAUC=1.0'); sys.exit(7)"),
    )

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.EXIT_NONZERO
    assert result.exit_code == 7
    assert not result.succeeded
    assert result.candidate_metrics_accepted is False


def test_environment_additions_are_an_allowlist_not_inherited_shell_state(tmp_path: Path) -> None:
    root = tmp_path / "invalid-env"
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    with pytest.raises(RunnerInputError, match="not allowlisted"):
        ExecutionSpec(
            execution_id="invalid-env",
            nonce="0123456789abcdef0123456789abcdef",
            interpreter=Path(sys.executable),
            arguments=("-c", "pass"),
            workspace=workspace,
            control_dir=root / "control",
            timeout_seconds=1.0,
            memory_limit_bytes=128 * 1024 * 1024,
            workspace_disk_limit_bytes=1024 * 1024,
            stdout_limit_bytes=1024,
            stderr_limit_bytes=1024,
            threads=1,
            source_digest="1" * 64,
            config_digest="2" * 64,
            data_digest="3" * 64,
            checkpoint_digest="4" * 64,
            extra_environment=(("OPENAI_API_KEY", "secret"),),
        )


def test_mps_is_optional_recorded_device_while_interface_remains_cpu_capable(
    tmp_path: Path,
) -> None:
    spec = _spec(
        tmp_path / "mps-record",
        ("-c", "print('device metadata only')"),
        device="mps",
    )

    result = Runner().run(spec, commit_launch=lambda _record: None)

    assert result.outcome is ExecutionOutcome.SUCCEEDED
    assert result.device == "mps"
