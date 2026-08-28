"""Bounded local subprocess supervision for generated candidate workspaces.

This module is process-level robustness containment, not a hostile-code security sandbox.  It
does not claim filesystem or network isolation.  The trusted workspace policy must ensure that a
candidate sees only approved inputs and no credentials or protected outcomes.  Within that seam,
``Runner`` provides a blocked pre-exec launch handshake, a from-scratch environment, bounded
logs, resource monitoring, and identity-checked process-tree cleanup.

The runner never parses candidate stdout or accepts a candidate-produced metric.  Callers receive
only process/resource evidence and paths to bounded logs.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Final, cast

import psutil  # type: ignore[import-untyped]

RUNNER_SCHEMA_VERSION: Final = 2
_READ_CHUNK_BYTES: Final = 64 * 1024
_IDENTIFIER_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_NONCE_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EXTRA_ENVIRONMENT: Final = frozenset(
    {
        "KUAIRAND_MODE",
        "KUAIRAND_SEED",
        "KUAIRAND_SPLIT_ROLE",
    }
)
_THREAD_ENVIRONMENT: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

type LaunchCommit = Callable[[ProcessRecord], None]


class RunnerInputError(ValueError):
    """Raised before spawning when an execution specification is unsafe or inconsistent."""


class ExecutionOutcome(StrEnum):
    """Trusted runner outcome; candidate text cannot influence this value."""

    SUCCEEDED = "succeeded"
    EXIT_NONZERO = "exit_nonzero"
    TIMED_OUT = "timed_out"
    MEMORY_LIMIT = "memory_limit"
    DISK_LIMIT = "disk_limit"
    PROCESS_LIMIT = "process_limit"
    CANCELLED = "cancelled"
    ORPHANED_DESCENDANT = "orphaned_descendant"
    INSPECTION_FAILED = "inspection_failed"
    LAUNCH_COMMIT_FAILED = "launch_commit_failed"
    LAUNCHER_FAILED = "launcher_failed"
    SPAWN_FAILED = "spawn_failed"
    CLEANUP_FAILED = "cleanup_failed"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object, namespace: bytes) -> str:
    result = hashlib.sha256(namespace + b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


def _sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RunnerInputError(f"{name} must be a positive integer")
    return value


def _require_digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise RunnerInputError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _recorded_absolute_path(value: object, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value or not Path(value).is_absolute():
        raise RunnerInputError(f"{name} must be an absolute path string")
    return os.path.abspath(value)


def _expected_absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise RunnerInputError(f"{name} must be an absolute pathlib.Path")
    return Path(os.path.abspath(value))


def _require_seconds(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerInputError(f"{name} must be a finite number")
    normalized = float(value)
    minimum = 0.0 if allow_zero else 0.001
    if not math.isfinite(normalized) or normalized < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise RunnerInputError(f"{name} must be a finite {qualifier} number")
    return normalized


def _absolute_interpreter(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise RunnerInputError("interpreter must be an absolute pathlib.Path")
    # Preserve a virtual-environment symlink as the launch path: resolving it can lose pyvenv.cfg.
    normalized = Path(os.path.abspath(path))
    try:
        mode = normalized.stat().st_mode
    except OSError as exc:
        raise RunnerInputError(f"interpreter cannot be inspected: {normalized}") from exc
    if not stat.S_ISREG(mode) or not os.access(normalized, os.X_OK):
        raise RunnerInputError("interpreter must resolve to an executable regular file")
    return normalized


def active_python_interpreter() -> Path:
    """Return the active Python launch path without dereferencing a virtualenv symlink.

    CPython discovers ``pyvenv.cfg`` from the invoked executable path.  Resolving that path to
    the base interpreter before ``execve`` silently drops the active environment's site-packages.
    Process evidence still records the resolved target separately as ``interpreter_real_path``.
    """

    return _absolute_interpreter(Path(sys.executable))


def _existing_directory(path: Path, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise RunnerInputError(f"{name} must be an absolute pathlib.Path")
    if path.is_symlink():
        raise RunnerInputError(f"{name} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RunnerInputError(f"{name} cannot be resolved: {path}") from exc
    if not resolved.is_dir():
        raise RunnerInputError(f"{name} must be an existing directory")
    return resolved


def _new_control_directory(path: Path, workspace: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise RunnerInputError("control_dir must be an absolute pathlib.Path")
    if not path.name or path.name in {".", ".."}:
        raise RunnerInputError("control_dir must name one new directory")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise RunnerInputError(f"control_dir parent cannot be resolved: {path.parent}") from exc
    normalized = parent / path.name
    if normalized.exists() or normalized.is_symlink():
        raise RunnerInputError("control_dir must not already exist")
    if normalized.is_relative_to(workspace) or workspace.is_relative_to(normalized):
        raise RunnerInputError("control_dir and candidate workspace must be disjoint")
    return normalized


def _arguments(values: Sequence[object]) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized or len(normalized) > 256:
        raise RunnerInputError("arguments must contain between 1 and 256 argv entries")
    if any(type(value) is not str or not value or "\x00" in value for value in normalized):
        raise RunnerInputError("arguments must be non-empty strings without NUL bytes")
    if sum(len(cast(str, value).encode("utf-8")) for value in normalized) > 128 * 1024:
        raise RunnerInputError("encoded argument vector exceeds 128 KiB")
    return cast(tuple[str, ...], normalized)


def _extra_environment(values: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    normalized = tuple(values)
    keys: list[str] = []
    for item in normalized:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RunnerInputError("extra_environment must contain (name, value) tuples")
        key, value = item
        if key not in _SAFE_EXTRA_ENVIRONMENT:
            raise RunnerInputError(f"candidate environment variable is not allowlisted: {key!r}")
        if type(value) is not str or "\x00" in value or len(value.encode("utf-8")) > 4096:
            raise RunnerInputError(f"candidate environment value is invalid for {key}")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise RunnerInputError("extra_environment contains duplicate keys")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Complete, validated local execution request.

    ``arguments`` is the argv tail passed to the absolute Python interpreter.  It is never joined
    into a shell string.  ``control_dir`` is trusted evidence storage and must be disjoint from the
    candidate workspace.  Private HOME/TMP/XDG directories are created afresh inside the bounded
    workspace for every execution.
    """

    execution_id: str
    nonce: str
    interpreter: Path
    arguments: tuple[str, ...]
    workspace: Path
    control_dir: Path
    timeout_seconds: float
    memory_limit_bytes: int
    workspace_disk_limit_bytes: int
    stdout_limit_bytes: int
    stderr_limit_bytes: int
    threads: int
    source_digest: str
    config_digest: str
    data_digest: str
    checkpoint_digest: str
    device: str = "cpu"
    process_limit: int = 64
    poll_interval_seconds: float = 0.05
    disk_poll_interval_seconds: float = 0.25
    termination_grace_seconds: float = 0.5
    python_hash_seed: int = 0
    extra_environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.execution_id) is not str
            or _IDENTIFIER_PATTERN.fullmatch(self.execution_id) is None
        ):
            raise RunnerInputError("execution_id must use 1-128 ASCII letters, digits, '_' or '-'")
        if type(self.nonce) is not str or _NONCE_PATTERN.fullmatch(self.nonce) is None:
            raise RunnerInputError("nonce must use 16-128 ASCII letters, digits, '_' or '-'")
        interpreter = _absolute_interpreter(self.interpreter)
        workspace = _existing_directory(self.workspace, "workspace")
        control = _new_control_directory(self.control_dir, workspace)
        arguments = _arguments(self.arguments)
        if self.device not in {"cpu", "mps"}:
            raise RunnerInputError("device must be 'cpu' or 'mps'")
        if type(self.python_hash_seed) is not int or not 0 <= self.python_hash_seed <= 2**32 - 1:
            raise RunnerInputError("python_hash_seed must be a uint32-compatible integer")
        object.__setattr__(self, "interpreter", interpreter)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "control_dir", control)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(
            self,
            "timeout_seconds",
            _require_seconds(self.timeout_seconds, "timeout_seconds"),
        )
        object.__setattr__(
            self,
            "memory_limit_bytes",
            _require_positive_int(self.memory_limit_bytes, "memory_limit_bytes"),
        )
        object.__setattr__(
            self,
            "workspace_disk_limit_bytes",
            _require_positive_int(self.workspace_disk_limit_bytes, "workspace_disk_limit_bytes"),
        )
        object.__setattr__(
            self,
            "stdout_limit_bytes",
            _require_positive_int(self.stdout_limit_bytes, "stdout_limit_bytes"),
        )
        object.__setattr__(
            self,
            "stderr_limit_bytes",
            _require_positive_int(self.stderr_limit_bytes, "stderr_limit_bytes"),
        )
        object.__setattr__(self, "threads", _require_positive_int(self.threads, "threads"))
        for digest_name in (
            "source_digest",
            "config_digest",
            "data_digest",
            "checkpoint_digest",
        ):
            object.__setattr__(
                self,
                digest_name,
                _require_digest(getattr(self, digest_name), digest_name),
            )
        object.__setattr__(
            self, "process_limit", _require_positive_int(self.process_limit, "process_limit")
        )
        poll = _require_seconds(self.poll_interval_seconds, "poll_interval_seconds")
        disk_poll = _require_seconds(self.disk_poll_interval_seconds, "disk_poll_interval_seconds")
        grace = _require_seconds(
            self.termination_grace_seconds,
            "termination_grace_seconds",
            allow_zero=True,
        )
        if poll > 5.0 or disk_poll > 30.0 or grace > 30.0:
            raise RunnerInputError("runner polling and termination intervals exceed safe bounds")
        object.__setattr__(self, "poll_interval_seconds", poll)
        object.__setattr__(self, "disk_poll_interval_seconds", disk_poll)
        object.__setattr__(self, "termination_grace_seconds", grace)
        object.__setattr__(self, "extra_environment", _extra_environment(self.extra_environment))

    @property
    def command(self) -> tuple[str, ...]:
        return (str(self.interpreter), *self.arguments)


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """PID plus creation time, sufficient to reject ordinary PID reuse before signaling."""

    pid: int
    create_time: float

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise RunnerInputError("process identity pid must be positive")
        if (
            isinstance(self.create_time, bool)
            or not isinstance(self.create_time, (int, float))
            or not math.isfinite(self.create_time)
            or self.create_time <= 0.0
        ):
            raise RunnerInputError("process identity create_time must be finite and positive")
        object.__setattr__(self, "create_time", float(self.create_time))


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    """Trusted pre-release identity persisted by the launch-commit callback."""

    execution_id: str
    nonce: str
    identity: ProcessIdentity
    process_group_id: int
    command: tuple[str, ...]
    command_digest: str
    environment_digest: str
    interpreter_real_path: str
    workspace: str
    control_dir: str
    source_digest: str
    config_digest: str
    data_digest: str
    checkpoint_digest: str
    started_at_utc: str
    launcher_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.execution_id) is not str
            or _IDENTIFIER_PATTERN.fullmatch(self.execution_id) is None
        ):
            raise RunnerInputError("process execution_id is invalid")
        if type(self.nonce) is not str or _NONCE_PATTERN.fullmatch(self.nonce) is None:
            raise RunnerInputError("process nonce is invalid")
        if not isinstance(self.identity, ProcessIdentity):
            raise RunnerInputError("process identity must be ProcessIdentity")
        if type(self.process_group_id) is not int or self.process_group_id <= 0:
            raise RunnerInputError("process_group_id must be positive")
        command = _arguments(self.command)
        if not Path(command[0]).is_absolute():
            raise RunnerInputError("recorded command must start with an absolute executable")
        command_digest = _require_digest(self.command_digest, "process command_digest")
        expected_command_digest = _digest(list(command), b"kuairand-runner-command-v1")
        if command_digest != expected_command_digest:
            raise RunnerInputError("process command_digest does not match the recorded command")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "command_digest", command_digest)
        object.__setattr__(
            self,
            "environment_digest",
            _require_digest(self.environment_digest, "process environment_digest"),
        )
        object.__setattr__(
            self,
            "interpreter_real_path",
            _recorded_absolute_path(self.interpreter_real_path, "process interpreter_real_path"),
        )
        workspace = _recorded_absolute_path(self.workspace, "process workspace")
        control_dir = _recorded_absolute_path(self.control_dir, "process control_dir")
        if Path(control_dir).is_relative_to(Path(workspace)) or Path(workspace).is_relative_to(
            Path(control_dir)
        ):
            raise RunnerInputError("recorded workspace and control_dir must be disjoint")
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "control_dir", control_dir)
        for digest_name in (
            "source_digest",
            "config_digest",
            "data_digest",
            "checkpoint_digest",
        ):
            object.__setattr__(
                self,
                digest_name,
                _require_digest(getattr(self, digest_name), f"process {digest_name}"),
            )
        if type(self.started_at_utc) is not str or not self.started_at_utc:
            raise RunnerInputError("process started_at_utc must be non-empty")
        try:
            started = datetime.fromisoformat(self.started_at_utc)
        except ValueError as exc:
            raise RunnerInputError("process started_at_utc is invalid") from exc
        if started.tzinfo is None:
            raise RunnerInputError("process started_at_utc must be timezone-aware")
        object.__setattr__(
            self,
            "launcher_sha256",
            _require_digest(self.launcher_sha256, "process launcher_sha256"),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "nonce": self.nonce,
            "pid": self.identity.pid,
            "process_create_time": self.identity.create_time,
            "process_group_id": self.process_group_id,
            "command": list(self.command),
            "command_digest": self.command_digest,
            "environment_digest": self.environment_digest,
            "interpreter_real_path": self.interpreter_real_path,
            "workspace": self.workspace,
            "control_dir": self.control_dir,
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "data_digest": self.data_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "started_at_utc": self.started_at_utc,
            "launcher_sha256": self.launcher_sha256,
        }

    @property
    def digest(self) -> str:
        return _digest(self.manifest(), b"kuairand-runner-process-record-v2")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> ProcessRecord:
        """Load one exact schema-v2 process record without accepting unknown fields."""

        if not isinstance(manifest, Mapping):
            raise RunnerInputError("process manifest must be a mapping")
        expected_keys = {
            "schema_version",
            "execution_id",
            "nonce",
            "pid",
            "process_create_time",
            "process_group_id",
            "command",
            "command_digest",
            "environment_digest",
            "interpreter_real_path",
            "workspace",
            "control_dir",
            "source_digest",
            "config_digest",
            "data_digest",
            "checkpoint_digest",
            "started_at_utc",
            "launcher_sha256",
        }
        if (
            set(manifest) != expected_keys
            or manifest.get("schema_version") != RUNNER_SCHEMA_VERSION
        ):
            raise RunnerInputError("process manifest schema or fields are not exact")
        raw_command = manifest["command"]
        if not isinstance(raw_command, list):
            raise RunnerInputError("process manifest command must be a JSON array")
        try:
            return cls(
                execution_id=cast(str, manifest["execution_id"]),
                nonce=cast(str, manifest["nonce"]),
                identity=ProcessIdentity(
                    cast(int, manifest["pid"]),
                    cast(float, manifest["process_create_time"]),
                ),
                process_group_id=cast(int, manifest["process_group_id"]),
                command=cast(tuple[str, ...], tuple(raw_command)),
                command_digest=cast(str, manifest["command_digest"]),
                environment_digest=cast(str, manifest["environment_digest"]),
                interpreter_real_path=cast(str, manifest["interpreter_real_path"]),
                workspace=cast(str, manifest["workspace"]),
                control_dir=cast(str, manifest["control_dir"]),
                source_digest=cast(str, manifest["source_digest"]),
                config_digest=cast(str, manifest["config_digest"]),
                data_digest=cast(str, manifest["data_digest"]),
                checkpoint_digest=cast(str, manifest["checkpoint_digest"]),
                started_at_utc=cast(str, manifest["started_at_utc"]),
                launcher_sha256=cast(str, manifest["launcher_sha256"]),
            )
        except (TypeError, ValueError) as exc:
            raise RunnerInputError("process manifest contains invalid value types") from exc


@dataclass(frozen=True, slots=True)
class ReconciliationExpectation:
    """Controller-owned identities that a persisted process record must match exactly."""

    execution_id: str
    nonce: str
    command_digest: str
    environment_digest: str
    interpreter_real_path: Path
    workspace: Path
    control_dir: Path
    source_digest: str
    config_digest: str
    data_digest: str
    checkpoint_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.execution_id) is not str
            or _IDENTIFIER_PATTERN.fullmatch(self.execution_id) is None
        ):
            raise RunnerInputError("expected execution_id is invalid")
        if type(self.nonce) is not str or _NONCE_PATTERN.fullmatch(self.nonce) is None:
            raise RunnerInputError("expected nonce is invalid")
        for digest_name in (
            "command_digest",
            "environment_digest",
            "source_digest",
            "config_digest",
            "data_digest",
            "checkpoint_digest",
        ):
            object.__setattr__(
                self,
                digest_name,
                _require_digest(getattr(self, digest_name), f"expected {digest_name}"),
            )
        object.__setattr__(
            self,
            "interpreter_real_path",
            _expected_absolute_path(
                self.interpreter_real_path,
                "expected interpreter_real_path",
            ),
        )
        workspace = _expected_absolute_path(self.workspace, "expected workspace")
        control_dir = _expected_absolute_path(self.control_dir, "expected control_dir")
        if control_dir.is_relative_to(workspace) or workspace.is_relative_to(control_dir):
            raise RunnerInputError("expected workspace and control_dir must be disjoint")
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "control_dir", control_dir)

    def manifest(self) -> dict[str, object]:
        return {
            "execution_id": self.execution_id,
            "nonce": self.nonce,
            "command_digest": self.command_digest,
            "environment_digest": self.environment_digest,
            "interpreter_real_path": str(self.interpreter_real_path),
            "workspace": str(self.workspace),
            "control_dir": str(self.control_dir),
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "data_digest": self.data_digest,
            "checkpoint_digest": self.checkpoint_digest,
        }

    @property
    def digest(self) -> str:
        return _digest(self.manifest(), b"kuairand-runner-reconciliation-expectation-v1")


class ReconciliationOutcome(StrEnum):
    """Typed disposition of one abandoned persisted execution."""

    ALREADY_DEAD = "already_dead"
    INTERRUPTED = "interrupted"
    TERMINATED = "terminated"
    IDENTITY_MISMATCH = "identity_mismatch"
    INSPECTION_FAILED = "inspection_failed"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Immutable process evidence returned by resume reconciliation.

    ``INTERRUPTED`` means the matching tree exited after SIGTERM; ``TERMINATED`` means cleanup
    required SIGKILL.  Neither outcome adopts, resumes, or relaunches candidate code.
    """

    execution_id: str
    outcome: ReconciliationOutcome
    process_record_digest: str
    expectation_digest: str
    started_at_utc: str
    ended_at_utc: str
    wall_seconds: float
    root_identity_matched: bool
    root_was_live: bool
    signal_sent: int | None
    observed_identities: tuple[ProcessIdentity, ...]
    surviving_identities: tuple[ProcessIdentity, ...]
    cleanup_verified: bool
    detail: str

    def manifest(self) -> dict[str, object]:
        def identity_manifest(identity: ProcessIdentity) -> dict[str, object]:
            return {"pid": identity.pid, "process_create_time": identity.create_time}

        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "outcome": self.outcome.value,
            "process_record_digest": self.process_record_digest,
            "expectation_digest": self.expectation_digest,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "wall_seconds": self.wall_seconds,
            "root_identity_matched": self.root_identity_matched,
            "root_was_live": self.root_was_live,
            "signal_sent": self.signal_sent,
            "observed_identities": [
                identity_manifest(identity) for identity in self.observed_identities
            ],
            "surviving_identities": [
                identity_manifest(identity) for identity in self.surviving_identities
            ],
            "cleanup_verified": self.cleanup_verified,
            "detail": self.detail,
            "candidate_adopted": False,
            "candidate_relaunched": False,
        }

    @property
    def digest(self) -> str:
        return _digest(self.manifest(), b"kuairand-runner-reconciliation-result-v1")


@dataclass(frozen=True, slots=True)
class LogEvidence:
    path: Path
    retained_bytes: int
    observed_bytes: int
    truncated: bool
    sha256: str

    def manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "retained_bytes": self.retained_bytes,
            "observed_bytes": self.observed_bytes,
            "truncated": self.truncated,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Process-only evidence.  Candidate metrics are intentionally absent."""

    execution_id: str
    outcome: ExecutionOutcome
    process: ProcessRecord | None
    candidate_released: bool
    exit_code: int | None
    terminating_signal: int | None
    started_at_utc: str
    ended_at_utc: str
    wall_seconds: float
    peak_tree_rss_bytes: int
    peak_workspace_bytes: int
    peak_process_count: int
    stdout: LogEvidence
    stderr: LogEvidence
    cleanup_verified: bool
    device: str
    threads: int
    detail: str | None = None
    candidate_metrics_accepted: bool = field(init=False, default=False)

    @property
    def succeeded(self) -> bool:
        return self.outcome is ExecutionOutcome.SUCCEEDED

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "outcome": self.outcome.value,
            "process": None if self.process is None else self.process.manifest(),
            "candidate_released": self.candidate_released,
            "exit_code": self.exit_code,
            "terminating_signal": self.terminating_signal,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "wall_seconds": self.wall_seconds,
            "peak_tree_rss_bytes": self.peak_tree_rss_bytes,
            "peak_workspace_bytes": self.peak_workspace_bytes,
            "peak_process_count": self.peak_process_count,
            "stdout": self.stdout.manifest(),
            "stderr": self.stderr.manifest(),
            "cleanup_verified": self.cleanup_verified,
            "device": self.device,
            "threads": self.threads,
            "detail": self.detail,
            "candidate_metrics_accepted": False,
        }


class _BoundedCapture:
    def __init__(self, source: IO[bytes], path: Path, limit: int) -> None:
        self._source = source
        self._path = path
        self._limit = limit
        self._observed = 0
        self._retained = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._drain,
            name=f"capture-{path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _drain(self) -> None:
        try:
            descriptor = os.open(
                self._path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as destination:
                while chunk := self._source.read(_READ_CHUNK_BYTES):
                    self._observed += len(chunk)
                    remaining = self._limit - self._retained
                    if remaining > 0:
                        kept = chunk[:remaining]
                        destination.write(kept)
                        self._retained += len(kept)
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException as exc:  # retained and converted into trusted runner failure
            self._error = exc
        finally:
            with suppress(OSError):
                self._source.close()

    def finish(self, timeout: float) -> tuple[LogEvidence, bool]:
        self._thread.join(timeout)
        if self._thread.is_alive():
            with suppress(OSError):
                self._source.close()
            self._thread.join(timeout)
        complete = not self._thread.is_alive() and self._error is None
        if not self._path.exists():
            self._path.touch(mode=0o600, exist_ok=False)
        evidence = LogEvidence(
            path=self._path,
            retained_bytes=self._path.stat().st_size,
            observed_bytes=self._observed,
            truncated=self._observed > self._limit,
            sha256=_sha256_file(self._path),
        )
        return evidence, complete


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_json(dict(value)) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _workspace_size(root: Path) -> int:
    total = 0
    seen_regular_files: set[tuple[int, int]] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                if stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(mode):
                    identity = (metadata.st_dev, metadata.st_ino)
                    if identity not in seen_regular_files:
                        seen_regular_files.add(identity)
                        total += metadata.st_size
                else:
                    # Count the directory-entry payload for links/special files without following
                    # it.  WorkspacePolicy separately rejects undeclared entry types.
                    total += metadata.st_size
    return total


def _identity_process(identity: ProcessIdentity) -> psutil.Process | None:
    try:
        process = psutil.Process(identity.pid)
        if process.create_time() != identity.create_time:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process if process.is_running() else None
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except (psutil.AccessDenied, PermissionError):
        return None


def _identity_matches(identity: ProcessIdentity) -> bool:
    """Return whether a PID still denotes the exact recorded process, zombies included."""

    try:
        return bool(psutil.Process(identity.pid).create_time() == identity.create_time)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, PermissionError):
        return False


def _darwin_child_pids(parent_pid: int) -> tuple[int, ...]:
    """List one process's direct children without macOS-wide PID enumeration.

    ``psutil.Process.children`` calls ``psutil.pids`` on macOS.  That global sysctl is denied by
    several legitimate local containment profiles even when inspecting our own child is allowed.
    ``proc_listchildpids`` is the narrower kernel interface intended for this exact query.
    """

    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    list_children = library.proc_listchildpids
    list_children.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    list_children.restype = ctypes.c_int
    capacity = 32
    while capacity <= 65_536:
        values = (ctypes.c_int * capacity)()
        ctypes.set_errno(0)
        count = int(
            list_children(
                parent_pid,
                ctypes.cast(values, ctypes.c_void_p),
                ctypes.sizeof(values),
            )
        )
        if count < 0:
            error_number = ctypes.get_errno()
            if error_number == errno.ESRCH:
                return ()
            if error_number in {errno.EACCES, errno.EPERM}:
                raise PermissionError(error_number, os.strerror(error_number))
            raise OSError(error_number, os.strerror(error_number))
        if count < capacity:
            return tuple(int(values[index]) for index in range(count) if values[index] > 0)
        capacity *= 2
    raise OSError(errno.EOVERFLOW, "direct child list exceeded runner safety bound")


def _direct_children(process: psutil.Process) -> tuple[psutil.Process, ...]:
    if sys.platform != "darwin":
        return tuple(process.children(recursive=False))
    children: list[psutil.Process] = []
    for pid in _darwin_child_pids(process.pid):
        try:
            child = psutil.Process(pid)
            # Reject an ordinary PID-reuse race between the kernel listing and Process creation.
            if child.create_time() + 1.0 < process.create_time():
                continue
            children.append(child)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return tuple(children)


class _ProcessTracker:
    def __init__(self, root: ProcessIdentity, process_group_id: int) -> None:
        self._identities: dict[int, ProcessIdentity] = {root.pid: root}
        self._root = root
        self._process_group_id = process_group_id
        self.inspection_failed = False

    @property
    def root(self) -> ProcessIdentity:
        return self._root

    def _remember(self, process: psutil.Process) -> ProcessIdentity | None:
        try:
            identity = ProcessIdentity(process.pid, process.create_time())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None
        except (psutil.AccessDenied, PermissionError):
            self.inspection_failed = True
            return None
        previous = self._identities.get(identity.pid)
        if previous is None or previous.create_time == identity.create_time:
            self._identities[identity.pid] = identity
            return identity
        return None

    def refresh(self) -> None:
        # Refresh from every known live process so a descendant that called setsid remains tracked
        # after its original parent exits.
        pending = list(self._identities.values())
        visited: set[tuple[int, float]] = set()
        while pending:
            identity = pending.pop()
            identity_key = (identity.pid, identity.create_time)
            if identity_key in visited:
                continue
            visited.add(identity_key)
            process = _identity_process(identity)
            if process is None:
                continue
            try:
                for child in _direct_children(process):
                    child_identity = self._remember(child)
                    if child_identity is not None:
                        pending.append(child_identity)
            except (psutil.AccessDenied, PermissionError):
                self.inspection_failed = True
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except OSError:
                self.inspection_failed = True

    def add_process_group_members(self) -> None:
        # Every process that can join the freshly-created candidate group is a descendant.  A
        # direct-child walk is therefore both narrower and more reliable than global PID scanning;
        # killpg covers a short-lived in-group child that races between two observations.
        self.refresh()

    def identities(self) -> tuple[ProcessIdentity, ...]:
        return tuple(self._identities.values())

    def live(self) -> tuple[tuple[ProcessIdentity, psutil.Process], ...]:
        result: list[tuple[ProcessIdentity, psutil.Process]] = []
        for identity in self._identities.values():
            process = _identity_process(identity)
            if process is not None:
                result.append((identity, process))
        return tuple(result)

    def sample(self) -> tuple[int, int]:
        self.refresh()
        rss = 0
        count = 0
        for _identity, process in self.live():
            try:
                rss += int(process.memory_info().rss)
                count += 1
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, PermissionError):
                self.inspection_failed = True
        return rss, count


@dataclass(frozen=True, slots=True)
class _CleanupResult:
    verified: bool
    survivors: tuple[int, ...]
    signal_sent: int | None = None


def _signal_individuals(
    processes: Sequence[tuple[ProcessIdentity, psutil.Process]],
    *,
    force: bool,
    pgid: int,
    group_signaled: bool,
) -> bool:
    signaled = False
    for identity, _process in processes:
        current = _identity_process(identity)
        if current is None:
            continue
        try:
            # Processes still in the isolated group were already covered by killpg.  Signal only
            # verified escaped descendants individually.
            if group_signaled and os.getpgid(identity.pid) == pgid:
                continue
        except OSError:
            pass
        try:
            current.kill() if force else current.terminate()
            signaled = True
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, PermissionError):
            continue
    return signaled


def _validated_group_signal(
    record: ProcessRecord,
    tracker: _ProcessTracker,
    sig: signal.Signals,
) -> bool:
    # Never signal a bare PGID.  First prove that at least one identity we observed in this launch
    # still exists and belongs to the recorded group.  This remains safe when the root is a zombie
    # or has exited but a tracked child keeps the group alive.
    for identity in tracker.identities():
        if not _identity_matches(identity):
            continue
        try:
            if os.getpgid(identity.pid) != record.process_group_id:
                continue
            os.killpg(record.process_group_id, sig)
            return True
        except (OSError, ProcessLookupError):
            continue
    return False


def _wait_for_tree_exit(tracker: _ProcessTracker, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        tracker.refresh()
        tracker.add_process_group_members()
        if not tracker.live():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.025, remaining))


def _terminate_tree(
    record: ProcessRecord, tracker: _ProcessTracker, grace: float
) -> _CleanupResult:
    tracker.refresh()
    tracker.add_process_group_members()
    term_group_signaled = _validated_group_signal(record, tracker, signal.SIGTERM)
    signal_sent: int | None = signal.SIGTERM if term_group_signaled else None
    term_individual_signaled = _signal_individuals(
        tracker.live(),
        force=False,
        pgid=record.process_group_id,
        group_signaled=term_group_signaled,
    )
    if term_individual_signaled:
        signal_sent = signal.SIGTERM
    if not _wait_for_tree_exit(tracker, grace):
        kill_group_signaled = _validated_group_signal(record, tracker, signal.SIGKILL)
        if kill_group_signaled:
            signal_sent = signal.SIGKILL
        tracker.refresh()
        tracker.add_process_group_members()
        kill_individual_signaled = _signal_individuals(
            tracker.live(),
            force=True,
            pgid=record.process_group_id,
            group_signaled=kill_group_signaled,
        )
        if kill_individual_signaled:
            signal_sent = signal.SIGKILL
        _wait_for_tree_exit(tracker, max(grace, 0.25))
    tracker.refresh()
    tracker.add_process_group_members()
    survivors = tuple(sorted(identity.pid for identity, _process in tracker.live()))
    return _CleanupResult(
        verified=not survivors and not tracker.inspection_failed,
        survivors=survivors,
        signal_sent=signal_sent,
    )


def _environment(spec: ExecutionSpec, runtime_root: Path) -> dict[str, str]:
    home = runtime_root / "home"
    temporary = runtime_root / "tmp"
    xdg_cache = runtime_root / "xdg-cache"
    xdg_config = runtime_root / "xdg-config"
    xdg_data = runtime_root / "xdg-data"
    matplotlib = runtime_root / "matplotlib"
    for directory in (home, temporary, xdg_cache, xdg_config, xdg_data, matplotlib):
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)

    thread_value = str(spec.threads)
    environment = {
        "PATH": os.pathsep.join((str(spec.interpreter.parent), "/usr/bin", "/bin")),
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "TMP": str(temporary),
        "TEMP": str(temporary),
        "XDG_CACHE_HOME": str(xdg_cache),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_DATA_HOME": str(xdg_data),
        "MPLCONFIGDIR": str(matplotlib),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONHASHSEED": str(spec.python_hash_seed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TOKENIZERS_PARALLELISM": "false",
        "KUAIRAND_EXECUTION_ID": spec.execution_id,
        "KUAIRAND_EXECUTION_NONCE": spec.nonce,
        "KUAIRAND_DEVICE": spec.device,
    }
    environment.update({name: thread_value for name in _THREAD_ENVIRONMENT})
    environment.update(dict(spec.extra_environment))
    return environment


def _empty_log(path: Path) -> LogEvidence:
    if not path.exists():
        path.touch(mode=0o600, exist_ok=False)
    return LogEvidence(path, 0, 0, False, _sha256_file(path))


def _reconciliation_mismatches(
    record: ProcessRecord,
    expected: ReconciliationExpectation,
) -> tuple[str, ...]:
    comparisons = {
        "execution_id": (record.execution_id, expected.execution_id),
        "nonce": (record.nonce, expected.nonce),
        "command_digest": (record.command_digest, expected.command_digest),
        "environment_digest": (record.environment_digest, expected.environment_digest),
        "interpreter_real_path": (
            record.interpreter_real_path,
            str(expected.interpreter_real_path),
        ),
        "workspace": (record.workspace, str(expected.workspace)),
        "control_dir": (record.control_dir, str(expected.control_dir)),
        "source_digest": (record.source_digest, expected.source_digest),
        "config_digest": (record.config_digest, expected.config_digest),
        "data_digest": (record.data_digest, expected.data_digest),
        "checkpoint_digest": (record.checkpoint_digest, expected.checkpoint_digest),
    }
    return tuple(name for name, (actual, wanted) in comparisons.items() if actual != wanted)


def _reconciliation_result(
    *,
    record: ProcessRecord,
    expected: ReconciliationExpectation,
    outcome: ReconciliationOutcome,
    started_at_utc: str,
    started: float,
    root_identity_matched: bool,
    root_was_live: bool,
    signal_sent: int | None,
    observed_identities: Sequence[ProcessIdentity] = (),
    surviving_identities: Sequence[ProcessIdentity] = (),
    cleanup_verified: bool,
    detail: str,
) -> ReconciliationResult:
    return ReconciliationResult(
        execution_id=record.execution_id,
        outcome=outcome,
        process_record_digest=record.digest,
        expectation_digest=expected.digest,
        started_at_utc=started_at_utc,
        ended_at_utc=_utc_now(),
        wall_seconds=time.monotonic() - started,
        root_identity_matched=root_identity_matched,
        root_was_live=root_was_live,
        signal_sent=signal_sent,
        observed_identities=tuple(observed_identities),
        surviving_identities=tuple(surviving_identities),
        cleanup_verified=cleanup_verified,
        detail=detail,
    )


class Runner:
    """Execute one candidate behind a durable pre-release launch seam.

    ``commit_launch`` must durably charge the launch and persist ``ProcessRecord``.  Candidate code
    is still blocked in the private launcher when the callback runs.  If it raises, the release
    pipe is closed and candidate code is never executed; callers should conservatively retain any
    charge whose transaction outcome is uncertain.
    """

    def reconcile(
        self,
        record: ProcessRecord,
        *,
        expected: ReconciliationExpectation,
        termination_grace_seconds: float = 0.5,
    ) -> ReconciliationResult:
        """Interrupt one exact abandoned tree without adopting or relaunching it.

        Invalid caller objects raise ``RunnerInputError`` before process inspection.  After valid
        objects are accepted, all process races, permission failures, mismatches, and cleanup
        failures return typed immutable evidence.  No signal is sent until the complete persisted
        scientific identity matches and the live PID, creation time, and PGID all match.
        """

        if not isinstance(record, ProcessRecord):
            raise RunnerInputError("record must be ProcessRecord")
        if not isinstance(expected, ReconciliationExpectation):
            raise RunnerInputError("expected must be ReconciliationExpectation")
        grace = _require_seconds(
            termination_grace_seconds,
            "termination_grace_seconds",
            allow_zero=True,
        )
        if grace > 30.0:
            raise RunnerInputError("termination_grace_seconds exceeds the 30-second safety bound")

        started_at_utc = _utc_now()
        started = time.monotonic()
        mismatches = _reconciliation_mismatches(record, expected)
        if mismatches:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.IDENTITY_MISMATCH,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=False,
                root_was_live=False,
                signal_sent=None,
                cleanup_verified=False,
                detail=f"persisted identities differ: {', '.join(mismatches)}",
            )

        try:
            live_root = psutil.Process(record.identity.pid)
            live_create_time = float(live_root.create_time())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.ALREADY_DEAD,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=False,
                root_was_live=False,
                signal_sent=None,
                cleanup_verified=False,
                detail="recorded root is absent; cold descendant absence cannot be proven",
            )
        except (psutil.AccessDenied, PermissionError, OSError) as exc:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.INSPECTION_FAILED,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=False,
                root_was_live=False,
                signal_sent=None,
                cleanup_verified=False,
                detail=f"recorded PID identity inspection failed: {type(exc).__name__}",
            )

        if live_create_time != record.identity.create_time:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.IDENTITY_MISMATCH,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=False,
                root_was_live=True,
                signal_sent=None,
                cleanup_verified=False,
                detail="recorded PID has been reused by a different process",
            )

        try:
            live_group_id = os.getpgid(record.identity.pid)
        except (OSError, ProcessLookupError) as exc:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.INSPECTION_FAILED,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=True,
                root_was_live=False,
                signal_sent=None,
                observed_identities=(record.identity,),
                cleanup_verified=False,
                detail=f"recorded process-group inspection failed: {type(exc).__name__}",
            )
        if live_group_id != record.process_group_id:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.IDENTITY_MISMATCH,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=True,
                root_was_live=True,
                signal_sent=None,
                observed_identities=(record.identity,),
                cleanup_verified=False,
                detail="recorded process group no longer matches the exact live process",
            )

        try:
            root_was_live = bool(
                live_root.is_running() and live_root.status() != psutil.STATUS_ZOMBIE
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            root_was_live = False
        except (psutil.AccessDenied, PermissionError, OSError):
            root_was_live = True

        if not root_was_live:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.ALREADY_DEAD,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=True,
                root_was_live=False,
                signal_sent=None,
                observed_identities=(record.identity,),
                cleanup_verified=False,
                detail="recorded root is no longer live; cold descendant absence cannot be proven",
            )

        tracker = _ProcessTracker(record.identity, record.process_group_id)
        try:
            tracker.refresh()
        except (OSError, psutil.Error):
            tracker.inspection_failed = True
        cleanup = _terminate_tree(record, tracker, grace)
        observed = tuple(
            sorted(tracker.identities(), key=lambda item: (item.pid, item.create_time))
        )
        survivors = tuple(
            sorted(
                (identity for identity, _process in tracker.live()),
                key=lambda item: (item.pid, item.create_time),
            )
        )
        signal_sent = cleanup.signal_sent
        if tracker.inspection_failed:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.INSPECTION_FAILED,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=True,
                root_was_live=True,
                signal_sent=signal_sent,
                observed_identities=observed,
                surviving_identities=survivors,
                cleanup_verified=False,
                detail="matching group received best-effort cleanup after tree inspection failed",
            )
        if not cleanup.verified:
            return _reconciliation_result(
                record=record,
                expected=expected,
                outcome=ReconciliationOutcome.CLEANUP_FAILED,
                started_at_utc=started_at_utc,
                started=started,
                root_identity_matched=True,
                root_was_live=True,
                signal_sent=signal_sent,
                observed_identities=observed,
                surviving_identities=survivors,
                cleanup_verified=False,
                detail="matching abandoned process tree survived bounded cleanup",
            )
        outcome = (
            ReconciliationOutcome.TERMINATED
            if signal_sent == signal.SIGKILL
            else ReconciliationOutcome.INTERRUPTED
        )
        detail = (
            "matching abandoned process tree required SIGKILL"
            if outcome is ReconciliationOutcome.TERMINATED
            else "matching abandoned process tree exited after bounded interruption"
        )
        return _reconciliation_result(
            record=record,
            expected=expected,
            outcome=outcome,
            started_at_utc=started_at_utc,
            started=started,
            root_identity_matched=True,
            root_was_live=True,
            signal_sent=signal_sent,
            observed_identities=observed,
            surviving_identities=survivors,
            cleanup_verified=True,
            detail=detail,
        )

    def run(
        self,
        spec: ExecutionSpec,
        *,
        commit_launch: LaunchCommit,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionResult:
        if not isinstance(spec, ExecutionSpec):
            raise RunnerInputError("spec must be ExecutionSpec")
        if not callable(commit_launch):
            raise RunnerInputError("commit_launch must be a callable durable launch seam")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise RunnerInputError("cancel_event must be threading.Event or None")
        if os.name != "posix":
            raise RunnerInputError("the local runner currently requires POSIX process groups")

        spec.control_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        stdout_path = spec.control_dir / "stdout.log"
        stderr_path = spec.control_dir / "stderr.log"
        runtime_root = spec.workspace / f".runner-runtime-{spec.nonce}"
        runtime_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        environment = _environment(spec, runtime_root)
        environment_digest = _digest(environment, b"kuairand-runner-environment-v1")
        command_digest = _digest(list(spec.command), b"kuairand-runner-command-v1")
        launcher = Path(__file__).with_name("_launcher.py").resolve(strict=True)
        launcher_sha256 = _sha256_file(launcher)
        initial_workspace_bytes = _workspace_size(spec.workspace)
        if initial_workspace_bytes > spec.workspace_disk_limit_bytes:
            raise RunnerInputError(
                "workspace already exceeds workspace_disk_limit_bytes before launch"
            )

        started_at_utc = _utc_now()
        started = time.monotonic()
        if cancel_event is not None and cancel_event.is_set():
            return self._early_result(
                spec=spec,
                outcome=ExecutionOutcome.CANCELLED,
                detail="trusted controller requested cancellation before launch admission",
                started_at_utc=started_at_utc,
                started=started,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                initial_workspace_bytes=initial_workspace_bytes,
            )
        read_fd, write_fd = os.pipe()
        process: subprocess.Popen[bytes] | None = None
        record: ProcessRecord | None = None
        stdout_capture: _BoundedCapture | None = None
        stderr_capture: _BoundedCapture | None = None
        tracker: _ProcessTracker | None = None
        candidate_released = False
        outcome: ExecutionOutcome | None = None
        detail: str | None = None
        peak_rss = 0
        peak_workspace = initial_workspace_bytes
        peak_process_count = 0
        cleanup = _CleanupResult(verified=True, survivors=())

        try:
            launcher_command = (
                str(spec.interpreter),
                "-I",
                str(launcher),
                str(read_fd),
                spec.nonce,
                str(spec.interpreter),
                *spec.arguments,
            )
            try:
                process = subprocess.Popen(
                    launcher_command,
                    cwd=spec.workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    close_fds=True,
                    pass_fds=(read_fd,),
                    start_new_session=True,
                    restore_signals=True,
                )
            except OSError as exc:
                outcome = ExecutionOutcome.SPAWN_FAILED
                detail = f"launcher spawn failed: {type(exc).__name__}"
                return self._early_result(
                    spec=spec,
                    outcome=outcome,
                    detail=detail,
                    started_at_utc=started_at_utc,
                    started=started,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    initial_workspace_bytes=initial_workspace_bytes,
                )
            finally:
                os.close(read_fd)
                read_fd = -1

            assert process.stdout is not None and process.stderr is not None
            stdout_capture = _BoundedCapture(process.stdout, stdout_path, spec.stdout_limit_bytes)
            stderr_capture = _BoundedCapture(process.stderr, stderr_path, spec.stderr_limit_bytes)
            stdout_capture.start()
            stderr_capture.start()

            ps_process = psutil.Process(process.pid)
            identity = ProcessIdentity(process.pid, ps_process.create_time())
            process_group_id = os.getpgid(process.pid)
            if process_group_id != process.pid:
                outcome = ExecutionOutcome.LAUNCHER_FAILED
                detail = "launcher did not create the required isolated process group"
            else:
                record = ProcessRecord(
                    execution_id=spec.execution_id,
                    nonce=spec.nonce,
                    identity=identity,
                    process_group_id=process_group_id,
                    command=spec.command,
                    command_digest=command_digest,
                    environment_digest=environment_digest,
                    interpreter_real_path=str(spec.interpreter.resolve(strict=True)),
                    workspace=str(spec.workspace),
                    control_dir=str(spec.control_dir),
                    source_digest=spec.source_digest,
                    config_digest=spec.config_digest,
                    data_digest=spec.data_digest,
                    checkpoint_digest=spec.checkpoint_digest,
                    started_at_utc=started_at_utc,
                    launcher_sha256=launcher_sha256,
                )
                tracker = _ProcessTracker(identity, process_group_id)
                _atomic_json(spec.control_dir / "process.json", record.manifest())

            if outcome is None and cancel_event is not None and cancel_event.is_set():
                outcome = ExecutionOutcome.CANCELLED
                detail = "trusted controller requested cancellation before launch commit"

            if outcome is None:
                assert record is not None
                try:
                    commit_launch(record)
                except BaseException as exc:
                    outcome = ExecutionOutcome.LAUNCH_COMMIT_FAILED
                    detail = f"launch commit callback failed: {type(exc).__name__}"
                if outcome is None and process.poll() is not None:
                    outcome = ExecutionOutcome.LAUNCHER_FAILED
                    detail = "blocked launcher exited before durable release"

            if outcome is None and cancel_event is not None and cancel_event.is_set():
                outcome = ExecutionOutcome.CANCELLED
                detail = "trusted controller requested cancellation before candidate release"

            if outcome is None:
                release = f"{spec.nonce}\n".encode("ascii")
                try:
                    written = os.write(write_fd, release)
                    if written != len(release):
                        raise OSError("short release-pipe write")
                    candidate_released = True
                    _atomic_json(
                        spec.control_dir / "release.json",
                        {
                            "schema_version": RUNNER_SCHEMA_VERSION,
                            "execution_id": spec.execution_id,
                            "nonce": spec.nonce,
                            "released_at_utc": _utc_now(),
                        },
                    )
                except OSError as exc:
                    outcome = ExecutionOutcome.LAUNCHER_FAILED
                    detail = f"candidate release failed: {type(exc).__name__}"
            os.close(write_fd)
            write_fd = -1

            last_disk_poll = started
            while outcome is None:
                assert process is not None and tracker is not None
                rss, process_count = tracker.sample()
                peak_rss = max(peak_rss, rss)
                peak_process_count = max(peak_process_count, process_count)
                now = time.monotonic()
                if tracker.inspection_failed:
                    outcome = ExecutionOutcome.INSPECTION_FAILED
                    detail = "process-tree inspection was denied"
                elif rss > spec.memory_limit_bytes:
                    outcome = ExecutionOutcome.MEMORY_LIMIT
                    detail = "aggregate process-tree RSS exceeded the configured limit"
                elif process_count > spec.process_limit:
                    outcome = ExecutionOutcome.PROCESS_LIMIT
                    detail = "candidate process count exceeded the configured limit"
                elif now - last_disk_poll >= spec.disk_poll_interval_seconds:
                    try:
                        workspace_bytes = _workspace_size(spec.workspace)
                    except OSError as exc:
                        outcome = ExecutionOutcome.INSPECTION_FAILED
                        detail = f"workspace inspection failed: {type(exc).__name__}"
                    else:
                        peak_workspace = max(peak_workspace, workspace_bytes)
                        last_disk_poll = now
                        if workspace_bytes > spec.workspace_disk_limit_bytes:
                            outcome = ExecutionOutcome.DISK_LIMIT
                            detail = "workspace bytes exceeded the configured limit"
                if outcome is not None:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    outcome = ExecutionOutcome.CANCELLED
                    detail = "trusted controller requested cancellation"
                    break
                if now - started >= spec.timeout_seconds:
                    outcome = ExecutionOutcome.TIMED_OUT
                    detail = "wall-clock timeout exceeded"
                    break
                return_code = process.poll()
                if return_code is not None:
                    tracker.refresh()
                    tracker.add_process_group_members()
                    descendants = [
                        identity
                        for identity, _descendant in tracker.live()
                        if identity != tracker.root
                    ]
                    if descendants:
                        outcome = ExecutionOutcome.ORPHANED_DESCENDANT
                        detail = "candidate parent exited while a descendant remained alive"
                    else:
                        outcome = (
                            ExecutionOutcome.SUCCEEDED
                            if return_code == 0
                            else ExecutionOutcome.EXIT_NONZERO
                        )
                    break
                if cancel_event is None:
                    time.sleep(spec.poll_interval_seconds)
                else:
                    cancel_event.wait(spec.poll_interval_seconds)

        except KeyboardInterrupt:
            outcome = ExecutionOutcome.CANCELLED
            detail = "runner interrupted by KeyboardInterrupt"
        except (OSError, psutil.Error) as exc:
            outcome = ExecutionOutcome.INSPECTION_FAILED
            detail = f"runner supervision failed: {type(exc).__name__}"
        finally:
            if read_fd >= 0:
                os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            if process is not None and record is not None and tracker is not None:
                should_terminate = outcome not in {
                    ExecutionOutcome.SUCCEEDED,
                    ExecutionOutcome.EXIT_NONZERO,
                }
                if should_terminate or process.poll() is None:
                    cleanup = _terminate_tree(record, tracker, spec.termination_grace_seconds)
                else:
                    tracker.refresh()
                    tracker.add_process_group_members()
                    live = tracker.live()
                    cleanup = _CleanupResult(
                        verified=not live and not tracker.inspection_failed,
                        survivors=tuple(identity.pid for identity, _item in live),
                    )
                try:
                    process.wait(timeout=max(spec.termination_grace_seconds, 0.25))
                except subprocess.TimeoutExpired:
                    cleanup = _terminate_tree(record, tracker, spec.termination_grace_seconds)
                    try:
                        process.wait(timeout=max(spec.termination_grace_seconds, 0.25))
                    except subprocess.TimeoutExpired:
                        cleanup = _CleanupResult(False, (process.pid,))
            elif process is not None and process.poll() is None:
                # No validated record means group signaling is forbidden.  The exact Popen PID is
                # still the child we just created, so terminate it directly and do not touch any
                # other PID.
                process.terminate()
                try:
                    process.wait(timeout=max(spec.termination_grace_seconds, 0.25))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

        if not cleanup.verified:
            if outcome is ExecutionOutcome.INSPECTION_FAILED:
                detail = "process-tree inspection was denied; cleanup could not be fully verified"
            else:
                outcome = ExecutionOutcome.CLEANUP_FAILED
                detail = "process-tree cleanup could not be verified"

        return_code = process.poll() if process is not None else None
        terminating_signal = -return_code if return_code is not None and return_code < 0 else None
        exit_code = return_code if return_code is not None and return_code >= 0 else None
        if stdout_capture is None:
            stdout_evidence, stdout_complete = _empty_log(stdout_path), True
        else:
            stdout_evidence, stdout_complete = stdout_capture.finish(2.0)
        if stderr_capture is None:
            stderr_evidence, stderr_complete = _empty_log(stderr_path), True
        else:
            stderr_evidence, stderr_complete = stderr_capture.finish(2.0)
        if not stdout_complete or not stderr_complete:
            outcome = ExecutionOutcome.CLEANUP_FAILED
            cleanup = _CleanupResult(False, cleanup.survivors)
            detail = "bounded log drains did not finish cleanly"
        try:
            final_workspace = _workspace_size(spec.workspace)
        except OSError:
            final_workspace = peak_workspace
        peak_workspace = max(peak_workspace, final_workspace)

        result = ExecutionResult(
            execution_id=spec.execution_id,
            outcome=outcome,
            process=record,
            candidate_released=candidate_released,
            exit_code=exit_code,
            terminating_signal=terminating_signal,
            started_at_utc=started_at_utc,
            ended_at_utc=_utc_now(),
            wall_seconds=time.monotonic() - started,
            peak_tree_rss_bytes=peak_rss,
            peak_workspace_bytes=peak_workspace,
            peak_process_count=peak_process_count,
            stdout=stdout_evidence,
            stderr=stderr_evidence,
            cleanup_verified=cleanup.verified,
            device=spec.device,
            threads=spec.threads,
            detail=detail,
        )
        _atomic_json(spec.control_dir / "result.json", result.manifest())
        return result

    @staticmethod
    def _early_result(
        *,
        spec: ExecutionSpec,
        outcome: ExecutionOutcome,
        detail: str,
        started_at_utc: str,
        started: float,
        stdout_path: Path,
        stderr_path: Path,
        initial_workspace_bytes: int,
    ) -> ExecutionResult:
        result = ExecutionResult(
            execution_id=spec.execution_id,
            outcome=outcome,
            process=None,
            candidate_released=False,
            exit_code=None,
            terminating_signal=None,
            started_at_utc=started_at_utc,
            ended_at_utc=_utc_now(),
            wall_seconds=time.monotonic() - started,
            peak_tree_rss_bytes=0,
            peak_workspace_bytes=initial_workspace_bytes,
            peak_process_count=0,
            stdout=_empty_log(stdout_path),
            stderr=_empty_log(stderr_path),
            cleanup_verified=True,
            device=spec.device,
            threads=spec.threads,
            detail=detail,
        )
        _atomic_json(spec.control_dir / "result.json", result.manifest())
        return result


__all__ = [
    "RUNNER_SCHEMA_VERSION",
    "ExecutionOutcome",
    "ExecutionResult",
    "ExecutionSpec",
    "LaunchCommit",
    "LogEvidence",
    "ProcessIdentity",
    "ProcessRecord",
    "ReconciliationExpectation",
    "ReconciliationOutcome",
    "ReconciliationResult",
    "Runner",
    "RunnerInputError",
    "active_python_interpreter",
]
