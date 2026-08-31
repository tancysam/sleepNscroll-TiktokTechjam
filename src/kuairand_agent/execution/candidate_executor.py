"""Trusted local train/predict execution for self-contained generated candidates.

This module is the narrow seam between generated source and the durable campaign controller.  It
materializes only controller-approved numeric capabilities, runs the source in a fresh supervised
process, validates the exact candidate protocol, and commits immutable output evidence.  Metrics
and labels never cross the prediction side of this seam.

The checkpoint passed to prediction is a verified content-addressed object.  It is deliberately
not represented as a data capability: its digest is bound independently in the command, runner
receipt, prediction request, and validated prediction manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from kuairand_agent.candidate_api.protocol import (
    PREDICTION_RESULT_FILENAME,
    SCORES_DTYPE,
    TRAIN_RESULT_FILENAME,
    PredictionExpectation,
    TrainExpectation,
    validate_prediction_outputs,
    validate_train_outputs,
)
from kuairand_agent.candidate_api.runtime_contract import CANDIDATE_RUNTIME_CONTRACT
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
)
from kuairand_agent.execution.policy import (
    ApprovedInput,
    CandidateInputRole,
    SplitRole,
)
from kuairand_agent.execution.runner import (
    ExecutionResult,
    ExecutionSpec,
    ProcessRecord,
    Runner,
)
from kuairand_agent.execution.workspace import (
    CandidateWorkspace,
    WorkspaceMaterializer,
    WorkspaceSpec,
)

_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_TRAIN_ROLES: Final = frozenset({SplitRole.TRAIN, SplitRole.INNER_TRAIN})
_PREDICTION_INPUT_ROLE: Final = {
    SplitRole.INNER_VALID: CandidateInputRole.INNER_VALID_INPUTS,
    SplitRole.OUTER_VALID: CandidateInputRole.OUTER_VALID_INPUTS,
    SplitRole.FINAL: CandidateInputRole.FINAL_INPUTS,
}
NO_CHECKPOINT_DIGEST: Final = hashlib.sha256(b"kuairand-no-checkpoint-v1\0").hexdigest()


class CandidateExecutionError(RuntimeError):
    """A generated execution failed supervision or its output contract."""

    def __init__(
        self,
        message: str,
        *,
        result: ExecutionResult | None = None,
        artifacts: CandidateExecutionArtifacts | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.artifacts = artifacts


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _token(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > 256
        or not value.isascii()
        or not value.isprintable()
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a printable ASCII token without whitespace")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _artifact_closure(entries: Sequence[tuple[str, ArtifactRef]]) -> str:
    payload = [
        {"role": role, **reference.manifest()}
        for role, reference in sorted(entries, key=lambda item: item[0])
    ]
    return hashlib.sha256(
        b"kuairand-candidate-execution-closure-v1\0" + _canonical_json(payload)
    ).hexdigest()


def _input(reference: ArtifactRef, name: str) -> ArtifactRef:
    if not isinstance(reference, ArtifactRef) or reference.kind is not ArtifactKind.INPUT:
        raise ValueError(f"{name} must be an ArtifactKind.INPUT reference")
    return reference


@dataclass(frozen=True, slots=True)
class GeneratedCandidateIdentity:
    """One material generated source tree and immutable config identity."""

    source_snapshot: DirectoryArtifactRef
    source_digest: str
    config_digest: str
    entry_point: str = CANDIDATE_RUNTIME_CONTRACT.entry_point
    config_path: str = CANDIDATE_RUNTIME_CONTRACT.config_path
    checkpoint_path: str = CANDIDATE_RUNTIME_CONTRACT.checkpoint_path

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_snapshot, DirectoryArtifactRef)
            or self.source_snapshot.kind is not ArtifactKind.SOURCE
        ):
            raise ValueError("source_snapshot must be a source directory artifact")
        _digest(self.source_digest, "source_digest")
        _digest(self.config_digest, "config_digest")
        entries = {entry.path: entry.artifact for entry in self.source_snapshot.entries}
        if self.entry_point not in entries:
            raise ValueError("generated source snapshot does not contain candidate.py")
        config = entries.get(self.config_path)
        if config is None or config.sha256 != self.config_digest:
            raise ValueError("generated source snapshot config.json identity mismatch")
        if self.entry_point != CANDIDATE_RUNTIME_CONTRACT.entry_point:
            raise ValueError("generated entry point differs from the runtime contract")
        if self.config_path != CANDIDATE_RUNTIME_CONTRACT.config_path:
            raise ValueError("generated config path differs from the runtime contract")
        if self.checkpoint_path != CANDIDATE_RUNTIME_CONTRACT.checkpoint_path:
            raise ValueError("generated checkpoint path differs from the runtime contract")


@dataclass(frozen=True, slots=True)
class LocalCandidateLimits:
    timeout_seconds: float
    memory_limit_bytes: int
    workspace_disk_limit_bytes: int
    output_limit_bytes: int
    temp_limit_bytes: int
    threads: int
    stdout_limit_bytes: int = 8 * 1024 * 1024
    stderr_limit_bytes: int = 8 * 1024 * 1024
    process_limit: int = 64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0.0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        for name in (
            "memory_limit_bytes",
            "workspace_disk_limit_bytes",
            "output_limit_bytes",
            "temp_limit_bytes",
            "threads",
            "stdout_limit_bytes",
            "stderr_limit_bytes",
            "process_limit",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.device not in {"cpu", "mps"}:
            raise ValueError("device must be cpu or mps")


@dataclass(frozen=True, slots=True)
class GeneratedTrainRequest:
    execution_id: str
    identity: GeneratedCandidateIdentity
    split_role: SplitRole
    data_digest: str
    split_token: str
    seed: int
    features: ArtifactRef
    targets: ArtifactRef
    user_groups: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.identity, GeneratedCandidateIdentity):
            raise ValueError("identity must be GeneratedCandidateIdentity")
        if self.split_role not in _TRAIN_ROLES:
            raise ValueError("generated training is allowed only for train or inner_train")
        _digest(self.data_digest, "data_digest")
        _token(self.split_token, "split_token")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        _input(self.features, "features")
        _input(self.targets, "targets")
        _input(self.user_groups, "user_groups")
        if len({self.features.sha256, self.targets.sha256, self.user_groups.sha256}) != 3:
            raise ValueError("training capability artifacts must be distinct")


@dataclass(frozen=True, slots=True)
class GeneratedPredictRequest:
    execution_id: str
    identity: GeneratedCandidateIdentity
    split_role: SplitRole
    data_digest: str
    split_token: str
    expected_count: int
    features: ArtifactRef
    checkpoint: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.identity, GeneratedCandidateIdentity):
            raise ValueError("identity must be GeneratedCandidateIdentity")
        if self.split_role not in _PREDICTION_INPUT_ROLE:
            raise ValueError("prediction role must be inner_valid, outer_valid, or final")
        _digest(self.data_digest, "data_digest")
        _token(self.split_token, "split_token")
        if type(self.expected_count) is not int or self.expected_count <= 0:
            raise ValueError("expected_count must be a positive integer")
        _input(self.features, "features")
        if (
            not isinstance(self.checkpoint, ArtifactRef)
            or self.checkpoint.kind is not ArtifactKind.CHECKPOINT
        ):
            raise ValueError("checkpoint must be an ArtifactKind.CHECKPOINT reference")


class CandidateAction(StrEnum):
    TRAIN = "train"
    PREDICT = "predict"


@dataclass(frozen=True, slots=True)
class CandidateExecutionArtifacts:
    entries: tuple[tuple[str, ArtifactRef], ...]
    output_validated: bool
    diagnostic: str | None = None
    closure_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.output_validated) is not bool:
            raise ValueError("output_validated must be boolean")
        if self.diagnostic is not None and (
            type(self.diagnostic) is not str or not self.diagnostic or len(self.diagnostic) > 4096
        ):
            raise ValueError("diagnostic must be absent or bounded non-empty text")
        normalized = tuple(sorted(self.entries, key=lambda item: item[0]))
        roles = tuple(role for role, _ in normalized)
        if len(roles) != len(set(roles)) or any(
            type(role) is not str or not role for role in roles
        ):
            raise ValueError("artifact roles must be distinct non-empty text")
        if any(not isinstance(reference, ArtifactRef) for _, reference in normalized):
            raise ValueError("execution artifacts must contain ArtifactRef values")
        object.__setattr__(self, "entries", normalized)
        object.__setattr__(self, "closure_digest", _artifact_closure(normalized))

    def artifact(self, role: str) -> ArtifactRef:
        for candidate_role, reference in self.entries:
            if candidate_role == role:
                return reference
        raise KeyError(role)


class CandidateExecutionJournal(Protocol):
    """Durable prelaunch/start/terminal seam implemented by the campaign store adapter."""

    def prepare(
        self,
        *,
        action: CandidateAction,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
    ) -> None: ...

    def commit(self, process: ProcessRecord) -> None: ...

    def finish(
        self,
        *,
        action: CandidateAction,
        result: ExecutionResult,
        artifacts: CandidateExecutionArtifacts,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class GeneratedTrainRun:
    execution: ExecutionResult
    checkpoint: ArtifactRef
    seed: int
    artifacts: CandidateExecutionArtifacts
    checkpoint_digest: str
    diagnostics: object


@dataclass(frozen=True, slots=True)
class GeneratedPredictionRun:
    execution: ExecutionResult
    prediction: ArtifactRef
    scores: NDArray[np.float64] = field(repr=False)
    artifacts: CandidateExecutionArtifacts
    prediction_file_digest: str
    logical_prediction_digest: str


def put_numpy_capability(
    artifact_store: ArtifactStore,
    values: NDArray[np.generic] | Sequence[object],
    *,
    max_bytes: int | None = None,
) -> ArtifactRef:
    """Commit one finite numeric `.npy` capability without building an in-memory file image."""

    if not isinstance(artifact_store, ArtifactStore):
        raise ValueError("artifact_store must be ArtifactStore")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NumPy capability must be a numeric array") from exc
    if array.ndim not in {1, 2} or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError("NumPy capability must be a non-empty 1D or 2D numeric array")
    if not bool(np.isfinite(array).all()):
        raise ValueError("NumPy capability must contain only finite values")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="candidate-capability-",
        suffix=".npy",
        dir=artifact_store.staging_root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        return artifact_store.put_file(temporary, kind=ArtifactKind.INPUT, max_bytes=max_bytes)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


class GeneratedCandidateExecutor:
    """Materialize, supervise, protocol-validate, persist, and clean one candidate action."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        workspace_materializer: WorkspaceMaterializer,
        control_root: Path,
        interpreter: Path,
        limits: LocalCandidateLimits,
        runner: Runner | None = None,
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise ValueError("artifact_store must be ArtifactStore")
        if not isinstance(workspace_materializer, WorkspaceMaterializer):
            raise ValueError("workspace_materializer must be WorkspaceMaterializer")
        if (
            not isinstance(control_root, Path)
            or not control_root.is_dir()
            or control_root.is_symlink()
        ):
            raise ValueError("control_root must be an existing real directory")
        if not isinstance(interpreter, Path):
            raise ValueError("interpreter must be pathlib.Path")
        if not isinstance(limits, LocalCandidateLimits):
            raise ValueError("limits must be LocalCandidateLimits")
        self.artifact_store = artifact_store
        self.workspace_materializer = workspace_materializer
        self.control_root = control_root.resolve(strict=True)
        self.interpreter = interpreter
        self.limits = limits
        self.runner = Runner() if runner is None else runner

    def _workspace(
        self,
        *,
        execution_id: str,
        identity: GeneratedCandidateIdentity,
        split_role: SplitRole,
        inputs: tuple[ApprovedInput, ...],
        payload: dict[str, object],
    ) -> CandidateWorkspace:
        self.artifact_store.verify_directory(identity.source_snapshot)
        return self.workspace_materializer.materialize(
            WorkspaceSpec(
                execution_id=execution_id,
                split_role=split_role,
                source_snapshot=identity.source_snapshot,
                approved_inputs=inputs,
                request_payload=payload,
                output_limit_bytes=self.limits.output_limit_bytes,
                temp_limit_bytes=self.limits.temp_limit_bytes,
            )
        )

    def _spec(
        self,
        *,
        action: CandidateAction,
        execution_id: str,
        identity: GeneratedCandidateIdentity,
        workspace: CandidateWorkspace,
        data_digest: str,
        checkpoint_digest: str,
        checkpoint_path: Path | None,
        seed: int,
    ) -> ExecutionSpec:
        nonce = hashlib.sha256(
            b"kuairand-generated-execution-v1\0"
            + execution_id.encode("ascii")
            + action.value.encode("ascii")
            + data_digest.encode("ascii")
            + checkpoint_digest.encode("ascii")
        ).hexdigest()[:32]
        arguments = [
            f"source/{identity.entry_point}",
            action.value,
            "--request",
            "request.json",
        ]
        if checkpoint_path is not None:
            arguments.extend(("--checkpoint", str(checkpoint_path)))
        arguments.extend(("--output", "output"))
        return ExecutionSpec(
            execution_id=execution_id,
            nonce=nonce,
            interpreter=self.interpreter,
            arguments=tuple(arguments),
            workspace=workspace.root,
            control_dir=self._next_control_dir(execution_id),
            timeout_seconds=self.limits.timeout_seconds,
            memory_limit_bytes=self.limits.memory_limit_bytes,
            workspace_disk_limit_bytes=self.limits.workspace_disk_limit_bytes,
            stdout_limit_bytes=self.limits.stdout_limit_bytes,
            stderr_limit_bytes=self.limits.stderr_limit_bytes,
            threads=self.limits.threads,
            source_digest=identity.source_digest,
            config_digest=identity.config_digest,
            data_digest=data_digest,
            checkpoint_digest=checkpoint_digest,
            device=self.limits.device,
            process_limit=self.limits.process_limit,
            python_hash_seed=seed,
            extra_environment=(
                ("KUAIRAND_MODE", action.value),
                ("KUAIRAND_SEED", str(seed)),
                ("KUAIRAND_SPLIT_ROLE", workspace.split_role.value),
            ),
        )

    def _next_control_dir(self, execution_id: str) -> Path:
        primary = self.control_root / execution_id
        if not os.path.lexists(primary):
            return primary
        for attempt in range(1, 10_000):
            candidate = self.control_root / f"{execution_id}.retry-{attempt:04d}"
            if not os.path.lexists(candidate):
                return candidate
        raise CandidateExecutionError("candidate execution exhausted bounded control retries")

    def _prepare(
        self,
        *,
        action: CandidateAction,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
        journal: CandidateExecutionJournal,
    ) -> None:
        try:
            journal.prepare(action=action, spec=spec, workspace=workspace)
        except Exception:
            self.workspace_materializer.cleanup(workspace)
            raise

    def _base_artifacts(self, result: ExecutionResult) -> list[tuple[str, ArtifactRef]]:
        return [
            (
                "execution_manifest",
                self.artifact_store.put_bytes(
                    _canonical_json(result.manifest()), kind=ArtifactKind.MANIFEST
                ),
            ),
            ("stderr", self.artifact_store.put_file(result.stderr.path, kind=ArtifactKind.LOG)),
            ("stdout", self.artifact_store.put_file(result.stdout.path, kind=ArtifactKind.LOG)),
        ]

    def _cleanup_workspace(
        self,
        workspace: CandidateWorkspace,
    ) -> tuple[ArtifactRef, str | None]:
        error_type: str | None = None
        try:
            self.workspace_materializer.cleanup(workspace)
        except Exception as exc:
            error_type = type(exc).__name__
        receipt = self.artifact_store.put_bytes(
            _canonical_json(
                {
                    "schema_version": 1,
                    "execution_id": workspace.execution_id,
                    "workspace_removed": error_type is None,
                    "error_type": error_type,
                }
            ),
            kind=ArtifactKind.MANIFEST,
        )
        return receipt, error_type

    def _failed(
        self,
        *,
        action: CandidateAction,
        result: ExecutionResult,
        journal: CandidateExecutionJournal,
        workspace: CandidateWorkspace,
        diagnostic: str,
        cleanup: tuple[ArtifactRef, str | None] | None = None,
    ) -> CandidateExecutionError:
        entries = self._base_artifacts(result)
        cleanup_receipt, cleanup_error = (
            self._cleanup_workspace(workspace) if cleanup is None else cleanup
        )
        entries.append(("workspace_cleanup", cleanup_receipt))
        if cleanup_error is not None:
            diagnostic = f"{diagnostic}; trusted workspace cleanup failed: {cleanup_error}"
        bounded_diagnostic = diagnostic[:4096]
        entries.append(
            (
                "failure_diagnostic",
                self.artifact_store.put_bytes(
                    bounded_diagnostic.encode("utf-8"),
                    kind=ArtifactKind.LOG,
                ),
            )
        )
        artifacts = CandidateExecutionArtifacts(
            tuple(entries),
            output_validated=False,
            diagnostic=bounded_diagnostic,
        )
        journal.finish(action=action, result=result, artifacts=artifacts)
        return CandidateExecutionError(
            bounded_diagnostic,
            result=result,
            artifacts=artifacts,
        )

    def train(
        self,
        request: GeneratedTrainRequest,
        *,
        journal: CandidateExecutionJournal,
        cancel_event: threading.Event | None = None,
    ) -> GeneratedTrainRun:
        if not isinstance(request, GeneratedTrainRequest):
            raise ValueError("request must be GeneratedTrainRequest")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise ValueError("cancel_event must be threading.Event or None")
        payload = CANDIDATE_RUNTIME_CONTRACT.training_payload(
            source_digest=request.identity.source_digest,
            config_digest=request.identity.config_digest,
            data_digest=request.data_digest,
            split_token=request.split_token,
            seed=request.seed,
        )
        workspace = self._workspace(
            execution_id=request.execution_id,
            identity=request.identity,
            split_role=request.split_role,
            inputs=(
                ApprovedInput(
                    CANDIDATE_RUNTIME_CONTRACT.features_handle,
                    CandidateInputRole.TRAIN_INPUTS,
                    request.features,
                ),
                ApprovedInput(
                    CANDIDATE_RUNTIME_CONTRACT.targets_handle,
                    CandidateInputRole.TRAIN_TARGETS,
                    request.targets,
                ),
                ApprovedInput(
                    CANDIDATE_RUNTIME_CONTRACT.user_groups_handle,
                    CandidateInputRole.TRAIN_INPUTS,
                    request.user_groups,
                ),
            ),
            payload=payload,
        )
        spec = self._spec(
            action=CandidateAction.TRAIN,
            execution_id=request.execution_id,
            identity=request.identity,
            workspace=workspace,
            data_digest=request.data_digest,
            checkpoint_digest=NO_CHECKPOINT_DIGEST,
            checkpoint_path=None,
            seed=request.seed,
        )
        self._prepare(
            action=CandidateAction.TRAIN,
            spec=spec,
            workspace=workspace,
            journal=journal,
        )
        result = self.runner.run(
            spec,
            commit_launch=journal.commit,
            cancel_event=cancel_event,
        )
        if not result.succeeded:
            raise self._failed(
                action=CandidateAction.TRAIN,
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic=f"generated training process failed: {result.outcome.value}",
            )
        try:
            validated = validate_train_outputs(
                workspace.output_dir.resolve(strict=True),
                TrainExpectation(
                    source_digest=request.identity.source_digest,
                    config_digest=request.identity.config_digest,
                    data_digest=request.data_digest,
                    split_token=request.split_token,
                    checkpoint_path=request.identity.checkpoint_path,
                ),
            )
        except Exception as exc:
            raise self._failed(
                action=CandidateAction.TRAIN,
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic=(
                    f"generated training output validation failed: {type(exc).__name__}: {exc}"
                ),
            ) from exc
        entries = self._base_artifacts(result)
        checkpoint = self.artifact_store.put_file(
            validated.checkpoint_path, kind=ArtifactKind.CHECKPOINT
        )
        result_manifest = self.artifact_store.put_file(
            workspace.output_dir / TRAIN_RESULT_FILENAME,
            kind=ArtifactKind.MANIFEST,
        )
        entries.extend((("checkpoint", checkpoint), ("candidate_result", result_manifest)))
        cleanup = self._cleanup_workspace(workspace)
        if cleanup[1] is not None:
            raise self._failed(
                action=CandidateAction.TRAIN,
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic="generated training workspace cleanup failed",
                cleanup=cleanup,
            )
        entries.append(("workspace_cleanup", cleanup[0]))
        artifacts = CandidateExecutionArtifacts(tuple(entries), output_validated=True)
        journal.finish(action=CandidateAction.TRAIN, result=result, artifacts=artifacts)
        return GeneratedTrainRun(
            execution=result,
            checkpoint=checkpoint,
            seed=request.seed,
            artifacts=artifacts,
            checkpoint_digest=validated.checkpoint_digest,
            diagnostics=validated.manifest.diagnostics,
        )

    def predict(
        self,
        request: GeneratedPredictRequest,
        *,
        journal: CandidateExecutionJournal,
        cancel_event: threading.Event | None = None,
    ) -> GeneratedPredictionRun:
        if not isinstance(request, GeneratedPredictRequest):
            raise ValueError("request must be GeneratedPredictRequest")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise ValueError("cancel_event must be threading.Event or None")
        checkpoint_path = self.artifact_store.verify(request.checkpoint)
        payload = CANDIDATE_RUNTIME_CONTRACT.prediction_payload(
            source_digest=request.identity.source_digest,
            config_digest=request.identity.config_digest,
            data_digest=request.data_digest,
            split_token=request.split_token,
            expected_count=request.expected_count,
            checkpoint_digest=request.checkpoint.sha256,
        )
        workspace = self._workspace(
            execution_id=request.execution_id,
            identity=request.identity,
            split_role=request.split_role,
            inputs=(
                ApprovedInput(
                    CANDIDATE_RUNTIME_CONTRACT.features_handle,
                    _PREDICTION_INPUT_ROLE[request.split_role],
                    request.features,
                ),
            ),
            payload=payload,
        )
        spec = self._spec(
            action=CandidateAction.PREDICT,
            execution_id=request.execution_id,
            identity=request.identity,
            workspace=workspace,
            data_digest=request.data_digest,
            checkpoint_digest=request.checkpoint.sha256,
            checkpoint_path=checkpoint_path,
            seed=0,
        )
        self._prepare(
            action=CandidateAction.PREDICT,
            spec=spec,
            workspace=workspace,
            journal=journal,
        )
        result = self.runner.run(
            spec,
            commit_launch=journal.commit,
            cancel_event=cancel_event,
        )
        if not result.succeeded:
            raise self._failed(
                action=CandidateAction.PREDICT,
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic=f"generated prediction process failed: {result.outcome.value}",
            )
        try:
            validated = validate_prediction_outputs(
                workspace.output_dir.resolve(strict=True),
                PredictionExpectation(
                    source_digest=request.identity.source_digest,
                    config_digest=request.identity.config_digest,
                    data_digest=request.data_digest,
                    split_token=request.split_token,
                    checkpoint_digest=request.checkpoint.sha256,
                    expected_count=request.expected_count,
                    dtype=SCORES_DTYPE,
                ),
            )
        except Exception as exc:
            raise self._failed(
                action=CandidateAction.PREDICT,
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic=(
                    f"generated prediction output validation failed: {type(exc).__name__}: {exc}"
                ),
            ) from exc
        entries = self._base_artifacts(result)
        prediction = self.artifact_store.put_file(
            validated.scores_path, kind=ArtifactKind.PREDICTION
        )
        result_manifest = self.artifact_store.put_file(
            workspace.output_dir / PREDICTION_RESULT_FILENAME,
            kind=ArtifactKind.MANIFEST,
        )
        entries.extend((("prediction", prediction), ("prediction_result", result_manifest)))
        cleanup = self._cleanup_workspace(workspace)
        if cleanup[1] is not None:
            raise self._failed(
                action=CandidateAction.PREDICT,
                result=result,
                journal=journal,
                workspace=workspace,
                diagnostic="generated prediction workspace cleanup failed",
                cleanup=cleanup,
            )
        entries.append(("workspace_cleanup", cleanup[0]))
        artifacts = CandidateExecutionArtifacts(tuple(entries), output_validated=True)
        journal.finish(action=CandidateAction.PREDICT, result=result, artifacts=artifacts)
        logical_prediction_digest = hashlib.sha256(
            b"kuairand-generated-predictions-v1\0" + validated.scores.tobytes(order="C")
        ).hexdigest()
        return GeneratedPredictionRun(
            execution=result,
            prediction=prediction,
            scores=validated.scores,
            artifacts=artifacts,
            prediction_file_digest=validated.scores_sha256,
            logical_prediction_digest=logical_prediction_digest,
        )


__all__ = [
    "NO_CHECKPOINT_DIGEST",
    "CandidateAction",
    "CandidateExecutionArtifacts",
    "CandidateExecutionError",
    "CandidateExecutionJournal",
    "GeneratedCandidateExecutor",
    "GeneratedCandidateIdentity",
    "GeneratedPredictRequest",
    "GeneratedPredictionRun",
    "GeneratedTrainRequest",
    "GeneratedTrainRun",
    "LocalCandidateLimits",
    "put_numpy_capability",
]
