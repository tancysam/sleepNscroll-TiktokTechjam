"""Provider-independent autonomous research-loop composition.

The module deliberately owns orchestration, not scientific authority.  Generated code receives
only a materialized source child and approved workspace inputs.  ``Runner`` returns process
evidence without metrics.  Only an injected trusted evaluator may turn validated outputs into a
``TrustedEvaluation`` and only an injected selector may replace the durable incumbent.

Campaign creation, resume reconciliation, deadlines, and finalization remain controller work.
``CampaignStoreResearchLedger`` is the narrow public adapter used after that controller has
created or resumed a store.
"""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

from kuairand_agent.campaign.budgets import LaunchCategory
from kuairand_agent.campaign.convergence import ConvergenceState
from kuairand_agent.campaign.store import (
    ArtifactSpec,
    CampaignStore,
    IncumbentRecord,
)
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
)
from kuairand_agent.execution.policy import ApprovedInput, SplitRole
from kuairand_agent.execution.runner import (
    ExecutionResult,
    ExecutionSpec,
    ProcessRecord,
)
from kuairand_agent.execution.workspace import (
    CandidateWorkspace,
    WorkspaceSpec,
)
from kuairand_agent.research.context import SafeResearchContext
from kuairand_agent.research.interface import ResearchModel
from kuairand_agent.research.materialize import (
    CandidateMaterializationError,
    CandidateStaticError,
    MaterialChangeEvidence,
    MaterializedCandidate,
    materialize_candidate,
    require_material_executable_change,
    snapshot_materialized_candidate,
    validate_candidate_static,
)
from kuairand_agent.research.schemas import (
    ExperimentResultSummary,
    FailureCategory,
    GeneratedPackage,
    ImplementationRequest,
    ParentSnapshot,
    Proposal,
    ProposalRequest,
    ReflectionRequest,
    RejectedPackageSnapshot,
    RepairRequest,
    ResearchOperation,
    canonical_digest,
    canonical_json_bytes,
)
from kuairand_agent.research.source_policy import (
    CandidateManifestPolicyError,
    classify_method_family,
    proposal_novelty_signature,
)

_DIGEST_RE: Final = __import__("re").compile(r"[0-9a-f]{64}\Z")
_MAX_ITERATIONS: Final = 50


class ResearchLoopError(RuntimeError):
    """The trusted research controller cannot safely continue the requested iteration."""


def _digest(value: object, location: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ResearchLoopError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ResearchLoopError(f"{location} must be non-empty text without NUL bytes")
    return value


def artifact_spec(
    reference: ArtifactRef,
    *,
    metadata: Mapping[str, object] = MappingProxyType({}),
) -> ArtifactSpec:
    """Convert one already committed artifact into the store's immutable link record."""

    if not isinstance(reference, ArtifactRef):
        raise ResearchLoopError("artifact reference must be an ArtifactRef")
    return ArtifactSpec(
        digest=reference.sha256,
        kind=reference.kind.value,
        relative_path=reference.object_relative_path.as_posix(),
        size_bytes=reference.size_bytes,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class TrustedEvaluation:
    """Metrics and replay identities produced only by the trusted evaluation callback."""

    summary: ExperimentResultSummary
    scorer_digest: str
    prediction_digest: str
    eligible: bool
    replay_verified: bool
    checkpoint_digest: str
    artifact_closure_digest: str
    artifacts: tuple[tuple[str, ArtifactSpec], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.summary, ExperimentResultSummary):
            raise ResearchLoopError("trusted evaluation requires ExperimentResultSummary")
        for name in (
            "scorer_digest",
            "prediction_digest",
            "checkpoint_digest",
            "artifact_closure_digest",
        ):
            _digest(getattr(self, name), f"trusted evaluation {name}")
        if type(self.eligible) is not bool or type(self.replay_verified) is not bool:
            raise ResearchLoopError("trusted evaluation eligibility flags must be boolean")
        if type(self.artifacts) is not tuple:
            raise ResearchLoopError("trusted evaluation artifacts must be a tuple")
        for role, spec in self.artifacts:
            _text(role, "trusted evaluation artifact role")
            if not isinstance(spec, ArtifactSpec):
                raise ResearchLoopError("trusted evaluation artifact must be ArtifactSpec")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    source_digest: str
    diff_digest: str
    material_change: MaterialChangeEvidence
    evaluation: TrustedEvaluation

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate evidence candidate_id")
        _digest(self.source_digest, "candidate evidence source_digest")
        _digest(self.diff_digest, "candidate evidence diff_digest")
        if not isinstance(self.material_change, MaterialChangeEvidence):
            raise ResearchLoopError("candidate evidence requires material-change evidence")
        if not isinstance(self.evaluation, TrustedEvaluation):
            raise ResearchLoopError("candidate evidence requires a trusted evaluation")


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    promote: bool
    eligibility: str
    reason: str

    def __post_init__(self) -> None:
        if type(self.promote) is not bool:
            raise ResearchLoopError("selection promote must be boolean")
        _text(self.eligibility, "selection eligibility")
        _text(self.reason, "selection reason")


class TrustedEvaluator(Protocol):
    """Trusted output validator plus protected scorer; generated code cannot implement this."""

    def evaluate(
        self,
        *,
        candidate: MaterializedCandidate,
        workspace: CandidateWorkspace,
        execution: ExecutionResult,
    ) -> TrustedEvaluation: ...


class TrustedSelector(Protocol):
    """Narrow seam for the deterministic WP4 selector or a fixture test policy."""

    def decide(
        self,
        incumbent_primary: float | None,
        candidate: CandidateEvidence,
    ) -> SelectionDecision: ...


class WorkspacePort(Protocol):
    def materialize(self, spec: WorkspaceSpec) -> CandidateWorkspace: ...

    def cleanup(self, workspace: CandidateWorkspace) -> None: ...


class RunnerPort(Protocol):
    def run(
        self,
        spec: ExecutionSpec,
        *,
        commit_launch: Callable[[ProcessRecord], None],
    ) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class LocalExecutionTemplate:
    """Fixed trusted limits and approved inputs used to build each local child execution."""

    approved_inputs: tuple[ApprovedInput, ...]
    split_role: SplitRole
    request_payload: Mapping[str, object]
    interpreter: Path
    arguments: tuple[str, ...]
    control_root: Path
    config_digest: str
    environment_digest: str
    data_digest: str
    checkpoint_digest: str
    timeout_seconds: float
    memory_limit_bytes: int
    workspace_disk_limit_bytes: int
    stdout_limit_bytes: int
    stderr_limit_bytes: int
    output_limit_bytes: int
    temp_limit_bytes: int
    threads: int
    device: str = "cpu"
    process_limit: int = 64
    python_hash_seed: int = 0
    extra_environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.approved_inputs) is not tuple or any(
            not isinstance(value, ApprovedInput) for value in self.approved_inputs
        ):
            raise ResearchLoopError("approved_inputs must be a tuple of ApprovedInput")
        if not isinstance(self.split_role, SplitRole):
            raise ResearchLoopError("split_role must be a SplitRole")
        if not isinstance(self.request_payload, Mapping):
            raise ResearchLoopError("request_payload must be a mapping")
        if type(self.arguments) is not tuple:
            raise ResearchLoopError("execution arguments must be a tuple")
        for name in (
            "config_digest",
            "environment_digest",
            "data_digest",
            "checkpoint_digest",
        ):
            _digest(getattr(self, name), f"execution template {name}")
        try:
            metadata = self.control_root.lstat()
        except OSError as exc:
            raise ResearchLoopError("control_root must already exist") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ResearchLoopError("control_root must be a real directory")

    def workspace_spec(
        self,
        *,
        execution_id: str,
        source_snapshot: DirectoryArtifactRef,
        source_digest: str,
    ) -> WorkspaceSpec:
        request_payload = {
            **dict(self.request_payload),
            "source_digest": source_digest,
            "config_digest": self.config_digest,
            "data_digest": self.data_digest,
            "checkpoint_digest": self.checkpoint_digest,
        }
        return WorkspaceSpec(
            execution_id=execution_id,
            split_role=self.split_role,
            source_snapshot=source_snapshot,
            approved_inputs=self.approved_inputs,
            request_payload=request_payload,
            output_limit_bytes=self.output_limit_bytes,
            temp_limit_bytes=self.temp_limit_bytes,
        )

    def execution_spec(
        self,
        *,
        execution_id: str,
        nonce: str,
        workspace: CandidateWorkspace,
        source_digest: str,
    ) -> ExecutionSpec:
        return ExecutionSpec(
            execution_id=execution_id,
            nonce=nonce,
            interpreter=self.interpreter,
            arguments=self.arguments,
            workspace=workspace.root,
            control_dir=self.control_root / execution_id,
            timeout_seconds=self.timeout_seconds,
            memory_limit_bytes=self.memory_limit_bytes,
            workspace_disk_limit_bytes=self.workspace_disk_limit_bytes,
            stdout_limit_bytes=self.stdout_limit_bytes,
            stderr_limit_bytes=self.stderr_limit_bytes,
            threads=self.threads,
            source_digest=source_digest,
            config_digest=self.config_digest,
            data_digest=self.data_digest,
            checkpoint_digest=self.checkpoint_digest,
            device=self.device,
            process_limit=self.process_limit,
            python_hash_seed=self.python_hash_seed,
            extra_environment=self.extra_environment,
        )


class ResearchLedger(Protocol):
    """Per-experiment persistence surface expected from a resumed campaign controller."""

    @property
    def campaign_id(self) -> str: ...

    @property
    def fallback_candidate_id(self) -> str: ...

    def incumbent(self) -> IncumbentRecord: ...

    def convergence(self) -> ConvergenceState: ...

    def create_iteration(self, experiment_id: str, iteration: int, proposal: Proposal) -> None: ...

    def record_proposal(
        self,
        experiment_id: str,
        request_digest: str,
        proposal: Proposal,
        transcript: ArtifactSpec,
    ) -> None: ...

    def transition(
        self,
        experiment_id: str,
        from_state: str,
        to_state: str,
        *,
        operation: str,
        reason: str,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> None: ...

    def record_source(
        self,
        *,
        snapshot_id: str,
        experiment_id: str,
        candidate: MaterializedCandidate,
        parent_digest: str,
        source_manifest: ArtifactSpec,
        diff: ArtifactSpec,
        transcript_role: str,
        transcript: ArtifactSpec,
    ) -> None: ...

    def record_failure(
        self,
        *,
        failure_id: str,
        experiment_id: str,
        execution_id: str | None,
        category: str,
        diagnostic: str,
        retry_ordinal: int,
        traceback_digest: str | None,
        repair_action: str | None,
    ) -> None: ...

    def begin_execution(
        self,
        *,
        experiment_id: str,
        iteration: int,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
        environment_digest: str,
    ) -> None: ...

    def mark_execution_running(self, execution_id: str, process: ProcessRecord) -> None: ...

    def finish_execution(
        self,
        execution: ExecutionResult,
        *,
        artifacts: Sequence[tuple[str, ArtifactSpec]],
    ) -> None: ...

    def record_metric(
        self,
        *,
        experiment_id: str,
        execution_id: str,
        evaluation: TrustedEvaluation,
    ) -> None: ...

    def promote(
        self,
        *,
        experiment_id: str,
        candidate: CandidateEvidence,
        decision: SelectionDecision,
    ) -> None: ...

    def update_convergence(self, state: ConvergenceState) -> None: ...


class CampaignStoreResearchLedger:
    """Public CampaignStore adapter; no controller private helper is imported or duplicated."""

    def __init__(
        self,
        store: CampaignStore,
        *,
        provider_name: str,
        fallback_candidate_id: str | None = None,
    ) -> None:
        if not isinstance(store, CampaignStore):
            raise ResearchLoopError("store must be CampaignStore")
        self.store = store
        self.provider_name = _text(provider_name, "provider_name")
        self._pending_executions: dict[
            str, tuple[str, int, ExecutionSpec, CandidateWorkspace, str]
        ] = {}
        current = self.incumbent()
        if fallback_candidate_id is None:
            if not current.is_fallback:
                raise ResearchLoopError(
                    "fallback_candidate_id is required when resuming after a promotion"
                )
            fallback_candidate_id = current.incumbent_id
        self._fallback_candidate_id = _text(fallback_candidate_id, "fallback_candidate_id")

    @property
    def campaign_id(self) -> str:
        return self.store.campaign_id

    @property
    def fallback_candidate_id(self) -> str:
        return self._fallback_candidate_id

    def _revision(self) -> int:
        return self.store.snapshot().revision

    def incumbent(self) -> IncumbentRecord:
        incumbent = self.store.current_incumbent()
        if incumbent is None:
            raise ResearchLoopError("a replay-verified fallback must exist before research")
        return incumbent

    def convergence(self) -> ConvergenceState:
        return ConvergenceState.from_manifest(self.store.snapshot().convergence_state)

    def create_iteration(self, experiment_id: str, iteration: int, proposal: Proposal) -> None:
        self.store.create_experiment(
            experiment_id=experiment_id,
            iteration_number=iteration,
            hypothesis=proposal.hypothesis,
            mechanism=proposal.mechanism,
            method_attribution="; ".join(proposal.attributions),
            expected_revision=self._revision(),
            parent_experiment_id=None,
            status="PLANNED",
            metadata={
                "proposal_id": proposal.proposal_id,
                "parent_candidate_id": proposal.parent_candidate_id,
            },
        )

    def record_proposal(
        self,
        experiment_id: str,
        request_digest: str,
        proposal: Proposal,
        transcript: ArtifactSpec,
    ) -> None:
        self.store.record_proposal(
            proposal_id=proposal.proposal_id,
            experiment_id=experiment_id,
            request_digest=request_digest,
            response_digest=proposal.digest,
            provider=self.provider_name,
            expected_revision=self._revision(),
            artifacts=(("proposal_transcript", transcript),),
            metadata={"operation": ResearchOperation.PROPOSE.value},
        )

    def transition(
        self,
        experiment_id: str,
        from_state: str,
        to_state: str,
        *,
        operation: str,
        reason: str,
        artifacts: Sequence[tuple[str, ArtifactSpec]] = (),
        metadata: Mapping[str, object] = MappingProxyType({}),
    ) -> None:
        self.store.transition_entity(
            "experiment",
            experiment_id,
            from_state=from_state,
            to_state=to_state,
            expected_revision=self._revision(),
            reason=reason,
            artifacts=artifacts,
            metadata={"operation": operation, **dict(metadata)},
        )

    def record_source(
        self,
        *,
        snapshot_id: str,
        experiment_id: str,
        candidate: MaterializedCandidate,
        parent_digest: str,
        source_manifest: ArtifactSpec,
        diff: ArtifactSpec,
        transcript_role: str,
        transcript: ArtifactSpec,
    ) -> None:
        self.store.record_source_snapshot(
            snapshot_id=snapshot_id,
            experiment_id=experiment_id,
            source_digest=candidate.source_digest,
            parent_source_digest=parent_digest,
            diff_digest=candidate.diff_digest,
            expected_revision=self._revision(),
            artifacts=(
                ("source_manifest", source_manifest),
                ("source_diff", diff),
                (transcript_role, transcript),
            ),
            metadata={
                "package_digest": candidate.package_digest,
                "changed_paths": list(candidate.changed_paths),
            },
        )

    def record_failure(
        self,
        *,
        failure_id: str,
        experiment_id: str,
        execution_id: str | None,
        category: str,
        diagnostic: str,
        retry_ordinal: int,
        traceback_digest: str | None,
        repair_action: str | None,
    ) -> None:
        fingerprint = hashlib.sha256(f"{category}\0{diagnostic}".encode()).hexdigest()
        self.store.record_failure(
            failure_id=failure_id,
            category=category,
            fingerprint=fingerprint,
            retry_ordinal=retry_ordinal,
            expected_revision=self._revision(),
            experiment_id=experiment_id,
            execution_id=execution_id,
            traceback_digest=traceback_digest,
            repair_action=repair_action,
            recovery_outcome=None,
            metadata={"diagnostic": diagnostic[:512]},
        )

    def begin_execution(
        self,
        *,
        experiment_id: str,
        iteration: int,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
        environment_digest: str,
    ) -> None:
        _digest(environment_digest, "locked execution environment digest")
        launch_id = f"{spec.execution_id}-launch"
        self.store.reserve_launch(
            launch_id=launch_id,
            reservation_key=f"research:{spec.execution_id}",
            category=LaunchCategory.DIVERSE_INNER_SCREEN.value,
            purpose="autonomous generated candidate execution",
            expected_revision=self._revision(),
            experiment_id=experiment_id,
            scientific_iteration=iteration,
            seed=spec.python_hash_seed,
            metadata={"source_digest": spec.source_digest},
        )
        if spec.execution_id in self._pending_executions:
            raise ResearchLoopError("execution is already pending")
        self._pending_executions[spec.execution_id] = (
            experiment_id,
            iteration,
            spec,
            workspace,
            launch_id,
        )
        # STARTING exists before Runner is invoked.  Its environment identity is the locked
        # campaign/scientific environment, not the runner's later dynamic sanitized-runtime
        # digest.  A crash from this point is therefore independently reconcilable.
        self._create_execution(
            spec.execution_id,
            status="STARTING",
            environment_digest=environment_digest,
        )

    def _create_execution(
        self,
        execution_id: str,
        *,
        status: str,
        environment_digest: str | None,
    ) -> None:
        try:
            experiment_id, _iteration, spec, workspace, launch_id = self._pending_executions[
                execution_id
            ]
        except KeyError as exc:
            raise ResearchLoopError("execution was not reserved by this ledger") from exc
        self.store.create_execution(
            execution_id=spec.execution_id,
            kind="generated_candidate",
            tier=workspace.split_role.value,
            command=spec.command,
            expected_revision=self._revision(),
            experiment_id=experiment_id,
            launch_id=launch_id,
            seed=spec.python_hash_seed,
            status=status,
            source_digest=spec.source_digest,
            config_digest=spec.config_digest,
            capability_digest=workspace.manifest_digest,
            environment_digest=environment_digest,
            data_digest=spec.data_digest,
            checkpoint_digest=spec.checkpoint_digest,
            nonce=spec.nonce,
            metadata={"workspace_manifest_digest": workspace.manifest_digest},
        )

    def mark_execution_running(self, execution_id: str, process: ProcessRecord) -> None:
        # The process remains blocked while the exact dynamic receipt is committed.  Preserve
        # the distinct prelaunch environment identity already stored on STARTING; the store
        # projects ``process.environment_digest`` separately as ``process_environment_digest``.
        execution = next(
            value for value in self.store.executions() if value.execution_id == execution_id
        )
        if execution.launch_id is None:
            raise ResearchLoopError("research execution lost its launch identity")
        self.store.transition_execution(
            execution_id,
            from_state="STARTING",
            to_state="RUNNING",
            expected_revision=self._revision(),
            reason="persist exact runner process receipt before candidate release",
            process_record_digest=process.digest,
            process_record=process.manifest(),
            metadata={"candidate_released": False},
        )
        self.store.transition_launch(
            execution.launch_id,
            to_state="STARTED",
            expected_revision=self._revision(),
            start_receipt_digest=process.digest,
            metadata={"execution_id": execution_id},
        )

    def finish_execution(
        self,
        execution: ExecutionResult,
        *,
        artifacts: Sequence[tuple[str, ArtifactSpec]],
    ) -> None:
        records = {value.execution_id: value for value in self.store.executions()}
        record = records.get(execution.execution_id)
        if record is None:
            raise ResearchLoopError("runner result has no prelaunch STARTING execution")
        terminal = "SUCCEEDED" if execution.succeeded else "FAILED"
        self.store.transition_execution(
            execution.execution_id,
            from_state=record.status,
            to_state=terminal,
            expected_revision=self._revision(),
            reason="persist trusted runner terminal evidence",
            result_digest=canonical_digest(execution.manifest()),
            finished_at=execution.ended_at_utc,
            artifacts=artifacts,
            metadata={
                "outcome": execution.outcome.value,
                "candidate_metrics_accepted": False,
            },
        )
        if record.launch_id is None:
            raise ResearchLoopError("research execution lost its launch identity")
        launch = next(
            value for value in self.store.launches() if value.launch_id == record.launch_id
        )
        if launch.state == "RESERVED":
            self.store.transition_launch(
                record.launch_id,
                to_state="NOT_STARTED",
                expected_revision=self._revision(),
                metadata={"execution_id": execution.execution_id},
            )
        else:
            self.store.transition_launch(
                record.launch_id,
                to_state="FINISHED",
                expected_revision=self._revision(),
                metadata={"execution_id": execution.execution_id},
            )
        self._pending_executions.pop(execution.execution_id, None)

    def record_metric(
        self,
        *,
        experiment_id: str,
        execution_id: str,
        evaluation: TrustedEvaluation,
    ) -> None:
        summary = evaluation.summary
        self.store.record_metric(
            metric_id=f"{execution_id}-trusted-metric",
            split_role=summary.tier,
            gauc=summary.gauc,
            ndcg_at_5=summary.ndcg_at_5,
            primary=summary.primary,
            scorer_digest=evaluation.scorer_digest,
            prediction_digest=evaluation.prediction_digest,
            expected_revision=self._revision(),
            experiment_id=experiment_id,
            execution_id=execution_id,
            seed=0,
            artifacts=evaluation.artifacts,
            metadata={"authority": "trusted_protected_scorer"},
        )

    def promote(
        self,
        *,
        experiment_id: str,
        candidate: CandidateEvidence,
        decision: SelectionDecision,
    ) -> None:
        evaluation = candidate.evaluation
        if (
            not evaluation.eligible
            or not evaluation.replay_verified
            or evaluation.summary.tier != "outer"
        ):
            raise ResearchLoopError("selector cannot promote ineligible or unreplayed evidence")
        self.store.record_incumbent(
            incumbent_id=candidate.candidate_id,
            eligibility=decision.eligibility,
            source_digest=candidate.source_digest,
            checkpoint_digest=evaluation.checkpoint_digest,
            artifact_closure_digest=evaluation.artifact_closure_digest,
            replay_verified=True,
            is_fallback=False,
            expected_revision=self._revision(),
            reason=decision.reason,
            experiment_id=experiment_id,
            outer_primary_mean=evaluation.summary.primary,
            metadata={"diff_digest": candidate.diff_digest},
        )

    def update_convergence(self, state: ConvergenceState) -> None:
        self.store.set_convergence_state(
            state.manifest(),
            expected_revision=self._revision(),
            reason="exactly one update after closing one scientific iteration",
        )


@dataclass(frozen=True, slots=True)
class IterationResult:
    iteration: int
    experiment_id: str
    proposal_id: str
    candidate_id: str | None
    status: str
    repairs: int
    trusted_evaluation: TrustedEvaluation | None
    diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignLoopResult:
    selected_candidate_id: str
    fallback_candidate_id: str
    iterations: tuple[IterationResult, ...]
    final_parent: ParentSnapshot = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ReadyCandidate:
    candidate_id: str
    candidate: MaterializedCandidate
    material_change: MaterialChangeEvidence
    source_artifact: DirectoryArtifactRef


class ResearchCampaignLoop:
    """Run bounded scientific iterations without granting the model repository or scorer access."""

    def __init__(
        self,
        *,
        model: ResearchModel,
        safe_context: SafeResearchContext,
        ledger: ResearchLedger,
        artifacts: ArtifactStore,
        generated_root: Path | str,
        workspace_materializer: WorkspacePort,
        runner: RunnerPort,
        execution: LocalExecutionTemplate,
        evaluator: TrustedEvaluator,
        selector: TrustedSelector,
    ) -> None:
        if not isinstance(safe_context, SafeResearchContext):
            raise ResearchLoopError("safe_context must be SafeResearchContext")
        if not isinstance(artifacts, ArtifactStore):
            raise ResearchLoopError("artifacts must be ArtifactStore")
        self.model = model
        self.safe_context = safe_context
        self.ledger = ledger
        self.artifacts = artifacts
        self.workspace_materializer = workspace_materializer
        self.runner = runner
        self.execution = execution
        self.evaluator = evaluator
        self.selector = selector
        self.generated_root = Path(generated_root)
        self.generated_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = self.generated_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ResearchLoopError("generated_root must be a real directory")
        self._training_evidence_cursor = safe_context.digest
        self._proposal_signatures: set[str] = set()
        self._family_implementation_admissions: dict[str, int] = {}

    def _put_json(self, value: Mapping[str, object]) -> ArtifactSpec:
        ref = self.artifacts.put_bytes(canonical_json_bytes(dict(value)), kind=ArtifactKind.OTHER)
        return artifact_spec(ref)

    def _transcript(
        self,
        operation: ResearchOperation,
        request: Mapping[str, object],
        response: Mapping[str, object],
    ) -> ArtifactSpec:
        return self._put_json(
            {
                "schema_version": 1,
                "operation": operation.value,
                "request": dict(request),
                "response": dict(response),
            }
        )

    def _failure(
        self,
        *,
        experiment_id: str,
        ordinal: int,
        category: str,
        diagnostic: str,
        execution_id: str | None = None,
        repair_action: str | None = None,
    ) -> None:
        bounded = diagnostic[:4096] or category
        trace_ref = self.artifacts.put_bytes(bounded.encode("utf-8"), kind=ArtifactKind.LOG)
        self.ledger.record_failure(
            failure_id=f"{experiment_id}-failure-{ordinal}-{category}",
            experiment_id=experiment_id,
            execution_id=execution_id,
            category=category,
            diagnostic=bounded,
            retry_ordinal=ordinal,
            traceback_digest=trace_ref.sha256,
            repair_action=repair_action,
        )

    @staticmethod
    def _runner_failure_diagnostic(execution: ExecutionResult) -> str:
        """Return only the runner's already-bounded, integrity-checked stderr evidence."""

        detail = execution.detail or execution.outcome.value
        try:
            stderr = execution.stderr.path.read_bytes()
        except OSError:
            return detail
        if (
            len(stderr) != execution.stderr.retained_bytes
            or hashlib.sha256(stderr).hexdigest() != execution.stderr.sha256
        ):
            return detail
        rendered = stderr.decode("utf-8", errors="replace").strip()
        return detail if not rendered else f"{detail}\n{rendered}"[:4096]

    @staticmethod
    def _runner_failure_category(diagnostic: str) -> str:
        import_markers = ("ModuleNotFoundError", "ImportError", "No module named")
        if any(marker in diagnostic for marker in import_markers):
            return FailureCategory.IMPORT_ERROR.value
        return "runtime_error"

    def _record_child(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        parent: ParentSnapshot,
        package: GeneratedPackage,
        transcript_role: str,
        transcript: ArtifactSpec,
    ) -> tuple[MaterializedCandidate, DirectoryArtifactRef]:
        candidate = materialize_candidate(
            parent,
            package,
            self.generated_root / candidate_id,
        )
        source = self.artifacts.put_directory(candidate.destination, kind=ArtifactKind.SOURCE)
        diff_ref = self.artifacts.put_bytes(
            candidate.unified_diff.encode("utf-8"), kind=ArtifactKind.SOURCE
        )
        self.ledger.record_source(
            snapshot_id=f"{candidate_id}-source",
            experiment_id=experiment_id,
            candidate=candidate,
            parent_digest=parent.digest,
            source_manifest=artifact_spec(source.manifest_artifact),
            diff=artifact_spec(diff_ref),
            transcript_role=transcript_role,
            transcript=transcript,
        )
        return candidate, source

    def _repair_after_execution_failure(
        self,
        *,
        experiment_id: str,
        proposal: Proposal,
        original_parent: ParentSnapshot,
        failed: _ReadyCandidate,
        repair_count: int,
        state: str,
        failure_category: str,
        diagnostic: str,
    ) -> tuple[_ReadyCandidate | None, int, str, str | None]:
        """Use remaining child budget for a candidate-controlled smoke/import failure."""

        maximum_repairs = min(proposal.maximum_repairs, 2)
        if repair_count >= maximum_repairs:
            return None, repair_count, state, diagnostic
        context = self.safe_context.to_wire()
        failed_candidate = failed.candidate
        failed_candidate_id = failed.candidate_id
        current_diagnostic = diagnostic
        current_category = failure_category
        if state != "REPAIRING":
            self.ledger.transition(
                experiment_id,
                state,
                "REPAIRING",
                operation="run",
                reason=(
                    "generated child failed candidate-controlled execution; enter bounded repair"
                ),
                metadata={"failure_category": failure_category},
            )
            state = "REPAIRING"

        while repair_count < maximum_repairs:
            repair_count += 1
            failed_snapshot = snapshot_materialized_candidate(
                failed_candidate,
                candidate_id=failed_candidate_id,
            )
            repair_request = RepairRequest.create(
                request_id=f"{experiment_id}-repair-{repair_count}",
                proposal_id=proposal.proposal_id,
                failed_candidate_id=failed_candidate_id,
                failed_child=failed_snapshot,
                failure_category=current_category,
                diagnostics=current_diagnostic,
                remaining_repairs=maximum_repairs - repair_count + 1,
                safe_context=context,
            )
            try:
                package = self.model.repair(repair_request)
            except Exception as exc:
                provider_diagnostic = f"repair provider failure: {type(exc).__name__}: {exc}"
                self._failure(
                    experiment_id=experiment_id,
                    ordinal=repair_count,
                    category="provider_error",
                    diagnostic=provider_diagnostic,
                    repair_action="close failed child",
                )
                return None, repair_count, state, provider_diagnostic
            transcript = self._transcript(
                ResearchOperation.REPAIR,
                repair_request.to_wire(),
                package.to_wire(),
            )
            candidate_id = f"{experiment_id}-repair-{repair_count}"
            child: MaterializedCandidate | None = None
            try:
                child, source = self._record_child(
                    experiment_id=experiment_id,
                    candidate_id=candidate_id,
                    parent=failed_snapshot,
                    package=package,
                    transcript_role="repair_transcript",
                    transcript=transcript,
                )
                validate_candidate_static(child)
                material = require_material_executable_change(original_parent, child)
            except (CandidateMaterializationError, CandidateStaticError) as exc:
                current_diagnostic = str(exc)
                current_category = (
                    FailureCategory.SYNTAX_ERROR.value
                    if "invalid Python" in current_diagnostic
                    else FailureCategory.STATIC_POLICY.value
                )
                self._failure(
                    experiment_id=experiment_id,
                    ordinal=repair_count,
                    category=current_category,
                    diagnostic=current_diagnostic,
                    repair_action=(
                        "request bounded exact-child repair"
                        if repair_count < maximum_repairs
                        else "close failed child"
                    ),
                )
                if child is None or repair_count >= maximum_repairs:
                    return None, repair_count, state, current_diagnostic
                failed_candidate = child
                failed_candidate_id = candidate_id
                continue

            self.ledger.transition(
                experiment_id,
                state,
                "MATERIALIZED",
                operation=ResearchOperation.REPAIR.value,
                reason="bounded exact-child repair passed source and static gates",
                metadata={
                    "candidate_id": candidate_id,
                    "recovered_failure_category": failure_category,
                    "source_digest": child.source_digest,
                    "diff_digest": child.diff_digest,
                    "changed_symbols": list(material.changed_symbols),
                },
            )
            return (
                _ReadyCandidate(candidate_id, child, material, source),
                repair_count,
                "MATERIALIZED",
                None,
            )
        return None, repair_count, state, current_diagnostic

    def _prepare_candidate(
        self,
        *,
        experiment_id: str,
        proposal: Proposal,
        original_parent: ParentSnapshot,
        state: str,
    ) -> tuple[_ReadyCandidate | None, int, str, str | None]:
        context = self.safe_context.to_wire()
        implementation_request = ImplementationRequest.create(
            request_id=f"{experiment_id}-implement",
            proposal=proposal,
            parent=original_parent,
            safe_context=context,
        )
        try:
            package = self.model.implement(implementation_request)
        except Exception as exc:
            diagnostic = f"implementation provider failure: {type(exc).__name__}: {exc}"
            self._failure(
                experiment_id=experiment_id,
                ordinal=0,
                category="provider_error",
                diagnostic=diagnostic,
            )
            return None, 0, state, diagnostic
        transcript = self._transcript(
            ResearchOperation.IMPLEMENT,
            implementation_request.to_wire(),
            package.to_wire(),
        )
        current_parent = original_parent
        current_package = package
        repair_count = 0
        maximum_repairs = min(proposal.maximum_repairs, 2)

        while True:
            child: MaterializedCandidate | None = None
            candidate_id = (
                f"{experiment_id}-child-0"
                if repair_count == 0
                else f"{experiment_id}-repair-{repair_count}"
            )
            role = "implementation_transcript" if repair_count == 0 else "repair_transcript"
            try:
                child, source = self._record_child(
                    experiment_id=experiment_id,
                    candidate_id=candidate_id,
                    parent=current_parent,
                    package=current_package,
                    transcript_role=role,
                    transcript=transcript,
                )
                validate_candidate_static(child)
                material = require_material_executable_change(original_parent, child)
            except (CandidateMaterializationError, CandidateStaticError) as exc:
                diagnostic = str(exc)
                category = (
                    FailureCategory.SYNTAX_ERROR.value
                    if "invalid Python" in diagnostic
                    else FailureCategory.STATIC_POLICY.value
                )
                self._failure(
                    experiment_id=experiment_id,
                    ordinal=repair_count,
                    category=category,
                    diagnostic=diagnostic,
                    repair_action=(
                        "request bounded exact-child repair"
                        if repair_count < maximum_repairs
                        else "close failed child"
                    ),
                )
                if repair_count >= maximum_repairs:
                    return None, repair_count, state, diagnostic
                if state != "REPAIRING":
                    self.ledger.transition(
                        experiment_id,
                        state,
                        "REPAIRING",
                        operation=ResearchOperation.IMPLEMENT.value,
                        reason="generated child failed static policy; enter bounded repair",
                    )
                    state = "REPAIRING"
                failed_snapshot = (
                    snapshot_materialized_candidate(child, candidate_id=candidate_id)
                    if child is not None
                    else original_parent
                )
                repair_count += 1
                repair_request = RepairRequest.create(
                    request_id=f"{experiment_id}-repair-{repair_count}",
                    proposal_id=proposal.proposal_id,
                    failed_candidate_id=candidate_id,
                    failed_child=failed_snapshot,
                    failure_category=category,
                    diagnostics=diagnostic,
                    remaining_repairs=maximum_repairs - repair_count + 1,
                    safe_context=context,
                    rejected_package=RejectedPackageSnapshot.from_generated_package(
                        current_package
                    ),
                )
                try:
                    repaired = self.model.repair(repair_request)
                except Exception as repair_exc:
                    repair_diagnostic = (
                        f"repair provider failure: {type(repair_exc).__name__}: {repair_exc}"
                    )
                    self._failure(
                        experiment_id=experiment_id,
                        ordinal=repair_count,
                        category="provider_error",
                        diagnostic=repair_diagnostic,
                        repair_action="close failed child",
                    )
                    return None, repair_count, state, repair_diagnostic
                transcript = self._transcript(
                    ResearchOperation.REPAIR,
                    repair_request.to_wire(),
                    repaired.to_wire(),
                )
                current_parent = original_parent
                current_package = repaired
                continue

            operation = (
                ResearchOperation.IMPLEMENT.value
                if repair_count == 0
                else ResearchOperation.REPAIR.value
            )
            self.ledger.transition(
                experiment_id,
                state,
                "MATERIALIZED",
                operation=operation,
                reason="generated child passed source, static, and material-change gates",
                metadata={
                    "candidate_id": candidate_id,
                    "source_digest": child.source_digest,
                    "diff_digest": child.diff_digest,
                    "changed_symbols": list(material.changed_symbols),
                },
            )
            return (
                _ReadyCandidate(candidate_id, child, material, source),
                repair_count,
                "MATERIALIZED",
                None,
            )

    def _runner_artifacts(self, execution: ExecutionResult) -> tuple[tuple[str, ArtifactSpec], ...]:
        result_ref = self.artifacts.put_bytes(
            canonical_json_bytes(execution.manifest()), kind=ArtifactKind.MANIFEST
        )
        stdout_ref = self.artifacts.put_file(execution.stdout.path, kind=ArtifactKind.LOG)
        stderr_ref = self.artifacts.put_file(execution.stderr.path, kind=ArtifactKind.LOG)
        return (
            ("runner_result", artifact_spec(result_ref)),
            ("stdout", artifact_spec(stdout_ref)),
            ("stderr", artifact_spec(stderr_ref)),
        )

    def _execute(
        self,
        *,
        experiment_id: str,
        iteration: int,
        ready: _ReadyCandidate,
        failure_ordinal: int,
        repair_available: bool,
    ) -> tuple[ExecutionResult | None, TrustedEvaluation | None, str | None, str | None]:
        execution_id = (
            f"{experiment_id}-execution"
            if failure_ordinal == 0
            else f"{experiment_id}-execution-repair-{failure_ordinal}"
        )
        workspace: CandidateWorkspace | None = None
        try:
            workspace = self.workspace_materializer.materialize(
                self.execution.workspace_spec(
                    execution_id=execution_id,
                    source_snapshot=ready.source_artifact,
                    source_digest=ready.candidate.source_digest,
                )
            )
            nonce = hashlib.sha256(
                f"{self.ledger.campaign_id}:{execution_id}:{ready.candidate.source_digest}".encode()
            ).hexdigest()[:32]
            spec = self.execution.execution_spec(
                execution_id=execution_id,
                nonce=nonce,
                workspace=workspace,
                source_digest=ready.candidate.source_digest,
            )
            self.ledger.begin_execution(
                experiment_id=experiment_id,
                iteration=iteration,
                spec=spec,
                workspace=workspace,
                environment_digest=self.execution.environment_digest,
            )
            result = self.runner.run(
                spec,
                commit_launch=lambda process: self.ledger.mark_execution_running(
                    execution_id, process
                ),
            )
            self.ledger.finish_execution(result, artifacts=self._runner_artifacts(result))
            if not result.succeeded:
                diagnostic = self._runner_failure_diagnostic(result)
                category = self._runner_failure_category(diagnostic)
                self._failure(
                    experiment_id=experiment_id,
                    ordinal=failure_ordinal,
                    category=category,
                    diagnostic=diagnostic,
                    execution_id=execution_id,
                    repair_action=(
                        "request bounded exact-child repair"
                        if repair_available
                        else "close failed child"
                    ),
                )
                return result, None, diagnostic, category
            if result.candidate_metrics_accepted:
                raise ResearchLoopError("runner must never accept candidate-declared metrics")
            evaluation = self.evaluator.evaluate(
                candidate=ready.candidate,
                workspace=workspace,
                execution=result,
            )
            if not isinstance(evaluation, TrustedEvaluation):
                raise ResearchLoopError("trusted evaluator returned an unsupported result")
            self.ledger.record_metric(
                experiment_id=experiment_id,
                execution_id=execution_id,
                evaluation=evaluation,
            )
            return result, evaluation, None, None
        except Exception as exc:
            diagnostic = f"trusted execution/evaluation failure: {type(exc).__name__}: {exc}"
            self._failure(
                experiment_id=experiment_id,
                ordinal=failure_ordinal,
                category=FailureCategory.OUTPUT_CONTRACT.value,
                diagnostic=diagnostic,
                execution_id=None,
            )
            return None, None, diagnostic, FailureCategory.OUTPUT_CONTRACT.value
        finally:
            if workspace is not None:
                self.workspace_materializer.cleanup(workspace)

    def _close(
        self,
        *,
        experiment_id: str,
        state: str,
        eligible_outer_primary: float | None,
        operation: str,
        reason: str,
    ) -> None:
        self.ledger.transition(
            experiment_id,
            state,
            "CLOSED",
            operation=operation,
            reason=reason,
        )
        current = self.ledger.convergence()
        updated = current.update_after_iteration(eligible_outer_primary)
        self.ledger.update_convergence(updated)

    def _close_pre_admission_rejection(
        self,
        *,
        iteration: int,
        experiment_id: str,
        proposal: Proposal,
        parent: ParentSnapshot,
        state: str,
        diagnostic: str,
        fingerprint: str,
    ) -> tuple[IterationResult, ParentSnapshot]:
        self._failure(
            experiment_id=experiment_id,
            ordinal=0,
            category=FailureCategory.STATIC_POLICY.value,
            diagnostic=diagnostic,
            repair_action="request a legal and novel proposal",
        )
        self.ledger.transition(
            experiment_id,
            state,
            "FAILED",
            operation=ResearchOperation.PROPOSE.value,
            reason="proposal failed controller admission before implementation",
            metadata={"admission_fingerprint": fingerprint},
        )
        self._close(
            experiment_id=experiment_id,
            state="FAILED",
            eligible_outer_primary=None,
            operation="close",
            reason="close rejected proposal without invoking implementation",
        )
        return (
            IterationResult(
                iteration,
                experiment_id,
                proposal.proposal_id,
                None,
                "failed",
                0,
                None,
                diagnostic,
            ),
            parent,
        )

    def _run_iteration(
        self,
        *,
        iteration: int,
        parent: ParentSnapshot,
    ) -> tuple[IterationResult, ParentSnapshot]:
        experiment_id = f"iteration-{iteration:02d}"
        context = self.safe_context.to_wire()
        proposal_request = ProposalRequest.create(
            request_id=f"{experiment_id}-propose",
            campaign_id=self.ledger.campaign_id,
            scientific_iteration=iteration,
            parent_candidate_id=parent.candidate_id,
            safe_context=context,
        )
        proposal = self.model.propose(proposal_request)
        if proposal.parent_candidate_id != parent.candidate_id:
            raise ResearchLoopError("proposal did not preserve the exact incumbent parent")
        self.ledger.create_iteration(experiment_id, iteration, proposal)
        proposal_transcript = self._transcript(
            ResearchOperation.PROPOSE,
            proposal_request.to_wire(),
            proposal.to_wire(),
        )
        self.ledger.record_proposal(
            experiment_id,
            proposal_request.digest,
            proposal,
            proposal_transcript,
        )
        self.ledger.transition(
            experiment_id,
            "PLANNED",
            "PROPOSED",
            operation=ResearchOperation.PROPOSE.value,
            reason="persist accepted typed proposal and complete transcript",
        )
        state = "PROPOSED"
        try:
            legal_manifest = proposal_request.source_policy.validate_manifest(
                proposal.files_expected,
                parent_paths=(),
            )
        except CandidateManifestPolicyError as exc:
            policy_diagnostic = f"{exc.fingerprint}: {exc}"
            return self._close_pre_admission_rejection(
                iteration=iteration,
                experiment_id=experiment_id,
                proposal=proposal,
                parent=parent,
                state=state,
                diagnostic=policy_diagnostic,
                fingerprint=exc.fingerprint,
            )
        method_family = classify_method_family(proposal.mechanism, proposal.objective)
        signature = proposal_novelty_signature(
            parent_digest=parent.digest,
            method_family=method_family,
            mechanism=proposal.mechanism,
            objective=proposal.objective,
            sampling=proposal.sampling,
            required_source_fields=(value.source_field for value in proposal.required_fields),
            legal_manifest=legal_manifest,
            evidence_cursor=self._training_evidence_cursor,
        )
        if signature in self._proposal_signatures:
            fingerprint = f"proposal_novelty:duplicate:{method_family}"
            return self._close_pre_admission_rejection(
                iteration=iteration,
                experiment_id=experiment_id,
                proposal=proposal,
                parent=parent,
                state=state,
                diagnostic=(
                    f"{fingerprint}: normalized proposal duplicate has no new trusted "
                    "training evidence"
                ),
                fingerprint=fingerprint,
            )
        family_admissions = self._family_implementation_admissions.get(method_family, 0)
        if family_admissions >= 2:
            fingerprint = f"proposal_novelty:family_pretraining_limit:{method_family}"
            return self._close_pre_admission_rejection(
                iteration=iteration,
                experiment_id=experiment_id,
                proposal=proposal,
                parent=parent,
                state=state,
                diagnostic=(
                    f"{fingerprint}: two proposals from this scientific family already "
                    "reached implementation without trusted training evidence"
                ),
                fingerprint=fingerprint,
            )
        self._proposal_signatures.add(signature)
        self._family_implementation_admissions[method_family] = family_admissions + 1
        ready, repairs, state, diagnostic = self._prepare_candidate(
            experiment_id=experiment_id,
            proposal=proposal,
            original_parent=parent,
            state=state,
        )
        if ready is None:
            self.ledger.transition(
                experiment_id,
                state,
                "FAILED",
                operation=ResearchOperation.IMPLEMENT.value,
                reason="implementation or bounded repair failed",
            )
            self._close(
                experiment_id=experiment_id,
                state="FAILED",
                eligible_outer_primary=None,
                operation="close",
                reason="close failed scientific iteration without changing incumbent",
            )
            return (
                IterationResult(
                    iteration,
                    experiment_id,
                    proposal.proposal_id,
                    None,
                    "failed",
                    repairs,
                    None,
                    diagnostic,
                ),
                parent,
            )

        evaluation: TrustedEvaluation | None = None
        run_diagnostic: str | None = None
        attempted_candidate_id = ready.candidate_id
        while evaluation is None:
            self.ledger.transition(
                experiment_id,
                "MATERIALIZED",
                "RUNNING",
                operation="run",
                reason="launch the statically valid material generated child locally",
            )
            attempted_candidate_id = ready.candidate_id
            maximum_repairs = min(proposal.maximum_repairs, 2)
            execution_result, evaluation, run_diagnostic, failure_category = self._execute(
                experiment_id=experiment_id,
                iteration=iteration,
                ready=ready,
                failure_ordinal=repairs,
                repair_available=repairs < maximum_repairs,
            )
            if evaluation is not None:
                break
            if (
                execution_result is not None
                and not execution_result.succeeded
                and failure_category
                in {
                    FailureCategory.IMPORT_ERROR.value,
                }
                and repairs < maximum_repairs
                and run_diagnostic is not None
            ):
                repaired, repairs, state, repair_diagnostic = self._repair_after_execution_failure(
                    experiment_id=experiment_id,
                    proposal=proposal,
                    original_parent=parent,
                    failed=ready,
                    repair_count=repairs,
                    state="RUNNING",
                    failure_category=failure_category,
                    diagnostic=run_diagnostic,
                )
                if repaired is not None:
                    ready = repaired
                    continue
                run_diagnostic = repair_diagnostic
            else:
                state = "RUNNING"
            self.ledger.transition(
                experiment_id,
                state,
                "FAILED",
                operation="run",
                reason="runner or trusted evaluation rejected the child",
            )
            self._close(
                experiment_id=experiment_id,
                state="FAILED",
                eligible_outer_primary=None,
                operation="close",
                reason="close failed execution without changing incumbent",
            )
            return (
                IterationResult(
                    iteration,
                    experiment_id,
                    proposal.proposal_id,
                    attempted_candidate_id,
                    "failed",
                    repairs,
                    None,
                    run_diagnostic,
                ),
                parent,
            )

        assert evaluation is not None

        self._training_evidence_cursor = canonical_digest(
            {
                "schema_version": 1,
                "previous_cursor": self._training_evidence_cursor,
                "candidate_source_digest": ready.candidate.source_digest,
                "trusted_evaluation": evaluation.summary.to_wire(),
            }
        )
        self._family_implementation_admissions[method_family] = 0

        self.ledger.transition(
            experiment_id,
            "RUNNING",
            "EVALUATED",
            operation="evaluate",
            reason="accept only trusted output validation and protected scorer metrics",
            artifacts=evaluation.artifacts,
            metadata={"primary": evaluation.summary.primary},
        )
        reflection_request = ReflectionRequest.create(
            request_id=f"{experiment_id}-reflect",
            proposal_id=proposal.proposal_id,
            candidate_id=ready.candidate_id,
            source_digest=ready.candidate.source_digest,
            diff_digest=ready.candidate.diff_digest,
            result=evaluation.summary,
            safe_context=context,
        )
        try:
            reflection = self.model.reflect(reflection_request)
        except Exception as exc:
            reflection_diagnostic = f"reflection provider failure: {type(exc).__name__}: {exc}"
            self._failure(
                experiment_id=experiment_id,
                ordinal=0,
                category="provider_error",
                diagnostic=reflection_diagnostic,
            )
            self.ledger.transition(
                experiment_id,
                "EVALUATED",
                "FAILED",
                operation=ResearchOperation.REFLECT.value,
                reason="reflection provider failed without changing incumbent",
            )
            eligible = (
                evaluation.summary.primary
                if evaluation.eligible and evaluation.summary.tier == "outer"
                else None
            )
            self._close(
                experiment_id=experiment_id,
                state="FAILED",
                eligible_outer_primary=eligible,
                operation="close",
                reason="close evaluated iteration after reflection failure",
            )
            return (
                IterationResult(
                    iteration,
                    experiment_id,
                    proposal.proposal_id,
                    ready.candidate_id,
                    "failed",
                    repairs,
                    evaluation,
                    reflection_diagnostic,
                ),
                parent,
            )
        reflection_artifact = self._transcript(
            ResearchOperation.REFLECT,
            reflection_request.to_wire(),
            reflection.to_wire(),
        )
        self.ledger.transition(
            experiment_id,
            "EVALUATED",
            "REFLECTED",
            operation=ResearchOperation.REFLECT.value,
            reason="persist typed reflection after trusted evaluation",
            artifacts=(("reflection_transcript", reflection_artifact),),
            metadata={"reflection_digest": reflection.digest},
        )
        evidence = CandidateEvidence(
            ready.candidate_id,
            ready.candidate.source_digest,
            ready.candidate.diff_digest,
            ready.material_change,
            evaluation,
        )
        incumbent = self.ledger.incumbent()
        decision = self.selector.decide(incumbent.outer_primary_mean, evidence)
        if not isinstance(decision, SelectionDecision):
            raise ResearchLoopError("trusted selector returned an unsupported decision")
        next_parent = parent
        status = "retained"
        if decision.promote:
            self.ledger.promote(
                experiment_id=experiment_id,
                candidate=evidence,
                decision=decision,
            )
            next_parent = snapshot_materialized_candidate(
                ready.candidate, candidate_id=ready.candidate_id
            )
            status = "promoted"
        eligible_outer = (
            evaluation.summary.primary
            if evaluation.eligible and evaluation.summary.tier == "outer"
            else None
        )
        self._close(
            experiment_id=experiment_id,
            state="REFLECTED",
            eligible_outer_primary=eligible_outer,
            operation="selection",
            reason=decision.reason,
        )
        return (
            IterationResult(
                iteration,
                experiment_id,
                proposal.proposal_id,
                ready.candidate_id,
                status,
                repairs,
                evaluation,
            ),
            next_parent,
        )

    def run(self, *, parent: ParentSnapshot, max_iterations: int) -> CampaignLoopResult:
        """Run at most ``max_iterations`` while preserving the incumbent across every failure."""

        if not isinstance(parent, ParentSnapshot):
            raise ResearchLoopError("parent must be ParentSnapshot")
        if type(max_iterations) is not int or not 1 <= max_iterations <= _MAX_ITERATIONS:
            raise ResearchLoopError("max_iterations must be an integer in [1, 50]")
        incumbent = self.ledger.incumbent()
        if parent.candidate_id != incumbent.incumbent_id:
            raise ResearchLoopError("initial parent must be the ledger's replay-verified incumbent")
        results: list[IterationResult] = []
        current_parent = parent
        for _ in range(max_iterations):
            convergence = self.ledger.convergence()
            if not convergence.may_launch_scientific_iteration:
                break
            iteration = convergence.completed_iterations + 1
            item, current_parent = self._run_iteration(
                iteration=iteration,
                parent=current_parent,
            )
            results.append(item)
        return CampaignLoopResult(
            selected_candidate_id=self.ledger.incumbent().incumbent_id,
            fallback_candidate_id=self.ledger.fallback_candidate_id,
            iterations=tuple(results),
            final_parent=current_parent,
        )


__all__ = [
    "CampaignLoopResult",
    "CampaignStoreResearchLedger",
    "CandidateEvidence",
    "IterationResult",
    "LocalExecutionTemplate",
    "ResearchCampaignLoop",
    "ResearchLedger",
    "ResearchLoopError",
    "SelectionDecision",
    "TrustedEvaluation",
    "TrustedEvaluator",
    "TrustedSelector",
    "artifact_spec",
]
