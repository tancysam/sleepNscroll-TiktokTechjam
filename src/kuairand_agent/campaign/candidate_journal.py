"""CampaignStore-backed durable journal for generated candidate execution.

The adapter is deliberately narrow: it reconstructs launch admission from the append-only
campaign store, reserves charged training work before Runner is entered, records the blocked
process receipt before release, and commits terminal artifact closure.  Prediction/replay work is
durable but never consumes a training launch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

from kuairand_agent.campaign.budgets import (
    AdmissionReason,
    BudgetLedger,
    BudgetReallocation,
    LaunchCategory,
    LaunchCharge,
    LaunchRequest,
    WorkKind,
    WorkPhase,
)
from kuairand_agent.campaign.clock import DeadlineObservation
from kuairand_agent.campaign.convergence import ConvergenceState
from kuairand_agent.campaign.models import ScientificIterationCount
from kuairand_agent.campaign.store import (
    ArtifactSpec,
    CampaignStore,
    ExecutionRecord,
)
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactRef, ArtifactStore
from kuairand_agent.execution.candidate_executor import (
    CandidateAction,
    CandidateExecutionArtifacts,
    CandidateExecutionJournal,
)
from kuairand_agent.execution.runner import ExecutionResult, ExecutionSpec, ProcessRecord
from kuairand_agent.execution.workspace import CandidateWorkspace

_QUALIFICATION_CATEGORY: Final = "baseline_qualification"
_RESULT_DOMAIN: Final = b"kuairand-candidate-execution-result-v1\0"


class CandidateJournalError(RuntimeError):
    """Durable generated-candidate execution cannot safely continue."""


class CandidateAdmissionError(CandidateJournalError):
    """Budget or deadline policy rejected an execution before reservation."""

    def __init__(self, reason: AdmissionReason) -> None:
        self.reason = reason
        super().__init__(f"candidate execution admission rejected: {reason.value}")


class CandidateExecutionPendingError(CandidateJournalError):
    """An exact execution is already STARTING or RUNNING and needs reconciliation."""


@dataclass(frozen=True, slots=True)
class RehydratedCandidateExecution:
    """Exact terminal evidence restored without rerunning generated source."""

    execution: ExecutionRecord
    action: CandidateAction
    artifacts: CandidateExecutionArtifacts


class CandidateExecutionTerminalError(CandidateJournalError):
    """Prepare found an exact terminal execution whose artifacts can be rehydrated."""

    def __init__(self, terminal: RehydratedCandidateExecution) -> None:
        self.terminal = terminal
        super().__init__(
            f"candidate execution {terminal.execution.execution_id!r} is already "
            f"{terminal.execution.status}"
        )


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise CandidateJournalError(f"{location} must be non-empty text without NUL bytes")
    return value


def _seconds(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateJournalError(f"{location} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CandidateJournalError(f"{location} must be a non-negative finite number")
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _result_digest(result: ExecutionResult) -> str:
    return hashlib.sha256(_RESULT_DOMAIN + _canonical_json(result.manifest())).hexdigest()


def _result_manifest_digest(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(_RESULT_DOMAIN + _canonical_json(manifest)).hexdigest()


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateJournalError("campaign hard deadline is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise CandidateJournalError("campaign hard deadline is not timezone-aware")
    return parsed


def _category(value: str) -> LaunchCategory:
    normalized = (
        LaunchCategory.BASELINE_QUALIFICATION_REPLAY.value
        if value == _QUALIFICATION_CATEGORY
        else value
    )
    try:
        return LaunchCategory(normalized)
    except ValueError as exc:
        raise CandidateJournalError(f"stored launch has unknown category {value!r}") from exc


def reconstruct_budget_ledger(store: CampaignStore) -> BudgetLedger:
    """Rebuild the immutable budget projection solely from authoritative store records.

    ``NOT_STARTED`` reservations are absent from charges.  Consequently, physical launch-number
    gaps are intentionally compacted into BudgetLedger's logical one-based charge sequence while
    the original append-only numbers remain available in :meth:`CampaignStore.launches`.
    """

    if not isinstance(store, CampaignStore):
        raise CandidateJournalError("store must be a CampaignStore")
    snapshot = store.snapshot()
    identity = store.identity()
    convergence = ConvergenceState.from_manifest(snapshot.convergence_state)
    charged_records = tuple(record for record in store.launches() if record.charged)
    if len(charged_records) != snapshot.launches_used:
        raise CandidateJournalError("campaign launch summary differs from append-only records")

    charges: list[LaunchCharge] = []
    for logical_number, record in enumerate(charged_records, start=1):
        category = _category(record.category)
        original = _category(record.original_category)
        imported = record.category == _QUALIFICATION_CATEGORY
        if category is LaunchCategory.RECOVERY_RESERVE:
            if original is LaunchCategory.RECOVERY_RESERVE:
                raise CandidateJournalError(
                    "stored recovery launch lost its non-recovery original category"
                )
        elif original is not category:
            raise CandidateJournalError("stored non-repair launch changed its original category")
        charges.append(
            LaunchCharge(
                launch_number=logical_number,
                execution_id=record.launch_id,
                family=record.purpose,
                category=category,
                original_category=original,
                repair_child=category is LaunchCategory.RECOVERY_RESERVE,
                imported_from_qualification=imported,
            )
        )

    if snapshot.qualification_digest is not None:
        imported_positions = tuple(
            index
            for index, charge in enumerate(charges, start=1)
            if charge.imported_from_qualification
        )
        if imported_positions != tuple(range(1, 7)):
            raise CandidateJournalError(
                "qualified campaign does not retain exactly six leading qualification charges"
            )

    transfers = tuple(
        BudgetReallocation(
            source=_category(item.from_category),
            target=_category(item.to_category),
            amount=item.launch_count,
            reason=item.reason,
        )
        for item in store.reallocations()
    )
    return BudgetLedger(
        charges=tuple(charges),
        scientific_iterations=ScientificIterationCount(convergence.completed_iterations),
        reallocations=transfers,
        max_training_launches=identity.max_launches,
    )


@dataclass(frozen=True, slots=True)
class CandidateJournalPolicy:
    """Frozen admission and attribution for one train or prediction execution."""

    family: str
    phase: WorkPhase
    p95_runtime_seconds: float
    cleanup_seconds: float = 0.0
    category: LaunchCategory | None = None
    original_category: LaunchCategory | None = None
    repair_child: bool = False
    experiment_id: str | None = None
    scientific_iteration: int | None = None

    def __post_init__(self) -> None:
        _text(self.family, "policy family")
        if not isinstance(self.phase, WorkPhase):
            raise CandidateJournalError("policy phase must be WorkPhase")
        _seconds(self.p95_runtime_seconds, "policy p95_runtime_seconds")
        _seconds(self.cleanup_seconds, "policy cleanup_seconds")
        if self.category is not None and not isinstance(self.category, LaunchCategory):
            raise CandidateJournalError("policy category must be LaunchCategory or absent")
        if self.original_category is not None and not isinstance(
            self.original_category, LaunchCategory
        ):
            raise CandidateJournalError("policy original_category must be LaunchCategory or absent")
        if type(self.repair_child) is not bool:
            raise CandidateJournalError("policy repair_child must be boolean")
        if self.experiment_id is not None:
            _text(self.experiment_id, "policy experiment_id")
        if self.scientific_iteration is not None and (
            type(self.scientific_iteration) is not int or self.scientific_iteration < 0
        ):
            raise CandidateJournalError(
                "policy scientific_iteration must be a non-negative integer or absent"
            )


def _artifact_spec(reference: ArtifactRef) -> ArtifactSpec:
    return ArtifactSpec(
        digest=reference.sha256,
        kind=reference.kind.value,
        relative_path=reference.object_relative_path.as_posix(),
        size_bytes=reference.size_bytes,
    )


class CampaignStoreCandidateJournal(CandidateExecutionJournal):
    """Durable CandidateExecutionJournal backed by one campaign database."""

    def __init__(
        self,
        *,
        store: CampaignStore,
        artifact_store: ArtifactStore,
        deadline: DeadlineObservation,
        policy: CandidateJournalPolicy,
    ) -> None:
        if not isinstance(store, CampaignStore):
            raise CandidateJournalError("store must be a CampaignStore")
        if not isinstance(artifact_store, ArtifactStore):
            raise CandidateJournalError("artifact_store must be an ArtifactStore")
        if not isinstance(deadline, DeadlineObservation):
            raise CandidateJournalError("deadline must be a DeadlineObservation")
        if not isinstance(policy, CandidateJournalPolicy):
            raise CandidateJournalError("policy must be CandidateJournalPolicy")
        identity_deadline = _parse_timestamp(store.identity().hard_deadline_utc)
        if identity_deadline != deadline.state.utc_deadline:
            raise CandidateJournalError(
                "deadline observation differs from the campaign's immutable hard deadline"
            )
        expected_remaining = max(
            0.0, deadline.state.wall_clock_seconds - deadline.state.last_elapsed_seconds
        )
        if (
            deadline.elapsed_seconds != deadline.state.last_elapsed_seconds
            or deadline.remaining_seconds != expected_remaining
        ):
            raise CandidateJournalError(
                "deadline observation is not its persisted state projection"
            )
        self.store = store
        self.artifact_store = artifact_store
        self.deadline = deadline
        self.policy = policy
        self._prepared_execution_id: str | None = None

    def _request(self, action: CandidateAction, execution_id: str) -> LaunchRequest:
        if not isinstance(action, CandidateAction):
            raise CandidateJournalError("action must be CandidateAction")
        kind = (
            WorkKind.FULL_TRAIN_EVALUATE
            if action is CandidateAction.TRAIN
            else WorkKind.CHECKPOINT_INFERENCE_REPLAY
        )
        return LaunchRequest(
            execution_id=execution_id,
            family=self.policy.family,
            kind=kind,
            phase=self.policy.phase,
            p95_runtime_seconds=self.policy.p95_runtime_seconds,
            cleanup_seconds=self.policy.cleanup_seconds,
            category=self.policy.category,
            original_category=self.policy.original_category,
            repair_child=self.policy.repair_child,
        )

    @staticmethod
    def _launch_id(execution_id: str) -> str:
        return f"{execution_id}-launch"

    @staticmethod
    def _reservation_key(action: CandidateAction, spec: ExecutionSpec) -> str:
        material = {
            "action": action.value,
            "execution_id": spec.execution_id,
            "source_digest": spec.source_digest,
            "config_digest": spec.config_digest,
            "data_digest": spec.data_digest,
            "checkpoint_digest": spec.checkpoint_digest,
        }
        return "candidate:" + hashlib.sha256(_canonical_json(material)).hexdigest()

    def _validate_existing(
        self,
        record: ExecutionRecord,
        *,
        action: CandidateAction,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
    ) -> None:
        expected_launch = (
            self._launch_id(spec.execution_id) if action is CandidateAction.TRAIN else None
        )
        exact = {
            "kind": (record.kind, f"generated_candidate_{action.value}"),
            "tier": (record.tier, workspace.split_role.value),
            "command": (record.command, spec.command),
            "seed": (record.seed, spec.python_hash_seed),
            "nonce": (record.nonce, spec.nonce),
            "source_digest": (record.source_digest, spec.source_digest),
            "config_digest": (record.config_digest, spec.config_digest),
            "capability_digest": (record.capability_digest, workspace.manifest_digest),
            "environment_digest": (
                record.environment_digest,
                self.store.identity().environment_digest,
            ),
            "data_digest": (record.data_digest, spec.data_digest),
            "checkpoint_digest": (record.checkpoint_digest, spec.checkpoint_digest),
            "launch_id": (record.launch_id, expected_launch),
        }
        mismatches = [name for name, (observed, expected) in exact.items() if observed != expected]
        if mismatches:
            raise CandidateJournalError(
                "execution_id already names different immutable evidence: " + ", ".join(mismatches)
            )

    def prepare(
        self,
        *,
        action: CandidateAction,
        spec: ExecutionSpec,
        workspace: CandidateWorkspace,
    ) -> None:
        if not isinstance(spec, ExecutionSpec) or not isinstance(workspace, CandidateWorkspace):
            raise CandidateJournalError("prepare requires ExecutionSpec and CandidateWorkspace")
        if spec.execution_id != workspace.execution_id:
            raise CandidateJournalError("execution and workspace identities differ")
        existing = self.store.execution(spec.execution_id)
        if existing is not None:
            self._validate_existing(existing, action=action, spec=spec, workspace=workspace)
            if existing.status in {"SUCCEEDED", "FAILED"}:
                raise CandidateExecutionTerminalError(self.rehydrate_terminal(spec.execution_id))
            raise CandidateExecutionPendingError(
                f"candidate execution {spec.execution_id!r} is already {existing.status}"
            )

        request = self._request(action, spec.execution_id)
        launch_id: str | None = None
        if action is CandidateAction.TRAIN:
            launch_id = self._launch_id(spec.execution_id)
            stored_launch = next(
                (item for item in self.store.launches() if item.launch_id == launch_id),
                None,
            )
            if stored_launch is None:
                admission = reconstruct_budget_ledger(self.store).admit(
                    request,
                    remaining_seconds=self.deadline.remaining_seconds,
                    finalization_reserve_seconds=self.deadline.state.finalization_reserve_seconds,
                )
                if not admission.allowed:
                    raise CandidateAdmissionError(admission.reason)
            assert self.policy.category is not None  # enforced by LaunchRequest
            expected_original = self.policy.original_category or self.policy.category
            if stored_launch is not None:
                exact_launch = {
                    "reservation_key": (
                        stored_launch.reservation_key,
                        self._reservation_key(action, spec),
                    ),
                    "category": (stored_launch.category, self.policy.category.value),
                    "original_category": (
                        stored_launch.original_category,
                        expected_original.value,
                    ),
                    "purpose": (stored_launch.purpose, self.policy.family),
                    "experiment_id": (
                        stored_launch.experiment_id,
                        self.policy.experiment_id,
                    ),
                    "scientific_iteration": (
                        stored_launch.scientific_iteration,
                        self.policy.scientific_iteration,
                    ),
                    "seed": (stored_launch.seed, spec.python_hash_seed),
                }
                mismatches = [
                    name
                    for name, (observed, expected) in exact_launch.items()
                    if observed != expected
                ]
                if mismatches:
                    raise CandidateJournalError(
                        "orphan launch differs from immutable execution request: "
                        + ", ".join(mismatches)
                    )
                if stored_launch.state != "RESERVED" or not stored_launch.charged:
                    raise CandidateJournalError("orphan launch is not a resumable RESERVED charge")
            self.store.reserve_launch(
                launch_id=launch_id,
                reservation_key=self._reservation_key(action, spec),
                category=self.policy.category.value,
                original_category=expected_original.value,
                purpose=self.policy.family,
                expected_revision=self.store.snapshot().revision,
                experiment_id=self.policy.experiment_id,
                scientific_iteration=self.policy.scientific_iteration,
                seed=spec.python_hash_seed,
                metadata={
                    "action": action.value,
                    "repair_child": self.policy.repair_child,
                    "source_digest": spec.source_digest,
                },
            )
        else:
            admission = reconstruct_budget_ledger(self.store).admit(
                request,
                remaining_seconds=self.deadline.remaining_seconds,
                finalization_reserve_seconds=self.deadline.state.finalization_reserve_seconds,
            )
            if not admission.allowed:
                raise CandidateAdmissionError(admission.reason)

        self.store.create_execution(
            execution_id=spec.execution_id,
            kind=f"generated_candidate_{action.value}",
            tier=workspace.split_role.value,
            command=spec.command,
            expected_revision=self.store.snapshot().revision,
            experiment_id=self.policy.experiment_id,
            launch_id=launch_id,
            seed=spec.python_hash_seed,
            status="STARTING",
            source_digest=spec.source_digest,
            config_digest=spec.config_digest,
            capability_digest=workspace.manifest_digest,
            environment_digest=self.store.identity().environment_digest,
            data_digest=spec.data_digest,
            checkpoint_digest=spec.checkpoint_digest,
            nonce=spec.nonce,
            metadata={
                "action": action.value,
                "family": self.policy.family,
                "workspace_manifest_digest": workspace.manifest_digest,
            },
        )
        self._prepared_execution_id = spec.execution_id

    def commit(self, process: ProcessRecord) -> None:
        if not isinstance(process, ProcessRecord):
            raise CandidateJournalError("commit requires a ProcessRecord")
        if self._prepared_execution_id != process.execution_id:
            raise CandidateJournalError("process was not prepared by this journal")
        record = self.store.execution(process.execution_id)
        if record is None:
            raise CandidateJournalError("prepared execution disappeared before process commit")
        if record.status not in {"STARTING", "RUNNING"}:
            raise CandidateJournalError("terminal execution cannot accept a process receipt")
        self.store.transition_execution(
            process.execution_id,
            from_state="STARTING",
            to_state="RUNNING",
            expected_revision=self.store.snapshot().revision,
            reason="persist exact generated-candidate process receipt before release",
            process_record_digest=process.digest,
            process_record=process.manifest(),
            metadata={"candidate_released": False},
        )
        record = self.store.execution(process.execution_id)
        assert record is not None
        if record.launch_id is not None:
            self.store.transition_launch(
                record.launch_id,
                to_state="STARTED",
                expected_revision=self.store.snapshot().revision,
                start_receipt_digest=process.digest,
                metadata={"execution_id": process.execution_id},
            )

    def _verified_artifact_specs(
        self, artifacts: CandidateExecutionArtifacts
    ) -> tuple[tuple[str, ArtifactSpec], ...]:
        specs: list[tuple[str, ArtifactSpec]] = []
        for role, reference in artifacts.entries:
            self.artifact_store.verify(reference)
            specs.append((role, _artifact_spec(reference)))
        return tuple(specs)

    def _validate_runner_artifacts(
        self,
        *,
        action: CandidateAction,
        result: ExecutionResult,
        artifacts: CandidateExecutionArtifacts,
    ) -> None:
        base = {"execution_manifest", "stderr", "stdout"}
        success = (
            {"candidate_result", "checkpoint"}
            if action is CandidateAction.TRAIN
            else {"prediction", "prediction_result"}
        )
        observed_roles = {role for role, _ in artifacts.entries}
        optional_roles = observed_roles & {"failure_diagnostic", "workspace_cleanup"}
        if artifacts.output_validated and "failure_diagnostic" in optional_roles:
            raise CandidateJournalError("successful execution cannot retain a failure diagnostic")
        expected_roles = base | (success if artifacts.output_validated else set()) | optional_roles
        if observed_roles != expected_roles:
            raise CandidateJournalError(
                "execution artifact roles differ from the frozen candidate action closure"
            )
        expected_kinds = {
            "execution_manifest": ArtifactKind.MANIFEST,
            "stderr": ArtifactKind.LOG,
            "stdout": ArtifactKind.LOG,
            "failure_diagnostic": ArtifactKind.LOG,
            "workspace_cleanup": ArtifactKind.MANIFEST,
            "candidate_result": ArtifactKind.MANIFEST,
            "checkpoint": ArtifactKind.CHECKPOINT,
            "prediction": ArtifactKind.PREDICTION,
            "prediction_result": ArtifactKind.MANIFEST,
        }
        for role, reference in artifacts.entries:
            if reference.kind is not expected_kinds[role]:
                raise CandidateJournalError(
                    f"execution artifact role {role!r} has the wrong immutable kind"
                )
        manifest_reference = artifacts.artifact("execution_manifest")
        if self.artifact_store.read_bytes(
            manifest_reference, max_bytes=manifest_reference.size_bytes
        ) != _canonical_json(result.manifest()):
            raise CandidateJournalError(
                "execution manifest artifact does not match the trusted runner result"
            )
        for role, evidence in (("stdout", result.stdout), ("stderr", result.stderr)):
            reference = artifacts.artifact(role)
            if (
                reference.sha256 != evidence.sha256
                or reference.size_bytes != evidence.retained_bytes
            ):
                raise CandidateJournalError(
                    f"{role} artifact does not match the trusted bounded log evidence"
                )

    def finish(
        self,
        *,
        action: CandidateAction,
        result: ExecutionResult,
        artifacts: CandidateExecutionArtifacts,
    ) -> None:
        if not isinstance(action, CandidateAction):
            raise CandidateJournalError("finish action must be CandidateAction")
        if not isinstance(result, ExecutionResult) or not isinstance(
            artifacts, CandidateExecutionArtifacts
        ):
            raise CandidateJournalError(
                "finish requires ExecutionResult and CandidateExecutionArtifacts"
            )
        record = self.store.execution(result.execution_id)
        if record is None:
            raise CandidateJournalError("runner result has no durable prepared execution")
        if record.kind != f"generated_candidate_{action.value}":
            raise CandidateJournalError("runner result action differs from prepared execution")
        if (action is CandidateAction.TRAIN) != (record.launch_id is not None):
            raise CandidateJournalError("prepared execution launch ownership differs from action")
        if artifacts.output_validated and not result.succeeded:
            raise CandidateJournalError("failed runner result cannot have validated output")
        if result.succeeded and not result.candidate_released:
            raise CandidateJournalError("successful runner result was never released")
        if result.candidate_released and record.process_record_digest is None:
            raise CandidateJournalError("released candidate has no durable process receipt")
        if record.process_record_digest is not None and (
            result.process is None or result.process.digest != record.process_record_digest
        ):
            raise CandidateJournalError("runner result process differs from durable receipt")

        terminal_status = (
            "SUCCEEDED" if result.succeeded and artifacts.output_validated else "FAILED"
        )
        result_digest = _result_digest(result)
        self._validate_runner_artifacts(action=action, result=result, artifacts=artifacts)
        artifact_specs = self._verified_artifact_specs(artifacts)
        if record.status in {"SUCCEEDED", "FAILED"}:
            if (
                record.status != terminal_status
                or record.result_digest != result_digest
                or self.store.artifacts_for(owner_type="execution", owner_id=result.execution_id)
                != artifact_specs
            ):
                raise CandidateJournalError(
                    "terminal execution retry differs from immutable result or artifacts"
                )
        elif record.status in {"STARTING", "RUNNING"}:
            self.store.transition_execution(
                result.execution_id,
                from_state=record.status,
                to_state=terminal_status,
                expected_revision=self.store.snapshot().revision,
                reason="persist generated-candidate terminal result and artifact closure",
                result_digest=result_digest,
                finished_at=result.ended_at_utc,
                artifacts=artifact_specs,
                metadata={
                    "action": action.value,
                    "artifact_closure_digest": artifacts.closure_digest,
                    "candidate_metrics_accepted": False,
                    "candidate_released": result.candidate_released,
                    "output_validated": artifacts.output_validated,
                    "outcome": result.outcome.value,
                },
            )
        else:
            raise CandidateJournalError(f"unsupported execution state {record.status!r}")

        refreshed = self.store.execution(result.execution_id)
        assert refreshed is not None
        if refreshed.launch_id is not None:
            launch = next(
                item for item in self.store.launches() if item.launch_id == refreshed.launch_id
            )
            desired: Literal["FINISHED", "NOT_STARTED"]
            if result.candidate_released:
                if launch.state not in {"STARTED", "START_UNCERTAIN", "FINISHED"}:
                    raise CandidateJournalError(
                        "released training result has no durable STARTED launch"
                    )
                desired = "FINISHED"
            else:
                if launch.state in {"STARTED", "START_UNCERTAIN"}:
                    desired = "FINISHED"
                elif launch.state in {"RESERVED", "NOT_STARTED"}:
                    desired = "NOT_STARTED"
                else:
                    desired = "FINISHED"
            if launch.state != desired:
                self.store.transition_launch(
                    launch.launch_id,
                    to_state=desired,
                    expected_revision=self.store.snapshot().revision,
                    metadata={"execution_id": result.execution_id},
                )
        self._prepared_execution_id = None

    def rehydrate_terminal(self, execution_id: str) -> RehydratedCandidateExecution:
        """Verify and restore exact terminal artifact references without candidate execution."""

        identifier = _text(execution_id, "execution_id")
        record = self.store.execution(identifier)
        if record is None or record.status not in {"SUCCEEDED", "FAILED"}:
            raise CandidateJournalError("execution is absent or not terminal")
        try:
            action = CandidateAction(record.kind.removeprefix("generated_candidate_"))
        except ValueError as exc:
            raise CandidateJournalError(
                "terminal execution kind is not a candidate action"
            ) from exc
        entries: list[tuple[str, ArtifactRef]] = []
        for role, artifact in self.store.artifacts_for(owner_type="execution", owner_id=identifier):
            try:
                kind = ArtifactKind(artifact.kind)
            except ValueError as exc:
                raise CandidateJournalError("terminal artifact has an unknown kind") from exc
            reference = ArtifactRef(artifact.digest, artifact.size_bytes, kind)
            if artifact.relative_path != reference.object_relative_path.as_posix():
                raise CandidateJournalError("terminal artifact path differs from its digest path")
            self.artifact_store.verify(reference)
            entries.append((role, reference))
        if not entries:
            raise CandidateJournalError("terminal execution has no immutable artifacts")
        expected_roles = {"execution_manifest", "stderr", "stdout"}
        if record.status == "SUCCEEDED":
            expected_roles |= (
                {"candidate_result", "checkpoint"}
                if action is CandidateAction.TRAIN
                else {"prediction", "prediction_result"}
            )
        observed_roles = {role for role, _ in entries}
        optional_roles = observed_roles & {"failure_diagnostic", "workspace_cleanup"}
        if record.status == "SUCCEEDED" and "failure_diagnostic" in optional_roles:
            raise CandidateJournalError(
                "successful terminal execution retained a failure diagnostic"
            )
        expected_roles |= optional_roles
        if observed_roles != expected_roles:
            raise CandidateJournalError(
                "terminal artifacts differ from the frozen candidate action closure"
            )
        entry_map = dict(entries)
        try:
            result_reference = entry_map["execution_manifest"]
        except KeyError as exc:
            raise CandidateJournalError(
                "terminal execution is missing its trusted execution manifest"
            ) from exc
        payload = self.artifact_store.read_bytes(
            result_reference, max_bytes=result_reference.size_bytes
        )
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CandidateJournalError("terminal execution manifest is not JSON") from exc
        if not isinstance(decoded, dict) or payload != _canonical_json(decoded):
            raise CandidateJournalError("terminal execution manifest is not canonical JSON")
        manifest = decoded
        candidate_released = manifest.get("candidate_released")
        if (
            manifest.get("execution_id") != identifier
            or type(candidate_released) is not bool
            or record.result_digest != _result_manifest_digest(manifest)
        ):
            raise CandidateJournalError(
                "terminal execution manifest differs from its durable result identity"
            )
        if record.status == "SUCCEEDED" and manifest.get("outcome") != "succeeded":
            raise CandidateJournalError(
                "successful terminal execution has a non-success runner outcome"
            )

        if action is CandidateAction.TRAIN:
            if record.launch_id is None:
                raise CandidateJournalError("terminal training execution lost its launch identity")
            launch = next(
                (item for item in self.store.launches() if item.launch_id == record.launch_id),
                None,
            )
            if launch is None:
                raise CandidateJournalError("terminal training execution launch is missing")
            if candidate_released:
                if launch.state in {"RESERVED", "NOT_STARTED"}:
                    raise CandidateJournalError(
                        "released terminal candidate lacks a durable STARTED launch"
                    )
                desired: Literal["FINISHED", "NOT_STARTED"] = "FINISHED"
            elif launch.state in {"STARTED", "START_UNCERTAIN", "FINISHED"}:
                desired = "FINISHED"
            else:
                desired = "NOT_STARTED"
            if launch.state != desired:
                self.store.transition_launch(
                    launch.launch_id,
                    to_state=desired,
                    expected_revision=self.store.snapshot().revision,
                    metadata={"execution_id": identifier},
                )
        elif record.launch_id is not None:
            raise CandidateJournalError("terminal prediction unexpectedly owns a training launch")
        restored = CandidateExecutionArtifacts(
            tuple(entries),
            output_validated=record.status == "SUCCEEDED",
        )
        return RehydratedCandidateExecution(record, action, restored)


__all__ = [
    "CampaignStoreCandidateJournal",
    "CandidateAdmissionError",
    "CandidateExecutionPendingError",
    "CandidateExecutionTerminalError",
    "CandidateJournalError",
    "CandidateJournalPolicy",
    "RehydratedCandidateExecution",
    "reconstruct_budget_ledger",
]
