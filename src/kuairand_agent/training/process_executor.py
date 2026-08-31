"""Model-agnostic process adapter over the existing trusted :class:`Runner`.

This module deliberately contains no process-launch or signal logic.  It preserves the complete
lower-level runner result, adds the shared trainer resource shape, and maps only trusted runner
outcomes into the closed trainer failure taxonomy.  Candidate stdout and stderr are never parsed
to refine a failure category.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from kuairand_agent.execution.runner import (
    ExecutionOutcome,
    ExecutionResult,
    ExecutionSpec,
    ProcessRecord,
    ReconciliationExpectation,
    ReconciliationResult,
    Runner,
)
from kuairand_agent.training.protocol import ResourceReceipt, TrainerFailureCode

_FAILURE_CODES: dict[ExecutionOutcome, TrainerFailureCode | None] = {
    ExecutionOutcome.SUCCEEDED: None,
    ExecutionOutcome.TIMED_OUT: TrainerFailureCode.TIMEOUT,
    ExecutionOutcome.MEMORY_LIMIT: TrainerFailureCode.OOM,
    ExecutionOutcome.CANCELLED: TrainerFailureCode.CANCELLED,
    ExecutionOutcome.EXIT_NONZERO: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.DISK_LIMIT: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.PROCESS_LIMIT: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.ORPHANED_DESCENDANT: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.INSPECTION_FAILED: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.LAUNCH_COMMIT_FAILED: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.LAUNCHER_FAILED: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.SPAWN_FAILED: TrainerFailureCode.INTERNAL_ERROR,
    ExecutionOutcome.CLEANUP_FAILED: TrainerFailureCode.INTERNAL_ERROR,
}


@dataclass(frozen=True, slots=True)
class ProcessExecutionReceipt:
    """Complete supervisor evidence and its conservative trainer classification."""

    execution: ExecutionResult
    resources: ResourceReceipt
    failure_code: TrainerFailureCode | None

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ExecutionResult):
            raise TypeError("execution must be ExecutionResult")
        if not isinstance(self.resources, ResourceReceipt):
            raise TypeError("resources must be ResourceReceipt")
        expected = _FAILURE_CODES[self.execution.outcome]
        if self.failure_code is not expected:
            raise ValueError("failure_code differs from the trusted runner outcome")

    @property
    def succeeded(self) -> bool:
        return self.failure_code is None and self.execution.succeeded


class ProcessExecutor:
    """Thin composition over ``Runner``; it is intentionally not another supervisor."""

    def __init__(self, runner: Runner | None = None) -> None:
        self._runner = Runner() if runner is None else runner
        if not isinstance(self._runner, Runner):
            raise TypeError("runner must be Runner")

    @property
    def runner(self) -> Runner:
        return self._runner

    def execute(
        self,
        spec: ExecutionSpec,
        *,
        commit_launch: Callable[[ProcessRecord], None],
        cancel_event: threading.Event | None = None,
    ) -> ProcessExecutionReceipt:
        result = self._runner.run(
            spec,
            commit_launch=commit_launch,
            cancel_event=cancel_event,
        )
        return ProcessExecutionReceipt(
            execution=result,
            resources=ResourceReceipt(
                wall_seconds=result.wall_seconds,
                # Runner currently does not expose process-tree CPU time.  Zero is retained with
                # an explicit measurement bit instead of pretending parent CPU is child CPU.
                cpu_seconds=0.0,
                peak_rss_bytes=result.peak_tree_rss_bytes,
                peak_disk_bytes=result.peak_workspace_bytes,
                peak_process_count=max(result.peak_process_count, 1),
                threads=result.threads,
                device=result.device,
                cpu_seconds_measured=False,
            ),
            failure_code=_FAILURE_CODES[result.outcome],
        )

    def reconcile(
        self,
        record: ProcessRecord,
        *,
        expected: ReconciliationExpectation,
        termination_grace_seconds: float = 0.5,
    ) -> ReconciliationResult:
        return self._runner.reconcile(
            record,
            expected=expected,
            termination_grace_seconds=termination_grace_seconds,
        )


__all__ = ["ProcessExecutionReceipt", "ProcessExecutor"]
