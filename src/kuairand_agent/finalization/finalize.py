"""Fail-closed finalization with replayable-ancestor fallback and closed publication."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from kuairand_agent.execution.artifacts import ArtifactStore
from kuairand_agent.finalization.organizer_check import (
    OrganizerCheckError,
    OrganizerCheckEvidence,
    check_final_submission,
)
from kuairand_agent.finalization.replay import (
    CleanReplayEvidence,
    CleanReplayRequest,
    MetricEvaluator,
    ReplayBackend,
    ReplayCancelledError,
    ReplayCapabilities,
    ReplayError,
    run_clean_replay,
)
from kuairand_agent.finalization.report import (
    FinalReportContext,
    FinalReportError,
    ReproduceInstructions,
    write_final_report,
    write_reproduce_script,
)
from kuairand_agent.finalization.submission_bundle import (
    FinalBundleCancelledError,
    FinalBundleError,
    FinalBundleMetadata,
    FinalBundleResult,
    FinalBundleSources,
    FinalStatus,
    create_final_bundle,
)

FINALIZATION_SCHEMA_VERSION: Final = 1


class FinalizationError(RuntimeError):
    """Raised when no candidate finalizes or the closed bundle cannot be published."""


class FinalizationCancelledError(FinalizationError):
    """Cooperative cancellation stopped finalization without publishing a partial bundle."""


def _check_cancellation(cancel_event: threading.Event | None, *, stage: str) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise FinalizationCancelledError(f"finalization cancelled before {stage}")


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise FinalizationError(f"{location} must be one non-empty line of text")
    return value


@dataclass(frozen=True, slots=True)
class FinalizationCandidate:
    """One eligible candidate, ordered from selected incumbent back to official FM."""

    candidate_id: str
    lineage: tuple[str, ...]
    status: FinalStatus
    replay_request: CleanReplayRequest
    backend: ReplayBackend
    bundle_metadata: FinalBundleMetadata
    report_context: FinalReportContext
    is_official_fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if not self.lineage:
            raise FinalizationError("candidate lineage must not be empty")
        lineage = tuple(_text(item, f"lineage[{index}]") for index, item in enumerate(self.lineage))
        if len(lineage) != len(set(lineage)) or lineage[-1] != self.candidate_id:
            raise FinalizationError("candidate lineage must be unique and end at candidate_id")
        object.__setattr__(self, "lineage", lineage)
        if not isinstance(self.status, FinalStatus):
            raise FinalizationError("candidate status must be FinalStatus")
        if not isinstance(self.replay_request, CleanReplayRequest):
            raise FinalizationError("replay_request must be CleanReplayRequest")
        if self.replay_request.candidate_id != self.candidate_id:
            raise FinalizationError("replay_request candidate differs from candidate_id")
        if not isinstance(self.bundle_metadata, FinalBundleMetadata):
            raise FinalizationError("bundle_metadata must be FinalBundleMetadata")
        if self.bundle_metadata.selected_experiment != self.candidate_id:
            raise FinalizationError("bundle metadata selected experiment differs from candidate")
        if self.bundle_metadata.lineage != self.lineage:
            raise FinalizationError("bundle metadata lineage differs from candidate")
        if self.bundle_metadata.status is not self.status:
            raise FinalizationError("bundle metadata status differs from candidate")
        if not isinstance(self.report_context, FinalReportContext):
            raise FinalizationError("report_context must be FinalReportContext")
        if type(self.is_official_fallback) is not bool:
            raise FinalizationError("is_official_fallback must be boolean")


@dataclass(frozen=True, slots=True)
class FinalizationRequest:
    """Shared trusted resources for one bounded finalization attempt chain."""

    destination: Path
    artifact_store: ArtifactStore
    capabilities: ReplayCapabilities
    protected_metric_evaluator: MetricEvaluator
    data_dir: Path
    starter_dir: Path
    experiments_jsonl: Path
    experiments_csv: Path
    candidates: tuple[FinalizationCandidate, ...]
    scratch_dir: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "destination",
            "data_dir",
            "starter_dir",
            "experiments_jsonl",
            "experiments_csv",
        ):
            if not isinstance(getattr(self, name), Path):
                raise FinalizationError(f"{name} must be a pathlib.Path")
        if self.scratch_dir is not None and not isinstance(self.scratch_dir, Path):
            raise FinalizationError("scratch_dir must be a pathlib.Path or None")
        if not isinstance(self.artifact_store, ArtifactStore):
            raise FinalizationError("artifact_store must be ArtifactStore")
        if not isinstance(self.capabilities, ReplayCapabilities):
            raise FinalizationError("capabilities must be ReplayCapabilities")
        if not callable(self.protected_metric_evaluator):
            raise FinalizationError("protected_metric_evaluator must be callable")
        if not self.candidates or any(
            not isinstance(candidate, FinalizationCandidate) for candidate in self.candidates
        ):
            raise FinalizationError("candidates must contain FinalizationCandidate entries")
        identifiers = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(identifiers) != len(set(identifiers)):
            raise FinalizationError("finalization candidates must be unique")
        fallbacks = tuple(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.is_official_fallback
        )
        if fallbacks != (len(self.candidates) - 1,):
            raise FinalizationError("the official fallback must exist exactly once and be last")


@dataclass(frozen=True, slots=True)
class FinalizationFailureEvidence:
    """Bounded diagnostic for one rejected replay/check candidate."""

    candidate_id: str
    stage: str
    exception_type: str
    diagnostic_sha256: str

    def manifest(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "diagnostic_sha256": self.diagnostic_sha256,
        }

    def report_line(self) -> str:
        return (
            f"Candidate {self.candidate_id} failed {self.stage} "
            f"({self.exception_type}; diagnostic SHA-256 {self.diagnostic_sha256}); "
            "the prior replayable incumbent remained eligible."
        )


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    """Immutable identity and fallback evidence for the published final bundle."""

    selected_candidate_id: str
    selected_status: FinalStatus
    fallback_count: int
    failures: tuple[FinalizationFailureEvidence, ...]
    replay: CleanReplayEvidence
    organizer_check: OrganizerCheckEvidence
    bundle: FinalBundleResult

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_status": self.selected_status.value,
            "fallback_count": self.fallback_count,
            "failures": [failure.manifest() for failure in self.failures],
            "replay": self.replay.manifest(),
            "organizer_check": self.organizer_check.manifest(),
            "bundle": {
                "root": str(self.bundle.root),
                "manifest_sha256": self.bundle.manifest_sha256,
                "submission_sha256": self.bundle.submission_sha256,
                "file_count": self.bundle.file_count,
                "total_size_bytes": self.bundle.total_size_bytes,
            },
        }


def _failure(candidate_id: str, stage: str, error: Exception) -> FinalizationFailureEvidence:
    rendered = f"{type(error).__name__}:{error}".encode("utf-8", errors="replace")
    return FinalizationFailureEvidence(
        candidate_id=candidate_id,
        stage=stage,
        exception_type=type(error).__name__,
        diagnostic_sha256=hashlib.sha256(rendered).hexdigest(),
    )


def _validate_candidate_closure(candidate: FinalizationCandidate) -> None:
    identity = candidate.replay_request.identity
    hashes = candidate.bundle_metadata.scientific_artifact_hashes
    expected = {
        "source": identity.source_sha256,
        "config": identity.config_sha256,
        "features": identity.features_sha256,
        "checkpoint": identity.checkpoint_sha256,
        "predictions": identity.validation_prediction_artifact_sha256,
    }
    if any(hashes.get(name) != digest for name, digest in expected.items()):
        raise FinalizationError("bundle scientific hashes differ from frozen replay identity")
    data_identity = candidate.bundle_metadata.data_identity
    if identity.data_sha256 not in data_identity.values():
        raise FinalizationError("bundle data identity omits the replay data digest")
    environment = candidate.bundle_metadata.environment_and_resource_usage
    if environment.get("environment_sha256") != identity.environment_sha256:
        raise FinalizationError("bundle environment identity differs from frozen replay")
    selected = candidate.report_context.selected
    metrics = candidate.bundle_metadata.validation_metrics
    for name, value in (
        ("GAUC", selected.gauc),
        ("nDCG@5", selected.ndcg_at_5),
        ("primary", selected.primary),
    ):
        observed = metrics.get(name)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise FinalizationError(f"bundle validation metric {name} is missing")
        if not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=0.0):
            raise FinalizationError(f"bundle validation metric {name} differs from report")


def _remove_work_root(path: Path) -> None:
    """Remove only the private temporary finalization tree, including read-only snapshots."""

    if not path.exists():
        return
    os.chmod(path, 0o700, follow_symlinks=False)
    for candidate in path.rglob("*"):
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            os.chmod(candidate, 0o700, follow_symlinks=False)
    shutil.rmtree(path)


def run_finalization(
    request: FinalizationRequest,
    *,
    cancel_event: threading.Event | None = None,
) -> FinalizationResult:
    """Publish the first candidate that passes the complete finalization closure.

    Replay alone does not select a candidate.  Closure validation, the untouched organizer
    check, judge-readable reporting, reproduction instructions, and closed-bundle construction
    are all candidate-specific eligibility gates.  Any one may walk the lineage back to the
    next replayable ancestor without weakening the immutable official fallback.
    """

    if not isinstance(request, FinalizationRequest):
        raise FinalizationError("request must be FinalizationRequest")
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise FinalizationError("cancel_event must be threading.Event or None")
    _check_cancellation(cancel_event, stage="workspace creation")
    destination = request.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FinalizationError(f"refusing to overwrite existing final bundle: {destination}")
    work_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.finalize-", dir=destination.parent)
    )
    failures: list[FinalizationFailureEvidence] = []
    try:
        for index, candidate in enumerate(request.candidates):
            _check_cancellation(cancel_event, stage=f"candidate {candidate.candidate_id} admission")
            try:
                _validate_candidate_closure(candidate)
            except FinalizationError as exc:
                failures.append(_failure(candidate.candidate_id, "candidate closure", exc))
                continue
            replay_request = replace(
                candidate.replay_request,
                output_dir=work_root / f"attempt-{index:02d}-{candidate.candidate_id}",
            )
            try:
                attempt = run_clean_replay(
                    replay_request,
                    artifact_store=request.artifact_store,
                    capabilities=request.capabilities,
                    backend=candidate.backend,
                    protected_metric_evaluator=request.protected_metric_evaluator,
                    cancel_event=cancel_event,
                )
            except ReplayCancelledError as exc:
                raise FinalizationCancelledError(
                    f"finalization cancelled during {candidate.candidate_id} clean replay"
                ) from exc
            except ReplayError as exc:
                failures.append(_failure(candidate.candidate_id, "clean replay", exc))
                continue
            try:
                _check_cancellation(cancel_event, stage="organizer checking")
                check = check_final_submission(
                    attempt.final_submission,
                    data_dir=request.data_dir,
                    starter_dir=request.starter_dir,
                    scratch_dir=request.scratch_dir,
                )
            except OrganizerCheckError as exc:
                failures.append(_failure(candidate.candidate_id, "organizer check", exc))
                continue
            support_root = work_root / f"support-{index:02d}-{candidate.candidate_id}"
            try:
                _check_cancellation(cancel_event, stage="final report construction")
                support_root.mkdir(mode=0o700)
                report_path = write_final_report(
                    support_root / "report.md",
                    candidate.report_context,
                    attempt.evidence,
                    check,
                    fallback_failures=tuple(failure.report_line() for failure in failures),
                )
                runtime_identity = candidate.bundle_metadata.environment_and_resource_usage.get(
                    "runtime_identity"
                )
                if not isinstance(runtime_identity, Mapping):
                    raise FinalReportError("bundle runtime_identity is missing")
                dependency_groups = runtime_identity.get("dependency_groups")
                if not isinstance(dependency_groups, list) or any(
                    type(group) is not str for group in dependency_groups
                ):
                    raise FinalReportError("bundle dependency_groups are malformed")
                reproduce_path = write_reproduce_script(
                    support_root / "reproduce.sh",
                    ReproduceInstructions(
                        expected_data_sha256=attempt.evidence.identity.data_sha256,
                        dependency_groups=tuple(dependency_groups),
                    ),
                )
            except (FinalReportError, OSError) as exc:
                failures.append(_failure(candidate.candidate_id, "final report", exc))
                continue
            sources = FinalBundleSources(
                submission=attempt.final_submission,
                report=report_path,
                experiments_jsonl=request.experiments_jsonl,
                experiments_csv=request.experiments_csv,
                environment=attempt.environment_path,
                reproduce=reproduce_path,
                config=attempt.config_dir,
                source=attempt.source_dir,
                model=attempt.model_dir,
                preprocessing=attempt.preprocessing_dir,
                validation_evidence=attempt.validation_evidence_dir,
                replay=attempt.replay_dir,
            )
            try:
                _check_cancellation(cancel_event, stage="closed bundle publication")
                bundle = create_final_bundle(
                    destination,
                    sources=sources,
                    metadata=candidate.bundle_metadata,
                    organizer_check=check,
                    cancel_event=cancel_event,
                )
            except FinalBundleCancelledError as exc:
                raise FinalizationCancelledError(
                    f"finalization cancelled during {candidate.candidate_id} bundle publication"
                ) from exc
            except FinalBundleError as exc:
                failures.append(_failure(candidate.candidate_id, "closed bundle", exc))
                continue
            return FinalizationResult(
                selected_candidate_id=candidate.candidate_id,
                selected_status=candidate.status,
                fallback_count=len(failures),
                failures=tuple(failures),
                replay=attempt.evidence,
                organizer_check=check,
                bundle=bundle,
            )
        raise FinalizationError(
            "all replayable ancestors, including the official FM fallback, failed finalization"
        )
    finally:
        try:
            _remove_work_root(work_root)
        except OSError as exc:
            raise FinalizationError("temporary finalization workspace cleanup failed") from exc


__all__ = [
    "FINALIZATION_SCHEMA_VERSION",
    "FinalizationCancelledError",
    "FinalizationCandidate",
    "FinalizationError",
    "FinalizationFailureEvidence",
    "FinalizationRequest",
    "FinalizationResult",
    "run_finalization",
]
