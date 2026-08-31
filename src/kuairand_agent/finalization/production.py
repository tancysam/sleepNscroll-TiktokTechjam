"""Production composition for provider-free campaign finalization and closed replay.

This module is the deep adapter between the durable research outcome and the lower-level replay,
organizer-check, reporting, and bundle modules.  Its public interface is intentionally small:

``finalize_provider_free_campaign``
    Strictly reopen one finalization-ready campaign, run the reserved deterministic training
    replay when a generated winner exists, walk back to the immutable official-FM fallback on
    any generated-branch failure, publish a closed bundle, and complete durable campaign state.

``replay_final_bundle``
    Verify every member of a closed bundle, rebuild only label-free capabilities from a verified
    KuaiRand-Pure directory, replay the allowlisted frozen backend without a provider or network,
    rerun the untouched organizer checker, and prove the final submission bytes are identical.

Final-period outcomes are never represented by this module.  The canonical loader skips them,
the candidate backend receives only :class:`CandidateInputs`, and the organizer checker receives
only its private outcome-masked view.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import shutil
import stat
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.campaign.budgets import LaunchCategory, WorkPhase
from kuairand_agent.campaign.candidate_journal import (
    CampaignStoreCandidateJournal,
    CandidateExecutionPendingError,
    CandidateExecutionTerminalError,
    CandidateJournalPolicy,
)
from kuairand_agent.campaign.controller import (
    CAMPAIGN_DATABASE_NAME,
    CampaignCreateRequest,
    CampaignEngine,
)
from kuairand_agent.campaign.full_campaign import (
    FinalizationSelectionPlan,
    FullCampaignOutcome,
    FullCampaignOutcomeRepository,
    FullCampaignProgressLedger,
    FullCampaignStage,
    QualifiedFMMemberPlan,
    build_finalization_candidate_inputs,
)
from kuairand_agent.campaign.generated_scientific_runner import (
    FileScientificRunEvidenceRepository,
)
from kuairand_agent.campaign.models import CampaignState
from kuairand_agent.campaign.provenance import capture_environment_identity, hash_source_tree
from kuairand_agent.campaign.qualification_evidence import (
    OfficialFMFallbackEvidence,
    OfficialFMQualificationEvidence,
    OfficialFMSeedEvidence,
    QualificationExpectations,
    load_official_fm_qualification,
)
from kuairand_agent.campaign.selector import OrganizerMetrics
from kuairand_agent.campaign.store import ArtifactSpec, CampaignStore
from kuairand_agent.candidates.bootstrap import paired_user_cluster_bootstrap
from kuairand_agent.contract import (
    BENCHMARK_CONTRACT,
    STARTER_FILE_SHA256,
    sha256_file,
    verify_starter_kit,
)
from kuairand_agent.data.canonical import (
    CanonicalDataset,
    ProtectedTargets,
    load_canonical_dataset,
)
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
)
from kuairand_agent.execution.candidate_executor import (
    GeneratedCandidateExecutor,
    GeneratedCandidateIdentity,
    GeneratedTrainRequest,
    LocalCandidateLimits,
)
from kuairand_agent.execution.policy import SplitRole
from kuairand_agent.execution.runner import active_python_interpreter
from kuairand_agent.execution.workspace import WorkspaceMaterializer
from kuairand_agent.finalization.backends import build_replay_backend, load_replay_backend
from kuairand_agent.finalization.finalize import (
    FinalizationCandidate,
    FinalizationRequest,
    FinalizationResult,
    run_finalization,
)
from kuairand_agent.finalization.iteration_evidence import (
    IterationEvidence,
    collect_iteration_narratives,
    count_recorded_iterations,
)
from kuairand_agent.finalization.organizer_check import (
    OrganizerCheckEvidence,
    check_final_submission,
)
from kuairand_agent.finalization.recipe import (
    GeneratedLambdaRankReplayRecipe,
    OfficialFMMemberRecipe,
    OfficialFMReplayRecipe,
    ReplayRecipe,
    load_replay_recipe,
    write_replay_recipe,
)
from kuairand_agent.finalization.replay import (
    CleanReplayEvidence,
    CleanReplayRequest,
    FrozenReplayIdentity,
    ReplayArtifacts,
    ReplayCapabilities,
    ReplayEquality,
    environment_identity_digest,
    run_clean_replay,
)
from kuairand_agent.finalization.replay import (
    _remove_private_tree as _remove_replay_private_tree,
)
from kuairand_agent.finalization.report import (
    ExperimentNarrative,
    FinalReportContext,
    MetricEvidence,
    ResourceEvidence,
)
from kuairand_agent.finalization.submission_bundle import (
    FINAL_BUNDLE_SCHEMA_VERSION,
    REQUIRED_DIRECTORY_PATHS,
    REQUIRED_FILE_PATHS,
    FinalBundleMetadata,
    FinalStatus,
)
from kuairand_agent.scoring.protected import (
    Alignment,
    ProtectedScorer,
    ScoreResult,
    SplitIdentity,
)
from kuairand_agent.scoring.submission import AlignmentRow, prediction_digest

PRODUCTION_FINALIZATION_SCHEMA_VERSION: Final = 2
PRODUCTION_BUNDLE_REPLAY_SCHEMA_VERSION: Final = 2
_MAX_MANIFEST_BYTES: Final = 16 * 1024 * 1024
_MAX_LEDGER_BYTES: Final = 64 * 1024 * 1024
_MAX_PREDICTION_BYTES: Final = 2 * 1024 * 1024 * 1024
_OUTCOME_RELATIVE_PATH: Final = Path("production/finalization/outcome.json")
_FINAL_BUNDLE_RELATIVE_PATH: Final = Path("final")
_FINAL_TRAINING_EXECUTION_PREFIX: Final = "final-train-"
_DIGEST_RE: Final = frozenset("0123456789abcdef")
_MATERIAL_PRIMARY_DELTA: Final = Decimal("0.002")
_WORST_INNER_DELTA: Final = Decimal("-0.002")
_FINALIZATION_RESOURCE_SCHEMA_VERSION: Final = 1
_FINALIZATION_COVERAGE: Final = (
    "canonical_context_reopen",
    "generated_final_training_if_selected",
    "validation_replay",
    "final_inference",
    "clean_replay_evidence",
    "untouched_organizer_check",
    "judge_report_and_reproduce_script",
    "bundle_construction_fsync_publication_and_closure",
)

type Float64Vector = npt.NDArray[np.float64]


class ProductionFinalizationError(RuntimeError):
    """A production finalization or closed-bundle replay gate failed closed."""


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProductionFinalizationError("evidence must be finite canonical JSON") from exc
    return payload + (b"\n" if newline else b"")


def _digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _DIGEST_RE for character in value)
    ):
        raise ProductionFinalizationError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ProductionFinalizationError(f"{location} must be one non-empty line of text")
    return value


def _bounded_int(value: object, location: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ProductionFinalizationError(f"{location} is outside its supported bound")
    return value


def _metric(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ProductionFinalizationError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProductionFinalizationError(f"{location} must be finite in [0, 1]")
    return result


def _cancel_event(value: threading.Event | None) -> threading.Event | None:
    if value is not None and not isinstance(value, threading.Event):
        raise ProductionFinalizationError("cancel_event must be threading.Event or None")
    return value


def _raise_if_cancelled(value: threading.Event | None, stage: str) -> None:
    if value is not None and value.is_set():
        raise ProductionFinalizationError(f"production operation cancelled before {stage}")


class _FinalizationDeadlineEvent(threading.Event):
    """Read-only composition of user cancellation and the immutable campaign deadline."""

    def __init__(
        self,
        *,
        signal_event: threading.Event | None,
        engine: CampaignEngine,
        run_dir: Path,
    ) -> None:
        super().__init__()
        self._signal_event = signal_event
        self._engine = engine
        self._run_dir = run_dir

    def deadline_expired(self) -> bool:
        return self._engine.inspect_deadline(self._run_dir).hard_expired

    def is_set(self) -> bool:
        return bool(
            super().is_set()
            or (self._signal_event is not None and self._signal_event.is_set())
            or self.deadline_expired()
        )


def _rss_bytes(value: int) -> int:
    observed = value * 1024 if sys.platform.startswith("linux") else value
    return max(observed, 0)


def _trusted_process_resources() -> tuple[float, int]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = float(own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime)
    # The sum of the independent high-water marks is a conservative upper bound for the
    # single-child finalization topology, even when their peaks occurred at different times.
    peak_rss_upper_bound = _rss_bytes(int(own.ru_maxrss)) + _rss_bytes(int(children.ru_maxrss))
    return cpu_seconds, max(peak_rss_upper_bound, 1)


@dataclass(slots=True)
class _FinalizationResourceTracker:
    started_campaign_elapsed_seconds: float
    started_monotonic_ns: int = field(init=False)
    started_cpu_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.started_campaign_elapsed_seconds, bool)
            or not isinstance(self.started_campaign_elapsed_seconds, (int, float))
            or not math.isfinite(float(self.started_campaign_elapsed_seconds))
            or float(self.started_campaign_elapsed_seconds) < 0.0
        ):
            raise ProductionFinalizationError("finalization start elapsed time is invalid")
        self.started_campaign_elapsed_seconds = float(self.started_campaign_elapsed_seconds)
        self.started_monotonic_ns = time.monotonic_ns()
        self.started_cpu_seconds = _trusted_process_resources()[0]

    def finish(
        self,
        *,
        campaign_elapsed_seconds: float,
        hard_wall_seconds: int,
        finalization_reserve_seconds: int,
        rows: int,
        bundle_manifest_sha256: str,
        training_replay: Mapping[str, object],
        selected_generated: bool,
    ) -> Mapping[str, object]:
        elapsed = float(campaign_elapsed_seconds)
        finalization_elapsed = max(0.0, elapsed - self.started_campaign_elapsed_seconds)
        cpu_now, peak_rss_upper_bound = _trusted_process_resources()
        cpu_seconds = max(0.0, cpu_now - self.started_cpu_seconds)
        monotonic_wall = max(0.0, (time.monotonic_ns() - self.started_monotonic_ns) / 1e9)
        # The durable campaign clock is conservative across UTC/monotonic observations and is
        # therefore the authoritative bounded aggregate.  The local monotonic sample is retained
        # as a diagnostic and cannot reduce the charged elapsed duration.
        aggregate: dict[str, object] = {
            "family": "production_finalization_total",
            "wall_seconds": finalization_elapsed,
            "local_monotonic_wall_seconds": monotonic_wall,
            "cpu_seconds": cpu_seconds,
            "peak_rss_bytes": peak_rss_upper_bound,
            "rows": rows,
            "evidence_digest": _digest(bundle_manifest_sha256, "bundle manifest"),
            "rss_accounting": (
                "conservative_sum_of_process_lifetime_self_and_child_high_water_marks"
            ),
        }
        evidence: dict[str, object] = {
            "schema_version": _FINALIZATION_RESOURCE_SCHEMA_VERSION,
            "clock_basis": "durable_max_of_monotonic_and_utc_elapsed",
            "campaign_elapsed_seconds": elapsed,
            "hard_wall_seconds": hard_wall_seconds,
            "finalization_reserve_seconds": finalization_reserve_seconds,
            "finalization_started_elapsed_seconds": self.started_campaign_elapsed_seconds,
            "finalization_elapsed_seconds": finalization_elapsed,
            "coverage": list(_FINALIZATION_COVERAGE),
            "within_reserve": finalization_elapsed <= float(finalization_reserve_seconds),
            "within_hard_wall": elapsed <= float(hard_wall_seconds),
            "aggregate": aggregate,
        }
        resources = training_replay.get("resources")
        if selected_generated:
            if not isinstance(resources, Mapping):
                raise ProductionFinalizationError(
                    "generated finalization lacks retained final-training resources"
                )
            evidence["final_training"] = {
                "family": "generated_final_training_replay",
                "wall_seconds": resources.get("wall_seconds"),
                "peak_rss_bytes": resources.get("peak_rss_bytes"),
                "disk_bytes": resources.get("disk_bytes"),
                "device": training_replay.get("device"),
                "evidence_digest": training_replay.get("checkpoint_sha256"),
            }
        return MappingProxyType(evidence)


def _normalize_resource_evidence(
    value: Mapping[str, object],
    *,
    bundle_manifest_sha256: str,
    selected_status: FinalStatus,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProductionFinalizationError("resource_evidence must be a mapping")
    normalized = json.loads(_canonical_json(dict(value)))
    required = {
        "schema_version",
        "clock_basis",
        "campaign_elapsed_seconds",
        "hard_wall_seconds",
        "finalization_reserve_seconds",
        "finalization_started_elapsed_seconds",
        "finalization_elapsed_seconds",
        "coverage",
        "within_reserve",
        "within_hard_wall",
        "aggregate",
    }
    has_training = "final_training" in normalized
    if (
        not isinstance(normalized, dict)
        or set(normalized) != required | ({"final_training"} if has_training else set())
        or normalized["schema_version"] != _FINALIZATION_RESOURCE_SCHEMA_VERSION
        or normalized["clock_basis"] != "durable_max_of_monotonic_and_utc_elapsed"
        or normalized["coverage"] != list(_FINALIZATION_COVERAGE)
    ):
        raise ProductionFinalizationError("resource_evidence schema or coverage is not exact")
    hard = normalized["hard_wall_seconds"]
    reserve = normalized["finalization_reserve_seconds"]
    if type(hard) is not int or not 60 <= hard <= 21_600:
        raise ProductionFinalizationError("production resource hard wall is invalid")
    if type(reserve) is not int or not 600 <= reserve < hard:
        raise ProductionFinalizationError("production resource finalization reserve is invalid")

    def seconds(name: str) -> float:
        raw = normalized[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProductionFinalizationError(f"resource_evidence {name} must be numeric")
        result = float(raw)
        if not math.isfinite(result) or result < 0.0:
            raise ProductionFinalizationError(f"resource_evidence {name} is invalid")
        return result

    campaign_elapsed = seconds("campaign_elapsed_seconds")
    started_elapsed = seconds("finalization_started_elapsed_seconds")
    finalization_elapsed = seconds("finalization_elapsed_seconds")
    if not math.isclose(
        finalization_elapsed,
        campaign_elapsed - started_elapsed,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ProductionFinalizationError("resource_evidence elapsed arithmetic is inconsistent")
    if (
        normalized["within_reserve"] is not (finalization_elapsed <= float(reserve))
        or normalized["within_hard_wall"] is not (campaign_elapsed <= float(hard))
        or not normalized["within_reserve"]
        or not normalized["within_hard_wall"]
    ):
        raise ProductionFinalizationError("production finalization exceeded its immutable limits")
    aggregate = normalized["aggregate"]
    aggregate_keys = {
        "family",
        "wall_seconds",
        "local_monotonic_wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "rows",
        "evidence_digest",
        "rss_accounting",
    }
    if (
        not isinstance(aggregate, dict)
        or set(aggregate) != aggregate_keys
        or aggregate["family"] != "production_finalization_total"
        or aggregate["evidence_digest"] != bundle_manifest_sha256
        or aggregate["rss_accounting"]
        != "conservative_sum_of_process_lifetime_self_and_child_high_water_marks"
        or type(aggregate["peak_rss_bytes"]) is not int
        or aggregate["peak_rss_bytes"] <= 0
        or type(aggregate["rows"]) is not int
        or aggregate["rows"] <= 0
    ):
        raise ProductionFinalizationError("aggregate finalization resource receipt is malformed")
    for name in ("wall_seconds", "local_monotonic_wall_seconds", "cpu_seconds"):
        raw = aggregate[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ProductionFinalizationError(f"aggregate receipt {name} must be numeric")
        rendered = float(raw)
        if not math.isfinite(rendered) or rendered < 0.0:
            raise ProductionFinalizationError(f"aggregate receipt {name} is invalid")
    if not math.isclose(
        float(aggregate["wall_seconds"]), finalization_elapsed, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ProductionFinalizationError("aggregate wall time differs from durable elapsed time")
    generated = selected_status is not FinalStatus.BASELINE_REPRODUCED
    if generated != has_training:
        raise ProductionFinalizationError(
            "final-training resource receipt presence differs from selected lineage"
        )
    if has_training:
        training = normalized["final_training"]
        if (
            not isinstance(training, dict)
            or set(training)
            != {
                "family",
                "wall_seconds",
                "peak_rss_bytes",
                "disk_bytes",
                "device",
                "evidence_digest",
            }
            or training["family"] != "generated_final_training_replay"
            or type(training["peak_rss_bytes"]) is not int
            or training["peak_rss_bytes"] < 0
            or type(training["disk_bytes"]) is not int
            or training["disk_bytes"] < 0
            or training["device"] != "cpu"
        ):
            raise ProductionFinalizationError("final-training resource receipt is malformed")
        wall = training["wall_seconds"]
        if (
            isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) < 0.0
        ):
            raise ProductionFinalizationError("final-training wall time is invalid")
        _digest(training["evidence_digest"], "final-training evidence digest")
    return MappingProxyType(normalized)


def _manifest_digest(domain: bytes, body: Mapping[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json(dict(body))).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    committed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        _fsync_directory(path.parent)
        committed = True
    except FileExistsError:
        raise
    finally:
        os.close(descriptor)
        if not committed:
            path.unlink(missing_ok=True)


def _read_regular(path: Path, *, maximum: int, location: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionFinalizationError(f"{location} is not safely readable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ProductionFinalizationError(
                f"{location} must be one bounded regular non-symlink file"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(payload) != before.st_size or not stable:
            raise ProductionFinalizationError(f"{location} changed while being read")
        return payload
    finally:
        os.close(descriptor)


def _strict_json(path: Path, *, maximum: int, location: str) -> dict[str, object]:
    payload = _read_regular(path, maximum=maximum, location=location)
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProductionFinalizationError(f"{location} is not strict ASCII JSON") from exc
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ProductionFinalizationError(f"{location} must be a JSON object")
    normalized = payload[:-1] if payload.endswith(b"\n") else payload
    if _canonical_json(value) != normalized:
        raise ProductionFinalizationError(f"{location} is not canonical JSON")
    return cast(dict[str, object], value)


def _safe_relative(value: object, location: str) -> str:
    text = _text(value, location)
    if "\\" in text:
        raise ProductionFinalizationError(f"{location} must be a canonical POSIX path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise ProductionFinalizationError(f"{location} is unsafe")
    return text


@dataclass(frozen=True, slots=True)
class ProductionFailureEvidence:
    """Bounded, non-secret diagnostic for a rejected production branch."""

    candidate_id: str
    stage: str
    exception_type: str
    diagnostic_sha256: str

    def __post_init__(self) -> None:
        for name in ("candidate_id", "stage", "exception_type"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _digest(self.diagnostic_sha256, "diagnostic_sha256")

    def manifest(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "diagnostic_sha256": self.diagnostic_sha256,
        }


def _failure(candidate_id: str, stage: str, error: BaseException) -> ProductionFailureEvidence:
    diagnostic = f"{type(error).__name__}:{error}".encode("utf-8", errors="replace")
    return ProductionFailureEvidence(
        candidate_id=candidate_id,
        stage=stage,
        exception_type=type(error).__name__,
        diagnostic_sha256=hashlib.sha256(diagnostic).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ProductionFinalizationOutcome:
    """Restart-safe terminal receipt for a published and store-completed final bundle."""

    run_dir: Path
    campaign_id: str
    research_outcome_digest: str
    selected_candidate_id: str
    selected_status: FinalStatus
    fallback_count: int
    failures: tuple[ProductionFailureEvidence, ...]
    training_replay: Mapping[str, object]
    resource_evidence: Mapping[str, object]
    bundle_root: Path
    bundle_manifest_sha256: str
    submission_sha256: str
    replay_evidence_sha256: str
    organizer_verification_sha256: str
    campaign_revision: int
    campaign_status: str = CampaignState.COMPLETED.value
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("run_dir", "bundle_root"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ProductionFinalizationError(f"{name} must be an absolute Path")
        for name in ("campaign_id", "selected_candidate_id", "campaign_status"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.selected_status, FinalStatus):
            raise ProductionFinalizationError("selected_status must be FinalStatus")
        for name in (
            "research_outcome_digest",
            "bundle_manifest_sha256",
            "submission_sha256",
            "replay_evidence_sha256",
            "organizer_verification_sha256",
        ):
            _digest(getattr(self, name), name)
        _bounded_int(self.fallback_count, "fallback_count", maximum=50)
        if self.fallback_count != len(self.failures):
            raise ProductionFinalizationError("fallback_count differs from retained failures")
        if any(not isinstance(item, ProductionFailureEvidence) for item in self.failures):
            raise ProductionFinalizationError("failures contain unsupported evidence")
        if not isinstance(self.training_replay, Mapping) or not self.training_replay:
            raise ProductionFinalizationError("training_replay must be a non-empty mapping")
        normalized = json.loads(_canonical_json(dict(self.training_replay)))
        if not isinstance(normalized, dict):  # pragma: no cover - mapping encodes to an object.
            raise ProductionFinalizationError("training_replay must encode to an object")
        object.__setattr__(self, "training_replay", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "resource_evidence",
            _normalize_resource_evidence(
                self.resource_evidence,
                bundle_manifest_sha256=self.bundle_manifest_sha256,
                selected_status=self.selected_status,
            ),
        )
        _bounded_int(self.campaign_revision, "campaign_revision", maximum=2**63 - 1)
        if self.campaign_status != CampaignState.COMPLETED.value:
            raise ProductionFinalizationError("production outcome must be campaign-complete")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-production-finalization-v1\0", self.body()),
        )

    def body(self) -> dict[str, object]:
        return {
            "schema_version": PRODUCTION_FINALIZATION_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "campaign_id": self.campaign_id,
            "research_outcome_digest": self.research_outcome_digest,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_status": self.selected_status.value,
            "fallback_count": self.fallback_count,
            "failures": [item.manifest() for item in self.failures],
            "training_replay": dict(self.training_replay),
            "resource_evidence": dict(self.resource_evidence),
            "bundle": {
                "root": str(self.bundle_root),
                "manifest_sha256": self.bundle_manifest_sha256,
                "submission_sha256": self.submission_sha256,
                "replay_evidence_sha256": self.replay_evidence_sha256,
                "organizer_verification_sha256": self.organizer_verification_sha256,
            },
            "campaign_revision": self.campaign_revision,
            "campaign_status": self.campaign_status,
        }

    def manifest(self) -> dict[str, object]:
        return self.body() | {"digest": self.digest}

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.manifest())

    @classmethod
    def from_bytes(cls, payload: bytes) -> ProductionFinalizationOutcome:
        try:
            raw = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProductionFinalizationError("production outcome is not strict JSON") from exc
        expected = {
            "schema_version",
            "run_dir",
            "campaign_id",
            "research_outcome_digest",
            "selected_candidate_id",
            "selected_status",
            "fallback_count",
            "failures",
            "training_replay",
            "resource_evidence",
            "bundle",
            "campaign_revision",
            "campaign_status",
            "digest",
        }
        if not isinstance(raw, dict) or set(raw) != expected or _canonical_json(raw) != payload:
            raise ProductionFinalizationError("production outcome does not use the exact schema")
        if raw["schema_version"] != PRODUCTION_FINALIZATION_SCHEMA_VERSION:
            raise ProductionFinalizationError("production outcome schema is unsupported")
        failures_raw = raw["failures"]
        bundle_raw = raw["bundle"]
        training_raw = raw["training_replay"]
        resource_raw = raw["resource_evidence"]
        if (
            not isinstance(failures_raw, list)
            or not isinstance(bundle_raw, dict)
            or set(bundle_raw)
            != {
                "root",
                "manifest_sha256",
                "submission_sha256",
                "replay_evidence_sha256",
                "organizer_verification_sha256",
            }
            or not isinstance(training_raw, dict)
            or not isinstance(resource_raw, dict)
        ):
            raise ProductionFinalizationError("production outcome nested evidence is malformed")
        failures: list[ProductionFailureEvidence] = []
        for index, item in enumerate(failures_raw):
            if not isinstance(item, dict) or set(item) != {
                "candidate_id",
                "stage",
                "exception_type",
                "diagnostic_sha256",
            }:
                raise ProductionFinalizationError(f"production failure {index} is malformed")
            failures.append(
                ProductionFailureEvidence(
                    candidate_id=item["candidate_id"],
                    stage=item["stage"],
                    exception_type=item["exception_type"],
                    diagnostic_sha256=item["diagnostic_sha256"],
                )
            )
        try:
            selected_status = FinalStatus(raw["selected_status"])
        except (TypeError, ValueError) as exc:
            raise ProductionFinalizationError("production selected status is unsupported") from exc
        outcome = cls(
            run_dir=Path(raw["run_dir"]),
            campaign_id=raw["campaign_id"],
            research_outcome_digest=raw["research_outcome_digest"],
            selected_candidate_id=raw["selected_candidate_id"],
            selected_status=selected_status,
            fallback_count=raw["fallback_count"],
            failures=tuple(failures),
            training_replay=training_raw,
            resource_evidence=resource_raw,
            bundle_root=Path(bundle_raw["root"]),
            bundle_manifest_sha256=bundle_raw["manifest_sha256"],
            submission_sha256=bundle_raw["submission_sha256"],
            replay_evidence_sha256=bundle_raw["replay_evidence_sha256"],
            organizer_verification_sha256=bundle_raw["organizer_verification_sha256"],
            campaign_revision=raw["campaign_revision"],
            campaign_status=raw["campaign_status"],
        )
        if outcome.digest != raw["digest"]:
            raise ProductionFinalizationError("production outcome digest mismatch")
        return outcome


@dataclass(frozen=True, slots=True)
class FinalBundleReplayOutcome:
    """Provider-free proof that a closed bundle reproduces its exact final submission."""

    candidate_id: str
    bundle_root: Path
    bundle_manifest_sha256: str
    expected_data_sha256: str
    replay: CleanReplayEvidence
    organizer_check: OrganizerCheckEvidence
    bundled_submission_sha256: str
    reproduced_submission_sha256: str
    project_root: Path
    project_source_digest: str
    environment_digest: str
    uv_lock_sha256: str
    dependency_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        for name in ("bundle_root", "project_root"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ProductionFinalizationError(f"{name} must be an absolute Path")
        for name in (
            "bundle_manifest_sha256",
            "expected_data_sha256",
            "bundled_submission_sha256",
            "reproduced_submission_sha256",
            "project_source_digest",
            "environment_digest",
            "uv_lock_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.dependency_groups not in {
            ("research-tree",),
            ("research-tree", "research-neural"),
        }:
            raise ProductionFinalizationError("dependency_groups are unsupported")
        if not isinstance(self.replay, CleanReplayEvidence):
            raise ProductionFinalizationError("replay must be CleanReplayEvidence")
        if not isinstance(self.organizer_check, OrganizerCheckEvidence):
            raise ProductionFinalizationError("organizer_check must be OrganizerCheckEvidence")
        if self.bundled_submission_sha256 != self.reproduced_submission_sha256:
            raise ProductionFinalizationError("closed replay changed final submission bytes")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": PRODUCTION_BUNDLE_REPLAY_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "bundle_root": str(self.bundle_root),
            "project_root": str(self.project_root),
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "expected_data_sha256": self.expected_data_sha256,
            "replay": self.replay.manifest(),
            "organizer_check": self.organizer_check.manifest(),
            "bundled_submission_sha256": self.bundled_submission_sha256,
            "reproduced_submission_sha256": self.reproduced_submission_sha256,
            "runtime_identity": {
                "project_source_digest": self.project_source_digest,
                "environment_digest": self.environment_digest,
                "uv_lock_sha256": self.uv_lock_sha256,
                "dependency_groups": list(self.dependency_groups),
                "project_source_identity_reproved": True,
                "environment_identity_reproved": True,
                "uv_lock_identity_reproved": True,
            },
            "submission_bytes_identical": True,
            "provider_used": False,
            "network_used": False,
            "final_outcomes_accessed": False,
            "final_outcomes_scored": False,
        }


@dataclass(frozen=True, slots=True)
class _VerifiedBundle:
    root: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    files: Mapping[str, Mapping[str, object]]


# The organizer evaluator is float32-sensitive: NumPy 2.x preserves float32 through its nDCG
# arithmetic, so the primary it returns is the float32 mean of GAUC and nDCG@5, while any float64
# recomputation of those same two numbers lands one float32 ulp away.  This tolerance exists to
# absorb exactly that, and it is why a derived primary may never be compared for exact equality
# against a scorer-reported one.  See scoring/protected.py score_with_encoded_labels.
_ORGANIZER_PRIMARY_DTYPE_TOLERANCE: Final = 2e-7


def _manifest_metrics(value: object, location: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"GAUC", "nDCG@5", "primary"}:
        raise ProductionFinalizationError(f"{location} organizer metrics are incomplete")
    result = {
        name: _metric(value[name], f"{location} {name}") for name in ("GAUC", "nDCG@5", "primary")
    }
    if not math.isclose(
        result["primary"],
        (result["GAUC"] + result["nDCG@5"]) / 2.0,
        rel_tol=0.0,
        abs_tol=_ORGANIZER_PRIMARY_DTYPE_TOLERANCE,
    ):
        raise ProductionFinalizationError(f"{location} primary is not the organizer mean")
    return result


def _require_close(left: float, right: float, location: str, *, abs_tol: float = 1e-15) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=abs_tol):
        raise ProductionFinalizationError(f"{location} differs from reconstructed evidence")


def _require_close_organizer_primary(left: float, right: float, location: str) -> None:
    """Compare two organizer primaries that were not necessarily derived the same way.

    Use this only where one side originates from the scorer and the other is recomputed, and only
    for ``primary``.  ``GAUC`` and ``nDCG@5`` are carried through unchanged on both sides and must
    still match exactly, which is what proves the difference is confined to the derived mean.
    """

    if not math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=_ORGANIZER_PRIMARY_DTYPE_TOLERANCE,
    ):
        raise ProductionFinalizationError(f"{location} differs from reconstructed evidence")


def _derive_bundle_status(manifest: Mapping[str, object]) -> FinalStatus:
    selection = manifest.get("selection")
    validation = manifest.get("validation")
    if not isinstance(selection, dict) or not isinstance(validation, dict):
        raise ProductionFinalizationError("bundle selection confirmation is missing")
    try:
        declared = FinalStatus(selection["status"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionFinalizationError("bundle declares an unsupported final status") from exc
    metrics = _manifest_metrics(validation.get("metrics"), "bundle representative")
    summary = validation.get("seed_summary")
    inner = validation.get("inner_fold_results")
    if not isinstance(summary, dict) or not isinstance(inner, list):
        raise ProductionFinalizationError("bundle confirmation structure is malformed")
    if declared is FinalStatus.BASELINE_REPRODUCED:
        if set(summary) != {
            "schema_version",
            "seeds",
            "representative_seed",
            "matched_confirmation",
            "derived_status",
            "confirmation_is_controller_derived",
        } or summary != {
            "schema_version": 1,
            "seeds": [4],
            "representative_seed": 4,
            "matched_confirmation": False,
            "derived_status": FinalStatus.BASELINE_REPRODUCED.value,
            "confirmation_is_controller_derived": True,
        }:
            raise ProductionFinalizationError(
                "baseline bundle confirmation is forged or incomplete"
            )
        if len(inner) != 1:
            raise ProductionFinalizationError("baseline bundle fold disclosure is incomplete")
        return declared

    expected_summary_fields = {
        "schema_version",
        "seeds",
        "representative_seed",
        "candidate_mean",
        "official_fm_mean",
        "per_seed",
        "primary_delta_summary",
        "inner_delta_summary",
        "paired_user_cluster_bootstrap",
        "derived_status",
        "status_thresholds",
        "confirmation_is_controller_derived",
    }
    if (
        set(summary) != expected_summary_fields
        or summary.get("schema_version") != 1
        or summary.get("seeds") != [0, 1, 2]
        or summary.get("confirmation_is_controller_derived") is not True
        or summary.get("status_thresholds")
        != {
            "material_primary_delta_strictly_greater_than": 0.002,
            "mean_inner_primary_delta_strictly_greater_than": 0.0,
            "worst_inner_primary_delta_minimum": -0.002,
        }
    ):
        raise ProductionFinalizationError("generated bundle confirmation is incomplete")
    representative_seed = summary.get("representative_seed")
    if type(representative_seed) is not int or representative_seed not in {0, 1, 2}:
        raise ProductionFinalizationError("bundle representative seed is invalid")
    records = summary.get("per_seed")
    if not isinstance(records, list) or len(records) != 3:
        raise ProductionFinalizationError("bundle matched-seed records are incomplete")
    expected_record_fields = {
        "seed",
        "candidate_metrics",
        "official_fm_metrics",
        "paired_deltas",
        "candidate_resources",
        "official_fm_resources",
        "resource_deltas",
        "candidate_prediction_artifact_sha256",
        "candidate_prediction_digest",
        "official_fm_prediction_artifact_sha256",
        "official_fm_prediction_digest",
        "scientific_request_digest",
        "scientific_record_digest",
        "checkpoint_sha256",
    }
    candidate_rows: list[Mapping[str, float]] = []
    fm_rows: list[Mapping[str, float]] = []
    paired_primary: list[Decimal] = []
    representative: Mapping[str, float] | None = None
    representative_fm: Mapping[str, float] | None = None
    representative_record: Mapping[str, object] | None = None
    for expected_seed, record in zip((0, 1, 2), records, strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != expected_record_fields
            or record.get("seed") != expected_seed
        ):
            raise ProductionFinalizationError("bundle matched-seed record schema changed")
        candidate = _manifest_metrics(record["candidate_metrics"], "bundle candidate seed")
        fm = _manifest_metrics(record["official_fm_metrics"], "bundle FM seed")
        deltas = record.get("paired_deltas")
        if not isinstance(deltas, dict) or set(deltas) != {"GAUC", "nDCG@5", "primary"}:
            raise ProductionFinalizationError("bundle paired seed deltas are incomplete")
        for name in ("GAUC", "nDCG@5", "primary"):
            delta = deltas[name]
            if isinstance(delta, bool) or not isinstance(delta, (int, float)):
                raise ProductionFinalizationError("bundle paired seed delta is not numeric")
            _require_close(float(delta), candidate[name] - fm[name], f"seed {expected_seed} {name}")
        for name in (
            "candidate_prediction_artifact_sha256",
            "candidate_prediction_digest",
            "official_fm_prediction_artifact_sha256",
            "official_fm_prediction_digest",
            "scientific_request_digest",
            "scientific_record_digest",
            "checkpoint_sha256",
        ):
            _digest(record.get(name), f"bundle seed {expected_seed} {name}")
        candidate_resource = record.get("candidate_resources")
        fm_resource = record.get("official_fm_resources")
        resource_delta = record.get("resource_deltas")
        if (
            not isinstance(candidate_resource, dict)
            or not isinstance(fm_resource, dict)
            or not isinstance(resource_delta, dict)
            or set(resource_delta) != {"wall_seconds", "peak_rss_bytes"}
        ):
            raise ProductionFinalizationError("bundle seed resource comparison is incomplete")
        try:
            _require_close(
                float(resource_delta["wall_seconds"]),
                float(candidate_resource["wall_seconds"]) - float(fm_resource["wall_seconds"]),
                f"seed {expected_seed} wall time delta",
            )
            if resource_delta["peak_rss_bytes"] != (
                candidate_resource["peak_rss_bytes"] - fm_resource["peak_rss_bytes"]
            ):
                raise ProductionFinalizationError(
                    f"seed {expected_seed} RSS delta differs from resource receipts"
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionFinalizationError("bundle seed resource values are invalid") from exc
        candidate_rows.append(candidate)
        fm_rows.append(fm)
        paired_primary.append(Decimal(str(deltas["primary"])))
        if expected_seed == representative_seed:
            representative = candidate
            representative_fm = fm
            representative_record = record
    if representative is None or representative_fm is None or representative_record is None:
        raise ProductionFinalizationError("bundle representative matched seed is absent")
    # The declared representative metrics come from the scorer; the matched-seed row recomputes
    # primary as the float64 mean of the same GAUC and nDCG@5.  Those two must match exactly, and
    # the derived mean is compared with the organizer dtype tolerance instead.
    for name in ("GAUC", "nDCG@5"):
        _require_close(metrics[name], representative[name], f"representative {name}")
    _require_close_organizer_primary(
        metrics["primary"],
        representative["primary"],
        "representative primary",
    )
    candidate_mean = _manifest_metrics(summary.get("candidate_mean"), "candidate seed mean")
    fm_mean = _manifest_metrics(summary.get("official_fm_mean"), "FM seed mean")
    reconstructed_candidate_mean = _mean_metric_rows(candidate_rows)
    reconstructed_fm_mean = _mean_metric_rows(fm_rows)
    for name in ("GAUC", "nDCG@5", "primary"):
        _require_close(
            candidate_mean[name],
            reconstructed_candidate_mean[name],
            f"candidate {name} mean",
        )
        _require_close(fm_mean[name], reconstructed_fm_mean[name], f"FM {name} mean")

    mean_primary = sum(paired_primary, Decimal(0)) / len(paired_primary)
    expected_primary_summary = {
        "mean": float(mean_primary),
        "median": float(statistics.median(paired_primary)),
        "minimum": float(min(paired_primary)),
        "population_std": float(statistics.pstdev(paired_primary)),
    }
    primary_summary = summary.get("primary_delta_summary")
    if not isinstance(primary_summary, dict) or set(primary_summary) != set(
        expected_primary_summary
    ):
        raise ProductionFinalizationError("bundle primary delta summary is incomplete")
    for name, expected in expected_primary_summary.items():
        value = primary_summary[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProductionFinalizationError("bundle primary delta summary is non-numeric")
        _require_close(float(value), expected, f"primary delta {name}")

    if len(inner) != 2:
        raise ProductionFinalizationError("bundle inner-fold evidence is incomplete")
    inner_deltas: list[Decimal] = []
    expected_inner_fields = {
        "fold",
        "candidate_metrics",
        "parent_metrics",
        "official_fm_reference_metrics",
        "primary_delta_to_parent",
        "primary_delta_to_reference",
        "evidence_digest",
        "weights_selected_on_public_validation",
    }
    for expected_fold, row in zip(("A", "B"), inner, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != expected_inner_fields
            or row.get("fold") != expected_fold
            or row.get("weights_selected_on_public_validation") is not False
        ):
            raise ProductionFinalizationError("bundle inner-fold record schema changed")
        candidate = _manifest_metrics(row["candidate_metrics"], "inner candidate")
        parent = _manifest_metrics(row["parent_metrics"], "inner parent")
        reference = _manifest_metrics(row["official_fm_reference_metrics"], "inner reference")
        for key, expected in (
            ("primary_delta_to_parent", candidate["primary"] - parent["primary"]),
            ("primary_delta_to_reference", candidate["primary"] - reference["primary"]),
        ):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProductionFinalizationError("bundle inner-fold delta is non-numeric")
            _require_close(float(value), expected, f"inner fold {expected_fold} {key}")
        _digest(row.get("evidence_digest"), f"inner fold {expected_fold} evidence")
        inner_deltas.append(Decimal(str(row["primary_delta_to_reference"])))
    mean_inner = sum(inner_deltas, Decimal(0)) / len(inner_deltas)
    worst_inner = min(inner_deltas)
    inner_summary = summary.get("inner_delta_summary")
    if not isinstance(inner_summary, dict) or set(inner_summary) != {
        "mean_primary_delta",
        "worst_primary_delta",
    }:
        raise ProductionFinalizationError("bundle inner delta summary is incomplete")
    _require_close(
        float(inner_summary["mean_primary_delta"]),
        float(mean_inner),
        "mean inner primary delta",
    )
    _require_close(
        float(inner_summary["worst_primary_delta"]),
        float(worst_inner),
        "worst inner primary delta",
    )
    if mean_primary <= Decimal(0) or mean_inner <= Decimal(0) or worst_inner < _WORST_INNER_DELTA:
        raise ProductionFinalizationError("bundle generated selection fails promotion guards")
    derived = (
        FinalStatus.MATERIALLY_CONFIRMED
        if mean_primary > _MATERIAL_PRIMARY_DELTA
        else FinalStatus.VALIDATION_IMPROVED
    )
    if summary.get("derived_status") != derived.value or declared is not derived:
        raise ProductionFinalizationError("bundle final status is not independently derivable")
    bootstrap = summary.get("paired_user_cluster_bootstrap")
    expected_bootstrap_fields = {
        "schema_version",
        "decision_use",
        "gating_eligible",
        "phase",
        "rows",
        "users",
        "gauc_eligible_users",
        "resamples",
        "seed",
        "point_estimate_source",
        "point_estimate_provenance",
        "metrics",
    }
    if (
        not isinstance(bootstrap, dict)
        or set(bootstrap) != expected_bootstrap_fields
        or bootstrap.get("schema_version") != 2
        or bootstrap.get("decision_use") != "diagnostic_only"
        or bootstrap.get("gating_eligible") is not False
        or bootstrap.get("phase") != DataPhase.OUTER_VALID.value
        or bootstrap.get("seed") != representative_seed
    ):
        raise ProductionFinalizationError("bundle bootstrap diagnostic is incomplete or gating")
    if bootstrap.get("point_estimate_source") != "protected_organizer_scorer":
        raise ProductionFinalizationError(
            "bundle bootstrap points are not owned by the protected organizer scorer"
        )
    rows = bootstrap.get("rows")
    users = bootstrap.get("users")
    eligible_users = bootstrap.get("gauc_eligible_users")
    resamples = bootstrap.get("resamples")
    if (
        type(rows) is not int
        or rows <= 0
        or type(users) is not int
        or not 1 <= users <= rows
        or type(eligible_users) is not int
        or not 0 <= eligible_users <= users
        or type(resamples) is not int
        or resamples <= 0
    ):
        raise ProductionFinalizationError("bundle bootstrap diagnostic dimensions are invalid")
    provenance = bootstrap.get("point_estimate_provenance")
    expected_provenance_fields = {
        "scorer_digest",
        "candidate_prediction_digest",
        "control_prediction_digest",
        "rows",
        "users",
        "primary_aggregation",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance_fields:
        raise ProductionFinalizationError("bundle protected bootstrap provenance is incomplete")
    if provenance.get("scorer_digest") != STARTER_FILE_SHA256["evaluate.py"]:
        raise ProductionFinalizationError("bundle bootstrap scorer identity is not pinned")
    if provenance.get("primary_aggregation") != (
        "selector_decimal_mean_of_protected_gauc_and_ndcg_at_5"
    ):
        raise ProductionFinalizationError(
            "bundle bootstrap protected primary aggregation convention changed"
        )
    if provenance.get("rows") != rows or provenance.get("users") != users:
        raise ProductionFinalizationError("bundle bootstrap provenance dimensions changed")
    if provenance.get("candidate_prediction_digest") != representative_record.get(
        "candidate_prediction_digest"
    ):
        raise ProductionFinalizationError(
            "bundle bootstrap candidate prediction provenance changed"
        )
    if provenance.get("control_prediction_digest") != representative_record.get(
        "official_fm_prediction_digest"
    ):
        raise ProductionFinalizationError("bundle bootstrap control prediction provenance changed")
    bootstrap_metrics = bootstrap.get("metrics")
    if not isinstance(bootstrap_metrics, dict) or set(bootstrap_metrics) != {
        "GAUC",
        "nDCG@5",
        "primary",
    }:
        raise ProductionFinalizationError("bundle bootstrap metric diagnostics are missing")
    for name in ("GAUC", "nDCG@5", "primary"):
        diagnostic = bootstrap_metrics.get(name)
        if not isinstance(diagnostic, dict) or set(diagnostic) != {
            "candidate",
            "control",
            "delta",
            "confidence_interval",
        }:
            raise ProductionFinalizationError("bundle bootstrap metric diagnostic is malformed")
        candidate_point = diagnostic.get("candidate")
        control_point = diagnostic.get("control")
        delta = diagnostic.get("delta")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (candidate_point, control_point, delta)
        ):
            raise ProductionFinalizationError("bundle bootstrap point evidence is non-numeric")
        candidate_value = float(cast(int | float, candidate_point))
        control_value = float(cast(int | float, control_point))
        delta_value = float(cast(int | float, delta))
        _require_close(
            candidate_value,
            representative[name],
            f"bootstrap candidate {name}",
        )
        _require_close(
            control_value,
            representative_fm[name],
            f"bootstrap control {name}",
        )
        _require_close(
            delta_value,
            candidate_value - control_value,
            f"bootstrap delta {name}",
        )
        interval = diagnostic.get("confidence_interval")
        if (
            not isinstance(interval, dict)
            or set(interval) != {"lower", "upper", "confidence_level", "method"}
            or interval.get("method") != "percentile-linear"
        ):
            raise ProductionFinalizationError("bundle bootstrap confidence interval is malformed")
        lower = interval.get("lower")
        upper = interval.get("upper")
        confidence = interval.get("confidence_level")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (lower, upper, confidence)
        ):
            raise ProductionFinalizationError("bundle bootstrap confidence interval is invalid")
        lower_value = float(cast(int | float, lower))
        upper_value = float(cast(int | float, upper))
        confidence_value = float(cast(int | float, confidence))
        if lower_value > upper_value or not 0.0 < confidence_value < 1.0:
            raise ProductionFinalizationError("bundle bootstrap confidence interval is invalid")
    return derived


def _verify_closed_bundle(
    bundle_dir: Path,
    *,
    require_immutable_directories: bool = True,
) -> _VerifiedBundle:
    try:
        supplied_metadata = bundle_dir.lstat()
        if stat.S_ISLNK(supplied_metadata.st_mode):
            raise ProductionFinalizationError("closed bundle path must not be a symlink")
        root = bundle_dir.resolve(strict=True)
        root_metadata = root.lstat()
    except (OSError, RuntimeError) as exc:
        raise ProductionFinalizationError("closed bundle directory is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ProductionFinalizationError("closed bundle must be a real directory")
    if require_immutable_directories and root_metadata.st_mode & 0o222:
        raise ProductionFinalizationError("closed bundle root retains directory write bits")
    manifest_path = root / "manifest.json"
    manifest = _strict_json(
        manifest_path,
        maximum=_MAX_MANIFEST_BYTES,
        location="closed bundle manifest",
    )
    expected_top = {
        "schema_version",
        "benchmark_identity",
        "starter_identity",
        "data_identity",
        "selection",
        "validation",
        "scientific_artifact_hashes",
        "prepublication_resource_receipt",
        "environment_and_resource_usage",
        "campaign_totals",
        "components",
        "known_limitations",
        "unresolved_organizer_questions",
    }
    if (
        set(manifest) != expected_top
        or manifest.get("schema_version") != FINAL_BUNDLE_SCHEMA_VERSION
    ):
        raise ProductionFinalizationError("closed bundle manifest schema is unsupported")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != {
        "required_paths",
        "roots",
        "files",
    }:
        raise ProductionFinalizationError("closed bundle component closure is malformed")
    required = components["required_paths"]
    expected_required = ["manifest.json", *REQUIRED_FILE_PATHS, *REQUIRED_DIRECTORY_PATHS]
    if required != expected_required:
        raise ProductionFinalizationError("closed bundle required-path contract changed")
    records = components["files"]
    if not isinstance(records, list) or not records:
        raise ProductionFinalizationError("closed bundle file inventory is empty")
    indexed: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(records):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "component",
            "sha256",
            "size_bytes",
        }:
            raise ProductionFinalizationError(f"closed bundle file record {index} is malformed")
        relative = _safe_relative(item["path"], f"closed bundle file {index} path")
        if relative == "manifest.json" or relative in indexed:
            raise ProductionFinalizationError("closed bundle file paths are not unique")
        _text(item["component"], f"closed bundle file {index} component")
        digest = _digest(item["sha256"], f"closed bundle file {index} digest")
        size = _bounded_int(
            item["size_bytes"],
            f"closed bundle file {index} size",
            maximum=2 * 1024**3,
        )
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_relative_to(root):  # defensive; PurePosixPath checks already reject '..'.
            raise ProductionFinalizationError("closed bundle file escapes its root")
        payload = _read_regular(
            path,
            maximum=max(1, size),
            location=f"closed bundle member {relative}",
        )
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ProductionFinalizationError(f"closed bundle member {relative} changed")
        indexed[relative] = MappingProxyType(dict(item))
    observed: set[str] = set()
    for candidate in root.rglob("*"):
        metadata = candidate.lstat()
        relative = candidate.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode) or (
            not stat.S_ISREG(metadata.st_mode) and not stat.S_ISDIR(metadata.st_mode)
        ):
            raise ProductionFinalizationError("closed bundle contains an unsafe member")
        if stat.S_ISDIR(metadata.st_mode) and (
            require_immutable_directories and metadata.st_mode & 0o222
        ):
            raise ProductionFinalizationError(
                f"closed bundle directory {relative} retains write bits"
            )
        if stat.S_ISREG(metadata.st_mode) and metadata.st_mode & 0o222:
            raise ProductionFinalizationError(f"closed bundle file {relative} retains write bits")
        if stat.S_ISREG(metadata.st_mode):
            observed.add(relative)
    if observed != {"manifest.json", *indexed}:
        raise ProductionFinalizationError("closed bundle inventory is not exact")
    prepublication_resource = manifest.get("prepublication_resource_receipt")
    if not isinstance(prepublication_resource, dict) or set(prepublication_resource) != {
        "path",
        "sha256",
    }:
        raise ProductionFinalizationError(
            "closed bundle prepublication resource receipt is malformed"
        )
    if prepublication_resource.get("path") != "prepublication-resource.json":
        raise ProductionFinalizationError(
            "closed bundle prepublication resource receipt path changed"
        )
    prepublication_record = indexed.get("prepublication-resource.json")
    if (
        prepublication_record is None
        or _digest(
            prepublication_resource.get("sha256"),
            "closed bundle prepublication resource receipt digest",
        )
        != prepublication_record["sha256"]
    ):
        raise ProductionFinalizationError(
            "closed bundle prepublication resource receipt identities differ"
        )
    scientific = manifest.get("scientific_artifact_hashes")
    if not isinstance(scientific, dict) or set(scientific) != {
        "source",
        "config",
        "features",
        "checkpoint",
        "predictions",
        "submission",
    }:
        raise ProductionFinalizationError("closed bundle scientific closure is malformed")
    for name, value in scientific.items():
        _digest(value, f"scientific artifact {name}")
    submission_record = indexed.get("submission.csv")
    if submission_record is None or submission_record["sha256"] != scientific["submission"]:
        raise ProductionFinalizationError("closed bundle submission identities differ")
    _derive_bundle_status(manifest)
    return _VerifiedBundle(
        root=root,
        manifest=MappingProxyType(manifest),
        manifest_sha256=sha256_file(manifest_path),
        files=MappingProxyType(indexed),
    )


def _close_bundle_directories(bundle_dir: Path) -> _VerifiedBundle:
    """Make the atomically published bundle non-replaceable through writable parents."""

    verified = _verify_closed_bundle(
        bundle_dir,
        require_immutable_directories=False,
    )
    directories = sorted(
        (path for path in verified.root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
        os.chmod(directory, 0o555, follow_symlinks=False)
    _fsync_directory(verified.root)
    os.chmod(verified.root, 0o555, follow_symlinks=False)
    _fsync_directory(verified.root.parent)
    return _verify_closed_bundle(verified.root)


def _resolve_project_path(project_root: Path, value: Path, location: str) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    try:
        supplied = candidate.lstat()
        if stat.S_ISLNK(supplied.st_mode):
            raise ProductionFinalizationError(f"{location} path must not be a symlink")
        resolved = candidate.resolve(strict=True)
        metadata = resolved.lstat()
    except (OSError, RuntimeError) as exc:
        raise ProductionFinalizationError(f"{location} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionFinalizationError(f"{location} must be a real directory")
    return resolved


def _alignment_rows(dataset: CanonicalDataset, *, final: bool) -> tuple[AlignmentRow, ...]:
    split = dataset.final if final else dataset.valid
    return tuple(
        AlignmentRow(row_id, user_id, video_id)
        for row_id, user_id, video_id in zip(
            split.alignment.row_id,
            split.alignment.user_id,
            split.alignment.video_id,
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class _TrustedReplayContext:
    dataset: CanonicalDataset
    capabilities: ReplayCapabilities
    scorer: ProtectedScorer
    split: SplitIdentity
    alignment: Alignment
    labels: tuple[int, ...]

    def score_result(self, scores: Float64Vector) -> ScoreResult:
        """Return the fresh protected aggregate with its exact prediction provenance."""

        return self.scorer.score_with_encoded_labels(
            alignment=self.alignment,
            split=self.split,
            labels=self.labels,
            scores=scores,
            expected_count=len(self.labels),
        )

    def score(self, scores: Float64Vector) -> Mapping[str, object]:
        return self.score_result(scores).as_dict()


def _trusted_replay_context(
    *, data_dir: Path, starter_dir: Path, expected_data_sha256: str
) -> _TrustedReplayContext:
    dataset = load_canonical_dataset(data_dir)
    if dataset.digest != _digest(expected_data_sha256, "expected_data_sha256"):
        raise ProductionFinalizationError("canonical dataset identity differs from finalization")
    if dataset.final.targets is not None:
        raise ProductionFinalizationError("final dataset unexpectedly exposes target capability")
    if not isinstance(dataset.valid.targets, ProtectedTargets):
        raise ProductionFinalizationError("public validation targets are not scorer-protected")
    validation_inputs = build_finalization_candidate_inputs(
        DataPhase.OUTER_VALID,
        dataset.valid.inputs,
    )
    final_inputs = build_finalization_candidate_inputs(
        DataPhase.FINAL,
        dataset.final.inputs,
    )
    split = SplitIdentity(
        name="outer_valid",
        token=validation_inputs.digest,
        expected_count=dataset.valid.row_count,
    )
    alignment = Alignment.from_ids(
        split=split,
        row_ids=dataset.valid.alignment.row_id,
        user_ids=dataset.valid.alignment.user_id,
        video_ids=dataset.valid.alignment.video_id,
    )
    scorer = ProtectedScorer(starter_dir=starter_dir, trusted_alignment=alignment)
    capabilities = ReplayCapabilities(
        data_sha256=dataset.digest,
        validation_inputs=validation_inputs,
        final_inputs=final_inputs,
        validation_alignment=_alignment_rows(dataset, final=False),
        final_alignment=_alignment_rows(dataset, final=True),
    )
    return _TrustedReplayContext(
        dataset=dataset,
        capabilities=capabilities,
        scorer=scorer,
        split=split,
        alignment=alignment,
        labels=dataset.valid.targets.reveal_for_scorer(),
    )


def _load_npy(reference: ArtifactRef, store: ArtifactStore, *, rows: int) -> Float64Vector:
    path = store.verify(reference)
    if path.stat().st_size > _MAX_PREDICTION_BYTES:
        raise ProductionFinalizationError("prediction artifact exceeds its size bound")
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ProductionFinalizationError("prediction artifact is not a safe NumPy array") from exc
    if (
        not isinstance(values, np.ndarray)
        or values.shape != (rows,)
        or values.dtype.kind not in "iuf"
    ):
        raise ProductionFinalizationError("prediction artifact shape or dtype is invalid")
    result = np.ascontiguousarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ProductionFinalizationError("prediction artifact contains NaN or infinity")
    result.setflags(write=False)
    return result


def _put_directory_from_members(
    store: ArtifactStore,
    *,
    kind: ArtifactKind,
    members: Sequence[tuple[str, Path, str]],
    prefix: str,
) -> DirectoryArtifactRef:
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=store.staging_root))
    try:
        for relative, source, expected in members:
            _safe_relative(relative, "artifact member path")
            if sha256_file(source) != _digest(expected, f"{relative} SHA-256"):
                raise ProductionFinalizationError(f"artifact member {relative} changed")
            target = root.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, target)
            if sha256_file(target) != expected:
                raise ProductionFinalizationError(
                    f"artifact member {relative} changed while copied"
                )
        return store.put_directory(root, kind=kind)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _artifact_spec(reference: ArtifactRef | DirectoryArtifactRef, *, kind: str) -> ArtifactSpec:
    if isinstance(reference, DirectoryArtifactRef):
        metadata = reference.manifest() | {
            "logical_closure_digest": reference.sha256,
            "entry_count": len(reference.entries),
        }
        relative = reference.manifest_artifact.object_relative_path.as_posix()
        size = reference.total_size_bytes
    else:
        metadata = reference.manifest()
        relative = reference.object_relative_path.as_posix()
        size = reference.size_bytes
    return ArtifactSpec(
        digest=reference.sha256,
        kind=kind,
        relative_path=relative,
        size_bytes=size,
        metadata=metadata,
    )


def _official_member(
    run: OfficialFMSeedEvidence | OfficialFMFallbackEvidence,
    *,
    starter_manifest_digest: str,
) -> OfficialFMMemberRecipe:
    return OfficialFMMemberRecipe(
        checkpoint_sha256=run.checkpoint_file_sha256,
        checkpoint_digest=run.checkpoint_digest,
        encoding_sha256=run.encoding_file_sha256,
        encoding_digest=run.encoding_digest,
        config_digest=run.config_digest,
        starter_manifest_sha256=starter_manifest_digest,
        seed=run.seed,
    )


def _require_planned_member(
    planned: QualifiedFMMemberPlan,
    run: OfficialFMSeedEvidence,
    qualification: OfficialFMQualificationEvidence,
) -> OfficialFMMemberRecipe:
    exact = {
        "seed": (planned.seed, run.seed),
        "checkpoint_sha256": (planned.checkpoint_sha256, run.checkpoint_file_sha256),
        "checkpoint_digest": (planned.checkpoint_digest, run.checkpoint_digest),
        "encoding_sha256": (planned.encoding_sha256, run.encoding_file_sha256),
        "encoding_digest": (planned.encoding_digest, run.encoding_digest),
        "config_digest": (planned.config_digest, run.config_digest),
        "starter_manifest_digest": (
            planned.starter_manifest_digest,
            qualification.starter_manifest_digest,
        ),
        "validation_prediction_digest": (
            planned.validation_prediction_digest,
            run.validation_prediction_digest,
        ),
    }
    mismatches = [name for name, values in exact.items() if values[0] != values[1]]
    if mismatches:
        raise ProductionFinalizationError(
            "selected matched FM identity changed: " + ", ".join(mismatches)
        )
    return _official_member(run, starter_manifest_digest=qualification.starter_manifest_digest)


def _require_selected_member(
    selection: FinalizationSelectionPlan,
    run: OfficialFMSeedEvidence,
    qualification: OfficialFMQualificationEvidence,
) -> OfficialFMMemberRecipe:
    return _require_planned_member(selection.fm_member, run, qualification)


def _load_research_outcome(
    run_dir: Path,
    *,
    engine: CampaignEngine,
) -> tuple[CampaignCreateRequest, ArtifactStore, FullCampaignOutcome]:
    request = engine.load_request(run_dir)
    store = ArtifactStore(run_dir / "artifacts")
    progress = FullCampaignProgressLedger(
        run_dir / "production" / "progress",
        create=False,
    )
    outcome = FullCampaignOutcomeRepository(
        run_dir=run_dir,
        artifact_store=store,
        progress=progress,
    ).load(request_digest=request.digest)
    if (
        outcome.run_dir != run_dir
        or outcome.campaign_id != request.campaign_id
        or outcome.request_digest != request.digest
        or outcome.qualification_manifest_digest != request.qualification_manifest_digest
        or not outcome.finalization_required
        or not outcome.fallback_preserved
    ):
        raise ProductionFinalizationError("research outcome differs from the campaign request")
    return request, store, outcome


def _qualification(
    request: CampaignCreateRequest,
    context: _TrustedReplayContext,
) -> OfficialFMQualificationEvidence:
    evidence = load_official_fm_qualification(
        request.qualification_run_dir,
        expectations=QualificationExpectations(
            canonical_digest=context.dataset.digest,
            starter_manifest_digest=request.starter_manifest_digest,
            scorer_digest=context.scorer.scorer_digest,
            validation_row_count=context.dataset.valid.row_count,
            final_row_count=context.dataset.final.row_count,
        ),
    )
    if evidence.manifest_digest != request.qualification_manifest_digest:
        raise ProductionFinalizationError("qualification manifest differs from campaign request")
    return evidence


def _environment(request: CampaignCreateRequest, project_root: Path) -> Mapping[str, object]:
    source = hash_source_tree(project_root)
    if source.digest != request.source_digest:
        raise ProductionFinalizationError("trusted project source differs from campaign creation")
    identity = capture_environment_identity(project_root)
    if identity.digest != request.environment_digest:
        raise ProductionFinalizationError("locked environment differs from campaign creation")
    manifest = identity.manifest()
    if environment_identity_digest(manifest) != request.environment_digest:
        raise ProductionFinalizationError("environment manifest cannot reprove its identity")
    return MappingProxyType(manifest)


def _dependency_groups(environment: Mapping[str, object]) -> tuple[str, ...]:
    packages = environment.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != {
        "lightgbm",
        "numpy",
        "psutil",
        "torch",
    }:
        raise ProductionFinalizationError("environment package profile is incomplete")
    if type(packages.get("lightgbm")) is not str:
        raise ProductionFinalizationError("LambdaRank replay requires the research-tree group")
    torch = packages.get("torch")
    if torch is not None and type(torch) is not str:
        raise ProductionFinalizationError("environment torch profile is malformed")
    return ("research-tree", "research-neural") if torch is not None else ("research-tree",)


def _training_execution_id(selection: FinalizationSelectionPlan) -> str:
    return _FINAL_TRAINING_EXECUTION_PREFIX + selection.digest[:40]


def _verify_charged_final_training_terminal(
    store: CampaignStore,
    selection: FinalizationSelectionPlan,
) -> None:
    execution_id = _training_execution_id(selection)
    execution = store.execution(execution_id)
    if (
        execution is None
        or execution.status != "SUCCEEDED"
        or execution.experiment_id != selection.experiment_id
        or execution.seed != selection.representative_seed
        or execution.source_digest != selection.source_digest
        or execution.config_digest != selection.config_digest
        or execution.data_digest != selection.dataset_digest
        or execution.launch_category != LaunchCategory.FINAL_TRAINING_REPLAY.value
        or execution.original_launch_category != LaunchCategory.FINAL_TRAINING_REPLAY.value
    ):
        raise ProductionFinalizationError(
            "generated final bundle lacks its exact charged successful training terminal"
        )
    artifacts = store.artifacts_for(owner_type="execution", owner_id=execution_id)
    checkpoint = tuple(artifact for role, artifact in artifacts if role == "checkpoint")
    if (
        len(checkpoint) != 1
        or checkpoint[0].digest != selection.tree_checkpoint.sha256
        or checkpoint[0].kind != ArtifactKind.CHECKPOINT.value
    ):
        raise ProductionFinalizationError(
            "charged final training terminal checkpoint differs from selected evidence"
        )


def _rehydrate_checkpoint(
    journal: CampaignStoreCandidateJournal,
    execution_id: str,
) -> tuple[ArtifactRef, Mapping[str, object]]:
    terminal = journal.rehydrate_terminal(execution_id)
    if terminal.execution.status != "SUCCEEDED":
        raise ProductionFinalizationError("retained final training execution failed")
    try:
        checkpoint = terminal.artifacts.artifact("checkpoint")
    except KeyError as exc:
        raise ProductionFinalizationError(
            "retained final training execution lacks a checkpoint"
        ) from exc
    if checkpoint.kind is not ArtifactKind.CHECKPOINT:
        raise ProductionFinalizationError("retained final checkpoint has the wrong artifact kind")
    try:
        execution_manifest = terminal.artifacts.artifact("execution_manifest")
    except KeyError as exc:
        raise ProductionFinalizationError(
            "retained final training execution lacks its resource manifest"
        ) from exc
    manifest_path = journal.artifact_store.verify(execution_manifest)
    manifest = _strict_json(
        manifest_path,
        maximum=_MAX_MANIFEST_BYTES,
        location="retained final training execution manifest",
    )
    if (
        manifest.get("execution_id") != execution_id
        or manifest.get("outcome") != "succeeded"
        or manifest.get("device") != "cpu"
    ):
        raise ProductionFinalizationError(
            "retained final training resource manifest differs from its terminal"
        )
    wall = manifest.get("wall_seconds")
    peak_rss = manifest.get("peak_tree_rss_bytes")
    disk = manifest.get("peak_workspace_bytes")
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0.0
        or type(peak_rss) is not int
        or peak_rss < 0
        or type(disk) is not int
        or disk < 0
    ):
        raise ProductionFinalizationError("retained final training resources are malformed")
    return checkpoint, MappingProxyType(
        {
            "wall_seconds": float(wall),
            "peak_rss_bytes": peak_rss,
            "disk_bytes": disk,
        }
    )


def _fresh_training_replay(
    *,
    run_dir: Path,
    request: CampaignCreateRequest,
    selection: FinalizationSelectionPlan,
    artifact_store: ArtifactStore,
    engine: CampaignEngine,
    store: CampaignStore,
    cancel_event: threading.Event | None,
) -> tuple[ArtifactRef, Mapping[str, object]]:
    _raise_if_cancelled(cancel_event, "final training admission")
    execution_resources: Mapping[str, object] | None = None
    execution_id = _training_execution_id(selection)
    deadline = engine.observe_deadline(run_dir)
    policy = CandidateJournalPolicy(
        family="generated-causal-lambdarank-final-training-replay",
        phase=WorkPhase.FINALIZATION,
        p95_runtime_seconds=float(selection.timeout_seconds),
        cleanup_seconds=60.0,
        category=LaunchCategory.FINAL_TRAINING_REPLAY,
        original_category=LaunchCategory.FINAL_TRAINING_REPLAY,
        experiment_id=selection.experiment_id,
        scientific_iteration=None,
    )
    journal = CampaignStoreCandidateJournal(
        store=store,
        artifact_store=artifact_store,
        deadline=deadline,
        policy=policy,
    )
    existing = store.execution(execution_id)
    if existing is not None:
        if existing.status in {"STARTING", "RUNNING"}:
            raise CandidateExecutionPendingError(
                f"final training execution {execution_id!r} still requires reconciliation"
            )
        checkpoint, execution_resources = _rehydrate_checkpoint(journal, execution_id)
        mode = "rehydrated_exact_terminal_training_replay"
    else:
        workspaces = WorkspaceMaterializer(
            run_dir / "production" / "finalization-workspaces",
            artifact_store=artifact_store,
        )
        control_root = run_dir / "production" / "finalization-control"
        control_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = control_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ProductionFinalizationError("finalization control root is unsafe")
        limits = LocalCandidateLimits(
            timeout_seconds=float(selection.timeout_seconds),
            memory_limit_bytes=selection.memory_limit_bytes,
            workspace_disk_limit_bytes=request.config.runner.disk_mb * 1024**2,
            output_limit_bytes=min(request.config.runner.disk_mb * 1024**2, 2 * 1024**3),
            temp_limit_bytes=min(request.config.runner.disk_mb * 1024**2, 2 * 1024**3),
            threads=selection.threads,
            device="cpu",
        )
        executor = GeneratedCandidateExecutor(
            artifact_store=artifact_store,
            workspace_materializer=workspaces,
            control_root=control_root,
            interpreter=active_python_interpreter(),
            limits=limits,
        )
        training = GeneratedTrainRequest(
            execution_id=execution_id,
            identity=GeneratedCandidateIdentity(
                source_snapshot=selection.source_snapshot,
                source_digest=selection.source_digest,
                config_digest=selection.config_digest,
            ),
            split_role=SplitRole.TRAIN,
            data_digest=selection.dataset_digest,
            split_token=selection.training_policy_digest,
            seed=selection.representative_seed,
            features=selection.training_features,
            targets=selection.training_targets,
            user_groups=selection.training_user_groups,
        )
        try:
            run = executor.train(training, journal=journal, cancel_event=cancel_event)
            checkpoint = run.checkpoint
        except CandidateExecutionTerminalError:
            checkpoint, execution_resources = _rehydrate_checkpoint(journal, execution_id)
            mode = "rehydrated_exact_terminal_training_replay"
        else:
            mode = "fresh_subprocess_official_train_replay"
            execution_resources = MappingProxyType(
                {
                    "wall_seconds": run.execution.wall_seconds,
                    "peak_rss_bytes": run.execution.peak_tree_rss_bytes,
                    "disk_bytes": run.execution.peak_workspace_bytes,
                }
            )
    artifact_store.verify(checkpoint)
    if checkpoint.sha256 != selection.tree_checkpoint.sha256:
        raise ProductionFinalizationError(
            "fresh official-train checkpoint bytes differ from selected outer checkpoint"
        )
    evidence: dict[str, object] = {
        "required": True,
        "completed": True,
        "mode": mode,
        "execution_id": execution_id,
        "seed": selection.representative_seed,
        "training_rows_include_public_validation": False,
        "training_period": "official_train_20220408_to_20220421",
        "checkpoint_sha256": checkpoint.sha256,
        "selected_checkpoint_sha256": selection.tree_checkpoint.sha256,
        "exact_checkpoint_bytes": True,
        "charged_launch": True,
        "device": "cpu",
    }
    if execution_resources is not None:
        evidence["resources"] = dict(execution_resources)
    return checkpoint, MappingProxyType(evidence)


def _recipe_ref(
    store: ArtifactStore,
    recipe: ReplayRecipe,
) -> ArtifactRef:
    directory = Path(tempfile.mkdtemp(prefix="replay-recipe-", dir=store.staging_root))
    try:
        path = write_replay_recipe(directory / "recipe.json", recipe)
        reference = store.put_file(path, kind=ArtifactKind.INPUT)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    if reference.sha256 != recipe.digest:
        raise ProductionFinalizationError("replay recipe artifact identity changed")
    return reference


def _organizer_metrics(value: OrganizerMetrics) -> dict[str, float]:
    if not isinstance(value, OrganizerMetrics):
        raise ProductionFinalizationError("organizer metric evidence is incomplete")
    gauc = _metric(value.gauc, "organizer GAUC")
    ndcg = _metric(value.ndcg_at_5, "organizer nDCG@5")
    primary = _metric(value.primary, "organizer primary")
    if not math.isclose(primary, (gauc + ndcg) / 2.0, rel_tol=0.0, abs_tol=2e-7):
        raise ProductionFinalizationError("organizer primary is not mean(GAUC, nDCG@5)")
    return {"GAUC": gauc, "nDCG@5": ndcg, "primary": primary}


def _require_metric_parity(
    observed: Mapping[str, object],
    recorded: Mapping[str, float],
    location: str,
) -> None:
    for name in ("GAUC", "nDCG@5", "primary"):
        if not math.isclose(
            _metric(observed.get(name), f"{location} rescored {name}"),
            recorded[name],
            rel_tol=0.0,
            abs_tol=2e-7,
        ):
            raise ProductionFinalizationError(
                f"{location} recorded {name} differs from protected rescore"
            )


def _qualification_seed_resources(
    qualification: OfficialFMQualificationEvidence,
) -> Mapping[int, Mapping[str, object]]:
    resources_by_seed: dict[int, Mapping[str, object]] = {}
    for run in qualification.outer_runs:
        resource = run.resources
        resources_by_seed[run.seed] = MappingProxyType(
            {
                "wall_seconds": float(resource.wall_seconds),
                "cpu_seconds": float(resource.cpu_seconds),
                "peak_rss_bytes": resource.peak_rss_bytes,
                "disk_bytes": resource.disk_bytes,
                "device": resource.device,
            }
        )
    if not all(seed in resources_by_seed for seed in (0, 1, 2)):
        raise ProductionFinalizationError("qualification FM resource seeds 0, 1, 2 are incomplete")
    return MappingProxyType(resources_by_seed)


def _mean_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ProductionFinalizationError("matched metric rows cannot be empty")
    return {
        name: float(math.fsum(row[name] for row in rows) / len(rows))
        for name in ("GAUC", "nDCG@5", "primary")
    }


@dataclass(frozen=True, slots=True)
class _GeneratedConfirmation:
    status: FinalStatus
    representative_metrics: Mapping[str, float]
    candidate_mean: Mapping[str, float]
    fm_mean: Mapping[str, float]
    seed_rows: tuple[Mapping[str, object], ...]
    primary_summary: Mapping[str, object]
    inner_rows: tuple[Mapping[str, object], ...]
    inner_summary: Mapping[str, float]
    bootstrap: Mapping[str, object]

    @property
    def retained_training_peak_rss_bytes(self) -> int:
        """Maximum peak RSS attributable to the retained generated matched-seed runs.

        The matched official-FM resources remain in ``seed_rows`` for paired resource-delta
        evidence, but they are a separate lineage.  Deriving this value directly from the
        generated candidate rows prevents a larger baseline peak from being relabelled as
        selected-candidate training usage in the final bundle.
        """

        peaks: list[int] = []
        seeds: list[int] = []
        for index, row in enumerate(self.seed_rows):
            seed = row.get("seed")
            resources = row.get("candidate_resources")
            if type(seed) is not int or not isinstance(resources, Mapping):
                raise ProductionFinalizationError(
                    f"retained generated seed row {index} has malformed resource provenance"
                )
            peak = resources.get("peak_rss_bytes")
            if type(peak) is not int or peak < 0:
                raise ProductionFinalizationError(
                    f"retained generated seed {seed} peak RSS is invalid"
                )
            seeds.append(seed)
            peaks.append(peak)
        if tuple(seeds) != (0, 1, 2):
            raise ProductionFinalizationError(
                "retained generated resource provenance must cover seeds 0, 1, 2"
            )
        return max(peaks)

    def seed_summary(self, representative_seed: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "seeds": [0, 1, 2],
            "representative_seed": representative_seed,
            "candidate_mean": dict(self.candidate_mean),
            "official_fm_mean": dict(self.fm_mean),
            "per_seed": [dict(item) for item in self.seed_rows],
            "primary_delta_summary": dict(self.primary_summary),
            "inner_delta_summary": dict(self.inner_summary),
            "paired_user_cluster_bootstrap": dict(self.bootstrap),
            "derived_status": self.status.value,
            "status_thresholds": {
                "material_primary_delta_strictly_greater_than": 0.002,
                "mean_inner_primary_delta_strictly_greater_than": 0.0,
                "worst_inner_primary_delta_minimum": -0.002,
            },
            "confirmation_is_controller_derived": True,
        }

    def seed_report_lines(self) -> tuple[str, ...]:
        rows = tuple(
            "Seed {seed}: candidate GAUC={cg}, nDCG@5={cn}, primary={cp}; "
            "official FM GAUC={fg}, nDCG@5={fn}, primary={fp}; paired deltas "
            "GAUC={dg}, nDCG@5={dn}, primary={dp}; runtime delta={dw}s; "
            "peak-RSS delta={dr} bytes.".format(
                seed=item["seed"],
                cg=cast(dict[str, float], item["candidate_metrics"])["GAUC"],
                cn=cast(dict[str, float], item["candidate_metrics"])["nDCG@5"],
                cp=cast(dict[str, float], item["candidate_metrics"])["primary"],
                fg=cast(dict[str, float], item["official_fm_metrics"])["GAUC"],
                fn=cast(dict[str, float], item["official_fm_metrics"])["nDCG@5"],
                fp=cast(dict[str, float], item["official_fm_metrics"])["primary"],
                dg=cast(dict[str, float], item["paired_deltas"])["GAUC"],
                dn=cast(dict[str, float], item["paired_deltas"])["nDCG@5"],
                dp=cast(dict[str, float], item["paired_deltas"])["primary"],
                dw=cast(dict[str, object], item["resource_deltas"])["wall_seconds"],
                dr=cast(dict[str, object], item["resource_deltas"])["peak_rss_bytes"],
            )
            for item in self.seed_rows
        )
        primary = self.primary_summary
        bootstrap_metrics = self.bootstrap.get("metrics")
        if not isinstance(bootstrap_metrics, dict):  # guarded by bootstrap primitive.
            raise ProductionFinalizationError("bootstrap confirmation metrics are missing")
        primary_bootstrap = bootstrap_metrics.get("primary")
        if not isinstance(primary_bootstrap, dict):
            raise ProductionFinalizationError("bootstrap primary diagnostic is missing")
        interval = primary_bootstrap.get("confidence_interval")
        if not isinstance(interval, dict):
            raise ProductionFinalizationError("bootstrap primary interval is missing")
        return (
            *rows,
            "Paired primary delta summary: mean={mean}, median={median}, minimum={minimum}, "
            "population std={population_std}.".format(**primary),
            "Representative-seed paired user-cluster bootstrap: primary delta={delta}, "
            "95% percentile interval=[{lower}, {upper}], diagnostic only.".format(
                delta=primary_bootstrap["delta"],
                lower=interval["lower"],
                upper=interval["upper"],
            ),
        )


def _reconstruct_generated_confirmation(
    *,
    selection: FinalizationSelectionPlan,
    qualification: OfficialFMQualificationEvidence,
    context: _TrustedReplayContext,
    artifact_store: ArtifactStore,
) -> _GeneratedConfirmation:
    if selection.source_digest == selection.parent_source_digest:
        raise ProductionFinalizationError("selected generated source is not a material change")
    fm_resources = _qualification_seed_resources(qualification)
    seed_rows: list[Mapping[str, object]] = []
    candidate_metrics: list[Mapping[str, float]] = []
    fm_metrics: list[Mapping[str, float]] = []
    representative_candidate: Float64Vector | None = None
    representative_fm: Float64Vector | None = None
    representative_candidate_result: ScoreResult | None = None
    representative_fm_result: ScoreResult | None = None
    for matched in selection.matched_seeds:
        run = qualification.outer_seed(matched.seed)
        _require_planned_member(matched.fm_member, run, qualification)
        if matched.fm_validation_prediction.sha256 != run.validation_predictions_file_sha256:
            raise ProductionFinalizationError(
                f"matched seed {matched.seed} FM prediction artifact changed"
            )
        candidate_scores = _load_npy(
            matched.candidate_validation_prediction,
            artifact_store,
            rows=context.dataset.valid.row_count,
        )
        fm_scores = _load_npy(
            matched.fm_validation_prediction,
            artifact_store,
            rows=context.dataset.valid.row_count,
        )
        if prediction_digest(fm_scores) != run.validation_prediction_digest:
            raise ProductionFinalizationError(
                f"matched seed {matched.seed} FM logical prediction changed"
            )
        candidate_recorded = _organizer_metrics(matched.candidate_metrics)
        fm_recorded = _organizer_metrics(matched.fm_metrics)
        candidate_result = context.score_result(candidate_scores)
        fm_result = context.score_result(fm_scores)
        _require_metric_parity(
            candidate_result.as_dict(),
            candidate_recorded,
            f"matched seed {matched.seed} candidate",
        )
        _require_metric_parity(
            fm_result.as_dict(),
            fm_recorded,
            f"matched seed {matched.seed} FM",
        )
        qualified = {
            "GAUC": run.metrics.gauc,
            "nDCG@5": run.metrics.ndcg_at_5,
            "primary": run.metrics.primary,
        }
        _require_metric_parity(fm_recorded, qualified, f"matched seed {matched.seed} qualified FM")
        deltas = {
            name: candidate_recorded[name] - fm_recorded[name]
            for name in ("GAUC", "nDCG@5", "primary")
        }
        baseline_resource = fm_resources[matched.seed]
        candidate_resource = {
            "wall_seconds": float(matched.candidate_wall_seconds),
            "peak_rss_bytes": matched.candidate_peak_rss_bytes,
            "disk_bytes": matched.candidate_disk_bytes,
            "device": "cpu",
        }
        resource_deltas = {
            "wall_seconds": (
                float(matched.candidate_wall_seconds)
                - cast(float, baseline_resource["wall_seconds"])
            ),
            "peak_rss_bytes": (
                matched.candidate_peak_rss_bytes - cast(int, baseline_resource["peak_rss_bytes"])
            ),
        }
        seed_rows.append(
            MappingProxyType(
                {
                    "seed": matched.seed,
                    "candidate_metrics": candidate_recorded,
                    "official_fm_metrics": fm_recorded,
                    "paired_deltas": deltas,
                    "candidate_resources": candidate_resource,
                    "official_fm_resources": dict(baseline_resource),
                    "resource_deltas": resource_deltas,
                    "candidate_prediction_artifact_sha256": (
                        matched.candidate_validation_prediction.sha256
                    ),
                    "candidate_prediction_digest": prediction_digest(candidate_scores),
                    "official_fm_prediction_artifact_sha256": (
                        matched.fm_validation_prediction.sha256
                    ),
                    "official_fm_prediction_digest": prediction_digest(fm_scores),
                    "scientific_request_digest": matched.scientific_request_digest,
                    "scientific_record_digest": matched.scientific_record_digest,
                    "checkpoint_sha256": matched.checkpoint.sha256,
                }
            )
        )
        candidate_metrics.append(candidate_recorded)
        fm_metrics.append(fm_recorded)
        if matched.seed == selection.representative_seed:
            representative_candidate = candidate_scores
            representative_fm = fm_scores
            representative_candidate_result = candidate_result
            representative_fm_result = fm_result
    if (
        representative_candidate is None
        or representative_fm is None
        or representative_candidate_result is None
        or representative_fm_result is None
    ):
        raise ProductionFinalizationError("representative matched seed is unavailable")

    paired_primary = tuple(
        Decimal(str(cast(dict[str, float], row["paired_deltas"])["primary"])) for row in seed_rows
    )
    mean_primary = sum(paired_primary, Decimal(0)) / len(paired_primary)
    primary_summary: dict[str, object] = {
        "mean": float(mean_primary),
        "median": float(statistics.median(paired_primary)),
        "minimum": float(min(paired_primary)),
        "population_std": float(statistics.pstdev(paired_primary)),
    }
    inner_rows: list[Mapping[str, object]] = []
    inner_deltas: list[Decimal] = []
    for fold in selection.inner_folds:
        candidate = _organizer_metrics(fold.candidate)
        parent = _organizer_metrics(fold.parent)
        reference = _organizer_metrics(fold.reference)
        delta_parent = candidate["primary"] - parent["primary"]
        delta_reference = candidate["primary"] - reference["primary"]
        inner_deltas.append(Decimal(str(delta_reference)))
        inner_rows.append(
            MappingProxyType(
                {
                    "fold": fold.fold_id,
                    "candidate_metrics": candidate,
                    "parent_metrics": parent,
                    "official_fm_reference_metrics": reference,
                    "primary_delta_to_parent": delta_parent,
                    "primary_delta_to_reference": delta_reference,
                    "evidence_digest": fold.digest,
                    "weights_selected_on_public_validation": False,
                }
            )
        )
    mean_inner = sum(inner_deltas, Decimal(0)) / len(inner_deltas)
    worst_inner = min(inner_deltas)
    inner_summary = {
        "mean_primary_delta": float(mean_inner),
        "worst_primary_delta": float(worst_inner),
    }
    inner_guards = mean_inner > Decimal(0) and worst_inner >= _WORST_INNER_DELTA
    if not inner_guards or mean_primary <= Decimal(0):
        raise ProductionFinalizationError(
            "selected generated candidate fails independently reconstructed promotion guards"
        )
    status = (
        FinalStatus.MATERIALLY_CONFIRMED
        if mean_primary > _MATERIAL_PRIMARY_DELTA
        else FinalStatus.VALIDATION_IMPROVED
    )
    diagnostic = paired_user_cluster_bootstrap(
        context.alignment.user_ids,
        context.labels,
        representative_candidate,
        representative_fm,
        phase=DataPhase.OUTER_VALID,
        seed=selection.representative_seed,
        candidate_protected_result=representative_candidate_result,
        control_protected_result=representative_fm_result,
    )
    representative_recorded = next(
        cast(dict[str, float], row["candidate_metrics"])
        for row in seed_rows
        if row["seed"] == selection.representative_seed
    )
    diagnostic_manifest = diagnostic.as_dict()
    diagnostic_metrics = diagnostic_manifest.get("metrics")
    if not isinstance(diagnostic_metrics, dict):
        raise ProductionFinalizationError("paired bootstrap metric evidence is malformed")
    diagnostic_candidate: dict[str, object] = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        item = diagnostic_metrics.get(name)
        if not isinstance(item, dict):
            raise ProductionFinalizationError("paired bootstrap metric evidence is incomplete")
        diagnostic_candidate[name] = item.get("candidate")
    _require_metric_parity(
        diagnostic_candidate,
        representative_recorded,
        "paired bootstrap representative candidate",
    )
    return _GeneratedConfirmation(
        status=status,
        representative_metrics=MappingProxyType(representative_recorded),
        candidate_mean=MappingProxyType(_mean_metric_rows(candidate_metrics)),
        fm_mean=MappingProxyType(_mean_metric_rows(fm_metrics)),
        seed_rows=tuple(seed_rows),
        primary_summary=MappingProxyType(primary_summary),
        inner_rows=tuple(inner_rows),
        inner_summary=MappingProxyType(inner_summary),
        bootstrap=MappingProxyType(diagnostic_manifest),
    )


def _generated_replay_candidate(
    *,
    selection: FinalizationSelectionPlan,
    checkpoint: ArtifactRef,
    qualification: OfficialFMQualificationEvidence,
    context: _TrustedReplayContext,
    request: CampaignCreateRequest,
    environment: Mapping[str, object],
    artifact_store: ArtifactStore,
    outcome: FullCampaignOutcome,
    report_failures: Sequence[ProductionFailureEvidence],
    campaign_wall_seconds: float,
    launch_count: int,
    cancel_event: threading.Event | None,
) -> FinalizationCandidate:
    if selection.dataset_digest != context.dataset.digest:
        raise ProductionFinalizationError("selected generated data identity changed")
    if (
        selection.validation_inputs_digest != context.capabilities.validation_inputs.digest
        or selection.final_inputs_digest != context.capabilities.final_inputs.digest
    ):
        raise ProductionFinalizationError("selected generated capability identity changed")
    run = qualification.outer_seed(selection.representative_seed)
    fm_member = _require_selected_member(selection, run, qualification)
    confirmation = _reconstruct_generated_confirmation(
        selection=selection,
        qualification=qualification,
        context=context,
        artifact_store=artifact_store,
    )
    preprocessing = _put_directory_from_members(
        artifact_store,
        kind=ArtifactKind.INPUT,
        prefix="final-preprocessing-",
        members=(
            (
                "validation.npy",
                artifact_store.verify(selection.validation_features),
                selection.validation_features.sha256,
            ),
            (
                "final.npy",
                artifact_store.verify(selection.final_features),
                selection.final_features.sha256,
            ),
            ("fm-encoding.npz", run.encoding_path, run.encoding_file_sha256),
        ),
    )
    model = _put_directory_from_members(
        artifact_store,
        kind=ArtifactKind.CHECKPOINT,
        prefix="final-model-",
        members=(
            ("tree-model.txt", artifact_store.verify(checkpoint), checkpoint.sha256),
            ("fm-checkpoint.npz", run.checkpoint_path, run.checkpoint_file_sha256),
        ),
    )
    source_entries = {entry.path: entry.artifact for entry in selection.source_snapshot.entries}
    candidate_source = source_entries.get("candidate.py")
    candidate_config = source_entries.get("config.json")
    if candidate_source is None or candidate_config is None:
        raise ProductionFinalizationError("generated source snapshot is incomplete")
    recipe = GeneratedLambdaRankReplayRecipe(
        source_artifact_sha256=selection.source_snapshot.sha256,
        candidate_source_sha256=candidate_source.sha256,
        candidate_config_sha256=candidate_config.sha256,
        feature_artifact_sha256=preprocessing.sha256,
        validation_features_sha256=selection.validation_features.sha256,
        final_features_sha256=selection.final_features.sha256,
        checkpoint_artifact_sha256=model.sha256,
        tree_checkpoint_sha256=checkpoint.sha256,
        data_sha256=context.dataset.digest,
        validation_inputs_digest=context.capabilities.validation_inputs.digest,
        final_inputs_digest=context.capabilities.final_inputs.digest,
        feature_count=selection.feature_count,
        timeout_seconds=selection.timeout_seconds,
        memory_limit_bytes=selection.memory_limit_bytes,
        threads=selection.threads,
        fm_member=fm_member,
        fusion_weights=selection.frozen_fusion_weights,
    )
    recipe_reference = _recipe_ref(artifact_store, recipe)
    reference = _load_npy(
        selection.validation_prediction,
        artifact_store,
        rows=context.dataset.valid.row_count,
    )
    metrics = context.score(reference)
    _require_metric_parity(
        metrics,
        confirmation.representative_metrics,
        "selected representative seed",
    )
    identity = FrozenReplayIdentity(
        source_sha256=selection.source_snapshot.sha256,
        config_sha256=recipe_reference.sha256,
        features_sha256=preprocessing.sha256,
        checkpoint_sha256=model.sha256,
        validation_prediction_artifact_sha256=selection.validation_prediction.sha256,
        validation_prediction_digest=prediction_digest(reference),
        data_sha256=context.dataset.digest,
        environment_sha256=request.environment_digest,
    )
    status = confirmation.status
    return FinalizationCandidate(
        candidate_id=selection.candidate_id,
        lineage=(outcome.fallback_candidate_id, selection.candidate_id),
        status=status,
        replay_request=CleanReplayRequest(
            candidate_id=selection.candidate_id,
            output_dir=outcome.run_dir / "production" / "unused-generated-replay",
            identity=identity,
            artifacts=ReplayArtifacts(
                source=selection.source_snapshot,
                config=recipe_reference,
                features=preprocessing,
                checkpoint=model,
                validation_predictions=selection.validation_prediction,
            ),
            environment=environment,
            equality=ReplayEquality.EXACT,
            training_replay="fresh official-train subprocess; exact checkpoint-byte identity",
        ),
        backend=build_replay_backend(recipe, cancel_event=cancel_event),
        bundle_metadata=_bundle_metadata(
            candidate_id=selection.candidate_id,
            lineage=(outcome.fallback_candidate_id, selection.candidate_id),
            status=status,
            metrics=metrics,
            identity=identity,
            environment=environment,
            request=request,
            outcome=outcome,
            seeds=(0, 1, 2),
            generated=True,
            confirmation=confirmation,
            campaign_wall_seconds=campaign_wall_seconds,
            launch_count=launch_count,
        ),
        report_context=_report_context(
            candidate_id=selection.candidate_id,
            parent_id=outcome.fallback_candidate_id,
            run_dir=outcome.run_dir,
            campaign_id=outcome.campaign_id,
            artifact_store=artifact_store,
            metrics=metrics,
            qualification=qualification,
            outcome=outcome,
            generated=True,
            failures=report_failures,
            confirmation=confirmation,
            campaign_wall_seconds=campaign_wall_seconds,
            launch_count=launch_count,
        ),
        is_official_fallback=False,
    )


def _fallback_replay_candidate(
    *,
    qualification: OfficialFMQualificationEvidence,
    context: _TrustedReplayContext,
    request: CampaignCreateRequest,
    environment: Mapping[str, object],
    artifact_store: ArtifactStore,
    outcome: FullCampaignOutcome,
    starter_dir: Path,
    report_failures: Sequence[ProductionFailureEvidence],
    campaign_wall_seconds: float,
    launch_count: int,
    cancel_event: threading.Event | None,
) -> FinalizationCandidate:
    fallback = qualification.fallback
    verify_starter_kit(starter_dir)
    source = artifact_store.put_directory(starter_dir, kind=ArtifactKind.SOURCE)
    encoding = artifact_store.put_file(fallback.encoding_path, kind=ArtifactKind.INPUT)
    checkpoint = artifact_store.put_file(
        fallback.checkpoint_path,
        kind=ArtifactKind.CHECKPOINT,
    )
    reference = artifact_store.put_file(
        fallback.validation_predictions_path,
        kind=ArtifactKind.PREDICTION,
    )
    if (
        encoding.sha256 != fallback.encoding_file_sha256
        or checkpoint.sha256 != fallback.checkpoint_file_sha256
        or reference.sha256 != fallback.validation_predictions_file_sha256
    ):
        raise ProductionFinalizationError("official fallback files changed after qualification")
    member = _official_member(
        fallback,
        starter_manifest_digest=qualification.starter_manifest_digest,
    )
    recipe = OfficialFMReplayRecipe(
        source_artifact_sha256=source.sha256,
        feature_artifact_sha256=encoding.sha256,
        checkpoint_artifact_sha256=checkpoint.sha256,
        data_sha256=context.dataset.digest,
        validation_inputs_digest=context.capabilities.validation_inputs.digest,
        final_inputs_digest=context.capabilities.final_inputs.digest,
        fm_member=member,
    )
    recipe_reference = _recipe_ref(artifact_store, recipe)
    reference_scores = _load_npy(reference, artifact_store, rows=context.dataset.valid.row_count)
    metrics = context.score(reference_scores)
    qualified_metrics = fallback.metrics.manifest()
    if any(
        not math.isclose(
            _metric(metrics[name], name),
            _metric(qualified_metrics[name], name),
            rel_tol=0.0,
            abs_tol=2e-7,
        )
        for name in ("GAUC", "nDCG@5", "primary")
    ):
        raise ProductionFinalizationError("official fallback metrics changed on protected replay")
    identity = FrozenReplayIdentity(
        source_sha256=source.sha256,
        config_sha256=recipe_reference.sha256,
        features_sha256=encoding.sha256,
        checkpoint_sha256=checkpoint.sha256,
        validation_prediction_artifact_sha256=reference.sha256,
        validation_prediction_digest=prediction_digest(reference_scores),
        data_sha256=context.dataset.digest,
        environment_sha256=request.environment_digest,
    )
    status = FinalStatus.BASELINE_REPRODUCED
    candidate_id = outcome.fallback_candidate_id
    return FinalizationCandidate(
        candidate_id=candidate_id,
        lineage=(candidate_id,),
        status=status,
        replay_request=CleanReplayRequest(
            candidate_id=candidate_id,
            output_dir=outcome.run_dir / "production" / "unused-fallback-replay",
            identity=identity,
            artifacts=ReplayArtifacts(
                source=source,
                config=recipe_reference,
                features=encoding,
                checkpoint=checkpoint,
                validation_predictions=reference,
            ),
            environment=environment,
            equality=ReplayEquality.EXACT,
            training_replay="immutable qualified official-FM seed-4 checkpoint replay",
        ),
        backend=build_replay_backend(recipe, cancel_event=cancel_event),
        bundle_metadata=_bundle_metadata(
            candidate_id=candidate_id,
            lineage=(candidate_id,),
            status=status,
            metrics=metrics,
            identity=identity,
            environment=environment,
            request=request,
            outcome=outcome,
            seeds=(4,),
            generated=False,
            confirmation=None,
            campaign_wall_seconds=campaign_wall_seconds,
            launch_count=launch_count,
        ),
        report_context=_report_context(
            candidate_id=candidate_id,
            parent_id=candidate_id,
            run_dir=outcome.run_dir,
            campaign_id=outcome.campaign_id,
            artifact_store=artifact_store,
            metrics=metrics,
            qualification=qualification,
            outcome=outcome,
            generated=False,
            failures=report_failures,
            confirmation=None,
            campaign_wall_seconds=campaign_wall_seconds,
            launch_count=launch_count,
        ),
        is_official_fallback=True,
    )


def _baseline_mean(qualification: OfficialFMQualificationEvidence) -> MetricEvidence:
    manifest = _strict_json(
        qualification.root / "manifest.json",
        maximum=_MAX_MANIFEST_BYTES,
        location="verified qualification manifest",
    )
    if manifest.get("digest") != qualification.manifest_digest:
        raise ProductionFinalizationError("qualification baseline manifest identity changed")
    fm = manifest.get("fm")
    summary = fm.get("five_seed_mean") if isinstance(fm, dict) else None
    if not isinstance(summary, dict) or set(summary) != {"GAUC", "nDCG@5", "primary"}:
        raise ProductionFinalizationError("qualification five-seed mean is missing")
    gauc = _metric(summary["GAUC"], "five-seed FM GAUC")
    ndcg = _metric(summary["nDCG@5"], "five-seed FM nDCG@5")
    primary = _metric(summary["primary"], "five-seed FM primary")
    if not math.isclose(primary, (gauc + ndcg) / 2.0, rel_tol=0.0, abs_tol=2e-7):
        raise ProductionFinalizationError("five-seed FM primary is not the organizer mean")
    return MetricEvidence(
        label="official FM five-seed mean",
        tier="public validation qualification",
        gauc=float(gauc),
        ndcg_at_5=float(ndcg),
        primary=float(primary),
        seeds=(0, 1, 2, 3, 4),
        note="Replayable organizer-compatible baseline.",
    )


def _selection_report_language(
    *,
    generated: bool,
    status: FinalStatus,
) -> tuple[str, tuple[str, ...]]:
    """Return status-derived judge wording without upgrading validation evidence."""

    if generated:
        if status is FinalStatus.MATERIALLY_CONFIRMED:
            return (
                "Highest materially confirmed replayable generated lineage; fall back to the "
                "official FM on any training, replay, serialization, checker, or bundle failure.",
                (),
            )
        if status is FinalStatus.VALIDATION_IMPROVED:
            return (
                "Highest replayable generated lineage with a positive matched-seed "
                "public-validation delta. It is materially unconfirmed because the strict "
                ">0.002 mean-primary target was not cleared; the official FM remains retained "
                "in its lineage as the immutable fallback.",
                (
                    "The generated selection has a positive matched-seed public-validation "
                    "delta but is materially unconfirmed; the strict >0.002 mean-primary "
                    "target was not cleared.",
                ),
            )
        raise ProductionFinalizationError(
            "a generated report must be validation_improved or materially_confirmed"
        )
    if status is not FinalStatus.BASELINE_REPRODUCED:
        raise ProductionFinalizationError("an official-FM report must be baseline_reproduced")
    return (
        "Immutable official FM seed 4 was the best remaining fully replayable incumbent; "
        "it is the protected baseline fallback and is not agent-generated.",
        (),
    )


@dataclass(frozen=True, slots=True)
class _JudgeProgressFacts:
    provider_usage: str
    portfolio_count: int
    portfolio_cap: int | None
    portfolio_cap_reason: str
    advanced_branch_disposition: str
    research_progress: str = (
        "Research admission stage counts are unavailable in this historical bundle."
    )
    research_outcome: str = "Research outcome details are unavailable in this historical bundle."
    research_rejections: tuple[str, ...] = ()


_RESEARCH_STAGE_COUNT_FIELDS: Final = (
    "branches_attempted",
    "proposal_responses_accepted",
    "implementation_responses_accepted",
    "repair_responses_accepted",
    "branches_rejected_pre_execution",
    "candidates_admitted",
    "training_started",
    "inner_evaluations_completed",
    "outer_evaluations_completed",
)


def _research_progress_summary(science: Mapping[str, object]) -> str:
    raw = science.get("research_stage_counts")
    if raw is None:
        return "Research admission stage counts are unavailable in this historical bundle."
    if not isinstance(raw, Mapping) or set(raw) != set(_RESEARCH_STAGE_COUNT_FIELDS):
        raise ProductionFinalizationError("research stage counts are malformed")
    counts: dict[str, int] = {}
    for name in _RESEARCH_STAGE_COUNT_FIELDS:
        value = raw[name]
        if type(value) is not int or value < 0:
            raise ProductionFinalizationError("research stage counts are malformed")
        counts[name] = value
    if not (
        counts["proposal_responses_accepted"] <= counts["branches_attempted"]
        and counts["implementation_responses_accepted"] <= counts["proposal_responses_accepted"]
        and counts["branches_rejected_pre_execution"] <= counts["branches_attempted"]
        and counts["candidates_admitted"] <= counts["branches_attempted"]
        and counts["inner_evaluations_completed"] <= counts["training_started"]
        and counts["outer_evaluations_completed"] <= counts["training_started"]
        and counts["inner_evaluations_completed"] + counts["outer_evaluations_completed"]
        <= counts["training_started"]
    ):
        raise ProductionFinalizationError("research stage counts are inconsistent")
    return (
        f"Research admission: branches attempted={counts['branches_attempted']}; "
        "proposal responses accepted="
        f"{counts['proposal_responses_accepted']}; implementation responses accepted="
        f"{counts['implementation_responses_accepted']}; repair responses accepted="
        f"{counts['repair_responses_accepted']}; rejected pre-execution="
        f"{counts['branches_rejected_pre_execution']}; candidates admitted="
        f"{counts['candidates_admitted']}; training started={counts['training_started']}; "
        f"inner evaluations completed={counts['inner_evaluations_completed']}; "
        f"outer evaluations completed={counts['outer_evaluations_completed']}."
    )


def _research_outcome_summary(
    *,
    lineage: Mapping[str, object],
    science: Mapping[str, object],
    cap_reason: str,
) -> str:
    if cap_reason == "configured_provider_unavailable":
        return "Research did not start because the configured provider was unavailable."
    if cap_reason == "runtime_provider_unavailable":
        return "Research started, but the provider failed at runtime after durable attempts."
    admission_closed = science.get("admission_closed", lineage.get("admission_closed"))
    if admission_closed is not None and type(admission_closed) is not bool:
        raise ProductionFinalizationError("research admission closure evidence is malformed")
    if admission_closed:
        reason = science.get("reason", lineage.get("reason", cap_reason))
        if type(reason) is not str or not reason or reason != cap_reason:
            raise ProductionFinalizationError("research admission closure evidence is inconsistent")
        return (
            "Research admission closed before a candidate was admitted; "
            f"controller reason={reason}."
        )
    return f"Research portfolio completed with controller reason={cap_reason}."


def _bounded_rejection_text(value: object, location: str, *, maximum: int) -> str:
    text = _text(value, location)
    if len(text) > maximum:
        raise ProductionFinalizationError(f"{location} exceeds its supported bound")
    return text


def _research_rejection_lines(science: Mapping[str, object]) -> tuple[str, ...]:
    raw = science.get("research_rejection_summary")
    if raw is None:
        return ()
    expected_fields = {
        "branches_rejected_pre_execution",
        "root_counts",
        "terminal_counts",
        "examples",
        "counts_truncated",
        "examples_truncated",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ProductionFinalizationError("research rejection summary is malformed")
    rejected = _bounded_int(
        raw["branches_rejected_pre_execution"],
        "research rejected branch count",
        maximum=1_000_000,
    )
    stage_counts = science.get("research_stage_counts")
    if isinstance(stage_counts, Mapping) and (
        stage_counts.get("branches_rejected_pre_execution") != rejected
    ):
        raise ProductionFinalizationError(
            "research rejection summary differs from research stage counts"
        )
    if type(raw["counts_truncated"]) is not bool or type(raw["examples_truncated"]) is not bool:
        raise ProductionFinalizationError("research rejection truncation evidence is malformed")

    count_fields = {"fingerprint", "stage", "category", "code", "subject", "count"}
    rendered_counts: dict[str, list[str]] = {"root": [], "terminal": []}
    known_fingerprints: dict[str, set[str]] = {"root": set(), "terminal": set()}
    for role, field_name in (("root", "root_counts"), ("terminal", "terminal_counts")):
        entries = raw[field_name]
        if not isinstance(entries, list) or len(entries) > 8:
            raise ProductionFinalizationError("research rejection counts exceed their bound")
        ordering: list[tuple[int, str]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) != count_fields:
                raise ProductionFinalizationError("research rejection count is malformed")
            fingerprint = _bounded_rejection_text(
                entry["fingerprint"],
                f"research {role} rejection {index} fingerprint",
                maximum=512,
            )
            _digest(fingerprint, f"research {role} rejection {index} fingerprint")
            stage = _bounded_rejection_text(
                entry["stage"], f"research {role} rejection {index} stage", maximum=128
            )
            category = _bounded_rejection_text(
                entry["category"],
                f"research {role} rejection {index} category",
                maximum=128,
            )
            code = _bounded_rejection_text(
                entry["code"], f"research {role} rejection {index} code", maximum=128
            )
            subject = _bounded_rejection_text(
                entry["subject"],
                f"research {role} rejection {index} subject",
                maximum=256,
            )
            count = _bounded_int(
                entry["count"],
                f"research {role} rejection {index} count",
                maximum=1_000_000,
            )
            if count == 0:
                raise ProductionFinalizationError("research rejection count must be positive")
            if fingerprint in known_fingerprints[role]:
                raise ProductionFinalizationError("research rejection fingerprint is duplicated")
            known_fingerprints[role].add(fingerprint)
            ordering.append((count, fingerprint))
            rendered_counts[role].append(
                f"{stage}/{category}/{code}/{subject} [{fingerprint}] x{count}"
            )
        if ordering != sorted(ordering, key=lambda item: (-item[0], item[1])):
            raise ProductionFinalizationError("research rejection counts are not canonical")

    examples = raw["examples"]
    if not isinstance(examples, list) or len(examples) > 6:
        raise ProductionFinalizationError("research rejection examples exceed their bound")
    lines: list[str] = []
    for role, title in (("root", "roots"), ("terminal", "terminals")):
        values = rendered_counts[role]
        if values:
            truncation = (
                " Top counts only; additional counts were truncated."
                if raw["counts_truncated"]
                else ""
            )
            lines.append(f"Research rejection {title}: {'; '.join(values)}.{truncation}")
    example_fields = {
        "scientific_iteration",
        "candidate_id",
        "proposal_family",
        "proposal_signature",
        "role",
        "fingerprint",
        "diagnostic",
    }
    for index, entry in enumerate(examples):
        if not isinstance(entry, Mapping) or set(entry) != example_fields:
            raise ProductionFinalizationError("research rejection example is malformed")
        iteration = _bounded_int(
            entry["scientific_iteration"],
            f"research rejection example {index} iteration",
            maximum=1_000_000,
        )
        if iteration == 0:
            raise ProductionFinalizationError("research rejection iteration must be positive")
        candidate_id = _bounded_rejection_text(
            entry["candidate_id"],
            f"research rejection example {index} candidate",
            maximum=256,
        )
        _bounded_rejection_text(
            entry["proposal_family"],
            f"research rejection example {index} proposal family",
            maximum=256,
        )
        proposal_signature = entry["proposal_signature"]
        if proposal_signature is not None:
            _digest(
                proposal_signature,
                f"research rejection example {index} proposal signature",
            )
        role = entry["role"]
        if role not in {"root", "terminal"}:
            raise ProductionFinalizationError("research rejection example role is malformed")
        fingerprint = _bounded_rejection_text(
            entry["fingerprint"],
            f"research rejection example {index} fingerprint",
            maximum=512,
        )
        _digest(fingerprint, f"research rejection example {index} fingerprint")
        if fingerprint not in known_fingerprints[cast(str, role)]:
            raise ProductionFinalizationError(
                "research rejection example lacks a retained fingerprint count"
            )
        diagnostic = _bounded_rejection_text(
            entry["diagnostic"],
            f"research rejection example {index} diagnostic",
            maximum=2_048,
        )
        # Iteration and candidate are part of the rendered line because two distinct rejections
        # can otherwise collapse into byte-identical text -- the deterministic proposal-family
        # circuit breaker refuses the same family with the same fingerprint and the same
        # diagnostic on every attempt -- and `FinalReportContext` requires unique failure lines.
        lines.append(
            f"Research rejection example ({role}) at iteration {iteration} "
            f"for {candidate_id}: {fingerprint}; {diagnostic}"
        )
    if raw["examples_truncated"]:
        lines.append("Additional research rejection examples were retained in the durable ledger.")
    return tuple(lines)


def _bundle_known_limitations(
    *,
    generated: bool,
    status: FinalStatus,
    judge_facts: _JudgeProgressFacts,
) -> tuple[str, ...]:
    return (
        "Hidden-test improvement is unverified until organizer scoring.",
        "campaign_totals.elapsed_seconds and the report resource wall time are sampled "
        "before clean replay and bundle publication; the signed terminal outcome at "
        "production/finalization/outcome.json is the inclusive production resource receipt.",
        judge_facts.advanced_branch_disposition,
        *_selection_report_language(generated=generated, status=status)[1],
    )


def _judge_progress_facts(outcome: FullCampaignOutcome) -> _JudgeProgressFacts:
    progress = FullCampaignProgressLedger(
        outcome.run_dir / "production" / "progress",
        create=False,
    )
    checkpoints = progress.checkpoints()
    if (
        len(checkpoints) < 2
        or checkpoints[-1].stage is not FullCampaignStage.FINALIZATION_REQUIRED
        or checkpoints[-1].request_digest != outcome.request_digest
        or checkpoints[-2].digest != outcome.progress_predecessor_digest
    ):
        raise ProductionFinalizationError("judge report progress differs from research outcome")
    by_stage = {checkpoint.stage: checkpoint for checkpoint in checkpoints}
    if len(by_stage) != len(checkpoints):
        raise ProductionFinalizationError("judge report progress contains duplicate stages")
    lineage = by_stage[FullCampaignStage.LINEAGE_READY].evidence
    science = by_stage[FullCampaignStage.SCIENCE_COMPLETE].evidence
    reflected = by_stage[FullCampaignStage.REFLECTED].evidence

    operations: list[str] = []
    provider = "none"
    live_provider_used = False
    lineage_manifest = lineage.get("lineage")
    if lineage_manifest is not None:
        if not isinstance(lineage_manifest, Mapping):
            raise ProductionFinalizationError("judge report lineage manifest is malformed")
        provider = _text(lineage_manifest.get("provider"), "lineage provider")
        live = lineage_manifest.get("live_provider_used")
        calls = lineage_manifest.get("model_calls")
        if type(live) is not bool or not isinstance(calls, list):
            raise ProductionFinalizationError("judge report provider call evidence is malformed")
        live_provider_used = live
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                raise ProductionFinalizationError("judge report provider call is malformed")
            operation = _text(call.get("operation"), f"provider call {index} operation")
            _digest(call.get("request_digest"), f"provider call {index} request")
            _digest(call.get("response_digest"), f"provider call {index} response")
            operations.append(operation.upper())
    elif isinstance(lineage.get("provider_diagnostic"), Mapping):
        provider_diagnostic = cast(Mapping[str, object], lineage["provider_diagnostic"])
        provider = (
            "runtime_provider_failure"
            if "attempts" in provider_diagnostic
            else "configured_provider_unavailable"
        )

    reflection_request = reflected.get("reflection_request_digest")
    reflection_response = reflected.get("reflection_response_digest")
    if reflection_request is not None or reflection_response is not None:
        _digest(reflection_request, "reflection request")
        _digest(reflection_response, "reflection response")
        if outcome.reflection_request_digest != reflection_request or (
            outcome.reflection_response_digest != reflection_response
        ):
            raise ProductionFinalizationError("reflection call differs from retained outcome")
        operations.append("REFLECT")
    operation_summary = "+".join(operations) if operations else "none"
    usage = science.get("provider_usage")
    if live_provider_used or isinstance(usage, Mapping):

        def valid_context_limits(value: object) -> bool:
            if value is None:
                return True
            if not isinstance(value, Mapping) or set(value) != {
                "context_length",
                "max_completion_tokens",
                "source",
            }:
                return False
            context_length = value["context_length"]
            maximum = value["max_completion_tokens"]
            source = value["source"]
            return (
                type(context_length) is int
                and 1 <= context_length <= 10_000_000
                and type(maximum) is int
                and 1 <= maximum <= context_length
                and type(source) is str
                and 1 <= len(source) <= 128
            )

        legacy_usage_fields = {
            "model",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "unaccounted_attempts",
            "transcript_count",
            "provider_wall_seconds",
        }
        chain_usage_fields = {
            "base_url",
            "active_slot",
            "failover_count",
            "failover_events",
            "provider_chain",
        }
        retry_usage_fields = {"retry_wait_seconds"}
        usage_fields = frozenset(usage) if isinstance(usage, Mapping) else frozenset()
        usage_fields_without_limits = usage_fields - {"context_limits"}
        if (
            not isinstance(usage, Mapping)
            or usage_fields_without_limits
            not in {
                frozenset(legacy_usage_fields),
                frozenset(legacy_usage_fields | retry_usage_fields),
                frozenset(legacy_usage_fields | chain_usage_fields),
                frozenset(legacy_usage_fields | chain_usage_fields | retry_usage_fields),
            }
            or (
                "context_limits" in usage_fields
                and not valid_context_limits(usage["context_limits"])
            )
        ):
            raise ProductionFinalizationError("live provider usage evidence is malformed")
        integer_fields = legacy_usage_fields - {
            "model",
            "estimated_cost_usd",
            "provider_wall_seconds",
        }
        if any(type(usage[name]) is not int or usage[name] < 0 for name in integer_fields):
            raise ProductionFinalizationError("live provider token counts are malformed")
        model = _text(usage["model"], "live provider model")
        estimated_cost = _text(usage["estimated_cost_usd"], "estimated provider cost")
        provider_wall = usage["provider_wall_seconds"]
        if (
            isinstance(provider_wall, bool)
            or not isinstance(provider_wall, (int, float))
            or not math.isfinite(float(provider_wall))
            or provider_wall < 0
        ):
            raise ProductionFinalizationError("live provider wall time is malformed")
        call_count = cast(int, usage["transcript_count"])
        retry_wait = usage.get("retry_wait_seconds", 0.0)
        if (
            isinstance(retry_wait, bool)
            or not isinstance(retry_wait, (int, float))
            or not math.isfinite(float(retry_wait))
            or retry_wait < 0
        ):
            raise ProductionFinalizationError("provider retry wait is malformed")
        provider_route = ""
        if chain_usage_fields.issubset(usage_fields):
            base_url = _text(usage["base_url"], "active provider base URL")
            active_slot = _text(usage["active_slot"], "active provider slot")
            failover_count = usage["failover_count"]
            failover_events = usage["failover_events"]
            provider_chain = usage["provider_chain"]
            if (
                active_slot not in {"main", "fallback"}
                or type(failover_count) is not int
                or failover_count < 0
                or not isinstance(failover_events, list)
                or len(failover_events) != failover_count
                or not isinstance(provider_chain, list)
                or not 1 <= len(provider_chain) <= 2
            ):
                raise ProductionFinalizationError("provider-chain usage evidence is malformed")
            for event in failover_events:
                if not isinstance(event, Mapping) or set(event) != {
                    "operation",
                    "from_slot",
                    "to_slot",
                    "failure",
                }:
                    raise ProductionFinalizationError("provider failover event is malformed")
                failure = event["failure"]
                if (
                    _text(event["operation"], "provider failover operation")
                    not in {"propose", "implement", "repair", "reflect"}
                    or event["from_slot"] != "main"
                    or event["to_slot"] != "fallback"
                    or not isinstance(failure, Mapping)
                    or set(failure) != {"slot", "code", "operation", "attempts", "status_code"}
                    or failure["slot"] != "main"
                    or type(failure["attempts"]) is not int
                    or failure["attempts"] < 0
                ):
                    raise ProductionFinalizationError("provider failover event is malformed")
            chain_integer_fields = {
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
                "unaccounted_attempts",
                "transcript_count",
            }
            expected_chain_fields = chain_integer_fields | {
                "slot",
                "model",
                "base_url",
                "credential_env",
                "estimated_cost_usd",
            }
            if "retry_wait_seconds" in usage_fields:
                expected_chain_fields.add("retry_wait_seconds")
            chain_totals = {name: 0 for name in chain_integer_fields}
            chain_retry_wait = 0.0
            slots: list[str] = []
            for index, profile in enumerate(provider_chain):
                if not isinstance(profile, Mapping) or set(profile) not in {
                    frozenset(expected_chain_fields),
                    frozenset(expected_chain_fields | {"context_limits"}),
                }:
                    raise ProductionFinalizationError(
                        "provider-chain profile usage evidence is malformed"
                    )
                if "context_limits" in profile and not valid_context_limits(
                    profile["context_limits"]
                ):
                    raise ProductionFinalizationError(
                        "provider-chain profile context limits are malformed"
                    )
                slot = _text(profile["slot"], f"provider-chain profile {index} slot")
                _text(profile["model"], f"provider-chain profile {index} model")
                _text(profile["base_url"], f"provider-chain profile {index} base URL")
                _text(
                    profile["credential_env"],
                    f"provider-chain profile {index} credential environment",
                )
                _text(
                    profile["estimated_cost_usd"],
                    f"provider-chain profile {index} estimated cost",
                )
                if slot not in {"main", "fallback"} or slot in slots:
                    raise ProductionFinalizationError("provider-chain profile slots are malformed")
                slots.append(slot)
                for name in chain_integer_fields:
                    value = profile[name]
                    if type(value) is not int or value < 0:
                        raise ProductionFinalizationError(
                            "provider-chain profile token counts are malformed"
                        )
                    chain_totals[name] += value
                if "retry_wait_seconds" in usage_fields:
                    profile_retry_wait = profile["retry_wait_seconds"]
                    if (
                        isinstance(profile_retry_wait, bool)
                        or not isinstance(profile_retry_wait, (int, float))
                        or not math.isfinite(float(profile_retry_wait))
                        or profile_retry_wait < 0
                    ):
                        raise ProductionFinalizationError("provider-chain retry wait is malformed")
                    chain_retry_wait += float(profile_retry_wait)
            if slots[0] != "main" or (len(slots) == 2 and slots != ["main", "fallback"]):
                raise ProductionFinalizationError("provider-chain profile order is malformed")
            for name in chain_integer_fields:
                if chain_totals[name] != usage[name]:
                    raise ProductionFinalizationError(
                        "provider-chain totals differ from aggregate provider usage"
                    )
            if "retry_wait_seconds" in usage_fields and not math.isclose(
                chain_retry_wait,
                float(retry_wait),
                rel_tol=0.0,
                abs_tol=0.000002 * len(provider_chain),
            ):
                raise ProductionFinalizationError(
                    "provider-chain retry wait differs from aggregate provider usage"
                )
            if active_slot not in slots:
                raise ProductionFinalizationError("active provider slot is absent from the chain")
            if failover_count > 1 or (active_slot == "main") != (failover_count == 0):
                raise ProductionFinalizationError("provider failover state is inconsistent")
            provider_route = (
                f"; active slot={active_slot}; failovers={failover_count}; "
                f"active base URL={base_url}"
            )
        provider_usage = (
            f"Research-model attempts={call_count}; provider={provider}; model={model}; "
            f"input tokens={usage['input_tokens']} (cached={usage['cached_input_tokens']}); "
            f"output tokens={usage['output_tokens']} "
            f"(reasoning={usage['reasoning_tokens']}); total tokens={usage['total_tokens']}; "
            f"estimated API cost USD={estimated_cost}; "
            f"provider wall seconds={provider_wall}; "
            f"retry wait seconds={retry_wait}; "
            f"unaccounted attempts={usage['unaccounted_attempts']}"
            f"{provider_route}; replay provider calls=0."
        )
    else:
        provider_usage = (
            f"Research-model calls={len(operations)} ({operation_summary}); "
            f"provider={provider}; network/API calls=0; replay provider calls=0."
        )

    portfolio_count = science.get("portfolio_count", reflected.get("portfolio_count"))
    cap_reason = science.get("portfolio_cap_reason", reflected.get("portfolio_cap_reason"))
    portfolio_cap = science.get("portfolio_cap")
    if (
        type(portfolio_count) is not int
        or portfolio_count < 0
        or type(cap_reason) is not str
        or (portfolio_cap is not None and (type(portfolio_cap) is not int or portfolio_cap < 0))
    ):
        raise ProductionFinalizationError("judge report portfolio closure is malformed")
    if reflected.get("portfolio_count", portfolio_count) != portfolio_count or (
        reflected.get("portfolio_cap_reason", cap_reason) != cap_reason
    ):
        raise ProductionFinalizationError("science and reflection portfolio evidence differ")
    advanced = (
        "The predeclared one-candidate high-value LambdaRank portfolio closed after its bounded "
        "branch. Advanced WP7 branches were not entered; this was a portfolio-cap decision, "
        "not an advanced-branch failure."
        if portfolio_count == 1
        and portfolio_cap == 1
        and cap_reason == "bounded_high_value_lambdarank_branch_prioritized"
        else (
            "Advanced WP7 branches were not entered; the retained portfolio count was "
            f"{portfolio_count} and the exact controller reason was {cap_reason}."
        )
    )
    return _JudgeProgressFacts(
        provider_usage=provider_usage,
        portfolio_count=portfolio_count,
        portfolio_cap=portfolio_cap,
        portfolio_cap_reason=cap_reason,
        advanced_branch_disposition=advanced,
        research_progress=_research_progress_summary(science),
        research_outcome=_research_outcome_summary(
            lineage=lineage,
            science=science,
            cap_reason=cap_reason,
        ),
        research_rejections=_research_rejection_lines(science),
    )


def _report_peak_rss_bytes(
    qualification: OfficialFMQualificationEvidence,
    confirmation: _GeneratedConfirmation | None,
) -> int:
    """Return a conservative model-training peak for the judge-readable report.

    Unlike selected-lineage provenance in the bundle metadata, the campaign-level report must
    conservatively cover both the official qualification and the retained generated runs.
    """

    resources = _qualification_seed_resources(qualification)
    qualification_peak = max(cast(int, item["peak_rss_bytes"]) for item in resources.values())
    if confirmation is None:
        return qualification_peak
    return max(qualification_peak, confirmation.retained_training_peak_rss_bytes)


def _retained_candidate_outcomes(
    run_dir: Path,
    outcome: FullCampaignOutcome,
    artifact_store: ArtifactStore,
) -> list[object]:
    """Read the retained scientific result and return its per-candidate outcome entries.

    The durable progress checkpoints carry the result artifact reference, so this recovers the
    document without threading the science checkpoint mapping through the report builder.  The
    stored digest is checked against the retained outcome so a mismatched document is ignored
    rather than reported.
    """

    progress_root = run_dir / "production" / "progress"
    if not progress_root.is_dir():
        return []
    for path in sorted(progress_root.iterdir(), reverse=True):
        if not path.is_file() or path.suffix != ".json":
            continue
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        evidence = checkpoint.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        reference_raw = evidence.get("scientific_result_artifact")
        if not isinstance(reference_raw, Mapping):
            continue
        if evidence.get("scientific_result_digest") != outcome.scientific_result_digest:
            continue
        reference = ArtifactRef.from_manifest(reference_raw)
        if reference.kind is not ArtifactKind.MANIFEST:
            return []
        payload = artifact_store.read_bytes(reference, max_bytes=_MAX_MANIFEST_BYTES)
        document = json.loads(payload.decode("ascii"))
        if not isinstance(document, dict):
            return []
        if document.get("digest") != outcome.scientific_result_digest:
            return []
        entries = document.get("candidate_outcomes")
        return list(entries) if isinstance(entries, list) else []
    return []


_INNER_TIER_RUN_COUNT: Final = 2


def _candidate_measured_primaries(
    run_dir: Path,
    outcome: FullCampaignOutcome,
    artifact_store: ArtifactStore,
) -> dict[str, tuple[float | None, float | None]]:
    """Recover each generated candidate's measured inner and outer primary.

    The trajectory table is the one place a judge can see that a candidate actually ran, and
    lineage records carry a proposal and a package but no score.  The scientific result names
    each candidate's run evidence digests and the durable record repository holds the exact
    metrics, so joining the two recovers the numbers without trusting the rounded public
    feedback.  Runs are recorded in tier order: the Fold B screen, the Fold A confirmation,
    then the matched outer seeds.

    This is presentation evidence.  Every inconsistency degrades to an unknown score rather than
    failing a finalization that has a valid bundle to publish.
    """

    if outcome.scientific_result_digest is None:
        return {}
    record_root = run_dir / "production" / "scientific-records"
    if not record_root.is_dir():
        return {}
    try:
        candidate_outcomes = _retained_candidate_outcomes(run_dir, outcome, artifact_store)
        if not candidate_outcomes:
            return {}
        repository = FileScientificRunEvidenceRepository(record_root)
        primary_by_run: dict[str, float] = {}
        for path in sorted(record_root.iterdir()):
            if not path.is_file() or path.suffix != ".json":
                continue
            record = repository.load(path.stem)
            if record is None or record.evidence.metrics is None:
                continue
            primary_by_run[record.evidence.digest] = float(record.evidence.metrics.primary)
    except Exception:
        return {}

    measured: dict[str, tuple[float | None, float | None]] = {}
    for candidate in candidate_outcomes:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = candidate.get("candidate_id")
        run_digests = candidate.get("run_digests")
        if not isinstance(candidate_id, str) or not isinstance(run_digests, list):
            continue
        scored = [
            primary_by_run[item]
            for item in run_digests
            if isinstance(item, str) and item in primary_by_run
        ]
        if not scored:
            continue
        inner = scored[:_INNER_TIER_RUN_COUNT]
        outer = scored[_INNER_TIER_RUN_COUNT:]
        measured[candidate_id] = (
            sum(inner) / len(inner) if inner else None,
            sum(outer) / len(outer) if outer else None,
        )
    return measured


def _candidate_outcome_status(outcome: FullCampaignOutcome) -> dict[str, str]:
    """Map candidate id to its measured campaign outcome for the trajectory table.

    The selection plan is the only outcome-bearing structure reachable here without
    reopening the campaign store, so an iteration the selector never promoted keeps the
    neutral default supplied by the collector."""

    selection = outcome.selection
    if selection is None:
        return {}
    return {selection.candidate_id: "promoted"}


def _with_selected_metric(
    narratives: tuple[ExperimentNarrative, ...],
    *,
    candidate_id: str,
    outer_primary: float,
) -> tuple[ExperimentNarrative, ...]:
    """Attach the measured outer primary to the narrative for the selected candidate.

    Recovered narratives are built from lineage records, which carry a proposal and a package
    but no score.  Without this the trajectory table renders a dash for the one iteration
    whose score is actually known.
    """

    return tuple(
        replace(item, outer_primary=outer_primary)
        if item.experiment_id == candidate_id and item.outer_primary is None
        else item
        for item in narratives
    )


def _report_context(
    *,
    candidate_id: str,
    parent_id: str,
    run_dir: Path,
    campaign_id: str,
    artifact_store: ArtifactStore,
    metrics: Mapping[str, object],
    qualification: OfficialFMQualificationEvidence,
    outcome: FullCampaignOutcome,
    generated: bool,
    failures: Sequence[ProductionFailureEvidence],
    confirmation: _GeneratedConfirmation | None,
    campaign_wall_seconds: float,
    launch_count: int,
) -> FinalReportContext:
    if generated != (confirmation is not None):
        raise ProductionFinalizationError("generated report confirmation presence is inconsistent")
    report_status = (
        confirmation.status if confirmation is not None else FinalStatus.BASELINE_REPRODUCED
    )
    selection_rationale, status_limitations = _selection_report_language(
        generated=generated,
        status=report_status,
    )
    judge_facts = _judge_progress_facts(outcome)
    selected = MetricEvidence(
        label=candidate_id,
        tier=("matched-seed outer validation" if generated else "qualified fallback"),
        gauc=_metric(metrics["GAUC"], "selected.GAUC"),
        ndcg_at_5=_metric(metrics["nDCG@5"], "selected.nDCG@5"),
        primary=_metric(metrics["primary"], "selected.primary"),
        seeds=(
            (outcome.selection.representative_seed,)
            if generated and outcome.selection is not None
            else (4,)
        ),
        note=(
            "Representative clean-replay seed; all matched-seed means and deltas follow below."
            if generated
            else "Immutable official fallback seed."
        ),
    )
    # The campaign durably records one lineage file per admitted iteration and one
    # hash-chained journal entry per rejected branch.  Prefer that real trajectory; the
    # literal below remains only for runs that never reached research at all.
    try:
        recovered = collect_iteration_narratives(
            run_dir,
            campaign_id=campaign_id,
            fallback_parent_id=parent_id,
            candidate_outcomes=_candidate_outcome_status(outcome),
            candidate_metrics=_candidate_measured_primaries(run_dir, outcome, artifact_store),
        )
    except Exception:
        # The recovered trajectory is presentation; the campaign store and the immutable
        # lineage files remain the evidence of record.  Degrade to the single narrative
        # below rather than aborting a finalization that has a valid bundle to publish.
        recovered = IterationEvidence((), ())
    experiment = ExperimentNarrative(
        iteration=1,
        experiment_id=candidate_id,
        parent_id=parent_id,
        hypothesis=(
            "Causal history features and LambdaRank can improve long-view impression ranking."
            if generated
            else "The immutable official FM remains a valid finalizable fallback."
        ),
        mechanism=(
            "Use train-frozen causal features, user-grouped LambdaRank, and a Fold-B-frozen "
            "rank fusion with a matched official FM."
            if generated
            else "Restore the hash-pinned organizer FM encoding and seed-4 checkpoint."
        ),
        material_changes=(
            (
                "Generated candidate.py is retained as material executable source.",
                "Fusion weights were selected only on train-derived Fold B.",
            )
            if generated
            else ("No generated challenger survived finalization; fallback stayed immutable.",)
        ),
        attributions=(
            "Organizer starter evaluator and baseline",
            "Repository-generated causal LambdaRank implementation" if generated else "Official FM",
        ),
        status=(confirmation.status.value if confirmation is not None else "baseline_reproduced"),
        outer_primary=selected.primary,
    )
    finalization_failure_lines = tuple(
        f"{item.candidate_id} failed {item.stage} ({item.exception_type}; "
        f"diagnostic SHA-256 {item.diagnostic_sha256}); incumbent protection remained active."
        for item in failures
    ) or ("No production finalization fallback was required.",)
    failure_lines = (
        judge_facts.research_outcome,
        judge_facts.research_progress,
        *judge_facts.research_rejections,
        *recovered.failure_lines,
        *finalization_failure_lines,
    )
    limitations = [
        "Hidden-test improvement is unverified until organizer scoring.",
        "The signed terminal production outcome reports one inclusive finalization resource "
        "receipt; the bundle's pre-publication report is not the terminal campaign total.",
        judge_facts.advanced_branch_disposition,
        *status_limitations,
    ]
    peak_rss = _report_peak_rss_bytes(qualification, confirmation)
    if confirmation is not None:
        fold_lines = tuple(
            "Fold {fold}: candidate primary={candidate}; parent primary={parent}; official-FM "
            "reference primary={reference}; delta to parent={delta_parent}; delta to reference="
            "{delta_reference}.".format(
                fold=row["fold"],
                candidate=cast(dict[str, float], row["candidate_metrics"])["primary"],
                parent=cast(dict[str, float], row["parent_metrics"])["primary"],
                reference=cast(dict[str, float], row["official_fm_reference_metrics"])["primary"],
                delta_parent=row["primary_delta_to_parent"],
                delta_reference=row["primary_delta_to_reference"],
            )
            for row in confirmation.inner_rows
        )
        inner_evidence = (
            *fold_lines,
            "Fold summary: mean primary delta to official-FM reference={mean}; worst reference "
            "fold delta={worst}; "
            "guards require mean > 0 and worst >= -0.002.".format(
                mean=confirmation.inner_summary["mean_primary_delta"],
                worst=confirmation.inner_summary["worst_primary_delta"],
            ),
        )
    else:
        inner_evidence = ("Official FM qualification and checkpoint replay passed.",)
    seed_confirmation = (
        confirmation.seed_report_lines()
        if confirmation is not None
        else (
            "Five official FM seeds and a clean seed-0 retrain were qualified; seed 4 is the "
            "immutable fallback.",
        )
    )
    return FinalReportContext(
        benchmark_contract={
            "dataset": "KuaiRand-Pure",
            "target": "long_view",
            "ranking_unit": "logged impressions grouped by user",
            "metrics": ["GAUC", "nDCG@5"],
            "primary": "mean(GAUC,nDCG@5)",
            "benchmark_digest": qualification.benchmark_digest,
        },
        baselines=(_baseline_mean(qualification),),
        selected=selected,
        experiments=(
            _with_selected_metric(
                recovered.narratives,
                candidate_id=candidate_id,
                outer_primary=selected.primary,
            )
            or (experiment,)
        ),
        inner_fold_evidence=inner_evidence,
        seed_confirmation=seed_confirmation,
        failures_and_recoveries=failure_lines,
        leakage_controls=(
            "Final-period outcomes were skipped by the canonical loader and no final target "
            "capability exists.",
            "Public-validation labels remained inside the protected scorer and were never "
            "provided to candidate training or inference.",
            "Generated replay ran in a fresh CPU subprocess with allowlisted recipe semantics.",
        ),
        test_evidence=(
            "Clean validation replay, high-precision CSV round-trip, exact within-user top-5, "
            "and protected metric parity are bundle gates.",
            "The untouched organizer checker runs with --check against a private outcome-masked "
            "final-period view.",
        ),
        selection_rationale=selection_rationale,
        resources=ResourceEvidence(
            wall_seconds=float(campaign_wall_seconds),
            peak_rss_bytes=peak_rss,
            launch_count=launch_count,
            intervention_count=outcome.manual_interventions,
            provider_usage=judge_facts.provider_usage,
            device_usage=("CPU official qualification, final training replay, and clean replay",),
        ),
        known_limitations=tuple(limitations),
    )


def _bundle_metadata(
    *,
    candidate_id: str,
    lineage: tuple[str, ...],
    status: FinalStatus,
    metrics: Mapping[str, object],
    identity: FrozenReplayIdentity,
    environment: Mapping[str, object],
    request: CampaignCreateRequest,
    outcome: FullCampaignOutcome,
    seeds: tuple[int, ...],
    generated: bool,
    confirmation: _GeneratedConfirmation | None,
    campaign_wall_seconds: float,
    launch_count: int,
) -> FinalBundleMetadata:
    if generated != (confirmation is not None):
        raise ProductionFinalizationError("bundle confirmation presence is inconsistent")
    if confirmation is not None and status is not confirmation.status:
        raise ProductionFinalizationError("bundle status differs from reconstructed confirmation")
    if confirmation is None and status is not FinalStatus.BASELINE_REPRODUCED:
        raise ProductionFinalizationError("fallback bundle must be baseline_reproduced")
    judge_facts = _judge_progress_facts(outcome)
    uv_lock_sha256 = _digest(environment.get("uv_lock_sha256"), "environment uv.lock")
    if environment_identity_digest(environment) != request.environment_digest:
        raise ProductionFinalizationError("bundle environment cannot reprove campaign identity")
    seed_summary: Mapping[str, object] = (
        confirmation.seed_summary(
            outcome.selection.representative_seed if outcome.selection is not None else seeds[0]
        )
        if confirmation is not None
        else {
            "schema_version": 1,
            "seeds": list(seeds),
            "representative_seed": seeds[0],
            "matched_confirmation": False,
            "derived_status": FinalStatus.BASELINE_REPRODUCED.value,
            "confirmation_is_controller_derived": True,
        }
    )
    inner_results: Sequence[Mapping[str, object]] = (
        confirmation.inner_rows
        if confirmation is not None
        else (
            {
                "fold": "official qualification",
                "weights_selected_on_public_validation": False,
                "evidence_receipt": outcome.fallback_receipt_digest,
            },
        )
    )
    return FinalBundleMetadata(
        benchmark_identity={
            "name": BENCHMARK_CONTRACT.task.dataset,
            "target": BENCHMARK_CONTRACT.task.target,
            "digest": request.benchmark_digest,
            "primary": "mean(GAUC,nDCG@5)",
        },
        starter_identity={"manifest_sha256": request.starter_manifest_digest},
        data_identity={
            "canonical_digest": identity.data_sha256,
            "final_outcomes_accessed": False,
            "final_outcomes_scored": False,
        },
        selected_experiment=candidate_id,
        lineage=lineage,
        status=status,
        validation_metrics={
            "GAUC": _metric(metrics["GAUC"], "GAUC"),
            "nDCG@5": _metric(metrics["nDCG@5"], "nDCG@5"),
            "primary": _metric(metrics["primary"], "primary"),
        },
        seed_summary=seed_summary,
        inner_fold_results=inner_results,
        scientific_artifact_hashes={
            "source": identity.source_sha256,
            "config": identity.config_sha256,
            "features": identity.features_sha256,
            "checkpoint": identity.checkpoint_sha256,
            "predictions": identity.validation_prediction_artifact_sha256,
        },
        environment_and_resource_usage={
            "environment_sha256": identity.environment_sha256,
            "runtime_identity": {
                "schema_version": 1,
                "project_source_digest": request.source_digest,
                "environment_digest": request.environment_digest,
                "uv_lock_sha256": uv_lock_sha256,
                "dependency_groups": list(_dependency_groups(environment)),
            },
            "device": "cpu",
            "locked_environment": True,
            "final_outcomes_accessed": False,
            "retained_training_peak_rss_bytes": (
                confirmation.retained_training_peak_rss_bytes if confirmation is not None else 0
            ),
            "seed_resource_deltas_in_seed_summary": confirmation is not None,
            "production_resource_receipt": {
                "schema_version": _FINALIZATION_RESOURCE_SCHEMA_VERSION,
                "terminal_receipt": "production/finalization/outcome.json",
                "aggregate_evidence_binding": "closed_bundle_manifest_sha256",
                "coverage": list(_FINALIZATION_COVERAGE),
            },
        },
        campaign_totals={
            "attempt_count": launch_count,
            "scientific_iteration_count": (
                count_recorded_iterations(outcome.run_dir)
                or (1 if outcome.scientific_result_digest else 0)
            ),
            "launch_count": launch_count,
            "elapsed_seconds": float(campaign_wall_seconds),
            "manual_intervention_count": outcome.manual_interventions,
            "portfolio_count": judge_facts.portfolio_count,
            "portfolio_cap": judge_facts.portfolio_cap,
            "portfolio_cap_reason": judge_facts.portfolio_cap_reason,
            "advanced_wp7_branches_entered": False,
        },
        known_limitations=_bundle_known_limitations(
            generated=generated,
            status=status,
            judge_facts=judge_facts,
        ),
        unresolved_organizer_questions=(),
    )


_EXPERIMENT_LEDGER_SCHEMA_VERSION: Final = 2
_EXPERIMENT_CSV_FIELDS: Final = (
    "record_type",
    "record_id",
    "candidate_id",
    "parent_id",
    "tier",
    "seed",
    "status",
    "decision",
    "primary",
    "wall_seconds",
    "peak_rss_bytes",
    "evidence_digest",
)


def _ledger_json_value(value: object, location: str) -> object:
    """Recursively normalize store projections without accepting opaque objects."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionFinalizationError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ProductionFinalizationError(f"{location} contains a non-text key")
            result[key] = _ledger_json_value(item, f"{location}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _ledger_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)
        ]
    raise ProductionFinalizationError(
        f"{location} contains unsupported evidence type {type(value).__name__}"
    )


def _ledger_row(
    record_type: str,
    record_id: str,
    evidence: Mapping[str, object],
    *,
    summary: Mapping[str, object] = MappingProxyType({}),
) -> dict[str, object]:
    normalized_evidence = _ledger_json_value(evidence, f"ledger {record_id} evidence")
    normalized_summary = _ledger_json_value(summary, f"ledger {record_id} summary")
    if not isinstance(normalized_evidence, dict) or not isinstance(normalized_summary, dict):
        raise ProductionFinalizationError("ledger rows require object evidence and summary")
    evidence_digest = hashlib.sha256(
        b"kuairand-judge-ledger-evidence-v2\0" + _canonical_json(normalized_evidence)
    ).hexdigest()
    return {
        "schema_version": _EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "record_type": _text(record_type, "ledger record_type"),
        "record_id": _text(record_id, "ledger record_id"),
        "evidence_digest": evidence_digest,
        "summary": normalized_summary,
        "evidence": normalized_evidence,
    }


def _artifact_links(
    store: CampaignStore,
    *,
    owner_type: str,
    owner_id: str,
) -> list[dict[str, object]]:
    return [
        {
            "role": role,
            "artifact": {
                "digest": artifact.digest,
                "kind": artifact.kind,
                "relative_path": artifact.relative_path,
                "size_bytes": artifact.size_bytes,
                "metadata": _ledger_json_value(
                    artifact.metadata,
                    f"{owner_type} {owner_id} artifact metadata",
                ),
            },
        }
        for role, artifact in store.artifacts_for(owner_type=owner_type, owner_id=owner_id)
    ]


def _scientific_result_document(
    *,
    artifact_store: ArtifactStore,
    checkpoint_evidence: Mapping[str, object],
    outcome: FullCampaignOutcome,
) -> dict[str, object] | None:
    if outcome.scientific_result_digest is None:
        if checkpoint_evidence.get("scientific_result_digest") is not None:
            raise ProductionFinalizationError(
                "scientific progress and retained outcome disagree about result absence"
            )
        return None
    reference_raw = checkpoint_evidence.get("scientific_result_artifact")
    if not isinstance(reference_raw, Mapping):
        raise ProductionFinalizationError("scientific result artifact reference is missing")
    try:
        reference = ArtifactRef.from_manifest(reference_raw)
        if reference.kind is not ArtifactKind.MANIFEST:
            raise ProductionFinalizationError("scientific result artifact has the wrong kind")
        payload = artifact_store.read_bytes(reference, max_bytes=_MAX_MANIFEST_BYTES)
        decoded = json.loads(payload.decode("ascii"))
    except ProductionFinalizationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProductionFinalizationError("scientific result artifact is malformed") from exc
    if not isinstance(decoded, dict) or _canonical_json(decoded) != payload:
        raise ProductionFinalizationError("scientific result artifact is not canonical JSON")
    if (
        decoded.get("digest") != outcome.scientific_result_digest
        or checkpoint_evidence.get("scientific_result_digest") != outcome.scientific_result_digest
    ):
        raise ProductionFinalizationError("scientific result identity differs from progress")
    return cast(dict[str, object], decoded)


def _candidate_run_ownership(
    candidate_outcomes: Sequence[object],
    *,
    selected_candidate_id: str,
) -> tuple[dict[str, str], set[str], set[str]]:
    """Attribute every retained scientific run to the candidate that produced it.

    A campaign evaluates several candidates, so the record repository holds runs from all of
    them. Returns the run-to-candidate map, every expected run digest, and the subset belonging
    to the selected candidate. Only that subset can be expected to carry the selection's own
    source and config identity.
    """

    owners: dict[str, str] = {}
    expected: set[str] = set()
    selected: set[str] = set()
    for index, candidate in enumerate(candidate_outcomes):
        if not isinstance(candidate, Mapping):
            raise ProductionFinalizationError("scientific candidate outcome is malformed")
        run_digests = candidate.get("run_digests")
        if not isinstance(run_digests, list):
            raise ProductionFinalizationError("scientific candidate run digests are malformed")
        owner = candidate.get("candidate_id")
        for run_index, digest in enumerate(run_digests):
            normalized = _digest(digest, f"candidate outcome {index} run {run_index}")
            expected.add(normalized)
            if isinstance(owner, str) and owner:
                owners[normalized] = owner
                if owner == selected_candidate_id:
                    selected.add(normalized)
    return owners, expected, selected


def _scientific_record_rows(
    *,
    run_dir: Path,
    request: CampaignCreateRequest,
    outcome: FullCampaignOutcome,
    artifact_store: ArtifactStore,
    scientific_result: Mapping[str, object],
) -> list[dict[str, object]]:
    selection = outcome.selection
    if selection is None:
        raise ProductionFinalizationError("scientific run export requires a retained selection")
    candidate_outcomes = scientific_result.get("candidate_outcomes")
    if not isinstance(candidate_outcomes, list):
        raise ProductionFinalizationError("scientific result candidate outcomes are missing")
    candidate_by_run, expected_run_digests, selected_run_digests = _candidate_run_ownership(
        candidate_outcomes, selected_candidate_id=selection.candidate_id
    )

    record_root = run_dir / "production" / "scientific-records"
    try:
        metadata = record_root.lstat()
    except OSError as exc:
        if expected_run_digests:
            raise ProductionFinalizationError("scientific record repository is missing") from exc
        return []
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionFinalizationError("scientific record repository is unsafe")
    repository = FileScientificRunEvidenceRepository(record_root)
    # The repository holds every scientific run this campaign performed, across *all* generated
    # candidates -- see the `retained_unreferenced_scientific_run` decision below, which exists
    # precisely for records outside the selected candidate's run set. Source-snapshot, source, and
    # config digests are per-candidate identities, so they are pinned to the selection only for the
    # records that actually belong to it; the environment digest is the one campaign-wide identity
    # and stays checked on every record.
    rows: list[dict[str, object]] = []
    observed_run_digests: set[str] = set()
    records_by_request: dict[str, object] = {}
    for path in sorted(record_root.iterdir(), key=lambda item: item.name):
        member = path.lstat()
        if stat.S_ISLNK(member.st_mode) or not stat.S_ISREG(member.st_mode):
            raise ProductionFinalizationError("scientific record repository member is unsafe")
        if path.suffix != ".json":
            raise ProductionFinalizationError("scientific record repository has an unknown member")
        request_digest = _digest(path.stem, "scientific record filename")
        record = repository.load(request_digest)
        if record is None:
            raise ProductionFinalizationError("scientific record disappeared during ledger export")
        if record.evidence.digest in observed_run_digests:
            raise ProductionFinalizationError("scientific result has duplicate run evidence")
        observed_run_digests.add(record.evidence.digest)
        records_by_request[request_digest] = record
        # The environment is campaign-wide, so every retained run must share it.
        if record.evidence.identities.environment_digest != request.environment_digest:
            raise ProductionFinalizationError("scientific record environment identity changed")
        # Source and config identity are per candidate. Only the selected candidate's own runs
        # can match the selection, and requiring every run to match it made finalization
        # impossible for any campaign that promoted a generated candidate after evaluating more
        # than one. Ownership comes from the campaign's own candidate_outcomes rather than from
        # digest equality, so a record cannot exempt itself by differing.
        if record.evidence.digest in selected_run_digests and (
            record.source_snapshot_digest != selection.source_snapshot.sha256
            or record.evidence.identities.source_digest != selection.source_digest
            or record.evidence.identities.config_digest != selection.config_digest
        ):
            raise ProductionFinalizationError(
                "selected candidate scientific record source or config identity changed"
            )
        for reference in (
            record.checkpoint,
            record.raw_prediction,
            record.replay_prediction,
            record.scored_prediction,
        ):
            artifact_store.verify(reference)
        for execution_artifacts in (
            record.train_artifacts,
            record.prediction_artifacts,
            record.replay_artifacts,
        ):
            for _, reference in execution_artifacts.entries:
                artifact_store.verify(reference)
        metrics = record.evidence.metrics
        referenced = record.evidence.digest in expected_run_digests
        rows.append(
            _ledger_row(
                "scientific_run",
                request_digest,
                record.manifest() | {"digest": record.digest},
                summary={
                    "candidate_id": candidate_by_run.get(
                        record.evidence.digest, selection.candidate_id
                    ),
                    "tier": "bound_in_request_digest_and_execution_records",
                    "seed": next(
                        (
                            matched.seed
                            for matched in selection.matched_seeds
                            if matched.scientific_request_digest == request_digest
                        ),
                        None,
                    ),
                    "status": "succeeded" if metrics is not None else "failed",
                    "decision": (
                        "replay_verified_scientific_result_run"
                        if referenced and record.evidence.replay_verified
                        else "retained_unreferenced_scientific_run"
                        if record.evidence.replay_verified
                        else "rejected"
                    ),
                    "primary": None if metrics is None else metrics.primary,
                    "wall_seconds": record.evidence.resources.wall_seconds,
                    "peak_rss_bytes": record.evidence.resources.peak_rss_bytes,
                },
            )
        )
    if not expected_run_digests.issubset(observed_run_digests):
        raise ProductionFinalizationError(
            "scientific result and durable run-record repository are incomplete"
        )
    for matched in selection.matched_seeds:
        retained = records_by_request.get(matched.scientific_request_digest)
        if (
            retained is None
            or getattr(retained, "digest", None) != matched.scientific_record_digest
        ):
            raise ProductionFinalizationError(
                f"matched seed {matched.seed} scientific record identity changed"
            )
    return rows


def _write_experiment_ledgers(
    *,
    run_dir: Path,
    request: CampaignCreateRequest,
    artifact_store: ArtifactStore,
    store: CampaignStore,
    outcome: FullCampaignOutcome,
    failures: Sequence[ProductionFailureEvidence],
    selected_metadata: FinalBundleMetadata | None,
) -> tuple[Path, Path]:
    """Export exact controller evidence, not a hand-authored experiment summary."""

    progress = FullCampaignProgressLedger(
        run_dir / "production" / "progress",
        create=False,
    )
    checkpoints = progress.checkpoints()
    if (
        not checkpoints
        or checkpoints[-1].stage is not FullCampaignStage.FINALIZATION_REQUIRED
        or checkpoints[-1].request_digest != request.digest
        or len(checkpoints) < 2
        or checkpoints[-2].digest != outcome.progress_predecessor_digest
    ):
        raise ProductionFinalizationError("judge ledger progress chain differs from outcome")
    by_stage = {checkpoint.stage: checkpoint for checkpoint in checkpoints}
    if len(by_stage) != len(checkpoints):
        raise ProductionFinalizationError("judge ledger progress contains duplicate stages")

    rows: list[dict[str, object]] = [
        _ledger_row(
            "campaign_progress",
            f"{checkpoint.sequence:02d}:{checkpoint.stage.value}",
            checkpoint.manifest(),
            summary={
                "status": checkpoint.stage.value,
                "decision": "durable_hash_chained_stage",
            },
        )
        for checkpoint in checkpoints
    ]
    identity = store.identity()
    if (
        identity.campaign_id != request.campaign_id
        or identity.config_digest != request.config.digest
        or identity.benchmark_digest != request.benchmark_digest
        or identity.starter_manifest_digest != request.starter_manifest_digest
        or identity.dataset_manifest_digest != request.dataset_manifest_digest
        or identity.source_digest != request.source_digest
        or identity.environment_digest != request.environment_digest
    ):
        raise ProductionFinalizationError("judge ledger store identity differs from request")
    health = store.health()
    rows.insert(
        0,
        _ledger_row(
            "campaign_identity",
            identity.campaign_id,
            {
                "campaign_id": identity.campaign_id,
                "config_digest": identity.config_digest,
                "benchmark_digest": identity.benchmark_digest,
                "starter_manifest_digest": identity.starter_manifest_digest,
                "dataset_manifest_digest": identity.dataset_manifest_digest,
                "source_digest": identity.source_digest,
                "environment_digest": identity.environment_digest,
                "hard_deadline_utc": identity.hard_deadline_utc,
                "max_launches": identity.max_launches,
                "outer_query_limit": identity.outer_query_limit,
                "created_at": identity.created_at,
                "store_health": {
                    "journal_mode": health.journal_mode,
                    "foreign_keys": health.foreign_keys,
                    "synchronous": health.synchronous,
                    "user_version": health.user_version,
                    "quick_check": health.quick_check,
                    "schema_digest": health.schema_digest,
                    "catalog_digest": health.catalog_digest,
                },
            },
            summary={
                "status": "identity_verified",
                "decision": "locked_source_config_environment_and_store",
            },
        ),
    )
    snapshot = store.snapshot()
    rows.append(
        _ledger_row(
            "campaign_budget_and_convergence",
            request.campaign_id,
            {
                "campaign_id": snapshot.campaign_id,
                "revision": snapshot.revision,
                "status": snapshot.status,
                "phase": snapshot.phase,
                "max_launches": snapshot.max_launches,
                "launches_used": snapshot.launches_used,
                "launches_remaining": snapshot.launches_remaining,
                "outer_query_limit": snapshot.outer_query_limit,
                "outer_queries_used": snapshot.outer_queries_used,
                "outer_queries_remaining": snapshot.outer_queries_remaining,
                "convergence": dict(snapshot.convergence_state),
                "manual_interventions": outcome.manual_interventions,
            },
            summary={
                "status": snapshot.status,
                "decision": "authoritative_remaining_budget_and_convergence",
            },
        )
    )
    rows.append(
        _ledger_row(
            "fallback_incumbent",
            outcome.fallback_candidate_id,
            {
                "candidate_id": outcome.fallback_candidate_id,
                "receipt_digest": outcome.fallback_receipt_digest,
                "qualification_manifest_digest": outcome.qualification_manifest_digest,
                "eligible": True,
                "replay_verified": True,
            },
            summary={
                "candidate_id": outcome.fallback_candidate_id,
                "status": "eligible_fallback",
                "decision": "retained_immutable_official_fm",
            },
        )
    )

    science_checkpoint = by_stage[FullCampaignStage.SCIENCE_COMPLETE]
    reflected_checkpoint = by_stage[FullCampaignStage.REFLECTED]
    scientific_result = _scientific_result_document(
        artifact_store=artifact_store,
        checkpoint_evidence=science_checkpoint.evidence,
        outcome=outcome,
    )
    if scientific_result is not None:
        if (
            scientific_result.get("convergence")
            != _ledger_json_value(snapshot.convergence_state, "store convergence")
            or reflected_checkpoint.evidence.get("scientific_result_digest")
            != outcome.scientific_result_digest
        ):
            raise ProductionFinalizationError(
                "scientific result convergence or reflection binding changed"
            )
        rows.append(
            _ledger_row(
                "scientific_result",
                cast(str, outcome.scientific_result_digest),
                scientific_result,
                summary={
                    "candidate_id": scientific_result.get("incumbent_candidate_id"),
                    "status": scientific_result.get("stop_reason"),
                    "decision": "trusted_scientific_campaign_closure",
                    "wall_seconds": scientific_result.get("elapsed_seconds"),
                },
            )
        )

    lineage_checkpoint = by_stage[FullCampaignStage.LINEAGE_READY]
    rows.append(
        _ledger_row(
            "generated_lineage",
            lineage_checkpoint.digest,
            {
                "progress_digest": lineage_checkpoint.digest,
                "evidence": dict(lineage_checkpoint.evidence),
                "selection_source_snapshot": (
                    None
                    if outcome.selection is None
                    else outcome.selection.source_snapshot.manifest()
                ),
            },
            summary={
                "candidate_id": (
                    None if outcome.selection is None else outcome.selection.candidate_id
                ),
                "parent_id": (
                    None if outcome.selection is None else outcome.selection.parent_source_digest
                ),
                "status": (
                    "material_generated_source"
                    if lineage_checkpoint.evidence.get("material_generated_source") is True
                    else "generated_branch_not_entered"
                ),
                "decision": "provider_request_response_source_diff_closure",
            },
        )
    )
    rows.append(
        _ledger_row(
            "reflection",
            reflected_checkpoint.digest,
            {
                "progress_digest": reflected_checkpoint.digest,
                "evidence": dict(reflected_checkpoint.evidence),
                "outcome_request_digest": outcome.reflection_request_digest,
                "outcome_response_digest": outcome.reflection_response_digest,
                "outcome_transcript": (
                    None
                    if outcome.reflection_transcript is None
                    else outcome.reflection_transcript.manifest()
                ),
            },
            summary={
                "candidate_id": reflected_checkpoint.evidence.get("selected_candidate_id"),
                "status": "reflected",
                "decision": "provider_request_response_transcript_bound",
            },
        )
    )

    portfolio_count = science_checkpoint.evidence.get(
        "portfolio_count",
        reflected_checkpoint.evidence.get("portfolio_count"),
    )
    cap_reason = science_checkpoint.evidence.get(
        "portfolio_cap_reason",
        reflected_checkpoint.evidence.get("portfolio_cap_reason"),
    )
    portfolio_cap = science_checkpoint.evidence.get("portfolio_cap")
    if type(portfolio_count) is not int or portfolio_count < 0 or type(cap_reason) is not str:
        raise ProductionFinalizationError("scientific portfolio closure is incomplete")
    if (
        reflected_checkpoint.evidence.get("portfolio_count", portfolio_count) != portfolio_count
        or reflected_checkpoint.evidence.get("portfolio_cap_reason", cap_reason) != cap_reason
    ):
        raise ProductionFinalizationError("science and reflection portfolio closure differ")
    rows.append(
        _ledger_row(
            "portfolio_closure",
            science_checkpoint.digest,
            {
                "portfolio_count": portfolio_count,
                "portfolio_cap": portfolio_cap,
                "portfolio_cap_reason": cap_reason,
                "declared_portfolio_cap_reached": science_checkpoint.evidence.get(
                    "declared_portfolio_cap_reached"
                ),
                "advanced_wp7_branches_entered": False,
                "advanced_wp7_disposition": (
                    "not_entered_because_bounded_high_value_one_lambdarank_portfolio_closed"
                ),
                "source_progress_digest": science_checkpoint.digest,
            },
            summary={
                "status": "bounded_portfolio_closed",
                "decision": cap_reason,
            },
        )
    )

    launches = store.launches()
    for launch in launches:
        rows.append(
            _ledger_row(
                "launch",
                launch.launch_id,
                {
                    "launch_id": launch.launch_id,
                    "launch_number": launch.launch_number,
                    "reservation_key": launch.reservation_key,
                    "category": launch.category,
                    "original_category": launch.original_category,
                    "purpose": launch.purpose,
                    "state": launch.state,
                    "charged": launch.charged,
                    "event_seq": launch.event_seq,
                    "experiment_id": launch.experiment_id,
                    "scientific_iteration": launch.scientific_iteration,
                    "seed": launch.seed,
                    "start_receipt_digest": launch.start_receipt_digest,
                },
                summary={
                    "candidate_id": launch.experiment_id,
                    "seed": launch.seed,
                    "status": launch.state,
                    "decision": launch.category,
                },
            )
        )

    executions = store.executions()
    linked_experiment_ids = {
        launch.experiment_id for launch in launches if launch.experiment_id is not None
    } | {execution.experiment_id for execution in executions if execution.experiment_id is not None}
    experiment_ids = sorted(
        {
            identifier
            for identifier in linked_experiment_ids
            if store.experiment(identifier) is not None
        }
    )
    for experiment_id in experiment_ids:
        experiment = store.experiment(experiment_id)
        if experiment is None:
            raise ProductionFinalizationError("research experiment disappeared during export")
        rows.append(
            _ledger_row(
                "research_experiment",
                experiment.experiment_id,
                {
                    "experiment_id": experiment.experiment_id,
                    "iteration_number": experiment.iteration_number,
                    "parent_experiment_id": experiment.parent_experiment_id,
                    "hypothesis": experiment.hypothesis,
                    "mechanism": experiment.mechanism,
                    "method_attribution": experiment.method_attribution,
                    "status": experiment.status,
                    "metadata": dict(experiment.metadata),
                    "created_at": experiment.created_at,
                    "artifacts": _artifact_links(
                        store,
                        owner_type="experiment",
                        owner_id=experiment.experiment_id,
                    ),
                },
                summary={
                    "candidate_id": outcome.selection.candidate_id
                    if outcome.selection is not None
                    else None,
                    "parent_id": experiment.parent_experiment_id,
                    "status": experiment.status,
                    "decision": "research_iteration_state",
                },
            )
        )
        proposal_id = experiment.metadata.get("proposal_id")
        if proposal_id is not None:
            proposal = store.proposal(_text(proposal_id, "experiment proposal_id"))
            if proposal is None or proposal.experiment_id != experiment.experiment_id:
                raise ProductionFinalizationError("research proposal link is incomplete")
            rows.append(
                _ledger_row(
                    "research_proposal",
                    proposal.proposal_id,
                    {
                        "proposal_id": proposal.proposal_id,
                        "experiment_id": proposal.experiment_id,
                        "request_digest": proposal.request_digest,
                        "response_digest": proposal.response_digest,
                        "provider": proposal.provider,
                        "metadata": dict(proposal.metadata),
                        "created_at": proposal.created_at,
                        "artifacts": _artifact_links(
                            store,
                            owner_type="proposal",
                            owner_id=proposal.proposal_id,
                        ),
                    },
                    summary={
                        "candidate_id": outcome.selection.candidate_id
                        if outcome.selection is not None
                        else None,
                        "status": "recorded",
                        "decision": proposal.provider,
                    },
                )
            )

    for execution in executions:
        evidence = {
            name: getattr(execution, name)
            for name in (
                "execution_id",
                "experiment_id",
                "launch_id",
                "launch_number",
                "launch_category",
                "original_launch_category",
                "kind",
                "tier",
                "seed",
                "command",
                "status",
                "nonce",
                "source_digest",
                "config_digest",
                "capability_digest",
                "environment_digest",
                "data_digest",
                "checkpoint_digest",
                "process_record_digest",
                "process_record",
                "process_id",
                "process_create_time",
                "process_group_id",
                "process_command_digest",
                "process_environment_digest",
                "result_digest",
                "created_at",
                "updated_at",
                "started_at",
                "finished_at",
                "metadata",
            )
        }
        evidence["artifacts"] = _artifact_links(
            store,
            owner_type="execution",
            owner_id=execution.execution_id,
        )
        process = execution.process_record
        rows.append(
            _ledger_row(
                "execution",
                execution.execution_id,
                evidence,
                summary={
                    "candidate_id": execution.experiment_id,
                    "tier": execution.tier,
                    "seed": execution.seed,
                    "status": execution.status,
                    "decision": execution.launch_category or execution.kind,
                    "wall_seconds": (
                        process.get("wall_seconds") if isinstance(process, Mapping) else None
                    ),
                    "peak_rss_bytes": (
                        process.get("peak_tree_rss_bytes") if isinstance(process, Mapping) else None
                    ),
                },
            )
        )

    for reallocation in store.reallocations():
        rows.append(
            _ledger_row(
                "budget_reallocation",
                reallocation.reallocation_id,
                {
                    "reallocation_id": reallocation.reallocation_id,
                    "from_category": reallocation.from_category,
                    "to_category": reallocation.to_category,
                    "launch_count": reallocation.launch_count,
                    "reason": reallocation.reason,
                    "metadata": dict(reallocation.metadata),
                },
                summary={
                    "status": "recorded",
                    "decision": reallocation.reason,
                },
            )
        )

    if outcome.selection is not None:
        if scientific_result is None:
            raise ProductionFinalizationError("generated selection lacks scientific result")
        rows.extend(
            _scientific_record_rows(
                run_dir=run_dir,
                request=request,
                outcome=outcome,
                artifact_store=artifact_store,
                scientific_result=scientific_result,
            )
        )
        rows.append(
            _ledger_row(
                "finalization_selection",
                outcome.selection.candidate_id,
                {
                    "selection": outcome.selection.manifest(),
                    "derived_final_status": (
                        selected_metadata.status.value if selected_metadata is not None else None
                    ),
                    "matched_seed_confirmation": (
                        dict(selected_metadata.seed_summary)
                        if selected_metadata is not None
                        else None
                    ),
                    "inner_fold_results": (
                        [dict(item) for item in selected_metadata.inner_fold_results]
                        if selected_metadata is not None
                        else None
                    ),
                },
                summary={
                    "candidate_id": outcome.selection.candidate_id,
                    "parent_id": outcome.selection.parent_source_digest,
                    "seed": outcome.selection.representative_seed,
                    "status": (
                        selected_metadata.status.value
                        if selected_metadata is not None
                        else "generated_branch_failed_finalization"
                    ),
                    "decision": "selected_for_finalization",
                    "primary": (
                        None
                        if selected_metadata is None
                        else selected_metadata.validation_metrics.get("primary")
                    ),
                },
            )
        )

    for index, failure in enumerate(failures, start=1):
        rows.append(
            _ledger_row(
                "finalization_failure",
                f"{index:02d}:{failure.candidate_id}",
                failure.manifest(),
                summary={
                    "candidate_id": failure.candidate_id,
                    "status": "failed_closed",
                    "decision": failure.stage,
                },
            )
        )

    support = run_dir / "production" / "finalization-support"
    support.mkdir(parents=True, exist_ok=True, mode=0o700)
    jsonl_payload = b"".join(_canonical_json(row, newline=True) for row in rows)
    jsonl = support / "experiments.jsonl"
    csv_path = support / "experiments.csv"
    csv_rows: list[dict[str, object]] = []
    for row in rows:
        summary = cast(dict[str, object], row["summary"])
        csv_rows.append(
            {
                "record_type": row["record_type"],
                "record_id": row["record_id"],
                "candidate_id": summary.get("candidate_id", ""),
                "parent_id": summary.get("parent_id", ""),
                "tier": summary.get("tier", ""),
                "seed": summary.get("seed", ""),
                "status": summary.get("status", ""),
                "decision": summary.get("decision", ""),
                "primary": summary.get("primary", ""),
                "wall_seconds": summary.get("wall_seconds", ""),
                "peak_rss_bytes": summary.get("peak_rss_bytes", ""),
                "evidence_digest": row["evidence_digest"],
            }
        )
    temporary = support / ".experiments.csv.staging"
    if not jsonl.exists():
        _write_exclusive(jsonl, jsonl_payload)
    elif (
        _read_regular(jsonl, maximum=_MAX_LEDGER_BYTES, location="experiments JSONL")
        != jsonl_payload
    ):
        raise ProductionFinalizationError("retained experiments JSONL contradicts finalization")

    def write_csv(path: Path, *, exclusive: bool) -> None:
        with path.open("x" if exclusive else "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=_EXPERIMENT_CSV_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(csv_rows)
            handle.flush()
            os.fsync(handle.fileno())

    if not csv_path.exists():
        write_csv(temporary, exclusive=True)
        os.chmod(temporary, 0o444, follow_symlinks=False)
        os.rename(temporary, csv_path)
        _fsync_directory(support)
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix="ledger-", dir=support)
        os.close(descriptor)
        temporary_compare = Path(temporary_name)
        try:
            write_csv(temporary_compare, exclusive=False)
            if csv_path.read_bytes() != temporary_compare.read_bytes():
                raise ProductionFinalizationError(
                    "retained experiments CSV contradicts finalization"
                )
        finally:
            temporary_compare.unlink(missing_ok=True)
    return jsonl, csv_path


def _phase_for_finalization(store: CampaignStore) -> None:
    snapshot = store.snapshot()
    if snapshot.status == CampaignState.FINALIZATION_REQUIRED.value:
        store.set_campaign_phase(
            phase="finalizing",
            status=CampaignState.FINALIZING.value,
            expected_revision=snapshot.revision,
            reason="begin provider-free production finalization",
        )
        return
    if snapshot.status in {CampaignState.FINALIZING.value, CampaignState.COMPLETED.value}:
        return
    raise ProductionFinalizationError(
        f"campaign status {snapshot.status!r} is not finalization-ready"
    )


def _complete_campaign(
    *,
    store: CampaignStore,
    request: CampaignCreateRequest,
    outcome: FullCampaignOutcome,
    result: FinalizationResult | None,
    bundle: _VerifiedBundle,
) -> int:
    selection = outcome.selection
    selection_raw = bundle.manifest.get("selection")
    metrics_raw = bundle.manifest.get("validation")
    if not isinstance(selection_raw, dict) or not isinstance(metrics_raw, dict):
        raise ProductionFinalizationError("bundle selection or validation evidence is malformed")
    derived_status = _derive_bundle_status(bundle.manifest)
    if result is not None and result.selected_status is not derived_status:
        raise ProductionFinalizationError("finalization result status differs from bundle evidence")
    selected_id = _text(selection_raw.get("selected_experiment"), "selected experiment")
    if selected_id not in {outcome.fallback_candidate_id, getattr(selection, "candidate_id", None)}:
        raise ProductionFinalizationError("bundle selected candidate differs from research outcome")
    if selected_id != outcome.fallback_candidate_id:
        if selection is None:
            raise ProductionFinalizationError("bundle selected an absent generated candidate")
        scientific = bundle.manifest.get("scientific_artifact_hashes")
        validation = metrics_raw.get("metrics")
        seed_summary = metrics_raw.get("seed_summary")
        if (
            not isinstance(scientific, dict)
            or not isinstance(validation, dict)
            or not isinstance(seed_summary, dict)
        ):
            raise ProductionFinalizationError("bundle generated incumbent closure is malformed")
        candidate_mean = _manifest_metrics(
            seed_summary.get("candidate_mean"),
            "bundle generated candidate mean",
        )
        checkpoint_digest = _digest(scientific.get("checkpoint"), "bundle checkpoint")
        closure = bundle.manifest_sha256
        confirmation_receipt = hashlib.sha256(
            b"kuairand-final-confirmation-v1\0" + _canonical_json(seed_summary)
        ).hexdigest()
        incumbent = store.record_incumbent(
            incumbent_id=f"final:{selected_id}:{bundle.manifest_sha256[:16]}",
            eligibility=f"{derived_status.value}_clean_replay",
            source_digest=selection.source_digest,
            checkpoint_digest=checkpoint_digest,
            artifact_closure_digest=closure,
            replay_verified=True,
            is_fallback=False,
            expected_revision=store.snapshot().revision,
            reason=(
                f"{derived_status.value} confirmation, clean replay, organizer check, and "
                "closed final bundle passed"
            ),
            experiment_id=selection.experiment_id,
            outer_primary_mean=candidate_mean["primary"],
            artifacts=(
                (
                    "closed_bundle_manifest",
                    ArtifactSpec(
                        digest=bundle.manifest_sha256,
                        kind="closed_final_bundle_manifest",
                        relative_path=((_FINAL_BUNDLE_RELATIVE_PATH / "manifest.json").as_posix()),
                        size_bytes=(bundle.root / "manifest.json").stat().st_size,
                        metadata={
                            "selected_candidate_id": selected_id,
                            "submission_sha256": _digest(
                                cast(dict[str, object], scientific).get("submission"),
                                "bundle submission",
                            ),
                        },
                    ),
                ),
            ),
            metadata={
                "research_outcome_digest": outcome.digest,
                "bundle_manifest_sha256": bundle.manifest_sha256,
                "derived_final_status": derived_status.value,
                "confirmation_receipt_digest": confirmation_receipt,
                "finalization_result_available": result is not None,
            },
        )
        current = store.current_incumbent()
        if current != incumbent or current is None:
            raise ProductionFinalizationError("generated incumbent retry changed durable evidence")
    else:
        if derived_status is not FinalStatus.BASELINE_REPRODUCED:
            raise ProductionFinalizationError("official fallback bundle has a generated status")
        fallback_incumbent = store.current_incumbent()
        if (
            fallback_incumbent is None
            or not fallback_incumbent.is_fallback
            or not fallback_incumbent.replay_verified
        ):
            raise ProductionFinalizationError("official fallback incumbent is no longer eligible")
    snapshot = store.snapshot()
    if snapshot.status == CampaignState.FINALIZING.value:
        snapshot = store.set_campaign_phase(
            phase="completed",
            status=CampaignState.COMPLETED.value,
            expected_revision=snapshot.revision,
            reason="closed final bundle published and incumbent replay verified",
        )
    elif snapshot.status != CampaignState.COMPLETED.value:
        raise ProductionFinalizationError("campaign left the finalizing state unexpectedly")
    return snapshot.revision


def _outcome_from_bundle(
    *,
    run_dir: Path,
    request: CampaignCreateRequest,
    research: FullCampaignOutcome,
    bundle: _VerifiedBundle,
    revision: int,
    failures: Sequence[ProductionFailureEvidence],
    training_replay: Mapping[str, object],
    resource_evidence: Mapping[str, object],
) -> ProductionFinalizationOutcome:
    selection = bundle.manifest.get("selection")
    scientific = bundle.manifest.get("scientific_artifact_hashes")
    if not isinstance(selection, dict) or not isinstance(scientific, dict):
        raise ProductionFinalizationError("closed bundle selection evidence is malformed")
    status = _derive_bundle_status(bundle.manifest)
    if request.campaign_id != research.campaign_id:
        raise ProductionFinalizationError("campaign and research outcome identity differ")
    if (
        resource_evidence.get("hard_wall_seconds") != request.config.benchmark.wall_clock_seconds
        or resource_evidence.get("finalization_reserve_seconds")
        != request.config.runner.finalization_reserve_seconds
    ):
        raise ProductionFinalizationError("production resource evidence changed configured limits")
    return ProductionFinalizationOutcome(
        run_dir=run_dir,
        campaign_id=request.campaign_id,
        research_outcome_digest=research.digest,
        selected_candidate_id=_text(selection.get("selected_experiment"), "selected experiment"),
        selected_status=status,
        fallback_count=len(failures),
        failures=tuple(failures),
        training_replay=training_replay,
        resource_evidence=resource_evidence,
        bundle_root=bundle.root,
        bundle_manifest_sha256=bundle.manifest_sha256,
        submission_sha256=_digest(scientific.get("submission"), "bundle submission"),
        replay_evidence_sha256=sha256_file(bundle.root / "replay" / "evidence.json"),
        organizer_verification_sha256=sha256_file(bundle.root / "verification.json"),
        campaign_revision=revision,
    )


def _verify_generated_final_training_binding(
    *,
    bundle: _VerifiedBundle,
    research: FullCampaignOutcome,
    outcome: ProductionFinalizationOutcome,
) -> None:
    selection = research.selection
    if selection is None:
        raise ProductionFinalizationError("generated production outcome lost its selection")
    scientific = bundle.manifest.get("scientific_artifact_hashes")
    final_training = outcome.resource_evidence.get("final_training")
    recipe = (
        load_replay_recipe(
            bundle.root / "config" / "artifact",
            expected_sha256=_digest(
                scientific.get("config") if isinstance(scientific, dict) else None,
                "bundle generated config",
            ),
        )
        if isinstance(scientific, dict)
        else None
    )
    if (
        not isinstance(scientific, dict)
        or not isinstance(final_training, Mapping)
        or not isinstance(recipe, GeneratedLambdaRankReplayRecipe)
        or final_training.get("evidence_digest") != recipe.tree_checkpoint_sha256
        or final_training.get("evidence_digest") != selection.tree_checkpoint.sha256
        or outcome.training_replay.get("checkpoint_sha256") != recipe.tree_checkpoint_sha256
    ):
        raise ProductionFinalizationError(
            "final-training resources are not bound to the selected tree checkpoint"
        )


def _load_production_outcome(
    path: Path,
    *,
    run_dir: Path,
    request: CampaignCreateRequest,
    research: FullCampaignOutcome,
    store: CampaignStore,
) -> ProductionFinalizationOutcome:
    payload = _read_regular(
        path,
        maximum=_MAX_MANIFEST_BYTES,
        location="production finalization outcome",
    )
    outcome = ProductionFinalizationOutcome.from_bytes(payload)
    if (
        outcome.run_dir != run_dir
        or outcome.campaign_id != request.campaign_id
        or outcome.research_outcome_digest != research.digest
        or outcome.bundle_root != (run_dir / _FINAL_BUNDLE_RELATIVE_PATH).resolve(strict=True)
    ):
        raise ProductionFinalizationError("production outcome differs from campaign identity")
    if (
        outcome.resource_evidence.get("hard_wall_seconds")
        != request.config.benchmark.wall_clock_seconds
        or outcome.resource_evidence.get("finalization_reserve_seconds")
        != request.config.runner.finalization_reserve_seconds
    ):
        raise ProductionFinalizationError("production resource evidence changed configured limits")
    bundle = _verify_closed_bundle(outcome.bundle_root)
    if (
        bundle.manifest_sha256 != outcome.bundle_manifest_sha256
        or sha256_file(bundle.root / "submission.csv") != outcome.submission_sha256
        or sha256_file(bundle.root / "replay" / "evidence.json") != outcome.replay_evidence_sha256
        or sha256_file(bundle.root / "verification.json") != outcome.organizer_verification_sha256
    ):
        raise ProductionFinalizationError("production outcome bundle closure changed")
    if outcome.selected_candidate_id != research.fallback_candidate_id:
        selection = research.selection
        if selection is None:
            raise ProductionFinalizationError("generated production outcome lost its selection")
        _verify_charged_final_training_terminal(store, selection)
        _verify_generated_final_training_binding(
            bundle=bundle,
            research=research,
            outcome=outcome,
        )
    snapshot = store.snapshot()
    if (
        snapshot.status != CampaignState.COMPLETED.value
        or snapshot.revision != outcome.campaign_revision
    ):
        raise ProductionFinalizationError("production outcome and campaign completion differ")
    return outcome


def _finalize_provider_free_campaign_impl(
    run_dir: Path,
    *,
    project_root: Path,
    engine: CampaignEngine,
    cancel_event: _FinalizationDeadlineEvent,
    resource_tracker: _FinalizationResourceTracker,
) -> ProductionFinalizationOutcome:
    """Finalize one retained campaign exactly once, with immutable official-FM fallback.

    The function is restart-safe.  Exact terminal retries verify and return the retained outcome;
    a crash after closed-bundle publication but before store completion reopens the bundle,
    promotes the same incumbent if needed, completes the campaign, and writes the terminal receipt.
    """

    if not isinstance(run_dir, Path) or not isinstance(project_root, Path):
        raise ProductionFinalizationError("run_dir and project_root must be pathlib.Path values")
    cancellation = _cancel_event(cancel_event)
    _raise_if_cancelled(cancellation, "campaign finalization")
    selected_engine = engine
    try:
        if stat.S_ISLNK(run_dir.lstat().st_mode) or stat.S_ISLNK(project_root.lstat().st_mode):
            raise ProductionFinalizationError(
                "campaign and project root paths must not be symlinks"
            )
        run = run_dir.resolve(strict=True)
        root = project_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionFinalizationError("campaign or project root is unavailable") from exc
    if run.is_symlink() or not run.is_dir() or root.is_symlink() or not root.is_dir():
        raise ProductionFinalizationError("campaign and project roots must be real directories")
    request, artifact_store, research = _load_research_outcome(
        run,
        engine=selected_engine,
    )
    data_dir = _resolve_project_path(root, request.config.benchmark.data_dir, "data directory")
    starter_dir = _resolve_project_path(
        root,
        request.config.benchmark.starter_dir,
        "starter directory",
    )
    context = _trusted_replay_context(
        data_dir=data_dir,
        starter_dir=starter_dir,
        expected_data_sha256=request.dataset_manifest_digest,
    )
    environment = _environment(request, root)
    qualification = _qualification(request, context)
    outcome_path = run / _OUTCOME_RELATIVE_PATH
    destination = run / _FINAL_BUNDLE_RELATIVE_PATH
    with CampaignStore.open(
        run / CAMPAIGN_DATABASE_NAME,
        campaign_id=request.campaign_id,
    ) as store:
        if outcome_path.exists():
            return _load_production_outcome(
                outcome_path,
                run_dir=run,
                request=request,
                research=research,
                store=store,
            )
        _phase_for_finalization(store)
        if store.snapshot().status == CampaignState.COMPLETED.value and not destination.exists():
            raise ProductionFinalizationError(
                "completed campaign is missing both terminal outcome and closed bundle"
            )
        failures: list[ProductionFailureEvidence] = []
        training_replay: Mapping[str, object] = MappingProxyType(
            {
                "required": research.selection is not None,
                "completed": False,
                "mode": "not_applicable_official_fm_fallback",
                "training_rows_include_public_validation": False,
                "device": "cpu",
            }
        )
        if destination.exists():
            bundle = _close_bundle_directories(destination)
            selected = bundle.manifest.get("selection")
            if not isinstance(selected, dict):
                raise ProductionFinalizationError("recovered bundle selection is missing")
            selected_id = _text(
                selected.get("selected_experiment"),
                "recovered selected experiment",
            )
            if selected_id != research.fallback_candidate_id:
                if research.selection is None:
                    raise ProductionFinalizationError(
                        "recovered generated bundle has no retained selection"
                    )
                _, training_replay = _fresh_training_replay(
                    run_dir=run,
                    request=request,
                    selection=research.selection,
                    artifact_store=artifact_store,
                    engine=selected_engine,
                    store=store,
                    cancel_event=cancellation,
                )
                _verify_charged_final_training_terminal(store, research.selection)
            else:
                training_replay = MappingProxyType(
                    {
                        "required": research.selection is not None,
                        "completed": False,
                        "mode": "recovered_published_official_fm_fallback",
                        "training_rows_include_public_validation": False,
                        "device": "cpu",
                    }
                )
            _raise_if_cancelled(cancellation, "recovered bundle campaign completion")
            final_observation = selected_engine.observe_deadline(run)
            if final_observation.hard_expired:
                raise ProductionFinalizationError(
                    "campaign hard deadline expired before recovered bundle completion"
                )
            resource_evidence = resource_tracker.finish(
                campaign_elapsed_seconds=final_observation.elapsed_seconds,
                hard_wall_seconds=request.config.benchmark.wall_clock_seconds,
                finalization_reserve_seconds=request.config.runner.finalization_reserve_seconds,
                rows=context.dataset.final.row_count,
                bundle_manifest_sha256=bundle.manifest_sha256,
                training_replay=training_replay,
                selected_generated=selected_id != research.fallback_candidate_id,
            )
            revision = _complete_campaign(
                store=store,
                request=request,
                outcome=research,
                result=None,
                bundle=bundle,
            )
            recovered = _outcome_from_bundle(
                run_dir=run,
                request=request,
                research=research,
                bundle=bundle,
                revision=revision,
                failures=failures,
                training_replay=training_replay,
                resource_evidence=resource_evidence,
            )
            _write_exclusive(outcome_path, recovered.canonical_bytes)
            return recovered

        candidates: list[FinalizationCandidate] = []
        if research.selection is not None:
            try:
                checkpoint, training_replay = _fresh_training_replay(
                    run_dir=run,
                    request=request,
                    selection=research.selection,
                    artifact_store=artifact_store,
                    engine=selected_engine,
                    store=store,
                    cancel_event=cancellation,
                )
                _verify_charged_final_training_terminal(store, research.selection)
                _raise_if_cancelled(cancellation, "generated confirmation reconstruction")
                observation = selected_engine.observe_deadline(run)
                launches = store.snapshot().launches_used
                candidates.append(
                    _generated_replay_candidate(
                        selection=research.selection,
                        checkpoint=checkpoint,
                        qualification=qualification,
                        context=context,
                        request=request,
                        environment=environment,
                        artifact_store=artifact_store,
                        outcome=research,
                        report_failures=failures,
                        campaign_wall_seconds=observation.elapsed_seconds,
                        launch_count=launches,
                        cancel_event=cancellation,
                    )
                )
            except Exception as exc:
                _raise_if_cancelled(cancellation, "generated finalization fallback")
                failures.append(
                    _failure(
                        research.selection.candidate_id,
                        "fresh official-train replay",
                        exc,
                    )
                )
                training_replay = MappingProxyType(
                    {
                        "required": True,
                        "completed": False,
                        "mode": "failed_closed_to_official_fm",
                        "diagnostic_sha256": failures[-1].diagnostic_sha256,
                        "training_rows_include_public_validation": False,
                        "device": "cpu",
                    }
                )
        observation = selected_engine.observe_deadline(run)
        launches = store.snapshot().launches_used
        # The guarantee-of-last-resort path.  The generated builder above is wrapped; this
        # one must be too, or a defect shared by both takes down the fallback itself.
        try:
            fallback = _fallback_replay_candidate(
                qualification=qualification,
                context=context,
                request=request,
                environment=environment,
                artifact_store=artifact_store,
                outcome=research,
                starter_dir=starter_dir,
                report_failures=failures,
                campaign_wall_seconds=observation.elapsed_seconds,
                launch_count=launches,
                cancel_event=cancellation,
            )
        except ProductionFinalizationError:
            # Already the right type, and cancellation is signalled this way too.
            raise
        except Exception as exc:
            raise ProductionFinalizationError(
                "official FM fallback candidate could not be prepared"
            ) from exc
        candidates.append(fallback)
        experiments_jsonl, experiments_csv = _write_experiment_ledgers(
            run_dir=run,
            request=request,
            artifact_store=artifact_store,
            store=store,
            outcome=research,
            failures=failures,
            selected_metadata=(
                candidates[0].bundle_metadata
                if research.selection is not None
                and candidates[0].candidate_id == research.selection.candidate_id
                else None
            ),
        )
        organizer_scratch = run / "production" / "organizer-check-scratch"
        organizer_scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
        _raise_if_cancelled(cancellation, "clean replay and publication")
        finalization = run_finalization(
            FinalizationRequest(
                destination=destination,
                artifact_store=artifact_store,
                capabilities=context.capabilities,
                protected_metric_evaluator=context.score,
                data_dir=data_dir,
                starter_dir=starter_dir,
                experiments_jsonl=experiments_jsonl,
                experiments_csv=experiments_csv,
                candidates=tuple(candidates),
                scratch_dir=organizer_scratch,
            ),
            cancel_event=cancellation,
        )
        failures.extend(
            ProductionFailureEvidence(
                candidate_id=item.candidate_id,
                stage=item.stage,
                exception_type=item.exception_type,
                diagnostic_sha256=item.diagnostic_sha256,
            )
            for item in finalization.failures
        )
        bundle = _close_bundle_directories(finalization.bundle.root)
        if (
            bundle.manifest_sha256 != finalization.bundle.manifest_sha256
            or sha256_file(bundle.root / "submission.csv") != finalization.bundle.submission_sha256
        ):
            raise ProductionFinalizationError("published final bundle identity changed")
        _raise_if_cancelled(cancellation, "campaign completion")
        final_observation = selected_engine.observe_deadline(run)
        if final_observation.hard_expired:
            raise ProductionFinalizationError(
                "campaign hard deadline expired before durable campaign completion"
            )
        selected_generated = finalization.selected_candidate_id != research.fallback_candidate_id
        resource_evidence = resource_tracker.finish(
            campaign_elapsed_seconds=final_observation.elapsed_seconds,
            hard_wall_seconds=request.config.benchmark.wall_clock_seconds,
            finalization_reserve_seconds=request.config.runner.finalization_reserve_seconds,
            rows=context.dataset.final.row_count,
            bundle_manifest_sha256=bundle.manifest_sha256,
            training_replay=training_replay,
            selected_generated=selected_generated,
        )
        revision = _complete_campaign(
            store=store,
            request=request,
            outcome=research,
            result=finalization,
            bundle=bundle,
        )
        completed = _outcome_from_bundle(
            run_dir=run,
            request=request,
            research=research,
            bundle=bundle,
            revision=revision,
            failures=failures,
            training_replay=training_replay,
            resource_evidence=resource_evidence,
        )
        _write_exclusive(outcome_path, completed.canonical_bytes)
        return completed


def finalize_provider_free_campaign(
    run_dir: Path,
    *,
    project_root: Path,
    engine: CampaignEngine | None = None,
    cancel_event: threading.Event | None = None,
) -> ProductionFinalizationOutcome:
    """Reconcile the immutable deadline, then run one deadline-aware finalization."""

    if not isinstance(run_dir, Path) or not isinstance(project_root, Path):
        raise ProductionFinalizationError("run_dir and project_root must be pathlib.Path values")
    signal_event = _cancel_event(cancel_event)
    _raise_if_cancelled(signal_event, "campaign finalization")
    selected_engine = CampaignEngine() if engine is None else engine
    try:
        if stat.S_ISLNK(run_dir.lstat().st_mode) or stat.S_ISLNK(project_root.lstat().st_mode):
            raise ProductionFinalizationError(
                "campaign and project root paths must not be symlinks"
            )
        run = run_dir.resolve(strict=True)
        root = project_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionFinalizationError("campaign or project root is unavailable") from exc
    if run.is_symlink() or not run.is_dir() or root.is_symlink() or not root.is_dir():
        raise ProductionFinalizationError("campaign and project roots must be real directories")

    # Resume is deliberately unconditional: it reconciles abandoned children and durably appends
    # a fresh deadline observation before data/context or finalization work can begin.
    lifecycle = selected_engine.resume(run)
    if lifecycle.status == CampaignState.INCOMPLETE.value:
        raise ProductionFinalizationError(
            "campaign hard deadline reached; durable campaign status is INCOMPLETE"
        )
    if (
        lifecycle.status != CampaignState.COMPLETED.value
        and lifecycle.deadline_remaining_seconds <= 0.0
    ):
        raise ProductionFinalizationError(
            "campaign hard deadline reached without a durable INCOMPLETE transition"
        )
    deadline_event = _FinalizationDeadlineEvent(
        signal_event=signal_event,
        engine=selected_engine,
        run_dir=run,
    )
    tracker = _FinalizationResourceTracker(lifecycle.deadline_elapsed_seconds)
    try:
        result = _finalize_provider_free_campaign_impl(
            run,
            project_root=root,
            engine=selected_engine,
            cancel_event=deadline_event,
            resource_tracker=tracker,
        )
    except BaseException as exc:
        if lifecycle.status != CampaignState.COMPLETED.value and deadline_event.deadline_expired():
            terminal = selected_engine.resume(run)
            if terminal.status != CampaignState.INCOMPLETE.value:
                raise ProductionFinalizationError(
                    "hard deadline crossed but campaign did not become INCOMPLETE"
                ) from exc
            raise ProductionFinalizationError(
                "campaign hard deadline crossed during finalization; durable status is INCOMPLETE"
            ) from exc
        raise
    return result


def _bundle_identity(manifest: Mapping[str, object]) -> FrozenReplayIdentity:
    scientific = manifest.get("scientific_artifact_hashes")
    data = manifest.get("data_identity")
    environment = manifest.get("environment_and_resource_usage")
    if (
        not isinstance(scientific, dict)
        or not isinstance(data, dict)
        or not isinstance(environment, dict)
    ):
        raise ProductionFinalizationError("bundle replay identity is incomplete")
    canonical = _digest(data.get("canonical_digest"), "bundle canonical data")
    return FrozenReplayIdentity(
        source_sha256=_digest(scientific.get("source"), "bundle source"),
        config_sha256=_digest(scientific.get("config"), "bundle config"),
        features_sha256=_digest(scientific.get("features"), "bundle features"),
        checkpoint_sha256=_digest(scientific.get("checkpoint"), "bundle checkpoint"),
        validation_prediction_artifact_sha256=_digest(
            scientific.get("predictions"),
            "bundle validation predictions",
        ),
        validation_prediction_digest="0" * 64,  # replaced after reference is decoded
        data_sha256=canonical,
        environment_sha256=_digest(
            environment.get("environment_sha256"),
            "bundle environment",
        ),
    )


@dataclass(frozen=True, slots=True)
class _RuntimeIdentityClosure:
    project_source_digest: str
    environment_digest: str
    uv_lock_sha256: str
    dependency_groups: tuple[str, ...]


def _runtime_identity_closure(
    manifest: Mapping[str, object],
    environment: Mapping[str, object],
) -> _RuntimeIdentityClosure:
    resource = manifest.get("environment_and_resource_usage")
    if not isinstance(resource, Mapping):
        raise ProductionFinalizationError("bundle runtime evidence is missing")
    runtime = resource.get("runtime_identity")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "schema_version",
        "project_source_digest",
        "environment_digest",
        "uv_lock_sha256",
        "dependency_groups",
    }:
        raise ProductionFinalizationError("bundle runtime identity closure is incomplete")
    if runtime.get("schema_version") != 1:
        raise ProductionFinalizationError("bundle runtime identity schema is unsupported")
    raw_groups = runtime.get("dependency_groups")
    if not isinstance(raw_groups, list) or any(type(group) is not str for group in raw_groups):
        raise ProductionFinalizationError("bundle dependency group profile is malformed")
    groups = tuple(cast(list[str], raw_groups))
    expected_groups = _dependency_groups(environment)
    project_source_digest = _digest(
        runtime.get("project_source_digest"),
        "bundle trusted project source",
    )
    environment_digest = _digest(runtime.get("environment_digest"), "bundle runtime environment")
    uv_lock_sha256 = _digest(runtime.get("uv_lock_sha256"), "bundle runtime uv.lock")
    if (
        groups != expected_groups
        or environment_digest != _digest(resource.get("environment_sha256"), "bundle environment")
        or environment_digest != environment_identity_digest(environment)
        or uv_lock_sha256 != _digest(environment.get("uv_lock_sha256"), "environment uv.lock")
    ):
        raise ProductionFinalizationError("bundle runtime identity cross-links changed")
    return _RuntimeIdentityClosure(
        project_source_digest=project_source_digest,
        environment_digest=environment_digest,
        uv_lock_sha256=uv_lock_sha256,
        dependency_groups=groups,
    )


def _verify_current_runtime(
    project_root: Path,
    closure: _RuntimeIdentityClosure,
) -> tuple[Path, Mapping[str, object]]:
    try:
        if stat.S_ISLNK(project_root.lstat().st_mode):
            raise ProductionFinalizationError("replay project root must not be a symlink")
        root = project_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionFinalizationError("replay project root is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise ProductionFinalizationError("replay project root must be a real directory")
    try:
        source = hash_source_tree(root)
        environment_identity = capture_environment_identity(root)
    except Exception as exc:
        raise ProductionFinalizationError("replay runtime identity cannot be recaptured") from exc
    environment = environment_identity.manifest()
    if (
        source.digest != closure.project_source_digest
        or environment_identity.digest != closure.environment_digest
        or environment_identity_digest(environment) != closure.environment_digest
        or _digest(environment.get("uv_lock_sha256"), "current uv.lock") != closure.uv_lock_sha256
        or _dependency_groups(environment) != closure.dependency_groups
    ):
        raise ProductionFinalizationError(
            "current project source, environment, lock, or dependency profile differs from bundle"
        )
    return root, MappingProxyType(environment)


def _find_starter(bundle: _VerifiedBundle, project_root: Path) -> Path:
    starter_identity = bundle.manifest.get("starter_identity")
    if not isinstance(starter_identity, dict):
        raise ProductionFinalizationError("bundle starter identity is missing")
    expected = _digest(starter_identity.get("manifest_sha256"), "bundle starter manifest")
    candidates = [bundle.root / "source", project_root / "kuairand-starter-kit"]
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            verification = verify_starter_kit(resolved)
        except Exception:
            continue
        if verification.manifest_sha256 == expected:
            return resolved
    raise ProductionFinalizationError("replay cannot locate the hash-pinned organizer starter kit")


def replay_final_bundle(
    bundle_dir: Path,
    data_dir: Path,
    expected_data_sha256: str,
    *,
    project_root: Path,
    cancel_event: threading.Event | None = None,
) -> FinalBundleReplayOutcome:
    """Reproduce a closed bundle locally without provider, network, or final labels."""

    if not all(isinstance(path, Path) for path in (bundle_dir, data_dir, project_root)):
        raise ProductionFinalizationError(
            "bundle_dir, data_dir, and project_root must be pathlib.Path values"
        )
    cancellation = _cancel_event(cancel_event)
    _raise_if_cancelled(cancellation, "closed bundle verification")
    expected = _digest(expected_data_sha256, "expected_data_sha256")
    bundle = _verify_closed_bundle(bundle_dir)
    historical_environment = _strict_json(
        bundle.root / "environment.json",
        maximum=_MAX_MANIFEST_BYTES,
        location="bundle environment",
    )
    runtime_closure = _runtime_identity_closure(bundle.manifest, historical_environment)
    _raise_if_cancelled(cancellation, "current replay runtime verification")
    verified_project_root, environment = _verify_current_runtime(project_root, runtime_closure)
    _raise_if_cancelled(cancellation, "organizer starter verification")
    starter_dir = _find_starter(bundle, verified_project_root)
    try:
        if stat.S_ISLNK(data_dir.lstat().st_mode):
            raise ProductionFinalizationError("replay data path must not be a symlink")
        verified_data = data_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionFinalizationError("replay data directory is unavailable") from exc
    context = _trusted_replay_context(
        data_dir=verified_data,
        starter_dir=starter_dir,
        expected_data_sha256=expected,
    )
    base_identity = _bundle_identity(bundle.manifest)
    if base_identity.data_sha256 != expected:
        raise ProductionFinalizationError("bundle and requested canonical data identities differ")
    if environment_identity_digest(environment) != base_identity.environment_sha256:
        raise ProductionFinalizationError("current replay environment identity changed")
    selection = bundle.manifest.get("selection")
    if not isinstance(selection, dict):
        raise ProductionFinalizationError("bundle selection is missing")
    candidate_id = _text(selection.get("selected_experiment"), "selected experiment")
    scratch = Path(tempfile.mkdtemp(prefix="kuairand-closed-replay-"))
    try:
        store = ArtifactStore(scratch / "artifacts")
        recipe = load_replay_recipe(
            bundle.root / "config" / "artifact",
            expected_sha256=base_identity.config_sha256,
        )
        source = store.put_directory(bundle.root / "source", kind=ArtifactKind.SOURCE)
        config = store.put_file(bundle.root / "config" / "artifact", kind=ArtifactKind.INPUT)
        features: ArtifactRef | DirectoryArtifactRef
        checkpoint: ArtifactRef | DirectoryArtifactRef
        if isinstance(recipe, OfficialFMReplayRecipe):
            features = store.put_file(
                bundle.root / "preprocessing" / "artifact",
                kind=ArtifactKind.INPUT,
            )
            checkpoint = store.put_file(
                bundle.root / "model" / "artifact",
                kind=ArtifactKind.CHECKPOINT,
            )
        else:
            features = store.put_directory(
                bundle.root / "preprocessing",
                kind=ArtifactKind.INPUT,
            )
            checkpoint = store.put_directory(
                bundle.root / "model",
                kind=ArtifactKind.CHECKPOINT,
            )
        reference = store.put_file(
            bundle.root / "validation-evidence" / "reference-validation-predictions.npy",
            kind=ArtifactKind.PREDICTION,
        )
        reference_scores = _load_npy(
            reference,
            store,
            rows=context.dataset.valid.row_count,
        )
        identity = FrozenReplayIdentity(
            source_sha256=base_identity.source_sha256,
            config_sha256=base_identity.config_sha256,
            features_sha256=base_identity.features_sha256,
            checkpoint_sha256=base_identity.checkpoint_sha256,
            validation_prediction_artifact_sha256=(
                base_identity.validation_prediction_artifact_sha256
            ),
            validation_prediction_digest=prediction_digest(reference_scores),
            data_sha256=base_identity.data_sha256,
            environment_sha256=base_identity.environment_sha256,
        )
        actual = (
            source.sha256,
            config.sha256,
            features.sha256,
            checkpoint.sha256,
            reference.sha256,
        )
        frozen = (
            identity.source_sha256,
            identity.config_sha256,
            identity.features_sha256,
            identity.checkpoint_sha256,
            identity.validation_prediction_artifact_sha256,
        )
        if actual != frozen:
            raise ProductionFinalizationError("bundle artifact reconstruction changed identity")
        _raise_if_cancelled(cancellation, "closed replay subprocess")
        backend = load_replay_backend(
            bundle.root / "config" / "artifact",
            expected_sha256=identity.config_sha256,
            cancel_event=cancellation,
        )
        if recipe.data_sha256 != expected:
            raise ProductionFinalizationError("bundle recipe data identity changed")
        replay = run_clean_replay(
            CleanReplayRequest(
                candidate_id=candidate_id,
                output_dir=scratch / "replayed",
                identity=identity,
                artifacts=ReplayArtifacts(
                    source=source,
                    config=config,
                    features=features,
                    checkpoint=checkpoint,
                    validation_predictions=reference,
                ),
                environment=environment,
                equality=ReplayEquality.EXACT,
                training_replay="closed checkpoint replay; no research provider or network",
            ),
            artifact_store=store,
            capabilities=context.capabilities,
            backend=backend,
            protected_metric_evaluator=context.score,
            cancel_event=cancellation,
        )
        bundled_submission = sha256_file(bundle.root / "submission.csv")
        reproduced_submission = sha256_file(replay.final_submission)
        if bundled_submission != reproduced_submission:
            raise ProductionFinalizationError("closed replay final submission differs from bundle")
        _raise_if_cancelled(cancellation, "organizer submission check")
        organizer_scratch = scratch / "organizer-check"
        organizer_scratch.mkdir(mode=0o700)
        organizer_scratch_metadata = organizer_scratch.lstat()
        if (
            stat.S_ISLNK(organizer_scratch_metadata.st_mode)
            or not stat.S_ISDIR(organizer_scratch_metadata.st_mode)
            or stat.S_IMODE(organizer_scratch_metadata.st_mode) != 0o700
        ):
            raise ProductionFinalizationError("closed replay organizer scratch is unsafe")
        organizer = check_final_submission(
            replay.final_submission,
            data_dir=verified_data,
            starter_dir=starter_dir,
            scratch_dir=organizer_scratch,
        )
        if organizer.submission_sha256 != bundled_submission:
            raise ProductionFinalizationError(
                "organizer checker observed different replay submission bytes"
            )
        return FinalBundleReplayOutcome(
            candidate_id=candidate_id,
            bundle_root=bundle.root,
            bundle_manifest_sha256=bundle.manifest_sha256,
            expected_data_sha256=expected,
            replay=replay.evidence,
            organizer_check=organizer,
            bundled_submission_sha256=bundled_submission,
            reproduced_submission_sha256=reproduced_submission,
            project_root=verified_project_root,
            project_source_digest=runtime_closure.project_source_digest,
            environment_digest=runtime_closure.environment_digest,
            uv_lock_sha256=runtime_closure.uv_lock_sha256,
            dependency_groups=runtime_closure.dependency_groups,
        )
    finally:
        try:
            _remove_replay_private_tree(scratch)
        except OSError as exc:
            raise ProductionFinalizationError(
                "closed replay scratch workspace cleanup failed"
            ) from exc


__all__ = [
    "PRODUCTION_BUNDLE_REPLAY_SCHEMA_VERSION",
    "PRODUCTION_FINALIZATION_SCHEMA_VERSION",
    "FinalBundleReplayOutcome",
    "ProductionFailureEvidence",
    "ProductionFinalizationError",
    "ProductionFinalizationOutcome",
    "finalize_provider_free_campaign",
    "replay_final_bundle",
]
