"""Fail-closed admission for the bounded, provider-free CPU fallback.

This module deliberately performs every expensive or mutable-input verification before a caller
may open :class:`~kuairand_agent.state.repository.StateRepository` or create a run directory.  It
does not import the legacy campaign engine, its stores, or its filesystem ledgers.  Successful
admission exposes the already verified runtime objects separately from a path-free canonical
receipt, so local locations never become scientific identity.
"""

from __future__ import annotations

import ast
import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from kuairand_agent.campaign.qualification_evidence import (
    OfficialFMFallbackEvidence,
    OfficialFMQualificationEvidence,
    QualificationEvidenceError,
    QualificationExpectations,
    load_official_fm_qualification,
)
from kuairand_agent.contract import (
    OrganizerIntegrityError,
    StarterVerification,
    verify_starter_kit,
)
from kuairand_agent.data.audit import DataAuditError, DataAuditReport, audit_dataset
from kuairand_agent.data.canonical import (
    OUTCOME_FIELDS,
    CanonicalDataError,
    CanonicalDataset,
    load_canonical_dataset,
)
from kuairand_agent.domain.identity import canonical_json_bytes
from kuairand_agent.observability.receipts import StartupReceipt
from kuairand_agent.performance import (
    PerformanceAcceptanceError,
    PerformanceProfile,
    load_performance_profile,
)
from kuairand_agent.resource_profiles import ResourceProfile, load_resource_profile

PRODUCTION_ADMISSION_SCHEMA_VERSION: Final = 1
CONTROLLER_CAPABILITY_SCHEMA_VERSION: Final = 1
_ADMISSION_DOMAIN: Final = b"kuairand-production-cpu-fallback-admission-v1\0"
_CONTROLLER_DOMAIN: Final = b"kuairand-production-controller-capability-v1\0"
_CONTROLLER_MODULE: Final = "kuairand_agent.production.controller"
_CONTROLLER_SOURCE_RELATIVE: Final = Path("src/kuairand_agent/production/controller.py")
_PROFILE_SOURCE_RELATIVE: Final = Path("configs/competition-cpu.toml")
_ZERO_OUTCOME_COUNTERS: Final = (
    "outcome_cells_materialized",
    "outcome_cells_decoded",
    "outcome_cells_converted",
    "outcome_cells_validated",
    "outcome_cells_aggregated",
    "outcome_cells_logged",
    "outcome_cells_scored",
)
_REQUIRED_MEASURED_FAMILIES: Final = (
    "data_audit",
    "causal_feature_cold",
    "causal_feature_warm",
    "full_data_grouping",
    "pairwise_sampler",
    "official_fm",
    "final_replay",
    "controller_admission",
)
_FINALIZATION_FAMILIES: Final = (
    "data_audit",
    "causal_feature_warm",
    "final_replay",
)
_FORBIDDEN_LEGACY_WRITER_MODULES: Final = (
    "kuairand_agent.campaign.candidate_journal",
    "kuairand_agent.campaign.controller",
    "kuairand_agent.campaign.full_campaign",
    "kuairand_agent.campaign.full_campaign_runtime",
    "kuairand_agent.campaign.scientific_store",
    "kuairand_agent.campaign.store",
)


class ProductionAdmissionError(RuntimeError):
    """Raised before mutable authority opens when production evidence is insufficient."""


def _digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionAdmissionError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _receipt_id(domain: bytes, body: Mapping[str, object]) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(body)).hexdigest()


def _regular_file_sha256(path: Path, *, location: str) -> str:
    """Hash one stable regular file without following a final-component symlink."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise ProductionAdmissionError(f"cannot inspect {location}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProductionAdmissionError(f"{location} must be a regular non-symlink file")
    descriptor = -1
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ProductionAdmissionError(f"{location} changed while being opened")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ProductionAdmissionError(f"{location} changed while being hashed")
    except OSError as exc:
        raise ProductionAdmissionError(f"cannot hash {location}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _directory(path: Path, location: str) -> Path:
    if not isinstance(path, Path):
        raise ProductionAdmissionError(f"{location} must be pathlib.Path")
    try:
        root = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionAdmissionError(f"{location} is unavailable") from exc
    if not root.is_dir():
        raise ProductionAdmissionError(f"{location} must be a directory")
    return root


def _file(path: Path, location: str) -> Path:
    if not isinstance(path, Path):
        raise ProductionAdmissionError(f"{location} must be pathlib.Path")
    candidate = path.expanduser().absolute()
    try:
        inspected = candidate.lstat()
    except OSError as exc:
        raise ProductionAdmissionError(f"{location} is unavailable") from exc
    if stat.S_ISLNK(inspected.st_mode) or not stat.S_ISREG(inspected.st_mode):
        raise ProductionAdmissionError(f"{location} must be a regular non-symlink file")
    return candidate


@dataclass(frozen=True, slots=True)
class ControllerCapabilityReceipt:
    """Internal proof that the admitted controller has exactly one mutable authority."""

    controller_module: str
    controller_source_sha256: str
    authority_module: str = "kuairand_agent.state.repository"
    authority_type: str = "StateRepository"
    state_repository_only: bool = True
    legacy_writer_imports: tuple[str, ...] = ()
    provider_request_limit: int = 0
    protected_query_limit: int = 0
    schema_version: int = CONTROLLER_CAPABILITY_SCHEMA_VERSION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CONTROLLER_CAPABILITY_SCHEMA_VERSION:
            raise ProductionAdmissionError("controller capability schema_version must be 1")
        if self.controller_module != _CONTROLLER_MODULE:
            raise ProductionAdmissionError("controller capability names an unknown controller")
        _digest(self.controller_source_sha256, "controller source digest")
        if (
            self.authority_module != "kuairand_agent.state.repository"
            or self.authority_type != "StateRepository"
            or not self.state_repository_only
            or self.legacy_writer_imports
        ):
            raise ProductionAdmissionError(
                "production controller must declare StateRepository as its sole authority"
            )
        if self.provider_request_limit != 0 or self.protected_query_limit != 0:
            raise ProductionAdmissionError(
                "bounded CPU fallback admission must allow zero provider and protected calls"
            )
        object.__setattr__(
            self,
            "receipt_id",
            _receipt_id(_CONTROLLER_DOMAIN, self._body()),
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "controller_module": self.controller_module,
            "controller_source_sha256": self.controller_source_sha256,
            "authority_module": self.authority_module,
            "authority_type": self.authority_type,
            "state_repository_only": self.state_repository_only,
            "legacy_writer_imports": list(self.legacy_writer_imports),
            "provider_request_limit": self.provider_request_limit,
            "protected_query_limit": self.protected_query_limit,
        }

    def manifest(self) -> dict[str, object]:
        return {**self._body(), "receipt_id": self.receipt_id}


@dataclass(frozen=True, slots=True)
class MeasuredFamilyCapability:
    """Path-free p95/resource binding for one required measured operation family."""

    family: str
    sample_count: int
    p95_seconds: float
    peak_rss_bytes: int
    maximum_rows: int
    evidence_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.family or "\x00" in self.family:
            raise ProductionAdmissionError("measured family name must be non-empty")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ProductionAdmissionError("measured family sample_count must be positive")
        if not isinstance(self.p95_seconds, float) or not (0.0 < self.p95_seconds < float("inf")):
            raise ProductionAdmissionError("measured family p95 must be positive and finite")
        if type(self.peak_rss_bytes) is not int or self.peak_rss_bytes <= 0:
            raise ProductionAdmissionError("measured family peak RSS must be positive")
        if type(self.maximum_rows) is not int or self.maximum_rows <= 0:
            raise ProductionAdmissionError("measured family row count must be positive")
        if not self.evidence_digests:
            raise ProductionAdmissionError("measured family must bind evidence digests")
        for digest in self.evidence_digests:
            _digest(digest, f"{self.family} evidence digest")

    def manifest(self) -> dict[str, object]:
        return {
            "family": self.family,
            "sample_count": self.sample_count,
            "p95_seconds": self.p95_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "maximum_rows": self.maximum_rows,
            "evidence_digests": list(self.evidence_digests),
        }


@dataclass(frozen=True, slots=True)
class ProductionRuntimeCapabilities:
    """Verified live objects for the zero-provider, zero-protected fallback controller."""

    resource_profile: ResourceProfile
    audit: DataAuditReport
    dataset: CanonicalDataset
    starter: StarterVerification
    qualification: OfficialFMQualificationEvidence
    performance: PerformanceProfile
    measured_families: tuple[MeasuredFamilyCapability, ...]
    controller: ControllerCapabilityReceipt
    provider_request_limit: int = 0
    protected_query_limit: int = 0

    @property
    def fallback(self) -> OfficialFMFallbackEvidence:
        """Expose the qualified official-FM seed-4 runtime closure."""

        return self.qualification.fallback


@dataclass(frozen=True, slots=True)
class ProductionAdmission:
    """Successful runtime capabilities plus their canonical, path-free admission identity."""

    startup_receipt_id: str
    contract_id: str
    resource_profile_digest: str
    resource_profile_file_sha256: str
    audit_digest: str
    canonical_dataset_digest: str
    starter_manifest_sha256: str
    qualification_manifest_digest: str
    qualification_input_digest: str
    fallback_manifest_digest: str
    performance_profile_digest: str
    performance_profile_file_sha256: str
    measured_families: tuple[MeasuredFamilyCapability, ...]
    controller_receipt_id: str
    train_rows: int
    validation_rows: int
    final_rows: int
    runtime: ProductionRuntimeCapabilities = field(repr=False, compare=False)
    schema_version: int = PRODUCTION_ADMISSION_SCHEMA_VERSION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_ADMISSION_SCHEMA_VERSION:
            raise ProductionAdmissionError("production admission schema_version must be 1")
        for name in (
            "startup_receipt_id",
            "contract_id",
            "resource_profile_digest",
            "resource_profile_file_sha256",
            "audit_digest",
            "canonical_dataset_digest",
            "starter_manifest_sha256",
            "qualification_manifest_digest",
            "qualification_input_digest",
            "fallback_manifest_digest",
            "performance_profile_digest",
            "performance_profile_file_sha256",
            "controller_receipt_id",
        ):
            _digest(getattr(self, name), name)
        for name in ("train_rows", "validation_rows", "final_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ProductionAdmissionError(f"{name} must be a positive integer")
        if tuple(item.family for item in self.measured_families) != _REQUIRED_MEASURED_FAMILIES:
            raise ProductionAdmissionError("production admission measured-family order changed")
        object.__setattr__(self, "receipt_id", _receipt_id(_ADMISSION_DOMAIN, self._body()))

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": "OFFICIAL_FM_CPU_FALLBACK_ONLY",
            "startup_receipt_id": self.startup_receipt_id,
            "contract_id": self.contract_id,
            "resource_profile": {
                "name": "competition-cpu",
                "digest": self.resource_profile_digest,
                "file_sha256": self.resource_profile_file_sha256,
            },
            "data": {
                "audit_digest": self.audit_digest,
                "canonical_dataset_digest": self.canonical_dataset_digest,
                "train_rows": self.train_rows,
                "validation_rows": self.validation_rows,
                "final_rows": self.final_rows,
                "final_outcome_parsed_fields": [],
                "final_outcome_parsed_cells": 0,
            },
            "starter_manifest_sha256": self.starter_manifest_sha256,
            "qualification": {
                "manifest_digest": self.qualification_manifest_digest,
                "input_digest": self.qualification_input_digest,
                "fallback_manifest_digest": self.fallback_manifest_digest,
            },
            "performance": {
                "profile_digest": self.performance_profile_digest,
                "file_sha256": self.performance_profile_file_sha256,
                "families": [item.manifest() for item in self.measured_families],
            },
            "controller_receipt_id": self.controller_receipt_id,
            "provider_request_limit": 0,
            "protected_query_limit": 0,
            "mutable_authority": "StateRepository",
        }

    def manifest(self) -> dict[str, object]:
        """Return canonical admission evidence with no local filesystem paths."""

        return {**self._body(), "receipt_id": self.receipt_id}


def _verify_final_no_outcomes(audit: DataAuditReport, dataset: CanonicalDataset) -> None:
    audit_trace = audit.final_outcome_trace.manifest()
    if audit_trace.get("split") != "test":
        raise ProductionAdmissionError("audit final outcome trace is not the test split")
    skipped = audit_trace.get("skipped_fields")
    if (
        not isinstance(skipped, list)
        or len(skipped) != len(OUTCOME_FIELDS)
        or set(skipped) != set(OUTCOME_FIELDS)
    ):
        raise ProductionAdmissionError("audit did not skip every final-period outcome field")
    selected = audit_trace.get("selected_fields")
    if not isinstance(selected, list) or set(selected) & set(OUTCOME_FIELDS):
        raise ProductionAdmissionError("audit selected a final-period outcome field")
    if any(audit_trace.get(name) != 0 for name in _ZERO_OUTCOME_COUNTERS):
        raise ProductionAdmissionError("audit final-period outcome counter is nonzero")
    if audit_trace.get("skipped_values_recorded") is not False:
        raise ProductionAdmissionError("audit recorded skipped final-period outcome values")

    final = dataset.final
    canonical_trace = final.outcome_trace
    if (
        canonical_trace.parsed_fields
        or canonical_trace.parsed_cell_count != 0
        or canonical_trace.skipped_fields != OUTCOME_FIELDS
        or canonical_trace.row_count != final.row_count
    ):
        raise ProductionAdmissionError("canonical final split accessed an outcome")
    final_manifest = final.manifest()
    if "target_access" in final_manifest or "target_digest" in final_manifest:
        raise ProductionAdmissionError("canonical final split contains target-shaped metadata")
    if hasattr(final, "targets"):
        raise ProductionAdmissionError("canonical final split structurally exposes targets")


def _controller_receipt(repository_root: Path) -> ControllerCapabilityReceipt:
    source_path = repository_root / _CONTROLLER_SOURCE_RELATIVE
    source_sha256 = _regular_file_sha256(source_path, location="production controller source")
    try:
        parsed = ast.parse(source_path.read_bytes(), filename=_CONTROLLER_MODULE)
    except (OSError, SyntaxError) as exc:
        raise ProductionAdmissionError("production controller source cannot be inspected") from exc
    imports: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    forbidden = tuple(
        module
        for module in _FORBIDDEN_LEGACY_WRITER_MODULES
        if any(imported == module or imported.startswith(f"{module}.") for imported in imports)
    )
    if forbidden:
        raise ProductionAdmissionError(
            "production controller imports legacy mutable authority: " + ", ".join(forbidden)
        )
    if "kuairand_agent.state.repository" not in imports:
        raise ProductionAdmissionError("production controller does not import StateRepository")
    if _regular_file_sha256(source_path, location="production controller source") != source_sha256:
        raise ProductionAdmissionError("production controller source changed during admission")
    return ControllerCapabilityReceipt(
        controller_module=_CONTROLLER_MODULE,
        controller_source_sha256=source_sha256,
        legacy_writer_imports=forbidden,
    )


def _measured_capabilities(
    performance: PerformanceProfile,
    *,
    profile: ResourceProfile,
    audit: DataAuditReport,
    dataset: CanonicalDataset,
    qualification: OfficialFMQualificationEvidence,
) -> tuple[MeasuredFamilyCapability, ...]:
    missing = tuple(
        name for name in _REQUIRED_MEASURED_FAMILIES if name not in performance.families
    )
    if missing:
        raise ProductionAdmissionError(f"performance profile lacks required families: {missing!r}")
    if tuple(performance.finalization_families) != _FINALIZATION_FAMILIES:
        raise ProductionAdmissionError("performance finalization family set changed")
    if performance.finalization_reserve_seconds < profile.finalization_reserve_seconds:
        raise ProductionAdmissionError("measured finalization reserve is below the CPU profile")
    if performance.campaign_limit_seconds != float(profile.wall_clock_seconds):
        raise ProductionAdmissionError("performance campaign limit differs from the CPU profile")
    if not performance.finalization_reserve_sufficient or not performance.campaign_time_sufficient:
        raise ProductionAdmissionError("measured CPU campaign does not fit required reserves")
    if performance.controller_overhead_ratio >= 0.01:
        raise ProductionAdmissionError("controller overhead exceeds one percent of measured work")
    peak_rss_bytes = max(item.peak_rss_bytes for item in performance.receipts)
    if peak_rss_bytes > profile.candidate_rss_target_bytes:
        raise ProductionAdmissionError("measured peak RSS exceeds the conservative CPU target")

    total_rows = dataset.train.row_count + dataset.valid.row_count + dataset.final.row_count
    expected_rows = {
        "data_audit": total_rows,
        "causal_feature_cold": total_rows,
        "causal_feature_warm": total_rows,
        "full_data_grouping": dataset.train.row_count,
        "pairwise_sampler": dataset.train.row_count,
        "official_fm": dataset.train.row_count,
        "final_replay": dataset.train.row_count + dataset.valid.row_count,
    }
    for family, rows in expected_rows.items():
        if performance.family(family).maximum_rows != rows:
            raise ProductionAdmissionError(f"performance {family} row count differs from full data")
    audit_summary = performance.family("data_audit")
    if audit_summary.evidence_digests != (audit.digest,):
        raise ProductionAdmissionError("performance data audit is not bound to current data")
    official_summary = performance.family("official_fm")
    official_digests = tuple(item.checkpoint_digest for item in qualification.outer_runs)
    if (
        official_summary.sample_count < len(official_digests)
        or tuple(official_summary.evidence_digests) != official_digests
    ):
        raise ProductionAdmissionError(
            "performance official-FM samples differ from current qualification"
        )
    return tuple(
        MeasuredFamilyCapability(
            family=name,
            sample_count=performance.family(name).sample_count,
            p95_seconds=float(performance.family(name).p95_seconds),
            peak_rss_bytes=performance.family(name).peak_rss_bytes,
            maximum_rows=performance.family(name).maximum_rows,
            evidence_digests=tuple(performance.family(name).evidence_digests),
        )
        for name in _REQUIRED_MEASURED_FAMILIES
    )


def admit_cpu_fallback(
    *,
    repository_root: Path,
    startup_receipt: StartupReceipt,
    resource_profile: ResourceProfile,
    resource_profile_file_sha256: str,
    data_root: Path,
    starter_root: Path,
    qualification_run_dir: Path,
    performance_profile_path: Path,
) -> ProductionAdmission:
    """Verify and admit the official-FM CPU fallback without any mutable state write."""

    repository = _directory(repository_root, "repository_root")
    if not isinstance(startup_receipt, StartupReceipt):
        raise ProductionAdmissionError("startup_receipt must be StartupReceipt")
    if startup_receipt.state_writes_started:
        raise ProductionAdmissionError("startup receipt was created after mutable state opened")
    if startup_receipt.profile != "competition-cpu":
        raise ProductionAdmissionError("CPU fallback requires a competition-cpu startup receipt")
    if not isinstance(resource_profile, ResourceProfile):
        raise ProductionAdmissionError("resource_profile must be ResourceProfile")
    if (
        resource_profile.name != "competition-cpu"
        or resource_profile.device != "cpu"
        or resource_profile.dependency_group != "tree-cpu"
        or not resource_profile.requires_measured_p95
        or resource_profile.in_attempt_device_fallback
        or resource_profile.gpu_probe_allowed
    ):
        raise ProductionAdmissionError("CPU fallback requires the frozen competition-cpu profile")

    expected_profile_sha256 = _digest(
        resource_profile_file_sha256,
        "resource profile file digest",
    )
    profile_path = repository / _PROFILE_SOURCE_RELATIVE
    observed_profile_sha256 = _regular_file_sha256(
        profile_path,
        location="competition-cpu resource profile",
    )
    if observed_profile_sha256 != expected_profile_sha256:
        raise ProductionAdmissionError("competition-cpu resource profile digest mismatch")
    try:
        loaded_profile = load_resource_profile(profile_path)
    except ValueError as exc:
        raise ProductionAdmissionError("competition-cpu resource profile is invalid") from exc
    if loaded_profile.digest != resource_profile.digest:
        raise ProductionAdmissionError("resource profile object differs from its checked-in file")

    # No StateRepository or run-directory operation may move above these evidence loads.
    requested_data_root = _directory(data_root, "KuaiRand-Pure data root")
    try:
        audit = audit_dataset(requested_data_root)
        dataset = load_canonical_dataset(audit.data_dir)
    except (DataAuditError, CanonicalDataError) as exc:
        raise ProductionAdmissionError("full-data audit or canonical load failed") from exc
    _verify_final_no_outcomes(audit, dataset)
    starter_dir = _directory(starter_root, "organizer starter root")
    try:
        starter = verify_starter_kit(starter_dir)
    except OrganizerIntegrityError as exc:
        raise ProductionAdmissionError("organizer starter verification failed") from exc

    repository_inputs = startup_receipt.repository_inputs
    if repository_inputs is not None:
        recorded_starter = repository_inputs.get("organizer_manifest_sha256")
        if recorded_starter != starter.manifest_sha256:
            raise ProductionAdmissionError("starter root differs from startup repository inputs")

    qualification_dir = _directory(qualification_run_dir, "official-FM qualification run")
    scorer_digest = starter.files["evaluate.py"]
    try:
        qualification = load_official_fm_qualification(
            qualification_dir,
            expectations=QualificationExpectations(
                canonical_digest=dataset.digest,
                starter_manifest_digest=starter.manifest_sha256,
                scorer_digest=scorer_digest,
                validation_row_count=dataset.valid.row_count,
                final_row_count=dataset.final.row_count,
            ),
        )
    except QualificationEvidenceError as exc:
        raise ProductionAdmissionError("official-FM qualification verification failed") from exc
    if qualification.audit_digest != audit.digest:
        raise ProductionAdmissionError("qualification audit differs from current full data")
    if tuple(item.seed for item in qualification.outer_runs) != (0, 1, 2):
        raise ProductionAdmissionError("qualification lacks exact outer seeds 0, 1, and 2")
    if qualification.fallback.seed != 4:
        raise ProductionAdmissionError("qualification fallback is not official-FM seed 4")

    performance_path = _file(performance_profile_path, "performance profile")
    performance_file_sha256 = _regular_file_sha256(
        performance_path,
        location="performance profile",
    )
    try:
        loaded_performance = load_performance_profile(performance_path)
    except PerformanceAcceptanceError as exc:
        raise ProductionAdmissionError("performance profile verification failed") from exc
    performance = loaded_performance.profile
    if loaded_performance.physical_sha256 != performance_file_sha256:
        raise ProductionAdmissionError("performance loader file digest differs from admission")
    if (
        _regular_file_sha256(performance_path, location="performance profile")
        != performance_file_sha256
    ):
        raise ProductionAdmissionError("performance profile changed while being admitted")
    measured = _measured_capabilities(
        performance,
        profile=resource_profile,
        audit=audit,
        dataset=dataset,
        qualification=qualification,
    )
    controller = _controller_receipt(repository)
    runtime = ProductionRuntimeCapabilities(
        resource_profile=resource_profile,
        audit=audit,
        dataset=dataset,
        starter=starter,
        qualification=qualification,
        performance=performance,
        measured_families=measured,
        controller=controller,
    )
    return ProductionAdmission(
        startup_receipt_id=startup_receipt.receipt_id,
        contract_id=startup_receipt.contract_id,
        resource_profile_digest=resource_profile.digest,
        resource_profile_file_sha256=observed_profile_sha256,
        audit_digest=audit.digest,
        canonical_dataset_digest=dataset.digest,
        starter_manifest_sha256=starter.manifest_sha256,
        qualification_manifest_digest=qualification.manifest_digest,
        qualification_input_digest=qualification.qualification_input_digest,
        fallback_manifest_digest=qualification.fallback.manifest_digest,
        performance_profile_digest=performance.digest,
        performance_profile_file_sha256=performance_file_sha256,
        measured_families=measured,
        controller_receipt_id=controller.receipt_id,
        train_rows=dataset.train.row_count,
        validation_rows=dataset.valid.row_count,
        final_rows=dataset.final.row_count,
        runtime=runtime,
    )


__all__ = [
    "CONTROLLER_CAPABILITY_SCHEMA_VERSION",
    "PRODUCTION_ADMISSION_SCHEMA_VERSION",
    "ControllerCapabilityReceipt",
    "MeasuredFamilyCapability",
    "ProductionAdmission",
    "ProductionAdmissionError",
    "ProductionRuntimeCapabilities",
    "admit_cpu_fallback",
]
