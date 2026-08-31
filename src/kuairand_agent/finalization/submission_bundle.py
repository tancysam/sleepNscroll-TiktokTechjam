"""No-overwrite construction of a closed, judge-readable final bundle."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import resource
import shutil
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Final

from kuairand_agent.contract import sha256_file
from kuairand_agent.finalization.organizer_check import OrganizerCheckEvidence
from kuairand_agent.finalization.replay import ReplayError, environment_identity_digest

FINAL_BUNDLE_SCHEMA_VERSION: Final = 1
REQUIRED_FILE_PATHS: Final = (
    "report.md",
    "submission.csv",
    "experiments.jsonl",
    "experiments.csv",
    "environment.json",
    "reproduce.sh",
    "verification.json",
    "prepublication-resource.json",
)
REQUIRED_DIRECTORY_PATHS: Final = (
    "config",
    "source",
    "model",
    "preprocessing",
    "validation-evidence",
    "replay",
)
_REQUIRED_SCIENTIFIC_HASHES: Final = frozenset(
    {"source", "config", "features", "checkpoint", "predictions"}
)
_REQUIRED_TOTALS: Final = frozenset(
    {
        "attempt_count",
        "scientific_iteration_count",
        "launch_count",
        "elapsed_seconds",
        "manual_intervention_count",
    }
)
_MAX_FILE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_BUNDLE_BYTES: Final = 8 * 1024 * 1024 * 1024
_MAX_REPLAY_EVIDENCE_BYTES: Final = 4 * 1024 * 1024
_COPY_CHUNK_BYTES: Final = 1024 * 1024
_RENAME_NOREPLACE: Final = 1
_AT_FDCWD: Final = -100


class FinalBundleError(RuntimeError):
    """Raised when a final bundle is incomplete, unsafe, or cannot be closed."""


class FinalBundleCancelledError(FinalBundleError):
    """Cooperative cancellation stopped construction before exclusive publication."""


def _check_cancellation(cancel_event: threading.Event | None, *, stage: str) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise FinalBundleCancelledError(f"final bundle construction cancelled before {stage}")


class FinalStatus(StrEnum):
    """Honest validation status labels allowed by the plan."""

    BASELINE_REPRODUCED = "baseline_reproduced"
    VALIDATION_IMPROVED = "validation_improved"
    MATERIALLY_CONFIRMED = "materially_confirmed"
    #: A controller-derived rank ensemble of already-qualified official FM seeds was
    #: validation-best, so it is what the campaign designates as final. It performs no new
    #: training and is not an agent result; the bundle says so rather than borrowing the
    #: vocabulary of one, and its confirmation shape names every seed it actually contains.
    ENSEMBLE_SELECTED = "ensemble_selected"


@dataclass(frozen=True, slots=True)
class FinalBundleSources:
    """Existing regular files and trees copied into the fixed final layout."""

    submission: Path
    report: Path
    experiments_jsonl: Path
    experiments_csv: Path
    environment: Path
    reproduce: Path
    config: Path
    source: Path
    model: Path
    preprocessing: Path
    validation_evidence: Path
    replay: Path

    def files(self) -> Mapping[str, Path]:
        return {
            "report.md": self.report,
            "submission.csv": self.submission,
            "experiments.jsonl": self.experiments_jsonl,
            "experiments.csv": self.experiments_csv,
            "environment.json": self.environment,
            "reproduce.sh": self.reproduce,
        }

    def directories(self) -> Mapping[str, Path]:
        return {
            "config": self.config,
            "source": self.source,
            "model": self.model,
            "preprocessing": self.preprocessing,
            "validation-evidence": self.validation_evidence,
            "replay": self.replay,
        }


@dataclass(frozen=True, slots=True)
class FinalBundleMetadata:
    """Scientific and operational facts required by ``manifest.json``."""

    benchmark_identity: Mapping[str, object]
    starter_identity: Mapping[str, object]
    data_identity: Mapping[str, object]
    selected_experiment: str
    lineage: tuple[str, ...]
    status: FinalStatus
    validation_metrics: Mapping[str, object]
    seed_summary: Mapping[str, object]
    inner_fold_results: Sequence[Mapping[str, object]]
    scientific_artifact_hashes: Mapping[str, str]
    environment_and_resource_usage: Mapping[str, object]
    campaign_totals: Mapping[str, object]
    known_limitations: tuple[str, ...] = ()
    unresolved_organizer_questions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BundleFileEvidence:
    """Exact identity of one non-manifest file in the final bundle."""

    relative_path: str
    component: str
    sha256: str
    size_bytes: int

    def manifest(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "component": self.component,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FinalBundleResult:
    """Identity of one exclusively published final bundle."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    submission_sha256: str
    file_count: int
    total_size_bytes: int


def _canonical_json(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FinalBundleError("final bundle metadata is not canonical JSON") from exc
    return (rendered + "\n").encode("ascii")


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not sys.platform.startswith(("darwin", "ios")):
        observed *= 1024
    return max(observed, 1)


def _prepublication_resource_receipt(
    *,
    started_monotonic_ns: int,
    started_cpu_ns: int,
) -> dict[str, object]:
    wall_seconds = max(time.monotonic_ns() - started_monotonic_ns, 0) / 1_000_000_000.0
    cpu_seconds = max(time.process_time_ns() - started_cpu_ns, 0) / 1_000_000_000.0
    return {
        "schema_version": 1,
        "scope": "final_bundle_prepublication",
        "coverage": {
            "began_at": "create_final_bundle_admission",
            "included_through": "scientific_identity_closure",
            "excluded_tail": [
                "receipt_and_manifest_serialization",
                "closed_file_verification_and_durability_flush",
                "exclusive_publication_and_postpublication_verification",
            ],
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "rss_semantics": "process_lifetime_peak_upper_bound",
        },
        "publication_state": "not_yet_published",
    }


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative(value: str) -> None:
    if not value or "\\" in value or "\x00" in value:
        raise FinalBundleError("bundle path must be a non-empty canonical POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FinalBundleError(f"bundle path is unsafe: {value!r}")
    if path.as_posix() != value:
        raise FinalBundleError(f"bundle path is not canonical: {value!r}")


def _validate_text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise FinalBundleError(f"{location} must be one non-empty line of text")
    return value


def _validate_json_mapping(value: Mapping[str, object], location: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise FinalBundleError(f"{location} must be a non-empty mapping")
    if any(type(key) is not str or not key for key in value):
        raise FinalBundleError(f"{location} keys must be non-empty strings")
    _canonical_json(dict(value))


def _metric(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FinalBundleError(f"{location} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise FinalBundleError(f"{location} must be finite in [0, 1]")
    return rendered


def _validate_metadata(
    metadata: FinalBundleMetadata, organizer_check: OrganizerCheckEvidence
) -> dict[str, str]:
    if not isinstance(metadata, FinalBundleMetadata):
        raise FinalBundleError("metadata must be FinalBundleMetadata")
    for name, identity in (
        ("benchmark_identity", metadata.benchmark_identity),
        ("starter_identity", metadata.starter_identity),
        ("data_identity", metadata.data_identity),
        ("seed_summary", metadata.seed_summary),
        ("environment_and_resource_usage", metadata.environment_and_resource_usage),
        ("campaign_totals", metadata.campaign_totals),
    ):
        _validate_json_mapping(identity, name)

    starter_digest = metadata.starter_identity.get("manifest_sha256")
    if starter_digest != organizer_check.starter_manifest_sha256:
        raise FinalBundleError("starter identity does not match the successful organizer check")
    selected = _validate_text(metadata.selected_experiment, "selected_experiment")
    if not metadata.lineage:
        raise FinalBundleError("selected candidate lineage must not be empty")
    lineage = tuple(
        _validate_text(experiment, f"lineage[{index}]")
        for index, experiment in enumerate(metadata.lineage)
    )
    if len(lineage) != len(set(lineage)) or lineage[-1] != selected:
        raise FinalBundleError("lineage must be unique, ordered, and end at selected_experiment")
    if not isinstance(metadata.status, FinalStatus):
        raise FinalBundleError("status must be an allowed FinalStatus")

    required_metrics = {"GAUC", "nDCG@5", "primary"}
    if not required_metrics.issubset(metadata.validation_metrics):
        raise FinalBundleError("validation_metrics must include GAUC, nDCG@5, and primary")
    gauc = _metric(metadata.validation_metrics["GAUC"], "validation_metrics.GAUC")
    ndcg = _metric(metadata.validation_metrics["nDCG@5"], "validation_metrics.nDCG@5")
    primary = _metric(metadata.validation_metrics["primary"], "validation_metrics.primary")
    if not math.isclose(primary, (gauc + ndcg) / 2.0, rel_tol=0.0, abs_tol=2e-7):
        raise FinalBundleError("validation primary must be the organizer mean of GAUC and nDCG@5")
    _canonical_json(dict(metadata.validation_metrics))
    _canonical_json([dict(result) for result in metadata.inner_fold_results])

    hashes = dict(metadata.scientific_artifact_hashes)
    missing_hashes = _REQUIRED_SCIENTIFIC_HASHES - hashes.keys()
    unknown_hashes = hashes.keys() - (_REQUIRED_SCIENTIFIC_HASHES | {"submission"})
    if missing_hashes or unknown_hashes:
        raise FinalBundleError(
            "scientific_artifact_hashes must contain exactly source, config, features, "
            "checkpoint, predictions, and optional submission"
        )
    for name, digest in hashes.items():
        if not _is_sha256(digest):
            raise FinalBundleError(f"scientific_artifact_hashes.{name} must be a SHA-256")
    supplied_submission = hashes.get("submission")
    if supplied_submission is not None and supplied_submission != organizer_check.submission_sha256:
        raise FinalBundleError("scientific submission hash does not match the organizer check")
    hashes["submission"] = organizer_check.submission_sha256

    missing_totals = _REQUIRED_TOTALS - metadata.campaign_totals.keys()
    if missing_totals:
        raise FinalBundleError(f"campaign_totals is missing {sorted(missing_totals)!r}")
    for name in _REQUIRED_TOTALS:
        value = metadata.campaign_totals[name]
        if name != "elapsed_seconds" and type(value) is not int:
            raise FinalBundleError(f"campaign_totals.{name} must be an integer")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise FinalBundleError(f"campaign_totals.{name} must be numeric")
        rendered = float(value)
        if not math.isfinite(rendered) or rendered < 0:
            raise FinalBundleError(f"campaign_totals.{name} must be finite and non-negative")

    for location, values in (
        ("known_limitations", metadata.known_limitations),
        ("unresolved_organizer_questions", metadata.unresolved_organizer_questions),
    ):
        if len(values) != len(set(values)):
            raise FinalBundleError(f"{location} entries must be unique")
        for index, value in enumerate(values):
            _validate_text(value, f"{location}[{index}]")
    return hashes


def _open_stable_regular(path: Path) -> tuple[int, os.stat_result]:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise FinalBundleError(f"bundle source is unavailable: {path}") from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise FinalBundleError(f"bundle source must be a regular non-symlink file: {path}")
    if initial.st_size <= 0:
        raise FinalBundleError(f"bundle source must not be empty: {path}")
    if initial.st_size > _MAX_FILE_BYTES:
        raise FinalBundleError(f"bundle source exceeds the file-size bound: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FinalBundleError(f"bundle source could not be opened safely: {path}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
        initial.st_dev,
        initial.st_ino,
    ):
        os.close(descriptor)
        raise FinalBundleError(f"bundle source changed before opening: {path}")
    return descriptor, opened


def _copy_file(
    source: Path, destination: Path, relative: str, component: str
) -> BundleFileEvidence:
    _safe_relative(relative)
    descriptor, opened = _open_stable_regular(source)
    digest = hashlib.sha256()
    size_bytes = 0
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        try:
            target = destination.open("xb")
        except OSError as exc:
            raise FinalBundleError(f"bundle staging path already exists: {relative}") from exc
        with target:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while chunk := handle.read(_COPY_CHUNK_BYTES):
                    target.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        final = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, field) != getattr(final, field) for field in fields):
            raise FinalBundleError(f"bundle source changed while being copied: {source}")
    finally:
        os.close(descriptor)
    destination.chmod(0o555 if relative == "reproduce.sh" else 0o444)
    return BundleFileEvidence(relative, component, digest.hexdigest(), size_bytes)


def _tree_snapshot(source: Path) -> tuple[str, ...]:
    try:
        root = source.lstat()
    except OSError as exc:
        raise FinalBundleError(f"bundle directory source is unavailable: {source}") from exc
    if stat.S_ISLNK(root.st_mode) or not stat.S_ISDIR(root.st_mode):
        raise FinalBundleError(f"bundle directory source must be a real directory: {source}")
    entries: list[str] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        _safe_relative(relative)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise FinalBundleError(f"bundle directory source contains a symlink: {relative}")
        if not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
            raise FinalBundleError(f"bundle directory source contains a special file: {relative}")
        entries.append(relative + ("/" if stat.S_ISDIR(metadata.st_mode) else ""))
    return tuple(entries)


def _copy_tree(source: Path, destination: Path, component: str) -> list[BundleFileEvidence]:
    before = _tree_snapshot(source)
    destination.mkdir(mode=0o700)
    evidence: list[BundleFileEvidence] = []
    for relative in before:
        stripped = relative.removesuffix("/")
        target = destination.joinpath(*PurePosixPath(stripped).parts)
        if relative.endswith("/"):
            target.mkdir(parents=True, exist_ok=False, mode=0o700)
            continue
        evidence.append(
            _copy_file(
                source.joinpath(*PurePosixPath(relative).parts),
                target,
                f"{component}/{relative}",
                component,
            )
        )
    if not evidence:
        raise FinalBundleError(f"required bundle component {component!r} contains no files")
    if _tree_snapshot(source) != before:
        raise FinalBundleError(f"bundle directory source changed while copied: {source}")
    return evidence


def _write_generated(
    destination: Path, relative: str, component: str, payload: bytes
) -> BundleFileEvidence:
    _safe_relative(relative)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FinalBundleError(f"could not create generated bundle file: {relative}") from exc
    destination.chmod(0o444)
    return BundleFileEvidence(
        relative,
        component,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


def _component_summaries(files: Sequence[BundleFileEvidence]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    required = (*REQUIRED_FILE_PATHS, *REQUIRED_DIRECTORY_PATHS)
    for component in required:
        selected = tuple(entry for entry in files if entry.component == component)
        if not selected:
            raise FinalBundleError(f"required component {component!r} has no retained file")
        payload = [
            entry.manifest() for entry in sorted(selected, key=lambda entry: entry.relative_path)
        ]
        summaries[component] = {
            "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
            "file_count": len(selected),
            "size_bytes": sum(entry.size_bytes for entry in selected),
        }
    return summaries


def _artifact_manifest_digest(
    files: Sequence[BundleFileEvidence],
    *,
    component: str,
    kind: str,
) -> str:
    selected = sorted(
        (entry for entry in files if entry.component == component),
        key=lambda entry: entry.relative_path,
    )
    if not selected:
        raise FinalBundleError(f"scientific component {component!r} is empty")
    prefix = f"{component}/"
    entries: list[dict[str, object]] = []
    for item in selected:
        if not item.relative_path.startswith(prefix):
            raise FinalBundleError(f"scientific component {component!r} path is malformed")
        relative = item.relative_path.removeprefix(prefix)
        _safe_relative(relative)
        entries.append(
            {
                "path": relative,
                "artifact": {
                    "schema_version": 1,
                    "algorithm": "sha256",
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "kind": kind,
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "total_size_bytes": sum(item.size_bytes for item in selected),
        "entries": entries,
    }
    return hashlib.sha256(_canonical_json(manifest).removesuffix(b"\n")).hexdigest()


def _scientific_component_digest(
    files: Sequence[BundleFileEvidence],
    *,
    component: str,
    kind: str,
    file_artifact_allowed: bool,
) -> str:
    selected = tuple(entry for entry in files if entry.component == component)
    if (
        file_artifact_allowed
        and len(selected) == 1
        and selected[0].relative_path == f"{component}/artifact"
    ):
        return selected[0].sha256
    return _artifact_manifest_digest(files, component=component, kind=kind)


def _file_evidence(
    files: Sequence[BundleFileEvidence],
    relative_path: str,
) -> BundleFileEvidence:
    selected = tuple(item for item in files if item.relative_path == relative_path)
    if len(selected) != 1:
        raise FinalBundleError(
            f"scientific evidence file is missing or duplicated: {relative_path}"
        )
    return selected[0]


def _canonical_staged_json(
    staging: Path,
    files: Sequence[BundleFileEvidence],
    relative_path: str,
) -> dict[str, object]:
    evidence = _file_evidence(files, relative_path)
    if not 0 < evidence.size_bytes <= _MAX_REPLAY_EVIDENCE_BYTES:
        raise FinalBundleError("replay evidence size is outside the trusted bound")
    path = staging.joinpath(*PurePosixPath(relative_path).parts)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise FinalBundleError(f"could not read copied canonical JSON: {relative_path}") from exc
    if (
        len(payload) != evidence.size_bytes
        or hashlib.sha256(payload).hexdigest() != evidence.sha256
    ):
        raise FinalBundleError("replay evidence bytes differ from the copied inventory")
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FinalBundleError("replay evidence is not canonical JSON") from exc
    if not isinstance(decoded, dict) or _canonical_json(decoded) != payload:
        raise FinalBundleError("replay evidence is not canonical JSON")
    return decoded


def _verify_scientific_identity_closure(
    staging: Path,
    files: Sequence[BundleFileEvidence],
    *,
    metadata: FinalBundleMetadata,
    scientific_hashes: Mapping[str, str],
) -> None:
    copied = {
        "source": _scientific_component_digest(
            files,
            component="source",
            kind="source",
            file_artifact_allowed=False,
        ),
        "config": _scientific_component_digest(
            files,
            component="config",
            kind="input",
            file_artifact_allowed=True,
        ),
        "features": _scientific_component_digest(
            files,
            component="preprocessing",
            kind="input",
            file_artifact_allowed=True,
        ),
        "checkpoint": _scientific_component_digest(
            files,
            component="model",
            kind="checkpoint",
            file_artifact_allowed=True,
        ),
        "predictions": _file_evidence(
            files,
            "validation-evidence/reference-validation-predictions.npy",
        ).sha256,
        "submission": _file_evidence(files, "submission.csv").sha256,
    }
    for name, observed in copied.items():
        if scientific_hashes.get(name) != observed:
            raise FinalBundleError(
                f"scientific_artifact_hashes.{name} differs from copied bundle evidence"
            )

    replay = _canonical_staged_json(staging, files, "replay/evidence.json")
    if replay.get("candidate_id") != metadata.selected_experiment:
        raise FinalBundleError("replay evidence candidate differs from selected_experiment")
    identity = replay.get("identity")
    required_identity = {
        "source_sha256",
        "config_sha256",
        "features_sha256",
        "checkpoint_sha256",
        "validation_prediction_artifact_sha256",
        "validation_prediction_digest",
        "data_sha256",
        "environment_sha256",
    }
    if not isinstance(identity, dict) or set(identity) != required_identity:
        raise FinalBundleError("replay evidence frozen identity is incomplete")
    for name, value in identity.items():
        if not _is_sha256(value):
            raise FinalBundleError(f"replay evidence identity {name} is not a SHA-256")
    replay_to_scientific = {
        "source_sha256": "source",
        "config_sha256": "config",
        "features_sha256": "features",
        "checkpoint_sha256": "checkpoint",
        "validation_prediction_artifact_sha256": "predictions",
    }
    for replay_name, scientific_name in replay_to_scientific.items():
        if identity[replay_name] != scientific_hashes[scientific_name]:
            raise FinalBundleError(
                f"replay evidence {replay_name} differs from scientific artifact identity"
            )
    if identity["data_sha256"] != metadata.data_identity.get("canonical_digest"):
        raise FinalBundleError(
            "replay evidence data identity differs from data_identity.canonical_digest"
        )
    if identity["environment_sha256"] != metadata.environment_and_resource_usage.get(
        "environment_sha256"
    ):
        raise FinalBundleError(
            "replay evidence environment identity differs from declared environment identity"
        )

    runtime_identity = metadata.environment_and_resource_usage.get("runtime_identity")
    required_runtime_identity = {
        "schema_version",
        "project_source_digest",
        "environment_digest",
        "uv_lock_sha256",
        "dependency_groups",
    }
    if (
        not isinstance(runtime_identity, Mapping)
        or set(runtime_identity) != required_runtime_identity
    ):
        raise FinalBundleError("runtime_identity must be the exact trusted runtime identity object")
    if runtime_identity.get("schema_version") != 1:
        raise FinalBundleError("runtime_identity.schema_version must be 1")
    for name in ("project_source_digest", "environment_digest", "uv_lock_sha256"):
        if not _is_sha256(runtime_identity.get(name)):
            raise FinalBundleError(f"runtime_identity.{name} must be a SHA-256")
    dependency_groups = runtime_identity.get("dependency_groups")
    if dependency_groups not in (
        ["research-tree"],
        ["research-tree", "research-neural"],
    ):
        raise FinalBundleError("runtime_identity dependency_groups are unsupported")
    if runtime_identity["environment_digest"] != identity["environment_sha256"]:
        raise FinalBundleError("runtime environment identity differs from replay identity")

    environment = _canonical_staged_json(staging, files, "environment.json")
    try:
        copied_environment_digest = environment_identity_digest(environment)
    except ReplayError as exc:
        raise FinalBundleError(
            "copied environment.json has an invalid provenance identity"
        ) from exc
    if copied_environment_digest != runtime_identity["environment_digest"]:
        raise FinalBundleError("runtime environment identity differs from copied environment.json")
    if environment.get("uv_lock_sha256") != runtime_identity["uv_lock_sha256"]:
        raise FinalBundleError("runtime uv_lock identity differs from copied environment.json")

    validation = replay.get("validation")
    final = replay.get("final")
    if not isinstance(validation, Mapping) or not isinstance(final, Mapping):
        raise FinalBundleError("replay validation and final evidence must be objects")
    if validation.get("reference_prediction_digest") != identity["validation_prediction_digest"]:
        raise FinalBundleError("replay reference prediction semantics differ from frozen identity")
    evidence_links = (
        (
            "validation prediction file",
            validation.get("replay_prediction_file_sha256"),
            _file_evidence(files, "replay/validation-predictions.npy").sha256,
        ),
        (
            "public-validation CSV",
            validation.get("public_submission_sha256"),
            _file_evidence(files, "validation-evidence/public-validation.csv").sha256,
        ),
        (
            "final prediction file",
            final.get("prediction_file_sha256"),
            _file_evidence(files, "replay/final-predictions.npy").sha256,
        ),
        (
            "final submission",
            final.get("submission_sha256"),
            scientific_hashes["submission"],
        ),
    )
    for location, declared, observed in evidence_links:
        if declared != observed:
            raise FinalBundleError(f"replay {location} identity differs from copied bytes")
    if validation.get("public_submission_prediction_digest") != validation.get(
        "replay_prediction_digest"
    ):
        raise FinalBundleError(
            "replay public-validation CSV semantics differ from replayed predictions"
        )
    if final.get("submission_prediction_digest") != final.get("prediction_digest"):
        raise FinalBundleError("replay final CSV semantics differ from final predictions")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_exclusive(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:  # pragma: no cover - supported macOS contract.
            raise FinalBundleError("renamex_np is unavailable for exclusive bundle publication")
        renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex.restype = ctypes.c_int
        result = int(renamex(source_bytes, destination_bytes, 0x00000004))
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - modern glibc contract.
            raise FinalBundleError("renameat2 is unavailable for exclusive bundle publication")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = int(
            renameat2(
                _AT_FDCWD,
                source_bytes,
                _AT_FDCWD,
                destination_bytes,
                _RENAME_NOREPLACE,
            )
        )
    else:  # pragma: no cover - supported reference platforms are macOS and Linux.
        raise FinalBundleError("platform lacks atomic no-overwrite directory publication")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise FinalBundleError(f"final bundle destination already exists: {destination}")
    raise FinalBundleError(f"could not publish final bundle: {os.strerror(error)}")


def _verify_closed_files(root: Path, evidence: Sequence[BundleFileEvidence]) -> None:
    expected = {entry.relative_path for entry in evidence} | {"manifest.json"}
    observed: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise FinalBundleError("closed bundle contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise FinalBundleError("closed bundle contains a special file")
        observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise FinalBundleError("closed bundle member set differs from its manifest")
    for entry in evidence:
        path = root.joinpath(*PurePosixPath(entry.relative_path).parts)
        if sha256_file(path) != entry.sha256:
            raise FinalBundleError(f"closed bundle digest mismatch: {entry.relative_path}")


def create_final_bundle(
    destination: str | Path,
    *,
    sources: FinalBundleSources,
    metadata: FinalBundleMetadata,
    organizer_check: OrganizerCheckEvidence,
    cancel_event: threading.Event | None = None,
) -> FinalBundleResult:
    """Create and exclusively publish the fixed ``final/`` bundle layout."""

    started_monotonic_ns = time.monotonic_ns()
    started_cpu_ns = time.process_time_ns()
    if not isinstance(sources, FinalBundleSources):
        raise FinalBundleError("sources must be FinalBundleSources")
    if not isinstance(organizer_check, OrganizerCheckEvidence):
        raise FinalBundleError("organizer_check must be successful OrganizerCheckEvidence")
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise FinalBundleError("cancel_event must be threading.Event or None")
    _check_cancellation(cancel_event, stage="bundle staging")
    if organizer_check.checker_returncode != 0 or organizer_check.checker_command[-1] != "--check":
        raise FinalBundleError("organizer check evidence is not a successful check-only result")
    if "--score" in organizer_check.checker_command:
        raise FinalBundleError("score-mode organizer evidence is forbidden")
    scientific_hashes = _validate_metadata(metadata, organizer_check)

    target = Path(destination)
    if target.name in {"", ".", ".."}:
        raise FinalBundleError("final bundle destination must name one directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_metadata = target.parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise FinalBundleError("final bundle parent must be a real directory")
    if os.path.lexists(target):
        raise FinalBundleError(f"final bundle destination already exists: {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    os.chmod(staging, 0o700, follow_symlinks=False)
    published = False
    try:
        evidence: list[BundleFileEvidence] = []
        for relative, source in sources.files().items():
            _check_cancellation(cancel_event, stage=f"copying {relative}")
            evidence.append(_copy_file(source, staging / relative, relative, relative))
        submission_entry = next(
            entry for entry in evidence if entry.relative_path == "submission.csv"
        )
        if (
            submission_entry.sha256 != organizer_check.submission_sha256
            or submission_entry.size_bytes != organizer_check.submission_size_bytes
        ):
            raise FinalBundleError("submission bytes changed after the successful organizer check")

        for component, source in sources.directories().items():
            _check_cancellation(cancel_event, stage=f"copying {component}")
            evidence.extend(_copy_tree(source, staging / component, component))

        verification_payload = _canonical_json(
            {
                "schema_version": FINAL_BUNDLE_SCHEMA_VERSION,
                "organizer_check": organizer_check.manifest(),
                "assertions": {
                    "checker_mode": "check_only",
                    "checker_split": "test",
                    "final_outcomes_masked_before_organizer_load": True,
                    "private_masked_view_deleted": True,
                    "starter_manifest_unchanged": True,
                    "submission_bytes_reverified": True,
                },
            }
        )
        evidence.append(
            _write_generated(
                staging / "verification.json",
                "verification.json",
                "verification.json",
                verification_payload,
            )
        )
        _verify_scientific_identity_closure(
            staging,
            evidence,
            metadata=metadata,
            scientific_hashes=scientific_hashes,
        )
        _check_cancellation(cancel_event, stage="prepublication resource receipt")
        prepublication_receipt = _write_generated(
            staging / "prepublication-resource.json",
            "prepublication-resource.json",
            "prepublication-resource.json",
            _canonical_json(
                _prepublication_resource_receipt(
                    started_monotonic_ns=started_monotonic_ns,
                    started_cpu_ns=started_cpu_ns,
                )
            ),
        )
        evidence.append(prepublication_receipt)
        evidence.sort(key=lambda entry: entry.relative_path)
        total_size = sum(entry.size_bytes for entry in evidence)
        if total_size > _MAX_BUNDLE_BYTES:
            raise FinalBundleError("final bundle exceeds the aggregate size bound")
        components = _component_summaries(evidence)

        manifest = {
            "schema_version": FINAL_BUNDLE_SCHEMA_VERSION,
            "benchmark_identity": dict(metadata.benchmark_identity),
            "starter_identity": dict(metadata.starter_identity),
            "data_identity": dict(metadata.data_identity),
            "selection": {
                "selected_experiment": metadata.selected_experiment,
                "lineage": list(metadata.lineage),
                "status": metadata.status.value,
            },
            "validation": {
                "metrics": dict(metadata.validation_metrics),
                "seed_summary": dict(metadata.seed_summary),
                "inner_fold_results": [dict(result) for result in metadata.inner_fold_results],
            },
            "scientific_artifact_hashes": scientific_hashes,
            "prepublication_resource_receipt": {
                "path": prepublication_receipt.relative_path,
                "sha256": prepublication_receipt.sha256,
            },
            "environment_and_resource_usage": dict(metadata.environment_and_resource_usage),
            "campaign_totals": dict(metadata.campaign_totals),
            "components": {
                "required_paths": [
                    "manifest.json",
                    *REQUIRED_FILE_PATHS,
                    *REQUIRED_DIRECTORY_PATHS,
                ],
                "roots": components,
                "files": [entry.manifest() for entry in evidence],
            },
            "known_limitations": list(metadata.known_limitations),
            "unresolved_organizer_questions": list(metadata.unresolved_organizer_questions),
        }
        manifest_payload = _canonical_json(manifest)
        manifest_entry = _write_generated(
            staging / "manifest.json", "manifest.json", "manifest.json", manifest_payload
        )
        _verify_closed_files(staging, evidence)
        _check_cancellation(cancel_event, stage="bundle durability flush")
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()), reverse=True
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        _check_cancellation(cancel_event, stage="exclusive bundle publication")
        _rename_exclusive(staging, target)
        published = True
        _fsync_directory(target.parent)
        _verify_closed_files(target, evidence)
        manifest_path = target / "manifest.json"
        if sha256_file(manifest_path) != manifest_entry.sha256:
            raise FinalBundleError("published bundle manifest digest mismatch")
        return FinalBundleResult(
            root=target.resolve(),
            manifest_path=manifest_path.resolve(),
            manifest_sha256=manifest_entry.sha256,
            submission_sha256=submission_entry.sha256,
            file_count=len(evidence) + 1,
            total_size_bytes=total_size + manifest_entry.size_bytes,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "FINAL_BUNDLE_SCHEMA_VERSION",
    "REQUIRED_DIRECTORY_PATHS",
    "REQUIRED_FILE_PATHS",
    "BundleFileEvidence",
    "FinalBundleCancelledError",
    "FinalBundleError",
    "FinalBundleMetadata",
    "FinalBundleResult",
    "FinalBundleSources",
    "FinalStatus",
    "create_final_bundle",
]
