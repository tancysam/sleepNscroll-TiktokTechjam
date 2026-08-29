"""Durable lifecycle controller for one autonomous KuaiRand-Pure campaign.

This module owns the small trusted vertical slice around an already-qualified official FM:

* create one immutable run without replacing any existing path;
* bind the normalized configuration and every scientific/environment identity;
* import the exact six qualification launches and install the seed-4 FM fallback;
* preserve the original monotonic/UTC deadline across process and machine restarts;
* conservatively reconcile abandoned ``STARTING``/``RUNNING`` executions; and
* report a stable, read-only status projection.

The research loop intentionally remains separate.  It uses the campaign store's explicit APIs
for experiments, executions, metrics, convergence, and incumbent promotion.  Resume never adopts
an old process and never changes the incumbent.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Self, cast

from kuairand_agent.campaign.budgets import MAX_TRAINING_LAUNCHES
from kuairand_agent.campaign.clock import (
    MIN_FINALIZATION_RESERVE_SECONDS,
    Clock,
    DeadlineObservation,
    DeadlineState,
    SystemClock,
)
from kuairand_agent.campaign.convergence import ConvergenceState
from kuairand_agent.campaign.models import CampaignState
from kuairand_agent.campaign.store import (
    ArtifactSpec,
    CampaignIdentityRecord,
    CampaignSnapshot,
    CampaignStore,
    ExecutionRecord,
    LaunchRecord,
)
from kuairand_agent.config import AgentConfig, parse_config
from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
)
from kuairand_agent.execution.runner import (
    ProcessRecord,
    ReconciliationExpectation,
    ReconciliationResult,
    Runner,
)

CONTROLLER_SCHEMA_VERSION: Final = 1
CAMPAIGN_DATABASE_NAME: Final = "campaign.sqlite3"
CONTROLLER_DIRECTORY_NAME: Final = "controller"
REQUEST_MANIFEST_NAME: Final = "create-request.json"
RUN_MANIFEST_NAME: Final = "run-manifest.json"
DEADLINE_DIRECTORY_NAME: Final = "deadline"
RECONCILIATION_DIRECTORY_NAME: Final = "reconciliations"
MAX_CONTROLLER_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_QUALIFICATION_MANIFEST_BYTES: Final = 64 * 1024 * 1024
_AT_FDCWD: Final = -2
_RENAME_NOREPLACE: Final = 1
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_QUALIFICATION_PATTERN: Final = (
    (1, "official_fm_training", 0),
    (2, "official_fm_training", 1),
    (3, "official_fm_training", 2),
    (4, "official_fm_training", 3),
    (5, "official_fm_training", 4),
    (6, "clean_source_retrain", 0),
)


class CampaignControllerError(RuntimeError):
    """Base class for trusted campaign lifecycle failures."""


class CampaignRunExistsError(CampaignControllerError):
    """Raised when create would replace an existing run path."""


class CampaignIntegrityError(CampaignControllerError):
    """Raised when persisted campaign evidence is missing, corrupt, or inconsistent."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CampaignIntegrityError("campaign evidence must be finite canonical JSON") from exc


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _thaw_json(value: object) -> object:
    """Convert an immutable store projection back to ordinary JSON container types."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise CampaignIntegrityError("store projection contains a non-JSON value")


def _signed_manifest(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if "digest" in result:
        raise CampaignIntegrityError("unsigned manifest body cannot already contain digest")
    result["digest"] = _canonical_digest(result)
    return result


def _verify_signed_manifest(
    value: Mapping[str, object], *, expected_keys: frozenset[str], location: str
) -> dict[str, object]:
    if set(value) != {*expected_keys, "digest"}:
        raise CampaignIntegrityError(f"{location} has missing or unknown fields")
    digest = _require_digest(value["digest"], f"{location}.digest")
    body = {key: item for key, item in value.items() if key != "digest"}
    if _canonical_digest(body) != digest:
        raise CampaignIntegrityError(f"{location} digest does not match its contents")
    return body


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignIntegrityError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CampaignIntegrityError(f"non-finite JSON number {value!r} is forbidden")


def _read_regular_file(path: Path, *, max_bytes: int, location: str) -> bytes:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise CampaignIntegrityError("internal manifest byte ceiling is invalid")
    try:
        before = path.lstat()
    except OSError as exc:
        raise CampaignIntegrityError(f"{location} is missing or unreadable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CampaignIntegrityError(f"{location} must be a regular non-symlink file")
    if before.st_nlink != 1:
        raise CampaignIntegrityError(f"{location} must not be hardlinked")
    if before.st_size > max_bytes:
        raise CampaignIntegrityError(f"{location} exceeds its byte ceiling")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CampaignIntegrityError(f"{location} could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise CampaignIntegrityError(f"{location} changed before it was opened")
        chunks: list[bytes] = []
        consumed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - consumed))
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > max_bytes:
                raise CampaignIntegrityError(f"{location} exceeds its byte ceiling")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, field) != getattr(after, field) for field in stable):
            raise CampaignIntegrityError(f"{location} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, max_bytes: int, location: str) -> dict[str, object]:
    payload = _read_regular_file(path, max_bytes=max_bytes, location=location)
    try:
        decoded: object = json.loads(
            payload,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CampaignIntegrityError(f"{location} is not strict JSON") from exc
    if not isinstance(decoded, dict) or any(type(key) is not str for key in decoded):
        raise CampaignIntegrityError(f"{location} must contain one JSON object")
    value = cast(dict[str, object], decoded)
    if payload != _canonical_json(value) + b"\n":
        raise CampaignIntegrityError(f"{location} is not canonical JSON with one newline")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_managed_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CampaignIntegrityError(f"managed path must be a real directory: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise CampaignIntegrityError(f"managed directory must be private: {path}")


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    _ensure_managed_directory(path.parent)
    payload = _canonical_json(dict(value)) + b"\n"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError as exc:
        raise CampaignIntegrityError(
            f"immutable controller evidence already exists: {path}"
        ) from exc
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise CampaignIntegrityError(f"short write while persisting {path}")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _require_text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise CampaignIntegrityError(f"{location} must be non-empty text without NUL")
    return value


def _require_digest(value: object, location: str) -> str:
    text = _require_text(value, location)
    if _DIGEST_RE.fullmatch(text) is None:
        raise CampaignIntegrityError(f"{location} must be a lowercase SHA-256 digest")
    return text


def _require_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CampaignIntegrityError(f"{location} must be an object")
    return cast(Mapping[str, object], value)


def _require_list(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise CampaignIntegrityError(f"{location} must be an array")
    return cast(list[object], value)


def _require_int(value: object, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignIntegrityError(f"{location} must be an integer >= {minimum}")
    return value


def _require_float(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignIntegrityError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignIntegrityError(f"{location} must be finite")
    return result


def _require_absolute_path(value: Path, location: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise CampaignIntegrityError(f"{location} must be an absolute Path")
    return value


@dataclass(frozen=True, slots=True)
class FallbackIdentity:
    """All immutable seed-4 FM identities required to reconstruct the fallback."""

    manifest_digest: str
    source_digest: str
    checkpoint_digest: str
    artifact_closure_digest: str
    config_digest: str
    encoding_digest: str
    outer_primary: float
    seed: int = 4

    def __post_init__(self) -> None:
        for name in (
            "manifest_digest",
            "source_digest",
            "checkpoint_digest",
            "artifact_closure_digest",
            "config_digest",
            "encoding_digest",
        ):
            _require_digest(getattr(self, name), f"fallback.{name}")
        if self.seed != 4:
            raise CampaignIntegrityError("the immutable fallback must be official FM seed 4")
        primary = _require_float(self.outer_primary, "fallback.outer_primary")
        if not 0.0 <= primary <= 1.0:
            raise CampaignIntegrityError("fallback.outer_primary must be in [0, 1]")

    def manifest(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "manifest_digest": self.manifest_digest,
            "source_digest": self.source_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "artifact_closure_digest": self.artifact_closure_digest,
            "config_digest": self.config_digest,
            "encoding_digest": self.encoding_digest,
            "outer_primary": self.outer_primary,
        }

    @classmethod
    def from_manifest(cls, raw: Mapping[str, object]) -> Self:
        expected = {
            "seed",
            "manifest_digest",
            "source_digest",
            "checkpoint_digest",
            "artifact_closure_digest",
            "config_digest",
            "encoding_digest",
            "outer_primary",
        }
        if set(raw) != expected:
            raise CampaignIntegrityError("fallback identity has missing or unknown fields")
        seed = _require_int(raw["seed"], "fallback.seed")
        return cls(
            seed=seed,
            manifest_digest=_require_digest(raw["manifest_digest"], "fallback.manifest_digest"),
            source_digest=_require_digest(raw["source_digest"], "fallback.source_digest"),
            checkpoint_digest=_require_digest(
                raw["checkpoint_digest"], "fallback.checkpoint_digest"
            ),
            artifact_closure_digest=_require_digest(
                raw["artifact_closure_digest"], "fallback.artifact_closure_digest"
            ),
            config_digest=_require_digest(raw["config_digest"], "fallback.config_digest"),
            encoding_digest=_require_digest(raw["encoding_digest"], "fallback.encoding_digest"),
            outer_primary=_require_float(raw["outer_primary"], "fallback.outer_primary"),
        )


_REQUEST_BODY_KEYS: Final = frozenset(
    {
        "schema_version",
        "campaign_id",
        "run_dir",
        "qualification_run_dir",
        "qualification_manifest_digest",
        "benchmark_digest",
        "starter_manifest_digest",
        "dataset_manifest_digest",
        "source_digest",
        "environment_digest",
        "config_digest",
        "config",
        "fallback",
    }
)


@dataclass(frozen=True, slots=True)
class CampaignCreateRequest:
    """Complete immutable identity for creating one new campaign."""

    run_dir: Path
    campaign_id: str
    config: AgentConfig
    qualification_run_dir: Path
    qualification_manifest_digest: str
    fallback: FallbackIdentity
    benchmark_digest: str
    starter_manifest_digest: str
    dataset_manifest_digest: str
    source_digest: str
    environment_digest: str
    schema_version: int = CONTROLLER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_absolute_path(self.run_dir, "run_dir")
        _require_absolute_path(self.qualification_run_dir, "qualification_run_dir")
        _require_text(self.campaign_id, "campaign_id")
        if not isinstance(self.config, AgentConfig):
            raise CampaignIntegrityError("config must be a validated AgentConfig")
        if not isinstance(self.fallback, FallbackIdentity):
            raise CampaignIntegrityError("fallback must be FallbackIdentity")
        if self.schema_version != CONTROLLER_SCHEMA_VERSION:
            raise CampaignIntegrityError("unsupported campaign request schema_version")
        for name in (
            "qualification_manifest_digest",
            "benchmark_digest",
            "starter_manifest_digest",
            "dataset_manifest_digest",
            "source_digest",
            "environment_digest",
        ):
            _require_digest(getattr(self, name), name)
        if self.config.runner.finalization_reserve_seconds < MIN_FINALIZATION_RESERVE_SECONDS:
            raise CampaignIntegrityError(
                "campaign finalization reserve must be at least 600 seconds"
            )
        if self.config.benchmark.max_iterations > MAX_TRAINING_LAUNCHES:
            raise CampaignIntegrityError("campaign scientific iteration cap cannot exceed 50")
        run = self.run_dir
        qualification = self.qualification_run_dir
        if (
            run == qualification
            or run.is_relative_to(qualification)
            or qualification.is_relative_to(run)
        ):
            raise CampaignIntegrityError("run_dir and qualification_run_dir must be disjoint")

    def identity_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "run_dir": str(self.run_dir),
            "qualification_run_dir": str(self.qualification_run_dir),
            "qualification_manifest_digest": self.qualification_manifest_digest,
            "benchmark_digest": self.benchmark_digest,
            "starter_manifest_digest": self.starter_manifest_digest,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "source_digest": self.source_digest,
            "environment_digest": self.environment_digest,
            "config_digest": self.config.digest,
            "config": self.config.normalized(),
            "fallback": self.fallback.manifest(),
        }

    @property
    def digest(self) -> str:
        return _canonical_digest(self.identity_manifest())

    def manifest(self) -> dict[str, object]:
        return self.identity_manifest() | {"digest": self.digest}

    @classmethod
    def from_manifest(cls, raw: Mapping[str, object]) -> Self:
        body = _verify_signed_manifest(
            raw,
            expected_keys=_REQUEST_BODY_KEYS,
            location="campaign create request",
        )
        if body["schema_version"] != CONTROLLER_SCHEMA_VERSION:
            raise CampaignIntegrityError("unsupported campaign create request schema_version")
        config_raw = _require_mapping(body["config"], "request.config")
        try:
            config = parse_config(cast(Mapping[str, Any], config_raw))
        except ValueError as exc:
            raise CampaignIntegrityError("stored normalized configuration is invalid") from exc
        if config.digest != _require_digest(body["config_digest"], "request.config_digest"):
            raise CampaignIntegrityError("normalized configuration digest does not match")
        fallback = FallbackIdentity.from_manifest(
            _require_mapping(body["fallback"], "request.fallback")
        )
        return cls(
            schema_version=CONTROLLER_SCHEMA_VERSION,
            run_dir=Path(_require_text(body["run_dir"], "request.run_dir")),
            campaign_id=_require_text(body["campaign_id"], "request.campaign_id"),
            config=config,
            qualification_run_dir=Path(
                _require_text(body["qualification_run_dir"], "request.qualification_run_dir")
            ),
            qualification_manifest_digest=_require_digest(
                body["qualification_manifest_digest"],
                "request.qualification_manifest_digest",
            ),
            fallback=fallback,
            benchmark_digest=_require_digest(body["benchmark_digest"], "request.benchmark_digest"),
            starter_manifest_digest=_require_digest(
                body["starter_manifest_digest"], "request.starter_manifest_digest"
            ),
            dataset_manifest_digest=_require_digest(
                body["dataset_manifest_digest"], "request.dataset_manifest_digest"
            ),
            source_digest=_require_digest(body["source_digest"], "request.source_digest"),
            environment_digest=_require_digest(
                body["environment_digest"], "request.environment_digest"
            ),
        )


@dataclass(frozen=True, slots=True)
class _QualificationEvidence:
    root_manifest_path: Path
    fallback_manifest_path: Path
    fallback_model_path: Path
    launch_records: tuple[Mapping[str, object], ...]


def _logical_manifest_digest(value: Mapping[str, object], location: str) -> str:
    observed = _require_digest(value.get("digest"), f"{location}.digest")
    body = {key: item for key, item in value.items() if key != "digest"}
    if _canonical_digest(body) != observed:
        raise CampaignIntegrityError(f"{location} logical digest mismatch")
    return observed


def _hash_regular_file(path: Path, *, expected_size: int, location: str) -> str:
    payload = _read_regular_file(path, max_bytes=expected_size, location=location)
    if len(payload) != expected_size:
        raise CampaignIntegrityError(f"{location} size does not match qualification evidence")
    return hashlib.sha256(payload).hexdigest()


def _verify_fallback_tree(
    qualification_root: Path,
    model_root: Path,
    artifacts_raw: object,
    expected_tree_digest: str,
) -> None:
    artifacts = _require_list(artifacts_raw, "qualification.artifacts")
    expected: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(artifacts):
        item = _require_mapping(raw, f"qualification.artifacts[{index}]")
        if set(item) != {"path", "size", "sha256"}:
            raise CampaignIntegrityError("qualification artifact record has invalid fields")
        path = _require_text(item["path"], f"qualification.artifacts[{index}].path")
        if path in expected:
            raise CampaignIntegrityError("qualification artifact paths must be unique")
        size = _require_int(item["size"], f"qualification.artifacts[{index}].size")
        digest = _require_digest(item["sha256"], f"qualification.artifacts[{index}].sha256")
        if path.startswith("fallback/model/"):
            expected[path.removeprefix("fallback/model/")] = (size, digest)

    observed: list[dict[str, object]] = []
    for candidate in sorted(model_root.rglob("*")):
        relative = candidate.relative_to(model_root).as_posix()
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CampaignIntegrityError(f"fallback model contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CampaignIntegrityError(f"fallback model contains an unsafe file: {relative}")
        if relative not in expected:
            raise CampaignIntegrityError(f"fallback model file is absent from manifest: {relative}")
        size, digest = expected[relative]
        observed_digest = _hash_regular_file(
            candidate,
            expected_size=size,
            location=f"fallback model file {relative}",
        )
        if observed_digest != digest:
            raise CampaignIntegrityError(f"fallback model file digest changed: {relative}")
        observed.append({"path": relative, "size": size, "sha256": digest})
    if set(expected) != {cast(str, item["path"]) for item in observed}:
        raise CampaignIntegrityError("fallback model is missing a qualification artifact")
    if _canonical_digest(observed) != expected_tree_digest:
        raise CampaignIntegrityError("fallback model tree digest does not match seed-4 identity")
    resolved_root = qualification_root.resolve(strict=True)
    if not model_root.resolve(strict=True).is_relative_to(resolved_root):
        raise CampaignIntegrityError("fallback model resolves outside qualification directory")


def _verify_qualification(request: CampaignCreateRequest) -> _QualificationEvidence:
    root = request.qualification_run_dir
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise CampaignIntegrityError("qualification run directory is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise CampaignIntegrityError("qualification run path must be a real directory")
    root_manifest_path = root / "manifest.json"
    fallback_manifest_path = root / "fallback" / "manifest.json"
    fallback_model_path = root / "fallback" / "model"
    root_manifest = _read_json(
        root_manifest_path,
        max_bytes=MAX_QUALIFICATION_MANIFEST_BYTES,
        location="qualification root manifest",
    )
    if _logical_manifest_digest(root_manifest, "qualification root manifest") != (
        request.qualification_manifest_digest
    ):
        raise CampaignIntegrityError("qualification manifest digest differs from request")
    if root_manifest.get("schema_version") != 1:
        raise CampaignIntegrityError("qualification manifest schema_version must be 1")
    if root_manifest.get("status") != "baseline_reproduced":
        raise CampaignIntegrityError("qualification did not reproduce the official baseline")
    if root_manifest.get("double_build_identity") is not True:
        raise CampaignIntegrityError("qualification double-build identity is not proven")
    if root_manifest.get("benchmark_digest") != request.benchmark_digest:
        raise CampaignIntegrityError("qualification benchmark identity differs from request")

    final_period = _require_mapping(root_manifest.get("final_period"), "qualification.final_period")
    if (
        final_period.get("outcomes_accessed") is not False
        or final_period.get("outcomes_scored") is not False
        or final_period.get("target_capability") is not None
    ):
        raise CampaignIntegrityError("qualification final-period outcome boundary was violated")

    accounting = _require_mapping(
        root_manifest.get("launch_accounting"), "qualification.launch_accounting"
    )
    if accounting.get("charged_launches") != 6 or accounting.get("expected_launches") != 6:
        raise CampaignIntegrityError("qualification must charge exactly six launches")
    if (
        accounting.get("checkpoint_replays_charged") is not False
        or accounting.get("random_rungs_charged") is not False
        or accounting.get("popularity_rung_charged") is not False
    ):
        raise CampaignIntegrityError("qualification charged an unapproved non-training action")
    records_raw = _require_list(accounting.get("records"), "qualification.launch records")
    if len(records_raw) != 6:
        raise CampaignIntegrityError("qualification must contain exactly six launch records")
    records: list[Mapping[str, object]] = []
    for raw, (number, kind, seed) in zip(records_raw, _QUALIFICATION_PATTERN, strict=True):
        item = _require_mapping(raw, f"qualification launch {number}")
        if set(item) != {"launch_number", "kind", "seed", "charged"} or dict(item) != {
            "launch_number": number,
            "kind": kind,
            "seed": seed,
            "charged": True,
        }:
            raise CampaignIntegrityError("qualification launch pattern is not the frozen six")
        records.append(item)

    root_fallback = _require_mapping(root_manifest.get("fallback"), "qualification.fallback")
    fallback_manifest = _read_json(
        fallback_manifest_path,
        max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
        location="qualification fallback manifest",
    )
    if dict(root_fallback) != fallback_manifest:
        raise CampaignIntegrityError("root and standalone fallback manifests differ")
    fallback_digest = _logical_manifest_digest(fallback_manifest, "fallback manifest")
    fallback = request.fallback
    if fallback_digest != fallback.manifest_digest:
        raise CampaignIntegrityError("fallback manifest digest differs from request")
    expected_fallback_fields: tuple[tuple[str, object], ...] = (
        ("schema_version", 1),
        ("kind", "immutable_official_fm_fallback"),
        ("seed", 4),
        ("checkpoint_digest", fallback.checkpoint_digest),
        ("config_digest", fallback.config_digest),
        ("encoding_digest", fallback.encoding_digest),
        ("source_model_tree_digest", fallback.source_digest),
        ("fallback_model_tree_digest", fallback.artifact_closure_digest),
        ("replay_verified", True),
        ("clean_seed_zero_retrain_verified", True),
    )
    for key, expected_value in expected_fallback_fields:
        if fallback_manifest.get(key) != expected_value:
            raise CampaignIntegrityError(f"fallback {key} differs from the seed-4 request")
    validation_metrics = _require_mapping(
        fallback_manifest.get("validation_metrics"), "fallback.validation_metrics"
    )
    if _require_float(validation_metrics.get("primary"), "fallback validation primary") != (
        fallback.outer_primary
    ):
        raise CampaignIntegrityError("fallback primary metric differs from request")
    final_submission = _require_mapping(
        fallback_manifest.get("final_submission"), "fallback.final_submission"
    )
    if final_submission.get("final_outcomes_accessed") is not False:
        raise CampaignIntegrityError("fallback claims access to final outcomes")

    fm = _require_mapping(root_manifest.get("fm"), "qualification.fm")
    runs = _require_list(fm.get("runs"), "qualification.fm.runs")
    seed_four = [
        _require_mapping(item, "qualification.fm.runs item")
        for item in runs
        if isinstance(item, Mapping) and item.get("seed") == 4
    ]
    if len(seed_four) != 1:
        raise CampaignIntegrityError("qualification must contain exactly one seed-4 FM run")
    seed_run = seed_four[0]
    seed_identity: tuple[tuple[str, object], ...] = (
        ("starter_manifest_digest", request.starter_manifest_digest),
        ("checkpoint_digest", fallback.checkpoint_digest),
        ("config_digest", fallback.config_digest),
        ("encoding_digest", fallback.encoding_digest),
        ("organizer_parity_passed", True),
    )
    for key, expected_value in seed_identity:
        if seed_run.get(key) != expected_value:
            raise CampaignIntegrityError(f"seed-4 FM {key} differs from request")

    _verify_fallback_tree(
        root,
        fallback_model_path,
        root_manifest.get("artifacts"),
        fallback.artifact_closure_digest,
    )
    return _QualificationEvidence(
        root_manifest_path=root_manifest_path,
        fallback_manifest_path=fallback_manifest_path,
        fallback_model_path=fallback_model_path,
        launch_records=tuple(records),
    )


def _artifact_spec(ref: ArtifactRef, *, semantic_kind: str) -> ArtifactSpec:
    return ArtifactSpec(
        digest=ref.sha256,
        kind=semantic_kind,
        relative_path=ref.object_relative_path.as_posix(),
        size_bytes=ref.size_bytes,
        metadata=ref.manifest(),
    )


def _directory_artifact_spec(
    ref: DirectoryArtifactRef,
    *,
    semantic_kind: str,
    logical_closure_digest: str,
) -> ArtifactSpec:
    manifest_ref = ref.manifest_artifact
    return ArtifactSpec(
        digest=manifest_ref.sha256,
        kind=semantic_kind,
        relative_path=manifest_ref.object_relative_path.as_posix(),
        size_bytes=manifest_ref.size_bytes,
        metadata={
            "artifact": manifest_ref.manifest(),
            "directory": ref.manifest(),
            "logical_closure_digest": logical_closure_digest,
        },
    )


@dataclass(frozen=True, slots=True)
class _InstalledArtifacts:
    qualification_manifest: ArtifactRef
    fallback_manifest: ArtifactRef
    fallback_model_manifest: ArtifactRef
    effective_config: ArtifactRef
    create_request: ArtifactRef
    fallback_model_total_size_bytes: int

    def body(self, *, request: CampaignCreateRequest) -> dict[str, object]:
        return {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "request_digest": request.digest,
            "qualification_manifest_digest": request.qualification_manifest_digest,
            "fallback_logical_closure_digest": request.fallback.artifact_closure_digest,
            "fallback_model_total_size_bytes": self.fallback_model_total_size_bytes,
            "artifacts": {
                "qualification_manifest": self.qualification_manifest.manifest(),
                "fallback_manifest": self.fallback_manifest.manifest(),
                "fallback_model_manifest": self.fallback_model_manifest.manifest(),
                "effective_config": self.effective_config.manifest(),
                "create_request": self.create_request.manifest(),
            },
        }


_RUN_MANIFEST_BODY_KEYS: Final = frozenset(
    {
        "schema_version",
        "request_digest",
        "qualification_manifest_digest",
        "fallback_logical_closure_digest",
        "fallback_model_total_size_bytes",
        "artifacts",
    }
)
_RUN_ARTIFACT_ROLES: Final = frozenset(
    {
        "qualification_manifest",
        "fallback_manifest",
        "fallback_model_manifest",
        "effective_config",
        "create_request",
    }
)


def _load_run_manifest(run_dir: Path, request: CampaignCreateRequest) -> dict[str, ArtifactRef]:
    path = run_dir / CONTROLLER_DIRECTORY_NAME / RUN_MANIFEST_NAME
    raw = _read_json(path, max_bytes=MAX_CONTROLLER_MANIFEST_BYTES, location="run manifest")
    body = _verify_signed_manifest(
        raw,
        expected_keys=_RUN_MANIFEST_BODY_KEYS,
        location="run manifest",
    )
    if body["schema_version"] != CONTROLLER_SCHEMA_VERSION:
        raise CampaignIntegrityError("unsupported run manifest schema_version")
    if body["request_digest"] != request.digest:
        raise CampaignIntegrityError("run manifest request identity mismatch")
    if body["qualification_manifest_digest"] != request.qualification_manifest_digest:
        raise CampaignIntegrityError("run manifest qualification identity mismatch")
    if body["fallback_logical_closure_digest"] != request.fallback.artifact_closure_digest:
        raise CampaignIntegrityError("run manifest fallback closure identity mismatch")
    _require_int(
        body["fallback_model_total_size_bytes"],
        "run manifest fallback_model_total_size_bytes",
    )
    artifacts_raw = _require_mapping(body["artifacts"], "run manifest artifacts")
    if set(artifacts_raw) != _RUN_ARTIFACT_ROLES:
        raise CampaignIntegrityError("run manifest has missing or unknown artifact roles")
    result: dict[str, ArtifactRef] = {}
    for role in sorted(_RUN_ARTIFACT_ROLES):
        try:
            result[role] = ArtifactRef.from_manifest(
                _require_mapping(artifacts_raw[role], f"run artifact {role}")
            )
        except ValueError as exc:
            raise CampaignIntegrityError(f"run artifact {role} is invalid") from exc
    if result["fallback_model_manifest"].kind is not ArtifactKind.MANIFEST:
        raise CampaignIntegrityError("fallback model reference must name a directory manifest")
    return result


def _verify_installed_artifacts(
    run_dir: Path,
    request: CampaignCreateRequest,
    refs: Mapping[str, ArtifactRef],
) -> None:
    store = ArtifactStore(run_dir / "artifacts")
    for role, ref in refs.items():
        store.verify(ref)
        expected_kind = ArtifactKind.CHECKPOINT if role == "fallback_model_manifest" else None
        if expected_kind is None:
            continue
        directory = store.load_directory(ref)
        if directory.kind is not expected_kind:
            raise CampaignIntegrityError("fallback directory artifact kind changed")
        verified = store.verify_directory(directory)
        if verified.sha256 != ref.sha256:
            raise CampaignIntegrityError("fallback directory manifest identity changed")
    request_payload = store.read_bytes(
        refs["create_request"], max_bytes=MAX_CONTROLLER_MANIFEST_BYTES
    )
    if request_payload != _canonical_json(request.manifest()) + b"\n":
        raise CampaignIntegrityError(
            "content-addressed create request differs from controller copy"
        )
    config_payload = store.read_bytes(
        refs["effective_config"], max_bytes=MAX_CONTROLLER_MANIFEST_BYTES
    )
    if config_payload != _canonical_json(request.config.normalized()) + b"\n":
        raise CampaignIntegrityError("content-addressed effective config differs from request")


_DEADLINE_BODY_KEYS: Final = frozenset({"schema_version", "sequence", "previous_digest", "state"})
_DEADLINE_FILE_RE: Final = re.compile(r"checkpoint-(\d{8})\.json\Z")


@dataclass(frozen=True, slots=True)
class _DeadlineCheckpoint:
    sequence: int
    digest: str
    state: DeadlineState


def _deadline_checkpoint_body(
    *, sequence: int, previous_digest: str | None, state: DeadlineState
) -> dict[str, object]:
    return {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "sequence": sequence,
        "previous_digest": previous_digest,
        "state": state.manifest(),
    }


def _deadline_directory(run_dir: Path) -> Path:
    return run_dir / CONTROLLER_DIRECTORY_NAME / DEADLINE_DIRECTORY_NAME


def _write_initial_deadline(run_dir: Path, state: DeadlineState) -> _DeadlineCheckpoint:
    body = _deadline_checkpoint_body(sequence=0, previous_digest=None, state=state)
    manifest = _signed_manifest(body)
    path = _deadline_directory(run_dir) / "checkpoint-00000000.json"
    _write_new_json(path, manifest)
    return _DeadlineCheckpoint(sequence=0, digest=cast(str, manifest["digest"]), state=state)


def _load_deadline(run_dir: Path) -> _DeadlineCheckpoint:
    directory = _deadline_directory(run_dir)
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise CampaignIntegrityError("deadline evidence directory is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CampaignIntegrityError("deadline evidence path must be a real directory")
    entries = sorted(directory.iterdir())
    if not entries:
        raise CampaignIntegrityError("deadline evidence has no checkpoint")
    previous: _DeadlineCheckpoint | None = None
    immutable_fields = (
        "wall_clock_seconds",
        "finalization_reserve_seconds",
        "started_utc",
        "utc_deadline",
        "original_boot_identity",
        "monotonic_started_ns",
        "monotonic_deadline_ns",
        "schema_version",
    )
    for expected_sequence, path in enumerate(entries):
        match = _DEADLINE_FILE_RE.fullmatch(path.name)
        if match is None or int(match.group(1)) != expected_sequence:
            raise CampaignIntegrityError("deadline checkpoint names must be contiguous and exact")
        raw = _read_json(
            path,
            max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
            location=f"deadline checkpoint {expected_sequence}",
        )
        body = _verify_signed_manifest(
            raw,
            expected_keys=_DEADLINE_BODY_KEYS,
            location=f"deadline checkpoint {expected_sequence}",
        )
        if body["schema_version"] != CONTROLLER_SCHEMA_VERSION:
            raise CampaignIntegrityError("unsupported deadline checkpoint schema_version")
        if body["sequence"] != expected_sequence:
            raise CampaignIntegrityError("deadline checkpoint sequence mismatch")
        expected_previous = None if previous is None else previous.digest
        if body["previous_digest"] != expected_previous:
            raise CampaignIntegrityError("deadline checkpoint chain is broken")
        try:
            state = DeadlineState.from_manifest(
                _require_mapping(body["state"], "deadline checkpoint state")
            )
        except ValueError as exc:
            raise CampaignIntegrityError("deadline checkpoint state is invalid") from exc
        if previous is not None:
            if any(
                getattr(state, field) != getattr(previous.state, field)
                for field in immutable_fields
            ):
                raise CampaignIntegrityError("deadline checkpoint reset an immutable field")
            if state.last_elapsed_seconds < previous.state.last_elapsed_seconds:
                raise CampaignIntegrityError("deadline elapsed time moved backwards")
            if state.last_observed_utc < previous.state.last_observed_utc:
                raise CampaignIntegrityError("deadline UTC observation moved backwards")
        previous = _DeadlineCheckpoint(
            sequence=expected_sequence,
            digest=_require_digest(raw["digest"], "deadline checkpoint digest"),
            state=state,
        )
    assert previous is not None
    return previous


def _append_deadline(
    run_dir: Path, previous: _DeadlineCheckpoint, observation: DeadlineObservation
) -> _DeadlineCheckpoint:
    sequence = previous.sequence + 1
    body = _deadline_checkpoint_body(
        sequence=sequence,
        previous_digest=previous.digest,
        state=observation.state,
    )
    manifest = _signed_manifest(body)
    path = _deadline_directory(run_dir) / f"checkpoint-{sequence:08d}.json"
    _write_new_json(path, manifest)
    return _DeadlineCheckpoint(
        sequence=sequence,
        digest=cast(str, manifest["digest"]),
        state=observation.state,
    )


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise CampaignIntegrityError("controller timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: str, location: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignIntegrityError(f"{location} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CampaignIntegrityError(f"{location} must include a timezone")
    return parsed.astimezone(UTC)


def _verify_database_identity(
    identity: CampaignIdentityRecord,
    request: CampaignCreateRequest,
    deadline: DeadlineState,
) -> None:
    exact: tuple[tuple[str, object, object], ...] = (
        ("campaign_id", identity.campaign_id, request.campaign_id),
        ("config_digest", identity.config_digest, request.config.digest),
        ("benchmark_digest", identity.benchmark_digest, request.benchmark_digest),
        (
            "starter_manifest_digest",
            identity.starter_manifest_digest,
            request.starter_manifest_digest,
        ),
        (
            "dataset_manifest_digest",
            identity.dataset_manifest_digest,
            request.dataset_manifest_digest,
        ),
        ("source_digest", identity.source_digest, request.source_digest),
        ("environment_digest", identity.environment_digest, request.environment_digest),
        ("max_launches", identity.max_launches, MAX_TRAINING_LAUNCHES),
        (
            "outer_query_limit",
            identity.outer_query_limit,
            request.config.validation.outer_promotion_limit,
        ),
    )
    for name, observed, expected in exact:
        if observed != expected:
            raise CampaignIntegrityError(f"campaign database {name} differs from create request")
    if _parse_utc(identity.hard_deadline_utc, "database hard deadline") != deadline.utc_deadline:
        raise CampaignIntegrityError("campaign database hard deadline differs from deadline chain")
    if deadline.wall_clock_seconds != request.config.benchmark.wall_clock_seconds:
        raise CampaignIntegrityError("deadline wall-clock budget differs from create request")
    if deadline.finalization_reserve_seconds != request.config.runner.finalization_reserve_seconds:
        raise CampaignIntegrityError("deadline finalization reserve differs from create request")


def _verify_exact_qualification_launches(
    launches: Sequence[LaunchRecord], qualification_digest: str
) -> None:
    if len(launches) < 6:
        raise CampaignIntegrityError("campaign lost one or more qualification launches")
    for record, (number, kind, seed) in zip(launches[:6], _QUALIFICATION_PATTERN, strict=True):
        if (
            record.launch_id != f"qualification-{number:02d}"
            or record.launch_number != number
            or record.reservation_key != f"qualification:{qualification_digest}:{number}"
            or record.category != "baseline_qualification"
            or record.original_category != "baseline_qualification"
            or record.purpose != kind
            or record.seed != seed
            or record.state != "FINISHED"
            or record.charged is not True
        ):
            raise CampaignIntegrityError("stored qualification launch pattern changed")
    if any(record.launch_number <= 6 for record in launches[6:]):
        raise CampaignIntegrityError("campaign launch numbers are not strictly ordered")


def _validate_run_directory(run_dir: Path) -> Path:
    path = _require_absolute_path(run_dir, "run_dir")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CampaignIntegrityError(f"campaign run directory does not exist: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CampaignIntegrityError("campaign run path must be a real directory")
    return path


def _load_request(run_dir: Path) -> CampaignCreateRequest:
    raw = _read_json(
        run_dir / CONTROLLER_DIRECTORY_NAME / REQUEST_MANIFEST_NAME,
        max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
        location="campaign create request",
    )
    request = CampaignCreateRequest.from_manifest(raw)
    if request.run_dir != run_dir:
        raise CampaignIntegrityError("stored create request names a different run directory")
    return request


def _rename_directory_exclusive(source: Path, destination: Path) -> None:
    """Atomically install ``source`` without replacing any destination object."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:  # pragma: no cover - required by supported macOS.
            raise CampaignIntegrityError("renamex_np is unavailable")
        renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex.restype = ctypes.c_int
        result = int(renamex(source_bytes, destination_bytes, 0x00000004))
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - required by supported Linux.
            raise CampaignIntegrityError("renameat2 is unavailable")
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
    else:  # pragma: no cover - local reference platforms are macOS and Linux.
        raise CampaignIntegrityError("platform lacks an atomic no-overwrite directory rename")
    if result == 0:
        _fsync_directory(destination.parent)
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise CampaignRunExistsError(f"campaign run path already exists: {destination}")
    raise CampaignIntegrityError(f"cannot commit campaign run: {os.strerror(error)}")


def _reconciliation_directory(run_dir: Path) -> Path:
    return run_dir / CONTROLLER_DIRECTORY_NAME / RECONCILIATION_DIRECTORY_NAME


def _reconciliation_files(run_dir: Path) -> tuple[Path, ...]:
    directory = _reconciliation_directory(run_dir)
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise CampaignIntegrityError("reconciliation evidence directory is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CampaignIntegrityError("reconciliation evidence path must be a real directory")
    paths = tuple(sorted(directory.iterdir()))
    for sequence, path in enumerate(paths, start=1):
        if path.name != f"reconciliation-{sequence:08d}.json":
            raise CampaignIntegrityError("reconciliation evidence names must be contiguous")
        raw = _read_json(
            path,
            max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
            location=f"reconciliation evidence {sequence}",
        )
        if raw.get("sequence") != sequence:
            raise CampaignIntegrityError("reconciliation evidence sequence mismatch")
        digest = _require_digest(raw.get("digest"), "reconciliation evidence digest")
        body = {key: value for key, value in raw.items() if key != "digest"}
        if _canonical_digest(body) != digest:
            raise CampaignIntegrityError("reconciliation evidence digest mismatch")
    return paths


def _require_execution_digest(value: str | None, location: str) -> str:
    if value is None:
        raise CampaignIntegrityError(f"unfinished execution is missing {location}")
    return _require_digest(value, f"execution.{location}")


def _process_reconciliation(
    run_dir: Path,
    runner: Runner,
    record: ExecutionRecord,
) -> ReconciliationResult:
    if record.process_record is None or record.process_record_digest is None:
        raise CampaignIntegrityError("RUNNING execution is missing its process receipt")
    try:
        # Store projections recursively freeze JSON arrays as tuples.  Re-decode the already
        # canonical mapping so the runner receives the exact JSON array types its strict loader
        # requires; no value or member can change through this round trip.
        thawed: object = _thaw_json(record.process_record)
        process = ProcessRecord.from_manifest(
            _require_mapping(thawed, "RUNNING execution process receipt")
        )
    except ValueError as exc:
        raise CampaignIntegrityError("RUNNING execution process receipt is invalid") from exc
    if process.digest != record.process_record_digest:
        raise CampaignIntegrityError("RUNNING execution process receipt digest mismatch")
    nonce = record.nonce
    if nonce is None:
        raise CampaignIntegrityError("RUNNING execution has no immutable nonce")
    process_environment = _require_execution_digest(
        record.process_environment_digest,
        "process_environment_digest",
    )
    expectation = ReconciliationExpectation(
        execution_id=record.execution_id,
        nonce=nonce,
        command_digest=_require_execution_digest(
            record.process_command_digest,
            "process_command_digest",
        ),
        environment_digest=process_environment,
        interpreter_real_path=Path(process.interpreter_real_path),
        workspace=Path(process.workspace),
        control_dir=Path(process.control_dir),
        source_digest=_require_execution_digest(record.source_digest, "source_digest"),
        config_digest=_require_execution_digest(record.config_digest, "config_digest"),
        data_digest=_require_execution_digest(record.data_digest, "data_digest"),
        checkpoint_digest=_require_execution_digest(
            record.checkpoint_digest,
            "checkpoint_digest",
        ),
    )
    workspace_root = run_dir / "workspaces"
    control_root = run_dir / "controls"
    if not expectation.workspace.is_relative_to(workspace_root):
        raise CampaignIntegrityError("persisted execution workspace is outside the campaign")
    if not expectation.control_dir.is_relative_to(control_root):
        raise CampaignIntegrityError("persisted execution control path is outside the campaign")
    return runner.reconcile(process, expected=expectation)


def _finish_reconciled_launch(store: CampaignStore, launch_id: str | None) -> str | None:
    if launch_id is None:
        return None
    launches = {item.launch_id: item for item in store.launches()}
    launch = launches.get(launch_id)
    if launch is None:
        raise CampaignIntegrityError("unfinished execution references an unknown launch")
    if launch.state == "RESERVED":
        launch = store.transition_launch(
            launch_id,
            to_state="START_UNCERTAIN",
            expected_revision=store.snapshot().revision,
            metadata={"reason": "controller restart cannot prove candidate release"},
        )
    if launch.state in {"STARTED", "START_UNCERTAIN"}:
        launch = store.transition_launch(
            launch_id,
            to_state="FINISHED",
            expected_revision=store.snapshot().revision,
            metadata={"reason": "abandoned execution reconciled as interrupted"},
        )
    if launch.state not in {"FINISHED", "NOT_STARTED"}:
        raise CampaignIntegrityError(f"cannot reconcile launch state {launch.state!r}")
    return launch.state


@dataclass(frozen=True, slots=True)
class _ReconciliationOutcome:
    count: int
    unsafe_cleanup: bool


def _reconcile_unfinished(
    run_dir: Path,
    store: CampaignStore,
    runner: Runner,
    clock: Clock,
) -> _ReconciliationOutcome:
    existing = len(_reconciliation_files(run_dir))
    unsafe = False
    count = 0
    artifact_store = ArtifactStore(run_dir / "artifacts")
    for execution in store.unfinished_executions():
        incumbent_before = store.current_incumbent()
        if incumbent_before is None:
            raise CampaignIntegrityError("campaign has no replayable incumbent during resume")
        runner_result: ReconciliationResult | None = None
        if execution.status == "RUNNING":
            runner_result = _process_reconciliation(run_dir, runner, execution)
            unsafe = unsafe or runner_result.outcome.value in {
                "inspection_failed",
                "cleanup_failed",
            }
        elif execution.status != "STARTING":  # pragma: no cover - store query is exact.
            raise CampaignIntegrityError("store returned a non-active unfinished execution")

        transition_body: dict[str, object] = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "execution_id": execution.execution_id,
            "from_state": execution.status,
            "to_state": "INTERRUPTED",
            "incumbent_before": incumbent_before.incumbent_id,
            "runner_reconciliation": (None if runner_result is None else runner_result.manifest()),
        }
        transition_manifest = _signed_manifest(transition_body)
        transition_ref = artifact_store.put_bytes(
            _canonical_json(transition_manifest) + b"\n",
            kind=ArtifactKind.MANIFEST,
            max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
        )
        now = clock.utc_now()
        if execution.started_at is not None:
            started = _parse_utc(execution.started_at, "execution started_at")
            if now.astimezone(UTC) < started:
                now = started
        store.transition_execution(
            execution.execution_id,
            from_state=execution.status,
            to_state="INTERRUPTED",
            expected_revision=store.snapshot().revision,
            reason="controller restart conservatively interrupts abandoned execution",
            finished_at=_utc_timestamp(now),
            artifacts=(
                (
                    "resume_reconciliation",
                    _artifact_spec(
                        transition_ref,
                        semantic_kind="resume_reconciliation",
                    ),
                ),
            ),
            metadata={
                "runner_outcome": (
                    "not_released_or_unknown"
                    if runner_result is None
                    else runner_result.outcome.value
                ),
                "incumbent_preserved": incumbent_before.incumbent_id,
            },
        )
        launch_state = _finish_reconciled_launch(store, execution.launch_id)
        incumbent_after = store.current_incumbent()
        if incumbent_after is None or incumbent_after.incumbent_id != incumbent_before.incumbent_id:
            raise CampaignIntegrityError("restart reconciliation changed the incumbent")
        count += 1
        sequence = existing + count
        report = _signed_manifest(
            {
                "schema_version": CONTROLLER_SCHEMA_VERSION,
                "sequence": sequence,
                "execution_id": execution.execution_id,
                "execution_from_state": execution.status,
                "execution_to_state": "INTERRUPTED",
                "launch_id": execution.launch_id,
                "launch_state": launch_state,
                "incumbent_before": incumbent_before.incumbent_id,
                "incumbent_after": incumbent_after.incumbent_id,
                "transition_evidence_digest": transition_ref.sha256,
                "runner_reconciliation": (
                    None if runner_result is None else runner_result.manifest()
                ),
            }
        )
        _write_new_json(
            _reconciliation_directory(run_dir) / f"reconciliation-{sequence:08d}.json",
            report,
        )
    # A controller can crash after atomically committing the terminal execution transition but
    # before appending the matching launch event.  Such executions are no longer returned by
    # unfinished_executions(), yet their conservative launch charge still needs a durable terminal
    # event.  Never infer NOT_STARTED from this split state: without the live supervising runner,
    # candidate release is uncertain.
    launches = {item.launch_id: item for item in store.launches()}
    for execution in store.executions():
        if execution.status in {"STARTING", "RUNNING"} or execution.launch_id is None:
            continue
        launch = launches.get(execution.launch_id)
        if launch is None:
            raise CampaignIntegrityError("terminal execution references an unknown launch")
        if launch.state in {"FINISHED", "NOT_STARTED"}:
            continue
        incumbent_before = store.current_incumbent()
        if incumbent_before is None:
            raise CampaignIntegrityError("campaign has no incumbent during launch reconciliation")
        launch_state = _finish_reconciled_launch(store, execution.launch_id)
        incumbent_after = store.current_incumbent()
        if incumbent_after is None or incumbent_after.incumbent_id != incumbent_before.incumbent_id:
            raise CampaignIntegrityError("split-state launch reconciliation changed the incumbent")
        count += 1
        sequence = existing + count
        report = _signed_manifest(
            {
                "schema_version": CONTROLLER_SCHEMA_VERSION,
                "sequence": sequence,
                "execution_id": execution.execution_id,
                "execution_from_state": execution.status,
                "execution_to_state": execution.status,
                "launch_id": execution.launch_id,
                "launch_state": launch_state,
                "incumbent_before": incumbent_before.incumbent_id,
                "incumbent_after": incumbent_after.incumbent_id,
                "transition_evidence_digest": None,
                "runner_reconciliation": None,
            }
        )
        _write_new_json(
            _reconciliation_directory(run_dir) / f"reconciliation-{sequence:08d}.json",
            report,
        )
    # The runner's blocked-launch callback is deliberately narrow, but a hard controller kill can
    # still land after a reservation and before the execution row.  With no durable positive proof
    # that the process never started, retain the charge and close the orphan launch uncertainly.
    linked_launches = {
        execution.launch_id for execution in store.executions() if execution.launch_id is not None
    }
    for launch in store.launches():
        if (
            launch.launch_number <= 6
            or launch.launch_id in linked_launches
            or launch.state in {"FINISHED", "NOT_STARTED"}
        ):
            continue
        incumbent_before = store.current_incumbent()
        if incumbent_before is None:
            raise CampaignIntegrityError("campaign has no incumbent during orphan reconciliation")
        launch_state = _finish_reconciled_launch(store, launch.launch_id)
        incumbent_after = store.current_incumbent()
        if incumbent_after is None or incumbent_after.incumbent_id != incumbent_before.incumbent_id:
            raise CampaignIntegrityError("orphan launch reconciliation changed the incumbent")
        count += 1
        sequence = existing + count
        report = _signed_manifest(
            {
                "schema_version": CONTROLLER_SCHEMA_VERSION,
                "sequence": sequence,
                "execution_id": None,
                "execution_from_state": None,
                "execution_to_state": None,
                "launch_id": launch.launch_id,
                "launch_state": launch_state,
                "incumbent_before": incumbent_before.incumbent_id,
                "incumbent_after": incumbent_after.incumbent_id,
                "transition_evidence_digest": None,
                "runner_reconciliation": None,
            }
        )
        _write_new_json(
            _reconciliation_directory(run_dir) / f"reconciliation-{sequence:08d}.json",
            report,
        )
    return _ReconciliationOutcome(count=count, unsafe_cleanup=unsafe)


@dataclass(frozen=True, slots=True)
class CampaignStatus:
    """Stable, JSON-compatible controller projection returned by all lifecycle operations."""

    campaign_id: str
    request_digest: str
    revision: int
    status: str
    phase: str
    launches_used: int
    launches_remaining: int
    scientific_iterations: int
    scientific_iterations_remaining: int
    outer_queries_used: int
    outer_queries_remaining: int
    qualification_manifest_digest: str
    incumbent_id: str
    incumbent_is_fallback: bool
    incumbent_replay_verified: bool
    deadline_elapsed_seconds: float
    deadline_remaining_seconds: float
    finalization_reserve_seconds: int
    finalization_required: bool
    unfinished_execution_ids: tuple[str, ...]
    reconciliation_count: int
    store_schema_digest: str
    store_catalog_digest: str
    schema_version: int = CONTROLLER_SCHEMA_VERSION

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "request_digest": self.request_digest,
            "revision": self.revision,
            "status": self.status,
            "phase": self.phase,
            "launches_used": self.launches_used,
            "launches_remaining": self.launches_remaining,
            "scientific_iterations": self.scientific_iterations,
            "scientific_iterations_remaining": self.scientific_iterations_remaining,
            "outer_queries_used": self.outer_queries_used,
            "outer_queries_remaining": self.outer_queries_remaining,
            "qualification_manifest_digest": self.qualification_manifest_digest,
            "incumbent_id": self.incumbent_id,
            "incumbent_is_fallback": self.incumbent_is_fallback,
            "incumbent_replay_verified": self.incumbent_replay_verified,
            "deadline_elapsed_seconds": self.deadline_elapsed_seconds,
            "deadline_remaining_seconds": self.deadline_remaining_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "finalization_required": self.finalization_required,
            "unfinished_execution_ids": list(self.unfinished_execution_ids),
            "reconciliation_count": self.reconciliation_count,
            "store_schema_digest": self.store_schema_digest,
            "store_catalog_digest": self.store_catalog_digest,
        }


def _status_from_open_store(
    *,
    run_dir: Path,
    store: CampaignStore,
    request: CampaignCreateRequest,
    deadline: _DeadlineCheckpoint,
    clock: Clock,
) -> CampaignStatus:
    identity = store.identity()
    _verify_database_identity(identity, request, deadline.state)
    snapshot = store.snapshot()
    health = store.health()
    if health.quick_check != "ok" or not health.foreign_keys or health.journal_mode != "wal":
        raise CampaignIntegrityError("campaign database safety settings are invalid")
    if snapshot.qualification_digest != request.qualification_manifest_digest:
        raise CampaignIntegrityError("campaign database qualification identity mismatch")
    _verify_exact_qualification_launches(store.launches(), request.qualification_manifest_digest)
    convergence = ConvergenceState.from_manifest(snapshot.convergence_state)
    incumbent = snapshot.incumbent
    if incumbent is None or not incumbent.replay_verified:
        raise CampaignIntegrityError("campaign has no replay-verified incumbent")
    observation = deadline.state.observe(clock)
    unfinished = store.unfinished_executions()
    reconciliations = _reconciliation_files(run_dir)
    maximum_iterations = request.config.benchmark.max_iterations
    return CampaignStatus(
        campaign_id=snapshot.campaign_id,
        request_digest=request.digest,
        revision=snapshot.revision,
        status=snapshot.status,
        phase=snapshot.phase,
        launches_used=snapshot.launches_used,
        launches_remaining=snapshot.launches_remaining,
        scientific_iterations=convergence.completed_iterations,
        scientific_iterations_remaining=max(
            0,
            maximum_iterations - convergence.completed_iterations,
        ),
        outer_queries_used=snapshot.outer_queries_used,
        outer_queries_remaining=snapshot.outer_queries_remaining,
        qualification_manifest_digest=request.qualification_manifest_digest,
        incumbent_id=incumbent.incumbent_id,
        incumbent_is_fallback=incumbent.is_fallback,
        incumbent_replay_verified=incumbent.replay_verified,
        deadline_elapsed_seconds=observation.elapsed_seconds,
        deadline_remaining_seconds=observation.remaining_seconds,
        finalization_reserve_seconds=deadline.state.finalization_reserve_seconds,
        finalization_required=snapshot.status
        in {CampaignState.FINALIZATION_REQUIRED.value, CampaignState.FINALIZING.value},
        unfinished_execution_ids=tuple(item.execution_id for item in unfinished),
        reconciliation_count=len(reconciliations),
        store_schema_digest=health.schema_digest,
        store_catalog_digest=health.catalog_digest,
    )


def _next_campaign_disposition(
    snapshot: CampaignSnapshot,
    observation: DeadlineObservation,
    request: CampaignCreateRequest,
    *,
    unsafe_cleanup: bool,
) -> tuple[str, str, str] | None:
    status = snapshot.status
    known = {item.value for item in CampaignState}
    if status not in known:
        raise CampaignIntegrityError(f"unknown stored campaign status {status!r}")
    if status in {CampaignState.COMPLETED.value, CampaignState.INCOMPLETE.value}:
        return None
    convergence = ConvergenceState.from_manifest(snapshot.convergence_state)
    if unsafe_cleanup:
        target = CampaignState.INCOMPLETE.value
        return ("reconciliation_failed", target, "abandoned process cleanup could not be proven")
    if observation.hard_expired:
        target = CampaignState.INCOMPLETE.value
        return ("deadline_exhausted", target, "campaign hard deadline expired")
    if status == CampaignState.FINALIZING.value:
        return None
    must_finalize = (
        observation.finalization_reserve_active
        or snapshot.launches_used >= snapshot.max_launches
        or convergence.should_stop
        or convergence.completed_iterations >= request.config.benchmark.max_iterations
    )
    if must_finalize:
        if status == CampaignState.FINALIZATION_REQUIRED.value:
            return None
        return (
            "finalization_required",
            CampaignState.FINALIZATION_REQUIRED.value,
            "research admission closed by time, launch, iteration, or convergence policy",
        )
    if status == CampaignState.FINALIZATION_REQUIRED.value:
        return None
    if status == CampaignState.RUNNING.value and snapshot.phase == "researching":
        return None
    return ("researching", CampaignState.RUNNING.value, "resume eligible research campaign")


def _load_verified_runtime_identity(
    run_dir: Path,
) -> tuple[CampaignCreateRequest, _DeadlineCheckpoint]:
    """Load the request and deadline only after all persisted identities agree."""

    request = _load_request(run_dir)
    deadline = _load_deadline(run_dir)
    _load_run_manifest(run_dir, request)
    with CampaignStore.open(
        run_dir / CAMPAIGN_DATABASE_NAME,
        read_only=True,
        campaign_id=request.campaign_id,
    ) as store:
        _verify_database_identity(store.identity(), request, deadline.state)
        snapshot = store.snapshot()
        if snapshot.qualification_digest != request.qualification_manifest_digest:
            raise CampaignIntegrityError("campaign database qualification identity mismatch")
        _verify_exact_qualification_launches(
            store.launches(), request.qualification_manifest_digest
        )
    return request, deadline


def _verified_deadline_observation(
    run_dir: Path, clock: Clock
) -> tuple[_DeadlineCheckpoint, DeadlineObservation]:
    """Observe one deadline only after all persisted campaign identities agree."""

    _, deadline = _load_verified_runtime_identity(run_dir)
    return deadline, deadline.state.observe(clock)


class CampaignEngine:
    """Trusted create/resume/status lifecycle over durable controller components."""

    def __init__(self, *, clock: Clock | None = None, runner: Runner | None = None) -> None:
        self._clock = SystemClock() if clock is None else clock
        self._runner = Runner() if runner is None else runner

    def create(self, request: CampaignCreateRequest) -> CampaignStatus:
        """Create one qualified campaign and refuse every existing destination path."""

        if not isinstance(request, CampaignCreateRequest):
            raise CampaignIntegrityError("request must be CampaignCreateRequest")
        destination = request.run_dir
        if os.path.lexists(destination):
            raise CampaignRunExistsError(f"campaign run path already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        parent_metadata = destination.parent.lstat()
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
            raise CampaignIntegrityError("campaign run parent must be a real directory")

        deadline_state = DeadlineState.start(
            self._clock,
            wall_clock_seconds=request.config.benchmark.wall_clock_seconds,
            finalization_reserve_seconds=request.config.runner.finalization_reserve_seconds,
        )
        evidence = _verify_qualification(request)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.creating-",
                dir=destination.parent,
            )
        )
        store: CampaignStore | None = None
        committed = False
        try:
            _ensure_managed_directory(staging)
            controller_dir = staging / CONTROLLER_DIRECTORY_NAME
            _ensure_managed_directory(controller_dir)
            _ensure_managed_directory(_reconciliation_directory(staging))
            _write_new_json(controller_dir / REQUEST_MANIFEST_NAME, request.manifest())
            initial_deadline = _write_initial_deadline(staging, deadline_state)

            store = CampaignStore.create(
                staging / CAMPAIGN_DATABASE_NAME,
                campaign_id=request.campaign_id,
                config_digest=request.config.digest,
                benchmark_digest=request.benchmark_digest,
                starter_digest=request.starter_manifest_digest,
                dataset_digest=request.dataset_manifest_digest,
                environment_digest=request.environment_digest,
                source_digest=request.source_digest,
                hard_deadline_utc=_utc_timestamp(deadline_state.utc_deadline),
                initial_convergence=ConvergenceState.initial(
                    request.fallback.outer_primary
                ).manifest(),
                max_launches=MAX_TRAINING_LAUNCHES,
                outer_query_limit=request.config.validation.outer_promotion_limit,
            )
            artifact_store = ArtifactStore(staging / "artifacts")
            qualification_ref = artifact_store.put_file(
                evidence.root_manifest_path,
                kind=ArtifactKind.MANIFEST,
                max_bytes=MAX_QUALIFICATION_MANIFEST_BYTES,
            )
            fallback_ref = artifact_store.put_file(
                evidence.fallback_manifest_path,
                kind=ArtifactKind.MANIFEST,
                max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
            )
            fallback_directory = artifact_store.put_directory(
                evidence.fallback_model_path,
                kind=ArtifactKind.CHECKPOINT,
            )
            if request.fallback.artifact_closure_digest != _canonical_digest(
                [
                    {
                        "path": item.path,
                        "size": item.artifact.size_bytes,
                        "sha256": item.artifact.sha256,
                    }
                    for item in fallback_directory.entries
                ]
            ):
                raise CampaignIntegrityError(
                    "installed fallback directory changed logical identity"
                )
            config_ref = artifact_store.put_bytes(
                _canonical_json(request.config.normalized()) + b"\n",
                kind=ArtifactKind.MANIFEST,
                max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
            )
            request_ref = artifact_store.put_file(
                controller_dir / REQUEST_MANIFEST_NAME,
                kind=ArtifactKind.MANIFEST,
                max_bytes=MAX_CONTROLLER_MANIFEST_BYTES,
            )
            installed = _InstalledArtifacts(
                qualification_manifest=qualification_ref,
                fallback_manifest=fallback_ref,
                fallback_model_manifest=fallback_directory.manifest_artifact,
                effective_config=config_ref,
                create_request=request_ref,
                fallback_model_total_size_bytes=fallback_directory.total_size_bytes,
            )
            run_manifest = _signed_manifest(installed.body(request=request))
            _write_new_json(controller_dir / RUN_MANIFEST_NAME, run_manifest)

            store.import_qualification_launches(
                evidence.launch_records,
                manifest_digest=request.qualification_manifest_digest,
                expected_revision=store.snapshot().revision,
            )
            store.record_incumbent(
                incumbent_id="official-fm-fallback-seed-4",
                eligibility="official_fm_qualified_fallback",
                source_digest=request.fallback.source_digest,
                checkpoint_digest=request.fallback.checkpoint_digest,
                artifact_closure_digest=request.fallback.artifact_closure_digest,
                replay_verified=True,
                is_fallback=True,
                expected_revision=store.snapshot().revision,
                reason="install immutable replay-verified WP3 seed-4 fallback",
                outer_primary_mean=request.fallback.outer_primary,
                artifacts=(
                    (
                        "qualification_manifest",
                        _artifact_spec(
                            qualification_ref,
                            semantic_kind="qualification_manifest",
                        ),
                    ),
                    (
                        "fallback_manifest",
                        _artifact_spec(fallback_ref, semantic_kind="fallback_manifest"),
                    ),
                    (
                        "fallback_model",
                        _directory_artifact_spec(
                            fallback_directory,
                            semantic_kind="fallback_model_directory",
                            logical_closure_digest=request.fallback.artifact_closure_digest,
                        ),
                    ),
                    (
                        "effective_config",
                        _artifact_spec(config_ref, semantic_kind="effective_config"),
                    ),
                    (
                        "create_request",
                        _artifact_spec(request_ref, semantic_kind="create_request"),
                    ),
                ),
                metadata={
                    "qualification_digest": request.qualification_manifest_digest,
                    "fallback_manifest_digest": request.fallback.manifest_digest,
                    "fallback_seed": 4,
                },
            )
            store.set_campaign_phase(
                phase="researching",
                status=CampaignState.RUNNING.value,
                expected_revision=store.snapshot().revision,
                reason="qualification imported and immutable fallback installed",
            )
            observation = initial_deadline.state.observe(self._clock)
            _append_deadline(staging, initial_deadline, observation)
            _verify_database_identity(store.identity(), request, observation.state)
            _verify_exact_qualification_launches(
                store.launches(),
                request.qualification_manifest_digest,
            )
            store.close()
            store = None
            _rename_directory_exclusive(staging, destination)
            committed = True
        finally:
            if store is not None:
                store.close()
            if not committed and staging.exists():
                shutil.rmtree(staging)
        return self.status(destination)

    def resume(self, run_dir: Path) -> CampaignStatus:
        """Reconcile abandoned work and continue the original campaign deadline."""

        run = _validate_run_directory(run_dir)
        request = _load_request(run)
        deadline = _load_deadline(run)
        refs = _load_run_manifest(run, request)
        _verify_installed_artifacts(run, request, refs)
        with CampaignStore.open(
            run / CAMPAIGN_DATABASE_NAME,
            campaign_id=request.campaign_id,
        ) as store:
            _verify_database_identity(store.identity(), request, deadline.state)
            snapshot = store.snapshot()
            if snapshot.qualification_digest != request.qualification_manifest_digest:
                raise CampaignIntegrityError("campaign qualification identity changed")
            _verify_exact_qualification_launches(
                store.launches(),
                request.qualification_manifest_digest,
            )
            incumbent_before = store.current_incumbent()
            if incumbent_before is None:
                raise CampaignIntegrityError("campaign has no incumbent to preserve")
            terminal_statuses = {
                CampaignState.COMPLETED.value,
                CampaignState.INCOMPLETE.value,
            }
            if snapshot.status in terminal_statuses:
                if store.unfinished_executions():
                    raise CampaignIntegrityError(
                        "terminal campaign contains an unfinished execution"
                    )
                return _status_from_open_store(
                    run_dir=run,
                    store=store,
                    request=request,
                    deadline=deadline,
                    clock=self._clock,
                )
            observation = deadline.state.observe(self._clock)
            advanced = _append_deadline(run, deadline, observation)
            reconciled = _reconcile_unfinished(run, store, self._runner, self._clock)
            incumbent_after = store.current_incumbent()
            if (
                incumbent_after is None
                or incumbent_after.incumbent_id != incumbent_before.incumbent_id
            ):
                raise CampaignIntegrityError("campaign resume changed the incumbent")
            disposition = _next_campaign_disposition(
                store.snapshot(),
                observation,
                request,
                unsafe_cleanup=reconciled.unsafe_cleanup,
            )
            if disposition is not None:
                phase, status, reason = disposition
                store.set_campaign_phase(
                    phase=phase,
                    status=status,
                    expected_revision=store.snapshot().revision,
                    reason=reason,
                )
            return _status_from_open_store(
                run_dir=run,
                store=store,
                request=request,
                deadline=advanced,
                clock=self._clock,
            )

    def status(self, run_dir: Path) -> CampaignStatus:
        """Return a read-only stable campaign projection without reconciling or writing."""

        run = _validate_run_directory(run_dir)
        request = _load_request(run)
        deadline = _load_deadline(run)
        _load_run_manifest(run, request)
        with CampaignStore.open(
            run / CAMPAIGN_DATABASE_NAME,
            read_only=True,
            campaign_id=request.campaign_id,
        ) as store:
            return _status_from_open_store(
                run_dir=run,
                store=store,
                request=request,
                deadline=deadline,
                clock=self._clock,
            )

    def inspect_deadline(self, run_dir: Path) -> DeadlineObservation:
        """Return a verified current observation without changing persisted evidence."""

        run = _validate_run_directory(run_dir)
        _, observation = _verified_deadline_observation(run, self._clock)
        return observation

    def load_request(self, run_dir: Path) -> CampaignCreateRequest:
        """Return the verified immutable create request without changing the run."""

        run = _validate_run_directory(run_dir)
        request, _ = _load_verified_runtime_identity(run)
        return request

    def observe_deadline(self, run_dir: Path) -> DeadlineObservation:
        """Verify and durably append one observation to the original deadline chain."""

        run = _validate_run_directory(run_dir)
        deadline, observation = _verified_deadline_observation(run, self._clock)
        _append_deadline(run, deadline, observation)
        return observation


__all__ = [
    "CAMPAIGN_DATABASE_NAME",
    "CONTROLLER_SCHEMA_VERSION",
    "CampaignControllerError",
    "CampaignCreateRequest",
    "CampaignEngine",
    "CampaignIntegrityError",
    "CampaignRunExistsError",
    "CampaignStatus",
    "DeadlineObservation",
    "FallbackIdentity",
]
