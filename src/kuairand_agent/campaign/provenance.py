"""Locked source, environment, and qualification identities for campaign creation.

The durable controller deliberately accepts already-computed scientific identities.  This module
is the small trusted bridge used by the CLI: it hashes the executable repository slice, records
the locked local environment, extracts the immutable WP3 fallback identity, and constructs the
fully bound :class:`~kuairand_agent.campaign.controller.CampaignCreateRequest`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from kuairand_agent.campaign.controller import CampaignCreateRequest, FallbackIdentity
from kuairand_agent.config import AgentConfig
from kuairand_agent.contract import benchmark_digest, verify_starter_kit

PROVENANCE_SCHEMA_VERSION: Final = 1
ENVIRONMENT_IDENTITY_SCHEMA_VERSION: Final = 2
ENVIRONMENT_IDENTITY_PACKAGES: Final = ("lightgbm", "numpy", "psutil", "torch")
MAX_PROVENANCE_FILE_BYTES: Final = 512 * 1024 * 1024
MAX_QUALIFICATION_MANIFEST_BYTES: Final = 8 * 1024 * 1024
_SOURCE_ROOT_FILES: Final = (".python-version", "pyproject.toml", "uv.lock")
_SOURCE_TREES: Final = {
    "candidate_seed": frozenset({".json", ".md", ".py"}),
    "candidate_templates": frozenset({".json", ".md", ".py"}),
    "configs": frozenset({".toml"}),
    "scripts": frozenset({".py", ".sh"}),
    "src": frozenset({".py"}),
}


class ProvenanceError(RuntimeError):
    """Raised when a campaign cannot be bound to immutable local evidence."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(domain: bytes, value: object) -> str:
    result = hashlib.sha256()
    result.update(domain)
    result.update(_canonical_json(value))
    return result.hexdigest()


def _require_digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProvenanceError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _read_regular(path: Path, *, maximum: int, location: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ProvenanceError(f"cannot inspect {location}: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProvenanceError(f"{location} must be a regular file without symlinks")
    if before.st_size < 0 or before.st_size > maximum:
        raise ProvenanceError(f"{location} exceeds its {maximum}-byte bound")
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
            descriptor = os.fstat(handle.fileno())
    except OSError as exc:
        raise ProvenanceError(f"cannot read {location}: {path}") from exc
    if len(payload) > maximum:
        raise ProvenanceError(f"{location} exceeds its {maximum}-byte bound")
    if (
        descriptor.st_dev != before.st_dev
        or descriptor.st_ino != before.st_ino
        or descriptor.st_size != before.st_size
        or len(payload) != before.st_size
    ):
        raise ProvenanceError(f"{location} changed while it was being hashed")
    return payload


@dataclass(frozen=True, slots=True)
class SourceFileIdentity:
    path: str
    size_bytes: int
    sha256: str

    def manifest(self) -> dict[str, object]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SourceTreeIdentity:
    root: Path
    files: tuple[SourceFileIdentity, ...]
    digest: str
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "files": [item.manifest() for item in self.files],
            "digest": self.digest,
        }


def _source_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in _SOURCE_ROOT_FILES:
        candidate = root / name
        if not candidate.exists():
            raise ProvenanceError(f"required locked source file is missing: {name}")
        paths.append(candidate)
    for tree_name, suffixes in _SOURCE_TREES.items():
        tree = root / tree_name
        if tree_name == "candidate_seed" and not tree.exists():
            continue
        if not tree.is_dir() or tree.is_symlink():
            raise ProvenanceError(f"required source tree is missing or unsafe: {tree_name}")
        for candidate in tree.rglob("*"):
            if candidate.is_symlink():
                raise ProvenanceError(
                    f"source tree contains a symlink: {candidate.relative_to(root).as_posix()}"
                )
            if (
                candidate.is_file()
                and candidate.suffix in suffixes
                and "__pycache__" not in candidate.parts
            ):
                paths.append(candidate)
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def hash_source_tree(repository_root: str | Path) -> SourceTreeIdentity:
    """Hash the locked executable/configuration slice without organizer or runtime artifacts."""

    try:
        root = Path(repository_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError("cannot resolve repository root") from exc
    if not root.is_dir() or root.is_symlink():
        raise ProvenanceError("repository root must be a real directory")
    identities: list[SourceFileIdentity] = []
    for path in _source_paths(root):
        relative = path.relative_to(root).as_posix()
        payload = _read_regular(
            path,
            maximum=MAX_PROVENANCE_FILE_BYTES,
            location=f"source file {relative}",
        )
        identities.append(
            SourceFileIdentity(
                path=relative,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    body = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "files": [item.manifest() for item in identities],
    }
    return SourceTreeIdentity(
        root=root,
        files=tuple(identities),
        digest=_digest(b"kuairand-source-tree-v1\0", body),
    )


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    manifest_data: dict[str, object]
    digest: str

    def manifest(self) -> dict[str, object]:
        return dict(self.manifest_data) | {"digest": self.digest}


def capture_environment_identity(repository_root: str | Path) -> EnvironmentIdentity:
    """Record deterministic locked-runtime facts; secrets and mutable shell state are excluded."""

    root = Path(repository_root).resolve(strict=True)
    lock = _read_regular(
        root / "uv.lock",
        maximum=MAX_PROVENANCE_FILE_BYTES,
        location="uv lock file",
    )
    packages: dict[str, str | None] = {}
    for package in ENVIRONMENT_IDENTITY_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    body: dict[str, object] = {
        "schema_version": ENVIRONMENT_IDENTITY_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "uv_lock_sha256": hashlib.sha256(lock).hexdigest(),
    }
    return EnvironmentIdentity(
        manifest_data=body,
        digest=_digest(b"kuairand-environment-v2\0", body),
    )


def _json_object(path: Path, *, location: str) -> dict[str, object]:
    payload = _read_regular(
        path,
        maximum=MAX_QUALIFICATION_MANIFEST_BYTES,
        location=location,
    )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"{location} is not strict JSON") from exc
    if not isinstance(decoded, dict) or any(type(key) is not str for key in decoded):
        raise ProvenanceError(f"{location} must be a JSON object")
    return cast(dict[str, object], decoded)


def _logical_digest(value: dict[str, object], *, location: str) -> str:
    observed = _require_digest(value.get("digest"), f"{location}.digest")
    body = {key: item for key, item in value.items() if key != "digest"}
    if hashlib.sha256(_canonical_json(body)).hexdigest() != observed:
        raise ProvenanceError(f"{location} logical digest mismatch")
    return observed


@dataclass(frozen=True, slots=True)
class QualificationIdentity:
    root: Path
    manifest_digest: str
    benchmark_digest: str
    starter_manifest_digest: str
    fallback: FallbackIdentity


def load_qualification_identity(qualification_run_dir: str | Path) -> QualificationIdentity:
    """Extract controller inputs from the signed WP3 manifest without weakening its verifier."""

    try:
        root = Path(qualification_run_dir).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError("cannot resolve qualification run directory") from exc
    if not root.is_dir() or root.is_symlink():
        raise ProvenanceError("qualification run directory must be a real directory")
    manifest = _json_object(root / "manifest.json", location="qualification manifest")
    manifest_digest = _logical_digest(manifest, location="qualification manifest")
    if manifest.get("status") != "baseline_reproduced":
        raise ProvenanceError("qualification manifest did not reproduce the official baseline")
    benchmark = _require_digest(manifest.get("benchmark_digest"), "qualification benchmark")
    if benchmark != benchmark_digest():
        raise ProvenanceError("qualification benchmark differs from the executable contract")
    raw_fallback = manifest.get("fallback")
    if not isinstance(raw_fallback, dict):
        raise ProvenanceError("qualification fallback must be an object")
    fallback_manifest = cast(dict[str, object], raw_fallback)
    fallback_digest = _logical_digest(fallback_manifest, location="qualification fallback")
    metrics = fallback_manifest.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ProvenanceError("qualification fallback metrics must be an object")
    outer_primary = metrics.get("primary")
    if isinstance(outer_primary, bool) or not isinstance(outer_primary, (int, float)):
        raise ProvenanceError("qualification fallback primary must be numeric")
    runs = manifest.get("fm")
    if not isinstance(runs, dict) or not isinstance(runs.get("runs"), list):
        raise ProvenanceError("qualification FM runs are missing")
    seed_four = [
        item
        for item in cast(list[object], runs["runs"])
        if isinstance(item, dict) and item.get("seed") == 4
    ]
    if len(seed_four) != 1:
        raise ProvenanceError("qualification must contain exactly one seed-4 FM run")
    seed_identity = cast(dict[str, object], seed_four[0])
    starter = _require_digest(
        seed_identity.get("starter_manifest_digest"),
        "seed-4 starter manifest",
    )
    fallback = FallbackIdentity(
        manifest_digest=fallback_digest,
        source_digest=_require_digest(
            fallback_manifest.get("source_model_tree_digest"),
            "fallback source model tree",
        ),
        checkpoint_digest=_require_digest(
            fallback_manifest.get("checkpoint_digest"), "fallback checkpoint"
        ),
        artifact_closure_digest=_require_digest(
            fallback_manifest.get("fallback_model_tree_digest"),
            "fallback artifact closure",
        ),
        config_digest=_require_digest(fallback_manifest.get("config_digest"), "fallback config"),
        encoding_digest=_require_digest(
            fallback_manifest.get("encoding_digest"), "fallback encoding"
        ),
        outer_primary=float(outer_primary),
    )
    return QualificationIdentity(
        root=root,
        manifest_digest=manifest_digest,
        benchmark_digest=benchmark,
        starter_manifest_digest=starter,
        fallback=fallback,
    )


@dataclass(frozen=True, slots=True)
class CampaignProvenance:
    request: CampaignCreateRequest
    source: SourceTreeIdentity
    environment: EnvironmentIdentity
    qualification: QualificationIdentity


def build_campaign_request(
    *,
    repository_root: str | Path,
    run_dir: str | Path,
    qualification_run_dir: str | Path,
    config: AgentConfig,
    dataset_manifest_digest: str,
) -> CampaignProvenance:
    """Build the exact controller request from independently verifiable local evidence."""

    if not isinstance(config, AgentConfig):
        raise ProvenanceError("config must be a validated AgentConfig")
    dataset_digest = _require_digest(dataset_manifest_digest, "dataset manifest digest")
    root = Path(repository_root).resolve(strict=True)
    source = hash_source_tree(root)
    environment = capture_environment_identity(root)
    qualification = load_qualification_identity(qualification_run_dir)
    starter_dir = config.benchmark.starter_dir
    if not starter_dir.is_absolute():
        starter_dir = root / starter_dir
    starter = verify_starter_kit(starter_dir)
    if starter.manifest_sha256 != qualification.starter_manifest_digest:
        raise ProvenanceError("current starter manifest differs from qualification")
    target = Path(run_dir)
    if not target.is_absolute():
        target = root / target
    target = target.resolve(strict=False)
    campaign_fingerprint = _digest(
        b"kuairand-campaign-request-id-v1\0",
        {
            "run_dir": str(target),
            "qualification": qualification.manifest_digest,
            "config": config.digest,
            "dataset": dataset_digest,
            "source": source.digest,
            "environment": environment.digest,
        },
    )
    request = CampaignCreateRequest(
        run_dir=target,
        campaign_id=f"kuairand-{campaign_fingerprint[:20]}",
        config=config,
        qualification_run_dir=qualification.root,
        qualification_manifest_digest=qualification.manifest_digest,
        fallback=qualification.fallback,
        benchmark_digest=qualification.benchmark_digest,
        starter_manifest_digest=qualification.starter_manifest_digest,
        dataset_manifest_digest=dataset_digest,
        source_digest=source.digest,
        environment_digest=environment.digest,
    )
    return CampaignProvenance(
        request=request,
        source=source,
        environment=environment,
        qualification=qualification,
    )


__all__ = [
    "ENVIRONMENT_IDENTITY_PACKAGES",
    "ENVIRONMENT_IDENTITY_SCHEMA_VERSION",
    "MAX_PROVENANCE_FILE_BYTES",
    "PROVENANCE_SCHEMA_VERSION",
    "CampaignProvenance",
    "EnvironmentIdentity",
    "ProvenanceError",
    "QualificationIdentity",
    "SourceFileIdentity",
    "SourceTreeIdentity",
    "build_campaign_request",
    "capture_environment_identity",
    "hash_source_tree",
    "load_qualification_identity",
]
