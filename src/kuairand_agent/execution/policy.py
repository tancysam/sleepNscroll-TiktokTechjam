"""Pure candidate source, request, workspace, and output policy.

This module provides robustness containment for cooperative locally generated code.  It rejects
common accidental escape and integrity hazards, but it is not an operating-system security
sandbox and must not be described as one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    DirectoryArtifactRef,
)

if TYPE_CHECKING:
    from kuairand_agent.execution.workspace import CandidateWorkspace

POLICY_SCHEMA_VERSION: Final = 1
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_HANDLE_RE: Final = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SOURCE_SUFFIXES: Final = frozenset({".py", ".json", ".toml", ".md", ".txt"})
_OUTPUT_SUFFIXES: Final = frozenset(
    {".bin", ".ckpt", ".csv", ".json", ".npy", ".npz", ".pt", ".pth", ".txt"}
)
_ARCHIVE_SUFFIXES: Final = (".tar", ".tar.gz", ".tgz", ".zip", ".7z")
_TRUSTED_BASENAMES: Final = frozenset(
    {
        "ablation_features.py",
        "baseline.py",
        "conftest.py",
        "contract.py",
        "data.py",
        "evaluate.py",
        "protected.py",
        "pyproject.toml",
        "scorer.py",
        "setup.cfg",
        "setup.py",
        "sitecustomize.py",
        "submit.py",
        "usercustomize.py",
        "uv.lock",
    }
)
_TRUSTED_SEGMENTS: Final = frozenset(
    {
        "__pycache__",
        "campaign",
        "finalization",
        "kuairand-starter-kit",
        "kuairand_agent",
        "scoring",
    }
)
_SECRET_KEYS: Final = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credentials",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)
_ALIGNMENT_KEYS: Final = frozenset(
    {
        "alignment",
        "alignment_hash",
        "provenance_hash",
        "row_id",
        "source_ordinal",
        "source_record_ordinal",
    }
)
_PROTECTED_REQUEST_KEYS: Final = frozenset(
    {
        "labels",
        "long_view",
        "outcomes",
        "protected_scorer",
        "scorer",
        "target_artifact",
        "target_handle",
        "target",
        "targets",
    }
)


class WorkspacePolicyError(ValueError):
    """A source, request, path, workspace, or output violated the frozen policy."""


class SplitRole(StrEnum):
    TRAIN = "train"
    INNER_TRAIN = "inner_train"
    INNER_VALID = "inner_valid"
    OUTER_VALID = "outer_valid"
    FINAL = "final"


class CandidateInputRole(StrEnum):
    TRAIN_INPUTS = "train_inputs"
    TRAIN_TARGETS = "train_targets"
    INNER_VALID_INPUTS = "inner_valid_inputs"
    OUTER_VALID_INPUTS = "outer_valid_inputs"
    FINAL_INPUTS = "final_inputs"


_INPUTS_BY_SPLIT: Final[Mapping[SplitRole, frozenset[CandidateInputRole]]] = {
    SplitRole.TRAIN: frozenset({CandidateInputRole.TRAIN_INPUTS, CandidateInputRole.TRAIN_TARGETS}),
    SplitRole.INNER_TRAIN: frozenset(
        {CandidateInputRole.TRAIN_INPUTS, CandidateInputRole.TRAIN_TARGETS}
    ),
    SplitRole.INNER_VALID: frozenset({CandidateInputRole.INNER_VALID_INPUTS}),
    SplitRole.OUTER_VALID: frozenset({CandidateInputRole.OUTER_VALID_INPUTS}),
    SplitRole.FINAL: frozenset({CandidateInputRole.FINAL_INPUTS}),
}


@dataclass(frozen=True, slots=True)
class SourceEntry:
    path: str
    sha256: str
    size_bytes: int

    def manifest(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class SourceManifest:
    entries: tuple[SourceEntry, ...]
    artifact_manifest_sha256: str
    schema_version: int = POLICY_SCHEMA_VERSION

    @classmethod
    def from_directory_artifact(cls, snapshot: DirectoryArtifactRef) -> SourceManifest:
        return cls(
            entries=tuple(
                SourceEntry(entry.path, entry.artifact.sha256, entry.artifact.size_bytes)
                for entry in snapshot.entries
            ),
            artifact_manifest_sha256=snapshot.sha256,
        )

    def identity_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "entries": [entry.manifest() for entry in self.entries],
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.identity_manifest())).hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovedInput:
    """One controller-approved candidate capability object, never a raw path."""

    name: str
    role: CandidateInputRole
    artifact: ArtifactRef

    def manifest(self, workspace_path: str) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "workspace_path": workspace_path,
            "artifact": self.artifact.manifest(),
        }


@dataclass(frozen=True, slots=True)
class DeclaredOutput:
    path: str
    max_bytes: int


@dataclass(frozen=True, slots=True)
class OutputDeclaration:
    files: tuple[DeclaredOutput, ...]


@dataclass(frozen=True, slots=True)
class OutputFile:
    path: str
    sha256: str
    size_bytes: int

    def manifest(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class OutputInventory:
    files: tuple[OutputFile, ...]
    total_size_bytes: int
    digest: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def validate_relative_path(
    value: str,
    *,
    allowed_suffixes: frozenset[str] | None = None,
    reject_hidden: bool = True,
) -> PurePosixPath:
    """Validate an exact canonical relative POSIX path without normalization."""

    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\0" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise WorkspacePolicyError("path must be a non-empty canonical POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise WorkspacePolicyError(f"path must be canonical and relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise WorkspacePolicyError(f"path contains a traversal component: {value!r}")
    if reject_hidden and any(part.startswith(".") for part in path.parts):
        raise WorkspacePolicyError(f"hidden path components are forbidden: {value!r}")
    if path.name.lower().endswith(_ARCHIVE_SUFFIXES):
        raise WorkspacePolicyError(f"raw archive paths are forbidden: {value!r}")
    if path.name.lower().endswith(".pth") and allowed_suffixes == _SOURCE_SUFFIXES:
        raise WorkspacePolicyError(f"Python path control files are forbidden: {value!r}")
    if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
        raise WorkspacePolicyError(f"path suffix is not allowed: {value!r}")
    return path


def _hash_file(path: Path, *, ceiling: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > ceiling:
                raise WorkspacePolicyError(f"file exceeds its {ceiling}-byte ceiling: {path.name}")
            digest.update(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    """Frozen local robustness and bounded-inventory policy."""

    max_source_files: int = 32
    max_source_file_bytes: int = 512 * 1024
    max_source_total_bytes: int = 4 * 1024 * 1024
    max_input_files: int = 32
    max_input_file_bytes: int = 2 * 1024 * 1024 * 1024
    max_input_total_bytes: int = 8 * 1024 * 1024 * 1024
    max_request_bytes: int = 256 * 1024
    max_output_files: int = 64
    max_output_file_bytes: int = 2 * 1024 * 1024 * 1024
    max_output_total_bytes: int = 8 * 1024 * 1024 * 1024
    max_temp_bytes: int = 4 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_source_files",
            "max_source_file_bytes",
            "max_source_total_bytes",
            "max_input_files",
            "max_input_file_bytes",
            "max_input_total_bytes",
            "max_request_bytes",
            "max_output_files",
            "max_output_file_bytes",
            "max_output_total_bytes",
            "max_temp_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise WorkspacePolicyError(f"{name} must be a positive integer")

    def validate_source_manifest(self, manifest: SourceManifest) -> None:
        if manifest.schema_version != POLICY_SCHEMA_VERSION:
            raise WorkspacePolicyError("source manifest schema_version must be 1")
        if _DIGEST_RE.fullmatch(manifest.artifact_manifest_sha256) is None:
            raise WorkspacePolicyError("source artifact manifest digest is invalid")
        if not manifest.entries:
            raise WorkspacePolicyError("source manifest must contain at least one file")
        if len(manifest.entries) > self.max_source_files:
            raise WorkspacePolicyError("source manifest exceeds the file-count ceiling")
        paths = tuple(entry.path for entry in manifest.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise WorkspacePolicyError("source paths must be unique and sorted")
        total = 0
        for entry in manifest.entries:
            path = validate_relative_path(entry.path, allowed_suffixes=_SOURCE_SUFFIXES)
            lowered_parts = {part.lower() for part in path.parts}
            if path.name.lower() in _TRUSTED_BASENAMES or lowered_parts & _TRUSTED_SEGMENTS:
                raise WorkspacePolicyError(f"source path shadows trusted code: {entry.path!r}")
            if path.name.lower().startswith("requirements"):
                raise WorkspacePolicyError(
                    f"dependency control files are forbidden: {entry.path!r}"
                )
            if _DIGEST_RE.fullmatch(entry.sha256) is None:
                raise WorkspacePolicyError(f"source digest is invalid: {entry.path!r}")
            if (
                type(entry.size_bytes) is not int
                or not 0 <= entry.size_bytes <= self.max_source_file_bytes
            ):
                raise WorkspacePolicyError(f"source file size is invalid: {entry.path!r}")
            total += entry.size_bytes
        if total > self.max_source_total_bytes:
            raise WorkspacePolicyError("source manifest exceeds the total-byte ceiling")

    def validate_approved_inputs(
        self, split_role: SplitRole, approved_inputs: Sequence[ApprovedInput]
    ) -> None:
        if not isinstance(split_role, SplitRole):
            raise WorkspacePolicyError("split_role must be a SplitRole")
        inputs = tuple(approved_inputs)
        if len(inputs) > self.max_input_files:
            raise WorkspacePolicyError("approved inputs exceed the file-count ceiling")
        names = tuple(item.name for item in inputs)
        if len(names) != len(set(names)):
            raise WorkspacePolicyError("approved input names must be unique")
        total = 0
        for item in inputs:
            if not isinstance(item, ApprovedInput):
                raise WorkspacePolicyError("approved inputs must use ApprovedInput records")
            if _HANDLE_RE.fullmatch(item.name) is None:
                raise WorkspacePolicyError(f"approved input name is invalid: {item.name!r}")
            protected_name_tokens = {"archive", "credential", "outcome", "scorer"}
            if split_role not in {SplitRole.TRAIN, SplitRole.INNER_TRAIN}:
                protected_name_tokens.update({"label", "long_view", "target"})
            if any(token in item.name for token in protected_name_tokens):
                raise WorkspacePolicyError(f"approved input name is protected: {item.name!r}")
            if not isinstance(item.role, CandidateInputRole):
                raise WorkspacePolicyError("approved input role must be a CandidateInputRole")
            if item.role not in _INPUTS_BY_SPLIT[split_role]:
                raise WorkspacePolicyError(
                    f"input role {item.role.value!r} is forbidden for split {split_role.value!r}"
                )
            if item.artifact.kind is not ArtifactKind.INPUT:
                raise WorkspacePolicyError("candidate input artifacts must use ArtifactKind.INPUT")
            if item.artifact.size_bytes > self.max_input_file_bytes:
                raise WorkspacePolicyError("approved input exceeds the individual-byte ceiling")
            total += item.artifact.size_bytes
        if total > self.max_input_total_bytes:
            raise WorkspacePolicyError("approved inputs exceed the total-byte ceiling")

    def validate_request_payload(
        self, split_role: SplitRole, payload: Mapping[str, object]
    ) -> bytes:
        if not isinstance(payload, Mapping):
            raise WorkspacePolicyError("candidate request payload must be a mapping")
        self._scan_request_value(payload, split_role=split_role, path="request")
        try:
            encoded = _canonical_json(payload)
        except (TypeError, ValueError, OverflowError) as error:
            raise WorkspacePolicyError(
                "candidate request payload must be finite canonical JSON"
            ) from error
        if len(encoded) > self.max_request_bytes:
            raise WorkspacePolicyError("candidate request payload exceeds the byte ceiling")
        return encoded

    def validate_workspace_identity(self, workspace: CandidateWorkspace) -> None:
        """Verify trusted workspace identity without traversing mutable runtime outputs."""

        root = workspace.root
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise WorkspacePolicyError("workspace root must be a real directory")
        if stat.S_IMODE(root_metadata.st_mode) & 0o077:
            raise WorkspacePolicyError("workspace root must be private to its owner")
        request_metadata = workspace.request_path.lstat()
        request_limit = self.max_request_bytes + self.max_input_files * 1024 + 16 * 1024
        if (
            stat.S_ISLNK(request_metadata.st_mode)
            or not stat.S_ISREG(request_metadata.st_mode)
            or request_metadata.st_nlink != 1
            or request_metadata.st_size > request_limit
            or request_metadata.st_mode & 0o222
        ):
            raise WorkspacePolicyError("workspace request is not an immutable bounded file")
        request_bytes = workspace.request_path.read_bytes()
        if hashlib.sha256(request_bytes).hexdigest() != workspace.request_sha256:
            raise WorkspacePolicyError("workspace request digest does not match")

        manifest_body = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "execution_id": workspace.execution_id,
            "split_role": workspace.split_role.value,
            "source_snapshot_sha256": workspace.source_snapshot_sha256,
            "source_files": [file.manifest() for file in workspace.source_files],
            "input_files": [file.manifest() for file in workspace.input_files],
            "request_sha256": workspace.request_sha256,
            "output_limit_bytes": workspace.output_limit_bytes,
            "temp_limit_bytes": workspace.temp_limit_bytes,
        }
        expected_digest = hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
        if expected_digest != workspace.manifest_digest:
            raise WorkspacePolicyError("workspace manifest identity is inconsistent")
        expected_manifest = _canonical_json(
            {**manifest_body, "manifest_digest": workspace.manifest_digest}
        )
        manifest_metadata = workspace.manifest_path.lstat()
        if (
            stat.S_ISLNK(manifest_metadata.st_mode)
            or not stat.S_ISREG(manifest_metadata.st_mode)
            or manifest_metadata.st_nlink != 1
            or manifest_metadata.st_size > 1024 * 1024
            or manifest_metadata.st_mode & 0o222
        ):
            raise WorkspacePolicyError("workspace manifest is not an immutable bounded file")
        if workspace.manifest_path.read_bytes() != expected_manifest:
            raise WorkspacePolicyError("workspace manifest bytes do not match")

    def validate_workspace(self, workspace: CandidateWorkspace) -> None:
        """Validate the exact freshly materialized immutable inventory."""

        self.validate_workspace_identity(workspace)
        root = workspace.root
        expected_top = {
            "home",
            "inputs",
            "output",
            "process.json",
            "request.json",
            "source",
            "tmp",
            "workspace-manifest.json",
        }
        actual_top = {path.name for path in root.iterdir()}
        if actual_top != expected_top:
            raise WorkspacePolicyError("workspace top-level inventory is not exact")

        for name in ("source", "inputs", "output", "tmp", "home"):
            directory = root / name
            metadata = directory.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise WorkspacePolicyError(f"workspace {name} must be a real directory")
            if name in {"source", "inputs"} and metadata.st_mode & 0o222:
                raise WorkspacePolicyError(f"workspace {name} must be read-only")
            if name in {"output", "tmp", "home"} and stat.S_IMODE(metadata.st_mode) & 0o077:
                raise WorkspacePolicyError(f"workspace {name} must be private")

        expected_files = {
            "process.json",
            "request.json",
            "workspace-manifest.json",
            *(file.relative_path for file in workspace.source_files),
            *(file.relative_path for file in workspace.input_files),
        }
        actual_files: set[str] = set()
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise WorkspacePolicyError(f"workspace contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                if relative.startswith(("source/", "inputs/")) and metadata.st_mode & 0o222:
                    raise WorkspacePolicyError(
                        f"workspace immutable directory is writable: {relative}"
                    )
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise WorkspacePolicyError(f"workspace contains a special file: {relative}")
            if metadata.st_nlink != 1:
                raise WorkspacePolicyError(f"workspace contains a hardlinked file: {relative}")
            actual_files.add(relative)
        if actual_files != expected_files:
            raise WorkspacePolicyError("workspace file inventory is not exact")

        for file in (*workspace.source_files, *workspace.input_files):
            destination = root / file.relative_path
            digest, size = _hash_file(destination, ceiling=file.size_bytes)
            if digest != file.sha256 or size != file.size_bytes:
                raise WorkspacePolicyError(
                    f"workspace immutable copy changed: {file.relative_path}"
                )
            if destination.stat().st_mode & 0o222:
                raise WorkspacePolicyError(
                    f"workspace immutable copy is writable: {file.relative_path}"
                )

        process_metadata = workspace.process_record_path.lstat()
        if process_metadata.st_size > 64 * 1024 or stat.S_IMODE(process_metadata.st_mode) & 0o077:
            raise WorkspacePolicyError("workspace process record is oversized or not private")
        try:
            process_record = json.loads(workspace.process_record_path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise WorkspacePolicyError("workspace process record is invalid JSON") from error
        if process_record != {
            "schema_version": POLICY_SCHEMA_VERSION,
            "execution_id": workspace.execution_id,
            "state": "materialized",
        }:
            raise WorkspacePolicyError("workspace process record is not the initial trusted record")

    def validate_outputs(
        self,
        workspace: CandidateWorkspace,
        declaration: OutputDeclaration,
    ) -> OutputInventory:
        """Reject undeclared, linked, special, or oversized candidate outputs."""

        declared = tuple(declaration.files)
        if len(declared) > self.max_output_files:
            raise WorkspacePolicyError("output declaration exceeds the file-count ceiling")
        declared_by_path: dict[str, DeclaredOutput] = {}
        expected_dirs: set[str] = set()
        declared_total_ceiling = 0
        for entry in declared:
            declared_path = validate_relative_path(entry.path, allowed_suffixes=_OUTPUT_SUFFIXES)
            if entry.path in declared_by_path:
                raise WorkspacePolicyError("output declaration paths must be unique")
            if type(entry.max_bytes) is not int or entry.max_bytes <= 0:
                raise WorkspacePolicyError("declared output max_bytes must be positive")
            if entry.max_bytes > self.max_output_file_bytes:
                raise WorkspacePolicyError("declared output exceeds the per-file policy ceiling")
            declared_by_path[entry.path] = entry
            declared_total_ceiling += entry.max_bytes
            for parent in declared_path.parents:
                if parent != PurePosixPath("."):
                    expected_dirs.add(parent.as_posix())

        if declared_total_ceiling > min(workspace.output_limit_bytes, self.max_output_total_bytes):
            raise WorkspacePolicyError("declared outputs exceed the total workspace ceiling")

        output_root = workspace.output_dir
        output_metadata = output_root.lstat()
        if stat.S_ISLNK(output_metadata.st_mode) or not stat.S_ISDIR(output_metadata.st_mode):
            raise WorkspacePolicyError("candidate output root must be a real directory")
        actual_files: dict[str, Path] = {}
        actual_dirs: set[str] = set()
        for actual_path in sorted(output_root.rglob("*")):
            relative = actual_path.relative_to(output_root).as_posix()
            metadata = actual_path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise WorkspacePolicyError(f"candidate output contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                actual_dirs.add(relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise WorkspacePolicyError(f"candidate output contains a special file: {relative}")
            if metadata.st_nlink != 1:
                raise WorkspacePolicyError(f"candidate output contains a hardlink: {relative}")
            actual_files[relative] = actual_path
        if set(actual_files) != set(declared_by_path):
            raise WorkspacePolicyError("candidate output inventory differs from its declaration")
        if actual_dirs != expected_dirs:
            raise WorkspacePolicyError("candidate output directory inventory is not declared")

        inventory: list[OutputFile] = []
        total = 0
        effective_total_limit = min(workspace.output_limit_bytes, self.max_output_total_bytes)
        for relative, path in sorted(actual_files.items()):
            declared_file = declared_by_path[relative]
            ceiling = min(declared_file.max_bytes, self.max_output_file_bytes)
            digest, size = _hash_file(path, ceiling=ceiling)
            total += size
            if total > effective_total_limit:
                raise WorkspacePolicyError("candidate outputs exceed the total-byte ceiling")
            inventory.append(OutputFile(relative, digest, size))
        manifest = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "total_size_bytes": total,
            "files": [file.manifest() for file in inventory],
        }
        return OutputInventory(
            files=tuple(inventory),
            total_size_bytes=total,
            digest=hashlib.sha256(_canonical_json(manifest)).hexdigest(),
        )

    def _scan_request_value(self, value: object, *, split_role: SplitRole, path: str) -> None:
        if value is None or isinstance(value, (str, bool)) or type(value) is int:
            if isinstance(value, str):
                lowered = value.lower()
                if "\0" in value or "\\" in value or value.startswith("/"):
                    raise WorkspacePolicyError(f"request contains a filesystem path at {path}")
                if lowered.startswith(("http://", "https://", "bearer ", "sk-")):
                    raise WorkspacePolicyError(
                        f"request contains network or credential data at {path}"
                    )
                if lowered.endswith(_ARCHIVE_SUFFIXES):
                    raise WorkspacePolicyError(f"request contains a raw archive handle at {path}")
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise WorkspacePolicyError(f"request contains a non-finite number at {path}")
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str) or not key:
                    raise WorkspacePolicyError(f"request object keys must be strings at {path}")
                lowered_key = key.lower()
                if lowered_key in _SECRET_KEYS or lowered_key.endswith(
                    ("_api_key", "_password", "_secret")
                ):
                    raise WorkspacePolicyError(f"request contains a credential key at {path}.{key}")
                if lowered_key in _ALIGNMENT_KEYS:
                    raise WorkspacePolicyError(
                        f"request contains trusted alignment at {path}.{key}"
                    )
                if lowered_key in {"archive", "archive_path", "raw_archive"}:
                    raise WorkspacePolicyError(
                        f"request contains a raw archive key at {path}.{key}"
                    )
                if lowered_key in {"scorer", "protected_scorer", "evaluator"}:
                    raise WorkspacePolicyError(f"request contains scorer authority at {path}.{key}")
                if (
                    split_role in {SplitRole.INNER_VALID, SplitRole.OUTER_VALID, SplitRole.FINAL}
                    and lowered_key in _PROTECTED_REQUEST_KEYS
                ):
                    raise WorkspacePolicyError(
                        f"request contains protected outcomes at {path}.{key}"
                    )
                self._scan_request_value(child, split_role=split_role, path=f"{path}.{key}")
            return
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for index, child in enumerate(value):
                self._scan_request_value(child, split_role=split_role, path=f"{path}[{index}]")
            return
        raise WorkspacePolicyError(f"request contains a non-JSON value at {path}")
