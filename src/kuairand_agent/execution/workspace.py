"""Deterministic, private candidate workspace materialization.

Candidate-visible source and capability artifacts are verified and copied byte-for-byte; they are
never hardlinked to canonical objects.  Read-only modes prevent accidental mutation, while the
separate output, temporary, and home directories are writable and explicitly bounded in the
request contract.  Because candidate code runs as the same local user, these modes are robustness
controls rather than a hostile-code security boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kuairand_agent.execution.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
)
from kuairand_agent.execution.policy import (
    ApprovedInput,
    SourceManifest,
    SplitRole,
    WorkspacePolicy,
    WorkspacePolicyError,
)

WORKSPACE_SCHEMA_VERSION: Final = 1
_EXECUTION_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class WorkspaceError(RuntimeError):
    """Workspace creation, persistence, or copy verification failed."""


class WorkspaceExistsError(WorkspaceError):
    """The execution already owns a workspace and it will not be overwritten."""


@dataclass(frozen=True, slots=True)
class WorkspaceSpec:
    execution_id: str
    split_role: SplitRole
    source_snapshot: DirectoryArtifactRef
    approved_inputs: tuple[ApprovedInput, ...]
    request_payload: Mapping[str, object]
    output_limit_bytes: int
    temp_limit_bytes: int


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    relative_path: str
    sha256: str
    size_bytes: int
    source_artifact: ArtifactRef

    def manifest(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_artifact": self.source_artifact.manifest(),
        }


@dataclass(frozen=True, slots=True)
class CandidateWorkspace:
    root: Path
    execution_id: str
    split_role: SplitRole
    source_snapshot_sha256: str
    source_files: tuple[WorkspaceFile, ...]
    input_files: tuple[WorkspaceFile, ...]
    output_limit_bytes: int
    temp_limit_bytes: int
    request_sha256: str
    manifest_digest: str

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def home_dir(self) -> Path:
        return self.root / "home"

    @property
    def request_path(self) -> Path:
        return self.root / "request.json"

    @property
    def process_record_path(self) -> Path:
        return self.root / "process.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "workspace-manifest.json"

    def private_environment(self) -> dict[str, str]:
        """Return only private workspace path overrides; the runner owns the full allowlist."""

        return {
            "HOME": str(self.home_dir),
            "TMPDIR": str(self.tmp_dir),
            "XDG_CACHE_HOME": str(self.home_dir / ".cache"),
            "XDG_CONFIG_HOME": str(self.home_dir / ".config"),
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_rmtree(path: Path, *, parent: Path) -> None:
    """Remove only a known unpublished child of the configured workspace root."""

    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_path_parent = path.parent.resolve(strict=True)
    except OSError:
        return
    if resolved_path_parent != resolved_parent or not path.name.startswith(".staging-"):
        raise WorkspaceError("refusing to clean a path outside workspace staging")
    shutil.rmtree(path, ignore_errors=True)


class WorkspaceMaterializer:
    """Build and atomically publish fresh candidate workspaces."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        artifact_store: ArtifactStore,
        policy: WorkspacePolicy | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.artifact_store = artifact_store
        self.policy = policy or WorkspacePolicy()
        self.workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = self.workspace_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WorkspaceError("workspace root must be a real directory")
        os.chmod(self.workspace_root, 0o700, follow_symlinks=False)

    def materialize(self, spec: WorkspaceSpec) -> CandidateWorkspace:
        """Verify, copy, fsync, validate, and publish one non-overwriting workspace."""

        self._validate_spec(spec)
        target = self.workspace_root / spec.execution_id
        if os.path.lexists(target):
            raise WorkspaceExistsError(f"workspace already exists: {spec.execution_id}")
        lock_path = self.workspace_root / f".{spec.execution_id}.publish.lock"
        lock_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        try:
            lock_descriptor = os.open(lock_path, lock_flags, 0o600)
        except FileExistsError as error:
            raise WorkspaceExistsError(
                f"workspace publication is already in progress: {spec.execution_id}"
            ) from error

        staging: Path | None = None
        try:
            os.write(lock_descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(lock_descriptor)
            os.close(lock_descriptor)
            lock_descriptor = -1
            if os.path.lexists(target):
                raise WorkspaceExistsError(f"workspace already exists: {spec.execution_id}")
            staging = Path(
                tempfile.mkdtemp(prefix=f".staging-{spec.execution_id}-", dir=self.workspace_root)
            )
            os.chmod(staging, 0o700, follow_symlinks=False)
            candidate = self._populate(staging, spec)
            self.policy.validate_workspace(candidate)
            if os.path.lexists(target):
                raise WorkspaceExistsError(f"workspace already exists: {spec.execution_id}")
            os.rename(staging, target)
            staging = None
            _fsync_directory(self.workspace_root)
            published = CandidateWorkspace(
                root=target,
                execution_id=candidate.execution_id,
                split_role=candidate.split_role,
                source_snapshot_sha256=candidate.source_snapshot_sha256,
                source_files=candidate.source_files,
                input_files=candidate.input_files,
                output_limit_bytes=candidate.output_limit_bytes,
                temp_limit_bytes=candidate.temp_limit_bytes,
                request_sha256=candidate.request_sha256,
                manifest_digest=candidate.manifest_digest,
            )
            self.policy.validate_workspace(published)
            return published
        finally:
            if "lock_descriptor" in locals() and lock_descriptor >= 0:
                with suppress(OSError):
                    os.close(lock_descriptor)
            if staging is not None and staging.exists():
                _safe_rmtree(staging, parent=self.workspace_root)
            with suppress(FileNotFoundError):
                lock_path.unlink()
            with suppress(OSError):
                _fsync_directory(self.workspace_root)

    def remove(self, workspace: CandidateWorkspace) -> None:
        """Safely remove one exact trusted workspace without following links.

        Only the deterministic request and workspace manifest are trusted for removal identity;
        mutable output, temporary, home, and process contents may legitimately have changed.  The
        method restores write permission on real directories only.  It never chmods files, so a
        candidate-created hardlink cannot alter permissions on an inode outside the workspace.
        """

        expected = self.workspace_root / workspace.execution_id
        if workspace.root != expected:
            raise WorkspaceError("workspace does not belong to this materializer")
        if workspace.root.parent.resolve(strict=True) != self.workspace_root.resolve(strict=True):
            raise WorkspaceError("workspace parent identity does not match")
        self.policy.validate_workspace_identity(workspace)
        self._remove_tree_without_following_links(workspace.root)
        _fsync_directory(self.workspace_root)

    def cleanup(self, workspace: CandidateWorkspace) -> None:
        """Alias for :meth:`remove` for fixture and runner finalizers."""

        self.remove(workspace)

    def _validate_spec(self, spec: WorkspaceSpec) -> None:
        if not isinstance(spec, WorkspaceSpec):
            raise WorkspacePolicyError("workspace spec must be a WorkspaceSpec")
        if _EXECUTION_ID_RE.fullmatch(spec.execution_id) is None or spec.execution_id in {
            ".",
            "..",
        }:
            raise WorkspacePolicyError("execution_id is not a safe workspace name")
        if not isinstance(spec.split_role, SplitRole):
            raise WorkspacePolicyError("split_role must be a SplitRole")
        if spec.source_snapshot.kind is not ArtifactKind.SOURCE:
            raise WorkspacePolicyError("source snapshot must use ArtifactKind.SOURCE")
        source_manifest = SourceManifest.from_directory_artifact(spec.source_snapshot)
        self.policy.validate_source_manifest(source_manifest)
        self.artifact_store.verify_directory(spec.source_snapshot)
        self.policy.validate_approved_inputs(spec.split_role, spec.approved_inputs)
        for approved in spec.approved_inputs:
            self.artifact_store.verify(approved.artifact)
        self.policy.validate_request_payload(spec.split_role, spec.request_payload)
        if (
            type(spec.output_limit_bytes) is not int
            or not 0 < spec.output_limit_bytes <= self.policy.max_output_total_bytes
        ):
            raise WorkspacePolicyError("output_limit_bytes is outside the workspace policy")
        if (
            type(spec.temp_limit_bytes) is not int
            or not 0 < spec.temp_limit_bytes <= self.policy.max_temp_bytes
        ):
            raise WorkspacePolicyError("temp_limit_bytes is outside the workspace policy")

    def _populate(self, root: Path, spec: WorkspaceSpec) -> CandidateWorkspace:
        source_dir = root / "source"
        inputs_dir = root / "inputs"
        output_dir = root / "output"
        tmp_dir = root / "tmp"
        home_dir = root / "home"
        for directory in (source_dir, inputs_dir, output_dir, tmp_dir, home_dir):
            directory.mkdir(mode=0o700)

        source_files: list[WorkspaceFile] = []
        for entry in spec.source_snapshot.entries:
            destination = source_dir.joinpath(*Path(entry.path).parts)
            copied = self._copy_artifact(entry.artifact, destination)
            source_files.append(
                WorkspaceFile(
                    relative_path=(Path("source") / entry.path).as_posix(),
                    sha256=copied.sha256,
                    size_bytes=copied.size_bytes,
                    source_artifact=entry.artifact,
                )
            )

        input_files: list[WorkspaceFile] = []
        input_manifests: list[dict[str, object]] = []
        for index, approved in enumerate(spec.approved_inputs):
            relative = f"inputs/{index:03d}-{approved.name}.artifact"
            copied = self._copy_artifact(approved.artifact, root / relative)
            input_files.append(
                WorkspaceFile(
                    relative_path=relative,
                    sha256=copied.sha256,
                    size_bytes=copied.size_bytes,
                    source_artifact=approved.artifact,
                )
            )
            input_manifests.append(approved.manifest(relative))

        payload_bytes = self.policy.validate_request_payload(spec.split_role, spec.request_payload)
        trusted_request = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "execution_id": spec.execution_id,
            "split_role": spec.split_role.value,
            "source_snapshot_sha256": spec.source_snapshot.sha256,
            "approved_inputs": input_manifests,
            "budgets": {
                "output_limit_bytes": spec.output_limit_bytes,
                "temp_limit_bytes": spec.temp_limit_bytes,
            },
            "request": json.loads(payload_bytes),
        }
        request_bytes = _canonical_json(trusted_request)
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        self._write_file(root / "request.json", request_bytes, mode=0o444)

        process_record = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "execution_id": spec.execution_id,
            "state": "materialized",
        }
        self._write_file(root / "process.json", _canonical_json(process_record), mode=0o600)

        manifest_body = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "execution_id": spec.execution_id,
            "split_role": spec.split_role.value,
            "source_snapshot_sha256": spec.source_snapshot.sha256,
            "source_files": [file.manifest() for file in source_files],
            "input_files": [file.manifest() for file in input_files],
            "request_sha256": request_sha256,
            "output_limit_bytes": spec.output_limit_bytes,
            "temp_limit_bytes": spec.temp_limit_bytes,
        }
        manifest_digest = hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
        workspace_manifest = {**manifest_body, "manifest_digest": manifest_digest}
        self._write_file(
            root / "workspace-manifest.json", _canonical_json(workspace_manifest), mode=0o444
        )

        self._make_tree_read_only(source_dir)
        self._make_tree_read_only(inputs_dir)
        for directory in (output_dir, tmp_dir, home_dir):
            os.chmod(directory, 0o700, follow_symlinks=False)
        os.chmod(root, 0o700, follow_symlinks=False)
        self._fsync_tree(root)
        return CandidateWorkspace(
            root=root,
            execution_id=spec.execution_id,
            split_role=spec.split_role,
            source_snapshot_sha256=spec.source_snapshot.sha256,
            source_files=tuple(source_files),
            input_files=tuple(input_files),
            output_limit_bytes=spec.output_limit_bytes,
            temp_limit_bytes=spec.temp_limit_bytes,
            request_sha256=request_sha256,
            manifest_digest=manifest_digest,
        )

    def _copy_artifact(self, ref: ArtifactRef, destination: Path) -> ArtifactRef:
        source = self.artifact_store.verify(ref)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_metadata = destination.parent.lstat()
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
            raise WorkspaceError("workspace copy parent must be a real directory")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(destination, flags, 0o600)
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                source.open("rb") as input_handle,
                os.fdopen(descriptor, "wb", closefd=False) as output_handle,
            ):
                while chunk := input_handle.read(1024 * 1024):
                    size += len(chunk)
                    if size > ref.size_bytes:
                        raise WorkspaceError("artifact grew while being copied into a workspace")
                    output_handle.write(chunk)
                    digest.update(chunk)
                output_handle.flush()
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            if size != ref.size_bytes or digest.hexdigest() != ref.sha256:
                raise WorkspaceError("workspace copy does not match its artifact reference")
        finally:
            os.close(descriptor)
        destination_metadata = destination.lstat()
        source_metadata = source.lstat()
        if (destination_metadata.st_dev, destination_metadata.st_ino) == (
            source_metadata.st_dev,
            source_metadata.st_ino,
        ):
            raise WorkspaceError("candidate-visible artifacts must not be hardlinked")
        return ref

    @staticmethod
    def _write_file(path: Path, payload: bytes, *, mode: int) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _make_tree_read_only(root: Path) -> None:
        directories: list[Path] = []
        for path in root.rglob("*"):
            if path.is_dir():
                directories.append(path)
            else:
                os.chmod(path, 0o444, follow_symlinks=False)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            os.chmod(directory, 0o555, follow_symlinks=False)
        os.chmod(root, 0o555, follow_symlinks=False)

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _fsync_directory(directory)

    @classmethod
    def _remove_tree_without_following_links(cls, directory: Path) -> None:
        metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WorkspaceError("cleanup target must remain a real directory")
        os.chmod(directory, 0o700, follow_symlinks=False)
        with os.scandir(directory) as entries:
            children = tuple(entries)
        for entry in children:
            child = Path(entry.path)
            child_metadata = child.lstat()
            if stat.S_ISDIR(child_metadata.st_mode) and not stat.S_ISLNK(child_metadata.st_mode):
                cls._remove_tree_without_following_links(child)
            else:
                child.unlink()
        directory.rmdir()
