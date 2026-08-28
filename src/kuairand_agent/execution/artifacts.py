"""Crash-safe content-addressed artifacts for trusted campaign evidence.

Objects are immutable regular files addressed by their SHA-256 digest.  New bytes are streamed
into a private same-filesystem staging file, flushed, atomically linked into the object namespace
without replacement, and only then returned to callers.  Directory artifacts are canonical JSON
manifests of file-object references; mutable directory trees are never artifact identities.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final

ARTIFACT_SCHEMA_VERSION: Final = 1
DEFAULT_MAX_OBJECT_BYTES: Final = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_DIRECTORY_BYTES: Final = 8 * 1024 * 1024 * 1024
DEFAULT_CHUNK_BYTES: Final = 1024 * 1024
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactError(RuntimeError):
    """Base class for artifact policy, persistence, and integrity failures."""


class ArtifactPolicyError(ArtifactError):
    """The requested source or reference violates the artifact contract."""


class ArtifactTooLargeError(ArtifactPolicyError):
    """An individual or aggregate artifact exceeded its declared ceiling."""


class ArtifactIntegrityError(ArtifactError):
    """Committed object bytes or metadata no longer match their reference."""


class ArtifactCollisionError(ArtifactIntegrityError):
    """A digest path already exists but cannot be verified as the same object."""


class ArtifactPersistenceError(ArtifactError):
    """The local filesystem could not durably stage an otherwise valid artifact."""


class ArtifactKind(StrEnum):
    """Evidence categories; object identity itself remains content-only."""

    SOURCE = "source"
    INPUT = "input"
    CHECKPOINT = "checkpoint"
    PREDICTION = "prediction"
    LOG = "log"
    OUTPUT = "output"
    MANIFEST = "manifest"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """One immutable content object with validated portable metadata."""

    sha256: str
    size_bytes: int
    kind: ArtifactKind
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactPolicyError("artifact reference schema_version must be 1")
        if not isinstance(self.sha256, str) or _DIGEST_RE.fullmatch(self.sha256) is None:
            raise ArtifactPolicyError("artifact sha256 must be a lowercase SHA-256 digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ArtifactPolicyError("artifact size_bytes must be a non-negative integer")
        if not isinstance(self.kind, ArtifactKind):
            raise ArtifactPolicyError("artifact kind must be an ArtifactKind")

    @property
    def object_relative_path(self) -> PurePosixPath:
        return PurePosixPath("objects", "sha256", self.sha256[:2], self.sha256)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "algorithm": "sha256",
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind.value,
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> ArtifactRef:
        """Strictly restore a portable reference without accepting schema drift."""

        expected_keys = {"schema_version", "algorithm", "sha256", "size_bytes", "kind"}
        if set(value) != expected_keys or value.get("algorithm") != "sha256":
            raise ArtifactPolicyError("artifact manifest schema is invalid")
        schema_version = value.get("schema_version")
        digest = value.get("sha256")
        size_bytes = value.get("size_bytes")
        kind_value = value.get("kind")
        if type(schema_version) is not int:
            raise ArtifactPolicyError("artifact manifest schema_version must be an integer")
        if not isinstance(digest, str):
            raise ArtifactPolicyError("artifact manifest sha256 must be a string")
        if type(size_bytes) is not int:
            raise ArtifactPolicyError("artifact manifest size_bytes must be an integer")
        if not isinstance(kind_value, str):
            raise ArtifactPolicyError("artifact manifest kind must be a string")
        try:
            kind = ArtifactKind(kind_value)
        except ValueError as error:
            raise ArtifactPolicyError("artifact manifest kind is unsupported") from error
        return cls(digest, size_bytes, kind, schema_version)


@dataclass(frozen=True, slots=True)
class DirectoryEntryRef:
    """One portable path-to-object edge inside a directory artifact."""

    path: str
    artifact: ArtifactRef

    def manifest(self) -> dict[str, object]:
        return {"path": self.path, "artifact": self.artifact.manifest()}


@dataclass(frozen=True, slots=True)
class DirectoryArtifactRef:
    """A deterministic manifest plus all referenced immutable file objects."""

    kind: ArtifactKind
    manifest_artifact: ArtifactRef
    entries: tuple[DirectoryEntryRef, ...]
    total_size_bytes: int
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactPolicyError("directory artifact schema_version must be 1")
        if not isinstance(self.kind, ArtifactKind) or self.kind is ArtifactKind.MANIFEST:
            raise ArtifactPolicyError("directory artifact kind must be a non-manifest ArtifactKind")
        if self.manifest_artifact.kind is not ArtifactKind.MANIFEST:
            raise ArtifactPolicyError("directory manifest must use ArtifactKind.MANIFEST")
        if type(self.total_size_bytes) is not int or self.total_size_bytes < 0:
            raise ArtifactPolicyError("directory total_size_bytes must be non-negative")
        paths = tuple(entry.path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ArtifactPolicyError("directory entries must have unique sorted paths")
        if sum(entry.artifact.size_bytes for entry in self.entries) != self.total_size_bytes:
            raise ArtifactPolicyError("directory total_size_bytes does not match its entries")
        if any(entry.artifact.kind is not self.kind for entry in self.entries):
            raise ArtifactPolicyError("directory file artifact kinds must match the directory kind")

    @property
    def sha256(self) -> str:
        """The directory identity is the canonical manifest object's digest."""

        return self.manifest_artifact.sha256

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "total_size_bytes": self.total_size_bytes,
            "entries": [entry.manifest() for entry in self.entries],
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_relative_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\0" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ArtifactPolicyError("directory artifact path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactPolicyError(f"directory artifact path is unsafe: {value!r}")
    if path.as_posix() != value:
        raise ArtifactPolicyError(f"directory artifact path is not canonical: {value!r}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactPolicyError(f"artifact store path must be a real directory: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ArtifactPolicyError(f"artifact store path must be private: {path}")


class ArtifactStore:
    """Streaming SHA-256 object store rooted inside one run directory."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
        max_directory_bytes: int = DEFAULT_MAX_DIRECTORY_BYTES,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        staging_writer: Callable[[BinaryIO, bytes], int | None] | None = None,
    ) -> None:
        if type(max_object_bytes) is not int or max_object_bytes <= 0:
            raise ArtifactPolicyError("max_object_bytes must be a positive integer")
        if type(max_directory_bytes) is not int or max_directory_bytes <= 0:
            raise ArtifactPolicyError("max_directory_bytes must be a positive integer")
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ArtifactPolicyError("chunk_bytes must be a positive integer")
        if staging_writer is not None and not callable(staging_writer):
            raise ArtifactPolicyError("staging_writer must be callable or absent")
        self.root = Path(root)
        self.objects_root = self.root / "objects" / "sha256"
        self.staging_root = self.root / "staging"
        self.max_object_bytes = max_object_bytes
        self.max_directory_bytes = max_directory_bytes
        self.chunk_bytes = chunk_bytes
        self._staging_writer = staging_writer
        _ensure_directory(self.root)
        _ensure_directory(self.root / "objects")
        _ensure_directory(self.objects_root)
        _ensure_directory(self.staging_root)

    def _validate_layout(self) -> None:
        """Fail closed if a managed path was replaced or made non-private."""

        for path in (
            self.root,
            self.root / "objects",
            self.objects_root,
            self.staging_root,
        ):
            _ensure_directory(path)

    def object_path(self, ref: ArtifactRef) -> Path:
        """Return the derived path only after reference validation."""

        # ArtifactRef validates itself at construction; rebuilding is a defense against unsafe
        # objects created through serialization tricks that bypassed normal construction.
        validated = ArtifactRef(ref.sha256, ref.size_bytes, ref.kind, ref.schema_version)
        return self.root.joinpath(*validated.object_relative_path.parts)

    def put_bytes(
        self,
        payload: bytes,
        *,
        kind: ArtifactKind,
        max_bytes: int | None = None,
    ) -> ArtifactRef:
        """Store in-memory bytes through the same staged streaming path as files."""

        if not isinstance(payload, bytes):
            raise ArtifactPolicyError("artifact payload must be bytes")
        limit = self._effective_limit(max_bytes)
        if len(payload) > limit:
            raise ArtifactTooLargeError(
                f"artifact size {len(payload)} exceeds the {limit}-byte ceiling"
            )
        return self._put_chunks(
            (
                payload[index : index + self.chunk_bytes]
                for index in range(0, len(payload), self.chunk_bytes)
            ),
            kind=kind,
            limit=limit,
        )

    def put_file(
        self,
        source: Path | str,
        *,
        kind: ArtifactKind,
        max_bytes: int | None = None,
    ) -> ArtifactRef:
        """Stream a stable regular non-symlink source into the store."""

        source_path = Path(source)
        try:
            initial = source_path.lstat()
        except OSError as error:
            raise ArtifactPolicyError(
                f"artifact source cannot be inspected: {source_path}"
            ) from error
        if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
            raise ArtifactPolicyError("artifact source must be a regular non-symlink file")
        limit = self._effective_limit(max_bytes)
        if initial.st_size > limit:
            raise ArtifactTooLargeError(
                f"artifact size {initial.st_size} exceeds the {limit}-byte ceiling"
            )

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source_path, flags)
        except OSError as error:
            raise ArtifactPolicyError("artifact source could not be opened safely") from error
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ArtifactPolicyError("artifact source changed before it was opened")
            if (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino):
                raise ArtifactPolicyError("artifact source changed before it was opened")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                ref = self._put_stream(handle, kind=kind, limit=limit)
            final = os.fstat(descriptor)
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(opened, field) != getattr(final, field) for field in stable_fields):
                raise ArtifactPolicyError("artifact source changed while it was being copied")
            return ref
        finally:
            os.close(descriptor)

    def put_directory(
        self,
        source: Path | str,
        *,
        kind: ArtifactKind,
        max_total_bytes: int | None = None,
    ) -> DirectoryArtifactRef:
        """Store a tree as sorted file objects plus a canonical manifest object."""

        if kind is ArtifactKind.MANIFEST:
            raise ArtifactPolicyError("directory content kind cannot be manifest")
        source_root = Path(source)
        try:
            root_metadata = source_root.lstat()
        except OSError as error:
            raise ArtifactPolicyError("directory artifact source cannot be inspected") from error
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise ArtifactPolicyError("directory artifact source must be a real directory")
        total_limit = self.max_directory_bytes if max_total_bytes is None else max_total_bytes
        if type(total_limit) is not int or total_limit <= 0:
            raise ArtifactPolicyError("max_total_bytes must be a positive integer")

        files: list[tuple[str, Path, os.stat_result]] = []
        seen_inodes: set[tuple[int, int]] = set()
        for candidate in sorted(source_root.rglob("*")):
            relative = candidate.relative_to(source_root).as_posix()
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ArtifactPolicyError(f"directory artifact contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactPolicyError(f"directory artifact contains a special file: {relative}")
            _validate_relative_path(relative)
            inode = (metadata.st_dev, metadata.st_ino)
            if metadata.st_nlink != 1 or inode in seen_inodes:
                raise ArtifactPolicyError(
                    f"directory artifact contains a hardlinked file: {relative}"
                )
            seen_inodes.add(inode)
            files.append((relative, candidate, metadata))

        total_size = sum(metadata.st_size for _, _, metadata in files)
        if total_size > total_limit:
            raise ArtifactTooLargeError(
                f"directory artifact size {total_size} exceeds the {total_limit}-byte ceiling"
            )
        entries = tuple(
            DirectoryEntryRef(relative, self.put_file(path, kind=kind))
            for relative, path, _ in files
        )
        directory = DirectoryArtifactRef(
            kind=kind,
            manifest_artifact=ArtifactRef("0" * 64, 0, ArtifactKind.MANIFEST),
            entries=entries,
            total_size_bytes=total_size,
        )
        manifest_ref = self.put_bytes(
            _canonical_json(directory.manifest()), kind=ArtifactKind.MANIFEST
        )
        return DirectoryArtifactRef(kind, manifest_ref, entries, total_size)

    def verify(self, ref: ArtifactRef) -> Path:
        """Rehash one object and reject replacement, links, permission drift, or truncation."""

        self._validate_layout()
        path = self.object_path(ref)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ArtifactIntegrityError(f"artifact object is missing: {ref.sha256}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactIntegrityError("artifact object must remain a regular non-symlink file")
        if metadata.st_nlink != 1:
            raise ArtifactIntegrityError("artifact object must not have external hardlinks")
        if metadata.st_mode & 0o222:
            raise ArtifactIntegrityError("artifact object must remain read-only")
        if metadata.st_size != ref.size_bytes:
            raise ArtifactIntegrityError("artifact object size does not match its reference")
        digest, size = self._hash_path(path, limit=ref.size_bytes)
        if size != ref.size_bytes or digest != ref.sha256:
            raise ArtifactIntegrityError("artifact object digest does not match its reference")
        return path

    def verify_directory(self, ref: DirectoryArtifactRef) -> DirectoryArtifactRef:
        """Verify the canonical manifest and every file object it names."""

        for entry in ref.entries:
            _validate_relative_path(entry.path)
            self.verify(entry.artifact)
        expected_manifest = _canonical_json(ref.manifest())
        actual_manifest = self.read_bytes(ref.manifest_artifact, max_bytes=len(expected_manifest))
        if actual_manifest != expected_manifest:
            raise ArtifactIntegrityError(
                "directory artifact manifest bytes do not match its reference"
            )
        return ref

    def load_directory(self, manifest_ref: ArtifactRef) -> DirectoryArtifactRef:
        """Strictly restore and verify a directory from its one durable manifest reference."""

        if manifest_ref.kind is not ArtifactKind.MANIFEST:
            raise ArtifactPolicyError("directory reference must point to a manifest artifact")
        payload = self.read_bytes(manifest_ref, max_bytes=16 * 1024 * 1024)
        try:
            raw = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ArtifactIntegrityError("directory manifest is not valid JSON") from error
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "kind",
            "total_size_bytes",
            "entries",
        }:
            raise ArtifactIntegrityError("directory manifest schema is invalid")
        if payload != _canonical_json(raw):
            raise ArtifactIntegrityError("directory manifest is not canonical JSON")
        schema_version = raw.get("schema_version")
        kind_value = raw.get("kind")
        total_size = raw.get("total_size_bytes")
        entries_value = raw.get("entries")
        if type(schema_version) is not int or not isinstance(kind_value, str):
            raise ArtifactIntegrityError("directory manifest identity fields are invalid")
        if type(total_size) is not int or not isinstance(entries_value, list):
            raise ArtifactIntegrityError("directory manifest inventory fields are invalid")
        try:
            kind = ArtifactKind(kind_value)
        except ValueError as error:
            raise ArtifactIntegrityError("directory manifest kind is unsupported") from error
        entries: list[DirectoryEntryRef] = []
        for item in entries_value:
            if not isinstance(item, dict) or set(item) != {"path", "artifact"}:
                raise ArtifactIntegrityError("directory manifest entry schema is invalid")
            relative_path = item.get("path")
            artifact_value = item.get("artifact")
            if not isinstance(relative_path, str) or not isinstance(artifact_value, dict):
                raise ArtifactIntegrityError("directory manifest entry values are invalid")
            entries.append(
                DirectoryEntryRef(relative_path, ArtifactRef.from_manifest(artifact_value))
            )
        try:
            restored = DirectoryArtifactRef(
                kind=kind,
                manifest_artifact=manifest_ref,
                entries=tuple(entries),
                total_size_bytes=total_size,
                schema_version=schema_version,
            )
        except ArtifactPolicyError as error:
            raise ArtifactIntegrityError("directory manifest violates artifact policy") from error
        return self.verify_directory(restored)

    def read_bytes(self, ref: ArtifactRef, *, max_bytes: int | None = None) -> bytes:
        """Read a previously verified bounded object."""

        limit = self._effective_limit(max_bytes)
        if ref.size_bytes > limit:
            raise ArtifactTooLargeError(
                f"artifact size {ref.size_bytes} exceeds the {limit}-byte read ceiling"
            )
        return self.verify(ref).read_bytes()

    def _effective_limit(self, requested: int | None) -> int:
        if requested is None:
            return self.max_object_bytes
        if type(requested) is not int or requested <= 0:
            raise ArtifactPolicyError("max_bytes must be a positive integer")
        return min(requested, self.max_object_bytes)

    def _put_stream(self, source: BinaryIO, *, kind: ArtifactKind, limit: int) -> ArtifactRef:
        def chunks() -> Iterable[bytes]:
            while chunk := source.read(self.chunk_bytes):
                if not isinstance(chunk, bytes):
                    raise ArtifactPolicyError("artifact stream must yield bytes")
                yield chunk

        return self._put_chunks(chunks(), kind=kind, limit=limit)

    def _put_chunks(
        self, chunks: Iterable[bytes], *, kind: ArtifactKind, limit: int
    ) -> ArtifactRef:
        if not isinstance(kind, ArtifactKind):
            raise ArtifactPolicyError("artifact kind must be an ArtifactKind")
        self._validate_layout()
        descriptor, temporary_name = tempfile.mkstemp(prefix="object-", dir=self.staging_root)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as destination:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise ArtifactPolicyError("artifact stream must yield bytes")
                    size += len(chunk)
                    if size > limit:
                        raise ArtifactTooLargeError(
                            f"artifact size exceeds the {limit}-byte ceiling"
                        )
                    try:
                        written = (
                            destination.write(chunk)
                            if self._staging_writer is None
                            else self._staging_writer(destination, chunk)
                        )
                    except OSError as error:
                        code = errno.errorcode.get(error.errno or 0, "EIO")
                        raise ArtifactPersistenceError(
                            f"artifact staging write failed ({code})"
                        ) from error
                    if written != len(chunk):
                        raise ArtifactPersistenceError(
                            "artifact staging write did not persist the complete chunk"
                        )
                    digest.update(chunk)
                destination.flush()
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            hex_digest = digest.hexdigest()
            ref = ArtifactRef(hex_digest, size, kind)
            target = self.object_path(ref)
            _ensure_directory(target.parent)
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                try:
                    existing = ArtifactRef(hex_digest, size, kind)
                    self.verify(existing)
                except ArtifactError as error:
                    raise ArtifactCollisionError(
                        f"existing digest path is not the expected object: {hex_digest}"
                    ) from error
            except OSError as error:
                if error.errno == errno.EXDEV:
                    raise ArtifactError(
                        "artifact staging and object roots must share a filesystem"
                    ) from error
                raise
            else:
                _fsync_directory(target.parent)
            temporary.unlink()
            _fsync_directory(self.staging_root)
            return ref
        finally:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()

    def _hash_path(self, path: Path, *, limit: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while chunk := handle.read(self.chunk_bytes):
                    size += len(chunk)
                    if size > limit:
                        raise ArtifactIntegrityError("artifact object exceeds its referenced size")
                    digest.update(chunk)
        finally:
            os.close(descriptor)
        return digest.hexdigest(), size
