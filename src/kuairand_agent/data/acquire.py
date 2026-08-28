"""Hash-pinned, fail-closed KuaiRand-Pure archive verification and installation.

Acquisition deliberately treats member payloads as opaque bytes.  The only content inspection in
this module is a byte-for-byte comparison of each declared CSV header; no data row is parsed.
"""

from __future__ import annotations

import ctypes
import errno
import gzip
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import unicodedata
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Final, Self, cast

from kuairand_agent.contract import DATASET_ARCHIVE_MD5, DATASET_ARCHIVE_SHA256

ARCHIVE_FILENAME: Final = "KuaiRand-Pure.tar.gz"
ARCHIVE_SIZE_BYTES: Final = 47_432_272
ARCHIVE_TAR_SIZE_BYTES: Final = 203_547_136
ARCHIVE_PAYLOAD_SIZE_BYTES: Final = 203_539_133
ARCHIVE_SOURCE: Final = "zenodo:10439422:file:8d31ed3f-6639-4649-9201-96d87a107e1f"
OFFICIAL_ARCHIVE_URL: Final = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
INTEGRITY_MANIFEST_FILENAME: Final = "acquisition-integrity.json"

_BLOCK_SIZE: Final = 512
_COPY_CHUNK_SIZE: Final = 1024 * 1024
_MAX_HEADER_BYTES: Final = 64 * 1024
_MAX_ARCHIVE_BYTES: Final = 64 * 1024 * 1024
_MAX_TAR_BYTES: Final = 256 * 1024 * 1024
_MAX_MEMBER_COUNT: Final = 64
_LOWER_HEX: Final = frozenset("0123456789abcdef")


class ArchiveIntegrityError(RuntimeError):
    """Raised when an archive or extraction destination violates the frozen contract."""


class MemberType(StrEnum):
    """The only tar member types accepted by the acquisition boundary."""

    DIRECTORY = "directory"
    REGULAR = "regular"


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """One ordered, byte-pinned tar member."""

    name: str
    type: MemberType
    size: int
    sha256: str | None = None
    header: tuple[str, ...] | None = None

    def manifest(self) -> dict[str, object]:
        """Return the deterministic logical representation saved after extraction."""

        return {
            "name": self.name,
            "type": self.type.value,
            "size": self.size,
            "sha256": self.sha256,
            "header": None if self.header is None else list(self.header),
        }


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    """Complete expected identity, including the ordered logical tar manifest.

    ``spec`` parameters on :func:`verify_archive` and :func:`prepare_archive` are the intentional
    synthetic-fixture seam.  Production callers use :data:`OFFICIAL_ARCHIVE_SPEC`.
    """

    source: str
    filename: str
    size: int
    md5: str
    sha256: str
    tar_size: int
    members: tuple[ArchiveMember, ...]

    @property
    def payload_size(self) -> int:
        """Return the declared sum of regular-file bytes."""

        return sum(member.size for member in self.members if member.type is MemberType.REGULAR)

    @property
    def root_name(self) -> str:
        """Return the single top-level archive directory without its trailing slash."""

        return self.members[0].name[:-1]

    def validate(self) -> Self:
        """Reject malformed fixture specs before opening any archive or destination."""

        if not self.source or any(character in self.source for character in "\x00\r\n"):
            raise ArchiveIntegrityError("archive source identity is empty or malformed")
        if not self.filename or self.filename != Path(self.filename).name:
            raise ArchiveIntegrityError("archive filename must be one plain basename")
        if not 0 < self.size <= _MAX_ARCHIVE_BYTES:
            raise ArchiveIntegrityError("compressed archive size exceeds the acquisition bound")
        if not _is_lower_hex(self.md5, 32):
            raise ArchiveIntegrityError("archive MD5 must be 32 lowercase hexadecimal characters")
        if not _is_lower_hex(self.sha256, 64):
            raise ArchiveIntegrityError(
                "archive SHA-256 must be 64 lowercase hexadecimal characters"
            )
        if (
            self.tar_size < 2 * _BLOCK_SIZE
            or self.tar_size > _MAX_TAR_BYTES
            or self.tar_size % _BLOCK_SIZE != 0
        ):
            raise ArchiveIntegrityError("uncompressed tar size violates the fixed block bound")
        if not self.members or len(self.members) > _MAX_MEMBER_COUNT:
            raise ArchiveIntegrityError("ordered member manifest is empty or too large")

        seen: set[str] = set()
        casefolded: set[str] = set()
        minimum_tar_size = (len(self.members) + 2) * _BLOCK_SIZE
        for member in self.members:
            _validate_member_name(member.name, expected_type=member.type)
            logical_name = member.name.removesuffix("/")
            collision_key = _collision_key(logical_name)
            if logical_name in seen:
                raise ArchiveIntegrityError(f"duplicate expected member: {member.name}")
            if collision_key in casefolded:
                raise ArchiveIntegrityError(f"case-colliding expected member: {member.name}")
            seen.add(logical_name)
            casefolded.add(collision_key)

            if member.size < 0:
                raise ArchiveIntegrityError(f"negative expected member size: {member.name}")
            if member.type is MemberType.DIRECTORY:
                if member.size != 0 or member.sha256 is not None or member.header is not None:
                    raise ArchiveIntegrityError(
                        "directory manifest must have zero size and no content identity: "
                        f"{member.name}"
                    )
            else:
                if member.sha256 is None or not _is_lower_hex(member.sha256, 64):
                    raise ArchiveIntegrityError(
                        f"regular member lacks a lowercase SHA-256: {member.name}"
                    )
                if member.header is not None:
                    _validate_expected_header(member)
                minimum_tar_size += _padded_size(member.size)

        if self.members[0].type is not MemberType.DIRECTORY:
            raise ArchiveIntegrityError("first archive member must be the top-level directory")
        root = self.members[0].name
        if root.count("/") != 1:
            raise ArchiveIntegrityError("archive root must be one top-level directory")
        for member in self.members[1:]:
            if not member.name.startswith(root):
                raise ArchiveIntegrityError(
                    f"member escapes the declared archive root: {member.name}"
                )
        if self.payload_size > ARCHIVE_PAYLOAD_SIZE_BYTES:
            raise ArchiveIntegrityError("declared member payload exceeds the KuaiRand-Pure bound")
        if self.tar_size < minimum_tar_size:
            raise ArchiveIntegrityError(
                "declared tar size cannot contain its members and terminator"
            )
        return self

    def manifest(self) -> dict[str, object]:
        """Return a path- and time-independent integrity manifest."""

        self.validate()
        return {
            "schema_version": 1,
            "source": self.source,
            "archive": {
                "filename": self.filename,
                "size": self.size,
                "md5": self.md5,
                "sha256": self.sha256,
                "tar_size": self.tar_size,
                "payload_size": self.payload_size,
            },
            "members": [member.manifest() for member in self.members],
        }


@dataclass(frozen=True, slots=True)
class ArchiveVerification:
    """Evidence returned after one-fd archive and member verification."""

    archive: Path
    archive_size: int
    archive_md5: str
    archive_sha256: str
    tar_size: int
    payload_size: int
    members: tuple[ArchiveMember, ...]
    member_sha256: MappingProxyType[str, str]


@dataclass(frozen=True, slots=True)
class PreparedArchive:
    """A fully fsynced archive installation and its deterministic evidence path."""

    destination: Path
    dataset_root: Path
    integrity_manifest: Path
    manifest_sha256: str
    verification: ArchiveVerification


@dataclass(frozen=True, slots=True)
class _ParsedHeader:
    name: str
    type: MemberType
    size: int
    mode: int


@dataclass(frozen=True, slots=True)
class _StatIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _StatIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )


class _BoundedTarReader:
    """Count every decompressed byte and forbid reads beyond the pinned tar size."""

    def __init__(self, stream: gzip.GzipFile, expected_size: int) -> None:
        self._stream = stream
        self._expected_size = expected_size
        self.total = 0

    def read_exact(self, size: int) -> bytes:
        if size < 0 or self.total + size > self._expected_size:
            raise ArchiveIntegrityError("tar stream exceeds its pinned uncompressed size")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._stream.read(remaining)
            if not chunk:
                raise ArchiveIntegrityError("tar stream ended before its pinned size")
            chunks.append(chunk)
            self.total += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def require_eof(self) -> None:
        if self.total != self._expected_size:
            raise ArchiveIntegrityError("tar stream size differs from its pinned identity")
        if self._stream.read(1):
            raise ArchiveIntegrityError("tar stream contains bytes beyond its pinned size")


class _ExtractionTarget:
    """Directory-fd-anchored writer for a private staging directory."""

    def __init__(self, root: Path) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | _o_nofollow()
        self._root_fd = os.open(root, flags)
        self._directories: dict[str, int] = {"": self._root_fd}
        self._closed = False

    def create_directory(self, member_name: str) -> None:
        logical = member_name.removesuffix("/")
        parent, _, basename = logical.rpartition("/")
        parent_fd = self._directories.get(parent)
        if parent_fd is None:
            raise ArchiveIntegrityError(
                f"archive directory parent appears out of order: {member_name}"
            )
        try:
            os.mkdir(basename, mode=0o700, dir_fd=parent_fd)
            os.chmod(basename, 0o700, dir_fd=parent_fd, follow_symlinks=False)
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | _o_nofollow()
            child_fd = os.open(basename, flags, dir_fd=parent_fd)
            os.fchmod(child_fd, 0o700)
        except OSError as exc:
            raise ArchiveIntegrityError(
                f"cannot exclusively create archive directory {member_name}: {exc.strerror}"
            ) from exc
        self._directories[logical] = child_fd

    def open_regular(self, member_name: str) -> int:
        parent, _, basename = member_name.rpartition("/")
        parent_fd = self._directories.get(parent)
        if parent_fd is None:
            raise ArchiveIntegrityError(f"archive file parent appears out of order: {member_name}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | _o_nofollow()
        try:
            descriptor = os.open(basename, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(descriptor, 0o600)
            return descriptor
        except OSError as exc:
            raise ArchiveIntegrityError(
                f"cannot exclusively create archive file {member_name}: {exc.strerror}"
            ) from exc

    def write_manifest(self, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | _o_nofollow()
        try:
            descriptor = os.open(INTEGRITY_MANIFEST_FILENAME, flags, 0o600, dir_fd=self._root_fd)
        except OSError as exc:
            raise ArchiveIntegrityError("cannot exclusively create integrity manifest") from exc
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def sync(self) -> None:
        for descriptor in reversed(tuple(self._directories.values())):
            os.fsync(descriptor)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for logical_name, descriptor in reversed(tuple(self._directories.items())):
            if logical_name == "":
                continue
            os.close(descriptor)
        os.close(self._root_fd)


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in _LOWER_HEX for character in value)


def _collision_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _validate_member_name(name: str, *, expected_type: MemberType) -> None:
    if not name or "\x00" in name:
        raise ArchiveIntegrityError("archive member path is empty or contains NUL")
    if "\\" in name:
        raise ArchiveIntegrityError(f"archive member path contains a backslash: {name!r}")
    if name.startswith("/") or (len(name) >= 2 and name[0].isalpha() and name[1] == ":"):
        raise ArchiveIntegrityError(f"archive member path is absolute: {name!r}")
    if (expected_type is MemberType.DIRECTORY) != name.endswith("/"):
        raise ArchiveIntegrityError(f"archive member slash/type mismatch: {name!r}")
    logical_name = name.removesuffix("/")
    parts = logical_name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArchiveIntegrityError(
            f"archive member has dot, traversal, or empty segment: {name!r}"
        )
    if unicodedata.normalize("NFC", logical_name) != logical_name:
        raise ArchiveIntegrityError(f"archive member path is not canonical Unicode NFC: {name!r}")
    if logical_name == INTEGRITY_MANIFEST_FILENAME:
        raise ArchiveIntegrityError("archive member collides with the trusted integrity manifest")


def _validate_expected_header(member: ArchiveMember) -> None:
    assert member.header is not None
    if not member.header or not member.name.endswith(".csv"):
        raise ArchiveIntegrityError(f"only CSV regular files may declare a header: {member.name}")
    for column in member.header:
        if not column or any(character in column for character in ",\r\n\x00"):
            raise ArchiveIntegrityError(f"malformed expected CSV header for {member.name}")
        try:
            column.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ArchiveIntegrityError(f"CSV header is not ASCII for {member.name}") from exc
    if len(_header_bytes(member)) > _MAX_HEADER_BYTES:
        raise ArchiveIntegrityError(f"CSV header exceeds the acquisition bound: {member.name}")


def _header_bytes(member: ArchiveMember) -> bytes:
    assert member.header is not None
    return ",".join(member.header).encode("ascii") + b"\n"


def _padded_size(size: int) -> int:
    return ((size + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE


def _o_nofollow() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if value is None:
        raise ArchiveIntegrityError("this platform does not provide O_NOFOLLOW")
    return int(value)


def _parse_octal(field: bytes, *, label: str) -> int:
    stripped = field.strip(b"\x00 ")
    if not stripped:
        return 0
    if any(character not in b"01234567" for character in stripped):
        raise ArchiveIntegrityError(f"tar {label} is not canonical octal")
    return int(stripped, 8)


def _decode_tar_string(field: bytes, *, label: str) -> str:
    terminator = field.find(b"\x00")
    if terminator >= 0:
        if any(field[terminator + 1 :]):
            raise ArchiveIntegrityError(f"tar {label} contains embedded NUL data")
        field = field[:terminator]
    try:
        return field.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArchiveIntegrityError(f"tar {label} is not valid UTF-8") from exc


def _parse_header(block: bytes) -> _ParsedHeader:
    if len(block) != _BLOCK_SIZE or not any(block):
        raise ArchiveIntegrityError("expected a non-empty 512-byte tar header")
    expected_checksum = _parse_octal(block[148:156], label="checksum")
    checksum_block = bytearray(block)
    checksum_block[148:156] = b" " * 8
    if sum(checksum_block) != expected_checksum:
        raise ArchiveIntegrityError("tar header checksum mismatch")
    if block[257:263] != b"ustar\x00" or block[263:265] != b"00":
        raise ArchiveIntegrityError("only canonical POSIX ustar headers are accepted")

    raw_name = _decode_tar_string(block[0:100], label="name")
    prefix = _decode_tar_string(block[345:500], label="prefix")
    link_name = _decode_tar_string(block[157:257], label="link name")
    name = f"{prefix}/{raw_name}" if prefix else raw_name
    type_flag = block[156:157]
    if type_flag == b"0":
        member_type = MemberType.REGULAR
    elif type_flag == b"5":
        member_type = MemberType.DIRECTORY
    else:
        raise ArchiveIntegrityError(
            "tar links, sparse files, devices, FIFOs, and extension members are forbidden"
        )
    if link_name:
        raise ArchiveIntegrityError("tar member link target must be empty")
    mode = _parse_octal(block[100:108], label="mode")
    size = _parse_octal(block[124:136], label="size")
    if member_type is MemberType.REGULAR and mode & 0o7111:
        raise ArchiveIntegrityError(f"executable or special permission bits are forbidden: {name}")
    return _ParsedHeader(name=name, type=member_type, size=size, mode=mode)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ArchiveIntegrityError("short write while extracting an archive member")
        view = view[written:]


def _hash_open_file(handle: BinaryIO, *, expected_size: int) -> tuple[int, str, str]:
    handle.seek(0)
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    remaining = expected_size
    while remaining:
        chunk = handle.read(min(_COPY_CHUNK_SIZE, remaining))
        if not chunk:
            raise ArchiveIntegrityError("archive ended before its pinned compressed size")
        size += len(chunk)
        remaining -= len(chunk)
        md5.update(chunk)
        sha256.update(chunk)
    if handle.read(1):
        raise ArchiveIntegrityError("archive exceeds its pinned compressed size")
    return size, md5.hexdigest(), sha256.hexdigest()


def _scan_tar(
    handle: BinaryIO,
    spec: ArchiveSpec,
    extraction: _ExtractionTarget | None,
) -> MappingProxyType[str, str]:
    handle.seek(0)
    observed_hashes: dict[str, str] = {}
    seen: set[str] = set()
    casefolded: set[str] = set()
    try:
        with gzip.GzipFile(fileobj=handle, mode="rb") as gzip_stream:
            reader = _BoundedTarReader(gzip_stream, spec.tar_size)
            for expected in spec.members:
                header_block = reader.read_exact(_BLOCK_SIZE)
                if not any(header_block):
                    raise ArchiveIntegrityError("tar ended before every expected member")
                parsed = _parse_header(header_block)
                _validate_member_name(parsed.name, expected_type=parsed.type)
                logical_name = parsed.name.removesuffix("/")
                collision_key = _collision_key(logical_name)
                if logical_name in seen:
                    raise ArchiveIntegrityError(f"duplicate tar member: {parsed.name}")
                if collision_key in casefolded:
                    raise ArchiveIntegrityError(f"case-colliding tar member: {parsed.name}")
                seen.add(logical_name)
                casefolded.add(collision_key)
                if parsed.name != expected.name:
                    raise ArchiveIntegrityError(
                        f"unexpected or out-of-order tar member: expected {expected.name}, "
                        f"got {parsed.name}"
                    )
                if parsed.type is not expected.type:
                    raise ArchiveIntegrityError(f"tar member type mismatch: {parsed.name}")
                if parsed.size != expected.size:
                    raise ArchiveIntegrityError(
                        f"tar member size mismatch for {parsed.name}: "
                        f"expected {expected.size}, got {parsed.size}"
                    )

                if parsed.type is MemberType.DIRECTORY:
                    if parsed.size != 0:
                        raise ArchiveIntegrityError(f"tar directory has a payload: {parsed.name}")
                    if extraction is not None:
                        extraction.create_directory(parsed.name)
                    continue

                descriptor = None if extraction is None else extraction.open_regular(parsed.name)
                digest = hashlib.sha256()
                expected_header = None if expected.header is None else _header_bytes(expected)
                header_position = 0
                remaining = parsed.size
                try:
                    while remaining:
                        chunk = reader.read_exact(min(remaining, _COPY_CHUNK_SIZE))
                        remaining -= len(chunk)
                        digest.update(chunk)
                        if expected_header is not None and header_position < len(expected_header):
                            inspected = min(len(chunk), len(expected_header) - header_position)
                            if (
                                chunk[:inspected]
                                != expected_header[header_position : header_position + inspected]
                            ):
                                raise ArchiveIntegrityError(
                                    f"CSV header mismatch for {parsed.name}"
                                )
                            header_position += inspected
                        if descriptor is not None:
                            _write_all(descriptor, chunk)
                    if expected_header is not None and header_position != len(expected_header):
                        raise ArchiveIntegrityError(f"CSV header is truncated for {parsed.name}")
                    actual_digest = digest.hexdigest()
                    if actual_digest != expected.sha256:
                        raise ArchiveIntegrityError(
                            f"tar member digest mismatch for {parsed.name}: "
                            f"expected {expected.sha256}, got {actual_digest}"
                        )
                    observed_hashes[parsed.name] = actual_digest
                    if descriptor is not None:
                        extracted_stat = os.fstat(descriptor)
                        if (
                            not stat.S_ISREG(extracted_stat.st_mode)
                            or extracted_stat.st_size != parsed.size
                        ):
                            raise ArchiveIntegrityError(
                                f"extracted member identity mismatch: {parsed.name}"
                            )
                        os.fsync(descriptor)
                finally:
                    if descriptor is not None:
                        os.close(descriptor)

                padding = _padded_size(parsed.size) - parsed.size
                if padding and any(reader.read_exact(padding)):
                    raise ArchiveIntegrityError(f"tar member padding is non-zero: {parsed.name}")

            zero_blocks = 0
            while reader.total < spec.tar_size:
                trailer = reader.read_exact(_BLOCK_SIZE)
                if any(trailer):
                    raise ArchiveIntegrityError("unexpected tar member or non-zero trailer data")
                zero_blocks += 1
            if zero_blocks < 2:
                raise ArchiveIntegrityError("tar is missing its two zero terminator blocks")
            reader.require_eof()
    except ArchiveIntegrityError:
        raise
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        raise ArchiveIntegrityError(f"invalid gzip/tar stream: {exc}") from exc
    return MappingProxyType(observed_hashes)


def _process_archive(
    archive: str | Path,
    *,
    spec: ArchiveSpec,
    extraction: _ExtractionTarget | None,
) -> ArchiveVerification:
    expected = spec.validate()
    archive_path = Path(archive)
    if archive_path.name != expected.filename:
        raise ArchiveIntegrityError(
            f"archive basename mismatch: expected {expected.filename}, got {archive_path.name}"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | _o_nofollow()
    try:
        descriptor = os.open(archive_path, flags)
    except OSError as exc:
        raise ArchiveIntegrityError(f"cannot securely open archive: {archive_path}") from exc
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        initial_stat = _StatIdentity.from_stat(os.fstat(handle.fileno()))
        if not stat.S_ISREG(initial_stat.mode):
            raise ArchiveIntegrityError("archive is not a regular file")
        if initial_stat.size != expected.size:
            raise ArchiveIntegrityError(
                f"archive size mismatch: expected {expected.size}, got {initial_stat.size}"
            )
        size, md5, sha256 = _hash_open_file(handle, expected_size=expected.size)
        if size != expected.size or md5 != expected.md5 or sha256 != expected.sha256:
            raise ArchiveIntegrityError(
                f"archive byte identity mismatch: size={size}, md5={md5}, sha256={sha256}"
            )
        member_hashes = _scan_tar(handle, expected, extraction)
        final_size, final_md5, final_sha256 = _hash_open_file(handle, expected_size=expected.size)
        final_stat = _StatIdentity.from_stat(os.fstat(handle.fileno()))
        if (
            final_stat != initial_stat
            or final_size != size
            or final_md5 != md5
            or final_sha256 != sha256
        ):
            raise ArchiveIntegrityError("archive changed while it was being verified and extracted")
    return ArchiveVerification(
        archive=archive_path.absolute(),
        archive_size=size,
        archive_md5=md5,
        archive_sha256=sha256,
        tar_size=expected.tar_size,
        payload_size=expected.payload_size,
        members=expected.members,
        member_sha256=member_hashes,
    )


def _manifest_bytes(spec: ArchiveSpec) -> bytes:
    return (
        json.dumps(spec.manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex = libc.renamex_np
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        status_code = int(renamex(source_bytes, destination_bytes, 0x00000004))
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        status_code = int(renameat2(-100, source_bytes, -100, destination_bytes, 0x00000001))
    else:
        raise ArchiveIntegrityError(
            "platform lacks an atomic no-replace directory rename primitive"
        )
    if status_code == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ArchiveIntegrityError(f"destination already exists: {destination}")
    raise ArchiveIntegrityError(
        f"cannot atomically install destination: {os.strerror(error_number)}"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | _o_nofollow())
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_archive(
    archive: str | Path,
    *,
    spec: ArchiveSpec | None = None,
) -> ArchiveVerification:
    """Verify an archive and all opaque members without writing extracted payloads."""

    return _process_archive(
        archive,
        spec=OFFICIAL_ARCHIVE_SPEC if spec is None else spec,
        extraction=None,
    )


def prepare_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    spec: ArchiveSpec | None = None,
) -> PreparedArchive:
    """Verify and atomically install an archive into a new destination.

    The destination's parent must already exist.  Existing files, directories, and symlinks are
    never replaced.  A failure removes only this call's private sibling staging directory.
    """

    expected = OFFICIAL_ARCHIVE_SPEC if spec is None else spec
    expected.validate()
    requested_destination = Path(destination)
    if requested_destination.name in {"", ".", ".."}:
        raise ArchiveIntegrityError("destination must name one new directory")
    try:
        parent = requested_destination.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArchiveIntegrityError("destination parent does not exist") from exc
    if not parent.is_dir():
        raise ArchiveIntegrityError("destination parent is not a directory")
    resolved_destination = parent / requested_destination.name
    if _destination_exists(resolved_destination):
        raise ArchiveIntegrityError(f"destination already exists: {resolved_destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{resolved_destination.name}.staging-", dir=parent))
    os.chmod(staging, 0o700)
    extraction: _ExtractionTarget | None = None
    installed = False
    try:
        extraction = _ExtractionTarget(staging)
        manifest_payload = _manifest_bytes(expected)
        extraction.write_manifest(manifest_payload)
        verification = _process_archive(archive, spec=expected, extraction=extraction)
        extraction.sync()
        extraction.close()
        extraction = None
        _rename_exclusive(staging, resolved_destination)
        installed = True
        _fsync_directory(parent)
    finally:
        if extraction is not None:
            extraction.close()
        if not installed:
            try:
                shutil.rmtree(staging)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ArchiveIntegrityError(
                    f"failed to remove private staging directory: {staging}"
                ) from exc

    manifest_path = resolved_destination / INTEGRITY_MANIFEST_FILENAME
    return PreparedArchive(
        destination=resolved_destination,
        dataset_root=resolved_destination / expected.root_name,
        integrity_manifest=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        verification=verification,
    )


def _open_official_download(url: str, timeout_seconds: float) -> BinaryIO:
    if url != OFFICIAL_ARCHIVE_URL:
        raise ArchiveIntegrityError("download URL differs from the pinned organizer source")
    response = urllib.request.urlopen(url, timeout=timeout_seconds)
    final_url = response.geturl()
    if final_url != OFFICIAL_ARCHIVE_URL:
        response.close()
        raise ArchiveIntegrityError(
            f"organizer download redirected away from the pinned URL: {final_url}"
        )
    return cast(BinaryIO, response)


def _stream_download(
    source: BinaryIO,
    destination: Path,
    *,
    spec: ArchiveSpec,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | _o_nofollow()
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as exc:
        raise ArchiveIntegrityError("cannot exclusively create private download file") from exc
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    total = 0
    try:
        os.fchmod(descriptor, 0o600)
        while total < spec.size:
            chunk = source.read(min(_COPY_CHUNK_SIZE, spec.size - total))
            if not isinstance(chunk, bytes):
                raise ArchiveIntegrityError("download stream returned a non-bytes payload")
            if not chunk:
                raise ArchiveIntegrityError(
                    f"download ended early: expected {spec.size} bytes, got {total}"
                )
            if total + len(chunk) > spec.size:
                raise ArchiveIntegrityError("download exceeds the pinned compressed size")
            _write_all(descriptor, chunk)
            total += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
        overflow = source.read(1)
        if not isinstance(overflow, bytes):
            raise ArchiveIntegrityError("download stream returned a non-bytes payload")
        if overflow:
            raise ArchiveIntegrityError("download exceeds the pinned compressed size")
        if md5.hexdigest() != spec.md5 or sha256.hexdigest() != spec.sha256:
            raise ArchiveIntegrityError("download digest differs from the pinned archive identity")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def download_and_prepare(
    destination: str | Path,
    *,
    timeout_seconds: float = 60.0,
    opener: Callable[[str, float], BinaryIO] | None = None,
    spec: ArchiveSpec | None = None,
) -> PreparedArchive:
    """Download the pinned Zenodo artifact privately, then securely prepare it.

    ``opener`` and ``spec`` are solely the deterministic synthetic-test seam.  Normal callers pass
    neither and can reach only the exact HTTPS URL and official archive contract.
    """

    expected = OFFICIAL_ARCHIVE_SPEC if spec is None else spec
    expected.validate()
    if opener is None and expected is not OFFICIAL_ARCHIVE_SPEC:
        raise ArchiveIntegrityError("a synthetic archive spec requires an injected opener")
    if not math.isfinite(timeout_seconds) or not 0.0 < timeout_seconds <= 300.0:
        raise ArchiveIntegrityError("download timeout must be finite and in (0, 300] seconds")

    requested_destination = Path(destination)
    if requested_destination.name in {"", ".", ".."}:
        raise ArchiveIntegrityError("destination must name one new directory")
    try:
        parent = requested_destination.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArchiveIntegrityError("destination parent does not exist") from exc
    if not parent.is_dir():
        raise ArchiveIntegrityError("destination parent is not a directory")
    resolved_destination = parent / requested_destination.name
    if _destination_exists(resolved_destination):
        raise ArchiveIntegrityError(f"destination already exists: {resolved_destination}")

    download_root = Path(
        tempfile.mkdtemp(prefix=f".{resolved_destination.name}.download-", dir=parent)
    )
    os.chmod(download_root, 0o700)
    downloaded_archive = download_root / expected.filename
    open_download = _open_official_download if opener is None else opener
    try:
        try:
            source = open_download(OFFICIAL_ARCHIVE_URL, timeout_seconds)
        except ArchiveIntegrityError:
            raise
        except Exception as exc:
            raise ArchiveIntegrityError("cannot open the pinned organizer download") from exc
        try:
            _stream_download(source, downloaded_archive, spec=expected)
        finally:
            source.close()
        _fsync_directory(download_root)
        return prepare_archive(downloaded_archive, resolved_destination, spec=expected)
    finally:
        try:
            shutil.rmtree(download_root)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ArchiveIntegrityError(
                f"failed to remove private download directory: {download_root}"
            ) from exc


def _columns(value: str) -> tuple[str, ...]:
    return tuple(value.split(","))


_LOG_HEADER: Final = _columns(
    "user_id,video_id,date,hourmin,time_ms,is_click,is_like,is_follow,is_comment,is_forward,"
    "is_hate,long_view,play_time_ms,duration_ms,profile_stay_time,comment_stay_time,"
    "is_profile_enter,is_rand,tab"
)
_USER_HEADER: Final = _columns(
    "user_id,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,"
    "follow_user_num,follow_user_num_range,fans_user_num,fans_user_num_range,friend_user_num,"
    "friend_user_num_range,register_days,register_days_range,onehot_feat0,onehot_feat1,"
    "onehot_feat2,onehot_feat3,onehot_feat4,onehot_feat5,onehot_feat6,onehot_feat7,"
    "onehot_feat8,onehot_feat9,onehot_feat10,onehot_feat11,onehot_feat12,onehot_feat13,"
    "onehot_feat14,onehot_feat15,onehot_feat16,onehot_feat17"
)
_STATISTICS_HEADER: Final = _columns(
    "video_id,counts,show_cnt,show_user_num,play_cnt,play_user_num,play_duration,"
    "complete_play_cnt,complete_play_user_num,valid_play_cnt,valid_play_user_num,"
    "long_time_play_cnt,long_time_play_user_num,short_time_play_cnt,short_time_play_user_num,"
    "play_progress,comment_stay_duration,like_cnt,like_user_num,click_like_cnt,double_click_cnt,"
    "cancel_like_cnt,cancel_like_user_num,comment_cnt,comment_user_num,direct_comment_cnt,"
    "reply_comment_cnt,delete_comment_cnt,delete_comment_user_num,comment_like_cnt,"
    "comment_like_user_num,follow_cnt,follow_user_num,cancel_follow_cnt,cancel_follow_user_num,"
    "share_cnt,share_user_num,download_cnt,download_user_num,report_cnt,report_user_num,"
    "reduce_similar_cnt,reduce_similar_user_num,collect_cnt,collect_user_num,cancel_collect_cnt,"
    "cancel_collect_user_num,direct_comment_user_num,reply_comment_user_num,share_all_cnt,"
    "share_all_user_num,outsite_share_all_cnt"
)
_BASIC_HEADER: Final = _columns(
    "video_id,author_id,video_type,upload_dt,upload_type,visible_status,video_duration,"
    "server_width,server_height,music_id,music_type,tag"
)

OFFICIAL_MEMBER_MANIFEST: Final = (
    ArchiveMember("KuaiRand-Pure/", MemberType.DIRECTORY, 0),
    ArchiveMember(
        "KuaiRand-Pure/LICENSE",
        MemberType.REGULAR,
        20_138,
        "187442db4df3afd21f2f0525739fd4beac28a62daaba3ee8d3533f60e7c33ec7",
    ),
    ArchiveMember("KuaiRand-Pure/data/", MemberType.DIRECTORY, 0),
    ArchiveMember(
        "KuaiRand-Pure/load_data_pure.py",
        MemberType.REGULAR,
        1_608,
        "19b6117c9c82a6480af72603e66579f1e0e824e16ce826eb9e6ac98fbf1ce6af",
    ),
    ArchiveMember(
        "KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv",
        MemberType.REGULAR,
        83_961_282,
        "5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c",
        _LOG_HEADER,
    ),
    ArchiveMember(
        "KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv",
        MemberType.REGULAR,
        87_086_116,
        "60b80994da969cd53da4d50c37ba3dafd6fb185df804c92c8410df34845a9d2c",
        _LOG_HEADER,
    ),
    ArchiveMember(
        "KuaiRand-Pure/data/user_features_pure.csv",
        MemberType.REGULAR,
        3_519_028,
        "dc729a656301b4c6d07f713fe41d05ec9bfaab670b90e531c70037caf033c011",
        _USER_HEADER,
    ),
    ArchiveMember(
        "KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv",
        MemberType.REGULAR,
        21_765_075,
        "429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd",
        _LOG_HEADER,
    ),
    ArchiveMember(
        "KuaiRand-Pure/data/video_features_statistic_pure.csv",
        MemberType.REGULAR,
        6_559_217,
        "d5c9e237ef2c6c1fc0e7f27e952f215d6626ecd934b01a6c53ecfcc72540f6b6",
        _STATISTICS_HEADER,
    ),
    ArchiveMember(
        "KuaiRand-Pure/data/video_features_basic_pure.csv",
        MemberType.REGULAR,
        626_669,
        "a6f7ee02684c5777422306cdc416e170302288aa89aca9dfea995edbd625bcc2",
        _BASIC_HEADER,
    ),
)

OFFICIAL_ARCHIVE_SPEC: Final = ArchiveSpec(
    source=ARCHIVE_SOURCE,
    filename=ARCHIVE_FILENAME,
    size=ARCHIVE_SIZE_BYTES,
    md5=DATASET_ARCHIVE_MD5,
    sha256=DATASET_ARCHIVE_SHA256,
    tar_size=ARCHIVE_TAR_SIZE_BYTES,
    members=OFFICIAL_MEMBER_MANIFEST,
).validate()
