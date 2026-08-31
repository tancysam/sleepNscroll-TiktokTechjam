"""Safe invocation of the immutable organizer submission checker.

The organizer ``submit.py`` checker is authoritative for CSV structure and row alignment, but
its loader also converts ``long_view`` for every split.  Passing the raw late-period log to that
loader would therefore cross the final-outcome boundary even when ``--check`` is selected.

This module constructs a private, short-lived data view first.  It scans standard logs as binary
records, decodes only the safe date cell, and replaces every registered outcome cell in final-date
rows with the ASCII token ``0``.  Outcome tokens are never sliced, decoded, converted, validated,
logged, or hashed.  The untouched, hash-pinned checker is then invoked with ``--check`` only.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from kuairand_agent.contract import OrganizerIntegrityError, verify_starter_kit
from kuairand_agent.data.canonical import LOG_HEADER, OUTCOME_FIELDS

ORGANIZER_CHECK_SCHEMA_VERSION: Final = 1
STANDARD_LOG_FILENAMES: Final = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
VIDEO_BASIC_FILENAME: Final = "video_features_basic_pure.csv"
REQUIRED_DATA_FILENAMES: Final = (*STANDARD_LOG_FILENAMES, VIDEO_BASIC_FILENAME)
FINAL_START_DATE: Final = 20_220_429
FINAL_END_DATE: Final = 20_220_508
_DATE_INDEX: Final = LOG_HEADER.index("date")
_OUTCOME_INDICES: Final = frozenset(LOG_HEADER.index(name) for name in OUTCOME_FIELDS)
_LOG_HEADER_BYTES: Final = ",".join(LOG_HEADER).encode("ascii")
_MAX_STANDARD_ROW_BYTES: Final = 1024 * 1024
_COPY_CHUNK_BYTES: Final = 1024 * 1024
_MAX_CHECKER_OUTPUT_BYTES: Final = 64 * 1024
_MAX_CHECKER_TIMEOUT_SECONDS: Final = 3600
_STABLE_COMMAND: Final = (
    "python",
    "-B",
    "submit.py",
    "submission.csv",
    "--data_dir",
    "<private-masked-data-view>",
    "--split",
    "test",
    "--check",
)


class OrganizerCheckError(RuntimeError):
    """Raised when safe view construction or the immutable checker fails closed."""


def _sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OrganizerCheckError(f"{location} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MaskedFileEvidence:
    """Content identity for one file inside the private masked data view."""

    relative_path: str
    sha256: str
    size_bytes: int
    data_rows: int | None
    final_rows_masked: int | None

    def manifest(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "data_rows": self.data_rows,
            "final_rows_masked": self.final_rows_masked,
        }


@dataclass(frozen=True, slots=True)
class MaskedViewEvidence:
    """Value-free proof that the organizer saw no final-period outcome token."""

    files: tuple[MaskedFileEvidence, ...]
    final_rows_masked: int
    final_outcome_cells_replaced: int
    digest: str

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": ORGANIZER_CHECK_SCHEMA_VERSION,
            "files": [entry.manifest() for entry in self.files],
            "final_outcome_isolation": {
                "registered_fields": list(OUTCOME_FIELDS),
                "final_rows_masked": self.final_rows_masked,
                "final_outcome_cells_replaced": self.final_outcome_cells_replaced,
                "outcome_cells_sliced": 0,
                "outcome_cells_decoded": 0,
                "outcome_cells_converted": 0,
                "outcome_cells_validated": 0,
                "outcome_cells_logged": 0,
                "outcome_cells_hashed": 0,
                "outcome_cells_scored": 0,
            },
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class OrganizerCheckEvidence:
    """Stable, bounded evidence from one successful official structural check."""

    starter_manifest_sha256: str
    submission_sha256: str
    submission_size_bytes: int
    masked_view: MaskedViewEvidence
    checker_command: tuple[str, ...]
    checker_returncode: int
    checker_stdout: str
    checker_stderr: str
    checker_stdout_sha256: str
    checker_stderr_sha256: str

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": ORGANIZER_CHECK_SCHEMA_VERSION,
            "checker": "hash-pinned organizer submit.py",
            "mode": "check_only",
            "split": "test",
            "starter_manifest_sha256": self.starter_manifest_sha256,
            "submission": {
                "sha256": self.submission_sha256,
                "size_bytes": self.submission_size_bytes,
            },
            "masked_data_view": self.masked_view.manifest(),
            "command": list(self.checker_command),
            "returncode": self.checker_returncode,
            "stdout": self.checker_stdout,
            "stderr": self.checker_stderr,
            "stdout_sha256": self.checker_stdout_sha256,
            "stderr_sha256": self.checker_stderr_sha256,
        }


@dataclass(frozen=True, slots=True)
class _CopiedFile:
    sha256: str
    size_bytes: int


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _open_stable_regular(path: Path) -> tuple[int, os.stat_result]:
    """Open one non-symlink file and bind the descriptor to its lstat identity."""

    try:
        initial = path.lstat()
    except OSError as exc:
        raise OrganizerCheckError(f"required input is unavailable: {path.name}") from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise OrganizerCheckError(f"required input must be a regular non-symlink file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OrganizerCheckError(
            f"required input could not be opened safely: {path.name}"
        ) from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
        initial.st_dev,
        initial.st_ino,
    ):
        os.close(descriptor)
        raise OrganizerCheckError(f"required input changed before opening: {path.name}")
    return descriptor, opened


def _require_unchanged(descriptor: int, opened: os.stat_result, name: str) -> None:
    final = os.fstat(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened, field) != getattr(final, field) for field in fields):
        raise OrganizerCheckError(f"required input changed while being read: {name}")


def _copy_regular_file(source: Path, destination: Path) -> _CopiedFile:
    descriptor, opened = _open_stable_regular(source)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        try:
            target = destination.open("xb")
        except OSError as exc:
            raise OrganizerCheckError(
                f"private view destination could not be created: {destination.name}"
            ) from exc
        with target:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while chunk := handle.read(_COPY_CHUNK_BYTES):
                    target.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        _require_unchanged(descriptor, opened, source.name)
    finally:
        os.close(descriptor)
    return _CopiedFile(digest.hexdigest(), size_bytes)


def _record_without_ending(line: bytes) -> tuple[bytes, bytes]:
    if line.endswith(b"\n"):
        if line.endswith(b"\r\n"):
            return line[:-2], b"\r\n"
        return line[:-1], b"\n"
    return line, b""


def _field_boundaries(record: bytes, source_name: str, line_number: int) -> tuple[int, ...]:
    """Return comma offsets without materializing any field payload."""

    boundaries = tuple(index for index, byte in enumerate(record) if byte == 0x2C)
    if len(boundaries) != len(LOG_HEADER) - 1:
        raise OrganizerCheckError(
            f"{source_name} line {line_number} has the wrong number of CSV fields"
        )
    return boundaries


def _field_span(boundaries: tuple[int, ...], field_index: int, record_size: int) -> tuple[int, int]:
    start = 0 if field_index == 0 else boundaries[field_index - 1] + 1
    end = record_size if field_index == len(boundaries) else boundaries[field_index]
    return start, end


def _safe_date(record: bytes, boundaries: tuple[int, ...], source: str, line: int) -> int:
    start, end = _field_span(boundaries, _DATE_INDEX, len(record))
    raw = record[start:end]
    try:
        rendered = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OrganizerCheckError(f"{source} line {line} has a non-ASCII safe date") from exc
    if len(rendered) != 8 or not rendered.isdigit():
        raise OrganizerCheckError(f"{source} line {line} has an invalid safe date")
    return int(rendered)


def _write_masked_record(
    target: object,
    digest: object,
    record: bytes,
    ending: bytes,
    boundaries: tuple[int, ...],
) -> int:
    """Write one final row without ever slicing an outcome cell."""

    written = 0
    for field_index in range(len(LOG_HEADER)):
        if field_index in _OUTCOME_INDICES:
            payload = b"0"
        else:
            start, end = _field_span(boundaries, field_index, len(record))
            payload = record[start:end]
        target.write(payload)  # type: ignore[attr-defined]
        digest.update(payload)  # type: ignore[attr-defined]
        written += len(payload)
        if field_index != len(LOG_HEADER) - 1:
            target.write(b",")  # type: ignore[attr-defined]
            digest.update(b",")  # type: ignore[attr-defined]
            written += 1
    target.write(ending)  # type: ignore[attr-defined]
    digest.update(ending)  # type: ignore[attr-defined]
    return written + len(ending)


def _mask_standard_log(source: Path, destination: Path) -> MaskedFileEvidence:
    descriptor, opened = _open_stable_regular(source)
    digest = hashlib.sha256()
    size_bytes = 0
    data_rows = 0
    final_rows = 0
    try:
        try:
            target = destination.open("xb")
        except OSError as exc:
            raise OrganizerCheckError(
                f"private view destination could not be created: {destination.name}"
            ) from exc
        with target:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                header = handle.readline(_MAX_STANDARD_ROW_BYTES + 1)
                if len(header) > _MAX_STANDARD_ROW_BYTES:
                    raise OrganizerCheckError(f"{source.name} header exceeds the row-size bound")
                header_record, _ = _record_without_ending(header)
                if header_record != _LOG_HEADER_BYTES:
                    raise OrganizerCheckError(f"{source.name} has an unexpected header")
                target.write(header)
                digest.update(header)
                size_bytes += len(header)

                for line_number, line in enumerate(handle, start=2):
                    if len(line) > _MAX_STANDARD_ROW_BYTES:
                        raise OrganizerCheckError(
                            f"{source.name} line {line_number} exceeds the row-size bound"
                        )
                    record, ending = _record_without_ending(line)
                    boundaries = _field_boundaries(record, source.name, line_number)
                    date = _safe_date(record, boundaries, source.name, line_number)
                    if FINAL_START_DATE <= date <= FINAL_END_DATE:
                        size_bytes += _write_masked_record(
                            target, digest, record, ending, boundaries
                        )
                        final_rows += 1
                    else:
                        target.write(line)
                        digest.update(line)
                        size_bytes += len(line)
                    data_rows += 1
            target.flush()
            os.fsync(target.fileno())
        _require_unchanged(descriptor, opened, source.name)
    finally:
        os.close(descriptor)
    return MaskedFileEvidence(
        relative_path=source.name,
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
        data_rows=data_rows,
        final_rows_masked=final_rows,
    )


def _build_masked_view(source_data: Path, destination_data: Path) -> MaskedViewEvidence:
    try:
        source_metadata = source_data.lstat()
    except OSError as exc:
        raise OrganizerCheckError("dataset data directory is unavailable") from exc
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(source_metadata.st_mode):
        raise OrganizerCheckError("dataset data directory must be a real directory")

    destination_data.mkdir(mode=0o700)
    files: list[MaskedFileEvidence] = []
    for filename in STANDARD_LOG_FILENAMES:
        files.append(_mask_standard_log(source_data / filename, destination_data / filename))
    copied = _copy_regular_file(
        source_data / VIDEO_BASIC_FILENAME,
        destination_data / VIDEO_BASIC_FILENAME,
    )
    files.append(
        MaskedFileEvidence(
            relative_path=VIDEO_BASIC_FILENAME,
            sha256=copied.sha256,
            size_bytes=copied.size_bytes,
            data_rows=None,
            final_rows_masked=None,
        )
    )
    ordered = tuple(sorted(files, key=lambda entry: entry.relative_path))
    final_rows = sum(entry.final_rows_masked or 0 for entry in ordered)
    stable = {
        "schema_version": ORGANIZER_CHECK_SCHEMA_VERSION,
        "files": [entry.manifest() for entry in ordered],
        "registered_outcome_fields": list(OUTCOME_FIELDS),
        "final_rows_masked": final_rows,
        "final_outcome_cells_replaced": final_rows * len(OUTCOME_FIELDS),
    }
    return MaskedViewEvidence(
        files=ordered,
        final_rows_masked=final_rows,
        final_outcome_cells_replaced=final_rows * len(OUTCOME_FIELDS),
        digest=hashlib.sha256(_canonical_json(stable)).hexdigest(),
    )


def _verify_starter(starter_dir: Path, phase: str) -> str:
    try:
        return verify_starter_kit(starter_dir).manifest_sha256
    except OrganizerIntegrityError as exc:
        raise OrganizerCheckError(
            f"organizer starter integrity failed {phase} checker execution"
        ) from exc


def validate_organizer_check_manifest(
    value: object,
    *,
    expected_submission_sha256: str,
    expected_submission_size_bytes: int | None,
    expected_starter_manifest_sha256: str,
    expected_final_rows: int,
) -> dict[str, object]:
    """Authenticate retained evidence from the hash-pinned structural checker.

    This validates the value-free masking proof as well as the checker outcome.  It is shared by
    the state authority and public bundle validator so a digest-shaped organizer claim cannot be
    accepted without the exact retained manifest behind it.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "checker",
        "mode",
        "split",
        "starter_manifest_sha256",
        "submission",
        "masked_data_view",
        "command",
        "returncode",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
    }:
        raise OrganizerCheckError("organizer check evidence has an unexpected schema")
    if (
        value.get("schema_version") != ORGANIZER_CHECK_SCHEMA_VERSION
        or value.get("checker") != "hash-pinned organizer submit.py"
        or value.get("mode") != "check_only"
        or value.get("split") != "test"
        or value.get("returncode") != 0
        or value.get("command") != list(_STABLE_COMMAND)
    ):
        raise OrganizerCheckError("organizer check evidence does not prove check-only success")
    if type(expected_final_rows) is not int or expected_final_rows <= 0:
        raise OrganizerCheckError("expected_final_rows must be positive")
    starter = _sha256(value.get("starter_manifest_sha256"), "starter_manifest_sha256")
    if starter != _sha256(expected_starter_manifest_sha256, "expected_starter_manifest_sha256"):
        raise OrganizerCheckError("organizer checker used a different starter manifest")
    submission = value.get("submission")
    if not isinstance(submission, Mapping) or set(submission) != {"sha256", "size_bytes"}:
        raise OrganizerCheckError("organizer submission evidence has an unexpected schema")
    if _sha256(submission.get("sha256"), "submission.sha256") != _sha256(
        expected_submission_sha256, "expected_submission_sha256"
    ):
        raise OrganizerCheckError("organizer checker observed different submission bytes")
    submission_size = submission.get("size_bytes")
    if type(submission_size) is not int or submission_size <= 0:
        raise OrganizerCheckError("organizer submission size is invalid")
    if expected_submission_size_bytes is not None and (
        type(expected_submission_size_bytes) is not int
        or expected_submission_size_bytes <= 0
        or submission_size != expected_submission_size_bytes
    ):
        raise OrganizerCheckError("organizer submission size differs from retained bytes")
    for stream in ("stdout", "stderr"):
        observed = value.get(stream)
        if type(observed) is not str:
            raise OrganizerCheckError(f"organizer {stream} must be text")
        expected_digest = hashlib.sha256(observed.encode("utf-8")).hexdigest()
        if _sha256(value.get(f"{stream}_sha256"), f"{stream}_sha256") != expected_digest:
            raise OrganizerCheckError(f"organizer {stream} digest is invalid")

    masked = value.get("masked_data_view")
    if not isinstance(masked, Mapping) or set(masked) != {
        "schema_version",
        "files",
        "final_outcome_isolation",
        "digest",
    }:
        raise OrganizerCheckError("masked data-view evidence has an unexpected schema")
    if masked.get("schema_version") != ORGANIZER_CHECK_SCHEMA_VERSION:
        raise OrganizerCheckError("masked data-view schema_version is unsupported")
    files = masked.get("files")
    if not isinstance(files, list) or len(files) != len(REQUIRED_DATA_FILENAMES):
        raise OrganizerCheckError("masked data-view file inventory is incomplete")
    normalized_files: list[dict[str, object]] = []
    counted_final_rows = 0
    for index, candidate in enumerate(files):
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "relative_path",
            "sha256",
            "size_bytes",
            "data_rows",
            "final_rows_masked",
        }:
            raise OrganizerCheckError(f"masked file evidence {index} has an unexpected schema")
        relative_path = candidate.get("relative_path")
        if type(relative_path) is not str:
            raise OrganizerCheckError("masked file relative_path must be text")
        size_bytes = candidate.get("size_bytes")
        if type(size_bytes) is not int or size_bytes <= 0:
            raise OrganizerCheckError("masked file size is invalid")
        _sha256(candidate.get("sha256"), f"masked files[{index}].sha256")
        data_rows = candidate.get("data_rows")
        final_rows = candidate.get("final_rows_masked")
        if relative_path == VIDEO_BASIC_FILENAME:
            if data_rows is not None or final_rows is not None:
                raise OrganizerCheckError("video feature masking counters must be absent")
        else:
            if (
                type(data_rows) is not int
                or data_rows <= 0
                or type(final_rows) is not int
                or final_rows < 0
                or final_rows > data_rows
            ):
                raise OrganizerCheckError("standard-log masking counters are invalid")
            counted_final_rows += final_rows
        normalized_files.append(dict(candidate))
    if [entry["relative_path"] for entry in normalized_files] != sorted(REQUIRED_DATA_FILENAMES):
        raise OrganizerCheckError("masked data-view file inventory differs from the contract")

    isolation = masked.get("final_outcome_isolation")
    counter_names = (
        "outcome_cells_sliced",
        "outcome_cells_decoded",
        "outcome_cells_converted",
        "outcome_cells_validated",
        "outcome_cells_logged",
        "outcome_cells_hashed",
        "outcome_cells_scored",
    )
    expected_cells = expected_final_rows * len(OUTCOME_FIELDS)
    if not isinstance(isolation, Mapping) or set(isolation) != {
        "registered_fields",
        "final_rows_masked",
        "final_outcome_cells_replaced",
        *counter_names,
    }:
        raise OrganizerCheckError("final outcome-isolation evidence has an unexpected schema")
    if (
        isolation.get("registered_fields") != list(OUTCOME_FIELDS)
        or isolation.get("final_rows_masked") != expected_final_rows
        or isolation.get("final_outcome_cells_replaced") != expected_cells
        or counted_final_rows != expected_final_rows
        or any(isolation.get(name) != 0 for name in counter_names)
    ):
        raise OrganizerCheckError("final outcome-isolation evidence is unsafe")
    digest_body = {
        "schema_version": ORGANIZER_CHECK_SCHEMA_VERSION,
        "files": normalized_files,
        "registered_outcome_fields": list(OUTCOME_FIELDS),
        "final_rows_masked": expected_final_rows,
        "final_outcome_cells_replaced": expected_cells,
    }
    if (
        _sha256(masked.get("digest"), "masked_data_view.digest")
        != hashlib.sha256(_canonical_json(digest_body)).hexdigest()
    ):
        raise OrganizerCheckError("masked data-view digest is invalid")
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OrganizerCheckError("organizer evidence is not finite canonical JSON") from exc
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping input guarantees object.
        raise OrganizerCheckError("organizer evidence must normalize to an object")
    return normalized


def _decode_bounded_output(payload: bytes, stream: str) -> str:
    if len(payload) > _MAX_CHECKER_OUTPUT_BYTES:
        raise OrganizerCheckError(
            f"organizer checker {stream} exceeded the {_MAX_CHECKER_OUTPUT_BYTES}-byte bound"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OrganizerCheckError(f"organizer checker {stream} was not UTF-8") from exc


def check_final_submission(
    submission_path: str | Path,
    *,
    data_dir: str | Path,
    starter_dir: str | Path,
    scratch_dir: str | Path | None = None,
    timeout_seconds: int = 300,
) -> OrganizerCheckEvidence:
    """Run the untouched organizer ``--check --split test`` through a masked data view.

    The returned object contains no physical temporary path and no raw-data digest.  It is stable
    under arbitrary changes to well-framed final-period outcome tokens because those tokens are
    replaced before any retained hash or text decoding occurs.
    """

    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= _MAX_CHECKER_TIMEOUT_SECONDS:
        raise OrganizerCheckError(
            f"timeout_seconds must be an integer in [1, {_MAX_CHECKER_TIMEOUT_SECONDS}]"
        )
    submission = Path(submission_path)
    source_data = Path(data_dir)
    # The checker runs from its private temporary root, so bind the verified starter path to the
    # caller's current directory before changing the child process working directory.
    starter = Path(starter_dir).absolute()
    scratch = None if scratch_dir is None else Path(scratch_dir)
    if scratch is not None:
        try:
            scratch_metadata = scratch.lstat()
        except OSError as exc:
            raise OrganizerCheckError("scratch directory is unavailable") from exc
        if stat.S_ISLNK(scratch_metadata.st_mode) or not stat.S_ISDIR(scratch_metadata.st_mode):
            raise OrganizerCheckError("scratch directory must be a real directory")

    starter_before = _verify_starter(starter, "before")
    result: subprocess.CompletedProcess[bytes] | None = None
    masked_view: MaskedViewEvidence | None = None
    snapshot = _CopiedFile("", 0)
    try:
        with tempfile.TemporaryDirectory(prefix="kuairand-final-check-", dir=scratch) as raw_root:
            private_root = Path(raw_root)
            os.chmod(private_root, 0o700, follow_symlinks=False)
            private_data = private_root / "data"
            masked_view = _build_masked_view(source_data, private_data)
            private_submission = private_root / "submission.csv"
            snapshot = _copy_regular_file(submission, private_submission)

            command = (
                sys.executable,
                "-B",
                str(starter / "submit.py"),
                str(private_submission),
                "--data_dir",
                str(private_data),
                "--split",
                "test",
                "--check",
            )
            environment = MappingProxyType(
                {
                    "LC_ALL": "C",
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "PYTHONHASHSEED": "0",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONUTF8": "1",
                }
            )
            try:
                result = subprocess.run(
                    command,
                    cwd=private_root,
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise OrganizerCheckError(
                    f"organizer checker exceeded the {timeout_seconds}-second timeout"
                ) from exc
    finally:
        starter_after = _verify_starter(starter, "after")
        if starter_after != starter_before:
            raise OrganizerCheckError("organizer starter manifest changed during checker execution")

    if result is None or masked_view is None:
        raise OrganizerCheckError("organizer checker did not produce a result")
    stdout = _decode_bounded_output(result.stdout, "stdout")
    stderr = _decode_bounded_output(result.stderr, "stderr")
    if result.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "no bounded diagnostic output"
        raise OrganizerCheckError(
            "organizer checker rejected the submission with exit code "
            f"{result.returncode}: {detail}"
        )
    return OrganizerCheckEvidence(
        starter_manifest_sha256=starter_before,
        submission_sha256=snapshot.sha256,
        submission_size_bytes=snapshot.size_bytes,
        masked_view=masked_view,
        checker_command=_STABLE_COMMAND,
        checker_returncode=result.returncode,
        checker_stdout=stdout,
        checker_stderr=stderr,
        checker_stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
        checker_stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
    )


__all__ = [
    "FINAL_END_DATE",
    "FINAL_START_DATE",
    "ORGANIZER_CHECK_SCHEMA_VERSION",
    "MaskedFileEvidence",
    "MaskedViewEvidence",
    "OrganizerCheckError",
    "OrganizerCheckEvidence",
    "check_final_submission",
    "validate_organizer_check_manifest",
]
