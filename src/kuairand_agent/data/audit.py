"""Deterministic, leakage-safe audit of the verified KuaiRand-Pure dataset.

The audit has two separate responsibilities:

* re-verify the exact extracted source members against their pinned byte identities; and
* stream high-signal semantic checks over fields the phase policy permits it to inspect.

The second responsibility deliberately does **not** use :mod:`csv` for standard-log data.
Instead, a binary selected-cell scanner counts delimiters but only slices and decodes requested
cells.  For final-period rows none of the outcome cells are requested, so their bytes are never
materialized, decoded, converted, validated, aggregated, logged, or scored.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from kuairand_agent.contract import (
    BENCHMARK_CONTRACT,
    DATASET_ARCHIVE_MD5,
    DATASET_ARCHIVE_SHA256,
    SplitName,
)
from kuairand_agent.data.fields import (
    CSV_HEADERS,
    FIELD_POLICY_DIGEST,
    RANDOMIZED_MEMBER,
    STANDARD_LATE_MEMBER,
    STANDARD_LOG_HEADER,
    STANDARD_TRAIN_MEMBER,
    USER_SNAPSHOT_MEMBER,
    VIDEO_BASIC_HEADER,
    VIDEO_BASIC_MEMBER,
    VIDEO_STATISTIC_MEMBER,
    field_policy_manifest,
)

AUDIT_SCHEMA_VERSION: Final = 1
DATASET_ARCHIVE_FILENAME: Final = "KuaiRand-Pure.tar.gz"
DATASET_ARCHIVE_SIZE: Final = 47_432_272
DATASET_SOURCE_ROOT: Final = "KuaiRand-Pure/data"
MAX_STANDARD_ROW_BYTES: Final = 1_048_576

_MEMBER_ORDER: Final = (
    STANDARD_TRAIN_MEMBER,
    RANDOMIZED_MEMBER,
    USER_SNAPSHOT_MEMBER,
    STANDARD_LATE_MEMBER,
    VIDEO_STATISTIC_MEMBER,
    VIDEO_BASIC_MEMBER,
)
_OUTCOME_FIELDS: Final = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
)
_BINARY_AUXILIARY_FIELDS: Final = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)
_NONNEGATIVE_AUXILIARY_FIELDS: Final = (
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
)
_SAFE_LOG_FIELDS: Final = (
    "user_id",
    "video_id",
    "date",
    "time_ms",
    "duration_ms",
    "tab",
)
_TRAIN_SELECTED_FIELDS: Final = (
    *_SAFE_LOG_FIELDS,
    "long_view",
    *_BINARY_AUXILIARY_FIELDS,
    *_NONNEGATIVE_AUXILIARY_FIELDS,
)
_VALID_SELECTED_FIELDS: Final = (*_SAFE_LOG_FIELDS, "long_view")
_FINAL_SELECTED_FIELDS: Final = _SAFE_LOG_FIELDS
_CHECKS_PERFORMED: Final = (
    {
        "scope": "all extracted CSV members",
        "checks": [
            "exact member set",
            "regular non-symlink files",
            "size",
            "SHA-256",
            "header",
            "row count",
        ],
    },
    {
        "scope": "standard logs in all splits",
        "checks": [
            "non-negative canonical int64 user_id/video_id/time_ms",
            "organizer date interval and physical-member placement",
            "finite non-negative duration_ms",
            "tab in [0, 14]",
            "binary-safe author lookup without row expansion",
        ],
    },
    {
        "scope": "training outcomes",
        "checks": [
            "binary long_view and approved binary auxiliary targets",
            "finite non-negative approved numeric auxiliary targets",
            "class balance and user support",
            "aggregate-only auxiliary association planning evidence",
        ],
    },
    {
        "scope": "public validation outcomes",
        "checks": [
            "binary long_view",
            "class balance and user support",
            "no auxiliary outcomes selected",
        ],
    },
    {
        "scope": "final-period outcomes",
        "checks": [
            "binary scanner selects no outcome cells",
            "zero outcome interpretation counters",
        ],
    },
)
_LOG_INDEX: Final = MappingProxyType(
    {name: index for index, name in enumerate(STANDARD_LOG_HEADER)}
)
_VIDEO_INDEX: Final = MappingProxyType(
    {name: index for index, name in enumerate(VIDEO_BASIC_HEADER)}
)
_SPLIT_INTERVALS: Final = MappingProxyType(
    {split.name: (split.start_date, split.end_date) for split in BENCHMARK_CONTRACT.splits}
)


class DataAuditError(ValueError):
    """Raised when source identity, schema, or permitted semantic data is invalid."""


@dataclass(frozen=True, slots=True)
class ExpectedSourceIdentity:
    """Pinned byte and shape identity for one extracted CSV member."""

    member: str
    size: int
    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if self.member not in CSV_HEADERS:
            raise DataAuditError(f"unregistered expected source member {self.member!r}")
        if self.size < 0 or self.row_count < 0:
            raise DataAuditError("expected source size and row_count must be non-negative")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise DataAuditError("expected source SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class ExpectedSplitIdentity:
    """Optional exact logical split facts pinned after the source identity was verified."""

    split: SplitName
    row_count: int
    first_date: int
    last_date: int

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise DataAuditError("expected split row_count must be positive")
        if self.first_date > self.last_date:
            raise DataAuditError("expected split date interval is reversed")


@dataclass(frozen=True, slots=True)
class AuditContract:
    """Exact data identities required before an audit result is eligible."""

    sources: tuple[ExpectedSourceIdentity, ...]
    expected_splits: tuple[ExpectedSplitIdentity, ...] = ()

    def __post_init__(self) -> None:
        members = tuple(source.member for source in self.sources)
        if members != _MEMBER_ORDER:
            raise DataAuditError(
                "audit contract sources must contain every registered CSV in archive order"
            )
        split_names = tuple(item.split for item in self.expected_splits)
        if len(split_names) != len(set(split_names)):
            raise DataAuditError("audit contract split identities must be unique")

    @property
    def source_by_member(self) -> Mapping[str, ExpectedSourceIdentity]:
        return MappingProxyType({source.member: source for source in self.sources})

    @property
    def split_by_name(self) -> Mapping[SplitName, ExpectedSplitIdentity]:
        return MappingProxyType({item.split: item for item in self.expected_splits})


OFFICIAL_SOURCE_IDENTITIES: Final = (
    ExpectedSourceIdentity(
        STANDARD_TRAIN_MEMBER,
        83_961_282,
        "5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c",
        1_141_112,
    ),
    ExpectedSourceIdentity(
        RANDOMIZED_MEMBER,
        87_086_116,
        "60b80994da969cd53da4d50c37ba3dafd6fb185df804c92c8410df34845a9d2c",
        1_186_059,
    ),
    ExpectedSourceIdentity(
        USER_SNAPSHOT_MEMBER,
        3_519_028,
        "dc729a656301b4c6d07f713fe41d05ec9bfaab670b90e531c70037caf033c011",
        27_285,
    ),
    ExpectedSourceIdentity(
        STANDARD_LATE_MEMBER,
        21_765_075,
        "429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd",
        295_497,
    ),
    ExpectedSourceIdentity(
        VIDEO_STATISTIC_MEMBER,
        6_559_217,
        "d5c9e237ef2c6c1fc0e7f27e952f215d6626ecd934b01a6c53ecfcc72540f6b6",
        7_583,
    ),
    ExpectedSourceIdentity(
        VIDEO_BASIC_MEMBER,
        626_669,
        "a6f7ee02684c5777422306cdc416e170302288aa89aca9dfea995edbd625bcc2",
        7_583,
    ),
)

# These logical identities were derived by the binary selected-cell scanner after every extracted
# member matched the pinned official byte manifest.  The source filename starts on April 8, while
# the first retained physical row is April 9; the audit records data facts rather than inferring a
# row that is not present.
OFFICIAL_SPLIT_IDENTITIES: Final = (
    ExpectedSplitIdentity(SplitName.TRAIN, 1_141_112, 20220409, 20220421),
    ExpectedSplitIdentity(SplitName.VALID, 124_909, 20220422, 20220428),
    ExpectedSplitIdentity(SplitName.TEST, 170_588, 20220429, 20220508),
)
OFFICIAL_AUDIT_CONTRACT: Final = AuditContract(
    sources=OFFICIAL_SOURCE_IDENTITIES,
    expected_splits=OFFICIAL_SPLIT_IDENTITIES,
)


@dataclass(frozen=True, slots=True)
class SourceAudit:
    """Verified source-qualified facts for one CSV member."""

    member: str
    size: int
    sha256: str
    row_count: int
    header: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "member": self.member,
            "size": self.size,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "header": list(self.header),
            "identity_verified": True,
        }


@dataclass(frozen=True, slots=True)
class FinalOutcomeSkipTrace:
    """Value-free instrumentation proving that final outcome cells stayed opaque."""

    member: str
    split: SplitName
    row_count: int
    skipped_fields: tuple[str, ...]
    selected_fields: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "member": self.member,
            "split": self.split.value,
            "row_count": self.row_count,
            "scanner": "binary-selected-cell-v1",
            "selected_fields": list(self.selected_fields),
            "skipped_fields": list(self.skipped_fields),
            "skipped_cell_count": self.row_count * len(self.skipped_fields),
            "outcome_cells_materialized": 0,
            "outcome_cells_decoded": 0,
            "outcome_cells_converted": 0,
            "outcome_cells_validated": 0,
            "outcome_cells_aggregated": 0,
            "outcome_cells_logged": 0,
            "outcome_cells_scored": 0,
            "skipped_values_recorded": False,
        }


@dataclass(frozen=True, slots=True)
class DataAuditReport:
    """Complete deterministic audit evidence with separate semantic/source digests."""

    data_dir: Path = field(repr=False)
    sources: tuple[SourceAudit, ...]
    splits: tuple[Mapping[str, object], ...]
    train_associations: Mapping[str, object]
    final_outcome_trace: FinalOutcomeSkipTrace
    field_policy: Mapping[str, object]

    def _semantic_manifest(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "checks_performed": list(_CHECKS_PERFORMED),
            "source_shapes": [
                {
                    "member": source.member,
                    "row_count": source.row_count,
                    "header": list(source.header),
                }
                for source in self.sources
            ],
            "splits": [dict(split) for split in self.splits],
            "train_associations": dict(self.train_associations),
            "final_outcome_trace": self.final_outcome_trace.manifest(),
            "field_policy_digest": FIELD_POLICY_DIGEST,
        }

    @property
    def semantic_digest(self) -> str:
        """Digest logical audit facts without source bytes or filesystem location.

        This digest is intentionally unchanged when only skipped final-outcome bytes change in a
        controlled metamorphic fixture.  The full report digest still catches that byte change.
        """

        return _json_digest(self._semantic_manifest())

    def _base_manifest(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "dataset": "KuaiRand-Pure",
            "source_root": DATASET_SOURCE_ROOT,
            "archive_identity": {
                "filename": DATASET_ARCHIVE_FILENAME,
                "size": DATASET_ARCHIVE_SIZE,
                "md5": DATASET_ARCHIVE_MD5,
                "sha256": DATASET_ARCHIVE_SHA256,
                "archive_verification_owner": "trusted acquisition gate",
                "extracted_member_bytes_reverified": True,
            },
            "checks_performed": list(_CHECKS_PERFORMED),
            "sources": [source.manifest() for source in self.sources],
            "splits": [dict(split) for split in self.splits],
            "train_associations": dict(self.train_associations),
            "final_outcome_trace": self.final_outcome_trace.manifest(),
            "field_policy": dict(self.field_policy),
            "field_policy_digest": FIELD_POLICY_DIGEST,
            "semantic_digest": self.semantic_digest,
        }

    @property
    def digest(self) -> str:
        """SHA-256 of every report fact except its own digest and local path."""

        return _json_digest(self._base_manifest())

    def manifest(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible report without absolute local paths."""

        manifest = self._base_manifest()
        manifest["digest"] = self.digest
        return manifest

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the evidence in stable key order."""

        return json.dumps(
            self.manifest(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            ensure_ascii=True,
        )

    def readable_report(self) -> str:
        """Render a judge-readable Markdown audit with the full field-policy table."""

        lines = [
            "# KuaiRand-Pure leakage-safe data audit",
            "",
            f"Report digest: `{self.digest}`",
            f"Semantic digest: `{self.semantic_digest}`",
            "",
            "## Verified source members",
            "",
            "| Member | Bytes | Rows | SHA-256 |",
            "| --- | ---: | ---: | --- |",
        ]
        for source in self.sources:
            lines.append(
                f"| `{source.member}` | {source.size} | {source.row_count} | `{source.sha256}` |"
            )
        lines.extend(
            [
                "",
                "## Checks performed",
                "",
            ]
        )
        for check in _CHECKS_PERFORMED:
            checks = cast(list[str], check["checks"])
            lines.append(f"- {check['scope']}: {', '.join(checks)}")
        lines.extend(
            [
                "",
                "## Logical split profile",
                "",
                "| Split | Rows | Date range | Users | Videos | Authors | Target rate |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for split in self.splits:
            target = split.get("primary_target")
            target_rate = "withheld"
            if isinstance(target, Mapping):
                target_rate = str(target["positive_rate"])
            lines.append(
                f"| {split['split']} | {split['row_count']} | "
                f"{split['first_date']}..{split['last_date']} | "
                f"{split['user_cardinality']} | {split['video_cardinality']} | "
                f"{split['author_cardinality']} | {target_rate} |"
            )
        trace = self.final_outcome_trace.manifest()
        lines.extend(
            [
                "",
                "## Final-period outcome isolation",
                "",
                "Final-period outcome values were byte-skipped by the binary selected-cell "
                "scanner. They were not materialized, decoded, converted, validated, "
                "aggregated, logged, or scored.",
                "",
                f"- Final rows: {trace['row_count']}",
                f"- Skipped outcome cells: {trace['skipped_cell_count']}",
                f"- Materialized outcome cells: {trace['outcome_cells_materialized']}",
                "",
                "## Source-qualified field policy",
                "",
                "| Member | Field | Role | Enabled | History eligible |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        fields = self.field_policy.get("fields")
        if not isinstance(fields, Sequence):  # pragma: no cover - registry owns this shape.
            raise DataAuditError("field policy manifest has no field table")
        for entry in fields:
            if not isinstance(entry, Mapping):  # pragma: no cover
                raise DataAuditError("field policy table contains an invalid row")
            lines.append(
                f"| `{entry['member']}` | `{entry['column']}` | {entry['role']} | "
                f"{entry['enabled']} | {entry['history_eligible']} |"
            )
        lines.extend(["", f"Field-policy digest: `{FIELD_POLICY_DIGEST}`", ""])
        return "\n".join(lines)


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _data_directory(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    try:
        requested_mode = requested.lstat().st_mode
    except FileNotFoundError as exc:
        raise DataAuditError(f"data path does not exist: {requested}") from exc
    if stat.S_ISLNK(requested_mode):
        raise DataAuditError("data audit root must not be a symbolic link")
    root = requested / "data" if (requested / "data").is_dir() else requested
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError as exc:  # pragma: no cover - guarded by requested path above.
        raise DataAuditError(f"data directory does not exist: {root}") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise DataAuditError("resolved audit data path must be a real directory")
    return root.resolve(strict=True)


def _physical_filename(member: str) -> str:
    prefix = "data/"
    if not member.startswith(prefix):  # pragma: no cover - frozen field registry invariant.
        raise DataAuditError(f"CSV member is not rooted under data/: {member}")
    return member.removeprefix(prefix)


def _strip_record_ending(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith(b"\n"):
        return line[:-1]
    return line


def _verify_filesystem_shape(root: Path) -> None:
    expected = {_physical_filename(member) for member in _MEMBER_ORDER}
    observed = {entry.name for entry in root.iterdir()}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise DataAuditError(
            f"data directory member set differs; missing={missing!r}, unexpected={unexpected!r}"
        )
    for name in sorted(expected):
        path = root / name
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise DataAuditError(f"expected data member must be a real regular file: {name}")


def _verify_source(root: Path, expected: ExpectedSourceIdentity) -> SourceAudit:
    path = root / _physical_filename(expected.member)
    digest = hashlib.sha256()
    physical_rows = -1
    header_bytes: bytes | None = None
    with path.open("rb", buffering=1024 * 1024) as handle:
        for physical_rows, line in enumerate(handle):
            digest.update(line)
            if physical_rows == 0:
                header_bytes = _strip_record_ending(line)
    row_count = max(physical_rows, 0)
    if header_bytes is None:
        raise DataAuditError(f"{expected.member} is empty and has no header")
    expected_header = ",".join(CSV_HEADERS[expected.member]).encode("ascii")
    if header_bytes != expected_header:
        raise DataAuditError(f"{expected.member} header differs from the exact field registry")
    observed_size = path.stat().st_size
    observed_sha = digest.hexdigest()
    if observed_size != expected.size:
        raise DataAuditError(
            f"{expected.member} size mismatch: expected {expected.size}, got {observed_size}"
        )
    if observed_sha != expected.sha256:
        raise DataAuditError(f"{expected.member} SHA-256 differs from the pinned identity")
    if row_count != expected.row_count:
        raise DataAuditError(
            f"{expected.member} row count mismatch: expected {expected.row_count}, got {row_count}"
        )
    return SourceAudit(
        member=expected.member,
        size=observed_size,
        sha256=observed_sha,
        row_count=row_count,
        header=CSV_HEADERS[expected.member],
    )


def _selected_cells(
    raw_line: bytes,
    selected: Mapping[int, str],
    *,
    expected_field_count: int,
    source: str,
) -> dict[str, bytes]:
    """Return only selected cell byte slices, never skipped cell contents."""

    if len(raw_line) > MAX_STANDARD_ROW_BYTES:
        raise DataAuditError(f"{source} exceeds the maximum standard-log row size")
    line = _strip_record_ending(raw_line)
    result: dict[str, bytes] = {}
    field_index = 0
    field_start = 0
    in_quotes = False
    offset = 0
    while offset < len(line):
        byte = line[offset]
        if byte == 0x22:  # double quote
            if in_quotes and offset + 1 < len(line) and line[offset + 1] == 0x22:
                offset += 2
                continue
            in_quotes = not in_quotes
            offset += 1
            continue
        if byte != 0x2C or in_quotes:  # comma outside a quoted CSV cell
            offset += 1
            continue
        name = selected.get(field_index)
        if name is not None:
            result[name] = _unquote_selected_cell(line[field_start:offset], source)
        field_index += 1
        field_start = offset + 1
        offset += 1
    if in_quotes:
        raise DataAuditError(f"{source} has an unterminated quoted CSV field")
    name = selected.get(field_index)
    if name is not None:
        result[name] = _unquote_selected_cell(line[field_start:], source)
    field_count = field_index + 1
    if field_count != expected_field_count:
        raise DataAuditError(f"{source} has the wrong number of CSV fields")
    if len(result) != len(selected):  # pragma: no cover - follows field-count validation.
        raise DataAuditError(f"{source} did not produce every selected field")
    return result


def _unquote_selected_cell(cell: bytes, source: str) -> bytes:
    """Unquote one selected cell; skipped cells are never passed to this function."""

    if not cell.startswith(b'"'):
        if b'"' in cell:
            raise DataAuditError(f"{source} has a malformed quoted CSV field")
        return cell
    if len(cell) < 2 or not cell.endswith(b'"'):
        raise DataAuditError(f"{source} has a malformed quoted CSV field")
    return cell[1:-1].replace(b'""', b'"')


def _parse_nonnegative_int(cell: bytes, field_name: str, source: str) -> int:
    if not cell or not cell.isdigit():
        raise DataAuditError(f"{source} has invalid integer field {field_name}")
    value = int(cell)
    if str(value).encode("ascii") != cell:
        raise DataAuditError(f"{source} has non-canonical integer field {field_name}")
    if value > 2**63 - 1:
        raise DataAuditError(f"{source} has int64 overflow in field {field_name}")
    return value


def _parse_binary(cell: bytes, field_name: str, source: str) -> int:
    if cell == b"0":
        return 0
    if cell == b"1":
        return 1
    raise DataAuditError(f"{source} has non-binary field {field_name}")


def _parse_nonnegative_float(cell: bytes, field_name: str, source: str) -> float:
    try:
        value = float(cell)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataAuditError(f"{source} has invalid numeric field {field_name}") from exc
    if not math.isfinite(value) or value < 0:
        raise DataAuditError(f"{source} has invalid non-negative field {field_name}")
    return value


def _split_for_date(date: int) -> SplitName:
    for split in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST):
        low, high = _SPLIT_INTERVALS[split]
        if low <= date <= high:
            return split
    raise DataAuditError("standard-log row has a date outside every organizer split")


def _nearest_rank(values: Sequence[int], numerator: int, denominator: int) -> int:
    if not values:
        return 0
    rank = max(1, (len(values) * numerator + denominator - 1) // denominator)
    return values[rank - 1]


@dataclass(slots=True)
class _CompensatedSum:
    total: float = 0.0
    correction: float = 0.0

    def add(self, value: float) -> None:
        adjusted = value - self.correction
        updated = self.total + adjusted
        self.correction = (updated - self.total) - adjusted
        self.total = updated


@dataclass(slots=True)
class _BinaryAssociation:
    counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def add(self, target: int, auxiliary: int) -> None:
        self.counts[target * 2 + auxiliary] += 1

    def manifest(self) -> dict[str, object]:
        y0a0, y0a1, y1a0, y1a1 = self.counts
        positive = y1a0 + y1a1
        negative = y0a0 + y0a1
        aux_positive = y0a1 + y1a1
        aux_negative = y0a0 + y1a0
        denominator = math.sqrt(positive * negative * aux_positive * aux_negative)
        phi = ((y1a1 * y0a0 - y1a0 * y0a1) / denominator) if denominator else None
        return {
            "joint_counts": {
                "target_0_aux_0": y0a0,
                "target_0_aux_1": y0a1,
                "target_1_aux_0": y1a0,
                "target_1_aux_1": y1a1,
            },
            "aux_positive_rate_given_target_0": y0a1 / negative if negative else None,
            "aux_positive_rate_given_target_1": y1a1 / positive if positive else None,
            "phi": phi,
        }


@dataclass(slots=True)
class _NumericAssociation:
    count_by_target: list[int] = field(default_factory=lambda: [0, 0])
    sum_by_target: tuple[_CompensatedSum, _CompensatedSum] = field(
        default_factory=lambda: (_CompensatedSum(), _CompensatedSum())
    )

    def add(self, target: int, value: float) -> None:
        self.count_by_target[target] += 1
        self.sum_by_target[target].add(value)

    def manifest(self) -> dict[str, object]:
        return {
            "count_by_target": {
                "0": self.count_by_target[0],
                "1": self.count_by_target[1],
            },
            "mean_given_target_0": (
                self.sum_by_target[0].total / self.count_by_target[0]
                if self.count_by_target[0]
                else None
            ),
            "mean_given_target_1": (
                self.sum_by_target[1].total / self.count_by_target[1]
                if self.count_by_target[1]
                else None
            ),
        }


@dataclass(slots=True)
class _SplitAccumulator:
    split: SplitName
    row_count: int = 0
    first_date: int | None = None
    last_date: int | None = None
    first_time_ms: int | None = None
    last_time_ms: int | None = None
    users: set[int] = field(default_factory=set)
    videos: set[int] = field(default_factory=set)
    authors: set[int | str] = field(default_factory=set)
    per_user_rows: Counter[int] = field(default_factory=Counter)
    per_user_positives: Counter[int] = field(default_factory=Counter)
    pair_counts: dict[int, int] = field(default_factory=dict)
    primary_positives: int = 0
    missing_author_rows: int = 0
    cold_user_rows: int = 0
    cold_video_rows: int = 0
    cold_author_rows: int = 0
    cold_users: set[int] = field(default_factory=set)
    cold_videos: set[int] = field(default_factory=set)
    cold_authors: set[int | str] = field(default_factory=set)

    def add(
        self,
        *,
        user_id: int,
        video_id: int,
        date: int,
        time_ms: int,
        author_id: int | str,
        target: int | None,
        train_entities: tuple[set[int], set[int], set[int | str]] | None,
    ) -> None:
        self.row_count += 1
        self.first_date = date if self.first_date is None else min(self.first_date, date)
        self.last_date = date if self.last_date is None else max(self.last_date, date)
        self.first_time_ms = (
            time_ms if self.first_time_ms is None else min(self.first_time_ms, time_ms)
        )
        self.last_time_ms = (
            time_ms if self.last_time_ms is None else max(self.last_time_ms, time_ms)
        )
        self.users.add(user_id)
        self.videos.add(video_id)
        self.authors.add(author_id)
        self.per_user_rows[user_id] += 1
        pair_key = (user_id << 64) | video_id
        self.pair_counts[pair_key] = self.pair_counts.get(pair_key, 0) + 1
        if author_id == "UNK":
            self.missing_author_rows += 1
        if target is not None:
            self.primary_positives += target
            self.per_user_positives[user_id] += target
        if train_entities is not None:
            train_users, train_videos, train_authors = train_entities
            if user_id not in train_users:
                self.cold_user_rows += 1
                self.cold_users.add(user_id)
            if video_id not in train_videos:
                self.cold_video_rows += 1
                self.cold_videos.add(video_id)
            if author_id not in train_authors:
                self.cold_author_rows += 1
                self.cold_authors.add(author_id)

    def manifest(self) -> dict[str, object]:
        if self.row_count == 0 or self.first_date is None or self.last_date is None:
            raise DataAuditError(f"{self.split.value} split contains no rows")
        slate_sizes = sorted(self.per_user_rows.values())
        repeated_multiplicities = [count for count in self.pair_counts.values() if count > 1]
        user_support = {"all_negative": 0, "mixed": 0, "all_positive": 0}
        if self.split is not SplitName.TEST:
            for user, count in self.per_user_rows.items():
                positives = self.per_user_positives[user]
                if positives == 0:
                    user_support["all_negative"] += 1
                elif positives == count:
                    user_support["all_positive"] += 1
                else:
                    user_support["mixed"] += 1
        result: dict[str, object] = {
            "split": self.split.value,
            "row_count": self.row_count,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "min_time_ms": self.first_time_ms,
            "max_time_ms": self.last_time_ms,
            "user_cardinality": len(self.users),
            "video_cardinality": len(self.videos),
            "author_cardinality": len(self.authors),
            "missing_author_rows": self.missing_author_rows,
            "per_user_slate_size": {
                "method": "nearest-rank",
                "min": slate_sizes[0],
                "p25": _nearest_rank(slate_sizes, 25, 100),
                "p50": _nearest_rank(slate_sizes, 50, 100),
                "p75": _nearest_rank(slate_sizes, 75, 100),
                "p95": _nearest_rank(slate_sizes, 95, 100),
                "p99": _nearest_rank(slate_sizes, 99, 100),
                "max": slate_sizes[-1],
                "mean": self.row_count / len(slate_sizes),
                "users_with_at_least_5_rows": sum(value >= 5 for value in slate_sizes),
            },
            "repeated_user_video_pairs": {
                "distinct_repeated_pairs": len(repeated_multiplicities),
                "rows_in_repeated_pairs": sum(repeated_multiplicities),
                "duplicate_rows_beyond_first": sum(value - 1 for value in repeated_multiplicities),
                "maximum_multiplicity": max(repeated_multiplicities, default=1),
            },
            "cold_relative_to_train": {
                "cold_user_cardinality": len(self.cold_users),
                "cold_user_rows": self.cold_user_rows,
                "cold_video_cardinality": len(self.cold_videos),
                "cold_video_rows": self.cold_video_rows,
                "cold_author_cardinality": len(self.cold_authors),
                "cold_author_rows": self.cold_author_rows,
            },
        }
        if self.split is SplitName.TEST:
            result["primary_target"] = None
            result["target_access"] = "forbidden_during_development"
            result["user_label_support"] = None
        else:
            negatives = self.row_count - self.primary_positives
            result["primary_target"] = {
                "field": (
                    f"{STANDARD_TRAIN_MEMBER}:long_view"
                    if self.split is SplitName.TRAIN
                    else f"{STANDARD_LATE_MEMBER}:long_view"
                ),
                "domain": [0, 1],
                "positives": self.primary_positives,
                "negatives": negatives,
                "positive_rate": self.primary_positives / self.row_count,
            }
            result["target_access"] = (
                "training" if self.split is SplitName.TRAIN else "trusted_audit_aggregate_only"
            )
            result["user_label_support"] = user_support
        return result


def _load_author_map(root: Path) -> dict[int, int]:
    path = root / _physical_filename(VIDEO_BASIC_MEMBER)
    selected = {
        _VIDEO_INDEX["video_id"]: "video_id",
        _VIDEO_INDEX["author_id"]: "author_id",
    }
    authors: dict[int, int] = {}
    with path.open("rb", buffering=1024 * 1024) as handle:
        next(handle)  # exact header was already verified byte-for-byte.
        for line_number, raw_line in enumerate(handle, start=2):
            source = f"{VIDEO_BASIC_MEMBER}:{line_number}"
            cells = _selected_cells(
                raw_line,
                selected,
                expected_field_count=len(VIDEO_BASIC_HEADER),
                source=source,
            )
            video_id = _parse_nonnegative_int(cells["video_id"], "video_id", source)
            author_id = _parse_nonnegative_int(cells["author_id"], "author_id", source)
            if video_id in authors:
                raise DataAuditError(f"{source} duplicates video_id in the author mapping")
            authors[video_id] = author_id
    return authors


def _selected_mapping(fields: Sequence[str]) -> Mapping[int, str]:
    return MappingProxyType({_LOG_INDEX[name]: name for name in fields})


def _audit_standard_logs(
    root: Path, author_map: Mapping[int, int]
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object], FinalOutcomeSkipTrace]:
    accumulators = {
        split: _SplitAccumulator(split)
        for split in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST)
    }
    binary_associations = {
        field_name: _BinaryAssociation() for field_name in _BINARY_AUXILIARY_FIELDS
    }
    numeric_associations = {
        field_name: _NumericAssociation() for field_name in _NONNEGATIVE_AUXILIARY_FIELDS
    }
    final_rows_from_late_member = 0

    for member in (STANDARD_TRAIN_MEMBER, STANDARD_LATE_MEMBER):
        path = root / _physical_filename(member)
        with path.open("rb", buffering=1024 * 1024) as handle:
            next(handle)  # exact header was already verified byte-for-byte.
            for line_number, raw_line in enumerate(handle, start=2):
                source = f"{member}:{line_number}"
                # Route on date using a first scan that materializes only that safe cell.  The
                # phase-specific second scan still never slices any disallowed outcome field.
                date_cell = _selected_cells(
                    raw_line,
                    {_LOG_INDEX["date"]: "date"},
                    expected_field_count=len(STANDARD_LOG_HEADER),
                    source=source,
                )["date"]
                date = _parse_nonnegative_int(date_cell, "date", source)
                split = _split_for_date(date)
                if member == STANDARD_TRAIN_MEMBER and split is not SplitName.TRAIN:
                    raise DataAuditError(f"{source} is outside the training member interval")
                if member == STANDARD_LATE_MEMBER and split is SplitName.TRAIN:
                    raise DataAuditError(
                        f"{source} is inside training but stored in the late member"
                    )

                selected_fields = {
                    SplitName.TRAIN: _TRAIN_SELECTED_FIELDS,
                    SplitName.VALID: _VALID_SELECTED_FIELDS,
                    SplitName.TEST: _FINAL_SELECTED_FIELDS,
                }[split]
                cells = _selected_cells(
                    raw_line,
                    _selected_mapping(selected_fields),
                    expected_field_count=len(STANDARD_LOG_HEADER),
                    source=source,
                )
                user_id = _parse_nonnegative_int(cells["user_id"], "user_id", source)
                video_id = _parse_nonnegative_int(cells["video_id"], "video_id", source)
                parsed_date = _parse_nonnegative_int(cells["date"], "date", source)
                if parsed_date != date:  # pragma: no cover - same immutable byte slice.
                    raise AssertionError("date changed between selected-cell scans")
                time_ms = _parse_nonnegative_int(cells["time_ms"], "time_ms", source)
                _parse_nonnegative_float(cells["duration_ms"], "duration_ms", source)
                tab = _parse_nonnegative_int(cells["tab"], "tab", source)
                if not 0 <= tab <= 14:
                    raise DataAuditError(f"{source} has tab outside the published range [0, 14]")
                target = (
                    _parse_binary(cells["long_view"], "long_view", source)
                    if split is not SplitName.TEST
                    else None
                )
                author_id: int | str = author_map.get(video_id, "UNK")
                train_entities: tuple[set[int], set[int], set[int | str]] | None = None
                if split is not SplitName.TRAIN:
                    train_accumulator = accumulators[SplitName.TRAIN]
                    train_entities = (
                        train_accumulator.users,
                        train_accumulator.videos,
                        train_accumulator.authors,
                    )
                accumulators[split].add(
                    user_id=user_id,
                    video_id=video_id,
                    date=date,
                    time_ms=time_ms,
                    author_id=author_id,
                    target=target,
                    train_entities=train_entities,
                )
                if split is SplitName.TRAIN:
                    if target is None:  # pragma: no cover - split branch establishes target.
                        raise AssertionError("train target unexpectedly absent")
                    for field_name, association in binary_associations.items():
                        auxiliary = _parse_binary(cells[field_name], field_name, source)
                        association.add(target, auxiliary)
                    for field_name, numeric_association in numeric_associations.items():
                        auxiliary_value = _parse_nonnegative_float(
                            cells[field_name], field_name, source
                        )
                        numeric_association.add(target, auxiliary_value)
                elif split is SplitName.TEST:
                    final_rows_from_late_member += 1

    split_manifests = tuple(
        accumulators[split].manifest()
        for split in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST)
    )
    associations: dict[str, object] = {
        "scope": "training rows only",
        "primary": f"{STANDARD_TRAIN_MEMBER}:long_view",
        "binary": {
            f"{STANDARD_TRAIN_MEMBER}:{name}": accumulator.manifest()
            for name, accumulator in binary_associations.items()
        },
        "nonnegative_numeric": {
            f"{STANDARD_TRAIN_MEMBER}:{name}": accumulator.manifest()
            for name, accumulator in numeric_associations.items()
        },
    }
    final_trace = FinalOutcomeSkipTrace(
        member=STANDARD_LATE_MEMBER,
        split=SplitName.TEST,
        row_count=final_rows_from_late_member,
        skipped_fields=_OUTCOME_FIELDS,
        selected_fields=_FINAL_SELECTED_FIELDS,
    )
    return split_manifests, MappingProxyType(associations), final_trace


def _validate_split_identities(
    splits: Sequence[Mapping[str, object]], contract: AuditContract
) -> None:
    observed_by_name = {SplitName(str(split["split"])): split for split in splits}
    for expected in contract.expected_splits:
        observed = observed_by_name[expected.split]
        facts = (
            int(str(observed["row_count"])),
            int(str(observed["first_date"])),
            int(str(observed["last_date"])),
        )
        expected_facts = (expected.row_count, expected.first_date, expected.last_date)
        if facts != expected_facts:
            raise DataAuditError(
                f"{expected.split.value} split identity mismatch: "
                f"expected {expected_facts}, got {facts}"
            )


def audit_dataset(
    data_dir: str | Path,
    *,
    contract: AuditContract = OFFICIAL_AUDIT_CONTRACT,
) -> DataAuditReport:
    """Audit an extracted dataset without ever interpreting final-period outcomes.

    The default contract is the official, hash-pinned KuaiRand-Pure identity.  Tests may pass an
    equally explicit synthetic :class:`AuditContract`; there is no "trust whatever is present"
    mode in production code.
    """

    root = _data_directory(data_dir)
    _verify_filesystem_shape(root)
    source_by_member = contract.source_by_member
    sources = tuple(_verify_source(root, source_by_member[member]) for member in _MEMBER_ORDER)
    author_map = _load_author_map(root)
    splits, associations, final_trace = _audit_standard_logs(root, author_map)
    _validate_split_identities(splits, contract)
    return DataAuditReport(
        data_dir=root,
        sources=sources,
        splits=splits,
        train_associations=associations,
        final_outcome_trace=final_trace,
        field_policy=field_policy_manifest(),
    )


def write_audit_report(
    report: DataAuditReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically write paired JSON and Markdown audit evidence."""

    outputs = (
        (Path(json_path), report.to_json(indent=2) + "\n"),
        (Path(markdown_path), report.readable_report()),
    )
    written: list[Path] = []
    for requested, content in outputs:
        path = requested.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        written.append(path)
    return written[0], written[1]


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DATASET_ARCHIVE_FILENAME",
    "DATASET_ARCHIVE_SIZE",
    "OFFICIAL_AUDIT_CONTRACT",
    "OFFICIAL_SOURCE_IDENTITIES",
    "OFFICIAL_SPLIT_IDENTITIES",
    "AuditContract",
    "DataAuditError",
    "DataAuditReport",
    "ExpectedSourceIdentity",
    "ExpectedSplitIdentity",
    "FinalOutcomeSkipTrace",
    "SourceAudit",
    "audit_dataset",
    "write_audit_report",
]
