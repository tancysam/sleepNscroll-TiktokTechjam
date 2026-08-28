"""Canonical, leakage-safe KuaiRand-Pure standard-log loading.

This module is part of the trusted data boundary.  It deliberately reads the two standard logs
with :mod:`csv` instead of a dataframe library so physical file order is explicit and outcome
columns can be skipped by index.  In particular, final-period outcome cells are never indexed,
converted, validated, included in a digest, or placed in an exception message.
"""

from __future__ import annotations

import csv
import hashlib
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from kuairand_agent.contract import BENCHMARK_CONTRACT, SplitName

CANONICAL_SCHEMA_VERSION: Final = 1
STANDARD_LOG_FILENAMES: Final = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
VIDEO_BASIC_FILENAME: Final = "video_features_basic_pure.csv"

# The order is the published KuaiRand archive schema, not merely a required-column subset.
LOG_HEADER: Final = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
)
VIDEO_BASIC_HEADER: Final = (
    "video_id",
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "video_duration",
    "server_width",
    "server_height",
    "music_id",
    "music_type",
    "tag",
)

PRIMARY_TARGET: Final = "long_view"
BINARY_AUXILIARY_TARGETS: Final = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
)
NONNEGATIVE_AUXILIARY_TARGETS: Final = (
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
)
APPROVED_AUXILIARY_TARGETS: Final = (
    *BINARY_AUXILIARY_TARGETS,
    *NONNEGATIVE_AUXILIARY_TARGETS,
)
OUTCOME_FIELDS: Final = (PRIMARY_TARGET, *APPROVED_AUXILIARY_TARGETS)
SAFE_INPUT_FIELDS: Final = (
    "user_id",
    "video_id",
    "date",
    "duration_ms",
    "tab",
    "author_id",
)
TRUSTED_BUILDER_FIELDS: Final = ("time_ms",)
UNKNOWN_AUTHOR: Final = "UNK"

_LOG_INDEX: Final = MappingProxyType({name: index for index, name in enumerate(LOG_HEADER)})
_VIDEO_INDEX: Final = MappingProxyType(
    {name: index for index, name in enumerate(VIDEO_BASIC_HEADER)}
)
_SPLIT_INTERVALS: Final = MappingProxyType(
    {split.name: (split.start_date, split.end_date) for split in BENCHMARK_CONTRACT.splits}
)
_SAFE_LOG_PROJECTION: Final = tuple(
    _LOG_INDEX[name] for name in ("user_id", "video_id", "date", "duration_ms", "tab", "time_ms")
)
_LOG_HEADER_BYTES: Final = ",".join(LOG_HEADER).encode("ascii")

type Identifier = str
type AuthorIdentifier = str
type TargetValue = int | float


class CanonicalDataError(ValueError):
    """Raised when archive data violates the frozen canonical-data contract."""


class TargetAccess(StrEnum):
    """The only ways a canonical target bundle may be consumed."""

    TRAINING = "training"
    PROTECTED_SCORER = "protected_scorer_only"


def _digest_columns(
    namespace: bytes,
    columns: Iterable[tuple[str, Sequence[int | float | str]]],
) -> str:
    """Hash typed column values without constructing a second full-data JSON representation."""

    digest = hashlib.sha256(namespace + b"\0")
    for name, values in columns:
        encoded_name = name.encode("ascii")
        digest.update(struct.pack("<H", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(values)))
        for value in values:
            if type(value) is int:
                digest.update(b"i")
                try:
                    digest.update(struct.pack("<q", value))
                except struct.error as exc:
                    raise CanonicalDataError(f"{name} contains an integer outside int64") from exc
            elif type(value) is float:
                digest.update(b"f")
                digest.update(struct.pack("<d", value))
            elif type(value) is str:
                encoded = value.encode("utf-8")
                digest.update(b"s")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
            else:  # pragma: no cover - all public constructors normalize before hashing.
                raise CanonicalDataError(f"{name} contains an unsupported canonical value type")
    return digest.hexdigest()


def _stable_digest(namespace: bytes, parts: Sequence[str]) -> str:
    digest = hashlib.sha256(namespace + b"\0")
    for part in parts:
        encoded = part.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _same_length(name: str, columns: Mapping[str, Sequence[object]]) -> int:
    lengths = {field_name: len(values) for field_name, values in columns.items()}
    unique = set(lengths.values())
    if len(unique) != 1:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(lengths.items()))
        raise CanonicalDataError(f"{name} columns must have identical lengths; got {rendered}")
    return next(iter(unique), 0)


def _canonical_int_sequence(values: Sequence[object], field_name: str) -> tuple[int, ...]:
    normalized: list[int] = []
    for index, value in enumerate(values):
        if type(value) is not int:
            raise CanonicalDataError(f"{field_name}[{index}] must be an integer")
        normalized.append(value)
    return tuple(normalized)


def _canonical_text_sequence(values: Sequence[object], field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, value in enumerate(values):
        if type(value) is not str or not value or "\x00" in value:
            raise CanonicalDataError(f"{field_name}[{index}] must be non-empty canonical text")
        normalized.append(value)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class CanonicalInputs:
    """Immutable, fixed-order safe input columns plus trusted event time.

    ``time_ms`` is retained for causal builders in the trusted data layer.  Candidate capability
    builders must project the explicitly declared safe columns and cannot expose this object as a
    monolithic raw table.
    """

    user_id: Sequence[str]
    video_id: Sequence[str]
    date: Sequence[int]
    duration_ms: Sequence[float]
    tab: Sequence[str]
    author_id: Sequence[AuthorIdentifier]
    time_ms: Sequence[int]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        users = _canonical_text_sequence(cast(Sequence[object], self.user_id), "user_id")
        videos = _canonical_text_sequence(cast(Sequence[object], self.video_id), "video_id")
        dates = _canonical_int_sequence(cast(Sequence[object], self.date), "date")
        tabs = _canonical_text_sequence(cast(Sequence[object], self.tab), "tab")
        times = _canonical_int_sequence(cast(Sequence[object], self.time_ms), "time_ms")

        durations: list[float] = []
        for index, duration_raw in enumerate(self.duration_ms):
            if (
                type(duration_raw) is not float
                or not math.isfinite(duration_raw)
                or duration_raw < 0
            ):
                raise CanonicalDataError(
                    f"duration_ms[{index}] must be a finite non-negative float"
                )
            durations.append(duration_raw)

        authors: list[str] = []
        for index, author_raw in enumerate(self.author_id):
            if type(author_raw) is not str or not author_raw or "\x00" in author_raw:
                raise CanonicalDataError(f"author_id[{index}] must be non-empty canonical text")
            authors.append(author_raw)

        columns: dict[str, Sequence[object]] = {
            "user_id": users,
            "video_id": videos,
            "date": dates,
            "duration_ms": durations,
            "tab": tabs,
            "author_id": authors,
            "time_ms": times,
        }
        _same_length("canonical input", columns)
        if any(value < 0 for value in times):
            raise CanonicalDataError("time_ms must be non-negative")
        for index, value in enumerate(tabs):
            try:
                numeric_tab = int(value)
            except ValueError as exc:
                raise CanonicalDataError(f"tab[{index}] must be numeric text") from exc
            if not 0 <= numeric_tab <= 14:
                raise CanonicalDataError("tab must be numeric text in the published range [0, 14]")

        object.__setattr__(self, "user_id", users)
        object.__setattr__(self, "video_id", videos)
        object.__setattr__(self, "date", dates)
        object.__setattr__(self, "duration_ms", tuple(durations))
        object.__setattr__(self, "tab", tabs)
        object.__setattr__(self, "author_id", tuple(authors))
        object.__setattr__(self, "time_ms", times)
        object.__setattr__(
            self,
            "digest",
            _digest_columns(
                b"kuairand-canonical-inputs-v1",
                (
                    ("user_id", users),
                    ("video_id", videos),
                    ("date", dates),
                    ("duration_ms", tuple(durations)),
                    ("tab", tabs),
                    ("author_id", tuple(authors)),
                    ("time_ms", times),
                ),
            ),
        )

    def __len__(self) -> int:
        return len(self.user_id)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the trusted schema; capability code must still apply its field registry."""

        return (*SAFE_INPUT_FIELDS, *TRUSTED_BUILDER_FIELDS)

    def column(self, name: str) -> Sequence[int | float | str]:
        """Return one declared input column without accepting arbitrary/raw field names."""

        if name not in self.field_names:
            raise CanonicalDataError(f"unknown canonical input field {name!r}")
        return cast(Sequence[int | float | str], getattr(self, name))


@dataclass(frozen=True, slots=True)
class CanonicalAlignment:
    """Trusted row/user/video identity, separate from candidate-visible input columns."""

    split: SplitName
    row_id: Sequence[int]
    user_id: Sequence[str]
    video_id: Sequence[str]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.split, SplitName):
            raise CanonicalDataError("alignment split must be a SplitName")
        rows = _canonical_int_sequence(cast(Sequence[object], self.row_id), "row_id")
        users = _canonical_text_sequence(cast(Sequence[object], self.user_id), "user_id")
        videos = _canonical_text_sequence(cast(Sequence[object], self.video_id), "video_id")
        _same_length("alignment", {"row_id": rows, "user_id": users, "video_id": videos})
        if rows != tuple(range(len(rows))):
            raise CanonicalDataError("row_id must be contiguous canonical split order from zero")
        object.__setattr__(self, "row_id", rows)
        object.__setattr__(self, "user_id", users)
        object.__setattr__(self, "video_id", videos)
        object.__setattr__(
            self,
            "digest",
            _digest_columns(
                f"kuairand-alignment-v1:{self.split.value}".encode("ascii"),
                (("row_id", rows), ("user_id", users), ("video_id", videos)),
            ),
        )

    def __len__(self) -> int:
        return len(self.row_id)

    @property
    def row_ids(self) -> tuple[int, ...]:
        """Plural alias used by scorer/finalizer adapters."""

        return cast(tuple[int, ...], self.row_id)

    @property
    def user_ids(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.user_id)

    @property
    def video_ids(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.video_id)

    @property
    def split_digest(self) -> str:
        """Opaque alignment token; target values never contribute to it."""

        return self.digest


@dataclass(frozen=True, slots=True)
class TrainingTargets:
    """Immutable train-only primary and approved auxiliary target columns."""

    _columns: Mapping[str, Sequence[TargetValue]] = field(repr=False)
    digest: str = field(init=False)
    access: TargetAccess = field(init=False, default=TargetAccess.TRAINING)

    def __post_init__(self) -> None:
        expected_names = (PRIMARY_TARGET, *APPROVED_AUXILIARY_TARGETS)
        if tuple(self._columns) != expected_names:
            raise CanonicalDataError(
                "training target columns must exactly match the approved target registry"
            )
        normalized = {name: tuple(values) for name, values in self._columns.items()}
        _same_length("training target", normalized)
        immutable = MappingProxyType(normalized)
        object.__setattr__(self, "_columns", immutable)
        object.__setattr__(
            self,
            "digest",
            _digest_columns(
                b"kuairand-training-targets-v1",
                ((name, immutable[name]) for name in expected_names),
            ),
        )

    def __len__(self) -> int:
        return len(self._columns[PRIMARY_TARGET])

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self._columns)

    @property
    def long_view(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self._columns[PRIMARY_TARGET])

    def column(self, name: str) -> tuple[TargetValue, ...]:
        """Expose registered targets only to the training capability builder."""

        try:
            return cast(tuple[TargetValue, ...], self._columns[name])
        except KeyError as exc:
            raise CanonicalDataError(f"unapproved training target {name!r}") from exc


@dataclass(frozen=True, slots=True)
class ProtectedTargets:
    """Opaque public-validation labels intended only for the trusted scorer.

    The class is deliberately non-iterable and has no generic ``column`` method, preventing
    accidental reuse by a candidate capability builder.  Trusted scoring code must make the
    explicit access decision expressed by :meth:`reveal_for_scorer`.
    """

    _long_view: Sequence[int] = field(repr=False)
    digest: str = field(init=False)
    access: TargetAccess = field(init=False, default=TargetAccess.PROTECTED_SCORER)

    def __post_init__(self) -> None:
        values = _canonical_int_sequence(cast(Sequence[object], self._long_view), PRIMARY_TARGET)
        if any(value not in (0, 1) for value in values):
            raise CanonicalDataError("protected long_view must be binary")
        object.__setattr__(self, "_long_view", values)
        object.__setattr__(
            self,
            "digest",
            _digest_columns(b"kuairand-protected-outer-targets-v1", ((PRIMARY_TARGET, values),)),
        )

    def __len__(self) -> int:
        return len(self._long_view)

    def reveal_for_scorer(self) -> tuple[int, ...]:
        """Return labels after the caller has entered the trusted scorer path."""

        return cast(tuple[int, ...], self._long_view)


@dataclass(frozen=True, slots=True)
class OutcomeAccessTrace:
    """Value-free, machine-readable proof of phase-specific outcome handling."""

    split: SplitName
    row_count: int
    parsed_fields: tuple[str, ...]
    skipped_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise CanonicalDataError("outcome trace row_count cannot be negative")
        if set(self.parsed_fields) & set(self.skipped_fields):
            raise CanonicalDataError("outcome trace fields cannot be both parsed and skipped")
        if set((*self.parsed_fields, *self.skipped_fields)) != set(OUTCOME_FIELDS):
            raise CanonicalDataError("outcome trace must account for every registered outcome")

    @property
    def parsed_cell_count(self) -> int:
        return self.row_count * len(self.parsed_fields)

    @property
    def skipped_cell_count(self) -> int:
        return self.row_count * len(self.skipped_fields)

    def manifest(self) -> dict[str, object]:
        """Return counts and names only; skipped values can never enter diagnostics."""

        return {
            "split": self.split.value,
            "row_count": self.row_count,
            "parsed_fields": list(self.parsed_fields),
            "skipped_fields": list(self.skipped_fields),
            "parsed_cell_count": self.parsed_cell_count,
            "skipped_cell_count": self.skipped_cell_count,
            "skipped_values_recorded": False,
        }


@dataclass(frozen=True, slots=True)
class CanonicalSplit:
    """One immutable organizer split with inputs, targets, and alignment separated."""

    name: SplitName
    inputs: CanonicalInputs
    alignment: CanonicalAlignment
    targets: TrainingTargets | ProtectedTargets | None
    outcome_trace: OutcomeAccessTrace
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        count = len(self.inputs)
        if self.alignment.split != self.name or self.outcome_trace.split != self.name:
            raise CanonicalDataError("split identity differs across canonical components")
        if len(self.alignment) != count or self.outcome_trace.row_count != count:
            raise CanonicalDataError("canonical split component row counts differ")
        expected_target_type: type[TrainingTargets] | type[ProtectedTargets] | None
        if self.name is SplitName.TRAIN:
            expected_target_type = TrainingTargets
        elif self.name is SplitName.VALID:
            expected_target_type = ProtectedTargets
        else:
            expected_target_type = None
        if expected_target_type is None:
            if self.targets is not None:
                raise CanonicalDataError("final split must not contain targets")
            target_digest = "no-final-targets"
        elif not isinstance(self.targets, expected_target_type):
            raise CanonicalDataError(f"{self.name.value} split has the wrong target capability")
        else:
            if len(self.targets) != count:
                raise CanonicalDataError("canonical split and target row counts differ")
            target_digest = self.targets.digest
        if self.inputs.user_id != self.alignment.user_id:
            raise CanonicalDataError("input and alignment user order differ")
        if self.inputs.video_id != self.alignment.video_id:
            raise CanonicalDataError("input and alignment video order differ")
        trace_digest = _stable_digest(
            b"kuairand-outcome-trace-v1",
            (
                self.name.value,
                str(self.outcome_trace.row_count),
                ",".join(self.outcome_trace.parsed_fields),
                ",".join(self.outcome_trace.skipped_fields),
            ),
        )
        object.__setattr__(
            self,
            "digest",
            _stable_digest(
                b"kuairand-canonical-split-v1",
                (
                    self.name.value,
                    self.inputs.digest,
                    self.alignment.digest,
                    target_digest,
                    trace_digest,
                ),
            ),
        )

    @property
    def row_count(self) -> int:
        return len(self.inputs)

    @property
    def split_digest(self) -> str:
        return self.digest

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "name": self.name.value,
            "row_count": self.row_count,
            "input_fields": list(self.inputs.field_names),
            "inputs_digest": self.inputs.digest,
            "alignment_digest": self.alignment.digest,
            "target_access": self.targets.access.value if self.targets is not None else "none",
            "target_digest": self.targets.digest if self.targets is not None else None,
            "outcome_access": self.outcome_trace.manifest(),
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class CanonicalDataset:
    """Complete immutable train/public-validation/final logical dataset."""

    train: CanonicalSplit
    valid: CanonicalSplit
    final: CanonicalSplit
    author_map_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.train.name, self.valid.name, self.final.name) != (
            SplitName.TRAIN,
            SplitName.VALID,
            SplitName.TEST,
        ):
            raise CanonicalDataError("canonical dataset splits must be train, valid, final")
        object.__setattr__(
            self,
            "digest",
            _stable_digest(
                b"kuairand-canonical-dataset-v1",
                (
                    str(CANONICAL_SCHEMA_VERSION),
                    self.author_map_digest,
                    self.train.digest,
                    self.valid.digest,
                    self.final.digest,
                ),
            ),
        )

    @property
    def test(self) -> CanonicalSplit:
        """Organizer-compatible alias for the final evaluation split."""

        return self.final

    def split(self, name: SplitName | str) -> CanonicalSplit:
        try:
            normalized = name if isinstance(name, SplitName) else SplitName(name)
        except ValueError as exc:
            raise CanonicalDataError(f"unknown canonical split {name!r}") from exc
        return {
            SplitName.TRAIN: self.train,
            SplitName.VALID: self.valid,
            SplitName.TEST: self.final,
        }[normalized]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "standard_log_order": list(STANDARD_LOG_FILENAMES),
            "log_header": list(LOG_HEADER),
            "video_basic_header": list(VIDEO_BASIC_HEADER),
            "author_map_digest": self.author_map_digest,
            "splits": [self.train.manifest(), self.valid.manifest(), self.final.manifest()],
            "digest": self.digest,
        }


@dataclass(slots=True)
class _SplitBuilder:
    user_id: list[str] = field(default_factory=list)
    video_id: list[str] = field(default_factory=list)
    date: list[int] = field(default_factory=list)
    duration_ms: list[float] = field(default_factory=list)
    tab: list[str] = field(default_factory=list)
    author_id: list[AuthorIdentifier] = field(default_factory=list)
    time_ms: list[int] = field(default_factory=list)
    targets: dict[str, list[TargetValue]] = field(default_factory=dict)


def _source(path: Path, line_number: int) -> str:
    return f"{path.name}:{line_number}"


def _parse_int(cell: str, field_name: str, source: str, *, nonnegative: bool = True) -> int:
    try:
        value = int(cell)
    except (TypeError, ValueError) as exc:
        raise CanonicalDataError(f"{source} has invalid integer field {field_name}") from exc
    if str(value) != cell:
        raise CanonicalDataError(f"{source} has non-canonical integer field {field_name}")
    if nonnegative and value < 0:
        raise CanonicalDataError(f"{source} has negative field {field_name}")
    return value


def _parse_identifier(cell: str, field_name: str, source: str) -> str:
    """Validate the published int64 domain while retaining exact organizer CSV text."""

    if not cell or "\x00" in cell:
        raise CanonicalDataError(f"{source} has empty identifier field {field_name}")
    try:
        value = int(cell)
    except ValueError as exc:
        raise CanonicalDataError(f"{source} has invalid identifier field {field_name}") from exc
    if value < 0:
        raise CanonicalDataError(f"{source} has negative identifier field {field_name}")
    return cell


def _parse_tab(cell: str, source: str) -> str:
    try:
        value = int(cell)
    except ValueError as exc:
        raise CanonicalDataError(f"{source} has invalid integer field tab") from exc
    if not 0 <= value <= 14:
        raise CanonicalDataError(f"{source} has tab outside the published range [0, 14]")
    return cell


def _parse_binary(cell: str, field_name: str, source: str) -> int:
    if cell == "0":
        return 0
    if cell == "1":
        return 1
    raise CanonicalDataError(f"{source} has non-binary target {field_name}")


def _parse_nonnegative_float(cell: str, field_name: str, source: str) -> float:
    try:
        value = float(cell)
    except (TypeError, ValueError) as exc:
        raise CanonicalDataError(f"{source} has invalid numeric field {field_name}") from exc
    if not math.isfinite(value) or value < 0:
        raise CanonicalDataError(f"{source} has invalid non-negative field {field_name}")
    return value


def _check_header(path: Path, observed: Sequence[str] | None, expected: tuple[str, ...]) -> None:
    if observed is None:
        raise CanonicalDataError(f"{path.name} is empty and has no header")
    if tuple(observed) != expected:
        raise CanonicalDataError(
            f"{path.name} header mismatch; expected {expected!r}, observed {tuple(observed)!r}"
        )


def _without_record_ending(line: bytes) -> bytes:
    if line.endswith(b"\n"):
        line = line[:-1]
        if line.endswith(b"\r"):
            line = line[:-1]
    return line


def _check_binary_log_header(path: Path, observed: bytes) -> None:
    if not observed:
        raise CanonicalDataError(f"{path.name} is empty and has no header")
    if _without_record_ending(observed) != _LOG_HEADER_BYTES:
        raise CanonicalDataError(f"{path.name} header mismatch")


def _project_binary_log_fields(
    line: bytes,
    selected_indices: Sequence[int],
    source: str,
) -> dict[str, str]:
    """Decode selected numeric cells while treating all other bytes as framing only.

    Official standard-log fields are unquoted numeric tokens.  The scanner records delimiter
    boundaries but creates byte slices and text objects only for the selected indices.  Thus an
    invalid UTF-8 byte sequence in a skipped final outcome is neither decoded nor surfaced.
    """

    record = _without_record_ending(line)
    selected = frozenset(selected_indices)
    projected: dict[str, str] = {}
    field_index = 0
    field_start = 0
    for position in range(len(record) + 1):
        if position != len(record) and record[position] != 0x2C:  # comma
            continue
        if field_index in selected:
            raw = record[field_start:position]
            try:
                projected[LOG_HEADER[field_index]] = raw.decode("ascii")
            except UnicodeDecodeError as exc:
                raise CanonicalDataError(
                    f"{source} has non-ASCII data in selected field {LOG_HEADER[field_index]}"
                ) from exc
        field_index += 1
        field_start = position + 1
    if field_index != len(LOG_HEADER):
        raise CanonicalDataError(f"{source} has the wrong number of CSV fields")
    if len(projected) != len(selected):
        raise CanonicalDataError(f"{source} is missing a selected canonical field")
    return projected


def _load_author_map(data_dir: Path) -> tuple[dict[str, str], str]:
    path = data_dir / VIDEO_BASIC_FILENAME
    authors: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        _check_header(path, next(reader, None), VIDEO_BASIC_HEADER)
        for line_number, row in enumerate(reader, start=2):
            source = _source(path, line_number)
            if len(row) != len(VIDEO_BASIC_HEADER):
                raise CanonicalDataError(f"{source} has the wrong number of CSV fields")
            video_id = _parse_identifier(row[_VIDEO_INDEX["video_id"]], "video_id", source)
            author_id = _parse_identifier(row[_VIDEO_INDEX["author_id"]], "author_id", source)
            if video_id in authors:
                raise CanonicalDataError(
                    f"{source} duplicates video_id in the unique author mapping"
                )
            authors[video_id] = author_id
    keys = tuple(authors)
    values = tuple(authors[key] for key in keys)
    digest = _digest_columns(b"kuairand-author-map-v1", (("video_id", keys), ("author_id", values)))
    return authors, digest


def _split_for_date(date: int) -> SplitName | None:
    for name in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST):
        low, high = _SPLIT_INTERVALS[name]
        if low <= date <= high:
            return name
    return None


def _target_fields_for_split(split: SplitName) -> tuple[str, ...]:
    if split is SplitName.TRAIN:
        return (PRIMARY_TARGET, *APPROVED_AUXILIARY_TARGETS)
    if split is SplitName.VALID:
        return (PRIMARY_TARGET,)
    return ()


def _append_row(
    *,
    row: Mapping[str, str],
    source: str,
    split: SplitName,
    builder: _SplitBuilder,
    author_map: Mapping[str, str],
) -> None:
    # The canonical identity is fixed here, before the author lookup (the only join).
    row_id = len(builder.user_id)
    user_id = _parse_identifier(row["user_id"], "user_id", source)
    video_id = _parse_identifier(row["video_id"], "video_id", source)
    date = _parse_int(row["date"], "date", source)
    duration = _parse_nonnegative_float(row["duration_ms"], "duration_ms", source)
    tab = _parse_tab(row["tab"], source)
    time_ms = _parse_int(row["time_ms"], "time_ms", source)

    # A missing basic-video row maps to one value and never changes cardinality.
    author_id: AuthorIdentifier = author_map.get(video_id, UNKNOWN_AUTHOR)
    builder.user_id.append(user_id)
    builder.video_id.append(video_id)
    builder.date.append(date)
    builder.duration_ms.append(duration)
    builder.tab.append(tab)
    builder.author_id.append(author_id)
    builder.time_ms.append(time_ms)

    for name in _target_fields_for_split(split):
        if name in BINARY_AUXILIARY_TARGETS or name == PRIMARY_TARGET:
            value: TargetValue = _parse_binary(row[name], name, source)
        else:
            value = _parse_nonnegative_float(row[name], name, source)
        builder.targets[name].append(value)
    if row_id != len(builder.user_id) - 1:  # pragma: no cover - explicit invariant documentation.
        raise AssertionError("canonical row identity changed during author join")


def _build_split(name: SplitName, builder: _SplitBuilder) -> CanonicalSplit:
    inputs = CanonicalInputs(
        user_id=builder.user_id,
        video_id=builder.video_id,
        date=builder.date,
        duration_ms=builder.duration_ms,
        tab=builder.tab,
        author_id=builder.author_id,
        time_ms=builder.time_ms,
    )
    alignment = CanonicalAlignment(
        split=name,
        row_id=tuple(range(len(inputs))),
        user_id=inputs.user_id,
        video_id=inputs.video_id,
    )
    parsed = _target_fields_for_split(name)
    skipped = tuple(field_name for field_name in OUTCOME_FIELDS if field_name not in parsed)
    trace = OutcomeAccessTrace(
        split=name,
        row_count=len(inputs),
        parsed_fields=parsed,
        skipped_fields=skipped,
    )
    if name is SplitName.TRAIN:
        targets: TrainingTargets | ProtectedTargets | None = TrainingTargets(builder.targets)
    elif name is SplitName.VALID:
        targets = ProtectedTargets(cast(Sequence[int], builder.targets[PRIMARY_TARGET]))
    else:
        targets = None
    return CanonicalSplit(
        name=name,
        inputs=inputs,
        alignment=alignment,
        targets=targets,
        outcome_trace=trace,
    )


def load_canonical_dataset(data_dir: str | Path) -> CanonicalDataset:
    """Load official standard logs into immutable, phase-specific canonical splits.

    Files are consumed in ``STANDARD_LOG_FILENAMES`` order.  Retained rows never pass through a
    sort, and repeated user/video pairs remain separate because identity is positional.
    """

    root = Path(data_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise CanonicalDataError(f"canonical data path is not a directory: {root}")
    author_map, author_map_digest = _load_author_map(root)
    builders = {
        name: _SplitBuilder(
            targets={field_name: [] for field_name in _target_fields_for_split(name)}
        )
        for name in (SplitName.TRAIN, SplitName.VALID, SplitName.TEST)
    }

    for filename in STANDARD_LOG_FILENAMES:
        path = root / filename
        with path.open("rb") as handle:
            _check_binary_log_header(path, handle.readline())
            for line_number, line in enumerate(handle, start=2):
                source = _source(path, line_number)
                row = _project_binary_log_fields(line, _SAFE_LOG_PROJECTION, source)
                date = _parse_int(row["date"], "date", source)
                split = _split_for_date(date)
                if split is None:
                    continue
                target_indices = tuple(_LOG_INDEX[name] for name in _target_fields_for_split(split))
                if target_indices:
                    row.update(_project_binary_log_fields(line, target_indices, source))
                _append_row(
                    row=row,
                    source=source,
                    split=split,
                    builder=builders[split],
                    author_map=author_map,
                )

    train = _build_split(SplitName.TRAIN, builders[SplitName.TRAIN])
    valid = _build_split(SplitName.VALID, builders[SplitName.VALID])
    final = _build_split(SplitName.TEST, builders[SplitName.TEST])
    return CanonicalDataset(
        train=train,
        valid=valid,
        final=final,
        author_map_digest=author_map_digest,
    )


def load_canonical(data_dir: str | Path) -> CanonicalDataset:
    """Concise alias for :func:`load_canonical_dataset`."""

    return load_canonical_dataset(data_dir)


def canonical_dataset_from_manifest(manifest: Mapping[str, object]) -> None:
    """Fail explicitly: logical manifests prove identity but never contain row-level data."""

    del manifest
    raise CanonicalDataError("canonical datasets must be replayed from verified source data")


__all__ = [
    "APPROVED_AUXILIARY_TARGETS",
    "BINARY_AUXILIARY_TARGETS",
    "CANONICAL_SCHEMA_VERSION",
    "LOG_HEADER",
    "NONNEGATIVE_AUXILIARY_TARGETS",
    "OUTCOME_FIELDS",
    "PRIMARY_TARGET",
    "SAFE_INPUT_FIELDS",
    "STANDARD_LOG_FILENAMES",
    "TRUSTED_BUILDER_FIELDS",
    "UNKNOWN_AUTHOR",
    "VIDEO_BASIC_FILENAME",
    "VIDEO_BASIC_HEADER",
    "CanonicalAlignment",
    "CanonicalDataError",
    "CanonicalDataset",
    "CanonicalInputs",
    "CanonicalSplit",
    "OutcomeAccessTrace",
    "ProtectedTargets",
    "TargetAccess",
    "TrainingTargets",
    "load_canonical",
    "load_canonical_dataset",
]
