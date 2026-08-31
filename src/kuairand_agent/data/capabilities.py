"""Leakage-safe, immutable data capabilities exposed at the candidate seam.

The factories in this module copy only explicitly requested, source-qualified fields.  Values
from outcome sidecars are never inspected while candidate inputs are built, so protected and
skipped outcome mutations cannot influence input content or manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.data.fields import (
    FIELD_POLICY_DIGEST,
    STANDARD_LATE_MEMBER,
    STANDARD_TRAIN_MEMBER,
    VIDEO_BASIC_MEMBER,
    FieldKey,
    FieldPolicyError,
    FieldRole,
    LogicalDType,
    field_spec,
)

if TYPE_CHECKING:
    from kuairand_agent.data.canonical import (
        CanonicalFinalSplit,
        CanonicalInputs,
        CanonicalTrainingSplit,
        CanonicalValidationSplit,
    )


class CapabilityError(ValueError):
    """Raised when a capability request violates schema, phase, or field policy."""


class DataPhase(StrEnum):
    """Outcome-access phases used by the trusted data controller."""

    TRAIN = "train"
    INNER_TRAIN = "inner_train"
    INNER_VALID = "inner_valid"
    OUTER_VALID = "outer_valid"
    FINAL = "final"


class CandidateCapabilityName(StrEnum):
    """The complete capability names a generated candidate may request."""

    TRAIN_INPUTS = "train_inputs"
    TRAIN_TARGETS = "train_targets"
    INNER_VALID_INPUTS = "inner_valid_inputs"
    OUTER_VALID_INPUTS = "outer_valid_inputs"
    FINAL_INPUTS = "final_inputs"


_INPUT_CAPABILITY_BY_PHASE: Final[Mapping[DataPhase, CandidateCapabilityName]] = MappingProxyType(
    {
        DataPhase.TRAIN: CandidateCapabilityName.TRAIN_INPUTS,
        DataPhase.INNER_TRAIN: CandidateCapabilityName.TRAIN_INPUTS,
        DataPhase.INNER_VALID: CandidateCapabilityName.INNER_VALID_INPUTS,
        DataPhase.OUTER_VALID: CandidateCapabilityName.OUTER_VALID_INPUTS,
        DataPhase.FINAL: CandidateCapabilityName.FINAL_INPUTS,
    }
)

_LOG_MEMBER_BY_PHASE: Final[Mapping[DataPhase, str]] = MappingProxyType(
    {
        DataPhase.TRAIN: STANDARD_TRAIN_MEMBER,
        DataPhase.INNER_TRAIN: STANDARD_TRAIN_MEMBER,
        DataPhase.INNER_VALID: STANDARD_TRAIN_MEMBER,
        DataPhase.OUTER_VALID: STANDARD_LATE_MEMBER,
        DataPhase.FINAL: STANDARD_LATE_MEMBER,
    }
)

_STORAGE_DTYPES: Final[Mapping[LogicalDType, np.dtype[np.generic]]] = MappingProxyType(
    {
        LogicalDType.INTEGER: np.dtype("<i8"),
        LogicalDType.NUMBER: np.dtype("<f8"),
        LogicalDType.BINARY: np.dtype("i1"),
        LogicalDType.STRING: np.dtype("<U64"),
    }
)

_RESERVED_CANDIDATE_NAMES: Final = frozenset(
    {
        "row_id",
        "alignment",
        "alignment_hash",
        "split_digest",
        "source_member_id",
        "source_ordinal",
        "source_record_ordinal",
        "provenance_hash",
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityColumn:
    """One exact ordered output column and its source-qualified policy identity."""

    name: str
    field_key: FieldKey
    logical_dtype: LogicalDType
    storage_dtype: str

    def manifest(self) -> dict[str, str]:
        """Return stable, value-free column metadata."""

        return {
            "name": self.name,
            "source_field": self.field_key.wire_name,
            "logical_dtype": self.logical_dtype.value,
            "storage_dtype": self.storage_dtype,
        }


@dataclass(frozen=True, slots=True)
class CandidateInputs:
    """Immutable, fixed-order candidate input arrays with a value-free manifest."""

    capability_name: CandidateCapabilityName
    phase: DataPhase
    schema: tuple[CapabilityColumn, ...]
    row_count: int
    field_policy_digest: str
    schema_digest: str
    logical_content_digest: str
    digest: str
    _columns: Mapping[str, npt.NDArray[np.generic]] = dataclass_field(repr=False)

    @property
    def columns(self) -> Mapping[str, npt.NDArray[np.generic]]:
        """Return the immutable name-to-array view in schema order."""

        return self._columns

    def column(self, name: str) -> npt.NDArray[np.generic]:
        """Return one read-only array, rejecting undeclared names."""

        try:
            return self._columns[name]
        except KeyError as exc:
            raise CapabilityError(f"candidate input column is not declared: {name!r}") from exc

    def manifest(self) -> dict[str, object]:
        """Return a fresh value-free manifest suitable for candidate/research context."""

        return {
            "schema_version": 1,
            "capability_name": self.capability_name.value,
            "phase": self.phase.value,
            "row_count": self.row_count,
            "columns": [column.manifest() for column in self.schema],
            "field_policy_digest": self.field_policy_digest,
            "capability_schema_digest": self.schema_digest,
            "logical_content_digest": self.logical_content_digest,
            "capability_digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class TrainTargets:
    """Immutable targets available only to official-train candidate fitting."""

    phase: DataPhase
    schema: tuple[CapabilityColumn, ...]
    row_count: int
    field_policy_digest: str
    schema_digest: str
    logical_content_digest: str
    digest: str
    _columns: Mapping[str, npt.NDArray[np.generic]] = dataclass_field(repr=False)

    @property
    def columns(self) -> Mapping[str, npt.NDArray[np.generic]]:
        """Return immutable primary/approved-auxiliary target arrays."""

        return self._columns

    @property
    def primary(self) -> npt.NDArray[np.generic]:
        """Return the native ``long_view`` target."""

        try:
            return self._columns["long_view"]
        except KeyError as exc:  # defended even for manually constructed objects
            raise CapabilityError("training targets do not contain long_view") from exc

    def manifest(self) -> dict[str, object]:
        """Return schema and content identity, never row-level target values."""

        return {
            "schema_version": 1,
            "capability_name": CandidateCapabilityName.TRAIN_TARGETS.value,
            "phase": self.phase.value,
            "row_count": self.row_count,
            "columns": [column.manifest() for column in self.schema],
            "field_policy_digest": self.field_policy_digest,
            "capability_schema_digest": self.schema_digest,
            "logical_content_digest": self.logical_content_digest,
            "capability_digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class TrainingCapabilities:
    """Candidate-facing training inputs and targets built from one typed split."""

    inputs: CandidateInputs
    targets: TrainTargets

    def __post_init__(self) -> None:
        if self.inputs.phase is not DataPhase.TRAIN or self.targets.phase is not DataPhase.TRAIN:
            raise CapabilityError("training capabilities require the train phase")
        if self.inputs.row_count != self.targets.row_count:
            raise CapabilityError("training input and target row counts differ")


def validate_candidate_capability_request(name: str) -> CandidateCapabilityName:
    """Accept only the small allowlist; raw history, alignment, and protected targets fail."""

    try:
        return CandidateCapabilityName(name)
    except ValueError as exc:
        raise CapabilityError(f"candidate capability is not allowed: {name!r}") from exc


def default_candidate_fields(phase: DataPhase) -> tuple[FieldKey, ...]:
    """Return the ordered organizer-FM-safe raw input slice for a phase."""

    try:
        log_member = _LOG_MEMBER_BY_PHASE[phase]
    except KeyError as exc:
        raise CapabilityError(f"unsupported data phase: {phase!r}") from exc
    return (
        FieldKey(log_member, "user_id"),
        FieldKey(log_member, "video_id"),
        FieldKey(VIDEO_BASIC_MEMBER, "author_id"),
        FieldKey(log_member, "tab"),
        FieldKey(log_member, "duration_ms"),
    )


def default_train_target_fields() -> tuple[FieldKey, ...]:
    """Return the initial primary-only target schema."""

    return (FieldKey(STANDARD_TRAIN_MEMBER, "long_view"),)


def _validate_available_keys(columns: Mapping[FieldKey, object]) -> None:
    # Deliberately iterate keys only.  Unrequested sidecar values, especially skipped final
    # outcomes, must never be fetched, converted, validated, hashed, or included in errors.
    for key in columns:
        if not isinstance(key, FieldKey):
            raise CapabilityError("capability mappings must use source-qualified FieldKey keys")
        try:
            field_spec(key)
        except FieldPolicyError as exc:
            raise CapabilityError(str(exc)) from exc


def _validate_requested_fields(requested_fields: Sequence[FieldKey]) -> tuple[FieldKey, ...]:
    requested = tuple(requested_fields)
    if not requested:
        raise CapabilityError("capability schema must contain at least one field")
    if any(not isinstance(key, FieldKey) for key in requested):
        raise CapabilityError("capability schema must use source-qualified FieldKey entries")
    if len(requested) != len(set(requested)):
        raise CapabilityError("capability schema contains a duplicate source-qualified field")
    output_names = [key.column for key in requested]
    if len(output_names) != len(set(output_names)):
        raise CapabilityError("capability schema contains duplicate candidate column names")
    reserved = sorted(set(output_names) & _RESERVED_CANDIDATE_NAMES)
    if reserved:
        raise CapabilityError(f"candidate schema contains trusted-only name(s): {reserved!r}")
    return requested


def _require_input_field(key: FieldKey, phase: DataPhase) -> LogicalDType:
    try:
        spec = field_spec(key)
    except FieldPolicyError as exc:
        raise CapabilityError(str(exc)) from exc
    allowed_log_member = _LOG_MEMBER_BY_PHASE[phase]
    if key.member not in {allowed_log_member, VIDEO_BASIC_MEMBER}:
        raise CapabilityError(
            f"field source {key.member!r} is not legal for {phase.value} candidate inputs"
        )
    if spec.role is not FieldRole.INFERENCE_INPUT:
        raise CapabilityError(
            f"field {key.wire_name} has role {spec.role.value}, not inference_input"
        )
    if not spec.enabled:
        raise CapabilityError(f"field {key.wire_name} is disabled by the current policy")
    return spec.logical_dtype


def _require_train_target(key: FieldKey, phase: DataPhase) -> LogicalDType:
    if phase not in {DataPhase.TRAIN, DataPhase.INNER_TRAIN}:
        raise CapabilityError("candidate training targets exist only for train or inner_train")
    try:
        spec = field_spec(key)
    except FieldPolicyError as exc:
        raise CapabilityError(str(exc)) from exc
    if key.member != STANDARD_TRAIN_MEMBER:
        raise CapabilityError("candidate training targets must come from the official train member")
    if spec.role not in {
        FieldRole.TRAINING_PRIMARY_TARGET,
        FieldRole.TRAINING_AUXILIARY_TARGET,
    }:
        raise CapabilityError(f"field {key.wire_name} is not a training target")
    if not spec.enabled:
        raise CapabilityError(f"training target {key.wire_name} is disabled by current policy")
    return spec.logical_dtype


def _as_1d_array(values: object, logical_dtype: LogicalDType, name: str) -> np.ndarray:
    raw = np.asarray(cast(npt.ArrayLike, values))
    if raw.ndim != 1:
        raise CapabilityError(f"column {name!r} must be one-dimensional, got shape {raw.shape!r}")

    storage_dtype = _STORAGE_DTYPES[logical_dtype]
    if logical_dtype is LogicalDType.INTEGER:
        if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
            raise CapabilityError(f"column {name!r} must contain exact integer values")
        try:
            converted = raw.astype(storage_dtype, copy=True, casting="safe")
        except TypeError as exc:
            raise CapabilityError(f"column {name!r} cannot be represented as int64") from exc
    elif logical_dtype is LogicalDType.NUMBER:
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            raise CapabilityError(f"column {name!r} must contain numeric values")
        converted = raw.astype(storage_dtype, copy=True)
        if any(not math.isfinite(float(value)) for value in converted):
            raise CapabilityError(f"column {name!r} must contain only finite values")
    elif logical_dtype is LogicalDType.BINARY:
        if raw.dtype.kind not in "biuf":
            raise CapabilityError(f"column {name!r} must contain binary scalar values")
        numeric = raw.astype(np.dtype("<f8"), copy=False)
        if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0) | (numeric == 1)):
            raise CapabilityError(f"column {name!r} must contain only 0 or 1")
        converted = numeric.astype(storage_dtype, copy=True)
    else:
        flat_values = raw.tolist()
        normalized: list[str] = []
        for value in flat_values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)):
                raise CapabilityError(
                    f"column {name!r} categorical values must be strings or integers"
                )
            text = str(value)
            if len(text) > 64 or "\x00" in text:
                raise CapabilityError(
                    f"column {name!r} categorical values must be at most 64 characters without NUL"
                )
            normalized.append(text)
        converted = np.asarray(normalized, dtype=storage_dtype)

    # An immutable bytes buffer prevents callers from re-enabling NumPy's WRITEABLE flag.
    immutable_buffer = converted.tobytes(order="C")
    frozen = np.frombuffer(immutable_buffer, dtype=converted.dtype).reshape(converted.shape)
    frozen.setflags(write=False)
    return frozen


def _schema_digest(schema: tuple[CapabilityColumn, ...]) -> str:
    payload = json.dumps(
        [column.manifest() for column in schema],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _content_digest(
    schema: tuple[CapabilityColumn, ...], columns: Mapping[str, npt.NDArray[np.generic]]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"kuairand-capability-content-v1\0")
    for column in schema:
        values = columns[column.name]
        digest.update(column.field_key.wire_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(values.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _capability_digest(
    *,
    capability_name: str,
    phase: DataPhase,
    row_count: int,
    schema: tuple[CapabilityColumn, ...],
    schema_digest: str,
    content_digest: str,
) -> str:
    manifest = {
        "schema_version": 1,
        "capability_name": capability_name,
        "phase": phase.value,
        "row_count": row_count,
        "columns": [column.manifest() for column in schema],
        "field_policy_digest": FIELD_POLICY_DIGEST,
        "capability_schema_digest": schema_digest,
        "logical_content_digest": content_digest,
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _materialize(
    *,
    columns: Mapping[FieldKey, object],
    requested_fields: tuple[FieldKey, ...],
    logical_dtypes: tuple[LogicalDType, ...],
) -> tuple[
    tuple[CapabilityColumn, ...],
    Mapping[str, npt.NDArray[np.generic]],
    int,
    str,
    str,
]:
    missing = [key.wire_name for key in requested_fields if key not in columns]
    if missing:
        raise CapabilityError(f"capability data is missing declared field(s): {missing!r}")

    schema: list[CapabilityColumn] = []
    materialized: dict[str, npt.NDArray[np.generic]] = {}
    row_count: int | None = None
    for key, logical_dtype in zip(requested_fields, logical_dtypes, strict=True):
        values = _as_1d_array(columns[key], logical_dtype, key.column)
        if row_count is None:
            row_count = len(values)
            if row_count == 0:
                raise CapabilityError("capability columns must contain at least one row")
        elif len(values) != row_count:
            raise CapabilityError(
                f"column {key.column!r} has length {len(values)}, expected {row_count}"
            )
        storage_dtype = values.dtype.str
        schema.append(CapabilityColumn(key.column, key, logical_dtype, storage_dtype))
        materialized[key.column] = values

    frozen_schema = tuple(schema)
    frozen_columns: Mapping[str, npt.NDArray[np.generic]] = MappingProxyType(materialized)
    schema_digest = _schema_digest(frozen_schema)
    content_digest = _content_digest(frozen_schema, frozen_columns)
    assert row_count is not None
    return frozen_schema, frozen_columns, row_count, schema_digest, content_digest


def build_candidate_inputs(
    phase: DataPhase,
    columns: Mapping[FieldKey, object],
    *,
    requested_fields: Sequence[FieldKey] | None = None,
) -> CandidateInputs:
    """Build one immutable input package without inspecting unrequested sidecars."""

    if not isinstance(phase, DataPhase):
        raise CapabilityError(f"unsupported data phase: {phase!r}")
    _validate_available_keys(columns)
    requested = _validate_requested_fields(
        default_candidate_fields(phase) if requested_fields is None else requested_fields
    )
    logical_dtypes = tuple(_require_input_field(key, phase) for key in requested)
    schema, frozen_columns, row_count, schema_digest, content_digest = _materialize(
        columns=columns,
        requested_fields=requested,
        logical_dtypes=logical_dtypes,
    )
    capability_name = _INPUT_CAPABILITY_BY_PHASE[phase]
    digest = _capability_digest(
        capability_name=capability_name.value,
        phase=phase,
        row_count=row_count,
        schema=schema,
        schema_digest=schema_digest,
        content_digest=content_digest,
    )
    return CandidateInputs(
        capability_name=capability_name,
        phase=phase,
        schema=schema,
        row_count=row_count,
        field_policy_digest=FIELD_POLICY_DIGEST,
        schema_digest=schema_digest,
        logical_content_digest=content_digest,
        digest=digest,
        _columns=frozen_columns,
    )


def build_train_targets(
    phase: DataPhase,
    columns: Mapping[FieldKey, object],
    *,
    requested_fields: Sequence[FieldKey] | None = None,
) -> TrainTargets:
    """Build primary/approved auxiliary targets only for official-train fitting phases."""

    if not isinstance(phase, DataPhase):
        raise CapabilityError(f"unsupported data phase: {phase!r}")
    _validate_available_keys(columns)
    requested = _validate_requested_fields(
        default_train_target_fields() if requested_fields is None else requested_fields
    )
    logical_dtypes = tuple(_require_train_target(key, phase) for key in requested)
    roles = tuple(field_spec(key).role for key in requested)
    if roles.count(FieldRole.TRAINING_PRIMARY_TARGET) != 1:
        raise CapabilityError("training target schema must contain exactly one primary target")
    if requested[0] != FieldKey(STANDARD_TRAIN_MEMBER, "long_view"):
        raise CapabilityError("long_view must be the first training target column")

    schema, frozen_columns, row_count, schema_digest, content_digest = _materialize(
        columns=columns,
        requested_fields=requested,
        logical_dtypes=logical_dtypes,
    )
    digest = _capability_digest(
        capability_name=CandidateCapabilityName.TRAIN_TARGETS.value,
        phase=phase,
        row_count=row_count,
        schema=schema,
        schema_digest=schema_digest,
        content_digest=content_digest,
    )
    return TrainTargets(
        phase=phase,
        schema=schema,
        row_count=row_count,
        field_policy_digest=FIELD_POLICY_DIGEST,
        schema_digest=schema_digest,
        logical_content_digest=content_digest,
        digest=digest,
        _columns=frozen_columns,
    )


def _canonical_input_columns(
    inputs: CanonicalInputs,
    *,
    log_member: str,
) -> Mapping[FieldKey, object]:
    """Project only candidate-approved canonical fields into the capability factory."""

    return MappingProxyType(
        {
            FieldKey(log_member, "user_id"): inputs.user_id,
            FieldKey(log_member, "video_id"): inputs.video_id,
            FieldKey(VIDEO_BASIC_MEMBER, "author_id"): inputs.author_id,
            FieldKey(log_member, "tab"): inputs.tab,
            FieldKey(log_member, "duration_ms"): inputs.duration_ms,
        }
    )


def build_training_capabilities(split: CanonicalTrainingSplit) -> TrainingCapabilities:
    """Build train inputs and targets from a structurally training-only split."""

    from kuairand_agent.data.canonical import CanonicalTrainingSplit as TrainingSplitType

    if not isinstance(split, TrainingSplitType):
        raise CapabilityError("training capabilities require CanonicalTrainingSplit")
    input_columns = _canonical_input_columns(split.inputs, log_member=STANDARD_TRAIN_MEMBER)
    target_columns = MappingProxyType(
        {FieldKey(STANDARD_TRAIN_MEMBER, "long_view"): split.targets.long_view}
    )
    return TrainingCapabilities(
        inputs=build_candidate_inputs(DataPhase.TRAIN, input_columns),
        targets=build_train_targets(DataPhase.TRAIN, target_columns),
    )


def build_validation_inputs(split: CanonicalValidationSplit) -> CandidateInputs:
    """Build validation inputs without touching the split's protected label capability."""

    from kuairand_agent.data.canonical import CanonicalValidationSplit as ValidationSplitType

    if not isinstance(split, ValidationSplitType):
        raise CapabilityError("validation inputs require CanonicalValidationSplit")
    return build_candidate_inputs(
        DataPhase.OUTER_VALID,
        _canonical_input_columns(split.inputs, log_member=STANDARD_LATE_MEMBER),
    )


def build_final_inputs(split: CanonicalFinalSplit) -> CandidateInputs:
    """Build final-period inputs from a type that cannot represent outcome labels."""

    from kuairand_agent.data.canonical import CanonicalFinalSplit as FinalSplitType

    if not isinstance(split, FinalSplitType):
        raise CapabilityError("final inputs require CanonicalFinalSplit")
    return build_candidate_inputs(
        DataPhase.FINAL,
        _canonical_input_columns(split.inputs, log_member=STANDARD_LATE_MEMBER),
    )
