"""Strict, canonical wire schemas for the research-model seam.

Provider responses are untrusted data.  Every public ``from_mapping`` constructor therefore
requires an exact field set and exact Python scalar types; permissive coercion would make a
recorded response mean something different on replay.  Canonical JSON bytes are also the sole
source of request and response identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Self, cast

SCHEMA_VERSION: Final = 1
MAX_TEXT_CHARS: Final = 16_384
MAX_DIAGNOSTIC_CHARS: Final = 32_768
MAX_SOURCE_CHARS: Final = 512 * 1024
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class SchemaValidationError(ValueError):
    """An untrusted request or response does not match its frozen schema."""


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_json_object(payload: str) -> Mapping[str, object]:
    """Parse one strict JSON object while rejecting duplicate member names and NaN."""

    if type(payload) is not str:
        raise SchemaValidationError("JSON payload must be text")
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SchemaValidationError(f"non-finite JSON number is forbidden: {value}")
            ),
        )
    except SchemaValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SchemaValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise SchemaValidationError("research response must be a JSON object")
    return cast(Mapping[str, object], decoded)


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic finite JSON bytes for persisted research records."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"value is not canonical JSON data: {exc}") from exc


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_fields(raw: Mapping[str, object], expected: set[str], location: str) -> None:
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown:
        raise SchemaValidationError(f"unknown {location} field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise SchemaValidationError(f"missing {location} field(s): {', '.join(sorted(missing))}")


def _schema_version(value: object, location: str) -> int:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise SchemaValidationError(f"{location}.schema_version must be {SCHEMA_VERSION}")
    return value


def _text(
    value: object,
    location: str,
    *,
    maximum: int = MAX_TEXT_CHARS,
    identifier: bool = False,
) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise SchemaValidationError(f"{location} must be non-empty text without NUL bytes")
    if len(value) > maximum:
        raise SchemaValidationError(f"{location} exceeds the {maximum}-character limit")
    if identifier and _IDENTIFIER_RE.fullmatch(value) is None:
        raise SchemaValidationError(f"{location} is not a portable identifier")
    return value


def _integer(value: object, location: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SchemaValidationError(f"{location} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{location} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{location} must be a finite real number")
    return result


def _string_tuple(
    value: object,
    location: str,
    *,
    maximum_items: int,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SchemaValidationError(f"{location} must be a non-empty JSON array")
    if len(value) > maximum_items:
        raise SchemaValidationError(f"{location} exceeds the {maximum_items}-item limit")
    result = tuple(_text(item, f"{location}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise SchemaValidationError(f"{location} contains duplicates")
    if allowed is not None and not set(result) <= allowed:
        raise SchemaValidationError(f"{location} contains an unsupported value")
    return result


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise SchemaValidationError(f"{location} must be a JSON object")
    return cast(Mapping[str, object], value)


@dataclass(frozen=True, slots=True)
class RequiredField:
    """One source-qualified field requested by a proposal for one declared role."""

    source_field: str
    role: str
    purpose: str

    def __post_init__(self) -> None:
        _text(self.source_field, "required_field.source_field")
        _text(self.role, "required_field.role", identifier=True)
        _text(self.purpose, "required_field.purpose")

    def to_wire(self) -> dict[str, str]:
        return {
            "source_field": self.source_field,
            "role": self.role,
            "purpose": self.purpose,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _exact_fields(raw, {"source_field", "role", "purpose"}, "required_field")
        return cls(
            source_field=_text(raw["source_field"], "required_field.source_field"),
            role=_text(raw["role"], "required_field.role", identifier=True),
            purpose=_text(raw["purpose"], "required_field.purpose"),
        )


@dataclass(frozen=True, slots=True)
class Proposal:
    """One falsifiable, atomic scientific change returned by a research model."""

    proposal_id: str
    hypothesis: str
    mechanism: str
    expected_metric_effects: tuple[str, ...]
    parent_candidate_id: str
    principal_change: str
    files_expected: tuple[str, ...]
    required_fields: tuple[RequiredField, ...]
    objective: str
    sampling: str
    grouping: str
    weighting: str
    causal_cutoff: str
    estimated_runtime_seconds: int
    estimated_memory_mb: int
    smoke_plan: str
    inner_fold_plan: str
    falsification_criteria: str
    promotion_criteria: str
    maximum_repairs: int
    rollback_parent_id: str
    attributions: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "proposal")
        _text(self.proposal_id, "proposal.proposal_id", identifier=True)
        for field_name in (
            "hypothesis",
            "mechanism",
            "principal_change",
            "objective",
            "sampling",
            "grouping",
            "weighting",
            "causal_cutoff",
            "smoke_plan",
            "inner_fold_plan",
            "falsification_criteria",
            "promotion_criteria",
        ):
            _text(getattr(self, field_name), f"proposal.{field_name}")
        _text(self.parent_candidate_id, "proposal.parent_candidate_id", identifier=True)
        _text(self.rollback_parent_id, "proposal.rollback_parent_id", identifier=True)
        if type(self.expected_metric_effects) is not tuple or not self.expected_metric_effects:
            raise SchemaValidationError("proposal.expected_metric_effects must be a tuple")
        if not set(self.expected_metric_effects) <= {
            "GAUC",
            "nDCG@5",
        }:
            raise SchemaValidationError(
                "proposal.expected_metric_effects must contain GAUC and/or nDCG@5"
            )
        if len(self.expected_metric_effects) != len(set(self.expected_metric_effects)):
            raise SchemaValidationError("proposal.expected_metric_effects contains duplicates")
        if type(self.files_expected) is not tuple or not self.files_expected:
            raise SchemaValidationError("proposal.files_expected must be a non-empty tuple")
        for index, value in enumerate(self.files_expected):
            _text(value, f"proposal.files_expected[{index}]")
        if len(self.files_expected) != len(set(self.files_expected)):
            raise SchemaValidationError("proposal.files_expected must be non-empty and unique")
        if type(self.required_fields) is not tuple or not self.required_fields:
            raise SchemaValidationError("proposal.required_fields cannot be empty")
        if any(not isinstance(value, RequiredField) for value in self.required_fields):
            raise SchemaValidationError("proposal.required_fields contains an invalid record")
        if type(self.attributions) is not tuple or not self.attributions:
            raise SchemaValidationError("proposal.attributions cannot be empty")
        for index, value in enumerate(self.attributions):
            _text(value, f"proposal.attributions[{index}]")
        _integer(
            self.estimated_runtime_seconds,
            "proposal.estimated_runtime_seconds",
            minimum=1,
            maximum=21_600,
        )
        _integer(
            self.estimated_memory_mb,
            "proposal.estimated_memory_mb",
            minimum=1,
            maximum=262_144,
        )
        _integer(self.maximum_repairs, "proposal.maximum_repairs", minimum=0, maximum=2)

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "expected_metric_effects": list(self.expected_metric_effects),
            "parent_candidate_id": self.parent_candidate_id,
            "principal_change": self.principal_change,
            "files_expected": list(self.files_expected),
            "required_fields": [value.to_wire() for value in self.required_fields],
            "objective": self.objective,
            "sampling": self.sampling,
            "grouping": self.grouping,
            "weighting": self.weighting,
            "causal_cutoff": self.causal_cutoff,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "estimated_memory_mb": self.estimated_memory_mb,
            "smoke_plan": self.smoke_plan,
            "inner_fold_plan": self.inner_fold_plan,
            "falsification_criteria": self.falsification_criteria,
            "promotion_criteria": self.promotion_criteria,
            "maximum_repairs": self.maximum_repairs,
            "rollback_parent_id": self.rollback_parent_id,
            "attributions": list(self.attributions),
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_wire()).decode("ascii")

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.from_mapping(parse_json_object(payload))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        expected = {
            "schema_version",
            "proposal_id",
            "hypothesis",
            "mechanism",
            "expected_metric_effects",
            "parent_candidate_id",
            "principal_change",
            "files_expected",
            "required_fields",
            "objective",
            "sampling",
            "grouping",
            "weighting",
            "causal_cutoff",
            "estimated_runtime_seconds",
            "estimated_memory_mb",
            "smoke_plan",
            "inner_fold_plan",
            "falsification_criteria",
            "promotion_criteria",
            "maximum_repairs",
            "rollback_parent_id",
            "attributions",
        }
        _exact_fields(raw, expected, "proposal")
        required_raw = raw["required_fields"]
        if not isinstance(required_raw, list) or not required_raw or len(required_raw) > 64:
            raise SchemaValidationError(
                "proposal.required_fields must be a non-empty array with at most 64 entries"
            )
        required = tuple(
            RequiredField.from_mapping(_mapping(value, f"proposal.required_fields[{index}]"))
            for index, value in enumerate(required_raw)
        )
        return cls(
            schema_version=_schema_version(raw["schema_version"], "proposal"),
            proposal_id=_text(raw["proposal_id"], "proposal.proposal_id", identifier=True),
            hypothesis=_text(raw["hypothesis"], "proposal.hypothesis"),
            mechanism=_text(raw["mechanism"], "proposal.mechanism"),
            expected_metric_effects=_string_tuple(
                raw["expected_metric_effects"],
                "proposal.expected_metric_effects",
                maximum_items=2,
                allowed=frozenset({"GAUC", "nDCG@5"}),
            ),
            parent_candidate_id=_text(
                raw["parent_candidate_id"], "proposal.parent_candidate_id", identifier=True
            ),
            principal_change=_text(raw["principal_change"], "proposal.principal_change"),
            files_expected=_string_tuple(
                raw["files_expected"], "proposal.files_expected", maximum_items=12
            ),
            required_fields=required,
            objective=_text(raw["objective"], "proposal.objective"),
            sampling=_text(raw["sampling"], "proposal.sampling"),
            grouping=_text(raw["grouping"], "proposal.grouping"),
            weighting=_text(raw["weighting"], "proposal.weighting"),
            causal_cutoff=_text(raw["causal_cutoff"], "proposal.causal_cutoff"),
            estimated_runtime_seconds=_integer(
                raw["estimated_runtime_seconds"],
                "proposal.estimated_runtime_seconds",
                minimum=1,
                maximum=21_600,
            ),
            estimated_memory_mb=_integer(
                raw["estimated_memory_mb"],
                "proposal.estimated_memory_mb",
                minimum=1,
                maximum=262_144,
            ),
            smoke_plan=_text(raw["smoke_plan"], "proposal.smoke_plan"),
            inner_fold_plan=_text(raw["inner_fold_plan"], "proposal.inner_fold_plan"),
            falsification_criteria=_text(
                raw["falsification_criteria"], "proposal.falsification_criteria"
            ),
            promotion_criteria=_text(raw["promotion_criteria"], "proposal.promotion_criteria"),
            maximum_repairs=_integer(
                raw["maximum_repairs"], "proposal.maximum_repairs", minimum=0, maximum=2
            ),
            rollback_parent_id=_text(
                raw["rollback_parent_id"], "proposal.rollback_parent_id", identifier=True
            ),
            attributions=_string_tuple(
                raw["attributions"], "proposal.attributions", maximum_items=16
            ),
        )


def _source_text(value: object, location: str) -> str:
    if type(value) is not str or "\x00" in value:
        raise SchemaValidationError(f"{location} must be UTF-8 text without NUL bytes")
    if len(value) > MAX_SOURCE_CHARS:
        raise SchemaValidationError(f"{location} exceeds the generated-source character limit")
    # Python strings are Unicode already; the encode check catches isolated surrogate code
    # points, which cannot be written as complete UTF-8 source.
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise SchemaValidationError(f"{location} is not valid UTF-8 text") from exc
    return value


@dataclass(frozen=True, slots=True)
class ParentSourceFile:
    """One exact, content-addressed file supplied to an implementation request."""

    path: str
    sha256: str
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.path, "parent_file.path")
        content = _source_text(self.content, f"parent_file[{self.path!r}].content")
        if type(self.sha256) is not str or _SHA256_RE.fullmatch(self.sha256) is None:
            raise SchemaValidationError("parent_file.sha256 must be a lowercase SHA-256 digest")
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise SchemaValidationError(f"parent file digest mismatch for {self.path!r}")

    @classmethod
    def create(cls, path: str, content: str) -> Self:
        validated = _source_text(content, f"parent_file[{path!r}].content")
        return cls(
            path=path,
            sha256=hashlib.sha256(validated.encode("utf-8")).hexdigest(),
            content=validated,
        )

    def to_wire(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "content": self.content}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _exact_fields(raw, {"path", "sha256", "content"}, "parent_file")
        return cls(
            path=_text(raw["path"], "parent_file.path"),
            sha256=_text(raw["sha256"], "parent_file.sha256"),
            content=_source_text(raw["content"], "parent_file.content"),
        )


@dataclass(frozen=True, slots=True)
class ParentSnapshot:
    """Immutable logical parent source; its identity excludes filesystem location and metadata."""

    candidate_id: str
    files: tuple[ParentSourceFile, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.candidate_id, "parent_snapshot.candidate_id", identifier=True)
        if type(self.files) is not tuple or not self.files or len(self.files) > 64:
            raise SchemaValidationError("parent snapshot must contain between 1 and 64 files")
        if any(not isinstance(value, ParentSourceFile) for value in self.files):
            raise SchemaValidationError("parent snapshot contains an invalid file record")
        ordered = tuple(sorted(self.files, key=lambda value: value.path))
        paths = tuple(value.path for value in ordered)
        if len(paths) != len(set(paths)):
            raise SchemaValidationError("parent snapshot contains duplicate paths")
        object.__setattr__(self, "files", ordered)
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": self.candidate_id,
                    "files": [value.to_wire() for value in ordered],
                }
            ),
        )

    def file(self, path: str) -> ParentSourceFile:
        for value in self.files:
            if value.path == path:
                return value
        raise SchemaValidationError(f"parent snapshot does not contain {path!r}")

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "files": [value.to_wire() for value in self.files],
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class GeneratedFile:
    """One provider-generated complete file, never a patch or filesystem reference."""

    path: str
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.path, "generated_file.path")
        _source_text(self.content, f"generated_file[{self.path!r}].content")

    def to_wire(self) -> dict[str, str]:
        return {"path": self.path, "content": self.content}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _exact_fields(raw, {"path", "content"}, "generated_file")
        return cls(
            path=_text(raw["path"], "generated_file.path"),
            content=_source_text(raw["content"], "generated_file.content"),
        )


@dataclass(frozen=True, slots=True)
class GeneratedPackage:
    """Bounded complete-file implementation response from a research model."""

    request_id: str
    response_id: str
    files: tuple[GeneratedFile, ...]
    material_change_summary: str
    material_symbols: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "generated_package")
        _text(self.request_id, "generated_package.request_id", identifier=True)
        _text(self.response_id, "generated_package.response_id", identifier=True)
        _text(self.material_change_summary, "generated_package.material_change_summary")
        if type(self.files) is not tuple or not self.files or len(self.files) > 12:
            raise SchemaValidationError("generated package must contain between 1 and 12 files")
        if any(not isinstance(value, GeneratedFile) for value in self.files):
            raise SchemaValidationError("generated package contains an invalid file record")
        ordered = tuple(sorted(self.files, key=lambda value: value.path))
        paths = tuple(value.path for value in ordered)
        if len(paths) != len(set(paths)):
            raise SchemaValidationError("generated package contains duplicate paths")
        object.__setattr__(self, "files", ordered)
        if (
            type(self.material_symbols) is not tuple
            or not self.material_symbols
            or len(self.material_symbols) > 32
        ):
            raise SchemaValidationError(
                "generated package must declare between 1 and 32 material symbols"
            )
        for index, value in enumerate(self.material_symbols):
            _text(value, f"generated_package.material_symbols[{index}]", identifier=True)
        if len(self.material_symbols) != len(set(self.material_symbols)):
            raise SchemaValidationError("generated package material symbols contain duplicates")

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "files": [value.to_wire() for value in self.files],
            "material_change_summary": self.material_change_summary,
            "material_symbols": list(self.material_symbols),
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_wire()).decode("ascii")

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.from_mapping(parse_json_object(payload))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        expected = {
            "schema_version",
            "request_id",
            "response_id",
            "files",
            "material_change_summary",
            "material_symbols",
        }
        _exact_fields(raw, expected, "generated_package")
        files_raw = raw["files"]
        if not isinstance(files_raw, list) or not files_raw or len(files_raw) > 12:
            raise SchemaValidationError(
                "generated_package.files must contain between 1 and 12 entries"
            )
        files = tuple(
            GeneratedFile.from_mapping(_mapping(value, f"generated_package.files[{index}]"))
            for index, value in enumerate(files_raw)
        )
        return cls(
            schema_version=_schema_version(raw["schema_version"], "generated_package"),
            request_id=_text(raw["request_id"], "generated_package.request_id", identifier=True),
            response_id=_text(raw["response_id"], "generated_package.response_id", identifier=True),
            files=files,
            material_change_summary=_text(
                raw["material_change_summary"],
                "generated_package.material_change_summary",
            ),
            material_symbols=_string_tuple(
                raw["material_symbols"],
                "generated_package.material_symbols",
                maximum_items=32,
            ),
        )


class ResearchOperation(StrEnum):
    PROPOSE = "propose"
    IMPLEMENT = "implement"
    REPAIR = "repair"
    REFLECT = "reflect"


class FailureCategory(StrEnum):
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    STATIC_POLICY = "static_policy"
    SMOKE_FAILURE = "smoke_failure"
    TRAINING_FAILURE = "training_failure"
    OUTPUT_CONTRACT = "output_contract"


def _safe_context(value: Mapping[str, object]) -> tuple[Mapping[str, object], str]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise SchemaValidationError("safe_context must be a JSON object")
    encoded = canonical_json_bytes(dict(value))
    if len(encoded) > 512 * 1024:
        raise SchemaValidationError("safe_context exceeds the request byte limit")
    copied = cast(dict[str, object], json.loads(encoded))
    return MappingProxyType(copied), hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    request_id: str
    campaign_id: str
    scientific_iteration: int
    parent_candidate_id: str
    safe_context: Mapping[str, object] = field(repr=False)
    safe_context_digest: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "proposal_request")
        _text(self.request_id, "proposal_request.request_id", identifier=True)
        _text(self.campaign_id, "proposal_request.campaign_id", identifier=True)
        _text(
            self.parent_candidate_id,
            "proposal_request.parent_candidate_id",
            identifier=True,
        )
        _integer(
            self.scientific_iteration,
            "proposal_request.scientific_iteration",
            minimum=1,
            maximum=50,
        )
        context, digest = _safe_context(self.safe_context)
        if self.safe_context_digest != digest:
            raise SchemaValidationError("proposal_request safe-context digest mismatch")
        object.__setattr__(self, "safe_context", context)

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        campaign_id: str,
        scientific_iteration: int,
        parent_candidate_id: str,
        safe_context: Mapping[str, object],
    ) -> Self:
        context, digest = _safe_context(safe_context)
        return cls(
            request_id=request_id,
            campaign_id=campaign_id,
            scientific_iteration=scientific_iteration,
            parent_candidate_id=parent_candidate_id,
            safe_context=context,
            safe_context_digest=digest,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "campaign_id": self.campaign_id,
            "scientific_iteration": self.scientific_iteration,
            "parent_candidate_id": self.parent_candidate_id,
            "safe_context": dict(self.safe_context),
            "safe_context_digest": self.safe_context_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


@dataclass(frozen=True, slots=True)
class ImplementationRequest:
    request_id: str
    proposal: Proposal
    parent: ParentSnapshot
    safe_context: Mapping[str, object] = field(repr=False)
    safe_context_digest: str
    max_changed_files: int = 12
    max_total_utf8_bytes: int = 1024 * 1024
    allowed_suffixes: tuple[str, ...] = (".json", ".md", ".py")
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "implementation_request")
        _text(self.request_id, "implementation_request.request_id", identifier=True)
        if not isinstance(self.proposal, Proposal) or not isinstance(self.parent, ParentSnapshot):
            raise SchemaValidationError("implementation request requires typed proposal and parent")
        if self.proposal.parent_candidate_id != self.parent.candidate_id:
            raise SchemaValidationError("proposal parent does not match implementation parent")
        _integer(
            self.max_changed_files,
            "implementation_request.max_changed_files",
            minimum=1,
            maximum=12,
        )
        _integer(
            self.max_total_utf8_bytes,
            "implementation_request.max_total_utf8_bytes",
            minimum=1,
            maximum=1024 * 1024,
        )
        if self.allowed_suffixes != (".json", ".md", ".py"):
            raise SchemaValidationError("implementation request suffix allowlist is frozen")
        context, digest = _safe_context(self.safe_context)
        if self.safe_context_digest != digest:
            raise SchemaValidationError("implementation_request safe-context digest mismatch")
        object.__setattr__(self, "safe_context", context)

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        proposal: Proposal,
        parent: ParentSnapshot,
        safe_context: Mapping[str, object],
    ) -> Self:
        context, digest = _safe_context(safe_context)
        return cls(
            request_id=request_id,
            proposal=proposal,
            parent=parent,
            safe_context=context,
            safe_context_digest=digest,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "proposal": self.proposal.to_wire(),
            "parent": self.parent.to_wire(),
            "safe_context": dict(self.safe_context),
            "safe_context_digest": self.safe_context_digest,
            "limits": {
                "max_changed_files": self.max_changed_files,
                "max_total_utf8_bytes": self.max_total_utf8_bytes,
                "allowed_suffixes": list(self.allowed_suffixes),
            },
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


@dataclass(frozen=True, slots=True)
class ExperimentResultSummary:
    tier: str
    status: str
    gauc: float
    ndcg_at_5: float
    primary: float
    runtime_seconds: float
    peak_memory_mb: float

    def __post_init__(self) -> None:
        _text(self.tier, "experiment_result.tier", identifier=True)
        if self.tier not in {"fixture", "inner", "outer"}:
            raise SchemaValidationError("experiment_result.tier is unsupported")
        _text(self.status, "experiment_result.status", identifier=True)
        if self.status not in {"passed", "rejected", "promoted"}:
            raise SchemaValidationError("experiment_result.status is unsupported")
        for name, value in (
            ("gauc", self.gauc),
            ("ndcg_at_5", self.ndcg_at_5),
            ("primary", self.primary),
        ):
            metric = _finite_number(value, f"experiment_result.{name}")
            if not 0.0 <= metric <= 1.0:
                raise SchemaValidationError(f"experiment_result.{name} must be in [0, 1]")
        if abs(self.primary - (self.gauc + self.ndcg_at_5) / 2.0) > 1e-12:
            raise SchemaValidationError("experiment_result.primary must be the metric mean")
        for name, value in (
            ("runtime_seconds", self.runtime_seconds),
            ("peak_memory_mb", self.peak_memory_mb),
        ):
            if _finite_number(value, f"experiment_result.{name}") < 0:
                raise SchemaValidationError(f"experiment_result.{name} cannot be negative")

    def to_wire(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "status": self.status,
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass(frozen=True, slots=True)
class ReflectionRequest:
    request_id: str
    proposal_id: str
    candidate_id: str
    source_digest: str
    diff_digest: str
    result: ExperimentResultSummary
    safe_context: Mapping[str, object] = field(repr=False)
    safe_context_digest: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "reflection_request")
        for name in ("request_id", "proposal_id", "candidate_id"):
            _text(getattr(self, name), f"reflection_request.{name}", identifier=True)
        for name in ("source_digest", "diff_digest"):
            value = getattr(self, name)
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise SchemaValidationError(f"reflection_request.{name} must be SHA-256")
        if not isinstance(self.result, ExperimentResultSummary):
            raise SchemaValidationError("reflection request requires ExperimentResultSummary")
        context, digest = _safe_context(self.safe_context)
        if self.safe_context_digest != digest:
            raise SchemaValidationError("reflection_request safe-context digest mismatch")
        object.__setattr__(self, "safe_context", context)

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        proposal_id: str,
        candidate_id: str,
        source_digest: str,
        diff_digest: str,
        result: ExperimentResultSummary,
        safe_context: Mapping[str, object],
    ) -> Self:
        context, digest = _safe_context(safe_context)
        return cls(
            request_id=request_id,
            proposal_id=proposal_id,
            candidate_id=candidate_id,
            source_digest=source_digest,
            diff_digest=diff_digest,
            result=result,
            safe_context=context,
            safe_context_digest=digest,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "source_digest": self.source_digest,
            "diff_digest": self.diff_digest,
            "result": self.result.to_wire(),
            "safe_context": dict(self.safe_context),
            "safe_context_digest": self.safe_context_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


@dataclass(frozen=True, slots=True)
class RepairRequest:
    request_id: str
    proposal_id: str
    failed_candidate_id: str
    failed_child: ParentSnapshot
    failure_category: FailureCategory
    diagnostics: str
    remaining_repairs: int
    safe_context: Mapping[str, object] = field(repr=False)
    safe_context_digest: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "repair_request")
        for name in ("request_id", "proposal_id", "failed_candidate_id"):
            _text(getattr(self, name), f"repair_request.{name}", identifier=True)
        if not isinstance(self.failed_child, ParentSnapshot):
            raise SchemaValidationError("repair request requires the exact failed child snapshot")
        if not isinstance(self.failure_category, FailureCategory):
            raise SchemaValidationError("repair request failure_category is unsupported")
        _text(
            self.diagnostics,
            "repair_request.diagnostics",
            maximum=MAX_DIAGNOSTIC_CHARS,
        )
        _integer(
            self.remaining_repairs,
            "repair_request.remaining_repairs",
            minimum=1,
            maximum=2,
        )
        context, digest = _safe_context(self.safe_context)
        if self.safe_context_digest != digest:
            raise SchemaValidationError("repair_request safe-context digest mismatch")
        object.__setattr__(self, "safe_context", context)

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        proposal_id: str,
        failed_candidate_id: str,
        failed_child: ParentSnapshot,
        failure_category: str | FailureCategory,
        diagnostics: str,
        remaining_repairs: int,
        safe_context: Mapping[str, object],
    ) -> Self:
        try:
            category = FailureCategory(failure_category)
        except ValueError as exc:
            raise SchemaValidationError("repair request failure_category is unsupported") from exc
        context, digest = _safe_context(safe_context)
        return cls(
            request_id=request_id,
            proposal_id=proposal_id,
            failed_candidate_id=failed_candidate_id,
            failed_child=failed_child,
            failure_category=category,
            diagnostics=diagnostics,
            remaining_repairs=remaining_repairs,
            safe_context=context,
            safe_context_digest=digest,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "proposal_id": self.proposal_id,
            "failed_candidate_id": self.failed_candidate_id,
            "failed_child": self.failed_child.to_wire(),
            "failure_category": self.failure_category.value,
            "diagnostics": self.diagnostics,
            "remaining_repairs": self.remaining_repairs,
            "safe_context": dict(self.safe_context),
            "safe_context_digest": self.safe_context_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


@dataclass(frozen=True, slots=True)
class Reflection:
    response_id: str
    summary: str
    recommendation: str
    lessons: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "reflection")
        _text(self.response_id, "reflection.response_id", identifier=True)
        _text(self.summary, "reflection.summary")
        if self.recommendation not in {
            "close_branch",
            "retain_specialist",
            "propose_next",
        }:
            raise SchemaValidationError("reflection.recommendation is unsupported")
        if type(self.lessons) is not tuple or not self.lessons or len(self.lessons) > 16:
            raise SchemaValidationError("reflection.lessons must contain between 1 and 16 entries")
        for index, lesson in enumerate(self.lessons):
            _text(lesson, f"reflection.lessons[{index}]")

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "response_id": self.response_id,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "lessons": list(self.lessons),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())

    @classmethod
    def from_json(cls, payload: str) -> Self:
        return cls.from_mapping(parse_json_object(payload))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _exact_fields(
            raw,
            {"schema_version", "response_id", "summary", "recommendation", "lessons"},
            "reflection",
        )
        return cls(
            schema_version=_schema_version(raw["schema_version"], "reflection"),
            response_id=_text(raw["response_id"], "reflection.response_id", identifier=True),
            summary=_text(raw["summary"], "reflection.summary"),
            recommendation=_text(raw["recommendation"], "reflection.recommendation"),
            lessons=_string_tuple(raw["lessons"], "reflection.lessons", maximum_items=16),
        )


def _strict_object_schema(title: str, properties: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


_TEXT_SCHEMA: Final[dict[str, Any]] = {
    "type": "string",
    "minLength": 1,
    "maxLength": MAX_TEXT_CHARS,
}
_IDENTIFIER_SCHEMA: Final[dict[str, Any]] = {
    "type": "string",
    "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
}
_REQUIRED_FIELD_SCHEMA: Final = _strict_object_schema(
    "RequiredField",
    {
        "source_field": _TEXT_SCHEMA,
        "role": _IDENTIFIER_SCHEMA,
        "purpose": _TEXT_SCHEMA,
    },
)
_PROPOSAL_SCHEMA: Final = _strict_object_schema(
    "Proposal",
    {
        "schema_version": {"type": "integer", "const": SCHEMA_VERSION},
        "proposal_id": _IDENTIFIER_SCHEMA,
        "hypothesis": _TEXT_SCHEMA,
        "mechanism": _TEXT_SCHEMA,
        "expected_metric_effects": {
            "type": "array",
            "items": {"type": "string", "enum": ["GAUC", "nDCG@5"]},
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
        },
        "parent_candidate_id": _IDENTIFIER_SCHEMA,
        "principal_change": _TEXT_SCHEMA,
        "files_expected": {
            "type": "array",
            "items": _TEXT_SCHEMA,
            "minItems": 1,
            "maxItems": 12,
            "uniqueItems": True,
        },
        "required_fields": {
            "type": "array",
            "items": _REQUIRED_FIELD_SCHEMA,
            "minItems": 1,
            "maxItems": 64,
        },
        "objective": _TEXT_SCHEMA,
        "sampling": _TEXT_SCHEMA,
        "grouping": _TEXT_SCHEMA,
        "weighting": _TEXT_SCHEMA,
        "causal_cutoff": _TEXT_SCHEMA,
        "estimated_runtime_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 21_600,
        },
        "estimated_memory_mb": {
            "type": "integer",
            "minimum": 1,
            "maximum": 262_144,
        },
        "smoke_plan": _TEXT_SCHEMA,
        "inner_fold_plan": _TEXT_SCHEMA,
        "falsification_criteria": _TEXT_SCHEMA,
        "promotion_criteria": _TEXT_SCHEMA,
        "maximum_repairs": {"type": "integer", "minimum": 0, "maximum": 2},
        "rollback_parent_id": _IDENTIFIER_SCHEMA,
        "attributions": {
            "type": "array",
            "items": _TEXT_SCHEMA,
            "minItems": 1,
            "maxItems": 16,
            "uniqueItems": True,
        },
    },
)
_GENERATED_FILE_SCHEMA: Final = _strict_object_schema(
    "GeneratedFile",
    {
        "path": _TEXT_SCHEMA,
        "content": {"type": "string", "maxLength": MAX_SOURCE_CHARS},
    },
)
_GENERATED_PACKAGE_SCHEMA: Final = _strict_object_schema(
    "GeneratedPackage",
    {
        "schema_version": {"type": "integer", "const": SCHEMA_VERSION},
        "request_id": _IDENTIFIER_SCHEMA,
        "response_id": _IDENTIFIER_SCHEMA,
        "files": {
            "type": "array",
            "items": _GENERATED_FILE_SCHEMA,
            "minItems": 1,
            "maxItems": 12,
        },
        "material_change_summary": _TEXT_SCHEMA,
        "material_symbols": {
            "type": "array",
            "items": _IDENTIFIER_SCHEMA,
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
        },
    },
)
_REFLECTION_SCHEMA: Final = _strict_object_schema(
    "Reflection",
    {
        "schema_version": {"type": "integer", "const": SCHEMA_VERSION},
        "response_id": _IDENTIFIER_SCHEMA,
        "summary": _TEXT_SCHEMA,
        "recommendation": {
            "type": "string",
            "enum": ["close_branch", "retain_specialist", "propose_next"],
        },
        "lessons": {
            "type": "array",
            "items": _TEXT_SCHEMA,
            "minItems": 1,
            "maxItems": 16,
        },
    },
)


def response_json_schema(operation: ResearchOperation) -> dict[str, Any]:
    """Return a defensive copy of the strict provider response schema for one operation."""

    if not isinstance(operation, ResearchOperation):
        raise SchemaValidationError("research operation is unsupported")
    if operation is ResearchOperation.PROPOSE:
        schema = _PROPOSAL_SCHEMA
    elif operation in {ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR}:
        schema = _GENERATED_PACKAGE_SCHEMA
    else:
        schema = _REFLECTION_SCHEMA
    return cast(dict[str, Any], json.loads(canonical_json_bytes(schema)))
