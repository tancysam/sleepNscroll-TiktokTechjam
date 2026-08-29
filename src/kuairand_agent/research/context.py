"""Leakage-safe aggregate context exposed to a research model.

The builder deliberately accepts only value-free capability manifests and scalar aggregate
records.  Benchmark and field-policy metadata are sourced inside the trusted process instead of
being caller-supplied mappings, keeping row-level labels and predictions out of the interface.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Final, cast

from kuairand_agent.contract import benchmark_digest, benchmark_manifest
from kuairand_agent.data.fields import (
    FieldKey,
    FieldPolicyError,
    FieldRole,
    field_policy_digest,
    field_policy_manifest,
    field_spec,
)
from kuairand_agent.research.schemas import canonical_digest, canonical_json_bytes

type AggregateScalar = str | int | float | bool | None

_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_BEARER_RE: Final = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+")
_ASSIGNMENT_SECRET_RE: Final = re.compile(
    r"(?i)((?:[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD))\s*=\s*)[^\s]+"
)
_PROTECTED_AGGREGATE_KEY_PARTS: Final = (
    "public_validation_label",
    "outer_valid_label",
    "final_label",
    "final_outcome",
    "residual",
    "worst_user",
    "row_id",
    "label_prediction",
    "prediction_label",
)
_CAPABILITY_KEYS: Final = {
    "schema_version",
    "capability_name",
    "phase",
    "row_count",
    "columns",
    "field_policy_digest",
    "capability_schema_digest",
    "logical_content_digest",
    "capability_digest",
}
_CAPABILITY_COLUMN_KEYS: Final = {
    "name",
    "source_field",
    "logical_dtype",
    "storage_dtype",
}
_CAPABILITY_PHASES: Final = {
    "train_inputs": frozenset({"train", "inner_train"}),
    "inner_valid_inputs": frozenset({"inner_valid"}),
    "outer_valid_inputs": frozenset({"outer_valid"}),
    "final_inputs": frozenset({"final"}),
}


def _provider_field_policy_manifest() -> dict[str, object]:
    """Compact the frozen policy for inference without changing its authority or digest."""

    full = field_policy_manifest()
    raw_fields = full["fields"]
    if not isinstance(raw_fields, list):
        raise SafeContextError("trusted field policy fields must be a list")
    enabled: list[dict[str, object]] = []
    disabled_by_member: dict[str, list[str]] = {}
    for index, raw_field in enumerate(raw_fields):
        if not isinstance(raw_field, dict):
            raise SafeContextError(f"trusted field policy field {index} must be an object")
        if raw_field.get("enabled") is True:
            enabled.append(dict(raw_field))
            continue
        member = raw_field.get("member")
        column = raw_field.get("column")
        if type(member) is not str or type(column) is not str:
            raise SafeContextError(f"trusted field policy field {index} has invalid identity")
        disabled_by_member.setdefault(member, []).append(column)
    return {
        "schema_version": full["schema_version"],
        "policy_semantics": (
            "fields contains every enabled source-qualified field with complete enforcement "
            "metadata; disabled_columns_by_member is exhaustive; every unspecified field is "
            "forbidden"
        ),
        "fields": enabled,
        "disabled_columns_by_member": disabled_by_member,
        "role_counts": full["role_counts"],
    }


class SafeContextError(ValueError):
    """A proposed research-context value could expose protected or row-level data."""


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise SafeContextError(f"{location} must be non-empty text without NUL bytes")
    if len(value) > 16_384:
        raise SafeContextError(f"{location} exceeds the context text limit")
    return value


def _digest(value: object, location: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise SafeContextError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _scalar(value: object, location: str) -> AggregateScalar:
    if value is None or type(value) in {str, int, bool}:
        if isinstance(value, str) and ("\x00" in value or len(value) > 16_384):
            raise SafeContextError(f"{location} contains invalid aggregate text")
        return cast(AggregateScalar, value)
    if type(value) is float and math.isfinite(value):
        return value
    raise SafeContextError(f"{location} must be an aggregate scalar, never a row-level array")


def redact_text(value: str, *, secrets: Sequence[str] = ()) -> str:
    """Redact explicit secret values and common credential-bearing text forms."""

    if type(value) is not str:
        raise SafeContextError("redaction input must be text")
    result = value
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            result = result.replace(secret, "[REDACTED]")
    result = _BEARER_RE.sub(r"\1[REDACTED]", result)
    return _ASSIGNMENT_SECRET_RE.sub(r"\1[REDACTED]", result)


def _redact_json(value: object, secrets: Sequence[str]) -> object:
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    if isinstance(value, list):
        return [_redact_json(item, secrets) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redact_json(item, secrets) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class AggregateRecord:
    """One named aggregate summary with scalar values only."""

    name: str
    values: Mapping[str, AggregateScalar]

    def __post_init__(self) -> None:
        _text(self.name, "aggregate.name")
        if not isinstance(self.values, Mapping) or not self.values:
            raise SafeContextError("aggregate.values must be a non-empty object")
        validated: dict[str, AggregateScalar] = {}
        for raw_key, raw_value in self.values.items():
            key = _text(raw_key, "aggregate key")
            value = _scalar(raw_value, f"aggregate.{key}")
            normalized = key.casefold()
            if any(part in normalized for part in _PROTECTED_AGGREGATE_KEY_PARTS):
                raise SafeContextError(
                    f"aggregate key {key!r} names row-level or protected information"
                )
            validated[key] = value
        object.__setattr__(self, "values", MappingProxyType(validated))

    def to_wire(self) -> dict[str, object]:
        return {"name": self.name, "values": dict(self.values)}


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate organizer metrics, exact for inner folds and rounded for outer context."""

    name: str
    gauc: float
    ndcg_at_5: float
    primary: float
    exact: bool = False

    def __post_init__(self) -> None:
        _text(self.name, "metric.name")
        for location, value in (
            ("metric.GAUC", self.gauc),
            ("metric.nDCG@5", self.ndcg_at_5),
            ("metric.primary", self.primary),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SafeContextError(f"{location} must be finite in [0, 1]")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise SafeContextError(f"{location} must be finite in [0, 1]")
        float64_primary = (self.gauc + self.ndcg_at_5) / 2.0
        float32_gauc = struct.unpack("!f", struct.pack("!f", self.gauc))[0]
        float32_ndcg = struct.unpack("!f", struct.pack("!f", self.ndcg_at_5))[0]
        float32_sum = struct.unpack("!f", struct.pack("!f", float32_gauc + float32_ndcg))[0]
        float32_primary = struct.unpack("!f", struct.pack("!f", float32_sum / 2.0))[0]
        # The untouched organizer evaluates one legal path with float32 labels and therefore
        # computes primary with float32 arithmetic.  Accept exactly either organizer arithmetic
        # path; this is not a tolerance and does not admit an independently supplied metric.
        if self.primary not in {float64_primary, float32_primary}:
            raise SafeContextError("metric.primary must be the mean of GAUC and nDCG@5")
        if type(self.exact) is not bool:
            raise SafeContextError("metric.exact must be boolean")

    @staticmethod
    def _rounded(value: float) -> float:
        return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

    def to_wire(self, *, outer: bool) -> dict[str, object]:
        if outer:
            values = tuple(self._rounded(value) for value in (self.gauc, self.ndcg_at_5))
            # Preserve the trusted aggregate primary's independently rounded value rather than
            # recomputing it from already-rounded components.
            primary = self._rounded(self.primary)
            precision = "rounded_4dp"
        else:
            values = (self.gauc, self.ndcg_at_5)
            primary = self.primary
            precision = "exact"
        return {
            "name": self.name,
            "GAUC": values[0],
            "nDCG@5": values[1],
            "primary": primary,
            "precision": precision,
        }


@dataclass(frozen=True, slots=True)
class ResearchBudgetContext:
    remaining_attempts: int
    remaining_wall_seconds: int
    remaining_outer_promotions: int
    intervention_count: int

    def __post_init__(self) -> None:
        limits = (
            ("remaining_attempts", self.remaining_attempts, 50),
            ("remaining_wall_seconds", self.remaining_wall_seconds, 21_600),
            ("remaining_outer_promotions", self.remaining_outer_promotions, 6),
            ("intervention_count", self.intervention_count, 1_000_000),
        )
        for name, value, maximum in limits:
            if type(value) is not int or not 0 <= value <= maximum:
                raise SafeContextError(f"{name} must be an integer in [0, {maximum}]")

    def to_wire(self) -> dict[str, int]:
        return {
            "remaining_attempts": self.remaining_attempts,
            "remaining_wall_seconds": self.remaining_wall_seconds,
            "remaining_outer_promotions": self.remaining_outer_promotions,
            "intervention_count": self.intervention_count,
        }


def _capability_manifest(raw: Mapping[str, object]) -> dict[str, object]:
    if set(raw) != _CAPABILITY_KEYS:
        raise SafeContextError("capability manifest has unknown or missing fields")
    if raw["schema_version"] != 1:
        raise SafeContextError("capability manifest schema_version must be 1")
    name = _text(raw["capability_name"], "capability_name")
    if "target" in name.casefold():
        raise SafeContextError("target capability cannot enter research-model context")
    phase = _text(raw["phase"], "capability.phase")
    if name not in _CAPABILITY_PHASES or phase not in _CAPABILITY_PHASES[name]:
        raise SafeContextError("capability name and phase are not a legal input pairing")
    row_count = raw["row_count"]
    if type(row_count) is not int or row_count < 0:
        raise SafeContextError("capability.row_count must be a non-negative integer")
    columns_raw = raw["columns"]
    if not isinstance(columns_raw, list) or len(columns_raw) > 256:
        raise SafeContextError("capability.columns must be a bounded array")
    columns: list[dict[str, str]] = []
    output_names: set[str] = set()
    for index, value in enumerate(columns_raw):
        if not isinstance(value, Mapping) or set(value) != _CAPABILITY_COLUMN_KEYS:
            raise SafeContextError(f"capability.columns[{index}] has unknown or missing fields")
        column = {
            key: _text(value[key], f"capability.columns[{index}].{key}")
            for key in sorted(_CAPABILITY_COLUMN_KEYS)
        }
        if column["name"] in output_names:
            raise SafeContextError("capability columns contain duplicate output names")
        output_names.add(column["name"])
        member, separator, source_column = column["source_field"].rpartition(":")
        if not separator or not member or not source_column:
            raise SafeContextError("capability source_field is not source-qualified")
        try:
            specification = field_spec(FieldKey(member, source_column))
        except FieldPolicyError as exc:
            raise SafeContextError(f"capability source field is not registered: {exc}") from exc
        if specification.role is not FieldRole.INFERENCE_INPUT or not specification.enabled:
            raise SafeContextError(
                f"capability source field {column['source_field']!r} is not an enabled "
                "inference input"
            )
        if column["logical_dtype"] != specification.logical_dtype.value:
            raise SafeContextError("capability logical dtype differs from the field policy")
        columns.append(column)
    manifest_policy_digest = _digest(raw["field_policy_digest"], "capability.field_policy_digest")
    if manifest_policy_digest != field_policy_digest():
        raise SafeContextError("capability field-policy digest differs from trusted policy")
    return {
        "schema_version": 1,
        "capability_name": name,
        "phase": phase,
        "row_count": row_count,
        "columns": columns,
        "field_policy_digest": manifest_policy_digest,
        "capability_schema_digest": _digest(
            raw["capability_schema_digest"], "capability.capability_schema_digest"
        ),
        "logical_content_digest": _digest(
            raw["logical_content_digest"], "capability.logical_content_digest"
        ),
        "capability_digest": _digest(raw["capability_digest"], "capability.capability_digest"),
    }


@dataclass(frozen=True, slots=True)
class SafeResearchContext:
    """Canonical, redacted context payload with no row-level data interface."""

    _payload: Mapping[str, object] = field(repr=False)
    digest: str

    def __post_init__(self) -> None:
        _digest(self.digest, "safe_context.digest")
        if canonical_digest(dict(self._payload)) != self.digest:
            raise SafeContextError("safe-context digest mismatch")

    def to_wire(self) -> dict[str, object]:
        # Canonical JSON is also a small defensive deep copy: callers cannot mutate the stored
        # context or smuggle non-JSON values through a mapping proxy.
        return cast(dict[str, object], json.loads(canonical_json_bytes(dict(self._payload))))


def build_safe_research_context(
    *,
    starter_manifest_sha256: str,
    dataset_manifest_sha256: str,
    capability_manifests: Sequence[Mapping[str, object]],
    budgets: ResearchBudgetContext,
    train_eda: Sequence[AggregateRecord] = (),
    validation_input_eda: Sequence[AggregateRecord] = (),
    method_cards: Sequence[AggregateRecord] = (),
    inner_metrics: Sequence[MetricSummary] = (),
    outer_metrics: Sequence[MetricSummary] = (),
    campaign_records: Sequence[AggregateRecord] = (),
    secrets: Sequence[str] = (),
) -> SafeResearchContext:
    """Build one provider-ready context from trusted metadata and scalar aggregates only."""

    if not isinstance(budgets, ResearchBudgetContext):
        raise SafeContextError("budgets must be ResearchBudgetContext")
    for value in inner_metrics:
        if not value.exact:
            raise SafeContextError("inner metric summaries must explicitly declare exact=True")
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": benchmark_manifest(),
        "benchmark_digest": benchmark_digest(),
        "starter_manifest_sha256": _digest(starter_manifest_sha256, "starter_manifest_sha256"),
        "dataset_manifest_sha256": _digest(dataset_manifest_sha256, "dataset_manifest_sha256"),
        "field_policy": _provider_field_policy_manifest(),
        "field_policy_digest": field_policy_digest(),
        "capabilities": [_capability_manifest(value) for value in capability_manifests],
        "train_eda": [value.to_wire() for value in train_eda],
        "validation_input_eda": [value.to_wire() for value in validation_input_eda],
        "method_cards": [value.to_wire() for value in method_cards],
        "inner_metrics": [value.to_wire(outer=False) for value in inner_metrics],
        "outer_metrics": [value.to_wire(outer=True) for value in outer_metrics],
        "campaign_records": [value.to_wire() for value in campaign_records],
        "budgets": budgets.to_wire(),
    }
    redacted = cast(dict[str, object], _redact_json(payload, secrets))
    digest = canonical_digest(redacted)
    return SafeResearchContext(MappingProxyType(redacted), digest)
