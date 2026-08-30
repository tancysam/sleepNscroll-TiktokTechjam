"""Leakage-safe expanding-prefix aggregate features.

This module is part of the trusted data boundary.  Candidate code receives the
resulting numeric matrices, never the outcome events used to construct them.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from kuairand_agent.data.canonical import APPROVED_AUXILIARY_TARGETS, PRIMARY_TARGET

_SCHEMA_VERSION: Final = 1
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# Keep this boundary derived from the canonical target registry.  The two
# legacy spellings remain blocked because older callers used them before the
# archive schema was centralized; they are aliases, not approved targets.
_COMPATIBILITY_OUTCOME_ALIASES: Final = frozenset({"hate", "play_time_truncate"})
_OUTCOME_FIELDS: Final = frozenset(
    (PRIMARY_TARGET, *APPROVED_AUXILIARY_TARGETS, *_COMPATIBILITY_OUTCOME_ALIASES)
)
_ROW_ID_FIELDS: Final = frozenset(
    {"row_id", "split_row_id", "source_row_id", "source_ordinal", "record_ordinal"}
)

type SafeScalar = str | int | float | bool | None
type TypedKeyPart = tuple[str, str]
type TypedKey = tuple[TypedKeyPart, ...]
type _CountPair = tuple[int, int]


class CausalFeatureError(ValueError):
    """Raised when a causal-feature request violates the trusted contract."""


class CausalFeatureCacheError(RuntimeError):
    """Raised when a committed content-addressed cache entry is invalid."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest_manifest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_name(name: object, *, context: str) -> str:
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise CausalFeatureError(f"{context} must match {_SAFE_NAME.pattern!r}; got {name!r}")
    lowered = name.casefold()
    if lowered in _ROW_ID_FIELDS:
        raise CausalFeatureError(f"{context} cannot expose row_id or provenance identity")
    if lowered in _OUTCOME_FIELDS or lowered.endswith("_stay_time"):
        raise CausalFeatureError(f"{context} cannot expose a current-row outcome field")
    return name


def _normalize_time(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise CausalFeatureError("time_ms values must be integral and cannot be bool")
    normalized = int(value)
    if not _INT64_MIN <= normalized <= _INT64_MAX:
        raise CausalFeatureError("time_ms values must fit signed int64")
    return normalized


def _normalize_scalar(value: object) -> SafeScalar:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise CausalFeatureError("causal key values must be finite")
        if normalized == 0.0:
            return 0.0
        return normalized
    raise CausalFeatureError("causal key values must be str, integral, finite real, bool, or None")


def _typed_part(value: SafeScalar) -> TypedKeyPart:
    if value is None:
        return ("none", "")
    if isinstance(value, bool):
        return ("bool", "1" if value else "0")
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", value.hex())
    return ("str", value)


def _digest_inputs(time_ms: tuple[int, ...], fields: Mapping[str, tuple[SafeScalar, ...]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"causal-inputs-v1\0")
    digest.update(_canonical_json(len(time_ms)))
    for timestamp in time_ms:
        digest.update(b"t")
        digest.update(str(timestamp).encode("ascii"))
        digest.update(b"\0")
    for name in sorted(fields):
        digest.update(b"f")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        for field_value in fields[name]:
            digest.update(_canonical_json(_typed_part(field_value)))
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class CausalInputs:
    """Immutable pre-impression fields in canonical split order.

    ``time_ms`` is chronology, not a candidate feature.  ``fields`` cannot
    contain outcomes, row identity, or provenance columns.
    """

    time_ms: tuple[int, ...]
    fields: Mapping[str, tuple[SafeScalar, ...]]
    logical_digest: str

    def __init__(
        self,
        *,
        time_ms: Iterable[object],
        fields: Mapping[str, Iterable[object]],
    ) -> None:
        times = tuple(_normalize_time(value) for value in time_ms)
        if not times:
            raise CausalFeatureError("causal inputs must contain at least one row")
        if not isinstance(fields, Mapping) or not fields:
            raise CausalFeatureError("causal inputs require at least one safe field")

        normalized_fields: dict[str, tuple[SafeScalar, ...]] = {}
        for raw_name, raw_values in fields.items():
            name = _validate_name(raw_name, context="causal input field name")
            values = tuple(_normalize_scalar(value) for value in raw_values)
            if len(values) != len(times):
                raise CausalFeatureError(
                    f"field {name!r} has {len(values)} rows; expected {len(times)}"
                )
            normalized_fields[name] = values

        ordered = dict(sorted(normalized_fields.items()))
        frozen_fields = cast(Mapping[str, tuple[SafeScalar, ...]], MappingProxyType(ordered))
        object.__setattr__(self, "time_ms", times)
        object.__setattr__(self, "fields", frozen_fields)
        object.__setattr__(self, "logical_digest", _digest_inputs(times, frozen_fields))

    @property
    def row_count(self) -> int:
        """Number of impressions in canonical order."""

        return len(self.time_ms)


@dataclass(frozen=True, slots=True, init=False)
class OutcomeEvents:
    """Authorized training-prefix outcomes; no query outcome type exists."""

    long_view: tuple[int, ...]
    logical_digest: str

    def __init__(self, *, long_view: Iterable[object]) -> None:
        values: list[int] = []
        for value in long_view:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise CausalFeatureError("long_view prefix outcomes must be binary integers")
            normalized = int(value)
            if normalized not in (0, 1):
                raise CausalFeatureError(
                    "long_view prefix outcomes must be binary integers in {0, 1}"
                )
            values.append(normalized)
        if not values:
            raise CausalFeatureError("prefix outcomes must contain at least one event")
        frozen = tuple(values)
        object.__setattr__(self, "long_view", frozen)
        object.__setattr__(
            self,
            "logical_digest",
            _digest_manifest({"schema_version": _SCHEMA_VERSION, "long_view": frozen}),
        )

    @property
    def row_count(self) -> int:
        """Number of authorized training-prefix outcomes."""

        return len(self.long_view)


def _finite_real(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CausalFeatureError(f"{context} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CausalFeatureError(f"{context} must be a finite real number")
    return normalized


@dataclass(frozen=True, slots=True)
class AggregateSpec:
    """One ordered, typed grouping and its hierarchical smoothing policy."""

    key_fields: tuple[str, ...]
    name: str | None = None
    smoothing: float = 20.0
    initial_prior: float = 0.5

    def __post_init__(self) -> None:
        keys = tuple(self.key_fields)
        if not keys:
            raise CausalFeatureError("aggregate key_fields cannot be empty")
        normalized_keys = tuple(_validate_name(key, context="aggregate key field") for key in keys)
        if len(set(normalized_keys)) != len(normalized_keys):
            raise CausalFeatureError("aggregate key_fields cannot contain duplicates")
        object.__setattr__(self, "key_fields", normalized_keys)

        if self.name is not None:
            _validate_name(self.name, context="aggregate name")
        resolved = self.resolved_name
        _validate_name(resolved, context="resolved aggregate name")

        smoothing = _finite_real(self.smoothing, context="aggregate smoothing")
        if smoothing <= 0.0:
            raise CausalFeatureError("aggregate smoothing must be finite and greater than zero")
        object.__setattr__(self, "smoothing", smoothing)

        prior = _finite_real(self.initial_prior, context="initial_prior")
        if not 0.0 <= prior <= 1.0:
            raise CausalFeatureError("initial_prior must be finite and in [0, 1]")
        object.__setattr__(self, "initial_prior", prior)

    @property
    def resolved_name(self) -> str:
        """Stable feature prefix, independent of mapping insertion order."""

        return self.name if self.name is not None else "__by__".join(self.key_fields)

    def manifest(self) -> dict[str, object]:
        """Canonical cache identity for this aggregate."""

        return {
            "name": self.resolved_name,
            "key_fields": list(self.key_fields),
            "smoothing": self.smoothing,
            "initial_prior": self.initial_prior,
        }


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    """Caller-reviewed logical identities; never cache paths or wall-clock data."""

    dataset: str
    split: str
    field_policy: str
    builder_source: str

    def __post_init__(self) -> None:
        for field_name in ("dataset", "split", "field_policy", "builder_source"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 4096
                or "\x00" in value
                or "\n" in value
                or "\r" in value
            ):
                raise CausalFeatureError(
                    f"build identity {field_name} must be a non-empty single-line string"
                )

    def manifest(self) -> dict[str, str]:
        """Canonical identity included in every cache and logical manifest."""

        return {
            "dataset": self.dataset,
            "split": self.split,
            "field_policy": self.field_policy,
            "builder_source": self.builder_source,
        }


def _matrix_digest(names: tuple[str, ...], values: NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    digest.update(b"causal-feature-matrix-v1\0")
    digest.update(_canonical_json({"feature_names": names, "shape": values.shape}))
    digest.update(values.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class FeatureMatrix:
    """Immutable, finite float64 feature matrix in canonical row order."""

    values: NDArray[np.float64] = field(repr=False)
    feature_names: tuple[str, ...]
    logical_digest: str

    def __init__(
        self, values: Sequence[Sequence[float]] | NDArray[np.float64], feature_names: Iterable[str]
    ) -> None:
        names = tuple(feature_names)
        if not names or len(set(names)) != len(names):
            raise CausalFeatureError("feature names must be non-empty and unique")
        for name in names:
            _validate_name(name, context="feature name")

        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CausalFeatureError("feature values must form a float64 matrix") from exc
        if array.ndim != 2:
            raise CausalFeatureError("feature values must be two-dimensional")
        if array.shape[0] == 0 or array.shape[1] != len(names):
            raise CausalFeatureError("feature matrix shape does not match rows and feature names")
        if not np.all(np.isfinite(array)):
            raise CausalFeatureError("feature values must all be finite")

        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        immutable_bytes = contiguous.tobytes(order="C")
        frozen = np.frombuffer(immutable_bytes, dtype=np.float64).reshape(contiguous.shape)
        object.__setattr__(self, "values", frozen)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "logical_digest", _matrix_digest(names, frozen))

    @property
    def row_count(self) -> int:
        """Number of canonical rows."""

        return int(self.values.shape[0])

    @property
    def feature_count(self) -> int:
        """Number of safe derived features."""

        return int(self.values.shape[1])

    @property
    def columns(self) -> tuple[str, ...]:
        """Alias used by tabular candidate adapters."""

        return self.feature_names

    @property
    def digest(self) -> str:
        """Alias for the logical feature artifact digest."""

        return self.logical_digest

    def manifest(self) -> dict[str, object]:
        """Logical matrix metadata without filesystem or build-time fields."""

        return {
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "logical_digest": self.logical_digest,
        }


@dataclass(frozen=True, slots=True)
class CausalFeaturePair:
    """Expanding-prefix training features and an optional frozen query matrix."""

    identity: BuildIdentity
    specs: tuple[AggregateSpec, ...]
    prefix: FeatureMatrix
    query: FeatureMatrix | None
    cache_key: str
    prefix_input_digest: str
    prefix_outcome_digest: str
    query_input_digest: str | None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.cache_key):
            raise CausalFeatureError("cache_key must be a lowercase SHA-256 digest")
        if self.query is None and self.query_input_digest is not None:
            raise CausalFeatureError("query digest cannot exist without query features")
        if self.query is not None and self.query_input_digest is None:
            raise CausalFeatureError("query features require a query input digest")
        if self.query is not None and self.query.feature_names != self.prefix.feature_names:
            raise CausalFeatureError("prefix and query feature schemas must match")

    @property
    def prefix_features(self) -> FeatureMatrix:
        """Explicit alias for callers that distinguish features from events."""

        return self.prefix

    @property
    def query_features(self) -> FeatureMatrix | None:
        """Explicit alias for the frozen query matrix."""

        return self.query

    def manifest(self) -> dict[str, object]:
        """Deterministic logical build manifest."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity.manifest(),
            "specs": [spec.manifest() for spec in self.specs],
            "cache_key": self.cache_key,
            "inputs": {
                "prefix": self.prefix_input_digest,
                "prefix_long_view": self.prefix_outcome_digest,
                "query": self.query_input_digest,
            },
            "prefix": self.prefix.manifest(),
            "query": None if self.query is None else self.query.manifest(),
        }

    @property
    def logical_digest(self) -> str:
        """SHA-256 identity of the path- and time-free logical artifact."""

        return _digest_manifest(self.manifest())


@dataclass(frozen=True, slots=True)
class _BuildRequest:
    prefix_inputs: CausalInputs
    prefix_outcomes: OutcomeEvents
    specs: tuple[AggregateSpec, ...]
    identity: BuildIdentity
    query_inputs: CausalInputs | None
    cache_key: str

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity.manifest(),
            "specs": [spec.manifest() for spec in self.specs],
            "inputs": {
                "prefix": self.prefix_inputs.logical_digest,
                "prefix_long_view": self.prefix_outcomes.logical_digest,
                "query": (None if self.query_inputs is None else self.query_inputs.logical_digest),
            },
        }


def _validate_request(
    *,
    prefix_inputs: CausalInputs,
    prefix_outcomes: OutcomeEvents,
    specs: Sequence[AggregateSpec],
    identity: BuildIdentity,
    query_inputs: CausalInputs | None,
) -> _BuildRequest:
    if not isinstance(prefix_inputs, CausalInputs):
        raise CausalFeatureError("prefix_inputs must be CausalInputs")
    if not isinstance(prefix_outcomes, OutcomeEvents):
        raise CausalFeatureError("prefix_outcomes must be OutcomeEvents")
    if not isinstance(identity, BuildIdentity):
        raise CausalFeatureError("identity must be BuildIdentity")
    normalized_specs = tuple(specs)
    if not normalized_specs or not all(
        isinstance(spec, AggregateSpec) for spec in normalized_specs
    ):
        raise CausalFeatureError("at least one AggregateSpec is required")
    if prefix_inputs.row_count != prefix_outcomes.row_count:
        raise CausalFeatureError("prefix inputs and outcomes must have equal row counts")
    if query_inputs is not None and not isinstance(query_inputs, CausalInputs):
        raise CausalFeatureError("query_inputs must be CausalInputs or None")

    names = tuple(spec.resolved_name for spec in normalized_specs)
    if len(set(names)) != len(names):
        raise CausalFeatureError("aggregate names must be unique")
    priors = {spec.initial_prior for spec in normalized_specs}
    if len(priors) != 1:
        raise CausalFeatureError("all aggregate specs must share one global initial_prior")

    for spec in normalized_specs:
        missing_prefix = set(spec.key_fields).difference(prefix_inputs.fields)
        if missing_prefix:
            raise CausalFeatureError(
                f"prefix inputs are missing aggregate fields {sorted(missing_prefix)!r}"
            )
        if query_inputs is not None:
            missing_query = set(spec.key_fields).difference(query_inputs.fields)
            if missing_query:
                raise CausalFeatureError(
                    f"query inputs are missing aggregate fields {sorted(missing_query)!r}"
                )

    if query_inputs is not None and max(prefix_inputs.time_ms) >= min(query_inputs.time_ms):
        raise CausalFeatureError("frozen query requires max(prefix time_ms) < min(query time_ms)")

    request_manifest = {
        "schema_version": _SCHEMA_VERSION,
        "identity": identity.manifest(),
        "specs": [spec.manifest() for spec in normalized_specs],
        "inputs": {
            "prefix": prefix_inputs.logical_digest,
            "prefix_long_view": prefix_outcomes.logical_digest,
            "query": None if query_inputs is None else query_inputs.logical_digest,
        },
    }
    cache_key = _digest_manifest(request_manifest)
    return _BuildRequest(
        prefix_inputs=prefix_inputs,
        prefix_outcomes=prefix_outcomes,
        specs=normalized_specs,
        identity=identity,
        query_inputs=query_inputs,
        cache_key=cache_key,
    )


def _feature_names(specs: tuple[AggregateSpec, ...]) -> tuple[str, ...]:
    names = ["global__long_view_prior"]
    for spec in specs:
        names.extend(
            (
                f"{spec.resolved_name}__exposure",
                f"{spec.resolved_name}__positive",
                f"{spec.resolved_name}__smoothed_rate",
            )
        )
    return tuple(names)


def _key(inputs: CausalInputs, key_fields: tuple[str, ...], index: int) -> TypedKey:
    return tuple(_typed_part(inputs.fields[name][index]) for name in key_fields)


def _add_count(current: _CountPair | None, exposure: int, positive: int) -> _CountPair:
    if current is None:
        return (exposure, positive)
    return (current[0] + exposure, current[1] + positive)


def _build_prefix_matrix(
    request: _BuildRequest,
) -> tuple[FeatureMatrix, list[dict[TypedKey, _CountPair]], int, int]:
    inputs = request.prefix_inputs
    labels = request.prefix_outcomes.long_view
    specs = request.specs
    names = _feature_names(specs)
    order = np.argsort(np.asarray(inputs.time_ms, dtype=np.int64), kind="stable")
    chronological = np.empty((inputs.row_count, len(names)), dtype=np.float64)
    states: list[dict[TypedKey, _CountPair]] = [dict() for _ in specs]
    global_exposure = 0
    global_positive = 0
    initial_prior = specs[0].initial_prior

    start = 0
    while start < inputs.row_count:
        bucket_time = inputs.time_ms[int(order[start])]
        end = start + 1
        while end < inputs.row_count and inputs.time_ms[int(order[end])] == bucket_time:
            end += 1

        prior = initial_prior if global_exposure == 0 else global_positive / global_exposure
        for position in range(start, end):
            canonical_index = int(order[position])
            chronological[position, 0] = prior
            column = 1
            for spec_index, spec in enumerate(specs):
                exposure, positive = states[spec_index].get(
                    _key(inputs, spec.key_fields, canonical_index), (0, 0)
                )
                smoothed = (positive + spec.smoothing * prior) / (exposure + spec.smoothing)
                chronological[position, column : column + 3] = (
                    exposure,
                    positive,
                    smoothed,
                )
                column += 3

        # Updates happen only after every simultaneous row has been featurized.
        # Aggregate bucket deltas before touching persistent state so the update
        # is commutative and independent of canonical order inside the bucket.
        for spec_index, spec in enumerate(specs):
            deltas: dict[TypedKey, _CountPair] = {}
            for position in range(start, end):
                canonical_index = int(order[position])
                key = _key(inputs, spec.key_fields, canonical_index)
                deltas[key] = _add_count(deltas.get(key), 1, labels[canonical_index])
            for key, (exposure, positive) in deltas.items():
                states[spec_index][key] = _add_count(
                    states[spec_index].get(key), exposure, positive
                )

        global_exposure += end - start
        global_positive += sum(labels[int(order[position])] for position in range(start, end))
        start = end

    canonical = np.empty_like(chronological)
    canonical[order, :] = chronological
    return (
        FeatureMatrix(canonical, names),
        states,
        global_exposure,
        global_positive,
    )


def _build_query_matrix(
    request: _BuildRequest,
    states: list[dict[TypedKey, _CountPair]],
    global_exposure: int,
    global_positive: int,
) -> FeatureMatrix | None:
    inputs = request.query_inputs
    if inputs is None:
        return None
    specs = request.specs
    names = _feature_names(specs)
    values = np.empty((inputs.row_count, len(names)), dtype=np.float64)
    prior = specs[0].initial_prior if global_exposure == 0 else global_positive / global_exposure
    for index in range(inputs.row_count):
        values[index, 0] = prior
        column = 1
        for spec_index, spec in enumerate(specs):
            exposure, positive = states[spec_index].get(
                _key(inputs, spec.key_fields, index), (0, 0)
            )
            smoothed = (positive + spec.smoothing * prior) / (exposure + spec.smoothing)
            values[index, column : column + 3] = (exposure, positive, smoothed)
            column += 3
    return FeatureMatrix(values, names)


def _build_uncached(request: _BuildRequest) -> CausalFeaturePair:
    prefix, states, global_exposure, global_positive = _build_prefix_matrix(request)
    query = _build_query_matrix(request, states, global_exposure, global_positive)
    return CausalFeaturePair(
        identity=request.identity,
        specs=request.specs,
        prefix=prefix,
        query=query,
        cache_key=request.cache_key,
        prefix_input_digest=request.prefix_inputs.logical_digest,
        prefix_outcome_digest=request.prefix_outcomes.logical_digest,
        query_input_digest=(
            None if request.query_inputs is None else request.query_inputs.logical_digest
        ),
    )


class CausalFeatureCache:
    """Content-addressed local cache with JSON commit metadata and NPZ values."""

    def __init__(self, cache_dir: Path | str) -> None:
        path = Path(cache_dir)
        if path.exists() and not path.is_dir():
            raise CausalFeatureCacheError("causal feature cache path must be a directory")
        self._cache_dir = path

    @property
    def cache_dir(self) -> Path:
        """Storage root; deliberately absent from every logical digest."""

        return self._cache_dir

    def build_or_load(
        self,
        *,
        prefix_inputs: CausalInputs,
        prefix_outcomes: OutcomeEvents,
        specs: Sequence[AggregateSpec],
        identity: BuildIdentity,
        query_inputs: CausalInputs | None = None,
    ) -> CausalFeaturePair:
        """Return a verified hit or atomically commit a deterministic cold build."""

        request = _validate_request(
            prefix_inputs=prefix_inputs,
            prefix_outcomes=prefix_outcomes,
            specs=specs,
            identity=identity,
            query_inputs=query_inputs,
        )
        metadata_path, values_path = self._paths(request.cache_key)
        metadata_exists = metadata_path.exists()
        values_exist = values_path.exists()
        if metadata_exists != values_exist:
            raise CausalFeatureCacheError(
                f"incomplete causal feature cache entry {request.cache_key}"
            )
        if metadata_exists:
            return self._load(request, metadata_path, values_path)

        pair = _build_uncached(request)
        self._commit(request, pair, metadata_path, values_path)
        return pair

    def _paths(self, cache_key: str) -> tuple[Path, Path]:
        return (
            self._cache_dir / f"{cache_key}.json",
            self._cache_dir / f"{cache_key}.npz",
        )

    @staticmethod
    def _metadata(request: _BuildRequest, pair: CausalFeaturePair) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "request": request.manifest(),
            "cache_key": request.cache_key,
            "pair": pair.manifest(),
            "pair_logical_digest": pair.logical_digest,
        }

    def _load(
        self,
        request: _BuildRequest,
        metadata_path: Path,
        values_path: Path,
    ) -> CausalFeaturePair:
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(raw_metadata, dict):
                raise TypeError("metadata root is not an object")
            if raw_metadata.get("request") != request.manifest():
                raise ValueError("request manifest mismatch")
            if raw_metadata.get("cache_key") != request.cache_key:
                raise ValueError("cache key mismatch")

            with np.load(values_path, allow_pickle=False) as archive:
                expected_files = {"prefix_values"}
                if request.query_inputs is not None:
                    expected_files.add("query_values")
                if set(archive.files) != expected_files:
                    raise ValueError("unexpected NPZ members")
                prefix_values = np.array(archive["prefix_values"], copy=True)
                query_values = (
                    None
                    if request.query_inputs is None
                    else np.array(archive["query_values"], copy=True)
                )

            names = _feature_names(request.specs)
            prefix = FeatureMatrix(prefix_values, names)
            query = None if query_values is None else FeatureMatrix(query_values, names)
            pair = CausalFeaturePair(
                identity=request.identity,
                specs=request.specs,
                prefix=prefix,
                query=query,
                cache_key=request.cache_key,
                prefix_input_digest=request.prefix_inputs.logical_digest,
                prefix_outcome_digest=request.prefix_outcomes.logical_digest,
                query_input_digest=(
                    None if request.query_inputs is None else request.query_inputs.logical_digest
                ),
            )
            if raw_metadata != self._metadata(request, pair):
                raise ValueError("logical artifact manifest mismatch")
            return pair
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            EOFError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ) as exc:
            raise CausalFeatureCacheError(
                f"corrupt causal feature cache entry {request.cache_key}: {exc}"
            ) from exc

    def _commit(
        self,
        request: _BuildRequest,
        pair: CausalFeaturePair,
        metadata_path: Path,
        values_path: Path,
    ) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        values_temp: Path | None = None
        metadata_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{request.cache_key}.",
                suffix=".npz.tmp",
                dir=self._cache_dir,
                delete=False,
            ) as handle:
                values_temp = Path(handle.name)
                if pair.query is not None:
                    np.savez(
                        handle,
                        prefix_values=pair.prefix.values,
                        query_values=pair.query.values,
                    )
                else:
                    np.savez(handle, prefix_values=pair.prefix.values)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(values_temp, values_path)
            values_temp = None

            encoded = _canonical_json(self._metadata(request, pair)) + b"\n"
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{request.cache_key}.",
                suffix=".json.tmp",
                dir=self._cache_dir,
                delete=False,
            ) as handle:
                metadata_temp = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            # JSON is the commit marker and is always installed last.
            os.replace(metadata_temp, metadata_path)
            metadata_temp = None
        finally:
            for path in (values_temp, metadata_temp):
                if path is not None:
                    with suppress(FileNotFoundError):
                        path.unlink()


def build_causal_feature_pair(
    *,
    prefix_inputs: CausalInputs,
    prefix_outcomes: OutcomeEvents,
    specs: Sequence[AggregateSpec],
    identity: BuildIdentity,
    query_inputs: CausalInputs | None = None,
    cache_dir: Path | str | None = None,
) -> CausalFeaturePair:
    """Build expanding-prefix features and an optional frozen query matrix.

    No query outcome parameter exists by design.  When ``cache_dir`` is
    supplied, a content-addressed cache is verified before values are returned.
    """

    if cache_dir is not None:
        return CausalFeatureCache(cache_dir).build_or_load(
            prefix_inputs=prefix_inputs,
            prefix_outcomes=prefix_outcomes,
            specs=specs,
            identity=identity,
            query_inputs=query_inputs,
        )
    request = _validate_request(
        prefix_inputs=prefix_inputs,
        prefix_outcomes=prefix_outcomes,
        specs=specs,
        identity=identity,
        query_inputs=query_inputs,
    )
    return _build_uncached(request)


def build_causal_features(
    *,
    prefix_inputs: CausalInputs,
    prefix_outcomes: OutcomeEvents,
    specs: Sequence[AggregateSpec],
    identity: BuildIdentity,
    query_inputs: CausalInputs | None = None,
    cache_dir: Path | str | None = None,
) -> CausalFeaturePair:
    """Compatibility spelling for :func:`build_causal_feature_pair`."""

    return build_causal_feature_pair(
        prefix_inputs=prefix_inputs,
        prefix_outcomes=prefix_outcomes,
        specs=specs,
        identity=identity,
        query_inputs=query_inputs,
        cache_dir=cache_dir,
    )
