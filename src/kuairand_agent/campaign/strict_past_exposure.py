"""Input-only strict-past exposure features for the trusted campaign data plane.

The public builder intentionally accepts only canonical impression inputs.  It has no outcome,
label, scorer, or target argument, so public-query inputs can warm later query rows without
creating a route to public or final outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.causal_features import FeatureMatrix

STRICT_PAST_EXPOSURE_SCHEMA_VERSION: Final = 1
STRICT_PAST_EXPOSURE_POLICY: Final = "strict-past-input-exposure-v1"
_DATE_CHRONOLOGY_STRIDE: Final = 10_000_000_000_000
_DATE_ORIGIN: Final = 20220408
_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_SCOPES: Final = (
    ("user", ("user_id",)),
    ("video", ("video_id",)),
    ("author", ("author_id",)),
    ("user_video", ("user_id", "video_id")),
)
STRICT_PAST_EXPOSURE_FEATURE_NAMES: Final = tuple(
    feature_name
    for scope, _fields in _SCOPES
    for feature_name in (
        f"{scope}__strict_earlier_exposure_count",
        f"{scope}__first_seen",
        f"{scope}__log1p_time_since_last_exposure",
    )
)

type _ScopeKey = tuple[str, ...]
type _ExposureState = tuple[int, int]


class StrictPastExposureError(ValueError):
    """Raised when an input-only strict-past exposure request is invalid."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(b"strict-past-input-exposure-v1\0" + _canonical_json(value)).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise StrictPastExposureError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_inputs(value: object, *, name: str) -> CanonicalInputs:
    if not isinstance(value, CanonicalInputs):
        raise StrictPastExposureError(f"{name} must be CanonicalInputs")
    if len(value) == 0:
        raise StrictPastExposureError(f"{name} must not be empty")
    return value


def _order(inputs: CanonicalInputs) -> tuple[int, ...]:
    """Return a deterministic date-first chronology with canonical order only as a tie-breaker."""

    return tuple(
        sorted(
            range(len(inputs)),
            key=lambda index: (inputs.date[index], inputs.time_ms[index], index),
        )
    )


def _date_time(inputs: CanonicalInputs, index: int) -> tuple[int, int]:
    return (inputs.date[index], inputs.time_ms[index])


def _chronology(inputs: CanonicalInputs, index: int) -> int:
    """Map the reviewed date-first order into a stable, candidate-hidden gap axis."""

    return (inputs.date[index] - _DATE_ORIGIN) * _DATE_CHRONOLOGY_STRIDE + inputs.time_ms[index]


def _scope_key(inputs: CanonicalInputs, fields: tuple[str, ...], index: int) -> _ScopeKey:
    return tuple(str(getattr(inputs, field)[index]) for field in fields)


def _write_bucket_features(
    *,
    inputs: CanonicalInputs,
    order: Sequence[int],
    states: list[dict[_ScopeKey, _ExposureState]],
) -> FeatureMatrix:
    values = np.empty((len(inputs), len(STRICT_PAST_EXPOSURE_FEATURE_NAMES)), dtype=np.float64)
    start = 0
    while start < len(order):
        first = order[start]
        bucket = _date_time(inputs, first)
        end = start + 1
        while end < len(order) and _date_time(inputs, order[end]) == bucket:
            end += 1

        at = _chronology(inputs, first)
        for position in range(start, end):
            row = order[position]
            column = 0
            for scope_index, (_scope, fields) in enumerate(_SCOPES):
                state = states[scope_index].get(_scope_key(inputs, fields, row))
                if state is None:
                    values[row, column : column + 3] = (0.0, 1.0, 0.0)
                else:
                    exposure_count, last_seen = state
                    gap = at - last_seen
                    if gap <= 0:
                        raise StrictPastExposureError(
                            "date-dominant chronology must strictly advance"
                        )
                    values[row, column : column + 3] = (
                        float(exposure_count),
                        0.0,
                        math.log1p(float(gap)),
                    )
                column += 3

        # Commit only after all simultaneous impressions have been featurized.  Aggregating
        # deltas makes results invariant to canonical ordering within a timestamp bucket.
        for scope_index, (_scope, fields) in enumerate(_SCOPES):
            deltas: dict[_ScopeKey, int] = {}
            for position in range(start, end):
                row = order[position]
                key = _scope_key(inputs, fields, row)
                deltas[key] = deltas.get(key, 0) + 1
            for key, increment in deltas.items():
                previous = states[scope_index].get(key)
                previous_count = 0 if previous is None else previous[0]
                states[scope_index][key] = (previous_count + increment, at)
        start = end

    if not np.isfinite(values).all():
        raise StrictPastExposureError("exposure feature transform produced a non-finite value")
    return FeatureMatrix(values, STRICT_PAST_EXPOSURE_FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class StrictPastExposurePair:
    """Immutable prefix/query matrices and complete path-independent exposure provenance."""

    prefix: FeatureMatrix
    query: FeatureMatrix
    prefix_input_digest: str
    query_input_digest: str
    builder_source_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.prefix.feature_names != STRICT_PAST_EXPOSURE_FEATURE_NAMES:
            raise StrictPastExposureError("prefix exposure schema differs from the fixed policy")
        if self.query.feature_names != STRICT_PAST_EXPOSURE_FEATURE_NAMES:
            raise StrictPastExposureError("query exposure schema differs from the fixed policy")
        _require_digest(self.prefix_input_digest, name="prefix_input_digest")
        _require_digest(self.query_input_digest, name="query_input_digest")
        _require_digest(self.builder_source_digest, name="builder_source_digest")
        object.__setattr__(self, "digest", _digest(self.manifest()))

    def manifest(self) -> dict[str, object]:
        """Return the full logical contract that produced these numeric matrices."""

        return {
            "schema_version": STRICT_PAST_EXPOSURE_SCHEMA_VERSION,
            "policy": STRICT_PAST_EXPOSURE_POLICY,
            "builder_source_digest": self.builder_source_digest,
            "prefix_input_digest": self.prefix_input_digest,
            "query_input_digest": self.query_input_digest,
            "source": "canonical_impression_inputs_only",
            "source_fields": ["user_id", "video_id", "author_id", "date", "time_ms"],
            "scopes": [{"name": name, "key_fields": list(fields)} for name, fields in _SCOPES],
            "feature_names": list(STRICT_PAST_EXPOSURE_FEATURE_NAMES),
            "timestamp_policy": "date_dominant_timestamp_buckets",
            "time_axis": "date_dominant_chronology",
            "gap_transform": "log1p",
            "unseen_gap_value": 0.0,
            "training_policy": "strictly_earlier_timestamp_buckets_only",
            "equal_timestamp_policy": "simultaneous_rows_are_isolated",
            "query_policy": "earlier_query_inputs_update_later_query_features",
            "outcomes_accepted_by_interface": False,
            "raw_identifiers_exposed_to_candidate": False,
            "prefix": self.prefix.manifest(),
            "query": self.query.manifest(),
        }


def build_strict_past_exposure_pair(
    *,
    prefix_inputs: CanonicalInputs,
    query_inputs: CanonicalInputs,
    builder_source_digest: str,
) -> StrictPastExposurePair:
    """Build strict-earlier input exposure matrices with query-period, input-only warm-up."""

    prefix = _validate_inputs(prefix_inputs, name="prefix_inputs")
    query = _validate_inputs(query_inputs, name="query_inputs")
    builder = _require_digest(builder_source_digest, name="builder_source_digest")
    prefix_order = _order(prefix)
    query_order = _order(query)
    if _date_time(prefix, prefix_order[-1]) >= _date_time(query, query_order[0]):
        raise StrictPastExposureError("query inputs must be strictly later than every prefix input")

    states: list[dict[_ScopeKey, _ExposureState]] = [{} for _ in _SCOPES]
    prefix_matrix = _write_bucket_features(inputs=prefix, order=prefix_order, states=states)
    query_matrix = _write_bucket_features(inputs=query, order=query_order, states=states)
    return StrictPastExposurePair(
        prefix=prefix_matrix,
        query=query_matrix,
        prefix_input_digest=prefix.digest,
        query_input_digest=query.digest,
        builder_source_digest=builder,
    )


__all__ = [
    "STRICT_PAST_EXPOSURE_FEATURE_NAMES",
    "STRICT_PAST_EXPOSURE_POLICY",
    "STRICT_PAST_EXPOSURE_SCHEMA_VERSION",
    "StrictPastExposureError",
    "StrictPastExposurePair",
    "build_strict_past_exposure_pair",
]
