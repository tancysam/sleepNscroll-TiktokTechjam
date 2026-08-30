"""Frozen leakage-safe feature table for the bounded KuaiRand-Pure campaign.

The production campaign deliberately keeps this module in the trusted data plane.  Candidate
code receives only the resulting numeric matrices; it never receives canonical targets,
alignment objects, chronology, or raw archive paths.  Aggregate features are rebuilt separately
for every temporal fold and are frozen at the official-train cutoff for public validation and
final inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain, pairwise
from pathlib import Path
from typing import Final

import numpy as np

from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.causal_features import (
    AggregateSpec,
    BuildIdentity,
    CausalInputs,
    FeatureMatrix,
    OutcomeEvents,
    build_causal_feature_pair,
)

PURE_FEATURE_SCHEMA_VERSION: Final = 1
PURE_FEATURE_POLICY: Final = "pure-causal-tree-features-v1"
PURE_AGGREGATE_SPECS: Final = (
    AggregateSpec(name="user", key_fields=("user_id",), smoothing=20.0),
    AggregateSpec(name="video", key_fields=("video_id",), smoothing=20.0),
    AggregateSpec(name="author", key_fields=("author_id",), smoothing=20.0),
    AggregateSpec(name="tab", key_fields=("tab",), smoothing=20.0),
    AggregateSpec(name="duration_bucket", key_fields=("duration_bucket",), smoothing=20.0),
    AggregateSpec(name="user_author", key_fields=("user_id", "author_id"), smoothing=20.0),
    AggregateSpec(name="user_tab", key_fields=("user_id", "tab"), smoothing=20.0),
    AggregateSpec(name="author_tab", key_fields=("author_id", "tab"), smoothing=20.0),
    AggregateSpec(
        name="user_duration_bucket",
        key_fields=("user_id", "duration_bucket"),
        smoothing=20.0,
    ),
)
_STATIC_FEATURE_NAMES: Final = (
    "duration_seconds",
    "duration_log1p",
    "duration_at_least_18_seconds",
    "date_offset_from_20220408",
    "tab_numeric",
)
# Train-fitted categorical codes.  The aggregate columns above summarize an identity's history;
# these carry the identity itself, so a candidate can learn a per-identity embedding the way the
# organizer FM does instead of only reweighting summaries.
#
# ``user_id`` leads the block.  It was previously omitted on the grounds that a user-constant term
# cannot reorder a within-user ranking, and that candidates already hold user identity through the
# trusted user_groups vector.  Both premises were too narrow.  The organizers measured that purely
# user-side *first-order* terms contribute exactly zero; a user *embedding* is a second-order term,
# and the user-by-video and user-by-author crosses it enables are where the organizer FM gets its
# ordering power.  And user_groups is a training-only capability: the prediction request carries a
# feature matrix and nothing else (candidate_api/runtime_contract.py prediction_request_fields), so
# without this column a candidate cannot evaluate any user-conditioned term at prediction time.
#
# The four original codes remain the trailing four columns, so code that slices a fixed trailing
# width is unaffected by this addition.
ID_CODE_FEATURE_NAMES: Final = (
    "user_id_code",
    "video_id_code",
    "author_id_code",
    "tab_code",
    "duration_bucket_code",
)
_DATE_CHRONOLOGY_STRIDE: Final = 10_000_000_000_000


class PureFeatureError(ValueError):
    """Raised when a production feature request violates the frozen policy."""


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(b"kuairand-pure-feature-pair-v1\0" + encoded).hexdigest()


def subset_canonical_inputs(
    inputs: CanonicalInputs,
    positions: Sequence[int],
) -> CanonicalInputs:
    """Copy one strictly increasing positional subset without accepting row identity as data."""

    if not isinstance(inputs, CanonicalInputs):
        raise PureFeatureError("inputs must be CanonicalInputs")
    normalized = tuple(positions)
    if not normalized:
        raise PureFeatureError("positions must not be empty")
    if any(type(position) is not int for position in normalized):
        raise PureFeatureError("positions must contain integers")
    if any(position < 0 or position >= len(inputs) for position in normalized):
        raise PureFeatureError("positions contain an out-of-range value")
    if any(left >= right for left, right in pairwise(normalized)):
        raise PureFeatureError("positions must be strictly increasing canonical positions")
    return CanonicalInputs(
        user_id=tuple(inputs.user_id[position] for position in normalized),
        video_id=tuple(inputs.video_id[position] for position in normalized),
        date=tuple(inputs.date[position] for position in normalized),
        duration_ms=tuple(inputs.duration_ms[position] for position in normalized),
        tab=tuple(inputs.tab[position] for position in normalized),
        author_id=tuple(inputs.author_id[position] for position in normalized),
        time_ms=tuple(inputs.time_ms[position] for position in normalized),
    )


def subset_values(values: Sequence[int], positions: Sequence[int]) -> tuple[int, ...]:
    """Subset trusted train labels by the same positional capability used for inputs."""

    normalized = tuple(positions)
    if not normalized or any(type(position) is not int for position in normalized):
        raise PureFeatureError("positions must contain at least one integer")
    if any(position < 0 or position >= len(values) for position in normalized):
        raise PureFeatureError("positions contain an out-of-range value")
    result = tuple(values[position] for position in normalized)
    if any(type(value) is not int or value not in (0, 1) for value in result):
        raise PureFeatureError("trusted target subset must remain binary integers")
    return result


def concat_canonical_inputs(parts: Sequence[CanonicalInputs]) -> CanonicalInputs:
    """Concatenate label-free split inputs without synthesizing row or provenance fields.

    The production path uses this once to build public-validation and final-query features from
    the same train-frozen causal state.  The split boundary remains controller-owned and is not
    exposed to generated candidate source.
    """

    normalized = tuple(parts)
    if not normalized:
        raise PureFeatureError("parts must contain at least one CanonicalInputs value")
    if any(not isinstance(part, CanonicalInputs) for part in normalized):
        raise PureFeatureError("parts must contain only CanonicalInputs values")
    return CanonicalInputs(
        user_id=tuple(chain.from_iterable(part.user_id for part in normalized)),
        video_id=tuple(chain.from_iterable(part.video_id for part in normalized)),
        date=tuple(chain.from_iterable(part.date for part in normalized)),
        duration_ms=tuple(chain.from_iterable(part.duration_ms for part in normalized)),
        tab=tuple(chain.from_iterable(part.tab for part in normalized)),
        author_id=tuple(chain.from_iterable(part.author_id for part in normalized)),
        time_ms=tuple(chain.from_iterable(part.time_ms for part in normalized)),
    )


def split_feature_matrix(
    matrix: FeatureMatrix,
    row_counts: Sequence[int],
) -> tuple[FeatureMatrix, ...]:
    """Split one trusted feature matrix into exact contiguous canonical partitions."""

    if not isinstance(matrix, FeatureMatrix):
        raise PureFeatureError("matrix must be a FeatureMatrix")
    normalized = tuple(row_counts)
    if not normalized or any(type(count) is not int or count <= 0 for count in normalized):
        raise PureFeatureError("row_counts must contain positive integers")
    if sum(normalized) != matrix.row_count:
        raise PureFeatureError("row_counts must sum to the feature matrix row count")
    result: list[FeatureMatrix] = []
    start = 0
    for count in normalized:
        stop = start + count
        result.append(FeatureMatrix(matrix.values[start:stop], matrix.feature_names))
        start = stop
    return tuple(result)


def _duration_bucket(milliseconds: float) -> str:
    seconds = milliseconds / 1000.0
    if seconds < 5.0:
        return "lt_5"
    if seconds < 10.0:
        return "5_to_10"
    if seconds < 18.0:
        return "10_to_18"
    if seconds < 30.0:
        return "18_to_30"
    if seconds < 60.0:
        return "30_to_60"
    return "ge_60"


def _causal_inputs(inputs: CanonicalInputs) -> CausalInputs:
    # The archive's millisecond clock has small UTC/local-date overlaps at date boundaries (for
    # example, late 18 April is numerically after early 19 April).  The benchmark fold contract
    # is date-based, so make date the dominant chronology component while preserving the exact
    # within-date millisecond ordering and simultaneous-event equality.  This value is trusted
    # builder state only and is never exposed as a candidate feature.
    chronology = tuple(
        (date - 20220408) * _DATE_CHRONOLOGY_STRIDE + time_ms
        for date, time_ms in zip(inputs.date, inputs.time_ms, strict=True)
    )
    return CausalInputs(
        time_ms=chronology,
        fields={
            "user_id": inputs.user_id,
            "video_id": inputs.video_id,
            "author_id": inputs.author_id,
            "tab": inputs.tab,
            "duration_bucket": tuple(_duration_bucket(value) for value in inputs.duration_ms),
        },
    )


def _static_matrix(inputs: CanonicalInputs) -> np.ndarray:
    duration_seconds = np.asarray(inputs.duration_ms, dtype=np.float64) / 1000.0
    dates = np.asarray(inputs.date, dtype=np.int64)
    tabs = np.asarray(tuple(int(value) for value in inputs.tab), dtype=np.float64)
    values = np.column_stack(
        (
            duration_seconds,
            np.log1p(duration_seconds),
            (duration_seconds >= 18.0).astype(np.float64),
            (dates - 20220408).astype(np.float64),
            tabs,
        )
    )
    if not np.isfinite(values).all():
        raise PureFeatureError("static feature transform produced a non-finite value")
    return np.ascontiguousarray(values, dtype=np.float64)


def _code_source_columns(inputs: CanonicalInputs) -> tuple[tuple[str, ...], ...]:
    """Return the categorical source column for each ID-code feature, in declared order."""

    return (
        tuple(str(value) for value in inputs.user_id),
        tuple(str(value) for value in inputs.video_id),
        tuple(str(value) for value in inputs.author_id),
        tuple(str(value) for value in inputs.tab),
        tuple(_duration_bucket(value) for value in inputs.duration_ms),
    )


def fit_code_vocabulary(inputs: CanonicalInputs) -> tuple[dict[str, int], ...]:
    """Fit one sorted categorical vocabulary per ID-code field.

    Codes are assigned in sorted value order rather than first-seen order.  Simultaneous events
    may arrive in either permutation, and this module guarantees path-independent features, so a
    first-seen assignment would make the matrix depend on input ordering.

    This must only ever see prefix rows.  Fitting on query rows would let a validation or final
    identity claim its own code, which is exactly the leak the frozen-query contract exists to
    prevent.  Values absent from the fitted vocabulary encode to the trailing unknown slot, so
    every code stays inside ``[0, cardinality)`` for both matrices.
    """

    if not isinstance(inputs, CanonicalInputs):
        raise PureFeatureError("inputs must be CanonicalInputs")
    return tuple(
        {value: index for index, value in enumerate(sorted(set(column)))}
        for column in _code_source_columns(inputs)
    )


def code_cardinalities(vocabulary: Sequence[dict[str, int]]) -> tuple[int, ...]:
    """Return each field's code count, including its trailing unknown slot."""

    return tuple(len(table) + 1 for table in vocabulary)


def _code_matrix(
    inputs: CanonicalInputs,
    vocabulary: Sequence[dict[str, int]],
) -> np.ndarray:
    columns = _code_source_columns(inputs)
    if len(vocabulary) != len(columns):
        raise PureFeatureError("code vocabulary does not match the declared categorical fields")
    encoded = tuple(
        np.asarray(
            [table.get(value, len(table)) for value in column],
            dtype=np.float64,
        )
        for column, table in zip(columns, vocabulary, strict=True)
    )
    values = np.column_stack(encoded)
    if not np.isfinite(values).all():
        raise PureFeatureError("categorical code transform produced a non-finite value")
    return np.ascontiguousarray(values, dtype=np.float64)


def _augment(
    causal: FeatureMatrix,
    inputs: CanonicalInputs,
    vocabulary: Sequence[dict[str, int]],
) -> FeatureMatrix:
    if causal.row_count != len(inputs):
        raise PureFeatureError("causal and static feature rows differ")
    values = np.concatenate(
        (causal.values, _static_matrix(inputs), _code_matrix(inputs, vocabulary)),
        axis=1,
    )
    return FeatureMatrix(
        values,
        (*causal.feature_names, *_STATIC_FEATURE_NAMES, *ID_CODE_FEATURE_NAMES),
    )


@dataclass(frozen=True, slots=True)
class PureFeaturePair:
    """One path-independent prefix/query feature artifact and its frozen policy identity."""

    prefix: FeatureMatrix
    query: FeatureMatrix
    dataset_digest: str
    split_role: str
    causal_cache_key: str
    code_cardinalities: tuple[int, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.prefix.feature_names != self.query.feature_names:
            raise PureFeatureError("prefix and query feature schemas differ")
        if type(self.code_cardinalities) is not tuple or any(
            type(value) is not int or value <= 0 for value in self.code_cardinalities
        ):
            raise PureFeatureError("code_cardinalities must be a tuple of positive integers")
        if (
            type(self.dataset_digest) is not str
            or len(self.dataset_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.dataset_digest)
        ):
            raise PureFeatureError("dataset_digest must be a lowercase SHA-256")
        if type(self.split_role) is not str or not self.split_role or "\n" in self.split_role:
            raise PureFeatureError("split_role must be non-empty single-line text")
        if (
            type(self.causal_cache_key) is not str
            or len(self.causal_cache_key) != 64
            or any(character not in "0123456789abcdef" for character in self.causal_cache_key)
        ):
            raise PureFeatureError("causal_cache_key must be a lowercase SHA-256")
        object.__setattr__(self, "digest", _canonical_digest(self.manifest()))

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": PURE_FEATURE_SCHEMA_VERSION,
            "policy": PURE_FEATURE_POLICY,
            "dataset_digest": self.dataset_digest,
            "split_role": self.split_role,
            "aggregate_specs": [spec.manifest() for spec in PURE_AGGREGATE_SPECS],
            "static_features": list(_STATIC_FEATURE_NAMES),
            "id_code_features": list(ID_CODE_FEATURE_NAMES),
            "code_cardinalities": list(self.code_cardinalities),
            "causal_cache_key": self.causal_cache_key,
            "prefix": self.prefix.manifest(),
            "query": self.query.manifest(),
        }


def build_pure_feature_pair(
    *,
    prefix_inputs: CanonicalInputs,
    prefix_labels: Sequence[int],
    query_inputs: CanonicalInputs,
    dataset_digest: str,
    split_role: str,
    builder_source_digest: str,
    cache_dir: Path | str | None = None,
) -> PureFeaturePair:
    """Build the fixed feature table with strict-past prefix and frozen-query semantics."""

    if not isinstance(prefix_inputs, CanonicalInputs) or not isinstance(
        query_inputs, CanonicalInputs
    ):
        raise PureFeatureError("prefix_inputs and query_inputs must be CanonicalInputs")
    labels = tuple(prefix_labels)
    if len(labels) != len(prefix_inputs):
        raise PureFeatureError("prefix labels and inputs must have identical row counts")
    if any(type(value) is not int or value not in (0, 1) for value in labels):
        raise PureFeatureError("prefix labels must contain binary integers")
    if (
        type(builder_source_digest) is not str
        or len(builder_source_digest) != 64
        or any(character not in "0123456789abcdef" for character in builder_source_digest)
    ):
        raise PureFeatureError("builder_source_digest must be a lowercase SHA-256")
    causal = build_causal_feature_pair(
        prefix_inputs=_causal_inputs(prefix_inputs),
        prefix_outcomes=OutcomeEvents(long_view=labels),
        specs=PURE_AGGREGATE_SPECS,
        identity=BuildIdentity(
            dataset=dataset_digest,
            split=split_role,
            field_policy=PURE_FEATURE_POLICY,
            builder_source=builder_source_digest,
        ),
        query_inputs=_causal_inputs(query_inputs),
        cache_dir=cache_dir,
    )
    if causal.query is None:  # pragma: no cover - query input above makes this defensive.
        raise PureFeatureError("causal builder did not return query features")
    # Fitted on prefix rows only, then applied unchanged to the query matrix, so a query-only
    # identity encodes to its field's unknown slot rather than gaining a code of its own.
    vocabulary = fit_code_vocabulary(prefix_inputs)
    return PureFeaturePair(
        prefix=_augment(causal.prefix, prefix_inputs, vocabulary),
        query=_augment(causal.query, query_inputs, vocabulary),
        dataset_digest=dataset_digest,
        split_role=split_role,
        causal_cache_key=causal.cache_key,
        code_cardinalities=code_cardinalities(vocabulary),
    )


def estimated_matrix_bytes(row_count: int, feature_count: int) -> int:
    """Return the exact float64 payload size used for admission/resource reporting."""

    if type(row_count) is not int or type(feature_count) is not int:
        raise PureFeatureError("row_count and feature_count must be integers")
    if row_count <= 0 or feature_count <= 0:
        raise PureFeatureError("row_count and feature_count must be positive")
    size = row_count * feature_count * np.dtype(np.float64).itemsize
    if not math.isfinite(float(size)):
        raise PureFeatureError("matrix size is not finite")
    return size


__all__ = [
    "PURE_AGGREGATE_SPECS",
    "PURE_FEATURE_POLICY",
    "PURE_FEATURE_SCHEMA_VERSION",
    "PureFeatureError",
    "PureFeaturePair",
    "build_pure_feature_pair",
    "concat_canonical_inputs",
    "estimated_matrix_bytes",
    "split_feature_matrix",
    "subset_canonical_inputs",
    "subset_values",
]
