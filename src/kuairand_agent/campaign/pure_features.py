"""Frozen leakage-safe feature table for the bounded KuaiRand-Pure campaign.

The production campaign deliberately keeps this module in the trusted data plane.  Candidate
code receives only the resulting numeric matrices; it never receives canonical targets,
alignment objects, chronology, or raw archive paths. Outcome-bearing aggregates are rebuilt
separately for every temporal fold and frozen at the official-train cutoff. The input-only
exposure family may advance on strictly earlier query inputs without accepting query outcomes.
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

from kuairand_agent.baselines.encoding import STARTER_FIELD_NAMES, StarterEncoding
from kuairand_agent.campaign.strict_past_exposure import (
    STRICT_PAST_EXPOSURE_FEATURE_NAMES,
    StrictPastExposurePair,
    build_strict_past_exposure_pair,
)
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.causal_features import (
    AggregateSpec,
    BuildIdentity,
    CausalInputs,
    FeatureMatrix,
    OutcomeEvents,
    build_causal_feature_pair,
)

PURE_FEATURE_SCHEMA_VERSION: Final = 8
PURE_FEATURE_POLICY: Final = "pure-causal-tree-features-v8"
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
PURE_CLICK_AGGREGATE_SPECS: Final = (
    AggregateSpec(name="click_user", key_fields=("user_id",), smoothing=20.0),
    AggregateSpec(name="click_video", key_fields=("video_id",), smoothing=20.0),
    AggregateSpec(name="click_author", key_fields=("author_id",), smoothing=20.0),
    AggregateSpec(
        name="click_user_author",
        key_fields=("user_id", "author_id"),
        smoothing=20.0,
    ),
)
PURE_WATCH_AGGREGATE_SPECS: Final = (
    AggregateSpec(name="watch_user", key_fields=("user_id",), smoothing=20.0),
    AggregateSpec(name="watch_video", key_fields=("video_id",), smoothing=20.0),
    AggregateSpec(name="watch_author", key_fields=("author_id",), smoothing=20.0),
    AggregateSpec(
        name="watch_user_author",
        key_fields=("user_id", "author_id"),
        smoothing=20.0,
    ),
)
PURE_WATCH_FEATURE_NAMES: Final = (
    "watch_global__mean",
    *(
        name
        for spec in PURE_WATCH_AGGREGATE_SPECS
        for name in (
            f"{spec.resolved_name}__exposure",
            f"{spec.resolved_name}__value_sum",
            f"{spec.resolved_name}__smoothed_mean",
        )
    ),
)
_STATIC_FEATURE_NAMES: Final = (
    "duration_seconds",
    "duration_log1p",
    "duration_at_least_18_seconds",
    "date_offset_from_20220408",
    "tab_numeric",
)
_RECENCY_SCOPES: Final = (
    ("user_recent", ("user_id",)),
    ("video_recent", ("video_id",)),
    ("user_author_recent", ("user_id", "author_id")),
)
_HISTORICAL_RECENCY_FEATURE_NAMES: Final = tuple(
    name
    for scope, _fields in _RECENCY_SCOPES
    for name in (f"{scope}__decayed_exposure", f"{scope}__smoothed_rate")
)
PURE_RECENCY_HALF_LIVES_DAYS: Final = (1.0, 3.0, 7.0)
PURE_RECENCY_FEATURE_NAMES: Final = tuple(
    name
    for scope, _fields in _RECENCY_SCOPES
    for half_life in PURE_RECENCY_HALF_LIVES_DAYS
    for name in (
        f"{scope}_h{int(half_life)}d__decayed_exposure",
        f"{scope}_h{int(half_life)}d__smoothed_rate",
    )
)
PURE_CATEGORICAL_CODE_FEATURE_NAMES: Final = tuple(
    f"starter_fm_code__{name}" for name in STARTER_FIELD_NAMES
)
PURE_VIDEO_TYPE_FEATURE_NAMES: Final = ("video_type_code",)
_HISTORICAL_RECENCY_HALF_LIFE_DAYS: Final = 3.0
_RECENCY_SMOOTHING: Final = 20.0
_MILLISECONDS_PER_DAY: Final = 86_400_000
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
        video_type=tuple(inputs.video_type[position] for position in normalized),
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
    one query stream. Outcome histories retain the same train-frozen state; label-free exposure
    state advances across public inputs before later final inputs. The split boundary remains
    controller-owned and is not exposed to generated candidate source.
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
        video_type=tuple(chain.from_iterable(part.video_type for part in normalized)),
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


def _recency_time_days(inputs: CanonicalInputs) -> np.ndarray:
    dates = np.asarray(inputs.date, dtype=np.float64) - 20220408.0
    milliseconds = np.asarray(inputs.time_ms, dtype=np.int64)
    within_date = np.empty(len(inputs), dtype=np.float64)
    raw_dates = np.asarray(inputs.date, dtype=np.int64)
    for date in np.unique(raw_dates):
        positions = raw_dates == date
        date_values = milliseconds[positions]
        elapsed = (date_values - np.min(date_values)).astype(np.float64)
        # The archive's raw Unix clock can cross a UTC midnight inside one benchmark-local date.
        # Anchor each date at its minimum raw clock and bound the fraction below one so the
        # reviewed date remains the dominant chronology component.
        within_date[positions] = np.minimum(
            elapsed / _MILLISECONDS_PER_DAY,
            np.nextafter(1.0, 0.0),
        )
    return np.ascontiguousarray(dates + within_date)


def _recency_key(inputs: CanonicalInputs, fields: tuple[str, ...], index: int) -> tuple[str, ...]:
    return tuple(str(getattr(inputs, field)[index]) for field in fields)


def _decayed_state(
    state: tuple[float, float, float] | None,
    *,
    at_days: float,
    half_life_days: float,
) -> tuple[float, float]:
    if state is None:
        return (0.0, 0.0)
    last_days, exposure, positive = state
    elapsed = at_days - last_days
    if elapsed < 0.0:
        raise PureFeatureError("recency chronology must be non-decreasing")
    decay = math.exp(-math.log(2.0) * elapsed / half_life_days)
    return (exposure * decay, positive * decay)


def _recency_pair(
    prefix_inputs: CanonicalInputs,
    prefix_labels: Sequence[int],
    query_inputs: CanonicalInputs,
) -> tuple[np.ndarray, np.ndarray]:
    """Build six strict-past recency companions without exposing outcome-bearing state."""

    labels = tuple(prefix_labels)
    prefix_days = _recency_time_days(prefix_inputs)
    order = np.lexsort(
        (
            np.arange(len(prefix_inputs), dtype=np.int64),
            np.asarray(prefix_inputs.time_ms, dtype=np.int64),
            np.asarray(prefix_inputs.date, dtype=np.int64),
        )
    )
    chronological = np.empty(
        (len(prefix_inputs), len(PURE_RECENCY_FEATURE_NAMES)), dtype=np.float64
    )
    recency_states = tuple(
        (fields, half_life)
        for _scope, fields in _RECENCY_SCOPES
        for half_life in PURE_RECENCY_HALF_LIVES_DAYS
    )
    states: list[dict[tuple[str, ...], tuple[float, float, float]]] = [{} for _ in recency_states]
    global_exposure = 0
    global_positive = 0

    start = 0
    while start < len(prefix_inputs):
        canonical = int(order[start])
        bucket_date = prefix_inputs.date[canonical]
        bucket_time = prefix_inputs.time_ms[canonical]
        end = start + 1
        while end < len(prefix_inputs):
            candidate = int(order[end])
            if (
                prefix_inputs.date[candidate] != bucket_date
                or prefix_inputs.time_ms[candidate] != bucket_time
            ):
                break
            end += 1

        at_days = float(prefix_days[canonical])
        prior = 0.5 if global_exposure == 0 else global_positive / global_exposure
        for position in range(start, end):
            row = int(order[position])
            column = 0
            for state_index, (fields, half_life) in enumerate(recency_states):
                key = _recency_key(prefix_inputs, fields, row)
                exposure, positive = _decayed_state(
                    states[state_index].get(key),
                    at_days=at_days,
                    half_life_days=half_life,
                )
                rate = (positive + _RECENCY_SMOOTHING * prior) / (exposure + _RECENCY_SMOOTHING)
                chronological[position, column : column + 2] = (exposure, rate)
                column += 2

        for state_index, (fields, half_life) in enumerate(recency_states):
            deltas: dict[tuple[str, ...], tuple[int, int]] = {}
            for position in range(start, end):
                row = int(order[position])
                key = _recency_key(prefix_inputs, fields, row)
                exposure_delta, positive_delta = deltas.get(key, (0, 0))
                deltas[key] = (exposure_delta + 1, positive_delta + labels[row])
            for key, (exposure_delta, positive_delta) in deltas.items():
                exposure, positive = _decayed_state(
                    states[state_index].get(key),
                    at_days=at_days,
                    half_life_days=half_life,
                )
                states[state_index][key] = (
                    at_days,
                    exposure + exposure_delta,
                    positive + positive_delta,
                )

        global_exposure += end - start
        global_positive += sum(labels[int(order[position])] for position in range(start, end))
        start = end

    prefix = np.empty_like(chronological)
    prefix[order, :] = chronological
    query_days = _recency_time_days(query_inputs)
    query = np.empty((len(query_inputs), len(PURE_RECENCY_FEATURE_NAMES)), dtype=np.float64)
    prior = 0.5 if global_exposure == 0 else global_positive / global_exposure
    for row in range(len(query_inputs)):
        column = 0
        at_days = float(query_days[row])
        for state_index, (fields, half_life) in enumerate(recency_states):
            key = _recency_key(query_inputs, fields, row)
            exposure, positive = _decayed_state(
                states[state_index].get(key),
                at_days=at_days,
                half_life_days=half_life,
            )
            rate = (positive + _RECENCY_SMOOTHING * prior) / (exposure + _RECENCY_SMOOTHING)
            query[row, column : column + 2] = (exposure, rate)
            column += 2
    if not np.isfinite(prefix).all() or not np.isfinite(query).all():
        raise PureFeatureError("recency feature transform produced a non-finite value")
    return (
        np.ascontiguousarray(prefix, dtype=np.float64),
        np.ascontiguousarray(query, dtype=np.float64),
    )


def _augment(
    causal: FeatureMatrix,
    recency: np.ndarray,
    categorical_codes: np.ndarray,
    click_history: FeatureMatrix,
    watch_history: FeatureMatrix,
    video_type_codes: np.ndarray,
    input_exposure: FeatureMatrix,
    inputs: CanonicalInputs,
) -> FeatureMatrix:
    if causal.row_count != len(inputs):
        raise PureFeatureError("causal and static feature rows differ")
    if recency.shape != (len(inputs), len(PURE_RECENCY_FEATURE_NAMES)):
        raise PureFeatureError("recency feature rows or columns differ")
    if categorical_codes.shape != (len(inputs), len(PURE_CATEGORICAL_CODE_FEATURE_NAMES)):
        raise PureFeatureError("categorical-code feature rows or columns differ")
    if click_history.row_count != len(inputs):
        raise PureFeatureError("click-history feature rows differ")
    if watch_history.row_count != len(inputs):
        raise PureFeatureError("watch-history feature rows differ")
    if video_type_codes.shape != (len(inputs), len(PURE_VIDEO_TYPE_FEATURE_NAMES)):
        raise PureFeatureError("video-type feature rows or columns differ")
    if (
        input_exposure.row_count != len(inputs)
        or input_exposure.feature_names != STRICT_PAST_EXPOSURE_FEATURE_NAMES
    ):
        raise PureFeatureError("input-exposure feature rows or columns differ")
    if (
        not np.isfinite(categorical_codes).all()
        or not np.equal(categorical_codes, np.floor(categorical_codes)).all()
        or not np.isfinite(video_type_codes).all()
        or not np.equal(video_type_codes, np.floor(video_type_codes)).all()
    ):
        raise PureFeatureError("categorical-code features must be finite integers")
    values = np.concatenate(
        (
            causal.values,
            recency,
            _static_matrix(inputs),
            categorical_codes,
            click_history.values,
            watch_history.values,
            video_type_codes,
            input_exposure.values,
        ),
        axis=1,
    )
    return FeatureMatrix(
        values,
        (
            *causal.feature_names,
            *PURE_RECENCY_FEATURE_NAMES,
            *_STATIC_FEATURE_NAMES,
            *PURE_CATEGORICAL_CODE_FEATURE_NAMES,
            *click_history.feature_names,
            *watch_history.feature_names,
            *PURE_VIDEO_TYPE_FEATURE_NAMES,
            *input_exposure.feature_names,
        ),
    )


def _video_type_code_pair(
    prefix_inputs: CanonicalInputs,
    query_inputs: CanonicalInputs,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one static basic-video field from the exact temporal-fold prefix vocabulary."""

    categories = sorted(set(prefix_inputs.video_type))
    mapping = {category: index + 1 for index, category in enumerate(categories)}
    prefix = np.fromiter(
        (mapping[value] for value in prefix_inputs.video_type),
        dtype=np.float64,
        count=len(prefix_inputs),
    ).reshape((-1, 1))
    query = np.fromiter(
        (mapping.get(value, 0) for value in query_inputs.video_type),
        dtype=np.float64,
        count=len(query_inputs),
    ).reshape((-1, 1))
    return np.ascontiguousarray(prefix), np.ascontiguousarray(query)


def _click_history_matrix(matrix: FeatureMatrix) -> FeatureMatrix:
    names = (
        "click_global__prior",
        *matrix.feature_names[1:],
    )
    return FeatureMatrix(matrix.values, names)


def _watch_history_pair(
    prefix_inputs: CanonicalInputs,
    prefix_watch_progress: Sequence[float],
    query_inputs: CanonicalInputs,
) -> tuple[FeatureMatrix, FeatureMatrix]:
    """Build strict-past graded watch-progress histories with a frozen query state."""

    progress = tuple(float(value) for value in prefix_watch_progress)
    if len(progress) != len(prefix_inputs):
        raise PureFeatureError("prefix watch progress and inputs must have identical row counts")
    if any(not math.isfinite(value) or not 0.0 <= value <= 2.0 for value in progress):
        raise PureFeatureError("prefix watch progress must be finite in [0, 2]")
    order = np.lexsort(
        (
            np.arange(len(prefix_inputs), dtype=np.int64),
            np.asarray(prefix_inputs.time_ms, dtype=np.int64),
            np.asarray(prefix_inputs.date, dtype=np.int64),
        )
    )
    chronological = np.empty((len(prefix_inputs), len(PURE_WATCH_FEATURE_NAMES)), dtype=np.float64)
    states: list[dict[tuple[str, ...], tuple[int, float]]] = [
        {} for _ in PURE_WATCH_AGGREGATE_SPECS
    ]
    global_count = 0
    global_sum = 0.0
    start = 0
    while start < len(prefix_inputs):
        canonical = int(order[start])
        bucket_date = prefix_inputs.date[canonical]
        bucket_time = prefix_inputs.time_ms[canonical]
        end = start + 1
        while end < len(prefix_inputs):
            candidate = int(order[end])
            if (
                prefix_inputs.date[candidate] != bucket_date
                or prefix_inputs.time_ms[candidate] != bucket_time
            ):
                break
            end += 1

        prior = 0.5 if global_count == 0 else global_sum / global_count
        for position in range(start, end):
            row = int(order[position])
            chronological[position, 0] = prior
            column = 1
            for state_index, spec in enumerate(PURE_WATCH_AGGREGATE_SPECS):
                key = _recency_key(prefix_inputs, spec.key_fields, row)
                count, value_sum = states[state_index].get(key, (0, 0.0))
                mean = (value_sum + spec.smoothing * prior) / (count + spec.smoothing)
                chronological[position, column : column + 3] = (count, value_sum, mean)
                column += 3

        for state_index, spec in enumerate(PURE_WATCH_AGGREGATE_SPECS):
            deltas: dict[tuple[str, ...], tuple[int, float]] = {}
            for position in range(start, end):
                row = int(order[position])
                key = _recency_key(prefix_inputs, spec.key_fields, row)
                count_delta, sum_delta = deltas.get(key, (0, 0.0))
                deltas[key] = (count_delta + 1, sum_delta + progress[row])
            for key, (count_delta, sum_delta) in deltas.items():
                count, value_sum = states[state_index].get(key, (0, 0.0))
                states[state_index][key] = (count + count_delta, value_sum + sum_delta)
        global_count += end - start
        global_sum += sum(progress[int(order[position])] for position in range(start, end))
        start = end

    prefix_values = np.empty_like(chronological)
    prefix_values[order, :] = chronological
    query_values = np.empty((len(query_inputs), len(PURE_WATCH_FEATURE_NAMES)), dtype=np.float64)
    prior = 0.5 if global_count == 0 else global_sum / global_count
    for row in range(len(query_inputs)):
        query_values[row, 0] = prior
        column = 1
        for state_index, spec in enumerate(PURE_WATCH_AGGREGATE_SPECS):
            key = _recency_key(query_inputs, spec.key_fields, row)
            count, value_sum = states[state_index].get(key, (0, 0.0))
            mean = (value_sum + spec.smoothing * prior) / (count + spec.smoothing)
            query_values[row, column : column + 3] = (count, value_sum, mean)
            column += 3
    return (
        FeatureMatrix(prefix_values, PURE_WATCH_FEATURE_NAMES),
        FeatureMatrix(query_values, PURE_WATCH_FEATURE_NAMES),
    )


@dataclass(frozen=True, slots=True)
class PureFeaturePair:
    """One path-independent prefix/query feature artifact and its frozen policy identity."""

    prefix: FeatureMatrix
    query: FeatureMatrix
    dataset_digest: str
    split_role: str
    causal_cache_key: str
    categorical_encoding_digest: str | None = None
    auxiliary_history_cache_key: str | None = None
    input_exposure: StrictPastExposurePair | None = None
    feature_schema_version: int = PURE_FEATURE_SCHEMA_VERSION
    feature_policy: str = PURE_FEATURE_POLICY
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.prefix.feature_names != self.query.feature_names:
            raise PureFeatureError("prefix and query feature schemas differ")
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
        expected_policy = {
            1: "pure-causal-tree-features-v1",
            2: "pure-causal-tree-features-v2",
            3: "pure-causal-tree-features-v3",
            4: "pure-causal-tree-features-v4",
            5: "pure-causal-tree-features-v5",
            6: "pure-causal-tree-features-v6",
            7: "pure-causal-tree-features-v7",
            8: PURE_FEATURE_POLICY,
        }
        if (
            type(self.feature_schema_version) is not int
            or self.feature_schema_version not in expected_policy
            or self.feature_policy != expected_policy[self.feature_schema_version]
        ):
            raise PureFeatureError("feature schema version and policy are incompatible")
        if self.feature_schema_version >= 3:
            if (
                type(self.categorical_encoding_digest) is not str
                or len(self.categorical_encoding_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.categorical_encoding_digest
                )
            ):
                raise PureFeatureError(
                    "schema-v3 categorical_encoding_digest must be a lowercase SHA-256"
                )
        elif self.categorical_encoding_digest is not None:
            raise PureFeatureError(
                "historical feature schemas cannot declare a categorical encoding digest"
            )
        if self.feature_schema_version >= 5:
            if (
                type(self.auxiliary_history_cache_key) is not str
                or len(self.auxiliary_history_cache_key) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.auxiliary_history_cache_key
                )
            ):
                raise PureFeatureError(
                    "schema-v5 auxiliary_history_cache_key must be a lowercase SHA-256"
                )
        elif self.auxiliary_history_cache_key is not None:
            raise PureFeatureError(
                "historical feature schemas cannot declare an auxiliary history cache key"
            )
        if self.feature_schema_version >= 8:
            if not isinstance(self.input_exposure, StrictPastExposurePair):
                raise PureFeatureError(
                    "schema-v8 input_exposure must be a StrictPastExposurePair"
                )
            width = len(STRICT_PAST_EXPOSURE_FEATURE_NAMES)
            if (
                self.prefix.feature_names[-width:] != STRICT_PAST_EXPOSURE_FEATURE_NAMES
                or self.query.feature_names[-width:] != STRICT_PAST_EXPOSURE_FEATURE_NAMES
            ):
                raise PureFeatureError(
                    "schema-v8 feature matrices must end with the fixed input-exposure schema"
                )
            if not np.array_equal(
                self.prefix.values[:, -width:], self.input_exposure.prefix.values
            ) or not np.array_equal(
                self.query.values[:, -width:], self.input_exposure.query.values
            ):
                raise PureFeatureError(
                    "schema-v8 feature matrices do not match input-exposure provenance"
                )
        elif self.input_exposure is not None:
            raise PureFeatureError(
                "historical feature schemas cannot declare input-exposure provenance"
            )
        object.__setattr__(self, "digest", _canonical_digest(self.manifest()))

    def manifest(self) -> dict[str, object]:
        manifest: dict[str, object] = {
            "schema_version": self.feature_schema_version,
            "policy": self.feature_policy,
            "dataset_digest": self.dataset_digest,
            "split_role": self.split_role,
            "aggregate_specs": [spec.manifest() for spec in PURE_AGGREGATE_SPECS],
            "static_features": list(_STATIC_FEATURE_NAMES),
            "causal_cache_key": self.causal_cache_key,
            "prefix": self.prefix.manifest(),
            "query": self.query.manifest(),
        }
        if self.feature_schema_version >= 2:
            recency_feature_names = (
                list(PURE_RECENCY_FEATURE_NAMES)
                if self.feature_schema_version >= 4
                else list(_HISTORICAL_RECENCY_FEATURE_NAMES)
            )
            manifest["recency"] = {
                "scopes": [
                    {"name": name, "key_fields": list(fields)} for name, fields in _RECENCY_SCOPES
                ],
                "feature_names": recency_feature_names,
                "half_life_days": (
                    list(PURE_RECENCY_HALF_LIVES_DAYS)
                    if self.feature_schema_version >= 4
                    else _HISTORICAL_RECENCY_HALF_LIFE_DAYS
                ),
                "smoothing": _RECENCY_SMOOTHING,
                "query_policy": "decay_frozen_prefix_state_without_query_updates",
            }
        if self.feature_schema_version >= 3:
            manifest["categorical_codes"] = {
                "feature_names": list(PURE_CATEGORICAL_CODE_FEATURE_NAMES),
                "field_names": list(STARTER_FIELD_NAMES),
                "encoding_digest": self.categorical_encoding_digest,
                "fit_policy": "fit_on_exact_prefix_in_first_seen_order",
                "query_policy": "prefix_vocab_with_one_unknown_slot_per_field",
                "storage": "exact_integer_ids_represented_as_float64",
                "raw_identifiers_exposed_to_candidate": False,
            }
        if self.feature_schema_version >= 5:
            manifest["auxiliary_history"] = {
                "source_target": "is_click",
                "feature_names": [
                    "click_global__prior",
                    *[
                        name
                        for spec in PURE_CLICK_AGGREGATE_SPECS
                        for name in (
                            f"{spec.resolved_name}__exposure",
                            f"{spec.resolved_name}__positive",
                            f"{spec.resolved_name}__smoothed_rate",
                        )
                    ],
                ],
                "aggregate_specs": [spec.manifest() for spec in PURE_CLICK_AGGREGATE_SPECS],
                "causal_cache_key": self.auxiliary_history_cache_key,
                "training_policy": (
                    "strictly earlier timestamp buckets only; simultaneous rows isolated"
                ),
                "query_policy": "frozen_prefix_state_without_query_updates",
                "same_row_outcome_exposed_to_candidate": False,
                "public_or_final_outcomes_used": False,
            }
        if self.feature_schema_version >= 6:
            manifest["watch_progress_history"] = {
                "source_target": "official_train_play_time_ms_only",
                "feature_names": list(PURE_WATCH_FEATURE_NAMES),
                "aggregate_specs": [spec.manifest() for spec in PURE_WATCH_AGGREGATE_SPECS],
                "transform": ("clip(play_time_ms / max(min(duration_ms, 18000), 1), 0, 2)"),
                "training_policy": (
                    "strictly earlier timestamp buckets only; simultaneous rows isolated"
                ),
                "query_policy": "frozen_prefix_state_without_query_updates",
                "same_row_outcome_exposed_to_candidate": False,
                "public_or_final_outcomes_used": False,
            }
        if self.feature_schema_version >= 7:
            manifest["video_type_code"] = {
                "source": "video_features_basic_pure.csv:video_type",
                "feature_names": list(PURE_VIDEO_TYPE_FEATURE_NAMES),
                "fit_policy": "lexical_categories_from_exact_prefix_only",
                "query_policy": "prefix_vocab_with_zero_unknown_slot",
                "raw_value_exposed_to_candidate": False,
                "public_or_final_outcomes_used": False,
            }
        if self.feature_schema_version >= 8:
            if self.input_exposure is None:  # pragma: no cover - validated above.
                raise PureFeatureError("schema-v8 input-exposure provenance is missing")
            manifest["input_exposure"] = {
                "build_digest": self.input_exposure.digest,
                **self.input_exposure.manifest(),
            }
        return manifest


def build_pure_feature_pair(
    *,
    prefix_inputs: CanonicalInputs,
    prefix_labels: Sequence[int],
    prefix_click_labels: Sequence[int],
    prefix_watch_progress: Sequence[float],
    query_inputs: CanonicalInputs,
    dataset_digest: str,
    split_role: str,
    builder_source_digest: str,
    cache_dir: Path | str | None = None,
) -> PureFeaturePair:
    """Build the fixed table with frozen outcome histories and input-only query warm-up."""

    if not isinstance(prefix_inputs, CanonicalInputs) or not isinstance(
        query_inputs, CanonicalInputs
    ):
        raise PureFeatureError("prefix_inputs and query_inputs must be CanonicalInputs")
    labels = tuple(prefix_labels)
    if len(labels) != len(prefix_inputs):
        raise PureFeatureError("prefix labels and inputs must have identical row counts")
    if any(type(value) is not int or value not in (0, 1) for value in labels):
        raise PureFeatureError("prefix labels must contain binary integers")
    click_labels = tuple(prefix_click_labels)
    if len(click_labels) != len(prefix_inputs):
        raise PureFeatureError("prefix click labels and inputs must have identical row counts")
    if any(type(value) is not int or value not in (0, 1) for value in click_labels):
        raise PureFeatureError("prefix click labels must contain binary integers")
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
    click_history = build_causal_feature_pair(
        prefix_inputs=_causal_inputs(prefix_inputs),
        prefix_outcomes=OutcomeEvents(long_view=click_labels),
        specs=PURE_CLICK_AGGREGATE_SPECS,
        identity=BuildIdentity(
            dataset=dataset_digest,
            split=f"{split_role}:strict-past-is-click",
            field_policy=PURE_FEATURE_POLICY,
            builder_source=builder_source_digest,
        ),
        query_inputs=_causal_inputs(query_inputs),
        cache_dir=cache_dir,
    )
    if click_history.query is None:  # pragma: no cover - query input above makes this defensive.
        raise PureFeatureError("click-history builder did not return query features")
    click_prefix = _click_history_matrix(click_history.prefix)
    click_query = _click_history_matrix(click_history.query)
    watch_prefix, watch_query = _watch_history_pair(
        prefix_inputs,
        prefix_watch_progress,
        query_inputs,
    )
    prefix_video_type_codes, query_video_type_codes = _video_type_code_pair(
        prefix_inputs,
        query_inputs,
    )
    input_exposure = build_strict_past_exposure_pair(
        prefix_inputs=prefix_inputs,
        query_inputs=query_inputs,
        builder_source_digest=builder_source_digest,
    )
    prefix_recency, query_recency = _recency_pair(prefix_inputs, labels, query_inputs)
    categorical_encoding = StarterEncoding.fit(prefix_inputs)
    prefix_categorical_codes = categorical_encoding.transform(prefix_inputs).astype(
        np.float64, copy=False
    )
    query_categorical_codes = categorical_encoding.transform(query_inputs).astype(
        np.float64, copy=False
    )
    return PureFeaturePair(
        prefix=_augment(
            causal.prefix,
            prefix_recency,
            prefix_categorical_codes,
            click_prefix,
            watch_prefix,
            prefix_video_type_codes,
            input_exposure.prefix,
            prefix_inputs,
        ),
        query=_augment(
            causal.query,
            query_recency,
            query_categorical_codes,
            click_query,
            watch_query,
            query_video_type_codes,
            input_exposure.query,
            query_inputs,
        ),
        dataset_digest=dataset_digest,
        split_role=split_role,
        causal_cache_key=causal.cache_key,
        categorical_encoding_digest=categorical_encoding.digest,
        auxiliary_history_cache_key=click_history.cache_key,
        input_exposure=input_exposure,
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
    "PURE_CATEGORICAL_CODE_FEATURE_NAMES",
    "PURE_CLICK_AGGREGATE_SPECS",
    "PURE_FEATURE_POLICY",
    "PURE_FEATURE_SCHEMA_VERSION",
    "PURE_RECENCY_FEATURE_NAMES",
    "PURE_RECENCY_HALF_LIVES_DAYS",
    "PURE_VIDEO_TYPE_FEATURE_NAMES",
    "PURE_WATCH_AGGREGATE_SPECS",
    "PURE_WATCH_FEATURE_NAMES",
    "PureFeatureError",
    "PureFeaturePair",
    "build_pure_feature_pair",
    "concat_canonical_inputs",
    "estimated_matrix_bytes",
    "split_feature_matrix",
    "subset_canonical_inputs",
    "subset_values",
]
