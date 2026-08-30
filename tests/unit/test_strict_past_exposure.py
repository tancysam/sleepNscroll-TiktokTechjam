from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from kuairand_agent.campaign.strict_past_exposure import (
    STRICT_PAST_EXPOSURE_FEATURE_NAMES,
    StrictPastExposureError,
    StrictPastExposurePair,
    build_strict_past_exposure_pair,
)
from kuairand_agent.data.canonical import CanonicalInputs


def _inputs(
    *,
    users: tuple[str, ...],
    videos: tuple[str, ...],
    authors: tuple[str, ...],
    dates: tuple[int, ...],
    times: tuple[int, ...],
) -> CanonicalInputs:
    count = len(users)
    return CanonicalInputs(
        user_id=users,
        video_id=videos,
        date=dates,
        duration_ms=(10_000.0,) * count,
        tab=("0",) * count,
        author_id=authors,
        time_ms=times,
    )


def _column(
    pair: StrictPastExposurePair,
    name: str,
    *,
    query: bool,
) -> npt.NDArray[np.float64]:
    matrix = pair.query if query else pair.prefix
    return matrix.values[:, matrix.feature_names.index(name)]


def test_builds_input_only_strict_earlier_features_for_all_fixed_scopes() -> None:
    prefix = _inputs(
        users=("u1", "u2", "u1"),
        videos=("v1", "v1", "v2"),
        authors=("a1", "a1", "a2"),
        dates=(20220408, 20220408, 20220408),
        times=(10, 20, 30),
    )
    query = _inputs(
        users=("u1", "u3"),
        videos=("v1", "v3"),
        authors=("a1", "a3"),
        dates=(20220409, 20220409),
        times=(40, 50),
    )

    pair = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=query,
        builder_source_digest="a" * 64,
    )

    assert pair.prefix.feature_names == STRICT_PAST_EXPOSURE_FEATURE_NAMES
    assert pair.query.feature_names == STRICT_PAST_EXPOSURE_FEATURE_NAMES
    assert pair.prefix.values.flags.writeable is False
    assert pair.query.values.flags.writeable is False
    np.testing.assert_array_equal(
        _column(pair, "user__strict_earlier_exposure_count", query=False),
        (0.0, 0.0, 1.0),
    )
    np.testing.assert_array_equal(
        _column(pair, "video__strict_earlier_exposure_count", query=False),
        (0.0, 1.0, 0.0),
    )
    np.testing.assert_array_equal(
        _column(pair, "author__strict_earlier_exposure_count", query=False),
        (0.0, 1.0, 0.0),
    )
    np.testing.assert_array_equal(
        _column(pair, "user_video__strict_earlier_exposure_count", query=False),
        (0.0, 0.0, 0.0),
    )
    np.testing.assert_array_equal(
        _column(pair, "user__first_seen", query=True),
        (0.0, 1.0),
    )
    assert _column(pair, "user__log1p_time_since_last_exposure", query=True)[0] > 0.0
    assert _column(pair, "user__log1p_time_since_last_exposure", query=True)[1] == 0.0

    manifest = pair.manifest()
    assert manifest["outcomes_accepted_by_interface"] is False
    assert manifest["query_policy"] == "earlier_query_inputs_update_later_query_features"
    assert manifest["timestamp_policy"] == "date_dominant_timestamp_buckets"
    assert manifest["prefix_input_digest"] == prefix.digest
    assert manifest["query_input_digest"] == query.digest


def test_equal_timestamps_are_isolated_and_earlier_query_inputs_warm_later_ones() -> None:
    prefix = _inputs(
        users=("u",),
        videos=("v0",),
        authors=("a",),
        dates=(20220408,),
        times=(10,),
    )
    query = _inputs(
        users=("u", "u", "u", "u"),
        videos=("v1", "v2", "v3", "v4"),
        authors=("a", "a", "a", "a"),
        dates=(20220409, 20220409, 20220409, 20220409),
        times=(20, 30, 30, 40),
    )

    pair = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=query,
        builder_source_digest="b" * 64,
    )

    np.testing.assert_array_equal(
        _column(pair, "user__strict_earlier_exposure_count", query=True),
        (1.0, 2.0, 2.0, 4.0),
    )
    np.testing.assert_array_equal(
        _column(pair, "user_video__strict_earlier_exposure_count", query=True),
        (0.0, 0.0, 0.0, 0.0),
    )


def test_date_dominates_raw_clock_overlap_at_the_prefix_query_boundary() -> None:
    prefix = _inputs(
        users=("u",),
        videos=("v",),
        authors=("a",),
        dates=(20220408,),
        times=(1_000_000,),
    )
    query = _inputs(
        users=("u",),
        videos=("q",),
        authors=("a",),
        dates=(20220409,),
        times=(1,),
    )

    pair = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=query,
        builder_source_digest="b" * 64,
    )

    assert _column(pair, "user__strict_earlier_exposure_count", query=True)[0] == 1.0
    assert _column(pair, "user__log1p_time_since_last_exposure", query=True)[0] > 0.0


def test_is_permutation_invariant_inside_an_equal_timestamp_bucket() -> None:
    prefix = _inputs(
        users=("u",),
        videos=("v0",),
        authors=("a",),
        dates=(20220408,),
        times=(10,),
    )
    left = _inputs(
        users=("u", "x", "u"),
        videos=("v1", "v2", "v3"),
        authors=("a", "b", "a"),
        dates=(20220409, 20220409, 20220409),
        times=(20, 20, 30),
    )
    right = _inputs(
        users=("x", "u", "u"),
        videos=("v2", "v1", "v3"),
        authors=("b", "a", "a"),
        dates=(20220409, 20220409, 20220409),
        times=(20, 20, 30),
    )

    first = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=left,
        builder_source_digest="c" * 64,
    )
    second = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=right,
        builder_source_digest="c" * 64,
    )

    np.testing.assert_array_equal(first.query.values, second.query.values[[1, 0, 2]])


def test_fails_closed_on_prefix_query_order_and_has_no_outcome_parameter() -> None:
    prefix = _inputs(
        users=("u",),
        videos=("v",),
        authors=("a",),
        dates=(20220409,),
        times=(20,),
    )
    too_early_query = _inputs(
        users=("u",),
        videos=("q",),
        authors=("a",),
        dates=(20220409,),
        times=(20,),
    )

    with pytest.raises(StrictPastExposureError, match="strictly later"):
        build_strict_past_exposure_pair(
            prefix_inputs=prefix,
            query_inputs=too_early_query,
            builder_source_digest="d" * 64,
        )

    parameters = inspect.signature(build_strict_past_exposure_pair).parameters
    assert set(parameters) == {"prefix_inputs", "query_inputs", "builder_source_digest"}
    unsafe_builder = cast(Callable[..., object], build_strict_past_exposure_pair)
    with pytest.raises(TypeError, match="query_outcomes"):
        unsafe_builder(
            prefix_inputs=prefix,
            query_inputs=_inputs(
                users=("u",),
                videos=("q",),
                authors=("a",),
                dates=(20220410,),
                times=(30,),
            ),
            builder_source_digest="d" * 64,
            query_outcomes=(1,),
        )


def test_digest_is_deterministic_and_source_identity_is_provenance() -> None:
    prefix = _inputs(
        users=("u",),
        videos=("v",),
        authors=("a",),
        dates=(20220408,),
        times=(10,),
    )
    query = _inputs(
        users=("u",),
        videos=("q",),
        authors=("a",),
        dates=(20220409,),
        times=(20,),
    )

    first = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=query,
        builder_source_digest="e" * 64,
    )
    replay = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=query,
        builder_source_digest="e" * 64,
    )
    changed_source = build_strict_past_exposure_pair(
        prefix_inputs=prefix,
        query_inputs=query,
        builder_source_digest="f" * 64,
    )

    assert first.digest == replay.digest
    assert first.manifest() == replay.manifest()
    assert first.digest != changed_source.digest
    assert first.manifest()["builder_source_digest"] == "e" * 64
