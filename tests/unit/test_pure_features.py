from __future__ import annotations

import numpy as np
import pytest

from kuairand_agent.campaign.pure_features import (
    PURE_AGGREGATE_SPECS,
    PureFeatureError,
    build_pure_feature_pair,
    concat_canonical_inputs,
    estimated_matrix_bytes,
    split_feature_matrix,
    subset_canonical_inputs,
    subset_values,
)
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.causal_features import FeatureMatrix


def _inputs(*, times: tuple[int, ...], suffix: str = "") -> CanonicalInputs:
    count = len(times)
    return CanonicalInputs(
        user_id=tuple(("u1", "u1", "u2", "u2")[:count]),
        video_id=tuple(f"v{index}{suffix}" for index in range(count)),
        date=tuple(20220408 if time < 30 else 20220409 for time in times),
        duration_ms=tuple((4_000.0, 18_000.0, 30_000.0, 61_000.0)[:count]),
        tab=tuple(("0", "1", "0", "1")[:count]),
        author_id=tuple(("a1", "a1", "a2", "a2")[:count]),
        time_ms=times,
    )


def test_subset_is_positional_and_never_introduces_row_identity() -> None:
    inputs = _inputs(times=(10, 20, 30, 40))
    subset = subset_canonical_inputs(inputs, (0, 2, 3))

    assert subset.user_id == ("u1", "u2", "u2")
    assert subset.time_ms == (10, 30, 40)
    assert "row_id" not in subset.field_names
    assert subset_values((1, 0, 1, 0), (0, 2, 3)) == (1, 1, 0)

    with pytest.raises(PureFeatureError, match="strictly increasing"):
        subset_canonical_inputs(inputs, (2, 1))
    with pytest.raises(PureFeatureError, match="out-of-range"):
        subset_values((1, 0), (2,))


def test_concat_and_split_preserve_canonical_order_and_exact_feature_identity() -> None:
    first = _inputs(times=(10, 20), suffix="-first")
    second = CanonicalInputs(
        user_id=("u3",),
        video_id=("v-final",),
        date=(20220422,),
        duration_ms=(25_000.0,),
        tab=("2",),
        author_id=("a3",),
        time_ms=(5,),
    )

    combined = concat_canonical_inputs((first, second))

    assert combined.user_id == ("u1", "u1", "u3")
    assert combined.video_id == ("v0-first", "v1-first", "v-final")
    assert combined.date == (20220408, 20220408, 20220422)
    assert "row_id" not in combined.field_names

    matrix = FeatureMatrix(
        np.asarray(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))),
        ("safe_a", "safe_b"),
    )
    left, right = split_feature_matrix(matrix, (2, 1))

    np.testing.assert_array_equal(left.values, matrix.values[:2])
    np.testing.assert_array_equal(right.values, matrix.values[2:])
    assert left.feature_names == right.feature_names == matrix.feature_names
    assert split_feature_matrix(matrix, (3,))[0].digest == matrix.digest


def test_concat_and_split_fail_closed_on_ambiguous_partition_requests() -> None:
    inputs = _inputs(times=(10, 20))
    matrix = FeatureMatrix(np.asarray(((1.0,), (2.0,))), ("safe",))

    with pytest.raises(PureFeatureError, match="at least one"):
        concat_canonical_inputs(())
    with pytest.raises(PureFeatureError, match="CanonicalInputs"):
        concat_canonical_inputs((inputs, object()))  # type: ignore[arg-type]
    with pytest.raises(PureFeatureError, match="positive integers"):
        split_feature_matrix(matrix, (1, 0, 1))
    with pytest.raises(PureFeatureError, match="sum"):
        split_feature_matrix(matrix, (1,))


def test_feature_pair_has_frozen_schema_and_strict_past_query_state() -> None:
    prefix = _inputs(times=(10, 20))
    query = CanonicalInputs(
        user_id=("u1", "u3"),
        video_id=("new1", "new2"),
        date=(20220409, 20220409),
        duration_ms=(18_000.0, 60_000.0),
        tab=("1", "2"),
        author_id=("a1", "a3"),
        time_ms=(30, 40),
    )
    pair = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1, 0),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )

    assert pair.prefix.row_count == 2
    assert pair.query.row_count == 2
    assert pair.prefix.feature_count == 1 + 3 * len(PURE_AGGREGATE_SPECS) + 5
    assert pair.prefix.feature_names == pair.query.feature_names
    assert "duration_at_least_18_seconds" in pair.query.feature_names
    assert all("row_id" not in name for name in pair.query.feature_names)
    assert np.isfinite(pair.prefix.values).all()
    assert np.isfinite(pair.query.values).all()


def test_date_fold_boundary_dominates_small_raw_clock_overlap() -> None:
    prefix = CanonicalInputs(
        user_id=("u",),
        video_id=("p",),
        date=(20220418,),
        duration_ms=(10_000.0,),
        tab=("0",),
        author_id=("a",),
        time_ms=(1_650_297_116_773,),
    )
    query = CanonicalInputs(
        user_id=("u",),
        video_id=("q",),
        date=(20220419,),
        duration_ms=(10_000.0,),
        tab=("0",),
        author_id=("a",),
        time_ms=(1_650_295_266_482,),
    )

    result = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1,),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-b",
        builder_source_digest="b" * 64,
    )

    assert result.query.row_count == 1


def test_query_features_do_not_change_when_only_unavailable_query_outcomes_change() -> None:
    prefix = _inputs(times=(10, 20))
    query = CanonicalInputs(
        user_id=("u1", "u1"),
        video_id=("q1", "q2"),
        date=(20220409, 20220409),
        duration_ms=(10_000.0, 20_000.0),
        tab=("0", "0"),
        author_id=("a1", "a1"),
        time_ms=(30, 40),
    )
    ordinary = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1, 0),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-b",
        builder_source_digest="b" * 64,
    )
    # There is intentionally no query-label argument.  A caller cannot feed either of these
    # hypothetical vectors into the builder, so changing them cannot change the artifact.
    unavailable_query_outcomes = (0, 1)
    mutated_unavailable_query_outcomes = (1, 0)
    assert unavailable_query_outcomes != mutated_unavailable_query_outcomes
    replay = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1, 0),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-b",
        builder_source_digest="b" * 64,
    )
    np.testing.assert_array_equal(ordinary.query.values, replay.query.values)
    assert ordinary.digest == replay.digest


def test_equal_timestamp_permutation_is_invariant_after_inverse_scatter() -> None:
    first = CanonicalInputs(
        user_id=("u", "v", "u"),
        video_id=("a", "b", "c"),
        date=(20220408, 20220408, 20220408),
        duration_ms=(10_000.0, 20_000.0, 30_000.0),
        tab=("0", "1", "0"),
        author_id=("x", "y", "x"),
        time_ms=(10, 10, 20),
    )
    second = CanonicalInputs(
        user_id=("v", "u", "u"),
        video_id=("b", "a", "c"),
        date=(20220408, 20220408, 20220408),
        duration_ms=(20_000.0, 10_000.0, 30_000.0),
        tab=("1", "0", "0"),
        author_id=("y", "x", "x"),
        time_ms=(10, 10, 20),
    )
    query = _inputs(times=(30, 40), suffix="-q")
    left = build_pure_feature_pair(
        prefix_inputs=first,
        prefix_labels=(1, 0, 1),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )
    right = build_pure_feature_pair(
        prefix_inputs=second,
        prefix_labels=(0, 1, 1),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )

    np.testing.assert_array_equal(left.prefix.values, right.prefix.values[[1, 0, 2]])
    np.testing.assert_array_equal(left.query.values, right.query.values)


def test_matrix_size_admission_is_exact_and_validated() -> None:
    assert estimated_matrix_bytes(1_000, 33) == 264_000
    with pytest.raises(PureFeatureError):
        estimated_matrix_bytes(0, 33)
