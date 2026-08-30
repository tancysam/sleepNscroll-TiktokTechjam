from __future__ import annotations

import numpy as np
import pytest

from kuairand_agent.campaign.pure_features import (
    PURE_AGGREGATE_SPECS,
    PURE_CATEGORICAL_CODE_FEATURE_NAMES,
    PURE_CLICK_AGGREGATE_SPECS,
    PURE_RECENCY_FEATURE_NAMES,
    PURE_VIDEO_TYPE_FEATURE_NAMES,
    PURE_WATCH_FEATURE_NAMES,
    PureFeatureError,
    PureFeaturePair,
    build_pure_feature_pair,
    concat_canonical_inputs,
    estimated_matrix_bytes,
    split_feature_matrix,
    subset_canonical_inputs,
    subset_values,
)
from kuairand_agent.campaign.strict_past_exposure import (
    STRICT_PAST_EXPOSURE_FEATURE_NAMES,
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
        prefix_click_labels=(1, 1),
        prefix_watch_progress=(1.2, 0.4),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )

    assert pair.prefix.row_count == 2
    assert pair.query.row_count == 2
    assert pair.prefix.feature_count == (
        1
        + 3 * len(PURE_AGGREGATE_SPECS)
        + len(PURE_RECENCY_FEATURE_NAMES)
        + 5
        + len(PURE_CATEGORICAL_CODE_FEATURE_NAMES)
        + 1
        + 3 * len(PURE_CLICK_AGGREGATE_SPECS)
        + len(PURE_WATCH_FEATURE_NAMES)
        + len(PURE_VIDEO_TYPE_FEATURE_NAMES)
        + len(STRICT_PAST_EXPOSURE_FEATURE_NAMES)
    )
    assert pair.prefix.feature_names == pair.query.feature_names
    assert "duration_at_least_18_seconds" in pair.query.feature_names
    assert pair.query.feature_names.index("video_type_code") == 82
    assert pair.query.feature_names[-12:] == STRICT_PAST_EXPOSURE_FEATURE_NAMES
    assert all("row_id" not in name for name in pair.query.feature_names)
    assert np.isfinite(pair.prefix.values).all()
    assert np.isfinite(pair.query.values).all()

    user_recency = pair.query.feature_names.index("user_recent_h3d__decayed_exposure")
    video_recency = pair.query.feature_names.index("video_recent_h3d__decayed_exposure")
    assert 1.0 < pair.query.values[0, user_recency] < 2.0
    assert pair.query.values[1, user_recency] == 0.0
    assert pair.query.values[0, video_recency] == 0.0
    user_code = pair.query.feature_names.index("starter_fm_code__user_id")
    video_code = pair.query.feature_names.index("starter_fm_code__video_id")
    assert pair.query.values[0, user_code] == pair.prefix.values[0, user_code]
    assert pair.query.values[1, user_code] == 1.0
    assert pair.query.values[0, video_code] == pair.query.values[1, video_code]
    assert pair.categorical_encoding_digest is not None
    one_day = pair.query.feature_names.index("user_recent_h1d__decayed_exposure")
    seven_day = pair.query.feature_names.index("user_recent_h7d__decayed_exposure")
    assert (
        pair.query.values[0, one_day]
        < pair.query.values[0, user_recency]
        < pair.query.values[0, seven_day]
    )
    assert pair.manifest()["recency"]["half_life_days"] == [1.0, 3.0, 7.0]  # type: ignore[index]
    input_exposure = pair.manifest()["input_exposure"]
    assert input_exposure["build_digest"] == pair.input_exposure.digest  # type: ignore[index,union-attr]
    assert input_exposure["outcomes_accepted_by_interface"] is False  # type: ignore[index]
    assert (
        input_exposure["query_policy"]  # type: ignore[index]
        == "earlier_query_inputs_update_later_query_features"
    )


def test_input_exposure_advances_query_inputs_while_outcome_histories_remain_frozen() -> None:
    prefix = CanonicalInputs(
        user_id=("u",),
        video_id=("v",),
        date=(20220408,),
        duration_ms=(10_000.0,),
        tab=("0",),
        author_id=("a",),
        time_ms=(10,),
    )
    query = CanonicalInputs(
        user_id=("u", "u"),
        video_id=("v", "v"),
        date=(20220409, 20220409),
        duration_ms=(10_000.0, 10_000.0),
        tab=("0", "0"),
        author_id=("a", "a"),
        time_ms=(20, 30),
    )

    pair = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1,),
        prefix_click_labels=(1,),
        prefix_watch_progress=(1.2,),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="outer-validation-plus-final",
        builder_source_digest="b" * 64,
    )

    exposure = pair.query.feature_names.index("user__strict_earlier_exposure_count")
    click_positive = pair.query.feature_names.index("click_user__positive")
    watch_sum = pair.query.feature_names.index("watch_user__value_sum")
    assert pair.query.values[:, exposure].tolist() == [1.0, 2.0]
    assert pair.query.values[:, click_positive].tolist() == [1.0, 1.0]
    assert pair.query.values[:, watch_sum].tolist() == [1.2, 1.2]


def test_video_type_extension_remains_at_column_82_in_schema_v8_identity() -> None:
    normal = CanonicalInputs(
        user_id=("u1", "u2"),
        video_id=("v1", "v2"),
        date=(20220408, 20220408),
        duration_ms=(10_000.0, 20_000.0),
        tab=("0", "1"),
        author_id=("a1", "a2"),
        time_ms=(1, 2),
        video_type=("NORMAL", "NORMAL"),
    )
    mixed = CanonicalInputs(
        user_id=("u1", "u2"),
        video_id=("v1", "v2"),
        date=(20220408, 20220408),
        duration_ms=(10_000.0, 20_000.0),
        tab=("0", "1"),
        author_id=("a1", "a2"),
        time_ms=(1, 2),
        video_type=("NORMAL", "AD"),
    )
    query = CanonicalInputs(
        user_id=("u3",),
        video_id=("v3",),
        date=(20220409,),
        duration_ms=(30_000.0,),
        tab=("2",),
        author_id=("a3",),
        time_ms=(3,),
        video_type=("NORMAL",),
    )

    def build(prefix: CanonicalInputs) -> PureFeaturePair:
        return build_pure_feature_pair(
            prefix_inputs=prefix,
            prefix_labels=(1, 0),
            prefix_click_labels=(1, 0),
            prefix_watch_progress=(1.0, 0.5),
            query_inputs=query,
            dataset_digest="a" * 64,
            split_role="fold-a",
            builder_source_digest="b" * 64,
        )

    normal_pair = build(normal)
    mixed_pair = build(mixed)

    assert normal.digest == mixed.digest  # Frozen qualification-compatible v1 identity.
    assert normal_pair.prefix.digest != mixed_pair.prefix.digest
    assert normal_pair.digest != mixed_pair.digest
    video_type = normal_pair.prefix.feature_names.index("video_type_code")
    without_video_type = tuple(
        index for index in range(normal_pair.prefix.feature_count) if index != video_type
    )
    np.testing.assert_array_equal(
        normal_pair.prefix.values[:, without_video_type],
        mixed_pair.prefix.values[:, without_video_type],
    )
    assert normal_pair.prefix.values[:, video_type].tolist() == [1.0, 1.0]
    assert mixed_pair.prefix.values[:, video_type].tolist() == [2.0, 1.0]


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
        prefix_click_labels=(1,),
        prefix_watch_progress=(1.2,),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-b",
        builder_source_digest="b" * 64,
    )

    assert result.query.row_count == 1


def test_recency_clock_does_not_wrap_at_utc_midnight_inside_one_local_date() -> None:
    prefix = CanonicalInputs(
        user_id=("u", "u"),
        video_id=("p1", "p2"),
        date=(20220418, 20220418),
        duration_ms=(10_000.0, 10_000.0),
        tab=("0", "0"),
        author_id=("a", "a"),
        time_ms=(1_650_326_399_000, 1_650_326_401_000),
    )
    query = CanonicalInputs(
        user_id=("u",),
        video_id=("q",),
        date=(20220419,),
        duration_ms=(10_000.0,),
        tab=("0",),
        author_id=("a",),
        time_ms=(1_650_326_402_000,),
    )

    result = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1, 0),
        prefix_click_labels=(1, 1),
        prefix_watch_progress=(1.2, 0.4),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-b",
        builder_source_digest="b" * 64,
    )

    index = result.query.feature_names.index("user_recent_h3d__decayed_exposure")
    assert 1.0 < result.query.values[0, index] < 2.0


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
        prefix_click_labels=(1, 0),
        prefix_watch_progress=(1.2, 0.4),
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
        prefix_click_labels=(1, 0),
        prefix_watch_progress=(1.2, 0.4),
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
        prefix_click_labels=(1, 0, 1),
        prefix_watch_progress=(1.2, 0.4, 1.5),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )
    right = build_pure_feature_pair(
        prefix_inputs=second,
        prefix_labels=(0, 1, 1),
        prefix_click_labels=(0, 1, 1),
        prefix_watch_progress=(0.4, 1.2, 1.5),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )

    # Causal long-view/click history and recency are permutation-invariant after inverse
    # scatter. The five organizer-compatible categorical codes deliberately retain the starter
    # encoder's first-seen physical-order vocabulary semantics, so omit only those columns.
    non_categorical = tuple(
        index
        for index, name in enumerate(left.prefix.feature_names)
        if name not in PURE_CATEGORICAL_CODE_FEATURE_NAMES
    )
    np.testing.assert_array_equal(
        left.prefix.values[:, non_categorical],
        right.prefix.values[[1, 0, 2]][:, non_categorical],
    )
    np.testing.assert_array_equal(
        left.query.values[:, non_categorical],
        right.query.values[:, non_categorical],
    )


def test_click_history_uses_its_own_strict_past_target_and_frozen_query_state() -> None:
    prefix = CanonicalInputs(
        user_id=("u", "u", "u"),
        video_id=("v1", "v2", "v3"),
        date=(20220408, 20220408, 20220408),
        duration_ms=(10_000.0, 10_000.0, 10_000.0),
        tab=("0", "0", "0"),
        author_id=("a", "a", "a"),
        time_ms=(10, 10, 20),
    )
    query = CanonicalInputs(
        user_id=("u", "u"),
        video_id=("q1", "q2"),
        date=(20220409, 20220409),
        duration_ms=(10_000.0, 10_000.0),
        tab=("0", "0"),
        author_id=("a", "a"),
        time_ms=(30, 40),
    )
    pair = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(0, 0, 0),
        prefix_click_labels=(1, 0, 1),
        prefix_watch_progress=(1.2, 0.4, 1.5),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )

    click_positive = pair.prefix.feature_names.index("click_user__positive")
    long_view_positive = pair.prefix.feature_names.index("user__positive")
    # Simultaneous rows cannot see one another; the later row sees both earlier clicks.
    assert pair.prefix.values[0, click_positive] == 0.0
    assert pair.prefix.values[1, click_positive] == 0.0
    assert pair.prefix.values[2, click_positive] == 1.0
    assert pair.prefix.values[2, long_view_positive] == 0.0
    # Query rows share one frozen prefix state and cannot update each other.
    assert pair.query.values[0, click_positive] == 2.0
    assert pair.query.values[1, click_positive] == 2.0


def test_watch_progress_is_graded_strict_past_and_query_frozen() -> None:
    prefix = CanonicalInputs(
        user_id=("u", "u", "u"),
        video_id=("v1", "v2", "v3"),
        date=(20220408, 20220408, 20220408),
        duration_ms=(10_000.0, 10_000.0, 10_000.0),
        tab=("0", "0", "0"),
        author_id=("a", "a", "a"),
        time_ms=(10, 10, 20),
    )
    query = CanonicalInputs(
        user_id=("u", "u"),
        video_id=("q1", "q2"),
        date=(20220409, 20220409),
        duration_ms=(10_000.0, 10_000.0),
        tab=("0", "0"),
        author_id=("a", "a"),
        time_ms=(30, 40),
    )
    pair = build_pure_feature_pair(
        prefix_inputs=prefix,
        prefix_labels=(1, 0, 1),
        prefix_click_labels=(1, 0, 1),
        prefix_watch_progress=(1.5, 0.25, 1.1),
        query_inputs=query,
        dataset_digest="a" * 64,
        split_role="fold-a",
        builder_source_digest="b" * 64,
    )

    user_sum = pair.prefix.feature_names.index("watch_user__value_sum")
    global_mean = pair.prefix.feature_names.index("watch_global__mean")
    # The simultaneous rows cannot see one another. The later row sees both graded values.
    assert pair.prefix.values[0, user_sum] == 0.0
    assert pair.prefix.values[1, user_sum] == 0.0
    assert pair.prefix.values[2, user_sum] == 1.75
    assert pair.prefix.values[2, global_mean] == pytest.approx(0.875)
    # Query rows share one frozen state and cannot update each other.
    assert pair.query.values[0, user_sum] == pytest.approx(2.85)
    assert pair.query.values[1, user_sum] == pytest.approx(2.85)


def test_matrix_size_admission_is_exact_and_validated() -> None:
    assert estimated_matrix_bytes(1_000, 33) == 264_000
    with pytest.raises(PureFeatureError):
        estimated_matrix_bytes(0, 33)
