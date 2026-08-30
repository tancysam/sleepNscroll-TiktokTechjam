from __future__ import annotations

from itertools import pairwise
from unittest.mock import patch

import numpy as np
import pytest

from kuairand_agent.candidates import fusion as fusion_module
from kuairand_agent.candidates.fusion import (
    FUSION_WEIGHT_GRID,
    LEGACY_FUSION_WEIGHT_GRID,
    FusionError,
    fuse_ranked_members,
    fuse_ranked_predictions,
    normalize_within_user_percentiles,
)
from kuairand_agent.candidates.grouping import GroupingError, build_user_grouping
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.submission import prediction_digest


def test_interleaved_users_group_stably_and_scatter_exactly_to_canonical_order() -> None:
    grouping = build_user_grouping(
        user_ids=["a", "b", "c", "b", "a"],
        video_ids=["repeat-a", "repeat-b", "only-c", "repeat-b", "repeat-a"],
        phase=DataPhase.INNER_VALID,
    )
    canonical = np.asarray(["row-0", "row-1", "row-2", "row-3", "row-4"])

    grouped = grouping.to_grouped(canonical)

    assert grouping.group_sizes == (2, 2, 1)
    np.testing.assert_array_equal(grouping.grouped_to_canonical, [0, 4, 1, 3, 2])
    np.testing.assert_array_equal(grouping.canonical_to_grouped, [0, 2, 4, 3, 1])
    np.testing.assert_array_equal(grouped, ["row-0", "row-4", "row-1", "row-3", "row-2"])
    np.testing.assert_array_equal(grouping.to_canonical(grouped), canonical)
    assert grouping.row_count == 5
    assert sum(grouping.group_sizes) == grouping.row_count

    # The repeated (user, video) impressions remain two distinct physical rows after round-trip.
    videos = np.asarray(["repeat-a", "repeat-b", "only-c", "repeat-b", "repeat-a"])
    np.testing.assert_array_equal(grouping.to_canonical(grouping.to_grouped(videos)), videos)


def test_within_user_percentiles_use_equal_midranks_for_a_tie_around_rank_five() -> None:
    normalized = normalize_within_user_percentiles(
        user_ids=["u", "u", "u", "u", "u", "u", "u", "singleton"],
        video_ids=["a", "b", "c", "d", "repeat", "repeat", "g", "only"],
        scores=[7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 1.0, 42.0],
        phase=DataPhase.OUTER_VALID,
    )
    expected = np.asarray([1.0, 5 / 6, 2 / 3, 0.5, 0.25, 0.25, 0.0, 0.5])

    np.testing.assert_allclose(normalized.scores, expected, rtol=0.0, atol=1e-15)
    assert normalized.prediction_digest == prediction_digest(expected)
    assert normalized.tie_policy == "descending-average-rank; singleton=0.5"
    assert not normalized.scores.flags.writeable


def test_rank_fusion_uses_only_frozen_grid_weights_and_has_exact_prediction_identity() -> None:
    assert LEGACY_FUSION_WEIGHT_GRID == (
        (1.0, 0.0),
        (0.75, 0.25),
        (0.5, 0.5),
        (0.25, 0.75),
        (0.0, 1.0),
    )
    assert len(FUSION_WEIGHT_GRID) == 21
    assert FUSION_WEIGHT_GRID[0] == (1.0, 0.0)
    assert FUSION_WEIGHT_GRID[-1] == (0.0, 1.0)
    assert (0.4, 0.6) in FUSION_WEIGHT_GRID
    assert all(
        first[0] - second[0] == pytest.approx(0.05)
        for first, second in pairwise(FUSION_WEIGHT_GRID)
    )
    result = fuse_ranked_predictions(
        user_ids=["u", "u", "u", "v", "v"],
        video_ids=["dup", "middle", "dup", "x", "x"],
        first_scores=[3.0, 2.0, 1.0, 9.0, 9.0],
        second_scores=[1.0, 2.0, 3.0, 0.0, 1.0],
        weights=(0.75, 0.25),
        phase=DataPhase.FINAL,
    )
    expected = np.asarray([0.75, 0.5, 0.25, 0.375, 0.625])

    np.testing.assert_array_equal(result.scores, expected)
    assert result.prediction_digest == prediction_digest(expected)
    assert result.weights == (0.75, 0.25)
    assert result.member_prediction_digests == (
        prediction_digest([3.0, 2.0, 1.0, 9.0, 9.0]),
        prediction_digest([1.0, 2.0, 3.0, 0.0, 1.0]),
    )
    assert result.fusion_digest == (
        "cb3c821086a5363d11511c1156b036117624a122a513e43f92757794f9e89a47"
    )
    assert not result.scores.flags.writeable

    replay = fuse_ranked_predictions(
        user_ids=["u", "u", "u", "v", "v"],
        video_ids=["dup", "middle", "dup", "x", "x"],
        first_scores=[3.0, 2.0, 1.0, 9.0, 9.0],
        second_scores=[1.0, 2.0, 3.0, 0.0, 1.0],
        weights=(0.75, 0.25),
        phase=DataPhase.FINAL,
    )
    assert replay.fusion_digest == result.fusion_digest


def test_direct_three_member_fusion_normalizes_each_ordered_member_exactly_once() -> None:
    user_ids = ["u", "u", "u", "v", "v", "singleton"]
    video_ids = ["repeat", "middle", "repeat", "x", "x", "only"]
    pointwise = [3.0, 2.0, 1.0, 9.0, 9.0, 5.0]
    video_type = [1.0, 3.0, 2.0, 0.0, 1.0, 4.0]
    control = [2.0, 1.0, 3.0, 7.0, 6.0, 4.0]

    with patch.object(
        fusion_module,
        "normalize_within_user_percentiles",
        wraps=fusion_module.normalize_within_user_percentiles,
    ) as normalize:
        direct = fuse_ranked_members(
            user_ids,
            video_ids,
            (pointwise, video_type, control),
            weights=(0.15, 0.30, 0.55),
            phase=DataPhase.INNER_VALID,
        )
    expected = np.asarray(
        [0.42500000000000004, 0.375, 0.7000000000000001, 0.625, 0.375, 0.5],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(direct.scores, expected)
    assert direct.scores.dtype == np.float64
    assert direct.scores.flags.c_contiguous
    assert np.isfinite(direct.scores).all()
    assert direct.weights == (0.15, 0.30, 0.55)
    assert direct.member_prediction_digests == tuple(
        prediction_digest(member) for member in (pointwise, video_type, control)
    )
    assert len(direct.normalized_prediction_digests) == 3
    assert normalize.call_count == 3
    assert not direct.scores.flags.writeable

    pointwise_rank = normalize_within_user_percentiles(
        user_ids, video_ids, pointwise, phase=DataPhase.INNER_VALID
    ).scores
    video_type_rank = normalize_within_user_percentiles(
        user_ids, video_ids, video_type, phase=DataPhase.INNER_VALID
    ).scores
    nested_generated = pointwise_rank / 3.0 + video_type_rank * 2.0 / 3.0
    nested = fuse_ranked_predictions(
        user_ids,
        video_ids,
        nested_generated,
        control,
        weights=(0.45, 0.55),
        phase=DataPhase.INNER_VALID,
    )
    assert not np.array_equal(direct.scores, nested.scores)


def test_n_member_order_and_weight_positions_are_semantic() -> None:
    user_ids = ["u", "u", "u"]
    video_ids = ["a", "b", "c"]
    first = [3.0, 2.0, 1.0]
    second = [1.0, 3.0, 2.0]
    third = [2.0, 1.0, 3.0]

    ordered = fuse_ranked_members(
        user_ids,
        video_ids,
        (first, second, third),
        weights=(0.15, 0.30, 0.55),
        phase=DataPhase.FINAL,
    )
    reordered = fuse_ranked_members(
        user_ids,
        video_ids,
        (third, second, first),
        weights=(0.15, 0.30, 0.55),
        phase=DataPhase.FINAL,
    )

    assert ordered.member_prediction_digests == tuple(
        prediction_digest(member) for member in (first, second, third)
    )
    assert reordered.member_prediction_digests == tuple(
        prediction_digest(member) for member in (third, second, first)
    )
    assert not np.array_equal(ordered.scores, reordered.scores)
    assert ordered.fusion_digest != reordered.fusion_digest


def test_uniform_three_seed_fusion_is_direct_and_deterministic() -> None:
    user_ids = ["u", "u", "u", "v", "v", "singleton"]
    video_ids = ["repeat", "middle", "repeat", "x", "x", "only"]
    seeds = (
        [3.0, 2.0, 1.0, 2.0, 1.0, 8.0],
        [1.0, 3.0, 2.0, 2.0, 1.0, 7.0],
        [2.0, 1.0, 3.0, 1.0, 2.0, 6.0],
    )
    uniform_weights = (1.0 / 3.0,) * 3

    result = fuse_ranked_members(
        user_ids,
        video_ids,
        seeds,
        weights=uniform_weights,
        phase=DataPhase.OUTER_VALID,
    )
    replay = fuse_ranked_members(
        user_ids,
        video_ids,
        seeds,
        weights=uniform_weights,
        phase=DataPhase.OUTER_VALID,
    )

    np.testing.assert_array_equal(result.scores, [0.5, 0.5, 0.5, 2.0 / 3.0, 1.0 / 3.0, 0.5])
    assert result.weights == uniform_weights
    assert result.fusion_digest == replay.fusion_digest
    assert result.prediction_digest == replay.prediction_digest


def test_n_member_two_member_form_is_v2_domain_separated_from_legacy_adapter() -> None:
    user_ids = ["u", "u", "u"]
    video_ids = ["a", "b", "c"]
    members = ([3.0, 2.0, 1.0], [1.0, 2.0, 3.0])

    generic = fuse_ranked_members(
        user_ids,
        video_ids,
        members,
        weights=(0.75, 0.25),
        phase=DataPhase.FINAL,
    )
    legacy = fuse_ranked_predictions(
        user_ids,
        video_ids,
        *members,
        weights=(0.75, 0.25),
        phase=DataPhase.FINAL,
    )

    np.testing.assert_array_equal(generic.scores, legacy.scores)
    assert generic.prediction_digest == legacy.prediction_digest
    assert generic.fusion_digest == (
        "59c8c1ca0f13f5d6864e4fe1aa170d14feb3678760e81f93c42f6ddcba4766f5"
    )
    assert generic.fusion_digest != legacy.fusion_digest


def test_n_member_fusion_rejects_noncanonical_negative_zero_weight() -> None:
    with pytest.raises(FusionError, match="finite non-negative"):
        fuse_ranked_members(
            ["u", "u"],
            ["x", "y"],
            ([1.0, 0.0], [0.0, 1.0]),
            weights=(-0.0, 1.0),
            phase=DataPhase.OUTER_VALID,
        )


@pytest.mark.parametrize(
    ("member_scores", "weights", "message"),
    [
        (([1.0, 0.0],), (1.0,), "at least two"),
        (([1.0, 0.0], [0.0, 1.0]), [0.5, 0.5], "finite non-negative"),
        (([1.0, 0.0], [0.0, 1.0]), (0.5,), "finite non-negative"),
        (([1.0, 0.0], [0.0, 1.0]), (1, 0.0), "finite non-negative"),
        (([1.0, 0.0], [0.0, 1.0]), (float("nan"), 1.0), "finite non-negative"),
        (([1.0, 0.0], [0.0, 1.0]), (-0.1, 1.1), "finite non-negative"),
        (([1.0, 0.0], [0.0, 1.0]), (0.4, 0.5), "finite non-negative"),
    ],
)
def test_n_member_fusion_rejects_malformed_members_and_weights(
    member_scores: object,
    weights: object,
    message: str,
) -> None:
    with pytest.raises(FusionError, match=message):
        fuse_ranked_members(
            ["u", "u"],
            ["x", "y"],
            member_scores,  # type: ignore[arg-type]
            weights=weights,  # type: ignore[arg-type]
            phase=DataPhase.OUTER_VALID,
        )


@pytest.mark.parametrize("bad_member", ([1.0], [1.0, np.inf], [[1.0], [0.0]]))
def test_n_member_fusion_rejects_malformed_prediction_vectors(bad_member: object) -> None:
    with pytest.raises(FusionError, match=r"length 2|finite|one-dimensional"):
        fuse_ranked_members(
            ["u", "u"],
            ["x", "y"],
            ([1.0, 0.0], bad_member),  # type: ignore[arg-type]
            weights=(0.5, 0.5),
            phase=DataPhase.OUTER_VALID,
        )


def test_grouping_handles_matrices_and_rejects_noncanonical_or_disallowed_views() -> None:
    grouping = build_user_grouping(
        ["a", "b", "c", "b", "a"],
        ["x", "y", "z", "y", "x"],
        phase=DataPhase.TRAIN,
    )
    matrix = np.arange(15).reshape(5, 3)
    np.testing.assert_array_equal(grouping.to_canonical(grouping.to_grouped(matrix)), matrix)
    assert (
        grouping.digest
        == build_user_grouping(
            ["a", "b", "c", "b", "a"],
            ["x", "y", "z", "y", "x"],
            phase=DataPhase.TRAIN,
        ).digest
    )

    with pytest.raises(GroupingError, match="row count"):
        grouping.to_grouped([1, 2])
    with pytest.raises(GroupingError, match="equal lengths"):
        build_user_grouping(["u", "u"], ["v"], phase=DataPhase.TRAIN)
    with pytest.raises(GroupingError, match="one-dimensional"):
        build_user_grouping(np.asarray([["u"]]), ["v"], phase=DataPhase.TRAIN)
    for phase in (DataPhase.OUTER_VALID, DataPhase.FINAL):
        with pytest.raises(GroupingError, match="allowed only"):
            build_user_grouping(["u"], ["v"], phase=phase)


def test_rank_normalization_is_invariant_to_strictly_increasing_score_transform() -> None:
    raw = normalize_within_user_percentiles(
        ["u", "v", "u", "v", "u"],
        ["a", "b", "c", "d", "e"],
        [1.0, -2.0, 3.0, 4.0, 2.0],
        phase=DataPhase.INNER_VALID,
    )
    transformed = normalize_within_user_percentiles(
        ["u", "v", "u", "v", "u"],
        ["a", "b", "c", "d", "e"],
        [11.0, 4.0, 19.0, 22.0, 14.0],
        phase=DataPhase.INNER_VALID,
    )
    np.testing.assert_array_equal(raw.scores, transformed.scores)


@pytest.mark.parametrize("phase", [DataPhase.TRAIN, DataPhase.INNER_TRAIN])
def test_rank_fusion_rejects_training_phases(phase: DataPhase) -> None:
    with pytest.raises(FusionError, match="allowed only"):
        normalize_within_user_percentiles(["u"], ["v"], [1.0], phase=phase)


def test_rank_fusion_rejects_malformed_predictions_alignment_and_adaptive_weights() -> None:
    with pytest.raises(FusionError, match="equal lengths"):
        normalize_within_user_percentiles(["u", "u"], ["v"], [1.0, 0.0], phase=DataPhase.FINAL)
    with pytest.raises(FusionError, match="length 2"):
        normalize_within_user_percentiles(["u", "u"], ["v", "v"], [1.0], phase=DataPhase.FINAL)
    with pytest.raises(FusionError, match="one-dimensional"):
        normalize_within_user_percentiles(
            ["u", "u"], ["v", "v"], [[1.0], [0.0]], phase=DataPhase.FINAL
        )
    with pytest.raises(FusionError, match="finite"):
        normalize_within_user_percentiles(
            ["u", "u"], ["v", "v"], [np.nan, 0.0], phase=DataPhase.FINAL
        )

    for weights in ((0.625, 0.375), (True, False), [0.5, 0.5]):
        with pytest.raises(FusionError, match="exact member"):
            fuse_ranked_predictions(
                ["u", "u"],
                ["v", "v"],
                [1.0, 0.0],
                [0.0, 1.0],
                weights=weights,  # type: ignore[arg-type]
                phase=DataPhase.OUTER_VALID,
            )
