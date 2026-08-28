from __future__ import annotations

import numpy as np
import pytest

from kuairand_agent.candidates.fusion import (
    FUSION_WEIGHT_GRID,
    FusionError,
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
    assert FUSION_WEIGHT_GRID == (
        (1.0, 0.0),
        (0.75, 0.25),
        (0.5, 0.5),
        (0.25, 0.75),
        (0.0, 1.0),
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
    assert len(result.fusion_digest) == 64
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

    for weights in ((0.6, 0.4), (True, False), [0.5, 0.5]):
        with pytest.raises(FusionError, match="exact member"):
            fuse_ranked_predictions(
                ["u", "u"],
                ["v", "v"],
                [1.0, 0.0],
                [0.0, 1.0],
                weights=weights,  # type: ignore[arg-type]
                phase=DataPhase.OUTER_VALID,
            )
