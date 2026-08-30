from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "candidate_seed"))

from reference_categorical_ranker import CATEGORICAL_RANK_FEATURE_POSITIONS  # noqa: E402
from reference_listnet_ranker import (  # noqa: E402
    _fit_reference_listnet_ranker,
    reference_listnet_ranker_diagnostics,
    reference_listnet_ranker_scores,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_groups = np.repeat(np.arange(16, dtype=np.int64), 8)
    within_user = np.tile(np.arange(8, dtype=np.int64), 16)
    targets = np.ascontiguousarray(within_user >= 4, dtype=np.float64)
    # Legacy ListNet composition remains first-82; the nested categorical specialist receives
    # this neutral integral video_type_code through its adapter.
    features = np.zeros((targets.size, 83), dtype="<f8")
    features[:, 0] = within_user / 7.0
    features[:, 3] = targets * 0.4 + within_user * 0.01
    features[:, 6] = (within_user % 5) / 5.0
    features[:, 21] = ((user_groups + within_user) % 11) / 11.0
    features[:, CATEGORICAL_RANK_FEATURE_POSITIONS] = np.column_stack(
        (
            user_groups,
            np.int64(100) + (within_user % 7),
            np.int64(200) + (user_groups % 9),
            np.int64(300) + (within_user % 3),
            np.int64(400) + (within_user % 6),
            np.zeros_like(within_user),
        )
    )
    return features, targets, user_groups


def test_protected_listnet_ranker_is_deterministic_and_replayable() -> None:
    features, targets, groups = _fixture()
    first = _fit_reference_listnet_ranker(
        features,
        targets,
        groups,
        pairs_per_epoch=8192,
        pairwise_epochs=1,
        tree_count=8,
        listnet_epochs=2,
    )
    second = _fit_reference_listnet_ranker(
        features,
        targets,
        groups,
        pairs_per_epoch=8192,
        pairwise_epochs=1,
        tree_count=8,
        listnet_epochs=2,
    )

    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert isinstance(first[name], np.ndarray)
        assert bool(np.isfinite(first[name]).all())

    scores = reference_listnet_ranker_scores(features, first)
    replay = reference_listnet_ranker_scores(
        np.array(features, copy=True),
        {name: np.array(value, copy=True) for name, value in first.items()},
    )
    np.testing.assert_array_equal(scores, replay)
    with_video_type = np.array(features, copy=True)
    with_video_type[:, 82] = np.arange(features.shape[0]) % 3
    np.testing.assert_array_equal(
        scores,
        reference_listnet_ranker_scores(
            np.ascontiguousarray(with_video_type, dtype="<f8"),
            first,
        ),
    )
    assert scores.shape == (features.shape[0],)
    assert scores.dtype == np.dtype("<f8")

    diagnostics = reference_listnet_ranker_diagnostics(first)
    assert diagnostics["listnet_eligible_queries"] == 16
    assert diagnostics["listnet_epochs"] == 8
    assert diagnostics["listnet_hidden_one"] == 192
    assert diagnostics["listnet_hidden_two"] == 64
    assert diagnostics["listnet_score_scale"] == 0.15
    assert diagnostics["reference_sampled_pairs"] == 8192
