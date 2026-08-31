from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "candidate_seed"))

from reference_categorical_ranker import CATEGORICAL_RANK_FEATURE_POSITIONS  # noqa: E402
from reference_pointwise_ranker import (  # noqa: E402
    _fit_reference_pointwise_ranker,
    reference_pointwise_ranker_diagnostics,
    reference_pointwise_ranker_scores,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_groups = np.repeat(np.arange(16, dtype=np.int64), 8)
    within_user = np.tile(np.arange(8, dtype=np.int64), 16)
    targets = np.ascontiguousarray(within_user >= 4, dtype=np.float64)
    features = np.zeros((targets.size, 83), dtype="<f8")
    features[:, 0] = within_user / 7.0
    features[:, 3] = targets * 0.4 + within_user * 0.01
    features[:, 6] = (within_user % 5) / 5.0
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
    features[:, 82] = within_user % 3
    return features, targets, user_groups


def test_protected_pointwise_ranker_is_deterministic_replayable_and_frozen_to_82_columns() -> None:
    features, targets, groups = _fixture()
    kwargs = {
        "pairs_per_epoch": 8192,
        "pairwise_epochs": 1,
        "tree_count": 8,
        "pointwise_epochs": 1,
    }
    first = _fit_reference_pointwise_ranker(features, targets, groups, **kwargs)
    second = _fit_reference_pointwise_ranker(features, targets, groups, **kwargs)

    assert set(first) == set(second)
    assert 1 <= len(first) <= 64
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert isinstance(first[name], np.ndarray)
        assert bool(np.isfinite(first[name]).all())

    scores = reference_pointwise_ranker_scores(features, first)
    replay = reference_pointwise_ranker_scores(
        np.array(features, copy=True),
        {name: np.array(value, copy=True) for name, value in first.items()},
    )
    np.testing.assert_array_equal(scores, replay)
    changed_video_type = np.array(features, copy=True)
    changed_video_type[:, 82] += 100.0
    np.testing.assert_array_equal(
        scores,
        reference_pointwise_ranker_scores(changed_video_type, first),
    )
    assert scores.shape == (features.shape[0],)
    assert scores.dtype == np.dtype("<f8")

    diagnostics = reference_pointwise_ranker_diagnostics(first)
    assert diagnostics["categorical_rank_feature_count"] == 83
    assert diagnostics["pointwise_batch_size"] == 4096
    assert diagnostics["pointwise_epochs"] == 3
    assert diagnostics["pointwise_hidden_one"] == 256
    assert diagnostics["pointwise_hidden_two"] == 64
    assert diagnostics["pointwise_residual_scale"] == 0.5
