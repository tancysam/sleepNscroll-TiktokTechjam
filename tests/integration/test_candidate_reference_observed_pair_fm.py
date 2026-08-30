from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "candidate_seed"))

from candidate_seed.reference_observed_pair_fm import (  # noqa: E402
    _fit_reference_observed_pair_fm,
    reference_observed_pair_fm_diagnostics,
    reference_observed_pair_fm_scores,
)
from candidate_seed.reference_pairwise_fm import (  # noqa: E402
    _STATE_KEYS,
    REFERENCE_FEATURE_POSITIONS,
    _fit_reference_pairwise_fm,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.repeat(np.arange(12, dtype=np.int64), 8)
    within = np.tile(np.arange(8, dtype=np.int64), 12)
    targets = np.ascontiguousarray(within % 2 == 0, dtype=np.float64)
    features = np.zeros((targets.size, 56), dtype="<f8")
    features[:, 46] = np.take(np.asarray([4.0, 3.0, 12.0, 11.0, 22.0, 25.0, 61.0, 65.0]), within)
    features[:, REFERENCE_FEATURE_POSITIONS] = np.column_stack(
        (
            groups,
            np.int64(100) + within,
            np.int64(200) + (groups % 5),
            np.int64(300) + (within % 3),
            np.int64(400) + (within % 6),
        )
    )
    return features, targets, groups


def test_uniform_arm_is_reference_exact_and_duration_arm_has_equal_training_budget() -> None:
    features, targets, groups = _fixture()
    expected = _fit_reference_pairwise_fm(
        features,
        targets,
        groups,
        pairs_per_epoch=8192,
        epochs=1,
        seed=17,
    )
    uniform = _fit_reference_observed_pair_fm(
        features,
        targets,
        groups,
        arm="uniform_control",
        pairs_per_epoch=8192,
        epochs=1,
        seed=17,
    )
    duration = _fit_reference_observed_pair_fm(
        features,
        targets,
        groups,
        arm="duration_conditioned",
        pairs_per_epoch=8192,
        epochs=1,
        seed=17,
    )

    for name in _STATE_KEYS:
        np.testing.assert_array_equal(uniform[name], expected[name])
    uniform_diagnostics = reference_observed_pair_fm_diagnostics(uniform)
    duration_diagnostics = reference_observed_pair_fm_diagnostics(duration)
    assert uniform_diagnostics["observed_pair_arm_code"] == 0
    assert duration_diagnostics["observed_pair_arm_code"] == 1
    assert uniform_diagnostics["observed_pair_pairs_per_epoch"] == 8192
    assert duration_diagnostics["observed_pair_pairs_per_epoch"] == 8192
    assert uniform_diagnostics["observed_pair_epochs"] == 1
    assert duration_diagnostics["observed_pair_epochs"] == 1
    assert uniform_diagnostics["observed_pair_optimizer_steps"] == 1
    assert duration_diagnostics["observed_pair_optimizer_steps"] == 1
    assert uniform_diagnostics["observed_pair_intervention_pairs"] == 0
    assert duration_diagnostics["observed_pair_intervention_pairs"] == 4096
    assert uniform_diagnostics["observed_pair_seed"] == 17
    assert duration_diagnostics["observed_pair_seed"] == 17
    assert uniform_diagnostics["observed_pair_factor_dim"] == 16
    assert duration_diagnostics["observed_pair_factor_dim"] == 16
    assert uniform_diagnostics["observed_pair_feature_count"] == 5
    assert duration_diagnostics["observed_pair_feature_count"] == 5
    assert uniform_diagnostics["observed_pair_batch_size"] == 8192
    assert duration_diagnostics["observed_pair_batch_size"] == 8192
    assert uniform_diagnostics["observed_pair_learning_rate"] == 0.001
    assert duration_diagnostics["observed_pair_learning_rate"] == 0.001
    assert uniform_diagnostics["observed_pair_l2"] == 0.000001
    assert duration_diagnostics["observed_pair_l2"] == 0.000001
    np.testing.assert_array_equal(
        uniform["observed_pair_initial_state_sha256"],
        duration["observed_pair_initial_state_sha256"],
    )
    assert not np.array_equal(uniform["reference_factors"], duration["reference_factors"])

    uniform_scores = reference_observed_pair_fm_scores(features, uniform)
    duration_scores = reference_observed_pair_fm_scores(features, duration)
    assert uniform_scores.shape == duration_scores.shape == (features.shape[0],)
    assert uniform_scores.dtype == duration_scores.dtype == np.dtype("<f8")
    assert bool(np.isfinite(duration_scores).all())


def test_duration_arm_replays_checkpoint_and_prediction_bytes() -> None:
    features, targets, groups = _fixture()
    first = _fit_reference_observed_pair_fm(
        features,
        targets,
        groups,
        arm="duration_conditioned",
        pairs_per_epoch=8192,
        epochs=2,
        seed=20260830,
    )
    replay = _fit_reference_observed_pair_fm(
        np.array(features, copy=True, order="C"),
        targets.copy(),
        groups.copy(),
        arm="duration_conditioned",
        pairs_per_epoch=8192,
        epochs=2,
        seed=20260830,
    )

    assert set(first) == set(replay)
    for name in first:
        np.testing.assert_array_equal(first[name], replay[name])
    np.testing.assert_array_equal(
        reference_observed_pair_fm_scores(features, first),
        reference_observed_pair_fm_scores(features, replay),
    )
