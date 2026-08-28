from __future__ import annotations

import math

import numpy as np
import pytest

from kuairand_agent.candidates.pairwise import (
    MAX_SAMPLED_PAIRS,
    GAUCPairSampler,
    PairwisePrimitiveError,
    brute_force_pair_distribution,
    pairwise_logistic_loss_and_gradient,
)
from kuairand_agent.data.capabilities import DataPhase


def test_brute_force_distribution_matches_worked_gauc_pair_weights() -> None:
    """P(u) proportional to positives gives each pair weight 1/(sum(P) * N_u)."""

    distribution = brute_force_pair_distribution(
        user_ids=np.asarray(["a", "a", "b", "b", "b", "b"]),
        labels=np.asarray([1, 0, 1, 1, 0, 0], dtype=np.int8),
        phase=DataPhase.TRAIN,
    )

    assert list(zip(distribution.positive_indices, distribution.negative_indices, strict=True)) == [
        (0, 1),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ]
    np.testing.assert_allclose(
        distribution.probabilities,
        np.asarray([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]),
        rtol=0.0,
        atol=1e-15,
    )
    assert distribution.pair_count == 5
    assert float(distribution.probabilities.sum()) == pytest.approx(1.0, abs=1e-15)


def test_sampler_is_seeded_and_draws_users_in_proportion_to_positive_count() -> None:
    users = np.asarray(["a", "a", "b", "b", "b", "b"])
    labels = np.asarray([1, 0, 1, 1, 0, 0], dtype=np.int8)
    sampler = GAUCPairSampler(users, labels, phase=DataPhase.INNER_TRAIN)

    first = sampler.sample(60_000, seed=913)
    replay = sampler.sample(60_000, seed=913)

    np.testing.assert_array_equal(first.user_group_indices, replay.user_group_indices)
    np.testing.assert_array_equal(first.positive_indices, replay.positive_indices)
    np.testing.assert_array_equal(first.negative_indices, replay.negative_indices)
    np.testing.assert_allclose(
        np.bincount(first.user_group_indices, minlength=2) / first.pair_count,
        np.asarray([1 / 3, 2 / 3]),
        rtol=0.0,
        atol=0.01,
    )
    assert np.all(labels[first.positive_indices] == 1)
    assert np.all(labels[first.negative_indices] == 0)
    assert np.all(users[first.positive_indices] == users[first.negative_indices])
    assert sampler.eligible_user_count == 2
    assert sampler.pair_space_size == 5


def test_sampler_empirical_pair_law_matches_independent_bounded_oracle() -> None:
    """Every concrete pair follows the GAUC law, not only the user marginal."""

    users = np.asarray(["a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "c"])
    labels = np.asarray([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.int8)
    oracle = brute_force_pair_distribution(users, labels, phase=DataPhase.TRAIN)
    sampled = GAUCPairSampler(users, labels, phase=DataPhase.TRAIN).sample(300_000, seed=73)

    expected = {
        (int(positive), int(negative)): float(probability)
        for positive, negative, probability in zip(
            oracle.positive_indices,
            oracle.negative_indices,
            oracle.probabilities,
            strict=True,
        )
    }
    pairs, counts = np.unique(
        np.column_stack((sampled.positive_indices, sampled.negative_indices)),
        axis=0,
        return_counts=True,
    )
    observed = {
        (int(pair[0]), int(pair[1])): int(count) / sampled.pair_count
        for pair, count in zip(pairs, counts, strict=True)
    }

    assert observed.keys() == expected.keys()
    for pair, probability in expected.items():
        assert observed[pair] == pytest.approx(probability, abs=0.0025)


def test_sampler_seed_has_locked_vectorized_golden_batch() -> None:
    """The lock-pinned NumPy generator and flat-row algorithm replay byte-for-byte."""

    sampler = GAUCPairSampler(
        user_ids=np.asarray(["a", "a", "a", "b", "b", "b", "b", "c", "c"]),
        labels=np.asarray([1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8),
        phase=DataPhase.INNER_TRAIN,
    )

    batch = sampler.sample(16, seed=240819)

    np.testing.assert_array_equal(
        batch.user_group_indices,
        [1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0, 2, 2, 1, 0],
    )
    np.testing.assert_array_equal(
        batch.positive_indices,
        [3, 6, 3, 7, 3, 3, 3, 6, 3, 0, 3, 2, 7, 7, 6, 2],
    )
    np.testing.assert_array_equal(
        batch.negative_indices,
        [4, 4, 4, 8, 5, 5, 4, 4, 4, 1, 4, 1, 8, 8, 5, 1],
    )
    assert not batch.user_group_indices.flags.writeable
    assert not batch.positive_indices.flags.writeable
    assert not batch.negative_indices.flags.writeable


def test_pairwise_logistic_loss_and_score_gradients_match_worked_weighted_case() -> None:
    result = pairwise_logistic_loss_and_gradient(
        positive_scores=np.asarray([0.0, 0.0]),
        negative_scores=np.asarray([0.0, 0.0]),
        weights=np.asarray([1.0, 3.0]),
    )

    assert result.loss == pytest.approx(float(np.log(2.0)), abs=1e-15)
    np.testing.assert_allclose(result.positive_gradient, [-0.125, -0.375], atol=1e-15)
    np.testing.assert_allclose(result.negative_gradient, [0.125, 0.375], atol=1e-15)
    np.testing.assert_array_equal(result.negative_gradient, -result.positive_gradient)


def test_weighted_loss_matches_exact_brute_force_pair_expectation() -> None:
    distribution = brute_force_pair_distribution(
        user_ids=["a", "a", "b", "b", "b", "b"],
        labels=[1, 0, 1, 1, 0, 0],
        phase=DataPhase.TRAIN,
    )
    scores = np.asarray([2.0, 0.0, 1.0, 3.0, -1.0, 0.5])

    result = pairwise_logistic_loss_and_gradient(
        scores[distribution.positive_indices],
        scores[distribution.negative_indices],
        weights=distribution.probabilities,
    )

    # One user-a pair has probability 1/3.  The four user-b pairs have probability 1/6 each.
    expected = (1 / 3) * math.log1p(math.exp(-2.0)) + (1 / 6) * sum(
        math.log1p(math.exp(-margin)) for margin in (2.0, 0.5, 4.0, 2.5)
    )
    assert result.loss == pytest.approx(expected, abs=1e-15)


def test_pairwise_logistic_loss_is_finite_for_extreme_finite_margins() -> None:
    result = pairwise_logistic_loss_and_gradient(
        positive_scores=np.asarray([-1_000.0, 1_000.0]),
        negative_scores=np.asarray([1_000.0, -1_000.0]),
    )

    assert result.loss == pytest.approx(1_000.0)
    assert np.isfinite(result.loss)
    assert np.isfinite(result.positive_gradient).all()
    assert np.isfinite(result.negative_gradient).all()
    np.testing.assert_allclose(result.positive_gradient, [-0.5, 0.0], atol=0.0)
    np.testing.assert_allclose(result.negative_gradient, [0.5, 0.0], atol=0.0)


def test_ineligible_edge_users_are_excluded_without_changing_row_positions() -> None:
    sampler = GAUCPairSampler(
        user_ids=["mixed", "mixed", "positive-only", "positive-only", "negative-only"],
        labels=[1, 0, 1, 1, 0],
        phase=DataPhase.TRAIN,
    )
    batch = sampler.sample(128, seed=4)

    assert sampler.eligible_user_count == 1
    assert sampler.stored_row_index_count == 2
    assert sampler.pair_space_size == 1
    assert set(batch.positive_indices.tolist()) == {0}
    assert set(batch.negative_indices.tolist()) == {1}


def test_sampler_memory_is_bounded_by_rows_and_requested_batch_not_pair_space() -> None:
    pair_side = 2_000
    sampler = GAUCPairSampler(
        user_ids=np.repeat("large", pair_side * 2),
        labels=np.concatenate(
            [np.ones(pair_side, dtype=np.int8), np.zeros(pair_side, dtype=np.int8)]
        ),
        phase=DataPhase.TRAIN,
    )

    assert sampler.pair_space_size == 4_000_000
    assert sampler.stored_row_index_count == 4_000
    assert sampler.sample(32, seed=8).pair_count == 32
    with pytest.raises(PairwisePrimitiveError, match="exceeding"):
        brute_force_pair_distribution(
            user_ids=np.repeat("large", pair_side * 2),
            labels=np.concatenate(
                [np.ones(pair_side, dtype=np.int8), np.zeros(pair_side, dtype=np.int8)]
            ),
            phase=DataPhase.TRAIN,
        )


@pytest.mark.parametrize("phase", [DataPhase.INNER_VALID, DataPhase.OUTER_VALID, DataPhase.FINAL])
def test_sampler_rejects_every_nontraining_label_phase(phase: DataPhase) -> None:
    with pytest.raises(PairwisePrimitiveError, match="only for train or inner_train"):
        GAUCPairSampler(["u", "u"], [1, 0], phase=phase)


@pytest.mark.parametrize(
    ("user_ids", "labels", "message"),
    [
        (["u"], [1, 0], "equal lengths"),
        (["u", "u"], [1, 2], "binary 0 and 1"),
        (["u", "u"], [1, np.nan], "binary 0 and 1"),
        (["u", "u"], [1, 1], "mixed-label user"),
        ([["u"], ["u"]], [1, 0], "user_ids must be one-dimensional"),
    ],
)
def test_sampler_rejects_invalid_shapes_domains_and_unusable_groups(
    user_ids: object, labels: object, message: str
) -> None:
    with pytest.raises(PairwisePrimitiveError, match=message):
        GAUCPairSampler(user_ids, labels, phase=DataPhase.TRAIN)  # type: ignore[arg-type]


def test_sample_and_loss_reject_invalid_bounds_and_numeric_inputs() -> None:
    sampler = GAUCPairSampler(["u", "u"], [1, 0], phase=DataPhase.TRAIN)
    for pair_count in (0, MAX_SAMPLED_PAIRS + 1, True):
        with pytest.raises(PairwisePrimitiveError, match="pair_count"):
            sampler.sample(pair_count, seed=0)
    for seed in (-1, 2**32, True):
        with pytest.raises(PairwisePrimitiveError, match="seed"):
            sampler.sample(1, seed=seed)

    with pytest.raises(PairwisePrimitiveError, match="equal lengths"):
        pairwise_logistic_loss_and_gradient([0.0], [0.0, 1.0])
    with pytest.raises(PairwisePrimitiveError, match="finite real"):
        pairwise_logistic_loss_and_gradient([np.inf], [0.0])
    with pytest.raises(PairwisePrimitiveError, match="non-negative"):
        pairwise_logistic_loss_and_gradient([0.0], [0.0], weights=[-1.0])
    with pytest.raises(PairwisePrimitiveError, match="positive sum"):
        pairwise_logistic_loss_and_gradient([0.0], [0.0], weights=[0.0])
