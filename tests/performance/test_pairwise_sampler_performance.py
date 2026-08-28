from __future__ import annotations

import time
import tracemalloc

import numpy as np

from kuairand_agent.candidates.pairwise import GAUCPairSampler
from kuairand_agent.data.capabilities import DataPhase


def _many_mixed_users(user_count: int) -> GAUCPairSampler:
    return GAUCPairSampler(
        user_ids=np.repeat(np.arange(user_count, dtype=np.int64), 2),
        labels=np.tile(np.asarray([1, 0], dtype=np.int8), user_count),
        phase=DataPhase.INNER_TRAIN,
    )


def test_full_scale_sampling_does_not_rescan_batch_for_every_user() -> None:
    """A representative batch must stay well below the old O(users * batch) path."""

    sampler = _many_mixed_users(20_000)
    started = time.perf_counter()
    batch = sampler.sample(200_000, seed=20260828)
    elapsed = time.perf_counter() - started

    assert batch.pair_count == 200_000
    assert elapsed < 0.5, f"sampling took {elapsed:.3f}s; batch appears to be rescanned per user"


def test_sample_peak_memory_is_bounded_by_batch_not_cartesian_pair_space() -> None:
    pair_side = 50_000
    sampler = GAUCPairSampler(
        user_ids=np.zeros(pair_side * 2, dtype=np.int64),
        labels=np.concatenate(
            (np.ones(pair_side, dtype=np.int8), np.zeros(pair_side, dtype=np.int8))
        ),
        phase=DataPhase.INNER_TRAIN,
    )
    sampler.sample(1, seed=0)  # Resolve NumPy lazy imports before measuring the sampling path.

    tracemalloc.start()
    try:
        batch = sampler.sample(100_000, seed=19)
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    output_bytes = sum(
        array.nbytes
        for array in (
            batch.positive_indices,
            batch.negative_indices,
            batch.user_group_indices,
        )
    )
    assert sampler.pair_space_size == 2_500_000_000
    assert sampler.stored_row_index_count == 100_000
    assert output_bytes == 2_400_000
    assert peak_bytes < 8_000_000
