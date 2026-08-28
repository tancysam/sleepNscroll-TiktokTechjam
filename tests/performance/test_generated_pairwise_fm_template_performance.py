from __future__ import annotations

import time
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "candidate_templates" / "pairwise_fm"


class _Batch(Protocol):
    positive_indices: npt.NDArray[np.int64]


class _Sampler(Protocol):
    stored_row_index_count: int
    pair_space_size: int

    def sample(self, pair_count: int, *, seed: int) -> _Batch: ...


class _CandidateModule(Protocol):
    def GAUCPairSampler(
        self,
        user_groups: npt.NDArray[np.int64],
        targets: npt.NDArray[np.int8],
    ) -> _Sampler: ...


def _candidate_module() -> _CandidateModule:
    source_path = TEMPLATE / "candidate.py"
    module = ModuleType("generated_pairwise_fm_candidate_performance")
    exec(compile(source_path.read_bytes(), str(source_path), "exec"), module.__dict__)
    return cast(_CandidateModule, module)


def test_pair_sampler_is_linear_memory_and_bounded_throughput() -> None:
    module = _candidate_module()
    # Each of 200 users has 100 positives and 100 negatives: two million possible
    # pairs, while the sampler should retain only the 40,000 logged row indices.
    users = np.repeat(np.arange(200, dtype=np.int64), 200)
    targets = np.tile(np.repeat(np.array([1, 0], dtype=np.int8), 100), 200)

    started = time.perf_counter()
    sampler = module.GAUCPairSampler(users, targets)
    batch = sampler.sample(250_000, seed=20260828)
    elapsed = time.perf_counter() - started

    assert sampler.stored_row_index_count == 40_000
    assert sampler.pair_space_size == 2_000_000
    assert batch.positive_indices.shape == (250_000,)
    assert elapsed < 3.0
