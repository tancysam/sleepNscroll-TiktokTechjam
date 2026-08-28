from __future__ import annotations

import numpy as np
import pytest

from kuairand_agent.candidates.neural import (
    NeuralArchitecture,
    NeuralCeilings,
    NeuralFeatureBatch,
    NeuralFitConfig,
    NeuralModelConfig,
    NeuralPairIndices,
    NeuralTrainingTargets,
    fit_neural,
)
from kuairand_agent.data.capabilities import DataPhase


def test_tiny_dcnv2_epoch_stays_inside_the_predeclared_cpu_family_ceiling() -> None:
    pytest.importorskip("torch")
    row_count = 512
    row = np.arange(row_count, dtype=np.int64)
    features = NeuralFeatureBatch(
        phase=DataPhase.INNER_TRAIN,
        categorical=np.column_stack((row % 31, row % 47)),
        dense=np.column_stack(
            (
                np.sin(row / 10.0),
                np.cos(row / 17.0),
                (row % 13) / 13.0,
            )
        ),
    )
    targets = NeuralTrainingTargets(
        phase=DataPhase.INNER_TRAIN,
        values=(row % 2 == 0).astype(np.int8),
    )
    pairs = NeuralPairIndices(row[::2], row[1::2])
    ceilings = NeuralCeilings(
        max_parameters=50_000,
        max_p95_epoch_seconds=10.0,
        min_examples_per_second=10.0,
        max_checkpoint_bytes=2_000_000,
    )

    checkpoint = fit_neural(
        NeuralModelConfig(
            architecture=NeuralArchitecture.DCNV2,
            categorical_cardinalities=(31, 47),
            dense_feature_count=3,
            embedding_dim=4,
            hidden_dims=(16, 8),
            cross_layers=2,
        ),
        features,
        targets,
        pairs,
        fit_config=NeuralFitConfig(
            seed=101,
            epochs=1,
            batch_size=128,
            learning_rate=0.005,
            requested_device="cpu",
            num_threads=1,
        ),
        ceilings=ceilings,
    )

    evidence = checkpoint.result.eligibility.evidence
    assert checkpoint.result.eligibility.passed
    assert evidence.parameter_count <= ceilings.max_parameters
    assert evidence.p95_epoch_seconds <= ceilings.max_p95_epoch_seconds
    assert evidence.minimum_examples_per_second >= ceilings.min_examples_per_second
    assert evidence.checkpoint_bytes <= ceilings.max_checkpoint_bytes
