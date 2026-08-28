"""Train one tiny deterministic LightGBM fixture in an isolated test process.

macOS native runtimes can abort when PyTorch and LightGBM are initialized in the same process.
The production replay backend already isolates generated-model inference; this helper keeps the
integration-test setup equally order-independent without weakening the real-model assertion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--final-features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path, required=True)
    return parser


def _load(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected a two-column feature array: {path}")
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    arguments = _parser().parse_args()
    validation_features = _load(arguments.validation_features)
    final_features = _load(arguments.final_features)
    labels = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    if validation_features.shape[0] != labels.shape[0]:
        raise ValueError("the LambdaRank fixture requires exactly six validation rows")
    dataset = lightgbm.Dataset(
        validation_features,
        label=labels,
        group=[3, 3],
        free_raw_data=False,
    )
    booster = lightgbm.train(
        {
            "objective": "lambdarank",
            "metric": "None",
            "device_type": "cpu",
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
            "seed": 7,
            "label_gain": [0, 1],
            "min_data_in_leaf": 1,
            "num_leaves": 7,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=5,
    )
    arguments.model.write_text(booster.model_to_string(), encoding="utf-8")
    with arguments.validation_predictions.open("xb") as handle:
        np.save(
            handle,
            np.asarray(booster.predict(validation_features), dtype=np.float64),
            allow_pickle=False,
        )
    with arguments.final_predictions.open("xb") as handle:
        np.save(
            handle,
            np.asarray(booster.predict(final_features), dtype=np.float64),
            allow_pickle=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
