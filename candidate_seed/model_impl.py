"""Small mutable model surface for autonomous candidate generation.

The stable ``candidate.py`` entrypoint owns capabilities, protocol validation, checkpoint I/O,
and result writing. Research implementations should normally replace only this module and
``config.json``.
"""

from __future__ import annotations

from typing import cast

import numpy as np

SCORES_DTYPE = "<f8"
CONFIG_KEYS = {
    "candidate_family",
    "epochs",
    "l2",
    "learning_rate",
    "logit_clip",
    "schema_version",
}


class CandidateModelError(ValueError):
    """The candidate-owned model configuration or numeric state is invalid."""


def _config_epochs(config: dict[str, object]) -> int:
    value = config["epochs"]
    if type(value) is not int or not 1 <= value <= 10_000:
        raise CandidateModelError("epochs is invalid")
    return value


def _config_float(config: dict[str, object], name: str) -> float:
    value = config[name]
    if type(value) not in {int, float}:
        raise CandidateModelError(f"{name} is invalid")
    numeric = float(cast(int | float, value))
    if not 0 < numeric < 1_000:
        raise CandidateModelError(f"{name} is invalid")
    return numeric


def validate_config(config: dict[str, object]) -> None:
    """Validate the model-owned configuration before training or prediction."""

    if set(config) != CONFIG_KEYS:
        raise CandidateModelError("model config keys do not match the implementation")
    if config["schema_version"] != 1:
        raise CandidateModelError("config schema_version must be 1")
    if config["candidate_family"] != "deterministic_logistic_seed":
        raise CandidateModelError("candidate_family is invalid")
    _config_epochs(config)
    for name in ("l2", "learning_rate", "logit_clip"):
        _config_float(config, name)


def _sigmoid(logits: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(logits, -clip, clip)
    result = np.reciprocal(1.0 + np.exp(-bounded), dtype=np.float64)
    return cast(np.ndarray, result)


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    """Fit a fixed-step standardized logistic model with deterministic full-batch updates."""

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise CandidateModelError("training features must have non-empty shape (N, D)")
    if targets.shape != (features.shape[0],):
        raise CandidateModelError("training targets must have shape (N,)")
    if not bool(np.logical_or(targets == 0.0, targets == 1.0).all()):
        raise CandidateModelError("training targets must be binary")
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    normalized = np.ascontiguousarray((features - mean) / scale, dtype=np.float64)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = np.float64(0.0)
    epochs = _config_epochs(config)
    learning_rate = np.float64(_config_float(config, "learning_rate"))
    l2 = np.float64(_config_float(config, "l2"))
    clip = _config_float(config, "logit_clip")
    row_count = np.float64(features.shape[0])
    for _ in range(epochs):
        probabilities = _sigmoid(normalized @ weights + bias, clip)
        error = probabilities - targets
        weights -= learning_rate * ((normalized.T @ error) / row_count + l2 * weights)
        bias -= learning_rate * np.mean(error, dtype=np.float64)
    probabilities = _sigmoid(normalized @ weights + bias, clip)
    epsilon = np.float64(1e-12)
    objective = -np.mean(
        targets * np.log(np.clip(probabilities, epsilon, 1.0))
        + (1.0 - targets) * np.log(np.clip(1.0 - probabilities, epsilon, 1.0)),
        dtype=np.float64,
    ) + np.float64(0.5) * l2 * np.dot(weights, weights)
    if not bool(np.isfinite(objective)):
        raise CandidateModelError("training objective became non-finite")
    return {
        "bias": np.asarray(bias, dtype=np.float64),
        "feature_mean": np.ascontiguousarray(mean, dtype=np.float64),
        "feature_scale": np.ascontiguousarray(scale, dtype=np.float64),
        "final_objective": np.asarray(objective, dtype=np.float64),
        "weights": np.ascontiguousarray(weights, dtype=np.float64),
    }


def predict_scores(features: np.ndarray, checkpoint: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the owned normalization and logistic interaction from a verified checkpoint."""

    expected = {"bias", "feature_mean", "feature_scale", "final_objective", "weights"}
    if set(checkpoint) != expected:
        raise CandidateModelError("checkpoint inventory is invalid")
    mean = checkpoint["feature_mean"]
    scale = checkpoint["feature_scale"]
    weights = checkpoint["weights"]
    bias = checkpoint["bias"]
    if features.ndim != 2 or features.shape[1:] != weights.shape:
        raise CandidateModelError("prediction feature shape does not match the checkpoint")
    if mean.shape != weights.shape or scale.shape != weights.shape or bias.shape != ():
        raise CandidateModelError("checkpoint array shapes are invalid")
    if not all(array.dtype == np.dtype(SCORES_DTYPE) for array in checkpoint.values()):
        raise CandidateModelError("checkpoint arrays must use float64")
    if not all(bool(np.isfinite(array).all()) for array in checkpoint.values()):
        raise CandidateModelError("checkpoint arrays must be finite")
    logits = ((features - mean) / scale) @ weights + bias
    return np.ascontiguousarray(_sigmoid(logits, 40.0), dtype=np.dtype(SCORES_DTYPE))


def training_diagnostics(
    config: dict[str, object], checkpoint: dict[str, np.ndarray]
) -> dict[str, int | float]:
    """Return bounded JSON diagnostics owned by the model implementation."""

    return {
        "epochs": _config_epochs(config),
        "final_objective": float(checkpoint["final_objective"]),
    }
