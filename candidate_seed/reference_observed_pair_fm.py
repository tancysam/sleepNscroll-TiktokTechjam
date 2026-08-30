"""Protected equal-budget FM experiment for duration-conditioned logged pairs.

The uniform arm delegates to the verified reference FM.  The duration arm shares its five fields,
factor capacity, initialization, dense-L2 Adam optimizer, batch schedule, and pair budget; only
the deterministic pair arrays differ.  Prediction consumes features and checkpoint state only.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from reference_observed_pair_objectives import (  # type: ignore[import-not-found]
    prepare_reference_duration_pair_ablation,
)
from reference_pairwise_fm import (  # type: ignore[import-not-found]
    _STATE_KEYS,
    REFERENCE_BATCH_SIZE,
    REFERENCE_EPOCHS,
    REFERENCE_FACTOR_DIM,
    REFERENCE_FEATURE_POSITIONS,
    REFERENCE_L2,
    REFERENCE_LEARNING_RATE,
    REFERENCE_PAIRS_PER_EPOCH,
    _batch_gradients,
    _encoded_codes,
    _fit_reference_pairwise_fm,
    reference_pairwise_fm_scores,
)

UNIFORM_CONTROL_ARM = "uniform_control"
DURATION_CONDITIONED_ARM = "duration_conditioned"
OBSERVED_PAIR_SCHEMA_VERSION = 1
_ARM_CODES = {UNIFORM_CONTROL_ARM: 0, DURATION_CONDITIONED_ARM: 1}
_OBSERVED_KEYS = {
    "observed_pair_arm_code",
    "observed_pair_conditioned_eligible_groups",
    "observed_pair_conditioned_eligible_positives",
    "observed_pair_epochs",
    "observed_pair_initial_state_sha256",
    "observed_pair_intervention_pairs",
    "observed_pair_optimizer_steps",
    "observed_pair_pairs_per_epoch",
    "observed_pair_schema_version",
    "observed_pair_seed",
    "observed_pair_treatment_seeds",
}
_CHECKPOINT_KEYS = _STATE_KEYS | _OBSERVED_KEYS


class ReferenceObservedPairFMError(ValueError):
    """The experimental observed-pair FM violates its frozen numeric contract."""


def _uint32(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ReferenceObservedPairFMError("seed must fit uint32")
    return value


def _budget(pairs_per_epoch: object, epochs: object) -> tuple[int, int]:
    if (
        type(pairs_per_epoch) is not int
        or not 2 <= pairs_per_epoch <= 1_000_000
        or pairs_per_epoch % 2 != 0
    ):
        raise ReferenceObservedPairFMError(
            "pairs_per_epoch must be an even integer in [2, 1000000]"
        )
    if type(epochs) is not int or not 1 <= epochs <= REFERENCE_EPOCHS:
        raise ReferenceObservedPairFMError(f"epochs must be an integer in [1, {REFERENCE_EPOCHS}]")
    return pairs_per_epoch, epochs


def _initial_state_digest(encoded: np.ndarray, seed: int) -> np.ndarray:
    total_dim = int(np.max(encoded)) + 2
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, size=(total_dim, REFERENCE_FACTOR_DIM)).astype(np.float32)
    linear = np.zeros(total_dim, dtype=np.float32)
    digest = hashlib.sha256(b"kuairand-observed-pair-fm-initial-state-v1\0")
    digest.update(factors.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(linear.astype("<f4", copy=False).tobytes(order="C"))
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def _observed_metadata(
    reference: dict[str, np.ndarray],
    *,
    arm: str,
    seed: int,
    pairs_per_epoch: int,
    epochs: int,
    initial_digest: np.ndarray,
    treatment_seeds: np.ndarray,
    intervention_pairs: int,
    eligible_groups: int,
    eligible_positives: int,
) -> dict[str, np.ndarray]:
    state = dict(reference)
    state.update(
        {
            "observed_pair_arm_code": np.asarray(_ARM_CODES[arm], dtype=np.int8),
            "observed_pair_conditioned_eligible_groups": np.asarray(
                eligible_groups, dtype=np.int64
            ),
            "observed_pair_conditioned_eligible_positives": np.asarray(
                eligible_positives, dtype=np.int64
            ),
            "observed_pair_epochs": np.asarray(epochs, dtype=np.int64),
            "observed_pair_initial_state_sha256": np.ascontiguousarray(
                initial_digest, dtype=np.uint8
            ),
            "observed_pair_intervention_pairs": np.asarray(intervention_pairs, dtype=np.int64),
            "observed_pair_optimizer_steps": np.asarray(
                epochs * math.ceil(pairs_per_epoch / REFERENCE_BATCH_SIZE), dtype=np.int64
            ),
            "observed_pair_pairs_per_epoch": np.asarray(pairs_per_epoch, dtype=np.int64),
            "observed_pair_schema_version": np.asarray(
                OBSERVED_PAIR_SCHEMA_VERSION, dtype=np.int64
            ),
            "observed_pair_seed": np.asarray(seed, dtype=np.uint64),
            "observed_pair_treatment_seeds": np.ascontiguousarray(treatment_seeds, dtype=np.uint32),
        }
    )
    return state


def _fit_duration_treatment(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    pairs_per_epoch: int,
    epochs: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    encoded = _encoded_codes(features)
    labels = np.asarray(targets)
    groups = np.asarray(user_groups)
    if labels.shape != (encoded.shape[0],) or groups.shape != (encoded.shape[0],):
        raise ReferenceObservedPairFMError("targets and user_groups must align to encoded features")
    total_dim = int(np.max(encoded)) + 2
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, size=(total_dim, REFERENCE_FACTOR_DIM)).astype(np.float32)
    linear = np.zeros(total_dim, dtype=np.float32)
    factor_first_moment = np.zeros_like(factors)
    factor_second_moment = np.zeros_like(factors)
    linear_first_moment = np.zeros_like(linear)
    linear_second_moment = np.zeros_like(linear)
    sampling_rng = np.random.default_rng(seed)
    treatment_seeds = np.empty(epochs, dtype=np.uint32)
    optimizer_steps = 0
    final_pairwise_loss = math.nan
    eligible_groups = 0
    eligible_positives = 0

    for epoch in range(epochs):
        sample_seed = int(sampling_rng.integers(0, 2**32, dtype=np.uint64))
        pairs = prepare_reference_duration_pair_ablation(
            features,
            groups,
            labels,
            pair_count=pairs_per_epoch,
            seed=sample_seed,
        )
        treatment_seeds[epoch] = pairs.treatment_seed
        eligible_groups = pairs.conditioned_eligible_group_count
        eligible_positives = pairs.conditioned_eligible_positive_count
        weighted_loss = 0.0
        for start in range(0, pairs.pair_count, REFERENCE_BATCH_SIZE):
            stop = min(start + REFERENCE_BATCH_SIZE, pairs.pair_count)
            positive = encoded[pairs.treatment_positive_indices[start:stop]]
            negative = encoded[pairs.treatment_negative_indices[start:stop]]
            loss, factor_gradient, linear_gradient = _batch_gradients(
                positive,
                negative,
                factors,
                linear,
            )
            batch_size = stop - start
            weighted_loss += loss * batch_size
            optimizer_steps += 1
            for parameter, gradient, first_moment, second_moment in (
                (factors, factor_gradient, factor_first_moment, factor_second_moment),
                (linear, linear_gradient, linear_first_moment, linear_second_moment),
            ):
                first_moment *= np.float32(0.9)
                first_moment += np.float32(0.1) * gradient
                second_moment *= np.float32(0.999)
                second_moment += np.float32(0.001) * (gradient * gradient)
                parameter -= (
                    REFERENCE_LEARNING_RATE
                    * (first_moment / (1.0 - 0.9**optimizer_steps))
                    / (np.sqrt(second_moment / (1.0 - 0.999**optimizer_steps)) + 1e-8)
                )
            if not bool(np.isfinite(factors).all()) or not bool(np.isfinite(linear).all()):
                raise ReferenceObservedPairFMError(
                    "duration-conditioned FM optimizer produced non-finite state"
                )
        final_pairwise_loss = weighted_loss / pairs.pair_count

    reference = {
        "reference_factors": np.ascontiguousarray(factors, dtype=np.float32),
        "reference_feature_positions": np.asarray(REFERENCE_FEATURE_POSITIONS, dtype=np.int64),
        "reference_final_pairwise_loss": np.asarray(final_pairwise_loss, dtype=np.float64),
        "reference_linear": np.ascontiguousarray(linear, dtype=np.float32),
        "reference_sampled_pairs": np.asarray(pairs_per_epoch * epochs, dtype=np.int64),
        "reference_schema_version": np.asarray(1, dtype=np.int64),
        "reference_total_dim": np.asarray(total_dim, dtype=np.int64),
        "reference_seed": np.asarray(seed, dtype=np.uint64),
    }
    return reference, treatment_seeds, eligible_groups, eligible_positives


def _fit_reference_observed_pair_fm(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    arm: str,
    pairs_per_epoch: int,
    epochs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit a bounded arm; public production entrypoints freeze the full reference budget."""

    if arm not in _ARM_CODES:
        raise ReferenceObservedPairFMError("arm must be uniform_control or duration_conditioned")
    pair_budget, epoch_budget = _budget(pairs_per_epoch, epochs)
    root_seed = _uint32(seed)
    encoded = _encoded_codes(features)
    initial_digest = _initial_state_digest(encoded, root_seed)
    if arm == UNIFORM_CONTROL_ARM:
        reference = _fit_reference_pairwise_fm(
            features,
            targets,
            user_groups,
            pairs_per_epoch=pair_budget,
            epochs=epoch_budget,
            seed=root_seed,
        )
        treatment_seeds = np.zeros(epoch_budget, dtype=np.uint32)
        intervention_pairs = 0
        eligible_groups = 0
        eligible_positives = 0
    else:
        reference, treatment_seeds, eligible_groups, eligible_positives = _fit_duration_treatment(
            features,
            targets,
            user_groups,
            pairs_per_epoch=pair_budget,
            epochs=epoch_budget,
            seed=root_seed,
        )
        intervention_pairs = (pair_budget // 2) * epoch_budget
    return _observed_metadata(
        reference,
        arm=arm,
        seed=root_seed,
        pairs_per_epoch=pair_budget,
        epochs=epoch_budget,
        initial_digest=initial_digest,
        treatment_seeds=treatment_seeds,
        intervention_pairs=intervention_pairs,
        eligible_groups=eligible_groups,
        eligible_positives=eligible_positives,
    )


def train_reference_uniform_pairwise_fm(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the exact protected uniform control with explicit seed and bound diagnostics."""

    return _fit_reference_observed_pair_fm(
        features,
        targets,
        user_groups,
        arm=UNIFORM_CONTROL_ARM,
        pairs_per_epoch=REFERENCE_PAIRS_PER_EPOCH,
        epochs=REFERENCE_EPOCHS,
        seed=seed,
    )


def train_reference_duration_pairwise_fm(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the equal-budget 50/50 duration-conditioned treatment."""

    return _fit_reference_observed_pair_fm(
        features,
        targets,
        user_groups,
        arm=DURATION_CONDITIONED_ARM,
        pairs_per_epoch=REFERENCE_PAIRS_PER_EPOCH,
        epochs=REFERENCE_EPOCHS,
        seed=seed,
    )


def _reference_state(checkpoint: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(checkpoint) != _CHECKPOINT_KEYS:
        raise ReferenceObservedPairFMError("observed-pair checkpoint inventory is invalid")
    return {name: checkpoint[name] for name in _STATE_KEYS}


def reference_observed_pair_fm_scores(
    features: np.ndarray,
    checkpoint: dict[str, np.ndarray],
) -> np.ndarray:
    """Predict in caller row order without target or group input."""

    try:
        scores = reference_pairwise_fm_scores(features, _reference_state(checkpoint))
        return np.ascontiguousarray(scores, dtype=np.float64)
    except ValueError as exc:
        raise ReferenceObservedPairFMError(str(exc)) from exc


def reference_observed_pair_fm_diagnostics(
    checkpoint: dict[str, np.ndarray],
) -> dict[str, int | float]:
    """Return bounded training/resource diagnostics without organizer metrics."""

    _reference_state(checkpoint)
    scalar_names = (
        "observed_pair_arm_code",
        "observed_pair_conditioned_eligible_groups",
        "observed_pair_conditioned_eligible_positives",
        "observed_pair_epochs",
        "observed_pair_intervention_pairs",
        "observed_pair_optimizer_steps",
        "observed_pair_pairs_per_epoch",
        "observed_pair_schema_version",
        "observed_pair_seed",
    )
    if any(checkpoint[name].shape != () for name in scalar_names):
        raise ReferenceObservedPairFMError("observed-pair scalar diagnostics are invalid")
    arm_code = int(checkpoint["observed_pair_arm_code"].item())
    pairs_per_epoch = int(checkpoint["observed_pair_pairs_per_epoch"].item())
    epochs = int(checkpoint["observed_pair_epochs"].item())
    optimizer_steps = int(checkpoint["observed_pair_optimizer_steps"].item())
    treatment_seeds = checkpoint["observed_pair_treatment_seeds"]
    initial_digest = checkpoint["observed_pair_initial_state_sha256"]
    if (
        arm_code not in (0, 1)
        or int(checkpoint["observed_pair_schema_version"].item()) != OBSERVED_PAIR_SCHEMA_VERSION
        or treatment_seeds.shape != (epochs,)
        or treatment_seeds.dtype != np.dtype("<u4")
        or initial_digest.shape != (32,)
        or initial_digest.dtype != np.dtype("uint8")
        or optimizer_steps != epochs * math.ceil(pairs_per_epoch / REFERENCE_BATCH_SIZE)
    ):
        raise ReferenceObservedPairFMError("observed-pair diagnostics are inconsistent")
    final_loss = float(checkpoint["reference_final_pairwise_loss"].item())
    if not math.isfinite(final_loss):
        raise ReferenceObservedPairFMError("observed-pair final loss is non-finite")
    return {
        "observed_pair_arm_code": arm_code,
        "observed_pair_batch_size": REFERENCE_BATCH_SIZE,
        "observed_pair_conditioned_eligible_groups": int(
            checkpoint["observed_pair_conditioned_eligible_groups"].item()
        ),
        "observed_pair_conditioned_eligible_positives": int(
            checkpoint["observed_pair_conditioned_eligible_positives"].item()
        ),
        "observed_pair_epochs": epochs,
        "observed_pair_factor_dim": REFERENCE_FACTOR_DIM,
        "observed_pair_feature_count": len(REFERENCE_FEATURE_POSITIONS),
        "observed_pair_final_training_loss": final_loss,
        "observed_pair_intervention_pairs": int(
            checkpoint["observed_pair_intervention_pairs"].item()
        ),
        "observed_pair_l2": REFERENCE_L2,
        "observed_pair_learning_rate": REFERENCE_LEARNING_RATE,
        "observed_pair_optimizer_steps": optimizer_steps,
        "observed_pair_pairs_per_epoch": pairs_per_epoch,
        "observed_pair_sampled_pairs": int(checkpoint["reference_sampled_pairs"].item()),
        "observed_pair_seed": int(checkpoint["observed_pair_seed"].item()),
    }


__all__ = [
    "DURATION_CONDITIONED_ARM",
    "OBSERVED_PAIR_SCHEMA_VERSION",
    "UNIFORM_CONTROL_ARM",
    "ReferenceObservedPairFMError",
    "reference_observed_pair_fm_diagnostics",
    "reference_observed_pair_fm_scores",
    "train_reference_duration_pairwise_fm",
    "train_reference_uniform_pairwise_fm",
]
