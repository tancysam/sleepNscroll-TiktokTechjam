from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from kuairand_agent.baselines.artifacts import StarterFMCheckpoint
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.starter_fm import (
    StarterFMAdapter,
    StarterFMConfig,
    StarterFMError,
    StarterFMRun,
)
from kuairand_agent.data.canonical import CanonicalInputs

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"


def _inputs(prefix: str, rows: int, *, start_time: int = 0) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u{index % 3}" for index in range(rows)),
        video_id=tuple(f"{prefix}-v{index % 4}" for index in range(rows)),
        date=tuple(20220408 for _ in range(rows)),
        duration_ms=tuple(float(1000 + index * 137) for index in range(rows)),
        tab=tuple(str(index % 2) for index in range(rows)),
        author_id=tuple(f"a{index % 2}" for index in range(rows)),
        time_ms=tuple(start_time + index for index in range(rows)),
    )


@dataclass(frozen=True)
class _Targets:
    primary: npt.NDArray[np.int8]
    training_inputs_digest: str
    digest: str = "d" * 64

    @property
    def row_count(self) -> int:
        return int(self.primary.size)


@dataclass(frozen=True)
class _GenericTargets:
    primary: npt.NDArray[np.generic]
    training_inputs_digest: str
    digest: str = "d" * 64

    @property
    def row_count(self) -> int:
        return int(self.primary.size)


@dataclass(frozen=True)
class _BoundScorer:
    validation_inputs_digest: str
    callback: Callable[[npt.NDArray[np.float64]], object]

    def __call__(self, scores: npt.NDArray[np.float64]) -> object:
        return self.callback(scores)


def _fit_with_scorer(
    scorer: Callable[[npt.NDArray[np.float64]], object],
    *,
    seed: int = 0,
) -> StarterFMRun:
    train = _inputs("train", 8)
    valid = _inputs("valid", 4, start_time=100)
    return StarterFMAdapter(
        starter_dir=STARTER,
        config=StarterFMConfig(seed=seed),
    ).fit(
        encoding=StarterEncoding.fit(train),
        train_inputs=train,
        train_targets=_Targets(
            np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8),
            training_inputs_digest=train.digest,
        ),
        validation_inputs=valid,
        validation_scorer=_BoundScorer(valid.digest, scorer),
    )


def test_exact_training_stops_after_four_bad_epochs_and_restores_best_state() -> None:
    train = _inputs("train", 8)
    valid = _inputs("valid", 4, start_time=100)
    encoding = StarterEncoding.fit(train)
    targets = _Targets(
        np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8),
        training_inputs_digest=train.digest,
    )
    observed: list[np.ndarray] = []

    def constant_scorer(scores: npt.NDArray[np.float64]) -> dict[str, float]:
        assert not scores.flags.writeable
        observed.append(scores.copy())
        return {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5}

    adapter = StarterFMAdapter(starter_dir=STARTER, config=StarterFMConfig(seed=0))
    run = adapter.fit(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=valid,
        validation_scorer=_BoundScorer(valid.digest, constant_scorer),
    )

    assert run.checkpoint.best_epoch == 1
    assert run.checkpoint.epochs_completed == 5
    assert run.checkpoint.optimizer_steps == 5
    assert [epoch.improved for epoch in run.trace] == [True, False, False, False, False]
    np.testing.assert_array_equal(observed[0], observed[-1])
    np.testing.assert_array_equal(run.validation_predictions.scores, observed[-1])
    assert run.validation_predictions.digest == run.trace[0].prediction_digest


def test_early_stopping_threshold_preserves_organizer_float32_scalar_semantics() -> None:
    calls = 0
    threshold_edge = float(np.float32(0.50001))
    assert threshold_edge > 0.5 + 1e-5
    assert not np.float32(threshold_edge) > np.float32(0.5) + 1e-5

    def scorer(_: npt.NDArray[np.float64]) -> dict[str, float]:
        nonlocal calls
        calls += 1
        primary = threshold_edge if calls == 2 else 0.5
        return {"GAUC": primary, "nDCG@5": primary, "primary": primary}

    run = _fit_with_scorer(scorer)

    assert run.checkpoint.best_epoch == 1
    assert run.checkpoint.epochs_completed == 5
    assert not run.trace[1].improved


def test_first_update_matches_untouched_organizer_float32_golden() -> None:
    train = _inputs("train", 8)
    valid = _inputs("valid", 4, start_time=100)
    encoding = StarterEncoding.fit(train)
    targets = _Targets(
        np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8),
        training_inputs_digest=train.digest,
    )

    run = StarterFMAdapter(starter_dir=STARTER, config=StarterFMConfig(seed=0)).fit(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=valid,
        validation_scorer=_BoundScorer(
            valid.digest,
            lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5},
        ),
    )

    # Frozen by running the hash-pinned organizer FM directly with its separate
    # initialization and permutation RNGs.  These are independent literals,
    # not values recomputed by a second implementation inside the test.
    assert run.trace[0].mean_loss == 0.693239152431488
    assert run.checkpoint.b.tobytes().hex() == "7dbf0134"
    assert hashlib.sha256(run.checkpoint.V.astype("<f4").tobytes()).hexdigest() == (
        "fbd83db397058ac1b01a93894d263e0a390e19a3a1532c27f577644a765b9db3"
    )
    assert hashlib.sha256(run.checkpoint.W.astype("<f4").tobytes()).hexdigest() == (
        "bcaacd30b34b44ebd877902ea3ae1050af30099fea26b57d32c688b1419e6a72"
    )
    np.testing.assert_array_equal(
        run.validation_predictions.scores,
        np.asarray(
            [
                0.002868118928745389,
                -0.0015314295887947083,
                0.004123099613934755,
                -0.005258315242826939,
            ],
            dtype=np.float64,
        ),
    )


def test_fit_api_has_no_validation_or_final_target_parameter() -> None:
    parameters = inspect.signature(StarterFMAdapter.fit).parameters
    assert tuple(parameters) == (
        "self",
        "encoding",
        "train_inputs",
        "train_targets",
        "validation_inputs",
        "validation_scorer",
    )
    assert not any("target" in name and name != "train_targets" for name in parameters)
    assert tuple(inspect.signature(StarterFMAdapter.predict).parameters) == (
        "self",
        "checkpoint",
        "encoding",
        "inputs",
        "expected_prediction_digest",
    )


def test_config_freezes_every_organizer_hyperparameter_except_seed() -> None:
    config = StarterFMConfig(seed=4)

    assert (config.k, config.lr, config.l2) == (16, 0.001, 1e-6)
    assert (config.bs, config.epochs, config.patience) == (8192, 40, 4)
    assert config.threshold == 1e-5
    assert config.predict_batch_size == 200_000
    assert config.manifest()["device"] == "cpu"
    with pytest.raises(TypeError):
        StarterFMConfig(seed=0, k=8)  # type: ignore[call-arg]
    for seed in (True, -1, 2**32):
        with pytest.raises(StarterFMError, match="uint32-compatible"):
            StarterFMConfig(seed=seed)


@pytest.mark.parametrize(
    "metrics",
    [
        {"GAUC": 0.5, "nDCG@5": 0.5},
        {"GAUC": np.nan, "nDCG@5": 0.5, "primary": 0.5},
        {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.6},
        {"GAUC": True, "nDCG@5": 0.5, "primary": 0.5},
        {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5, "labels": [1, 0]},
    ],
)
def test_fit_rejects_invalid_protected_metric_contract(metrics: object) -> None:
    with pytest.raises(StarterFMError):
        _fit_with_scorer(lambda _: metrics)


def test_fit_rejects_nondeterministic_protected_rescore_after_best_restore() -> None:
    calls = 0

    def changing_scorer(_: npt.NDArray[np.float64]) -> dict[str, float]:
        nonlocal calls
        calls += 1
        primary = 0.75 if calls == 6 else 0.5
        return {"GAUC": primary, "nDCG@5": primary, "primary": primary}

    with pytest.raises(StarterFMError, match="protected scorer changed"):
        _fit_with_scorer(changing_scorer)


@pytest.mark.parametrize(
    ("primary", "target_digest", "aligned"),
    [
        (np.asarray([1, 0, 1], dtype=np.int8), "d" * 64, True),
        (np.asarray([1, 0, 2, 0, 1, 0, 0, 1], dtype=np.int8), "d" * 64, True),
        (np.asarray([True, False] * 4, dtype=np.bool_), "d" * 64, True),
        (np.asarray([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8), "D" * 64, True),
        (np.asarray([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8), "d" * 64, False),
    ],
)
def test_fit_rejects_invalid_training_target_capabilities(
    primary: npt.NDArray[np.generic],
    target_digest: str,
    aligned: bool,
) -> None:
    train = _inputs("train", 8)
    valid = _inputs("valid", 4, start_time=100)
    targets = _GenericTargets(
        primary,
        training_inputs_digest=train.digest if aligned else "0" * 64,
        digest=target_digest,
    )
    with pytest.raises(StarterFMError):
        StarterFMAdapter(starter_dir=STARTER).fit(
            encoding=StarterEncoding.fit(train),
            train_inputs=train,
            train_targets=targets,
            validation_inputs=valid,
            validation_scorer=_BoundScorer(
                valid.digest,
                lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5},
            ),
        )


def test_fit_and_replay_reject_mismatched_data_encoding_config_and_digest() -> None:
    train = _inputs("train", 8)
    valid = _inputs("valid", 4, start_time=100)
    encoding = StarterEncoding.fit(train)
    targets = _Targets(
        np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8),
        training_inputs_digest=train.digest,
    )
    adapter = StarterFMAdapter(starter_dir=STARTER, config=StarterFMConfig(seed=0))
    run = adapter.fit(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=valid,
        validation_scorer=_BoundScorer(
            valid.digest,
            lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5},
        ),
    )
    wrong_encoding = StarterEncoding.fit(_inputs("other", 8, start_time=200))

    with pytest.raises(StarterFMError, match="not fitted from"):
        adapter.fit(
            encoding=wrong_encoding,
            train_inputs=train,
            train_targets=targets,
            validation_inputs=valid,
            validation_scorer=_BoundScorer(
                valid.digest,
                lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5},
            ),
        )
    with pytest.raises(StarterFMError, match="checkpoint config"):
        StarterFMAdapter(starter_dir=STARTER, config=StarterFMConfig(seed=1)).predict(
            checkpoint=run.checkpoint,
            encoding=encoding,
            inputs=valid,
        )
    with pytest.raises(StarterFMError, match="checkpoint encoding"):
        adapter.predict(
            checkpoint=run.checkpoint,
            encoding=wrong_encoding,
            inputs=valid,
        )
    with pytest.raises(StarterFMError, match="prediction digest mismatch"):
        adapter.predict(
            checkpoint=run.checkpoint,
            encoding=encoding,
            inputs=valid,
            expected_prediction_digest="0" * 64,
        )

    false_seed = StarterFMCheckpoint(
        V=run.checkpoint.V,
        W=run.checkpoint.W,
        b=run.checkpoint.b,
        encoding_digest=run.checkpoint.encoding_digest,
        config_digest=run.checkpoint.config_digest,
        starter_manifest_digest=run.checkpoint.starter_manifest_digest,
        seed=1,
        best_epoch=run.checkpoint.best_epoch,
        epochs_completed=run.checkpoint.epochs_completed,
        optimizer_steps=run.checkpoint.optimizer_steps,
    )
    with pytest.raises(StarterFMError, match="checkpoint seed"):
        adapter.predict(
            checkpoint=false_seed,
            encoding=encoding,
            inputs=valid,
        )

    with pytest.raises(StarterFMError, match="validation scorer is not aligned"):
        adapter.fit(
            encoding=encoding,
            train_inputs=train,
            train_targets=targets,
            validation_inputs=valid,
            validation_scorer=_BoundScorer(
                "0" * 64,
                lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5},
            ),
        )


def test_multi_batch_training_records_every_organizer_update() -> None:
    train = _inputs("train", 8193)
    valid = _inputs("valid", 4, start_time=20_000)
    labels = np.asarray([index % 2 for index in range(len(train))], dtype=np.int8)

    run = StarterFMAdapter(starter_dir=STARTER).fit(
        encoding=StarterEncoding.fit(train),
        train_inputs=train,
        train_targets=_Targets(labels, training_inputs_digest=train.digest),
        validation_inputs=valid,
        validation_scorer=_BoundScorer(
            valid.digest,
            lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5},
        ),
    )

    assert [epoch.batch_count for epoch in run.trace] == [2, 2, 2, 2, 2]
    assert [epoch.optimizer_steps for epoch in run.trace] == [2, 4, 6, 8, 10]
    assert run.checkpoint.optimizer_steps == 10


def test_run_rejects_resource_facts_that_disagree_with_hashed_evidence() -> None:
    run = _fit_with_scorer(lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5})

    with pytest.raises(StarterFMError, match="resource validation row count"):
        replace(
            run,
            resources=replace(
                run.resources,
                validation_rows=run.resources.validation_rows + 1,
            ),
        )
