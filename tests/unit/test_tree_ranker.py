from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pytest

from kuairand_agent.candidates.grouping import build_user_grouping
from kuairand_agent.candidates.tree_ranker import (
    BackendFitRequest,
    BackendFitResult,
    InnerValidationSet,
    LambdaRankConfig,
    TreeRankerDependencyError,
    TreeRankerError,
    fit_lambdarank,
    predict_lambdarank,
)
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.data.causal_features import FeatureMatrix


@dataclass
class RecordingBackend:
    identity: str = "fake-lightgbm:1"
    fit_requests: list[BackendFitRequest] = field(default_factory=list)
    predict_features: list[npt.NDArray[np.float64]] = field(default_factory=list)

    def fit(self, request: BackendFitRequest) -> BackendFitResult:
        self.fit_requests.append(request)
        return BackendFitResult(model_text="fake-model-v1", best_iteration=7)

    def predict(
        self,
        *,
        model_text: str,
        features: npt.NDArray[np.float64],
        num_iteration: int,
    ) -> object:
        assert model_text == "fake-model-v1"
        assert num_iteration == 7
        self.predict_features.append(features)
        return features[:, 0] - 0.25 * features[:, 1]


@dataclass
class MalformedPredictionBackend(RecordingBackend):
    prediction_result: object = None

    def predict(
        self,
        *,
        model_text: str,
        features: npt.NDArray[np.float64],
        num_iteration: int,
    ) -> object:
        return self.prediction_result


class LabelsMustNotBeInspected:
    def __array__(self) -> npt.NDArray[np.float64]:
        raise AssertionError("forbidden phase inspected labels")


def test_fit_groups_only_the_private_training_view_and_predicts_in_canonical_order() -> None:
    features = FeatureMatrix(
        [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]],
        ["duration_log", "item_rate"],
    )
    grouping = build_user_grouping(
        user_ids=["b", "a", "b", "a"],
        video_ids=["v0", "v1", "v2", "v3"],
        phase=DataPhase.TRAIN,
    )
    config = LambdaRankConfig(seed=2, num_threads=3, num_boost_round=17)
    backend = RecordingBackend()

    checkpoint = fit_lambdarank(
        features,
        labels=[1, 0, 0, 1],
        grouping=grouping,
        phase=DataPhase.TRAIN,
        config=config,
        backend=backend,
    )

    request = backend.fit_requests[0]
    np.testing.assert_array_equal(
        request.train_features,
        [[10.0, 1.0], [30.0, 3.0], [20.0, 2.0], [40.0, 4.0]],
    )
    np.testing.assert_array_equal(request.train_labels, [1, 0, 0, 1])
    assert request.train_group_sizes == (2, 2)
    assert request.eval_at == (5,)
    assert request.inner_valid_features is None
    assert request.early_stopping_rounds is None
    assert request.params["objective"] == "lambdarank"
    assert request.params["device_type"] == "cpu"
    assert request.params["deterministic"] is True
    assert request.params["force_col_wise"] is True
    assert "force_row_wise" not in request.params
    assert request.params["label_gain"] == (0, 1)
    assert request.params["num_threads"] == 3
    for name in (
        "seed",
        "data_random_seed",
        "feature_fraction_seed",
        "bagging_seed",
        "extra_seed",
    ):
        assert request.params[name] == 2

    replay = fit_lambdarank(
        features,
        labels=[1, 0, 0, 1],
        grouping=grouping,
        phase=DataPhase.TRAIN,
        config=config,
        backend=RecordingBackend(),
    )
    assert checkpoint.digest == replay.digest
    assert checkpoint.training_feature_digest == features.digest
    assert checkpoint.training_grouping_digest == grouping.digest
    assert len(checkpoint.training_target_digest) == 64
    assert checkpoint.feature_names == features.feature_names

    query = FeatureMatrix(
        [[4.0, 8.0], [3.0, 4.0], [2.0, 0.0]],
        ["duration_log", "item_rate"],
    )
    prediction = predict_lambdarank(
        checkpoint,
        query,
        phase=DataPhase.OUTER_VALID,
        backend=backend,
    )

    np.testing.assert_array_equal(prediction.scores, [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(backend.predict_features[0], query.values)
    assert prediction.checkpoint_digest == checkpoint.digest
    assert prediction.feature_digest == query.digest
    assert prediction.phase is DataPhase.OUTER_VALID
    assert not prediction.scores.flags.writeable


def test_early_stopping_is_available_only_for_a_train_derived_inner_fold() -> None:
    train_features = FeatureMatrix(
        [[1.0], [2.0], [3.0], [4.0]],
        ["causal_item_rate"],
    )
    train_grouping = build_user_grouping(
        ["u", "v", "u", "v"],
        ["a", "b", "c", "d"],
        phase=DataPhase.INNER_TRAIN,
    )
    valid_features = FeatureMatrix(
        [[5.0], [6.0], [7.0], [8.0]],
        ["causal_item_rate"],
    )
    valid_grouping = build_user_grouping(
        ["v", "u", "v", "u"],
        ["e", "f", "g", "h"],
        phase=DataPhase.INNER_VALID,
    )
    inner_valid = InnerValidationSet(
        valid_features,
        labels=[0, 1, 1, 0],
        grouping=valid_grouping,
    )
    backend = RecordingBackend()
    config = LambdaRankConfig(num_boost_round=20, early_stopping_rounds=4)

    checkpoint = fit_lambdarank(
        train_features,
        labels=[1, 0, 0, 1],
        grouping=train_grouping,
        phase=DataPhase.INNER_TRAIN,
        config=config,
        inner_valid=inner_valid,
        backend=backend,
    )

    request = backend.fit_requests[0]
    np.testing.assert_array_equal(request.inner_valid_features, [[5.0], [7.0], [6.0], [8.0]])
    np.testing.assert_array_equal(request.inner_valid_labels, [0, 1, 1, 0])
    assert request.inner_valid_group_sizes == (2, 2)
    assert request.early_stopping_rounds == 4
    assert checkpoint.inner_validation_digest == inner_valid.digest

    full_grouping = build_user_grouping(
        ["u", "v", "u", "v"],
        ["a", "b", "c", "d"],
        phase=DataPhase.TRAIN,
    )
    with pytest.raises(TreeRankerError, match="only with inner_train"):
        fit_lambdarank(
            train_features,
            labels=[1, 0, 0, 1],
            grouping=full_grouping,
            phase=DataPhase.TRAIN,
            config=config,
            inner_valid=inner_valid,
            backend=RecordingBackend(),
        )


def test_default_backend_imports_lightgbm_only_when_fit_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def unavailable(name: str) -> object:
        imported.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    features = FeatureMatrix([[1.0], [2.0]], ["duration_log"])
    grouping = build_user_grouping(
        ["u", "u"],
        ["a", "b"],
        phase=DataPhase.TRAIN,
    )

    assert imported == []
    with pytest.raises(TreeRankerDependencyError, match="research-tree"):
        fit_lambdarank(
            features,
            labels=[1, 0],
            grouping=grouping,
            phase=DataPhase.TRAIN,
        )
    assert imported == ["lightgbm"]


def test_default_backend_reports_an_unavailable_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_runtime(name: str) -> object:
        raise OSError(f"cannot load native runtime for {name}")

    monkeypatch.setattr(importlib, "import_module", unavailable_runtime)
    features = FeatureMatrix([[1.0], [2.0]], ["duration_log"])
    grouping = build_user_grouping(
        ["u", "u"],
        ["a", "b"],
        phase=DataPhase.TRAIN,
    )

    with pytest.raises(TreeRankerDependencyError, match="native runtime"):
        fit_lambdarank(
            features,
            labels=[1, 0],
            grouping=grouping,
            phase=DataPhase.TRAIN,
        )


def test_label_and_prediction_phase_contracts_fail_before_forbidden_label_access() -> None:
    features = FeatureMatrix([[1.0], [2.0]], ["duration_log"])
    grouping = build_user_grouping(
        ["u", "u"],
        ["a", "b"],
        phase=DataPhase.TRAIN,
    )
    for phase in (DataPhase.INNER_VALID, DataPhase.OUTER_VALID, DataPhase.FINAL):
        with pytest.raises(TreeRankerError, match="only for train or inner_train"):
            fit_lambdarank(
                features,
                labels=LabelsMustNotBeInspected(),  # type: ignore[arg-type]
                grouping=grouping,
                phase=phase,
                backend=RecordingBackend(),
            )

    checkpoint = fit_lambdarank(
        features,
        labels=[1, 0],
        grouping=grouping,
        phase=DataPhase.TRAIN,
        backend=RecordingBackend(),
    )
    for phase in (DataPhase.TRAIN, DataPhase.INNER_TRAIN):
        with pytest.raises(TreeRankerError, match="prediction is allowed only"):
            predict_lambdarank(
                checkpoint,
                features,
                phase=phase,
                backend=RecordingBackend(),
            )


@pytest.mark.parametrize(
    "forbidden_name",
    ["show_cnt", "fans_user_num", "onehot_feat0", "visible_status", "is_rand"],
)
def test_tree_ranker_rejects_raw_statistic_snapshot_and_randomized_features(
    forbidden_name: str,
) -> None:
    features = FeatureMatrix([[1.0], [2.0]], [forbidden_name])
    grouping = build_user_grouping(
        ["u", "u"],
        ["a", "b"],
        phase=DataPhase.TRAIN,
    )

    with pytest.raises(TreeRankerError, match="blocked raw feature"):
        fit_lambdarank(
            features,
            labels=[1, 0],
            grouping=grouping,
            phase=DataPhase.TRAIN,
            backend=RecordingBackend(),
        )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([1.0], "expected 2"),
        ([1.0, np.nan], "only finite"),
        ([[1.0], [2.0]], "one-dimensional"),
        ([True, False], "real numeric dtype"),
    ],
)
def test_prediction_rejects_malformed_backend_output(result: object, message: str) -> None:
    features = FeatureMatrix([[1.0], [2.0]], ["duration_log"])
    grouping = build_user_grouping(
        ["u", "u"],
        ["a", "b"],
        phase=DataPhase.TRAIN,
    )
    training_backend = RecordingBackend(identity="malformed-backend:1")
    checkpoint = fit_lambdarank(
        features,
        labels=[1, 0],
        grouping=grouping,
        phase=DataPhase.TRAIN,
        backend=training_backend,
    )

    with pytest.raises(TreeRankerError, match=message):
        predict_lambdarank(
            checkpoint,
            features,
            phase=DataPhase.FINAL,
            backend=MalformedPredictionBackend(
                identity="malformed-backend:1",
                prediction_result=result,
            ),
        )


def test_prediction_requires_checkpoint_feature_order_and_backend_identity() -> None:
    features = FeatureMatrix([[1.0, 2.0], [3.0, 4.0]], ["first", "second"])
    grouping = build_user_grouping(
        ["u", "u"],
        ["a", "b"],
        phase=DataPhase.TRAIN,
    )
    checkpoint = fit_lambdarank(
        features,
        labels=[1, 0],
        grouping=grouping,
        phase=DataPhase.TRAIN,
        backend=RecordingBackend(),
    )

    reordered = FeatureMatrix([[2.0, 1.0], [4.0, 3.0]], ["second", "first"])
    with pytest.raises(TreeRankerError, match="names/order"):
        predict_lambdarank(
            checkpoint,
            reordered,
            phase=DataPhase.FINAL,
            backend=RecordingBackend(),
        )
    with pytest.raises(TreeRankerError, match="backend identity"):
        predict_lambdarank(
            checkpoint,
            features,
            phase=DataPhase.FINAL,
            backend=RecordingBackend(identity="different:1"),
        )


def test_checkpoint_identity_changes_with_model_bytes_and_rejects_malformed_digests() -> None:
    features = FeatureMatrix([[1.0], [2.0]], ["duration_log"])
    grouping = build_user_grouping(
        ["u", "u"],
        ["a", "b"],
        phase=DataPhase.TRAIN,
    )
    checkpoint = fit_lambdarank(
        features,
        labels=[1, 0],
        grouping=grouping,
        phase=DataPhase.TRAIN,
        backend=RecordingBackend(),
    )

    changed_model = replace(checkpoint, model_text="different-model-bytes")
    assert changed_model.model_sha256 != checkpoint.model_sha256
    assert changed_model.digest != checkpoint.digest
    with pytest.raises(TreeRankerError, match="training_feature_digest"):
        replace(checkpoint, training_feature_digest="not-a-digest")
    with pytest.raises(TreeRankerError, match="training_phase"):
        replace(checkpoint, training_phase=DataPhase.FINAL)


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="optional research-tree dependency is not installed",
)
def test_pinned_lightgbm_backend_is_deterministic_on_a_tiny_cpu_fixture() -> None:
    try:
        importlib.import_module("lightgbm")
    except (ImportError, OSError) as exc:
        pytest.skip(f"optional LightGBM native runtime is unavailable: {exc}")

    features = FeatureMatrix(
        [
            [0.0, 0.1],
            [1.0, 0.2],
            [2.0, 0.3],
            [3.0, 0.4],
            [0.5, 0.8],
            [1.5, 0.7],
            [2.5, 0.6],
            [3.5, 0.5],
            [0.2, 0.9],
            [1.2, 1.0],
            [2.2, 1.1],
            [3.2, 1.2],
        ],
        ["duration_log", "causal_item_rate"],
    )
    users = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    grouping = build_user_grouping(
        users,
        [f"v{index}" for index in range(12)],
        phase=DataPhase.TRAIN,
    )
    labels = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    config = LambdaRankConfig(
        seed=17,
        num_threads=1,
        num_boost_round=8,
        early_stopping_rounds=3,
        num_leaves=4,
        min_data_in_leaf=1,
    )

    try:
        first = fit_lambdarank(
            features,
            labels,
            grouping=grouping,
            phase=DataPhase.TRAIN,
            config=config,
        )
    except TreeRankerDependencyError as exc:
        pytest.skip(f"native tree worker requires a fresh process: {exc}")
    second = fit_lambdarank(
        features,
        labels,
        grouping=grouping,
        phase=DataPhase.TRAIN,
        config=config,
    )
    first_prediction = predict_lambdarank(
        first,
        features,
        phase=DataPhase.INNER_VALID,
    )
    second_prediction = predict_lambdarank(
        second,
        features,
        phase=DataPhase.INNER_VALID,
    )

    np.testing.assert_array_equal(first_prediction.scores, second_prediction.scores)
    assert first.digest == second.digest
    assert first_prediction.prediction_digest == second_prediction.prediction_digest
