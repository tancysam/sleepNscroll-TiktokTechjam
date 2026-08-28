from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.baselines.qualification import (
    FinalPredictionEvidence,
    FMReplayEvidence,
    FMTrainingEvidence,
    QualificationError,
    QualificationMetrics,
    QualificationRequest,
    QualificationSnapshot,
    ResourceUsage,
    RungEvaluationEvidence,
    RungSummaryEvidence,
    run_qualification,
)
from kuairand_agent.scoring.submission import AlignmentRow, read_submission

_VALIDATION_ALIGNMENT = (
    AlignmentRow(0, "u0", "v0"),
    AlignmentRow(1, "u0", "v1"),
    AlignmentRow(2, "u0", "v2"),
    AlignmentRow(3, "u1", "v3"),
    AlignmentRow(4, "u1", "v4"),
    AlignmentRow(5, "u1", "v5"),
)
_FINAL_ALIGNMENT = (
    AlignmentRow(0, "final-u0", "final-v0"),
    AlignmentRow(1, "final-u0", "final-v1"),
    AlignmentRow(2, "final-u1", "final-v2"),
)
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metrics(delta: float = 0.0) -> QualificationMetrics:
    # The unrounded means live inside the published four-place bins and their exact formula mean
    # is above the 0.60155 half-up boundary, matching the official five-seed contract.
    gauc = 0.66744 + delta
    ndcg = 0.5357 + delta
    return QualificationMetrics(gauc, ndcg, (gauc + ndcg) / 2.0)


def _rung_metrics(name: str) -> QualificationMetrics:
    if name == "random":
        return QualificationMetrics(0.4993, 0.4675, 0.4834)
    return QualificationMetrics(0.6387, 0.5227, 0.5807)


class FixtureBackend:
    def __init__(
        self,
        *,
        mismatch_second_snapshot: bool = False,
        corrupt_replay_seed: int | None = None,
    ) -> None:
        self.mismatch_second_snapshot = mismatch_second_snapshot
        self.corrupt_replay_seed = corrupt_replay_seed
        self.snapshot_calls = 0
        self.train_calls: list[int] = []
        self.calls: list[str] = []
        self._metrics_by_prediction: dict[str, QualificationMetrics] = {}

    @staticmethod
    def _score_key(scores: np.ndarray[tuple[int], np.dtype[np.float64]]) -> str:
        return hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()

    def _register_scores(
        self,
        scores: np.ndarray[tuple[int], np.dtype[np.float64]],
        metrics: QualificationMetrics,
    ) -> None:
        self._metrics_by_prediction[self._score_key(scores)] = metrics

    def snapshot(self, request: QualificationRequest) -> QualificationSnapshot:
        del request
        self.snapshot_calls += 1
        self.calls.append(f"snapshot-{self.snapshot_calls}")
        canonical_digest = (
            _HASH_D if self.mismatch_second_snapshot and self.snapshot_calls == 2 else _HASH_C
        )
        return QualificationSnapshot(
            starter_manifest_digest=_HASH_A,
            audit_digest=_HASH_B,
            audit_manifest={"digest": _HASH_B, "final_outcome_cells_materialized": 0},
            canonical_digest=canonical_digest,
            canonical_manifest={
                "digest": canonical_digest,
                "final": {"target_access": "none"},
            },
            evaluator_golden_digest=_HASH_D,
            evaluator_golden_passed=True,
            validation_alignment=_VALIDATION_ALIGNMENT,
            validation_labels=(1, 0, 1, 0, 1, 0),
            final_alignment=_FINAL_ALIGNMENT,
            payload={"fixture": True},
        )

    def random_rungs(self, snapshot: QualificationSnapshot) -> RungSummaryEvidence:
        self.calls.append("random-rungs")
        metrics = _rung_metrics("random")
        evaluations = tuple(
            RungEvaluationEvidence(
                seed=seed,
                metrics=metrics,
                users=2,
                rows=snapshot.validation_count,
                scorer_digest=_HASH_A,
                prediction_digest=hashlib.sha256(f"random-{seed}".encode()).hexdigest(),
                split_digest=snapshot.canonical_digest,
                runtime_seconds=0.01,
            )
            for seed in range(5)
        )
        return RungSummaryEvidence("random", evaluations, metrics, True)

    def popularity_rung(self, snapshot: QualificationSnapshot) -> RungSummaryEvidence:
        self.calls.append("popularity-rung")
        metrics = _rung_metrics("item_popularity")
        evaluation = RungEvaluationEvidence(
            seed=None,
            metrics=metrics,
            users=2,
            rows=snapshot.validation_count,
            scorer_digest=_HASH_A,
            prediction_digest=_HASH_B,
            split_digest=snapshot.canonical_digest,
            runtime_seconds=0.01,
        )
        return RungSummaryEvidence("item_popularity", (evaluation,), metrics, True)

    def train_fm(
        self,
        snapshot: QualificationSnapshot,
        seed: int,
        artifact_dir: Path,
    ) -> FMTrainingEvidence:
        self.calls.append(f"train-{seed}")
        self.train_calls.append(seed)
        delta = (seed - 2) * 0.00001
        metrics = _metrics(delta)
        # Scores are close enough to exercise binary64 serialization but preserve a seed-specific
        # identity.  The second seed-0 call is the mandatory clean retrain and is byte-identical.
        scores = np.asarray(
            [
                0.900000000000001 + seed * 1e-5,
                0.3,
                0.1,
                0.8 + seed * 1e-5,
                0.2,
                -0.0,
            ],
            dtype=np.float64,
        )
        self._register_scores(scores, metrics)
        checkpoint_payload = f"fixture-checkpoint-seed-{seed}\n".encode()
        encoding_payload = b"fixture-encoding-v1\n"
        prediction_payload = scores.astype("<f8").tobytes()
        checkpoint_path = artifact_dir / "checkpoint.bin"
        encoding_path = artifact_dir / "encoding.bin"
        predictions_path = artifact_dir / "validation-predictions.bin"
        checkpoint_path.write_bytes(checkpoint_payload)
        encoding_path.write_bytes(encoding_payload)
        predictions_path.write_bytes(prediction_payload)
        return FMTrainingEvidence(
            seed=seed,
            validation_scores=scores,
            validation_metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_digest=_digest(checkpoint_payload),
            encoding_digest=_digest(encoding_payload),
            config_digest=_digest(f"config-{seed}".encode()),
            starter_manifest_digest=_HASH_A,
            artifact_paths=(checkpoint_path, encoding_path, predictions_path),
            artifact_sha256={
                checkpoint_path.name: _digest(checkpoint_payload),
                encoding_path.name: _digest(encoding_payload),
                predictions_path.name: _digest(prediction_payload),
            },
            training_trace={"best_epoch": 3, "epochs_completed": 5, "seed": seed},
            resources=ResourceUsage(1.0 + seed, 0.9 + seed, 10_000 + seed, "cpu"),
            organizer_parity_passed=True,
        )

    def replay_fm(
        self,
        snapshot: QualificationSnapshot,
        training: FMTrainingEvidence,
    ) -> FMReplayEvidence:
        del snapshot
        self.calls.append(f"replay-{training.seed}")
        scores = np.asarray(training.validation_scores, dtype=np.float64).copy()
        if self.corrupt_replay_seed == training.seed:
            scores[0] += 1.0
        return FMReplayEvidence(
            seed=training.seed,
            validation_scores=scores,
            validation_metrics=training.validation_metrics,
            checkpoint_digest=training.checkpoint_digest,
            resources=ResourceUsage(0.1, 0.09, 8_000, "cpu"),
        )

    def score_validation(
        self,
        snapshot: QualificationSnapshot,
        scores: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> QualificationMetrics:
        del snapshot
        try:
            return self._metrics_by_prediction[self._score_key(scores)]
        except KeyError as exc:
            raise QualificationError("fixture scorer received unknown predictions") from exc

    def predict_final(
        self,
        snapshot: QualificationSnapshot,
        training: FMTrainingEvidence,
    ) -> FinalPredictionEvidence:
        self.calls.append(f"final-{training.seed}")
        scores = np.asarray([0.12345640000000001, 0.12345649999999998, -0.0], dtype=np.float64)
        assert len(scores) == snapshot.final_count
        return FinalPredictionEvidence(
            scores=scores,
            checkpoint_digest=training.checkpoint_digest,
            resources=ResourceUsage(0.05, 0.04, 8_000, "cpu"),
        )


def _request(tmp_path: Path, name: str = "qualification") -> QualificationRequest:
    data = tmp_path / "data"
    starter = tmp_path / "starter"
    data.mkdir(parents=True, exist_ok=True)
    starter.mkdir(parents=True, exist_ok=True)
    return QualificationRequest(data, starter, tmp_path / name)


def test_fixture_qualification_publishes_replayable_immutable_fallback(tmp_path: Path) -> None:
    backend = FixtureBackend()
    request = _request(tmp_path)

    result = run_qualification(request, backend=backend)

    assert result.run_dir == request.run_dir.absolute()
    assert result.fallback_seed == 4
    assert result.launch_count == 6
    assert backend.train_calls == [0, 1, 2, 3, 4, 0]
    assert backend.calls[:4] == ["snapshot-1", "snapshot-2", "random-rungs", "popularity-rung"]
    assert backend.calls[4:9] == ["train-0", "train-1", "train-2", "train-3", "train-4"]
    assert backend.calls[9:14] == [
        "replay-0",
        "replay-1",
        "replay-2",
        "replay-3",
        "replay-4",
    ]
    assert backend.calls[-2:] == ["train-0", "final-4"]

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="ascii"))
    assert manifest["status"] == "baseline_reproduced"
    assert manifest["double_build_identity"] is True
    assert manifest["launch_accounting"]["charged_launches"] == 6
    assert [item["seed"] for item in manifest["launch_accounting"]["records"]] == [
        0,
        1,
        2,
        3,
        4,
        0,
    ]
    assert manifest["fallback"]["seed"] == 4
    assert manifest["fallback"]["replay_verified"] is True
    assert manifest["fallback"]["clean_seed_zero_retrain_verified"] is True
    assert manifest["final_period"] == {
        "input_rows": 3,
        "target_capability": None,
        "outcomes_accessed": False,
        "outcomes_scored": False,
    }
    assert manifest["fm"]["clean_seed_zero"]["prediction_identity"] is True
    assert manifest["fm"]["clean_seed_zero"]["within_user_order_identity"] is True
    assert manifest["digest"] == result.manifest_digest

    fallback_model = result.run_dir / "fallback" / "model"
    assert (fallback_model / "checkpoint.bin").read_bytes() == b"fixture-checkpoint-seed-4\n"
    assert stat_mode(fallback_model / "checkpoint.bin") == 0o400
    assert stat_mode(result.run_dir / "fallback") == 0o500
    validation = read_submission(result.validation_submission.path, _VALIDATION_ALIGNMENT)
    final = read_submission(result.final_submission.path, _FINAL_ALIGNMENT)
    assert validation.scores.tobytes() == result.validation_submission.scores.tobytes()
    assert final.scores.tobytes() == result.final_submission.scores.tobytes()
    assert final.scores[0] != final.scores[1]


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_two_clean_runs_have_identical_logical_manifest_and_artifact_bytes(
    tmp_path: Path,
) -> None:
    first_request = _request(tmp_path / "first")
    second_request = _request(tmp_path / "second")

    first = run_qualification(first_request, backend=FixtureBackend())
    second = run_qualification(second_request, backend=FixtureBackend())

    assert first.manifest_digest == second.manifest_digest
    assert (first.run_dir / "manifest.json").read_bytes() == (
        second.run_dir / "manifest.json"
    ).read_bytes()
    assert (first.run_dir / "final" / "submission.csv").read_bytes() == (
        second.run_dir / "final" / "submission.csv"
    ).read_bytes()
    assert (first.run_dir / "fallback" / "manifest.json").read_bytes() == (
        second.run_dir / "fallback" / "manifest.json"
    ).read_bytes()


def test_replay_failure_removes_private_staging_and_publishes_nothing(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(QualificationError, match="replay changed predictions"):
        run_qualification(request, backend=FixtureBackend(corrupt_replay_seed=2))

    assert not request.run_dir.exists()
    assert list(tmp_path.glob(f".{request.run_dir.name}.qualification-*")) == []


def test_double_build_identity_mismatch_stops_before_any_model_launch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = FixtureBackend(mismatch_second_snapshot=True)

    with pytest.raises(QualificationError, match="double build changed logical identity"):
        run_qualification(request, backend=backend)

    assert backend.train_calls == []
    assert not request.run_dir.exists()


def test_existing_run_directory_is_never_touched_or_overwritten(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.run_dir.mkdir()
    marker = request.run_dir / "user-owned.txt"
    marker.write_text("preserve me", encoding="utf-8")
    backend = FixtureBackend()

    with pytest.raises(QualificationError, match="already exists"):
        run_qualification(request, backend=backend)

    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert backend.calls == []


def test_final_prediction_length_or_nonfinite_value_fails_before_publication(
    tmp_path: Path,
) -> None:
    class InvalidFinalBackend(FixtureBackend):
        def predict_final(
            self,
            snapshot: QualificationSnapshot,
            training: FMTrainingEvidence,
        ) -> FinalPredictionEvidence:
            del snapshot
            return FinalPredictionEvidence(
                scores=(0.1, np.nan),
                checkpoint_digest=training.checkpoint_digest,
                resources=ResourceUsage(0.0, 0.0, 0, "cpu"),
            )

    request = _request(tmp_path)
    with pytest.raises(QualificationError, match="row count mismatch"):
        run_qualification(request, backend=InvalidFinalBackend())
    assert not request.run_dir.exists()


def test_fixture_report_contains_no_row_level_labels_or_final_outcomes(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = run_qualification(request, backend=FixtureBackend())

    report = (result.run_dir / "manifest.json").read_text(encoding="ascii")
    assert "validation_labels" not in report
    assert "final_targets" not in report
    assert "final_outcome_values" not in report
    assert "final_label_rate" not in report
    assert '"outcomes_accessed":false' in report
    assert '"outcomes_scored":false' in report


def test_artifact_symlink_from_backend_is_rejected(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("platform has no symlink support")

    class SymlinkBackend(FixtureBackend):
        def train_fm(
            self,
            snapshot: QualificationSnapshot,
            seed: int,
            artifact_dir: Path,
        ) -> FMTrainingEvidence:
            evidence = super().train_fm(snapshot, seed, artifact_dir)
            link = artifact_dir / "linked-checkpoint.bin"
            link.symlink_to(evidence.checkpoint_path)
            return FMTrainingEvidence(
                seed=evidence.seed,
                validation_scores=evidence.validation_scores,
                validation_metrics=evidence.validation_metrics,
                checkpoint_path=link,
                checkpoint_digest=evidence.checkpoint_digest,
                encoding_digest=evidence.encoding_digest,
                config_digest=evidence.config_digest,
                starter_manifest_digest=evidence.starter_manifest_digest,
                artifact_paths=(*evidence.artifact_paths, link),
                artifact_sha256={
                    **evidence.artifact_sha256,
                    link.name: evidence.artifact_sha256[evidence.checkpoint_path.name],
                },
                training_trace=evidence.training_trace,
                resources=evidence.resources,
                organizer_parity_passed=True,
            )

    request = _request(tmp_path)
    with pytest.raises(QualificationError, match="artifact_paths contain duplicates"):
        run_qualification(request, backend=SymlinkBackend())
    assert not request.run_dir.exists()
