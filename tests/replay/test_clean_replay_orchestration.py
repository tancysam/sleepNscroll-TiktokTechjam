from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.data.capabilities import CandidateInputs, DataPhase, build_candidate_inputs
from kuairand_agent.data.fields import STANDARD_LATE_MEMBER, VIDEO_BASIC_MEMBER, FieldKey
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.finalization.replay import (
    CleanReplayRequest,
    FrozenReplayIdentity,
    ReplayArtifacts,
    ReplayCancelledError,
    ReplayCapabilities,
    ReplayEquality,
    run_clean_replay,
)
from kuairand_agent.scoring.submission import AlignmentRow, read_submission


def _digest_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _environment_manifest() -> tuple[dict[str, object], str]:
    body: dict[str, object] = {
        "schema_version": 1,
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "platform": {"system": "fixture", "machine": "test"},
        "packages": {"numpy": "2.5.2"},
        "uv_lock_sha256": "f" * 64,
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    digest = hashlib.sha256(b"kuairand-environment-v1\0" + encoded).hexdigest()
    return body | {"digest": digest}, digest


def _inputs(phase: DataPhase, durations: list[float]) -> CandidateInputs:
    columns: dict[FieldKey, object] = {
        FieldKey(STANDARD_LATE_MEMBER, "user_id"): ["u1", "u1", "u2"],
        FieldKey(STANDARD_LATE_MEMBER, "video_id"): ["v1", "v2", "v3"],
        FieldKey(VIDEO_BASIC_MEMBER, "author_id"): ["a1", "a2", "a3"],
        FieldKey(STANDARD_LATE_MEMBER, "tab"): ["0", "1", "0"],
        FieldKey(STANDARD_LATE_MEMBER, "duration_ms"): durations,
    }
    return build_candidate_inputs(phase, columns)


def _write_npy(path: Path, values: np.ndarray) -> Path:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    return path


class _Backend:
    def replay_validation(self, *, workspace: object, inputs: CandidateInputs) -> np.ndarray:
        del workspace
        assert inputs.phase is DataPhase.OUTER_VALID
        return np.array([0.1, 0.9, 0.7], dtype=np.float64)

    def predict_final(self, *, workspace: object, inputs: CandidateInputs) -> np.ndarray:
        del workspace
        assert inputs.phase is DataPhase.FINAL
        assert not hasattr(inputs, "targets")
        return np.array([0.2, 0.8, 0.6], dtype=np.float64)


class _CancelBeforeFinalBackend(_Backend):
    def __init__(self, cancellation: threading.Event) -> None:
        self.cancellation = cancellation
        self.final_calls = 0

    def replay_validation(self, *, workspace: object, inputs: CandidateInputs) -> np.ndarray:
        result = super().replay_validation(workspace=workspace, inputs=inputs)
        self.cancellation.set()
        return result

    def predict_final(self, *, workspace: object, inputs: CandidateInputs) -> np.ndarray:
        self.final_calls += 1
        return super().predict_final(workspace=workspace, inputs=inputs)


def test_clean_replay_verifies_all_identities_and_writes_both_checked_csvs(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "candidate-source"
    source.mkdir()
    (source / "candidate.py").write_text("def score(rows): return rows\n", encoding="utf-8")
    source_ref = store.put_directory(source, kind=ArtifactKind.SOURCE)
    config_ref = store.put_bytes(b'{"seed":0}\n', kind=ArtifactKind.INPUT)
    features_ref = store.put_bytes(b'{"encoding":"v1"}\n', kind=ArtifactKind.INPUT)
    checkpoint_ref = store.put_bytes(b"checkpoint-v1", kind=ArtifactKind.CHECKPOINT)
    reference_path = _write_npy(
        tmp_path / "reference.npy", np.array([0.1, 0.9, 0.7], dtype=np.float64)
    )
    reference_ref = store.put_file(reference_path, kind=ArtifactKind.PREDICTION)
    environment, environment_digest = _environment_manifest()
    prediction_semantic_digest = "32592fb393d69088c854057a3f8435414d47fbfa87d09fa21c83f6a2cfe17689"
    identity = FrozenReplayIdentity(
        source_sha256=source_ref.sha256,
        config_sha256=config_ref.sha256,
        features_sha256=features_ref.sha256,
        checkpoint_sha256=checkpoint_ref.sha256,
        validation_prediction_artifact_sha256=reference_ref.sha256,
        validation_prediction_digest=prediction_semantic_digest,
        data_sha256="d" * 64,
        environment_sha256=environment_digest,
    )
    alignment = tuple(
        AlignmentRow(index, user, video)
        for index, (user, video) in enumerate((("u1", "v1"), ("u1", "v2"), ("u2", "v3")))
    )
    capabilities = ReplayCapabilities(
        data_sha256="d" * 64,
        validation_inputs=_inputs(DataPhase.OUTER_VALID, [100.0, 200.0, 300.0]),
        final_inputs=_inputs(DataPhase.FINAL, [110.0, 210.0, 310.0]),
        validation_alignment=alignment,
        final_alignment=alignment,
    )

    result = run_clean_replay(
        CleanReplayRequest(
            candidate_id="candidate-001",
            output_dir=tmp_path / "replay-output",
            identity=identity,
            artifacts=ReplayArtifacts(
                source=source_ref,
                config=config_ref,
                features=features_ref,
                checkpoint=checkpoint_ref,
                validation_predictions=reference_ref,
            ),
            environment=environment,
            equality=ReplayEquality.EXACT,
        ),
        artifact_store=store,
        capabilities=capabilities,
        backend=_Backend(),
        protected_metric_evaluator=lambda scores: {
            "GAUC": float(scores[1]),
            "nDCG@5": float(scores[2]),
            "primary": float((scores[1] + scores[2]) / 2.0),
        },
    )

    assert result.evidence.validation.exact_prediction_bytes is True
    assert result.evidence.validation.top5_order_identical is True
    assert result.evidence.validation.protected_metrics_identical is True
    assert result.evidence.validation.csv_round_trip_identity is True
    assert result.evidence.validation.csv_within_user_order_preserved is True
    assert result.evidence.validation.csv_top5_preserved is True
    assert result.evidence.validation.csv_protected_metrics_preserved is True
    assert result.evidence.final.final_outcomes_accessed is False
    assert result.evidence.final.csv_round_trip_identity is True
    assert result.evidence.final.row_count == 3
    assert result.evidence.identity == identity
    assert result.final_submission.exists()
    assert result.public_validation_submission.exists()
    assert read_submission(result.final_submission, alignment).row_count == 3
    assert read_submission(result.public_validation_submission, alignment).row_count == 3
    evidence = json.loads(result.evidence_path.read_text(encoding="ascii"))
    assert evidence["final"]["outcome_access"] == "none"
    assert evidence["workspace"]["clean_workspace_removed"] is True
    assert (result.source_dir / "candidate.py").read_text(encoding="utf-8").startswith("def")

    cancellation = threading.Event()
    cancelled_backend = _CancelBeforeFinalBackend(cancellation)
    with pytest.raises(ReplayCancelledError, match="final inference"):
        run_clean_replay(
            replace(
                CleanReplayRequest(
                    candidate_id="candidate-001",
                    output_dir=tmp_path / "unused",
                    identity=identity,
                    artifacts=ReplayArtifacts(
                        source=source_ref,
                        config=config_ref,
                        features=features_ref,
                        checkpoint=checkpoint_ref,
                        validation_predictions=reference_ref,
                    ),
                    environment=environment,
                    equality=ReplayEquality.EXACT,
                ),
                output_dir=tmp_path / "cancelled-replay-output",
            ),
            artifact_store=store,
            capabilities=capabilities,
            backend=cancelled_backend,
            protected_metric_evaluator=lambda scores: {
                "GAUC": float(scores[1]),
                "nDCG@5": float(scores[2]),
                "primary": float((scores[1] + scores[2]) / 2.0),
            },
            cancel_event=cancellation,
        )
    assert cancelled_backend.final_calls == 0
    assert not (tmp_path / "cancelled-replay-output").exists()
