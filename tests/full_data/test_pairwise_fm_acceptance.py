from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

import kuairand_agent.candidates.pairwise_fm as pairwise_fm_module
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.candidates.pairwise_fm import (
    EncodedFMInputs,
    PairwiseFMAdapter,
    PairwiseFMConfig,
    PairwiseFMTrainingData,
)
from kuairand_agent.data.audit import DataAuditReport
from kuairand_agent.data.canonical import CanonicalDataset, ProtectedTargets, TrainingTargets
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"
EVIDENCE_ENV = "KUAIRAND_PAIRWISE_ACCEPTANCE_OUTPUT"
PAIR_COUNT = 8_192
EPOCH_COUNT = 1
EXPECTED_ELIGIBLE_TRAIN_ROWS = 1_130_240


def _source_digest() -> str:
    source_path = Path(pairwise_fm_module.__file__).resolve(strict=True)
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _write_optional_evidence(evidence: dict[str, object]) -> None:
    requested = os.environ.get(EVIDENCE_ENV)
    if requested is None:
        return
    destination = Path(requested).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    if destination.exists():
        assert destination.read_bytes() == payload
        return
    destination.write_bytes(payload)


def test_pairwise_fm_objective_and_sampler_run_on_verified_official_data(
    official_dataset: CanonicalDataset,
    official_audit: DataAuditReport,
) -> None:
    """Bounded gate: full train scan/encoding, one epoch, and 8,192 logged pairs."""

    train = official_dataset.train
    valid = official_dataset.valid
    assert isinstance(train.targets, TrainingTargets)
    assert isinstance(valid.targets, ProtectedTargets)
    assert official_dataset.final.targets is None
    assert official_audit.final_outcome_trace.manifest()["outcome_cells_scored"] == 0

    encoding = StarterEncoding.fit(train.inputs)
    encoded_train = EncodedFMInputs.from_encoding(
        encoding,
        train.inputs,
        phase=DataPhase.TRAIN,
    )
    training = PairwiseFMTrainingData(
        inputs=encoded_train,
        labels=np.asarray(train.targets.long_view, dtype=np.int8),
        user_ids=train.inputs.user_id,
        training_targets_digest=train.targets.digest,
        target_inputs_digest=train.inputs.digest,
    )
    config = PairwiseFMConfig(
        seed=20260828,
        pair_batch_size=PAIR_COUNT,
        pairs_per_epoch=PAIR_COUNT,
        max_epochs=EPOCH_COUNT,
        device_metadata="official-full-data-acceptance-cpu",
    )
    adapter = PairwiseFMAdapter(source_digest=_source_digest(), config=config)
    run = adapter.fit(training)

    assert run.checkpoint.epochs_completed == EPOCH_COUNT
    assert run.checkpoint.sampled_pairs == PAIR_COUNT
    assert run.trace[0].batch_sizes == (PAIR_COUNT,)
    assert run.trace[0].optimizer_steps == 1
    assert run.stored_row_index_count == EXPECTED_ELIGIBLE_TRAIN_ROWS
    assert run.stored_row_index_count < train.row_count
    assert run.eligible_user_count > 0
    assert run.eligible_positive_count > 0
    assert run.pair_space_size >= PAIR_COUNT
    assert math.isfinite(run.trace[0].mean_pairwise_loss)
    assert run.trace[0].mean_pairwise_loss > 0.0

    encoded_valid = EncodedFMInputs.from_encoding(
        encoding,
        valid.inputs,
        phase=DataPhase.OUTER_VALID,
    )
    prediction = adapter.predict(run.checkpoint, encoded_valid)
    replay = adapter.predict(
        run.checkpoint,
        encoded_valid,
        expected_prediction_digest=prediction.prediction_digest,
    )
    np.testing.assert_array_equal(prediction.scores, replay.scores)
    assert prediction.scores.size == 124_909
    assert np.isfinite(prediction.scores).all()

    split = SplitIdentity(
        name="outer_valid",
        token="official-pairwise-bounded-acceptance-v1",
        expected_count=valid.row_count,
    )
    alignment = Alignment.from_ids(
        split=split,
        row_ids=valid.alignment.row_id,
        user_ids=valid.alignment.user_id,
        video_ids=valid.alignment.video_id,
    )
    scorer = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)
    score = scorer.score_with_encoded_labels(
        alignment=alignment,
        split=split,
        labels=valid.targets.reveal_for_scorer(),
        scores=prediction.scores,
    )
    assert score.prediction_digest == prediction.prediction_digest
    assert 0.0 <= score.gauc <= 1.0
    assert 0.0 <= score.ndcg_at_5 <= 1.0
    # The untouched organizer evaluates encoded float32 labels; NumPy preserves that scalar
    # dtype through its primary arithmetic, so compare at the same declared metric tolerance.
    assert math.isclose(
        score.primary,
        (score.gauc + score.ndcg_at_5) / 2.0,
        rel_tol=0.0,
        abs_tol=2e-7,
    )

    _write_optional_evidence(
        {
            "schema_version": 1,
            "gate": "official_pairwise_fm_bounded_acceptance",
            "dataset_audit_digest": official_audit.digest,
            "canonical_dataset_digest": official_dataset.digest,
            "source_digest": _source_digest(),
            "config": config.manifest(),
            "workload": {
                "training_rows_scanned": train.row_count,
                "validation_rows_predicted": valid.row_count,
                "epochs": EPOCH_COUNT,
                "sampled_logged_pairs": PAIR_COUNT,
                "full_catalog_negatives": False,
            },
            "run_logical_digest": run.logical_digest,
            "checkpoint_digest": run.checkpoint.digest,
            "prediction_digest": prediction.prediction_digest,
            "protected_score": {
                "GAUC": score.gauc,
                "nDCG@5": score.ndcg_at_5,
                "primary": score.primary,
                "scorer_digest": score.scorer_digest,
            },
            "final_period": {
                "outcomes_accessed": False,
                "outcomes_scored": False,
                "target_capability": None,
            },
        }
    )
