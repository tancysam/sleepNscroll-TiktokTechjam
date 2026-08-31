from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.starter_fm import StarterFMAdapter, StarterFMConfig
from kuairand_agent.candidates.fusion import fuse_ranked_predictions
from kuairand_agent.contract import verify_starter_kit
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity
from tests.support.scripted_campaign import run_scripted_three_iteration_acceptance

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "autonomous_lab"
STARTER = ROOT / "kuairand-starter-kit"
EXPECTED_FILE_HASHES = {
    "official_fm_golden.json": "38c0feece48ff90e37d63a2e644e90d8fe1940b759b567753c86bba7be75f761",
    "organizer_scoring_golden.json": (
        "ff69468bb7beb57d5e1d6260a4b3c5959c02f3dbe7bc7f57a61ba67dc092d5fc"
    ),
    "rank_fusion_golden.json": "5936e2bf3a8ece5e6c0cc3bc253f5cb31d6bf5ab5e40e650aa68c401b8a52edd",
    "scripted_campaign_golden.json": (
        "de6100d4a84b8909482636a041a28eb4156f8e386e142d11619faa55ea808939"
    ),
}


def _fixture(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIXTURES / name).read_text(encoding="utf-8")))


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
class _Scorer:
    validation_inputs_digest: str
    callback: Callable[[npt.NDArray[np.float64]], object]

    def __call__(self, scores: npt.NDArray[np.float64]) -> object:
        return self.callback(scores)


def test_phase_minus_one_fixture_files_are_content_frozen() -> None:
    for name, expected in EXPECTED_FILE_HASHES.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == expected


def test_official_fm_matches_frozen_first_update() -> None:
    golden = _fixture("official_fm_golden.json")
    train = _inputs("train", 8)
    valid = _inputs("valid", 4, start_time=100)
    run = StarterFMAdapter(starter_dir=STARTER, config=StarterFMConfig(seed=0)).fit(
        encoding=StarterEncoding.fit(train),
        train_inputs=train,
        train_targets=_Targets(
            np.asarray([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8),
            training_inputs_digest=train.digest,
        ),
        validation_inputs=valid,
        validation_scorer=_Scorer(
            valid.digest,
            lambda _: {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5},
        ),
    )

    assert run.trace[0].mean_loss == golden["first_epoch_mean_loss"]
    np.testing.assert_array_equal(run.validation_predictions.scores, golden["predictions_f64"])
    assert run.validation_predictions.digest == golden["prediction_digest"]
    checkpoint = golden["checkpoint"]
    assert isinstance(checkpoint, dict)
    assert run.checkpoint.b.tobytes().hex() == checkpoint["bias_f32_hex"]
    assert (
        hashlib.sha256(run.checkpoint.V.astype("<f4").tobytes()).hexdigest()
        == checkpoint["embedding_sha256"]
    )
    assert (
        hashlib.sha256(run.checkpoint.W.astype("<f4").tobytes()).hexdigest()
        == checkpoint["linear_sha256"]
    )


def test_organizer_scoring_matches_frozen_fixture() -> None:
    golden = _fixture("organizer_scoring_golden.json")
    alignment_data = golden["alignment"]
    assert isinstance(alignment_data, dict)
    split = SplitIdentity(
        name=str(alignment_data["split"]),
        token=str(alignment_data["split_token"]),
        expected_count=int(alignment_data["row_count"]),
    )
    alignment = Alignment.from_ids(
        split=split,
        user_ids=alignment_data["users"],
        video_ids=alignment_data["videos"],
    )
    result = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment).score(
        alignment=alignment,
        split=split,
        labels=golden["labels"],
        scores=golden["scores"],
    )
    metrics = golden["metrics"]
    assert isinstance(metrics, dict)
    assert result.gauc == metrics["GAUC"]
    assert result.ndcg_at_5 == metrics["nDCG@5"]
    assert result.primary == metrics["primary"]
    assert result.prediction_digest == golden["prediction_digest"]


def test_rank_fusion_matches_frozen_exact_vector() -> None:
    golden = _fixture("rank_fusion_golden.json")
    members = golden["members"]
    assert isinstance(members, list) and len(members) == 2
    result = fuse_ranked_predictions(
        user_ids=golden["user_ids"],
        video_ids=golden["video_ids"],
        first_scores=members[0],
        second_scores=members[1],
        weights=tuple(golden["weights"]),
        phase=DataPhase.FINAL,
    )
    np.testing.assert_array_equal(result.scores, golden["expected_scores_f64"])
    assert result.prediction_digest == golden["prediction_digest"]
    assert result.fusion_digest == golden["fusion_digest"]


def test_scripted_campaign_matches_frozen_behavior(tmp_path: Path) -> None:
    golden = _fixture("scripted_campaign_golden.json")
    starter_digest = verify_starter_kit(STARTER).manifest_sha256
    evidence = run_scripted_three_iteration_acceptance(
        tmp_path / "scripted",
        starter_dir=STARTER,
        verified_audit_digest="a" * 64,
        verified_starter_digest=starter_digest,
        verified_dataset_digest="b" * 64,
    )

    assert list(evidence.statuses) == golden["statuses"]
    assert evidence.selected_candidate_id == golden["selected_candidate_id"]
    assert evidence.fallback_candidate_id == golden["fallback_candidate_id"]
    assert list(evidence.repairs) == golden["repairs"]
    assert list(evidence.trusted_evaluation_present) == golden["trusted_evaluation_present"]
    assert evidence.evaluator_calls == golden["evaluator_calls"]
    assert evidence.incumbent_id == golden["incumbent_id"]
    assert evidence.incumbent_primary == golden["incumbent_primary"]
    assert evidence.launches_used == golden["launches_used"]
    assert evidence.completed_iterations == golden["completed_iterations"]
    assert evidence.best_primary == golden["best_primary"]
    assert evidence.execution_count == golden["execution_count"]
    assert set(golden["operations_required"]) <= evidence.operations
    assert set(golden["artifact_roles_required"]) <= evidence.artifact_roles
    assert set(golden["failure_categories_required"]) <= evidence.failure_categories
