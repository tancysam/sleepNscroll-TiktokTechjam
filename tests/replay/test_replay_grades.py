from __future__ import annotations

from dataclasses import replace

import pytest

from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import ContractId, PredictionId
from kuairand_agent.finalization.replay import (
    CleanReplayEvidence,
    FinalReplayEvidence,
    FrozenReplayIdentity,
    ReplayEquality,
    ValidationReplayEvidence,
)
from kuairand_agent.finalization.replay_grades import (
    ReplayGradeError,
    ReplayGradeReceipt,
    combine_replay_grade_receipts,
    derive_clean_replay_grade_receipts,
    derive_cross_backend_portability_grade,
)


def _clean_replay(
    *,
    equality: ReplayEquality = ReplayEquality.EXACT,
    environment_sha256: str = "8" * 64,
    exact: bool = True,
) -> CleanReplayEvidence:
    reference = "1" * 64
    replayed = reference if exact else "2" * 64
    validation = ValidationReplayEvidence(
        row_count=3,
        reference_prediction_digest=reference,
        replay_prediction_digest=replayed,
        replay_prediction_file_sha256="3" * 64,
        exact_prediction_bytes=exact,
        maximum_absolute_difference=0.0 if exact else 0.001,
        top5_order_identical=True,
        protected_metrics_identical=True,
        metrics={"GAUC": 0.6, "nDCG@5": 0.8, "primary": 0.7},
        public_submission_sha256="4" * 64,
        public_submission_prediction_digest=replayed,
        csv_round_trip_identity=True,
        csv_within_user_order_preserved=True,
        csv_top5_preserved=True,
        csv_protected_metrics_preserved=True,
    )
    final = FinalReplayEvidence(
        row_count=3,
        prediction_digest="5" * 64,
        prediction_file_sha256="6" * 64,
        submission_sha256="7" * 64,
        submission_prediction_digest="5" * 64,
        finite_scores=True,
        csv_round_trip_identity=True,
    )
    return CleanReplayEvidence(
        candidate_id="candidate-1",
        identity=FrozenReplayIdentity(
            source_sha256="a" * 64,
            config_sha256="b" * 64,
            features_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
            validation_prediction_artifact_sha256="e" * 64,
            validation_prediction_digest=reference,
            data_sha256="f" * 64,
            environment_sha256=environment_sha256,
        ),
        equality=equality,
        absolute_tolerance=0.0 if exact else 0.01,
        training_replay="checkpoint_replay",
        validation=validation,
        final=final,
        validation_capability_digest="9" * 64,
        final_capability_digest="0" * 64,
    )


def test_exact_grades_are_derived_from_clean_replay_not_caller_declarations() -> None:
    contract_id = ContractId("a" * 64)
    prediction_id = PredictionId("b" * 64)
    evidence = _clean_replay()

    first = derive_clean_replay_grade_receipts(
        contract_id=contract_id,
        prediction_id=prediction_id,
        evidence=evidence,
    )
    second = derive_clean_replay_grade_receipts(
        contract_id=contract_id.value,
        prediction_id=prediction_id.value,
        evidence=evidence,
    )

    assert tuple(receipt.grade for receipt in first) == (
        ReplayGrade.SCORING_EXACT,
        ReplayGrade.EXPERIMENT_SAME_BACKEND,
    )
    assert tuple(receipt.receipt_id for receipt in first) == tuple(
        receipt.receipt_id for receipt in second
    )
    assert all(receipt.contract_id == contract_id.value for receipt in first)
    assert all(receipt.prediction_id == prediction_id.value for receipt in first)
    with pytest.raises(TypeError):
        ReplayGradeReceipt(  # type: ignore[call-arg]
            grade=ReplayGrade.BUNDLE_EXACT,
            contract_id=contract_id.value,
            prediction_id=prediction_id.value,
            evidence_sha256="c" * 64,
            evidence={},
            receipt_id="d" * 64,
        )


def test_tolerant_replay_does_not_overclaim_exact_scoring() -> None:
    receipts = derive_clean_replay_grade_receipts(
        contract_id="a" * 64,
        prediction_id="b" * 64,
        evidence=_clean_replay(equality=ReplayEquality.TOLERANT_TOP5, exact=False),
    )

    assert tuple(receipt.grade for receipt in receipts) == (ReplayGrade.EXPERIMENT_TOLERANT,)


def test_cross_backend_grade_requires_distinct_exact_matching_replays() -> None:
    cpu = _clean_replay(environment_sha256="8" * 64)
    gpu = _clean_replay(environment_sha256="7" * 64)

    receipt = derive_cross_backend_portability_grade(
        contract_id="a" * 64,
        prediction_id="b" * 64,
        cpu_evidence=cpu,
        gpu_evidence=gpu,
    )

    assert receipt.grade is ReplayGrade.CROSS_BACKEND_PORTABILITY
    with pytest.raises(ReplayGradeError, match="distinct environment"):
        derive_cross_backend_portability_grade(
            contract_id="a" * 64,
            prediction_id="b" * 64,
            cpu_evidence=cpu,
            gpu_evidence=replace(gpu, identity=cpu.identity),
        )


def test_grade_report_is_canonical_and_rejects_duplicate_conclusions() -> None:
    receipts = derive_clean_replay_grade_receipts(
        contract_id="a" * 64,
        prediction_id="b" * 64,
        evidence=_clean_replay(),
    )

    report = combine_replay_grade_receipts(receipts)

    assert report.canonical_bytes().endswith(b"\n")
    assert report.manifest()["achieved_grades"] == sorted(
        (
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        )
    )
    with pytest.raises(ReplayGradeError, match="duplicate grades"):
        combine_replay_grade_receipts((receipts[0], receipts[0]))
