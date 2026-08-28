from __future__ import annotations

from pathlib import Path

import pytest

from kuairand_agent.baselines.rungs import (
    RANDOM_SEEDS,
    BaselineReferenceMismatch,
    BaselineRungError,
    RungEvaluation,
    RungMetrics,
    RungName,
    RungSummary,
    build_validation_scoring_context,
    evaluate_popularity_validation,
    evaluate_random_rungs,
    evaluate_random_validation,
    organizer_validation_fixture,
    require_reference_parity,
    select_best_seed,
)
from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import (
    APPROVED_AUXILIARY_TARGETS,
    BINARY_AUXILIARY_TARGETS,
    OUTCOME_FIELDS,
    PRIMARY_TARGET,
    CanonicalAlignment,
    CanonicalDataset,
    CanonicalInputs,
    CanonicalSplit,
    OutcomeAccessTrace,
    ProtectedTargets,
    TrainingTargets,
)

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"


def _inputs(
    users: tuple[str, ...],
    videos: tuple[str, ...],
    *,
    start_date: int,
    authors: tuple[str, ...] | None = None,
) -> CanonicalInputs:
    count = len(users)
    return CanonicalInputs(
        user_id=users,
        video_id=videos,
        date=tuple(start_date for _ in range(count)),
        duration_ms=tuple(10_000.0 + index for index in range(count)),
        tab=tuple("1" for _ in range(count)),
        author_id=authors if authors is not None else tuple("10" for _ in range(count)),
        time_ms=tuple(range(1, count + 1)),
    )


def _training_targets(labels: tuple[int, ...]) -> TrainingTargets:
    columns: dict[str, tuple[int | float, ...]] = {PRIMARY_TARGET: labels}
    for name in APPROVED_AUXILIARY_TARGETS:
        columns[name] = (
            labels if name in BINARY_AUXILIARY_TARGETS else tuple(float(value) for value in labels)
        )
    return TrainingTargets(columns)


def fixture_dataset() -> CanonicalDataset:
    train_inputs = _inputs(
        ("u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"),
        ("a", "a", "a", "b", "b", "c", "c", "c"),
        start_date=20220421,
    )
    train_targets = _training_targets((1, 0, 1, 0, 1, 0, 1, 0))
    train = CanonicalSplit(
        name=SplitName.TRAIN,
        inputs=train_inputs,
        alignment=CanonicalAlignment(
            split=SplitName.TRAIN,
            row_id=tuple(range(len(train_inputs))),
            user_id=train_inputs.user_id,
            video_id=train_inputs.video_id,
        ),
        targets=train_targets,
        outcome_trace=OutcomeAccessTrace(
            split=SplitName.TRAIN,
            row_count=len(train_inputs),
            parsed_fields=OUTCOME_FIELDS,
            skipped_fields=(),
        ),
    )

    valid_inputs = _inputs(
        ("u1", "u1", "u1", "u2", "u2", "u2", "u5", "u5"),
        ("a", "b", "new", "a", "b", "c", "new", "c"),
        start_date=20220422,
    )
    valid_targets = ProtectedTargets((1, 0, 1, 0, 1, 0, 1, 0))
    valid = CanonicalSplit(
        name=SplitName.VALID,
        inputs=valid_inputs,
        alignment=CanonicalAlignment(
            split=SplitName.VALID,
            row_id=tuple(range(len(valid_inputs))),
            user_id=valid_inputs.user_id,
            video_id=valid_inputs.video_id,
        ),
        targets=valid_targets,
        outcome_trace=OutcomeAccessTrace(
            split=SplitName.VALID,
            row_count=len(valid_inputs),
            parsed_fields=(PRIMARY_TARGET,),
            skipped_fields=APPROVED_AUXILIARY_TARGETS,
        ),
    )

    final_inputs = _inputs(
        ("must-not-be-used",),
        ("final-only-video",),
        start_date=20220429,
        authors=("final-only-author",),
    )
    final = CanonicalSplit(
        name=SplitName.TEST,
        inputs=final_inputs,
        alignment=CanonicalAlignment(
            split=SplitName.TEST,
            row_id=(0,),
            user_id=final_inputs.user_id,
            video_id=final_inputs.video_id,
        ),
        targets=None,
        outcome_trace=OutcomeAccessTrace(
            split=SplitName.TEST,
            row_count=1,
            parsed_fields=(),
            skipped_fields=OUTCOME_FIELDS,
        ),
    )
    return CanonicalDataset(
        train=train,
        valid=valid,
        final=final,
        author_map_digest="a" * 64,
    )


def _evaluation(
    seed: int,
    metrics: RungMetrics,
) -> RungEvaluation:
    return RungEvaluation(
        name=RungName.RANDOM,
        seed=seed,
        metrics=metrics,
        users=2,
        rows=4,
        scorer_digest="1" * 64,
        prediction_digest=f"{seed + 2:x}" * 64,
        split_digest="f" * 64,
        runtime_seconds=0.1,
    )


def test_validation_context_uses_only_canonical_valid_alignment_and_protected_target() -> None:
    dataset = fixture_dataset()
    context = build_validation_scoring_context(dataset, STARTER)

    assert context.split.name == "outer_valid"
    assert context.split.token == dataset.valid.digest
    assert context.alignment.row_ids == dataset.valid.alignment.row_id
    assert context.alignment.user_ids == dataset.valid.alignment.user_id
    assert context.alignment.video_ids == dataset.valid.alignment.video_id
    assert "must-not-be-used" not in context.alignment.user_ids

    result = context.score(tuple(float(index) for index in range(dataset.valid.row_count)))
    encoded = context.score_with_encoded_labels(
        tuple(float(index) for index in range(dataset.valid.row_count))
    )
    assert result.rows == dataset.valid.row_count
    assert encoded.rows == dataset.valid.row_count


def test_random_seeds_are_fixed_ordered_deterministic_and_validation_only() -> None:
    dataset = fixture_dataset()
    first_context = build_validation_scoring_context(dataset, STARTER)
    second_context = build_validation_scoring_context(dataset, STARTER)

    first = evaluate_random_rungs(first_context)
    replay = evaluate_random_rungs(second_context)
    assert tuple(run.seed for run in first.evaluations) == RANDOM_SEEDS
    assert [run.prediction_digest for run in first.evaluations] == [
        run.prediction_digest for run in replay.evaluations
    ]
    assert first.digest == replay.digest
    assert first.logical_manifest() == replay.logical_manifest()
    assert len(set(run.prediction_digest for run in first.evaluations)) == 5

    with pytest.raises(BaselineRungError, match="one of 0, 1, 2, 3, 4"):
        evaluate_random_validation(first_context, 5)


def test_popularity_is_deterministic_prior_twenty_and_unseen_items_use_global_rate() -> None:
    dataset = fixture_dataset()
    context = build_validation_scoring_context(dataset, STARTER)
    first = evaluate_popularity_validation(dataset, context)
    second = evaluate_popularity_validation(dataset, context)

    assert first.digest == second.digest
    assert first.evaluations[0].prediction_digest == second.evaluations[0].prediction_digest
    assert first.evaluations[0].seed is None
    # The fixture includes an unseen item and repeated train items; organizer parity is asserted in
    # the integration suite, including the exact prior-20 smoothing arithmetic.
    assert first.evaluations[0].rows == dataset.valid.row_count


def test_four_decimal_reference_gate_is_deterministic_and_fail_closed() -> None:
    observed = RungMetrics(
        gauc=0.63874,
        ndcg_at_5=0.52274,
        primary=(0.63874 + 0.52274) / 2.0,
    )
    passing_run = RungEvaluation(
        name=RungName.ITEM_POPULARITY,
        seed=None,
        metrics=observed,
        users=2,
        rows=4,
        scorer_digest="1" * 64,
        prediction_digest="2" * 64,
        split_digest="f" * 64,
        runtime_seconds=0.1,
    )
    expected = RungMetrics(gauc=0.6387, ndcg_at_5=0.5227, primary=0.5807)
    passing = RungSummary(
        name=RungName.ITEM_POPULARITY,
        evaluations=(passing_run,),
        reference_metrics=expected,
    )
    require_reference_parity(passing)
    assert passing.reference_passed

    failing = RungSummary(
        name=RungName.ITEM_POPULARITY,
        evaluations=(passing_run,),
        reference_metrics=RungMetrics(gauc=0.6388, ndcg_at_5=0.5227, primary=0.58075),
    )
    with pytest.raises(BaselineReferenceMismatch, match="four-decimal mismatch"):
        require_reference_parity(failing)


def test_best_seed_tie_policy_prefers_components_then_lowest_seed() -> None:
    tied = RungMetrics(gauc=0.6, ndcg_at_5=0.4, primary=0.5)
    assert select_best_seed((_evaluation(3, tied), _evaluation(1, tied))).seed == 1

    higher_gauc = RungMetrics(gauc=0.7, ndcg_at_5=0.3, primary=0.5)
    assert select_best_seed((_evaluation(0, tied), _evaluation(4, higher_gauc))).seed == 4

    higher_primary = RungMetrics(gauc=0.61, ndcg_at_5=0.41, primary=0.51)
    assert select_best_seed((_evaluation(0, tied), _evaluation(2, higher_primary))).seed == 2

    with pytest.raises(BaselineRungError, match="duplicate seeds"):
        select_best_seed((_evaluation(1, tied), _evaluation(1, tied)))


def test_organizer_parity_adapter_uses_train_placeholder_not_final_split() -> None:
    dataset = fixture_dataset()
    fixture = organizer_validation_fixture(dataset, placeholder_count=2)

    assert tuple(fixture) == ("train", "valid", "test")
    assert len(fixture["train"]) == dataset.train.row_count
    assert len(fixture["valid"]) == dataset.valid.row_count
    assert len(fixture["test"]) == 2
    assert fixture["test"][0] == fixture["train"][0]
    assert all(row[1] != "must-not-be-used" for row in fixture["test"])
