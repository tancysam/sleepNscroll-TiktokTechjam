from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from kuairand_agent.candidates.bootstrap import (
    MAX_BOOTSTRAP_RESAMPLES,
    BootstrapDiagnosticError,
    UserClusterBootstrapDiagnostic,
    paired_user_cluster_bootstrap,
)
from kuairand_agent.contract import STARTER_FILE_SHA256
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import ScoreResult
from kuairand_agent.scoring.submission import prediction_digest

type OrganizerResult = dict[str, float | int]
type OrganizerEvaluator = Callable[
    [Sequence[int | str], Sequence[int], Sequence[float], int], OrganizerResult
]

USERS = np.asarray([10, 10, 10, 20, 20, 30, 30, 40, 40, 40, 40], dtype=np.int64)
LABELS = np.asarray([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
CANDIDATE = np.asarray(
    [0.95, 0.05, 0.75, 0.3, 0.1, 0.7, 0.2, 0.15, 0.9, 0.25, 0.8],
    dtype=np.float64,
)
CONTROL = np.asarray(
    [0.4, 0.8, 0.6, 0.7, 0.2, 0.1, 0.9, 0.95, 0.3, 0.85, 0.5],
    dtype=np.float64,
)


def _organizer_evaluator() -> OrganizerEvaluator:
    """Load the untouched evaluator source as the independent tiny-fixture oracle."""

    path = Path(__file__).parents[2] / "kuairand-starter-kit" / "evaluate.py"
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return cast(OrganizerEvaluator, namespace["evaluate"])


def _evaluate(
    evaluator: OrganizerEvaluator,
    users: Sequence[int | str] | np.ndarray[tuple[int], np.dtype[np.int64]],
    labels: Sequence[int] | np.ndarray[tuple[int], np.dtype[np.int8]],
    scores: Sequence[float] | np.ndarray[tuple[int], np.dtype[np.float64]],
) -> OrganizerResult:
    return evaluator(
        [int(value) if isinstance(value, np.integer) else value for value in users],
        [int(value) for value in labels],
        [float(value) for value in scores],
        5,
    )


def _diagnostic(
    *,
    user_ids: Sequence[object] | np.ndarray[tuple[int], np.dtype[np.int64]] = USERS,
    labels: Sequence[object] | np.ndarray[tuple[int], np.dtype[np.int8]] = LABELS,
    candidate_scores: Sequence[object] | np.ndarray[tuple[int], np.dtype[np.float64]] = CANDIDATE,
    control_scores: Sequence[object] | np.ndarray[tuple[int], np.dtype[np.float64]] = CONTROL,
    resamples: int = 299,
    confidence_level: float = 0.9,
    seed: int = 123,
    phase: DataPhase = DataPhase.OUTER_VALID,
    candidate_protected_result: ScoreResult | None = None,
    control_protected_result: ScoreResult | None = None,
) -> UserClusterBootstrapDiagnostic:
    return paired_user_cluster_bootstrap(
        user_ids,
        labels,
        candidate_scores,
        control_scores,
        phase=phase,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
        candidate_protected_result=candidate_protected_result,
        control_protected_result=control_protected_result,
    )


def _encoded_score_result(
    scores: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> ScoreResult:
    raw = _organizer_evaluator()(
        [int(value) for value in USERS],
        cast(Sequence[int], np.asarray(LABELS, dtype=np.float32)),
        [float(value) for value in scores],
        5,
    )
    return ScoreResult(
        gauc=float(raw["GAUC"]),
        ndcg_at_5=float(raw["nDCG@5"]),
        primary=float(raw["primary"]),
        users=int(raw["users"]),
        rows=int(raw["rows"]),
        scorer_digest=STARTER_FILE_SHA256["evaluate.py"],
        prediction_digest=prediction_digest(scores),
        runtime_seconds=0.001,
    )


def _selector_primary(result: ScoreResult) -> float:
    return float((Decimal(str(result.gauc)) + Decimal(str(result.ndcg_at_5))) / 2)


def test_point_estimates_match_immutable_organizer_oracle() -> None:
    evaluator = _organizer_evaluator()
    diagnostic = _diagnostic()
    candidate = _evaluate(evaluator, USERS, LABELS, CANDIDATE)
    control = _evaluate(evaluator, USERS, LABELS, CONTROL)

    assert diagnostic.rows == len(LABELS)
    assert diagnostic.users == candidate["users"] == control["users"] == 4
    assert diagnostic.gauc_eligible_users == 2
    for observed, key in (
        (diagnostic.gauc, "GAUC"),
        (diagnostic.ndcg_at_5, "nDCG@5"),
        (diagnostic.primary, "primary"),
    ):
        assert observed.candidate == pytest.approx(candidate[key], abs=1e-15)
        assert observed.control == pytest.approx(control[key], abs=1e-15)
        assert observed.delta == pytest.approx(candidate[key] - control[key], abs=1e-15)
    assert diagnostic.primary.candidate == pytest.approx(
        (diagnostic.gauc.candidate + diagnostic.ndcg_at_5.candidate) / 2.0,
        abs=1e-15,
    )


def test_protected_score_results_own_points_while_intervals_remain_independent() -> None:
    independent = _diagnostic()
    candidate = _encoded_score_result(CANDIDATE)
    control = _encoded_score_result(CONTROL)
    protected = _diagnostic(
        candidate_protected_result=candidate,
        control_protected_result=control,
    )

    # NumPy 2.x preserves encoded float32 labels through the organizer's nDCG arithmetic, so
    # this fixture deliberately differs from the independent float64 reconstruction.
    assert independent.ndcg_at_5.control != control.ndcg_at_5
    assert protected.gauc.candidate == candidate.gauc
    assert protected.ndcg_at_5.candidate == candidate.ndcg_at_5
    assert protected.primary.candidate == _selector_primary(candidate)
    assert protected.gauc.control == control.gauc
    assert protected.ndcg_at_5.control == control.ndcg_at_5
    assert protected.primary.control == _selector_primary(control)
    assert protected.gauc.interval == independent.gauc.interval
    assert protected.ndcg_at_5.interval == independent.ndcg_at_5.interval
    assert protected.primary.interval == independent.primary.interval
    manifest = protected.as_dict()
    assert manifest["schema_version"] == 2
    assert manifest["point_estimate_source"] == "protected_organizer_scorer"
    assert manifest["point_estimate_provenance"] == {
        "scorer_digest": STARTER_FILE_SHA256["evaluate.py"],
        "candidate_prediction_digest": prediction_digest(CANDIDATE),
        "control_prediction_digest": prediction_digest(CONTROL),
        "rows": len(LABELS),
        "users": 4,
        "primary_aggregation": "selector_decimal_mean_of_protected_gauc_and_ndcg_at_5",
    }


def test_default_point_provenance_is_explicitly_independent() -> None:
    manifest = _diagnostic().as_dict()

    assert manifest["schema_version"] == 2
    assert manifest["point_estimate_source"] == "independent_diagnostic_reconstruction"
    assert manifest["point_estimate_provenance"] is None


def test_protected_raw_float32_primary_is_validated_then_selector_normalized() -> None:
    gauc = 0.6672250628471375
    ndcg = 0.5357140302658081
    raw_primary = float((np.float32(gauc) + np.float32(ndcg)) / np.float32(2.0))
    candidate = replace(
        _encoded_score_result(CANDIDATE),
        gauc=gauc,
        ndcg_at_5=ndcg,
        primary=raw_primary,
    )
    protected = _diagnostic(
        candidate_protected_result=candidate,
        control_protected_result=_encoded_score_result(CONTROL),
    )

    assert raw_primary != _selector_primary(candidate)
    assert protected.primary.candidate == _selector_primary(candidate)


def test_protected_point_results_must_be_supplied_as_a_pair() -> None:
    candidate = _encoded_score_result(CANDIDATE)

    with pytest.raises(BootstrapDiagnosticError, match="supplied together"):
        _diagnostic(candidate_protected_result=candidate)


@pytest.mark.parametrize(
    ("member", "replacement", "message"),
    [
        ("candidate", {"prediction_digest": "f" * 64}, "prediction digest"),
        ("control", {"rows": len(LABELS) - 1}, "rows"),
        ("candidate", {"users": 3}, "users"),
        ("control", {"scorer_digest": "e" * 64}, "scorer identity"),
        ("candidate", {"primary": 0.123}, "organizer primary"),
    ],
)
def test_protected_point_results_are_exactly_bound_to_bootstrap_inputs(
    member: str,
    replacement: dict[str, object],
    message: str,
) -> None:
    candidate = _encoded_score_result(CANDIDATE)
    control = _encoded_score_result(CONTROL)
    if member == "candidate":
        candidate = replace(candidate, **cast(Any, replacement))
    else:
        control = replace(control, **cast(Any, replacement))

    with pytest.raises(BootstrapDiagnosticError, match=message):
        _diagnostic(
            candidate_protected_result=candidate,
            control_protected_result=control,
        )


def _direct_cluster_bootstrap_oracle(
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], ...]:
    """Resample copied clusters and call the organizer for every tiny-fixture replicate."""

    evaluator = _organizer_evaluator()
    grouped: dict[int, list[int]] = {}
    for index, user in enumerate(USERS):
        grouped.setdefault(int(user), []).append(index)

    contribution_groups: list[tuple[tuple[int, float, float, float, float], list[int]]] = []
    for indices in grouped.values():
        labels = LABELS[indices]
        candidate = _evaluate(evaluator, [0] * len(indices), labels, CANDIDATE[indices])
        control = _evaluate(evaluator, [0] * len(indices), labels, CONTROL[indices])
        positives = int(labels.sum())
        denominator = positives if 0 < positives < len(labels) else 0
        key = (
            denominator,
            denominator * float(candidate["GAUC"]),
            denominator * float(control["GAUC"]),
            float(candidate["nDCG@5"]),
            float(control["nDCG@5"]),
        )
        contribution_groups.append((key, indices))
    contribution_groups.sort(key=lambda item: item[0])

    deltas = tuple(np.empty(resamples, dtype=np.float64) for _ in range(3))
    rng = np.random.default_rng(seed)
    for replicate in range(resamples):
        draws = rng.integers(0, len(contribution_groups), size=len(contribution_groups))
        copied_users: list[int] = []
        copied_labels: list[int] = []
        copied_candidate: list[float] = []
        copied_control: list[float] = []
        for copied_user, group_index in enumerate(draws):
            for row in contribution_groups[int(group_index)][1]:
                copied_users.append(copied_user)
                copied_labels.append(int(LABELS[row]))
                copied_candidate.append(float(CANDIDATE[row]))
                copied_control.append(float(CONTROL[row]))
        candidate = _evaluate(evaluator, copied_users, copied_labels, copied_candidate)
        control = _evaluate(evaluator, copied_users, copied_labels, copied_control)
        for target, metric_name in zip(deltas, ("GAUC", "nDCG@5", "primary"), strict=True):
            target[replicate] = float(candidate[metric_name]) - float(control[metric_name])

    tail = (1.0 - confidence_level) / 2.0
    return tuple(np.quantile(values, (tail, 1.0 - tail), method="linear") for values in deltas)


def test_intervals_match_direct_cluster_resampling_oracle() -> None:
    resamples = 127
    confidence_level = 0.8
    seed = 19
    diagnostic = _diagnostic(
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    expected = _direct_cluster_bootstrap_oracle(
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    for metric, bounds in zip(
        (diagnostic.gauc, diagnostic.ndcg_at_5, diagnostic.primary), expected, strict=True
    ):
        assert metric.interval.lower == pytest.approx(bounds[0], abs=1e-15)
        assert metric.interval.upper == pytest.approx(bounds[1], abs=1e-15)
        assert metric.interval.confidence_level == confidence_level
        assert metric.interval.method == "percentile-linear"


def test_deterministic_and_machine_readably_non_gating() -> None:
    first = _diagnostic()
    second = _diagnostic()

    assert first == second
    assert first.gating_eligible is False
    assert first.decision_use == "diagnostic_only"
    assert first.as_dict() == second.as_dict()
    assert first.as_dict()["metrics"] == {
        "GAUC": first.gauc.as_dict(),
        "nDCG@5": first.ndcg_at_5.as_dict(),
        "primary": first.primary.as_dict(),
    }


def test_row_permutation_with_aligned_unique_scores_is_invariant() -> None:
    permutation = np.asarray([8, 0, 6, 3, 10, 1, 5, 9, 2, 7, 4], dtype=np.int64)
    original = _diagnostic()
    permuted = _diagnostic(
        user_ids=USERS[permutation],
        labels=LABELS[permutation],
        candidate_scores=CANDIDATE[permutation],
        control_scores=CONTROL[permutation],
    )

    assert permuted == original


def test_strictly_monotone_score_transforms_are_invariant() -> None:
    original = _diagnostic()
    transformed = _diagnostic(
        candidate_scores=CANDIDATE * 7.0 - 13.0,
        control_scores=CONTROL * 0.25 + 31.0,
    )

    assert transformed == original


def test_bijective_user_relabeling_is_invariant() -> None:
    relabel = {10: "zeta", 20: "alpha", 30: "mu", 40: "beta"}
    relabeled = [relabel[int(user)] for user in USERS]

    assert _diagnostic(user_ids=relabeled) == _diagnostic()


def test_equal_scores_preserve_organizer_row_order_tie_semantics() -> None:
    evaluator = _organizer_evaluator()
    users = [1, 1, 1, 2, 2]
    labels = [0, 1, 0, 1, 0]
    tied = [0.5, 0.5, 0.5, 0.2, 0.9]
    control = [0.1, 0.9, 0.2, 0.8, 0.3]
    diagnostic = _diagnostic(
        user_ids=users,
        labels=labels,
        candidate_scores=tied,
        control_scores=control,
        resamples=31,
    )
    expected = _evaluate(evaluator, users, labels, tied)

    assert diagnostic.gauc.candidate == pytest.approx(expected["GAUC"], abs=1e-15)
    assert diagnostic.ndcg_at_5.candidate == pytest.approx(expected["nDCG@5"], abs=1e-15)


@pytest.mark.parametrize("labels", ([0, 0, 0, 0], [1, 1, 1, 1]))
def test_no_mixed_label_users_use_organizer_gauc_fallback(labels: list[int]) -> None:
    diagnostic = _diagnostic(
        user_ids=[1, 1, 2, 2],
        labels=labels,
        candidate_scores=[0.9, 0.1, 0.8, 0.2],
        control_scores=[0.1, 0.9, 0.2, 0.8],
        resamples=51,
    )

    assert diagnostic.gauc_eligible_users == 0
    assert diagnostic.gauc.candidate == 0.5
    assert diagnostic.gauc.control == 0.5
    assert diagnostic.gauc.delta == 0.0
    assert diagnostic.gauc.interval.lower == diagnostic.gauc.interval.upper == 0.0


def test_identical_predictions_have_zero_paired_intervals() -> None:
    diagnostic = _diagnostic(control_scores=CANDIDATE)

    for metric in (diagnostic.gauc, diagnostic.ndcg_at_5, diagnostic.primary):
        assert metric.delta == 0.0
        assert metric.interval.lower == 0.0
        assert metric.interval.upper == 0.0


def test_integer_and_string_user_ids_remain_distinct() -> None:
    diagnostic = _diagnostic(
        user_ids=[1, 1, "1", "1"],
        labels=[1, 0, 1, 0],
        candidate_scores=[0.9, 0.1, 0.8, 0.2],
        control_scores=[0.1, 0.9, 0.2, 0.8],
        resamples=11,
        phase=DataPhase.INNER_VALID,
    )

    assert diagnostic.users == 2
    assert diagnostic.phase is DataPhase.INNER_VALID


class _ExplodingVector:
    def __array__(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise AssertionError("final outcome vector was inspected")


def test_final_phase_is_rejected_before_any_vector_is_inspected() -> None:
    hidden = cast(Sequence[object], _ExplodingVector())

    with pytest.raises(BootstrapDiagnosticError, match="allowed only"):
        paired_user_cluster_bootstrap(
            hidden,
            hidden,
            hidden,
            hidden,
            phase=DataPhase.FINAL,
            resamples=1,
        )


@pytest.mark.parametrize("phase", [DataPhase.TRAIN, DataPhase.INNER_TRAIN, DataPhase.FINAL])
def test_non_validation_phases_are_rejected(phase: DataPhase) -> None:
    with pytest.raises(BootstrapDiagnosticError, match="allowed only"):
        _diagnostic(phase=phase)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user_ids", [], "cannot be empty"),
        ("user_ids", [[1], [2]], "one-dimensional"),
        ("user_ids", [True] * len(USERS), "integer or non-empty string"),
        ("user_ids", [np.bool_(True)] * len(USERS), "integer or non-empty string"),
        ("user_ids", [1.5] * len(USERS), "integer or non-empty string"),
        ("user_ids", [""] * len(USERS), "integer or non-empty string"),
        ("user_ids", ["bad\x00id"] * len(USERS), "integer or non-empty string"),
        ("labels", [0, 2] + [0] * (len(USERS) - 2), "binary 0 and 1"),
        ("labels", [math.nan] + [0] * (len(USERS) - 1), "binary 0 and 1"),
        ("labels", ["0"] * len(USERS), "numeric binary"),
        ("candidate_scores", [True] * len(USERS), "finite real numbers"),
        ("candidate_scores", [math.inf] + [0.0] * (len(USERS) - 1), "finite real"),
        ("control_scores", [math.nan] + [0.0] * (len(USERS) - 1), "finite real"),
        ("control_scores", ["0"] * len(USERS), "finite real numbers"),
    ],
)
def test_malformed_vectors_fail_closed(field: str, value: list[object], message: str) -> None:
    arguments: dict[str, object] = {
        "user_ids": USERS,
        "labels": LABELS,
        "candidate_scores": CANDIDATE,
        "control_scores": CONTROL,
    }
    arguments[field] = value

    with pytest.raises(BootstrapDiagnosticError, match=message):
        paired_user_cluster_bootstrap(
            cast(Sequence[object], arguments["user_ids"]),
            cast(Sequence[object], arguments["labels"]),
            cast(Sequence[object], arguments["candidate_scores"]),
            cast(Sequence[object], arguments["control_scores"]),
            phase=DataPhase.OUTER_VALID,
            resamples=1,
        )


@pytest.mark.parametrize("field", ["labels", "candidate_scores", "control_scores"])
def test_vector_length_mismatch_fails_closed(field: str) -> None:
    arguments: dict[str, object] = {
        "user_ids": USERS,
        "labels": LABELS,
        "candidate_scores": CANDIDATE,
        "control_scores": CONTROL,
    }
    arguments[field] = [0]

    with pytest.raises(BootstrapDiagnosticError, match="equal lengths"):
        paired_user_cluster_bootstrap(
            cast(Sequence[object], arguments["user_ids"]),
            cast(Sequence[object], arguments["labels"]),
            cast(Sequence[object], arguments["candidate_scores"]),
            cast(Sequence[object], arguments["control_scores"]),
            phase=DataPhase.OUTER_VALID,
            resamples=1,
        )


@pytest.mark.parametrize("resamples", [True, 0, -1, MAX_BOOTSTRAP_RESAMPLES + 1])
def test_invalid_resample_count_is_rejected(resamples: int) -> None:
    with pytest.raises(BootstrapDiagnosticError, match="resamples"):
        _diagnostic(resamples=resamples)


@pytest.mark.parametrize("confidence_level", [True, 0.0, 1.0, -0.1, math.nan, math.inf])
def test_invalid_confidence_level_is_rejected(confidence_level: float) -> None:
    with pytest.raises(BootstrapDiagnosticError, match="confidence_level"):
        _diagnostic(confidence_level=confidence_level)


@pytest.mark.parametrize("seed", [True, -1, 2**32])
def test_invalid_seed_is_rejected(seed: int) -> None:
    with pytest.raises(BootstrapDiagnosticError, match="seed"):
        _diagnostic(seed=seed)


def test_non_enum_phase_is_rejected() -> None:
    with pytest.raises(BootstrapDiagnosticError, match="DataPhase"):
        paired_user_cluster_bootstrap(
            USERS,
            LABELS,
            CANDIDATE,
            CONTROL,
            phase=cast(DataPhase, "outer_valid"),
            resamples=1,
        )
