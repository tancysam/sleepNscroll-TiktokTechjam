"""The agent may ask questions of the training split, and only of the training split.

Every finding that moved this project came from someone looking at data, and none of them were
reachable by the agent. This capability closes that gap without granting authority: a typed query
over train-only arrays, answered in scalars.

These tests pin the three properties that make it safe to expose -- train-only inputs, aggregate-
only outputs, and named columns rather than expressions -- and the behaviour that makes it useful:
a bad question is evidence for the next iteration, not a dead campaign.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any, cast

from kuairand_agent.campaign.analysis import (
    MAX_ANALYSIS_REQUESTS,
    AnalysisError,
    AnalysisKind,
    AnalysisQuery,
    TrainAnalysisInputs,
    run_analysis,
    run_requested_analyses,
)


def _inputs(rows: int = 400) -> TrainAnalysisInputs:
    rng = np.random.default_rng(7)
    users = [f"u{index % 40}" for index in range(rows)]
    item = rng.normal(size=rows)
    # A column that is constant inside each user, so it cannot reorder that user's own slate.
    user_constant = np.asarray([float(index % 40) for index in range(rows)])
    labels = (item + 0.3 * rng.normal(size=rows) > 0).astype(np.float64)
    values = np.column_stack((item, user_constant, item * user_constant))
    return TrainAnalysisInputs(
        feature_names=("item_signal", "user_constant", "cross"),
        feature_values=values,
        labels=labels,
        user_ids=tuple(users),
    )


def test_an_interaction_query_reports_the_cross_against_both_single_columns() -> None:
    """The open question all session has been which crosses carry signal the columns do not."""

    record = run_analysis(
        _inputs(),
        AnalysisQuery(
            kind=AnalysisKind.WITHIN_USER_INTERACTION,
            feature="item_signal",
            second_feature="user_constant",
        ),
    )

    assert record.name == "requested_within_user_interaction"
    values = dict(record.values)
    # The item column carries real within-user signal; the user-constant column cannot.
    assert abs(float(cast(Any, values["within_user_corr_a"]))) > 0.2
    assert abs(float(cast(Any, values["within_user_corr_b"]))) < 1e-9
    assert "exceeds both single-column" in str(values["note"])


def test_a_user_constant_column_is_reported_as_carrying_no_within_user_signal() -> None:
    """The organizers measured that such a term contributes exactly zero; the query must agree."""

    record = run_analysis(
        _inputs(),
        AnalysisQuery(
            kind=AnalysisKind.WITHIN_USER_INTERACTION,
            feature="user_constant",
            second_feature="item_signal",
        ),
    )

    assert abs(float(cast(Any, dict(record.values)["within_user_corr_a"]))) < 1e-9


def test_bucketed_queries_report_only_buckets_above_the_disclosure_floor() -> None:
    record = run_analysis(
        _inputs(rows=120),
        AnalysisQuery(kind=AnalysisKind.LABEL_RATE_BY_BUCKET, feature="item_signal", buckets=10),
    )

    values = dict(record.values)
    reported = [key for key in values if key.endswith("_rows")]
    for key in reported:
        assert int(cast(Any, values[key])) >= 50, (
            "no reported bucket may summarise fewer than 50 rows"
        )


def test_signal_by_slate_size_answers_where_a_column_earns_its_signal() -> None:
    record = run_analysis(
        _inputs(),
        AnalysisQuery(kind=AnalysisKind.SIGNAL_BY_SLATE_SIZE, feature="item_signal", buckets=2),
    )

    assert record.name == "requested_signal_by_slate_size"
    assert "GAUC weights by positive count" in str(dict(record.values)["note"])


def test_an_unknown_column_cannot_be_queried() -> None:
    """Feature names are checked against the bundle, so a query cannot smuggle an expression."""

    with pytest.raises(AnalysisError, match="unknown feature"):
        run_analysis(
            _inputs(),
            AnalysisQuery(kind=AnalysisKind.LABEL_RATE_BY_BUCKET, feature="item_signal * 2"),
        )


def test_malformed_queries_are_rejected_at_construction() -> None:
    with pytest.raises(AnalysisError, match="requires two features"):
        AnalysisQuery(kind=AnalysisKind.WITHIN_USER_INTERACTION, feature="item_signal")
    with pytest.raises(AnalysisError, match="buckets"):
        AnalysisQuery(kind=AnalysisKind.LABEL_RATE_BY_BUCKET, feature="item_signal", buckets=99)


def test_misaligned_training_arrays_are_refused() -> None:
    with pytest.raises(AnalysisError, match="must align"):
        TrainAnalysisInputs(
            feature_names=("a",),
            feature_values=np.zeros((10, 1)),
            labels=np.zeros(9),
            user_ids=tuple("u" for _ in range(10)),
        )


def test_a_bad_question_becomes_evidence_rather_than_ending_the_campaign() -> None:
    """A malformed request must tell the model what went wrong, not abort the run."""

    records = run_requested_analyses(
        _inputs(),
        (
            AnalysisQuery(kind=AnalysisKind.LABEL_RATE_BY_BUCKET, feature="does_not_exist"),
            AnalysisQuery(kind=AnalysisKind.LABEL_RATE_BY_BUCKET, feature="item_signal"),
        ),
    )

    assert len(records) == 2
    assert records[0].name.endswith("_failed")
    assert "unknown feature" in str(dict(records[0].values)["reason"])
    assert "label_rate_by_bucket" in records[1].name


def test_the_number_of_questions_per_iteration_is_bounded() -> None:
    query = AnalysisQuery(kind=AnalysisKind.LABEL_RATE_BY_BUCKET, feature="item_signal")
    records = run_requested_analyses(_inputs(), (query,) * (MAX_ANALYSIS_REQUESTS + 3))

    assert len(records) == MAX_ANALYSIS_REQUESTS
