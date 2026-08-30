from __future__ import annotations

import numpy as np
import pytest

from kuairand_agent.campaign.eda import (
    EdaError,
    within_user_feature_diagnostics,
    within_user_label_structure,
)


def _record(records: tuple, name: str) -> dict:
    for item in records:
        if item.name == name:
            return dict(item.values)
    raise AssertionError(f"missing record {name}")


def test_a_user_constant_feature_is_reported_as_inert() -> None:
    """The organizers measured that user-constant terms contribute exactly zero to ranking.

    A global correlation would rank such a feature highly and send the agent after something that
    provably cannot reorder anything, which is why these diagnostics are within-user.
    """

    rows = 200
    users = np.repeat(np.arange(rows // 4), 4)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=rows)
    # Perfectly predicts the user's mean label, but is identical across that user's own rows.
    user_constant = np.repeat(
        np.array([labels[i * 4 : i * 4 + 4].mean() for i in range(rows // 4)]), 4
    )
    values = np.column_stack([user_constant])

    records = within_user_feature_diagnostics(
        feature_names=["user_constant"],
        feature_values=values,
        labels=list(labels),
        user_ids=list(users),
    )

    inert = _record(records, "train_features_inert_within_user")
    assert inert["count"] == 1
    assert "user_constant" in inert["features_csv"]
    top = _record(records, "train_feature_signal_01")
    assert top["feature"] == "user_constant"
    assert top["within_user_variance_share"] == 0.0
    # No within-user variance means no ranking power, whatever its global association.
    assert top["within_user_corr"] == 0.0


def test_a_within_user_signal_is_surfaced_with_its_sign() -> None:
    rows = 400
    users = np.repeat(np.arange(rows // 4), 4)
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=rows)
    informative = labels + rng.normal(0.0, 0.05, size=rows)
    noise = rng.normal(0.0, 1.0, size=rows)
    values = np.column_stack([noise, informative])

    records = within_user_feature_diagnostics(
        feature_names=["noise", "informative"],
        feature_values=values,
        labels=list(labels),
        user_ids=list(users),
    )

    top = _record(records, "train_feature_signal_01")
    assert top["feature"] == "informative"
    assert top["within_user_corr"] > 0.9
    assert top["within_user_variance_share"] > 0.5
    assert "train_features_inert_within_user" not in {item.name for item in records}


def test_label_structure_identifies_the_users_that_can_move_the_metrics() -> None:
    # 16 users x 4 rows = 64, above the aggregation floor: 8 mixed, 5 all-zero, 3 all-positive.
    labels: list[int] = []
    users: list[int] = []
    for user in range(16):
        users.extend([user] * 4)
        if user < 8:
            labels.extend([1, 0, 0, 0])
        elif user < 13:
            labels.extend([0, 0, 0, 0])
        else:
            labels.extend([1, 1, 1, 1])

    records = within_user_label_structure(labels=labels, user_ids=users)

    row = _record(records, "train_within_user_label_structure")
    assert row["users"] == 16
    assert row["mixed_label_users"] == 8
    assert row["zero_positive_users"] == 5
    assert row["all_positive_users"] == 3
    assert row["slate_size_mean"] == 4.0


def test_diagnostics_are_withheld_for_a_group_too_small_to_aggregate() -> None:
    assert within_user_label_structure(labels=[1, 0], user_ids=[1, 1]) == ()
    assert (
        within_user_feature_diagnostics(
            feature_names=["a"],
            feature_values=np.zeros((2, 1)),
            labels=[1, 0],
            user_ids=[1, 1],
        )
        == ()
    )


@pytest.mark.parametrize(
    ("names", "shape", "labels", "users"),
    [
        (["a", "b"], (100, 1), [0] * 100, list(range(100))),
        (["a"], (100, 1), [0] * 99, list(range(100))),
    ],
)
def test_inconsistent_inputs_fail_closed(
    names: list[str], shape: tuple[int, int], labels: list[int], users: list[int]
) -> None:
    with pytest.raises(EdaError):
        within_user_feature_diagnostics(
            feature_names=names,
            feature_values=np.zeros(shape),
            labels=labels,
            user_ids=users,
        )
