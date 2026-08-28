from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import overload

import pytest

from kuairand_agent.baselines.fold_control_runner import (
    FoldFMControlExecutionRequest,
    SupervisedFoldFMError,
)
from kuairand_agent.data.canonical import CanonicalInputs


def _inputs(prefix: str, dates: tuple[int, ...]) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u-{index // 2}" for index in range(len(dates))),
        video_id=tuple(f"{prefix}-v-{index}" for index in range(len(dates))),
        date=dates,
        duration_ms=tuple(float(500 + index) for index in range(len(dates))),
        tab=tuple(str(index % 3) for index in range(len(dates))),
        author_id=tuple(f"a-{index % 4}" for index in range(len(dates))),
        time_ms=tuple(range(len(dates))),
    )


class _ExplodingLabels(Sequence[object]):
    def __len__(self) -> int:
        raise AssertionError("protected labels were inspected")

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        raise AssertionError(f"protected labels[{index}] were inspected")


def test_request_is_a_value_only_identity_and_never_reads_labels() -> None:
    prefix = _inputs("prefix", (20220408, 20220409))
    query = _inputs("query", (20220416, 20220417))

    request = FoldFMControlExecutionRequest(
        execution_id="fm-fold-a-seed-0",
        fold_name="A",
        fold_token="a" * 64,
        seed=0,
        prefix_inputs=prefix,
        prefix_labels=_ExplodingLabels(),
        query_inputs=query,
        query_labels=_ExplodingLabels(),
    )

    assert request.execution_id == "fm-fold-a-seed-0"
    assert request.fold_name == "A"


@pytest.mark.parametrize(
    ("execution_id", "fold_name", "fold_token", "seed", "message"),
    [
        ("has.dot", "A", "a" * 64, 0, "execution_id"),
        ("valid-id", "outer_valid", "a" * 64, 0, "fold_name"),
        ("valid-id", "B", "short", 0, "fold_token"),
        ("valid-id", "B", "a" * 64, -1, "seed"),
    ],
)
def test_request_rejects_noncanonical_execution_identity(
    execution_id: str,
    fold_name: str,
    fold_token: str,
    seed: int,
    message: str,
) -> None:
    prefix = _inputs("prefix", (20220408, 20220409))
    query = _inputs("query", (20220416, 20220417))

    with pytest.raises(SupervisedFoldFMError, match=message):
        FoldFMControlExecutionRequest(
            execution_id=execution_id,
            fold_name=fold_name,  # type: ignore[arg-type]
            fold_token=fold_token,
            seed=seed,
            prefix_inputs=prefix,
            prefix_labels=(0, 1),
            query_inputs=query,
            query_labels=(0, 1),
        )


def test_request_requires_canonical_inputs_without_inspecting_labels() -> None:
    inputs = _inputs("input", (20220408, 20220409))

    with pytest.raises(SupervisedFoldFMError, match="query_inputs"):
        FoldFMControlExecutionRequest(
            execution_id="valid-id",
            fold_name="A",
            fold_token="a" * 64,
            seed=0,
            prefix_inputs=inputs,
            prefix_labels=_ExplodingLabels(),
            query_inputs=Path("not-canonical"),  # type: ignore[arg-type]
            query_labels=_ExplodingLabels(),
        )
