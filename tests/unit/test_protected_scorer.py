from __future__ import annotations

import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kuairand_agent.contract import STARTER_FILE_SHA256, OrganizerIntegrityError
from kuairand_agent.scoring.protected import (
    Alignment,
    ProtectedScorer,
    ScoringInputError,
    SplitIdentity,
)

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"

USERS = (10, 20, 30, 10, 40, 20, 30, 10, 10)
VIDEOS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
LABELS = (1, 0, 1, 0, 1, 0, 1, 1, 0)
SCORES = (0.9, 0.7, 0.4, 0.8, 0.5, 0.6, 0.3, 0.1, 0.2)
SPLIT = SplitIdentity(name="outer_valid", token="fixture-public-valid-v1", expected_count=9)
ALIGNMENT = Alignment.from_ids(split=SPLIT, user_ids=USERS, video_ids=VIDEOS)


@pytest.fixture
def scorer() -> ProtectedScorer:
    return ProtectedScorer(starter_dir=STARTER, trusted_alignment=ALIGNMENT)


def test_hardcoded_golden_mixed_and_degenerate_users(
    scorer: ProtectedScorer,
) -> None:
    # Four users: one mixed, one zero-positive, one all-positive, and one single positive.
    # The constants were independently frozen from the pinned audit fixture, not recomputed by a
    # second metric implementation in the test.
    result = scorer.score(
        alignment=ALIGNMENT,
        split=SPLIT,
        labels=LABELS,
        scores=SCORES,
        expected_count=9,
    )

    assert result.gauc == pytest.approx(0.5, abs=1e-15)
    assert result.ndcg_at_5 == pytest.approx(0.7193038288345124, abs=1e-15)
    assert result.primary == pytest.approx(0.6096519144172562, abs=1e-15)
    assert result.users == 4
    assert result.rows == 9
    assert result.scorer_digest == STARTER_FILE_SHA256["evaluate.py"]
    assert len(result.prediction_digest) == 64
    assert result.runtime_seconds >= 0.0
    assert result.as_dict()["nDCG@5"] == result.ndcg5


@pytest.mark.parametrize(
    ("labels", "expected_gauc", "expected_ndcg", "expected_primary"),
    [
        ((0, 0), 0.5, 0.0, 0.25),
        ((1, 1), 0.5, 1.0, 0.75),
        ((1,), 0.5, 1.0, 0.75),
        ((0,), 0.5, 0.0, 0.25),
    ],
)
def test_isolated_zero_all_positive_and_single_row_degenerates(
    labels: tuple[int, ...],
    expected_gauc: float,
    expected_ndcg: float,
    expected_primary: float,
) -> None:
    split = SplitIdentity(
        name="inner_valid", token=f"degenerate-{labels}", expected_count=len(labels)
    )
    alignment = Alignment.from_ids(
        split=split,
        user_ids=(1,) * len(labels),
        video_ids=tuple(range(len(labels))),
    )
    local = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)
    result = local.score(
        alignment=alignment,
        split=split,
        labels=labels,
        scores=tuple(float(index) for index in range(len(labels))),
    )

    assert result.gauc == expected_gauc
    assert result.ndcg_at_5 == expected_ndcg
    assert result.primary == expected_primary


def test_wrapper_equals_untouched_organizer_on_interleaved_fixture(
    scorer: ProtectedScorer,
) -> None:
    organizer = runpy.run_path(str((STARTER / "evaluate.py").resolve()))["evaluate"]
    assert callable(organizer)
    direct = organizer(USERS, LABELS, SCORES, 5)
    wrapped = scorer.score(alignment=ALIGNMENT, split=SPLIT, labels=LABELS, scores=SCORES)

    assert wrapped.gauc == direct["GAUC"]
    assert wrapped.ndcg_at_5 == direct["nDCG@5"]
    assert wrapped.primary == direct["primary"]
    assert wrapped.users == direct["users"]
    assert wrapped.rows == direct["rows"]


def test_encoded_label_route_exactly_matches_organizer_float32_semantics(
    scorer: ProtectedScorer,
) -> None:
    organizer = runpy.run_path(str((STARTER / "evaluate.py").resolve()))["evaluate"]
    assert callable(organizer)
    encoded_labels = np.asarray(LABELS, dtype=np.float32)
    encoded_scores = np.asarray(SCORES, dtype=np.float32)
    direct = organizer(USERS, encoded_labels, encoded_scores, 5)
    wrapped = scorer.score_with_encoded_labels(
        alignment=ALIGNMENT,
        split=SPLIT,
        labels=LABELS,
        scores=encoded_scores,
    )

    assert wrapped.gauc == direct["GAUC"]
    assert wrapped.ndcg_at_5 == direct["nDCG@5"]
    assert wrapped.primary == direct["primary"]
    assert wrapped.users == direct["users"]
    assert wrapped.rows == direct["rows"]


def test_ties_including_rank_five_match_hardcoded_golden() -> None:
    split = SplitIdentity(name="inner_valid", token="fold-with-rank-five-tie", expected_count=7)
    alignment = Alignment.from_ids(
        split=split,
        user_ids=(7, 7, 7, 7, 7, 7, 7),
        video_ids=(70, 71, 72, 73, 74, 75, 76),
    )
    local = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)
    result = local.score(
        alignment=alignment,
        split=split,
        labels=(1, 0, 0, 0, 0, 1, 0),
        scores=(7, 6, 5, 4, 3, 3, 2),
    )

    assert result.gauc == pytest.approx(0.65, abs=1e-15)
    assert result.ndcg_at_5 == pytest.approx(0.6131471927654584, abs=1e-15)
    assert result.primary == pytest.approx(0.6315735963827291, abs=1e-15)


def test_positive_affine_transform_preserves_metrics_but_not_prediction_identity(
    scorer: ProtectedScorer,
) -> None:
    original = scorer.score(alignment=ALIGNMENT, split=SPLIT, labels=LABELS, scores=SCORES)
    transformed_scores = tuple(11.0 * score - 3.25 for score in SCORES)
    transformed = scorer.score(
        alignment=ALIGNMENT,
        split=SPLIT,
        labels=LABELS,
        scores=transformed_scores,
    )

    assert transformed.gauc == original.gauc
    assert transformed.ndcg_at_5 == original.ndcg_at_5
    assert transformed.primary == original.primary
    assert transformed.prediction_digest != original.prediction_digest


def test_duplicate_user_video_pairs_are_distinct_aligned_rows() -> None:
    split = SplitIdentity(name="inner_valid", token="duplicate-pairs", expected_count=4)
    alignment = Alignment.from_ids(
        split=split,
        user_ids=(1, 1, 1, 1),
        video_ids=(8, 8, 9, 9),
    )
    local = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)

    result = local.score(
        alignment=alignment,
        split=split,
        labels=(1, 0, 1, 0),
        scores=(0.9, 0.8, 0.7, 0.6),
    )
    assert result.rows == 4
    assert result.users == 1


def test_alignment_snapshots_mutable_inputs_and_has_type_stable_digest() -> None:
    users = [1, 2]
    videos = ["1", "2"]
    split = SplitIdentity(name="inner_valid", token="snapshot", expected_count=2)
    alignment = Alignment.from_ids(split=split, user_ids=users, video_ids=videos)
    original_digest = alignment.digest
    users[0] = 99
    videos[0] = "changed"

    assert alignment.user_ids == (1, 2)
    assert alignment.video_ids == ("1", "2")
    assert alignment.digest == original_digest
    integer_videos = Alignment.from_ids(split=split, user_ids=(1, 2), video_ids=(1, 2))
    assert integer_videos.digest != alignment.digest


@pytest.mark.parametrize(
    "row_ids",
    [
        (0, 0),
        (1, 0),
        (0, 2),
        (0, True),
    ],
)
def test_alignment_rejects_missing_duplicate_or_noncanonical_positions(
    row_ids: tuple[object, ...],
) -> None:
    split = SplitIdentity(name="inner_valid", token="bad-rows", expected_count=2)
    with pytest.raises(ScoringInputError, match="row_ids"):
        Alignment(
            split=split,
            row_ids=row_ids,
            user_ids=(1, 2),
            video_ids=(3, 4),
        )


def _spy_on_organizer(
    scorer: ProtectedScorer, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def forbidden(*args: object) -> Mapping[str, object]:
        calls.append(args)
        raise AssertionError("malformed input reached organizer evaluate")

    monkeypatch.setattr(scorer, "_evaluate", forbidden)
    return calls


@pytest.mark.parametrize(
    ("labels", "scores", "expected_count", "message"),
    [
        ((), (), None, "non-empty"),
        (LABELS, SCORES[:-1], None, "equal lengths"),
        (LABELS, (*SCORES, 0.0), None, "equal lengths"),
        (LABELS, SCORES, 8, "expected_count mismatch"),
        (np.asarray([LABELS]), SCORES, None, "one-dimensional"),
        (LABELS, np.asarray([SCORES]), None, "one-dimensional"),
        ((0, 1, 2, 0, 1, 0, 1, 0, 1), SCORES, None, "binary"),
        ((0, 1, "1", 0, 1, 0, 1, 0, 1), SCORES, None, "numeric binary"),
        (LABELS, ("0.9",) * 9, None, "real numeric"),
        (LABELS, (True,) * 9, None, "real numeric"),
        (LABELS, (*SCORES[:-1], float("nan")), None, "finite"),
        (LABELS, (*SCORES[:-1], float("inf")), None, "finite"),
        (LABELS, (*SCORES[:-1], -float("inf")), None, "finite"),
    ],
)
def test_invalid_vectors_never_reach_organizer(
    scorer: ProtectedScorer,
    monkeypatch: pytest.MonkeyPatch,
    labels: Any,
    scores: Any,
    expected_count: int | None,
    message: str,
) -> None:
    calls = _spy_on_organizer(scorer, monkeypatch)
    with pytest.raises(ScoringInputError, match=message):
        scorer.score(
            alignment=ALIGNMENT,
            split=SPLIT,
            labels=labels,
            scores=scores,
            expected_count=expected_count,
        )
    assert calls == []


def test_alignment_and_split_mismatches_never_reach_organizer(
    scorer: ProtectedScorer, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_on_organizer(scorer, monkeypatch)
    other_split = SplitIdentity(
        name=SPLIT.name,
        token="a-different-split-token",
        expected_count=SPLIT.expected_count,
    )
    other_alignment = Alignment.from_ids(
        split=SPLIT,
        user_ids=USERS,
        video_ids=(*VIDEOS[:-2], VIDEOS[-1], VIDEOS[-2]),
    )

    with pytest.raises(ScoringInputError, match="split identity"):
        scorer.score(
            alignment=ALIGNMENT,
            split=other_split,
            labels=LABELS,
            scores=SCORES,
        )
    with pytest.raises(ScoringInputError, match="alignment identity"):
        scorer.score(
            alignment=other_alignment,
            split=SPLIT,
            labels=LABELS,
            scores=SCORES,
        )
    assert calls == []


def test_unique_absolute_import_ignores_shadow_module_and_writes_no_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "evaluate.py").write_text(
        "def evaluate(*args, **kwargs): return {'primary': 999}\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(shadow))
    before = {path.name for path in STARTER.iterdir()}
    local = ProtectedScorer(starter_dir=STARTER, trusted_alignment=ALIGNMENT)
    result = local.score(alignment=ALIGNMENT, split=SPLIT, labels=LABELS, scores=SCORES)
    after = {path.name for path in STARTER.iterdir()}

    assert result.primary == pytest.approx(0.6096519144172562)
    assert local.organizer_module_name.startswith("_kuairand_pinned_evaluator_")
    assert local.organizer_module_name not in sys.modules
    assert local.organizer_module_name != "evaluate"
    assert after == before
    assert "__pycache__" not in after


def test_full_member_set_is_reverified_before_each_organizer_call(
    tmp_path: Path,
) -> None:
    starter_copy = tmp_path / "starter"
    starter_copy.mkdir()
    for source in STARTER.iterdir():
        (starter_copy / source.name).write_bytes(source.read_bytes())
    local = ProtectedScorer(starter_dir=starter_copy, trusted_alignment=ALIGNMENT)
    (starter_copy / "unexpected.py").write_text("# not organizer-owned\n", encoding="utf-8")

    with pytest.raises(OrganizerIntegrityError, match="unexpected"):
        local.score(alignment=ALIGNMENT, split=SPLIT, labels=LABELS, scores=SCORES)


def test_evaluator_digest_is_reverified_before_each_organizer_call(
    tmp_path: Path,
) -> None:
    starter_copy = tmp_path / "starter"
    starter_copy.mkdir()
    for source in STARTER.iterdir():
        (starter_copy / source.name).write_bytes(source.read_bytes())
    local = ProtectedScorer(starter_dir=starter_copy, trusted_alignment=ALIGNMENT)
    (starter_copy / "evaluate.py").write_text("# changed after load\n", encoding="utf-8")

    with pytest.raises(OrganizerIntegrityError, match="digest mismatch"):
        local.score(alignment=ALIGNMENT, split=SPLIT, labels=LABELS, scores=SCORES)
