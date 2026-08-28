from __future__ import annotations

import csv
import os
import runpy
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from kuairand_agent.baselines.rungs import (
    RANDOM_SEEDS,
    build_validation_scoring_context,
    evaluate_popularity_validation,
    evaluate_random_validation,
    organizer_validation_fixture,
    qualify_reference_rungs,
)
from kuairand_agent.contract import verify_starter_kit
from kuairand_agent.data.canonical import (
    LOG_HEADER,
    STANDARD_LOG_FILENAMES,
    VIDEO_BASIC_FILENAME,
    VIDEO_BASIC_HEADER,
    CanonicalDataset,
    load_canonical_dataset,
)

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"

type OrganizerSplits = Mapping[str, Sequence[tuple[int, str, str, str, str, float, int]]]
type OrganizerResult = Mapping[str, Mapping[str, float | int]]
type OrganizerRunPop = Callable[[OrganizerSplits, float], OrganizerResult]
type OrganizerRunRandom = Callable[[OrganizerSplits, int], OrganizerResult]


def _log_row(user: str, video: str, date: int, label: int, ordinal: int) -> list[str]:
    values = {
        "user_id": user,
        "video_id": video,
        "date": str(date),
        "hourmin": "400",
        "time_ms": str(1_650_000_000_000 + ordinal),
        "is_click": str(label),
        "is_like": "0",
        "is_follow": "0",
        "is_comment": "0",
        "is_forward": "0",
        "is_hate": "0",
        "long_view": str(label),
        "play_time_ms": "20000",
        "duration_ms": "30000",
        "profile_stay_time": "0",
        "comment_stay_time": "0",
        "is_profile_enter": "0",
        "is_rand": "0",
        "tab": "1",
    }
    return [values[name] for name in LOG_HEADER]


def _write_csv(path: Path, header: tuple[str, ...], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fixture_dataset(tmp_path: Path) -> CanonicalDataset:
    data = tmp_path / "data"
    data.mkdir()
    video_rows = [
        [
            video,
            author,
            "NORMAL",
            "2020-01-01",
            "ShortImport",
            "1",
            "30000",
            "720",
            "1280",
            "1",
            "4",
            "12,65",
        ]
        for video, author in (("10", "100"), ("11", "101"), ("12", "102"))
    ]
    _write_csv(data / VIDEO_BASIC_FILENAME, VIDEO_BASIC_HEADER, video_rows)
    train_pairs = (
        ("1", "10", 1),
        ("1", "10", 0),
        ("2", "10", 1),
        ("2", "11", 0),
        ("3", "11", 1),
        ("3", "12", 0),
        ("4", "12", 1),
        ("4", "12", 0),
    )
    valid_pairs = (
        ("1", "10", 1),
        ("1", "11", 0),
        ("1", "99", 1),
        ("2", "10", 0),
        ("2", "11", 1),
        ("2", "12", 0),
        ("5", "99", 1),
        ("5", "12", 0),
    )
    _write_csv(
        data / STANDARD_LOG_FILENAMES[0],
        LOG_HEADER,
        [
            _log_row(user, video, 20220421, label, ordinal)
            for ordinal, (user, video, label) in enumerate(train_pairs)
        ],
    )
    late_rows = [
        _log_row(user, video, 20220422, label, 100 + ordinal)
        for ordinal, (user, video, label) in enumerate(valid_pairs)
    ]
    late_rows.append(_log_row("999", "99", 20220429, 1, 999))
    _write_csv(data / STANDARD_LOG_FILENAMES[1], LOG_HEADER, late_rows)
    return load_canonical_dataset(data)


def _verified_organizer_functions() -> tuple[OrganizerRunPop, OrganizerRunRandom]:
    """Import untouched functions without executing ``data.load`` or leaving bytecode behind."""

    before = verify_starter_kit(STARTER).manifest_sha256
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in ("data", "evaluate")}
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(STARTER))
    try:
        sys.modules.pop("data", None)
        sys.modules.pop("evaluate", None)
        namespace = runpy.run_path(str(STARTER / "baseline.py"), run_name="_pinned_baseline")
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = previous_bytecode
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    after = verify_starter_kit(STARTER).manifest_sha256
    assert after == before
    return (
        cast(OrganizerRunPop, namespace["run_pop"]),
        cast(OrganizerRunRandom, namespace["run_random"]),
    )


def _metric(result: Mapping[str, float | int], key: str) -> float:
    value = result[key]
    assert isinstance(value, float)
    return value


def test_canonical_adapters_exactly_match_untouched_random_and_popularity_functions(
    tmp_path: Path,
) -> None:
    dataset = fixture_dataset(tmp_path)
    fixture = organizer_validation_fixture(dataset)
    organizer_pop, organizer_random = _verified_organizer_functions()
    context = build_validation_scoring_context(dataset, STARTER)

    direct_pop = organizer_pop(fixture, 20.0)["valid"]
    protected_pop = evaluate_popularity_validation(dataset, context).evaluations[0].metrics
    assert protected_pop.gauc == _metric(direct_pop, "GAUC")
    assert protected_pop.ndcg_at_5 == _metric(direct_pop, "nDCG@5")
    assert protected_pop.primary == _metric(direct_pop, "primary")

    for seed in RANDOM_SEEDS:
        direct_random = organizer_random(fixture, seed)["valid"]
        protected_random = evaluate_random_validation(context, seed).metrics
        assert protected_random.gauc == _metric(direct_random, "GAUC")
        assert protected_random.ndcg_at_5 == _metric(direct_random, "nDCG@5")
        assert protected_random.primary == _metric(direct_random, "primary")

    # The organizer's mandatory test result came from the one-row training placeholder.  The
    # canonical final split has no targets and was never adapted or scored.
    assert fixture["test"][0] == fixture["train"][0]


@pytest.mark.skipif(
    "KUAIRAND_PURE_DATA_DIR" not in os.environ,
    reason="set KUAIRAND_PURE_DATA_DIR to run full-data published-rung parity",
)
def test_full_data_random_and_popularity_reproduce_published_validation_references() -> None:
    dataset: CanonicalDataset = load_canonical_dataset(os.environ["KUAIRAND_PURE_DATA_DIR"])
    qualification = qualify_reference_rungs(dataset, STARTER)

    assert qualification.random.reference_passed
    assert qualification.random.rounded_mean.manifest() == {
        "GAUC": 0.4993,
        "nDCG@5": 0.4675,
        "primary": 0.4834,
    }
    assert qualification.popularity.reference_passed
    assert qualification.popularity.rounded_mean.manifest() == {
        "GAUC": 0.6387,
        "nDCG@5": 0.5227,
        "primary": 0.5807,
    }
    assert all(run.rows == dataset.valid.row_count for run in qualification.random.evaluations)
    assert qualification.popularity.evaluations[0].rows == dataset.valid.row_count


def test_no_organizer_raw_loader_is_called_by_parity_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    organizer_pop, _ = _verified_organizer_functions()
    organizer_globals = cast(dict[str, Any], cast(Any, organizer_pop).__globals__)

    def forbidden_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("organizer data.load must never be called by parity adapters")

    monkeypatch.setitem(organizer_globals, "load", forbidden_load)
    dataset = fixture_dataset(tmp_path)
    result: dict[str, Any] = dict(organizer_pop(organizer_validation_fixture(dataset), 20.0))
    assert "valid" in result
