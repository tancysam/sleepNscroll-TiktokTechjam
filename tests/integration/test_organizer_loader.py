from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.organizer import load_verified_organizer
from kuairand_agent.contract import verify_starter_kit
from kuairand_agent.data.canonical import CanonicalInputs

STARTER_DIR = Path(__file__).resolve().parents[2] / "kuairand-starter-kit"


def test_verified_loader_uses_private_modules_and_binds_exact_dependencies() -> None:
    before = verify_starter_kit(STARTER_DIR)
    loaded = load_verified_organizer(STARTER_DIR)
    after = verify_starter_kit(STARTER_DIR)

    assert loaded.root == STARTER_DIR.resolve()
    assert loaded.manifest_sha256 == before.manifest_sha256 == after.manifest_sha256
    assert loaded.data.__name__.startswith("_kuairand_pinned_data_")
    assert loaded.evaluate.__name__.startswith("_kuairand_pinned_evaluate_")
    assert loaded.baseline.__name__.startswith("_kuairand_pinned_baseline_")
    assert loaded.baseline.encode is loaded.data.encode
    assert loaded.baseline.evaluate is loaded.evaluate.evaluate
    assert not (STARTER_DIR / "__pycache__").exists()


def test_loader_restores_existing_aliases_and_bytecode_setting_by_identity() -> None:
    sentinel_data = ModuleType("data")
    sentinel_evaluate = ModuleType("evaluate")
    previous_data = sys.modules.get("data")
    previous_evaluate = sys.modules.get("evaluate")
    previous_bytecode = sys.dont_write_bytecode
    sys.modules["data"] = sentinel_data
    sys.modules["evaluate"] = sentinel_evaluate
    sys.dont_write_bytecode = False
    try:
        loaded = load_verified_organizer(STARTER_DIR)
        assert loaded.baseline.encode is loaded.data.encode
        assert sys.modules["data"] is sentinel_data
        assert sys.modules["evaluate"] is sentinel_evaluate
        assert sys.dont_write_bytecode is False
    finally:
        if previous_data is None:
            sys.modules.pop("data", None)
        else:
            sys.modules["data"] = previous_data
        if previous_evaluate is None:
            sys.modules.pop("evaluate", None)
        else:
            sys.modules["evaluate"] = previous_evaluate
        sys.dont_write_bytecode = previous_bytecode


def test_loading_modules_never_calls_organizer_data_load() -> None:
    loaded = load_verified_organizer(STARTER_DIR)

    # If baseline import had invoked load(), it would have needed a data directory and failed.
    # The callable remains available solely for fixture parity; qualification never calls it.
    assert callable(loaded.data.load)
    assert loaded.baseline.FM.__module__ == loaded.baseline.__name__


def _canonical(rows: list[tuple[int, str, str, str, str, float, int]]) -> CanonicalInputs:
    return CanonicalInputs(
        date=tuple(row[0] for row in rows),
        user_id=tuple(row[1] for row in rows),
        video_id=tuple(row[2] for row in rows),
        author_id=tuple(row[3] for row in rows),
        tab=tuple(row[4] for row in rows),
        duration_ms=tuple(row[5] for row in rows),
        time_ms=tuple(range(len(rows))),
    )


def test_starter_encoding_matches_untouched_organizer_encode_on_fixture() -> None:
    loaded = load_verified_organizer(STARTER_DIR)
    train = [
        (20220408, "u2", "v2", "a2", "1", 10.0, 1),
        (20220408, "u1", "v1", "a1", "0", 20.0, 0),
        (20220409, "u2", "v3", "a2", "1", 30.0, 1),
        (20220409, "u3", "v1", "UNK", "2", 40.0, 0),
    ]
    valid = [(20220422, "new-u", "new-v", "new-a", "14", 100.0, 0)]
    placeholder_test = [train[0]]
    organizer_encoded, organizer_dim = loaded.data.encode(
        {"train": train, "valid": valid, "test": placeholder_test}
    )

    encoding = StarterEncoding.fit(_canonical(train))
    assert encoding.total_dim == organizer_dim
    np.testing.assert_array_equal(
        encoding.transform(_canonical(train)), organizer_encoded["train"][0]
    )
    np.testing.assert_array_equal(
        encoding.transform(_canonical(valid)), organizer_encoded["valid"][0]
    )
    np.testing.assert_array_equal(
        encoding.transform(_canonical(placeholder_test)), organizer_encoded["test"][0]
    )
