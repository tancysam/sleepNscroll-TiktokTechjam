from __future__ import annotations

import ast
import json
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "candidate_templates" / "pairwise_fm"


class _PairBatch(Protocol):
    positive_indices: npt.NDArray[np.int64]
    negative_indices: npt.NDArray[np.int64]
    user_group_indices: npt.NDArray[np.int64]


class _Sampler(Protocol):
    eligible_user_count: int
    eligible_positive_count: int
    stored_row_index_count: int
    pair_space_size: int

    def sample(self, pair_count: int, *, seed: int) -> _PairBatch: ...


class _CandidateModule(Protocol):
    CONFIG_KEYS: set[str]
    TRAIN_KEYS: set[str]
    PREDICT_KEYS: set[str]

    def GAUCPairSampler(
        self,
        user_groups: npt.NDArray[np.int64],
        targets: npt.NDArray[np.int8],
    ) -> _Sampler: ...


def _candidate_module() -> _CandidateModule:
    source_path = TEMPLATE / "candidate.py"
    module = ModuleType("generated_pairwise_fm_candidate")
    exec(compile(source_path.read_bytes(), str(source_path), "exec"), module.__dict__)
    return cast(_CandidateModule, module)


def test_template_inventory_is_exactly_the_standalone_generated_package() -> None:
    assert sorted(path.name for path in TEMPLATE.iterdir()) == [
        "README.md",
        "candidate.py",
        "config.json",
    ]


def test_config_and_request_schemas_pin_a_bounded_seeded_pairwise_fm() -> None:
    module = _candidate_module()
    config = cast(
        dict[str, object],
        json.loads((TEMPLATE / "config.json").read_text(encoding="ascii")),
    )

    assert (
        set(config)
        == module.CONFIG_KEYS
        == {
            "candidate_family",
            "epochs",
            "factor_dim",
            "l2",
            "learning_rate",
            "pair_batch_size",
            "pairs_per_epoch",
            "schema_version",
            "seed_policy",
        }
    )
    assert config["candidate_family"] == "deterministic_pairwise_fm"
    assert config["seed_policy"] == "controller_request_uint32"
    assert {
        "config_digest",
        "data_digest",
        "features_handle",
        "protocol_schema_version",
        "seed",
        "source_digest",
        "split_token",
        "targets_handle",
        "user_groups_handle",
    } == module.TRAIN_KEYS
    assert {
        "checkpoint_digest",
        "config_digest",
        "data_digest",
        "expected_count",
        "features_handle",
        "protocol_schema_version",
        "source_digest",
        "split_token",
    } == module.PREDICT_KEYS


def test_sampler_is_deterministic_same_user_and_positive_count_weighted() -> None:
    module = _candidate_module()
    # User 10 has one positive and two negatives. User 20 has two positives and three
    # negatives. The target user law is therefore 1/3 versus 2/3, not uniform by user.
    users = np.array([10, 10, 10, 20, 20, 20, 20, 20], dtype=np.int64)
    targets = np.array([1, 0, 0, 1, 1, 0, 0, 0], dtype=np.int8)
    sampler = module.GAUCPairSampler(users, targets)

    first = sampler.sample(90_000, seed=17)
    replay = sampler.sample(90_000, seed=17)

    assert np.array_equal(first.positive_indices, replay.positive_indices)
    assert np.array_equal(first.negative_indices, replay.negative_indices)
    assert np.array_equal(users[first.positive_indices], users[first.negative_indices])
    assert np.all(targets[first.positive_indices] == 1)
    assert np.all(targets[first.negative_indices] == 0)
    user_20_fraction = float(np.mean(users[first.positive_indices] == 20))
    assert 0.655 < user_20_fraction < 0.678
    user_20_pairs = users[first.positive_indices] == 20
    user_20_positive_counts = np.bincount(first.positive_indices[user_20_pairs], minlength=5)[3:5]
    user_20_negative_counts = np.bincount(first.negative_indices[user_20_pairs], minlength=8)[5:8]
    assert int(np.ptp(user_20_positive_counts)) < 700
    assert int(np.ptp(user_20_negative_counts)) < 700
    assert sampler.eligible_user_count == 2
    assert sampler.eligible_positive_count == 3
    assert sampler.stored_row_index_count == 8
    assert sampler.pair_space_size == 8


def test_source_owns_the_mechanism_and_has_only_standard_library_plus_numpy_imports() -> None:
    source = (TEMPLATE / "candidate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "re",
        "stat",
        "typing",
        "numpy",
    }
    assert "class GAUCPairSampler" in source
    assert "def train_model(" in source
    assert "def predict_scores(" in source
    assert "positive_tickets" in source
    assert "full_catalog" not in source.lower()
    assert "from kuairand_agent" not in source
    assert "import kuairand_agent" not in source
    assert "OPENAI_API_KEY" not in source
    assert "requests." not in source
    assert "urllib" not in source
