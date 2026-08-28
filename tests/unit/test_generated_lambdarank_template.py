from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "candidate_templates" / "lambdarank"


class _CandidateModule(Protocol):
    CONFIG_KEYS: set[str]
    TRAIN_KEYS: set[str]
    PREDICT_KEYS: set[str]

    def _stable_user_grouping(
        self, user_groups: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], tuple[int, ...]]: ...


def _candidate_module() -> _CandidateModule:
    # Execute the immutable source bytes directly.  SourceFileLoader would write a
    # ``__pycache__`` member into the organizer-visible template tree, making the next
    # content-addressed source snapshot depend on which tests happened to run first.
    source_path = TEMPLATE / "candidate.py"
    module = ModuleType("generated_lambdarank_candidate")
    exec(compile(source_path.read_bytes(), str(source_path), "exec"), module.__dict__)
    return cast(_CandidateModule, module)


def test_template_config_and_request_schemas_are_exact_and_seed_replay_safe() -> None:
    module = _candidate_module()
    raw_config = json.loads((TEMPLATE / "config.json").read_text(encoding="ascii"))
    assert isinstance(raw_config, dict)
    config = cast(dict[str, object], raw_config)

    assert (
        set(config)
        == module.CONFIG_KEYS
        == {
            "candidate_family",
            "learning_rate",
            "min_data_in_leaf",
            "num_boost_round",
            "num_leaves",
            "num_threads",
            "schema_version",
            "seed_policy",
            "tree_count_policy",
        }
    )
    assert config["seed_policy"] == "controller_request_uint32"
    assert config["tree_count_policy"] == "frozen_train_derived"
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


def test_stable_user_grouping_preserves_first_seen_users_and_within_user_order() -> None:
    module = _candidate_module()
    groups = np.array([20, 10, 20, 30, 10, 30, 20], dtype=np.int64)

    permutation, sizes = module._stable_user_grouping(groups)

    assert permutation.tolist() == [0, 2, 6, 1, 4, 3, 5]
    assert sizes == (3, 2, 2)
    assert groups[permutation].tolist() == [20, 20, 20, 10, 10, 30, 30]


def test_template_owns_material_mechanism_without_trusted_or_provider_imports() -> None:
    source = (TEMPLATE / "candidate.py").read_text(encoding="utf-8")

    assert "def train_model(" in source
    assert "def predict_scores(" in source
    assert "def _stable_user_grouping(" in source
    assert '"device_type": "cpu"' in source
    assert '"deterministic": True' in source
    assert '"force_col_wise": True' in source
    assert "inverse_permutation" not in source
    assert "from kuairand_agent" not in source
    assert "import kuairand_agent" not in source
    assert "OPENAI_API_KEY" not in source
    assert "requests." not in source
    assert "urllib" not in source
    assert "KuaiRand-Pure" not in source
