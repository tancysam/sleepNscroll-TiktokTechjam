from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kuairand_agent.candidate_api.runtime_contract import (
    CANDIDATE_RUNTIME_CONTRACT,
    CandidateRuntimeContractError,
)

ROOT = Path(__file__).parents[2]
SEED_ENTRYPOINT = ROOT / "candidate_seed" / "candidate.py"


def _seed_literal(name: str) -> object:
    tree = ast.parse(SEED_ENTRYPOINT.read_text(encoding="utf-8"), filename=str(SEED_ENTRYPOINT))
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"seed entrypoint does not declare {name}")


def test_distributed_seed_wrapper_matches_the_canonical_controller_contract() -> None:
    contract = CANDIDATE_RUNTIME_CONTRACT

    assert _seed_literal("SCHEMA_VERSION") == contract.schema_version
    assert _seed_literal("TRAIN_KEYS") == set(contract.train_request_fields)
    assert _seed_literal("PREDICT_KEYS") == set(contract.prediction_request_fields)
    assert contract.to_wire()["stable_files"] == {
        "entry_point": "candidate.py",
        "protected_paths": [
            "candidate.py",
            "reference_categorical_ranker.py",
            "reference_listnet_ranker.py",
            "reference_observed_pair_fm.py",
            "reference_observed_pair_objectives.py",
            "reference_pairwise_fm.py",
            "reference_pointwise_ranker.py",
        ],
    }
    assert len(contract.digest) == 64


def test_executor_payload_builders_have_exact_fields_and_fixed_handles() -> None:
    contract = CANDIDATE_RUNTIME_CONTRACT
    train = contract.training_payload(
        source_digest="1" * 64,
        config_digest="2" * 64,
        data_digest="3" * 64,
        split_token="fold-b",
        seed=7,
    )
    prediction = contract.prediction_payload(
        source_digest="1" * 64,
        config_digest="2" * 64,
        data_digest="4" * 64,
        split_token="fold-b-valid",
        expected_count=12,
        checkpoint_digest="5" * 64,
    )

    assert set(train) == set(contract.train_request_fields)
    assert train["features_handle"] == "features"
    assert train["targets_handle"] == "targets"
    assert train["user_groups_handle"] == "user_groups"
    assert set(prediction) == set(contract.prediction_request_fields)
    assert prediction["features_handle"] == "features"


def test_runtime_contract_rejects_protected_model_overlays() -> None:
    with pytest.raises(CandidateRuntimeContractError, match="protected runtime file"):
        CANDIDATE_RUNTIME_CONTRACT.validate_overlay_paths(
            ("config.json", "candidate.py", "reference_categorical_ranker.py", "model_impl.py")
        )

    CANDIDATE_RUNTIME_CONTRACT.validate_overlay_paths(
        ("config.json", "model_impl.py", "pairwise.py")
    )
