from __future__ import annotations

import inspect
import math

import pytest

from kuairand_agent.domain.identity import (
    AttemptId,
    CampaignId,
    CanonicalJsonError,
    ContractId,
    ExperimentId,
    FamilyId,
    IdentityError,
    PredictionId,
    TrialId,
    canonical_json_bytes,
    canonical_json_sha256,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def test_canonical_json_is_utf8_sorted_and_normalizes_equivalent_numbers() -> None:
    left = {"z": [1.0, -0.0, 1e-7], "é": {"value": 0.1}}
    right = {"é": {"value": 0.1}, "z": (1, 0, 0.0000001)}

    assert canonical_json_bytes(left) == (b'{"z":[1,0,0.0000001],"\xc3\xa9":{"value":0.1}}')
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(CanonicalJsonError, match="finite"):
        canonical_json_bytes({"value": value})


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string key"},
        {"bytes": b"not JSON"},
        {"set": {1, 2}},
    ],
)
def test_canonical_json_rejects_implicit_python_coercions(value: object) -> None:
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes(value)


def test_nominal_ids_require_full_lowercase_sha_and_do_not_compare_across_types() -> None:
    contract_id = ContractId(DIGEST_A)
    campaign_id = CampaignId(DIGEST_A)

    assert str(contract_id) == DIGEST_A
    assert len({contract_id, campaign_id}) == 2
    with pytest.raises(IdentityError, match="full lowercase"):
        ContractId("A" * 64)
    with pytest.raises(IdentityError, match="full lowercase"):
        ContractId("a" * 32)


def test_identity_constructor_signatures_exclude_operational_metadata() -> None:
    family_parameters = set(inspect.signature(FamilyId.derive).parameters)
    experiment_parameters = set(inspect.signature(ExperimentId.derive).parameters)
    trial_parameters = set(inspect.signature(TrialId.derive).parameters)

    assert {"seed", "device", "attempt", "pid", "hostname"}.isdisjoint(family_parameters)
    assert {"device", "retry_count", "attempt", "pid", "hostname"}.isdisjoint(experiment_parameters)
    assert {"pid", "hostname", "retry_count", "infrastructure_attempt"}.isdisjoint(trial_parameters)


def test_identity_manifests_reject_absolute_machine_paths() -> None:
    with pytest.raises(IdentityError, match="absolute machine paths"):
        ContractId.from_manifest({"dataset_path": "/Users/example/data"})
    with pytest.raises(IdentityError, match="absolute machine paths"):
        CampaignId.derive(
            contract_id=ContractId(DIGEST_A),
            campaign_config={"cache": "C:\\machine\\cache"},
            start_nonce="nonce-1",
        )


def test_backend_changes_trial_not_experiment_and_attempt_is_separate() -> None:
    experiment = ExperimentId.derive(
        experiment_spec={"model": "lambda_rank", "learning_rate": 0.1},
        data_identities={"train": DIGEST_A},
        fold_identities={"fold_a": DIGEST_B},
        code_artifact_sha256=DIGEST_C,
    )
    common: dict[str, object] = {
        "experiment_id": experiment,
        "trainer_id": "lightgbm",
        "trainer_version": "4.7.0",
        "precision": "float64",
        "dependency_lock_sha256": DIGEST_A,
        "seed": 0,
        "fold": "fold_a",
        "fidelity": {"fraction": 1.0},
        "qualified_settings": {"deterministic": True},
    }
    cpu = TrialId.derive(backend="cpu", **common)  # type: ignore[arg-type]
    gpu = TrialId.derive(backend="gpu", **common)  # type: ignore[arg-type]

    assert cpu != gpu
    assert AttemptId.derive(trial_id=cpu, infrastructure_attempt=1) != AttemptId.derive(
        trial_id=cpu, infrastructure_attempt=2
    )
    assert (
        ExperimentId.derive(
            experiment_spec={"learning_rate": 0.1, "model": "lambda_rank"},
            data_identities={"train": DIGEST_A},
            fold_identities={"fold_a": DIGEST_B},
            code_artifact_sha256=DIGEST_C,
        )
        == experiment
    )


def test_prediction_identity_binds_order_exact_bytes_and_producer() -> None:
    experiment = ExperimentId(DIGEST_A)
    trial = TrialId.derive(
        experiment_id=experiment,
        trainer_id="scripted",
        trainer_version="1",
        backend="cpu",
        precision="float64",
        dependency_lock_sha256=DIGEST_B,
        seed=0,
        fold="fold_a",
        fidelity="full",
        qualified_settings={},
    )
    first = PredictionId.from_trial(
        ordered_row_ids=(0, 1, 2), prediction_sha256=DIGEST_C, trial_id=trial
    )

    assert first == PredictionId.from_trial(
        ordered_row_ids=[0, 1, 2], prediction_sha256=DIGEST_C, trial_id=trial
    )
    assert first != PredictionId.from_trial(
        ordered_row_ids=[2, 1, 0], prediction_sha256=DIGEST_C, trial_id=trial
    )
    assert first != PredictionId.from_rank_graph(
        ordered_row_ids=[0, 1, 2],
        prediction_sha256=DIGEST_C,
        rank_graph_sha256=DIGEST_B,
    )
