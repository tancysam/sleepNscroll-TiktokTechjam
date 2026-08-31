from __future__ import annotations

import dataclasses
import hashlib
import shutil
from pathlib import Path

import pytest

from kuairand_agent.challenge_contract import KUAI_PURE_CHALLENGE
from kuairand_agent.contract import (
    CONTRACT_ID,
    CONTRACT_MANIFEST,
    DATASET_ARCHIVE_SHA256,
    STARTER_FILE_SHA256,
    ContractManifestError,
    PinnedSplitInput,
    RepositoryContractVerificationReceipt,
    SplitName,
    contract_manifest,
    verify_contract_manifest,
    verify_repository_contract_inputs,
)
from kuairand_agent.domain.identity import ContractId, canonical_json_bytes


def _copy_repository_contract_inputs(destination: Path) -> Path:
    repository = destination / "repository"
    repository.mkdir()
    shutil.copytree(Path.cwd() / "kuairand-starter-kit", repository / "kuairand-starter-kit")
    shutil.copyfile(
        Path.cwd() / "kuairand-starter-kit.zip", repository / "kuairand-starter-kit.zip"
    )
    return repository


def test_complete_contract_manifest_has_exact_path_independent_contract_id() -> None:
    manifest = contract_manifest()

    assert set(manifest) == {
        "schema_version",
        "executable_benchmark_manifest",
        "challenge_manifest",
        "organizer_file_sha256",
        "dataset_sha256",
        "split_identities",
        "metric_implementation_sha256",
        "row_identity_policy",
        "submission_schema",
    }
    assert (
        ContractId("fedc7599f59c6cc1b3319c542d147dcaf499bdd08ceeba92551787d9c9bf4f93")
        == CONTRACT_ID
    )
    assert CONTRACT_MANIFEST.contract_id == CONTRACT_ID
    assert manifest["dataset_sha256"] == DATASET_ARCHIVE_SHA256
    assert manifest["organizer_file_sha256"] == dict(STARTER_FILE_SHA256)
    assert manifest["metric_implementation_sha256"] == STARTER_FILE_SHA256["evaluate.py"]
    assert b"/Users/" not in canonical_json_bytes(manifest)


def test_challenge_manifest_contains_every_admission_rule() -> None:
    manifest = KUAI_PURE_CHALLENGE.manifest()

    assert manifest["dataset"] == "KuaiRand-Pure"
    assert manifest["target"] == "long_view"
    assert manifest["metrics"] == {
        "names": ["GAUC", "nDCG@5"],
        "primary_formula": "(GAUC + nDCG@5) / 2",
        "ndcg_cutoff": 5,
    }
    assert manifest["scientific_iterations"] == {"maximum": 50}
    assert manifest["wall_clock_seconds"] == 21_600
    assert manifest["protected_evaluations"] == {"maximum": 6, "lineage": "ContractId"}
    assert manifest["external_training_data_allowed"] is False
    assert manifest["hidden_test_outcomes_representable"] is False
    assert manifest["prediction_context"] == "strict_past_only"


def test_contract_projection_is_a_copy_and_nested_mutation_cannot_change_identity() -> None:
    projected = contract_manifest()
    challenge = projected["challenge_manifest"]
    assert isinstance(challenge, dict)
    challenge["target"] = "is_click"

    assert CONTRACT_MANIFEST.contract_id == CONTRACT_ID
    assert contract_manifest()["challenge_manifest"] == KUAI_PURE_CHALLENGE.manifest()


def test_startup_verification_emits_receipt_and_rejects_contract_drift() -> None:
    receipt = verify_contract_manifest(CONTRACT_MANIFEST)
    assert receipt.verified is True
    assert receipt.contract_id == CONTRACT_ID
    assert receipt.expected_contract_id == CONTRACT_ID

    drifted = dataclasses.replace(CONTRACT_MANIFEST, dataset_sha256="f" * 64)
    assert drifted.contract_id != CONTRACT_ID
    with pytest.raises(ContractManifestError, match="contract mismatch"):
        verify_contract_manifest(drifted)


def test_contract_manifest_rejects_absolute_machine_paths() -> None:
    row_policy = dict(CONTRACT_MANIFEST.row_identity_policy)
    row_policy["debug_path"] = "/Users/example/repo"

    with pytest.raises(ContractManifestError, match="absolute machine paths"):
        dataclasses.replace(CONTRACT_MANIFEST, row_identity_policy=row_policy)


def test_live_repository_verification_hashes_actual_pinned_inputs_without_paths(
    tmp_path: Path,
) -> None:
    repository = _copy_repository_contract_inputs(tmp_path)
    split_path = tmp_path / "configured-train.split"
    split_path.write_bytes(b"explicitly pinned split bytes\n")
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()

    receipt = verify_repository_contract_inputs(
        repository,
        split_inputs=(PinnedSplitInput(SplitName.TRAIN, split_path, split_sha256),),
    )

    assert isinstance(receipt, RepositoryContractVerificationReceipt)
    assert receipt.verified is True
    assert receipt.contract_id == CONTRACT_ID
    assert receipt.repository_inputs.organizer_file_sha256 == STARTER_FILE_SHA256
    assert receipt.repository_inputs.starter_zip_sha256 == (
        "07237e62cc1a9cd8278556dab995dd5388516f10772724f582ef8320ac68b10b"
    )
    assert receipt.repository_inputs.dataset_archive_sha256 is None
    assert receipt.repository_inputs.split_input_sha256 == {"train": split_sha256}
    encoded = canonical_json_bytes(receipt.manifest())
    assert str(repository).encode() not in encoded
    assert str(split_path).encode() not in encoded


def test_live_repository_verification_fails_closed_when_organizer_artifact_is_missing(
    tmp_path: Path,
) -> None:
    repository = _copy_repository_contract_inputs(tmp_path)
    (repository / "kuairand-starter-kit" / "submit.py").unlink()

    with pytest.raises(ContractManifestError, match=r"missing=.*submit\.py"):
        verify_repository_contract_inputs(repository)


def test_live_repository_verification_fails_closed_when_scorer_is_tampered(
    tmp_path: Path,
) -> None:
    repository = _copy_repository_contract_inputs(tmp_path)
    with (repository / "kuairand-starter-kit" / "evaluate.py").open("ab") as handle:
        handle.write(b"\n# tampered\n")

    with pytest.raises(ContractManifestError, match=r"digest mismatch for evaluate\.py"):
        verify_repository_contract_inputs(repository)


def test_live_repository_verification_requires_the_pinned_source_zip(tmp_path: Path) -> None:
    repository = _copy_repository_contract_inputs(tmp_path)
    (repository / "kuairand-starter-kit.zip").unlink()

    with pytest.raises(ContractManifestError, match="starter ZIP is missing"):
        verify_repository_contract_inputs(repository)


def test_live_repository_verification_rejects_tampered_explicit_split_input(
    tmp_path: Path,
) -> None:
    repository = _copy_repository_contract_inputs(tmp_path)
    split_path = tmp_path / "configured-valid.split"
    split_path.write_bytes(b"before")
    pinned = hashlib.sha256(split_path.read_bytes()).hexdigest()
    split_path.write_bytes(b"after")

    with pytest.raises(ContractManifestError, match="valid split input digest mismatch"):
        verify_repository_contract_inputs(
            repository,
            split_inputs=(PinnedSplitInput(SplitName.VALID, split_path, pinned),),
        )


def test_live_repository_verification_rejects_missing_explicit_dataset_archive(
    tmp_path: Path,
) -> None:
    repository = _copy_repository_contract_inputs(tmp_path)

    with pytest.raises(ContractManifestError, match="dataset archive is missing"):
        verify_repository_contract_inputs(
            repository,
            dataset_archive=tmp_path / "not-present.tar.gz",
        )
