from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import kuairand_agent.lab as lab_module
from kuairand_agent.contract import CONTRACT_ID, ContractManifestError
from kuairand_agent.lab import (
    AutonomousExperimentLab,
    CampaignOptions,
    LabAdmissionError,
)
from kuairand_agent.observability.receipts import ReceiptError, ScriptedReplayReceipt


def test_open_verifies_contract_before_creating_state_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root = tmp_path / "state-that-must-not-exist"

    def reject_contract(_candidate: object) -> None:
        raise ContractManifestError("fixture contract drift")

    monkeypatch.setattr(lab_module, "verify_repository_contract_inputs", reject_contract)

    with pytest.raises(LabAdmissionError, match="before state open"):
        AutonomousExperimentLab.open(
            repository_root=Path.cwd(),
            state_root=state_root,
            run_root=tmp_path / "run",
            profile="cpu",
        )

    assert not state_root.exists()


def test_open_is_no_write_and_emits_exact_startup_receipt(tmp_path: Path) -> None:
    state_root = tmp_path / "state"

    lab = AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=state_root,
        run_root=tmp_path / "run",
        profile="competition-cpu",
    )

    assert not state_root.exists()
    assert lab.startup_receipt.contract_id == CONTRACT_ID.value
    assert lab.startup_receipt.expected_contract_id == CONTRACT_ID.value
    assert lab.startup_receipt.verified is True
    assert lab.startup_receipt.state_writes_started is False


def test_compete_reverifies_contract_after_open_before_creating_state(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    shutil.copytree(Path.cwd() / "kuairand-starter-kit", repository / "kuairand-starter-kit")
    shutil.copyfile(
        Path.cwd() / "kuairand-starter-kit.zip",
        repository / "kuairand-starter-kit.zip",
    )
    state_root = tmp_path / "state-that-must-not-exist"
    run_root = tmp_path / "run-that-must-not-exist"
    lab = AutonomousExperimentLab.open(
        repository_root=repository,
        state_root=state_root,
        run_root=run_root,
        profile="cpu",
    )
    with (repository / "kuairand-starter-kit" / "evaluate.py").open("ab") as handle:
        handle.write(b"\n# post-open contract tampering\n")

    with pytest.raises(LabAdmissionError, match="re-verification failed before state open"):
        lab.compete(
            options=CampaignOptions(
                Path.cwd() / "configs/competition-cpu.toml",
                execution="offline-scripted",
                allow_test_fixture=True,
            ),
            idempotency_key="post-open-tamper",
        )

    assert not state_root.exists()
    assert not run_root.exists()


def test_compete_reverifies_again_at_state_write_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    shutil.copytree(Path.cwd() / "kuairand-starter-kit", repository / "kuairand-starter-kit")
    shutil.copyfile(
        Path.cwd() / "kuairand-starter-kit.zip",
        repository / "kuairand-starter-kit.zip",
    )
    state_root = tmp_path / "state-that-must-not-exist"
    run_root = tmp_path / "run-that-must-not-exist"
    lab = AutonomousExperimentLab.open(
        repository_root=repository,
        state_root=state_root,
        run_root=run_root,
        profile="cpu",
    )
    original_load_profile = AutonomousExperimentLab._load_profile

    def load_profile_then_tamper(
        self: AutonomousExperimentLab,
        config_path: Path,
    ) -> object:
        profile = original_load_profile(self, config_path)
        with (repository / "kuairand-starter-kit" / "evaluate.py").open("ab") as handle:
            handle.write(b"\n# tampered between admission and state open\n")
        return profile

    monkeypatch.setattr(AutonomousExperimentLab, "_load_profile", load_profile_then_tamper)

    with pytest.raises(LabAdmissionError, match="re-verification failed before state open"):
        lab.compete(
            options=CampaignOptions(
                Path.cwd() / "configs/competition-cpu.toml",
                execution="offline-scripted",
                allow_test_fixture=True,
            ),
            idempotency_key="write-seam-tamper",
        )

    assert not state_root.exists()
    assert not run_root.exists()


def test_campaign_options_rejects_unknown_execution() -> None:
    with pytest.raises(LabAdmissionError, match="execution must be"):
        CampaignOptions(Path("profile.toml"), execution="live")


def test_scripted_fixture_requires_explicit_test_gate() -> None:
    with pytest.raises(LabAdmissionError, match="requires allow_test_fixture=True"):
        CampaignOptions(Path("profile.toml"), execution="offline-scripted")


def test_scripted_replay_receipt_rejects_different_prediction_bytes() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64

    with pytest.raises(ReceiptError, match="prediction bytes differ"):
        ScriptedReplayReceipt(
            contract_id=CONTRACT_ID.value,
            campaign_id="e" * 64,
            prediction_id="c" * 64,
            first_prediction_sha256=digest_a,
            replay_prediction_sha256=digest_b,
            first_result_sha256="d" * 64,
            replay_result_sha256="d" * 64,
        )
