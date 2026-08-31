from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import kuairand_agent.lab as lab_module
import kuairand_agent.production.controller as production_controller_module
from kuairand_agent.contract import CONTRACT_ID, ContractManifestError
from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import CampaignId
from kuairand_agent.lab import (
    AutonomousExperimentLab,
    CampaignOptions,
    CampaignResult,
    LabAdmissionError,
    LabConflictError,
)
from kuairand_agent.observability.receipts import ReceiptError, ScriptedReplayReceipt
from kuairand_agent.production.admission import ProductionAdmission
from kuairand_agent.production.controller import (
    ProductionCPUFallbackRequest,
    ProductionCPUFallbackResult,
)
from kuairand_agent.state.repository import StateRepository


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


def test_running_retry_reenters_controller_for_exact_sealed_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_id = CampaignId("a" * 64)
    campaign_revision = 1
    selected_prediction_id = "b" * 64
    qualification_manifest_digest = "c" * 64
    qualification_fallback_digest = "d" * 64
    preparation_id = "e" * 64
    prepared = {
        "preparation_id": preparation_id,
        "campaign_id": campaign_id.value,
        "contract_id": CONTRACT_ID.value,
        "source_state": "RUNNING",
        "source_campaign_revision": campaign_revision,
        "source_last_event_seq": 7,
        "terminal_state": "COMPLETED",
        "decision_id": "f" * 64,
        "selected_prediction_id": selected_prediction_id,
        "fallback_prediction_id": selected_prediction_id,
        "prepared_projection_sha256": "1" * 64,
        "projection_schema_version": 1,
        "redaction_policy_version": 1,
        "replay_payload": {
            "contract_id": CONTRACT_ID.value,
            "campaign_id": campaign_id.value,
            "prediction_id": selected_prediction_id,
            "qualification_manifest_digest": qualification_manifest_digest,
            "qualification_fallback_digest": qualification_fallback_digest,
            "qualification_scope": "FULL_DATA_CPU",
            "final_period_outcomes_accessed": False,
        },
        "bundle_claims": {
            "campaign_kind": "PRODUCTION_FULL_DATA",
            "qualification_scope": "FULL_DATA_CPU",
            "qualification_manifest_digest": qualification_manifest_digest,
            "protected_query_count": 0,
            "provider_operation_count": 0,
        },
    }
    running_snapshot = {
        "campaign": {
            "state": "RUNNING",
            "revision": campaign_revision,
            "terminal": False,
        },
        "entities": {
            "terminal_preparations": [prepared],
            "bundle_publications": [],
        },
    }
    completed_snapshot = {"campaign": {"state": "COMPLETED"}, "entities": {}}
    expected_campaign_id = campaign_id

    class _Repository:
        def inspect(self, *, campaign_id: object) -> dict[str, object]:
            assert campaign_id == expected_campaign_id
            return running_snapshot

    repository = _Repository()
    admission = cast(
        ProductionAdmission,
        SimpleNamespace(
            qualification_manifest_digest=qualification_manifest_digest,
            fallback_manifest_digest=qualification_fallback_digest,
        ),
    )
    lab = AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        profile="cpu",
    )
    destination = tmp_path / "run" / "campaigns" / campaign_id.value / "final" / "submission-bundle"
    destination.mkdir(parents=True)
    destination.chmod(0o555)
    expected_result = CampaignResult(
        campaign_id=campaign_id.value,
        contract_id=CONTRACT_ID.value,
        terminal_state="COMPLETED",
        submission_disposition="FALLBACK_RETAINED",
        scientific_disposition="INSUFFICIENT_VALID_EVIDENCE",
        selected_prediction_id=selected_prediction_id,
        fallback_prediction_id=selected_prediction_id,
        exact_metrics=None,
        bundle_path=destination,
        bundle_sha256="2" * 64,
        replay_grade=ReplayGrade.BUNDLE_EXACT.value,
        resource_receipt_id="3" * 64,
        protected_query_count=0,
        evidence_manifest_path=destination / "evidence-manifest.json",
        replay_grades=(
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
            ReplayGrade.BUNDLE_EXACT.value,
        ),
        campaign_kind="PRODUCTION_FULL_DATA",
        qualification_scope="FULL_DATA_CPU",
    )
    observed: list[ProductionCPUFallbackRequest] = []

    def resume(
        _controller: object,
        request: ProductionCPUFallbackRequest,
    ) -> ProductionCPUFallbackResult:
        observed.append(request)
        assert request.campaign_id == campaign_id.value
        assert request.campaign_revision == campaign_revision
        assert request.run_dir == destination.parents[1]
        assert destination.is_dir()
        return ProductionCPUFallbackResult(
            snapshot=completed_snapshot,
            bundle_path=destination,
            bundle_id="2" * 64,
            manifest_sha256="4" * 64,
            inventory_sha256="5" * 64,
            submission_sha256="6" * 64,
            resource_receipt_id="3" * 64,
        )

    def result_from_snapshot(
        _lab: AutonomousExperimentLab,
        snapshot: object,
    ) -> CampaignResult:
        assert snapshot is completed_snapshot
        return expected_result

    monkeypatch.setattr(production_controller_module.ProductionCPUFallbackController, "run", resume)
    monkeypatch.setattr(AutonomousExperimentLab, "_result_from_snapshot", result_from_snapshot)

    result = lab._resume_production_cpu_fallback(
        repository=cast(StateRepository, repository),
        campaign_id=campaign_id,
        campaign_revision=campaign_revision,
        admission=admission,
        campaign_run_root=destination.parents[1],
    )

    assert result is expected_result
    assert len(observed) == 1
    assert destination.stat().st_mode & 0o777 == 0o555


def test_running_retry_without_preparation_reenters_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_id = CampaignId("a" * 64)
    campaign_revision = 1
    running_snapshot = {
        "campaign": {
            "state": "RUNNING",
            "revision": campaign_revision,
            "terminal": False,
        },
        "entities": {
            "terminal_preparations": [],
            "bundle_publications": [],
        },
    }

    class _Repository:
        def inspect(self, *, campaign_id: object) -> dict[str, object]:
            return running_snapshot

    repository = _Repository()
    admission = cast(
        ProductionAdmission,
        SimpleNamespace(
            qualification_manifest_digest="c" * 64,
            fallback_manifest_digest="d" * 64,
        ),
    )
    lab = AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        profile="cpu",
    )
    campaign_run_root = tmp_path / "run" / "campaigns" / campaign_id.value
    completed_snapshot = {"campaign": {"state": "COMPLETED"}, "entities": {}}
    observed: list[ProductionCPUFallbackRequest] = []
    expected_result = object()

    def resume(
        _controller: object,
        request: ProductionCPUFallbackRequest,
    ) -> ProductionCPUFallbackResult:
        observed.append(request)
        return ProductionCPUFallbackResult(
            snapshot=completed_snapshot,
            bundle_path=campaign_run_root / "final" / "submission-bundle",
            bundle_id="2" * 64,
            manifest_sha256="4" * 64,
            inventory_sha256="5" * 64,
            submission_sha256="6" * 64,
            resource_receipt_id="3" * 64,
        )

    def result_from_snapshot(
        _lab: AutonomousExperimentLab,
        snapshot: object,
    ) -> CampaignResult:
        assert snapshot is completed_snapshot
        return cast(CampaignResult, expected_result)

    monkeypatch.setattr(production_controller_module.ProductionCPUFallbackController, "run", resume)
    monkeypatch.setattr(AutonomousExperimentLab, "_result_from_snapshot", result_from_snapshot)

    result = lab._resume_production_cpu_fallback(
        repository=cast(StateRepository, repository),
        campaign_id=campaign_id,
        campaign_revision=campaign_revision,
        admission=admission,
        campaign_run_root=campaign_run_root,
    )

    assert result is expected_result
    assert len(observed) == 1
    assert observed[0].run_dir == campaign_run_root


def test_running_retry_rejects_bundle_without_terminal_preparation(
    tmp_path: Path,
) -> None:
    campaign_id = CampaignId("a" * 64)
    campaign_revision = 1
    running_snapshot = {
        "campaign": {
            "state": "RUNNING",
            "revision": campaign_revision,
            "terminal": False,
        },
        "entities": {
            "terminal_preparations": [],
            "bundle_publications": [],
        },
    }

    class _Repository:
        def inspect(self, *, campaign_id: object) -> dict[str, object]:
            return running_snapshot

    lab = AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        profile="cpu",
    )
    campaign_run_root = tmp_path / "run" / "campaigns" / campaign_id.value
    destination = campaign_run_root / "final" / "submission-bundle"
    destination.mkdir(parents=True)

    with pytest.raises(
        LabConflictError,
        match="destination exists without a terminal preparation",
    ):
        lab._resume_production_cpu_fallback(
            repository=cast(StateRepository, _Repository()),
            campaign_id=campaign_id,
            campaign_revision=campaign_revision,
            admission=cast(
                ProductionAdmission,
                SimpleNamespace(
                    qualification_manifest_digest="c" * 64,
                    fallback_manifest_digest="d" * 64,
                ),
            ),
            campaign_run_root=campaign_run_root,
        )


def test_running_retry_rejects_preparation_from_different_admission() -> None:
    campaign_id = CampaignId("a" * 64)
    selected_prediction_id = "b" * 64
    prepared = {
        "preparation_id": "c" * 64,
        "campaign_id": campaign_id.value,
        "contract_id": CONTRACT_ID.value,
        "source_state": "RUNNING",
        "source_campaign_revision": 1,
        "source_last_event_seq": 7,
        "terminal_state": "COMPLETED",
        "decision_id": "d" * 64,
        "selected_prediction_id": selected_prediction_id,
        "fallback_prediction_id": selected_prediction_id,
        "prepared_projection_sha256": "e" * 64,
        "projection_schema_version": 1,
        "redaction_policy_version": 1,
        "replay_payload": {
            "contract_id": CONTRACT_ID.value,
            "campaign_id": campaign_id.value,
            "prediction_id": selected_prediction_id,
            "qualification_manifest_digest": "f" * 64,
            "qualification_fallback_digest": "1" * 64,
            "qualification_scope": "FULL_DATA_CPU",
            "final_period_outcomes_accessed": False,
        },
        "bundle_claims": {
            "campaign_kind": "PRODUCTION_FULL_DATA",
            "qualification_scope": "FULL_DATA_CPU",
            "qualification_manifest_digest": "f" * 64,
            "protected_query_count": 0,
            "provider_operation_count": 0,
        },
    }

    with pytest.raises(
        LabConflictError,
        match="differs from admitted fallback evidence",
    ):
        AutonomousExperimentLab._validate_production_resume_preparation(
            prepared,
            campaign_id=campaign_id,
            campaign_revision=1,
            admission=cast(
                ProductionAdmission,
                SimpleNamespace(
                    qualification_manifest_digest="2" * 64,
                    fallback_manifest_digest="1" * 64,
                ),
            ),
        )
