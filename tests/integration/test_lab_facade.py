from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import kuairand_agent.finalization.bundle as bundle_module
from kuairand_agent.domain.decisions import ReplayGrade, ScientificDisposition
from kuairand_agent.domain.identity import BundleId, PredictionId, canonical_json_bytes
from kuairand_agent.finalization.bundle import BundleProjectionError, EvidenceRole
from kuairand_agent.lab import (
    AutonomousExperimentLab,
    BundleValidationError,
    CampaignOptions,
    LabAdmissionError,
)
from kuairand_agent.state.repository import DurableRecord, RecordKind, StateRepository
from kuairand_agent.training.protocol import TrainerFailureCode


def _open_lab(tmp_path: Path) -> AutonomousExperimentLab:
    return AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        profile="cpu",
    )


def _fixture_options() -> CampaignOptions:
    return CampaignOptions(
        Path.cwd() / "configs/competition-cpu.toml",
        execution="offline-scripted",
        allow_test_fixture=True,
    )


def _reseal_json_evidence(
    source: Path,
    destination: Path,
    *,
    role: EvidenceRole,
    replacement: Mapping[str, object],
) -> None:
    shutil.copytree(source, destination)
    for path in destination.iterdir():
        os.chmod(path, 0o644)
    os.chmod(destination, 0o755)
    evidence_path = destination / role.value
    evidence_payload = canonical_json_bytes(dict(replacement)) + b"\n"
    evidence_path.write_bytes(evidence_payload)
    evidence_sha256 = hashlib.sha256(evidence_payload).hexdigest()
    manifest_path = destination / "bundle-manifest.json"
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text("ascii")))
    evidence = cast(list[dict[str, object]], manifest["evidence"])
    receipt = next(item for item in evidence if item["role"] == role.value)
    receipt_body = {
        "schema_version": 1,
        "role": role.value,
        "sha256": evidence_sha256,
        "size_bytes": len(evidence_payload),
    }
    receipt.update(receipt_body)
    receipt["receipt_id"] = hashlib.sha256(
        b"kuairand-frozen-bundle-file-v1\0" + canonical_json_bytes(receipt_body)
    ).hexdigest()
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    bundle_id = BundleId.derive(
        selected_prediction_id=PredictionId(cast(str, manifest["selected_prediction_id"])),
        replay_output_sha256={
            EvidenceRole.REPLAY_RECEIPT.value: cast(str, manifest["replay_receipt_sha256"])
        },
        submission_sha256=cast(str, manifest["submission_sha256"]),
        manifest_sha256=manifest_sha256,
    )
    (destination / "bundle.sha256").write_text(bundle_id.value + "\n", encoding="ascii")
    for path in destination.iterdir():
        os.chmod(path, 0o444)
    os.chmod(destination, 0o555)


def test_public_compete_refuses_missing_real_admission_before_state_creation(
    tmp_path: Path,
) -> None:
    lab = _open_lab(tmp_path)

    with pytest.raises(LabAdmissionError, match="before CampaignId creation") as failure:
        lab.compete(
            options=CampaignOptions(Path.cwd() / "configs/competition-cpu.toml"),
            idempotency_key="production-shaped-request",
        )

    assert failure.value.code is TrainerFailureCode.ADMISSION_REJECTED
    assert failure.value.selected_profile == "competition-cpu"
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "run").exists()


def test_gpu_preflight_selects_cpu_before_refusing_and_creates_no_campaign(
    tmp_path: Path,
) -> None:
    lab = AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        profile="gpu",
    )

    with pytest.raises(LabAdmissionError) as failure:
        lab.compete(
            options=CampaignOptions(Path.cwd() / "configs/competition-gpu.toml"),
            idempotency_key="gpu-production-shaped-request",
        )

    assert failure.value.requested_profile == "competition-gpu"
    assert failure.value.selected_profile == "competition-cpu"
    assert "GPU preflight selected CPU" in failure.value.missing_evidence[0]
    assert not (tmp_path / "state").exists()


def test_scripted_fallback_vertical_slice_is_terminal_replayable_and_idempotent(
    tmp_path: Path,
) -> None:
    lab = _open_lab(tmp_path)
    options = _fixture_options()

    first = lab.compete(options=options, idempotency_key="offline-fixture")
    second = lab.compete(options=options, idempotency_key="offline-fixture")

    assert second == first
    assert first.terminal_state == "COMPLETED_OFFLINE_FIXTURE"
    assert first.submission_disposition == "SCRIPTED_FALLBACK_RETAINED"
    assert first.scientific_disposition == ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE.value
    assert first.selected_prediction_id == first.fallback_prediction_id
    assert first.exact_metrics is None
    assert first.protected_query_count == 0
    assert first.qualification_scope == "SCRIPTED_FIXTURE_ONLY"
    assert first.bundle_path.is_dir()

    snapshot = lab.inspect(campaign_id=first.campaign_id)
    entities = cast(Mapping[str, list[object]], snapshot["entities"])
    assert entities["protected_query_reservations"] == []
    assert entities["protected_evaluations"] == []
    assert entities["provider_operations"] == []
    assert len(entities["predictions"]) == 1
    assert len(entities["resource_receipts"]) == 1
    resource_row = cast(Mapping[str, object], entities["resource_receipts"][0])
    resource_payload = cast(Mapping[str, object], resource_row["payload"])
    observed = cast(Mapping[str, object], resource_payload["observed_resources"])
    assert cast(int, observed["peak_rss_bytes"]) > 0
    assert observed["peak_rss_bytes_measured"] is True
    assert observed["peak_disk_bytes"] is None
    assert observed["peak_disk_bytes_measured"] is False
    assert observed["threads"] is None
    assert observed["threads_measured"] is False

    scoring = lab.replay(
        campaign_id=first.campaign_id,
        grade=ReplayGrade.EXPERIMENT_SAME_BACKEND,
    )
    bundle = lab.replay(campaign_id=first.campaign_id, grade="bundle-exact")
    assert scoring.verified is True
    assert scoring.grade == ReplayGrade.EXPERIMENT_SAME_BACKEND.value
    assert bundle.verified is True
    assert bundle.grade == ReplayGrade.BUNDLE_EXACT.value
    assert bundle.bundle_id == first.bundle_sha256


def test_inspect_does_not_mutate_authority_files(tmp_path: Path) -> None:
    lab = _open_lab(tmp_path)
    result = lab.compete(
        options=_fixture_options(),
        idempotency_key="read-only-inspection",
    )
    state_root = tmp_path / "state"
    before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in state_root.iterdir()
        if not path.name.endswith(("-shm", "-wal"))
    }

    snapshot = lab.inspect(campaign_id=result.campaign_id)

    after = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in state_root.iterdir()
        if not path.name.endswith(("-shm", "-wal"))
    }
    campaign = cast(Mapping[str, object], snapshot["campaign"])
    assert campaign["state"] == "COMPLETED_OFFLINE_FIXTURE"
    assert after == before


def test_validate_bundle_rejects_tampered_evidence(tmp_path: Path) -> None:
    lab = _open_lab(tmp_path)
    result = lab.compete(
        options=_fixture_options(),
        idempotency_key="tamper-detection",
    )
    submission = result.bundle_path / "submission.csv"
    os.chmod(submission, 0o644)
    submission.write_bytes(submission.read_bytes() + b"\n")

    with pytest.raises(BundleValidationError, match="digest changed"):
        lab.validate_bundle(result.bundle_path)


def test_validate_bundle_rejects_symlink_root_before_resolution(tmp_path: Path) -> None:
    lab = _open_lab(tmp_path)
    result = lab.compete(
        options=_fixture_options(),
        idempotency_key="symlink-root",
    )
    symlink = tmp_path / "bundle-symlink"
    symlink.symlink_to(result.bundle_path, target_is_directory=True)

    with pytest.raises(BundleValidationError, match="real directory, not a symlink"):
        lab.validate_bundle(symlink)


def test_validate_bundle_rejects_unsealed_root_and_member(tmp_path: Path) -> None:
    lab = _open_lab(tmp_path)
    result = lab.compete(
        options=_fixture_options(),
        idempotency_key="unsealed-bundle",
    )
    os.chmod(result.bundle_path, 0o755)
    with pytest.raises(BundleValidationError, match="root must be sealed"):
        lab.validate_bundle(result.bundle_path)

    os.chmod(result.bundle_path, 0o555)
    report = result.bundle_path / EvidenceRole.REPORT.value
    os.chmod(report, 0o644)
    with pytest.raises(BundleValidationError, match="member is not sealed"):
        lab.validate_bundle(result.bundle_path)


def test_validate_bundle_rejects_resealed_cross_campaign_lineage(tmp_path: Path) -> None:
    lab = _open_lab(tmp_path)
    result = lab.compete(
        options=_fixture_options(),
        idempotency_key="resealed-lineage",
    )
    campaign_manifest = cast(
        dict[str, object],
        json.loads((result.bundle_path / EvidenceRole.CAMPAIGN_MANIFEST.value).read_text("utf-8")),
    )
    campaign_manifest["campaign_id"] = "f" * 64
    malicious = tmp_path / "resealed-cross-campaign-bundle"
    _reseal_json_evidence(
        result.bundle_path,
        malicious,
        role=EvidenceRole.CAMPAIGN_MANIFEST,
        replacement=campaign_manifest,
    )

    with pytest.raises(BundleValidationError, match="CampaignId lineage"):
        lab.validate_bundle(malicious)


def test_same_authority_and_run_root_isolate_campaign_bundles_and_reuse_experiment(
    tmp_path: Path,
) -> None:
    first_lab = AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        profile="cpu",
    )
    second_lab = AutonomousExperimentLab.open(
        repository_root=Path.cwd(),
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        profile="cpu",
    )

    first = first_lab.compete(options=_fixture_options(), idempotency_key="first-campaign")
    second = second_lab.compete(options=_fixture_options(), idempotency_key="second-campaign")
    first_snapshot = first_lab.inspect(campaign_id=first.campaign_id)
    second_snapshot = second_lab.inspect(campaign_id=second.campaign_id)
    first_entities = cast(Mapping[str, list[Mapping[str, object]]], first_snapshot["entities"])
    second_entities = cast(Mapping[str, list[Mapping[str, object]]], second_snapshot["entities"])

    assert first.campaign_id != second.campaign_id
    assert (
        first_entities["experiments"][0]["experiment_id"]
        == second_entities["experiments"][0]["experiment_id"]
    )
    assert first.bundle_path != second.bundle_path
    assert first.campaign_id in first.bundle_path.parts
    assert second.campaign_id in second.bundle_path.parts
    assert first_lab.validate_bundle(first.bundle_path).campaign_id == first.campaign_id
    assert second_lab.validate_bundle(second.bundle_path).campaign_id == second.campaign_id


def test_published_bundle_recovers_after_crash_before_atomic_terminal_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = _open_lab(tmp_path)
    original = StateRepository.finalize_prepared_campaign

    def crash_before_commit(self: StateRepository, **_kwargs: object) -> None:
        raise RuntimeError("injected crash before terminal commit")

    monkeypatch.setattr(StateRepository, "finalize_prepared_campaign", crash_before_commit)
    with pytest.raises(RuntimeError, match="injected crash"):
        lab.compete(options=_fixture_options(), idempotency_key="recoverable-publication")

    monkeypatch.setattr(StateRepository, "finalize_prepared_campaign", original)
    recovered = lab.compete(options=_fixture_options(), idempotency_key="recoverable-publication")
    snapshot = lab.inspect(campaign_id=recovered.campaign_id)
    entities = cast(Mapping[str, list[Mapping[str, object]]], snapshot["entities"])
    bundle_payload = cast(Mapping[str, object], entities["bundles"][0]["payload"])

    assert recovered.terminal_state == "COMPLETED_OFFLINE_FIXTURE"
    assert recovered.replay_grades == (
        ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        ReplayGrade.BUNDLE_EXACT.value,
    )
    assert len(entities["resource_receipts"]) == 1
    assert (
        lab.validate_bundle(recovered.bundle_path).preparation_id
        == bundle_payload["preparation_id"]
    )


def test_campaign_retry_adopts_exact_orphan_after_bundle_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = _open_lab(tmp_path)
    original_publish = cast(
        Callable[[Path, Path], None],
        bundle_module.__dict__["_legacy_publish_exclusive"],
    )
    armed = True

    def publish_then_crash(source: Path, destination: Path) -> None:
        nonlocal armed
        original_publish(source, destination)
        if armed:
            armed = False
            raise OSError("injected crash after exclusive bundle rename")

    monkeypatch.setattr(bundle_module, "_legacy_publish_exclusive", publish_then_crash)
    with pytest.raises(BundleProjectionError, match="failed closed"):
        lab.compete(options=_fixture_options(), idempotency_key="bundle-rename-recovery")

    orphan_roots = list((tmp_path / "run" / "campaigns").glob("*/final/submission-bundle"))
    assert len(orphan_roots) == 1
    assert stat.S_IMODE(orphan_roots[0].lstat().st_mode) == 0o700

    monkeypatch.setattr(bundle_module, "_legacy_publish_exclusive", original_publish)
    recovered = lab.compete(options=_fixture_options(), idempotency_key="bundle-rename-recovery")

    assert recovered.terminal_state == "COMPLETED_OFFLINE_FIXTURE"
    assert recovered.bundle_path == orphan_roots[0].resolve()
    assert stat.S_IMODE(recovered.bundle_path.lstat().st_mode) == 0o555
    assert lab.validate_bundle(recovered.bundle_path).bundle_id == recovered.bundle_sha256


def test_execution_recovers_after_crash_before_attempt_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = _open_lab(tmp_path)
    original = StateRepository.transition
    armed = True

    def crash_before_attempt_completion(self: StateRepository, *args: Any, **kwargs: Any) -> Any:
        nonlocal armed
        if (
            armed
            and kwargs.get("entity_kind") == "attempt"
            and kwargs.get("expected_state") == "RUNNING"
            and kwargs.get("new_state") == "COMPLETED"
        ):
            armed = False
            raise RuntimeError("injected crash before attempt completion")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(StateRepository, "transition", crash_before_attempt_completion)
    with pytest.raises(RuntimeError, match="before attempt completion"):
        lab.compete(options=_fixture_options(), idempotency_key="attempt-running-recovery")

    monkeypatch.setattr(StateRepository, "transition", original)
    recovered = lab.compete(options=_fixture_options(), idempotency_key="attempt-running-recovery")
    snapshot = lab.inspect(campaign_id=recovered.campaign_id)
    entities = cast(Mapping[str, list[Mapping[str, object]]], snapshot["entities"])

    assert entities["attempts"][0]["state"] == "COMPLETED"
    assert entities["attempts"][0]["terminal"] is True
    assert entities["trials"][0]["state"] == "COMPLETED"
    assert len(entities["resource_receipts"]) == 1


def test_execution_recovers_after_prediction_before_resource_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = _open_lab(tmp_path)
    original = StateRepository.register
    armed = True

    def crash_before_resource_receipt(
        self: StateRepository,
        record: DurableRecord,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal armed
        if armed and record.kind is RecordKind.RESOURCE_RECEIPT:
            armed = False
            raise RuntimeError("injected crash before resource receipt")
        return original(self, record, *args, **kwargs)

    monkeypatch.setattr(StateRepository, "register", crash_before_resource_receipt)
    with pytest.raises(RuntimeError, match="before resource receipt"):
        lab.compete(options=_fixture_options(), idempotency_key="resource-receipt-recovery")

    monkeypatch.setattr(StateRepository, "register", original)
    recovered = lab.compete(options=_fixture_options(), idempotency_key="resource-receipt-recovery")
    snapshot = lab.inspect(campaign_id=recovered.campaign_id)
    entities = cast(Mapping[str, list[Mapping[str, object]]], snapshot["entities"])

    assert len(entities["predictions"]) == 1
    assert len(entities["resource_receipts"]) == 1
    assert entities["attempts"][0]["state"] == "COMPLETED"
    assert entities["trials"][0]["state"] == "COMPLETED"
