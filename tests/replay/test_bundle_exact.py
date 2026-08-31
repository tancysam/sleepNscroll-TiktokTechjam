from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import kuairand_agent.finalization.bundle as bundle_module
from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import BundleId, CampaignId, ContractId, PredictionId
from kuairand_agent.finalization.bundle import (
    REQUIRED_BUNDLE_PATHS,
    REQUIRED_EVIDENCE_ROLES,
    BundleFinalizationRequest,
    BundleFinalizer,
    BundleProjectionError,
    EvidenceRole,
    FrozenFileReceipt,
    TerminalProjectionBinding,
)


def _payload(role: EvidenceRole) -> bytes:
    if role is EvidenceRole.CAMPAIGN_STATE_SNAPSHOT:
        return b"SQLite format 3\x00frozen-fixture"
    if role in {EvidenceRole.EVENT_EXPORT, EvidenceRole.RESOURCE_RECEIPTS}:
        return (json.dumps({"role": role.value}, sort_keys=True) + "\n").encode("ascii")
    if role is EvidenceRole.SUBMISSION:
        return b"user_id,video_id,prediction\nu1,v1,0.5\n"
    if role is EvidenceRole.REPORT:
        return b"# Frozen campaign report\n"
    return (json.dumps({"role": role.value}, sort_keys=True) + "\n").encode("ascii")


def _receipts(tmp_path: Path) -> tuple[FrozenFileReceipt, ...]:
    source = tmp_path / "receipts"
    source.mkdir()
    receipts: list[FrozenFileReceipt] = []
    for role in REQUIRED_EVIDENCE_ROLES:
        path = source / role.value
        path.write_bytes(_payload(role))
        receipts.append(FrozenFileReceipt.capture(role, path))
    return tuple(receipts)


def _request(
    destination: Path,
    receipts: tuple[FrozenFileReceipt, ...],
) -> BundleFinalizationRequest:
    return BundleFinalizationRequest(
        destination=destination,
        contract_id=ContractId("a" * 64),
        campaign_id=CampaignId("b" * 64),
        selected_prediction_id=PredictionId("c" * 64),
        terminal_projection=TerminalProjectionBinding(
            preparation_id="d" * 64,
            projection_sha256="e" * 64,
            campaign_revision=7,
            last_event_seq=19,
        ),
        receipts=receipts,
    )


def test_bundle_projection_emits_exact_layout_and_domain_bundle_id(tmp_path: Path) -> None:
    receipts = _receipts(tmp_path)
    destination = tmp_path / "final"

    result = BundleFinalizer().finalize(_request(destination, receipts))

    assert {path.name for path in destination.iterdir()} == set(REQUIRED_BUNDLE_PATHS)
    assert result.replay_grade.grade is ReplayGrade.BUNDLE_EXACT
    assert result.regeneration_evidence.first_bundle_id == result.bundle_id
    assert result.regeneration_evidence.regenerated_bundle_id == result.bundle_id
    manifest_bytes = (destination / "bundle-manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == result.manifest_sha256
    assert (destination / "bundle.sha256").read_text(encoding="ascii") == (result.bundle_id + "\n")
    manifest = json.loads(manifest_bytes)
    assert manifest["schema_version"] == 2
    assert manifest["terminal_projection"] == {
        "preparation_id": "d" * 64,
        "projection_sha256": "e" * 64,
        "redaction_policy_version": 1,
        "schema_version": 1,
        "source_revision": {"campaign_revision": 7, "last_event_seq": 19},
    }
    replay_receipt = next(
        receipt for receipt in receipts if receipt.role is EvidenceRole.REPLAY_RECEIPT
    )
    submission = next(receipt for receipt in receipts if receipt.role is EvidenceRole.SUBMISSION)
    expected = BundleId.derive(
        selected_prediction_id=PredictionId("c" * 64),
        replay_output_sha256={EvidenceRole.REPLAY_RECEIPT.value: replay_receipt.sha256},
        submission_sha256=submission.sha256,
        manifest_sha256=result.manifest_sha256,
    )
    assert result.bundle_id == expected.value


def test_same_frozen_receipts_regenerate_same_bundle_without_runtime_metadata(
    tmp_path: Path,
) -> None:
    receipts = _receipts(tmp_path)
    finalizer = BundleFinalizer()

    first = finalizer.finalize(_request(tmp_path / "first", receipts))
    second = finalizer.finalize(_request(tmp_path / "second", receipts))

    assert first.bundle_id == second.bundle_id
    assert first.inventory_sha256 == second.inventory_sha256
    for name in REQUIRED_BUNDLE_PATHS:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes()
    manifest = json.loads(first.manifest_path.read_text(encoding="ascii"))
    assert "created_at" not in manifest
    assert "elapsed_seconds" not in manifest
    assert "resource_usage" not in manifest


def test_exact_sealed_publication_is_recovered_without_overwrite(tmp_path: Path) -> None:
    receipts = _receipts(tmp_path)
    destination = tmp_path / "final"
    finalizer = BundleFinalizer()
    first = finalizer.finalize(_request(destination, receipts))
    before = {
        member.name: (member.stat().st_ino, member.stat().st_mtime_ns)
        for member in destination.iterdir()
    }

    recovered = finalizer.finalize(_request(destination, receipts))

    assert recovered.root == first.root
    assert recovered.bundle_id == first.bundle_id
    assert recovered.inventory_sha256 == first.inventory_sha256
    assert recovered.replay_grade.grade is ReplayGrade.BUNDLE_EXACT
    assert (destination / "bundle.sha256").read_text(encoding="ascii") == (first.bundle_id + "\n")
    assert {
        member.name: (member.stat().st_ino, member.stat().st_mtime_ns)
        for member in destination.iterdir()
    } == before


def test_sealed_publication_recovery_rejects_different_frozen_inputs(tmp_path: Path) -> None:
    receipts = _receipts(tmp_path)
    destination = tmp_path / "final"
    BundleFinalizer().finalize(_request(destination, receipts))

    report = next(receipt for receipt in receipts if receipt.role is EvidenceRole.REPORT)
    report.source.write_text("# Mutated recovery input\n", encoding="utf-8")
    changed_report = FrozenFileReceipt.capture(EvidenceRole.REPORT, report.source)
    changed = tuple(
        changed_report if receipt.role is EvidenceRole.REPORT else receipt for receipt in receipts
    )

    with pytest.raises(BundleProjectionError, match="regeneration"):
        BundleFinalizer().finalize(_request(destination, changed))

    assert stat.S_IMODE(destination.lstat().st_mode) == 0o555


def test_exact_orphan_after_exclusive_rename_is_verified_sealed_and_adopted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = _receipts(tmp_path)
    destination = tmp_path / "final"
    request = _request(destination, receipts)
    original_publish = cast(
        Callable[[Path, Path], None],
        bundle_module.__dict__["_legacy_publish_exclusive"],
    )
    armed = True

    def publish_then_crash(source: Path, target: Path) -> None:
        nonlocal armed
        original_publish(source, target)
        if armed:
            armed = False
            raise OSError("injected crash after exclusive rename")

    monkeypatch.setattr(bundle_module, "_legacy_publish_exclusive", publish_then_crash)
    with pytest.raises(BundleProjectionError, match="failed closed"):
        BundleFinalizer().finalize(request)

    assert stat.S_IMODE(destination.lstat().st_mode) == 0o700
    monkeypatch.setattr(bundle_module, "_legacy_publish_exclusive", original_publish)
    recovered = BundleFinalizer().finalize(request)

    assert recovered.root == destination.resolve()
    assert stat.S_IMODE(destination.lstat().st_mode) == 0o555
    assert destination.lstat().st_mode & 0o222 == 0
    for member in destination.iterdir():
        assert member.lstat().st_mode & 0o222 == 0
        assert stat.S_ISREG(member.lstat().st_mode)
    assert os.path.lexists(destination)


def test_mutated_receipt_source_fails_closed_before_publication(tmp_path: Path) -> None:
    receipts = _receipts(tmp_path)
    report = next(receipt for receipt in receipts if receipt.role is EvidenceRole.REPORT)
    report.source.chmod(0o644)
    report.source.write_text("# changed after capture\n", encoding="utf-8")
    destination = tmp_path / "final"

    with pytest.raises(BundleProjectionError, match="frozen receipt"):
        BundleFinalizer().finalize(_request(destination, receipts))

    assert not destination.exists()


def test_request_rejects_an_incomplete_evidence_layout(tmp_path: Path) -> None:
    receipts = _receipts(tmp_path)

    with pytest.raises(BundleProjectionError, match="required evidence layout"):
        _request(tmp_path / "final", receipts[:-1])
