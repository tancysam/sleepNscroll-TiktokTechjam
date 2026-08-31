from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import kuairand_agent.lab as lab_module
from kuairand_agent.data.canonical import OUTCOME_FIELDS
from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import canonical_json_bytes, canonical_json_sha256
from kuairand_agent.finalization.bundle import EvidenceRole
from kuairand_agent.finalization.organizer_check import (
    MaskedFileEvidence,
    MaskedViewEvidence,
    OrganizerCheckEvidence,
)
from kuairand_agent.finalization.replay import (
    CleanReplayEvidence,
    FinalReplayEvidence,
    FrozenReplayIdentity,
    ReplayEquality,
    ValidationReplayEvidence,
)
from kuairand_agent.finalization.replay_grades import (
    BundleRegenerationEvidence,
    combine_replay_grade_receipts,
    derive_bundle_exact_grade,
    derive_clean_replay_grade_receipts,
)
from kuairand_agent.lab import BundleValidationResult, LabError
from kuairand_agent.production import controller
from kuairand_agent.production.controller import (
    ProductionControllerError,
    ProductionCPUFallbackRequest,
    ProductionCPUFallbackResult,
)
from kuairand_agent.state.repository import (
    DurableRecord,
    PostpublicationResourceReceipt,
    RecordKind,
    StateRepository,
)

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src" / "kuairand_agent" / "production" / "controller.py"


def test_public_controller_seam_is_frozen_and_stable() -> None:
    assert tuple(field.name for field in fields(ProductionCPUFallbackRequest)) == (
        "repository",
        "admission",
        "campaign_id",
        "contract_id",
        "campaign_revision",
        "run_dir",
        "repository_root",
        "startup_receipt",
    )
    assert tuple(field.name for field in fields(ProductionCPUFallbackResult)) == (
        "snapshot",
        "bundle_path",
        "bundle_id",
        "manifest_sha256",
        "inventory_sha256",
        "submission_sha256",
        "resource_receipt_id",
    )
    result = ProductionCPUFallbackResult(
        snapshot={},
        bundle_path=Path("bundle"),
        bundle_id="0" * 64,
        manifest_sha256="1" * 64,
        inventory_sha256="2" * 64,
        submission_sha256="3" * 64,
        resource_receipt_id="4" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        result.bundle_id = "5" * 64  # type: ignore[misc]


def test_controller_source_has_one_authority_and_no_legacy_or_network_path() -> None:
    parsed = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    assert "kuairand_agent.state.repository" in imports
    assert (
        not {
            "kuairand_agent.campaign.controller",
            "kuairand_agent.campaign.store",
            "kuairand_agent.campaign.full_campaign",
            "kuairand_agent.campaign.full_campaign_runtime",
            "kuairand_agent.campaign.candidate_journal",
            "kuairand_agent.campaign.scientific_store",
        }
        & imports
    )
    assert not {"CampaignEngine", "CampaignStore", "reserve_protected_query"} & calls
    assert "NotImplementedError" not in SOURCE.read_text(encoding="utf-8")


def test_production_terminal_claims_are_the_bounded_exact_pair() -> None:
    assert controller._REPLAY_GRADES == (
        "SCORING_EXACT",
        "EXPERIMENT_SAME_BACKEND",
        "BUNDLE_EXACT",
    )
    assert controller._PREPUBLICATION_REPLAY_GRADES == (
        "SCORING_EXACT",
        "EXPERIMENT_SAME_BACKEND",
    )
    assert controller._CAMPAIGN_KIND == "PRODUCTION_FULL_DATA"
    assert controller._QUALIFICATION_SCOPE == "FULL_DATA_CPU"
    assert controller._SUBMISSION_DISPOSITION == "FALLBACK_RETAINED"


def test_prediction_identity_is_deterministic_and_binds_rank_graph() -> None:
    alignment = canonical_json_sha256({"alignment": "final"})
    prediction = canonical_json_sha256({"prediction": "final"})
    rank_graph = controller._rank_graph_identity(
        final_alignment=alignment,
        final_prediction_digest=prediction,
    )
    first = controller._provisional_prediction_identity(
        final_alignment=alignment,
        final_prediction_digest=prediction,
        final_rows=3,
        rank_graph_id=rank_graph,
    )
    second = controller._provisional_prediction_identity(
        final_alignment=alignment,
        final_prediction_digest=prediction,
        final_rows=3,
    )
    changed = controller._provisional_prediction_identity(
        final_alignment=alignment,
        final_prediction_digest=prediction,
        final_rows=4,
        rank_graph_id=rank_graph,
    )
    assert first == second
    assert first != changed


def test_qualification_metric_reverification_is_exact() -> None:
    metrics = {"GAUC": 0.6, "nDCG@5": 0.4, "primary": 0.5}
    controller._assert_qualified_metrics(metrics, dict(metrics))
    with pytest.raises(ProductionControllerError, match="differ"):
        controller._assert_qualified_metrics(metrics, metrics | {"primary": 0.5000000001})


def _clean_replay() -> CleanReplayEvidence:
    validation = ValidationReplayEvidence(
        row_count=3,
        reference_prediction_digest="1" * 64,
        replay_prediction_digest="1" * 64,
        replay_prediction_file_sha256="3" * 64,
        exact_prediction_bytes=True,
        maximum_absolute_difference=0.0,
        top5_order_identical=True,
        protected_metrics_identical=True,
        metrics={"GAUC": 0.6, "nDCG@5": 0.8, "primary": 0.7},
        public_submission_sha256="4" * 64,
        public_submission_prediction_digest="1" * 64,
        csv_round_trip_identity=True,
        csv_within_user_order_preserved=True,
        csv_top5_preserved=True,
        csv_protected_metrics_preserved=True,
    )
    final = FinalReplayEvidence(
        row_count=3,
        prediction_digest="5" * 64,
        prediction_file_sha256="6" * 64,
        submission_sha256="7" * 64,
        submission_prediction_digest="5" * 64,
        finite_scores=True,
        csv_round_trip_identity=True,
    )
    return CleanReplayEvidence(
        candidate_id="official-fm-seed-4-qualified-fallback",
        identity=FrozenReplayIdentity(
            source_sha256="a" * 64,
            config_sha256="b" * 64,
            features_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
            validation_prediction_artifact_sha256="e" * 64,
            validation_prediction_digest="1" * 64,
            data_sha256="f" * 64,
            environment_sha256="8" * 64,
        ),
        equality=ReplayEquality.EXACT,
        absolute_tolerance=0.0,
        training_replay="immutable checkpoint replay",
        validation=validation,
        final=final,
        validation_capability_digest="9" * 64,
        final_capability_digest="0" * 64,
    )


def test_exact_replay_payload_preserves_clean_evidence_and_scoring_receipt() -> None:
    evidence = _clean_replay()
    receipts = derive_clean_replay_grade_receipts(
        contract_id="a" * 64,
        prediction_id="b" * 64,
        evidence=evidence,
    )
    scoring, same_backend = controller._select_exact_replay_receipts(
        receipts=receipts,
        expected_contract_id="a" * 64,
        expected_prediction_id="b" * 64,
    )

    payload = controller._exact_replay_evidence_payload(
        evidence=evidence,
        scoring_receipt=scoring,
        same_backend_receipt=same_backend,
    )

    assert payload["clean_replay_evidence"] == evidence.manifest()
    assert payload["clean_replay_evidence_sha256"] == canonical_json_sha256(evidence.manifest())
    scoring_manifest = payload["scoring_exact_receipt"]
    assert isinstance(scoring_manifest, dict)
    assert scoring_manifest["grade"] == "SCORING_EXACT"
    assert scoring_manifest["prediction_id"] == "b" * 64
    same_backend_manifest = payload["same_backend_receipt"]
    assert isinstance(same_backend_manifest, dict)
    assert same_backend_manifest["grade"] == "EXPERIMENT_SAME_BACKEND"
    assert same_backend_manifest["prediction_id"] == "b" * 64
    assert (
        scoring_manifest["evidence"]["clean_replay_sha256"]
        == same_backend_manifest["evidence"]["clean_replay_sha256"]
    )
    with pytest.raises(ProductionControllerError, match="outside campaign lineage"):
        controller._select_exact_replay_receipts(
            receipts=receipts,
            expected_contract_id="a" * 64,
            expected_prediction_id="c" * 64,
        )

    other_receipts = derive_clean_replay_grade_receipts(
        contract_id="a" * 64,
        prediction_id="b" * 64,
        evidence=replace(evidence, candidate_id="other-candidate"),
    )
    with pytest.raises(ProductionControllerError, match="same clean replay"):
        controller._select_exact_replay_receipts(
            receipts=(scoring, other_receipts[1]),
            expected_contract_id="a" * 64,
            expected_prediction_id="b" * 64,
        )


def test_lab_accepts_v3_embedded_scoring_and_same_backend_proof() -> None:
    contract_id = "a" * 64
    campaign_id = "c" * 64
    prediction_id = "b" * 64
    evidence = _clean_replay()
    scoring, same_backend = derive_clean_replay_grade_receipts(
        contract_id=contract_id,
        prediction_id=prediction_id,
        evidence=evidence,
    )
    exact = controller._exact_replay_evidence_payload(
        evidence=evidence,
        scoring_receipt=scoring,
        same_backend_receipt=same_backend,
    )
    organizer = replace(
        _organizer_evidence(final_rows=3),
        submission_sha256="7" * 64,
    )
    organizer_manifest = controller._validate_organizer_evidence(
        organizer,
        expected_submission_sha256="7" * 64,
        expected_starter_manifest_sha256="2" * 64,
        expected_final_rows=3,
    )
    replay = {
        "schema_version": 3,
        "contract_id": contract_id,
        "campaign_id": campaign_id,
        "prediction_id": prediction_id,
        "qualification_manifest_digest": "d" * 64,
        "qualification_fallback_digest": "e" * 64,
        "original_prediction_sha256": "5" * 64,
        "replay_prediction_sha256": "5" * 64,
        "row_alignment_sha256": "f" * 64,
        "submission_sha256": "7" * 64,
        "organizer_check_sha256": canonical_json_sha256(organizer_manifest),
        "organizer_check": organizer_manifest,
        **exact,
        "achieved_replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        ],
        "required_terminal_replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
            ReplayGrade.BUNDLE_EXACT.value,
        ],
        "qualification_scope": "FULL_DATA_CPU",
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
    }
    campaign = {
        "production_admission": {
            "starter_manifest_sha256": "2" * 64,
            "data": {"final_rows": 3},
        }
    }
    scientific = {
        "organizer_check": organizer_manifest,
        "exact_replay_evidence": exact,
    }

    lab_module._validate_embedded_production_replay(
        replay,
        campaign_evidence=campaign,
        scientific_evidence=scientific,
        contract_id=contract_id,
        campaign_id=campaign_id,
        prediction_id=prediction_id,
        submission_sha256="7" * 64,
        submission_size_bytes=organizer.submission_size_bytes,
    )

    stale = dict(replay)
    stale["schema_version"] = 2
    with pytest.raises(LabError, match="schema v3"):
        lab_module._validate_embedded_production_replay(
            stale,
            campaign_evidence=campaign,
            scientific_evidence=scientific,
            contract_id=contract_id,
            campaign_id=campaign_id,
            prediction_id=prediction_id,
            submission_sha256="7" * 64,
            submission_size_bytes=organizer.submission_size_bytes,
        )

    other_same_backend = derive_clean_replay_grade_receipts(
        contract_id=contract_id,
        prediction_id=prediction_id,
        evidence=replace(evidence, candidate_id="different-clean-replay"),
    )[1]
    crossed = dict(replay)
    crossed["same_backend_receipt"] = other_same_backend.manifest()
    with pytest.raises(LabError, match="exact replay receipt"):
        lab_module._validate_embedded_production_replay(
            crossed,
            campaign_evidence=campaign,
            scientific_evidence=scientific,
            contract_id=contract_id,
            campaign_id=campaign_id,
            prediction_id=prediction_id,
            submission_sha256="7" * 64,
            submission_size_bytes=organizer.submission_size_bytes,
        )


def _organizer_evidence(*, final_rows: int = 2) -> OrganizerCheckEvidence:
    masked_files = (
        MaskedFileEvidence(
            relative_path="log_standard_4_08_to_4_21_pure.csv",
            sha256="0" * 64,
            size_bytes=128,
            data_rows=5,
            final_rows_masked=0,
        ),
        MaskedFileEvidence(
            relative_path="log_standard_4_22_to_5_08_pure.csv",
            sha256="1" * 64,
            size_bytes=128,
            data_rows=5,
            final_rows_masked=final_rows,
        ),
        MaskedFileEvidence(
            relative_path="video_features_basic_pure.csv",
            sha256="4" * 64,
            size_bytes=128,
            data_rows=None,
            final_rows_masked=None,
        ),
    )
    cells = final_rows * len(OUTCOME_FIELDS)
    stable = {
        "schema_version": 1,
        "files": [entry.manifest() for entry in masked_files],
        "registered_outcome_fields": list(OUTCOME_FIELDS),
        "final_rows_masked": final_rows,
        "final_outcome_cells_replaced": cells,
    }
    masked = MaskedViewEvidence(
        files=masked_files,
        final_rows_masked=final_rows,
        final_outcome_cells_replaced=cells,
        digest=canonical_json_sha256(stable),
    )
    stdout = "OK\n"
    stderr = ""
    return OrganizerCheckEvidence(
        starter_manifest_sha256="2" * 64,
        submission_sha256="3" * 64,
        submission_size_bytes=256,
        masked_view=masked,
        checker_command=controller._ORGANIZER_COMMAND,
        checker_returncode=0,
        checker_stdout=stdout,
        checker_stderr=stderr,
        checker_stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
        checker_stderr_sha256=hashlib.sha256(stderr.encode()).hexdigest(),
    )


def test_lab_validates_three_grade_authority_and_postpublication_resource_lineage(
    tmp_path: Path,
) -> None:
    contract_id = "a" * 64
    campaign_id = "b" * 64
    prediction_id = "c" * 64
    bundle_id = "d" * 64
    submission_sha256 = "e" * 64
    inventory_sha256 = "f" * 64
    performance_profile_digest = "1" * 64
    scoring, same_backend = derive_clean_replay_grade_receipts(
        contract_id=contract_id,
        prediction_id=prediction_id,
        evidence=_clean_replay(),
    )
    regeneration = BundleRegenerationEvidence._from_verified_projection(
        contract_id=contract_id,
        prediction_id=prediction_id,
        first_bundle_id=bundle_id,
        regenerated_bundle_id=bundle_id,
        first_submission_sha256=submission_sha256,
        regenerated_submission_sha256=submission_sha256,
        first_inventory_sha256=inventory_sha256,
        regenerated_inventory_sha256=inventory_sha256,
    )
    bundle_exact = derive_bundle_exact_grade(regeneration)
    report = combine_replay_grade_receipts((scoring, same_backend, bundle_exact))
    prepublication_body = {
        "schema_version": 1,
        "contract_id": contract_id,
        "campaign_id": campaign_id,
        "prediction_id": prediction_id,
        "campaign_kind": "PRODUCTION_FULL_DATA",
        "qualification_scope": "FULL_DATA_CPU",
        "qualification_manifest_digest": "2" * 64,
        "qualification_fallback_digest": "3" * 64,
        "performance_profile_digest": performance_profile_digest,
        "declared_resource_profile": {
            "name": "competition-cpu",
            "preferred_backend": "lightgbm-cpu",
            "wall_clock_seconds": 100,
            "process_tree_rss_hard_cap_mb": 64,
            "candidate_disk_hard_cap_mb": 64,
        },
        "actual_trainer_backend": "organizer-numpy-fm",
        "actual_trainer_device": "cpu",
        "qualified_training_resources": {
            "wall_seconds": 10.0,
            "cpu_seconds": 9.0,
            "peak_rss_bytes": 2_000_000,
            "disk_bytes": 3_000_000,
            "device": "cpu",
        },
        "controller_resources": {
            "wall_seconds": 5.0,
            "cpu_seconds": 4.0,
            "peak_rss_bytes": 4_000_000,
            "disk_bytes": 5_000_000,
            "device": "cpu",
        },
        "controller_resource_scope": "PREPUBLICATION_SELF_EXCLUDING",
        "preferred_backend_qualified": False,
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
    }
    resource_receipt_id = canonical_json_sha256(prepublication_body)
    (tmp_path / EvidenceRole.RESOURCE_RECEIPTS.value).write_bytes(
        canonical_json_bytes(prepublication_body | {"receipt_id": resource_receipt_id}) + b"\n"
    )
    (tmp_path / EvidenceRole.CAMPAIGN_MANIFEST.value).write_bytes(
        canonical_json_bytes(
            {
                "production_admission": {
                    "performance": {"profile_digest": performance_profile_digest}
                }
            }
        )
    )
    (tmp_path / EvidenceRole.REPLAY_RECEIPT.value).write_bytes(
        canonical_json_bytes(
            {
                "scoring_exact_receipt": scoring.manifest(),
                "same_backend_receipt": same_backend.manifest(),
            }
        )
    )
    postpublication = PostpublicationResourceReceipt.from_measurement(
        contract_id=contract_id,
        campaign_id=campaign_id,
        prediction_id=prediction_id,
        prepublication_resource_receipt_id=resource_receipt_id,
        performance_profile_digest=performance_profile_digest,
        measurements={
            "wall_seconds": 20.0,
            "cpu_seconds": 18.0,
            "peak_rss_bytes": 6_000_000,
            "disk_bytes": 7_000_000,
            "device": "cpu",
        },
    )
    validation = BundleValidationResult(
        bundle_path=tmp_path,
        bundle_id=bundle_id,
        contract_id=contract_id,
        campaign_id=campaign_id,
        decision_id="4" * 64,
        resource_receipt_id=resource_receipt_id,
        preparation_id="5" * 64,
        projection_sha256="6" * 64,
        manifest_sha256="7" * 64,
        submission_sha256=submission_sha256,
        inventory_sha256=inventory_sha256,
        selected_prediction_id=prediction_id,
        file_count=16,
        total_size_bytes=1,
    )
    proof = {
        "schema_version": 1,
        "regeneration_evidence": regeneration.manifest(),
        "bundle_exact_receipt": bundle_exact.manifest(),
        "replay_grade_receipts": {
            ReplayGrade.SCORING_EXACT.value: scoring.manifest(),
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value: same_backend.manifest(),
            ReplayGrade.BUNDLE_EXACT.value: bundle_exact.manifest(),
        },
        "replay_grade_report": report.manifest(),
    }
    payload = {
        "bundle_exact_status": "VERIFIED",
        "prepublication_replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
        ],
        "required_replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
            ReplayGrade.BUNDLE_EXACT.value,
        ],
        "replay_grades": [
            ReplayGrade.SCORING_EXACT.value,
            ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
            ReplayGrade.BUNDLE_EXACT.value,
        ],
        "replay_grade": ReplayGrade.BUNDLE_EXACT.value,
        "resource_receipt_id": resource_receipt_id,
        "publication_proof": proof,
        "postpublication_resource_receipt": postpublication.manifest(),
    }

    lab_module._validate_authority_publication_proof(payload, validation)

    missing_same_backend = dict(payload)
    missing_proof = dict(proof)
    missing_receipts = dict(cast(dict[str, object], proof["replay_grade_receipts"]))
    del missing_receipts[ReplayGrade.EXPERIMENT_SAME_BACKEND.value]
    missing_proof["replay_grade_receipts"] = missing_receipts
    missing_same_backend["publication_proof"] = missing_proof
    with pytest.raises(LabError, match="receipt map"):
        lab_module._validate_authority_publication_proof(missing_same_backend, validation)

    over_cap = PostpublicationResourceReceipt.from_measurement(
        contract_id=contract_id,
        campaign_id=campaign_id,
        prediction_id=prediction_id,
        prepublication_resource_receipt_id=resource_receipt_id,
        performance_profile_digest=performance_profile_digest,
        measurements={
            "wall_seconds": 20.0,
            "cpu_seconds": 18.0,
            "peak_rss_bytes": 6_000_000,
            "disk_bytes": 65 * 1024 * 1024,
            "device": "cpu",
        },
    )
    over_cap_payload = dict(payload)
    over_cap_payload["postpublication_resource_receipt"] = over_cap.manifest()
    with pytest.raises(LabError, match="exceed declared caps"):
        lab_module._validate_authority_publication_proof(over_cap_payload, validation)


def test_organizer_evidence_is_cross_checked_before_it_is_retained() -> None:
    evidence = _organizer_evidence()

    manifest = controller._validate_organizer_evidence(
        evidence,
        expected_submission_sha256="3" * 64,
        expected_starter_manifest_sha256="2" * 64,
        expected_final_rows=2,
    )

    masked_data_view = manifest["masked_data_view"]
    assert isinstance(masked_data_view, dict)
    isolation = masked_data_view["final_outcome_isolation"]
    assert isinstance(isolation, dict)
    assert isolation["final_rows_masked"] == 2
    assert isolation["final_outcome_cells_replaced"] == 2 * len(OUTCOME_FIELDS)
    assert isolation["outcome_cells_scored"] == 0

    bad_digest = replace(
        evidence,
        masked_view=replace(evidence.masked_view, digest="f" * 64),
    )
    with pytest.raises(ProductionControllerError, match="masked-view digest"):
        controller._validate_organizer_evidence(
            bad_digest,
            expected_submission_sha256="3" * 64,
            expected_starter_manifest_sha256="2" * 64,
            expected_final_rows=2,
        )
    with pytest.raises(ProductionControllerError, match="outcome masking counts"):
        controller._validate_organizer_evidence(
            evidence,
            expected_submission_sha256="3" * 64,
            expected_starter_manifest_sha256="2" * 64,
            expected_final_rows=3,
        )


def test_controller_resource_measurement_has_two_truthful_scopes() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "_controller_prepublication_resources" in source
    assert "_controller_postpublication_resources" in source
    assert "PostpublicationResourceReceipt.from_measurement" in source
    assert "pre-publication, self-excluding observation" in source
    assert "complete_run_resources_claimed" in source
    assert '"controller_resource_scope": "PREPUBLICATION_SELF_EXCLUDING"' in source
    assert "THROUGH_BUNDLE_PUBLICATION_EXCLUDING_TERMINAL_COMMIT" in source
    assert '"preferred_backend_qualified": False' in source


def test_controller_monitor_uses_runner_simultaneous_process_tree_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _SampledTree:
        inspection_failed = False

        def __init__(self, _root: object, _process_group_id: int) -> None:
            pass

        def sample(self) -> tuple[int, int]:
            # Runner's tracker has already summed root + both live descendants in one observation.
            return 101 + 202 + 303, 3

    monkeypatch.setattr(controller, "_ProcessTracker", _SampledTree)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monitor = controller._ControllerProcessTreeMonitor(run_dir)
    monitor.start()
    try:
        measurement = monitor.measurement()
    finally:
        monitor.stop()

    assert measurement == {"peak_rss_bytes": 606, "disk_bytes": 0}
    assert "ru_maxrss" not in SOURCE.read_text(encoding="utf-8")


def test_deterministic_registration_resumes_after_family_boundary(tmp_path: Path) -> None:
    class _RecoveringRepository:
        def __init__(self) -> None:
            self.records: dict[RecordKind, DurableRecord] = {}
            self.fail_rank_once = True

        def register(self, record: DurableRecord) -> bool:
            existing = self.records.get(record.kind)
            if existing is not None:
                if existing != record:
                    raise AssertionError("same identity changed across recovery")
                return False
            if record.kind is RecordKind.RANK_GRAPH and self.fail_rank_once:
                self.fail_rank_once = False
                raise RuntimeError("synthetic crash after family registration")
            self.records[record.kind] = record
            return True

        def inspect(self, *, campaign_id: str) -> dict[str, object]:
            del campaign_id
            identities = {
                RecordKind.FAMILY: ("families", "family_id"),
                RecordKind.RANK_GRAPH: ("rank_graphs", "rank_graph_id"),
                RecordKind.ARTIFACT: ("artifacts", "artifact_id"),
                RecordKind.PREDICTION: ("predictions", "prediction_id"),
            }
            entities: dict[str, object] = {
                table: [] for table, _identity_field in identities.values()
            }
            for kind, record in self.records.items():
                table, identity_field = identities[kind]
                cast(list[object], entities[table]).append({identity_field: record.record_id})
            return {"entities": entities}

    repository = _RecoveringRepository()
    request = cast(
        ProductionCPUFallbackRequest,
        SimpleNamespace(
            repository=cast(StateRepository, cast(object, repository)),
            admission=SimpleNamespace(
                qualification_manifest_digest="1" * 64,
                fallback_manifest_digest="2" * 64,
                final_rows=3,
            ),
            campaign_id="3" * 64,
            contract_id="4" * 64,
        ),
    )
    submission = tmp_path / "submission.csv"
    submission.write_bytes(b"user_id,video_id,score\n")

    def register() -> tuple[str, str, str]:
        return controller._register_fallback_prediction(
            request=request,
            final_submission=submission,
            final_prediction_digest="5" * 64,
            final_alignment_digest="6" * 64,
        )

    with pytest.raises(RuntimeError, match="synthetic crash"):
        register()
    assert set(repository.records) == {RecordKind.FAMILY}

    first = register()
    second = register()

    assert first == second
    assert set(repository.records) == {
        RecordKind.FAMILY,
        RecordKind.RANK_GRAPH,
        RecordKind.ARTIFACT,
        RecordKind.PREDICTION,
    }


def test_running_recovery_rejects_ambiguous_deterministic_identity() -> None:
    class _AmbiguousRepository:
        def inspect(self, *, campaign_id: str) -> dict[str, object]:
            del campaign_id
            return {
                "entities": {
                    "families": [{"family_id": "1" * 64}, {"family_id": "2" * 64}],
                    "rank_graphs": [{"rank_graph_id": "3" * 64}],
                    "artifacts": [{"artifact_id": "4" * 64}],
                    "predictions": [{"prediction_id": "5" * 64}],
                }
            }

    with pytest.raises(ProductionControllerError, match="ambiguous families"):
        controller._assert_exact_registered_fallback(
            cast(StateRepository, cast(object, _AmbiguousRepository())),
            campaign_id="0" * 64,
            family_id="1" * 64,
            rank_graph_id="3" * 64,
            artifact_id="4" * 64,
            prediction_id="5" * 64,
        )
