"""StateRepository-only controller for the admitted full-data CPU fallback.

This module performs no research search, provider operation, protected-query reservation, or
final-period outcome access.  It replays the already-qualified official-FM seed-4 fallback,
publishes one exact flat evidence bundle, and atomically completes the existing campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Self, cast

import numpy as np
import numpy.typing as npt
import psutil  # type: ignore[import-untyped]

from kuairand_agent.campaign.provenance import capture_environment_identity
from kuairand_agent.contract import CONTRACT_ID, CONTRACT_MANIFEST, sha256_file
from kuairand_agent.data.canonical import OUTCOME_FIELDS, CanonicalAlignment, CanonicalInputs
from kuairand_agent.data.capabilities import CandidateInputs, DataPhase, build_candidate_inputs
from kuairand_agent.data.fields import STANDARD_LATE_MEMBER, VIDEO_BASIC_MEMBER, FieldKey
from kuairand_agent.domain.decisions import ReplayGrade, ScientificDisposition
from kuairand_agent.domain.identity import (
    BundleId,
    DecisionId,
    PredictionId,
    canonical_json_bytes,
    canonical_json_sha256,
)
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.execution.runner import ProcessIdentity, _ProcessTracker
from kuairand_agent.finalization.backends import build_replay_backend
from kuairand_agent.finalization.bundle import (
    REQUIRED_EVIDENCE_ROLES,
    BundleFinalizationRequest,
    BundleFinalizer,
    EvidenceRole,
    FrozenFileReceipt,
    TerminalProjectionBinding,
)
from kuairand_agent.finalization.organizer_check import (
    OrganizerCheckEvidence,
    check_final_submission,
)
from kuairand_agent.finalization.recipe import OfficialFMMemberRecipe, OfficialFMReplayRecipe
from kuairand_agent.finalization.replay import (
    CleanReplayEvidence,
    CleanReplayRequest,
    CleanReplayResult,
    FrozenReplayIdentity,
    ReplayArtifacts,
    ReplayBackend,
    ReplayCapabilities,
    ReplayEquality,
    run_clean_replay,
)
from kuairand_agent.finalization.replay_grades import (
    ReplayGradeReceipt,
    derive_clean_replay_grade_receipts,
)
from kuairand_agent.production.admission import ProductionAdmission
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity
from kuairand_agent.scoring.submission import AlignmentRow, prediction_digest
from kuairand_agent.state.repository import (
    DurableRecord,
    PostpublicationResourceReceipt,
    PublishedBundleReceipt,
    RecordKind,
    StateRepository,
    TerminalPreparation,
)

_CAMPAIGN_KIND: Final = "PRODUCTION_FULL_DATA"
_QUALIFICATION_SCOPE: Final = "FULL_DATA_CPU"
_SUBMISSION_DISPOSITION: Final = "FALLBACK_RETAINED"
_SCIENTIFIC_DISPOSITION: Final = ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE.value
_POSTPUBLICATION_RESOURCE_SCOPE: Final = "THROUGH_BUNDLE_PUBLICATION_EXCLUDING_TERMINAL_COMMIT"
_PREPUBLICATION_REPLAY_GRADES: Final = (
    ReplayGrade.SCORING_EXACT.value,
    ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
)
_REPLAY_GRADES: Final = (*_PREPUBLICATION_REPLAY_GRADES, ReplayGrade.BUNDLE_EXACT.value)
_ORGANIZER_COMMAND: Final = (
    "python",
    "-B",
    "submit.py",
    "submission.csv",
    "--data_dir",
    "<private-masked-data-view>",
    "--split",
    "test",
    "--check",
)
_ORGANIZER_ZERO_OUTCOME_COUNTERS: Final = (
    "outcome_cells_sliced",
    "outcome_cells_decoded",
    "outcome_cells_converted",
    "outcome_cells_validated",
    "outcome_cells_logged",
    "outcome_cells_hashed",
    "outcome_cells_scored",
)


class ProductionControllerError(RuntimeError):
    """The admitted fallback could not be replayed and finalized exactly."""


@dataclass(frozen=True, slots=True)
class ProductionCPUFallbackRequest:
    """Inputs for one already-created ``RUNNING`` production campaign."""

    repository: StateRepository
    admission: ProductionAdmission
    campaign_id: str
    contract_id: str
    campaign_revision: int
    run_dir: Path
    repository_root: Path
    startup_receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProductionCPUFallbackResult:
    """Terminal authority snapshot and exact published-bundle identities."""

    snapshot: Mapping[str, object]
    bundle_path: Path
    bundle_id: str
    manifest_sha256: str
    inventory_sha256: str
    submission_sha256: str
    resource_receipt_id: str


class ProductionCPUFallbackController:
    """Complete the bounded provider-free official-FM fallback campaign."""

    def run(self, request: ProductionCPUFallbackRequest) -> ProductionCPUFallbackResult:
        _validate_request(request)
        request.run_dir.mkdir(parents=True, exist_ok=True)
        _validate_run_directory(request.run_dir)
        started_wall = time.monotonic()
        started_cpu = _process_tree_cpu_seconds()
        monitor = _ControllerProcessTreeMonitor(request.run_dir)
        monitor.start()
        try:
            return self._run_monitored(
                request,
                monitor=monitor,
                started_wall=started_wall,
                started_cpu=started_cpu,
            )
        finally:
            monitor.stop()

    def _run_monitored(
        self,
        request: ProductionCPUFallbackRequest,
        *,
        monitor: _ControllerProcessTreeMonitor,
        started_wall: float,
        started_cpu: float,
    ) -> ProductionCPUFallbackResult:
        repository = request.repository
        admission = request.admission
        runtime = admission.runtime
        fallback = runtime.fallback
        dataset = runtime.dataset

        store = ArtifactStore(request.run_dir / "artifact-store")
        validation_inputs = _candidate_inputs(DataPhase.OUTER_VALID, dataset.valid.inputs)
        final_inputs = _candidate_inputs(DataPhase.FINAL, dataset.final.inputs)
        validation_alignment = _alignment_rows(dataset.valid.alignment)
        final_alignment = _alignment_rows(dataset.final.alignment)
        capabilities = ReplayCapabilities(
            data_sha256=dataset.digest,
            validation_inputs=validation_inputs,
            final_inputs=final_inputs,
            validation_alignment=validation_alignment,
            final_alignment=final_alignment,
        )

        source = store.put_directory(runtime.starter.root, kind=ArtifactKind.SOURCE)
        encoding = store.put_file(fallback.encoding_path, kind=ArtifactKind.INPUT)
        checkpoint = store.put_file(fallback.checkpoint_path, kind=ArtifactKind.CHECKPOINT)
        reference = store.put_file(
            fallback.validation_predictions_path,
            kind=ArtifactKind.PREDICTION,
        )
        _assert_artifact_closure(
            source_sha256=source.sha256,
            starter_manifest_sha256=runtime.starter.manifest_sha256,
            encoding_sha256=encoding.sha256,
            expected_encoding_sha256=fallback.encoding_file_sha256,
            checkpoint_sha256=checkpoint.sha256,
            expected_checkpoint_sha256=fallback.checkpoint_file_sha256,
            reference_sha256=reference.sha256,
            expected_reference_sha256=fallback.validation_predictions_file_sha256,
        )
        member = OfficialFMMemberRecipe(
            checkpoint_sha256=fallback.checkpoint_file_sha256,
            checkpoint_digest=fallback.checkpoint_digest,
            encoding_sha256=fallback.encoding_file_sha256,
            encoding_digest=fallback.encoding_digest,
            config_digest=fallback.config_digest,
            starter_manifest_sha256=runtime.qualification.starter_manifest_digest,
            seed=fallback.seed,
        )
        recipe = OfficialFMReplayRecipe(
            source_artifact_sha256=source.sha256,
            feature_artifact_sha256=encoding.sha256,
            checkpoint_artifact_sha256=checkpoint.sha256,
            data_sha256=dataset.digest,
            validation_inputs_digest=validation_inputs.digest,
            final_inputs_digest=final_inputs.digest,
            fm_member=member,
        )
        recipe_ref = store.put_bytes(recipe.canonical_bytes, kind=ArtifactKind.INPUT)
        environment_identity = capture_environment_identity(request.repository_root)
        environment = environment_identity.manifest()
        reference_values = _load_prediction_vector(
            fallback.validation_predictions_path,
            expected_rows=dataset.valid.row_count,
        )
        if prediction_digest(reference_values) != fallback.validation_prediction_digest:
            raise ProductionControllerError(
                "qualified fallback validation prediction semantics changed"
            )
        scorer = _qualification_reverification_scorer(request)
        original_metrics = scorer(reference_values)
        _assert_qualified_metrics(original_metrics, fallback.metrics.manifest())

        identity = FrozenReplayIdentity(
            source_sha256=source.sha256,
            config_sha256=recipe_ref.sha256,
            features_sha256=encoding.sha256,
            checkpoint_sha256=checkpoint.sha256,
            validation_prediction_artifact_sha256=reference.sha256,
            validation_prediction_digest=fallback.validation_prediction_digest,
            data_sha256=dataset.digest,
            environment_sha256=environment_identity.digest,
        )
        replay_request = CleanReplayRequest(
            candidate_id="official-fm-seed-4-qualified-fallback",
            output_dir=request.run_dir / "replay" / "official-fm-seed-4",
            identity=identity,
            artifacts=ReplayArtifacts(
                source=source,
                config=recipe_ref,
                features=encoding,
                checkpoint=checkpoint,
                validation_predictions=reference,
            ),
            environment=environment,
            equality=ReplayEquality.EXACT,
            training_replay="immutable qualified official-FM seed-4 checkpoint replay",
        )
        replay = _run_or_reuse_clean_replay(
            replay_request,
            artifact_store=store,
            capabilities=capabilities,
            backend=build_replay_backend(recipe),
            # This is a finalization-only re-verification of the imported qualification vector.
            # It is not a campaign protected query, is not recorded as one, and cannot influence
            # selection because the fallback is already frozen before this controller starts.
            protected_metric_evaluator=scorer,
        )
        if replay.evidence.validation.replay_prediction_digest != (
            fallback.validation_prediction_digest
        ):
            raise ProductionControllerError("clean replay changed the qualified validation vector")
        _assert_qualified_metrics(replay.evidence.validation.metrics, fallback.metrics.manifest())
        if replay.evidence.final.prediction_digest != fallback.final_prediction_digest:
            raise ProductionControllerError("clean replay changed the qualified final vector")
        if sha256_file(replay.final_submission) != fallback.final_submission_file_sha256:
            raise ProductionControllerError("clean replay changed the qualified final submission")

        organizer_scratch = request.run_dir / "organizer-check"
        organizer_scratch.mkdir(mode=0o700, exist_ok=True)
        organizer = check_final_submission(
            replay.final_submission,
            data_dir=runtime.audit.data_dir,
            starter_dir=runtime.starter.root,
            scratch_dir=organizer_scratch,
        )
        if organizer.submission_sha256 != fallback.final_submission_file_sha256:
            raise ProductionControllerError("organizer checker observed different submission bytes")

        _rank_graph_id, prediction_id, _artifact_id = _register_fallback_prediction(
            request=request,
            final_submission=replay.final_submission,
            final_prediction_digest=fallback.final_prediction_digest,
            final_alignment_digest=dataset.final.alignment.digest,
        )
        scoring_receipts = derive_clean_replay_grade_receipts(
            contract_id=request.contract_id,
            prediction_id=prediction_id,
            evidence=replay.evidence,
        )
        scoring_exact_receipt, same_backend_receipt = _select_exact_replay_receipts(
            receipts=scoring_receipts,
            expected_contract_id=request.contract_id,
            expected_prediction_id=prediction_id,
        )
        exact_replay_evidence = _exact_replay_evidence_payload(
            evidence=replay.evidence,
            scoring_receipt=scoring_exact_receipt,
            same_backend_receipt=same_backend_receipt,
        )
        organizer_manifest = _validate_organizer_evidence(
            organizer,
            expected_submission_sha256=fallback.final_submission_file_sha256,
            expected_starter_manifest_sha256=runtime.starter.manifest_sha256,
            expected_final_rows=dataset.final.row_count,
        )

        controller_prepublication_resources = _controller_prepublication_resources(
            request.run_dir,
            started_wall=started_wall,
            started_cpu=started_cpu,
            monitor=monitor,
        )
        resource_body, resource_receipt_id = _register_or_reuse_resource_receipt(
            request=request,
            prediction_id=prediction_id,
            controller_resources=controller_prepublication_resources,
        )

        decision_id = DecisionId.derive(
            policy_sha256=canonical_json_sha256(
                {
                    "schema_version": 1,
                    "selection": _SUBMISSION_DISPOSITION,
                    "qualification_scope": _QUALIFICATION_SCOPE,
                    "generated_candidates": 0,
                    "protected_queries": 0,
                    "provider_operations": 0,
                }
            ),
            evidence_ids={"qualified_fallback_prediction": PredictionId(prediction_id)},
        )
        organizer_digest = canonical_json_sha256(organizer_manifest)
        replay_payload = {
            "schema_version": 3,
            "contract_id": request.contract_id,
            "campaign_id": request.campaign_id,
            "prediction_id": prediction_id,
            "qualification_manifest_digest": admission.qualification_manifest_digest,
            "qualification_fallback_digest": admission.fallback_manifest_digest,
            "original_prediction_sha256": fallback.final_prediction_digest,
            "replay_prediction_sha256": replay.evidence.final.prediction_digest,
            "row_alignment_sha256": dataset.final.alignment.digest,
            "submission_sha256": organizer.submission_sha256,
            "organizer_check_sha256": organizer_digest,
            "organizer_check": organizer_manifest,
            **exact_replay_evidence,
            "achieved_replay_grades": list(_PREPUBLICATION_REPLAY_GRADES),
            "required_terminal_replay_grades": list(_REPLAY_GRADES),
            "qualification_scope": _QUALIFICATION_SCOPE,
            "official_fm_qualified": True,
            "full_data_qualified": True,
            "final_period_outcomes_accessed": False,
        }
        decision_payload = {
            "schema_version": 1,
            "contract_id": request.contract_id,
            "campaign_id": request.campaign_id,
            "decision_id": decision_id.value,
            "selected_prediction_id": prediction_id,
            "fallback_prediction_id": prediction_id,
            "submission_disposition": _SUBMISSION_DISPOSITION,
            "scientific_disposition": _SCIENTIFIC_DISPOSITION,
            "qualification_scope": _QUALIFICATION_SCOPE,
            "official_fm_qualified": True,
            "full_data_qualified": True,
            "generated_candidates_evaluated": 0,
            "protected_query_count": 0,
            "provider_operation_count": 0,
            "final_period_outcomes_accessed": False,
        }
        bundle_claims = {
            "schema_version": 2,
            "resource_receipt_id": resource_receipt_id,
            "prepublication_replay_grades": list(_PREPUBLICATION_REPLAY_GRADES),
            "required_replay_grades": list(_REPLAY_GRADES),
            "bundle_exact_status": "PENDING_PUBLICATION_PROOF",
            "submission_disposition": _SUBMISSION_DISPOSITION,
            "scientific_disposition": _SCIENTIFIC_DISPOSITION,
            "campaign_kind": _CAMPAIGN_KIND,
            "qualification_scope": _QUALIFICATION_SCOPE,
            "protected_query_count": 0,
            "provider_operation_count": 0,
            "exact_metrics": None,
            "official_fm_qualified": True,
            "full_data_qualified": True,
            "final_period_outcomes_accessed": False,
            "qualification_manifest_digest": admission.qualification_manifest_digest,
        }
        replay_id = canonical_json_sha256(replay_payload)
        prepared = repository.prepare_terminal_projection(
            campaign_id=request.campaign_id,
            contract_id=request.contract_id,
            expected_state="RUNNING",
            expected_revision=request.campaign_revision,
            preparation=TerminalPreparation(
                decision_id=decision_id,
                replay_id=replay_id,
                selected_prediction_id=prediction_id,
                fallback_prediction_id=prediction_id,
                terminal_state="COMPLETED",
                decision_payload=decision_payload,
                replay_payload=replay_payload,
                bundle_claims=bundle_claims,
                scoring_exact_receipt=scoring_exact_receipt,
                same_backend_receipt=same_backend_receipt,
            ),
        )
        terminal_binding = TerminalProjectionBinding(
            preparation_id=prepared.preparation_id,
            projection_sha256=prepared.projection_sha256,
            campaign_revision=prepared.source.campaign_revision,
            last_event_seq=prepared.source.last_event_seq,
            schema_version=prepared.projection_schema_version,
            redaction_policy_version=prepared.redaction_policy_version,
        )
        evidence_root = request.run_dir / "evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        repository.materialize_prepared_terminal_projection(
            preparation_id=prepared.preparation_id,
            snapshot_destination=evidence_root / EvidenceRole.CAMPAIGN_STATE_SNAPSHOT.value,
            event_export_destination=evidence_root / EvidenceRole.EVENT_EXPORT.value,
        )
        _write_evidence(
            request=request,
            evidence_root=evidence_root,
            terminal_binding=terminal_binding,
            decision_id=decision_id.value,
            prediction_id=prediction_id,
            replay_payload=replay_payload,
            exact_replay_evidence=exact_replay_evidence,
            resource_body=resource_body,
            resource_receipt_id=resource_receipt_id,
            organizer_manifest=organizer_manifest,
            final_submission=replay.final_submission,
        )
        finalizer_result = BundleFinalizer().finalize(
            BundleFinalizationRequest(
                destination=request.run_dir / "final" / "submission-bundle",
                contract_id=request.contract_id,
                campaign_id=request.campaign_id,
                selected_prediction_id=prediction_id,
                terminal_projection=terminal_binding,
                receipts=tuple(
                    FrozenFileReceipt.capture(role, evidence_root / role.value)
                    for role in REQUIRED_EVIDENCE_ROLES
                ),
            )
        )
        if finalizer_result.replay_grade.grade is not ReplayGrade.BUNDLE_EXACT:
            raise ProductionControllerError("bundle finalizer did not derive BUNDLE_EXACT")
        postpublication_resources = _controller_postpublication_resources(
            request.run_dir,
            started_wall=started_wall,
            started_cpu=started_cpu,
            monitor=monitor,
        )
        postpublication_resource_receipt = PostpublicationResourceReceipt.from_measurement(
            contract_id=request.contract_id,
            campaign_id=request.campaign_id,
            prediction_id=prediction_id,
            prepublication_resource_receipt_id=resource_receipt_id,
            performance_profile_digest=admission.performance_profile_digest,
            measurements=postpublication_resources,
        )
        if (
            postpublication_resource_receipt.manifest().get("scope")
            != _POSTPUBLICATION_RESOURCE_SCOPE
        ):
            raise ProductionControllerError("postpublication receipt overstates its measured scope")
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=PublishedBundleReceipt(
                root=finalizer_result.root,
                bundle_id=BundleId(finalizer_result.bundle_id),
                manifest_sha256=finalizer_result.manifest_sha256,
                inventory_sha256=finalizer_result.inventory_sha256,
                submission_sha256=finalizer_result.submission_sha256,
                file_count=finalizer_result.file_count,
                total_size_bytes=finalizer_result.total_size_bytes,
                regeneration_evidence=finalizer_result.regeneration_evidence,
                bundle_exact_receipt=finalizer_result.replay_grade,
                postpublication_resource_receipt=postpublication_resource_receipt,
            ),
        )
        snapshot = repository.inspect(campaign_id=request.campaign_id)
        return ProductionCPUFallbackResult(
            snapshot=snapshot,
            bundle_path=finalizer_result.root,
            bundle_id=finalizer_result.bundle_id,
            manifest_sha256=finalizer_result.manifest_sha256,
            inventory_sha256=finalizer_result.inventory_sha256,
            submission_sha256=finalizer_result.submission_sha256,
            resource_receipt_id=resource_receipt_id,
        )


def _validate_request(request: ProductionCPUFallbackRequest) -> None:
    if not isinstance(request, ProductionCPUFallbackRequest):
        raise ProductionControllerError("request must be ProductionCPUFallbackRequest")
    if not isinstance(request.repository, StateRepository):
        raise ProductionControllerError("repository must be StateRepository")
    if not isinstance(request.admission, ProductionAdmission):
        raise ProductionControllerError("admission must be ProductionAdmission")
    if (
        request.contract_id != CONTRACT_ID.value
        or request.contract_id != request.admission.contract_id
    ):
        raise ProductionControllerError("production campaign differs from admitted ContractId")
    for name in ("campaign_id", "contract_id"):
        value = getattr(request, name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ProductionControllerError(f"{name} must be a lowercase SHA-256 digest")
    if type(request.campaign_revision) is not int or request.campaign_revision < 0:
        raise ProductionControllerError("campaign_revision must be non-negative")
    for name in ("run_dir", "repository_root"):
        value = getattr(request, name)
        if not isinstance(value, Path):
            raise ProductionControllerError(f"{name} must be pathlib.Path")
    try:
        root = request.repository_root.resolve(strict=True)
    except OSError as exc:
        raise ProductionControllerError("repository_root is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise ProductionControllerError("repository_root must be a real directory")
    if not isinstance(request.startup_receipt, Mapping):
        raise ProductionControllerError("startup_receipt must be a mapping")
    startup = dict(request.startup_receipt)
    if (
        startup.get("receipt_id") != request.admission.startup_receipt_id
        or startup.get("contract_id") != request.contract_id
        or startup.get("verified") is not True
        or startup.get("state_writes_started") is not False
    ):
        raise ProductionControllerError("startup receipt differs from production admission")
    snapshot = request.repository.inspect(campaign_id=request.campaign_id)
    campaign = _mapping(snapshot.get("campaign"), "campaign")
    if (
        campaign.get("contract_id") != request.contract_id
        or campaign.get("state") != "RUNNING"
        or campaign.get("revision") != request.campaign_revision
        or campaign.get("terminal") is not False
    ):
        raise ProductionControllerError("campaign is not the admitted RUNNING revision")
    config = _mapping(campaign.get("config"), "campaign config")
    expected_digests = {
        "qualification_manifest_digest": request.admission.qualification_manifest_digest,
        "performance_profile_digest": request.admission.performance_profile_digest,
    }
    if any(config.get(name) != value for name, value in expected_digests.items()) or (
        canonical_json_bytes(config.get("resource_profile"))
        != canonical_json_bytes(request.admission.runtime.resource_profile.manifest())
    ):
        raise ProductionControllerError("campaign config differs from admitted evidence")
    entities = _mapping(snapshot.get("entities"), "entities")
    # Deterministic fallback identity records and its prepublication resource receipt are replay
    # safe.  They are validated against their exact expected IDs/content later in the controller.
    # Every other execution or terminal record remains forbidden on this zero-preparation path.
    for name in (
        "experiments",
        "trials",
        "attempts",
        "inner_evaluations",
        "protected_evaluations",
        "promotion_decisions",
        "selection_decisions",
        "replays",
        "bundles",
        "provider_operations",
        "failures",
        "bundle_publications",
    ):
        values = entities.get(name, [])
        if not isinstance(values, list) or values:
            raise ProductionControllerError(
                "fresh production campaign contains prior execution state"
            )
    preparations = entities.get("terminal_preparations", [])
    if not isinstance(preparations, list) or len(preparations) > 1:
        raise ProductionControllerError(
            "RUNNING recovery has ambiguous terminal preparation authority"
        )
    reservations = snapshot.get("contract_protected_query_reservations")
    if not isinstance(reservations, list):
        raise ProductionControllerError("authority protected-query projection is malformed")
    if any(
        isinstance(item, Mapping) and item.get("campaign_id") == request.campaign_id
        for item in reservations
    ):
        raise ProductionControllerError("production campaign already contains a protected query")


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProductionControllerError(f"{location} must be a mapping")
    return cast(Mapping[str, object], value)


def _validate_run_directory(run_dir: Path) -> None:
    try:
        metadata = run_dir.lstat()
    except OSError as exc:
        raise ProductionControllerError("controller run directory is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionControllerError("controller run directory must be a real directory")


def _candidate_inputs(phase: DataPhase, inputs: CanonicalInputs) -> CandidateInputs:
    return build_candidate_inputs(
        phase,
        {
            FieldKey(STANDARD_LATE_MEMBER, "user_id"): inputs.user_id,
            FieldKey(STANDARD_LATE_MEMBER, "video_id"): inputs.video_id,
            FieldKey(VIDEO_BASIC_MEMBER, "author_id"): inputs.author_id,
            FieldKey(STANDARD_LATE_MEMBER, "tab"): inputs.tab,
            FieldKey(STANDARD_LATE_MEMBER, "duration_ms"): inputs.duration_ms,
        },
    )


def _alignment_rows(alignment: CanonicalAlignment) -> tuple[AlignmentRow, ...]:
    return tuple(
        AlignmentRow(row_id, user_id, video_id)
        for row_id, user_id, video_id in zip(
            alignment.row_id,
            alignment.user_id,
            alignment.video_id,
            strict=True,
        )
    )


def _assert_artifact_closure(
    *,
    source_sha256: str,
    starter_manifest_sha256: str,
    encoding_sha256: str,
    expected_encoding_sha256: str,
    checkpoint_sha256: str,
    expected_checkpoint_sha256: str,
    reference_sha256: str,
    expected_reference_sha256: str,
) -> None:
    for name, value in (
        ("source artifact", source_sha256),
        ("starter manifest", starter_manifest_sha256),
    ):
        if len(value) != 64:
            raise ProductionControllerError(f"{name} identity is malformed")
    if (
        encoding_sha256 != expected_encoding_sha256
        or checkpoint_sha256 != expected_checkpoint_sha256
        or reference_sha256 != expected_reference_sha256
    ):
        raise ProductionControllerError("qualified fallback artifacts changed after admission")


def _load_prediction_vector(path: Path, *, expected_rows: int) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            loaded = np.load(stream, allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ProductionControllerError("qualified validation vector is unreadable") from exc
    if (
        not isinstance(loaded, np.ndarray)
        or loaded.shape != (expected_rows,)
        or loaded.dtype.kind not in "iuf"
    ):
        raise ProductionControllerError("qualified validation vector shape or dtype changed")
    values = np.ascontiguousarray(loaded, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ProductionControllerError("qualified validation vector contains non-finite values")
    values.setflags(write=False)
    return values


def _run_or_reuse_clean_replay(
    request: CleanReplayRequest,
    *,
    artifact_store: ArtifactStore,
    capabilities: ReplayCapabilities,
    backend: ReplayBackend,
    protected_metric_evaluator: Callable[[npt.NDArray[np.float64]], Mapping[str, object]],
) -> CleanReplayResult:
    """Publish once or prove that an earlier atomic replay publication is byte-identical."""

    destination = request.output_dir
    if not os.path.lexists(destination):
        return run_clean_replay(
            request,
            artifact_store=artifact_store,
            capabilities=capabilities,
            backend=backend,
            protected_metric_evaluator=protected_metric_evaluator,
        )
    try:
        metadata = destination.lstat()
    except OSError as exc:
        raise ProductionControllerError("existing clean replay output is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionControllerError("existing clean replay output is not a real directory")

    recovery_parent = Path(
        tempfile.mkdtemp(prefix=".official-fm-recovery-", dir=destination.parent)
    )
    try:
        regenerated = run_clean_replay(
            replace(request, output_dir=recovery_parent / "replay"),
            artifact_store=artifact_store,
            capabilities=capabilities,
            backend=backend,
            protected_metric_evaluator=protected_metric_evaluator,
        )
        if _directory_inventory(destination) != _directory_inventory(regenerated.root):
            raise ProductionControllerError(
                "existing clean replay output differs from exact regeneration"
            )
        root = destination.resolve(strict=True)
        return CleanReplayResult(
            root=root,
            evidence=regenerated.evidence,
            evidence_path=root / "replay" / "evidence.json",
            final_submission=root / "submission.csv",
            public_validation_submission=root / "validation-evidence" / "public-validation.csv",
            source_dir=root / "source",
            config_dir=root / "config",
            model_dir=root / "model",
            preprocessing_dir=root / "preprocessing",
            validation_evidence_dir=root / "validation-evidence",
            replay_dir=root / "replay",
            environment_path=root / "environment.json",
        )
    finally:
        shutil.rmtree(recovery_parent, ignore_errors=True)


def _directory_inventory(root: Path) -> tuple[tuple[str, str, int], ...]:
    """Hash one immutable replay tree while rejecting links and non-regular members."""

    entries: list[tuple[str, str, int]] = []
    for candidate in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()):
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise ProductionControllerError(
                "clean replay tree changed during verification"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProductionControllerError("clean replay tree contains a symlink")
        relative = candidate.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            entries.append((relative + "/", "directory", 0))
        elif stat.S_ISREG(metadata.st_mode):
            entries.append((relative, sha256_file(candidate), metadata.st_size))
        else:
            raise ProductionControllerError("clean replay tree contains a special file")
    return tuple(entries)


def _qualification_reverification_scorer(
    request: ProductionCPUFallbackRequest,
) -> Callable[[npt.NDArray[np.float64]], Mapping[str, object]]:
    """Close public-validation labels inside finalization-only qualification re-verification."""

    runtime = request.admission.runtime
    dataset = runtime.dataset
    validation_inputs = _candidate_inputs(DataPhase.OUTER_VALID, dataset.valid.inputs)
    split = SplitIdentity(
        name="outer_valid_qualification_reverification",
        token=validation_inputs.digest,
        expected_count=dataset.valid.row_count,
    )
    alignment = Alignment.from_ids(
        split=split,
        row_ids=dataset.valid.alignment.row_id,
        user_ids=dataset.valid.alignment.user_id,
        video_ids=dataset.valid.alignment.video_id,
    )
    scorer = ProtectedScorer(starter_dir=runtime.starter.root, trusted_alignment=alignment)
    labels = dataset.valid.targets.reveal_for_scorer()

    def score(scores: npt.NDArray[np.float64]) -> Mapping[str, object]:
        result = scorer.score_with_encoded_labels(
            alignment=alignment,
            split=split,
            labels=labels,
            scores=scores,
            expected_count=dataset.valid.row_count,
        )
        return result.as_dict()

    return score


def _assert_qualified_metrics(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    for name in ("GAUC", "nDCG@5", "primary"):
        left = observed.get(name)
        right = expected.get(name)
        if (
            isinstance(left, bool)
            or not isinstance(left, (int, float, np.number))
            or isinstance(right, bool)
            or not isinstance(right, (int, float, np.number))
            or float(left) != float(right)
        ):
            raise ProductionControllerError(
                "qualification re-verification metrics differ from qualified seed 4"
            )


def _select_exact_replay_receipts(
    *,
    receipts: Sequence[ReplayGradeReceipt],
    expected_contract_id: str,
    expected_prediction_id: str,
) -> tuple[ReplayGradeReceipt, ReplayGradeReceipt]:
    """Select the exact scoring and same-backend conclusions from one clean replay."""

    if any(not isinstance(receipt, ReplayGradeReceipt) for receipt in receipts):
        raise ProductionControllerError("clean replay returned malformed grade receipts")
    by_grade = {
        grade: tuple(receipt for receipt in receipts if receipt.grade is grade)
        for grade in (
            ReplayGrade.SCORING_EXACT,
            ReplayGrade.EXPERIMENT_SAME_BACKEND,
        )
    }
    if any(len(matches) != 1 for matches in by_grade.values()):
        raise ProductionControllerError(
            "clean replay must derive exactly one SCORING_EXACT and same-backend receipt"
        )
    selected = (
        by_grade[ReplayGrade.SCORING_EXACT][0],
        by_grade[ReplayGrade.EXPERIMENT_SAME_BACKEND][0],
    )
    if any(
        receipt.contract_id != expected_contract_id
        or receipt.prediction_id != expected_prediction_id
        for receipt in selected
    ):
        raise ProductionControllerError("exact replay receipt is outside campaign lineage")
    clean_replay_digests = {receipt.evidence.get("clean_replay_sha256") for receipt in selected}
    if len(clean_replay_digests) != 1:
        raise ProductionControllerError(
            "exact replay receipts do not authenticate the same clean replay"
        )
    return selected


def _select_scoring_exact_receipt(
    *,
    receipts: Sequence[ReplayGradeReceipt],
    expected_contract_id: str,
    expected_prediction_id: str,
) -> ReplayGradeReceipt:
    """Compatibility helper for callers that need only the scoring conclusion."""

    scoring, _same_backend = _select_exact_replay_receipts(
        receipts=receipts,
        expected_contract_id=expected_contract_id,
        expected_prediction_id=expected_prediction_id,
    )
    return scoring


def _exact_replay_evidence_payload(
    *,
    evidence: CleanReplayEvidence,
    scoring_receipt: ReplayGradeReceipt,
    same_backend_receipt: ReplayGradeReceipt,
) -> dict[str, object]:
    """Preserve both exact conclusions authenticated by the same clean replay."""

    if not isinstance(evidence, CleanReplayEvidence):
        raise ProductionControllerError("exact replay evidence must be CleanReplayEvidence")
    if (
        not isinstance(scoring_receipt, ReplayGradeReceipt)
        or scoring_receipt.grade is not ReplayGrade.SCORING_EXACT
        or not isinstance(same_backend_receipt, ReplayGradeReceipt)
        or same_backend_receipt.grade is not ReplayGrade.EXPERIMENT_SAME_BACKEND
    ):
        raise ProductionControllerError(
            "exact replay requires SCORING_EXACT and EXPERIMENT_SAME_BACKEND receipts"
        )
    clean_manifest = evidence.manifest()
    clean_digest = canonical_json_sha256(clean_manifest)
    # Replay-grade v1 predates the domain canonical-number encoder and retains JSON float
    # spellings such as ``0.0``.  Verify its own legacy proof digest while publishing the v2
    # clean-replay manifest under the current canonical digest below.
    receipt_clean_digest = _stdlib_canonical_json_sha256(clean_manifest)
    for receipt in (scoring_receipt, same_backend_receipt):
        if receipt.evidence.get("clean_replay_sha256") != receipt_clean_digest:
            raise ProductionControllerError(
                f"{receipt.grade.value} receipt does not bind clean replay evidence"
            )
        if receipt.evidence_sha256 != _stdlib_canonical_json_sha256(dict(receipt.evidence)):
            raise ProductionControllerError(
                f"{receipt.grade.value} receipt evidence digest is invalid"
            )
    return {
        "clean_replay_evidence_sha256": clean_digest,
        "clean_replay_evidence": clean_manifest,
        "scoring_exact_receipt": scoring_receipt.manifest(),
        "same_backend_receipt": same_backend_receipt.manifest(),
    }


def _stdlib_canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_organizer_evidence(
    evidence: OrganizerCheckEvidence,
    *,
    expected_submission_sha256: str,
    expected_starter_manifest_sha256: str,
    expected_final_rows: int,
) -> dict[str, object]:
    """Cross-check the exact successful, outcome-masked organizer-check evidence."""

    if not isinstance(evidence, OrganizerCheckEvidence):
        raise ProductionControllerError("organizer evidence must be OrganizerCheckEvidence")
    if type(expected_final_rows) is not int or expected_final_rows <= 0:
        raise ProductionControllerError("expected organizer final rows must be positive")
    manifest = evidence.manifest()
    if (
        manifest.get("mode") != "check_only"
        or manifest.get("split") != "test"
        or evidence.checker_returncode != 0
        or evidence.checker_command != _ORGANIZER_COMMAND
    ):
        raise ProductionControllerError("organizer evidence is not the exact check-only test run")
    if (
        evidence.submission_sha256 != expected_submission_sha256
        or evidence.starter_manifest_sha256 != expected_starter_manifest_sha256
    ):
        raise ProductionControllerError("organizer evidence differs from admitted artifacts")
    if evidence.submission_size_bytes <= 0:
        raise ProductionControllerError("organizer evidence reports an empty submission")
    if (
        evidence.checker_stdout_sha256
        != hashlib.sha256(evidence.checker_stdout.encode("utf-8")).hexdigest()
        or evidence.checker_stderr_sha256
        != hashlib.sha256(evidence.checker_stderr.encode("utf-8")).hexdigest()
    ):
        raise ProductionControllerError("organizer output digest is invalid")

    masked = evidence.masked_view
    counted_rows = 0
    for entry in masked.files:
        rows = entry.final_rows_masked
        if rows is not None:
            if type(rows) is not int or rows < 0:
                raise ProductionControllerError("organizer masked file row count is invalid")
            counted_rows += rows
    expected_cells = expected_final_rows * len(OUTCOME_FIELDS)
    if (
        masked.final_rows_masked != expected_final_rows
        or counted_rows != expected_final_rows
        or masked.final_outcome_cells_replaced != expected_cells
    ):
        raise ProductionControllerError("organizer outcome masking counts differ from final split")
    stable_masked_view = {
        "schema_version": 1,
        "files": [entry.manifest() for entry in masked.files],
        "registered_outcome_fields": list(OUTCOME_FIELDS),
        "final_rows_masked": expected_final_rows,
        "final_outcome_cells_replaced": expected_cells,
    }
    if masked.digest != canonical_json_sha256(stable_masked_view):
        raise ProductionControllerError("organizer masked-view digest is invalid")
    masked_manifest = _mapping(manifest.get("masked_data_view"), "organizer masked data view")
    isolation = _mapping(
        masked_manifest.get("final_outcome_isolation"),
        "organizer final outcome isolation",
    )
    if (
        isolation.get("registered_fields") != list(OUTCOME_FIELDS)
        or isolation.get("final_rows_masked") != expected_final_rows
        or isolation.get("final_outcome_cells_replaced") != expected_cells
        or any(isolation.get(name) != 0 for name in _ORGANIZER_ZERO_OUTCOME_COUNTERS)
    ):
        raise ProductionControllerError("organizer outcome-isolation evidence is unsafe")
    # Normalization also rejects non-finite or otherwise non-canonical retained values.
    canonical_json_bytes(manifest)
    return manifest


def _rank_graph_identity(*, final_alignment: str, final_prediction_digest: str) -> str:
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "kind": "official_fm_seed_4_qualified_fallback",
            "qualification_scope": _QUALIFICATION_SCOPE,
            "final_alignment_sha256": final_alignment,
            "final_prediction_sha256": final_prediction_digest,
        }
    )


def _provisional_prediction_identity(
    *,
    final_alignment: str,
    final_prediction_digest: str,
    final_rows: int,
    rank_graph_id: str | None = None,
) -> str:
    producer = rank_graph_id or _rank_graph_identity(
        final_alignment=final_alignment,
        final_prediction_digest=final_prediction_digest,
    )
    return PredictionId.from_rank_graph(
        ordered_row_ids=tuple(range(final_rows)),
        prediction_sha256=final_prediction_digest,
        rank_graph_sha256=producer,
    ).value


def _register_fallback_prediction(
    *,
    request: ProductionCPUFallbackRequest,
    final_submission: Path,
    final_prediction_digest: str,
    final_alignment_digest: str,
) -> tuple[str, str, str]:
    repository = request.repository
    family_id = canonical_json_sha256(
        {
            "schema_version": 1,
            "mechanism": "official_fm_seed_4_qualified_fallback",
            "qualification_manifest_digest": request.admission.qualification_manifest_digest,
        }
    )
    rank_graph_id = _rank_graph_identity(
        final_alignment=final_alignment_digest,
        final_prediction_digest=final_prediction_digest,
    )
    prediction_id = _provisional_prediction_identity(
        final_alignment=final_alignment_digest,
        final_prediction_digest=final_prediction_digest,
        final_rows=request.admission.final_rows,
        rank_graph_id=rank_graph_id,
    )
    artifact_sha256 = sha256_file(final_submission)
    artifact_id = canonical_json_sha256(
        {
            "schema_version": 1,
            "kind": "qualified_fallback_submission",
            "sha256": artifact_sha256,
            "prediction_sha256": final_prediction_digest,
        }
    )
    repository.register(
        DurableRecord(
            kind=RecordKind.FAMILY,
            record_id=family_id,
            campaign_id=request.campaign_id,
            contract_id=request.contract_id,
            attributes={"protected_eligible": False},
            payload={
                "campaign_kind": _CAMPAIGN_KIND,
                "qualification_scope": _QUALIFICATION_SCOPE,
                "generated": False,
            },
        )
    )
    repository.register(
        DurableRecord(
            kind=RecordKind.RANK_GRAPH,
            record_id=rank_graph_id,
            campaign_id=request.campaign_id,
            contract_id=request.contract_id,
            references={"family_id": family_id},
            payload={
                "kind": "qualified_fallback_identity_graph",
                "members": ["official_fm_seed_4"],
                "selection_performed": False,
            },
        )
    )
    repository.register(
        DurableRecord(
            kind=RecordKind.ARTIFACT,
            record_id=artifact_id,
            campaign_id=request.campaign_id,
            contract_id=request.contract_id,
            attributes={
                "kind": "qualified_fallback_submission",
                "relative_path": "replay/official-fm-seed-4/submission.csv",
                "sha256": artifact_sha256,
                "size_bytes": final_submission.stat().st_size,
                "verified_path": final_submission,
            },
            payload={
                "qualification_manifest_digest": request.admission.qualification_manifest_digest,
                "qualification_fallback_digest": request.admission.fallback_manifest_digest,
            },
        )
    )
    repository.register(
        DurableRecord(
            kind=RecordKind.PREDICTION,
            record_id=prediction_id,
            campaign_id=request.campaign_id,
            contract_id=request.contract_id,
            references={"artifact_id": artifact_id, "rank_graph_id": rank_graph_id},
            attributes={
                "ordered_rows_sha256": final_alignment_digest,
                "prediction_bytes_sha256": final_prediction_digest,
            },
            payload={
                "stage": "FINAL",
                "qualification_manifest_digest": request.admission.qualification_manifest_digest,
                "qualification_fallback_digest": request.admission.fallback_manifest_digest,
                "labels_accessed": False,
                "protected_metrics_evaluated": False,
                "final_period_outcomes_accessed": False,
            },
        )
    )
    _assert_exact_registered_fallback(
        repository,
        campaign_id=request.campaign_id,
        family_id=family_id,
        rank_graph_id=rank_graph_id,
        artifact_id=artifact_id,
        prediction_id=prediction_id,
    )
    return rank_graph_id, prediction_id, artifact_id


def _assert_exact_registered_fallback(
    repository: StateRepository,
    *,
    campaign_id: str,
    family_id: str,
    rank_graph_id: str,
    artifact_id: str,
    prediction_id: str,
) -> None:
    snapshot = repository.inspect(campaign_id=campaign_id)
    entities = _mapping(snapshot.get("entities"), "registered fallback entities")
    expected = {
        "families": ("family_id", family_id),
        "rank_graphs": ("rank_graph_id", rank_graph_id),
        "artifacts": ("artifact_id", artifact_id),
        "predictions": ("prediction_id", prediction_id),
    }
    for entity_name, (identity_field, expected_id) in expected.items():
        records = entities.get(entity_name)
        if not isinstance(records, list) or len(records) != 1:
            raise ProductionControllerError(
                f"RUNNING recovery has ambiguous {entity_name} authority"
            )
        record = _mapping(records[0], f"registered {entity_name} record")
        if record.get(identity_field) != expected_id:
            raise ProductionControllerError(
                f"RUNNING recovery {entity_name} identity differs from deterministic fallback"
            )


def _directory_bytes(root: Path) -> int:
    _validate_run_directory(root)
    total = 0
    for candidate in root.rglob("*"):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            # Staging trees are created and removed during replay/finalization.  A sampled peak
            # observation may legitimately race one such removal; the next poll observes the new
            # stable state.  Other inspection failures still fail closed.
            continue
        except OSError as exc:
            raise ProductionControllerError(
                "controller run tree changed during measurement"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProductionControllerError("controller run tree contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
    return total


class _ControllerProcessTreeMonitor:
    """Sample simultaneous aggregate RSS and workspace disk over the controller lifetime.

    The process-tree walk is the same identity-aware tracker used by :class:`Runner`.  A sample
    sums the resident bytes of the live controller and every live descendant at one observation;
    it never combines unrelated per-process lifetime maxima.  Disk observations are less frequent
    because a complete, symlink-rejecting tree walk is materially more expensive than RSS polling.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        poll_interval_seconds: float = 0.05,
        disk_poll_interval_seconds: float = 0.25,
    ) -> None:
        if not isinstance(run_dir, Path):
            raise ProductionControllerError("resource monitor run_dir must be pathlib.Path")
        if (
            not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0.0
            or not math.isfinite(disk_poll_interval_seconds)
            or disk_poll_interval_seconds <= 0.0
        ):
            raise ProductionControllerError(
                "resource monitor intervals must be finite and positive"
            )
        try:
            root = psutil.Process(os.getpid())
            identity = ProcessIdentity(root.pid, root.create_time())
            process_group_id = os.getpgid(root.pid)
        except (OSError, psutil.Error) as exc:
            raise ProductionControllerError("controller process identity is unavailable") from exc
        self._run_dir = run_dir
        self._poll_interval_seconds = poll_interval_seconds
        self._disk_poll_interval_seconds = disk_poll_interval_seconds
        self._tracker = _ProcessTracker(identity, process_group_id)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_disk_poll = 0.0
        self._peak_rss_bytes = 0
        self._peak_disk_bytes = 0
        self._peak_process_count = 0
        self._error: BaseException | None = None

    @property
    def poll_interval_seconds(self) -> float:
        return self._poll_interval_seconds

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def start(self) -> None:
        if self._thread is not None:
            raise ProductionControllerError("controller resource monitor was already started")
        self._sample(force_disk=True)
        thread = threading.Thread(
            target=self._run,
            name="production-process-tree-monitor",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval_seconds):
            self._sample(force_disk=False)
            if self._error is not None:
                self._stop.set()
                return

    def _sample(self, *, force_disk: bool) -> None:
        with self._lock:
            if self._error is not None:
                return
            try:
                rss_bytes, process_count = self._tracker.sample()
                if self._tracker.inspection_failed:
                    raise ProductionControllerError("controller process-tree inspection was denied")
                now = time.monotonic()
                disk_bytes: int | None = None
                if force_disk or now - self._last_disk_poll >= self._disk_poll_interval_seconds:
                    disk_bytes = _directory_bytes(self._run_dir)
                    self._last_disk_poll = now
            except (OSError, psutil.Error, ProductionControllerError) as exc:
                self._error = exc
                return
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)
            self._peak_process_count = max(self._peak_process_count, process_count)
            if disk_bytes is not None:
                self._peak_disk_bytes = max(self._peak_disk_bytes, disk_bytes)

    def measurement(self) -> dict[str, int]:
        self._sample(force_disk=True)
        with self._lock:
            if self._error is not None:
                raise ProductionControllerError(
                    f"controller resource monitoring failed: {type(self._error).__name__}"
                ) from self._error
            if self._peak_rss_bytes <= 0 or self._peak_process_count <= 0:
                raise ProductionControllerError("controller process-tree measurement is empty")
            return {
                "peak_rss_bytes": self._peak_rss_bytes,
                "disk_bytes": self._peak_disk_bytes,
            }

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, 4.0 * self._poll_interval_seconds))


class ProductionPublicationMonitor:
    """Trusted resource-measurement seam for exact prepared-publication recovery.

    Start immediately before the recovery operation and call :meth:`measurement` only after the
    finalizer has atomically published and verified its destination.  The returned values then
    truthfully cover that recovery pass through bundle publication, while excluding the later
    terminal SQLite commit that will persist the receipt.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._started_wall: float | None = None
        self._started_cpu: float | None = None
        self._monitor = _ControllerProcessTreeMonitor(run_dir)

    @classmethod
    def start(cls, run_dir: Path) -> Self:
        result = cls(run_dir)
        result._started_wall = time.monotonic()
        result._started_cpu = _process_tree_cpu_seconds()
        result._monitor.start()
        return result

    def measurement(self) -> Mapping[str, object]:
        if self._started_wall is None or self._started_cpu is None:
            raise ProductionControllerError("publication resource monitor is not active")
        return _controller_postpublication_resources(
            self._run_dir,
            started_wall=self._started_wall,
            started_cpu=self._started_cpu,
            monitor=self._monitor,
        )

    def stop(self) -> None:
        self._monitor.stop()
        self._started_wall = None
        self._started_cpu = None


def _process_tree_cpu_seconds() -> float:
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + float(children.ru_utime) + float(children.ru_stime)


def _controller_prepublication_resources(
    run_dir: Path,
    *,
    started_wall: float,
    started_cpu: float,
    monitor: _ControllerProcessTreeMonitor,
) -> dict[str, object]:
    """Measure work so far, excluding evidence publication and terminal commit.

    This is intentionally a self-excluding pre-publication observation.  It must not be read as
    whole-run resource usage or as measurement of the final bundle write and atomic publication.
    """

    return _controller_resource_measurement(
        run_dir,
        started_wall=started_wall,
        started_cpu=started_cpu,
        monitor=monitor,
    )


def _controller_postpublication_resources(
    run_dir: Path,
    *,
    started_wall: float,
    started_cpu: float,
    monitor: _ControllerProcessTreeMonitor,
) -> dict[str, object]:
    """Measure through sealed bundle publication, necessarily before the terminal commit.

    The resulting authority-only receipt must not be written into the bundle it measures: doing so
    would change the bundle bytes, requiring another measurement and creating a content cycle.
    StateRepository binds this receipt atomically with terminal publication instead.
    """

    return _controller_resource_measurement(
        run_dir,
        started_wall=started_wall,
        started_cpu=started_cpu,
        monitor=monitor,
    )


def _controller_resource_measurement(
    run_dir: Path,
    *,
    started_wall: float,
    started_cpu: float,
    monitor: _ControllerProcessTreeMonitor,
) -> dict[str, object]:
    if monitor.run_dir != run_dir:
        raise ProductionControllerError("resource monitor observes a different run directory")
    wall_seconds = max(time.monotonic() - started_wall, 0.0)
    cpu_seconds = max(_process_tree_cpu_seconds() - started_cpu, 0.0)
    if not math.isfinite(wall_seconds) or not math.isfinite(cpu_seconds):
        raise ProductionControllerError("controller resource timing is non-finite")
    observed = monitor.measurement()
    return {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": observed["peak_rss_bytes"],
        "disk_bytes": observed["disk_bytes"],
        "device": "cpu",
    }


def _resource_receipt_body(
    *,
    request: ProductionCPUFallbackRequest,
    prediction_id: str,
    controller_resources: Mapping[str, object],
) -> dict[str, object]:
    """Build the closed authority receipt with pre-publication controller measurements."""

    resource_evidence = request.admission.runtime.fallback.resources
    return {
        "schema_version": 1,
        "contract_id": request.contract_id,
        "campaign_id": request.campaign_id,
        "prediction_id": prediction_id,
        "campaign_kind": _CAMPAIGN_KIND,
        "qualification_scope": _QUALIFICATION_SCOPE,
        "qualification_manifest_digest": request.admission.qualification_manifest_digest,
        "qualification_fallback_digest": request.admission.fallback_manifest_digest,
        "performance_profile_digest": request.admission.performance_profile_digest,
        "declared_resource_profile": request.admission.runtime.resource_profile.manifest(),
        "actual_trainer_backend": "organizer-numpy-fm",
        "actual_trainer_device": "cpu",
        "qualified_training_resources": {
            "wall_seconds": resource_evidence.wall_seconds,
            "cpu_seconds": resource_evidence.cpu_seconds,
            "peak_rss_bytes": resource_evidence.peak_rss_bytes,
            "disk_bytes": resource_evidence.disk_bytes,
            "device": resource_evidence.device,
        },
        "controller_resources": dict(controller_resources),
        "controller_resource_scope": "PREPUBLICATION_SELF_EXCLUDING",
        "preferred_backend_qualified": False,
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "final_period_outcomes_accessed": False,
    }


def _register_or_reuse_resource_receipt(
    *,
    request: ProductionCPUFallbackRequest,
    prediction_id: str,
    controller_resources: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Register once or exact-reuse the sole prepublication authority receipt."""

    snapshot = request.repository.inspect(campaign_id=request.campaign_id)
    entities = _mapping(snapshot.get("entities"), "resource receipt recovery entities")
    existing = entities.get("resource_receipts")
    if not isinstance(existing, list):
        raise ProductionControllerError("resource receipt authority projection is malformed")
    if len(existing) > 1:
        raise ProductionControllerError("RUNNING recovery has ambiguous resource receipt authority")
    if existing:
        record = _mapping(existing[0], "existing resource receipt")
        receipt_id = record.get("receipt_id")
        payload = _mapping(record.get("payload"), "existing resource receipt payload")
        prior_controller = _mapping(
            payload.get("controller_resources"),
            "existing controller resource measurements",
        )
        expected = _resource_receipt_body(
            request=request,
            prediction_id=prediction_id,
            controller_resources=prior_controller,
        )
        if (
            type(receipt_id) is not str
            or canonical_json_sha256(dict(payload)) != receipt_id
            or canonical_json_bytes(dict(payload)) != canonical_json_bytes(expected)
        ):
            raise ProductionControllerError(
                "RUNNING recovery resource receipt differs from deterministic lineage"
            )
        return dict(payload), receipt_id

    body = _resource_receipt_body(
        request=request,
        prediction_id=prediction_id,
        controller_resources=controller_resources,
    )
    receipt_id = canonical_json_sha256(body)
    request.repository.register(
        DurableRecord(
            kind=RecordKind.RESOURCE_RECEIPT,
            record_id=receipt_id,
            campaign_id=request.campaign_id,
            contract_id=request.contract_id,
            payload=body,
        )
    )
    after = request.repository.inspect(campaign_id=request.campaign_id)
    after_entities = _mapping(after.get("entities"), "registered resource receipt entities")
    records = after_entities.get("resource_receipts")
    if (
        not isinstance(records, list)
        or len(records) != 1
        or _mapping(records[0], "registered resource receipt").get("receipt_id") != receipt_id
    ):
        raise ProductionControllerError("resource receipt registration did not converge exactly")
    return body, receipt_id


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ProductionControllerError(f"existing evidence differs: {path.name}")
        return
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ProductionControllerError(f"could not publish evidence: {path.name}") from exc


def _write_json(path: Path, value: object, *, newline: bool = False) -> None:
    payload = canonical_json_bytes(value) + (b"\n" if newline else b"")
    _write_exact(path, payload)


def _write_receipt_json(path: Path, value: object) -> None:
    """Preserve replay-grade v1 float spellings inside authenticated receipt manifests."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProductionControllerError("receipt evidence is not finite canonical JSON") from exc
    _write_exact(path, payload)


def _write_evidence(
    *,
    request: ProductionCPUFallbackRequest,
    evidence_root: Path,
    terminal_binding: TerminalProjectionBinding,
    decision_id: str,
    prediction_id: str,
    replay_payload: Mapping[str, object],
    exact_replay_evidence: Mapping[str, object],
    resource_body: Mapping[str, object],
    resource_receipt_id: str,
    organizer_manifest: Mapping[str, object],
    final_submission: Path,
) -> None:
    binding = terminal_binding.manifest()
    selection = {
        "schema_version": 1,
        "contract_id": request.contract_id,
        "campaign_id": request.campaign_id,
        "decision_id": decision_id,
        "selected_prediction_id": prediction_id,
        "fallback_prediction_id": prediction_id,
        "submission_disposition": _SUBMISSION_DISPOSITION,
        "qualification_scope": _QUALIFICATION_SCOPE,
        "protected_metrics_evaluated": False,
        "official_fm_qualified": True,
        "full_data_qualified": True,
        "terminal_projection": binding,
    }
    scientific = {
        "schema_version": 1,
        "contract_id": request.contract_id,
        "campaign_id": request.campaign_id,
        "decision_id": decision_id,
        "selected_prediction_id": prediction_id,
        "fallback_prediction_id": prediction_id,
        "scientific_disposition": _SCIENTIFIC_DISPOSITION,
        "reason": "bounded production run retained the pre-qualified seed-4 fallback",
        "material_improvement_claimed": False,
        "generated_candidates_evaluated": 0,
        "qualification_reverified_only": True,
        "exact_replay_evidence": dict(exact_replay_evidence),
        "organizer_check": dict(organizer_manifest),
        "controller_resource_measurement_scope": {
            "kind": "pre_publication_self_excluding",
            "includes_evidence_publication": False,
            "includes_bundle_publication": False,
            "includes_terminal_commit": False,
            "complete_run_resources_claimed": False,
        },
        "terminal_projection": binding,
    }
    _write_json(
        evidence_root / EvidenceRole.CONTRACT_MANIFEST.value,
        CONTRACT_MANIFEST.manifest(),
    )
    _write_json(
        evidence_root / EvidenceRole.CAMPAIGN_MANIFEST.value,
        {
            "schema_version": 1,
            "campaign_id": request.campaign_id,
            "contract_id": request.contract_id,
            "decision_id": decision_id,
            "campaign_kind": _CAMPAIGN_KIND,
            "qualification_scope": _QUALIFICATION_SCOPE,
            "startup_receipt": dict(request.startup_receipt),
            "production_admission": request.admission.manifest(),
            "resource_profile": request.admission.runtime.resource_profile.manifest(),
            "terminal_projection": binding,
        },
    )
    _write_json(evidence_root / EvidenceRole.SELECTION_EVIDENCE.value, selection)
    _write_json(evidence_root / EvidenceRole.SUBMISSION_DECISION.value, selection)
    _write_receipt_json(evidence_root / EvidenceRole.SCIENTIFIC_DECISION.value, scientific)
    _write_receipt_json(evidence_root / EvidenceRole.REPLAY_RECEIPT.value, dict(replay_payload))
    _write_json(
        evidence_root / EvidenceRole.RESOURCE_RECEIPTS.value,
        dict(resource_body) | {"receipt_id": resource_receipt_id},
        newline=True,
    )
    _write_json(
        evidence_root / EvidenceRole.PROTECTED_QUERY_ACCOUNTING.value,
        {
            "schema_version": 1,
            "authorized": False,
            "query_count": 0,
            "queries": [],
            "qualification_reverification_is_campaign_query": False,
        },
    )
    _write_json(
        evidence_root / EvidenceRole.PROVIDER_ACCOUNTING.value,
        {"schema_version": 1, "provider_enabled": False, "operation_count": 0},
    )
    _write_json(
        evidence_root / EvidenceRole.FAILURE_SUMMARY.value,
        {"schema_version": 1, "failure_count": 0, "failures": []},
    )
    _write_exact(
        evidence_root / EvidenceRole.SUBMISSION.value,
        final_submission.read_bytes(),
    )
    _write_exact(
        evidence_root / EvidenceRole.REPORT.value,
        b"# Full-data CPU fallback campaign\n\n"
        b"This bounded production run exactly replayed the already-qualified official-FM "
        b"seed-4 fallback on full KuaiRand-Pure inputs. Public validation labels were closed "
        b"inside finalization solely to re-verify imported qualification evidence; no campaign "
        b"protected query, provider operation, generated candidate, or final-period outcome "
        b"was used. The fallback was retained and no score-improvement claim is made.\n\n"
        b"The controller resource receipt is a pre-publication, self-excluding observation. "
        b"It does not claim to measure evidence publication, bundle publication, the terminal "
        b"commit, or complete-run resources.\n",
    )


__all__ = [
    "ProductionCPUFallbackController",
    "ProductionCPUFallbackRequest",
    "ProductionCPUFallbackResult",
    "ProductionControllerError",
    "ProductionPublicationMonitor",
]
