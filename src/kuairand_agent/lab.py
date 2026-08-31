"""One deep public facade for the autonomous KuaiRand experiment laboratory.

The first accepted vertical slice is intentionally offline and deterministic.  It exercises the
new contract, resource-profile, trainer, state-authority, replay, and bundle modules without a
provider call, protected score, final-period outcome, or full-data run.  Its evidence says
``SCRIPTED_FIXTURE_ONLY`` throughout; it is not official-FM qualification evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Self, cast

import numpy as np

from kuairand_agent.contract import (
    CONTRACT_ID,
    CONTRACT_MANIFEST,
    ContractManifestError,
    verify_repository_contract_inputs,
)
from kuairand_agent.domain.decisions import ReplayGrade, ScientificDisposition
from kuairand_agent.domain.identity import (
    AttemptId,
    BundleId,
    CampaignId,
    DecisionId,
    ExperimentId,
    FamilyId,
    PredictionId,
    TrialId,
    canonical_json_bytes,
    canonical_json_sha256,
)
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
from kuairand_agent.observability.receipts import (
    ReceiptError,
    ScriptedReplayReceipt,
    StartupReceipt,
)
from kuairand_agent.resource_profiles import ResourceProfile, load_resource_profile
from kuairand_agent.scoring.submission import (
    AlignmentRow,
    read_submission,
    write_submission,
)
from kuairand_agent.state.repository import (
    DurableRecord,
    PreparedTerminalProjection,
    PublishedBundleReceipt,
    RecordKind,
    StateRepository,
    TerminalPreparation,
)
from kuairand_agent.training.protocol import (
    ResourceLimits,
    TrainerFailureCode,
    TrialRequest,
    TrialResult,
)
from kuairand_agent.training.scripted import ScriptedTrainer, ScriptedTrialPayload

LAB_SCHEMA_VERSION: Final = 1
CAMPAIGN_KIND: Final = "OFFLINE_SCRIPTED_FIXTURE"
SCRIPTED_DISPOSITION: Final = "SCRIPTED_FALLBACK_RETAINED"
SCRIPTED_QUALIFICATION_SCOPE: Final = "SCRIPTED_FIXTURE_ONLY"
_EXPECTED_PROFILE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "cpu": "competition-cpu",
        "gpu": "competition-gpu",
        "competition-cpu": "competition-cpu",
        "competition-gpu": "competition-gpu",
    }
)
_FIXTURE_ALIGNMENT: Final = (
    AlignmentRow(0, "fixture-user-0", "fixture-video-0"),
    AlignmentRow(1, "fixture-user-0", "fixture-video-1"),
    AlignmentRow(2, "fixture-user-1", "fixture-video-2"),
    AlignmentRow(3, "fixture-user-1", "fixture-video-3"),
)
_FIXTURE_PREDICTIONS: Final = (0.875, 0.125, 0.75, 0.25)


class LabError(RuntimeError):
    """Base failure at the autonomous-laboratory interface."""


class LabAdmissionError(LabError):
    """Raised before campaign creation when immutable inputs are invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: TrainerFailureCode = TrainerFailureCode.ADMISSION_REJECTED,
        requested_profile: str | None = None,
        selected_profile: str | None = None,
        missing_evidence: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.requested_profile = requested_profile
        self.selected_profile = selected_profile
        self.missing_evidence = missing_evidence


class LabConflictError(LabError):
    """Raised when an idempotency key or run root is already bound incompatibly."""


class BundleValidationError(LabError):
    """Raised when a sealed bundle no longer matches its content-addressed manifest."""


@dataclass(frozen=True, slots=True)
class CampaignOptions:
    """Typed input to the one competition path plus an explicitly gated test seam."""

    config_path: Path
    execution: str = "competition"
    data_root: Path | None = None
    starter_root: Path | None = None
    qualification_receipt: Path | None = None
    allow_test_fixture: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, Path):
            raise LabAdmissionError("config_path must be pathlib.Path")
        if self.execution not in {"competition", "offline-scripted"}:
            raise LabAdmissionError(
                "execution must be 'competition' or the gated 'offline-scripted' test seam"
            )
        for name in ("data_root", "starter_root", "qualification_receipt"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise LabAdmissionError(f"{name} must be pathlib.Path when provided")
        if type(self.allow_test_fixture) is not bool:
            raise LabAdmissionError("allow_test_fixture must be bool")
        if self.execution == "offline-scripted" and not self.allow_test_fixture:
            raise LabAdmissionError(
                "offline-scripted is a test/development seam and requires allow_test_fixture=True"
            )
        if self.execution == "competition" and self.allow_test_fixture:
            raise LabAdmissionError("competition execution cannot enable the scripted test fixture")


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """Terminal campaign result using the plan's claim distinctions."""

    campaign_id: str
    contract_id: str
    terminal_state: str
    submission_disposition: str
    scientific_disposition: str
    selected_prediction_id: str
    fallback_prediction_id: str
    exact_metrics: Mapping[str, float] | None
    bundle_path: Path
    bundle_sha256: str
    replay_grade: str
    resource_receipt_id: str
    protected_query_count: int
    evidence_manifest_path: Path
    replay_grades: tuple[str, ...]
    campaign_kind: str = CAMPAIGN_KIND
    qualification_scope: str = SCRIPTED_QUALIFICATION_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "campaign_id",
            "contract_id",
            "selected_prediction_id",
            "fallback_prediction_id",
            "bundle_sha256",
            "resource_receipt_id",
        ):
            _require_sha256(getattr(self, name), name)
        if self.protected_query_count != 0:
            raise LabError("offline scripted campaigns cannot spend protected queries")
        if self.exact_metrics is not None:
            raise LabError("offline scripted campaigns have no protected exact metrics")
        if self.campaign_kind != CAMPAIGN_KIND:
            raise LabError("campaign_kind must identify the offline scripted fixture")
        if self.qualification_scope != SCRIPTED_QUALIFICATION_SCOPE:
            raise LabError("scripted campaigns must not claim production qualification")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": LAB_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "contract_id": self.contract_id,
            "terminal_state": self.terminal_state,
            "submission_disposition": self.submission_disposition,
            "scientific_disposition": self.scientific_disposition,
            "selected_prediction_id": self.selected_prediction_id,
            "fallback_prediction_id": self.fallback_prediction_id,
            "exact_metrics": None,
            "bundle_path": str(self.bundle_path),
            "bundle_sha256": self.bundle_sha256,
            "replay_grade": self.replay_grade,
            "replay_grades": list(self.replay_grades),
            "resource_receipt_id": self.resource_receipt_id,
            "protected_query_count": self.protected_query_count,
            "evidence_manifest_path": str(self.evidence_manifest_path),
            "campaign_kind": self.campaign_kind,
            "qualification_scope": self.qualification_scope,
            "official_fm_qualified": False,
            "full_data_qualified": False,
        }


@dataclass(frozen=True, slots=True)
class BundleValidationResult:
    """Read-only validation of one exact bundle projection."""

    bundle_path: Path
    bundle_id: str
    contract_id: str
    campaign_id: str
    decision_id: str
    resource_receipt_id: str
    preparation_id: str
    projection_sha256: str
    manifest_sha256: str
    submission_sha256: str
    inventory_sha256: str
    selected_prediction_id: str
    file_count: int
    total_size_bytes: int
    valid: bool = True

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": LAB_SCHEMA_VERSION,
            "bundle_path": str(self.bundle_path),
            "bundle_id": self.bundle_id,
            "contract_id": self.contract_id,
            "campaign_id": self.campaign_id,
            "decision_id": self.decision_id,
            "resource_receipt_id": self.resource_receipt_id,
            "preparation_id": self.preparation_id,
            "projection_sha256": self.projection_sha256,
            "manifest_sha256": self.manifest_sha256,
            "submission_sha256": self.submission_sha256,
            "inventory_sha256": self.inventory_sha256,
            "selected_prediction_id": self.selected_prediction_id,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """A named replay grade verified without mutating authority or bundle state."""

    campaign_id: str
    contract_id: str
    prediction_id: str
    grade: str
    bundle_id: str
    verified: bool
    qualification_scope: str = SCRIPTED_QUALIFICATION_SCOPE

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": LAB_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "contract_id": self.contract_id,
            "prediction_id": self.prediction_id,
            "grade": self.grade,
            "bundle_id": self.bundle_id,
            "verified": self.verified,
            "qualification_scope": self.qualification_scope,
        }


@dataclass(frozen=True, slots=True)
class _ScriptedExecution:
    request: TrialRequest
    result: TrialResult
    replay: TrialResult
    replay_receipt: ScriptedReplayReceipt
    family_id: FamilyId
    experiment_id: ExperimentId
    trial_id: TrialId
    attempt_id: AttemptId


@dataclass(frozen=True, slots=True)
class _RegisteredExecution:
    resource_receipt_id: str
    resource_receipt_body: Mapping[str, object]


class AutonomousExperimentLab:
    """Deep facade owning admission, deterministic execution, state, replay, and finalization."""

    def __init__(
        self,
        *,
        repository_root: Path,
        state_root: Path,
        run_root: Path,
        profile: str,
        startup_receipt: StartupReceipt,
    ) -> None:
        self._repository_root = repository_root
        self._state_root = state_root
        self._run_root = run_root
        self._profile = profile
        self._startup_receipt = startup_receipt

    @classmethod
    def open(
        cls,
        *,
        repository_root: Path,
        state_root: Path,
        run_root: Path,
        profile: str,
    ) -> Self:
        """Verify the frozen contract and create a no-write laboratory handle.

        ``StateRepository.open`` is deliberately deferred until :meth:`compete`.  Therefore
        contract failure cannot create a state directory or SQLite file, and inspection through a
        handle returned here remains read-only.
        """

        if not all(isinstance(path, Path) for path in (repository_root, state_root, run_root)):
            raise LabAdmissionError(
                "repository_root, state_root, and run_root must be pathlib.Path"
            )
        try:
            normalized_profile = _EXPECTED_PROFILE[profile]
        except (KeyError, TypeError) as exc:
            raise LabAdmissionError(
                "profile must be cpu, gpu, competition-cpu, or competition-gpu"
            ) from exc
        try:
            repository = repository_root.resolve(strict=True)
        except OSError as exc:
            raise LabAdmissionError(f"repository_root is unavailable: {repository_root}") from exc
        if not repository.is_dir():
            raise LabAdmissionError("repository_root must be a directory")
        try:
            verification = verify_repository_contract_inputs(repository)
        except ContractManifestError as exc:
            raise LabAdmissionError(
                f"contract verification failed before state open: {exc}"
            ) from exc
        startup = StartupReceipt.from_verification(
            verification,
            profile=normalized_profile,
        )
        return cls(
            repository_root=repository,
            state_root=state_root.expanduser().resolve(strict=False),
            run_root=run_root.expanduser().resolve(strict=False),
            profile=normalized_profile,
            startup_receipt=startup,
        )

    @property
    def startup_receipt(self) -> StartupReceipt:
        return self._startup_receipt

    def compete(self, *, options: CampaignOptions, idempotency_key: str) -> CampaignResult:
        """Admit the requested competition profile before creating a campaign identity.

        The scripted fixture remains reachable only through an explicit test/development option.
        Production-shaped calls never silently turn into fixture campaigns.
        """

        if not isinstance(options, CampaignOptions):
            raise LabAdmissionError("options must be CampaignOptions")
        key = _single_line(idempotency_key, "idempotency_key")
        self._reverify_contract_before_state()
        profile = self._load_profile(options.config_path)
        config_sha256 = _file_sha256(options.config_path)
        if options.execution == "competition":
            self._admit_competition_or_refuse(options, profile=profile)
            raise AssertionError("competition admission must either return a controller or refuse")
        return self._compete_scripted_fixture(
            options=options,
            key=key,
            profile=profile,
            config_sha256=config_sha256,
        )

    def _reverify_contract_before_state(self) -> None:
        """Close the time-of-check gap between opening the handle and campaign admission."""

        try:
            verification = verify_repository_contract_inputs(self._repository_root)
            current = StartupReceipt.from_verification(
                verification,
                profile=self._profile,
            )
        except (ContractManifestError, ReceiptError) as exc:
            raise LabAdmissionError(
                f"contract re-verification failed before state open: {exc}"
            ) from exc
        if current.manifest() != self._startup_receipt.manifest():
            raise LabAdmissionError(
                "contract re-verification differs from the startup receipt before state open"
            )

    def _compete_scripted_fixture(
        self,
        *,
        options: CampaignOptions,
        key: str,
        profile: ResourceProfile,
        config_sha256: str,
    ) -> CampaignResult:
        """Run the deterministic provider-free fixture behind the explicit dev/test gate."""

        identity_config = {
            "schema_version": LAB_SCHEMA_VERSION,
            "campaign_kind": CAMPAIGN_KIND,
            "execution": options.execution,
            "resource_profile": profile.manifest(),
            "resource_profile_file_sha256": config_sha256,
            "startup_receipt_id": self._startup_receipt.receipt_id,
            "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
            "provider_enabled": False,
            "protected_scoring_enabled": False,
            "full_data_enabled": False,
        }
        campaign_id = CampaignId.derive(
            contract_id=CONTRACT_ID,
            campaign_config=identity_config,
            start_nonce=key,
        )
        campaign_run_root = self._campaign_run_root(campaign_id)
        durable_config = identity_config | {"run_root": str(campaign_run_root)}

        # This is the first state write in the facade.  The startup receipt above already proves
        # that the candidate and expected ContractId were identical. Re-verify at this exact seam
        # as profile/config admission occurred after the earlier compete-level check.
        self._reverify_contract_before_state()
        repository = StateRepository.open(self._state_root)
        handle = repository.create_campaign(
            campaign_id=campaign_id,
            contract_id=CONTRACT_ID,
            contract_manifest=CONTRACT_MANIFEST.manifest(),
            config=durable_config,
            idempotency_key=key,
            protected_query_limit=0,
            initial_state="READY",
        )
        if not handle.created and handle.state == "COMPLETED_OFFLINE_FIXTURE":
            return self._result_from_snapshot(repository.inspect(campaign_id=campaign_id))
        if not handle.created and handle.state not in {"READY", "RUNNING"}:
            raise LabConflictError(f"campaign is not resumable from state {handle.state!r}")
        if handle.state == "READY":
            transitioned = repository.transition(
                campaign_id=campaign_id,
                entity_kind="campaign",
                entity_id=campaign_id,
                expected_state="READY",
                expected_revision=handle.revision,
                new_state="RUNNING",
                event_type="offline_scripted_campaign_started",
                payload={
                    "campaign_kind": CAMPAIGN_KIND,
                    "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
                },
            )
            campaign_revision = transitioned.revision
        else:
            campaign_revision = handle.revision

        campaign_run_root.mkdir(parents=True, exist_ok=True)
        execution = self._execute_scripted(
            profile,
            campaign_id=campaign_id,
            config_sha256=config_sha256,
        )
        prediction_artifact, artifact_id = self._persist_execution_artifacts(
            execution,
            campaign_id=campaign_id,
        )
        registered = self._register_execution(
            repository,
            campaign_id=campaign_id,
            execution=execution,
            prediction_artifact=prediction_artifact,
            artifact_id=artifact_id,
            profile=profile,
        )
        resource_receipt_id = registered.resource_receipt_id
        decision_id = DecisionId.derive(
            policy_sha256=canonical_json_sha256(
                {
                    "schema_version": 1,
                    "selection": SCRIPTED_DISPOSITION,
                    "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
                    "protected_metrics_required": False,
                }
            ),
            evidence_ids={"scripted_fallback_prediction": execution.result.prediction_id},
        )
        replay_id = canonical_json_sha256(
            {
                "schema_version": 1,
                "contract_id": CONTRACT_ID.value,
                "prediction_id": execution.result.prediction_id.value,
                "scripted_replay_receipt_id": execution.replay_receipt.receipt_id,
            }
        )
        decision_payload = {
            "contract_id": CONTRACT_ID.value,
            "campaign_id": campaign_id.value,
            "decision_id": decision_id.value,
            "selected_prediction_id": execution.result.prediction_id.value,
            "fallback_prediction_id": execution.result.prediction_id.value,
            "submission_disposition": SCRIPTED_DISPOSITION,
            "scientific_disposition": ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE.value,
            "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
            "official_fm_qualified": False,
        }
        replay_payload = {
            "contract_id": CONTRACT_ID.value,
            "campaign_id": campaign_id.value,
            "prediction_id": execution.result.prediction_id.value,
            "replay_grades": [
                ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
                ReplayGrade.BUNDLE_EXACT.value,
            ],
            "scripted_replay_receipt": execution.replay_receipt.manifest(),
        }
        bundle_claims = {
            "resource_receipt_id": resource_receipt_id,
            "replay_grade": ReplayGrade.BUNDLE_EXACT.value,
            "replay_grades": [
                ReplayGrade.EXPERIMENT_SAME_BACKEND.value,
                ReplayGrade.BUNDLE_EXACT.value,
            ],
            "submission_disposition": SCRIPTED_DISPOSITION,
            "scientific_disposition": ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE.value,
            "campaign_kind": CAMPAIGN_KIND,
            "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
            "protected_query_count": 0,
            "exact_metrics": None,
        }
        prepared = repository.prepare_terminal_projection(
            campaign_id=campaign_id,
            contract_id=CONTRACT_ID,
            expected_state="RUNNING",
            expected_revision=campaign_revision,
            preparation=TerminalPreparation(
                decision_id=decision_id,
                replay_id=replay_id,
                selected_prediction_id=execution.result.prediction_id,
                fallback_prediction_id=execution.result.prediction_id,
                terminal_state="COMPLETED_OFFLINE_FIXTURE",
                decision_payload=decision_payload,
                replay_payload=replay_payload,
                bundle_claims=bundle_claims,
            ),
        )
        bundle = self._ensure_bundle(
            repository,
            campaign_id=campaign_id,
            profile=profile,
            execution=execution,
            resource_receipt_id=resource_receipt_id,
            resource_receipt_body=registered.resource_receipt_body,
            decision_id=decision_id,
            prepared=prepared,
        )
        repository.finalize_prepared_campaign(
            preparation_id=prepared.preparation_id,
            publication=PublishedBundleReceipt(
                root=bundle.bundle_path,
                bundle_id=BundleId(bundle.bundle_id),
                manifest_sha256=bundle.manifest_sha256,
                inventory_sha256=bundle.inventory_sha256,
                submission_sha256=bundle.submission_sha256,
                file_count=bundle.file_count,
                total_size_bytes=bundle.total_size_bytes,
            ),
        )
        return self._result_from_snapshot(repository.inspect(campaign_id=campaign_id))

    def _admit_competition_or_refuse(
        self,
        options: CampaignOptions,
        *,
        profile: ResourceProfile,
    ) -> None:
        """Fail closed before ``CampaignId`` creation until real inputs are admissible.

        Local fixture qualifications prove the trainer adapters, but they do not provide the
        full-data official-FM parity, measured p95 resource envelope, or production controller
        receipt required by the frozen plan. GPU is therefore preflighted to the CPU policy before
        this typed refusal; it is never relabelled as a GPU campaign.
        """

        selected_profile = profile.name
        missing: list[str] = []
        if profile.name == "competition-gpu":
            cpu_config = self._repository_root / "configs/competition-cpu.toml"
            try:
                cpu_profile = load_resource_profile(cpu_config)
            except (OSError, ValueError) as exc:
                raise LabAdmissionError(
                    f"GPU is unsupported and the CPU fallback profile is unavailable: {exc}",
                    code=TrainerFailureCode.UNSUPPORTED,
                    requested_profile=profile.name,
                ) from exc
            if cpu_profile.scientific_policy.digest != profile.scientific_policy.digest:
                raise LabAdmissionError(
                    "GPU and CPU profiles do not share the frozen scientific policy",
                    requested_profile=profile.name,
                )
            selected_profile = cpu_profile.name
            missing.append("same-backend GPU qualification (GPU preflight selected CPU)")

        required_paths = (
            ("verified KuaiRand-Pure data root", options.data_root),
            ("hash-verified organizer starter root", options.starter_root),
            (
                "full-data parity and measured-p95 qualification receipt",
                options.qualification_receipt,
            ),
        )
        for label, path in required_paths:
            if path is None:
                missing.append(label)
                continue
            try:
                resolved = path.expanduser().resolve(strict=True)
            except OSError:
                missing.append(label)
                continue
            if label.endswith("receipt"):
                if not resolved.is_file():
                    missing.append(label)
            elif not resolved.is_dir():
                missing.append(label)

        # The replacement controller is not yet admitted to consume those capabilities. Keeping
        # this item explicit prevents a directory-shaped placeholder from becoming a live run.
        missing.append("admitted full-data scientific controller receipt")
        raise LabAdmissionError(
            "competition admission refused before CampaignId creation; missing verified evidence: "
            + ", ".join(missing),
            code=TrainerFailureCode.ADMISSION_REJECTED,
            requested_profile=profile.name,
            selected_profile=selected_profile,
            missing_evidence=tuple(missing),
        )

    def inspect(self, *, campaign_id: str | CampaignId) -> Mapping[str, object]:
        """Return a projection using SQLite's read-only mode and perform no reconciliation."""

        repository = StateRepository(self._state_root / "authority.sqlite3")
        return repository.inspect(campaign_id=campaign_id)

    def replay(
        self,
        *,
        campaign_id: str | CampaignId,
        grade: str | ReplayGrade = ReplayGrade.EXPERIMENT_SAME_BACKEND,
    ) -> ReplayResult:
        """Verify one already-achieved named grade without writing state or bundle files."""

        try:
            requested = (
                grade
                if isinstance(grade, ReplayGrade)
                else ReplayGrade(grade.upper().replace("-", "_"))
            )
        except (AttributeError, ValueError) as exc:
            raise LabAdmissionError("grade must name a supported ReplayGrade") from exc
        if requested not in {ReplayGrade.EXPERIMENT_SAME_BACKEND, ReplayGrade.BUNDLE_EXACT}:
            raise LabAdmissionError(
                "the unscored scripted slice supports EXPERIMENT_SAME_BACKEND and BUNDLE_EXACT"
            )
        snapshot = self.inspect(campaign_id=campaign_id)
        campaign = _mapping(snapshot.get("campaign"), "campaign")
        entities = _mapping(snapshot.get("entities"), "entities")
        bundles = _mapping_list(entities.get("bundles"), "entities.bundles")
        if campaign.get("state") != "COMPLETED_OFFLINE_FIXTURE" or len(bundles) != 1:
            raise LabError("campaign has no terminal scripted bundle to replay")
        bundle_payload = _mapping(bundles[0].get("payload"), "bundle payload")
        achieved = _string_tuple(bundle_payload.get("replay_grades"), "replay_grades")
        if requested.value not in achieved:
            raise LabError(f"campaign did not achieve replay grade {requested.value}")
        validation = self.validate_bundle(
            Path(_string(bundle_payload.get("bundle_path"), "bundle_path"))
        )
        campaign_identity = _string(campaign.get("campaign_id"), "campaign_id")
        contract_identity = _string(campaign.get("contract_id"), "contract_id")
        if validation.campaign_id != campaign_identity:
            raise LabError("bundle differs from authoritative CampaignId")
        if validation.contract_id != contract_identity:
            raise LabError("bundle differs from authoritative ContractId")
        if validation.bundle_id != _string(bundles[0].get("bundle_id"), "bundle_id"):
            raise LabError("bundle bytes differ from authoritative BundleId")
        replay_receipt = _read_json_object(
            validation.bundle_path / EvidenceRole.REPLAY_RECEIPT.value
        )
        if requested is ReplayGrade.EXPERIMENT_SAME_BACKEND:
            if replay_receipt.get("grade") != ReplayGrade.EXPERIMENT_SAME_BACKEND.value:
                raise LabError("scripted replay receipt does not prove EXPERIMENT_SAME_BACKEND")
            if replay_receipt.get("qualification_scope") != SCRIPTED_QUALIFICATION_SCOPE:
                raise LabError("scripted replay receipt overstates its qualification scope")
        return ReplayResult(
            campaign_id=campaign_identity,
            contract_id=contract_identity,
            prediction_id=validation.selected_prediction_id,
            grade=requested.value,
            bundle_id=validation.bundle_id,
            verified=True,
        )

    @staticmethod
    def validate_bundle(bundle: Path) -> BundleValidationResult:
        """Validate bundle membership, every evidence hash, and the derived ``BundleId``."""

        if not isinstance(bundle, Path):
            raise BundleValidationError("bundle must be pathlib.Path")
        expanded = bundle.expanduser()
        try:
            supplied_stat = expanded.lstat()
        except OSError as exc:
            raise BundleValidationError(f"bundle is unavailable: {expanded}") from exc
        if stat.S_ISLNK(supplied_stat.st_mode) or not stat.S_ISDIR(supplied_stat.st_mode):
            raise BundleValidationError("bundle must be a real directory, not a symlink")
        if stat.S_IMODE(supplied_stat.st_mode) != 0o555:
            raise BundleValidationError("bundle root must be sealed read-only with mode 0555")
        try:
            root = expanded.resolve(strict=True)
        except OSError as exc:
            raise BundleValidationError(f"bundle is unavailable: {expanded}") from exc
        members = tuple(sorted(path.name for path in root.iterdir()))
        expected = tuple(sorted(REQUIRED_BUNDLE_PATHS))
        if members != expected:
            raise BundleValidationError("bundle member set differs from the required layout")
        manifest_path = root / "bundle-manifest.json"
        manifest = _read_json_object(manifest_path)
        if manifest.get("schema_version") != 2:
            raise BundleValidationError(
                "prepared-terminal bundle manifest schema_version must be 2"
            )
        contract_id = _require_sha256(manifest.get("contract_id"), "contract_id")
        if contract_id != CONTRACT_ID.value:
            raise BundleValidationError("bundle ContractId differs from the frozen contract")
        campaign_id = _require_sha256(manifest.get("campaign_id"), "campaign_id")
        required_paths = manifest.get("required_paths")
        if not isinstance(required_paths, list) or any(
            type(value) is not str for value in required_paths
        ):
            raise BundleValidationError("bundle manifest required_paths must be text paths")
        if tuple(sorted(cast(list[str], required_paths))) != expected:
            raise BundleValidationError("bundle manifest required_paths differs from the layout")
        evidence = manifest.get("evidence")
        if not isinstance(evidence, list) or len(evidence) != len(REQUIRED_EVIDENCE_ROLES):
            raise BundleValidationError("bundle manifest has incomplete evidence receipts")
        by_role: dict[str, Mapping[str, object]] = {}
        for index, item in enumerate(evidence):
            receipt = _mapping(item, f"evidence[{index}]")
            role = _string(receipt.get("role"), f"evidence[{index}].role")
            if role in by_role:
                raise BundleValidationError(f"duplicate evidence role {role!r}")
            by_role[role] = receipt
        expected_roles = {role.value for role in REQUIRED_EVIDENCE_ROLES}
        if set(by_role) != expected_roles:
            raise BundleValidationError("bundle evidence roles differ from the required layout")
        for role, receipt in by_role.items():
            path = root / role
            observed_sha256, size_bytes = _regular_file_digest(path)
            if observed_sha256 != _require_sha256(receipt.get("sha256"), f"{role}.sha256"):
                raise BundleValidationError(f"bundle evidence digest changed: {role}")
            declared_size = receipt.get("size_bytes")
            if type(declared_size) is not int or declared_size != size_bytes:
                raise BundleValidationError(f"bundle evidence size changed: {role}")
            receipt_body = {
                "schema_version": 1,
                "role": role,
                "sha256": observed_sha256,
                "size_bytes": size_bytes,
            }
            expected_receipt_id = hashlib.sha256(
                b"kuairand-frozen-bundle-file-v1\0" + canonical_json_bytes(receipt_body)
            ).hexdigest()
            if receipt.get("receipt_id") != expected_receipt_id:
                raise BundleValidationError(f"bundle evidence receipt identity changed: {role}")
        manifest_sha256, _ = _regular_file_digest(manifest_path)
        submission_sha256, _ = _regular_file_digest(root / EvidenceRole.SUBMISSION.value)
        if manifest.get("submission_sha256") != submission_sha256:
            raise BundleValidationError("bundle manifest submission digest differs from bytes")
        selected_prediction = _require_sha256(
            manifest.get("selected_prediction_id"), "selected_prediction_id"
        )
        replay_sha256 = _require_sha256(
            by_role[EvidenceRole.REPLAY_RECEIPT.value].get("sha256"),
            "replay_receipt_sha256",
        )
        if manifest.get("replay_receipt_sha256") != replay_sha256:
            raise BundleValidationError("bundle manifest replay receipt digest differs")
        expected_bundle_id = BundleId.derive(
            selected_prediction_id=PredictionId(selected_prediction),
            replay_output_sha256={EvidenceRole.REPLAY_RECEIPT.value: replay_sha256},
            submission_sha256=submission_sha256,
            manifest_sha256=manifest_sha256,
        ).value
        digest_path = root / "bundle.sha256"
        try:
            digest_text = digest_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise BundleValidationError("bundle.sha256 is unreadable") from exc
        if digest_text != f"{expected_bundle_id}\n":
            raise BundleValidationError("bundle.sha256 differs from the derived BundleId")
        terminal_projection = _mapping(manifest.get("terminal_projection"), "terminal_projection")
        if terminal_projection.get("schema_version") != 1:
            raise BundleValidationError("terminal projection schema_version must be 1")
        if terminal_projection.get("redaction_policy_version") != 1:
            raise BundleValidationError("terminal projection redaction policy must be 1")
        preparation_id = _require_sha256(
            terminal_projection.get("preparation_id"), "preparation_id"
        )
        projection_sha256 = _require_sha256(
            terminal_projection.get("projection_sha256"), "projection_sha256"
        )
        source_revision = _mapping(
            terminal_projection.get("source_revision"), "terminal_projection.source_revision"
        )
        if (
            type(source_revision.get("campaign_revision")) is not int
            or cast(int, source_revision.get("campaign_revision")) < 0
            or type(source_revision.get("last_event_seq")) is not int
            or cast(int, source_revision.get("last_event_seq")) <= 0
        ):
            raise BundleValidationError("terminal projection source revision is invalid")

        contract_evidence = _read_json_object(root / EvidenceRole.CONTRACT_MANIFEST.value)
        if contract_evidence != CONTRACT_MANIFEST.manifest():
            raise BundleValidationError("contract evidence differs from the frozen manifest")
        campaign_evidence = _read_json_object(root / EvidenceRole.CAMPAIGN_MANIFEST.value)
        _require_lineage(
            campaign_evidence,
            contract_id=contract_id,
            campaign_id=campaign_id,
            location="campaign manifest",
        )
        startup = _mapping(campaign_evidence.get("startup_receipt"), "startup_receipt")
        if startup.get("contract_id") != contract_id or startup.get("verified") is not True:
            raise BundleValidationError("campaign startup receipt differs from contract lineage")

        selection = _read_json_object(root / EvidenceRole.SELECTION_EVIDENCE.value)
        submission_decision = _read_json_object(root / EvidenceRole.SUBMISSION_DECISION.value)
        scientific = _read_json_object(root / EvidenceRole.SCIENTIFIC_DECISION.value)
        for location, evidence_object in (
            ("selection evidence", selection),
            ("submission decision", submission_decision),
            ("scientific decision", scientific),
        ):
            _require_lineage(
                evidence_object,
                contract_id=contract_id,
                campaign_id=campaign_id,
                location=location,
            )
        if selection != submission_decision:
            raise BundleValidationError("submission decision differs from selection evidence")
        decision_id = _require_sha256(selection.get("decision_id"), "decision_id")
        if scientific.get("decision_id") != decision_id:
            raise BundleValidationError("scientific decision differs from DecisionId lineage")
        for name in ("selected_prediction_id", "fallback_prediction_id"):
            value = _require_sha256(selection.get(name), name)
            if name == "selected_prediction_id" and value != selected_prediction:
                raise BundleValidationError("selection differs from bundle PredictionId")
            if scientific.get(name) != value:
                raise BundleValidationError(f"scientific decision differs from {name} lineage")

        replay_receipt = _read_json_object(root / EvidenceRole.REPLAY_RECEIPT.value)
        _require_lineage(
            replay_receipt,
            contract_id=contract_id,
            campaign_id=campaign_id,
            location="replay receipt",
        )
        if replay_receipt.get("prediction_id") != selected_prediction:
            raise BundleValidationError("replay receipt differs from selected PredictionId")
        if replay_receipt.get("grade") == ReplayGrade.SCORING_EXACT.value:
            raise BundleValidationError("unscored scripted replay cannot claim SCORING_EXACT")

        resource_rows = _read_json_lines(root / EvidenceRole.RESOURCE_RECEIPTS.value)
        if len(resource_rows) != 1:
            raise BundleValidationError("scripted bundle must contain one resource receipt")
        resource_receipt = resource_rows[0]
        _require_lineage(
            resource_receipt,
            contract_id=contract_id,
            campaign_id=campaign_id,
            location="resource receipt",
        )
        resource_receipt_id = _require_sha256(
            resource_receipt.get("receipt_id"), "resource_receipt_id"
        )
        resource_body = dict(resource_receipt)
        del resource_body["receipt_id"]
        if canonical_json_sha256(resource_body) != resource_receipt_id:
            raise BundleValidationError("resource receipt id differs from its exact content")
        if resource_receipt.get("prediction_id") != selected_prediction:
            raise BundleValidationError("resource receipt differs from selected PredictionId")
        _validate_terminal_projection_evidence(
            root,
            terminal_projection=terminal_projection,
            preparation_id=preparation_id,
            projection_sha256=projection_sha256,
            contract_id=contract_id,
            campaign_id=campaign_id,
            decision_id=decision_id,
            resource_receipt_id=resource_receipt_id,
            selected_prediction_id=selected_prediction,
        )
        for member in root.iterdir():
            if stat.S_IMODE(member.lstat().st_mode) != 0o444:
                raise BundleValidationError(f"bundle member is not sealed read-only: {member.name}")
        inventory_sha256, total_size_bytes = _bundle_inventory(root)
        return BundleValidationResult(
            bundle_path=root,
            bundle_id=expected_bundle_id,
            contract_id=contract_id,
            campaign_id=campaign_id,
            decision_id=decision_id,
            resource_receipt_id=resource_receipt_id,
            preparation_id=preparation_id,
            projection_sha256=projection_sha256,
            manifest_sha256=manifest_sha256,
            submission_sha256=submission_sha256,
            inventory_sha256=inventory_sha256,
            selected_prediction_id=selected_prediction,
            file_count=len(members),
            total_size_bytes=total_size_bytes,
        )

    def _load_profile(self, config_path: Path) -> ResourceProfile:
        try:
            path = config_path.expanduser().resolve(strict=True)
            profile = load_resource_profile(path)
        except (OSError, ValueError) as exc:
            raise LabAdmissionError(f"resource profile admission failed: {exc}") from exc
        if profile.name != self._profile:
            raise LabAdmissionError(
                f"open profile {self._profile!r} differs from config profile {profile.name!r}"
            )
        return profile

    def _campaign_run_root(self, campaign_id: CampaignId) -> Path:
        """Keep every mutable projection and sealed bundle under one CampaignId."""

        return self._run_root / "campaigns" / campaign_id.value

    def _execute_scripted(
        self,
        profile: ResourceProfile,
        *,
        campaign_id: CampaignId,
        config_sha256: str,
    ) -> _ScriptedExecution:
        dependency_lock = _file_sha256(self._repository_root / "uv.lock")
        trainer = ScriptedTrainer(dependency_lock_sha256=dependency_lock)
        family_id = FamilyId.derive(
            mechanism="offline-scripted-fallback-fixture",
            feature_family="scripted-no-data",
            target_family="no-label-access",
            objective_family="deterministic-replay-only",
        )
        experiment_manifest = {
            "schema_version": 1,
            "campaign_kind": CAMPAIGN_KIND,
            "mechanism": "deterministic scripted fallback plumbing",
            "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
            "provider_enabled": False,
            "protected_metrics_evaluated": False,
            "official_fm_qualified": False,
        }
        scripted_source_sha256 = _file_sha256(
            self._repository_root / "src/kuairand_agent/training/scripted.py"
        )
        experiment_id = ExperimentId.derive(
            experiment_spec=experiment_manifest,
            data_identities={"fixture": _sha256_bytes(b"kuairand-offline-scripted-data-v1")},
            fold_identities={"fixture": _sha256_bytes(b"kuairand-offline-scripted-fold-v1")},
            code_artifact_sha256=scripted_source_sha256,
        )
        qualified_settings: dict[str, str | bool] = {
            "qualification_scope": "scripted_fixture_only",
            "declared_resource_profile": profile.name,
            "provider_enabled": False,
            # The experiment remains cross-campaign content addressed. A Trial is one admitted
            # execution association and therefore binds the campaign that owns its attempt.
            "campaign_id": campaign_id.value,
        }
        trial_id = TrialId.derive(
            experiment_id=experiment_id,
            trainer_id=trainer.identity.trainer_id,
            trainer_version=trainer.identity.trainer_version,
            backend=trainer.identity.backend,
            precision=trainer.identity.precision,
            dependency_lock_sha256=trainer.identity.dependency_lock_sha256,
            seed=0,
            fold="offline-fixture",
            fidelity="scripted",
            qualified_settings=qualified_settings,
        )
        attempt_id = AttemptId.derive(trial_id=trial_id, infrastructure_attempt=1)
        payload = ScriptedTrialPayload(
            predictions=_FIXTURE_PREDICTIONS,
            training_data_sha256=_sha256_bytes(b"scripted-training-data-v1"),
            prediction_data_sha256=_sha256_bytes(b"scripted-prediction-data-v1"),
            training_feature_sha256=_sha256_bytes(b"scripted-training-features-v1"),
            prediction_feature_sha256=_sha256_bytes(b"scripted-prediction-features-v1"),
            feature_schema_sha256=_sha256_bytes(b"scripted-feature-schema-v1"),
            config_sha256=config_sha256,
            model_sha256=_sha256_bytes(b"scripted-model-v1"),
            training_rows=len(_FIXTURE_ALIGNMENT),
            feature_count=1,
            diagnostics={
                "campaign_kind": CAMPAIGN_KIND,
                "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
                "official_fm_qualified": False,
            },
        )
        request = TrialRequest(
            experiment_id=experiment_id,
            trial_id=trial_id,
            attempt_id=attempt_id,
            trainer_identity=trainer.identity,
            seed=0,
            fold="offline-fixture",
            fidelity="scripted",
            qualified_settings=qualified_settings,
            infrastructure_attempt=1,
            ordered_row_ids=tuple(row.row_id for row in _FIXTURE_ALIGNMENT),
            resource_limits=ResourceLimits(
                timeout_seconds=30.0,
                memory_limit_bytes=profile.process_tree_rss_hard_cap_bytes,
                disk_limit_bytes=profile.candidate_disk_hard_cap_bytes,
                threads=profile.threads_per_candidate,
            ),
            payload=payload,
        )
        preflight = trainer.preflight(request)
        if preflight.status.value != "PREFLIGHT_PASSED":
            raise LabAdmissionError("scripted trainer did not pass deterministic preflight")
        result = trainer.fit_predict(request)
        replay = trainer.fit_predict(request)
        replay_receipt = ScriptedReplayReceipt(
            contract_id=CONTRACT_ID.value,
            campaign_id=campaign_id.value,
            prediction_id=result.prediction_id.value,
            first_prediction_sha256=result.prediction_sha256,
            replay_prediction_sha256=replay.prediction_sha256,
            first_result_sha256=_stable_trial_result_sha256(result),
            replay_result_sha256=_stable_trial_result_sha256(replay),
        )
        return _ScriptedExecution(
            request=request,
            result=result,
            replay=replay,
            replay_receipt=replay_receipt,
            family_id=family_id,
            experiment_id=experiment_id,
            trial_id=trial_id,
            attempt_id=attempt_id,
        )

    def _persist_execution_artifacts(
        self,
        execution: _ScriptedExecution,
        *,
        campaign_id: CampaignId,
    ) -> tuple[Path, str]:
        artifact = self._campaign_run_root(campaign_id) / "artifacts" / "scripted-predictions.f64le"
        values = np.ascontiguousarray(execution.result.predictions, dtype="<f8")
        _write_exact(artifact, values.tobytes(order="C"))
        artifact_sha256 = _file_sha256(artifact)
        artifact_id = canonical_json_sha256(
            {
                "schema_version": 1,
                "kind": "scripted_prediction_bytes",
                "attempt_id": execution.attempt_id.value,
                "sha256": artifact_sha256,
                "relative_path": "artifacts/scripted-predictions.f64le",
            }
        )
        return artifact, artifact_id

    def _register_execution(
        self,
        repository: StateRepository,
        *,
        campaign_id: CampaignId,
        execution: _ScriptedExecution,
        prediction_artifact: Path,
        artifact_id: str,
        profile: ResourceProfile,
    ) -> _RegisteredExecution:
        repository.register(
            DurableRecord(
                kind=RecordKind.FAMILY,
                record_id=execution.family_id,
                campaign_id=campaign_id,
                contract_id=CONTRACT_ID,
                attributes={"protected_eligible": False},
                payload={
                    "campaign_kind": CAMPAIGN_KIND,
                    "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
                },
            )
        )
        repository.register(
            DurableRecord(
                kind=RecordKind.EXPERIMENT,
                record_id=execution.experiment_id,
                campaign_id=campaign_id,
                contract_id=CONTRACT_ID,
                references={"family_id": execution.family_id},
                payload={
                    "campaign_kind": CAMPAIGN_KIND,
                    "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
                    "protected_metrics_evaluated": False,
                },
            )
        )
        repository.register(
            DurableRecord(
                kind=RecordKind.TRIAL,
                record_id=execution.trial_id,
                campaign_id=campaign_id,
                contract_id=CONTRACT_ID,
                references={"experiment_id": execution.experiment_id},
                payload={"request": _trial_request_manifest(execution.request)},
                state="PENDING",
            )
        )
        repository.register(
            DurableRecord(
                kind=RecordKind.ATTEMPT,
                record_id=execution.attempt_id,
                campaign_id=campaign_id,
                contract_id=CONTRACT_ID,
                references={"trial_id": execution.trial_id},
                attributes={"attempt_ordinal": 1},
                payload={"trainer_identity": execution.result.trainer_identity.manifest()},
                state="RUNNING",
            )
        )
        snapshot = repository.inspect(campaign_id=campaign_id)
        existing_entities = _mapping(snapshot.get("entities"), "entities")
        trial = _execution_entity(
            existing_entities,
            collection="trials",
            identity_key="trial_id",
            identity=execution.trial_id.value,
        )
        attempt = _execution_entity(
            existing_entities,
            collection="attempts",
            identity_key="attempt_id",
            identity=execution.attempt_id.value,
        )
        _validate_execution_state_lineage(
            trial,
            campaign_id=campaign_id,
            entity="trial",
        )
        _validate_execution_state_lineage(
            attempt,
            campaign_id=campaign_id,
            entity="attempt",
        )
        if trial.get("state") == "COMPLETED" and attempt.get("state") != "COMPLETED":
            raise LabConflictError("completed scripted trial has a non-terminal attempt")
        if attempt.get("state") == "RUNNING":
            _require_initial_execution_state(attempt, entity="attempt")
            repository.transition(
                campaign_id=campaign_id,
                entity_kind="attempt",
                entity_id=execution.attempt_id,
                expected_state="RUNNING",
                expected_revision=0,
                new_state="COMPLETED",
                event_type="scripted_attempt_completed",
                payload={"result": execution.result.manifest()},
                terminal=True,
            )
        elif not (
            attempt.get("state") == "COMPLETED"
            and attempt.get("revision") == 1
            and attempt.get("terminal") is True
        ):
            raise LabConflictError("scripted attempt is not deterministically resumable")
        if trial.get("state") == "PENDING":
            _require_initial_execution_state(trial, entity="trial")
            repository.transition(
                campaign_id=campaign_id,
                entity_kind="trial",
                entity_id=execution.trial_id,
                expected_state="PENDING",
                expected_revision=0,
                new_state="COMPLETED",
                event_type="scripted_trial_completed",
                payload={"prediction_id": execution.result.prediction_id.value},
                terminal=True,
            )
        elif not (
            trial.get("state") == "COMPLETED"
            and trial.get("revision") == 1
            and trial.get("terminal") is True
        ):
            raise LabConflictError("scripted trial is not deterministically resumable")
        artifact_sha256 = _file_sha256(prediction_artifact)
        repository.register(
            DurableRecord(
                kind=RecordKind.ARTIFACT,
                record_id=artifact_id,
                campaign_id=campaign_id,
                contract_id=CONTRACT_ID,
                references={"attempt_id": execution.attempt_id},
                attributes={
                    "kind": "scripted_prediction_bytes",
                    "relative_path": "artifacts/scripted-predictions.f64le",
                    "sha256": artifact_sha256,
                    "size_bytes": prediction_artifact.stat().st_size,
                    "verified_path": prediction_artifact,
                },
                payload={"qualification_scope": SCRIPTED_QUALIFICATION_SCOPE},
            )
        )
        repository.verify_artifact(artifact_id=artifact_id, path=prediction_artifact)
        snapshot = repository.inspect(campaign_id=campaign_id)
        existing_entities = _mapping(snapshot.get("entities"), "entities")
        existing_predictions = [
            row
            for row in _mapping_list(existing_entities.get("predictions"), "entities.predictions")
            if row.get("prediction_id") == execution.result.prediction_id.value
        ]
        if len(existing_predictions) > 1:
            raise LabConflictError("scripted PredictionId is duplicated in authority")
        if existing_predictions:
            existing_prediction = existing_predictions[0]
            if (
                existing_prediction.get("contract_id") != CONTRACT_ID.value
                or existing_prediction.get("campaign_id") != campaign_id.value
                or existing_prediction.get("trial_id") != execution.trial_id.value
                or existing_prediction.get("artifact_id") != artifact_id
                or existing_prediction.get("ordered_rows_sha256")
                != canonical_json_sha256(list(execution.result.ordered_row_ids))
                or existing_prediction.get("prediction_bytes_sha256")
                != execution.result.prediction_sha256
            ):
                raise LabConflictError("existing scripted prediction has different lineage")
            prediction_payload = _mapping(
                existing_prediction.get("payload"), "existing prediction payload"
            )
            if (
                prediction_payload.get("stage") != "FINAL"
                or prediction_payload.get("labels_accessed") is not False
                or prediction_payload.get("protected_metrics_evaluated") is not False
            ):
                raise LabConflictError("existing scripted prediction has different semantics")
            result_manifest = _mapping(
                prediction_payload.get("trial_result"), "existing trial result"
            )
        else:
            result_manifest = execution.result.manifest()
            repository.register(
                DurableRecord(
                    kind=RecordKind.PREDICTION,
                    record_id=execution.result.prediction_id,
                    campaign_id=campaign_id,
                    contract_id=CONTRACT_ID,
                    references={"artifact_id": artifact_id, "trial_id": execution.trial_id},
                    attributes={
                        "ordered_rows_sha256": canonical_json_sha256(
                            list(execution.result.ordered_row_ids)
                        ),
                        "prediction_bytes_sha256": execution.result.prediction_sha256,
                    },
                    payload={
                        "stage": "FINAL",
                        "labels_accessed": False,
                        "protected_metrics_evaluated": False,
                        "trial_result": result_manifest,
                    },
                )
            )
        resource_manifest = _scripted_resource_receipt_body_from_manifest(
            campaign_id=campaign_id,
            execution=execution,
            result_manifest=result_manifest,
            profile=profile,
        )
        receipt_id = canonical_json_sha256(resource_manifest)
        resources = _mapping_list(
            existing_entities.get("resource_receipts"), "entities.resource_receipts"
        )
        attempt_resources = [
            row for row in resources if row.get("attempt_id") == execution.attempt_id.value
        ]
        if any(row.get("receipt_id") != receipt_id for row in attempt_resources):
            raise LabConflictError("scripted attempt has a different resource receipt")
        repository.register(
            DurableRecord(
                kind=RecordKind.RESOURCE_RECEIPT,
                record_id=receipt_id,
                campaign_id=campaign_id,
                contract_id=CONTRACT_ID,
                references={"attempt_id": execution.attempt_id},
                payload=resource_manifest,
            )
        )
        return _RegisteredExecution(receipt_id, resource_manifest)

    def _ensure_bundle(
        self,
        repository: StateRepository,
        *,
        campaign_id: CampaignId,
        profile: ResourceProfile,
        execution: _ScriptedExecution,
        resource_receipt_id: str,
        resource_receipt_body: Mapping[str, object],
        decision_id: DecisionId,
        prepared: PreparedTerminalProjection,
    ) -> BundleValidationResult:
        campaign_root = self._campaign_run_root(campaign_id)
        destination = campaign_root / "final" / "submission-bundle"
        existing: BundleValidationResult | None = None
        unsealed_orphan = False
        if os.path.lexists(destination):
            destination_metadata = destination.lstat()
            unsealed_orphan = (
                not stat.S_ISLNK(destination_metadata.st_mode)
                and stat.S_ISDIR(destination_metadata.st_mode)
                and stat.S_IMODE(destination_metadata.st_mode) == 0o700
            )
            if not unsealed_orphan and stat.S_IMODE(destination_metadata.st_mode) != 0o555:
                raise LabConflictError(
                    "existing bundle destination is neither sealed nor an exact orphan candidate"
                )
        if os.path.lexists(destination) and not unsealed_orphan:
            existing = self.validate_bundle(destination)
            if existing.contract_id != CONTRACT_ID.value:
                raise LabConflictError("existing bundle belongs to a different ContractId")
            if existing.campaign_id != campaign_id.value:
                raise LabConflictError("existing bundle belongs to a different CampaignId")
            if existing.decision_id != decision_id.value:
                raise LabConflictError("existing bundle belongs to a different DecisionId")
            if existing.resource_receipt_id != resource_receipt_id:
                raise LabConflictError("existing bundle has a different resource receipt")
            if existing.preparation_id != prepared.preparation_id:
                raise LabConflictError("existing bundle has a different terminal preparation")
            if existing.projection_sha256 != prepared.projection_sha256:
                raise LabConflictError("existing bundle has a different terminal projection")
        evidence_root = campaign_root / "evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        repository.materialize_prepared_terminal_projection(
            preparation_id=prepared.preparation_id,
            snapshot_destination=evidence_root / EvidenceRole.CAMPAIGN_STATE_SNAPSHOT.value,
            event_export_destination=evidence_root / EvidenceRole.EVENT_EXPORT.value,
        )
        submission_path = evidence_root / EvidenceRole.SUBMISSION.value
        if submission_path.exists():
            submission = read_submission(submission_path, _FIXTURE_ALIGNMENT)
            if submission.prediction_digest != execution.result.prediction_sha256:
                raise LabConflictError("existing scripted submission has different predictions")
        else:
            submission = write_submission(
                submission_path,
                _FIXTURE_ALIGNMENT,
                execution.result.predictions,
            )
        terminal_projection = TerminalProjectionBinding(
            preparation_id=prepared.preparation_id,
            projection_sha256=prepared.projection_sha256,
            campaign_revision=prepared.source.campaign_revision,
            last_event_seq=prepared.source.last_event_seq,
            schema_version=prepared.projection_schema_version,
            redaction_policy_version=prepared.redaction_policy_version,
        )
        selection = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID.value,
            "campaign_id": campaign_id.value,
            "decision_id": decision_id.value,
            "selected_prediction_id": execution.result.prediction_id.value,
            "fallback_prediction_id": execution.result.prediction_id.value,
            "submission_disposition": SCRIPTED_DISPOSITION,
            "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
            "protected_metrics_evaluated": False,
            "official_fm_qualified": False,
            "terminal_projection": terminal_projection.manifest(),
        }
        scientific = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID.value,
            "campaign_id": campaign_id.value,
            "decision_id": decision_id.value,
            "selected_prediction_id": execution.result.prediction_id.value,
            "fallback_prediction_id": execution.result.prediction_id.value,
            "scientific_disposition": ScientificDisposition.INSUFFICIENT_VALID_EVIDENCE.value,
            "reason": "offline scripted fixture has no labels or protected metrics",
            "material_improvement_claimed": False,
            "terminal_projection": terminal_projection.manifest(),
        }
        _write_json_exact(
            evidence_root / EvidenceRole.CONTRACT_MANIFEST.value, CONTRACT_MANIFEST.manifest()
        )
        _write_json_exact(
            evidence_root / EvidenceRole.CAMPAIGN_MANIFEST.value,
            {
                "schema_version": 1,
                "campaign_id": campaign_id.value,
                "contract_id": CONTRACT_ID.value,
                "decision_id": decision_id.value,
                "campaign_kind": CAMPAIGN_KIND,
                "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
                "startup_receipt": self._startup_receipt.manifest(),
                "resource_profile": profile.manifest(),
                "terminal_projection": terminal_projection.manifest(),
            },
        )
        _write_json_exact(evidence_root / EvidenceRole.SELECTION_EVIDENCE.value, selection)
        _write_json_exact(evidence_root / EvidenceRole.SCIENTIFIC_DECISION.value, scientific)
        _write_json_exact(evidence_root / EvidenceRole.SUBMISSION_DECISION.value, selection)
        _write_json_exact(
            evidence_root / EvidenceRole.REPLAY_RECEIPT.value,
            execution.replay_receipt.manifest(),
        )
        _write_json_lines_exact(
            evidence_root / EvidenceRole.RESOURCE_RECEIPTS.value,
            [dict(resource_receipt_body) | {"receipt_id": resource_receipt_id}],
        )
        _write_json_exact(
            evidence_root / EvidenceRole.PROTECTED_QUERY_ACCOUNTING.value,
            {"schema_version": 1, "authorized": False, "query_count": 0, "queries": []},
        )
        _write_json_exact(
            evidence_root / EvidenceRole.PROVIDER_ACCOUNTING.value,
            {"schema_version": 1, "provider_enabled": False, "operation_count": 0},
        )
        _write_json_exact(
            evidence_root / EvidenceRole.FAILURE_SUMMARY.value,
            {"schema_version": 1, "failure_count": 0, "failures": []},
        )
        _write_exact(
            evidence_root / EvidenceRole.REPORT.value,
            b"# Offline scripted campaign evidence\n\n"
            b"This bundle verifies the new deterministic trainer/state/replay/finalization "
            b"plumbing only. It did not access full KuaiRand data, final-period outcomes, a "
            b"provider, or protected scoring. It does not qualify the official FM and makes "
            b"no score-improvement claim.\n",
        )
        receipts = tuple(
            FrozenFileReceipt.capture(role, evidence_root / role.value)
            for role in REQUIRED_EVIDENCE_ROLES
        )
        request = BundleFinalizationRequest(
            destination=destination,
            contract_id=CONTRACT_ID,
            campaign_id=campaign_id,
            selected_prediction_id=execution.result.prediction_id,
            terminal_projection=terminal_projection,
            receipts=receipts,
        )
        if existing is None:
            try:
                published = BundleFinalizer().finalize(request)
                if published.replay_grade.grade is not ReplayGrade.BUNDLE_EXACT:
                    raise BundleValidationError("bundle finalizer did not prove BUNDLE_EXACT")
            except BundleProjectionError:
                if not os.path.lexists(destination):
                    raise
                destination_metadata = destination.lstat()
                if (
                    not stat.S_ISLNK(destination_metadata.st_mode)
                    and stat.S_ISDIR(destination_metadata.st_mode)
                    and stat.S_IMODE(destination_metadata.st_mode) == 0o700
                ):
                    raise
                existing = self.validate_bundle(destination)
                _prove_existing_bundle_regeneration(request, expected=existing)
        else:
            _prove_existing_bundle_regeneration(request, expected=existing)
        validation = self.validate_bundle(destination)
        if validation.contract_id != CONTRACT_ID.value:
            raise LabConflictError("published bundle belongs to a different ContractId")
        if validation.campaign_id != campaign_id.value:
            raise LabConflictError("published bundle belongs to a different CampaignId")
        if validation.decision_id != decision_id.value:
            raise LabConflictError("published bundle belongs to a different DecisionId")
        if validation.resource_receipt_id != resource_receipt_id:
            raise LabConflictError("published bundle has a different resource receipt")
        if validation.preparation_id != prepared.preparation_id:
            raise LabConflictError("published bundle has a different terminal preparation")
        if validation.projection_sha256 != prepared.projection_sha256:
            raise LabConflictError("published bundle has a different terminal projection")
        if validation.selected_prediction_id != execution.result.prediction_id.value:
            raise LabConflictError("published bundle selects a different PredictionId")
        if validation.submission_sha256 != submission.submission_digest:
            raise BundleValidationError("published submission differs from validated source")
        return validation

    def _result_from_snapshot(self, snapshot: Mapping[str, object]) -> CampaignResult:
        campaign = _mapping(snapshot.get("campaign"), "campaign")
        entities = _mapping(snapshot.get("entities"), "entities")
        bundles = _mapping_list(entities.get("bundles"), "entities.bundles")
        decisions = _mapping_list(
            entities.get("selection_decisions"), "entities.selection_decisions"
        )
        resources = _mapping_list(entities.get("resource_receipts"), "entities.resource_receipts")
        if len(bundles) != 1:
            raise LabError("terminal scripted campaign must contain exactly one bundle")
        if len(decisions) != 1 or len(resources) != 1:
            raise LabError("terminal scripted campaign has incomplete decision/resource lineage")
        bundle_record = bundles[0]
        decision_record = decisions[0]
        resource_record = resources[0]
        payload = _mapping(bundle_record.get("payload"), "bundle payload")
        metrics = payload.get("exact_metrics")
        if metrics is not None:
            raise LabError("scripted campaign unexpectedly contains protected exact metrics")
        replay_grades = _string_tuple(payload.get("replay_grades"), "replay_grades")
        campaign_id = _string(campaign.get("campaign_id"), "campaign_id")
        contract_id = _string(campaign.get("contract_id"), "contract_id")
        selected_prediction_id = _string(
            campaign.get("selected_prediction_id"), "selected_prediction_id"
        )
        fallback_prediction_id = _string(
            campaign.get("fallback_prediction_id"), "fallback_prediction_id"
        )
        bundle_path = Path(_string(payload.get("bundle_path"), "bundle_path"))
        validation = self.validate_bundle(bundle_path)
        expected_lineage = (
            contract_id,
            campaign_id,
            selected_prediction_id,
            validation.bundle_id,
            validation.decision_id,
            validation.resource_receipt_id,
        )
        observed_lineage = (
            _string(bundle_record.get("contract_id"), "bundle.contract_id"),
            _string(bundle_record.get("campaign_id"), "bundle.campaign_id"),
            _string(bundle_record.get("selected_prediction_id"), "bundle.selected_prediction_id"),
            _string(bundle_record.get("bundle_id"), "bundle.bundle_id"),
            _string(decision_record.get("decision_id"), "decision_id"),
            _string(resource_record.get("receipt_id"), "resource_receipt_id"),
        )
        if observed_lineage != expected_lineage:
            raise LabError("sealed bundle differs from authoritative campaign lineage")
        if bundle_record.get("manifest_sha256") != validation.manifest_sha256:
            raise LabError("bundle payload differs from exact manifest digest")
        if payload.get("submission_sha256") != validation.submission_sha256:
            raise LabError("bundle payload differs from exact submission digest")
        return CampaignResult(
            campaign_id=campaign_id,
            contract_id=contract_id,
            terminal_state=_string(campaign.get("state"), "terminal_state"),
            submission_disposition=_string(
                payload.get("submission_disposition"), "submission_disposition"
            ),
            scientific_disposition=_string(
                payload.get("scientific_disposition"), "scientific_disposition"
            ),
            selected_prediction_id=selected_prediction_id,
            fallback_prediction_id=fallback_prediction_id,
            exact_metrics=None,
            bundle_path=bundle_path,
            bundle_sha256=_string(bundle_record.get("bundle_id"), "bundle_id"),
            replay_grade=_string(payload.get("replay_grade"), "replay_grade"),
            resource_receipt_id=_string(payload.get("resource_receipt_id"), "resource_receipt_id"),
            protected_query_count=cast(int, payload.get("protected_query_count")),
            evidence_manifest_path=bundle_path / "bundle-manifest.json",
            replay_grades=replay_grades,
            campaign_kind=_string(payload.get("campaign_kind"), "campaign_kind"),
            qualification_scope=_string(payload.get("qualification_scope"), "qualification_scope"),
        )


def _execution_entity(
    entities: Mapping[str, object],
    *,
    collection: str,
    identity_key: str,
    identity: str,
) -> Mapping[str, object]:
    matches = [
        row
        for row in _mapping_list(entities.get(collection), f"entities.{collection}")
        if row.get(identity_key) == identity
    ]
    if len(matches) != 1:
        raise LabConflictError(
            f"scripted {identity_key} must identify exactly one durable {collection} record"
        )
    return matches[0]


def _validate_execution_state_lineage(
    row: Mapping[str, object],
    *,
    campaign_id: CampaignId,
    entity: str,
) -> None:
    if row.get("contract_id") != CONTRACT_ID.value or row.get("campaign_id") != campaign_id.value:
        raise LabConflictError(f"existing scripted {entity} has different lineage")


def _require_initial_execution_state(row: Mapping[str, object], *, entity: str) -> None:
    if row.get("revision") != 0 or row.get("terminal") is not False:
        raise LabConflictError(
            f"scripted {entity} initial state is not deterministically resumable"
        )


def _trial_request_manifest(request: TrialRequest) -> dict[str, object]:
    return {
        "experiment_id": request.experiment_id.value,
        "trial_id": request.trial_id.value,
        "attempt_id": request.attempt_id.value,
        "trainer_identity": request.trainer_identity.manifest(),
        "seed": request.seed,
        "fold": request.fold,
        "fidelity": request.fidelity,
        "qualified_settings": dict(request.qualified_settings),
        "infrastructure_attempt": request.infrastructure_attempt,
        "ordered_row_ids": list(request.ordered_row_ids),
        "resource_limits": request.resource_limits.manifest(),
    }


def _stable_trial_result_sha256(result: TrialResult) -> str:
    """Hash deterministic result evidence while excluding observed timing/resource variation."""

    return canonical_json_sha256(
        {
            "trial_id": result.trial_id.value,
            "attempt_id": result.attempt_id.value,
            "prediction_id": result.prediction_id.value,
            "prediction_sha256": result.prediction_sha256,
            "ordered_row_ids": list(result.ordered_row_ids),
            "data": result.data.manifest(),
            "features": result.features.manifest(),
            "model": result.model.manifest(),
            "environment": result.environment.manifest(),
            "diagnostics": dict(result.diagnostics),
        }
    )


def _scripted_resource_evidence_from_manifest(
    resources: Mapping[str, object],
) -> dict[str, object]:
    """Expose only measurements persisted by the first completed scripted prediction."""

    return {
        "wall_seconds": resources.get("wall_seconds"),
        "wall_seconds_measured": True,
        "cpu_seconds": resources.get("cpu_seconds"),
        "cpu_seconds_measured": resources.get("cpu_seconds_measured"),
        "peak_rss_bytes": resources.get("peak_rss_bytes"),
        "peak_rss_bytes_measured": True,
        "peak_disk_bytes": None,
        "peak_disk_bytes_measured": False,
        "peak_process_count": resources.get("peak_process_count"),
        "peak_process_count_measured": True,
        "threads": None,
        "threads_measured": False,
        "declared_thread_limit": resources.get("threads"),
        "device": resources.get("device"),
    }


def _scripted_resource_receipt_body_from_manifest(
    *,
    campaign_id: CampaignId,
    execution: _ScriptedExecution,
    result_manifest: Mapping[str, object],
    profile: ResourceProfile,
) -> dict[str, object]:
    expected_result_lineage = (
        execution.trial_id.value,
        execution.attempt_id.value,
        execution.result.prediction_id.value,
        execution.result.prediction_sha256,
    )
    observed_result_lineage = tuple(
        result_manifest.get(name)
        for name in ("trial_id", "attempt_id", "prediction_id", "prediction_sha256")
    )
    if observed_result_lineage != expected_result_lineage:
        raise LabConflictError("persisted scripted trial result has different lineage")
    trainer = _mapping(result_manifest.get("trainer_identity"), "trial result trainer_identity")
    resources = _mapping(result_manifest.get("resources"), "trial result resources")
    timing = _mapping(result_manifest.get("timing"), "trial result timing")
    if (
        trainer.get("backend") != execution.result.trainer_identity.backend
        or trainer.get("device") != execution.result.trainer_identity.device
    ):
        raise LabConflictError("persisted scripted trainer identity differs from the request")
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID.value,
        "campaign_id": campaign_id.value,
        "prediction_id": execution.result.prediction_id.value,
        "attempt_id": execution.attempt_id.value,
        "campaign_kind": CAMPAIGN_KIND,
        "qualification_scope": SCRIPTED_QUALIFICATION_SCOPE,
        "declared_resource_profile": profile.manifest(),
        "actual_trainer_backend": trainer.get("backend"),
        "actual_trainer_device": trainer.get("device"),
        "observed_resources": _scripted_resource_evidence_from_manifest(resources),
        "timing": dict(timing),
        "preferred_backend_qualified": False,
        "official_fm_qualified": False,
        "full_data_qualified": False,
    }


def _write_json_exact(path: Path, value: object) -> None:
    _write_exact(path, canonical_json_bytes(value) + b"\n")


def _write_json_lines_exact(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        payload = b"{}\n"
    else:
        payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _write_exact(path, payload)


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise LabConflictError(f"existing artifact differs from idempotent content: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            if path.read_bytes() != payload:
                raise LabConflictError(
                    f"concurrent artifact differs from idempotent content: {path}"
                ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _regular_file_digest(path: Path) -> tuple[str, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BundleValidationError(f"bundle member is unavailable: {path.name}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BundleValidationError(f"bundle member must be a regular non-symlink: {path.name}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise BundleValidationError(f"bundle member is unreadable: {path.name}") from exc
    if size != metadata.st_size:
        raise BundleValidationError(f"bundle member changed while validating: {path.name}")
    return digest.hexdigest(), size


def _bundle_inventory(root: Path) -> tuple[str, int]:
    entries: list[dict[str, object]] = []
    total_size = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        sha256, size_bytes = _regular_file_digest(path)
        entries.append({"path": path.name, "sha256": sha256, "size_bytes": size_bytes})
        total_size += size_bytes
    return canonical_json_sha256(entries), total_size


def _prove_existing_bundle_regeneration(
    request: BundleFinalizationRequest,
    *,
    expected: BundleValidationResult,
) -> None:
    """Freshly double-regenerate an existing publication before retaining BUNDLE_EXACT."""

    verification_root = Path(
        tempfile.mkdtemp(prefix=".bundle-reverification-", dir=request.destination.parent)
    )
    try:
        regenerated = BundleFinalizer().finalize(
            BundleFinalizationRequest(
                destination=verification_root / "submission-bundle",
                contract_id=request.contract_id,
                campaign_id=request.campaign_id,
                selected_prediction_id=request.selected_prediction_id,
                terminal_projection=request.terminal_projection,
                receipts=request.receipts,
            )
        )
        if regenerated.replay_grade.grade is not ReplayGrade.BUNDLE_EXACT:
            raise BundleValidationError("clean regeneration did not prove BUNDLE_EXACT")
        observed = (
            regenerated.bundle_id,
            regenerated.manifest_sha256,
            regenerated.submission_sha256,
            regenerated.inventory_sha256,
            regenerated.file_count,
            regenerated.total_size_bytes,
        )
        expected_values = (
            expected.bundle_id,
            expected.manifest_sha256,
            expected.submission_sha256,
            expected.inventory_sha256,
            expected.file_count,
            expected.total_size_bytes,
        )
        if observed != expected_values:
            raise BundleValidationError("clean regeneration differs from published bundle")
    finally:
        if verification_root.exists():
            for child in sorted(
                verification_root.rglob("*"), key=lambda path: len(path.parts), reverse=True
            ):
                if not child.is_symlink():
                    os.chmod(child, 0o700 if child.is_dir() else 0o600)
            os.chmod(verification_root, 0o700)
            shutil.rmtree(verification_root)


def _validate_terminal_projection_evidence(
    root: Path,
    *,
    terminal_projection: Mapping[str, object],
    preparation_id: str,
    projection_sha256: str,
    contract_id: str,
    campaign_id: str,
    decision_id: str,
    resource_receipt_id: str,
    selected_prediction_id: str,
) -> None:
    snapshot_path = root / EvidenceRole.CAMPAIGN_STATE_SNAPSHOT.value
    try:
        snapshot = sqlite3.connect(f"file:{snapshot_path.as_posix()}?mode=ro", uri=True)
        metadata = snapshot.execute(
            """
            SELECT preparation_id, projection_sha256, projection_schema_version,
                   redaction_policy_version, projection_json
            FROM projection_metadata WHERE singleton = 1
            """
        ).fetchone()
        integrity = snapshot.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        raise BundleValidationError("terminal SQLite projection is invalid") from exc
    finally:
        if "snapshot" in locals():
            snapshot.close()
    expected_metadata = (
        preparation_id,
        projection_sha256,
        terminal_projection.get("schema_version"),
        terminal_projection.get("redaction_policy_version"),
    )
    if metadata is None or tuple(metadata[:4]) != expected_metadata or integrity != ("ok",):
        raise BundleValidationError("terminal SQLite projection differs from bundle manifest")
    projection_json = str(metadata[4])
    if hashlib.sha256(projection_json.encode()).hexdigest() != projection_sha256:
        raise BundleValidationError("terminal projection content digest differs")
    try:
        projection_value = json.loads(projection_json)
    except json.JSONDecodeError as exc:
        raise BundleValidationError("terminal projection contains invalid JSON") from exc
    projection = _mapping(projection_value, "terminal projection")
    projected_campaign = _mapping(projection.get("campaign"), "terminal projection campaign")
    if (
        projected_campaign.get("contract_id") != contract_id
        or projected_campaign.get("campaign_id") != campaign_id
        or projected_campaign.get("selected_prediction_id") != selected_prediction_id
        or projected_campaign.get("state") != "COMPLETED_OFFLINE_FIXTURE"
        or projected_campaign.get("terminal") is not True
    ):
        raise BundleValidationError("terminal projection differs from committed campaign lineage")
    projected_terminal = _mapping(
        projection.get("terminal_projection"), "terminal projection policy"
    )
    if projected_terminal.get("schema_version") != terminal_projection.get("schema_version"):
        raise BundleValidationError("terminal projection schema differs from bundle binding")
    if projected_terminal.get("source_revision") != terminal_projection.get("source_revision"):
        raise BundleValidationError("terminal projection horizon differs from bundle binding")
    redaction = _mapping(projected_terminal.get("redaction_policy"), "redaction policy")
    if redaction.get("version") != terminal_projection.get("redaction_policy_version"):
        raise BundleValidationError("terminal projection redaction differs from bundle binding")
    projected_entities = _mapping(projection.get("entities"), "terminal projection entities")
    decisions = _mapping_list(
        projected_entities.get("selection_decisions"), "terminal projection decisions"
    )
    resources = _mapping_list(
        projected_entities.get("resource_receipts"), "terminal projection resources"
    )
    if not any(
        row.get("decision_id") == decision_id
        and row.get("contract_id") == contract_id
        and row.get("campaign_id") == campaign_id
        and row.get("selected_prediction_id") == selected_prediction_id
        for row in decisions
    ):
        raise BundleValidationError("terminal projection lacks the exact selection decision")
    if not any(
        row.get("receipt_id") == resource_receipt_id
        and row.get("contract_id") == contract_id
        and row.get("campaign_id") == campaign_id
        for row in resources
    ):
        raise BundleValidationError("terminal projection lacks the exact resource receipt")
    events = projection.get("events")
    if not isinstance(events, list):
        raise BundleValidationError("terminal projection lacks its event horizon")
    expected_export = b"".join(canonical_json_bytes(event) + b"\n" for event in events)
    try:
        observed_export = (root / EvidenceRole.EVENT_EXPORT.value).read_bytes()
    except OSError as exc:
        raise BundleValidationError("terminal event export is unreadable") from exc
    if observed_export != expected_export:
        raise BundleValidationError("terminal event export differs from SQLite projection")


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"invalid JSON evidence: {path.name}") from exc
    return _mapping(value, path.name)


def _read_json_lines(path: Path) -> list[Mapping[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BundleValidationError(f"invalid JSONL evidence: {path.name}") from exc
    if not lines:
        raise BundleValidationError(f"JSONL evidence is empty: {path.name}")
    rows: list[Mapping[str, object]] = []
    for index, line in enumerate(lines):
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleValidationError(f"invalid JSONL evidence: {path.name}:{index + 1}") from exc
        rows.append(_mapping(decoded, f"{path.name}:{index + 1}"))
    return rows


def _require_lineage(
    value: Mapping[str, object],
    *,
    contract_id: str,
    campaign_id: str,
    location: str,
) -> None:
    if value.get("contract_id") != contract_id:
        raise BundleValidationError(f"{location} differs from ContractId lineage")
    if value.get("campaign_id") != campaign_id:
        raise BundleValidationError(f"{location} differs from CampaignId lineage")


def _file_sha256(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256")
    except OSError as exc:
        raise LabAdmissionError(f"required file is unavailable: {path}") from exc
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BundleValidationError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _single_line(value: object, location: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\n" in value
        or "\x00" in value
    ):
        raise LabAdmissionError(f"{location} must be non-empty normalized single-line text")
    return value


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise LabError(f"{location} must be an object")
    return cast(Mapping[str, object], value)


def _mapping_list(value: object, location: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise LabError(f"{location} must be a list")
    return [_mapping(item, f"{location}[{index}]") for index, item in enumerate(value)]


def _string(value: object, location: str) -> str:
    if type(value) is not str or not value:
        raise LabError(f"{location} must be non-empty text")
    return value


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        raise LabError(f"{location} must be a list of non-empty strings")
    return tuple(cast(list[str], value))


__all__ = [
    "AutonomousExperimentLab",
    "BundleValidationError",
    "BundleValidationResult",
    "CampaignOptions",
    "CampaignResult",
    "LabAdmissionError",
    "LabConflictError",
    "LabError",
    "ReplayResult",
]
