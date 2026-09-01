"""Recorded generated-source lineages used by the Pure campaign.

The production campaign must still cross the same research-model boundary when live provider
credentials are unavailable.  This module therefore feeds one fully recorded LambdaRank response
through :class:`ScriptedResearchModel`, the generated-package materializer, static policy, and the
material executable-change gate.  The returned source is content-addressed and can be executed by
the generated candidate runner without trusting or importing it in the controller process.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from kuairand_agent.execution.artifacts import (
    ArtifactError,
    ArtifactKind,
    ArtifactRef,
    ArtifactStore,
    DirectoryArtifactRef,
    DirectoryEntryRef,
)
from kuairand_agent.execution.candidate_executor import GeneratedCandidateIdentity
from kuairand_agent.research.context import SafeResearchContext
from kuairand_agent.research.interface import ResearchModel
from kuairand_agent.research.materialize import (
    CandidateMaterializationError,
    CandidateStaticError,
    MaterialChangeEvidence,
    MaterializedCandidate,
    describe_materialized_candidate,
    materialize_candidate,
    require_material_executable_change,
    snapshot_materialized_candidate,
    validate_candidate_static,
    validate_model_generated_overlay,
)
from kuairand_agent.research.schemas import (
    FailureCategory,
    GeneratedFile,
    GeneratedPackage,
    ImplementationRequest,
    ParentSnapshot,
    ParentSourceFile,
    Proposal,
    ProposalRequest,
    RejectedPackageSnapshot,
    RepairRequest,
    RequiredField,
    ResearchOperation,
    canonical_digest,
    canonical_json_bytes,
    parse_json_object,
)
from kuairand_agent.research.scripted import (
    ScriptedCall,
    ScriptedResearchModel,
    ScriptedResponse,
)
from kuairand_agent.research.source_policy import CandidateManifestPolicyError

SCRIPTED_PRODUCTION_SCHEMA_VERSION: Final = 1
SCRIPTED_CANDIDATE_ID: Final = "generated-causal-lambdarank-v1"
SCRIPTED_PARENT_ID: Final = "official-fm-fallback-seed-4"
_TEMPLATE_MEMBERS: Final = ("README.md", "candidate.py", "config.json")
_PARENT_SOURCE: Final = """\
def train_model(features, targets, user_groups, config, seed):
    raise NotImplementedError("official FM parent does not implement generated tree training")


def predict_scores(features, checkpoint_text):
    return [0.0 for _ in range(len(features))]
"""


class ProductionResearchError(RuntimeError):
    """The frozen scripted production lineage cannot be constructed exactly."""


_FAILURE_DIAGNOSTIC_LIMIT: Final = 2_000
_FAILURE_SUBJECT_LIMIT: Final = 160
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


def _bounded_failure_text(value: str, *, maximum: int) -> str:
    normalized = " ".join(
        "".join(character if 32 <= ord(character) < 127 else " " for character in value).split()
    )
    return normalized[:maximum] or "unspecified"


@dataclass(frozen=True, slots=True)
class ResearchFailureObservation:
    """Bounded, normalized evidence for one failed live-research admission stage."""

    stage: str
    category: str
    code: str
    subject: str
    fingerprint: str
    diagnostic: str

    def __post_init__(self) -> None:
        for name in ("stage", "category", "code"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value) > 64:
                raise ProductionResearchError(f"failure observation {name} is invalid")
        if type(self.subject) is not str or not self.subject or len(self.subject) > 160:
            raise ProductionResearchError("failure observation subject is invalid")
        if type(self.fingerprint) is not str or _SHA256_RE.fullmatch(self.fingerprint) is None:
            raise ProductionResearchError("failure observation fingerprint is invalid")
        if (
            type(self.diagnostic) is not str
            or not self.diagnostic
            or len(self.diagnostic) > _FAILURE_DIAGNOSTIC_LIMIT
        ):
            raise ProductionResearchError("failure observation diagnostic is invalid")

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        category: str,
        code: str,
        subject: str,
        diagnostic: str,
    ) -> ResearchFailureObservation:
        bounded_stage = _bounded_failure_text(stage, maximum=64)
        bounded_category = _bounded_failure_text(category, maximum=64)
        bounded_code = _bounded_failure_text(code, maximum=64)
        bounded_subject = _bounded_failure_text(subject, maximum=_FAILURE_SUBJECT_LIMIT)
        bounded_diagnostic = _bounded_failure_text(diagnostic, maximum=_FAILURE_DIAGNOSTIC_LIMIT)
        fingerprint = canonical_digest(
            {
                "schema_version": 1,
                "stage": bounded_stage,
                "category": bounded_category,
                "code": bounded_code,
                "subject": bounded_subject,
            }
        )
        return cls(
            stage=bounded_stage,
            category=bounded_category,
            code=bounded_code,
            subject=bounded_subject,
            fingerprint=fingerprint,
            diagnostic=bounded_diagnostic,
        )

    def to_wire(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "category": self.category,
            "code": self.code,
            "subject": self.subject,
            "fingerprint": self.fingerprint,
            "diagnostic": self.diagnostic,
        }


def _quoted_failure_subject(diagnostic: str) -> str:
    match = re.search(r"'([^'\r\n]{1,160})'", diagnostic)
    return match.group(1) if match is not None else "candidate"


def _observe_candidate_failure(
    error: CandidateMaterializationError | CandidateStaticError,
) -> ResearchFailureObservation:
    diagnostic = str(error)
    lowered = diagnostic.lower()
    subject = _quoted_failure_subject(diagnostic)
    if isinstance(error, CandidateMaterializationError):
        stage = "materialization"
        category = "static_policy"
        if "reserved candidate filename" in lowered:
            code = "reserved_filename"
        elif "suffix is not allowed" in lowered:
            code = "unsupported_suffix"
        elif "path" in lowered and "unsafe" in lowered:
            code = "unsafe_path"
        elif "byte limit" in lowered:
            code = "package_byte_limit"
        elif "file-count limit" in lowered:
            code = "package_file_limit"
        else:
            code = "materialization_rejected"
    elif "material symbol" in lowered or "material executable-source" in lowered:
        stage = "materiality"
        category = "materiality"
        code = "declared_symbol_unchanged" if "did not change" in lowered else "no_material_change"
    else:
        stage = "static_validation"
        category = "static_policy"
        if "forbidden import" in lowered:
            code = "forbidden_import"
        elif "forbidden call" in lowered:
            code = "forbidden_call"
        elif "invalid python" in lowered:
            code = "invalid_python"
        elif "invalid json" in lowered:
            code = "invalid_json"
        elif "entry point" in lowered:
            code = "missing_entrypoint"
        else:
            code = "static_validation_rejected"
    return ResearchFailureObservation.create(
        stage=stage,
        category=category,
        code=code,
        subject=subject,
        diagnostic=diagnostic,
    )


def proposal_family_of(proposal: Proposal) -> str:
    scientific_text = " ".join(
        (proposal.objective, proposal.mechanism, proposal.principal_change)
    ).lower()
    families = (
        ("pairwise", ("pairwise", "bpr")),
        ("listwise", ("listwise", "lambdarank", "lambda rank")),
        ("duration-bucket", ("duration bucket", "duration-bucket")),
        ("user-balanced", ("user balanced", "user-balanced")),
    )
    for family, markers in families:
        if any(marker in scientific_text for marker in markers):
            return family
    normalized = re.sub(r"[^a-z0-9]+", "-", proposal.objective.lower()).strip("-")
    return normalized[:64] or "unknown"


def _proposal_signature(proposal: Proposal) -> str:
    return canonical_digest(
        {
            "schema_version": 1,
            "parent_candidate_id": proposal.parent_candidate_id,
            "principal_change": proposal.principal_change,
            "objective": proposal.objective,
            "sampling": proposal.sampling,
            "grouping": proposal.grouping,
            "weighting": proposal.weighting,
            "files_expected": list(proposal.files_expected),
            "required_fields": [field.to_wire() for field in proposal.required_fields],
        }
    )


def _proposal_family_is_blocked(safe_context: SafeResearchContext, *, proposal_family: str) -> bool:
    records = safe_context.to_wire().get("campaign_records", [])
    if not isinstance(records, list):
        return False
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            continue
        values = raw_record.get("values")
        if (
            isinstance(values, Mapping)
            and values.get("proposal_family_blocked") is True
            and values.get("proposal_family") == proposal_family
        ):
            return True
    return False


class LiveResearchBranchRejected(ProductionResearchError):
    """One generated live-research branch exhausted its bounded repair allowance."""

    def __init__(
        self,
        *,
        failed_candidate_id: str,
        repairs_attempted: int,
        diagnostic: str,
        root_failure: ResearchFailureObservation | None = None,
        terminal_failure: ResearchFailureObservation | None = None,
        proposal_family: str = "unknown",
        proposal_signature: str = "",
    ) -> None:
        legacy_failure = ResearchFailureObservation.create(
            stage="research_admission",
            category="research_admission",
            code="branch_rejected",
            subject=failed_candidate_id,
            diagnostic=diagnostic,
        )
        root = root_failure or terminal_failure or legacy_failure
        terminal = terminal_failure or root
        super().__init__(terminal.diagnostic)
        self.failed_candidate_id = failed_candidate_id
        self.repairs_attempted = repairs_attempted
        self.diagnostic = terminal.diagnostic
        self.root_failure = root
        self.terminal_failure = terminal
        self.proposal_family = _bounded_failure_text(proposal_family, maximum=64)
        if proposal_signature and _SHA256_RE.fullmatch(proposal_signature) is None:
            raise ProductionResearchError("proposal signature must be SHA-256 when present")
        self.proposal_signature = proposal_signature


def load_parent_snapshot(parent_dir: Path, *, candidate_id: str) -> ParentSnapshot:
    """Load one bounded, symlink-free candidate protocol tree as a typed parent snapshot."""

    try:
        root = parent_dir.resolve(strict=True)
        root_metadata = root.lstat()
    except OSError as exc:
        raise ProductionResearchError("candidate parent directory is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ProductionResearchError("candidate parent must be a real directory")
    observed = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts
    )
    if not 1 <= len(observed) <= 12:
        raise ProductionResearchError("candidate parent must contain between 1 and 12 files")
    files: list[ParentSourceFile] = []
    for path in observed:
        relative = path.relative_to(root).as_posix()
        try:
            metadata = path.lstat()
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProductionResearchError(
                f"candidate parent member {relative} is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ProductionResearchError(f"candidate parent member {relative} must be a real file")
        if Path(relative).suffix not in {".py", ".json", ".md"}:
            raise ProductionResearchError(
                f"candidate parent member {relative} has an unsupported suffix"
            )
        files.append(ParentSourceFile.create(relative, content))
    paths = {item.path for item in files}
    if not {"candidate.py", "config.json"}.issubset(paths):
        raise ProductionResearchError("candidate parent requires candidate.py and config.json")
    return ParentSnapshot(candidate_id=candidate_id, files=tuple(files))


def _template_bytes(template_dir: Path) -> dict[str, bytes]:
    try:
        source_metadata = template_dir.lstat()
        if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(source_metadata.st_mode):
            raise ProductionResearchError("LambdaRank template must be a real directory")
        root = template_dir.resolve(strict=True)
        metadata = root.lstat()
    except OSError as exc:
        raise ProductionResearchError("LambdaRank template directory is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionResearchError("LambdaRank template must be a real directory")
    observed = tuple(path.relative_to(root).as_posix() for path in sorted(root.rglob("*")))
    if observed != _TEMPLATE_MEMBERS:
        raise ProductionResearchError(
            "LambdaRank template must contain exactly README.md, candidate.py, and config.json"
        )
    result: dict[str, bytes] = {}
    for name in _TEMPLATE_MEMBERS:
        path = root / name
        try:
            member = path.lstat()
            payload = path.read_bytes()
        except OSError as exc:
            raise ProductionResearchError(f"cannot read LambdaRank template member {name}") from exc
        if stat.S_ISLNK(member.st_mode) or not stat.S_ISREG(member.st_mode) or not payload:
            raise ProductionResearchError(f"LambdaRank template member {name} is not a real file")
        result[name] = payload
    return result


def _text(payload: bytes, name: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProductionResearchError(f"LambdaRank template member {name} is not UTF-8") from exc


def _proposal() -> Proposal:
    return Proposal(
        proposal_id="causal-lambdarank-v1",
        hypothesis=(
            "Stable user-grouped LambdaRank over leakage-safe causal history and current-item "
            "features will improve long-view GAUC and top-five ranking quality."
        ),
        mechanism=(
            "Fit deterministic CPU LambdaRank on a private stable grouping permutation while "
            "preserving canonical prediction order; optionally fuse only with a Fold-B-selected "
            "official-FM rank weight."
        ),
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id=SCRIPTED_PARENT_ID,
        principal_change=(
            "Replace the constant generated parent with executable deterministic grouped-tree "
            "training and checkpoint inference."
        ),
        files_expected=_TEMPLATE_MEMBERS,
        required_fields=(
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:user_id",
                "private_grouping",
                "Build stable ranking groups inside the candidate workspace.",
            ),
            RequiredField(
                "trusted/causal_feature_matrix:aggregate_and_current_item_features",
                "training_and_inference_input",
                "Use only controller-built leakage-safe numeric capabilities.",
            ),
            RequiredField(
                "data/log_standard_4_08_to_4_21_pure.csv:long_view",
                "train_only_target",
                "Train the ranking objective without exposing validation labels.",
            ),
        ),
        objective="Deterministic binary-gain LambdaRank aligned to nDCG@5 and user ranking.",
        sampling="All impressions in each train-derived prefix; no public-label sampling.",
        grouping="Stable first-seen user order with stable canonical order within each user.",
        weighting="Uniform logged impressions and binary gains; no external propensity data.",
        causal_cutoff="Every aggregate uses only rows earlier than the query event.",
        estimated_runtime_seconds=600,
        estimated_memory_mb=4096,
        smoke_plan="Train and replay a small multi-user numeric capability in a fresh process.",
        inner_fold_plan=(
            "Screen on Fold B, freeze tree and fusion policy, then confirm once on Fold A."
        ),
        falsification_criteria=(
            "Reject non-determinism, replay mismatch, failed gates, or non-positive inner evidence."
        ),
        promotion_criteria=(
            "Require structural gates, both temporal folds, and matched public seeds 0, 1, and 2."
        ),
        maximum_repairs=1,
        rollback_parent_id=SCRIPTED_PARENT_ID,
        attributions=(
            "LightGBM LambdaRank objective",
            "organizer GAUC and nDCG@5 benchmark contract",
            "controller-owned causal feature policy",
        ),
    )


def _parent(config_text: str) -> ParentSnapshot:
    return ParentSnapshot(
        candidate_id=SCRIPTED_PARENT_ID,
        files=(
            ParentSourceFile.create("candidate.py", _PARENT_SOURCE),
            ParentSourceFile.create("config.json", config_text),
        ),
    )


@dataclass(frozen=True, slots=True)
class _ExpectedScript:
    parent: ParentSnapshot
    proposal_request: ProposalRequest
    proposal: Proposal
    implementation_request: ImplementationRequest
    package: GeneratedPackage
    transcript_bytes: bytes
    calls: tuple[ScriptedCall, ...]


def _transcript_bytes(
    proposal_request: ProposalRequest,
    proposal: Proposal,
    implementation_request: ImplementationRequest,
    package: GeneratedPackage,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": SCRIPTED_PRODUCTION_SCHEMA_VERSION,
            "provider": "scripted",
            "calls": [
                {
                    "operation": ResearchOperation.PROPOSE.value,
                    "request": proposal_request.to_wire(),
                    "response": proposal.to_wire(),
                },
                {
                    "operation": ResearchOperation.IMPLEMENT.value,
                    "request": implementation_request.to_wire(),
                    "response": package.to_wire(),
                },
            ],
        }
    )


def _expected_script(
    *,
    campaign_id: str,
    scientific_iteration: int,
    members: dict[str, bytes],
    safe_context: SafeResearchContext,
) -> _ExpectedScript:
    config_text = _text(members["config.json"], "config.json")
    parent = _parent(config_text)
    proposal = _proposal()
    implementation_request_id = f"{SCRIPTED_CANDIDATE_ID}-implement-{scientific_iteration}"
    package = GeneratedPackage(
        request_id=implementation_request_id,
        response_id=f"{SCRIPTED_CANDIDATE_ID}-source-{scientific_iteration}",
        files=tuple(GeneratedFile(name, _text(members[name], name)) for name in _TEMPLATE_MEMBERS),
        material_change_summary=(
            "Implement stable grouped deterministic LightGBM LambdaRank train and predict source."
        ),
        material_symbols=("train_model", "predict_scores"),
    )
    context = safe_context.to_wire()
    proposal_request = ProposalRequest.create(
        request_id=f"{SCRIPTED_CANDIDATE_ID}-propose-{scientific_iteration}",
        campaign_id=campaign_id,
        scientific_iteration=scientific_iteration,
        parent_candidate_id=parent.candidate_id,
        safe_context=context,
    )
    implementation_request = ImplementationRequest.create(
        request_id=implementation_request_id,
        proposal=proposal,
        parent=parent,
        safe_context=context,
    )
    calls = (
        ScriptedCall(
            ResearchOperation.PROPOSE,
            proposal_request.digest,
            proposal.digest,
        ),
        ScriptedCall(
            ResearchOperation.IMPLEMENT,
            implementation_request.digest,
            package.digest,
        ),
    )
    return _ExpectedScript(
        parent=parent,
        proposal_request=proposal_request,
        proposal=proposal,
        implementation_request=implementation_request,
        package=package,
        transcript_bytes=_transcript_bytes(
            proposal_request,
            proposal,
            implementation_request,
            package,
        ),
        calls=calls,
    )


def _describe_expected_candidate(
    parent: ParentSnapshot,
    package: GeneratedPackage,
    destination: Path,
) -> MaterializedCandidate:
    """Recompute the materializer's normalized logical result without filesystem writes."""

    return describe_materialized_candidate(parent, package, destination)


def _verify_exact_generated_tree(candidate: MaterializedCandidate) -> None:
    destination = candidate.destination
    try:
        root_metadata = destination.lstat()
    except OSError as exc:
        raise ProductionResearchError("existing generated candidate is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ProductionResearchError("existing generated candidate must be a real directory")
    if root_metadata.st_mode & 0o222:
        raise ProductionResearchError("existing generated candidate directory must be read-only")

    expected = {value.path: value for value in candidate.files}
    observed: dict[str, Path] = {}
    try:
        for path in sorted(destination.rglob("*")):
            relative = path.relative_to(destination).as_posix()
            observed[relative] = path
    except OSError as exc:
        raise ProductionResearchError("existing generated candidate cannot be inventoried") from exc
    if set(observed) != set(expected):
        raise ProductionResearchError(
            "existing generated candidate must contain exactly the declared source files"
        )
    for relative, reference in expected.items():
        path = observed[relative]
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ProductionResearchError(
                f"existing generated member {relative} cannot be inspected"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ProductionResearchError(
                f"existing generated member {relative} must be a real non-symlink file"
            )
        if metadata.st_nlink != 1 or metadata.st_mode & 0o222:
            raise ProductionResearchError(
                f"existing generated member {relative} must be private and read-only"
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProductionResearchError(
                f"existing generated member {relative} cannot be read"
            ) from exc
        if payload != reference.content.encode("utf-8"):
            raise ProductionResearchError(
                f"existing generated member {relative} does not match its recorded response"
            )


def _artifact_ref(payload: bytes, kind: ArtifactKind) -> ArtifactRef:
    return ArtifactRef(hashlib.sha256(payload).hexdigest(), len(payload), kind)


@dataclass(frozen=True, slots=True)
class _ExpectedArtifacts:
    source_snapshot: DirectoryArtifactRef
    config_artifact: ArtifactRef
    diff_artifact: ArtifactRef
    transcript_artifact: ArtifactRef


def _expected_artifacts(
    candidate: MaterializedCandidate,
    transcript_bytes: bytes,
) -> _ExpectedArtifacts:
    entries = tuple(
        DirectoryEntryRef(
            value.path,
            _artifact_ref(value.content.encode("utf-8"), ArtifactKind.SOURCE),
        )
        for value in candidate.files
    )
    total_size = sum(entry.artifact.size_bytes for entry in entries)
    placeholder = ArtifactRef("0" * 64, 0, ArtifactKind.MANIFEST)
    draft = DirectoryArtifactRef(ArtifactKind.SOURCE, placeholder, entries, total_size)
    manifest_bytes = canonical_json_bytes(draft.manifest())
    source_snapshot = DirectoryArtifactRef(
        ArtifactKind.SOURCE,
        _artifact_ref(manifest_bytes, ArtifactKind.MANIFEST),
        entries,
        total_size,
    )
    config_artifact = next(entry.artifact for entry in entries if entry.path == "config.json")
    return _ExpectedArtifacts(
        source_snapshot=source_snapshot,
        config_artifact=config_artifact,
        diff_artifact=_artifact_ref(candidate.unified_diff.encode("utf-8"), ArtifactKind.SOURCE),
        transcript_artifact=_artifact_ref(transcript_bytes, ArtifactKind.LOG),
    )


@dataclass(frozen=True, slots=True)
class ScriptedLambdaRankLineage:
    """Complete provider, request/response, source-diff, and executable identity evidence."""

    provider: str
    live_provider_used: bool
    parent: ParentSnapshot
    proposal_request: ProposalRequest
    proposal: Proposal
    implementation_request: ImplementationRequest
    package: GeneratedPackage
    materialized: MaterializedCandidate
    material_change: MaterialChangeEvidence
    source_snapshot: DirectoryArtifactRef
    config_artifact: ArtifactRef
    diff_artifact: ArtifactRef
    transcript_artifact: ArtifactRef
    identity: GeneratedCandidateIdentity
    model_calls: tuple[ScriptedCall, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCRIPTED_PRODUCTION_SCHEMA_VERSION,
            "provider": self.provider,
            "live_provider_used": self.live_provider_used,
            "candidate_id": SCRIPTED_CANDIDATE_ID,
            "parent_id": self.parent.candidate_id,
            "parent_digest": self.parent.digest,
            "proposal_request_digest": self.proposal_request.digest,
            "proposal_digest": self.proposal.digest,
            "implementation_request_digest": self.implementation_request.digest,
            "package_digest": self.package.digest,
            "source_digest": self.materialized.source_digest,
            "source_snapshot_sha256": self.source_snapshot.sha256,
            "config_sha256": self.config_artifact.sha256,
            "diff_sha256": self.diff_artifact.sha256,
            "transcript_sha256": self.transcript_artifact.sha256,
            "material_changed_symbols": list(self.material_change.changed_symbols),
            "material_reachable_files": list(self.material_change.reachable_python_files),
            "model_calls": [
                {
                    "operation": call.operation.value,
                    "request_digest": call.request_digest,
                    "response_digest": call.response_digest,
                }
                for call in self.model_calls
            ],
        }


def prepare_scripted_lambdarank_lineage(
    *,
    campaign_id: str,
    scientific_iteration: int,
    template_dir: Path,
    generated_root: Path,
    artifact_store: ArtifactStore,
    safe_context: SafeResearchContext,
) -> ScriptedLambdaRankLineage:
    """Prepare or strictly rehydrate the one deterministic generated-source lineage.

    A first preparation always crosses :class:`ScriptedResearchModel`.  If the deterministic
    destination already exists, no model or persistence method is called: the source tree and all
    content-addressed source, diff, and transcript evidence must already be complete and exact.
    Any partial or drifted state fails closed so a retry cannot silently bless different source.
    """

    if not isinstance(artifact_store, ArtifactStore):
        raise ProductionResearchError("artifact_store must be ArtifactStore")
    if not isinstance(safe_context, SafeResearchContext):
        raise ProductionResearchError("safe_context must be SafeResearchContext")
    if type(scientific_iteration) is not int or not 1 <= scientific_iteration <= 50:
        raise ProductionResearchError("scientific_iteration must be in [1, 50]")
    members = _template_bytes(template_dir)
    expected = _expected_script(
        campaign_id=campaign_id,
        scientific_iteration=scientific_iteration,
        members=members,
        safe_context=safe_context,
    )
    destination = generated_root / SCRIPTED_CANDIDATE_ID
    if generated_root.exists() or generated_root.is_symlink():
        try:
            generated_root_metadata = generated_root.lstat()
        except OSError as exc:
            raise ProductionResearchError("generated-source root cannot be inspected") from exc
        if stat.S_ISLNK(generated_root_metadata.st_mode) or not stat.S_ISDIR(
            generated_root_metadata.st_mode
        ):
            raise ProductionResearchError("generated-source root must be a real directory")
    destination_exists = destination.exists() or destination.is_symlink()
    if destination_exists:
        candidate = _describe_expected_candidate(
            expected.parent,
            expected.package,
            destination,
        )
        try:
            _verify_exact_generated_tree(candidate)
            material_change = require_material_executable_change(expected.parent, candidate)
            artifacts = _expected_artifacts(candidate, expected.transcript_bytes)
            artifact_store.verify_directory(artifacts.source_snapshot)
            for entry, value in zip(
                artifacts.source_snapshot.entries,
                candidate.files,
                strict=True,
            ):
                if artifact_store.read_bytes(entry.artifact) != value.content.encode("utf-8"):
                    raise ProductionResearchError(
                        f"recorded source artifact {entry.path} does not match generated source"
                    )
            if artifact_store.read_bytes(artifacts.diff_artifact) != candidate.unified_diff.encode(
                "utf-8"
            ):
                raise ProductionResearchError("recorded source diff evidence does not match")
            if (
                artifact_store.read_bytes(artifacts.transcript_artifact)
                != expected.transcript_bytes
            ):
                raise ProductionResearchError(
                    "recorded scripted transcript evidence does not match"
                )
        except ProductionResearchError:
            raise
        except (
            ArtifactError,
            CandidateMaterializationError,
            CandidateStaticError,
            OSError,
        ) as exc:
            raise ProductionResearchError(
                "existing scripted lineage evidence is incomplete or invalid"
            ) from exc
        identity = GeneratedCandidateIdentity(
            source_snapshot=artifacts.source_snapshot,
            source_digest=candidate.source_digest,
            config_digest=artifacts.config_artifact.sha256,
        )
        return ScriptedLambdaRankLineage(
            provider="scripted",
            live_provider_used=False,
            parent=expected.parent,
            proposal_request=expected.proposal_request,
            proposal=expected.proposal,
            implementation_request=expected.implementation_request,
            package=expected.package,
            materialized=candidate,
            material_change=material_change,
            source_snapshot=artifacts.source_snapshot,
            config_artifact=artifacts.config_artifact,
            diff_artifact=artifacts.diff_artifact,
            transcript_artifact=artifacts.transcript_artifact,
            identity=identity,
            model_calls=expected.calls,
        )

    model = ScriptedResearchModel(
        (
            ScriptedResponse(ResearchOperation.PROPOSE, expected.proposal),
            ScriptedResponse(ResearchOperation.IMPLEMENT, expected.package),
        )
    )
    generated_proposal = model.propose(expected.proposal_request)
    implementation_request = ImplementationRequest.create(
        request_id=expected.implementation_request.request_id,
        proposal=generated_proposal,
        parent=expected.parent,
        safe_context=safe_context.to_wire(),
    )
    generated_package = model.implement(implementation_request)

    try:
        generated_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_metadata = generated_root.lstat()
    except OSError as exc:
        raise ProductionResearchError("cannot create generated-source root") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ProductionResearchError("generated-source root must be a real directory")
    materialized = materialize_candidate(
        expected.parent,
        generated_package,
        destination,
    )
    material_change = require_material_executable_change(expected.parent, materialized)
    source_snapshot = artifact_store.put_directory(
        materialized.destination,
        kind=ArtifactKind.SOURCE,
    )
    by_path = {entry.path: entry.artifact for entry in source_snapshot.entries}
    config_artifact = by_path.get("config.json")
    if config_artifact is None:
        raise ProductionResearchError("generated source snapshot lost config.json")
    diff_artifact = artifact_store.put_bytes(
        materialized.unified_diff.encode("utf-8"),
        kind=ArtifactKind.SOURCE,
    )
    transcript_artifact = artifact_store.put_bytes(
        _transcript_bytes(
            expected.proposal_request,
            generated_proposal,
            implementation_request,
            generated_package,
        ),
        kind=ArtifactKind.LOG,
    )
    identity = GeneratedCandidateIdentity(
        source_snapshot=source_snapshot,
        source_digest=materialized.source_digest,
        config_digest=config_artifact.sha256,
    )
    return ScriptedLambdaRankLineage(
        provider="scripted",
        live_provider_used=False,
        parent=expected.parent,
        proposal_request=expected.proposal_request,
        proposal=generated_proposal,
        implementation_request=implementation_request,
        package=generated_package,
        materialized=materialized,
        material_change=material_change,
        source_snapshot=source_snapshot,
        config_artifact=config_artifact,
        diff_artifact=diff_artifact,
        transcript_artifact=transcript_artifact,
        identity=identity,
        model_calls=model.calls,
    )


@dataclass(frozen=True, slots=True)
class LiveResearchLineage:
    """Complete durable evidence for one live-provider proposal and implementation."""

    provider: str
    live_provider_used: bool
    candidate_id: str
    parent: ParentSnapshot
    proposal_request: ProposalRequest
    proposal: Proposal
    implementation_request: ImplementationRequest
    package: GeneratedPackage
    materialized: MaterializedCandidate
    material_change: MaterialChangeEvidence
    source_snapshot: DirectoryArtifactRef
    config_artifact: ArtifactRef
    diff_artifact: ArtifactRef
    transcript_artifact: ArtifactRef
    identity: GeneratedCandidateIdentity
    model_calls: tuple[ScriptedCall, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCRIPTED_PRODUCTION_SCHEMA_VERSION,
            "provider": self.provider,
            "live_provider_used": self.live_provider_used,
            "candidate_id": self.candidate_id,
            "parent_id": self.parent.candidate_id,
            "parent_digest": self.parent.digest,
            "proposal_request_digest": self.proposal_request.digest,
            "proposal_digest": self.proposal.digest,
            "implementation_request_digest": self.implementation_request.digest,
            "package_digest": self.package.digest,
            "source_digest": self.materialized.source_digest,
            "source_snapshot_sha256": self.source_snapshot.sha256,
            "config_sha256": self.config_artifact.sha256,
            "diff_sha256": self.diff_artifact.sha256,
            "transcript_sha256": self.transcript_artifact.sha256,
            "material_changed_symbols": list(self.material_change.changed_symbols),
            "material_reachable_files": list(self.material_change.reachable_python_files),
            "model_calls": [
                {
                    "operation": call.operation.value,
                    "request_digest": call.request_digest,
                    "response_digest": call.response_digest,
                }
                for call in self.model_calls
            ],
        }


def _live_transcript_bytes(
    *,
    provider: str,
    proposal_request: ProposalRequest,
    proposal: Proposal,
    implementation_request: ImplementationRequest,
    implementation_package: GeneratedPackage,
    repair_calls: tuple[tuple[RepairRequest, GeneratedPackage], ...],
) -> bytes:
    calls: list[dict[str, object]] = [
        {
            "operation": ResearchOperation.PROPOSE.value,
            "request": proposal_request.to_wire(),
            "response": proposal.to_wire(),
        },
        {
            "operation": ResearchOperation.IMPLEMENT.value,
            "request": implementation_request.to_wire(),
            "response": implementation_package.to_wire(),
        },
    ]
    calls.extend(
        {
            "operation": ResearchOperation.REPAIR.value,
            "request": request.to_wire(),
            "response": response.to_wire(),
        }
        for request, response in repair_calls
    )
    return canonical_json_bytes(
        {
            "schema_version": SCRIPTED_PRODUCTION_SCHEMA_VERSION,
            "provider": provider,
            "live_provider_used": True,
            "calls": calls,
        }
    )


def _write_live_record(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
        path.chmod(0o400)
    except OSError as exc:
        raise ProductionResearchError("cannot persist live research lineage record") from exc


def prepare_or_rehydrate_live_lineage(
    *,
    campaign_id: str,
    scientific_iteration: int,
    parent: ParentSnapshot,
    generated_root: Path,
    artifact_store: ArtifactStore,
    safe_context: SafeResearchContext,
    model: ResearchModel,
    provider: str = "openai",
) -> LiveResearchLineage:
    """Generate or exactly rehydrate one provider-created immutable candidate source tree."""

    if provider != "openai":
        raise ProductionResearchError("live lineage requires the OpenAI provider")
    if not isinstance(parent, ParentSnapshot):
        raise ProductionResearchError("live lineage parent must be ParentSnapshot")
    if not isinstance(safe_context, SafeResearchContext):
        raise ProductionResearchError("safe_context must be SafeResearchContext")
    if not isinstance(model, ResearchModel):
        raise ProductionResearchError("live lineage model must implement ResearchModel")
    if type(scientific_iteration) is not int or not 1 <= scientific_iteration <= 50:
        raise ProductionResearchError("scientific_iteration must be in [1, 50]")
    try:
        generated_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_metadata = generated_root.lstat()
    except OSError as exc:
        raise ProductionResearchError("cannot create generated-source root") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ProductionResearchError("generated-source root must be a real directory")

    record_path = generated_root / f"iteration-{scientific_iteration:02d}-lineage.json"
    proposal_request_id = f"iteration-{scientific_iteration:02d}-propose"
    implementation_request_id = f"iteration-{scientific_iteration:02d}-implement"
    rehydrating = record_path.exists() or record_path.is_symlink()
    repair_calls: list[tuple[RepairRequest, GeneratedPackage]] = []
    if rehydrating:
        try:
            record_metadata = record_path.lstat()
            raw = parse_json_object(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ProductionResearchError("live research lineage record is unreadable") from exc
        if stat.S_ISLNK(record_metadata.st_mode) or not stat.S_ISREG(record_metadata.st_mode):
            raise ProductionResearchError("live research lineage record must be a real file")
        expected_fields = {
            "schema_version",
            "provider",
            "campaign_id",
            "scientific_iteration",
            "parent_digest",
            "safe_context_digest",
            "candidate_id",
            "proposal",
            "implementation_package",
            "repair_calls",
            "package",
        }
        if set(raw) != expected_fields:
            raise ProductionResearchError("live research lineage record fields are not exact")
        if (
            raw["schema_version"] != SCRIPTED_PRODUCTION_SCHEMA_VERSION
            or raw["provider"] != provider
            or raw["campaign_id"] != campaign_id
            or raw["scientific_iteration"] != scientific_iteration
            or raw["parent_digest"] != parent.digest
            or raw["safe_context_digest"] != safe_context.digest
        ):
            raise ProductionResearchError("live research lineage identity differs on resume")
        proposal_raw = raw["proposal"]
        implementation_package_raw = raw["implementation_package"]
        repair_calls_raw = raw["repair_calls"]
        package_raw = raw["package"]
        if (
            not isinstance(proposal_raw, dict)
            or not isinstance(implementation_package_raw, dict)
            or not isinstance(repair_calls_raw, list)
            or not isinstance(package_raw, dict)
        ):
            raise ProductionResearchError("live research lineage responses are malformed")
        proposal = Proposal.from_mapping(cast(dict[str, object], proposal_raw))
        implementation_package = GeneratedPackage.from_mapping(
            cast(dict[str, object], implementation_package_raw)
        )
        for index, call_raw in enumerate(repair_calls_raw):
            if not isinstance(call_raw, dict) or set(call_raw) != {"request", "response"}:
                raise ProductionResearchError(f"live research repair call {index} is malformed")
            request_raw = call_raw["request"]
            response_raw = call_raw["response"]
            if not isinstance(request_raw, dict) or not isinstance(response_raw, dict):
                raise ProductionResearchError(f"live research repair call {index} is malformed")
            repair_calls.append(
                (
                    RepairRequest.from_mapping(cast(dict[str, object], request_raw)),
                    GeneratedPackage.from_mapping(cast(dict[str, object], response_raw)),
                )
            )
        package = GeneratedPackage.from_mapping(cast(dict[str, object], package_raw))
        candidate_id_raw = raw["candidate_id"]
        if type(candidate_id_raw) is not str:
            raise ProductionResearchError("live research candidate identity is malformed")
        candidate_id = candidate_id_raw
    else:
        proposal_request = ProposalRequest.create(
            request_id=proposal_request_id,
            campaign_id=campaign_id,
            scientific_iteration=scientific_iteration,
            parent_candidate_id=parent.candidate_id,
            safe_context=safe_context.to_wire(),
        )
        proposal = model.propose(proposal_request)
        candidate_id = f"candidate-{scientific_iteration:02d}-{proposal.digest[:16]}"
        proposal_family = proposal_family_of(proposal)
        try:
            proposal_request.source_policy.validate_manifest(
                proposal.files_expected,
                require_final_entrypoint=True,
            )
        except CandidateManifestPolicyError as exc:
            failure = ResearchFailureObservation.create(
                stage="proposal_admission",
                category="static_policy",
                code="candidate_source_policy",
                subject=exc.fingerprint,
                diagnostic=str(exc),
            )
            raise LiveResearchBranchRejected(
                failed_candidate_id=candidate_id,
                repairs_attempted=0,
                diagnostic=str(exc),
                root_failure=failure,
                terminal_failure=failure,
                proposal_family=proposal_family,
                proposal_signature=_proposal_signature(proposal),
            ) from exc
        if _proposal_family_is_blocked(safe_context, proposal_family=proposal_family):
            diagnostic = (
                f"proposal family {proposal_family!r} is blocked by prior admission evidence"
            )
            failure = ResearchFailureObservation.create(
                stage="proposal_admission",
                category="novelty_policy",
                code="proposal_family_blocked",
                subject=proposal_family,
                diagnostic=diagnostic,
            )
            raise LiveResearchBranchRejected(
                failed_candidate_id=candidate_id,
                repairs_attempted=0,
                diagnostic=diagnostic,
                root_failure=failure,
                terminal_failure=failure,
                proposal_family=proposal_family,
                proposal_signature=_proposal_signature(proposal),
            )
        implementation_request = ImplementationRequest.create(
            request_id=implementation_request_id,
            proposal=proposal,
            parent=parent,
            safe_context=safe_context.to_wire(),
        )
        implementation_package = model.implement(implementation_request)
        package = implementation_package

    proposal_request = ProposalRequest.create(
        request_id=proposal_request_id,
        campaign_id=campaign_id,
        scientific_iteration=scientific_iteration,
        parent_candidate_id=parent.candidate_id,
        safe_context=safe_context.to_wire(),
    )
    if proposal.parent_candidate_id != parent.candidate_id:
        raise ProductionResearchError("live proposal names a different parent")
    implementation_request = ImplementationRequest.create(
        request_id=implementation_request_id,
        proposal=proposal,
        parent=parent,
        safe_context=safe_context.to_wire(),
    )
    if implementation_package.request_id != implementation_request.request_id:
        raise ProductionResearchError("live generated package names a different request")
    causative_package = implementation_package
    base_candidate_id = candidate_id.rsplit("-repair-", 1)[0] if repair_calls else candidate_id
    for index, (repair_request, repair_package) in enumerate(repair_calls, start=1):
        rejected_package = repair_request.rejected_package
        expected_failed_candidate_id = (
            base_candidate_id if index == 1 else f"{base_candidate_id}-repair-{index - 1}"
        )
        if (
            repair_request.proposal_id != proposal.proposal_id
            or repair_request.safe_context_digest != safe_context.digest
            or repair_request.request_id != f"iteration-{scientific_iteration:02d}-repair-{index}"
            or repair_request.failed_candidate_id != expected_failed_candidate_id
            or repair_package.request_id != repair_request.request_id
            or repair_request.remaining_repairs != min(proposal.maximum_repairs, 2) - index + 1
            or rejected_package is None
            or rejected_package.to_generated_package() != causative_package
        ):
            raise ProductionResearchError("live research repair lineage is inconsistent")
        causative_package = repair_package
    expected_final_package = repair_calls[-1][1] if repair_calls else implementation_package
    if package != expected_final_package:
        raise ProductionResearchError("live research final package differs from its call lineage")

    candidate: MaterializedCandidate
    if rehydrating:
        destination = generated_root / candidate_id
        try:
            validate_model_generated_overlay(package)
        except CandidateMaterializationError as exc:
            raise ProductionResearchError(
                "persisted live package violates the runtime contract"
            ) from exc
        candidate = _describe_expected_candidate(parent, package, destination)
        _verify_exact_generated_tree(candidate)
        validate_candidate_static(candidate)
        material_change = require_material_executable_change(parent, candidate)
    else:
        maximum_repairs = min(proposal.maximum_repairs, 2)
        base_candidate_id = candidate_id
        root_failure: ResearchFailureObservation | None = None
        while True:
            destination = generated_root / candidate_id
            attempted_candidate: MaterializedCandidate | None = None
            try:
                validate_model_generated_overlay(package)
                attempted_candidate = materialize_candidate(parent, package, destination)
                candidate = attempted_candidate
                validate_candidate_static(candidate)
                material_change = require_material_executable_change(parent, candidate)
                break
            except (CandidateMaterializationError, CandidateStaticError) as exc:
                observed_failure = _observe_candidate_failure(exc)
                if root_failure is None:
                    root_failure = observed_failure
                if len(repair_calls) >= maximum_repairs:
                    raise LiveResearchBranchRejected(
                        failed_candidate_id=candidate_id,
                        repairs_attempted=len(repair_calls),
                        diagnostic=str(exc),
                        root_failure=root_failure,
                        terminal_failure=observed_failure,
                        proposal_family=proposal_family_of(proposal),
                        proposal_signature=_proposal_signature(proposal),
                    ) from exc
                failed_child = (
                    snapshot_materialized_candidate(attempted_candidate, candidate_id=candidate_id)
                    if attempted_candidate is not None
                    else parent
                )
                repair_number = len(repair_calls) + 1
                repair_request = RepairRequest.create(
                    request_id=(f"iteration-{scientific_iteration:02d}-repair-{repair_number}"),
                    proposal_id=proposal.proposal_id,
                    failed_candidate_id=candidate_id,
                    failed_child=failed_child,
                    failure_category=FailureCategory.STATIC_POLICY,
                    diagnostics=str(exc),
                    remaining_repairs=maximum_repairs - len(repair_calls),
                    safe_context=safe_context.to_wire(),
                    rejected_package=RejectedPackageSnapshot.from_generated_package(package),
                )
                repair_package = model.repair(repair_request)
                if repair_package.request_id != repair_request.request_id:
                    raise ProductionResearchError(
                        "live repaired package names a different request"
                    ) from exc
                repair_calls.append((repair_request, repair_package))
                package = repair_package
                candidate_id = f"{base_candidate_id}-repair-{repair_number}"

        pending_record_payload = canonical_json_bytes(
            {
                "schema_version": SCRIPTED_PRODUCTION_SCHEMA_VERSION,
                "provider": provider,
                "campaign_id": campaign_id,
                "scientific_iteration": scientific_iteration,
                "parent_digest": parent.digest,
                "safe_context_digest": safe_context.digest,
                "candidate_id": candidate_id,
                "proposal": proposal.to_wire(),
                "implementation_package": implementation_package.to_wire(),
                "repair_calls": [
                    {"request": request.to_wire(), "response": response.to_wire()}
                    for request, response in repair_calls
                ],
                "package": package.to_wire(),
            }
        )
        _write_live_record(record_path, pending_record_payload)
    source_snapshot = artifact_store.put_directory(destination, kind=ArtifactKind.SOURCE)
    by_path = {entry.path: entry.artifact for entry in source_snapshot.entries}
    config_artifact = by_path.get("config.json")
    if config_artifact is None:
        raise ProductionResearchError("live generated source snapshot lost config.json")
    diff_artifact = artifact_store.put_bytes(
        candidate.unified_diff.encode("utf-8"), kind=ArtifactKind.SOURCE
    )
    transcript_bytes = _live_transcript_bytes(
        provider=provider,
        proposal_request=proposal_request,
        proposal=proposal,
        implementation_request=implementation_request,
        implementation_package=implementation_package,
        repair_calls=tuple(repair_calls),
    )
    transcript_artifact = artifact_store.put_bytes(transcript_bytes, kind=ArtifactKind.LOG)
    identity = GeneratedCandidateIdentity(
        source_snapshot=source_snapshot,
        source_digest=candidate.source_digest,
        config_digest=config_artifact.sha256,
    )
    return LiveResearchLineage(
        provider=provider,
        live_provider_used=True,
        candidate_id=candidate_id,
        parent=parent,
        proposal_request=proposal_request,
        proposal=proposal,
        implementation_request=implementation_request,
        package=package,
        materialized=candidate,
        material_change=material_change,
        source_snapshot=source_snapshot,
        config_artifact=config_artifact,
        diff_artifact=diff_artifact,
        transcript_artifact=transcript_artifact,
        identity=identity,
        model_calls=(
            ScriptedCall(ResearchOperation.PROPOSE, proposal_request.digest, proposal.digest),
            ScriptedCall(
                ResearchOperation.IMPLEMENT,
                implementation_request.digest,
                implementation_package.digest,
            ),
            *(
                ScriptedCall(ResearchOperation.REPAIR, request.digest, response.digest)
                for request, response in repair_calls
            ),
        ),
    )


# Explicit public name for orchestration code that wants to document the restart-safe behavior.
prepare_or_rehydrate_scripted_lambdarank_lineage = prepare_scripted_lambdarank_lineage


__all__ = [
    "SCRIPTED_CANDIDATE_ID",
    "SCRIPTED_PARENT_ID",
    "SCRIPTED_PRODUCTION_SCHEMA_VERSION",
    "LiveResearchBranchRejected",
    "LiveResearchLineage",
    "ProductionResearchError",
    "ResearchFailureObservation",
    "ScriptedLambdaRankLineage",
    "load_parent_snapshot",
    "prepare_or_rehydrate_live_lineage",
    "prepare_or_rehydrate_scripted_lambdarank_lineage",
    "prepare_scripted_lambdarank_lineage",
    "proposal_family_of",
]
