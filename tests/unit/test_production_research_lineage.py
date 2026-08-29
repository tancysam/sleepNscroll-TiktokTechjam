from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from kuairand_agent.execution.artifacts import ArtifactStore
from kuairand_agent.research.context import (
    AggregateRecord,
    ResearchBudgetContext,
    SafeResearchContext,
    build_safe_research_context,
)
from kuairand_agent.research.production import (
    LiveResearchBranchRejected,
    ProductionResearchError,
    ResearchFailureObservation,
    load_parent_snapshot,
    prepare_or_rehydrate_live_lineage,
    prepare_scripted_lambdarank_lineage,
)
from kuairand_agent.research.schemas import (
    FailureCategory,
    GeneratedFile,
    GeneratedPackage,
    ImplementationRequest,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
    RejectedPackageSnapshot,
    RepairRequest,
    RequiredField,
)
from kuairand_agent.research.scripted import ScriptedResearchModel

ROOT = Path(__file__).parents[2]


def _safe_context() -> SafeResearchContext:
    return build_safe_research_context(
        starter_manifest_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        capability_manifests=(),
        budgets=ResearchBudgetContext(44, 18_000, 1, 0),
    )


def _unexpected_model_call(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise AssertionError("rehydration must not invoke the research model")


class _LiveFixtureModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def propose(self, request: ProposalRequest) -> Proposal:
        self.calls.append("propose")
        return Proposal(
            proposal_id="live-fixture-v1",
            hypothesis="A deterministic executable marker verifies live material generation.",
            mechanism="Add one reachable Python function without changing the command contract.",
            expected_metric_effects=("GAUC", "nDCG@5"),
            parent_candidate_id=request.parent_candidate_id,
            principal_change="Add a reachable executable function to candidate.py.",
            files_expected=("candidate.py",),
            required_fields=(
                RequiredField("trusted/features", "inference_input", "Use approved features."),
            ),
            objective="binary long-view ranking",
            sampling="logged train impressions",
            grouping="user impression groups",
            weighting="uniform",
            causal_cutoff="No validation or future outcome is used.",
            estimated_runtime_seconds=1,
            estimated_memory_mb=64,
            smoke_plan="Run the unchanged deterministic candidate protocol.",
            inner_fold_plan="Screen Fold B, then confirm Fold A.",
            falsification_criteria="Reject failed static or protected metric gates.",
            promotion_criteria="Require the frozen selector gates.",
            maximum_repairs=0,
            rollback_parent_id=request.parent_candidate_id,
            attributions=("unit fixture",),
        )

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        self.calls.append("implement")
        parent_source = next(
            item.content for item in request.parent.files if item.path == "candidate.py"
        )
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-source-v1",
            files=(
                GeneratedFile(
                    "candidate.py",
                    parent_source
                    + "\n\ndef live_variant_marker():\n"
                    + "    return 'live-provider-generated'\n",
                ),
            ),
            material_change_summary="Add a reachable live-provider marker function.",
            material_symbols=("live_variant_marker",),
        )

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        raise AssertionError(request)

    def reflect(self, request: ReflectionRequest) -> Reflection:
        raise AssertionError(request)


class _InvalidLiveFixtureModel(_LiveFixtureModel):
    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        self.calls.append("implement")
        parent_source = next(
            item.content for item in request.parent.files if item.path == "candidate.py"
        )
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-forbidden-import-v1",
            files=(
                GeneratedFile(
                    "candidate.py",
                    parent_source
                    + "\n\nimport os\n\ndef invalid_live_variant_marker():\n"
                    + "    return os.getcwd()\n",
                ),
            ),
            material_change_summary="Exercise rejection of a forbidden generated import.",
            material_symbols=("invalid_live_variant_marker",),
        )


class _PolicyInvalidProposalModel(_LiveFixtureModel):
    def propose(self, request: ProposalRequest) -> Proposal:
        return replace(
            super().propose(request),
            files_expected=("baseline.py", "submission.csv"),
        )


class _RepairingReservedFilenameModel(_LiveFixtureModel):
    HELPER_SOURCE = """\
def pairwise_rank_score(positive_score, negative_score):
    return positive_score - negative_score
"""

    def __init__(self) -> None:
        super().__init__()
        self.implementation_package: GeneratedPackage | None = None
        self.repair_requests: list[RepairRequest] = []

    def propose(self, request: ProposalRequest) -> Proposal:
        return replace(super().propose(request), maximum_repairs=1)

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        self.calls.append("implement")
        parent_source = next(
            item.content for item in request.parent.files if item.path == "candidate.py"
        )
        package = GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-reserved-filename-v1",
            files=(
                GeneratedFile(
                    "baseline.py",
                    "from pairwise_helper import pairwise_rank_score\n"
                    "assert pairwise_rank_score(2.0, 1.0) > 0.0\n",
                ),
                GeneratedFile("pairwise_helper.py", self.HELPER_SOURCE),
                GeneratedFile(
                    "candidate.py",
                    parent_source
                    + "\n\nfrom pairwise_helper import pairwise_rank_score\n\n"
                    + "def repaired_live_variant_marker():\n"
                    + "    return pairwise_rank_score(2.0, 1.0)\n",
                ),
            ),
            material_change_summary="Exercise rejection of a reserved candidate filename.",
            material_symbols=("pairwise_rank_score",),
        )
        self.implementation_package = package
        return package

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        self.calls.append("repair")
        self.repair_requests.append(request)
        assert request.failure_category is FailureCategory.STATIC_POLICY
        assert request.remaining_repairs == 1
        assert request.failed_child.candidate_id == "official-fm-fallback-seed-4"
        assert "reserved candidate filename" in request.diagnostics
        assert self.implementation_package is not None
        assert request.rejected_package is not None
        assert request.rejected_package.package_digest == self.implementation_package.digest
        rejected_files = {value.path: value.content for value in request.rejected_package.files}
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-reserved-filename-repaired-v1",
            files=(
                GeneratedFile("candidate.py", rejected_files["candidate.py"]),
                GeneratedFile("pairwise_helper.py", rejected_files["pairwise_helper.py"]),
            ),
            material_change_summary="Preserve the rejected scientific change in legal files.",
            material_symbols=("pairwise_rank_score",),
        )


class _ExhaustingRepairModel(_RepairingReservedFilenameModel):
    def repair(self, request: RepairRequest) -> GeneratedPackage:
        self.calls.append("repair")
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-exhausted-repair-v1",
            files=(
                GeneratedFile(
                    "candidate.py",
                    request.failed_child.file("candidate.py").content,
                ),
            ),
            material_change_summary="Return no executable change.",
            material_symbols=("train_model",),
        )


class _TwiceRepairingReservedFilenameModel(_RepairingReservedFilenameModel):
    def __init__(self) -> None:
        super().__init__()
        self.first_repair_package: GeneratedPackage | None = None

    def propose(self, request: ProposalRequest) -> Proposal:
        return replace(_LiveFixtureModel.propose(self, request), maximum_repairs=2)

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        self.calls.append("repair")
        self.repair_requests.append(request)
        assert request.rejected_package is not None
        rejected_files = {value.path: value.content for value in request.rejected_package.files}
        if len(self.repair_requests) == 1:
            assert self.implementation_package is not None
            assert request.rejected_package.package_digest == self.implementation_package.digest
            package = GeneratedPackage(
                request_id=request.request_id,
                response_id="live-fixture-still-reserved-v1",
                files=(
                    GeneratedFile("baseline.py", rejected_files["baseline.py"]),
                    GeneratedFile("candidate.py", rejected_files["candidate.py"]),
                    GeneratedFile("pairwise_helper.py", rejected_files["pairwise_helper.py"]),
                ),
                material_change_summary="First repair retained the forbidden launcher.",
                material_symbols=("pairwise_rank_score",),
            )
            self.first_repair_package = package
            return package
        assert self.first_repair_package is not None
        assert request.rejected_package.package_digest == self.first_repair_package.digest
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-second-repair-legal-v1",
            files=(
                GeneratedFile("candidate.py", rejected_files["candidate.py"]),
                GeneratedFile("pairwise_helper.py", rejected_files["pairwise_helper.py"]),
            ),
            material_change_summary="Second repair removes the forbidden launcher.",
            material_symbols=("pairwise_rank_score",),
        )


class _RepairingMaterializedPoisonModel(_LiveFixtureModel):
    def __init__(self) -> None:
        super().__init__()
        self.parent_source = ""

    def propose(self, request: ProposalRequest) -> Proposal:
        return replace(super().propose(request), maximum_repairs=1)

    def implement(self, request: ImplementationRequest) -> GeneratedPackage:
        self.calls.append("implement")
        self.parent_source = next(
            item.content for item in request.parent.files if item.path == "candidate.py"
        )
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-materialized-poison-v1",
            files=(
                GeneratedFile(
                    "candidate.py",
                    self.parent_source + "\n\nimport poison_helper\n",
                ),
                GeneratedFile(
                    "poison_helper.py",
                    "import os\n\ndef poisoned_variant():\n    return os.getcwd()\n",
                ),
            ),
            material_change_summary="Generate a materialized child that fails static policy.",
            material_symbols=("poisoned_variant",),
        )

    def repair(self, request: RepairRequest) -> GeneratedPackage:
        self.calls.append("repair")
        assert request.failed_child.file("poison_helper.py")
        assert request.rejected_package is not None
        return GeneratedPackage(
            request_id=request.request_id,
            response_id="live-fixture-materialized-poison-repaired-v1",
            files=(
                GeneratedFile(
                    "candidate.py",
                    self.parent_source
                    + "\n\ndef clean_repair_marker():\n    return 'clean-parent-overlay'\n",
                ),
            ),
            material_change_summary="Apply the safe replacement over the trusted parent.",
            material_symbols=("clean_repair_marker",),
        )


def test_rejected_live_package_does_not_persist_immutable_lineage_record(
    tmp_path: Path,
) -> None:
    model = _InvalidLiveFixtureModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )
    generated = tmp_path / "generated"

    with pytest.raises(LiveResearchBranchRejected, match="forbidden import 'os'") as captured:
        prepare_or_rehydrate_live_lineage(
            campaign_id="live-rejected-campaign",
            scientific_iteration=1,
            parent=parent,
            generated_root=generated,
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            safe_context=_safe_context(),
            model=model,
        )

    assert model.calls == ["propose", "implement"]
    assert not (generated / "iteration-01-lineage.json").exists()
    rejection = captured.value
    assert isinstance(rejection.root_failure, ResearchFailureObservation)
    assert rejection.root_failure == rejection.terminal_failure
    assert rejection.root_failure.stage == "static_validation"
    assert rejection.root_failure.category == "static_policy"
    assert rejection.root_failure.code == "forbidden_import"
    assert rejection.root_failure.subject == "os"
    assert rejection.root_failure.diagnostic == "forbidden import 'os' in 'candidate.py'"
    assert rejection.root_failure.to_wire() == {
        "stage": "static_validation",
        "category": "static_policy",
        "code": "forbidden_import",
        "subject": "os",
        "fingerprint": rejection.root_failure.fingerprint,
        "diagnostic": "forbidden import 'os' in 'candidate.py'",
    }
    assert len(rejection.root_failure.fingerprint) == 64
    assert rejection.proposal_family == "binary-long-view-ranking"
    assert len(rejection.proposal_signature) == 64


def test_live_lineage_rejects_invalid_proposal_manifest_before_implementation(
    tmp_path: Path,
) -> None:
    model = _PolicyInvalidProposalModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )

    with pytest.raises(LiveResearchBranchRejected, match="reserved candidate filename") as captured:
        prepare_or_rehydrate_live_lineage(
            campaign_id="live-invalid-proposal-campaign",
            scientific_iteration=1,
            parent=parent,
            generated_root=tmp_path / "generated",
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            safe_context=_safe_context(),
            model=model,
        )

    assert model.calls == ["propose"]
    assert captured.value.repairs_attempted == 0
    assert captured.value.root_failure.stage == "proposal_admission"
    assert captured.value.root_failure.code == "candidate_source_policy"
    assert captured.value.root_failure.subject == (
        "candidate_path_policy:forbidden_basename:baseline.py"
    )


def test_live_research_branch_rejected_keeps_legacy_diagnostic_contract() -> None:
    rejection = LiveResearchBranchRejected(
        failed_candidate_id="candidate-legacy",
        repairs_attempted=0,
        diagnostic="legacy rejection",
    )

    assert str(rejection) == "legacy rejection"
    assert rejection.diagnostic == "legacy rejection"
    assert rejection.root_failure.diagnostic == "legacy rejection"
    assert rejection.terminal_failure == rejection.root_failure
    assert rejection.proposal_family == "unknown"
    assert rejection.proposal_signature == ""


def test_live_lineage_blocks_excluded_proposal_family_before_implementation(
    tmp_path: Path,
) -> None:
    model = _LiveFixtureModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )
    safe_context = build_safe_research_context(
        starter_manifest_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        capability_manifests=(),
        budgets=ResearchBudgetContext(44, 18_000, 1, 0),
        campaign_records=(
            AggregateRecord(
                "research_branch_rejection",
                {
                    "proposal_family_blocked": True,
                    "proposal_family": "binary-long-view-ranking",
                    "failure_fingerprint": "f" * 64,
                    "failure_count": 2,
                },
            ),
        ),
    )

    with pytest.raises(LiveResearchBranchRejected) as captured:
        prepare_or_rehydrate_live_lineage(
            campaign_id="live-family-block-campaign",
            scientific_iteration=1,
            parent=parent,
            generated_root=tmp_path / "generated",
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            safe_context=safe_context,
            model=model,
        )

    assert model.calls == ["propose"]
    assert captured.value.repairs_attempted == 0
    assert captured.value.proposal_family == "binary-long-view-ranking"
    assert captured.value.root_failure.code == "proposal_family_blocked"
    assert captured.value.root_failure.fingerprint == captured.value.terminal_failure.fingerprint


def test_live_lineage_routes_reserved_filename_through_agent_repair_and_rehydrates(
    tmp_path: Path,
) -> None:
    model = _RepairingReservedFilenameModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    generated = tmp_path / "generated"

    first = prepare_or_rehydrate_live_lineage(
        campaign_id="live-repair-campaign",
        scientific_iteration=1,
        parent=parent,
        generated_root=generated,
        artifact_store=artifacts,
        safe_context=_safe_context(),
        model=model,
    )

    assert model.calls == ["propose", "implement", "repair"]
    assert first.candidate_id.endswith("-repair-1")
    assert [call.operation.value for call in first.model_calls] == [
        "propose",
        "implement",
        "repair",
    ]
    assert first.package.response_id == "live-fixture-reserved-filename-repaired-v1"
    assert {entry.path for entry in first.source_snapshot.entries} == {
        "README.md",
        "candidate.py",
        "config.json",
        "pairwise_helper.py",
    }
    assert len(model.repair_requests) == 1
    assert model.implementation_package is not None
    rejected = model.repair_requests[0].rejected_package
    assert rejected is not None
    assert rejected.to_generated_package() == model.implementation_package
    assert rejected.package_digest == model.implementation_package.digest
    assert first.materialized.parent_digest == parent.digest
    assert first.material_change.changed_symbols == ("pairwise_helper.py:pairwise_rank_score",)
    assert (generated / first.candidate_id / "pairwise_helper.py").read_text(
        encoding="utf-8"
    ) == model.HELPER_SOURCE
    assert not (generated / first.candidate_id / "baseline.py").exists()
    assert not (generated / first.candidate_id.removesuffix("-repair-1")).exists()

    resumed = prepare_or_rehydrate_live_lineage(
        campaign_id="live-repair-campaign",
        scientific_iteration=1,
        parent=parent,
        generated_root=generated,
        artifact_store=artifacts,
        safe_context=_safe_context(),
        model=model,
    )
    assert model.calls == ["propose", "implement", "repair"]
    assert resumed.manifest() == first.manifest()


def test_live_lineage_rejects_rehydrated_repair_with_causal_package_mismatch(
    tmp_path: Path,
) -> None:
    model = _RepairingReservedFilenameModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )
    generated = tmp_path / "generated"
    prepare_or_rehydrate_live_lineage(
        campaign_id="live-repair-causal-campaign",
        scientific_iteration=1,
        parent=parent,
        generated_root=generated,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        safe_context=_safe_context(),
        model=model,
    )
    record_path = generated / "iteration-01-lineage.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    repair_response = GeneratedPackage.from_mapping(record["repair_calls"][0]["response"])
    mismatched = RejectedPackageSnapshot.from_generated_package(repair_response)
    record["repair_calls"][0]["request"]["rejected_package"] = mismatched.to_wire()
    record_path.chmod(0o600)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    record_path.chmod(0o400)

    with pytest.raises(ProductionResearchError, match="repair lineage is inconsistent"):
        prepare_or_rehydrate_live_lineage(
            campaign_id="live-repair-causal-campaign",
            scientific_iteration=1,
            parent=parent,
            generated_root=generated,
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            safe_context=_safe_context(),
            model=model,
        )

    assert model.calls == ["propose", "implement", "repair"]


def test_each_repair_snapshots_the_immediately_causative_package(tmp_path: Path) -> None:
    model = _TwiceRepairingReservedFilenameModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )

    lineage = prepare_or_rehydrate_live_lineage(
        campaign_id="live-two-repair-campaign",
        scientific_iteration=1,
        parent=parent,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        safe_context=_safe_context(),
        model=model,
    )

    assert model.calls == ["propose", "implement", "repair", "repair"]
    assert model.implementation_package is not None
    assert model.first_repair_package is not None
    first_snapshot = model.repair_requests[0].rejected_package
    second_snapshot = model.repair_requests[1].rejected_package
    assert first_snapshot is not None
    assert second_snapshot is not None
    assert [first_snapshot.package_digest, second_snapshot.package_digest] == [
        model.implementation_package.digest,
        model.first_repair_package.digest,
    ]
    assert lineage.candidate_id.endswith("-repair-2")
    assert lineage.materialized.parent_digest == parent.digest


def test_repair_package_is_freshly_applied_over_trusted_parent(tmp_path: Path) -> None:
    model = _RepairingMaterializedPoisonModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )

    lineage = prepare_or_rehydrate_live_lineage(
        campaign_id="live-trusted-parent-repair-campaign",
        scientific_iteration=1,
        parent=parent,
        generated_root=tmp_path / "generated",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        safe_context=_safe_context(),
        model=model,
    )

    assert model.calls == ["propose", "implement", "repair"]
    assert lineage.materialized.parent_digest == parent.digest
    assert "poison_helper.py" not in {entry.path for entry in lineage.source_snapshot.entries}
    assert lineage.material_change.changed_symbols == ("candidate.py:clean_repair_marker",)


def test_live_lineage_exposes_exhausted_repairs_as_one_rejected_research_branch(
    tmp_path: Path,
) -> None:
    model = _ExhaustingRepairModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )

    with pytest.raises(
        LiveResearchBranchRejected,
        match=r"declared material symbol.*did not change",
    ):
        prepare_or_rehydrate_live_lineage(
            campaign_id="live-exhausted-repair-campaign",
            scientific_iteration=1,
            parent=parent,
            generated_root=tmp_path / "generated",
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            safe_context=_safe_context(),
            model=model,
        )

    assert model.calls == ["propose", "implement", "repair"]
    assert not (tmp_path / "generated" / "iteration-01-lineage.json").exists()


def test_live_lineage_materializes_and_rehydrates_without_recalling_model(
    tmp_path: Path,
) -> None:
    model = _LiveFixtureModel()
    parent = load_parent_snapshot(
        ROOT / "candidate_seed", candidate_id="official-fm-fallback-seed-4"
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    generated = tmp_path / "generated"
    first = prepare_or_rehydrate_live_lineage(
        campaign_id="live-campaign",
        scientific_iteration=1,
        parent=parent,
        generated_root=generated,
        artifact_store=artifacts,
        safe_context=_safe_context(),
        model=model,
    )

    assert model.calls == ["propose", "implement"]
    assert first.live_provider_used is True
    assert first.provider == "openai"
    assert first.candidate_id.startswith("candidate-01-")
    assert first.materialized.source_digest != parent.digest
    assert first.material_change.changed_symbols == ("candidate.py:live_variant_marker",)
    assert artifacts.verify_directory(first.source_snapshot) == first.source_snapshot

    resumed = prepare_or_rehydrate_live_lineage(
        campaign_id="live-campaign",
        scientific_iteration=1,
        parent=parent,
        generated_root=generated,
        artifact_store=artifacts,
        safe_context=_safe_context(),
        model=model,
    )
    assert model.calls == ["propose", "implement"]
    assert resumed.manifest() == first.manifest()


def test_scripted_production_lineage_materially_generates_replayable_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    safe_context = build_safe_research_context(
        starter_manifest_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        capability_manifests=(),
        budgets=ResearchBudgetContext(44, 18_000, 1, 0),
    )
    operations: list[str] = []
    original_propose = ScriptedResearchModel.propose
    original_implement = ScriptedResearchModel.implement

    def recording_propose(self: ScriptedResearchModel, request: ProposalRequest) -> Proposal:
        operations.append("propose")
        return original_propose(self, request)

    def recording_implement(
        self: ScriptedResearchModel, request: ImplementationRequest
    ) -> GeneratedPackage:
        operations.append("implement")
        return original_implement(self, request)

    monkeypatch.setattr(ScriptedResearchModel, "propose", recording_propose)
    monkeypatch.setattr(ScriptedResearchModel, "implement", recording_implement)

    lineage = prepare_scripted_lambdarank_lineage(
        campaign_id="production-campaign",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=tmp_path / "generated",
        artifact_store=artifacts,
        safe_context=safe_context,
    )

    assert lineage.proposal.parent_candidate_id == "official-fm-fallback-seed-4"
    assert lineage.proposal_request.digest == lineage.model_calls[0].request_digest
    assert lineage.proposal.digest == lineage.model_calls[0].response_digest
    assert lineage.implementation_request.digest == lineage.model_calls[1].request_digest
    assert lineage.package.digest == lineage.model_calls[1].response_digest
    assert lineage.material_change.changed_symbols == (
        "candidate.py:predict_scores",
        "candidate.py:train_model",
    )
    assert lineage.materialized.parent_digest == lineage.parent.digest
    assert lineage.materialized.source_digest != lineage.parent.digest
    assert lineage.materialized.diff_digest == lineage.diff_artifact.sha256
    assert lineage.source_snapshot.kind.value == "source"
    assert {entry.path for entry in lineage.source_snapshot.entries} == {
        "README.md",
        "candidate.py",
        "config.json",
    }
    assert lineage.identity.source_snapshot == lineage.source_snapshot
    assert lineage.identity.source_digest == lineage.materialized.source_digest
    assert lineage.identity.config_digest == lineage.config_artifact.sha256
    assert artifacts.verify_directory(lineage.source_snapshot) == lineage.source_snapshot
    assert lineage.provider == "scripted"
    assert lineage.live_provider_used is False
    assert operations == ["propose", "implement"]


def test_scripted_production_lineage_is_exactly_replayable_from_recorded_response(
    tmp_path: Path,
) -> None:
    safe_context = build_safe_research_context(
        starter_manifest_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        capability_manifests=(),
        budgets=ResearchBudgetContext(44, 18_000, 1, 0),
    )

    first = prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-replay",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=tmp_path / "generated-a",
        artifact_store=ArtifactStore(tmp_path / "artifacts-a"),
        safe_context=safe_context,
    )
    second = prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-replay",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=tmp_path / "generated-b",
        artifact_store=ArtifactStore(tmp_path / "artifacts-b"),
        safe_context=safe_context,
    )

    assert second.parent == first.parent
    assert second.proposal == first.proposal
    assert second.package == first.package
    assert second.materialized.source_digest == first.materialized.source_digest
    assert second.materialized.diff_digest == first.materialized.diff_digest
    assert second.source_snapshot.sha256 == first.source_snapshot.sha256
    assert second.transcript_artifact.sha256 == first.transcript_artifact.sha256


def test_completed_scripted_lineage_rehydrates_after_controller_restart_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    safe_context = _safe_context()
    first = prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-resume",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=tmp_path / "generated",
        artifact_store=artifacts,
        safe_context=safe_context,
    )

    monkeypatch.setattr(ScriptedResearchModel, "propose", _unexpected_model_call)
    monkeypatch.setattr(ScriptedResearchModel, "implement", _unexpected_model_call)
    monkeypatch.setattr(ArtifactStore, "put_bytes", _unexpected_model_call)
    monkeypatch.setattr(ArtifactStore, "put_directory", _unexpected_model_call)

    resumed = prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-resume",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=tmp_path / "generated",
        artifact_store=artifacts,
        safe_context=safe_context,
    )

    assert resumed == first
    assert resumed.manifest() == first.manifest()


def test_existing_generated_tree_without_matching_artifact_evidence_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "generated"
    prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-partial",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=generated_root,
        artifact_store=ArtifactStore(tmp_path / "lost-artifacts"),
        safe_context=_safe_context(),
    )
    monkeypatch.setattr(ScriptedResearchModel, "propose", _unexpected_model_call)
    monkeypatch.setattr(ScriptedResearchModel, "implement", _unexpected_model_call)

    with pytest.raises(ProductionResearchError, match=r"incomplete|evidence"):
        prepare_scripted_lambdarank_lineage(
            campaign_id="campaign-partial",
            scientific_iteration=1,
            template_dir=ROOT / "candidate_templates" / "lambdarank",
            generated_root=generated_root,
            artifact_store=ArtifactStore(tmp_path / "replacement-artifacts"),
            safe_context=_safe_context(),
        )


def test_tampered_generated_source_is_rejected_on_rehydrate(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-tamper",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=generated_root,
        artifact_store=artifacts,
        safe_context=_safe_context(),
    )
    candidate_path = generated_root / "generated-causal-lambdarank-v1" / "candidate.py"
    candidate_path.chmod(0o600)
    candidate_path.write_text("tampered\n", encoding="utf-8")
    candidate_path.chmod(0o400)

    with pytest.raises(ProductionResearchError, match=r"tampered|match|changed"):
        prepare_scripted_lambdarank_lineage(
            campaign_id="campaign-tamper",
            scientific_iteration=1,
            template_dir=ROOT / "candidate_templates" / "lambdarank",
            generated_root=generated_root,
            artifact_store=artifacts,
            safe_context=_safe_context(),
        )


def test_bytecode_or_extra_directory_is_rejected_on_rehydrate(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-bytecode",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=generated_root,
        artifact_store=artifacts,
        safe_context=_safe_context(),
    )
    destination = generated_root / "generated-causal-lambdarank-v1"
    destination.chmod(0o700)
    cache = destination / "__pycache__"
    cache.mkdir()
    (cache / "candidate.cpython-312.pyc").write_bytes(b"forbidden-bytecode")
    destination.chmod(0o500)

    with pytest.raises(ProductionResearchError, match=r"exactly|undeclared|extra"):
        prepare_scripted_lambdarank_lineage(
            campaign_id="campaign-bytecode",
            scientific_iteration=1,
            template_dir=ROOT / "candidate_templates" / "lambdarank",
            generated_root=generated_root,
            artifact_store=artifacts,
            safe_context=_safe_context(),
        )


def test_symlinked_generated_member_is_rejected_on_rehydrate(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-symlink",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=generated_root,
        artifact_store=artifacts,
        safe_context=_safe_context(),
    )
    destination = generated_root / "generated-causal-lambdarank-v1"
    config_path = destination / "config.json"
    destination.chmod(0o700)
    config_path.unlink()
    os.symlink(ROOT / "candidate_templates" / "lambdarank" / "config.json", config_path)
    destination.chmod(0o500)

    with pytest.raises(ProductionResearchError, match=r"symlink|real file|nonregular"):
        prepare_scripted_lambdarank_lineage(
            campaign_id="campaign-symlink",
            scientific_iteration=1,
            template_dir=ROOT / "candidate_templates" / "lambdarank",
            generated_root=generated_root,
            artifact_store=artifacts,
            safe_context=_safe_context(),
        )


def test_tampered_transcript_artifact_is_rejected_on_rehydrate(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    lineage = prepare_scripted_lambdarank_lineage(
        campaign_id="campaign-artifact-tamper",
        scientific_iteration=1,
        template_dir=ROOT / "candidate_templates" / "lambdarank",
        generated_root=generated_root,
        artifact_store=artifacts,
        safe_context=_safe_context(),
    )
    transcript_path = artifacts.object_path(lineage.transcript_artifact)
    transcript_path.chmod(0o600)
    transcript_path.write_bytes(transcript_path.read_bytes() + b"tampered")
    transcript_path.chmod(0o444)

    with pytest.raises(ProductionResearchError, match=r"incomplete|invalid|evidence"):
        prepare_scripted_lambdarank_lineage(
            campaign_id="campaign-artifact-tamper",
            scientific_iteration=1,
            template_dir=ROOT / "candidate_templates" / "lambdarank",
            generated_root=generated_root,
            artifact_store=artifacts,
            safe_context=_safe_context(),
        )
