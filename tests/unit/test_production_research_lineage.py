from __future__ import annotations

import os
from pathlib import Path

import pytest

from kuairand_agent.execution.artifacts import ArtifactStore
from kuairand_agent.research.context import (
    ResearchBudgetContext,
    SafeResearchContext,
    build_safe_research_context,
)
from kuairand_agent.research.production import (
    ProductionResearchError,
    load_parent_snapshot,
    prepare_or_rehydrate_live_lineage,
    prepare_scripted_lambdarank_lineage,
)
from kuairand_agent.research.schemas import (
    GeneratedFile,
    GeneratedPackage,
    ImplementationRequest,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
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
