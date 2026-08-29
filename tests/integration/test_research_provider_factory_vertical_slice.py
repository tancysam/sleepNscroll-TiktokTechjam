from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kuairand_agent.config import ResearchConfig
from kuairand_agent.research.factory import (
    AvailableResearchProvider,
    select_research_provider,
)
from kuairand_agent.research.materialize import (
    materialize_candidate,
    require_material_executable_change,
    validate_candidate_static,
)
from kuairand_agent.research.provider import (
    OpenAIResponsesConfig,
    OpenAIResponsesModel,
    TokenPricing,
    TransportRequest,
    TransportResponse,
)
from kuairand_agent.research.schemas import (
    ExperimentResultSummary,
    GeneratedFile,
    GeneratedPackage,
    ImplementationRequest,
    ParentSnapshot,
    ParentSourceFile,
    Proposal,
    ProposalRequest,
    Reflection,
    ReflectionRequest,
    RequiredField,
)


def _proposal() -> Proposal:
    return Proposal(
        proposal_id="offline-provider-slice-v1",
        hypothesis="A legal scalar feature improves the fixture's within-user ordering.",
        mechanism="Replace the constant score with a deterministic scalar score.",
        expected_metric_effects=("GAUC", "nDCG@5"),
        parent_candidate_id="factory-parent",
        principal_change="Materially implement the candidate score function.",
        files_expected=("candidate.py",),
        required_fields=(
            RequiredField(
                "trusted/causal_feature_matrix:duration_ms",
                "inference_input",
                "Use only this controller-approved current-item feature.",
            ),
        ),
        objective="binary long-view ranking",
        sampling="logged impressions",
        grouping="user impression groups",
        weighting="uniform",
        causal_cutoff="No current-row, validation, or final-period outcome is used.",
        estimated_runtime_seconds=1,
        estimated_memory_mb=64,
        smoke_plan="Score a four-row controller fixture.",
        inner_fold_plan="Screen Fold B and confirm Fold A.",
        falsification_criteria="Reject failed static, replay, or protected-score gates.",
        promotion_criteria="Require positive matched evidence under the frozen selector.",
        maximum_repairs=1,
        rollback_parent_id="factory-parent",
        attributions=("offline factory transport fixture",),
    )


def _package() -> GeneratedPackage:
    return GeneratedPackage(
        request_id="factory-implement",
        response_id="offline-source-1",
        files=(
            GeneratedFile(
                "candidate.py",
                "def score(values):\n    return [float(value) * 0.25 for value in values]\n",
            ),
        ),
        material_change_summary="Replace the constant score with a legal scalar transformation.",
        material_symbols=("score",),
    )


def _reflection() -> Reflection:
    return Reflection(
        response_id="offline-reflection-1",
        summary="The material scalar candidate passed the supplied fixture evidence.",
        recommendation="propose_next",
        lessons=("Retain exact generated-source and protected-score evidence.",),
    )


def _response(payload: dict[str, object], response_id: str) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "object": "chat.completion",
            "model": "gpt-5.4",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload, separators=(",", ":")),
                        "refusal": None,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 150,
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass
class _FakeTransport:
    responses: list[TransportResponse]
    requests: list[TransportRequest] = field(default_factory=list)

    def send(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _offline_config() -> OpenAIResponsesConfig:
    return OpenAIResponsesConfig(
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        reasoning_effort="low",
        pricing=TokenPricing(
            input_usd_per_million="1",
            cached_input_usd_per_million="0.25",
            output_usd_per_million="4",
        ),
        api_key_env="KUAIRAND_FACTORY_SLICE_KEY",
        timeout_seconds=5.0,
        max_response_bytes=64 * 1024,
        max_output_tokens=4096,
    )


def test_openai_factory_drives_full_typed_materialization_and_reflection_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-proj-offline-factory-secret"
    monkeypatch.setenv("KUAIRAND_FACTORY_SLICE_KEY", secret)
    proposal = _proposal()
    package = _package()
    reflection = _reflection()
    transport = _FakeTransport(
        [
            TransportResponse(200, _response(proposal.to_wire(), "response-propose")),
            TransportResponse(200, _response(package.to_wire(), "response-implement")),
            TransportResponse(200, _response(reflection.to_wire(), "response-reflect")),
        ]
    )
    selection = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_config=_offline_config(),
        transport=transport,
    )
    assert isinstance(selection, AvailableResearchProvider)
    assert isinstance(selection.model, OpenAIResponsesModel)
    model = selection.model
    safe_context = {
        "schema_version": 1,
        "approved_fields": ["duration_ms"],
        "forbidden_fields": ["long_view", "is_click"],
    }
    proposed = model.propose(
        ProposalRequest.create(
            request_id="factory-propose",
            campaign_id="factory-campaign",
            scientific_iteration=1,
            parent_candidate_id="factory-parent",
            safe_context=safe_context,
        )
    )
    parent = ParentSnapshot(
        candidate_id="factory-parent",
        files=(
            ParentSourceFile.create(
                "candidate.py",
                "def score(values):\n    return [0.0 for _value in values]\n",
            ),
        ),
    )
    implemented = model.implement(
        ImplementationRequest.create(
            request_id="factory-implement",
            proposal=proposed,
            parent=parent,
            safe_context=safe_context,
        )
    )
    child = materialize_candidate(parent, implemented, tmp_path / "generated-child")
    validate_candidate_static(child)
    material = require_material_executable_change(parent, child)
    assert material.changed_symbols == ("candidate.py:score",)

    result = ExperimentResultSummary(
        tier="fixture",
        status="passed",
        gauc=1.0,
        ndcg_at_5=1.0,
        primary=1.0,
        runtime_seconds=0.01,
        peak_memory_mb=1.0,
    )
    reflected = model.reflect(
        ReflectionRequest.create(
            request_id="factory-reflect",
            proposal_id=proposed.proposal_id,
            candidate_id="factory-child",
            source_digest=child.source_digest,
            diff_digest=child.diff_digest,
            result=result,
            safe_context=safe_context,
        )
    )

    assert reflected is not None
    assert len(transport.requests) == 3
    assert [transcript.operation.value for transcript in model.transcripts] == [
        "propose",
        "implement",
        "reflect",
    ]
    assert secret not in repr(selection)
    assert secret not in repr(transport.requests)
    assert all(secret.encode("utf-8") not in item.json_bytes for item in model.transcripts)
    assert all(secret not in source.content for source in child.files)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is absent",
)
def test_credential_gated_real_provider_proposes_and_materially_implements(
    tmp_path: Path,
) -> None:
    """The real-provider acceptance slice; skipped cleanly when its credential is absent."""

    config = OpenAIResponsesConfig(
        model=os.environ.get("KUAIRAND_OPENAI_MODEL", "gpt-5.4"),
        base_url="https://api.openai.com/v1",
        reasoning_effort="low",
        # Smoke accounting is explicit and token-exact.  Monetary rates may be supplied by the
        # caller; zero is an intentional non-claim when no current price contract is configured.
        pricing=TokenPricing(
            input_usd_per_million=os.environ.get("KUAIRAND_OPENAI_INPUT_USD_PER_MILLION", "0"),
            cached_input_usd_per_million=os.environ.get(
                "KUAIRAND_OPENAI_CACHED_INPUT_USD_PER_MILLION", "0"
            ),
            output_usd_per_million=os.environ.get("KUAIRAND_OPENAI_OUTPUT_USD_PER_MILLION", "0"),
        ),
        max_output_tokens=4096,
    )
    selection = select_research_provider(
        ResearchConfig(provider="openai", max_repairs_per_experiment=1),
        openai_config=config,
    )
    assert isinstance(selection, AvailableResearchProvider), selection
    model = selection.model
    parent = ParentSnapshot(
        candidate_id="live-smoke-parent",
        files=(
            ParentSourceFile.create(
                "candidate.py",
                "def score(feature_values):\n    return [0.0 for _value in feature_values]\n",
            ),
        ),
    )
    safe_context = {
        "schema_version": 1,
        "acceptance_scope": (
            "Propose exactly one tiny deterministic candidate.py change to score a sequence of "
            "controller-approved finite scalar feature_values. Use no imports and no other files."
        ),
        "approved_fields": ["trusted/causal_feature_matrix:duration_ms"],
        "forbidden_fields": [
            "current-row outcomes",
            "public-validation outcomes",
            "final-period outcomes",
        ],
        "budgets": {
            "remaining_attempts": 1,
            "remaining_wall_seconds": 120,
            "remaining_outer_promotions": 0,
        },
    }
    proposal = model.propose(
        ProposalRequest.create(
            request_id="live-smoke-propose",
            campaign_id="live-provider-smoke",
            scientific_iteration=1,
            parent_candidate_id=parent.candidate_id,
            safe_context=safe_context,
        )
    )
    package = model.implement(
        ImplementationRequest.create(
            request_id="live-smoke-implement",
            proposal=proposal,
            parent=parent,
            safe_context=safe_context,
        )
    )
    child = materialize_candidate(parent, package, tmp_path / "live-generated-child")
    validate_candidate_static(child)
    evidence = require_material_executable_change(parent, child)

    assert proposal.parent_candidate_id == parent.candidate_id
    assert package.request_id == "live-smoke-implement"
    assert child.source_digest != parent.digest
    assert evidence.changed_symbols
