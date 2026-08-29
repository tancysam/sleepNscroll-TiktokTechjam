from __future__ import annotations

import pytest

from kuairand_agent.research.prompts import PROMPT_VERSION, instructions_for
from kuairand_agent.research.schemas import ProposalRequest, ResearchOperation
from kuairand_agent.research.source_policy import DEFAULT_CANDIDATE_SOURCE_POLICY


@pytest.mark.parametrize(
    "operation",
    (ResearchOperation.PROPOSE, ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR),
)
def test_candidate_generation_prompts_derive_the_enforced_source_contract(
    operation: ResearchOperation,
) -> None:
    instructions = instructions_for(operation)

    assert PROMPT_VERSION == 3
    assert DEFAULT_CANDIDATE_SOURCE_POLICY.digest in instructions
    assert "candidate.py" in instructions
    assert "baseline.py" in instructions
    assert ".csv" in instructions
    assert "submission.csv" in instructions
    assert "complete-file overlay" in instructions
    assert "pairwise_fm.py" in instructions
    assert "provider JSON acceptance is not candidate admission" in instructions


@pytest.mark.parametrize("operation", (ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR))
def test_source_generation_prompts_define_overlay_and_material_symbol_semantics(
    operation: ResearchOperation,
) -> None:
    instructions = instructions_for(operation)

    assert "Unmentioned trusted-parent files remain in the final tree" in instructions
    assert (
        "may omit candidate.py only when the supplied trusted parent already contains it"
        in instructions
    )
    assert "reachable top-level Python identifiers actually changed" in instructions
    assert (
        "Documentation, docstrings, filenames, whitespace, and unchanged symbols do not count"
        in instructions
    )


def test_proposal_prompt_requires_a_legal_final_manifest() -> None:
    instructions = instructions_for(ResearchOperation.PROPOSE)

    assert "files_expected describes the final candidate manifest" in instructions
    assert "must include candidate.py" in instructions


def test_delivered_prompt_uses_the_exact_request_policy_digest() -> None:
    request = ProposalRequest.create(
        request_id="propose-1",
        campaign_id="campaign-1",
        scientific_iteration=1,
        parent_candidate_id="fm-seed",
        safe_context={"evidence_cursor": "initial"},
    )

    instructions = instructions_for(ResearchOperation.PROPOSE, source_policy=request.source_policy)

    assert request.source_policy_digest in instructions
