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

    assert PROMPT_VERSION == 6
    assert DEFAULT_CANDIDATE_SOURCE_POLICY.digest in instructions
    assert "candidate.py" in instructions
    assert "baseline.py" in instructions
    assert ".csv" in instructions
    assert "submission.csv" in instructions
    assert "complete-file overlay" in instructions
    assert "pairwise_fm.py" in instructions
    assert "provider JSON acceptance is not candidate admission" in instructions
    assert "Candidate runtime contract digest" in instructions
    assert "finite float64 (N,D)" in instructions
    assert "feature_names_csv" in instructions
    assert "never assume features[:,0:5]" in instructions


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
    assert "strict parseable JSON" in instructions
    assert "double-quoted property names" in instructions
    assert "complete JSON object as the file content string" in instructions
    assert "executable definition responsible" in instructions
    if operation is ResearchOperation.IMPLEMENT:
        assert "Return only files whose content differs" in instructions
    else:
        assert "complete generated overlay relative to the trusted" in instructions
        assert "unmentioned rejected-overlay files are not implicitly retained" in instructions
    assert "Never return or replace" in instructions
    assert "protected_paths" in instructions
    assert "model_impl.py" in instructions


def test_model_generation_prompt_defines_one_general_model_interface() -> None:
    instructions = instructions_for(ResearchOperation.IMPLEMENT)

    assert "train_model(features, targets, user_groups, config, seed)" in instructions
    assert "predict_scores(features, checkpoint)" in instructions
    assert "1..64 named finite numeric NumPy arrays" in instructions
    assert "checkpoint keys may be model-specific" in instructions
    assert "The protected wrapper owns them" in instructions


def test_proposal_prompt_requires_a_legal_returnable_manifest() -> None:
    """files_expected must describe what the response RETURNS, not the final tree.

    This test previously asserted that PROPOSE tells the model files_expected "must include
    candidate.py". That instruction contradicted IMPLEMENT, which forbids returning any
    protected path -- and candidate.py is the protected path. A live campaign lost its first
    branch to it: the model listed candidate.py as instructed, returned it, and was rejected at
    materialization with "generated package cannot replace protected runtime file(s):
    candidate.py" after exhausting both repairs.

    The final tree does contain candidate.py, because overlay semantics retain unmentioned
    parent files. The response must not.
    """

    instructions = " ".join(instructions_for(ResearchOperation.PROPOSE).split())

    assert "files_expected lists exactly the files the implementation will RETURN" in instructions
    assert "never list or return candidate.py" in instructions
    assert "must include candidate.py" not in instructions


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
