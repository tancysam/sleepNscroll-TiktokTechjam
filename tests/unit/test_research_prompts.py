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

    assert PROMPT_VERSION == 18
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
    # The parent always contains candidate.py, so the old conditional wording ("may omit it only
    # when the parent already contains it") was vacuous and reintroduced the doubt that cost a
    # live campaign its first branch. The overlay is unconditionally legal without it.
    assert "a helper-only overlay such as pairwise_fm.py is legal and complete on its own" in (
        instructions
    )
    assert "only when the supplied trusted parent already contains it" not in instructions
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


def test_proposal_prompt_requires_a_legal_final_manifest() -> None:
    """files_expected is the FINAL TREE manifest, and the entrypoint is validated.

    provider._parse calls source_policy.validate_manifest(proposal.files_expected,
    require_final_entrypoint=True) with no parent_paths, so files_expected IS the final tree and
    omitting candidate.py fails validation on every proposal.

    The failure that made this look like a contradiction was at the other end: the model treated
    files_expected as the list of files to return, returned candidate.py, and was refused at
    materialization because it is a protected path. Both statements are true and they describe
    different things -- the manifest is the resulting tree, the response is an overlay over it.
    """

    instructions = " ".join(instructions_for(ResearchOperation.PROPOSE).split())

    assert "files_expected is the manifest of the FINAL candidate tree" in instructions
    assert "must include candidate.py" in instructions
    assert "never returns candidate.py itself" in instructions


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


def test_prompt_grants_in_matrix_feature_authority_and_bounds_it() -> None:
    """The candidate may transform the matrix it is given, and must be told so explicitly.

    `train_model` receives a mutable NumPy array and numpy is available, so in-matrix feature
    engineering was always technically possible. The briefing never said so, and across sixteen
    live campaigns every proposal preserved the controller bundle verbatim -- one even said it
    would "preserve the parent's causal controller feature bundle". That is the same
    constraint-transmission defect this project already documented in reverse: a model rejected
    for rules it was never told, and here a capability declined because it was never granted.
    """

    instructions = instructions_for(ResearchOperation.PROPOSE)

    assert "Feature authority inside your own code" in instructions
    assert "mutable NumPy array" in instructions
    # The grant must travel with its two bounds, or it invites a leak or a train/inference skew.
    assert "identical transformation in" in instructions
    assert "to read raw columns, current-row outcomes" in instructions
    # The stale claim that no history exists must not survive the widened bundle.
    assert "use no history at all" not in instructions
    assert "is_click_smoothed_rate" in instructions
