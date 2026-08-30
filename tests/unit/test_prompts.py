"""Guard the research instruction surface against the failures that actually happened.

``tests/unit/test_research_prompts.py`` covers the source-policy contract rendered from
:mod:`kuairand_agent.research.source_policy`.  This file covers the rest of the instruction
surface: the scientific briefing, the execution environment, and the two naming and output-path
mistakes that cost real campaign iterations.

Every assertion here traces to an observed failure:

* A live campaign repeatedly returned ``baseline.py`` and was rejected by the deterministic
  basename gate on every iteration, so no candidate was ever scored and the run finalized the
  immutable fallback.  The request context lists the organizer starter filenames as artifact
  identities, so the instructions must say explicitly that they are references, not targets.
* The candidate seed parsed a seven-key training request while the trusted executor always sends
  nine, and wrote ``checkpoint/model.npz`` while the protocol pins ``checkpoint/model.txt``.
* Two planning documents encoded ``click`` with NDCG@10/Recall@50 before the starter kit settled
  the contract; that framing must never reach the model.
"""

from __future__ import annotations

import pytest

from kuairand_agent.research.prompts import PROMPT_VERSION, instructions_for
from kuairand_agent.research.schemas import ResearchOperation
from kuairand_agent.research.source_policy import DEFAULT_CANDIDATE_SOURCE_POLICY

# Operations that return candidate source and therefore need the environment and worked example.
_SOURCE_OPERATIONS = (ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR)
# Operations that reason about the science and therefore need the benchmark briefing.
_SCIENCE_OPERATIONS = (
    ResearchOperation.PROPOSE,
    ResearchOperation.IMPLEMENT,
    ResearchOperation.REFLECT,
)


@pytest.mark.parametrize("operation", _SCIENCE_OPERATIONS)
def test_briefing_pins_the_operative_task_contract(operation: ResearchOperation) -> None:
    """The real label, metrics, and ceiling must reach every operation that reasons."""

    rendered = instructions_for(operation)
    for fragment in ("long_view", "GAUC", "nDCG@5"):
        assert fragment in rendered
    assert "0.8645" in rendered, "the oracle ceiling must be stated so 1.0 is not chased"
    assert "0.5946" in rendered, "the baseline to beat must be stated"
    assert "NDCG@10" not in rendered
    assert "Recall@50" not in rendered


@pytest.mark.parametrize("operation", _SCIENCE_OPERATIONS)
def test_briefing_carries_the_organizers_measured_dead_ends(
    operation: ResearchOperation,
) -> None:
    """Published negative results belong in the prompt, not in rediscovered iterations."""

    rendered = instructions_for(operation)
    assert "0.5940" in rendered, "the static-feature ablation result must be stated"
    assert "0.5895" in rendered, "the embedding-dimension sweep must be stated"
    assert "EXACTLY ZERO" in rendered, "user-side first-order terms cannot reorder within a user"
    assert "pointwise log loss" in rendered.lower() or "pointwise" in rendered


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_source_operations_state_the_pinned_output_paths(
    operation: ResearchOperation,
) -> None:
    """A candidate that writes the wrong checkpoint path fails after training completes."""

    rendered = instructions_for(operation)
    assert "checkpoint/model.txt" in rendered
    assert "scores.npy" in rendered


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_source_operations_state_the_full_training_request_keys(
    operation: ResearchOperation,
) -> None:
    """The executor always sends these; a parser that omits either one fails closed."""

    rendered = instructions_for(operation)
    assert "user_groups_handle" in rendered
    assert "seed" in rendered


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_source_operations_disambiguate_the_organizer_starter_filenames(
    operation: ResearchOperation,
) -> None:
    """This is the exact failure that wasted a full live campaign."""

    rendered = instructions_for(operation)
    assert "ORGANIZER REFERENCE FILES" in rendered
    assert "NOT as files for you to write" in rendered
    for name in ("baseline.py", "evaluate.py", "data.py", "submit.py"):
        assert name in rendered, f"{name} must be named as a reference, not left implicit"


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_source_operations_state_the_available_environment(
    operation: ResearchOperation,
) -> None:
    """Positive framing: the model needs to know what it *can* import, not only what it cannot."""

    rendered = instructions_for(operation)
    assert "numpy" in rendered
    assert "np.savez" in rendered
    assert "pickle" in rendered
    assert "import os.path" in rendered, "the first-dotted-component rule must be explicit"


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_source_operations_carry_the_worked_material_change_example(
    operation: ResearchOperation,
) -> None:
    """The material-change gate rejects more packages than any other; show a worked case."""

    rendered = instructions_for(operation)
    assert "Worked example" in rendered
    assert "material_symbols" in rendered
    assert "REJECTED" in rendered


@pytest.mark.parametrize(
    "operation",
    (ResearchOperation.PROPOSE, ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR),
)
def test_candidate_operations_still_carry_the_source_policy(
    operation: ResearchOperation,
) -> None:
    """Adding briefing content must not displace the enforced source policy."""

    rendered = instructions_for(operation)
    assert DEFAULT_CANDIDATE_SOURCE_POLICY.digest in rendered


def test_fileless_operations_are_not_charged_for_source_guidance() -> None:
    """Token budget guard: PROPOSE and REFLECT emit no files and run every iteration."""

    implement = instructions_for(ResearchOperation.IMPLEMENT)
    for operation in (ResearchOperation.PROPOSE, ResearchOperation.REFLECT):
        rendered = instructions_for(operation)
        assert "Worked example" not in rendered
        assert "Execution environment" not in rendered
        assert len(rendered) < len(implement)


def test_every_operation_renders_and_is_distinct() -> None:
    rendered = {operation: instructions_for(operation) for operation in ResearchOperation}
    assert len(rendered) == 4
    for text in rendered.values():
        assert text.strip()
    assert len(set(rendered.values())) == 4, "operations must not share identical instructions"


def test_schema_retry_appends_only_a_correction_notice() -> None:
    base = instructions_for(ResearchOperation.IMPLEMENT)
    retry = instructions_for(ResearchOperation.IMPLEMENT, schema_retry=True)
    assert retry.startswith(base)
    assert "rejected by the local strict parser" in retry[len(base) :]


def test_prompt_version_matches_the_response_schema_name() -> None:
    """``provider.py`` builds ``kuairand_<operation>_v{PROMPT_VERSION}``; tests pin ``_v3``."""

    assert PROMPT_VERSION == 10


@pytest.mark.parametrize("bad", [None, "propose", 0, object()])
def test_instructions_reject_a_non_operation(bad: object) -> None:
    with pytest.raises(ValueError, match="operation must be a ResearchOperation"):
        instructions_for(bad)  # type: ignore[arg-type]


def test_instructions_reject_a_non_boolean_retry_flag() -> None:
    with pytest.raises(ValueError, match="schema_retry must be bool"):
        instructions_for(ResearchOperation.PROPOSE, schema_retry=1)  # type: ignore[arg-type]


# A campaign closed with zero candidates built because the controller enforced a family block the
# proposer was never shown. These pin the guidance that turns that wall into a redirection.


@pytest.mark.parametrize("operation", (ResearchOperation.PROPOSE, ResearchOperation.REFLECT))
def test_axis_choosing_operations_are_shown_the_closed_families(
    operation: ResearchOperation,
) -> None:
    rendered = instructions_for(operation, blocked_families=(("pairwise", "already lost"),))
    assert "Closed proposal families" in rendered
    assert "pairwise: already lost" in rendered
    assert "Choose a different axis." in rendered


@pytest.mark.parametrize("operation", (ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR))
def test_committed_operations_are_not_charged_for_closures(
    operation: ResearchOperation,
) -> None:
    """IMPLEMENT and REPAIR serve an already-admitted proposal and cannot act on a closure."""

    rendered = instructions_for(operation, blocked_families=(("pairwise", "already lost"),))
    assert "Closed proposal families" not in rendered


def test_no_closures_renders_no_section() -> None:
    assert "Closed proposal families" not in instructions_for(ResearchOperation.PROPOSE)


@pytest.mark.parametrize("bad", [(("pairwise",),), (("pairwise", ""),), ((1, "why"),)])
def test_instructions_reject_malformed_closures(bad: object) -> None:
    with pytest.raises(ValueError, match="blocked_families entries"):
        instructions_for(ResearchOperation.PROPOSE, blocked_families=bad)  # type: ignore[arg-type]


def test_lightgbm_guidance_states_the_two_facts_that_would_fail_a_candidate() -> None:
    """`scikit-learn` is absent and a Booster is not a numeric array; both are hard failures."""

    rendered = instructions_for(ResearchOperation.IMPLEMENT)
    assert "scikit-learn` is NOT installed" in rendered
    assert "LGBMRanker" in rendered, "the unavailable sklearn entrypoint must be named"
    assert "lambdarank" in rendered
    for parameter in ("deterministic", "force_col_wise", "num_threads", "bagging_freq"):
        assert parameter in rendered, f"{parameter} is required for the replay gate"
    assert "model_to_string" in rendered, "the Booster must be encoded to satisfy the checkpoint"
    assert "np.uint8" in rendered


# The briefing carried slate statistics that were wrong for the scored period and told the agent
# the nDCG@5 cutoff was inert. Measured on the real test split: median 8, p90 24, and 66.2% of
# users above the cutoff. These pin the corrected facts so the wrong ones cannot silently return.


@pytest.mark.parametrize("operation", _SCIENCE_OPERATIONS)
def test_briefing_states_the_measured_slate_structure(operation: ResearchOperation) -> None:
    rendered = instructions_for(operation)
    assert "median 8" in rendered, "the scored-period median slate size is 8, not 4"
    assert "66.2%" in rendered, "the share of users the nDCG@5 cutoff actually binds on"
    assert "median 4 impressions" not in rendered, "the retracted training-window figure"
    assert "inert for the majority" not in rendered, "the retracted truncation claim"


@pytest.mark.parametrize("operation", _SCIENCE_OPERATIONS)
def test_briefing_states_the_user_weighting_asymmetry(operation: ResearchOperation) -> None:
    """GAUC weights by positive count; nDCG@5 gives every user one vote. This drives the trade."""

    rendered = instructions_for(operation)
    assert "31.7%" in rendered, "the GAUC weight concentration must be quantified"
    assert "0.386" in rendered, "the position-5 DCG discount makes the top-5 lever concrete"
    assert "-0.0215" in rendered, "the measured GAUC/nDCG trade must be stated"


@pytest.mark.parametrize("operation", _SCIENCE_OPERATIONS)
def test_briefing_rules_out_the_avenues_the_runtime_cannot_express(
    operation: ResearchOperation,
) -> None:
    """Multi-task and censored watch-time are organizer priorities this runtime cannot reach."""

    rendered = instructions_for(operation)
    assert "DO NOT\nPROPOSE EITHER" in rendered or "DO NOT PROPOSE EITHER" in rendered
    for absent_field in ("is_follow", "is_comment", "is_forward", "play_time_ms"):
        assert absent_field in rendered, f"{absent_field} must be named as unavailable"
    assert "exactly one binary `targets` vector" in rendered


def test_lightgbm_guidance_pins_the_objective_cutoff() -> None:
    """`ndcg_eval_at` sets the printed metric; the objective cutoff is a different parameter."""

    rendered = instructions_for(ResearchOperation.IMPLEMENT)
    assert "lambdarank_truncation_level" in rendered
    assert "defaults to 30" in rendered, "the default is the trap; state it"


def test_feature_authority_states_the_measured_cost_of_an_embedding_table() -> None:
    """The one candidate that used identities lost monotonically; say why, not just that."""

    rendered = instructions_for(ResearchOperation.PROPOSE)
    assert "0.5705" in rendered, "the measured identity-embedding result must be stated"
    assert "26,211" in rendered and "7,539" in rendered
    assert "id__tab" in rendered, "the low-cardinality escape hatch must be named"


def test_instructions_reject_a_foreign_source_policy() -> None:
    with pytest.raises(ValueError, match="source_policy must be CandidateSourcePolicy"):
        instructions_for(ResearchOperation.IMPLEMENT, source_policy=object())  # type: ignore[arg-type]
