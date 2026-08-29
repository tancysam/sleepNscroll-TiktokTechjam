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
    """``provider.py`` builds ``kuairand_<operation>_v{PROMPT_VERSION}``; tests pin ``_v6``."""

    assert PROMPT_VERSION == 6


@pytest.mark.parametrize("bad", [None, "propose", 0, object()])
def test_instructions_reject_a_non_operation(bad: object) -> None:
    with pytest.raises(ValueError, match="operation must be a ResearchOperation"):
        instructions_for(bad)  # type: ignore[arg-type]


def test_instructions_reject_a_non_boolean_retry_flag() -> None:
    with pytest.raises(ValueError, match="schema_retry must be bool"):
        instructions_for(ResearchOperation.PROPOSE, schema_retry=1)  # type: ignore[arg-type]


def test_instructions_reject_a_foreign_source_policy() -> None:
    with pytest.raises(ValueError, match="source_policy must be CandidateSourcePolicy"):
        instructions_for(ResearchOperation.IMPLEMENT, source_policy=object())  # type: ignore[arg-type]


def test_propose_never_asks_for_a_protected_path() -> None:
    """PROPOSE and IMPLEMENT must not contradict each other about candidate.py.

    A live campaign lost its first branch to exactly this. PROPOSE said files_expected "must
    include candidate.py"; IMPLEMENT forbids returning any protected path, and candidate.py is
    one. The model obeyed PROPOSE, implemented the manifest it had just proposed, and was
    rejected at materialization after burning both repair attempts:

        "generated package cannot replace protected runtime file(s): candidate.py"

    Nothing validates files_expected, so only this test stops the wording drifting back.
    """

    rendered = instructions_for(ResearchOperation.PROPOSE)
    assert "must include candidate.py" not in rendered
    assert "never list or return candidate.py" in " ".join(rendered.split())


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_source_operations_state_the_schema_rules_that_cost_an_attempt(
    operation: ResearchOperation,
) -> None:
    """Both rules below are enforced by the strict schema and were previously unstated.

    A repair returned model_impl.py twice and was rejected as malformed, consuming one of only
    two attempts, because nothing told the model paths must be unique.
    """

    rendered = " ".join(instructions_for(operation).split())
    assert "must be unique" in rendered
    assert "Never return candidate.py" in rendered
