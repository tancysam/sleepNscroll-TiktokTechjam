"""Guard the research instruction surface against drifting from the enforcement it describes.

The controller rejects generated packages using the constants in
``kuairand_agent.research.materialize``.  Historically the implementation request transmitted only
the file-count, byte, and suffix limits, so a model was rejected for import, basename, and path
rules it was never told.  These tests fail if any enforced constant stops reaching the model.
"""

from __future__ import annotations

import pytest

from kuairand_agent.research.materialize import (
    ALLOWED_SUFFIXES,
    MAX_GENERATED_FILE_BYTES,
    MAX_GENERATED_FILES,
    MAX_GENERATED_TOTAL_BYTES,
    _FORBIDDEN_BASENAMES,
    _FORBIDDEN_CALLS,
    _FORBIDDEN_IMPORT_ROOTS,
    _TRUSTED_ROOTS,
)
from kuairand_agent.research.prompts import PROMPT_VERSION, instructions_for
from kuairand_agent.research.schemas import ResearchOperation

# Operations whose response carries candidate source and therefore must state the full contract.
_SOURCE_OPERATIONS = (ResearchOperation.IMPLEMENT, ResearchOperation.REPAIR)


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
@pytest.mark.parametrize(
    ("group", "values"),
    [
        ("forbidden import roots", _FORBIDDEN_IMPORT_ROOTS),
        ("forbidden basenames", _FORBIDDEN_BASENAMES),
        ("forbidden calls", _FORBIDDEN_CALLS),
        ("trusted roots", _TRUSTED_ROOTS),
        ("allowed suffixes", ALLOWED_SUFFIXES),
    ],
)
def test_enforced_name_sets_reach_the_model(
    operation: ResearchOperation, group: str, values: frozenset[str]
) -> None:
    """Every name the controller rejects on must be named in the instructions."""

    rendered = instructions_for(operation)
    missing = sorted(value for value in values if value not in rendered)
    assert not missing, f"{operation.value} instructions omit {group}: {missing}"


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_enforced_numeric_limits_reach_the_model(operation: ResearchOperation) -> None:
    """The size limits must appear as their exact enforced values, not as prose approximations."""

    rendered = instructions_for(operation)
    for name, value in (
        ("MAX_GENERATED_FILES", MAX_GENERATED_FILES),
        ("MAX_GENERATED_FILE_BYTES", MAX_GENERATED_FILE_BYTES),
        ("MAX_GENERATED_TOTAL_BYTES", MAX_GENERATED_TOTAL_BYTES),
    ):
        assert str(value) in rendered, f"{operation.value} instructions omit {name}={value}"


@pytest.mark.parametrize("operation", _SOURCE_OPERATIONS)
def test_material_change_semantics_are_explained(operation: ResearchOperation) -> None:
    """The material-change gate rejects more packages than any other; its rules must be stated.

    ``require_material_executable_change`` compares only docstring-stripped top-level ``def`` and
    ``class`` nodes in files reachable from ``candidate.py``.  A model that does not know this
    declares a changed module-level constant and is rejected for a claim that looks dishonest.
    """

    rendered = instructions_for(operation)
    for fragment in ("material_symbols", "candidate.py", "top-level", "docstring", "reachable"):
        assert fragment.lower() in rendered.lower(), (
            f"{operation.value} instructions omit material-change concept {fragment!r}"
        )


def test_every_operation_renders_and_is_distinct() -> None:
    """All four operations must produce non-empty, operation-specific instructions."""

    rendered = {operation: instructions_for(operation) for operation in ResearchOperation}
    assert len(rendered) == 4
    for operation, text in rendered.items():
        assert text.strip()
        assert operation.value.upper() in text, f"{operation.value} instructions lack their header"
    assert len(set(rendered.values())) == 4, "operations must not share identical instructions"


def test_fileless_operations_omit_the_source_contract() -> None:
    """PROPOSE and REFLECT emit no files, so they must not be charged for the source contract.

    This is a token-budget guard: total consumption is scored, and these two operations run on
    every iteration.
    """

    for operation in (ResearchOperation.PROPOSE, ResearchOperation.REFLECT):
        rendered = instructions_for(operation)
        assert "Candidate source contract" not in rendered
        assert "Worked example" not in rendered
        assert len(rendered) < len(instructions_for(ResearchOperation.IMPLEMENT))


def test_benchmark_briefing_states_the_task_contract() -> None:
    """The briefing must pin the real label, metrics, and ceiling on every operation that gets it.

    Two planning documents in this repository encoded ``click`` with NDCG@10/Recall@50 before the
    starter kit resolved the contract.  The model must never be given that stale framing.
    """

    for operation in (ResearchOperation.PROPOSE, ResearchOperation.REFLECT):
        rendered = instructions_for(operation)
        assert "long_view" in rendered
        assert "GAUC" in rendered
        assert "nDCG@5" in rendered
        assert "0.8645" in rendered, "the oracle ceiling must be stated so 1.0 is not chased"
        assert "NDCG@10" not in rendered
        assert "Recall@50" not in rendered


def test_schema_retry_appends_only_a_correction_notice() -> None:
    """A retry must add guidance without altering the instructions already given."""

    base = instructions_for(ResearchOperation.IMPLEMENT)
    retry = instructions_for(ResearchOperation.IMPLEMENT, schema_retry=True)
    assert retry.startswith(base)
    assert "rejected by the local strict parser" in retry[len(base) :]


def test_prompt_version_is_pinned() -> None:
    """PROMPT_VERSION names the response JSON schema; provider tests assert the ``_v1`` suffix."""

    assert PROMPT_VERSION == 1


@pytest.mark.parametrize("bad", [None, "propose", 0, object()])
def test_instructions_reject_a_non_operation(bad: object) -> None:
    with pytest.raises(ValueError, match="operation must be a ResearchOperation"):
        instructions_for(bad)  # type: ignore[arg-type]


def test_instructions_reject_a_non_boolean_retry_flag() -> None:
    with pytest.raises(ValueError, match="schema_retry must be bool"):
        instructions_for(ResearchOperation.PROPOSE, schema_retry=1)  # type: ignore[arg-type]
