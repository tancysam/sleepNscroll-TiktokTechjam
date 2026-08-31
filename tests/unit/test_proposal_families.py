"""Guard the family verdicts that the proposer and the controller must agree on.

Two failures this file exists to prevent, both observed:

``runs/box-20260830T161225Z`` -- the controller blocked the ``pairwise`` family on cross-run
admission evidence, the propose instructions never mentioned the block, the model proposed
``pairwise`` three times, and the campaign closed at ``repeated_pre_admission_failure`` having
built zero candidates. The list shown and the list enforced must come from one derivation.

And the inverse: an absolute block after ONE loss contradicted the briefing's own instruction to
read a prior ``candidate_config_json`` before concluding a family is exhausted, and walled off
``pairwise`` at nineteen recorded events and five promotions -- the most-explored and
best-performing axis in the ledger. One loss now reopens for one refined attempt.
"""

from __future__ import annotations

from kuairand_agent.research.prompts import _blocked_family_constraints
from kuairand_agent.research.proposal_families import (
    ONCE_DISCARDED_BY_FUSION,
    ONCE_MEASURED_AND_LOST,
    REPEATEDLY_REJECTED,
    TWICE_MEASURED_AND_LOST,
    blocked_proposal_families,
    proposal_family_is_blocked,
)


def _record(**values: object) -> dict[str, object]:
    return {"name": "record", "values": values}


def _loss(family: str) -> dict[str, object]:
    return _record(proposal_family=family, branch_outcome="admitted", promoted=False)


def test_one_measured_loss_reopens_rather_than_closing() -> None:
    """One loss is evidence about a configuration, not about a direction."""

    records = [_loss("pairwise")]
    assert blocked_proposal_families(records) == (("pairwise", ONCE_MEASURED_AND_LOST),)
    assert not proposal_family_is_blocked(records, proposal_family="pairwise")


def test_a_second_measured_loss_closes_the_family_for_good() -> None:
    """The direction was built and scored twice and failed twice. That is the case for a wall."""

    records = [_loss("pairwise"), _loss("pairwise")]
    assert blocked_proposal_families(records) == (("pairwise", TWICE_MEASURED_AND_LOST),)
    assert proposal_family_is_blocked(records, proposal_family="pairwise")


def _discard(family: str) -> dict[str, object]:
    return _record(proposal_family=family, fusion_model_discarded=True)


def test_a_fusion_discard_counts_but_does_not_close_on_its_own() -> None:
    """It never set ``promoted``, so nothing used to close -- but it is the WEAKER result.

    Rank fusion gave the model weight 0.0, so the recorded primary is the control's score with
    none of the model's ordering in it, and the branch stopped at the Fold B screen without ever
    reaching Fold A. Closing permanently on that would invert the principle this module states:
    a family measured on both folds would get a refined re-attempt while one screened out on its
    first configuration would get none. That is the run 15 -> run 16 pattern, where the refinement
    of a -5.15 sigma direction became the best standalone result in the project.
    """

    records = [_discard("listwise")]
    assert blocked_proposal_families(records) == (("listwise", ONCE_DISCARDED_BY_FUSION),)
    assert not proposal_family_is_blocked(records, proposal_family="listwise")


def test_two_adverse_measurements_close_the_family_whatever_their_kinds() -> None:
    """A discard and a loss are one measurement each; the mix does not matter."""

    for records in (
        [_discard("listwise"), _discard("listwise")],
        [_discard("listwise"), _loss("listwise")],
        [_loss("listwise"), _discard("listwise")],
    ):
        assert blocked_proposal_families(records) == (("listwise", TWICE_MEASURED_AND_LOST),)
        assert proposal_family_is_blocked(records, proposal_family="listwise")


def test_a_pre_admission_wall_still_refuses_a_family_that_also_has_one_measurement() -> None:
    """The refusal decision must not be read off the reason chosen for display.

    Coupling them let a family holding both come back proposable, because the measurement
    outranked the wall for display and the measurement alone is reopenable.
    """

    records = [
        _record(proposal_family="pairwise", proposal_family_blocked=True),
        _loss("pairwise"),
    ]
    assert blocked_proposal_families(records) == (("pairwise", ONCE_MEASURED_AND_LOST),)
    assert proposal_family_is_blocked(records, proposal_family="pairwise")


def test_repeated_pre_admission_rejection_is_blocked() -> None:
    """Never measured, so refinement is not the issue: it is a wall the controller will not move."""

    records = [_record(proposal_family="listwise", proposal_family_blocked=True)]
    assert blocked_proposal_families(records) == (("listwise", REPEATEDLY_REJECTED),)
    assert proposal_family_is_blocked(records, proposal_family="listwise")


def test_stronger_evidence_wins_the_reason_regardless_of_record_order() -> None:
    """Several triggers can fire for one family; the reason shown must be the strongest."""

    weakest_last = [
        _loss("pairwise"),
        _loss("pairwise"),
        _record(proposal_family="pairwise", proposal_family_blocked=True),
    ]
    assert blocked_proposal_families(weakest_last) == (("pairwise", TWICE_MEASURED_AND_LOST),)
    weakest_first = [
        _record(proposal_family="pairwise", proposal_family_blocked=True),
        _loss("pairwise"),
        _loss("pairwise"),
    ]
    assert blocked_proposal_families(weakest_first) == (("pairwise", TWICE_MEASURED_AND_LOST),)


def test_promoted_and_pending_families_stay_open() -> None:
    """A family that won, or that has not been measured yet, must remain proposable."""

    records = [
        _record(proposal_family="pairwise", branch_outcome="admitted", promoted=True),
        _record(proposal_family="listwise", branch_outcome="admitted"),
        _record(proposal_family="duration-bucket"),
    ]
    assert blocked_proposal_families(records) == ()
    assert not proposal_family_is_blocked(records, proposal_family="pairwise")


def test_families_are_ordered_and_deduplicated() -> None:
    records = [
        _record(proposal_family="listwise", proposal_family_blocked=True),
        _record(proposal_family="pairwise", proposal_family_blocked=True),
        _record(proposal_family="listwise", proposal_family_blocked=True),
    ]
    assert [family for family, _reason in blocked_proposal_families(records)] == [
        "listwise",
        "pairwise",
    ]


def test_malformed_context_can_never_fabricate_a_block() -> None:
    """A block is a hard wall, so only a well-formed record may raise one."""

    malformed_contexts: tuple[object, ...] = (
        None,
        {},
        "pairwise",
        [None],
        [{"values": "pairwise"}],
        [{"values": {}}],
    )
    for malformed in malformed_contexts:
        assert blocked_proposal_families(malformed) == ()
    empty_family = [_record(proposal_family="", proposal_family_blocked=True)]
    assert blocked_proposal_families(empty_family) == ()


def test_the_directive_tells_the_model_which_verdict_applies() -> None:
    """Guidance and enforcement must say the same thing; conflating them is the whole defect."""

    directive = _blocked_family_constraints(
        blocked_proposal_families([_loss("pairwise"), _loss("listwise"), _loss("listwise")])
    )
    reopen = directive.index("REOPENABLE")
    closed = directive.index("CLOSED")
    assert closed < reopen, "closed families come first"
    assert "candidate_config_json" in directive[reopen:], "say how to reopen one"
    # The reopenable family must not be described under the wall.
    assert "pairwise" in directive[reopen:]
    assert "listwise" in directive[closed:reopen]


def test_a_directive_with_no_closed_families_omits_the_wall_language() -> None:
    """A campaign whose only verdict is reopenable must not read as though everything is walled."""

    directive = _blocked_family_constraints(blocked_proposal_families([_loss("pairwise")]))
    assert "CLOSED" not in directive
    assert "REOPENABLE" in directive
