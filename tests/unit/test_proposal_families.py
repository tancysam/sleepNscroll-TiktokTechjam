"""Guard the closed-family derivation that the proposer and the controller must agree on.

The failure this file exists to prevent was observed in ``runs/box-20260830T161225Z``: the
controller blocked the ``pairwise`` family on cross-run admission evidence, the propose
instructions never mentioned the block, the model proposed ``pairwise`` three times, and the
campaign closed at ``repeated_pre_admission_failure`` having built zero candidates. The list shown
and the list enforced must come from one derivation.
"""

from __future__ import annotations

from kuairand_agent.research.proposal_families import (
    ADMITTED_AND_LOST,
    REPEATEDLY_REJECTED,
    blocked_proposal_families,
    proposal_family_is_blocked,
)


def _record(**values: object) -> dict[str, object]:
    return {"name": "record", "values": values}


def test_admitted_and_unpromoted_family_is_blocked_with_the_measured_reason() -> None:
    """One full inner-fold loss is conclusive; it already cost a real training run."""

    records = [_record(proposal_family="pairwise", branch_outcome="admitted", promoted=False)]
    assert blocked_proposal_families(records) == (("pairwise", ADMITTED_AND_LOST),)
    assert proposal_family_is_blocked(records, proposal_family="pairwise")


def test_repeated_pre_admission_rejection_is_blocked() -> None:
    records = [_record(proposal_family="listwise", proposal_family_blocked=True)]
    assert blocked_proposal_families(records) == (("listwise", REPEATEDLY_REJECTED),)


def test_measured_loss_outranks_a_pre_admission_rejection_as_the_reason() -> None:
    """Both triggers can fire for one family; the reason shown should be the stronger evidence."""

    records = [
        _record(proposal_family="pairwise", proposal_family_blocked=True),
        _record(proposal_family="pairwise", branch_outcome="admitted", promoted=False),
    ]
    assert blocked_proposal_families(records) == (("pairwise", ADMITTED_AND_LOST),)


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
