"""What the accumulated evidence says about each proposal family.

The controller refuses a blocked family deterministically, but only *after* the model has proposed
it. That refusal is cheap in isolation and ruinous in repetition: a campaign whose every proposal
lands on one blocked family closes its portfolio with nothing built, having spent a full data
preparation and qualification cycle to produce zero candidates. Observed exactly that way in
``runs/box-20260830T161225Z``, where three consecutive ``pairwise`` proposals were each blocked and
the campaign ended at ``repeated_pre_admission_failure``.

The proposer therefore needs the same verdicts *before* it chooses an axis. Both the refusal in
:mod:`kuairand_agent.research.production` and the directive rendered into the propose instructions
read this one derivation, so what the model is shown cannot drift from what is enforced against it.

Not every verdict is a wall, and treating them alike was its own defect -- twice, in opposite
directions. Blocking on one measured loss took the most-explored axis in the ledger (nineteen
recorded ``pairwise`` events, five promotions) and walled it off while the briefing simultaneously
told the model that one loss does not exhaust a direction. And the first fix over-corrected the
other way: it made a Fold B fusion discard a permanent super-verdict outranking a full two-fold
loss, so a family screened out on its first configuration got no re-attempt while a family
measured on both folds got one.

One adverse measurement -- either kind -- is now evidence about a configuration and reopens for a
single refined attempt. Two close the family. A pre-admission wall closes it regardless, which is
why :func:`_verdicts` computes the reason shown and the refusal decision separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

# Two adverse measurements close a family for good: the direction was built and scored twice and
# failed twice, which is the case this breaker exists for.
TWICE_MEASURED_AND_LOST: Final = (
    "was built and measured twice and failed both times, so it is closed for good"
)
# One adverse measurement is evidence about a CONFIGURATION, not about a direction. The briefing
# tells the model exactly that -- read the prior candidate_config_json before concluding a family
# is exhausted -- and an absolute block after one result contradicted it, walling off `pairwise` at
# nineteen recorded events and five promotions, the most-explored axis in the ledger.
_REOPENS: Final = (
    " It reopens for ONE refined re-attempt: read that iteration's candidate_config_json, state "
    "what you are changing about it, and make the proposal materially different from the recorded "
    "one. Restating it closes the family"
)
ONCE_MEASURED_AND_LOST: Final = "lost one full inner-fold evaluation." + _REOPENS
# A screen discard is a WEAKER result than a two-fold loss, not a stronger one: the branch never
# reached Fold A. Treating it as a super-verdict inverted the whole principle -- a family measured
# on both folds got a re-attempt while one screened out got none, project-wide and forever. That is
# the run 15 -> run 16 pattern exactly: run 15's identity candidate at -5.15 sigma earned weight
# 0.0, and run 16's refinement of that same direction is the best standalone this project has
# produced. With 53 of 80 recorded grids selecting 0/100, a super-verdict here would close new
# families almost on contact.
ONCE_DISCARDED_BY_FUSION: Final = (
    "was discarded by rank fusion, which gave it weight 0.0 against the official control on the "
    "Fold B screen." + _REOPENS
)
# A family rejected twice before admission never got far enough to be measured, but repeating it a
# third time spends provider calls on a wall the controller will not move.
REPEATEDLY_REJECTED: Final = "was rejected twice in a row before it could be admitted"

#: Verdicts that permit one more attempt. The refusal decision does NOT read this alone -- see
#: :func:`_verdicts` -- because a family can hold a reopenable verdict and a hard wall at once,
#: and the wall must win.
REOPENABLE_REASONS: Final = frozenset({ONCE_MEASURED_AND_LOST, ONCE_DISCARDED_BY_FUSION})


def _measured_loss(values: Mapping[str, object]) -> bool:
    """Did this record measure the family against the incumbent and lose?"""

    return values.get("branch_outcome") == "admitted" and values.get("promoted") is False


def blocked_proposal_families(campaign_records: object) -> tuple[tuple[str, str], ...]:
    """Return ``(family, reason)`` pairs the evidence rules out, ordered by family name.

    ``campaign_records`` is the wire-form record list carried in the safe context -- the same
    records the model itself is shown. Anything that is not a well-formed record list yields no
    blocks, so a malformed context can never fabricate one.

    A family appears here whether it is closed for good or merely conditional; the reason string
    says which, and :func:`proposal_family_is_blocked` is what decides refusal.
    """

    return tuple(sorted((family, reason) for family, (reason, _) in _verdicts(campaign_records)))


def _verdicts(campaign_records: object) -> tuple[tuple[str, tuple[str, bool]], ...]:
    """Return ``family -> (reason, refused)``.

    ``reason`` is for display and ``refused`` is the decision, and they are computed separately on
    purpose. Coupling them -- refusing whichever reason happened to rank highest -- let a family
    holding a hard pre-admission wall AND one measured loss come back proposable, because the loss
    outranked the wall for display.
    """

    if not isinstance(campaign_records, list):
        return ()
    adverse: dict[str, list[str]] = {}
    walled: set[str] = set()
    seen: list[str] = []
    for raw_record in campaign_records:
        if not isinstance(raw_record, Mapping):
            continue
        values = raw_record.get("values")
        if not isinstance(values, Mapping):
            continue
        family = values.get("proposal_family")
        if not isinstance(family, str) or not family:
            continue
        # A two-fold loss and a screen discard are both one adverse measurement of one
        # configuration. They differ in how far the branch got, not in what they license.
        if _measured_loss(values):
            adverse.setdefault(family, []).append(ONCE_MEASURED_AND_LOST)
        elif values.get("fusion_model_discarded") is True:
            adverse.setdefault(family, []).append(ONCE_DISCARDED_BY_FUSION)
        elif values.get("proposal_family_blocked") is True:
            walled.add(family)
        else:
            continue
        if family not in seen:
            seen.append(family)
    return tuple(
        (family, _verdict_for(adverse.get(family, ()), walled=family in walled)) for family in seen
    )


def _verdict_for(adverse: Sequence[str], *, walled: bool) -> tuple[str, bool]:
    """Collapse one family's evidence into ``(reason to show, refuse)``."""

    if len(adverse) > 1:
        return TWICE_MEASURED_AND_LOST, True
    if adverse:
        # A measurement is more informative than a pre-admission wall, so it is what the model is
        # shown -- but the wall still refuses, which is why these are two separate values.
        return adverse[0], walled
    return REPEATEDLY_REJECTED, True


def proposal_family_is_blocked(campaign_records: object, *, proposal_family: str) -> bool:
    """Return whether ``proposal_family`` is refused outright by the accumulated evidence.

    A family with exactly one adverse measurement and no pre-admission wall is NOT refused: it
    reopens for one refined re-attempt, on the reasoning in :data:`ONCE_MEASURED_AND_LOST`.
    """

    return any(
        family == proposal_family and refused
        for family, (_reason, refused) in _verdicts(campaign_records)
    )


__all__ = [
    "ONCE_DISCARDED_BY_FUSION",
    "ONCE_MEASURED_AND_LOST",
    "REOPENABLE_REASONS",
    "REPEATEDLY_REJECTED",
    "TWICE_MEASURED_AND_LOST",
    "blocked_proposal_families",
    "proposal_family_is_blocked",
]
