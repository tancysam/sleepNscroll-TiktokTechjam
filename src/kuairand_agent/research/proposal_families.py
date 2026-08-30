"""Which proposal families the accumulated evidence has already closed.

The controller refuses a blocked family deterministically, but only *after* the model has proposed
it. That refusal is cheap in isolation and ruinous in repetition: a campaign whose every proposal
lands on one blocked family closes its portfolio with nothing built, having spent a full data
preparation and qualification cycle to produce zero candidates. Observed exactly that way in
``runs/box-20260830T161225Z``, where three consecutive ``pairwise`` proposals were each blocked and
the campaign ended at ``repeated_pre_admission_failure``.

The proposer therefore needs the same list *before* it chooses an axis. Both the refusal in
:mod:`kuairand_agent.research.production` and the directive rendered into the propose instructions
read this one derivation, so the list the model is shown cannot drift from the list enforced
against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

# A family that reached a full inner-fold evaluation and lost has already cost a real training run;
# the organizer's own comparison was run and the family lost it. One occurrence is conclusive.
ADMITTED_AND_LOST: Final = (
    "already reached a full inner-fold evaluation against the incumbent and lost"
)
# A family rejected twice before admission never got far enough to be measured, but repeating it a
# third time spends provider calls on a wall the controller will not move.
REPEATEDLY_REJECTED: Final = "was rejected twice in a row before it could be admitted"


def blocked_proposal_families(campaign_records: object) -> tuple[tuple[str, str], ...]:
    """Return ``(family, reason)`` pairs the evidence rules out, ordered by family name.

    ``campaign_records`` is the wire-form record list carried in the safe context -- the same
    records the model itself is shown. Anything that is not a well-formed record list yields no
    blocks, so a malformed context can never fabricate one.
    """

    if not isinstance(campaign_records, list):
        return ()
    reasons: dict[str, str] = {}
    for raw_record in campaign_records:
        if not isinstance(raw_record, Mapping):
            continue
        values = raw_record.get("values")
        if not isinstance(values, Mapping):
            continue
        family = values.get("proposal_family")
        if not isinstance(family, str) or not family:
            continue
        if values.get("branch_outcome") == "admitted" and values.get("promoted") is False:
            # Measured evidence outranks a pre-admission rejection, so it overwrites.
            reasons[family] = ADMITTED_AND_LOST
        elif values.get("proposal_family_blocked") is True:
            reasons.setdefault(family, REPEATEDLY_REJECTED)
    return tuple(sorted(reasons.items()))


def proposal_family_is_blocked(campaign_records: object, *, proposal_family: str) -> bool:
    """Return whether ``proposal_family`` is closed by the evidence in ``campaign_records``."""

    return any(
        family == proposal_family for family, _reason in blocked_proposal_families(campaign_records)
    )


__all__ = [
    "ADMITTED_AND_LOST",
    "REPEATEDLY_REJECTED",
    "blocked_proposal_families",
    "proposal_family_is_blocked",
]
