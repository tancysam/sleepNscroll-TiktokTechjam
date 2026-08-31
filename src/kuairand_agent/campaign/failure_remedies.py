"""Controller-authored remedies for candidate failures that produce no measurement.

A candidate that raises before evaluation tells the next proposer nothing unless it learns
what to do differently.  ``_invoke_runner`` deliberately discards the exception message,
because that message can quote candidate-authored text and must never re-enter a prompt.
This module restores the useful half of that information without the unsafe half: a fixed
marker written by trusted code selects a remedy also written by trusted code, so what
reaches the model is chosen by the candidate but never authored by it.

Only failures whose cause is a rule the candidate could have obeyed belong here.  A defect
with no general remedy is better reported as an unclassified crash than as a guess.
"""

from __future__ import annotations

from typing import Final

_DIAGNOSTICS_NAMING: Final = (
    "training_diagnostics was refused because a key or string named an official metric. "
    "The evaluator alone may name GAUC, nDCG or the primary score, so any diagnostics key "
    "containing the token gauc, ndcg or primary is rejected before the result is read, and "
    "the training run is discarded even though it succeeded. Report internal model selection "
    "under neutral names -- selected_members, inner_score, inner_score_by_size, guard_margin "
    "-- reporting the number selected on and never the metric's name."
)

_DIAGNOSTICS_NON_FINITE: Final = (
    "training_diagnostics returned a NaN or infinity. Every diagnostic value must be a "
    "finite scalar; guard divisions and log arguments, and omit a statistic rather than "
    "emitting a non-finite placeholder for it."
)

_DIAGNOSTICS_KEY_SHAPE: Final = (
    "training_diagnostics returned a key the schema rejects. Keys must be short "
    "lower-case identifiers, and the whole mapping must be a JSON object of scalars, "
    "lists and nested objects."
)

_DIAGNOSTICS_SIZE: Final = (
    "training_diagnostics exceeded its size limits. Report a summary of the internal "
    "search -- the winning setting and its score -- not a per-epoch or per-row trace."
)

#: Marker text is matched against the controller's own diagnostic. Every marker below is a
#: literal from ``candidate_api.protocol``; none of it originates with a candidate.
CONTROLLER_FAILURE_REMEDIES: Final[tuple[tuple[str, str], ...]] = (
    ("candidate-declared official metric", _DIAGNOSTICS_NAMING),
    ("diagnostics contain a non-finite number", _DIAGNOSTICS_NON_FINITE),
    ("diagnostic key is invalid", _DIAGNOSTICS_KEY_SHAPE),
    ("diagnostics must be a JSON object", _DIAGNOSTICS_KEY_SHAPE),
    ("diagnostics contain a non-JSON value", _DIAGNOSTICS_KEY_SHAPE),
    ("diagnostics exceed the node-count limit", _DIAGNOSTICS_SIZE),
    ("diagnostics exceed the nesting-depth limit", _DIAGNOSTICS_SIZE),
    ("diagnostic string is oversized", _DIAGNOSTICS_SIZE),
)


def remedy_for(diagnostic: object) -> str | None:
    """Return the trusted remedy for a controller-recognised failure, else ``None``.

    The returned text is a module constant. No part of ``diagnostic`` is echoed, so a
    candidate cannot reach the next prompt by choosing what its exception says.
    """

    if not isinstance(diagnostic, str):
        return None
    for marker, remedy in CONTROLLER_FAILURE_REMEDIES:
        if marker in diagnostic:
            return remedy
    return None


__all__ = ["CONTROLLER_FAILURE_REMEDIES", "remedy_for"]
