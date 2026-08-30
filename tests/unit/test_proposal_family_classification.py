"""Classify a proposal by the mechanism it asserts, never by a family it disclaims.

Showing the proposer its closed families made it state, in prose, which family it was avoiding.
A bare substring match then filed those proposals into the very family they were avoiding and the
controller blocked them -- so the guidance that redirected the model was punished by the classifier
that read it. Both texts below are verbatim from ``runs/guided-20260830T170124Z``, where a listwise
proposal and a feature-interaction proposal were each blocked as ``pairwise``.

The error costs are asymmetric, and the tests encode that: a false block wastes a whole campaign,
while a missed block spends one training run that then gets measured on its merits.
"""

from __future__ import annotations

from kuairand_agent.research.production import proposal_family_of
from kuairand_agent.research.schemas import Proposal


def _family(objective: str, mechanism: str, principal_change: str) -> str:
    proposal = Proposal.__new__(Proposal)
    object.__setattr__(proposal, "objective", objective)
    object.__setattr__(proposal, "mechanism", mechanism)
    object.__setattr__(proposal, "principal_change", principal_change)
    return proposal_family_of(proposal)


def test_listwise_proposal_disclaiming_pairwise_is_classified_listwise() -> None:
    """Verbatim from the blocked candidate-01; the mechanism asserted is listwise softmax."""

    family = _family(
        "Full-slate listwise softmax cross-entropy: for mixed user u with positive set P_u, "
        "q_ui=1/|P_u| for i in P_u and 0 otherwise, minimize sum_u w_u[-sum_i q_ui log "
        "softmax(s_u)_i] plus L2 regularization.",
        "For every mixed-label user slate, compute FM scores over the same safe feature matrix. "
        "This makes every negative compete with every positive in the logged slate without "
        "entering the closed pairwise family or sampling unexposed items.",
        "Change only the training objective from pointwise binary log loss to complete "
        "within-user listwise softmax cross-entropy for mixed-label logged slates.",
    )
    assert family == "listwise"


def test_feature_interaction_proposal_disclaiming_pairwise_is_not_pairwise() -> None:
    """Verbatim from the blocked candidate-02; it changes features, not the objective."""

    family = _family(
        "Retain the trusted parent's existing pointwise binary log-loss and regularization so "
        "the experiment isolates the causal engagement-history interaction block; do not use the "
        "closed pairwise family.",
        "Construct deterministic columns from the audited strictly-past click and like smoothed "
        "rates and their id__tab and id__duration_bucket interactions.",
        "Add a training-derived, inference-identical feature transformation containing "
        "strictly-past click/like smoothed-rate residuals and their low-cardinality "
        "id__tab/id__duration_bucket interaction columns.",
    )
    assert family != "pairwise"


def test_an_asserted_pairwise_proposal_is_still_classified_pairwise() -> None:
    """The block must keep working; only disclaimers are exempt."""

    family = _family(
        "Metric-matched pairwise BPR objective over sampled within-user eligible pairs.",
        "Sample a positive row uniformly from the pooled positives of GAUC-eligible users, then "
        "sample one logged negative uniformly from the same user.",
        "Change the training objective from pointwise binary log-loss to within-user BPR.",
    )
    assert family == "pairwise"


def test_a_clean_feature_proposal_is_unaffected() -> None:
    """Verbatim from the candidate that carried no family marker at all."""

    family = _family(
        "Use the exact pointwise binary log-loss objective and optimization procedure of "
        "official-fm-fallback-seed-4; only the engagement-history feature block changes.",
        "Construct tab-conditioned and duration-bucket-conditioned columns using deterministic "
        "one-hot encodings of id__tab and id__duration_bucket.",
        "Add training-derived interactions between strictly-past click/like smoothed-rate "
        "features and only id__tab/id__duration_bucket.",
    )
    assert family not in {"pairwise", "listwise"}
