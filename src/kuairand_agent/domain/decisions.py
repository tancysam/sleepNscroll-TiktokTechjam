"""Closed decision vocabularies shared across evaluation, replay, and finalization."""

from __future__ import annotations

from enum import StrEnum


class EvidenceStage(StrEnum):
    """Leakage boundary at which an immutable prediction/evaluation artifact is produced."""

    INNER = "INNER"
    OUTER = "OUTER"
    FINAL = "FINAL"


class SubmissionDisposition(StrEnum):
    """Which exact vector is selected for the competition submission."""

    CHALLENGER_SELECTED = "CHALLENGER_SELECTED"
    OFFICIAL_FM_RETAINED = "OFFICIAL_FM_RETAINED"


class ScientificDisposition(StrEnum):
    """What the frozen uncertainty evidence supports scientifically."""

    MATERIALLY_CONFIRMED = "MATERIALLY_CONFIRMED"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    INSUFFICIENT_VALID_EVIDENCE = "INSUFFICIENT_VALID_EVIDENCE"


class ReplayGrade(StrEnum):
    """Named replay guarantees; grades are capabilities, not an implicit ordinal scale."""

    SCORING_EXACT = "SCORING_EXACT"
    EXPERIMENT_SAME_BACKEND = "EXPERIMENT_SAME_BACKEND"
    EXPERIMENT_TOLERANT = "EXPERIMENT_TOLERANT"
    CROSS_BACKEND_PORTABILITY = "CROSS_BACKEND_PORTABILITY"
    BUNDLE_EXACT = "BUNDLE_EXACT"


__all__ = [
    "EvidenceStage",
    "ReplayGrade",
    "ScientificDisposition",
    "SubmissionDisposition",
]
