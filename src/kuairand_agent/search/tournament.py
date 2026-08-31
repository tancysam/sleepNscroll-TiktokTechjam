"""Deterministic matched-control tournament and mechanism-diverse Pareto archive."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum

from kuairand_agent.domain.identity import ExperimentId, FamilyId

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class TournamentError(ValueError):
    """Tournament evidence is incomplete, unmatched, or non-finite."""


class ScientificResult(StrEnum):
    COMPLETED = "completed"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class TournamentDisposition(StrEnum):
    ADVANCE = "advance"
    SCIENTIFIC_REJECTION = "scientific_rejection"
    INFRASTRUCTURE_RETRY = "infrastructure_retry"


@dataclass(frozen=True, slots=True)
class MatchedEvaluationContext:
    """Predeclared conditions which must be identical across a candidate and its controls."""

    row_set_sha256: str
    fold_protocol: str
    fidelity: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.row_set_sha256) is not str
            or _SHA256_RE.fullmatch(self.row_set_sha256) is None
        ):
            raise TournamentError("matched row set must be a lowercase SHA-256 digest")
        for name in ("fold_protocol", "fidelity"):
            value = getattr(self, name)
            if type(value) is not str or not value or "\x00" in value:
                raise TournamentError(f"matched {name} must be non-empty text")
        if type(self.seeds) is not tuple or not self.seeds:
            raise TournamentError("matched seeds must be non-empty")
        if any(type(value) is not int or not 0 <= value <= 2**32 - 1 for value in self.seeds):
            raise TournamentError("matched seeds must be uint32-compatible integers")
        if len(self.seeds) != len(set(self.seeds)):
            raise TournamentError("matched seeds contain duplicates")


@dataclass(frozen=True, slots=True)
class InnerMetrics:
    primary: float
    gauc: float
    ndcg5: float

    def __post_init__(self) -> None:
        for name in ("primary", "gauc", "ndcg5"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise TournamentError(f"inner metric {name} must be a finite float in [0, 1]")
        if not math.isclose(self.primary, (self.gauc + self.ndcg5) / 2.0, abs_tol=1e-12):
            raise TournamentError("inner primary must equal (GAUC + nDCG@5) / 2")


@dataclass(frozen=True, slots=True)
class TournamentEvidence:
    """Inner-only evidence; protected results have no field in this type."""

    experiment_id: ExperimentId
    family_id: FamilyId
    parent_experiment_id: ExperimentId | None
    context: MatchedEvaluationContext
    result: ScientificResult
    metrics: InnerMetrics | None
    complementarity: float | None
    stability: float | None
    runtime_seconds: float
    peak_memory_mb: float
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, ExperimentId):
            raise TournamentError("tournament evidence requires ExperimentId")
        if not isinstance(self.family_id, FamilyId):
            raise TournamentError("tournament evidence requires FamilyId")
        if self.parent_experiment_id is not None and not isinstance(
            self.parent_experiment_id, ExperimentId
        ):
            raise TournamentError("tournament parent requires ExperimentId")
        if not isinstance(self.context, MatchedEvaluationContext):
            raise TournamentError("tournament evidence requires a matched context")
        if not isinstance(self.result, ScientificResult):
            raise TournamentError("tournament result is invalid")
        for name in ("runtime_seconds", "peak_memory_mb"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise TournamentError(f"{name} must be a finite non-negative float")
        if (
            type(self.diagnostic) is not str
            or "\x00" in self.diagnostic
            or len(self.diagnostic) > 1000
        ):
            raise TournamentError("diagnostic must be bounded text without NUL")
        if self.result is ScientificResult.COMPLETED:
            if not isinstance(self.metrics, InnerMetrics):
                raise TournamentError("completed scientific evidence requires inner metrics")
            for name in ("complementarity", "stability"):
                value = getattr(self, name)
                if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise TournamentError(f"completed evidence {name} must be in [0, 1]")
        elif any(
            value is not None for value in (self.metrics, self.complementarity, self.stability)
        ):
            raise TournamentError("infrastructure failures cannot carry scientific measurements")


@dataclass(frozen=True, slots=True)
class TournamentDecision:
    experiment_id: ExperimentId
    disposition: TournamentDisposition
    matched_parent_delta: float | None
    fallback_delta: float | None
    protected_query_allowed: bool
    archive_admitted: bool
    rationale: str


def _dominates(left: TournamentEvidence, right: TournamentEvidence) -> bool:
    if left.metrics is None or right.metrics is None:
        return False
    assert left.complementarity is not None
    assert left.stability is not None
    assert right.complementarity is not None
    assert right.stability is not None
    no_worse = (
        left.metrics.primary >= right.metrics.primary
        and left.complementarity >= right.complementarity
        and left.stability >= right.stability
        and left.runtime_seconds <= right.runtime_seconds
        and left.peak_memory_mb <= right.peak_memory_mb
    )
    strictly_better = (
        left.metrics.primary > right.metrics.primary
        or left.complementarity > right.complementarity
        or left.stability > right.stability
        or left.runtime_seconds < right.runtime_seconds
        or left.peak_memory_mb < right.peak_memory_mb
    )
    return no_worse and strictly_better


def _archive_sort_key(value: TournamentEvidence) -> tuple[float, float, float, float, float, str]:
    if value.metrics is None or value.complementarity is None or value.stability is None:
        raise TournamentError("only completed scientific evidence can enter the archive")
    return (
        -value.metrics.primary,
        -value.complementarity,
        -value.stability,
        value.runtime_seconds,
        value.peak_memory_mb,
        value.experiment_id.value,
    )


@dataclass(slots=True)
class ParetoArchive:
    """Bounded nondominated archive with an explicit per-family diversity cap."""

    max_slots: int = 8
    max_per_family: int = 2
    _entries: list[TournamentEvidence] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if type(self.max_slots) is not int or self.max_slots <= 0:
            raise TournamentError("archive max_slots must be a positive integer")
        if type(self.max_per_family) is not int or not 1 <= self.max_per_family <= self.max_slots:
            raise TournamentError("archive max_per_family must be in [1, max_slots]")

    def add(self, candidate: TournamentEvidence) -> bool:
        if candidate.result is not ScientificResult.COMPLETED:
            return False
        if any(value.experiment_id == candidate.experiment_id for value in self._entries):
            return True
        if any(_dominates(value, candidate) for value in self._entries):
            return False
        survivors = [value for value in self._entries if not _dominates(candidate, value)]
        family_entries = sorted(
            (value for value in survivors if value.family_id == candidate.family_id),
            key=_archive_sort_key,
        )
        if len(family_entries) >= self.max_per_family:
            worst = family_entries[-1]
            if _archive_sort_key(candidate) >= _archive_sort_key(worst):
                return False
            survivors.remove(worst)
        survivors.append(candidate)
        survivors.sort(key=_archive_sort_key)
        self._entries = survivors[: self.max_slots]
        return any(value.experiment_id == candidate.experiment_id for value in self._entries)

    def entries(self) -> tuple[TournamentEvidence, ...]:
        return tuple(self._entries)


@dataclass(slots=True)
class MatchedControlTournament:
    """Compare one candidate with its declared parent and the official fallback."""

    practical_inner_margin: float = 0.002
    archive: ParetoArchive = field(default_factory=ParetoArchive)

    def __post_init__(self) -> None:
        if (
            type(self.practical_inner_margin) is not float
            or not math.isfinite(self.practical_inner_margin)
            or self.practical_inner_margin < 0.0
        ):
            raise TournamentError("practical_inner_margin must be a finite non-negative float")

    def assess(
        self,
        *,
        candidate: TournamentEvidence,
        parent: TournamentEvidence,
        official_fallback: TournamentEvidence,
    ) -> TournamentDecision:
        if candidate.result is ScientificResult.INFRASTRUCTURE_FAILURE:
            return TournamentDecision(
                experiment_id=candidate.experiment_id,
                disposition=TournamentDisposition.INFRASTRUCTURE_RETRY,
                matched_parent_delta=None,
                fallback_delta=None,
                protected_query_allowed=False,
                archive_admitted=False,
                rationale="infrastructure failure is not a scientific loss",
            )
        if (
            parent.result is not ScientificResult.COMPLETED
            or official_fallback.result is not ScientificResult.COMPLETED
        ):
            raise TournamentError("matched parent and official fallback require completed evidence")
        if candidate.parent_experiment_id != parent.experiment_id:
            raise TournamentError("candidate declared matched parent differs from supplied parent")
        if not (candidate.context == parent.context == official_fallback.context):
            raise TournamentError(
                "candidate, parent, and fallback evaluation contexts are not matched"
            )
        assert candidate.metrics is not None
        assert parent.metrics is not None
        assert official_fallback.metrics is not None
        parent_delta = candidate.metrics.primary - parent.metrics.primary
        fallback_delta = candidate.metrics.primary - official_fallback.metrics.primary
        matched_delta = min(parent_delta, fallback_delta)
        if matched_delta < self.practical_inner_margin:
            return TournamentDecision(
                experiment_id=candidate.experiment_id,
                disposition=TournamentDisposition.SCIENTIFIC_REJECTION,
                matched_parent_delta=parent_delta,
                fallback_delta=fallback_delta,
                protected_query_allowed=False,
                archive_admitted=False,
                rationale="candidate is submaterial against a matched control or official fallback",
            )
        admitted = self.archive.add(candidate)
        return TournamentDecision(
            experiment_id=candidate.experiment_id,
            disposition=TournamentDisposition.ADVANCE,
            matched_parent_delta=parent_delta,
            fallback_delta=fallback_delta,
            protected_query_allowed=True,
            archive_admitted=admitted,
            rationale="candidate clears the predeclared matched-control inner margin",
        )
