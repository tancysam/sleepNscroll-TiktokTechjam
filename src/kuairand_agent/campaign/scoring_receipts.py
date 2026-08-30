"""Pure lookup of reusable, trusted scorer receipts.

The public-query ledger records operational history (candidate, source, request, and query
identities) in addition to the scorer evidence.  This module deliberately projects only the
scorer-input identity from that history.  It never opens or mutates a store: callers provide an
already materialized :class:`~kuairand_agent.campaign.store.OuterQueryLedgerProjection`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Self

from kuairand_agent.campaign.selector import OrganizerMetrics
from kuairand_agent.campaign.store import (
    OuterQueryLedgerProjection,
    OuterQueryProjectionRecord,
)

SCORING_RECEIPT_SCHEMA_VERSION: Final = 1
DEFAULT_MATCHED_SEED_SPLIT_ROLE: Final = "outer_valid_matched_seed"
_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:  # pragma: no cover - constructors validate.
        raise ScoringReceiptError("receipt manifest must be finite JSON") from exc


def _manifest_digest(domain: bytes, manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json(manifest)).hexdigest()


class ScoringReceiptError(ValueError):
    """Raised when a completed scoring record cannot be trusted or is contradictory."""


class ScoringReceiptConflictError(ScoringReceiptError):
    """Raised when one scorer-input identity has different trusted metrics."""


def _digest(value: object, location: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ScoringReceiptError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ScoringReceiptError(f"{location} must be a non-empty string without NUL bytes")
    return value


def _unsigned_seed(value: object, location: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ScoringReceiptError(f"{location} must be an unsigned 32-bit integer")
    return value


def _metric(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringReceiptError(f"{location} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ScoringReceiptError(f"{location} must be a finite number in [0, 1]")
    return result


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ScoringReceiptError(f"{location} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class ScoringEquivalenceIdentity:
    """Exact identity of the inputs consumed by one trusted scorer invocation.

    Operational identities are intentionally absent.  A changed candidate, source, request,
    query, or score-evidence receipt can still refer to the same scorer input when all five fields
    below are equal.
    """

    benchmark_digest: str
    dataset_digest: str
    scorer_digest: str
    split_role: str
    prediction_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("benchmark_digest", "dataset_digest", "scorer_digest", "prediction_digest"):
            _digest(getattr(self, name), f"scoring identity {name}")
        _text(self.split_role, "scoring identity split_role")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-scoring-equivalence-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCORING_RECEIPT_SCHEMA_VERSION,
            "benchmark_digest": self.benchmark_digest,
            "dataset_digest": self.dataset_digest,
            "scorer_digest": self.scorer_digest,
            "split_role": self.split_role,
            "prediction_digest": self.prediction_digest,
        }


@dataclass(frozen=True, slots=True)
class TrustedScoreReceipt:
    """One immutable scorer result and the ledger origin that vouches for it."""

    identity: ScoringEquivalenceIdentity
    metrics: OrganizerMetrics
    origin_query_id: str
    origin_revision: int
    origin_result_digest: str
    origin_evidence_digest: str
    seed: int
    score_evidence_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ScoringEquivalenceIdentity):
            raise ScoringReceiptError("receipt identity must be ScoringEquivalenceIdentity")
        if not isinstance(self.metrics, OrganizerMetrics):
            raise ScoringReceiptError("receipt metrics must be OrganizerMetrics")
        _text(self.origin_query_id, "receipt origin_query_id")
        if type(self.origin_revision) is not int or self.origin_revision < 0:
            raise ScoringReceiptError("receipt origin_revision must be a non-negative integer")
        _digest(self.origin_result_digest, "receipt origin_result_digest")
        _digest(self.origin_evidence_digest, "receipt origin_evidence_digest")
        _unsigned_seed(self.seed, "receipt seed")
        _digest(self.score_evidence_digest, "receipt score_evidence_digest")
        object.__setattr__(
            self,
            "digest",
            _manifest_digest(b"kuairand-trusted-score-receipt-v1\0", self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCORING_RECEIPT_SCHEMA_VERSION,
            "identity": self.identity.manifest(),
            "metrics": {
                "GAUC": self.metrics.gauc,
                "nDCG@5": self.metrics.ndcg_at_5,
                "primary": self.metrics.primary,
            },
            "origin_query_id": self.origin_query_id,
            "origin_revision": self.origin_revision,
            "origin_result_digest": self.origin_result_digest,
            "origin_evidence_digest": self.origin_evidence_digest,
            "seed": self.seed,
            "score_evidence_digest": self.score_evidence_digest,
        }

    @property
    def gauc(self) -> float:
        return self.metrics.gauc

    @property
    def ndcg_at_5(self) -> float:
        return self.metrics.ndcg_at_5

    @property
    def primary(self) -> float:
        return self.metrics.primary

    @property
    def prediction_digest(self) -> str:
        return self.identity.prediction_digest

    @property
    def split_role(self) -> str:
        return self.identity.split_role

    @property
    def benchmark_digest(self) -> str:
        return self.identity.benchmark_digest

    @property
    def dataset_digest(self) -> str:
        return self.identity.dataset_digest

    @property
    def scorer_digest(self) -> str:
        return self.identity.scorer_digest

    # Short aliases make the provenance fields convenient without changing their explicit names.
    @property
    def query_id(self) -> str:
        return self.origin_query_id

    @property
    def revision(self) -> int:
        return self.origin_revision

    @property
    def result_digest(self) -> str:
        return self.origin_result_digest

    @property
    def evidence_digest(self) -> str:
        return self.origin_evidence_digest

    @property
    def origin_score_evidence_digest(self) -> str:
        return self.score_evidence_digest

    @property
    def origin_seed(self) -> int:
        return self.seed


@dataclass(frozen=True, slots=True)
class ScoringReceiptBook:
    """Deterministic, immutable index of completed scorer receipts."""

    _receipts: tuple[TrustedScoreReceipt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self._receipts, tuple) or any(
            not isinstance(item, TrustedScoreReceipt) for item in self._receipts
        ):
            raise ScoringReceiptError("receipt book must contain TrustedScoreReceipt values")
        ordered = tuple(sorted(self._receipts, key=lambda item: _identity_key(item.identity)))
        object.__setattr__(self, "_receipts", ordered)

    @classmethod
    def from_projection(cls, projection: OuterQueryLedgerProjection) -> Self:
        """Build a book from the exact read-only project-ledger projection."""

        if not isinstance(projection, OuterQueryLedgerProjection):
            raise ScoringReceiptError("projection must be OuterQueryLedgerProjection")
        return cls.from_completed_rows(projection.queries)

    @classmethod
    def from_completed_rows(
        cls, rows: Iterable[OuterQueryProjectionRecord | Mapping[str, object]]
    ) -> Self:
        """Build a book from projection rows without accessing persistence.

        The adapter accepts the repository's projection dataclass and mapping-shaped rows for
        callers deserializing a read-only projection.  Non-completed rows are ignored.  A
        completed successful row with malformed evidence is rejected rather than silently made
        reusable.
        """

        if isinstance(rows, (str, bytes, Mapping)):
            raise ScoringReceiptError("completed rows must be an iterable of rows")
        parsed: list[TrustedScoreReceipt] = []
        try:
            iterator = iter(rows)
        except TypeError as exc:
            raise ScoringReceiptError("completed rows must be iterable") from exc
        for row in iterator:
            parsed.extend(_parse_row(row))

        chosen: dict[ScoringEquivalenceIdentity, TrustedScoreReceipt] = {}
        for receipt in sorted(parsed, key=_receipt_origin_key):
            previous = chosen.get(receipt.identity)
            if previous is None:
                chosen[receipt.identity] = receipt
            elif previous.metrics != receipt.metrics:
                raise ScoringReceiptConflictError(
                    "conflicting metrics for one scorer-input identity"
                )
        return cls(tuple(chosen.values()))

    @property
    def receipts(self) -> tuple[TrustedScoreReceipt, ...]:
        """Receipts in canonical identity order."""

        return self._receipts

    def find(self, identity: ScoringEquivalenceIdentity) -> TrustedScoreReceipt | None:
        """Return the receipt for an exact scorer-input identity, if one exists."""

        if not isinstance(identity, ScoringEquivalenceIdentity):
            raise ScoringReceiptError("lookup identity must be ScoringEquivalenceIdentity")
        for receipt in self._receipts:
            if receipt.identity == identity:
                return receipt
        return None

    def __len__(self) -> int:
        return len(self._receipts)

    def __iter__(self) -> Iterator[TrustedScoreReceipt]:
        return iter(self._receipts)


def _identity_key(identity: ScoringEquivalenceIdentity) -> tuple[str, ...]:
    return (
        identity.benchmark_digest,
        identity.dataset_digest,
        identity.scorer_digest,
        identity.split_role,
        identity.prediction_digest,
    )


def _receipt_origin_key(receipt: TrustedScoreReceipt) -> tuple[object, ...]:
    return (
        _identity_key(receipt.identity),
        receipt.origin_revision,
        receipt.origin_query_id,
        receipt.seed,
        receipt.origin_result_digest,
        receipt.origin_evidence_digest,
        receipt.score_evidence_digest,
    )


def _row_value(row: OuterQueryProjectionRecord | Mapping[str, object], name: str) -> object:
    if isinstance(row, OuterQueryProjectionRecord):
        return getattr(row, name)
    if isinstance(row, Mapping):
        try:
            return row[name]
        except KeyError as exc:
            raise ScoringReceiptError(f"projection row is missing {name}") from exc
    raise ScoringReceiptError("completed row must be OuterQueryProjectionRecord or mapping")


def _parse_row(
    row: OuterQueryProjectionRecord | Mapping[str, object],
) -> tuple[TrustedScoreReceipt, ...]:
    state = _row_value(row, "state")
    if state != "COMPLETED":
        return ()
    query_id = _text(_row_value(row, "query_id"), "projection query_id")
    revision = _row_value(row, "reservation_revision")
    if type(revision) is not int or revision < 0:
        raise ScoringReceiptError("projection reservation_revision must be non-negative")
    result_digest = _digest(_row_value(row, "result_digest"), "projection result_digest")
    benchmark = _digest(_row_value(row, "benchmark_digest"), "projection benchmark_digest")
    dataset = _digest(_row_value(row, "dataset_digest"), "projection dataset_digest")
    scorer = _digest(_row_value(row, "scorer_digest"), "projection scorer_digest")
    metadata = _mapping(_row_value(row, "latest_metadata"), "completed metadata")

    if metadata.get("schema_version") != SCORING_RECEIPT_SCHEMA_VERSION:
        raise ScoringReceiptError("completed metadata schema_version is not version 1")
    if metadata.get("kind") != "scientific_outer_promotion_completion":
        raise ScoringReceiptError("completed metadata kind is not a scoring completion")
    if metadata.get("successful") is False:
        return ()
    if metadata.get("successful") is not True:
        raise ScoringReceiptError("completed metadata successful must be true")

    evidence_digest = _digest(metadata.get("evidence_digest"), "completion evidence_digest")
    metadata_revision = metadata.get("reservation_revision")
    if metadata_revision is not None and (
        type(metadata_revision) is not int or metadata_revision != revision
    ):
        raise ScoringReceiptError("completion reservation_revision differs from projection")

    request_digest = metadata.get("request_digest")
    if request_digest is not None:
        _digest(request_digest, "completion request_digest")
    request = metadata.get("request")
    if request is not None:
        request_mapping = _mapping(request, "completion request")
        for name, expected in (
            ("benchmark_digest", benchmark),
            ("dataset_digest", dataset),
            ("scorer_digest", scorer),
        ):
            nested = _digest(request_mapping.get(name), f"completion request {name}")
            if nested != expected:
                raise ScoringReceiptError(f"completion request {name} differs from projection")

    metrics_raw = metadata.get("seed_metrics")
    evidence_raw = metadata.get("trusted_seed_evidence")
    if not isinstance(metrics_raw, Sequence) or isinstance(metrics_raw, (str, bytes)):
        raise ScoringReceiptError("completion seed_metrics must be a list")
    if not isinstance(evidence_raw, Sequence) or isinstance(evidence_raw, (str, bytes)):
        raise ScoringReceiptError("completion trusted_seed_evidence must be a list")
    if len(metrics_raw) == 0 or len(metrics_raw) != len(evidence_raw):
        raise ScoringReceiptError("completion seed metrics and trusted evidence must align")

    split_role_default = metadata.get("split_role", DEFAULT_MATCHED_SEED_SPLIT_ROLE)
    _text(split_role_default, "completion split_role")
    metric_by_seed: dict[int, tuple[OrganizerMetrics, str]] = {}
    for index, (metric_raw, evidence_item_raw) in enumerate(
        zip(metrics_raw, evidence_raw, strict=True)
    ):
        metric_item = _mapping(metric_raw, f"completion seed_metrics[{index}]")
        evidence_item = _mapping(evidence_item_raw, f"completion trusted_seed_evidence[{index}]")
        seed = _unsigned_seed(metric_item.get("seed"), f"completion seed_metrics[{index}].seed")
        evidence_seed = _unsigned_seed(
            evidence_item.get("seed"), f"completion trusted_seed_evidence[{index}].seed"
        )
        if evidence_seed != seed or seed in metric_by_seed:
            raise ScoringReceiptError(
                "completion seed metrics contain duplicate or mismatched seeds"
            )
        gauc = _metric(metric_item.get("GAUC"), f"completion seed_metrics[{index}].GAUC")
        ndcg = _metric(metric_item.get("nDCG@5"), f"completion seed_metrics[{index}].nDCG@5")
        primary = _metric(metric_item.get("primary"), f"completion seed_metrics[{index}].primary")
        metrics = OrganizerMetrics(gauc, ndcg)
        if not math.isclose(primary, metrics.primary, rel_tol=0.0, abs_tol=1e-12):
            raise ScoringReceiptError("completion primary does not equal metric mean")
        prediction = _digest(
            evidence_item.get("prediction_digest"),
            f"completion trusted_seed_evidence[{index}].prediction_digest",
        )
        score_evidence = _digest(
            evidence_item.get("score_evidence_digest"),
            f"completion trusted_seed_evidence[{index}].score_evidence_digest",
        )
        split_role_value = metric_item.get(
            "split_role", evidence_item.get("split_role", split_role_default)
        )
        split_role = _text(split_role_value, f"completion seed {seed} split_role")
        metric_by_seed[seed] = (metrics, split_role)

    receipts: list[TrustedScoreReceipt] = []
    for index, evidence_item_raw in enumerate(evidence_raw):
        evidence_item = _mapping(evidence_item_raw, f"completion trusted_seed_evidence[{index}]")
        seed = _unsigned_seed(evidence_item.get("seed"), f"completion evidence[{index}].seed")
        metrics, split_role = metric_by_seed[seed]
        prediction = _digest(
            evidence_item.get("prediction_digest"),
            f"completion trusted_seed_evidence[{index}].prediction_digest",
        )
        score_evidence = _digest(
            evidence_item.get("score_evidence_digest"),
            f"completion trusted_seed_evidence[{index}].score_evidence_digest",
        )
        receipts.append(
            TrustedScoreReceipt(
                identity=ScoringEquivalenceIdentity(
                    benchmark, dataset, scorer, split_role, prediction
                ),
                metrics=metrics,
                origin_query_id=query_id,
                origin_revision=revision,
                origin_result_digest=result_digest,
                origin_evidence_digest=evidence_digest,
                seed=seed,
                score_evidence_digest=score_evidence,
            )
        )
    return tuple(receipts)


__all__ = [
    "DEFAULT_MATCHED_SEED_SPLIT_ROLE",
    "SCORING_RECEIPT_SCHEMA_VERSION",
    "ScoringEquivalenceIdentity",
    "ScoringReceiptBook",
    "ScoringReceiptConflictError",
    "ScoringReceiptError",
    "TrustedScoreReceipt",
]
