"""Fail-closed wrapper around the immutable KuaiRand organizer evaluator.

Candidate code is allowed to produce only a positional score vector.  This module binds that
vector to trusted row/user/video identity, validates it structurally, and only then invokes the
hash-pinned organizer implementation.  It deliberately does not contain a second metric
implementation: there must be exactly one source of truth for official scores.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.contract import (
    STARTER_FILE_SHA256,
    OrganizerIntegrityError,
    verify_starter_kit,
)
from kuairand_agent.scoring.submission import prediction_digest as canonical_prediction_digest

SCORER_SCHEMA_VERSION: Final = 1
_EVALUATOR_FILENAME: Final = "evaluate.py"
_EVALUATOR_DIGEST: Final = STARTER_FILE_SHA256[_EVALUATOR_FILENAME]
_IMPORT_LOCK: Final = threading.Lock()

type IdentityValue = int | str
type VectorInput = Sequence[object] | npt.NDArray[np.generic]
type OrganizerLabelVector = Sequence[object] | npt.NDArray[np.float32]
type OrganizerEvaluator = Callable[
    [Sequence[IdentityValue], OrganizerLabelVector, Sequence[float], int], Mapping[str, object]
]


class ScoringInputError(ValueError):
    """Raised before organizer code is called when a scoring request is malformed."""


class ScorerContractError(RuntimeError):
    """Raised when the pinned evaluator cannot be loaded or returns an invalid contract."""


def _require_nonempty_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ScoringInputError(f"{field_name} must be a non-empty string without NUL bytes")
    return value


@dataclass(frozen=True, slots=True)
class SplitIdentity:
    """Opaque identity of one scorer-eligible split or train-derived fold."""

    name: str
    token: str
    expected_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_nonempty_text(self.name, "split name"))
        object.__setattr__(self, "token", _require_nonempty_text(self.token, "split token"))
        if type(self.expected_count) is not int or self.expected_count <= 0:
            raise ScoringInputError("split expected_count must be a positive integer")


def _identity_value(value: object, location: str) -> IdentityValue:
    if type(value) is bool:
        raise ScoringInputError(f"{location} must be a non-boolean integer or non-empty string")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value and "\x00" not in value:
        return value
    raise ScoringInputError(f"{location} must be a non-boolean integer or non-empty string")


def _identity_wire_value(value: IdentityValue) -> list[object]:
    # Type tags make integer 1 and string "1" distinct alignment identities.
    return ["i", value] if type(value) is int else ["s", value]


@dataclass(frozen=True, slots=True)
class Alignment:
    """Trusted canonical row identity attached after candidate inference.

    The constructor snapshots all input sequences into immutable tuples.  ``row_ids`` must be the
    canonical zero-based physical split order, while repeated user/video pairs remain valid and
    distinct because their row positions differ.
    """

    split: SplitIdentity
    row_ids: Sequence[object]
    user_ids: Sequence[IdentityValue]
    video_ids: Sequence[IdentityValue]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.split, SplitIdentity):
            raise ScoringInputError("alignment split must be a SplitIdentity")

        normalized_rows: list[int] = []
        for index, value in enumerate(self.row_ids):
            if type(value) is bool or not isinstance(value, Integral):
                raise ScoringInputError(f"row_ids[{index}] must be an integer")
            normalized_rows.append(int(value))
        rows = tuple(normalized_rows)
        users = tuple(
            _identity_value(value, f"user_ids[{index}]")
            for index, value in enumerate(self.user_ids)
        )
        videos = tuple(
            _identity_value(value, f"video_ids[{index}]")
            for index, value in enumerate(self.video_ids)
        )

        expected = self.split.expected_count
        if not (len(rows) == len(users) == len(videos) == expected):
            raise ScoringInputError(
                "alignment lengths must all equal split expected_count "
                f"{expected}; got rows={len(rows)}, users={len(users)}, videos={len(videos)}"
            )
        canonical_rows = tuple(range(expected))
        if rows != canonical_rows:
            raise ScoringInputError("row_ids must be unique, contiguous, and ordered from zero")

        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "user_ids", users)
        object.__setattr__(self, "video_ids", videos)
        payload = {
            "schema_version": SCORER_SCHEMA_VERSION,
            "split": {
                "name": self.split.name,
                "token": self.split.token,
                "expected_count": expected,
            },
            "rows": list(rows),
            "users": [_identity_wire_value(value) for value in users],
            "videos": [_identity_wire_value(value) for value in videos],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        object.__setattr__(self, "digest", hashlib.sha256(encoded).hexdigest())

    @classmethod
    def from_ids(
        cls,
        *,
        split: SplitIdentity,
        user_ids: Sequence[IdentityValue],
        video_ids: Sequence[IdentityValue],
        row_ids: Sequence[object] | None = None,
    ) -> Alignment:
        """Build an alignment, deriving canonical fixture row IDs when they are omitted."""

        rows = tuple(range(split.expected_count)) if row_ids is None else row_ids
        return cls(split=split, row_ids=rows, user_ids=user_ids, video_ids=video_ids)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Typed organizer metric result plus exact scorer and prediction provenance."""

    gauc: float
    ndcg_at_5: float
    primary: float
    users: int
    rows: int
    scorer_digest: str
    prediction_digest: str
    runtime_seconds: float

    @property
    def ndcg5(self) -> float:
        """Concise alias for callers that cannot use the organizer's ``nDCG@5`` spelling."""

        return self.ndcg_at_5

    def as_dict(self) -> dict[str, float | int | str]:
        """Return the stable evidence-manifest representation."""

        return {
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
            "users": self.users,
            "rows": self.rows,
            "scorer_digest": self.scorer_digest,
            "prediction_digest": self.prediction_digest,
            "runtime_seconds": self.runtime_seconds,
        }


def _load_verified_evaluator(starter_dir: Path) -> tuple[OrganizerEvaluator, str, str]:
    """Load the pinned source path without importing a shadowable ``evaluate`` module."""

    root = starter_dir.expanduser().resolve(strict=True)
    verification = verify_starter_kit(root)
    evaluator_path = (root / _EVALUATOR_FILENAME).resolve(strict=True)
    if evaluator_path.parent != root:
        raise OrganizerIntegrityError("organizer evaluator resolved outside the starter directory")

    module_name = f"_kuairand_pinned_evaluator_{_EVALUATOR_DIGEST}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
    if spec is None or spec.loader is None:
        raise ScorerContractError(f"cannot create an import specification for {evaluator_path}")
    module = importlib.util.module_from_spec(spec)

    # SourceFileLoader normally writes __pycache__.  The starter member set is immutable, so run
    # its loader under a process-wide lock with bytecode writes disabled and never register the
    # module under a conventional/importable name.
    with _IMPORT_LOCK:
        if module_name in sys.modules:  # UUID makes this defensive branch practically unreachable.
            raise ScorerContractError("unique organizer module name unexpectedly already exists")
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous

    verification_after = verify_starter_kit(root)
    if verification_after.manifest_sha256 != verification.manifest_sha256:
        raise OrganizerIntegrityError("organizer starter changed while loading the evaluator")
    evaluator = getattr(module, "evaluate", None)
    if not callable(evaluator):
        raise ScorerContractError("pinned organizer module does not define callable evaluate")
    if module.__name__ != module_name:
        raise ScorerContractError("organizer evaluator loaded under an unexpected module name")
    return cast(OrganizerEvaluator, evaluator), verification.manifest_sha256, module_name


def _one_dimensional(value: VectorInput, name: str) -> npt.NDArray[np.generic]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ScoringInputError(f"{name} must be a rectangular one-dimensional vector") from exc
    if array.ndim != 1:
        raise ScoringInputError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ScoringInputError(f"{name} must be non-empty")
    return array


def _validated_labels(value: VectorInput) -> npt.NDArray[np.int8]:
    array = _one_dimensional(value, "labels")
    if array.dtype.kind not in "biuf":
        raise ScoringInputError("labels must be numeric binary values")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ScoringInputError("labels must contain only finite binary values")
    if not np.logical_or(array == 0, array == 1).all():
        raise ScoringInputError("labels must contain only binary values 0 and 1")
    return np.ascontiguousarray(array, dtype=np.int8)


def _validated_scores(value: VectorInput) -> npt.NDArray[np.float64]:
    array = _one_dimensional(value, "scores")
    # Strings that happen to parse as numbers, booleans, objects, and complex values are rejected
    # rather than silently coerced at the trusted boundary.
    if array.dtype.kind not in "iuf":
        raise ScoringInputError("scores must contain real numeric values")
    finite = np.isfinite(array)
    if not finite.all():
        raise ScoringInputError("scores must contain only finite values (no NaN or infinity)")
    converted = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ScoringInputError("scores must remain finite when represented as float64")
    return converted


def _prediction_digest(scores: npt.NDArray[np.float64]) -> str:
    return canonical_prediction_digest(scores)


def _metric(mapping: Mapping[str, object], key: str) -> float:
    raw = mapping.get(key)
    if isinstance(raw, (bool, np.bool_)) or not isinstance(
        raw, (int, float, np.integer, np.floating)
    ):
        raise ScorerContractError(f"organizer result {key!r} is not numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise ScorerContractError(f"organizer result {key!r} is not finite")
    return value


def _count(mapping: Mapping[str, object], key: str) -> int:
    raw = mapping.get(key)
    if type(raw) is not int or raw < 0:
        raise ScorerContractError(f"organizer result {key!r} is not a non-negative integer")
    return raw


class ProtectedScorer:
    """Organizer-compatible scorer bound to one immutable trusted alignment."""

    __slots__ = (
        "_evaluate",
        "_module_name",
        "_starter_dir",
        "_starter_manifest_digest",
        "_trusted_alignment",
    )

    def __init__(self, *, starter_dir: str | Path, trusted_alignment: Alignment) -> None:
        if not isinstance(trusted_alignment, Alignment):
            raise ScoringInputError("trusted_alignment must be an Alignment")
        try:
            root = Path(starter_dir).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise OrganizerIntegrityError(
                f"cannot resolve organizer starter directory: {starter_dir}"
            ) from exc
        evaluator, manifest_digest, module_name = _load_verified_evaluator(root)
        self._starter_dir = root
        self._trusted_alignment = trusted_alignment
        self._evaluate = evaluator
        self._starter_manifest_digest = manifest_digest
        self._module_name = module_name

    @property
    def trusted_alignment(self) -> Alignment:
        return self._trusted_alignment

    @property
    def scorer_digest(self) -> str:
        return _EVALUATOR_DIGEST

    @property
    def organizer_module_name(self) -> str:
        """Expose the non-shadowable import identity for audit evidence."""

        return self._module_name

    def _verify_organizer_unchanged(self) -> None:
        verification = verify_starter_kit(self._starter_dir)
        if verification.manifest_sha256 != self._starter_manifest_digest:
            raise OrganizerIntegrityError(
                "organizer starter manifest changed after scorer creation"
            )

    def score(
        self,
        *,
        alignment: Alignment,
        split: SplitIdentity,
        labels: VectorInput,
        scores: VectorInput,
        expected_count: int | None = None,
    ) -> ScoreResult:
        """Score with the integer labels used by organizer random/popularity rungs."""

        return self._score(
            alignment=alignment,
            split=split,
            labels=labels,
            scores=scores,
            expected_count=expected_count,
            encoded_float32_labels=False,
        )

    def score_with_encoded_labels(
        self,
        *,
        alignment: Alignment,
        split: SplitIdentity,
        labels: VectorInput,
        scores: VectorInput,
        expected_count: int | None = None,
    ) -> ScoreResult:
        """Score with the exact ``float32`` labels emitted by organizer ``data.encode``.

        The immutable organizer evaluator is accidentally scalar-dtype-sensitive: NumPy 2.x
        preserves ``float32`` through its nDCG arithmetic, and ``run_fm`` receives that dtype from
        ``data.encode``.  Trainable-model qualification and candidate scoring use this trusted
        route so aggregate metrics and the FM early-stopping decision match untouched ``run_fm``.
        Random and popularity qualification continue to use :meth:`score`, matching their raw
        integer-label organizer paths.
        """

        return self._score(
            alignment=alignment,
            split=split,
            labels=labels,
            scores=scores,
            expected_count=expected_count,
            encoded_float32_labels=True,
        )

    def _score(
        self,
        *,
        alignment: Alignment,
        split: SplitIdentity,
        labels: VectorInput,
        scores: VectorInput,
        expected_count: int | None,
        encoded_float32_labels: bool,
    ) -> ScoreResult:
        """Validate and score one aligned prediction vector with pinned label semantics."""

        if not isinstance(alignment, Alignment):
            raise ScoringInputError("alignment must be an Alignment")
        if not isinstance(split, SplitIdentity):
            raise ScoringInputError("split must be a SplitIdentity")
        if split != self._trusted_alignment.split:
            raise ScoringInputError("split identity does not match the trusted scoring split")
        if alignment.split != split:
            raise ScoringInputError("alignment split identity does not match the requested split")
        if (
            alignment != self._trusted_alignment
            or alignment.digest != self._trusted_alignment.digest
        ):
            raise ScoringInputError(
                "alignment identity does not match the trusted canonical alignment"
            )

        count = split.expected_count if expected_count is None else expected_count
        if type(count) is not int or count <= 0:
            raise ScoringInputError("expected_count must be a positive integer")
        if count != split.expected_count:
            raise ScoringInputError(
                f"expected_count mismatch: split requires {split.expected_count}, got {count}"
            )

        checked_labels = _validated_labels(labels)
        checked_scores = _validated_scores(scores)
        if checked_labels.size != checked_scores.size:
            raise ScoringInputError(
                "labels and scores must have equal lengths; "
                f"got {checked_labels.size} and {checked_scores.size}"
            )
        if checked_labels.size != count:
            raise ScoringInputError(
                f"scoring vectors must contain exactly {count} rows; got {checked_labels.size}"
            )

        # Reverify all seven members immediately before and after the only organizer call.  The
        # second check makes a concurrent mutation fail closed instead of returning a metric.
        self._verify_organizer_unchanged()
        users = alignment.user_ids
        label_values: OrganizerLabelVector
        if encoded_float32_labels:
            encoded_labels = np.ascontiguousarray(checked_labels, dtype=np.float32)
            encoded_labels.setflags(write=False)
            label_values = encoded_labels
        else:
            label_values = checked_labels.tolist()
        score_values = checked_scores.tolist()
        started = time.perf_counter()
        raw_result = self._evaluate(users, label_values, score_values, 5)
        runtime_seconds = time.perf_counter() - started
        self._verify_organizer_unchanged()

        if not isinstance(raw_result, Mapping):
            raise ScorerContractError("organizer evaluate() did not return a mapping")
        gauc = _metric(raw_result, "GAUC")
        ndcg = _metric(raw_result, "nDCG@5")
        primary = _metric(raw_result, "primary")
        users_count = _count(raw_result, "users")
        rows_count = _count(raw_result, "rows")
        if rows_count != count:
            raise ScorerContractError(
                f"organizer reported {rows_count} rows for an expected {count}-row split"
            )
        if users_count != len(set(users)):
            raise ScorerContractError("organizer reported a user count inconsistent with alignment")
        expected_primary = (
            float((np.float32(gauc) + np.float32(ndcg)) / 2.0)
            if encoded_float32_labels
            else (gauc + ndcg) / 2.0
        )
        if primary != expected_primary:
            raise ScorerContractError("organizer primary is not the mean of GAUC and nDCG@5")

        return ScoreResult(
            gauc=gauc,
            ndcg_at_5=ndcg,
            primary=primary,
            users=users_count,
            rows=rows_count,
            scorer_digest=_EVALUATOR_DIGEST,
            prediction_digest=_prediction_digest(checked_scores),
            runtime_seconds=runtime_seconds,
        )
