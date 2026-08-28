"""Trusted, high-precision KuaiRand submission I/O.

Candidate code returns one numeric prediction per canonical input row.  It never owns the
alignment columns: this module attaches those columns from :class:`AlignmentRow` values and
then reads the resulting CSV back before accepting it.  Repeated ``(user_id, video_id)`` pairs
are deliberately preserved because only the contiguous positional ``row_id`` is an identity.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import struct
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

HEADER: Final = ("row_id", "user_id", "video_id", "score")
DEFAULT_PROTECTED_METRICS: Final = ("GAUC", "nDCG@5", "primary")
_PREDICTION_DIGEST_DOMAIN: Final = b"kuairand-agent/predictions/float64-le/v1\0"

type Float64Array = NDArray[np.float64]
type MetricEvaluator = Callable[[Float64Array], Mapping[str, object]]


class SubmissionError(ValueError):
    """Raised when predictions, trusted alignment, or a submission are invalid."""


@dataclass(frozen=True, slots=True)
class AlignmentRow:
    """One trusted canonical alignment row, assigned before any reordering or joining."""

    row_id: int
    user_id: str
    video_id: str

    def __post_init__(self) -> None:
        if type(self.row_id) is not int or self.row_id < 0:
            raise SubmissionError("row_id must be a non-negative built-in int")
        for name, value in (("user_id", self.user_id), ("video_id", self.video_id)):
            if type(value) is not str or not value or "\x00" in value:
                raise SubmissionError(f"{name} must be a non-empty string without NUL bytes")


@dataclass(frozen=True, slots=True)
class ValidatedSubmission:
    """A structurally checked submission and its stable identities."""

    path: Path
    scores: Float64Array
    row_count: int
    prediction_digest: str
    submission_digest: str
    round_trip_identity: bool | None = None
    within_user_order_preserved: bool | None = None
    top5_preserved: bool | None = None
    protected_metrics_preserved: bool | None = None


def validate_alignment(alignment: Sequence[AlignmentRow]) -> None:
    """Require a non-empty, exactly contiguous trusted positional alignment."""

    if not alignment:
        raise SubmissionError("trusted alignment must contain at least one row")
    for expected_row_id, row in enumerate(alignment):
        if type(row) is not AlignmentRow:
            raise SubmissionError(
                f"alignment[{expected_row_id}] must be an AlignmentRow, got {type(row).__name__}"
            )
        if row.row_id != expected_row_id:
            raise SubmissionError(
                "trusted row_id values must be zero-based and contiguous: "
                f"alignment[{expected_row_id}] has row_id={row.row_id}"
            )


def _normalise_scores(
    scores: Iterable[object], *, expected_count: int | None = None
) -> Float64Array:
    if isinstance(scores, np.ndarray):
        if scores.ndim != 1:
            raise SubmissionError(f"scores must be one-dimensional, got shape {scores.shape}")
        if scores.dtype.kind not in "iuf":
            raise SubmissionError(f"scores must have a real numeric dtype, got {scores.dtype}")
        values = np.asarray(scores, dtype=np.float64)
    else:
        converted: list[float] = []
        for index, raw in enumerate(scores):
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
                raise SubmissionError(
                    f"score at index {index} must be a real number, got {type(raw).__name__}"
                )
            converted.append(float(raw))
        values = np.asarray(converted, dtype=np.float64)

    if values.ndim != 1:  # defensive for unusual ndarray subclasses
        raise SubmissionError("scores must be one-dimensional")
    if expected_count is not None and len(values) != expected_count:
        raise SubmissionError(f"score count mismatch: expected {expected_count}, got {len(values)}")
    if len(values) == 0:
        raise SubmissionError("scores must contain at least one value")
    nonfinite = np.flatnonzero(~np.isfinite(values))
    if len(nonfinite):
        index = int(nonfinite[0])
        raise SubmissionError(f"score at index {index} must be finite, got {values[index]!r}")
    return values


def _digest_normalised_predictions(scores: Float64Array) -> str:
    little_endian = np.ascontiguousarray(scores, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(_PREDICTION_DIGEST_DOMAIN)
    digest.update(struct.pack("<Q", len(little_endian)))
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def prediction_digest(scores: Iterable[object]) -> str:
    """Hash the count and canonical little-endian float64 prediction bytes."""

    return _digest_normalised_predictions(_normalise_scores(scores))


def submission_digest(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash the exact submission bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _format_score(score: float) -> str:
    # CPython's float repr is the shortest decimal string that round-trips to the identical
    # binary64 value.  Converting numpy scalars to built-in float avoids numpy-specific reprs.
    return repr(float(score))


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _readonly_owned(scores: Float64Array) -> Float64Array:
    owned = np.array(scores, dtype=np.float64, copy=True)
    owned.setflags(write=False)
    return owned


def read_submission(path: str | Path, alignment: Sequence[AlignmentRow]) -> ValidatedSubmission:
    """Read a CSV and require its exact header, count, row IDs, alignment, and finite scores."""

    validate_alignment(alignment)
    submission_path = Path(path)
    scores = np.empty(len(alignment), dtype=np.float64)
    rows_seen = 0
    try:
        with submission_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if header != list(HEADER):
                raise SubmissionError(
                    f"submission header must be exactly {','.join(HEADER)!r}, got {header!r}"
                )

            for record in reader:
                line_number = reader.line_num
                if len(record) != len(HEADER):
                    raise SubmissionError(
                        f"CSV record ending at line {line_number} has {len(record)} fields; "
                        "expected 4"
                    )
                if rows_seen >= len(alignment):
                    raise SubmissionError(
                        f"submission has extra row at CSV line {line_number}; "
                        f"expected exactly {len(alignment)} data rows"
                    )

                row_id, user_id, video_id, score_text = record
                expected = alignment[rows_seen]
                canonical_row_id = str(rows_seen)
                if row_id != canonical_row_id:
                    raise SubmissionError(
                        f"CSV line {line_number} row_id must be canonical {canonical_row_id!r}, "
                        f"got {row_id!r}"
                    )
                if user_id != expected.user_id or video_id != expected.video_id:
                    raise SubmissionError(
                        f"CSV line {line_number} alignment mismatch at row_id={rows_seen}: "
                        f"expected ({expected.user_id!r}, {expected.video_id!r}), "
                        f"got ({user_id!r}, {video_id!r})"
                    )
                try:
                    score = float(score_text)
                except (OverflowError, ValueError) as exc:
                    raise SubmissionError(
                        f"CSV line {line_number} score is not a float64 number: {score_text!r}"
                    ) from exc
                if not math.isfinite(score):
                    raise SubmissionError(
                        f"CSV line {line_number} score must be finite, got {score_text!r}"
                    )
                scores[rows_seen] = score
                rows_seen += 1
    except (csv.Error, UnicodeError) as exc:
        raise SubmissionError(f"submission is not strict UTF-8 CSV: {exc}") from exc

    if rows_seen != len(alignment):
        raise SubmissionError(
            f"submission is truncated: expected {len(alignment)} data rows, got {rows_seen}"
        )

    readonly_scores = _readonly_owned(scores)
    return ValidatedSubmission(
        path=submission_path,
        scores=readonly_scores,
        row_count=rows_seen,
        prediction_digest=_digest_normalised_predictions(readonly_scores),
        submission_digest=submission_digest(submission_path),
    )


def within_user_order(
    alignment: Sequence[AlignmentRow],
    scores: Iterable[object],
    *,
    top_k: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return stable descending row-ID order per user, optionally truncated to ``top_k``.

    Python's stable sort matches the immutable organizer evaluator's handling of tied scores:
    ties retain canonical input order.  Returning row IDs, rather than item IDs, keeps repeated
    user/video pairs distinct.
    """

    validate_alignment(alignment)
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        raise SubmissionError("top_k must be a positive built-in int or None")
    values = _normalise_scores(scores, expected_count=len(alignment))
    positions_by_user: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(alignment):
        positions_by_user[row.user_id].append(position)

    result: dict[str, tuple[int, ...]] = {}
    for user_id, positions in positions_by_user.items():
        ordered = sorted(positions, key=lambda position: values[position], reverse=True)
        if top_k is not None:
            ordered = ordered[:top_k]
        result[user_id] = tuple(alignment[position].row_id for position in ordered)
    return result


def compare_within_user_order(
    alignment: Sequence[AlignmentRow],
    expected_scores: Iterable[object],
    actual_scores: Iterable[object],
) -> bool:
    """Return whether all organizer-stable within-user orders are identical."""

    return within_user_order(alignment, expected_scores) == within_user_order(
        alignment, actual_scores
    )


def compare_within_user_top5(
    alignment: Sequence[AlignmentRow],
    expected_scores: Iterable[object],
    actual_scores: Iterable[object],
) -> bool:
    """Return whether every user's organizer-stable top-five row IDs are identical."""

    return within_user_order(alignment, expected_scores, top_k=5) == within_user_order(
        alignment, actual_scores, top_k=5
    )


def _metric_values(
    evaluator: MetricEvaluator,
    scores: Float64Array,
    metric_names: Sequence[str],
) -> dict[str, float]:
    evaluator_input = _readonly_owned(scores)
    result = evaluator(evaluator_input)
    if not isinstance(result, Mapping):
        raise SubmissionError("protected metric evaluator must return a mapping")
    values: dict[str, float] = {}
    for name in metric_names:
        if name not in result:
            raise SubmissionError(f"protected metric evaluator omitted {name!r}")
        raw = result[name]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise SubmissionError(f"protected metric {name!r} must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise SubmissionError(f"protected metric {name!r} must be finite")
        values[name] = value
    return values


def compare_protected_metrics(
    expected_scores: Iterable[object],
    actual_scores: Iterable[object],
    evaluator: MetricEvaluator,
    *,
    metric_names: Sequence[str] = DEFAULT_PROTECTED_METRICS,
    absolute_tolerance: float = 0.0,
) -> bool:
    """Compare selected protected aggregate metrics before and after serialization."""

    if not metric_names or any(type(name) is not str or not name for name in metric_names):
        raise SubmissionError("metric_names must contain non-empty strings")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise SubmissionError("absolute_tolerance must be finite and non-negative")
    expected = _normalise_scores(expected_scores)
    actual = _normalise_scores(actual_scores, expected_count=len(expected))
    before = _metric_values(evaluator, expected, metric_names)
    after = _metric_values(evaluator, actual, metric_names)
    return all(
        math.isclose(before[name], after[name], rel_tol=0.0, abs_tol=absolute_tolerance)
        for name in metric_names
    )


def validate_submission(
    path: str | Path,
    alignment: Sequence[AlignmentRow],
    *,
    reference_scores: Iterable[object] | None = None,
    protected_metric_evaluator: MetricEvaluator | None = None,
    metric_tolerance: float = 0.0,
) -> ValidatedSubmission:
    """Validate a file and, when predictions are supplied, prove rank and metric parity."""

    checked = read_submission(path, alignment)
    if reference_scores is None:
        if protected_metric_evaluator is not None:
            raise SubmissionError(
                "reference_scores are required when protected_metric_evaluator is supplied"
            )
        return checked

    reference = _normalise_scores(reference_scores, expected_count=len(alignment))
    round_trip_identity = reference.tobytes(order="C") == checked.scores.tobytes(order="C")
    order_preserved = compare_within_user_order(alignment, reference, checked.scores)
    top5_preserved = compare_within_user_top5(alignment, reference, checked.scores)
    if not order_preserved or not top5_preserved:
        raise SubmissionError("submission serialization changed within-user ordering or top-five")

    metric_preserved: bool | None = None
    if protected_metric_evaluator is not None:
        metric_preserved = compare_protected_metrics(
            reference,
            checked.scores,
            protected_metric_evaluator,
            absolute_tolerance=metric_tolerance,
        )
        if not metric_preserved:
            raise SubmissionError("submission serialization changed protected metrics")

    return replace(
        checked,
        round_trip_identity=round_trip_identity,
        within_user_order_preserved=order_preserved,
        top5_preserved=top5_preserved,
        protected_metrics_preserved=metric_preserved,
    )


def write_submission(
    path: str | Path,
    alignment: Sequence[AlignmentRow],
    scores: Iterable[object],
    *,
    protected_metric_evaluator: MetricEvaluator | None = None,
    metric_tolerance: float = 0.0,
) -> ValidatedSubmission:
    """Atomically write, read back, and validate a deterministic high-precision CSV."""

    validate_alignment(alignment)
    values = _normalise_scores(scores, expected_count=len(alignment))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    installed = False
    committed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            for row, score in zip(alignment, values, strict=True):
                writer.writerow(
                    (row.row_id, row.user_id, row.video_id, _format_score(float(score)))
                )
            handle.flush()
            os.fsync(handle.fileno())

        staged = validate_submission(
            temporary_path,
            alignment,
            reference_scores=values,
            protected_metric_evaluator=protected_metric_evaluator,
            metric_tolerance=metric_tolerance,
        )
        if not staged.round_trip_identity:
            raise SubmissionError("high-precision score serialization did not round-trip exactly")
        try:
            os.link(temporary_path, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise SubmissionError(
                f"refusing to overwrite existing submission: {destination}"
            ) from exc
        except OSError as exc:
            raise SubmissionError(f"cannot atomically install submission: {destination}") from exc
        installed = True
        _fsync_directory(destination.parent)

        # Validate the artifact through its final path; this also computes the digest of the exact
        # bytes that downstream finalization will consume.
        final = validate_submission(
            destination,
            alignment,
            reference_scores=values,
            protected_metric_evaluator=protected_metric_evaluator,
            metric_tolerance=metric_tolerance,
        )
        if not final.round_trip_identity:
            raise SubmissionError("final score bytes differ from the source float64 predictions")
        committed = True
        return final
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if installed and not committed:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
