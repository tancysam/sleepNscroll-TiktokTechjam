"""Fail-closed production loader for the immutable official-FM qualification.

The qualification coordinator writes a self-digesting logical manifest and an index of every
retained evidence file.  Campaign code must not cherry-pick paths from that JSON: doing so would
silently trust missing, replaced, or cross-dataset artifacts.  This module is the single bridge
from WP3 evidence into later scientific runs.  It verifies the complete physical closure, the
public benchmark/data/scorer identities, the exact five-seed qualification gates, and the copied
seed-4 fallback before exposing only the three permitted public-confirmation seeds and fallback.

No final-period outcome member is opened here.  The only final-period file in the qualification
closure is a prediction submission, whose bytes are hashed but never scored.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path, PurePosixPath
from typing import Final, cast

from kuairand_agent.baselines.artifacts import (
    BaselineArtifactError,
    load_checkpoint,
    load_predictions,
)
from kuairand_agent.baselines.encoding import StarterEncoding, StarterEncodingError
from kuairand_agent.baselines.qualification import QualificationError, QualificationMetrics
from kuairand_agent.campaign.provenance import (
    ProvenanceError,
    load_qualification_identity,
)
from kuairand_agent.contract import (
    BENCHMARK_CONTRACT,
    DATASET_ARCHIVE_MD5,
    DATASET_ARCHIVE_SHA256,
)

QUALIFICATION_EVIDENCE_SCHEMA_VERSION: Final = 1
OUTER_BASELINE_SEEDS: Final = (0, 1, 2)
_QUALIFICATION_SEEDS: Final = (0, 1, 2, 3, 4)
_FALLBACK_SEED: Final = 4
_MAX_JSON_BYTES: Final = 8 * 1024 * 1024
_MAX_ARTIFACT_BYTES: Final = 2 * 1024 * 1024 * 1024
_ARTIFACT_FILENAMES: Final = (
    "encoding.npz",
    "checkpoint.npz",
    "validation-predictions.npy",
)
_FINAL_ZERO_COUNTERS: Final = (
    "outcome_cells_materialized",
    "outcome_cells_decoded",
    "outcome_cells_converted",
    "outcome_cells_validated",
    "outcome_cells_aggregated",
    "outcome_cells_logged",
    "outcome_cells_scored",
)
_ROOT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "benchmark_digest",
        "qualification_input_digest",
        "double_build_identity",
        "rungs",
        "fm",
        "launch_accounting",
        "fallback",
        "resource_usage",
        "final_period",
        "artifacts",
        "digest",
    }
)
_RUN_FIELDS: Final = frozenset(
    {
        "artifact_file_sha256",
        "artifacts",
        "checkpoint_digest",
        "encoding_digest",
        "config_digest",
        "starter_manifest_digest",
        "organizer_parity_passed",
        "seed",
        "validation_metrics",
        "validation_prediction_digest",
        "training_trace",
        "resources",
    }
)
_FALLBACK_FIELDS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "seed",
        "validation_metrics",
        "checkpoint_digest",
        "encoding_digest",
        "config_digest",
        "validation_prediction_digest",
        "validation_submission",
        "final_submission",
        "source_model_tree_digest",
        "fallback_model_tree_digest",
        "replay_verified",
        "clean_seed_zero_retrain_verified",
        "digest",
    }
)

type JsonObject = dict[str, object]


class QualificationEvidenceError(RuntimeError):
    """Raised when retained qualification evidence is incomplete or inconsistent."""


def _require_digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualificationEvidenceError(f"{location} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: object, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise QualificationEvidenceError(f"{location} must be a positive integer")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise QualificationEvidenceError("qualification evidence is not canonical JSON") from exc


def _logical_digest(value: Mapping[str, object], location: str) -> str:
    observed = _require_digest(value.get("digest"), f"{location}.digest")
    body = {key: item for key, item in value.items() if key != "digest"}
    if hashlib.sha256(_canonical_json(body)).hexdigest() != observed:
        raise QualificationEvidenceError(f"{location} logical digest mismatch")
    return observed


def _hash_regular(path: Path, *, maximum: int, location: str) -> tuple[int, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise QualificationEvidenceError(f"cannot inspect {location}: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise QualificationEvidenceError(f"{location} must be a regular non-symlink file")
    if before.st_size <= 0 or before.st_size > maximum:
        raise QualificationEvidenceError(f"{location} size is outside the supported bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationEvidenceError(f"cannot open {location}: {path}") from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size
            ):
                raise QualificationEvidenceError(f"{location} changed while being opened")
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if (
                after.st_dev != opened.st_dev
                or after.st_ino != opened.st_ino
                or after.st_size != opened.st_size
            ):
                raise QualificationEvidenceError(f"{location} changed while being hashed")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return before.st_size, digest.hexdigest()


def _json_object(path: Path, location: str) -> JsonObject:
    size, _ = _hash_regular(path, maximum=_MAX_JSON_BYTES, location=location)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise QualificationEvidenceError(f"cannot read {location}: {path}") from exc
    if len(payload) != size:
        raise QualificationEvidenceError(f"{location} changed between hashing and reading")
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationEvidenceError(f"{location} must be strict ASCII JSON") from exc
    if not isinstance(decoded, dict) or any(type(key) is not str for key in decoded):
        raise QualificationEvidenceError(f"{location} must be a JSON object")
    return cast(JsonObject, decoded)


def _object(value: object, location: str) -> JsonObject:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise QualificationEvidenceError(f"{location} must be an object")
    return cast(JsonObject, value)


def _sequence(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise QualificationEvidenceError(f"{location} must be an array")
    return cast(list[object], value)


def _artifact_relative_path(value: object, location: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise QualificationEvidenceError(f"{location} must be a canonical relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise QualificationEvidenceError(f"{location} must stay inside the qualification root")
    if parsed.as_posix() != value:
        raise QualificationEvidenceError(f"{location} must be a canonical relative POSIX path")
    return value


def _artifact_index(root: Path, manifest: JsonObject) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(_sequence(manifest.get("artifacts"), "qualification artifacts")):
        record = _object(raw, f"qualification artifacts[{index}]")
        if set(record) != {"path", "size", "sha256"}:
            raise QualificationEvidenceError("qualification artifact records use an unknown schema")
        relative = _artifact_relative_path(record.get("path"), f"artifact[{index}].path")
        size = _require_positive_int(record.get("size"), f"artifact {relative} size")
        sha256 = _require_digest(record.get("sha256"), f"artifact {relative} SHA-256")
        if relative in indexed:
            raise QualificationEvidenceError(
                f"qualification artifact path is duplicated: {relative}"
            )
        indexed[relative] = {"path": relative, "size": size, "sha256": sha256}

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise QualificationEvidenceError(f"qualification tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise QualificationEvidenceError(f"qualification tree contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative != "manifest.json":
            actual.add(relative)
    if actual != set(indexed):
        missing = sorted(set(indexed) - actual)
        unexpected = sorted(actual - set(indexed))
        raise QualificationEvidenceError(
            f"qualification artifact closure mismatch: missing={missing}, unexpected={unexpected}"
        )
    for relative, record in indexed.items():
        size, sha256 = _hash_regular(
            root / relative,
            maximum=_MAX_ARTIFACT_BYTES,
            location=f"qualification artifact {relative}",
        )
        if size != record["size"] or sha256 != record["sha256"]:
            raise QualificationEvidenceError(f"qualification artifact changed: {relative}")
    return indexed


def _tree_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise QualificationEvidenceError(f"model tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise QualificationEvidenceError(f"model tree contains a special file: {path}")
        size, sha256 = _hash_regular(
            path,
            maximum=_MAX_ARTIFACT_BYTES,
            location=f"model artifact {path}",
        )
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": sha256,
            }
        )
    if not records:
        raise QualificationEvidenceError(f"model tree is empty: {root}")
    return records


def _tree_digest(root: Path) -> tuple[list[dict[str, object]], str]:
    records = _tree_records(root)
    return records, hashlib.sha256(_canonical_json(records)).hexdigest()


@dataclass(frozen=True, slots=True)
class QualificationExpectations:
    """Current trusted identities that retained WP3 evidence must match exactly."""

    canonical_digest: str
    starter_manifest_digest: str
    scorer_digest: str
    validation_row_count: int
    final_row_count: int

    def __post_init__(self) -> None:
        for name in ("canonical_digest", "starter_manifest_digest", "scorer_digest"):
            object.__setattr__(self, name, _require_digest(getattr(self, name), name))
        for name in ("validation_row_count", "final_row_count"):
            object.__setattr__(self, name, _require_positive_int(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class OfficialFMResourceEvidence:
    """Verified CPU training resource receipt for one qualified FM seed."""

    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    disk_bytes: int
    device: str

    def __post_init__(self) -> None:
        for name in ("wall_seconds", "cpu_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise QualificationEvidenceError(
                    f"official FM resource {name} must be finite and non-negative"
                )
        for name in ("peak_rss_bytes", "disk_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise QualificationEvidenceError(
                    f"official FM resource {name} must be a non-negative integer"
                )
        if self.device != "cpu":
            raise QualificationEvidenceError("official FM qualification must remain on CPU")


@dataclass(frozen=True, slots=True)
class OfficialFMSeedEvidence:
    """One decoded, physically verified public-validation official-FM run."""

    seed: int
    metrics: QualificationMetrics
    validation_prediction_digest: str
    checkpoint_digest: str
    config_digest: str
    encoding_digest: str
    checkpoint_path: Path
    encoding_path: Path
    validation_predictions_path: Path
    checkpoint_file_sha256: str
    encoding_file_sha256: str
    validation_predictions_file_sha256: str
    resources: OfficialFMResourceEvidence


@dataclass(frozen=True, slots=True)
class OfficialFMFallbackEvidence:
    """The immutable seed-4 copied fallback and its executable artifact closure."""

    seed: int
    metrics: QualificationMetrics
    manifest_digest: str
    source_model_tree_digest: str
    fallback_model_tree_digest: str
    validation_prediction_digest: str
    checkpoint_digest: str
    config_digest: str
    encoding_digest: str
    model_dir: Path
    checkpoint_path: Path
    encoding_path: Path
    validation_predictions_path: Path
    checkpoint_file_sha256: str
    encoding_file_sha256: str
    validation_predictions_file_sha256: str
    final_submission_path: Path
    final_submission_file_sha256: str
    final_prediction_digest: str
    final_row_count: int
    resources: OfficialFMResourceEvidence


@dataclass(frozen=True, slots=True)
class OfficialFMQualificationEvidence:
    """Production-safe qualified baseline evidence for matched-seed fusion and fallback."""

    root: Path
    manifest_digest: str
    qualification_input_digest: str
    benchmark_digest: str
    canonical_digest: str
    audit_digest: str
    starter_manifest_digest: str
    scorer_digest: str
    validation_row_count: int
    final_row_count: int
    outer_runs: tuple[OfficialFMSeedEvidence, ...]
    fallback: OfficialFMFallbackEvidence
    schema_version: int = QUALIFICATION_EVIDENCE_SCHEMA_VERSION

    def outer_seed(self, seed: int) -> OfficialFMSeedEvidence:
        """Return one of the three pre-authorized public confirmation seed baselines."""

        if type(seed) is not int or seed not in OUTER_BASELINE_SEEDS:
            raise QualificationEvidenceError(
                f"outer FM seed must be one of {OUTER_BASELINE_SEEDS}, got {seed!r}"
            )
        for run in self.outer_runs:
            if run.seed == seed:
                return run
        raise QualificationEvidenceError(f"qualified outer FM seed {seed} is unavailable")


def _metrics(value: object, location: str) -> QualificationMetrics:
    raw = _object(value, location)
    if set(raw) != {"GAUC", "nDCG@5", "primary"}:
        raise QualificationEvidenceError(f"{location} must contain GAUC, nDCG@5, and primary")
    converted: dict[str, float] = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        item = raw[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise QualificationEvidenceError(f"{location}.{name} must be numeric")
        converted[name] = float(item)
    try:
        return QualificationMetrics(
            converted["GAUC"],
            converted["nDCG@5"],
            converted["primary"],
            label_protocol="encoded_labels_float32",
        )
    except QualificationError as exc:
        raise QualificationEvidenceError(f"{location} is invalid: {exc}") from exc


def _metric_mapping(metrics: QualificationMetrics) -> dict[str, float]:
    return metrics.manifest()


def _require_mapping_equal(left: object, right: object, location: str) -> None:
    if left != right:
        raise QualificationEvidenceError(f"{location} identity mismatch")


def _canonical_split_identities(
    snapshot: JsonObject,
    expectations: QualificationExpectations,
) -> tuple[str, str]:
    canonical = _object(snapshot.get("canonical_manifest"), "canonical manifest")
    if _require_digest(canonical.get("digest"), "canonical manifest digest") != (
        expectations.canonical_digest
    ):
        raise QualificationEvidenceError("qualification canonical dataset identity mismatch")
    splits = _sequence(canonical.get("splits"), "canonical splits")
    if len(splits) != 3:
        raise QualificationEvidenceError("canonical manifest must contain train, valid, and test")
    by_name: dict[str, JsonObject] = {}
    for index, raw in enumerate(splits):
        split = _object(raw, f"canonical splits[{index}]")
        name = split.get("name")
        if type(name) is not str or name in by_name:
            raise QualificationEvidenceError("canonical split names are invalid or duplicated")
        by_name[name] = split
    if tuple(by_name) != ("train", "valid", "test"):
        raise QualificationEvidenceError("canonical split order must be train, valid, test")
    train = by_name["train"]
    valid = by_name["valid"]
    final = by_name["test"]
    if train.get("target_access") != "training" or valid.get("target_access") != (
        "protected_scorer_only"
    ):
        raise QualificationEvidenceError("qualification train/validation target policy changed")
    # ``CanonicalFinalSplit`` is structurally incapable of carrying targets and therefore omits
    # both target-shaped keys from its current manifest.  Accept missing/``None`` and the exact
    # legacy ``"none"`` sentinel retained by older sound qualification receipts.  Every other
    # access claim and every non-null digest still fails closed.
    final_target_access = final.get("target_access")
    if (
        (final_target_access is not None and final_target_access != "none")
        or final.get("target_digest") is not None
        or _require_positive_int(final.get("row_count"), "final split row count")
        != expectations.final_row_count
    ):
        raise QualificationEvidenceError("qualification final split exposed target capability")
    if (
        _require_positive_int(valid.get("row_count"), "validation split row count")
        != expectations.validation_row_count
    ):
        raise QualificationEvidenceError("qualification validation row count mismatch")
    final_access = _object(final.get("outcome_access"), "final outcome access")
    if (
        final_access.get("parsed_cell_count") != 0
        or final_access.get("skipped_values_recorded") is not False
    ):
        raise QualificationEvidenceError("qualification final outcomes were materialized")
    return (
        _require_digest(train.get("inputs_digest"), "train inputs digest"),
        _require_digest(valid.get("inputs_digest"), "validation inputs digest"),
    )


def _require_snapshot(
    root: Path,
    manifest: JsonObject,
    expectations: QualificationExpectations,
) -> tuple[str, str, str]:
    first = _json_object(root / "verification" / "snapshot-first.json", "first snapshot")
    second = _json_object(root / "verification" / "snapshot-second.json", "second snapshot")
    if first != second or manifest.get("double_build_identity") is not True:
        raise QualificationEvidenceError("qualification audit/canonical double build is not exact")
    if set(first) != {
        "starter_manifest_digest",
        "audit_digest",
        "audit_manifest",
        "canonical_digest",
        "canonical_manifest",
        "evaluator_golden_digest",
        "evaluator_golden_passed",
        "benchmark_digest",
        "validation_alignment_count",
        "validation_label_digest",
        "final_alignment_count",
        "final_target_capability",
    }:
        raise QualificationEvidenceError("qualification snapshot uses an unknown schema")
    input_digest = _require_digest(
        manifest.get("qualification_input_digest"), "qualification input digest"
    )
    if hashlib.sha256(_canonical_json(first)).hexdigest() != input_digest:
        raise QualificationEvidenceError("qualification snapshot logical digest mismatch")
    if first.get("benchmark_digest") != BENCHMARK_CONTRACT.digest:
        raise QualificationEvidenceError("snapshot benchmark identity mismatch")
    if first.get("evaluator_golden_passed") is not True:
        raise QualificationEvidenceError("qualification evaluator golden fixture did not pass")
    if first.get("final_target_capability") is not None:
        raise QualificationEvidenceError("qualification snapshot retained final target capability")
    if first.get("starter_manifest_digest") != expectations.starter_manifest_digest:
        raise QualificationEvidenceError("qualification starter-kit identity mismatch")
    if first.get("canonical_digest") != expectations.canonical_digest:
        raise QualificationEvidenceError("qualification canonical dataset identity mismatch")
    if first.get("validation_alignment_count") != expectations.validation_row_count:
        raise QualificationEvidenceError("qualification validation alignment count mismatch")
    if first.get("final_alignment_count") != expectations.final_row_count:
        raise QualificationEvidenceError("qualification final alignment count mismatch")
    _require_digest(first.get("validation_label_digest"), "validation label digest")

    audit_digest = _require_digest(first.get("audit_digest"), "audit digest")
    audit = _object(first.get("audit_manifest"), "audit manifest")
    if audit.get("digest") != audit_digest:
        raise QualificationEvidenceError("qualification audit manifest identity mismatch")
    archive = _object(audit.get("archive_identity"), "qualification archive identity")
    if archive.get("md5") != DATASET_ARCHIVE_MD5 or archive.get("sha256") != (
        DATASET_ARCHIVE_SHA256
    ):
        raise QualificationEvidenceError("qualification used another dataset archive identity")
    trace = _object(audit.get("final_outcome_trace"), "final outcome trace")
    if trace.get("row_count") != expectations.final_row_count:
        raise QualificationEvidenceError("qualification final trace row count mismatch")
    if any(trace.get(name) != 0 for name in _FINAL_ZERO_COUNTERS):
        raise QualificationEvidenceError("qualification final outcome trace is nonzero")
    if trace.get("skipped_values_recorded") is not False:
        raise QualificationEvidenceError("qualification retained skipped final outcome values")
    train_inputs_digest, validation_inputs_digest = _canonical_split_identities(first, expectations)
    return audit_digest, train_inputs_digest, validation_inputs_digest


def _require_scorer_rungs(manifest: JsonObject, expected_scorer_digest: str) -> None:
    rungs = _object(manifest.get("rungs"), "qualification rungs")
    if set(rungs) != {"random", "item_popularity"}:
        raise QualificationEvidenceError("qualification rungs are incomplete")
    for name in ("random", "item_popularity"):
        rung = _object(rungs[name], f"{name} rung")
        if rung.get("reference_passed") is not True:
            raise QualificationEvidenceError(f"{name} reference rung did not pass")
        evaluations = _sequence(rung.get("evaluations"), f"{name} evaluations")
        if not evaluations:
            raise QualificationEvidenceError(f"{name} scorer evidence is missing")
        for index, raw in enumerate(evaluations):
            evaluation = _object(raw, f"{name} evaluations[{index}]")
            if evaluation.get("scorer_digest") != expected_scorer_digest:
                raise QualificationEvidenceError(f"{name} used another protected scorer")


def _require_artifact_records(
    run: JsonObject,
    *,
    source_prefix: str,
    artifact_index: Mapping[str, dict[str, object]],
    artifact_filenames: Sequence[str] = _ARTIFACT_FILENAMES,
) -> dict[str, str]:
    hashes = _object(run.get("artifact_file_sha256"), "FM artifact file SHA-256")
    if set(hashes) != set(artifact_filenames):
        raise QualificationEvidenceError("FM run retained an unexpected artifact set")
    normalized_hashes = {
        name: _require_digest(hashes.get(name), f"FM artifact {name} SHA-256")
        for name in artifact_filenames
    }
    expected_records: list[dict[str, object]] = []
    for name in artifact_filenames:
        relative = f"{source_prefix}/{name}"
        try:
            record = artifact_index[relative]
        except KeyError as exc:
            raise QualificationEvidenceError(
                f"FM artifact is absent from root index: {relative}"
            ) from exc
        if record["sha256"] != normalized_hashes[name]:
            raise QualificationEvidenceError(
                f"FM artifact hash differs from root index: {relative}"
            )
        expected_records.append(dict(record))
    if run.get("artifacts") != expected_records:
        raise QualificationEvidenceError(
            "FM run artifact records differ from the signed root index"
        )
    return normalized_hashes


def _require_training_trace(
    run: JsonObject,
    *,
    metrics: QualificationMetrics,
    checkpoint_manifest: Mapping[str, object],
    prediction_manifest: Mapping[str, object],
    train_inputs_digest: str,
    validation_inputs_digest: str,
    expected_scorer_digest: str,
    expected_starter_digest: str,
    evaluator_golden_digest: str,
) -> None:
    trace = _object(run.get("training_trace"), "FM training trace")
    required = {
        "checkpoint",
        "config_digest",
        "encoding_digest",
        "schema_version",
        "starter_manifest_digest",
        "train_inputs_digest",
        "training_targets_digest",
        "trusted_fixture_proof",
        "validation_inputs_digest",
        "validation_metrics",
        "validation_predictions",
    }
    if not required <= set(trace):
        raise QualificationEvidenceError("FM training trace is incomplete")
    if trace.get("schema_version") != 1:
        raise QualificationEvidenceError("FM training trace schema changed")
    for name in ("config_digest", "encoding_digest", "starter_manifest_digest"):
        if trace.get(name) != run.get(name):
            raise QualificationEvidenceError(f"FM training trace changed {name}")
    if trace.get("train_inputs_digest") != train_inputs_digest:
        raise QualificationEvidenceError("FM training used another training input identity")
    if trace.get("validation_inputs_digest") != validation_inputs_digest:
        raise QualificationEvidenceError("FM training used another validation input identity")
    _require_digest(trace.get("training_targets_digest"), "FM training target digest")
    _require_mapping_equal(trace.get("checkpoint"), checkpoint_manifest, "FM checkpoint trace")
    _require_mapping_equal(
        trace.get("validation_predictions"),
        prediction_manifest,
        "FM prediction trace",
    )
    _require_mapping_equal(
        trace.get("validation_metrics"),
        _metric_mapping(metrics),
        "FM metric trace",
    )
    proof = _object(trace.get("trusted_fixture_proof"), "FM trusted fixture proof")
    if proof.get("digest") != evaluator_golden_digest:
        raise QualificationEvidenceError("FM run used another evaluator golden proof")
    evaluator = _object(proof.get("evaluator_golden"), "FM evaluator golden proof")
    parity = _object(
        proof.get("starter_fm_untouched_fixture_parity"),
        "FM organizer parity proof",
    )
    if evaluator.get("passed") is not True or evaluator.get("scorer_digest") != (
        expected_scorer_digest
    ):
        raise QualificationEvidenceError("FM evaluator golden scorer identity mismatch")
    if (
        parity.get("passed") is not True
        or parity.get("organizer_evaluator_sha256") != expected_scorer_digest
        or parity.get("organizer_manifest_sha256") != expected_starter_digest
    ):
        raise QualificationEvidenceError("FM run did not prove untouched organizer parity")


def _load_seed_run(
    root: Path,
    run: JsonObject,
    *,
    physical_prefix: str,
    source_prefix: str,
    artifact_index: Mapping[str, dict[str, object]],
    expectations: QualificationExpectations,
    train_inputs_digest: str,
    validation_inputs_digest: str,
    evaluator_golden_digest: str,
    allow_clean_fields: bool = False,
) -> OfficialFMSeedEvidence:
    expected_fields = _RUN_FIELDS | (
        {"source_seed", "prediction_identity", "within_user_order_identity", "absolute_tolerance"}
        if allow_clean_fields
        else set()
    )
    if set(run) != expected_fields:
        raise QualificationEvidenceError("FM run uses an unknown or incomplete schema")
    seed = run.get("seed")
    if type(seed) is not int or seed not in _QUALIFICATION_SEEDS:
        raise QualificationEvidenceError("FM run seed must be one of 0 through 4")
    if run.get("organizer_parity_passed") is not True:
        raise QualificationEvidenceError(f"FM seed {seed} did not pass organizer parity")
    if run.get("starter_manifest_digest") != expectations.starter_manifest_digest:
        raise QualificationEvidenceError(f"FM seed {seed} starter identity mismatch")
    checkpoint_digest = _require_digest(
        run.get("checkpoint_digest"), f"FM seed {seed} checkpoint digest"
    )
    encoding_digest = _require_digest(run.get("encoding_digest"), f"FM seed {seed} encoding digest")
    config_digest = _require_digest(run.get("config_digest"), f"FM seed {seed} config digest")
    prediction_digest = _require_digest(
        run.get("validation_prediction_digest"),
        f"FM seed {seed} validation prediction digest",
    )
    metrics = _metrics(run.get("validation_metrics"), f"FM seed {seed} metrics")
    hashes = _require_artifact_records(
        run,
        source_prefix=source_prefix,
        artifact_index=artifact_index,
        artifact_filenames=(
            (*_ARTIFACT_FILENAMES, "clean-worker-evidence.json")
            if allow_clean_fields
            else _ARTIFACT_FILENAMES
        ),
    )
    physical_dir = root / physical_prefix
    encoding_path = physical_dir / "encoding.npz"
    checkpoint_path = physical_dir / "checkpoint.npz"
    predictions_path = physical_dir / "validation-predictions.npy"
    try:
        encoding = StarterEncoding.load(
            encoding_path,
            expected_file_sha256=hashes["encoding.npz"],
        )
        if encoding.digest != encoding_digest:
            raise QualificationEvidenceError(f"FM seed {seed} encoding logical digest mismatch")
        if encoding.training_inputs_digest != train_inputs_digest:
            raise QualificationEvidenceError(f"FM seed {seed} encoding came from another train set")
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_file_sha256=hashes["checkpoint.npz"],
            expected_checkpoint_digest=checkpoint_digest,
            expected_encoding_digest=encoding_digest,
            expected_starter_manifest_digest=expectations.starter_manifest_digest,
            expected_config_digest=config_digest,
            expected_seed=seed,
        )
        predictions = load_predictions(
            predictions_path,
            expected_file_sha256=hashes["validation-predictions.npy"],
            expected_prediction_digest=prediction_digest,
            expected_row_count=expectations.validation_row_count,
        )
    except (BaselineArtifactError, StarterEncodingError) as exc:
        raise QualificationEvidenceError(
            f"FM seed {seed} artifact verification failed: {exc}"
        ) from exc
    _require_training_trace(
        run,
        metrics=metrics,
        checkpoint_manifest=checkpoint.manifest(),
        prediction_manifest=predictions.manifest(),
        train_inputs_digest=train_inputs_digest,
        validation_inputs_digest=validation_inputs_digest,
        expected_scorer_digest=expectations.scorer_digest,
        expected_starter_digest=expectations.starter_manifest_digest,
        evaluator_golden_digest=evaluator_golden_digest,
    )
    resources = _object(run.get("resources"), f"FM seed {seed} resources")
    if set(resources) != {"wall_seconds", "cpu_seconds", "peak_rss_bytes", "device"}:
        raise QualificationEvidenceError(f"FM seed {seed} resources have invalid fields")
    resource_evidence = OfficialFMResourceEvidence(
        wall_seconds=cast(float, resources["wall_seconds"]),
        cpu_seconds=cast(float, resources["cpu_seconds"]),
        peak_rss_bytes=cast(int, resources["peak_rss_bytes"]),
        disk_bytes=sum(
            path.stat().st_size for path in (checkpoint_path, encoding_path, predictions_path)
        ),
        device=cast(str, resources["device"]),
    )
    return OfficialFMSeedEvidence(
        seed=seed,
        metrics=metrics,
        validation_prediction_digest=prediction_digest,
        checkpoint_digest=checkpoint_digest,
        config_digest=config_digest,
        encoding_digest=encoding_digest,
        checkpoint_path=checkpoint_path.resolve(),
        encoding_path=encoding_path.resolve(),
        validation_predictions_path=predictions_path.resolve(),
        checkpoint_file_sha256=hashes["checkpoint.npz"],
        encoding_file_sha256=hashes["encoding.npz"],
        validation_predictions_file_sha256=hashes["validation-predictions.npy"],
        resources=resource_evidence,
    )


def _require_clean_worker(
    root: Path,
    clean: JsonObject,
    clean_loaded: OfficialFMSeedEvidence,
    *,
    expectations: QualificationExpectations,
    audit_digest: str,
    train_inputs_digest: str,
    validation_inputs_digest: str,
) -> None:
    trace = _object(clean.get("training_trace"), "clean seed-0 training trace")
    subprocess_evidence = _object(
        trace.get("clean_subprocess"),
        "clean seed-0 subprocess evidence",
    )
    if subprocess_evidence != {
        "evidence_digest": subprocess_evidence.get("evidence_digest"),
        "evidence_file_sha256": subprocess_evidence.get("evidence_file_sha256"),
        "fresh_interpreter": True,
        "identity_verified_by_parent": True,
        "source_retrain": True,
    }:
        raise QualificationEvidenceError("clean seed-0 retrain was not a fresh source subprocess")
    evidence_digest = _require_digest(
        subprocess_evidence.get("evidence_digest"),
        "clean worker evidence digest",
    )
    evidence_file_sha256 = _require_digest(
        subprocess_evidence.get("evidence_file_sha256"),
        "clean worker evidence file SHA-256",
    )
    worker_path = root / "clean-retrain-seed-0" / "clean-worker-evidence.json"
    _, actual_file_sha256 = _hash_regular(
        worker_path,
        maximum=_MAX_JSON_BYTES,
        location="clean worker evidence",
    )
    if actual_file_sha256 != evidence_file_sha256:
        raise QualificationEvidenceError("clean worker evidence file identity mismatch")
    worker = _json_object(worker_path, "clean worker evidence")
    if _logical_digest(worker, "clean worker evidence") != evidence_digest:
        raise QualificationEvidenceError("clean worker logical identity mismatch")
    if set(worker) != {
        "schema_version",
        "kind",
        "seed",
        "starter_manifest_digest",
        "canonical_digest",
        "audit_digest",
        "train_inputs_digest",
        "validation_inputs_digest",
        "checkpoint_digest",
        "encoding_digest",
        "config_digest",
        "validation_prediction_digest",
        "validation_metrics",
        "final_target_capability",
        "final_outcomes_accessed",
        "artifact_sha256",
        "logical_run",
        "logical_run_digest",
        "resources",
        "digest",
    }:
        raise QualificationEvidenceError("clean worker evidence uses an unknown schema")
    expected_identity: dict[str, object] = {
        "schema_version": 1,
        "kind": "clean_seed_zero_source_retrain_worker",
        "seed": 0,
        "starter_manifest_digest": expectations.starter_manifest_digest,
        "canonical_digest": expectations.canonical_digest,
        "audit_digest": audit_digest,
        "train_inputs_digest": train_inputs_digest,
        "validation_inputs_digest": validation_inputs_digest,
        "checkpoint_digest": clean_loaded.checkpoint_digest,
        "encoding_digest": clean_loaded.encoding_digest,
        "config_digest": clean_loaded.config_digest,
        "validation_prediction_digest": clean_loaded.validation_prediction_digest,
        "validation_metrics": _metric_mapping(clean_loaded.metrics),
        "final_target_capability": None,
        "final_outcomes_accessed": False,
    }
    if any(worker.get(name) != value for name, value in expected_identity.items()):
        raise QualificationEvidenceError("clean worker evidence is bound to another identity")
    artifact_sha256 = _object(worker.get("artifact_sha256"), "clean worker artifact hashes")
    clean_hashes = _object(clean.get("artifact_file_sha256"), "clean retrain artifact hashes")
    if artifact_sha256 != {name: clean_hashes.get(name) for name in _ARTIFACT_FILENAMES}:
        raise QualificationEvidenceError("clean worker model artifact identities changed")
    logical_run = _object(worker.get("logical_run"), "clean worker logical run")
    logical_run_digest = _require_digest(
        worker.get("logical_run_digest"),
        "clean worker logical run digest",
    )
    if hashlib.sha256(_canonical_json(logical_run)).hexdigest() != logical_run_digest:
        raise QualificationEvidenceError("clean worker logical run digest mismatch")
    for name, expected in (
        ("checkpoint", _object(trace.get("checkpoint"), "clean checkpoint trace")),
        ("config_digest", clean_loaded.config_digest),
        ("encoding_digest", clean_loaded.encoding_digest),
        ("starter_manifest_digest", expectations.starter_manifest_digest),
        ("train_inputs_digest", train_inputs_digest),
        ("validation_inputs_digest", validation_inputs_digest),
        ("validation_metrics", _metric_mapping(clean_loaded.metrics)),
    ):
        if logical_run.get(name) != expected:
            raise QualificationEvidenceError(f"clean worker logical run changed {name}")


def _require_fm_qualification(
    root: Path,
    manifest: JsonObject,
    *,
    artifact_index: Mapping[str, dict[str, object]],
    expectations: QualificationExpectations,
    train_inputs_digest: str,
    validation_inputs_digest: str,
    evaluator_golden_digest: str,
    audit_digest: str,
) -> tuple[OfficialFMSeedEvidence, ...]:
    fm = _object(manifest.get("fm"), "qualification FM evidence")
    if set(fm) != {
        "seeds",
        "runs",
        "five_seed_mean",
        "published_reference",
        "reference_passed",
        "checkpoint_replays",
        "clean_seed_zero",
    }:
        raise QualificationEvidenceError("qualification FM evidence uses an unknown schema")
    if fm.get("seeds") != list(_QUALIFICATION_SEEDS) or fm.get("reference_passed") is not True:
        raise QualificationEvidenceError("qualification did not pass exact FM seeds 0 through 4")
    published = None
    for rung in BENCHMARK_CONTRACT.reference_rungs:
        if rung.name == "fm_official":
            published = rung.validation.manifest()
            break
    if published is None or fm.get("published_reference") != published:
        raise QualificationEvidenceError("qualification official-FM published reference changed")
    raw_runs = _sequence(fm.get("runs"), "qualification FM runs")
    if len(raw_runs) != len(_QUALIFICATION_SEEDS):
        raise QualificationEvidenceError("qualification must retain exactly five FM runs")
    loaded: list[OfficialFMSeedEvidence] = []
    for expected_seed, raw in zip(_QUALIFICATION_SEEDS, raw_runs, strict=True):
        run = _object(raw, f"FM seed {expected_seed} run")
        if run.get("seed") != expected_seed:
            raise QualificationEvidenceError("qualification FM runs are not in exact seed order")
        standalone = _json_object(
            root / "fm" / f"seed-{expected_seed}" / "run.json",
            f"FM seed {expected_seed} run manifest",
        )
        if standalone != run:
            raise QualificationEvidenceError(f"FM seed {expected_seed} run manifest differs")
        loaded.append(
            _load_seed_run(
                root,
                run,
                physical_prefix=f"fm/seed-{expected_seed}",
                source_prefix=f"fm/seed-{expected_seed}",
                artifact_index=artifact_index,
                expectations=expectations,
                train_inputs_digest=train_inputs_digest,
                validation_inputs_digest=validation_inputs_digest,
                evaluator_golden_digest=evaluator_golden_digest,
            )
        )

    mean_metrics = _metrics(fm.get("five_seed_mean"), "FM five-seed mean")
    for name in ("GAUC", "nDCG@5", "primary"):
        observed = _metric_mapping(mean_metrics)[name]
        expected = sum(_metric_mapping(run.metrics)[name] for run in loaded) / len(loaded)
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise QualificationEvidenceError(f"FM five-seed mean changed {name}")
        rounded = Decimal(str(observed)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        reference = Decimal(str(published[name])).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        if rounded != reference:
            raise QualificationEvidenceError(f"FM qualification did not reproduce published {name}")

    replay_entries = _sequence(fm.get("checkpoint_replays"), "FM checkpoint replays")
    if len(replay_entries) != len(_QUALIFICATION_SEEDS):
        raise QualificationEvidenceError("qualification must retain five checkpoint replays")
    for seed, (raw, qualified_run) in enumerate(zip(replay_entries, loaded, strict=True)):
        replay = _object(raw, f"FM seed {seed} replay")
        if (
            replay.get("seed") != seed
            or replay.get("checkpoint_digest") != qualified_run.checkpoint_digest
            or replay.get("prediction_digest") != qualified_run.validation_prediction_digest
            or replay.get("prediction_identity") is not True
            or replay.get("charged_launch") is not False
            or replay.get("metrics") != _metric_mapping(qualified_run.metrics)
        ):
            raise QualificationEvidenceError(f"FM seed {seed} checkpoint replay is not exact")
        standalone = _json_object(
            root / "replays" / f"seed-{seed}.json",
            f"FM seed {seed} replay manifest",
        )
        if standalone != replay:
            raise QualificationEvidenceError(f"FM seed {seed} replay manifest differs")

    clean = _object(fm.get("clean_seed_zero"), "clean seed-0 retrain")
    standalone_clean = _json_object(
        root / "clean-retrain-seed-0" / "run.json",
        "clean seed-0 run manifest",
    )
    if standalone_clean != clean:
        raise QualificationEvidenceError("clean seed-0 retrain manifest differs")
    if (
        clean.get("source_seed") != 0
        or clean.get("prediction_identity") is not True
        or clean.get("within_user_order_identity") is not True
        or clean.get("absolute_tolerance") != 0.0
    ):
        raise QualificationEvidenceError("clean seed-0 retrain did not prove exact identity")
    clean_loaded = _load_seed_run(
        root,
        clean,
        physical_prefix="clean-retrain-seed-0",
        source_prefix="clean-retrain-seed-0",
        artifact_index=artifact_index,
        expectations=expectations,
        train_inputs_digest=train_inputs_digest,
        validation_inputs_digest=validation_inputs_digest,
        evaluator_golden_digest=evaluator_golden_digest,
        allow_clean_fields=True,
    )
    _require_clean_worker(
        root,
        clean,
        clean_loaded,
        expectations=expectations,
        audit_digest=audit_digest,
        train_inputs_digest=train_inputs_digest,
        validation_inputs_digest=validation_inputs_digest,
    )
    seed_zero = loaded[0]
    if (
        clean_loaded.checkpoint_digest != seed_zero.checkpoint_digest
        or clean_loaded.encoding_digest != seed_zero.encoding_digest
        or clean_loaded.config_digest != seed_zero.config_digest
        or clean_loaded.validation_prediction_digest != seed_zero.validation_prediction_digest
        or clean_loaded.metrics != seed_zero.metrics
    ):
        raise QualificationEvidenceError("clean seed-0 retrain differs from the source run")

    launch = _object(manifest.get("launch_accounting"), "qualification launch accounting")
    expected_records = [
        *(
            {
                "launch_number": seed + 1,
                "kind": "official_fm_training",
                "seed": seed,
                "charged": True,
            }
            for seed in _QUALIFICATION_SEEDS
        ),
        {
            "launch_number": 6,
            "kind": "clean_source_retrain",
            "seed": 0,
            "charged": True,
        },
    ]
    if (
        launch.get("charged_launches") != 6
        or launch.get("expected_launches") != 6
        or launch.get("records") != expected_records
        or launch.get("random_rungs_charged") is not False
        or launch.get("popularity_rung_charged") is not False
        or launch.get("checkpoint_replays_charged") is not False
    ):
        raise QualificationEvidenceError("qualification launch accounting is not exactly six")
    return tuple(loaded)


def _submission_identity(
    root: Path,
    value: object,
    *,
    expected_path: str,
    location: str,
) -> JsonObject:
    submission = _object(value, location)
    if submission.get("path") != expected_path:
        raise QualificationEvidenceError(f"{location} path changed")
    expected_hash = _require_digest(submission.get("sha256"), f"{location} SHA-256")
    _, actual_hash = _hash_regular(
        root / expected_path,
        maximum=_MAX_ARTIFACT_BYTES,
        location=location,
    )
    if actual_hash != expected_hash:
        raise QualificationEvidenceError(f"{location} bytes changed")
    if submission.get("round_trip_identity") is not True:
        raise QualificationEvidenceError(f"{location} did not preserve exact round trip")
    return submission


def _require_fallback(
    root: Path,
    manifest: JsonObject,
    *,
    artifact_index: Mapping[str, dict[str, object]],
    expectations: QualificationExpectations,
    seed_four: OfficialFMSeedEvidence,
    seed_four_run: JsonObject,
) -> OfficialFMFallbackEvidence:
    fallback = _object(manifest.get("fallback"), "qualification fallback")
    if set(fallback) != _FALLBACK_FIELDS:
        raise QualificationEvidenceError("qualification fallback uses an unknown schema")
    fallback_manifest = _json_object(root / "fallback" / "manifest.json", "fallback manifest")
    if fallback_manifest != fallback:
        raise QualificationEvidenceError("standalone fallback manifest differs from root manifest")
    fallback_digest = _logical_digest(fallback, "qualification fallback")
    if (
        fallback.get("schema_version") != 1
        or fallback.get("kind") != "immutable_official_fm_fallback"
        or fallback.get("seed") != _FALLBACK_SEED
        or fallback.get("replay_verified") is not True
        or fallback.get("clean_seed_zero_retrain_verified") is not True
    ):
        raise QualificationEvidenceError("qualification fallback is not the verified seed-4 FM")
    if (
        fallback.get("validation_metrics") != _metric_mapping(seed_four.metrics)
        or fallback.get("checkpoint_digest") != seed_four.checkpoint_digest
        or fallback.get("encoding_digest") != seed_four.encoding_digest
        or fallback.get("config_digest") != seed_four.config_digest
        or fallback.get("validation_prediction_digest") != seed_four.validation_prediction_digest
    ):
        raise QualificationEvidenceError("qualification fallback differs from FM seed 4")

    validation_submission = _submission_identity(
        root,
        fallback.get("validation_submission"),
        expected_path="validation/submission.csv",
        location="fallback validation submission",
    )
    if (
        validation_submission.get("prediction_digest") != seed_four.validation_prediction_digest
        or validation_submission.get("protected_metrics_preserved") is not True
    ):
        raise QualificationEvidenceError("fallback validation submission changed protected scores")
    final_submission = _submission_identity(
        root,
        fallback.get("final_submission"),
        expected_path="final/submission.csv",
        location="fallback final submission",
    )
    final_submission_file_sha256 = _require_digest(
        final_submission.get("sha256"),
        "fallback final submission SHA-256",
    )
    final_prediction_digest = _require_digest(
        final_submission.get("prediction_digest"),
        "fallback final prediction digest",
    )
    if final_submission.get("final_outcomes_accessed") is not False:
        raise QualificationEvidenceError("fallback final inference accessed outcomes")

    source_dir = root / "fm" / "seed-4"
    model_dir = root / "fallback" / "model"
    source_records, source_digest = _tree_digest(source_dir)
    fallback_records, fallback_tree_digest = _tree_digest(model_dir)
    if source_records != fallback_records:
        raise QualificationEvidenceError("fallback model tree differs from FM seed 4")
    if (
        fallback.get("source_model_tree_digest") != source_digest
        or fallback.get("fallback_model_tree_digest") != fallback_tree_digest
    ):
        raise QualificationEvidenceError("fallback model tree digest mismatch")
    model_run = _json_object(model_dir / "run.json", "fallback copied run manifest")
    if model_run != seed_four_run:
        raise QualificationEvidenceError("fallback copied run differs from FM seed 4")

    hashes = _require_artifact_records(
        seed_four_run,
        source_prefix="fm/seed-4",
        artifact_index=artifact_index,
    )
    try:
        encoding = StarterEncoding.load(
            model_dir / "encoding.npz",
            expected_file_sha256=hashes["encoding.npz"],
        )
        if encoding.digest != seed_four.encoding_digest:
            raise QualificationEvidenceError("fallback copied encoding logical digest mismatch")
        load_checkpoint(
            model_dir / "checkpoint.npz",
            expected_file_sha256=hashes["checkpoint.npz"],
            expected_checkpoint_digest=seed_four.checkpoint_digest,
            expected_encoding_digest=seed_four.encoding_digest,
            expected_starter_manifest_digest=expectations.starter_manifest_digest,
            expected_config_digest=seed_four.config_digest,
            expected_seed=4,
        )
        load_predictions(
            model_dir / "validation-predictions.npy",
            expected_file_sha256=hashes["validation-predictions.npy"],
            expected_prediction_digest=seed_four.validation_prediction_digest,
            expected_row_count=expectations.validation_row_count,
        )
    except (BaselineArtifactError, StarterEncodingError) as exc:
        raise QualificationEvidenceError(f"fallback artifact verification failed: {exc}") from exc
    return OfficialFMFallbackEvidence(
        seed=4,
        metrics=seed_four.metrics,
        manifest_digest=fallback_digest,
        source_model_tree_digest=source_digest,
        fallback_model_tree_digest=fallback_tree_digest,
        validation_prediction_digest=seed_four.validation_prediction_digest,
        checkpoint_digest=seed_four.checkpoint_digest,
        config_digest=seed_four.config_digest,
        encoding_digest=seed_four.encoding_digest,
        model_dir=model_dir.resolve(),
        checkpoint_path=(model_dir / "checkpoint.npz").resolve(),
        encoding_path=(model_dir / "encoding.npz").resolve(),
        validation_predictions_path=(model_dir / "validation-predictions.npy").resolve(),
        checkpoint_file_sha256=hashes["checkpoint.npz"],
        encoding_file_sha256=hashes["encoding.npz"],
        validation_predictions_file_sha256=hashes["validation-predictions.npy"],
        final_submission_path=(root / "final" / "submission.csv").resolve(),
        final_submission_file_sha256=final_submission_file_sha256,
        final_prediction_digest=final_prediction_digest,
        final_row_count=expectations.final_row_count,
        resources=seed_four.resources,
    )


def load_official_fm_qualification(
    run_dir: str | Path,
    *,
    expectations: QualificationExpectations,
) -> OfficialFMQualificationEvidence:
    """Verify and load one immutable official-FM qualification for production use.

    The caller supplies identities rebuilt from the current locked data/starter/scorer context.
    The loader fails closed on any logical, physical, semantic, or artifact-decoding mismatch.
    It intentionally exposes no validation labels and never opens final-period outcomes.
    """

    if not isinstance(expectations, QualificationExpectations):
        raise QualificationEvidenceError("expectations must be QualificationExpectations")
    try:
        identity = load_qualification_identity(run_dir)
    except ProvenanceError as exc:
        raise QualificationEvidenceError(
            f"qualification identity verification failed: {exc}"
        ) from exc
    root = identity.root
    manifest = _json_object(root / "manifest.json", "qualification manifest")
    if set(manifest) != _ROOT_FIELDS:
        raise QualificationEvidenceError("qualification root manifest uses an unknown schema")
    manifest_digest = _logical_digest(manifest, "qualification manifest")
    if manifest_digest != identity.manifest_digest:
        raise QualificationEvidenceError(
            "qualification identity changed between verification passes"
        )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "baseline_reproduced"
        or manifest.get("benchmark_digest") != BENCHMARK_CONTRACT.digest
        or identity.benchmark_digest != BENCHMARK_CONTRACT.digest
    ):
        raise QualificationEvidenceError("qualification is not the operative official benchmark")
    if identity.starter_manifest_digest != expectations.starter_manifest_digest:
        raise QualificationEvidenceError("qualification starter-kit identity mismatch")
    artifact_index = _artifact_index(root, manifest)
    audit_digest, train_inputs_digest, validation_inputs_digest = _require_snapshot(
        root, manifest, expectations
    )
    _require_scorer_rungs(manifest, expectations.scorer_digest)
    evaluator_golden_digest = _require_digest(
        _json_object(
            root / "verification" / "snapshot-first.json",
            "first snapshot",
        ).get("evaluator_golden_digest"),
        "evaluator golden digest",
    )
    runs = _require_fm_qualification(
        root,
        manifest,
        artifact_index=artifact_index,
        expectations=expectations,
        train_inputs_digest=train_inputs_digest,
        validation_inputs_digest=validation_inputs_digest,
        evaluator_golden_digest=evaluator_golden_digest,
        audit_digest=audit_digest,
    )
    final_period = _object(manifest.get("final_period"), "qualification final period")
    if final_period != {
        "input_rows": expectations.final_row_count,
        "target_capability": None,
        "outcomes_accessed": False,
        "outcomes_scored": False,
    }:
        raise QualificationEvidenceError("qualification final-period no-outcome gate changed")
    seed_four_run = _object(
        _sequence(_object(manifest["fm"], "qualification FM")["runs"], "FM runs")[4],
        "FM seed 4 run",
    )
    fallback = _require_fallback(
        root,
        manifest,
        artifact_index=artifact_index,
        expectations=expectations,
        seed_four=runs[4],
        seed_four_run=seed_four_run,
    )
    return OfficialFMQualificationEvidence(
        root=root,
        manifest_digest=manifest_digest,
        qualification_input_digest=_require_digest(
            manifest.get("qualification_input_digest"), "qualification input digest"
        ),
        benchmark_digest=BENCHMARK_CONTRACT.digest,
        canonical_digest=expectations.canonical_digest,
        audit_digest=audit_digest,
        starter_manifest_digest=expectations.starter_manifest_digest,
        scorer_digest=expectations.scorer_digest,
        validation_row_count=expectations.validation_row_count,
        final_row_count=expectations.final_row_count,
        outer_runs=tuple(runs[seed] for seed in OUTER_BASELINE_SEEDS),
        fallback=fallback,
    )


__all__ = [
    "OUTER_BASELINE_SEEDS",
    "QUALIFICATION_EVIDENCE_SCHEMA_VERSION",
    "OfficialFMFallbackEvidence",
    "OfficialFMQualificationEvidence",
    "OfficialFMResourceEvidence",
    "OfficialFMSeedEvidence",
    "QualificationEvidenceError",
    "QualificationExpectations",
    "load_official_fm_qualification",
]
