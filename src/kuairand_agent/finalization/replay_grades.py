"""Evidence-derived replay grades for finalization acceptance.

Replay grades are conclusions, not request parameters.  This module deliberately has no public
constructor that accepts a grade chosen by a caller.  Factories consume validated replay or bundle
evidence, prove the grade-specific invariants, and return a content-addressed receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.finalization.replay import CleanReplayEvidence, ReplayEquality

REPLAY_GRADE_SCHEMA_VERSION: Final = 1
_RECEIPT_DOMAIN: Final = b"kuairand-replay-grade-receipt-v1\0"
_REPORT_DOMAIN: Final = b"kuairand-replay-grade-report-v1\0"
_DIGEST_LENGTH: Final = 64


class ReplayGradeError(ValueError):
    """Raised when supplied evidence cannot prove the requested replay conclusion."""


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
        raise ReplayGradeError("replay grade evidence must be finite canonical JSON") from exc


def _digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplayGradeError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, location: str) -> str:
    """Accept a SHA-256 string or a nominal domain ID exposing ``.value``."""

    candidate = value if type(value) is str else getattr(value, "value", None)
    return _digest(candidate, location)


def _frozen_mapping(value: Mapping[str, object], location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ReplayGradeError(f"{location} must be a non-empty mapping")
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical encoder is authoritative.
        raise ReplayGradeError(f"{location} could not be normalized") from exc
    if not isinstance(normalized, dict):  # pragma: no cover - input Mapping guarantees this.
        raise ReplayGradeError(f"{location} must normalize to an object")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True, init=False)
class ReplayGradeReceipt:
    """A content-addressed conclusion created only after grade-specific verification."""

    grade: ReplayGrade
    contract_id: str
    prediction_id: str
    evidence_sha256: str
    evidence: Mapping[str, object]
    receipt_id: str

    @classmethod
    def _from_verified_evidence(
        cls,
        *,
        grade: ReplayGrade,
        contract_id: object,
        prediction_id: object,
        evidence: Mapping[str, object],
    ) -> ReplayGradeReceipt:
        if not isinstance(grade, ReplayGrade):
            raise ReplayGradeError("grade must be ReplayGrade")
        contract = _identifier(contract_id, "contract_id")
        prediction = _identifier(prediction_id, "prediction_id")
        frozen = _frozen_mapping(evidence, "evidence")
        evidence_sha256 = hashlib.sha256(_canonical_json(dict(frozen))).hexdigest()
        body: dict[str, object] = {
            "schema_version": REPLAY_GRADE_SCHEMA_VERSION,
            "grade": grade.value,
            "contract_id": contract,
            "prediction_id": prediction,
            "evidence_sha256": evidence_sha256,
            "evidence": dict(frozen),
        }
        receipt_id = hashlib.sha256(_RECEIPT_DOMAIN + _canonical_json(body)).hexdigest()
        result = object.__new__(cls)
        object.__setattr__(result, "grade", grade)
        object.__setattr__(result, "contract_id", contract)
        object.__setattr__(result, "prediction_id", prediction)
        object.__setattr__(result, "evidence_sha256", evidence_sha256)
        object.__setattr__(result, "evidence", frozen)
        object.__setattr__(result, "receipt_id", receipt_id)
        return result

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": REPLAY_GRADE_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "grade": self.grade.value,
            "contract_id": self.contract_id,
            "prediction_id": self.prediction_id,
            "evidence_sha256": self.evidence_sha256,
            "evidence": json.loads(_canonical_json(dict(self.evidence))),
        }


def _clean_replay_proof(evidence: CleanReplayEvidence) -> dict[str, object]:
    replay_manifest = evidence.manifest()
    return {
        "kind": "clean_replay",
        "clean_replay_sha256": hashlib.sha256(_canonical_json(replay_manifest)).hexdigest(),
        "candidate_id": evidence.candidate_id,
        "frozen_identity": evidence.identity.manifest(),
        "equality": evidence.equality.value,
        "absolute_tolerance": evidence.absolute_tolerance,
        "training_replay": evidence.training_replay,
        "validation": evidence.validation.manifest(),
        "final": evidence.final.manifest(),
        "capabilities": {
            "validation_digest": evidence.validation_capability_digest,
            "final_digest": evidence.final_capability_digest,
        },
        "clean_workspace_removed": evidence.clean_workspace_removed,
    }


def derive_clean_replay_grade_receipts(
    *,
    contract_id: object,
    prediction_id: object,
    evidence: CleanReplayEvidence,
) -> tuple[ReplayGradeReceipt, ...]:
    """Derive only the grades actually demonstrated by one validated clean replay.

    Exact evidence proves both scoring exactness and same-backend experiment replay.  Tolerant
    evidence proves only the explicitly tolerant experiment grade.  Portability and bundle
    exactness require independent evidence and are therefore never inferred here.
    """

    if not isinstance(evidence, CleanReplayEvidence):
        raise ReplayGradeError("evidence must be CleanReplayEvidence")
    proof = _clean_replay_proof(evidence)
    validation = evidence.validation
    if evidence.equality is ReplayEquality.EXACT:
        if not (
            validation.exact_prediction_bytes
            and validation.reference_prediction_digest == validation.replay_prediction_digest
            and validation.maximum_absolute_difference == 0.0
            and validation.protected_metrics_identical
            and validation.csv_round_trip_identity
            and validation.csv_protected_metrics_preserved
        ):
            raise ReplayGradeError("clean replay does not prove exact scoring identity")
        scoring = ReplayGradeReceipt._from_verified_evidence(
            grade=ReplayGrade.SCORING_EXACT,
            contract_id=contract_id,
            prediction_id=prediction_id,
            evidence=proof
            | {"conclusion": "stored and replayed prediction bytes have identical exact metrics"},
        )
        same_backend = ReplayGradeReceipt._from_verified_evidence(
            grade=ReplayGrade.EXPERIMENT_SAME_BACKEND,
            contract_id=contract_id,
            prediction_id=prediction_id,
            evidence=proof
            | {
                "conclusion": (
                    "frozen trial artifacts in the frozen environment reproduced exact predictions"
                )
            },
        )
        return (scoring, same_backend)

    if evidence.equality is ReplayEquality.TOLERANT_TOP5:
        if not (
            evidence.absolute_tolerance > 0.0
            and validation.maximum_absolute_difference <= evidence.absolute_tolerance
            and validation.top5_order_identical
            and validation.protected_metrics_identical
            and validation.csv_top5_preserved
            and validation.csv_protected_metrics_preserved
        ):
            raise ReplayGradeError("clean replay does not satisfy its frozen tolerance")
        return (
            ReplayGradeReceipt._from_verified_evidence(
                grade=ReplayGrade.EXPERIMENT_TOLERANT,
                contract_id=contract_id,
                prediction_id=prediction_id,
                evidence=proof
                | {
                    "conclusion": (
                        "approved numeric drift preserved frozen rank and metric tolerances"
                    )
                },
            ),
        )
    raise ReplayGradeError("clean replay uses an unsupported equality policy")


def derive_cross_backend_portability_grade(
    *,
    contract_id: object,
    prediction_id: object,
    cpu_evidence: CleanReplayEvidence,
    gpu_evidence: CleanReplayEvidence,
) -> ReplayGradeReceipt:
    """Derive strong CPU/GPU portability from two exact clean-replay receipts.

    The proof intentionally requires exact equality, which is stricter than the plan's permitted
    threshold-based portability receipt.  A future threshold adapter can add a separately typed
    evaluator without weakening this safe default.
    """

    for name, evidence in (("cpu_evidence", cpu_evidence), ("gpu_evidence", gpu_evidence)):
        if not isinstance(evidence, CleanReplayEvidence):
            raise ReplayGradeError(f"{name} must be CleanReplayEvidence")
        if evidence.equality is not ReplayEquality.EXACT:
            raise ReplayGradeError("cross-backend portability requires exact clean replays")
    if cpu_evidence.candidate_id != gpu_evidence.candidate_id:
        raise ReplayGradeError("CPU and GPU replay evidence refer to different candidates")
    if cpu_evidence.identity.environment_sha256 == gpu_evidence.identity.environment_sha256:
        raise ReplayGradeError("cross-backend evidence must use distinct environment identities")
    if (
        cpu_evidence.validation.reference_prediction_digest
        != gpu_evidence.validation.reference_prediction_digest
        or cpu_evidence.validation.replay_prediction_digest
        != gpu_evidence.validation.replay_prediction_digest
        or dict(cpu_evidence.validation.metrics) != dict(gpu_evidence.validation.metrics)
        or cpu_evidence.final.prediction_digest != gpu_evidence.final.prediction_digest
        or cpu_evidence.final.submission_sha256 != gpu_evidence.final.submission_sha256
    ):
        raise ReplayGradeError("CPU and GPU replay outputs are not exactly portable")
    proof = {
        "kind": "cross_backend_clean_replay",
        "candidate_id": cpu_evidence.candidate_id,
        "cpu": _clean_replay_proof(cpu_evidence),
        "gpu": _clean_replay_proof(gpu_evidence),
        "conclusion": "CPU and GPU clean replays produced identical validation and final outputs",
    }
    return ReplayGradeReceipt._from_verified_evidence(
        grade=ReplayGrade.CROSS_BACKEND_PORTABILITY,
        contract_id=contract_id,
        prediction_id=prediction_id,
        evidence=proof,
    )


@dataclass(frozen=True, slots=True, init=False)
class BundleRegenerationEvidence:
    """Exact two-pass projection evidence created internally by ``BundleFinalizer``."""

    contract_id: str
    prediction_id: str
    first_bundle_id: str
    regenerated_bundle_id: str
    first_submission_sha256: str
    regenerated_submission_sha256: str
    first_inventory_sha256: str
    regenerated_inventory_sha256: str

    @classmethod
    def _from_verified_projection(
        cls,
        *,
        contract_id: object,
        prediction_id: object,
        first_bundle_id: str,
        regenerated_bundle_id: str,
        first_submission_sha256: str,
        regenerated_submission_sha256: str,
        first_inventory_sha256: str,
        regenerated_inventory_sha256: str,
    ) -> BundleRegenerationEvidence:
        values = {
            "first_bundle_id": _digest(first_bundle_id, "first_bundle_id"),
            "regenerated_bundle_id": _digest(regenerated_bundle_id, "regenerated_bundle_id"),
            "first_submission_sha256": _digest(first_submission_sha256, "first_submission_sha256"),
            "regenerated_submission_sha256": _digest(
                regenerated_submission_sha256, "regenerated_submission_sha256"
            ),
            "first_inventory_sha256": _digest(first_inventory_sha256, "first_inventory_sha256"),
            "regenerated_inventory_sha256": _digest(
                regenerated_inventory_sha256, "regenerated_inventory_sha256"
            ),
        }
        if values["first_bundle_id"] != values["regenerated_bundle_id"]:
            raise ReplayGradeError("regenerated bundle identity differs")
        if values["first_submission_sha256"] != values["regenerated_submission_sha256"]:
            raise ReplayGradeError("regenerated submission bytes differ")
        if values["first_inventory_sha256"] != values["regenerated_inventory_sha256"]:
            raise ReplayGradeError("regenerated bundle inventory differs")
        result = object.__new__(cls)
        object.__setattr__(result, "contract_id", _identifier(contract_id, "contract_id"))
        object.__setattr__(result, "prediction_id", _identifier(prediction_id, "prediction_id"))
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "bundle_two_pass_regeneration",
            "contract_id": self.contract_id,
            "prediction_id": self.prediction_id,
            "first_bundle_id": self.first_bundle_id,
            "regenerated_bundle_id": self.regenerated_bundle_id,
            "first_submission_sha256": self.first_submission_sha256,
            "regenerated_submission_sha256": self.regenerated_submission_sha256,
            "first_inventory_sha256": self.first_inventory_sha256,
            "regenerated_inventory_sha256": self.regenerated_inventory_sha256,
            "conclusion": "clean finalization regenerated identical submission and bundle bytes",
        }


def derive_bundle_exact_grade(evidence: BundleRegenerationEvidence) -> ReplayGradeReceipt:
    """Derive ``BUNDLE_EXACT`` from finalizer-owned two-pass evidence."""

    if not isinstance(evidence, BundleRegenerationEvidence):
        raise ReplayGradeError("evidence must be BundleRegenerationEvidence")
    return ReplayGradeReceipt._from_verified_evidence(
        grade=ReplayGrade.BUNDLE_EXACT,
        contract_id=evidence.contract_id,
        prediction_id=evidence.prediction_id,
        evidence=evidence.manifest(),
    )


def validate_bundle_regeneration_evidence_manifest(
    value: object,
) -> BundleRegenerationEvidence:
    """Reconstruct and authenticate a serialized two-pass bundle proof.

    The public validator deliberately calls the same verified constructor used by the finalizer,
    then requires byte-equivalent canonical content.  Callers therefore cannot smuggle additional
    claims into an otherwise valid proof or weaken any of the exact-equality checks.
    """

    if not isinstance(value, Mapping):
        raise ReplayGradeError("bundle regeneration evidence must be a mapping")
    expected_fields = {
        "kind",
        "contract_id",
        "prediction_id",
        "first_bundle_id",
        "regenerated_bundle_id",
        "first_submission_sha256",
        "regenerated_submission_sha256",
        "first_inventory_sha256",
        "regenerated_inventory_sha256",
        "conclusion",
    }
    if set(value) != expected_fields:
        raise ReplayGradeError("bundle regeneration evidence has an unexpected schema")
    reconstructed = BundleRegenerationEvidence._from_verified_projection(
        contract_id=value.get("contract_id"),
        prediction_id=value.get("prediction_id"),
        first_bundle_id=_digest(value.get("first_bundle_id"), "first_bundle_id"),
        regenerated_bundle_id=_digest(value.get("regenerated_bundle_id"), "regenerated_bundle_id"),
        first_submission_sha256=_digest(
            value.get("first_submission_sha256"), "first_submission_sha256"
        ),
        regenerated_submission_sha256=_digest(
            value.get("regenerated_submission_sha256"),
            "regenerated_submission_sha256",
        ),
        first_inventory_sha256=_digest(
            value.get("first_inventory_sha256"), "first_inventory_sha256"
        ),
        regenerated_inventory_sha256=_digest(
            value.get("regenerated_inventory_sha256"),
            "regenerated_inventory_sha256",
        ),
    )
    if _canonical_json(dict(value)) != _canonical_json(reconstructed.manifest()):
        raise ReplayGradeError("bundle regeneration evidence contains forged fields")
    return reconstructed


def validate_replay_grade_receipt_manifest(
    value: object,
    *,
    expected_grade: ReplayGrade | None = None,
    expected_contract_id: object | None = None,
    expected_prediction_id: object | None = None,
    expected_evidence: Mapping[str, object] | None = None,
) -> ReplayGradeReceipt:
    """Authenticate a serialized evidence-derived replay-grade receipt.

    Receipt IDs use this module's historical ASCII JSON contract, which is intentionally not
    interchangeable with the domain identity canonicalizer.  Keeping validation here prevents
    persistence and UI layers from accidentally reimplementing the receipt formula differently.
    """

    if not isinstance(value, Mapping):
        raise ReplayGradeError("replay grade receipt must be a mapping")
    if set(value) != {
        "schema_version",
        "receipt_id",
        "grade",
        "contract_id",
        "prediction_id",
        "evidence_sha256",
        "evidence",
    }:
        raise ReplayGradeError("replay grade receipt has an unexpected schema")
    if value.get("schema_version") != REPLAY_GRADE_SCHEMA_VERSION:
        raise ReplayGradeError("replay grade receipt schema_version is unsupported")
    grade_value = value.get("grade")
    if type(grade_value) is not str:
        raise ReplayGradeError("replay grade receipt grade must be text")
    try:
        grade = ReplayGrade(grade_value)
    except (TypeError, ValueError) as exc:
        raise ReplayGradeError("replay grade receipt names an unsupported grade") from exc
    if expected_grade is not None and grade is not expected_grade:
        raise ReplayGradeError("replay grade receipt proves a different grade")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ReplayGradeError("replay grade receipt evidence must be a mapping")
    if expected_evidence is not None and _canonical_json(dict(evidence)) != _canonical_json(
        dict(expected_evidence)
    ):
        raise ReplayGradeError("replay grade receipt differs from expected evidence")
    contract_id = value.get("contract_id")
    prediction_id = value.get("prediction_id")
    if expected_contract_id is not None and _identifier(contract_id, "contract_id") != _identifier(
        expected_contract_id, "expected_contract_id"
    ):
        raise ReplayGradeError("replay grade receipt differs from expected contract")
    if expected_prediction_id is not None and _identifier(
        prediction_id, "prediction_id"
    ) != _identifier(expected_prediction_id, "expected_prediction_id"):
        raise ReplayGradeError("replay grade receipt differs from expected prediction")
    reconstructed = ReplayGradeReceipt._from_verified_evidence(
        grade=grade,
        contract_id=contract_id,
        prediction_id=prediction_id,
        evidence=evidence,
    )
    if _canonical_json(dict(value)) != _canonical_json(reconstructed.manifest()):
        raise ReplayGradeError("replay grade receipt identity or evidence digest is invalid")
    return reconstructed


@dataclass(frozen=True, slots=True)
class ReplayGradeReport:
    """Canonical collection of non-conflicting evidence-derived receipts."""

    receipts: tuple[ReplayGradeReceipt, ...]
    report_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.receipts or any(
            not isinstance(receipt, ReplayGradeReceipt) for receipt in self.receipts
        ):
            raise ReplayGradeError("report must contain ReplayGradeReceipt values")
        contracts = {receipt.contract_id for receipt in self.receipts}
        predictions = {receipt.prediction_id for receipt in self.receipts}
        grades = {receipt.grade for receipt in self.receipts}
        if len(contracts) != 1 or len(predictions) != 1:
            raise ReplayGradeError("report receipts must share one contract and prediction")
        if len(grades) != len(self.receipts):
            raise ReplayGradeError("report cannot contain duplicate grades")
        ordered = tuple(sorted(self.receipts, key=lambda receipt: receipt.grade.value))
        object.__setattr__(self, "receipts", ordered)
        object.__setattr__(
            self,
            "report_id",
            hashlib.sha256(_REPORT_DOMAIN + _canonical_json(self.body_manifest())).hexdigest(),
        )

    @property
    def achieved_grades(self) -> frozenset[ReplayGrade]:
        return frozenset(receipt.grade for receipt in self.receipts)

    @property
    def contract_id(self) -> str:
        return self.receipts[0].contract_id

    @property
    def prediction_id(self) -> str:
        return self.receipts[0].prediction_id

    def body_manifest(self) -> dict[str, object]:
        return {
            "schema_version": REPLAY_GRADE_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "prediction_id": self.prediction_id,
            "achieved_grades": sorted(grade.value for grade in self.achieved_grades),
            "receipts": [receipt.manifest() for receipt in self.receipts],
        }

    def manifest(self) -> dict[str, object]:
        return self.body_manifest() | {"report_id": self.report_id}

    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.manifest()) + b"\n"


def combine_replay_grade_receipts(
    receipts: Sequence[ReplayGradeReceipt],
) -> ReplayGradeReport:
    """Build a deterministic report from already derived receipts."""

    return ReplayGradeReport(tuple(receipts))


__all__ = [
    "REPLAY_GRADE_SCHEMA_VERSION",
    "BundleRegenerationEvidence",
    "ReplayGrade",
    "ReplayGradeError",
    "ReplayGradeReceipt",
    "ReplayGradeReport",
    "combine_replay_grade_receipts",
    "derive_bundle_exact_grade",
    "derive_clean_replay_grade_receipts",
    "derive_cross_backend_portability_grade",
    "validate_bundle_regeneration_evidence_manifest",
    "validate_replay_grade_receipt_manifest",
]
