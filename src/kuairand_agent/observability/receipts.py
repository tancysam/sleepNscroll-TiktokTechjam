"""Small, content-addressed receipts for the laboratory facade.

The receipts in this module describe what was actually verified.  In particular, the offline
scripted vertical slice is never described as an official-FM or full-data qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from kuairand_agent.contract import ContractVerificationReceipt
from kuairand_agent.domain.decisions import ReplayGrade
from kuairand_agent.domain.identity import canonical_json_bytes

RECEIPT_SCHEMA_VERSION: Final = 1
_STARTUP_DOMAIN: Final = b"kuairand-lab-startup-receipt-v1\0"
_SCRIPTED_REPLAY_DOMAIN: Final = b"kuairand-scripted-replay-receipt-v1\0"


class ReceiptError(ValueError):
    """Raised when observability evidence overstates or contradicts its verification."""


def _sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReceiptError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, location: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value:
        raise ReceiptError(f"{location} must be non-empty single-line text")
    return value


def _receipt_id(domain: bytes, manifest: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(manifest)).hexdigest()


@dataclass(frozen=True, slots=True)
class StartupReceipt:
    """Proof that the frozen organizer/challenge contract was checked before state opened."""

    contract_id: str
    expected_contract_id: str
    profile: str
    verified: bool
    repository_inputs: Mapping[str, object] | None = None
    state_writes_started: bool = False
    schema_version: int = RECEIPT_SCHEMA_VERSION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ReceiptError("startup receipt schema_version must be 1")
        _sha256(self.contract_id, "contract_id")
        _sha256(self.expected_contract_id, "expected_contract_id")
        _text(self.profile, "profile")
        if self.contract_id != self.expected_contract_id or not self.verified:
            raise ReceiptError("startup receipt requires an exact verified contract")
        if self.state_writes_started:
            raise ReceiptError("startup contract verification must precede every state write")
        if self.repository_inputs is not None:
            if not isinstance(self.repository_inputs, Mapping):
                raise ReceiptError("repository_inputs must be a mapping")
            try:
                normalized = json.loads(canonical_json_bytes(dict(self.repository_inputs)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ReceiptError("repository_inputs must be canonical JSON") from exc
            if not isinstance(normalized, dict):
                raise ReceiptError("repository_inputs must be an object")
            if normalized.get("contract_id") != self.contract_id:
                raise ReceiptError("repository_inputs differs from startup ContractId")
            object.__setattr__(self, "repository_inputs", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "receipt_id",
            _receipt_id(_STARTUP_DOMAIN, self._body()),
        )

    @classmethod
    def from_verification(
        cls,
        verification: ContractVerificationReceipt,
        *,
        profile: str,
    ) -> StartupReceipt:
        if not isinstance(verification, ContractVerificationReceipt):
            raise ReceiptError("verification must be a ContractVerificationReceipt")
        repository_inputs = getattr(verification, "repository_inputs", None)
        repository_manifest = (
            repository_inputs.manifest() if repository_inputs is not None else None
        )
        return cls(
            contract_id=verification.contract_id.value,
            expected_contract_id=verification.expected_contract_id.value,
            profile=profile,
            verified=verification.verified,
            repository_inputs=repository_manifest,
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "expected_contract_id": self.expected_contract_id,
            "profile": self.profile,
            "verified": self.verified,
            "state_writes_started": self.state_writes_started,
            "repository_inputs": (
                dict(self.repository_inputs) if self.repository_inputs is not None else None
            ),
        }

    def manifest(self) -> dict[str, object]:
        return {**self._body(), "receipt_id": self.receipt_id}


@dataclass(frozen=True, slots=True)
class ScriptedReplayReceipt:
    """Same-backend replay evidence for the deterministic fixture trainer.

    This receipt intentionally says ``SCRIPTED_FIXTURE_ONLY``. Equal scripted prediction and
    result bytes prove only that the same fixture trial reproduced on the scripted backend. No
    metric implementation or labels participated, so this can never claim ``SCORING_EXACT``.
    """

    contract_id: str
    campaign_id: str
    prediction_id: str
    first_prediction_sha256: str
    replay_prediction_sha256: str
    first_result_sha256: str
    replay_result_sha256: str
    grade: ReplayGrade = ReplayGrade.EXPERIMENT_SAME_BACKEND
    qualification_scope: str = "SCRIPTED_FIXTURE_ONLY"
    schema_version: int = RECEIPT_SCHEMA_VERSION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ReceiptError("scripted replay receipt schema_version must be 1")
        for name in (
            "contract_id",
            "campaign_id",
            "prediction_id",
            "first_prediction_sha256",
            "replay_prediction_sha256",
            "first_result_sha256",
            "replay_result_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.grade is not ReplayGrade.EXPERIMENT_SAME_BACKEND:
            raise ReceiptError("unscored scripted replay can conclude only EXPERIMENT_SAME_BACKEND")
        if self.qualification_scope != "SCRIPTED_FIXTURE_ONLY":
            raise ReceiptError("scripted replay must not claim production qualification")
        if self.first_prediction_sha256 != self.replay_prediction_sha256:
            raise ReceiptError("scripted replay prediction bytes differ")
        if self.first_result_sha256 != self.replay_result_sha256:
            raise ReceiptError("scripted replay result receipts differ")
        object.__setattr__(
            self,
            "receipt_id",
            _receipt_id(_SCRIPTED_REPLAY_DOMAIN, self._body()),
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "campaign_id": self.campaign_id,
            "prediction_id": self.prediction_id,
            "grade": self.grade.value,
            "qualification_scope": self.qualification_scope,
            "first_prediction_sha256": self.first_prediction_sha256,
            "replay_prediction_sha256": self.replay_prediction_sha256,
            "first_result_sha256": self.first_result_sha256,
            "replay_result_sha256": self.replay_result_sha256,
            "exact_prediction_bytes": True,
            "exact_metrics_recomputed": False,
            "protected_metrics_evaluated": False,
            "official_fm_qualified": False,
            "full_data_qualified": False,
        }

    def manifest(self) -> dict[str, object]:
        return {**self._body(), "receipt_id": self.receipt_id}


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "ReceiptError",
    "ScriptedReplayReceipt",
    "StartupReceipt",
]
