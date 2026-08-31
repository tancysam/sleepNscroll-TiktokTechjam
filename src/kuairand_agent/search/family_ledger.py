"""Cross-campaign scientific-family keys and replay-safe branch closure evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum

from kuairand_agent.domain.experiment import ExperimentSpec
from kuairand_agent.domain.identity import ContractId, FamilyId


class FamilyLedgerError(ValueError):
    """Family evidence does not satisfy the cross-campaign key contract."""


class BranchResult(StrEnum):
    IMPROVED = "improved"
    NO_IMPROVEMENT = "no_improvement"
    UNSUPPORTED = "unsupported"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


def family_id_for(spec: ExperimentSpec) -> FamilyId:
    """Derive a nominal family ID only from stable executable science."""

    family_id = spec.family_id
    if not isinstance(family_id, FamilyId):  # pragma: no cover - nominal constructor invariant.
        raise FamilyLedgerError("ExperimentSpec returned an invalid FamilyId")
    return family_id


@dataclass(frozen=True, slots=True, order=True)
class FamilyLedgerKey:
    """Evidence lineage key; campaign, provider, source tree, and run directory are absent."""

    contract_id: ContractId
    family_id: FamilyId

    def __post_init__(self) -> None:
        if not isinstance(self.contract_id, ContractId):
            raise FamilyLedgerError("family ledger key requires ContractId")
        if not isinstance(self.family_id, FamilyId):
            raise FamilyLedgerError("family ledger key requires FamilyId")

    @classmethod
    def for_spec(cls, contract_id: ContractId, spec: ExperimentSpec) -> FamilyLedgerKey:
        return cls(contract_id=contract_id, family_id=family_id_for(spec))

    def manifest(self) -> dict[str, str]:
        return {"contract_id": self.contract_id.value, "family_id": self.family_id.value}


@dataclass(frozen=True, slots=True, order=True)
class BranchFingerprint:
    """Exact closed-branch tuple required by the laboratory plan."""

    representation: str
    model_family: str
    objective: str
    temporal_policy: str
    fusion_member: str
    result: BranchResult

    def __post_init__(self) -> None:
        for name in (
            "representation",
            "model_family",
            "objective",
            "temporal_policy",
            "fusion_member",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or "\x00" in value:
                raise FamilyLedgerError(f"branch fingerprint {name} must be non-empty text")
        if not isinstance(self.result, BranchResult):
            raise FamilyLedgerError("branch fingerprint result is invalid")

    @classmethod
    def for_spec(cls, spec: ExperimentSpec, result: BranchResult) -> BranchFingerprint:
        return cls(
            representation=",".join(value.value for value in spec.feature_view_ids),
            model_family=spec.model_family.value,
            objective=spec.objective.value,
            temporal_policy=json.dumps(
                spec.strict_past.manifest(), sort_keys=True, separators=(",", ":")
            ),
            fusion_member=(
                ",".join(value.prediction_ref for value in spec.rank_fusion.members)
                if spec.rank_fusion is not None
                else "none"
            ),
            result=result,
        )

    @property
    def value(self) -> str:
        material = " | ".join(
            (
                self.representation,
                self.model_family,
                self.objective,
                self.temporal_policy,
                self.fusion_member,
                self.result.value,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, str]:
        return {
            "representation": self.representation,
            "model_family": self.model_family,
            "objective": self.objective,
            "temporal_policy": self.temporal_policy,
            "fusion_member": self.fusion_member,
            "result": self.result.value,
            "fingerprint": self.value,
        }


@dataclass(frozen=True, slots=True)
class FamilyLedgerEntry:
    key: FamilyLedgerKey
    fingerprints: tuple[BranchFingerprint, ...] = ()
    max_negative_results: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.key, FamilyLedgerKey):
            raise FamilyLedgerError("family ledger entry key is invalid")
        if type(self.fingerprints) is not tuple or any(
            not isinstance(value, BranchFingerprint) for value in self.fingerprints
        ):
            raise FamilyLedgerError("family ledger fingerprints are invalid")
        identities = tuple(value.value for value in self.fingerprints)
        if len(identities) != len(set(identities)):
            raise FamilyLedgerError("family ledger fingerprints contain duplicates")
        if type(self.max_negative_results) is not int or self.max_negative_results <= 0:
            raise FamilyLedgerError("max_negative_results must be a positive integer")

    @property
    def scientific_negative_count(self) -> int:
        return sum(
            value.result in {BranchResult.NO_IMPROVEMENT, BranchResult.UNSUPPORTED}
            for value in self.fingerprints
        )

    @property
    def improved(self) -> bool:
        return any(value.result is BranchResult.IMPROVED for value in self.fingerprints)

    @property
    def closed(self) -> bool:
        """Infrastructure failures never close a scientific family."""

        return self.improved or self.scientific_negative_count >= self.max_negative_results

    def append(self, fingerprint: BranchFingerprint) -> FamilyLedgerEntry:
        if fingerprint.value in {value.value for value in self.fingerprints}:
            return self
        return FamilyLedgerEntry(
            key=self.key,
            fingerprints=(*self.fingerprints, fingerprint),
            max_negative_results=self.max_negative_results,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "key": self.key.manifest(),
            "fingerprints": [value.manifest() for value in self.fingerprints],
            "max_negative_results": self.max_negative_results,
            "closed": self.closed,
        }


@dataclass(slots=True)
class FamilyLedger:
    """Small in-memory projection; durable writes belong to ``StateRepository``."""

    max_negative_results: int = 1
    _entries: dict[FamilyLedgerKey, FamilyLedgerEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if type(self.max_negative_results) is not int or self.max_negative_results <= 0:
            raise FamilyLedgerError("max_negative_results must be a positive integer")

    def record(
        self,
        *,
        contract_id: ContractId,
        spec: ExperimentSpec,
        result: BranchResult,
    ) -> FamilyLedgerEntry:
        key = FamilyLedgerKey.for_spec(contract_id, spec)
        current = self._entries.get(
            key,
            FamilyLedgerEntry(key=key, max_negative_results=self.max_negative_results),
        )
        updated = current.append(BranchFingerprint.for_spec(spec, result))
        self._entries[key] = updated
        return updated

    def entry(self, key: FamilyLedgerKey) -> FamilyLedgerEntry | None:
        return self._entries.get(key)

    def is_closed(self, *, contract_id: ContractId, spec: ExperimentSpec) -> bool:
        entry = self.entry(FamilyLedgerKey.for_spec(contract_id, spec))
        return entry.closed if entry is not None else False

    def entries(self) -> tuple[FamilyLedgerEntry, ...]:
        return tuple(
            self._entries[key]
            for key in sorted(
                self._entries, key=lambda value: (value.contract_id.value, value.family_id.value)
            )
        )
