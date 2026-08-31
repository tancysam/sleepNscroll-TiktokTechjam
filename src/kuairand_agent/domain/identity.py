"""Canonical JSON and nominal content identities for the autonomous laboratory.

The identity constructors in this module deliberately expose only fields that belong to each
identity.  Operational metadata such as host names, process IDs, retry counts, and report labels
cannot accidentally enter a scientific identity because the relevant constructors do not accept
them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

SHA256_HEX_LENGTH = 64
IDENTITY_SCHEMA_VERSION = 1


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be represented as finite canonical JSON."""


class IdentityError(ValueError):
    """Raised when an identity component or digest is malformed."""


def _number_text(value: int | float) -> str:
    """Render JSON numbers so mathematically equal Python ints/floats share one spelling."""

    if type(value) is int:
        return str(value)
    if not math.isfinite(value):
        raise CanonicalJsonError("canonical JSON numbers must be finite")
    if value == 0.0:
        return "0"
    if value.is_integer():
        return str(int(value))

    # ``repr`` is Python's shortest round-tripping decimal for a binary float.  Rendering that
    # decimal without exponent notation removes alternate spellings such as 1e-07/1E-7 while
    # retaining exact round-trip semantics.
    rendered = repr(value)
    if "e" in rendered or "E" in rendered:
        mantissa, exponent_text = rendered.lower().split("e", maxsplit=1)
        exponent = int(exponent_text)
        negative = mantissa.startswith("-")
        unsigned = mantissa.removeprefix("-")
        whole, dot, fraction = unsigned.partition(".")
        digits = whole + (fraction if dot else "")
        decimal_position = len(whole) + exponent
        if decimal_position <= 0:
            rendered = "0." + ("0" * -decimal_position) + digits
        elif decimal_position >= len(digits):
            rendered = digits + ("0" * (decimal_position - len(digits)))
        else:
            rendered = digits[:decimal_position] + "." + digits[decimal_position:]
        if negative:
            rendered = "-" + rendered
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _encode_canonical(value: object, *, location: str) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float):
        return _number_text(cast(int | float, value))
    if type(value) is str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        items: list[str] = []
        for key in value:
            if type(key) is not str:
                raise CanonicalJsonError(f"{location} object keys must be strings")
        for key in sorted(value):
            encoded_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            encoded_value = _encode_canonical(value[key], location=f"{location}.{key}")
            items.append(f"{encoded_key}:{encoded_value}")
        return "{" + ",".join(items) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return (
            "["
            + ",".join(
                _encode_canonical(item, location=f"{location}[{index}]")
                for index, item in enumerate(value)
            )
            + "]"
        )
    raise CanonicalJsonError(
        f"{location} contains unsupported canonical JSON type {type(value).__name__}"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return sorted-key, finite, numerically normalized UTF-8 JSON bytes.

    Tuples are accepted as immutable JSON arrays.  Bytes, sets, non-string mapping keys,
    non-finite floats, custom numeric classes, and arbitrary Python objects are rejected rather
    than being stringified or coerced.
    """

    return _encode_canonical(value, location="$").encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the ordinary SHA-256 digest of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IdentityError(f"{location} must be a full lowercase SHA-256 digest")
    return value


def _text(value: object, location: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise IdentityError(f"{location} must be one non-empty canonical line of text")
    return value


def _digest_mapping(value: Mapping[str, str], location: str) -> dict[str, str]:
    if not value:
        raise IdentityError(f"{location} must not be empty")
    rendered: dict[str, str] = {}
    for key, digest in value.items():
        name = _text(key, f"{location} key")
        if name in rendered:
            raise IdentityError(f"{location} contains duplicate key {name!r}")
        rendered[name] = _sha256(digest, f"{location}.{name}")
    return rendered


def _reject_absolute_machine_paths(value: object, location: str = "identity") -> None:
    if type(value) is str:
        windows_drive = (
            len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in ("/", "\\")
        )
        if (
            value.startswith("/")
            or value.startswith("\\\\")
            or value.lower().startswith("file://")
            or windows_drive
        ):
            raise IdentityError(f"{location} must not contain absolute machine paths")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_machine_paths(key, f"{location} key")
            _reject_absolute_machine_paths(item, f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_absolute_machine_paths(item, f"{location}[{index}]")


def _identity_digest(identity_type: str, components: Mapping[str, object]) -> str:
    _reject_absolute_machine_paths(components)
    return canonical_json_sha256(
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "identity_type": identity_type,
            "components": components,
        }
    )


@dataclass(frozen=True, slots=True, order=True)
class Sha256Id:
    """Base implementation for nominal, full-SHA identity value objects."""

    value: str

    def __post_init__(self) -> None:
        _sha256(self.value, type(self).__name__)

    def __str__(self) -> str:
        return self.value


class ContractId(Sha256Id):
    """Identity of the complete immutable competition contract."""

    @classmethod
    def from_manifest(cls, manifest: object) -> Self:
        _reject_absolute_machine_paths(manifest, "contract manifest")
        return cls(canonical_json_sha256(manifest))


class CampaignId(Sha256Id):
    """Identity of one campaign, excluding transient process state."""

    @classmethod
    def derive(
        cls,
        *,
        contract_id: ContractId,
        campaign_config: object,
        start_nonce: str,
    ) -> Self:
        if not isinstance(contract_id, ContractId):
            raise IdentityError("contract_id must be a ContractId")
        return cls(
            _identity_digest(
                "campaign",
                {
                    "contract_id": contract_id.value,
                    "campaign_config": campaign_config,
                    "start_nonce": _text(start_nonce, "start_nonce"),
                },
            )
        )


class FamilyId(Sha256Id):
    """Scientific mechanism family, excluding seed, device, and execution attempt."""

    @classmethod
    def derive(
        cls,
        *,
        mechanism: str,
        feature_family: str,
        target_family: str,
        objective_family: str,
    ) -> Self:
        return cls(
            _identity_digest(
                "family",
                {
                    "mechanism": _text(mechanism, "mechanism"),
                    "feature_family": _text(feature_family, "feature_family"),
                    "target_family": _text(target_family, "target_family"),
                    "objective_family": _text(objective_family, "objective_family"),
                },
            )
        )


class ExperimentId(Sha256Id):
    """Semantic experiment identity, excluding device and infrastructure retry state."""

    @classmethod
    def derive(
        cls,
        *,
        experiment_spec: object,
        data_identities: Mapping[str, str],
        fold_identities: Mapping[str, str],
        code_artifact_sha256: str,
    ) -> Self:
        return cls(
            _identity_digest(
                "experiment",
                {
                    "experiment_spec": experiment_spec,
                    "data_identities": _digest_mapping(data_identities, "data_identities"),
                    "fold_identities": _digest_mapping(fold_identities, "fold_identities"),
                    "code_artifact_sha256": _sha256(code_artifact_sha256, "code_artifact_sha256"),
                },
            )
        )


class TrialId(Sha256Id):
    """Trainer/backend realization of an experiment, excluding PID and host name."""

    @classmethod
    def derive(
        cls,
        *,
        experiment_id: ExperimentId,
        trainer_id: str,
        trainer_version: str,
        backend: str,
        precision: str,
        dependency_lock_sha256: str,
        seed: int,
        fold: str,
        fidelity: object,
        qualified_settings: object,
    ) -> Self:
        if not isinstance(experiment_id, ExperimentId):
            raise IdentityError("experiment_id must be an ExperimentId")
        if type(seed) is not int or seed < 0:
            raise IdentityError("seed must be a non-negative integer")
        return cls(
            _identity_digest(
                "trial",
                {
                    "experiment_id": experiment_id.value,
                    "trainer_id": _text(trainer_id, "trainer_id"),
                    "trainer_version": _text(trainer_version, "trainer_version"),
                    "backend": _text(backend, "backend"),
                    "precision": _text(precision, "precision"),
                    "dependency_lock_sha256": _sha256(
                        dependency_lock_sha256, "dependency_lock_sha256"
                    ),
                    "seed": seed,
                    "fold": _text(fold, "fold"),
                    "fidelity": fidelity,
                    "qualified_settings": qualified_settings,
                },
            )
        )


class AttemptId(Sha256Id):
    """One monotonically numbered infrastructure attempt for a trial."""

    @classmethod
    def derive(cls, *, trial_id: TrialId, infrastructure_attempt: int) -> Self:
        if not isinstance(trial_id, TrialId):
            raise IdentityError("trial_id must be a TrialId")
        if type(infrastructure_attempt) is not int or infrastructure_attempt < 1:
            raise IdentityError("infrastructure_attempt must be a positive integer")
        return cls(
            _identity_digest(
                "attempt",
                {
                    "trial_id": trial_id.value,
                    "infrastructure_attempt": infrastructure_attempt,
                },
            )
        )


def _ordered_row_ids(values: Sequence[int | str]) -> list[int | str]:
    if not values:
        raise IdentityError("ordered_row_ids must not be empty")
    rendered: list[int | str] = []
    for index, value in enumerate(values):
        if type(value) is int and value >= 0:
            rendered.append(value)
        elif type(value) is str:
            rendered.append(_text(value, f"ordered_row_ids[{index}]"))
        else:
            raise IdentityError("ordered_row_ids must contain non-negative ints or canonical text")
    if len(set(rendered)) != len(rendered):
        raise IdentityError("ordered_row_ids must not contain duplicates")
    return rendered


class PredictionId(Sha256Id):
    """Exact ordered prediction-vector identity, excluding human report labels."""

    @classmethod
    def from_trial(
        cls,
        *,
        ordered_row_ids: Sequence[int | str],
        prediction_sha256: str,
        trial_id: TrialId,
    ) -> Self:
        if not isinstance(trial_id, TrialId):
            raise IdentityError("trial_id must be a TrialId")
        return cls._derive(
            ordered_row_ids=ordered_row_ids,
            prediction_sha256=prediction_sha256,
            producer_type="trial",
            producer_sha256=trial_id.value,
        )

    @classmethod
    def from_rank_graph(
        cls,
        *,
        ordered_row_ids: Sequence[int | str],
        prediction_sha256: str,
        rank_graph_sha256: str,
    ) -> Self:
        return cls._derive(
            ordered_row_ids=ordered_row_ids,
            prediction_sha256=prediction_sha256,
            producer_type="rank_graph",
            producer_sha256=_sha256(rank_graph_sha256, "rank_graph_sha256"),
        )

    @classmethod
    def _derive(
        cls,
        *,
        ordered_row_ids: Sequence[int | str],
        prediction_sha256: str,
        producer_type: str,
        producer_sha256: str,
    ) -> Self:
        return cls(
            _identity_digest(
                "prediction",
                {
                    "ordered_row_ids": _ordered_row_ids(ordered_row_ids),
                    "prediction_sha256": _sha256(prediction_sha256, "prediction_sha256"),
                    "producer": {
                        "type": producer_type,
                        "sha256": producer_sha256,
                    },
                },
            )
        )


class DecisionId(Sha256Id):
    """Policy and exact evidence identity, excluding prose explanations."""

    @classmethod
    def derive(
        cls,
        *,
        policy_sha256: str,
        evidence_ids: Mapping[str, Sha256Id],
    ) -> Self:
        if not evidence_ids:
            raise IdentityError("evidence_ids must not be empty")
        rendered: dict[str, str] = {}
        for role, evidence_id in evidence_ids.items():
            name = _text(role, "evidence role")
            if not isinstance(evidence_id, Sha256Id):
                raise IdentityError(f"evidence_ids.{name} must be a nominal SHA identity")
            rendered[name] = evidence_id.value
        return cls(
            _identity_digest(
                "decision",
                {
                    "policy_sha256": _sha256(policy_sha256, "policy_sha256"),
                    "evidence_ids": rendered,
                },
            )
        )


class BundleId(Sha256Id):
    """Sealed bundle identity, excluding mutable filesystem paths."""

    @classmethod
    def derive(
        cls,
        *,
        selected_prediction_id: PredictionId,
        replay_output_sha256: Mapping[str, str],
        submission_sha256: str,
        manifest_sha256: str,
    ) -> Self:
        if not isinstance(selected_prediction_id, PredictionId):
            raise IdentityError("selected_prediction_id must be a PredictionId")
        return cls(
            _identity_digest(
                "bundle",
                {
                    "selected_prediction_id": selected_prediction_id.value,
                    "replay_output_sha256": _digest_mapping(
                        replay_output_sha256, "replay_output_sha256"
                    ),
                    "submission_sha256": _sha256(submission_sha256, "submission_sha256"),
                    "manifest_sha256": _sha256(manifest_sha256, "manifest_sha256"),
                },
            )
        )


__all__ = [
    "SHA256_HEX_LENGTH",
    "AttemptId",
    "BundleId",
    "CampaignId",
    "CanonicalJsonError",
    "ContractId",
    "DecisionId",
    "ExperimentId",
    "FamilyId",
    "IdentityError",
    "PredictionId",
    "Sha256Id",
    "TrialId",
    "canonical_json_bytes",
    "canonical_json_sha256",
]
