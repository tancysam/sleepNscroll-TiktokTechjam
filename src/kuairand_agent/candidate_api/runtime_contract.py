"""Canonical controller/model contract for generated candidate execution.

The generated candidate is intentionally self-contained, so it cannot import this module at
runtime.  This module is nevertheless the single controller-side authority used to build executor
requests, describe the model-facing interface, protect stable source, and test the distributed
seed wrapper for conformance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final


class CandidateRuntimeContractError(ValueError):
    """A model-generated overlay or controller request violates the runtime contract."""


@dataclass(frozen=True, slots=True)
class CandidateRuntimeContract:
    """Small, versioned interface at the generated-candidate execution seam."""

    schema_version: int = 1
    entry_point: str = "candidate.py"
    config_path: str = "config.json"
    checkpoint_path: str = "checkpoint/model.txt"
    train_result_path: str = "candidate_result.json"
    prediction_result_path: str = "prediction_result.json"
    scores_path: str = "scores.npy"
    protected_paths: tuple[str, ...] = (
        "candidate.py",
        "reference_categorical_ranker.py",
        "reference_listnet_ranker.py",
        "reference_observed_pair_fm.py",
        "reference_observed_pair_objectives.py",
        "reference_pairwise_fm.py",
        "reference_pointwise_ranker.py",
    )
    train_request_fields: tuple[str, ...] = (
        "config_digest",
        "data_digest",
        "features_handle",
        "protocol_schema_version",
        "seed",
        "source_digest",
        "split_token",
        "targets_handle",
        "user_groups_handle",
    )
    prediction_request_fields: tuple[str, ...] = (
        "checkpoint_digest",
        "config_digest",
        "data_digest",
        "expected_count",
        "features_handle",
        "protocol_schema_version",
        "source_digest",
        "split_token",
    )
    features_handle: str = "features"
    targets_handle: str = "targets"
    user_groups_handle: str = "user_groups"

    def to_wire(self) -> dict[str, object]:
        """Return the exact provider-visible interface description."""

        return {
            "schema_version": self.schema_version,
            "stable_files": {
                "entry_point": self.entry_point,
                "protected_paths": list(self.protected_paths),
            },
            "artifact_paths": {
                "config": self.config_path,
                "checkpoint": self.checkpoint_path,
                "train_result": self.train_result_path,
                "prediction_result": self.prediction_result_path,
                "scores": self.scores_path,
            },
            "train_request": {
                "exact_fields": list(self.train_request_fields),
                "fixed_handles": {
                    "features_handle": self.features_handle,
                    "targets_handle": self.targets_handle,
                    "user_groups_handle": self.user_groups_handle,
                },
            },
            "prediction_request": {
                "exact_fields": list(self.prediction_request_fields),
                "fixed_handles": {"features_handle": self.features_handle},
            },
            "model_interface": {
                "validate_config": "validate_config(config) -> None",
                "train_model": (
                    "train_model(features, targets, user_groups, config, seed) "
                    "-> dict[str, numpy.ndarray]"
                ),
                "predict_scores": (
                    "predict_scores(features, checkpoint) -> numpy.ndarray shape (N,)"
                ),
                "training_diagnostics": (
                    "training_diagnostics(config, checkpoint) -> dict[str, int | float]"
                ),
            },
            "numeric_semantics": {
                "features": (
                    "finite float64 matrix shape (N,D), engineered by the controller; column "
                    "order is exactly safe_context.method_cards[name="
                    "controller_causal_feature_bundle].feature_names_csv"
                ),
                "targets": "finite float64 binary vector shape (N,)",
                "user_groups": "finite numeric user-group vector shape (N,)",
                "seed": "unsigned 32-bit integer",
                "checkpoint": (
                    "1..64 named finite numeric NumPy arrays with at most 536870912 total bytes"
                ),
                "scores": "finite float64 vector shape (N,)",
            },
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_wire(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def validate_overlay_paths(self, paths: Iterable[str]) -> None:
        """Reject model output that attempts to replace controller-owned source."""

        protected = set(self.protected_paths)
        changed = sorted(protected.intersection(paths))
        if changed:
            joined = ", ".join(changed)
            raise CandidateRuntimeContractError(
                f"generated package cannot replace protected runtime file(s): {joined}"
            )

    def training_payload(
        self,
        *,
        source_digest: str,
        config_digest: str,
        data_digest: str,
        split_token: str,
        seed: int,
    ) -> dict[str, object]:
        return {
            "protocol_schema_version": self.schema_version,
            "source_digest": source_digest,
            "config_digest": config_digest,
            "data_digest": data_digest,
            "split_token": split_token,
            "seed": seed,
            "features_handle": self.features_handle,
            "targets_handle": self.targets_handle,
            "user_groups_handle": self.user_groups_handle,
        }

    def prediction_payload(
        self,
        *,
        source_digest: str,
        config_digest: str,
        data_digest: str,
        split_token: str,
        expected_count: int,
        checkpoint_digest: str,
    ) -> dict[str, object]:
        return {
            "protocol_schema_version": self.schema_version,
            "source_digest": source_digest,
            "config_digest": config_digest,
            "data_digest": data_digest,
            "split_token": split_token,
            "features_handle": self.features_handle,
            "expected_count": expected_count,
            "checkpoint_digest": checkpoint_digest,
        }


CANDIDATE_RUNTIME_CONTRACT: Final = CandidateRuntimeContract()


__all__ = [
    "CANDIDATE_RUNTIME_CONTRACT",
    "CandidateRuntimeContract",
    "CandidateRuntimeContractError",
]
