"""Explicit unavailable GPU trainer for the current CPU-only LightGBM execution build.

The existing tree configuration content-hashes ``device_type=cpu`` and the existing Runner hides
CUDA devices.  Re-labelling either primitive would create false evidence, so this adapter has a
distinct GPU identity and fails closed until a GPU-qualified primitive and process profile exist.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from kuairand_agent.training.protocol import (
    EnvironmentReceipt,
    QualificationReceipt,
    QualificationStatus,
    TrainerError,
    TrainerFailureCode,
    TrainerIdentity,
    TrialRequest,
    TrialResult,
    validate_request_for_trainer,
)

LIGHTGBM_GPU_TRAINER_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class GPUCapabilityEvidence:
    """Honest negative capability receipt for this source/build combination."""

    platform_system: str
    platform_machine: str
    tree_primitive_device: str = "cpu"
    runner_gpu_visible: bool = False
    gpu_build_qualified: bool = False
    detail: str = (
        "current LambdaRank config is content-bound to CPU and current Runner hides CUDA devices"
    )

    @classmethod
    def capture(cls) -> GPUCapabilityEvidence:
        return cls(
            platform_system=platform.system() or "unknown",
            platform_machine=platform.machine() or "unknown",
        )

    def manifest(self) -> dict[str, object]:
        return {
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "tree_primitive_device": self.tree_primitive_device,
            "runner_gpu_visible": self.runner_gpu_visible,
            "gpu_build_qualified": self.gpu_build_qualified,
            "detail": self.detail,
        }


class LightGBMGPUTrainer:
    """GPU identity and preflight that never falls through to CPU execution."""

    def __init__(self, *, dependency_lock_sha256: str) -> None:
        self._identity = TrainerIdentity(
            trainer_id="lightgbm-lambdarank",
            trainer_version=LIGHTGBM_GPU_TRAINER_VERSION,
            backend="lightgbm-gpu",
            device="gpu",
            precision="float32",
            dependency_lock_sha256=dependency_lock_sha256,
        )
        self._capability = GPUCapabilityEvidence.capture()

    @property
    def identity(self) -> TrainerIdentity:
        return self._identity

    @property
    def capability(self) -> GPUCapabilityEvidence:
        return self._capability

    def preflight(self, request: TrialRequest) -> QualificationReceipt:
        validate_request_for_trainer(request, self.identity)
        return QualificationReceipt(
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            status=QualificationStatus.UNSUPPORTED,
            checks=(
                "canonical-request",
                "distinct-gpu-trial-identity",
                "tree-primitive-device-capability",
                "runner-gpu-visibility",
            ),
            environment=EnvironmentReceipt.capture(self.identity),
            failure_code=TrainerFailureCode.UNSUPPORTED,
            detail=self.capability.detail,
        )

    def fit_predict(self, request: TrialRequest) -> TrialResult:
        validate_request_for_trainer(request, self.identity)
        raise TrainerError(
            TrainerFailureCode.UNSUPPORTED,
            self.capability.detail,
            trainer_identity=self.identity,
            trial_id=request.trial_id,
            attempt_id=request.attempt_id,
        )


__all__ = [
    "LIGHTGBM_GPU_TRAINER_VERSION",
    "GPUCapabilityEvidence",
    "LightGBMGPUTrainer",
]
