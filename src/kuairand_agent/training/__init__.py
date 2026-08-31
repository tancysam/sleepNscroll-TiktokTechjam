"""Qualified trainer, process, and replay-evidence seams."""

from kuairand_agent.training.lightgbm_cpu import (
    LIGHTGBM_CPU_TRAINER_VERSION,
    LightGBMCPUTrainer,
    LightGBMCPUTrialPayload,
)
from kuairand_agent.training.lightgbm_gpu import (
    LIGHTGBM_GPU_TRAINER_VERSION,
    GPUCapabilityEvidence,
    LightGBMGPUTrainer,
)
from kuairand_agent.training.official_fm import (
    OFFICIAL_FM_TRAINER_VERSION,
    OfficialFMTrainer,
    OfficialFMTrialPayload,
)
from kuairand_agent.training.process_executor import ProcessExecutionReceipt, ProcessExecutor
from kuairand_agent.training.protocol import (
    TRAINER_PROTOCOL_SCHEMA_VERSION,
    DataReceipt,
    EnvironmentReceipt,
    FeatureReceipt,
    ModelReceipt,
    QualificationReceipt,
    QualificationStatus,
    QualifiedTrainer,
    ResourceLimits,
    ResourceReceipt,
    TimingReceipt,
    TrainerContractError,
    TrainerError,
    TrainerFailureCode,
    TrainerIdentity,
    TrialRequest,
    TrialResult,
)
from kuairand_agent.training.qualification import qualify_trainer
from kuairand_agent.training.scripted import (
    SCRIPTED_TRAINER_VERSION,
    ScriptedTrainer,
    ScriptedTrialPayload,
)

__all__ = [
    "LIGHTGBM_CPU_TRAINER_VERSION",
    "LIGHTGBM_GPU_TRAINER_VERSION",
    "OFFICIAL_FM_TRAINER_VERSION",
    "SCRIPTED_TRAINER_VERSION",
    "TRAINER_PROTOCOL_SCHEMA_VERSION",
    "DataReceipt",
    "EnvironmentReceipt",
    "FeatureReceipt",
    "GPUCapabilityEvidence",
    "LightGBMCPUTrainer",
    "LightGBMCPUTrialPayload",
    "LightGBMGPUTrainer",
    "ModelReceipt",
    "OfficialFMTrainer",
    "OfficialFMTrialPayload",
    "ProcessExecutionReceipt",
    "ProcessExecutor",
    "QualificationReceipt",
    "QualificationStatus",
    "QualifiedTrainer",
    "ResourceLimits",
    "ResourceReceipt",
    "ScriptedTrainer",
    "ScriptedTrialPayload",
    "TimingReceipt",
    "TrainerContractError",
    "TrainerError",
    "TrainerFailureCode",
    "TrainerIdentity",
    "TrialRequest",
    "TrialResult",
    "qualify_trainer",
]
