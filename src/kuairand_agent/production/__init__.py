"""Production-only admission and execution capabilities.

The package is intentionally separate from the legacy campaign implementation.  Production
controllers may consume the typed values exported here, but admission never opens mutable state.
"""

from kuairand_agent.production.admission import (
    ControllerCapabilityReceipt,
    MeasuredFamilyCapability,
    ProductionAdmission,
    ProductionAdmissionError,
    ProductionRuntimeCapabilities,
    admit_cpu_fallback,
)

__all__ = [
    "ControllerCapabilityReceipt",
    "MeasuredFamilyCapability",
    "ProductionAdmission",
    "ProductionAdmissionError",
    "ProductionRuntimeCapabilities",
    "admit_cpu_fallback",
]
