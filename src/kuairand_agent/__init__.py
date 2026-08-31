"""Trusted controller for the KuaiRand-Pure autonomous research agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kuairand_agent.lab import (
        AutonomousExperimentLab,
        CampaignOptions,
        CampaignResult,
    )

__all__ = [
    "AutonomousExperimentLab",
    "CampaignOptions",
    "CampaignResult",
    "__version__",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """Load the laboratory facade lazily so package metadata imports remain lightweight."""

    if name in {"AutonomousExperimentLab", "CampaignOptions", "CampaignResult"}:
        from kuairand_agent import lab

        return getattr(lab, name)
    raise AttributeError(name)
