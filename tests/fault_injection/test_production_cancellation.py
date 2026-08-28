from __future__ import annotations

import threading
from pathlib import Path

import pytest

from kuairand_agent.campaign.full_campaign import (
    FullCampaignCancelled,
    run_provider_free_campaign,
)
from kuairand_agent.finalization.production import (
    ProductionFinalizationError,
    finalize_provider_free_campaign,
    replay_final_bundle,
)


def test_preset_cancellation_stops_full_campaign_before_any_stage_or_launch(
    tmp_path: Path,
) -> None:
    cancellation = threading.Event()
    cancellation.set()
    missing_run = tmp_path / "must-not-be-created"

    with pytest.raises(FullCampaignCancelled, match="cancelled"):
        run_provider_free_campaign(
            missing_run,
            project_root=tmp_path,
            cancel_event=cancellation,
        )

    assert not missing_run.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_preset_cancellation_stops_production_finalization_before_any_mutation(
    tmp_path: Path,
) -> None:
    cancellation = threading.Event()
    cancellation.set()
    missing_run = tmp_path / "must-not-be-created"

    with pytest.raises(ProductionFinalizationError, match="cancelled"):
        finalize_provider_free_campaign(
            missing_run,
            project_root=tmp_path,
            cancel_event=cancellation,
        )

    assert not missing_run.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_preset_cancellation_stops_closed_bundle_replay_before_verification_or_scratch(
    tmp_path: Path,
) -> None:
    cancellation = threading.Event()
    cancellation.set()
    missing_bundle = tmp_path / "must-not-be-opened"
    missing_data = tmp_path / "must-not-be-read"

    with pytest.raises(ProductionFinalizationError, match="cancelled"):
        replay_final_bundle(
            missing_bundle,
            missing_data,
            "d" * 64,
            project_root=tmp_path,
            cancel_event=cancellation,
        )

    assert not missing_bundle.exists()
    assert not missing_data.exists()
    assert tuple(tmp_path.iterdir()) == ()
