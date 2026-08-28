from __future__ import annotations

import json
import os
import stat
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import cast

import pytest

from kuairand_agent.baselines.qualification import QualificationRequest, run_qualification

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"
DATA_ENV = "KUAIRAND_PURE_DATA_DIR"
FOUR_PLACES = Decimal("0.0001")

pytestmark = pytest.mark.skipif(
    DATA_ENV not in os.environ,
    reason=f"set {DATA_ENV} to run the official five-seed qualification",
)


def _rounded(value: object) -> Decimal:
    return Decimal(str(value)).quantize(FOUR_PLACES, rounding=ROUND_HALF_UP)


def test_official_five_seed_qualification_and_clean_replay(tmp_path: Path) -> None:
    data_dir = Path(os.environ[DATA_ENV]).resolve(strict=True)
    run_dir = tmp_path / "official-qualification"

    result = run_qualification(QualificationRequest(data_dir, STARTER, run_dir))

    assert result.launch_count == 6
    assert result.run_dir == run_dir.absolute()
    manifest = cast(
        dict[str, object],
        json.loads((run_dir / "manifest.json").read_text(encoding="ascii")),
    )
    assert manifest["status"] == "baseline_reproduced"
    launch_accounting = cast(dict[str, object], manifest["launch_accounting"])
    assert launch_accounting["charged_launches"] == 6
    assert launch_accounting["expected_launches"] == 6
    fm = cast(dict[str, object], manifest["fm"])
    five_seed = cast(dict[str, object], fm["five_seed_mean"])
    assert _rounded(five_seed["GAUC"]) == Decimal("0.6674")
    assert _rounded(five_seed["nDCG@5"]) == Decimal("0.5357")
    assert _rounded(five_seed["primary"]) == Decimal("0.6016")
    assert fm["reference_passed"] is True
    clean = cast(dict[str, object], fm["clean_seed_zero"])
    assert clean["prediction_identity"] is True
    assert clean["within_user_order_identity"] is True
    clean_trace = cast(dict[str, object], clean["training_trace"])
    clean_process = cast(dict[str, object], clean_trace["clean_subprocess"])
    assert clean_process["fresh_interpreter"] is True
    assert clean_process["source_retrain"] is True
    assert clean_process["identity_verified_by_parent"] is True

    final_period = cast(dict[str, object], manifest["final_period"])
    assert final_period == {
        "input_rows": 170_588,
        "outcomes_accessed": False,
        "outcomes_scored": False,
        "target_capability": None,
    }
    assert result.final_submission.row_count == 170_588
    assert result.final_submission.round_trip_identity is True
    fallback = run_dir / "fallback"
    assert stat.S_IMODE(fallback.stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE((fallback / "manifest.json").stat().st_mode) & 0o222 == 0
