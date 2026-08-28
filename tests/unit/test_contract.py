from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kuairand_agent.contract import (
    STARTER_FILE_SHA256,
    OrganizerIntegrityError,
    verify_starter_kit,
)

ROOT = Path(__file__).parents[2]


def test_organizer_starter_matches_complete_pinned_manifest() -> None:
    result = verify_starter_kit(ROOT / "kuairand-starter-kit")
    assert result.files == STARTER_FILE_SHA256
    assert len(result.manifest_sha256) == 64


def test_starter_digest_change_is_rejected(tmp_path: Path) -> None:
    copy = tmp_path / "starter"
    shutil.copytree(ROOT / "kuairand-starter-kit", copy)
    (copy / "evaluate.py").write_text("# modified\n", encoding="utf-8")
    with pytest.raises(OrganizerIntegrityError, match="digest mismatch"):
        verify_starter_kit(copy)


def test_starter_extra_member_is_rejected(tmp_path: Path) -> None:
    copy = tmp_path / "starter"
    shutil.copytree(ROOT / "kuairand-starter-kit", copy)
    (copy / "__pycache__").mkdir()
    with pytest.raises(OrganizerIntegrityError, match="unexpected"):
        verify_starter_kit(copy)
