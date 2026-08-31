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
    (copy / "smuggled.py").write_text("# not organizer source\n", encoding="utf-8")
    with pytest.raises(OrganizerIntegrityError, match="unexpected"):
        verify_starter_kit(copy)


def test_starter_tolerates_bytecode_cache_written_by_the_organizer_workflow(
    tmp_path: Path,
) -> None:
    """The organizer README tells you to run submit.py from inside this directory.

    CPython then writes __pycache__ beside the pinned files.  Rejecting that halted a live
    campaign over a directory entry that changes no pinned byte, so the documented
    pre-submission check was lethal to any concurrent run.  hash_source_tree already skips
    __pycache__; this keeps the two consistent while every pinned file stays digest-verified.
    """

    copy = tmp_path / "starter"
    shutil.copytree(ROOT / "kuairand-starter-kit", copy)
    cache = copy / "__pycache__"
    cache.mkdir()
    (cache / "data.cpython-312.pyc").write_bytes(b"\x00compiled\x00")

    result = verify_starter_kit(copy)
    pinned = verify_starter_kit(ROOT / "kuairand-starter-kit")
    assert result.files == STARTER_FILE_SHA256
    assert result.manifest_sha256 == pinned.manifest_sha256
