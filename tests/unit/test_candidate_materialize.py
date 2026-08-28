from __future__ import annotations

import stat
from pathlib import Path

import pytest

from kuairand_agent.research.materialize import (
    CandidateMaterializationError,
    CandidateStaticError,
    materialize_candidate,
    require_material_executable_change,
    validate_candidate_static,
)
from kuairand_agent.research.schemas import (
    GeneratedFile,
    GeneratedPackage,
    ParentSnapshot,
    ParentSourceFile,
)

BASE_SOURCE = """\
def score(value: float) -> float:
    return value
"""
CHANGED_SOURCE = """\
def score(value: float) -> float:
    tab_bias = 0.125
    return value + tab_bias
"""


def parent_snapshot() -> ParentSnapshot:
    return ParentSnapshot(
        candidate_id="fm-seed",
        files=(
            ParentSourceFile.create("candidate.py", BASE_SOURCE),
            ParentSourceFile.create("config.json", '{"seed":0}\n'),
            ParentSourceFile.create("README.md", "# Candidate\n"),
        ),
    )


def package(path: str, content: str) -> GeneratedPackage:
    return GeneratedPackage(
        request_id="implement-tab-bias",
        response_id="scripted-implement-1",
        files=(GeneratedFile(path, content),),
        material_change_summary="Change the score function body.",
        material_symbols=("score",),
    )


def test_materialization_is_disposable_reproducible_and_parent_preserving(
    tmp_path: Path,
) -> None:
    parent = parent_snapshot()
    first = materialize_candidate(parent, package("candidate.py", CHANGED_SOURCE), tmp_path / "a")
    second = materialize_candidate(parent, package("candidate.py", CHANGED_SOURCE), tmp_path / "b")

    assert first.source_digest == second.source_digest
    assert first.diff_digest == second.diff_digest
    assert first.unified_diff == second.unified_diff
    assert first.changed_paths == ("candidate.py",)
    assert "return value + tab_bias" in first.unified_diff
    assert parent.file("candidate.py").content == BASE_SOURCE
    assert stat.S_IMODE(first.destination.stat().st_mode) == 0o500
    assert stat.S_IMODE((first.destination / "candidate.py").stat().st_mode) == 0o400
    assert validate_candidate_static(first).python_files == ("candidate.py",)
    material = require_material_executable_change(parent, first)
    assert material.changed_symbols == ("candidate.py:score",)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/candidate.py",
        "../candidate.py",
        ".hidden/candidate.py",
        "kuairand_agent/scoring/protected.py",
        "evaluate.py",
        "sitecustomize.py",
        "candidate.bin",
        "nested\\candidate.py",
        "C:/candidate.py",
    ],
)
def test_invalid_generated_path_is_rejected_before_destination_exists(
    tmp_path: Path, path: str
) -> None:
    destination = tmp_path / "child"

    with pytest.raises(CandidateMaterializationError):
        materialize_candidate(parent_snapshot(), package(path, CHANGED_SOURCE), destination)

    assert not destination.exists()


def test_symlinked_destination_parent_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(CandidateMaterializationError, match="real directory"):
        materialize_candidate(
            parent_snapshot(), package("candidate.py", CHANGED_SOURCE), link / "child"
        )

    assert not (real / "child").exists()


def test_config_or_comment_only_change_is_not_a_material_executable_change(
    tmp_path: Path,
) -> None:
    config_child = materialize_candidate(
        parent_snapshot(), package("config.json", '{"seed":1}\n'), tmp_path / "config"
    )
    with pytest.raises(CandidateStaticError, match="material executable-source"):
        require_material_executable_change(parent_snapshot(), config_child)

    comment_child = materialize_candidate(
        parent_snapshot(),
        package("candidate.py", "# comment only\n" + BASE_SOURCE),
        tmp_path / "comment",
    )
    with pytest.raises(CandidateStaticError, match="material executable-source"):
        require_material_executable_change(parent_snapshot(), comment_child)


def test_declared_material_symbol_must_exist_and_change(tmp_path: Path) -> None:
    wrong_declaration = GeneratedPackage(
        request_id="implement-tab-bias",
        response_id="scripted-implement-wrong-symbol",
        files=(GeneratedFile("candidate.py", CHANGED_SOURCE),),
        material_change_summary="Claim a symbol that was not implemented.",
        material_symbols=("missing_mechanism",),
    )
    child = materialize_candidate(parent_snapshot(), wrong_declaration, tmp_path / "wrong")

    with pytest.raises(CandidateStaticError, match="declared material symbol"):
        require_material_executable_change(parent_snapshot(), child)


def test_syntax_and_forbidden_import_fail_the_static_gate_after_materialization(
    tmp_path: Path,
) -> None:
    syntax_child = materialize_candidate(
        parent_snapshot(), package("candidate.py", "def score(:\n"), tmp_path / "syntax"
    )
    with pytest.raises(CandidateStaticError, match="invalid Python"):
        validate_candidate_static(syntax_child)

    import_child = materialize_candidate(
        parent_snapshot(),
        package("candidate.py", "import kuairand_agent.scoring.protected\n" + BASE_SOURCE),
        tmp_path / "import",
    )
    with pytest.raises(CandidateStaticError, match="forbidden import"):
        validate_candidate_static(import_child)


@pytest.mark.parametrize("module", ["socket", "subprocess", "os", "kuairand_agent"])
def test_network_process_and_trusted_controller_imports_are_rejected(
    tmp_path: Path, module: str
) -> None:
    child = materialize_candidate(
        parent_snapshot(),
        package("candidate.py", f"import {module}\n" + BASE_SOURCE),
        tmp_path / module,
    )

    with pytest.raises(CandidateStaticError, match="forbidden import"):
        validate_candidate_static(child)
