from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from kuairand_agent.research.materialize import (
    CandidateMaterializationError,
    CandidateStaticError,
    materialize_candidate,
    require_material_executable_change,
    validate_candidate_static,
    validate_model_generated_overlay,
)
from kuairand_agent.research.schemas import (
    GeneratedFile,
    GeneratedPackage,
    ParentSnapshot,
    ParentSourceFile,
)
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateManifestPolicyError,
    CandidateSourcePolicy,
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


def test_default_source_policy_has_canonical_wire_identity_and_manifest_fingerprints() -> None:
    policy = DEFAULT_CANDIDATE_SOURCE_POLICY

    assert policy.final_entrypoint == "candidate.py"
    assert policy.allowed_suffixes == (".json", ".md", ".py")
    assert "baseline.py" in policy.forbidden_basenames
    assert policy.overlay_semantics == "complete_file_overlay"
    assert CandidateSourcePolicy.from_mapping(policy.to_wire()) == policy
    assert len(policy.digest) == 64

    with pytest.raises(CandidateManifestPolicyError) as forbidden:
        policy.validate_manifest(("candidate.py", "baseline.py"))
    assert forbidden.value.fingerprint == ("candidate_path_policy:forbidden_basename:baseline.py")

    with pytest.raises(CandidateManifestPolicyError) as unsupported:
        policy.validate_manifest(("candidate.py", "submission.csv"))
    assert unsupported.value.fingerprint == ("candidate_path_policy:unsupported_suffix:.csv")


def test_source_policy_requires_entrypoint_in_final_tree_but_allows_helper_only_overlay() -> None:
    policy = DEFAULT_CANDIDATE_SOURCE_POLICY

    policy.validate_manifest(("pairwise_fm.py",), parent_paths=("candidate.py",))
    with pytest.raises(CandidateManifestPolicyError) as missing:
        policy.validate_manifest(("pairwise_fm.py",), parent_paths=())
    assert missing.value.fingerprint == "candidate_manifest:missing_entrypoint:candidate.py"


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


def test_live_model_overlay_cannot_replace_stable_candidate_wrapper() -> None:
    with pytest.raises(CandidateMaterializationError, match="protected runtime file"):
        validate_model_generated_overlay(package("candidate.py", CHANGED_SOURCE))
    with pytest.raises(CandidateMaterializationError, match="protected runtime file"):
        validate_model_generated_overlay(package("reference_pairwise_fm.py", CHANGED_SOURCE))
    with pytest.raises(CandidateMaterializationError, match="protected runtime file"):
        validate_model_generated_overlay(
            package("reference_categorical_ranker.py", CHANGED_SOURCE)
        )
    with pytest.raises(CandidateMaterializationError, match="protected runtime file"):
        validate_model_generated_overlay(package("reference_listnet_ranker.py", CHANGED_SOURCE))
    with pytest.raises(CandidateMaterializationError, match="protected runtime file"):
        validate_model_generated_overlay(
            package("reference_observed_pair_fm.py", CHANGED_SOURCE)
        )
    with pytest.raises(CandidateMaterializationError, match="protected runtime file"):
        validate_model_generated_overlay(
            package("reference_observed_pair_objectives.py", CHANGED_SOURCE)
        )
    with pytest.raises(CandidateMaterializationError, match="protected runtime file"):
        validate_model_generated_overlay(package("reference_pointwise_ranker.py", CHANGED_SOURCE))

    validate_model_generated_overlay(
        package("model_impl.py", "def score(value):\n    return value + 1\n")
    )


def test_json_equivalent_python_literal_from_provider_is_canonicalized(
    tmp_path: Path,
) -> None:
    captured_attempt_10_config = (
        "{'candidate_family':'calibrated_hierarchical_prior','epochs':64,"
        "'l2':0.1,'learning_rate':0.05,'logit_clip':40.0,'schema_version':1}"
    )
    generated = GeneratedPackage(
        request_id="implement-captured-provider-package",
        response_id="captured-attempt-10",
        files=(
            GeneratedFile("candidate.py", CHANGED_SOURCE),
            GeneratedFile("config.json", captured_attempt_10_config),
        ),
        material_change_summary="Change scoring and its JSON configuration.",
        material_symbols=("score",),
    )

    child = materialize_candidate(parent_snapshot(), generated, tmp_path / "normalized")

    assert validate_candidate_static(child).json_files == ("config.json",)
    assert json.loads(child.file("config.json").content) == {
        "candidate_family": "calibrated_hierarchical_prior",
        "epochs": 64,
        "l2": 0.1,
        "learning_rate": 0.05,
        "logit_clip": 40.0,
        "schema_version": 1,
    }


@pytest.mark.parametrize(
    "unsafe_content",
    (
        "{'value': __import__('os').getcwd()}",
        "{'value': (1, 2)}",
        "{'value': {1, 2}}",
        "{'value': float('nan')}",
        "{'duplicate': 1, 'duplicate': 2}",
    ),
)
def test_json_literal_normalization_never_executes_or_loosens_json_semantics(
    tmp_path: Path, unsafe_content: str
) -> None:
    generated = GeneratedPackage(
        request_id="implement-untrusted-json",
        response_id="untrusted-json",
        files=(
            GeneratedFile("candidate.py", CHANGED_SOURCE),
            GeneratedFile("config.json", unsafe_content),
        ),
        material_change_summary="Exercise the safe JSON normalization boundary.",
        material_symbols=("score",),
    )
    child = materialize_candidate(parent_snapshot(), generated, tmp_path / "unsafe-json")

    assert child.file("config.json").content == unsafe_content
    with pytest.raises(CandidateStaticError, match="invalid JSON"):
        validate_candidate_static(child)


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


def test_extra_incorrect_material_symbol_does_not_hide_a_real_declared_change(
    tmp_path: Path,
) -> None:
    noisy_declaration = GeneratedPackage(
        request_id="implement-tab-bias",
        response_id="scripted-implement-extra-symbol",
        files=(GeneratedFile("candidate.py", CHANGED_SOURCE),),
        material_change_summary="Change score but accidentally over-declare one symbol.",
        material_symbols=("score", "unchanged_or_missing"),
    )
    child = materialize_candidate(parent_snapshot(), noisy_declaration, tmp_path / "noisy")

    evidence = require_material_executable_change(parent_snapshot(), child)

    assert evidence.changed_symbols == ("candidate.py:score",)


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


def test_numpy_load_context_manager_bug_is_rejected_before_execution(tmp_path: Path) -> None:
    captured_attempt_7_loader = """\
import numpy as np

def score(value: float):
    with np.load('targets.npy', allow_pickle=False) as loaded:
        if hasattr(loaded, 'files'):
            return value
        return np.asarray(loaded)
"""
    child = materialize_candidate(
        parent_snapshot(),
        package("candidate.py", captured_attempt_7_loader),
        tmp_path / "numpy-load-context-manager",
    )

    with pytest.raises(CandidateStaticError, match=r"np.load.*context manager"):
        validate_candidate_static(child)


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
