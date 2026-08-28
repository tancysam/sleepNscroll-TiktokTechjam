from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kuairand_agent.campaign.provenance import (
    ProvenanceError,
    capture_environment_identity,
    hash_source_tree,
    load_qualification_identity,
)

ROOT = Path(__file__).parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _signed(body: dict[str, object]) -> dict[str, object]:
    return body | {"digest": hashlib.sha256(_canonical(body)).hexdigest()}


def test_real_source_and_environment_identities_are_stable_and_secret_free() -> None:
    first = hash_source_tree(ROOT)
    second = hash_source_tree(ROOT)
    assert first.digest == second.digest
    assert first.files == second.files
    assert "kuairand-starter-kit" not in repr(first.manifest())
    assert all(not item.path.startswith((".data/", "runs/")) for item in first.files)
    assert {
        "candidate_templates/lambdarank/candidate.py",
        "candidate_templates/lambdarank/config.json",
        "candidate_templates/lambdarank/README.md",
    } <= {item.path for item in first.files}
    environment = capture_environment_identity(ROOT)
    assert environment.digest == capture_environment_identity(ROOT).digest
    rendered = json.dumps(environment.manifest(), sort_keys=True)
    assert "OPENAI_API_KEY" not in rendered
    assert "HOME" not in rendered


def test_source_tree_rejects_symlinked_executable_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "candidate_templates").mkdir()
    for name in (".python-version", "pyproject.toml", "uv.lock"):
        (tmp_path / name).write_text("locked", encoding="utf-8")
    target = tmp_path / "outside.py"
    target.write_text("pass\n", encoding="utf-8")
    (tmp_path / "src" / "candidate.py").symlink_to(target)
    with pytest.raises(ProvenanceError, match="symlink"):
        hash_source_tree(tmp_path)


def test_qualification_identity_rejects_logical_tampering(tmp_path: Path) -> None:
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    fallback = _signed(
        {
            "source_model_tree_digest": "1" * 64,
            "checkpoint_digest": "2" * 64,
            "fallback_model_tree_digest": "3" * 64,
            "config_digest": "4" * 64,
            "encoding_digest": "5" * 64,
            "validation_metrics": {"primary": 0.6},
        }
    )
    body: dict[str, object] = {
        "status": "baseline_reproduced",
        "benchmark_digest": "6" * 64,
        "fallback": fallback,
        "fm": {"runs": [{"seed": 4, "starter_manifest_digest": "7" * 64}]},
    }
    manifest = _signed(body)
    manifest["status"] = "tampered"
    (qualification / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(ProvenanceError, match="logical digest mismatch"):
        load_qualification_identity(qualification)
