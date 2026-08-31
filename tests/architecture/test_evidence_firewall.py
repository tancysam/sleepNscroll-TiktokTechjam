from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from kuairand_agent.data.canonical import CanonicalFinalSplit
from kuairand_agent.evaluation.protected import ProtectedLabels, ProtectedResult

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src" / "kuairand_agent"
_FIREWALLED_MODULES = (
    "kuairand_agent.evaluation.protected",
    "kuairand_agent.scoring.protected",
)


def _imports(path: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.append((node.module or "", tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.Import):
            found.extend((alias.name, ()) for alias in node.names)
    return tuple(found)


def _class_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def test_protected_label_and_result_classes_are_owned_only_by_evaluation() -> None:
    evaluation_path = _SOURCE_ROOT / "evaluation" / "protected.py"
    canonical_path = _SOURCE_ROOT / "data" / "canonical.py"

    assert {"ProtectedLabels", "ProtectedResult"} <= _class_definitions(evaluation_path)
    assert "ProtectedTargets" not in _class_definitions(canonical_path)
    assert ProtectedLabels.__module__ == "kuairand_agent.evaluation.protected"
    assert ProtectedResult.__module__ == "kuairand_agent.evaluation.protected"


def test_proposal_training_and_search_cannot_import_protected_evidence_types() -> None:
    violations: list[str] = []
    for package_name in ("proposal", "training", "search"):
        package = _SOURCE_ROOT / package_name
        if not package.exists():
            continue
        for path in package.rglob("*.py"):
            for module, names in _imports(path):
                if module in _FIREWALLED_MODULES:
                    violations.append(f"{path.relative_to(_REPOSITORY_ROOT)} imports {module}")
                if module == "kuairand_agent.data.canonical" and {
                    "ProtectedLabels",
                    "ProtectedTargets",
                } & set(names):
                    violations.append(
                        f"{path.relative_to(_REPOSITORY_ROOT)} imports protected labels from data"
                    )

    assert violations == []


def test_candidate_implementation_modules_do_not_import_protected_result_modules() -> None:
    for relative_path in ("candidates/bootstrap.py", "candidates/neural.py"):
        path = _SOURCE_ROOT / relative_path
        imported_modules = {module for module, _names in _imports(path)}
        assert imported_modules.isdisjoint(_FIREWALLED_MODULES)


def test_final_split_runtime_and_ast_schema_have_no_target_member() -> None:
    runtime_fields = {field.name for field in dataclasses.fields(CanonicalFinalSplit)}
    assert "targets" not in runtime_fields

    canonical_path = _SOURCE_ROOT / "data" / "canonical.py"
    tree = ast.parse(canonical_path.read_text(encoding="utf-8"), filename=str(canonical_path))
    final_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CanonicalFinalSplit"
    )
    annotated_names = {
        node.target.id
        for node in final_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "targets" not in annotated_names
