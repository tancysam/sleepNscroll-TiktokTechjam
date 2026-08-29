"""Disposable generated-package materialization and controller-computed source diffs.

The research model returns complete UTF-8 file contents.  It never supplies patches, shell
commands, destination paths, or filesystem references.  This module validates the whole mapping
before creating a new child directory, preserves the in-memory parent, and computes every logical
identity independently of temporary paths and file metadata.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import os
import re
import shutil
import stat
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kuairand_agent.research.schemas import (
    GeneratedFile,
    GeneratedPackage,
    ParentSnapshot,
    ParentSourceFile,
    SchemaValidationError,
    canonical_digest,
    parse_json_object,
)
from kuairand_agent.research.source_policy import (
    DEFAULT_CANDIDATE_SOURCE_POLICY,
    CandidateManifestPolicyError,
)
ALLOWED_SUFFIXES: Final = frozenset({".py", ".json", ".md"})
_FORBIDDEN_BASENAMES: Final = frozenset(
    {
        "data.py",
        "evaluate.py",
        "baseline.py",
        "submit.py",
        "sitecustomize.py",
        "usercustomize.py",
        "conftest.py",
        "pyproject.toml",
    }
)
_FORBIDDEN_IMPORT_ROOTS: Final = frozenset(
    {
        "kuairand_agent",
        "subprocess",
        "socket",
        "urllib",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "os",
        "sys",
        "shutil",
        "glob",
        "tempfile",
        "ctypes",
        "importlib",
        "multiprocessing",
        "pickle",
        "marshal",
        "builtins",
        "webbrowser",
    }
)
_FORBIDDEN_CALLS: Final = frozenset({"eval", "exec", "compile", "__import__", "breakpoint"})
_FROZEN_PROTOCOL_SYMBOLS: Final = frozenset(
    {
        "SCHEMA_VERSION",
        "SCORES_DTYPE",
        "MAX_JSON_BYTES",
        "CONFIG_KEYS",
        "TRAIN_KEYS",
        "PREDICT_KEYS",
        "CandidateInputError",
        "_sha256",
        "_read_json",
        "_require_exact_keys",
        "_require_digest",
        "_write_json",
        "_write_scores",
        "_read_capability",
        "main",
    }
)
_PORTABLE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\Z")


class CandidateMaterializationError(ValueError):
    """A generated package cannot be safely represented as a disposable source child."""


class CandidateStaticError(ValueError):
    """A materialized child fails syntax, import, or executable-materiality policy."""


@dataclass(frozen=True, slots=True)
class MaterializedCandidate:
    """Path plus logical identities for one immutable disposable child source tree."""

    destination: Path
    parent_digest: str
    package_digest: str
    material_symbols: tuple[str, ...]
    files: tuple[ParentSourceFile, ...]
    changed_paths: tuple[str, ...]
    unified_diff: str
    source_digest: str
    diff_digest: str

    def file(self, path: str) -> ParentSourceFile:
        for value in self.files:
            if value.path == path:
                return value
        raise CandidateMaterializationError(f"materialized child does not contain {path!r}")


@dataclass(frozen=True, slots=True)
class StaticGateResult:
    python_files: tuple[str, ...]
    json_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MaterialChangeEvidence:
    changed_symbols: tuple[str, ...]
    reachable_python_files: tuple[str, ...]


def _validate_path(value: str, *, generated: bool) -> PurePosixPath:
    try:
        return DEFAULT_CANDIDATE_SOURCE_POLICY.validate_path(value, generated=generated)
    except CandidateManifestPolicyError as exc:
        raise CandidateMaterializationError(str(exc)) from exc


def _validate_package(parent: ParentSnapshot, package: GeneratedPackage) -> None:
    if not isinstance(parent, ParentSnapshot):
        raise CandidateMaterializationError("parent must be a ParentSnapshot")
    if not isinstance(package, GeneratedPackage):
        raise CandidateMaterializationError("package must be a GeneratedPackage")
    policy = DEFAULT_CANDIDATE_SOURCE_POLICY
    try:
        policy.validate_manifest(
            (value.path for value in package.files),
            parent_paths=(value.path for value in parent.files),
        )
    except CandidateManifestPolicyError as exc:
        raise CandidateMaterializationError(str(exc)) from exc
    total = 0
    for generated_file in package.files:
        size = len(generated_file.content.encode("utf-8"))
        if size > policy.max_generated_file_bytes:
            raise CandidateMaterializationError(
                f"generated file {generated_file.path!r} exceeds the byte limit"
            )
        total += size
    if total > policy.max_generated_total_bytes:
        raise CandidateMaterializationError("generated package exceeds the total byte limit")


def _tree_digest(files: Iterable[ParentSourceFile]) -> str:
    return canonical_digest(
        {
            "schema_version": 1,
            "files": [
                {"path": value.path, "sha256": value.sha256}
                for value in sorted(files, key=lambda item: item.path)
            ],
        }
    )


def _unified_diff(
    parent_files: Mapping[str, str], child_files: Mapping[str, str]
) -> tuple[tuple[str, ...], str]:
    changed = tuple(
        path
        for path in sorted(set(parent_files) | set(child_files))
        if parent_files.get(path) != child_files.get(path)
    )
    fragments: list[str] = []
    for path in changed:
        fragments.extend(
            difflib.unified_diff(
                parent_files.get(path, "").splitlines(keepends=True),
                child_files.get(path, "").splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="\n",
            )
        )
    return changed, "".join(fragments)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_read_only(root: Path) -> None:
    for value in sorted(root.rglob("*"), reverse=True):
        metadata = value.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            value.chmod(0o500)
        elif stat.S_ISREG(metadata.st_mode):
            value.chmod(0o400)
        else:  # defensive; this module creates only directories and regular files
            raise CandidateMaterializationError("candidate tree contains a special file")
    root.chmod(0o500)


def _cleanup(root: Path) -> None:
    if not root.exists():
        return
    for value in root.rglob("*"):
        try:
            if value.is_dir():
                value.chmod(0o700)
            else:
                value.chmod(0o600)
        except OSError:
            pass
    with suppress(OSError):
        root.chmod(0o700)
    shutil.rmtree(root, ignore_errors=True)


def materialize_candidate(
    parent: ParentSnapshot,
    package: GeneratedPackage,
    destination: Path | str,
) -> MaterializedCandidate:
    """Create one new read-only child tree after validating the complete response mapping."""


    _validate_package(parent, package)
    destination_path = Path(destination)
    if destination_path.exists() or destination_path.is_symlink():
        raise CandidateMaterializationError(
            f"candidate destination already exists: {destination_path}"
        )
    parent_directory = destination_path.parent
    try:
        metadata = parent_directory.lstat()
    except OSError as exc:
        raise CandidateMaterializationError("candidate destination parent does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CandidateMaterializationError("candidate destination parent must be a real directory")

    parent_content = {value.path: value.content for value in parent.files}
    child_content = dict(parent_content)
    child_content.update({value.path: value.content for value in package.files})
    changed_paths, unified = _unified_diff(parent_content, child_content)
    child_files = tuple(
        ParentSourceFile.create(path, content) for path, content in sorted(child_content.items())
    )
    source_digest = _tree_digest(child_files)
    diff_digest = hashlib.sha256(unified.encode("utf-8")).hexdigest()

    try:
        destination_path.mkdir(mode=0o700, exist_ok=False)
        for value in child_files:
            relative = _validate_path(value.path, generated=False)
            target = destination_path.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _write_exclusive(target, value.content.encode("utf-8"))
        _make_read_only(destination_path)
    except (OSError, CandidateMaterializationError) as exc:
        _cleanup(destination_path)
        if isinstance(exc, CandidateMaterializationError):
            raise
        raise CandidateMaterializationError(f"candidate materialization failed: {exc}") from exc

    return MaterializedCandidate(
        destination=destination_path,
        parent_digest=parent.digest,
        package_digest=package.digest,
        material_symbols=package.material_symbols,
        files=child_files,
        changed_paths=changed_paths,
        unified_diff=unified,
        source_digest=source_digest,
        diff_digest=diff_digest,
    )


def _verify_materialized_bytes(candidate: MaterializedCandidate) -> None:
    try:
        root_metadata = candidate.destination.lstat()
    except OSError as exc:
        raise CandidateStaticError("materialized candidate directory is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise CandidateStaticError("materialized candidate root must be a real directory")
    expected = {value.path: value for value in candidate.files}
    observed: set[str] = set()
    for path in candidate.destination.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        relative = path.relative_to(candidate.destination).as_posix()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CandidateStaticError(f"candidate tree contains a nonregular file: {relative}")
        try:
            reference = expected[relative]
        except KeyError as exc:
            raise CandidateStaticError(
                f"candidate tree contains an undeclared file: {relative}"
            ) from exc
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise CandidateStaticError(f"candidate file cannot be read: {relative}") from exc
        if hashlib.sha256(payload).hexdigest() != reference.sha256:
            raise CandidateStaticError(f"candidate file changed after materialization: {relative}")
        observed.add(relative)
    missing = set(expected) - observed
    if missing:
        raise CandidateStaticError(
            f"candidate tree is missing declared file(s): {sorted(missing)!r}"
        )


def _import_root(name: str | None) -> str:
    if not name:
        return ""
    return name.split(".", 1)[0]


def _validate_python(path: str, content: str) -> ast.Module:
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        raise CandidateStaticError(f"invalid Python in {path!r} at {location}: {exc.msg}") from exc
    for node in ast.walk(tree):
        imported: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            imported = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            imported = (node.module or "",)
        for name in imported:
            if _import_root(name) in DEFAULT_CANDIDATE_SOURCE_POLICY.forbidden_import_roots:
                raise CandidateStaticError(f"forbidden import {name!r} in {path!r}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in DEFAULT_CANDIDATE_SOURCE_POLICY.forbidden_calls
        ):
            raise CandidateStaticError(f"forbidden call {node.func.id!r} in {path!r}")
    return tree


def validate_candidate_static(candidate: MaterializedCandidate) -> StaticGateResult:
    """Run syntax, JSON, import, and immutable-tree checks after materialization."""

    _verify_materialized_bytes(candidate)
    python_files: list[str] = []
    json_files: list[str] = []
    for value in candidate.files:
        if value.path.endswith(".py"):
            _validate_python(value.path, value.content)
            python_files.append(value.path)
        elif value.path.endswith(".json"):
            try:
                parse_json_object(value.content)
            except SchemaValidationError as exc:
                raise CandidateStaticError(f"invalid JSON in {value.path!r}: {exc}") from exc
            json_files.append(value.path)
    entrypoint = DEFAULT_CANDIDATE_SOURCE_POLICY.final_entrypoint
    if entrypoint not in python_files:
        raise CandidateStaticError(f"candidate tree must contain the {entrypoint} entry point")
    return StaticGateResult(tuple(python_files), tuple(json_files))


class _DocstringStripper(ast.NodeTransformer):
    def _strip(
        self, node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        self._strip(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        self._strip(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        self._strip(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        self._strip(node)
        return node


def _normalized_tree(path: str, content: str) -> ast.Module:
    tree = _validate_python(path, content)
    stripped = _DocstringStripper().visit(tree)
    assert isinstance(stripped, ast.Module)
    return ast.fix_missing_locations(stripped)


def _top_level_symbols(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, annotate_fields=True, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = ast.dump(node, annotate_fields=True, include_attributes=False)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                result[node.target.id] = ast.dump(node, annotate_fields=True, include_attributes=False)
    return result


def _local_module_candidates(name: str) -> tuple[str, str]:
    base = name.replace(".", "/")
    return f"{base}.py", f"{base}/__init__.py"


def _reachable_python_files(files: Mapping[str, str]) -> tuple[str, ...]:
    if "candidate.py" not in files:
        return ()
    pending = ["candidate.py"]
    reached: set[str] = set()
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        reached.add(path)
        tree = _normalized_tree(path, files[path])
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = (node.module,)
            for module in modules:
                for candidate in _local_module_candidates(module):
                    if candidate in files and candidate not in reached:
                        pending.append(candidate)
    return tuple(sorted(reached))


def _extract_symbol_source(
    source: str, symbol_name: str, tree: ast.Module,
) -> str | None:
    """Extract the exact source lines for a top-level symbol from source text."""
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        name: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol_name:
                    name = symbol_name
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name == symbol_name and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno is not None else start + 1
            return "".join(lines[start:end])
    return None


def _restore_protocol_constants(
    parent: ParentSnapshot, candidate_content: str,
) -> str:
    """Restore frozen protocol symbols in candidate.py from the parent if the model changed them.

    Returns the (possibly corrected) source text for candidate.py.
    """
    parent_file = None
    for f in parent.files:
        if f.path == "candidate.py":
            parent_file = f
            break
    if parent_file is None:
        return candidate_content

    try:
        parent_tree = ast.parse(parent_file.content, filename="candidate.py")
        child_tree = ast.parse(candidate_content, filename="candidate.py")
    except SyntaxError:
        return candidate_content

    parent_symbols = _top_level_symbols(parent_tree)
    child_symbols = _top_level_symbols(child_tree)

    result = candidate_content
    for symbol_name in _FROZEN_PROTOCOL_SYMBOLS:
        if symbol_name not in parent_symbols:
            continue
        parent_dump = parent_symbols.get(symbol_name)
        child_dump = child_symbols.get(symbol_name)
        if parent_dump == child_dump:
            continue
        # The model changed a frozen symbol — restore the parent's version
        parent_source = _extract_symbol_source(parent_file.content, symbol_name, parent_tree)
        if parent_source is None:
            continue
        if child_dump is not None:
            # Symbol exists in child but was modified — replace it
            child_source = _extract_symbol_source(result, symbol_name, ast.parse(result))
            if child_source is not None:
                result = result.replace(child_source, parent_source, 1)
        else:
            # Symbol was deleted by the model — re-insert it before the first function/class
            lines = result.splitlines(keepends=True)
            insert_line = len(lines)
            current_tree = ast.parse(result)
            for node in current_tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    insert_line = node.lineno - 1
                    break
            lines.insert(insert_line, parent_source + "\n")
            result = "".join(lines)

    return result


def sanitize_generated_package(
    parent: ParentSnapshot,
    package: GeneratedPackage,
) -> GeneratedPackage:
    """Apply deterministic guardrails to fix common LLM code-generation mistakes.

    1. Drop files with forbidden basenames (e.g. baseline.py, data.py).
    2. Restore frozen protocol constants in candidate.py from the parent.
    3. Recompute material_symbols based on actual AST differences.
    """
    # Step 1: Drop forbidden-basename files
    cleaned_files: list[GeneratedFile] = []
    for f in package.files:
        p = PurePosixPath(f.path)
        if p.name in _FORBIDDEN_BASENAMES:
            continue
        if p.suffix not in ALLOWED_SUFFIXES:
            continue
        cleaned_files.append(f)

    # Step 2: Restore protocol constants in candidate.py
    final_files: list[GeneratedFile] = []
    for f in cleaned_files:
        if f.path == "candidate.py":
            restored = _restore_protocol_constants(parent, f.content)
            final_files.append(GeneratedFile(path=f.path, content=restored))
        else:
            final_files.append(f)

    if not final_files:
        raise CandidateMaterializationError(
            "sanitization removed all generated files — candidate is empty"
        )

    # Step 3: Recompute material_symbols from actual AST diff
    parent_files = {v.path: v.content for v in parent.files if v.path.endswith(".py")}
    child_files = dict(parent_files)
    child_files.update({f.path: f.content for f in final_files if f.path.endswith(".py")})
    actual_changed: list[str] = []
    for path in sorted(child_files):
        if not path.endswith(".py"):
            continue
        try:
            child_tree = _normalized_tree(path, child_files[path])
            child_syms = _top_level_symbols(child_tree)
        except CandidateStaticError:
            continue
        if path in parent_files:
            try:
                parent_tree = _normalized_tree(path, parent_files[path])
                parent_syms = _top_level_symbols(parent_tree)
            except CandidateStaticError:
                parent_syms = {}
        else:
            parent_syms = {}
        for name, dump in child_syms.items():
            if name.startswith("_"):
                continue  # schemas.py forbids material_symbols starting with _
            if parent_syms.get(name) != dump and name not in actual_changed:
                actual_changed.append(name)

    if not actual_changed:
        raise SchemaValidationError(
            "sanitized candidate is identical to parent — returning unmodified source is forbidden"
        )

    return GeneratedPackage(
        schema_version=package.schema_version,
        request_id=package.request_id,
        response_id=package.response_id,
        files=tuple(final_files),
        material_change_summary=package.material_change_summary,
        material_symbols=tuple(sorted(actual_changed)),
    )


def require_material_executable_change(
    parent: ParentSnapshot,
    candidate: MaterializedCandidate,
) -> MaterialChangeEvidence:
    """Require a declared changed symbol in executable source reachable from ``candidate.py``."""

    validate_candidate_static(candidate)
    parent_files = {
        value.path: value.content for value in parent.files if value.path.endswith(".py")
    }
    child_files = {
        value.path: value.content for value in candidate.files if value.path.endswith(".py")
    }
    reachable = _reachable_python_files(child_files)
    changed_by_name: dict[str, str] = {}
    for path in reachable:
        child_tree = _normalized_tree(path, child_files[path])
        child_symbols = _top_level_symbols(child_tree)
        parent_symbols = (
            _top_level_symbols(_normalized_tree(path, parent_files[path]))
            if path in parent_files
            else {}
        )
        for name, subtree in child_symbols.items():
            if parent_symbols.get(name) != subtree:
                qualified = f"{path}:{name}"
                if name in changed_by_name:
                    raise CandidateStaticError(
                        f"declared material symbol {name!r} is ambiguous across reachable files"
                    )
                changed_by_name[name] = qualified
    missing = sorted(set(candidate.material_symbols) - set(changed_by_name))
    if missing:
        raise CandidateStaticError(
            f"declared material symbol(s) did not change material executable-source: {missing!r}"
        )
    changed_symbols = tuple(changed_by_name[name] for name in candidate.material_symbols)
    if not changed_symbols:
        raise CandidateStaticError(
            "candidate does not contain a material executable-source symbol change"
        )
    return MaterialChangeEvidence(tuple(sorted(changed_symbols)), reachable)


def snapshot_materialized_candidate(
    candidate: MaterializedCandidate, *, candidate_id: str
) -> ParentSnapshot:
    """Promote a verified child source record to an immutable parent snapshot for repair/replay."""

    _verify_materialized_bytes(candidate)
    return ParentSnapshot(candidate_id=candidate_id, files=candidate.files)
