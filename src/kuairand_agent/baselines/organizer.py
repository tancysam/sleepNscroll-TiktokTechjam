"""Collision-safe access to the immutable organizer baseline modules.

The organizer starter uses conventional top-level imports (``data`` and ``evaluate``).  This
loader satisfies those imports only while executing the verified ``baseline.py`` under a private
module name.  It never adds the starter directory to ``sys.path``, never calls ``data.load``, and
restores all interpreter-global state before returning.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, cast

from kuairand_agent.contract import OrganizerIntegrityError, verify_starter_kit

_IMPORT_LOCK: Final = threading.RLock()
_TEMPORARY_ALIASES: Final = ("data", "evaluate")
_REQUIRED_CALLABLES: Final = {
    "data": ("load", "encode"),
    "evaluate": ("evaluate",),
    "baseline": ("FM", "run_random", "run_pop", "run_fm"),
}
_MISSING: Final = object()


class OrganizerLoadError(RuntimeError):
    """Raised when verified organizer source cannot be loaded under the expected contract."""


@dataclass(frozen=True, slots=True)
class OrganizerModules:
    """Verified private organizer modules and their aggregate source identity."""

    root: Path
    manifest_sha256: str
    data: ModuleType
    evaluate: ModuleType
    baseline: ModuleType

    def __post_init__(self) -> None:
        for role, module in (
            ("data", self.data),
            ("evaluate", self.evaluate),
            ("baseline", self.baseline),
        ):
            if not module.__name__.startswith("_kuairand_pinned_"):
                raise OrganizerLoadError(f"organizer {role} module does not have a private name")
            expected_path = (self.root / f"{role}.py").resolve()
            module_path = Path(str(getattr(module, "__file__", ""))).resolve()
            if module_path != expected_path:
                raise OrganizerLoadError(
                    f"organizer {role} module came from {module_path}, expected {expected_path}"
                )
            for name in _REQUIRED_CALLABLES[role]:
                if not callable(getattr(module, name, None)):
                    raise OrganizerLoadError(
                        f"organizer {role} module does not define callable {name}"
                    )


def _private_name(role: str) -> str:
    return f"_kuairand_pinned_{role}_{uuid.uuid4().hex}"


def _load_source(path: Path, role: str) -> ModuleType:
    module_name = _private_name(role)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OrganizerLoadError(f"cannot create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise OrganizerLoadError(f"failed to execute verified organizer {role}.py") from exc
    return module


def _restore_alias(name: str, previous: object) -> None:
    if previous is _MISSING:
        sys.modules.pop(name, None)
    else:
        # Preserve the exact prior interpreter state.  Although the annotated sys.modules
        # contract says ModuleType, import machinery can transiently store None sentinels.
        sys.modules[name] = cast(ModuleType, previous)


def load_verified_organizer(starter_dir: str | Path) -> OrganizerModules:
    """Load pinned organizer modules without leaking their conventional import aliases.

    Verification occurs before and after source execution.  ``baseline.py`` binds its imported
    ``encode`` and ``evaluate`` callables to the verified private module objects, while any
    pre-existing modules named ``data`` or ``evaluate`` are restored by identity.
    """

    with _IMPORT_LOCK:
        verification = verify_starter_kit(starter_dir)
        root = verification.root
        paths = {role: (root / f"{role}.py").resolve(strict=True) for role in _REQUIRED_CALLABLES}
        if any(path.parent != root for path in paths.values()):
            raise OrganizerIntegrityError("organizer module resolved outside the starter directory")

        previous_bytecode = sys.dont_write_bytecode
        previous_aliases = {name: sys.modules.get(name, _MISSING) for name in _TEMPORARY_ALIASES}
        sys.dont_write_bytecode = True
        try:
            data_module = _load_source(paths["data"], "data")
            evaluate_module = _load_source(paths["evaluate"], "evaluate")
            sys.modules["data"] = data_module
            sys.modules["evaluate"] = evaluate_module
            baseline_module = _load_source(paths["baseline"], "baseline")
        finally:
            for alias in _TEMPORARY_ALIASES:
                _restore_alias(alias, previous_aliases[alias])
            sys.dont_write_bytecode = previous_bytecode

        verification_after = verify_starter_kit(root)
        if verification_after.manifest_sha256 != verification.manifest_sha256:
            raise OrganizerIntegrityError(
                "organizer starter changed while loading baseline modules"
            )

        loaded = OrganizerModules(
            root=root,
            manifest_sha256=verification.manifest_sha256,
            data=data_module,
            evaluate=evaluate_module,
            baseline=baseline_module,
        )
        if loaded.baseline.encode is not loaded.data.encode:
            raise OrganizerLoadError("baseline.py did not bind the verified organizer encode()")
        if loaded.baseline.evaluate is not loaded.evaluate.evaluate:
            raise OrganizerLoadError("baseline.py did not bind the verified organizer evaluate()")
        return loaded


__all__ = ["OrganizerLoadError", "OrganizerModules", "load_verified_organizer"]
