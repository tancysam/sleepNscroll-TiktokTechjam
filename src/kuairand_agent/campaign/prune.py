"""Reclaim regenerable bulk from a finished campaign without touching its evidence.

A completed campaign occupies roughly 1.2-4 GB, of which the durable deliverable -- the published
bundle -- is about 17 MB. The remainder is a content-addressed artifact store and a causal-feature
cache, both of which exist to make the *run* fast and restartable rather than to make its result
verifiable. Replay operates on the bundle (`kuairand-agent replay --bundle`), never on the run
directory, so a finalized run's bundle is self-contained.

This module removes only those two directories, and only under conditions it verifies first. It
never removes a bundle, a campaign store, a provider-attempt journal, a scientific record, a
generated-source tree, or either project ledger -- those are the evidence the results documents
rest on, and the project's ground rule is that no claim appears without a retained artifact behind
it.
"""

from __future__ import annotations

import shutil
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Regenerable, per run directory. `artifacts` is only prunable once a bundle exists, because until
# then it is the sole record of what the campaign produced; `feature-cache` is a pure content
# addressed cache and is regenerable at any point.
_ALWAYS_PRUNABLE: Final = ("production/feature-cache",)
_PRUNABLE_ONCE_FINALIZED: Final = ("artifacts",)

# Required before anything is deleted, as proof this is a real campaign run whose record is
# intact. Deliberately minimal: a scripted campaign has no provider-attempt journal and a campaign
# that never admitted a candidate has no scientific records, so demanding those would refuse to
# prune legitimate runs forever. Every other path is protected simply by never being a target.
_REQUIRED: Final = ("campaign.sqlite3",)
_REQUIRED_WHEN_FINALIZED: Final = ("final/report.md",)


class PruneError(RuntimeError):
    """Raised when a run directory cannot be pruned safely."""


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """What pruning one run directory would remove, and why."""

    run_dir: Path
    finalized: bool
    targets: tuple[Path, ...]
    reclaimed_bytes: int

    def to_wire(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "finalized": self.finalized,
            "targets": [str(path) for path in self.targets],
            "reclaimed_bytes": self.reclaimed_bytes,
        }


def _directory_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        info = child.lstat()
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def _safe_child(run_dir: Path, relative: str) -> Path | None:
    """Resolve one prune target, refusing anything that escapes the run directory."""

    candidate = run_dir / relative
    if candidate.is_symlink() or not candidate.is_dir():
        return None
    resolved = candidate.resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise PruneError(f"prune target escapes the run directory: {relative}")
    return candidate


def plan_prune(run_dir: Path | str) -> PrunePlan:
    """Describe what pruning would remove, without removing anything."""

    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        raise PruneError("run directory must be a real directory")
    if not (root / "production").is_dir():
        raise PruneError("run directory does not look like a campaign run")

    finalized = (root / "final" / "report.md").is_file()
    relatives = list(_ALWAYS_PRUNABLE)
    if finalized:
        relatives.extend(_PRUNABLE_ONCE_FINALIZED)

    targets: list[Path] = []
    reclaimed = 0
    for relative in relatives:
        target = _safe_child(root, relative)
        if target is None:
            continue
        targets.append(target)
        reclaimed += _directory_bytes(target)
    return PrunePlan(
        run_dir=root,
        finalized=finalized,
        targets=tuple(targets),
        reclaimed_bytes=reclaimed,
    )


def archive_generated_source(run_dir: Path | str, runs_root: Path | str | None = None) -> int:
    """Mirror a run's generated candidate source into a durable archive; return files copied.

    ``prune`` never removes ``generated-source``, but that only protects it from this tool. A
    whole-directory delete -- reclaiming disk by hand, say -- takes the code with it, and the
    ledger keeps configurations rather than implementations. That happened: 129 candidate trees
    existed across the run directories and two survived, so the recipes behind every earlier
    measurement had to be reconstructed from config alone.

    The cost of preventing it is nil. Generated source is 244 KB against a 4.4 GB run, roughly
    five thousandths of one percent; the bulk is regenerable feature cache. Mirroring is
    idempotent and never overwrites, so an archived tree stays as it was first written.
    """

    root = Path(run_dir)
    source = root / "production" / "generated-source"
    if source.is_symlink() or not source.is_dir():
        return 0
    archive = (Path(runs_root) if runs_root is not None else root.parent) / "archive"
    destination = archive / "generated-source" / root.name
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        target = destination / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def prune_run(run_dir: Path | str) -> PrunePlan:
    """Remove regenerable bulk from one run directory and return what was removed."""

    plan = plan_prune(run_dir)
    root = plan.run_dir
    # Archive before deleting anything, so the irreplaceable half of a run outlives the run.
    archive_generated_source(root)
    # Verified immediately before deletion rather than only at plan time: the record must still be
    # present at the moment anything is removed.
    required = list(_REQUIRED)
    if plan.finalized:
        required.extend(_REQUIRED_WHEN_FINALIZED)
    for relative in required:
        if not (root / relative).exists():
            raise PruneError(f"refusing to prune a run missing protected evidence: {relative}")
    for target in plan.targets:
        shutil.rmtree(target)
    return plan


def iter_run_dirs(runs_root: Path | str) -> Iterator[Path]:
    """Yield campaign run directories under a runs root, in stable order."""

    root = Path(runs_root)
    if root.is_symlink() or not root.is_dir():
        raise PruneError("runs root must be a real directory")
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.is_symlink() and (child / "production").is_dir():
            yield child
