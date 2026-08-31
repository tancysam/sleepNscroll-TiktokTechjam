"""Deterministic projection of frozen campaign receipts into the final evidence bundle.

The legacy finalizer remains the adapter for model replay and organizer validation.  This module
owns the final, flat evidence projection required by the current campaign contract.  It accepts
only content-addressed file receipts, materializes the projection twice, proves byte-for-byte
regeneration, and exclusively publishes the first projection.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from kuairand_agent.domain.identity import BundleId, PredictionId
from kuairand_agent.finalization.replay import CleanReplayResult
from kuairand_agent.finalization.replay_grades import (
    BundleRegenerationEvidence,
    ReplayGradeReceipt,
    derive_bundle_exact_grade,
)
from kuairand_agent.finalization.submission_bundle import (
    _fsync_directory as _legacy_fsync_directory,
)
from kuairand_agent.finalization.submission_bundle import (
    _rename_exclusive as _legacy_publish_exclusive,
)

BUNDLE_PROJECTION_SCHEMA_VERSION: Final = 2
_FROZEN_FILE_RECEIPT_SCHEMA_VERSION: Final = 1
_RECEIPT_DOMAIN: Final = b"kuairand-frozen-bundle-file-v1\0"
_BUNDLE_MANIFEST_NAME: Final = "bundle-manifest.json"
_BUNDLE_DIGEST_NAME: Final = "bundle.sha256"
_COPY_CHUNK_BYTES: Final = 1024 * 1024
_MAX_FILE_BYTES: Final = 8 * 1024 * 1024 * 1024
_MAX_BUNDLE_BYTES: Final = 16 * 1024 * 1024 * 1024


class BundleProjectionError(RuntimeError):
    """Raised when frozen evidence cannot produce one exact closed projection."""


class BundleProjectionCancelledError(BundleProjectionError):
    """Raised when cancellation occurs before exclusive publication."""


class EvidenceRole(StrEnum):
    """Exact evidence paths required from every accepted rehearsal."""

    CONTRACT_MANIFEST = "contract-manifest.json"
    CAMPAIGN_MANIFEST = "campaign-manifest.json"
    CAMPAIGN_STATE_SNAPSHOT = "campaign-state-snapshot.sqlite3"
    EVENT_EXPORT = "event-export.jsonl"
    SELECTION_EVIDENCE = "selection-evidence.json"
    SCIENTIFIC_DECISION = "scientific-decision.json"
    SUBMISSION_DECISION = "submission-decision.json"
    REPLAY_RECEIPT = "replay-receipt.json"
    RESOURCE_RECEIPTS = "resource-receipts.jsonl"
    PROTECTED_QUERY_ACCOUNTING = "protected-query-accounting.json"
    PROVIDER_ACCOUNTING = "provider-accounting.json"
    FAILURE_SUMMARY = "failure-summary.json"
    SUBMISSION = "submission.csv"
    REPORT = "report.md"


REQUIRED_EVIDENCE_ROLES: Final = tuple(EvidenceRole)
REQUIRED_BUNDLE_PATHS: Final = (
    *(role.value for role in REQUIRED_EVIDENCE_ROLES),
    _BUNDLE_MANIFEST_NAME,
    _BUNDLE_DIGEST_NAME,
)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise BundleProjectionError("bundle projection must be finite canonical JSON") from exc


def _digest(value: object, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BundleProjectionError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, location: str) -> str:
    candidate = value if type(value) is str else getattr(value, "value", None)
    return _digest(candidate, location)


def _role(value: EvidenceRole | str) -> EvidenceRole:
    try:
        return value if isinstance(value, EvidenceRole) else EvidenceRole(value)
    except (TypeError, ValueError) as exc:
        raise BundleProjectionError("evidence role is not part of the required layout") from exc


def _check_cancellation(cancel_event: threading.Event | None, stage: str) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise BundleProjectionCancelledError(f"bundle projection cancelled before {stage}")


def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise BundleProjectionError(f"evidence source is unavailable: {path}") from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise BundleProjectionError(f"evidence source must be a regular non-symlink file: {path}")
    if not 0 < initial.st_size <= _MAX_FILE_BYTES:
        raise BundleProjectionError(f"evidence source size is outside the accepted bound: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BundleProjectionError(f"evidence source could not be opened safely: {path}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns):
        os.close(descriptor)
        raise BundleProjectionError(f"evidence source changed while it was opened: {path}")
    return descriptor, opened


def _read_digest(path: Path) -> tuple[str, int]:
    descriptor, opened = _open_regular(path)
    digest = hashlib.sha256()
    observed = 0
    try:
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            observed += len(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if observed != opened.st_size or (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
        raise BundleProjectionError(f"evidence source changed while it was hashed: {path}")
    return digest.hexdigest(), observed


@dataclass(frozen=True, slots=True, init=False)
class FrozenFileReceipt:
    """Verified content address for one required evidence file.

    ``source`` is transport only and is intentionally excluded from ``receipt_id`` and the bundle
    manifest.  Machine-specific paths therefore cannot perturb bundle identity.
    """

    role: EvidenceRole
    source: Path
    sha256: str
    size_bytes: int
    receipt_id: str

    @classmethod
    def capture(cls, role: EvidenceRole | str, source: str | Path) -> FrozenFileReceipt:
        normalized_role = _role(role)
        path = Path(source)
        sha256, size_bytes = _read_digest(path)
        body = {
            "schema_version": _FROZEN_FILE_RECEIPT_SCHEMA_VERSION,
            "role": normalized_role.value,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        result = object.__new__(cls)
        object.__setattr__(result, "role", normalized_role)
        object.__setattr__(result, "source", path)
        object.__setattr__(result, "sha256", sha256)
        object.__setattr__(result, "size_bytes", size_bytes)
        object.__setattr__(
            result,
            "receipt_id",
            hashlib.sha256(_RECEIPT_DOMAIN + _canonical_json(body)).hexdigest(),
        )
        return result

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": _FROZEN_FILE_RECEIPT_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "role": self.role.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class TerminalProjectionBinding:
    """Path-free identity of the prepared terminal authority projection."""

    preparation_id: str
    projection_sha256: str
    campaign_revision: int
    last_event_seq: int
    schema_version: int = 1
    redaction_policy_version: int = 1

    def __post_init__(self) -> None:
        _digest(self.preparation_id, "preparation_id")
        _digest(self.projection_sha256, "projection_sha256")
        if type(self.campaign_revision) is not int or self.campaign_revision < 0:
            raise BundleProjectionError("campaign_revision must be a non-negative integer")
        if type(self.last_event_seq) is not int or self.last_event_seq <= 0:
            raise BundleProjectionError("last_event_seq must be a positive integer")
        if self.schema_version != 1:
            raise BundleProjectionError("terminal projection schema_version must be 1")
        if self.redaction_policy_version != 1:
            raise BundleProjectionError("terminal projection redaction_policy_version must be 1")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "preparation_id": self.preparation_id,
            "projection_sha256": self.projection_sha256,
            "source_revision": {
                "campaign_revision": self.campaign_revision,
                "last_event_seq": self.last_event_seq,
            },
            "redaction_policy_version": self.redaction_policy_version,
        }


def capture_clean_replay_receipts(
    replay: CleanReplayResult,
) -> tuple[FrozenFileReceipt, FrozenFileReceipt]:
    """Adapt the established replay result into frozen projection receipts."""

    if not isinstance(replay, CleanReplayResult):
        raise BundleProjectionError("replay must be CleanReplayResult")
    return (
        FrozenFileReceipt.capture(EvidenceRole.REPLAY_RECEIPT, replay.evidence_path),
        FrozenFileReceipt.capture(EvidenceRole.SUBMISSION, replay.final_submission),
    )


@dataclass(frozen=True, slots=True)
class BundleFinalizationRequest:
    """Identity and complete frozen inputs for one no-overwrite publication."""

    destination: Path
    contract_id: object
    campaign_id: object
    selected_prediction_id: object
    terminal_projection: TerminalProjectionBinding
    receipts: tuple[FrozenFileReceipt, ...]
    _contract_value: str = field(init=False, repr=False)
    _campaign_value: str = field(init=False, repr=False)
    _prediction_value: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.destination, Path):
            raise BundleProjectionError("destination must be pathlib.Path")
        object.__setattr__(self, "_contract_value", _identifier(self.contract_id, "contract_id"))
        object.__setattr__(self, "_campaign_value", _identifier(self.campaign_id, "campaign_id"))
        object.__setattr__(
            self,
            "_prediction_value",
            _identifier(self.selected_prediction_id, "selected_prediction_id"),
        )
        if not isinstance(self.terminal_projection, TerminalProjectionBinding):
            raise BundleProjectionError("terminal_projection must be a TerminalProjectionBinding")
        if any(not isinstance(receipt, FrozenFileReceipt) for receipt in self.receipts):
            raise BundleProjectionError("receipts must contain FrozenFileReceipt values")
        roles = tuple(receipt.role for receipt in self.receipts)
        if len(roles) != len(set(roles)):
            raise BundleProjectionError("each evidence role must have exactly one receipt")
        missing = set(REQUIRED_EVIDENCE_ROLES) - set(roles)
        unknown = set(roles) - set(REQUIRED_EVIDENCE_ROLES)
        if missing or unknown:
            raise BundleProjectionError(
                "receipts must cover exactly the required evidence layout; "
                f"missing={sorted(role.value for role in missing)!r}, "
                f"unknown={sorted(role.value for role in unknown)!r}"
            )
        ordered = tuple(sorted(self.receipts, key=lambda receipt: receipt.role.value))
        object.__setattr__(self, "receipts", ordered)

    @property
    def contract_value(self) -> str:
        return self._contract_value

    @property
    def campaign_value(self) -> str:
        return self._campaign_value

    @property
    def prediction_value(self) -> str:
        return self._prediction_value

    def receipt(self, role: EvidenceRole) -> FrozenFileReceipt:
        return next(receipt for receipt in self.receipts if receipt.role is role)


@dataclass(frozen=True, slots=True)
class BundleFinalizationResult:
    """Published bundle identity and its independently derived exact-regeneration grade."""

    root: Path
    bundle_id: str
    manifest_sha256: str
    submission_sha256: str
    inventory_sha256: str
    file_count: int
    total_size_bytes: int
    regeneration_evidence: BundleRegenerationEvidence
    replay_grade: ReplayGradeReceipt

    @property
    def bundle_sha256(self) -> str:
        """Compatibility spelling used by ``CampaignResult``."""

        return self.bundle_id

    @property
    def manifest_path(self) -> Path:
        return self.root / _BUNDLE_MANIFEST_NAME

    @property
    def bundle_sha256_path(self) -> Path:
        return self.root / _BUNDLE_DIGEST_NAME


@dataclass(frozen=True, slots=True)
class _Projection:
    root: Path
    bundle_id: str
    manifest_sha256: str
    submission_sha256: str
    inventory_sha256: str
    file_count: int
    total_size_bytes: int


def _copy_receipt(receipt: FrozenFileReceipt, destination: Path) -> None:
    descriptor, opened = _open_regular(receipt.source)
    digest = hashlib.sha256()
    observed = 0
    try:
        with destination.open("xb") as target:
            while True:
                chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                observed += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        final = os.fstat(descriptor)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    if (
        observed != receipt.size_bytes
        or digest.hexdigest() != receipt.sha256
        or (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    ):
        destination.unlink(missing_ok=True)
        raise BundleProjectionError(
            f"evidence bytes differ from frozen receipt: {receipt.role.value}"
        )
    os.chmod(destination, 0o444, follow_symlinks=False)


def _write_generated(path: Path, payload: bytes) -> tuple[str, int]:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444, follow_symlinks=False)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _inventory(root: Path) -> tuple[str, int, int]:
    entries: list[dict[str, object]] = []
    total = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise BundleProjectionError("bundle projection contains a non-regular member")
        sha256, size_bytes = _read_digest(path)
        total += size_bytes
        entries.append({"path": path.name, "sha256": sha256, "size_bytes": size_bytes})
    if tuple(entry["path"] for entry in entries) != tuple(sorted(REQUIRED_BUNDLE_PATHS)):
        raise BundleProjectionError("bundle projection member set differs from the required layout")
    return hashlib.sha256(_canonical_json(entries)).hexdigest(), len(entries), total


def _project_once(
    root: Path,
    request: BundleFinalizationRequest,
    cancel_event: threading.Event | None,
) -> _Projection:
    for receipt in request.receipts:
        _check_cancellation(cancel_event, f"copying {receipt.role.value}")
        _copy_receipt(receipt, root / receipt.role.value)
    evidence_size = sum(receipt.size_bytes for receipt in request.receipts)
    if evidence_size > _MAX_BUNDLE_BYTES:
        raise BundleProjectionError("frozen evidence exceeds the aggregate bundle-size bound")
    submission = request.receipt(EvidenceRole.SUBMISSION)
    replay = request.receipt(EvidenceRole.REPLAY_RECEIPT)
    manifest = {
        "schema_version": BUNDLE_PROJECTION_SCHEMA_VERSION,
        "contract_id": request.contract_value,
        "campaign_id": request.campaign_value,
        "selected_prediction_id": request.prediction_value,
        "terminal_projection": request.terminal_projection.manifest(),
        "identity": {
            "algorithm": "sha256",
            "definition": (
                "domain BundleId over selected prediction, replay output, submission, and manifest"
            ),
        },
        "submission_sha256": submission.sha256,
        "replay_receipt_sha256": replay.sha256,
        "required_paths": list(REQUIRED_BUNDLE_PATHS),
        "evidence": [receipt.manifest() for receipt in request.receipts],
    }
    manifest_payload = _canonical_json(manifest) + b"\n"
    manifest_sha256, _ = _write_generated(root / _BUNDLE_MANIFEST_NAME, manifest_payload)
    bundle_id = BundleId.derive(
        selected_prediction_id=PredictionId(request.prediction_value),
        replay_output_sha256={EvidenceRole.REPLAY_RECEIPT.value: replay.sha256},
        submission_sha256=submission.sha256,
        manifest_sha256=manifest_sha256,
    ).value
    digest_payload = f"{bundle_id}\n".encode("ascii")
    _write_generated(root / _BUNDLE_DIGEST_NAME, digest_payload)
    _legacy_fsync_directory(root)
    inventory_sha256, file_count, total_size_bytes = _inventory(root)
    return _Projection(
        root=root,
        bundle_id=bundle_id,
        manifest_sha256=manifest_sha256,
        submission_sha256=submission.sha256,
        inventory_sha256=inventory_sha256,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
    )


def _assert_same_projection(first: _Projection, second: _Projection) -> None:
    if (
        first.bundle_id != second.bundle_id
        or first.manifest_sha256 != second.manifest_sha256
        or first.submission_sha256 != second.submission_sha256
        or first.inventory_sha256 != second.inventory_sha256
        or first.file_count != second.file_count
        or first.total_size_bytes != second.total_size_bytes
    ):
        raise BundleProjectionError("clean bundle regeneration did not reproduce exact bytes")
    for name in REQUIRED_BUNDLE_PATHS:
        if (first.root / name).read_bytes() != (second.root / name).read_bytes():
            raise BundleProjectionError(f"clean bundle regeneration changed {name}")


def _remove_projection(path: Path) -> None:
    if not path.exists():
        return
    os.chmod(path, 0o700, follow_symlinks=False)
    for child in path.iterdir():
        if not child.is_symlink():
            os.chmod(child, 0o600, follow_symlinks=False)
    shutil.rmtree(path)


def _inspect_exact_existing_publication(
    path: Path,
    *,
    expected_root_mode: int,
    description: str,
) -> _Projection:
    """Read an existing publication without trusting its claimed identity.

    A crash can leave the exclusively renamed directory either unsealed (0700) or already sealed
    (0555).  Both cases are recoverable only after the caller's frozen request regenerates every
    member byte and :func:`_assert_same_projection` proves exact equality.
    """

    initial = path.lstat()
    if (
        stat.S_ISLNK(initial.st_mode)
        or not stat.S_ISDIR(initial.st_mode)
        or stat.S_IMODE(initial.st_mode) != expected_root_mode
    ):
        raise BundleProjectionError(f"existing destination is not {description}")
    members = tuple(sorted(path.iterdir(), key=lambda item: item.name))
    if tuple(member.name for member in members) != tuple(sorted(REQUIRED_BUNDLE_PATHS)):
        raise BundleProjectionError(f"{description} member set differs from the exact layout")
    for member in members:
        metadata = member.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
        ):
            raise BundleProjectionError(f"{description} contains an unsealed member")
    manifest_sha256, _ = _read_digest(path / _BUNDLE_MANIFEST_NAME)
    submission_sha256, _ = _read_digest(path / EvidenceRole.SUBMISSION.value)
    inventory_sha256, file_count, total_size_bytes = _inventory(path)
    try:
        digest_text = (path / _BUNDLE_DIGEST_NAME).read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise BundleProjectionError(f"{description} bundle digest is unreadable") from exc
    if not digest_text.endswith("\n") or digest_text.count("\n") != 1:
        raise BundleProjectionError(f"{description} bundle digest has invalid framing")
    bundle_id = _digest(digest_text.removesuffix("\n"), f"{description} BundleId")
    final = path.lstat()
    if (
        final.st_dev,
        final.st_ino,
        final.st_mtime_ns,
        final.st_mode,
    ) != (
        initial.st_dev,
        initial.st_ino,
        initial.st_mtime_ns,
        initial.st_mode,
    ):
        raise BundleProjectionError(f"{description} changed during exact verification")
    return _Projection(
        root=path,
        bundle_id=bundle_id,
        manifest_sha256=manifest_sha256,
        submission_sha256=submission_sha256,
        inventory_sha256=inventory_sha256,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
    )


def _inspect_exact_unsealed_orphan(path: Path) -> _Projection:
    return _inspect_exact_existing_publication(
        path,
        expected_root_mode=0o700,
        description="unsealed publication orphan",
    )


def _inspect_exact_sealed_publication(path: Path) -> _Projection:
    return _inspect_exact_existing_publication(
        path,
        expected_root_mode=0o555,
        description="sealed publication",
    )


class BundleFinalizer:
    """Create, exactly regenerate, and exclusively publish one evidence bundle."""

    def finalize(
        self,
        request: BundleFinalizationRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> BundleFinalizationResult:
        if not isinstance(request, BundleFinalizationRequest):
            raise BundleProjectionError("request must be BundleFinalizationRequest")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise BundleProjectionError("cancel_event must be threading.Event or None")
        _check_cancellation(cancel_event, "bundle admission")
        destination = request.destination
        if destination.name in {"", ".", ".."}:
            raise BundleProjectionError("destination must name one bundle directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        parent = destination.parent.lstat()
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise BundleProjectionError("bundle parent must be a real directory")
        orphan_candidate = False
        sealed_candidate = False
        if os.path.lexists(destination):
            existing = destination.lstat()
            orphan_candidate = (
                not stat.S_ISLNK(existing.st_mode)
                and stat.S_ISDIR(existing.st_mode)
                and stat.S_IMODE(existing.st_mode) == 0o700
            )
            sealed_candidate = (
                not stat.S_ISLNK(existing.st_mode)
                and stat.S_ISDIR(existing.st_mode)
                and stat.S_IMODE(existing.st_mode) == 0o555
            )
            if not orphan_candidate and not sealed_candidate:
                raise BundleProjectionError(f"bundle destination already exists: {destination}")

        first_root = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.projection-", dir=destination.parent)
        )
        second_root = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.regeneration-", dir=destination.parent)
        )
        published = False
        try:
            first = _project_once(first_root, request, cancel_event)
            _check_cancellation(cancel_event, "clean regeneration")
            second = _project_once(second_root, request, cancel_event)
            _assert_same_projection(first, second)
            regeneration = BundleRegenerationEvidence._from_verified_projection(
                contract_id=request.contract_value,
                prediction_id=request.prediction_value,
                first_bundle_id=first.bundle_id,
                regenerated_bundle_id=second.bundle_id,
                first_submission_sha256=first.submission_sha256,
                regenerated_submission_sha256=second.submission_sha256,
                first_inventory_sha256=first.inventory_sha256,
                regenerated_inventory_sha256=second.inventory_sha256,
            )
            grade = derive_bundle_exact_grade(regeneration)
            _check_cancellation(cancel_event, "exclusive publication")
            if sealed_candidate:
                sealed = _inspect_exact_sealed_publication(destination)
                _assert_same_projection(sealed, first)
                return BundleFinalizationResult(
                    root=destination.resolve(),
                    bundle_id=sealed.bundle_id,
                    manifest_sha256=sealed.manifest_sha256,
                    submission_sha256=sealed.submission_sha256,
                    inventory_sha256=sealed.inventory_sha256,
                    file_count=sealed.file_count,
                    total_size_bytes=sealed.total_size_bytes,
                    regeneration_evidence=regeneration,
                    replay_grade=grade,
                )
            if orphan_candidate:
                orphan = _inspect_exact_unsealed_orphan(destination)
                _assert_same_projection(orphan, first)
                os.chmod(destination, 0o555, follow_symlinks=False)
                _legacy_fsync_directory(destination.parent)
                return BundleFinalizationResult(
                    root=destination.resolve(),
                    bundle_id=orphan.bundle_id,
                    manifest_sha256=orphan.manifest_sha256,
                    submission_sha256=orphan.submission_sha256,
                    inventory_sha256=orphan.inventory_sha256,
                    file_count=orphan.file_count,
                    total_size_bytes=orphan.total_size_bytes,
                    regeneration_evidence=regeneration,
                    replay_grade=grade,
                )
            _legacy_publish_exclusive(first_root, destination)
            published = True
            _legacy_fsync_directory(destination.parent)
            published_projection = _Projection(
                root=destination,
                bundle_id=first.bundle_id,
                manifest_sha256=first.manifest_sha256,
                submission_sha256=first.submission_sha256,
                inventory_sha256=first.inventory_sha256,
                file_count=first.file_count,
                total_size_bytes=first.total_size_bytes,
            )
            _assert_same_projection(published_projection, second)
            os.chmod(destination, 0o555, follow_symlinks=False)
            return BundleFinalizationResult(
                root=destination.resolve(),
                bundle_id=first.bundle_id,
                manifest_sha256=first.manifest_sha256,
                submission_sha256=first.submission_sha256,
                inventory_sha256=first.inventory_sha256,
                file_count=first.file_count,
                total_size_bytes=first.total_size_bytes,
                regeneration_evidence=regeneration,
                replay_grade=grade,
            )
        except BundleProjectionError:
            raise
        except OSError as exc:
            raise BundleProjectionError("bundle projection failed closed") from exc
        finally:
            if not published:
                _remove_projection(first_root)
            _remove_projection(second_root)


def finalize_bundle(
    request: BundleFinalizationRequest,
    *,
    cancel_event: threading.Event | None = None,
) -> BundleFinalizationResult:
    """Functional adapter for callers that do not retain a finalizer instance."""

    return BundleFinalizer().finalize(request, cancel_event=cancel_event)


__all__ = [
    "BUNDLE_PROJECTION_SCHEMA_VERSION",
    "REQUIRED_BUNDLE_PATHS",
    "REQUIRED_EVIDENCE_ROLES",
    "BundleFinalizationRequest",
    "BundleFinalizationResult",
    "BundleFinalizer",
    "BundleProjectionCancelledError",
    "BundleProjectionError",
    "EvidenceRole",
    "FrozenFileReceipt",
    "TerminalProjectionBinding",
    "capture_clean_replay_receipts",
    "finalize_bundle",
]
