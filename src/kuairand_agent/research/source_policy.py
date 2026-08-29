"""Canonical candidate-owned source policy shared by prompts and enforcement.

The policy is value-only: it contains no filesystem locations or controller authority.  Provider
requests carry its canonical manifest and digest, while the local materializer remains the
authoritative enforcement boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Self

SOURCE_POLICY_SCHEMA_VERSION: Final = 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PORTABLE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\Z")


class CandidateSourcePolicyError(ValueError):
    """A source-policy value or wire manifest is invalid."""


class CandidateManifestPolicyError(CandidateSourcePolicyError):
    """A proposed or generated manifest violates a normalized source-policy rule."""

    def __init__(self, message: str, *, fingerprint: str) -> None:
        super().__init__(message)
        self.fingerprint = fingerprint


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CandidateSourcePolicyError("source policy is not canonical JSON data") from exc


def _string_tuple(value: object, location: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise CandidateSourcePolicyError(f"{location} must be a {qualifier}JSON array")
    if any(type(item) is not str or not item for item in value):
        raise CandidateSourcePolicyError(f"{location} must contain non-empty strings")
    result = tuple(value)
    if tuple(sorted(result)) != result or len(result) != len(set(result)):
        raise CandidateSourcePolicyError(f"{location} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class CandidateSourcePolicy:
    """Immutable policy for candidate-owned complete-file overlays."""

    final_entrypoint: str
    allowed_suffixes: tuple[str, ...]
    forbidden_basenames: tuple[str, ...]
    forbidden_path_roots: tuple[str, ...]
    forbidden_import_roots: tuple[str, ...]
    forbidden_calls: tuple[str, ...]
    max_generated_files: int
    max_generated_file_bytes: int
    max_generated_total_bytes: int
    overlay_semantics: str
    schema_version: int = SOURCE_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SOURCE_POLICY_SCHEMA_VERSION
        ):
            raise CandidateSourcePolicyError(
                f"source_policy.schema_version must be {SOURCE_POLICY_SCHEMA_VERSION}"
            )
        if self.final_entrypoint != "candidate.py":
            raise CandidateSourcePolicyError("source policy final entrypoint is frozen")
        for name in (
            "allowed_suffixes",
            "forbidden_basenames",
            "forbidden_path_roots",
            "forbidden_import_roots",
            "forbidden_calls",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise CandidateSourcePolicyError(f"source_policy.{name} must be a non-empty tuple")
            if tuple(sorted(values)) != values or len(values) != len(set(values)):
                raise CandidateSourcePolicyError(f"source_policy.{name} must be sorted and unique")
            if any(type(value) is not str or not value for value in values):
                raise CandidateSourcePolicyError(
                    f"source_policy.{name} must contain non-empty strings"
                )
        if self.allowed_suffixes != (".json", ".md", ".py"):
            raise CandidateSourcePolicyError("source policy suffix allowlist is frozen")
        if self.overlay_semantics != "complete_file_overlay":
            raise CandidateSourcePolicyError("source policy overlay semantics are frozen")
        if type(self.max_generated_files) is not int or self.max_generated_files != 12:
            raise CandidateSourcePolicyError("source policy generated-file limit is frozen")
        if (
            type(self.max_generated_file_bytes) is not int
            or self.max_generated_file_bytes != 512 * 1024
        ):
            raise CandidateSourcePolicyError("source policy per-file byte limit is frozen")
        if (
            type(self.max_generated_total_bytes) is not int
            or self.max_generated_total_bytes != 1024 * 1024
        ):
            raise CandidateSourcePolicyError("source policy total byte limit is frozen")

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "final_entrypoint": self.final_entrypoint,
            "allowed_suffixes": list(self.allowed_suffixes),
            "forbidden_basenames": list(self.forbidden_basenames),
            "forbidden_path_roots": list(self.forbidden_path_roots),
            "forbidden_import_roots": list(self.forbidden_import_roots),
            "forbidden_calls": list(self.forbidden_calls),
            "max_generated_files": self.max_generated_files,
            "max_generated_file_bytes": self.max_generated_file_bytes,
            "max_generated_total_bytes": self.max_generated_total_bytes,
            "overlay_semantics": self.overlay_semantics,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_wire())).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        if not isinstance(raw, Mapping) or any(type(key) is not str for key in raw):
            raise CandidateSourcePolicyError("source_policy must be a JSON object")
        expected = {
            "schema_version",
            "final_entrypoint",
            "allowed_suffixes",
            "forbidden_basenames",
            "forbidden_path_roots",
            "forbidden_import_roots",
            "forbidden_calls",
            "max_generated_files",
            "max_generated_file_bytes",
            "max_generated_total_bytes",
            "overlay_semantics",
        }
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            unknown = sorted(set(raw) - expected)
            detail = f"missing={missing!r}, unknown={unknown!r}"
            raise CandidateSourcePolicyError(f"source_policy fields are not exact: {detail}")
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            final_entrypoint=raw["final_entrypoint"],  # type: ignore[arg-type]
            allowed_suffixes=_string_tuple(
                raw["allowed_suffixes"], "source_policy.allowed_suffixes"
            ),
            forbidden_basenames=_string_tuple(
                raw["forbidden_basenames"], "source_policy.forbidden_basenames"
            ),
            forbidden_path_roots=_string_tuple(
                raw["forbidden_path_roots"], "source_policy.forbidden_path_roots"
            ),
            forbidden_import_roots=_string_tuple(
                raw["forbidden_import_roots"], "source_policy.forbidden_import_roots"
            ),
            forbidden_calls=_string_tuple(raw["forbidden_calls"], "source_policy.forbidden_calls"),
            max_generated_files=raw["max_generated_files"],  # type: ignore[arg-type]
            max_generated_file_bytes=raw["max_generated_file_bytes"],  # type: ignore[arg-type]
            max_generated_total_bytes=raw["max_generated_total_bytes"],  # type: ignore[arg-type]
            overlay_semantics=raw["overlay_semantics"],  # type: ignore[arg-type]
        )

    def validate_path(self, value: str, *, generated: bool) -> PurePosixPath:
        """Validate one portable candidate-owned path and return its canonical form."""

        if type(value) is not str or not value or "\\" in value or "\x00" in value:
            raise CandidateManifestPolicyError(
                "candidate path must be a non-empty POSIX path",
                fingerprint="candidate_path_policy:invalid_posix_path",
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise CandidateManifestPolicyError(
                "candidate path contains a control character",
                fingerprint="candidate_path_policy:control_character",
            )
        if _PORTABLE_PATH_RE.fullmatch(value) is None:
            raise CandidateManifestPolicyError(
                f"candidate path is not portable: {value!r}",
                fingerprint="candidate_path_policy:nonportable_path",
            )
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise CandidateManifestPolicyError(
                f"candidate path is unsafe: {value!r}",
                fingerprint="candidate_path_policy:unsafe_path",
            )
        if path.as_posix() != value:
            raise CandidateManifestPolicyError(
                f"candidate path is not canonical: {value!r}",
                fingerprint="candidate_path_policy:noncanonical_path",
            )
        if any(part.startswith(".") for part in path.parts):
            raise CandidateManifestPolicyError(
                f"hidden candidate path is forbidden: {value!r}",
                fingerprint=f"candidate_path_policy:hidden_path:{path.parts[0]}",
            )
        if path.parts[0] in self.forbidden_path_roots:
            raise CandidateManifestPolicyError(
                f"trusted candidate path is forbidden: {value!r}",
                fingerprint=f"candidate_path_policy:forbidden_root:{path.parts[0]}",
            )
        if path.name in self.forbidden_basenames:
            raise CandidateManifestPolicyError(
                f"reserved candidate filename is forbidden: {value!r}",
                fingerprint=f"candidate_path_policy:forbidden_basename:{path.name}",
            )
        if path.suffix not in self.allowed_suffixes:
            kind = "generated" if generated else "parent"
            suffix = path.suffix or "<none>"
            raise CandidateManifestPolicyError(
                f"{kind} candidate file suffix is not allowed: {value!r}",
                fingerprint=f"candidate_path_policy:unsupported_suffix:{suffix}",
            )
        return path

    def validate_manifest(
        self,
        overlay_paths: Iterable[str],
        *,
        parent_paths: Iterable[str] = (),
        require_final_entrypoint: bool = True,
    ) -> tuple[str, ...]:
        """Validate an overlay and require the entrypoint in the resulting final tree."""

        overlay = tuple(overlay_paths)
        parent = tuple(parent_paths)
        if len(overlay) > self.max_generated_files:
            raise CandidateManifestPolicyError(
                "generated package exceeds the file-count limit",
                fingerprint="candidate_manifest:file_count_exceeded",
            )
        if len(overlay) != len(set(overlay)):
            raise CandidateManifestPolicyError(
                "candidate manifest contains duplicate paths",
                fingerprint="candidate_manifest:duplicate_path",
            )
        for path in parent:
            self.validate_path(path, generated=False)
        for path in overlay:
            self.validate_path(path, generated=True)
        final_paths = tuple(sorted(set(parent) | set(overlay)))
        if require_final_entrypoint and self.final_entrypoint not in final_paths:
            raise CandidateManifestPolicyError(
                f"candidate tree must contain the {self.final_entrypoint} entry point",
                fingerprint=f"candidate_manifest:missing_entrypoint:{self.final_entrypoint}",
            )
        return final_paths


DEFAULT_CANDIDATE_SOURCE_POLICY: Final = CandidateSourcePolicy(
    final_entrypoint="candidate.py",
    allowed_suffixes=(".json", ".md", ".py"),
    forbidden_basenames=tuple(
        sorted(
            {
                "baseline.py",
                "conftest.py",
                "data.py",
                "evaluate.py",
                "pyproject.toml",
                "sitecustomize.py",
                "submit.py",
                "usercustomize.py",
            }
        )
    ),
    forbidden_path_roots=tuple(
        sorted(
            {
                "configs",
                "docs",
                "kuairand-starter-kit",
                "kuairand_agent",
                "kuairand_starter_kit",
                "scripts",
                "tests",
            }
        )
    ),
    forbidden_import_roots=tuple(
        sorted(
            {
                "aiohttp",
                "builtins",
                "ctypes",
                "ftplib",
                "glob",
                "http",
                "httpx",
                "importlib",
                "kuairand_agent",
                "marshal",
                "multiprocessing",
                "os",
                "pickle",
                "requests",
                "shutil",
                "socket",
                "subprocess",
                "sys",
                "tempfile",
                "urllib",
                "webbrowser",
            }
        )
    ),
    forbidden_calls=tuple(sorted({"__import__", "breakpoint", "compile", "eval", "exec"})),
    max_generated_files=12,
    max_generated_file_bytes=512 * 1024,
    max_generated_total_bytes=1024 * 1024,
    overlay_semantics="complete_file_overlay",
)


def validate_policy_digest(policy: CandidateSourcePolicy, digest: object) -> None:
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        raise CandidateSourcePolicyError("source policy digest must be lowercase SHA-256")
    if digest != policy.digest:
        raise CandidateSourcePolicyError("source policy digest mismatch")


def classify_method_family(mechanism: str, objective: str) -> str:
    """Classify a proposal into a stable coarse novelty family."""

    normalized = f"{mechanism} {objective}".casefold()
    families = (
        ("pairwise", ("pairwise", "bpr", "ranking loss")),
        ("listwise", ("listwise", "listnet", "listmle")),
        ("calibration", ("calibrat", "isotonic", "platt")),
        ("optimization", ("optimizer", "learning rate", "regularization", "early stop")),
        ("feature", ("feature", "embedding", "interaction", "duration", "bucket")),
    )
    for family, markers in families:
        if any(marker in normalized for marker in markers):
            return family
    return "other"


def proposal_novelty_signature(
    *,
    parent_digest: str,
    method_family: str,
    mechanism: str,
    objective: str,
    sampling: str,
    required_source_fields: Iterable[str],
    legal_manifest: Iterable[str],
    evidence_cursor: str,
) -> str:
    """Return a deterministic signature for proposal-admission novelty control."""

    if method_family not in {
        "pairwise",
        "listwise",
        "feature",
        "calibration",
        "optimization",
        "other",
    }:
        raise CandidateSourcePolicyError("proposal method family is unsupported")

    def normalize(value: str) -> str:
        if type(value) is not str:
            raise CandidateSourcePolicyError("proposal novelty text must be text")
        return " ".join(value.casefold().split())

    value = {
        "schema_version": 1,
        "parent_digest": parent_digest,
        "method_family": method_family,
        "mechanism": normalize(mechanism),
        "objective": normalize(objective),
        "sampling": normalize(sampling),
        "required_source_fields": sorted(set(required_source_fields)),
        "legal_manifest": sorted(set(legal_manifest)),
        "evidence_cursor": evidence_cursor,
    }
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


__all__ = [
    "DEFAULT_CANDIDATE_SOURCE_POLICY",
    "CandidateManifestPolicyError",
    "CandidateSourcePolicy",
    "CandidateSourcePolicyError",
    "classify_method_family",
    "proposal_novelty_signature",
    "validate_policy_digest",
]
