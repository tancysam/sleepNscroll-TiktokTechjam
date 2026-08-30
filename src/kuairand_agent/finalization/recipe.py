"""Typed, non-executable recipes for provider-free frozen-model replay.

Recipes contain identities and fixed policy choices only.  They cannot name Python import paths,
shell commands, arbitrary entry points, or adaptive fusion code.  The trusted factory maps the
small backend enum to controller-owned implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from kuairand_agent.candidates.fusion import FUSION_WEIGHT_GRID

REPLAY_RECIPE_SCHEMA_VERSION: Final = 1
MAX_REPLAY_RECIPE_BYTES: Final = 256 * 1024
MAX_ENSEMBLE_REPLAY_MEMBERS: Final = 16
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class ReplayRecipeError(ValueError):
    """A replay recipe is malformed, unbound, or outside the backend allowlist."""


class ReplayBackendKind(StrEnum):
    """The complete executable backend allowlist."""

    GENERATED_LAMBDARANK = "generated_lambdarank_v1"
    GENERATED_LAMBDARANK_ENSEMBLE = "generated_lambdarank_ensemble_v2"
    OFFICIAL_FM = "official_fm_v1"


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
        raise ReplayRecipeError("replay recipe must be finite canonical JSON") from exc


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ReplayRecipeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str, *, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ReplayRecipeError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class OfficialFMMemberRecipe:
    """Exact immutable organizer-FM artifacts used alone or as a fusion member."""

    checkpoint_sha256: str
    checkpoint_digest: str
    encoding_sha256: str
    encoding_digest: str
    config_digest: str
    starter_manifest_sha256: str
    seed: int

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "checkpoint_digest",
            "encoding_sha256",
            "encoding_digest",
            "config_digest",
            "starter_manifest_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ReplayRecipeError("FM seed must be a uint32-compatible integer")

    def manifest(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_digest": self.checkpoint_digest,
            "encoding_sha256": self.encoding_sha256,
            "encoding_digest": self.encoding_digest,
            "config_digest": self.config_digest,
            "starter_manifest_sha256": self.starter_manifest_sha256,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class GeneratedLambdaRankReplayRecipe:
    """Frozen generated-tree inference, optionally fused with one official FM member."""

    source_artifact_sha256: str
    candidate_source_sha256: str
    candidate_config_sha256: str
    feature_artifact_sha256: str
    validation_features_sha256: str
    final_features_sha256: str
    checkpoint_artifact_sha256: str
    tree_checkpoint_sha256: str
    data_sha256: str
    validation_inputs_digest: str
    final_inputs_digest: str
    feature_count: int
    timeout_seconds: int
    memory_limit_bytes: int
    threads: int
    fm_member: OfficialFMMemberRecipe | None = None
    fusion_weights: tuple[float, float] | None = None
    schema_version: int = field(init=False, default=REPLAY_RECIPE_SCHEMA_VERSION)
    backend: ReplayBackendKind = field(init=False, default=ReplayBackendKind.GENERATED_LAMBDARANK)

    def __post_init__(self) -> None:
        for name in (
            "source_artifact_sha256",
            "candidate_source_sha256",
            "candidate_config_sha256",
            "feature_artifact_sha256",
            "validation_features_sha256",
            "final_features_sha256",
            "checkpoint_artifact_sha256",
            "tree_checkpoint_sha256",
            "data_sha256",
            "validation_inputs_digest",
            "final_inputs_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _positive_int(self.feature_count, "feature_count", maximum=1_000_000)
        _positive_int(self.timeout_seconds, "timeout_seconds", maximum=86_400)
        _positive_int(self.memory_limit_bytes, "memory_limit_bytes")
        _positive_int(self.threads, "threads", maximum=64)
        if (self.fm_member is None) != (self.fusion_weights is None):
            raise ReplayRecipeError("fm_member and fusion_weights must be declared together")
        if self.fm_member is not None and not isinstance(self.fm_member, OfficialFMMemberRecipe):
            raise ReplayRecipeError("fm_member must be an OfficialFMMemberRecipe")
        if self.fusion_weights is not None:
            weights = self.fusion_weights
            if (
                type(weights) is not tuple
                or len(weights) != 2
                or any(type(value) is not float or not math.isfinite(value) for value in weights)
                or weights not in FUSION_WEIGHT_GRID
            ):
                raise ReplayRecipeError(
                    f"fusion_weights must be one exact FUSION_WEIGHT_GRID point: "
                    f"{FUSION_WEIGHT_GRID}"
                )

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend.value,
            "source_artifact_sha256": self.source_artifact_sha256,
            "candidate_source_sha256": self.candidate_source_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "validation_features_sha256": self.validation_features_sha256,
            "final_features_sha256": self.final_features_sha256,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "tree_checkpoint_sha256": self.tree_checkpoint_sha256,
            "data_sha256": self.data_sha256,
            "validation_inputs_digest": self.validation_inputs_digest,
            "final_inputs_digest": self.final_inputs_digest,
            "feature_count": self.feature_count,
            "limits": {
                "timeout_seconds": self.timeout_seconds,
                "memory_limit_bytes": self.memory_limit_bytes,
                "threads": self.threads,
            },
            "fm_member": None if self.fm_member is None else self.fm_member.manifest(),
            "fusion_weights": (None if self.fusion_weights is None else list(self.fusion_weights)),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.manifest())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class GeneratedLambdaRankEnsembleReplayRecipe:
    """Ordered, all-or-nothing rank fusion of complete generated LambdaRank v1 recipes.

    ``members`` is intentionally an ordered tuple rather than a mapping: position is the frozen
    seed/member identity and must stay paired with the corresponding digest, validation output,
    and fusion weight.  A member remains a complete v1 recipe so its source, input, checkpoint,
    and bounded subprocess policy are replayed unchanged.  This supports a uniform seed ensemble
    today without encoding a seed-specific protocol, and can later carry distinct direct-portfolio
    members through the same immutable tuple.
    """

    members: tuple[GeneratedLambdaRankReplayRecipe, ...]
    member_recipe_digests: tuple[str, ...]
    validation_member_prediction_digests: tuple[str, ...]
    fusion_weights: tuple[float, ...]
    validation_fusion_digest: str
    schema_version: int = field(init=False, default=REPLAY_RECIPE_SCHEMA_VERSION)
    backend: ReplayBackendKind = field(
        init=False, default=ReplayBackendKind.GENERATED_LAMBDARANK_ENSEMBLE
    )

    def __post_init__(self) -> None:
        members = self.members
        if type(members) is not tuple or not 2 <= len(members) <= MAX_ENSEMBLE_REPLAY_MEMBERS:
            raise ReplayRecipeError(
                "ensemble members must be an ordered tuple containing between 2 and "
                f"{MAX_ENSEMBLE_REPLAY_MEMBERS} generated LambdaRank recipes"
            )
        if any(not isinstance(member, GeneratedLambdaRankReplayRecipe) for member in members):
            raise ReplayRecipeError("ensemble members must all be GeneratedLambdaRankReplayRecipe")
        if len({member.digest for member in members}) != len(members):
            raise ReplayRecipeError("ensemble members must not repeat one member recipe")
        for name in (
            "member_recipe_digests",
            "validation_member_prediction_digests",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != len(members):
                raise ReplayRecipeError(f"{name} must be an ordered digest tuple matching members")
            for index, value in enumerate(values):
                _digest(value, f"{name}[{index}]")
        if self.member_recipe_digests != tuple(member.digest for member in members):
            raise ReplayRecipeError("member_recipe_digests must exactly bind ordered members")
        weights = self.fusion_weights
        if (
            type(weights) is not tuple
            or len(weights) != len(members)
            or any(
                type(weight) is not float
                or not math.isfinite(weight)
                or weight < 0.0
                or (weight == 0.0 and math.copysign(1.0, weight) < 0.0)
                for weight in weights
            )
            or math.fsum(weights) != 1.0
        ):
            raise ReplayRecipeError(
                "fusion_weights must be a finite non-negative float tuple matching members and "
                "summing to one"
            )
        _digest(self.validation_fusion_digest, "validation_fusion_digest")
        first = members[0]
        for member in members[1:]:
            if member.data_sha256 != first.data_sha256:
                raise ReplayRecipeError("ensemble members must bind one shared data_sha256")
            if member.validation_inputs_digest != first.validation_inputs_digest:
                raise ReplayRecipeError(
                    "ensemble members must bind one shared validation input capability"
                )
            if member.final_inputs_digest != first.final_inputs_digest:
                raise ReplayRecipeError(
                    "ensemble members must bind one shared final input capability"
                )

    @property
    def data_sha256(self) -> str:
        return self.members[0].data_sha256

    @property
    def validation_inputs_digest(self) -> str:
        return self.members[0].validation_inputs_digest

    @property
    def final_inputs_digest(self) -> str:
        return self.members[0].final_inputs_digest

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend.value,
            "members": [member.manifest() for member in self.members],
            "member_recipe_digests": list(self.member_recipe_digests),
            "validation_member_prediction_digests": list(
                self.validation_member_prediction_digests
            ),
            "fusion_weights": list(self.fusion_weights),
            "validation_fusion_digest": self.validation_fusion_digest,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.manifest())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class OfficialFMReplayRecipe:
    """Frozen official FM fallback over label-free candidate input capabilities."""

    source_artifact_sha256: str
    feature_artifact_sha256: str
    checkpoint_artifact_sha256: str
    data_sha256: str
    validation_inputs_digest: str
    final_inputs_digest: str
    fm_member: OfficialFMMemberRecipe
    schema_version: int = field(init=False, default=REPLAY_RECIPE_SCHEMA_VERSION)
    backend: ReplayBackendKind = field(init=False, default=ReplayBackendKind.OFFICIAL_FM)

    def __post_init__(self) -> None:
        for name in (
            "source_artifact_sha256",
            "feature_artifact_sha256",
            "checkpoint_artifact_sha256",
            "data_sha256",
            "validation_inputs_digest",
            "final_inputs_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.fm_member, OfficialFMMemberRecipe):
            raise ReplayRecipeError("fm_member must be an OfficialFMMemberRecipe")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend.value,
            "source_artifact_sha256": self.source_artifact_sha256,
            "feature_artifact_sha256": self.feature_artifact_sha256,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "data_sha256": self.data_sha256,
            "validation_inputs_digest": self.validation_inputs_digest,
            "final_inputs_digest": self.final_inputs_digest,
            "fm_member": self.fm_member.manifest(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.manifest())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


type ReplayRecipe = (
    GeneratedLambdaRankReplayRecipe
    | GeneratedLambdaRankEnsembleReplayRecipe
    | OfficialFMReplayRecipe
)


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayRecipeError(f"replay recipe contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ReplayRecipeError(f"replay recipe contains non-finite constant {value}")


def _exact(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReplayRecipeError(f"{name} fields do not match the exact schema")
    return cast(dict[str, object], value)


def _fm_from_manifest(value: object) -> OfficialFMMemberRecipe:
    raw = _exact(
        value,
        {
            "checkpoint_sha256",
            "checkpoint_digest",
            "encoding_sha256",
            "encoding_digest",
            "config_digest",
            "starter_manifest_sha256",
            "seed",
        },
        "fm_member",
    )
    return OfficialFMMemberRecipe(
        checkpoint_sha256=cast(str, raw["checkpoint_sha256"]),
        checkpoint_digest=cast(str, raw["checkpoint_digest"]),
        encoding_sha256=cast(str, raw["encoding_sha256"]),
        encoding_digest=cast(str, raw["encoding_digest"]),
        config_digest=cast(str, raw["config_digest"]),
        starter_manifest_sha256=cast(str, raw["starter_manifest_sha256"]),
        seed=cast(int, raw["seed"]),
    )


def _generated_from_manifest(value: dict[str, object]) -> GeneratedLambdaRankReplayRecipe:
    raw = _exact(
        value,
        {
            "schema_version",
            "backend",
            "source_artifact_sha256",
            "candidate_source_sha256",
            "candidate_config_sha256",
            "feature_artifact_sha256",
            "validation_features_sha256",
            "final_features_sha256",
            "checkpoint_artifact_sha256",
            "tree_checkpoint_sha256",
            "data_sha256",
            "validation_inputs_digest",
            "final_inputs_digest",
            "feature_count",
            "limits",
            "fm_member",
            "fusion_weights",
        },
        "generated LambdaRank recipe",
    )
    limits = _exact(raw["limits"], {"timeout_seconds", "memory_limit_bytes", "threads"}, "limits")
    raw_member = raw["fm_member"]
    member = None if raw_member is None else _fm_from_manifest(raw_member)
    raw_weights = raw["fusion_weights"]
    weights: tuple[float, float] | None
    if raw_weights is None:
        weights = None
    elif (
        isinstance(raw_weights, list)
        and len(raw_weights) == 2
        and all(type(item) is float for item in raw_weights)
    ):
        weights = (cast(float, raw_weights[0]), cast(float, raw_weights[1]))
    else:
        raise ReplayRecipeError("fusion_weights must be null or two JSON floats")
    return GeneratedLambdaRankReplayRecipe(
        source_artifact_sha256=cast(str, raw["source_artifact_sha256"]),
        candidate_source_sha256=cast(str, raw["candidate_source_sha256"]),
        candidate_config_sha256=cast(str, raw["candidate_config_sha256"]),
        feature_artifact_sha256=cast(str, raw["feature_artifact_sha256"]),
        validation_features_sha256=cast(str, raw["validation_features_sha256"]),
        final_features_sha256=cast(str, raw["final_features_sha256"]),
        checkpoint_artifact_sha256=cast(str, raw["checkpoint_artifact_sha256"]),
        tree_checkpoint_sha256=cast(str, raw["tree_checkpoint_sha256"]),
        data_sha256=cast(str, raw["data_sha256"]),
        validation_inputs_digest=cast(str, raw["validation_inputs_digest"]),
        final_inputs_digest=cast(str, raw["final_inputs_digest"]),
        feature_count=cast(int, raw["feature_count"]),
        timeout_seconds=cast(int, limits["timeout_seconds"]),
        memory_limit_bytes=cast(int, limits["memory_limit_bytes"]),
        threads=cast(int, limits["threads"]),
        fm_member=member,
        fusion_weights=weights,
    )


def _ensemble_from_manifest(value: dict[str, object]) -> GeneratedLambdaRankEnsembleReplayRecipe:
    raw = _exact(
        value,
        {
            "schema_version",
            "backend",
            "members",
            "member_recipe_digests",
            "validation_member_prediction_digests",
            "fusion_weights",
            "validation_fusion_digest",
        },
        "generated LambdaRank ensemble recipe",
    )
    raw_members = raw["members"]
    if not isinstance(raw_members, list):
        raise ReplayRecipeError("ensemble members must be a JSON array")
    members: list[GeneratedLambdaRankReplayRecipe] = []
    for index, raw_member in enumerate(raw_members):
        if not isinstance(raw_member, dict):
            raise ReplayRecipeError(f"ensemble members[{index}] must be a generated recipe object")
        if raw_member.get("backend") != ReplayBackendKind.GENERATED_LAMBDARANK.value:
            raise ReplayRecipeError(f"ensemble members[{index}] must be generated_lambdarank_v1")
        members.append(_generated_from_manifest(cast(dict[str, object], raw_member)))

    def _digest_tuple(field: str) -> tuple[str, ...]:
        values = raw[field]
        if not isinstance(values, list) or any(type(item) is not str for item in values):
            raise ReplayRecipeError(f"{field} must be a JSON string array")
        return tuple(cast(str, item) for item in values)

    raw_weights = raw["fusion_weights"]
    if not isinstance(raw_weights, list) or any(type(item) is not float for item in raw_weights):
        raise ReplayRecipeError("fusion_weights must be a JSON float array")
    return GeneratedLambdaRankEnsembleReplayRecipe(
        members=tuple(members),
        member_recipe_digests=_digest_tuple("member_recipe_digests"),
        validation_member_prediction_digests=_digest_tuple(
            "validation_member_prediction_digests"
        ),
        fusion_weights=tuple(cast(float, item) for item in raw_weights),
        validation_fusion_digest=cast(str, raw["validation_fusion_digest"]),
    )


def _official_from_manifest(value: dict[str, object]) -> OfficialFMReplayRecipe:
    raw = _exact(
        value,
        {
            "schema_version",
            "backend",
            "source_artifact_sha256",
            "feature_artifact_sha256",
            "checkpoint_artifact_sha256",
            "data_sha256",
            "validation_inputs_digest",
            "final_inputs_digest",
            "fm_member",
        },
        "official FM recipe",
    )
    return OfficialFMReplayRecipe(
        source_artifact_sha256=cast(str, raw["source_artifact_sha256"]),
        feature_artifact_sha256=cast(str, raw["feature_artifact_sha256"]),
        checkpoint_artifact_sha256=cast(str, raw["checkpoint_artifact_sha256"]),
        data_sha256=cast(str, raw["data_sha256"]),
        validation_inputs_digest=cast(str, raw["validation_inputs_digest"]),
        final_inputs_digest=cast(str, raw["final_inputs_digest"]),
        fm_member=_fm_from_manifest(raw["fm_member"]),
    )


def parse_replay_recipe(payload: bytes) -> ReplayRecipe:
    """Parse one exact canonical JSON recipe through the executable backend allowlist."""

    if type(payload) is not bytes or not 0 < len(payload) <= MAX_REPLAY_RECIPE_BYTES:
        raise ReplayRecipeError("replay recipe size is outside the supported bound")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ReplayRecipeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ReplayRecipeError("replay recipe is not strict ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise ReplayRecipeError("replay recipe is not canonical JSON")
    raw = cast(dict[str, object], value)
    if raw.get("schema_version") != REPLAY_RECIPE_SCHEMA_VERSION:
        raise ReplayRecipeError("replay recipe schema_version must be integer 1")
    backend = raw.get("backend")
    try:
        kind = ReplayBackendKind(cast(str, backend))
    except (TypeError, ValueError) as exc:
        raise ReplayRecipeError("replay backend is not allowlisted") from exc
    if kind is ReplayBackendKind.GENERATED_LAMBDARANK:
        return _generated_from_manifest(raw)
    if kind is ReplayBackendKind.GENERATED_LAMBDARANK_ENSEMBLE:
        return _ensemble_from_manifest(raw)
    if kind is ReplayBackendKind.OFFICIAL_FM:
        return _official_from_manifest(raw)
    raise ReplayRecipeError("replay backend is not allowlisted")  # pragma: no cover


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReplayRecipeError("replay recipe must be a readable regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_REPLAY_RECIPE_BYTES:
            raise ReplayRecipeError("replay recipe must be a bounded regular file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(payload) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ReplayRecipeError("replay recipe changed while being read")
        return payload
    finally:
        os.close(descriptor)


def load_replay_recipe(path: str | Path, *, expected_sha256: str | None = None) -> ReplayRecipe:
    """Load, hash, and strictly parse one immutable replay recipe file."""

    payload = _read_regular(Path(path))
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != _digest(expected_sha256, "expected_sha256"):
        raise ReplayRecipeError("replay recipe SHA-256 mismatch")
    recipe = parse_replay_recipe(payload)
    if recipe.digest != observed:
        raise ReplayRecipeError("replay recipe logical identity differs from its bytes")
    return recipe


def write_replay_recipe(path: str | Path, recipe: ReplayRecipe) -> Path:
    """Exclusively persist one canonical recipe suitable for an INPUT artifact."""

    if not isinstance(
        recipe,
        (
            GeneratedLambdaRankReplayRecipe,
            GeneratedLambdaRankEnsembleReplayRecipe,
            OfficialFMReplayRecipe,
        ),
    ):
        raise ReplayRecipeError("recipe must be one allowlisted replay recipe")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise ReplayRecipeError(f"refusing to overwrite replay recipe: {destination}") from exc
    committed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(recipe.canonical_bytes)
            handle.flush()
            os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        committed = True
    finally:
        os.close(descriptor)
        if not committed:
            destination.unlink(missing_ok=True)
    return destination.resolve()


__all__ = [
    "MAX_ENSEMBLE_REPLAY_MEMBERS",
    "MAX_REPLAY_RECIPE_BYTES",
    "REPLAY_RECIPE_SCHEMA_VERSION",
    "GeneratedLambdaRankEnsembleReplayRecipe",
    "GeneratedLambdaRankReplayRecipe",
    "OfficialFMMemberRecipe",
    "OfficialFMReplayRecipe",
    "ReplayBackendKind",
    "ReplayRecipe",
    "ReplayRecipeError",
    "load_replay_recipe",
    "parse_replay_recipe",
    "write_replay_recipe",
]
