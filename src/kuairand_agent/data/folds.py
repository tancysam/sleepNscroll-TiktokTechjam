"""Frozen, rolling-origin temporal folds derived only from canonical train dates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Self, cast

from kuairand_agent.contract import SplitName
from kuairand_agent.data.canonical import CanonicalTrainingSplit, TrainingTargets

TEMPORAL_FOLD_SCHEMA_VERSION: Final = 1


class TemporalFoldError(ValueError):
    """Raised when a frozen fold or its minimum support contract is violated."""


@dataclass(frozen=True, slots=True)
class TemporalFoldSpec:
    """Inclusive train-prefix and validation-window dates for one rolling-origin fold."""

    name: str
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int

    def __post_init__(self) -> None:
        if not self.name:
            raise TemporalFoldError("fold name cannot be empty")
        if not self.train_start <= self.train_end < self.valid_start <= self.valid_end:
            raise TemporalFoldError("fold dates must be ordered, disjoint, and rolling-origin")

    @property
    def train_dates(self) -> tuple[int, ...]:
        return tuple(range(self.train_start, self.train_end + 1))

    @property
    def valid_dates(self) -> tuple[int, ...]:
        return tuple(range(self.valid_start, self.valid_end + 1))

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "train": {"start": self.train_start, "end": self.train_end},
            "valid": {"start": self.valid_start, "end": self.valid_end},
        }


FOLD_A_SPEC: Final = TemporalFoldSpec(
    name="A",
    train_start=20220408,
    train_end=20220415,
    valid_start=20220416,
    valid_end=20220418,
)
FOLD_B_SPEC: Final = TemporalFoldSpec(
    name="B",
    train_start=20220408,
    train_end=20220418,
    valid_start=20220419,
    valid_end=20220421,
)
FROZEN_FOLD_SPECS: Final = (FOLD_A_SPEC, FOLD_B_SPEC)


def _fold_digest(
    spec: TemporalFoldSpec,
    source_alignment_digest: str,
    train_positions: Sequence[int],
    valid_positions: Sequence[int],
) -> str:
    payload = {
        "schema_version": TEMPORAL_FOLD_SCHEMA_VERSION,
        "source_split": SplitName.TRAIN.value,
        "source_alignment_digest": source_alignment_digest,
        "spec": spec.manifest(),
        "train_positions": list(train_positions),
        "valid_positions": list(valid_positions),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def _strict_positions(values: Sequence[object], name: str) -> tuple[int, ...]:
    positions: list[int] = []
    previous = -1
    for index, raw in enumerate(values):
        if type(raw) is not int or raw < 0:
            raise TemporalFoldError(f"{name}[{index}] must be a non-negative integer")
        if raw <= previous:
            raise TemporalFoldError(f"{name} must be strictly increasing canonical positions")
        positions.append(raw)
        previous = raw
    if not positions:
        raise TemporalFoldError(f"{name} cannot be empty")
    return tuple(positions)


@dataclass(frozen=True, slots=True)
class TemporalFold:
    """One immutable fold expressed as positions in the canonical train split."""

    spec: TemporalFoldSpec
    source_alignment_digest: str
    train_positions: Sequence[int]
    valid_positions: Sequence[int]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.spec not in FROZEN_FOLD_SPECS:
            raise TemporalFoldError("fold spec is not one of the two frozen benchmark folds")
        if len(self.source_alignment_digest) != 64:
            raise TemporalFoldError("source alignment digest must be a SHA-256 hex digest")
        try:
            int(self.source_alignment_digest, 16)
        except ValueError as exc:
            raise TemporalFoldError("source alignment digest is not hexadecimal") from exc
        train = _strict_positions(cast(Sequence[object], self.train_positions), "train_positions")
        valid = _strict_positions(cast(Sequence[object], self.valid_positions), "valid_positions")
        if set(train) & set(valid):
            raise TemporalFoldError("fold train and validation positions overlap")
        object.__setattr__(self, "train_positions", train)
        object.__setattr__(self, "valid_positions", valid)
        object.__setattr__(
            self,
            "digest",
            _fold_digest(self.spec, self.source_alignment_digest, train, valid),
        )

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def train_indices(self) -> tuple[int, ...]:
        """Compatibility alias; values are positions, never user/video keys."""

        return cast(tuple[int, ...], self.train_positions)

    @property
    def valid_indices(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.valid_positions)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_FOLD_SCHEMA_VERSION,
            "source_split": SplitName.TRAIN.value,
            "source_alignment_digest": self.source_alignment_digest,
            "spec": self.spec.manifest(),
            "train_positions": list(self.train_positions),
            "valid_positions": list(self.valid_positions),
            "digest": self.digest,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> Self:
        """Restore a fold only when its complete frozen manifest and digest agree."""

        expected_keys = {
            "schema_version",
            "source_split",
            "source_alignment_digest",
            "spec",
            "train_positions",
            "valid_positions",
            "digest",
        }
        if set(manifest) != expected_keys:
            raise TemporalFoldError("fold manifest has unknown or missing fields")
        if manifest["schema_version"] != TEMPORAL_FOLD_SCHEMA_VERSION:
            raise TemporalFoldError("unsupported temporal-fold schema version")
        if manifest["source_split"] != SplitName.TRAIN.value:
            raise TemporalFoldError("temporal folds must derive from the train split")
        spec_manifest = manifest["spec"]
        if not isinstance(spec_manifest, Mapping):
            raise TemporalFoldError("fold manifest spec must be an object")
        spec = _spec_from_manifest(spec_manifest)
        source_digest = manifest["source_alignment_digest"]
        train_positions = manifest["train_positions"]
        valid_positions = manifest["valid_positions"]
        stored_digest = manifest["digest"]
        if type(source_digest) is not str or type(stored_digest) is not str:
            raise TemporalFoldError("fold manifest digests must be strings")
        if not isinstance(train_positions, list) or not isinstance(valid_positions, list):
            raise TemporalFoldError("fold manifest positions must be arrays")
        restored = cls(
            spec=spec,
            source_alignment_digest=source_digest,
            train_positions=train_positions,
            valid_positions=valid_positions,
        )
        if restored.digest != stored_digest:
            raise TemporalFoldError("fold manifest digest mismatch")
        return restored


def _spec_from_manifest(manifest: Mapping[str, object]) -> TemporalFoldSpec:
    if set(manifest) != {"name", "train", "valid"}:
        raise TemporalFoldError("fold spec manifest has unknown or missing fields")
    train = manifest["train"]
    valid = manifest["valid"]
    name = manifest["name"]
    if type(name) is not str or not isinstance(train, Mapping) or not isinstance(valid, Mapping):
        raise TemporalFoldError("fold spec manifest has invalid field types")
    if set(train) != {"start", "end"} or set(valid) != {"start", "end"}:
        raise TemporalFoldError("fold date interval has unknown or missing fields")
    values = (train["start"], train["end"], valid["start"], valid["end"])
    if any(type(value) is not int for value in values):
        raise TemporalFoldError("fold dates must be integers")
    spec = TemporalFoldSpec(
        name=name,
        train_start=cast(int, values[0]),
        train_end=cast(int, values[1]),
        valid_start=cast(int, values[2]),
        valid_end=cast(int, values[3]),
    )
    if spec not in FROZEN_FOLD_SPECS:
        raise TemporalFoldError("fold spec differs from the frozen benchmark boundary")
    return spec


@dataclass(frozen=True, slots=True)
class TemporalFoldSet:
    """The ordered A/B fold set with a restart-stable logical digest."""

    source_alignment_digest: str
    folds: tuple[TemporalFold, TemporalFold]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if tuple(fold.spec for fold in self.folds) != FROZEN_FOLD_SPECS:
            raise TemporalFoldError("fold set must contain frozen folds A then B")
        if any(fold.source_alignment_digest != self.source_alignment_digest for fold in self.folds):
            raise TemporalFoldError("fold source alignment digests differ")
        encoded = json.dumps(
            {
                "schema_version": TEMPORAL_FOLD_SCHEMA_VERSION,
                "source_split": SplitName.TRAIN.value,
                "source_alignment_digest": self.source_alignment_digest,
                "fold_digests": [fold.digest for fold in self.folds],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        object.__setattr__(self, "digest", hashlib.sha256(encoded).hexdigest())

    @property
    def fold_a(self) -> TemporalFold:
        return self.folds[0]

    @property
    def fold_b(self) -> TemporalFold:
        return self.folds[1]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_FOLD_SCHEMA_VERSION,
            "source_split": SplitName.TRAIN.value,
            "source_alignment_digest": self.source_alignment_digest,
            "folds": [fold.manifest() for fold in self.folds],
            "digest": self.digest,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> Self:
        expected_keys = {
            "schema_version",
            "source_split",
            "source_alignment_digest",
            "folds",
            "digest",
        }
        if set(manifest) != expected_keys:
            raise TemporalFoldError("fold-set manifest has unknown or missing fields")
        if manifest["schema_version"] != TEMPORAL_FOLD_SCHEMA_VERSION:
            raise TemporalFoldError("unsupported temporal-fold schema version")
        if manifest["source_split"] != SplitName.TRAIN.value:
            raise TemporalFoldError("fold set must derive from the train split")
        source_digest = manifest["source_alignment_digest"]
        folds_raw = manifest["folds"]
        stored_digest = manifest["digest"]
        if type(source_digest) is not str or type(stored_digest) is not str:
            raise TemporalFoldError("fold-set manifest digests must be strings")
        if not isinstance(folds_raw, list) or len(folds_raw) != 2:
            raise TemporalFoldError("fold-set manifest must contain exactly two folds")
        if not all(isinstance(value, Mapping) for value in folds_raw):
            raise TemporalFoldError("fold-set entries must be objects")
        restored_folds = tuple(
            TemporalFold.from_manifest(cast(Mapping[str, object], value)) for value in folds_raw
        )
        restored = cls(
            source_alignment_digest=source_digest,
            folds=cast(tuple[TemporalFold, TemporalFold], restored_folds),
        )
        if restored.digest != stored_digest:
            raise TemporalFoldError("fold-set manifest digest mismatch")
        return restored


@dataclass(frozen=True, slots=True)
class FoldSupportPolicy:
    """Predeclared minimum structural and target support for each inner validation fold."""

    min_train_rows: int = 1
    min_valid_rows: int = 2
    min_train_positives: int = 1
    min_valid_positives: int = 1
    min_mixed_label_users: int = 1
    min_valid_slate_depth: int = 2

    def __post_init__(self) -> None:
        values = (
            self.min_train_rows,
            self.min_valid_rows,
            self.min_train_positives,
            self.min_valid_positives,
            self.min_mixed_label_users,
            self.min_valid_slate_depth,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise TemporalFoldError("all fold support minima must be positive integers")


DEFAULT_FOLD_SUPPORT_POLICY: Final = FoldSupportPolicy()


@dataclass(frozen=True, slots=True)
class FoldSupport:
    train_rows: int
    valid_rows: int
    train_positives: int
    valid_positives: int
    mixed_label_users: int
    maximum_valid_slate_depth: int


def _validate_train_date_domain(split: CanonicalTrainingSplit) -> None:
    if any(not 20220408 <= date <= 20220421 for date in split.inputs.date):
        raise TemporalFoldError("temporal-fold source contains a date outside official training")


def _positions_for(
    spec: TemporalFoldSpec, dates: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train: list[int] = []
    valid: list[int] = []
    for position, date in enumerate(dates):
        if spec.train_start <= date <= spec.train_end:
            train.append(position)
        elif spec.valid_start <= date <= spec.valid_end:
            valid.append(position)
    return tuple(train), tuple(valid)


def validate_fold_support(
    split: CanonicalTrainingSplit,
    fold: TemporalFold,
    policy: FoldSupportPolicy = DEFAULT_FOLD_SUPPORT_POLICY,
) -> FoldSupport:
    """Fail closed when a date-correct fold cannot support ranking evaluation.

    These diagnostics are intentionally excluded from fold manifests and digests: changing labels
    cannot change position identity.  The validation step is trusted and applies only to official
    training targets.
    """

    if split.name is not SplitName.TRAIN or not isinstance(split.targets, TrainingTargets):
        raise TemporalFoldError("fold support requires canonical training targets")
    if fold.source_alignment_digest != split.alignment.digest:
        raise TemporalFoldError("fold and canonical train alignment differ")
    labels = split.targets.long_view
    users = split.alignment.user_id
    train_labels = [labels[position] for position in fold.train_positions]
    valid_labels = [labels[position] for position in fold.valid_positions]
    grouped: dict[str, list[int]] = defaultdict(list)
    for position in fold.valid_positions:
        grouped[users[position]].append(labels[position])
    mixed = sum(1 for values in grouped.values() if 0 < sum(values) < len(values))
    slate_counts = Counter(users[position] for position in fold.valid_positions)
    maximum_slate = max(slate_counts.values(), default=0)
    support = FoldSupport(
        train_rows=len(train_labels),
        valid_rows=len(valid_labels),
        train_positives=sum(train_labels),
        valid_positives=sum(valid_labels),
        mixed_label_users=mixed,
        maximum_valid_slate_depth=maximum_slate,
    )
    failures: list[str] = []
    if support.train_rows < policy.min_train_rows:
        failures.append("train rows")
    if support.valid_rows < policy.min_valid_rows:
        failures.append("validation rows")
    if support.train_positives < policy.min_train_positives:
        failures.append("train positives")
    if support.valid_positives < policy.min_valid_positives:
        failures.append("validation positives")
    if support.valid_rows - support.valid_positives < 1:
        failures.append("validation negatives")
    if support.mixed_label_users < policy.min_mixed_label_users:
        failures.append("mixed-label users")
    if support.maximum_valid_slate_depth < policy.min_valid_slate_depth:
        failures.append("validation slate depth")
    if failures:
        raise TemporalFoldError(
            f"fold {fold.name} fails predeclared support: {', '.join(failures)}"
        )
    return support


def build_temporal_folds(
    train_split: CanonicalTrainingSplit,
    *,
    validate_support: bool = True,
    support_policy: FoldSupportPolicy = DEFAULT_FOLD_SUPPORT_POLICY,
) -> TemporalFoldSet:
    """Build frozen A/B positions from canonical dates without consulting target values."""

    if train_split.name is not SplitName.TRAIN:
        raise TemporalFoldError("temporal folds can only be built from the train split")
    _validate_train_date_domain(train_split)
    folds: list[TemporalFold] = []
    for spec in FROZEN_FOLD_SPECS:
        train_positions, valid_positions = _positions_for(spec, train_split.inputs.date)
        fold = TemporalFold(
            spec=spec,
            source_alignment_digest=train_split.alignment.digest,
            train_positions=train_positions,
            valid_positions=valid_positions,
        )
        if validate_support:
            validate_fold_support(train_split, fold, support_policy)
        folds.append(fold)
    return TemporalFoldSet(
        source_alignment_digest=train_split.alignment.digest,
        folds=cast(tuple[TemporalFold, TemporalFold], tuple(folds)),
    )


__all__ = [
    "DEFAULT_FOLD_SUPPORT_POLICY",
    "FOLD_A_SPEC",
    "FOLD_B_SPEC",
    "FROZEN_FOLD_SPECS",
    "TEMPORAL_FOLD_SCHEMA_VERSION",
    "FoldSupport",
    "FoldSupportPolicy",
    "TemporalFold",
    "TemporalFoldError",
    "TemporalFoldSet",
    "TemporalFoldSpec",
    "build_temporal_folds",
    "validate_fold_support",
]
