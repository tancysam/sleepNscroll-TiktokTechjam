"""Canonical, declarative scientific proposals for the autonomous laboratory.

``ExperimentSpec`` is the only proposal language consumed by the new search path.  It deliberately
contains no source-code, filesystem, URL, provider, or protected-score field.  Provider prose lives
in :class:`MechanismMetadata`; changing that prose cannot change semantic experiment identity.
Every executable choice is instead selected from a bounded allowlist and validated at construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Self, cast

if TYPE_CHECKING:
    from kuairand_agent.domain.identity import ExperimentId, FamilyId

EXPERIMENT_SPEC_SCHEMA_VERSION: Final = 2
_IDENTIFIER_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")

type ParameterValue = str | int | float | bool


class ExperimentSpecError(ValueError):
    """A scientific proposal is outside the frozen declarative language."""


class FeatureViewId(StrEnum):
    """Trusted feature views which trainers may materialize."""

    OFFICIAL_FM_SPARSE = "official_fm_sparse_v1"
    LEAK_SAFE_BASE = "leak_safe_base_v1"
    STRICT_PAST_RECENCY = "strict_past_recency_v1"
    STRICT_PAST_FREQUENCY = "strict_past_frequency_v1"
    FROZEN_PREDICTIONS = "frozen_prediction_vectors_v1"


class TrainingTarget(StrEnum):
    """Native and explicitly reviewed training targets.

    Final-period and public-validation outcomes are intentionally not members of this enum.
    """

    LONG_VIEW = "long_view"


class AuxiliaryTarget(StrEnum):
    """Same-impression auxiliary targets which remain subject to data-capability policy."""

    CLICK = "is_click"
    LIKE = "is_like"
    FOLLOW = "is_follow"
    COMMENT = "is_comment"
    FORWARD = "is_forward"
    HATE = "is_hate"
    PLAY_TIME_MS = "play_time_ms"
    PROFILE_STAY_TIME = "profile_stay_time"
    COMMENT_STAY_TIME = "comment_stay_time"
    PROFILE_ENTER = "is_profile_enter"


class ModelFamily(StrEnum):
    OFFICIAL_FM = "official_fm"
    LIGHTGBM_LAMBDARANK = "lightgbm_lambdarank"
    LIGHTGBM_POINTWISE = "lightgbm_pointwise"
    PAIRWISE_FM = "pairwise_fm"
    EXACT_RANK_FUSION = "exact_rank_fusion"


class Objective(StrEnum):
    OFFICIAL_POINTWISE_LOGLOSS = "official_pointwise_logloss"
    LAMBDARANK_NDCG5 = "lambdarank_ndcg5"
    POINTWISE_BINARY_LOGLOSS = "pointwise_binary_logloss"
    PAIRWISE_LOGISTIC = "pairwise_logistic"
    EXACT_WEIGHTED_RANK = "exact_weighted_rank"


class GroupingUnit(StrEnum):
    USER = "user"
    IMPRESSION = "impression"


class FoldProtocol(StrEnum):
    TEMPORAL_AB = "temporal_ab_v1"
    FULL_OFFICIAL_TRAIN = "full_official_train_v1"


class ScreeningFidelity(StrEnum):
    SMOKE = "smoke"
    REDUCED = "reduced"
    FOLD_B = "fold_b"
    FOLD_A = "fold_a"
    MULTI_SEED = "multi_seed"
    PORTFOLIO = "portfolio"


class ResourceClass(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class RequiredAblation(StrEnum):
    MATCHED_PARENT = "matched_parent"
    NO_STRICT_PAST_FEATURES = "no_strict_past_features"
    POINTWISE_OBJECTIVE = "pointwise_objective"
    PRIMARY_TARGET_ONLY = "primary_target_only"
    SHUFFLED_AUXILIARY = "shuffled_auxiliary"
    SINGLE_MEMBER = "single_member"


class TemporalEntity(StrEnum):
    USER = "user"
    VIDEO = "video"
    USER_AUTHOR = "user_author"


class RankTransform(StrEnum):
    USER_PERCENTILE = "user_percentile"


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
        raise ExperimentSpecError(f"experiment value is not canonical JSON: {exc}") from exc


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _identifier(value: object, location: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ExperimentSpecError(f"{location} must be a portable lowercase identifier")
    return value


def _digest_or_identifier(value: object, location: str) -> str:
    result = _identifier(value, location)
    if (
        result.startswith("sha256:")
        and _SHA256_RE.fullmatch(result.removeprefix("sha256:")) is None
    ):
        raise ExperimentSpecError(f"{location} has an invalid SHA-256 reference")
    return result


def _bounded_text(value: object, location: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value or len(value) > maximum:
        raise ExperimentSpecError(
            f"{location} must be non-empty text without NUL and at most {maximum} characters"
        )
    return value


def _exact_fields(raw: Mapping[str, object], expected: frozenset[str], location: str) -> None:
    unknown = frozenset(raw) - expected
    missing = expected - frozenset(raw)
    if unknown:
        raise ExperimentSpecError(f"unknown {location} field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ExperimentSpecError(f"missing {location} field(s): {', '.join(sorted(missing))}")


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ExperimentSpecError(f"{location} must be an object")
    return cast(Mapping[str, object], value)


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise ExperimentSpecError(f"{location} must be an array")
    return value


def _enum_member[E: StrEnum](enum_type: type[E], value: object, location: str) -> E:
    if type(value) is not str:
        raise ExperimentSpecError(f"{location} must be text")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ExperimentSpecError(f"{location} is not allowlisted: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class MechanismMetadata:
    """Advisory prose and attribution, excluded from semantic experiment identity."""

    mechanism: str
    falsifiable_hypothesis: str
    expected_metric_effect: str
    leakage_argument: str
    rejection_criterion: str
    attributions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "mechanism",
            "falsifiable_hypothesis",
            "expected_metric_effect",
            "leakage_argument",
            "rejection_criterion",
        ):
            _bounded_text(getattr(self, name), f"metadata.{name}")
        if type(self.attributions) is not tuple or len(self.attributions) > 16:
            raise ExperimentSpecError("metadata.attributions must be a tuple of at most 16 entries")
        for index, attribution in enumerate(self.attributions):
            _bounded_text(attribution, f"metadata.attributions[{index}]", maximum=1024)

    def manifest(self) -> dict[str, object]:
        return {
            "mechanism": self.mechanism,
            "falsifiable_hypothesis": self.falsifiable_hypothesis,
            "expected_metric_effect": self.expected_metric_effect,
            "leakage_argument": self.leakage_argument,
            "rejection_criterion": self.rejection_criterion,
            "attributions": list(self.attributions),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        fields = frozenset(
            {
                "mechanism",
                "falsifiable_hypothesis",
                "expected_metric_effect",
                "leakage_argument",
                "rejection_criterion",
                "attributions",
            }
        )
        _exact_fields(raw, fields, "metadata")
        attributions = tuple(
            _bounded_text(value, f"metadata.attributions[{index}]", maximum=1024)
            for index, value in enumerate(_array(raw["attributions"], "metadata.attributions"))
        )
        return cls(
            mechanism=_bounded_text(raw["mechanism"], "metadata.mechanism"),
            falsifiable_hypothesis=_bounded_text(
                raw["falsifiable_hypothesis"], "metadata.falsifiable_hypothesis"
            ),
            expected_metric_effect=_bounded_text(
                raw["expected_metric_effect"], "metadata.expected_metric_effect"
            ),
            leakage_argument=_bounded_text(raw["leakage_argument"], "metadata.leakage_argument"),
            rejection_criterion=_bounded_text(
                raw["rejection_criterion"], "metadata.rejection_criterion"
            ),
            attributions=attributions,
        )


@dataclass(frozen=True, slots=True)
class StrictPastPolicy:
    """A declarative temporal feature policy; current-row outcomes are unrepresentable."""

    entities: tuple[TemporalEntity, ...] = ()
    lookback_windows_ms: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.entities) is not tuple or any(
            not isinstance(value, TemporalEntity) for value in self.entities
        ):
            raise ExperimentSpecError("strict-past entities must be an allowlisted tuple")
        if len(self.entities) != len(set(self.entities)):
            raise ExperimentSpecError("strict-past entities contain duplicates")
        if type(self.lookback_windows_ms) is not tuple or any(
            type(value) is not int or value <= 0 or value > 365 * 24 * 60 * 60 * 1000
            for value in self.lookback_windows_ms
        ):
            raise ExperimentSpecError("strict-past windows must be bounded positive integer ms")
        if tuple(sorted(set(self.lookback_windows_ms))) != self.lookback_windows_ms:
            raise ExperimentSpecError("strict-past windows must be sorted and unique")
        if bool(self.entities) != bool(self.lookback_windows_ms):
            raise ExperimentSpecError("strict-past entities and windows must be enabled together")

    @property
    def enabled(self) -> bool:
        return bool(self.entities)

    def manifest(self) -> dict[str, object]:
        return {
            "policy": "strict_past_only",
            "entities": [value.value for value in self.entities],
            "lookback_windows_ms": list(self.lookback_windows_ms),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _exact_fields(raw, frozenset({"policy", "entities", "lookback_windows_ms"}), "strict_past")
        if raw["policy"] != "strict_past_only":
            raise ExperimentSpecError("strict_past.policy must be 'strict_past_only'")
        entities = tuple(
            _enum_member(TemporalEntity, item, f"strict_past.entities[{index}]")
            for index, item in enumerate(_array(raw["entities"], "strict_past.entities"))
        )
        windows = tuple(
            cast(int, item)
            for item in _array(raw["lookback_windows_ms"], "strict_past.lookback_windows_ms")
        )
        return cls(entities=entities, lookback_windows_ms=windows)


@dataclass(frozen=True, slots=True)
class RankFusionMember:
    """One frozen prediction input to an exact weighted rank recipe."""

    prediction_ref: str
    weight: float
    transform: RankTransform = RankTransform.USER_PERCENTILE

    def __post_init__(self) -> None:
        _digest_or_identifier(self.prediction_ref, "rank member prediction_ref")
        if type(self.weight) is not float or not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ExperimentSpecError("rank member weight must be a finite positive float")
        if not isinstance(self.transform, RankTransform):
            raise ExperimentSpecError("rank member transform is not allowlisted")

    def manifest(self) -> dict[str, object]:
        return {
            "prediction_ref": self.prediction_ref,
            "weight": self.weight,
            "transform": self.transform.value,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _exact_fields(raw, frozenset({"prediction_ref", "weight", "transform"}), "rank member")
        weight = raw["weight"]
        if type(weight) is not float:
            raise ExperimentSpecError("rank member weight must be a JSON float")
        return cls(
            prediction_ref=_digest_or_identifier(raw["prediction_ref"], "rank member reference"),
            weight=weight,
            transform=_enum_member(RankTransform, raw["transform"], "rank member transform"),
        )


@dataclass(frozen=True, slots=True)
class RankFusionRecipe:
    """Fully specified label-free composition included in semantic identity."""

    members: tuple[RankFusionMember, ...]
    tie_policy: str = "canonical_row_order"

    def __post_init__(self) -> None:
        if type(self.members) is not tuple or len(self.members) < 2:
            raise ExperimentSpecError("rank fusion requires at least two members")
        if any(not isinstance(value, RankFusionMember) for value in self.members):
            raise ExperimentSpecError("rank fusion contains an invalid member")
        references = tuple(value.prediction_ref for value in self.members)
        if len(references) != len(set(references)):
            raise ExperimentSpecError("rank fusion member references must be unique")
        if not math.isclose(sum(value.weight for value in self.members), 1.0, abs_tol=1e-12):
            raise ExperimentSpecError("rank fusion member weights must sum to one")
        if self.tie_policy != "canonical_row_order":
            raise ExperimentSpecError("rank fusion tie policy is not allowlisted")

    def manifest(self) -> dict[str, object]:
        return {
            "members": [value.manifest() for value in self.members],
            "tie_policy": self.tie_policy,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _exact_fields(raw, frozenset({"members", "tie_policy"}), "rank fusion")
        members = tuple(
            RankFusionMember.from_mapping(_mapping(item, f"rank_fusion.members[{index}]"))
            for index, item in enumerate(_array(raw["members"], "rank_fusion.members"))
        )
        return cls(
            members=members,
            tie_policy=_bounded_text(raw["tie_policy"], "rank_fusion.tie_policy", maximum=64),
        )


_OBJECTIVE_BY_MODEL: Final[Mapping[ModelFamily, frozenset[Objective]]] = MappingProxyType(
    {
        ModelFamily.OFFICIAL_FM: frozenset({Objective.OFFICIAL_POINTWISE_LOGLOSS}),
        ModelFamily.LIGHTGBM_LAMBDARANK: frozenset({Objective.LAMBDARANK_NDCG5}),
        ModelFamily.LIGHTGBM_POINTWISE: frozenset({Objective.POINTWISE_BINARY_LOGLOSS}),
        ModelFamily.PAIRWISE_FM: frozenset({Objective.PAIRWISE_LOGISTIC}),
        ModelFamily.EXACT_RANK_FUSION: frozenset({Objective.EXACT_WEIGHTED_RANK}),
    }
)

_GROUPING_BY_MODEL: Final[Mapping[ModelFamily, frozenset[GroupingUnit]]] = MappingProxyType(
    {
        ModelFamily.OFFICIAL_FM: frozenset({GroupingUnit.IMPRESSION}),
        ModelFamily.LIGHTGBM_LAMBDARANK: frozenset({GroupingUnit.USER}),
        ModelFamily.LIGHTGBM_POINTWISE: frozenset({GroupingUnit.IMPRESSION}),
        ModelFamily.PAIRWISE_FM: frozenset({GroupingUnit.USER}),
        ModelFamily.EXACT_RANK_FUSION: frozenset({GroupingUnit.USER}),
    }
)

_PARAMETER_BOUNDS: Final[Mapping[ModelFamily, Mapping[str, tuple[type[object], float, float]]]] = (
    MappingProxyType(
        {
            ModelFamily.OFFICIAL_FM: MappingProxyType(
                {
                    "k": (int, 1, 512),
                    "max_epochs": (int, 1, 100),
                    "learning_rate": (float, 1e-6, 1.0),
                    "l2": (float, 0.0, 1_000_000.0),
                    "batch_size": (int, 1, 10_000_000),
                    "patience": (int, 1, 100),
                    "improvement_threshold": (float, 0.0, 1.0),
                    "predict_batch_size": (int, 1, 10_000_000),
                }
            ),
            ModelFamily.LIGHTGBM_LAMBDARANK: MappingProxyType(
                {
                    "num_boost_round": (int, 1, 10_000),
                    "early_stopping_rounds": (int, 1, 10_000),
                    "learning_rate": (float, 1e-6, 1.0),
                    "num_leaves": (int, 2, 4096),
                    "min_data_in_leaf": (int, 1, 10_000_000),
                    "lambda_l2": (float, 0.0, 1_000_000.0),
                    "lambdarank_truncation_level": (int, 6, 1000),
                }
            ),
            ModelFamily.LIGHTGBM_POINTWISE: MappingProxyType(
                {
                    "num_boost_round": (int, 1, 10_000),
                    "early_stopping_rounds": (int, 1, 10_000),
                    "learning_rate": (float, 1e-6, 1.0),
                    "num_leaves": (int, 2, 4096),
                    "min_data_in_leaf": (int, 1, 10_000_000),
                    "lambda_l2": (float, 0.0, 1_000_000.0),
                }
            ),
            ModelFamily.PAIRWISE_FM: MappingProxyType(
                {
                    "factors": (int, 1, 512),
                    "epochs": (int, 1, 100),
                    "learning_rate": (float, 1e-6, 1.0),
                    "regularization": (float, 0.0, 1_000_000.0),
                }
            ),
            ModelFamily.EXACT_RANK_FUSION: MappingProxyType({}),
        }
    )
)


def _freeze_parameters(
    model_family: ModelFamily, raw: Mapping[str, ParameterValue]
) -> Mapping[str, ParameterValue]:
    bounds = _PARAMETER_BOUNDS[model_family]
    unknown = frozenset(raw) - frozenset(bounds)
    if unknown:
        raise ExperimentSpecError(
            f"unsupported {model_family.value} hyperparameter(s): {', '.join(sorted(unknown))}"
        )
    normalized: dict[str, ParameterValue] = {}
    for name, value in raw.items():
        expected_type, minimum, maximum = bounds[name]
        if type(value) is not expected_type:
            raise ExperimentSpecError(
                f"hyperparameter {name} must have exact type {expected_type.__name__}"
            )
        numeric = cast(float | int, value)
        if isinstance(numeric, float) and not math.isfinite(numeric):
            raise ExperimentSpecError(f"hyperparameter {name} must be finite")
        if not minimum <= numeric <= maximum:
            raise ExperimentSpecError(f"hyperparameter {name} is outside its frozen bounds")
        normalized[name] = value
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Schema-versioned, prose-insensitive, declarative experiment definition."""

    proposal_key: str
    metadata: MechanismMetadata = field(compare=False, hash=False, repr=False)
    parent_experiment_refs: tuple[str, ...]
    fallback_prediction_ref: str
    feature_view_ids: tuple[FeatureViewId, ...]
    strict_past: StrictPastPolicy
    training_target: TrainingTarget
    auxiliary_targets: tuple[AuxiliaryTarget, ...]
    model_family: ModelFamily
    hyperparameters: Mapping[str, ParameterValue]
    objective: Objective
    grouping_unit: GroupingUnit
    fold_protocol: FoldProtocol
    seeds: tuple[int, ...]
    screening_fidelity: ScreeningFidelity
    required_ablations: tuple[RequiredAblation, ...]
    resource_class: ResourceClass
    promotion_policy_version: str
    rank_fusion: RankFusionRecipe | None = None
    schema_version: int = EXPERIMENT_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_SPEC_SCHEMA_VERSION:
            raise ExperimentSpecError(
                f"ExperimentSpec.schema_version must be {EXPERIMENT_SPEC_SCHEMA_VERSION}"
            )
        _identifier(self.proposal_key, "proposal_key")
        if not isinstance(self.metadata, MechanismMetadata):
            raise ExperimentSpecError("metadata must be MechanismMetadata")
        self._validate_identifier_tuple(self.parent_experiment_refs, "parent_experiment_refs", 8)
        _digest_or_identifier(self.fallback_prediction_ref, "fallback_prediction_ref")
        self._validate_enum_tuple(self.feature_view_ids, FeatureViewId, "feature_view_ids", 8)
        if not self.feature_view_ids:
            raise ExperimentSpecError("feature_view_ids cannot be empty")
        if not isinstance(self.strict_past, StrictPastPolicy):
            raise ExperimentSpecError("strict_past must be StrictPastPolicy")
        self._validate_enum_tuple(self.auxiliary_targets, AuxiliaryTarget, "auxiliary_targets", 10)
        if not isinstance(self.training_target, TrainingTarget):
            raise ExperimentSpecError("training_target is not allowlisted")
        if self.training_target is not TrainingTarget.LONG_VIEW:
            raise ExperimentSpecError("the native long_view target is mandatory")
        if not isinstance(self.model_family, ModelFamily):
            raise ExperimentSpecError("model_family is not allowlisted")
        if (
            not isinstance(self.objective, Objective)
            or self.objective not in _OBJECTIVE_BY_MODEL[self.model_family]
        ):
            raise ExperimentSpecError("objective is not compatible with model_family")
        if (
            not isinstance(self.grouping_unit, GroupingUnit)
            or self.grouping_unit not in _GROUPING_BY_MODEL[self.model_family]
        ):
            raise ExperimentSpecError("grouping_unit is not compatible with model_family")
        if not isinstance(self.fold_protocol, FoldProtocol):
            raise ExperimentSpecError("fold_protocol is not allowlisted")
        if type(self.seeds) is not tuple or not self.seeds or len(self.seeds) > 8:
            raise ExperimentSpecError("seeds must be a non-empty tuple of at most eight values")
        if any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in self.seeds):
            raise ExperimentSpecError("seeds must be uint32-compatible integers")
        if len(self.seeds) != len(set(self.seeds)):
            raise ExperimentSpecError("seeds contain duplicates")
        self._validate_enum_tuple(
            self.required_ablations, RequiredAblation, "required_ablations", 8
        )
        if not isinstance(self.screening_fidelity, ScreeningFidelity):
            raise ExperimentSpecError("screening_fidelity is not allowlisted")
        if not isinstance(self.resource_class, ResourceClass):
            raise ExperimentSpecError("resource_class is not allowlisted")
        _identifier(self.promotion_policy_version, "promotion_policy_version")
        frozen_parameters = _freeze_parameters(self.model_family, self.hyperparameters)
        object.__setattr__(self, "hyperparameters", frozen_parameters)
        uses_strict_past_view = bool(
            {FeatureViewId.STRICT_PAST_RECENCY, FeatureViewId.STRICT_PAST_FREQUENCY}
            & set(self.feature_view_ids)
        )
        if uses_strict_past_view != self.strict_past.enabled:
            raise ExperimentSpecError(
                "strict-past feature views and strict-past policy must be enabled together"
            )
        if self.model_family is ModelFamily.EXACT_RANK_FUSION:
            if self.rank_fusion is None:
                raise ExperimentSpecError("exact rank fusion requires a rank_fusion recipe")
            if self.hyperparameters:
                raise ExperimentSpecError("exact rank fusion cannot carry trainer hyperparameters")
        elif self.rank_fusion is not None:
            raise ExperimentSpecError("rank_fusion is valid only for exact_rank_fusion")

    @staticmethod
    def _validate_identifier_tuple(values: object, location: str, maximum: int) -> None:
        if type(values) is not tuple or len(values) > maximum:
            raise ExperimentSpecError(f"{location} must be a tuple of at most {maximum} values")
        for index, value in enumerate(values):
            _digest_or_identifier(value, f"{location}[{index}]")
        if len(values) != len(set(cast(tuple[object, ...], values))):
            raise ExperimentSpecError(f"{location} contains duplicates")

    @staticmethod
    def _validate_enum_tuple[E: StrEnum](
        values: object, enum_type: type[E], location: str, maximum: int
    ) -> None:
        if type(values) is not tuple or len(values) > maximum:
            raise ExperimentSpecError(f"{location} must be a tuple of at most {maximum} values")
        typed_values = cast(tuple[object, ...], values)
        if any(not isinstance(value, enum_type) for value in typed_values):
            raise ExperimentSpecError(f"{location} contains a value outside its allowlist")
        if len(typed_values) != len(set(typed_values)):
            raise ExperimentSpecError(f"{location} contains duplicates")

    def semantic_manifest(self) -> dict[str, object]:
        """Return executable science only; no prose, attribution, or provider state."""

        return {
            "schema_version": self.schema_version,
            "parent_experiment_refs": list(self.parent_experiment_refs),
            "fallback_prediction_ref": self.fallback_prediction_ref,
            "feature_view_ids": [value.value for value in self.feature_view_ids],
            "strict_past": self.strict_past.manifest(),
            "training_target": self.training_target.value,
            "auxiliary_targets": [value.value for value in self.auxiliary_targets],
            "model_family": self.model_family.value,
            "hyperparameters": dict(self.hyperparameters),
            "objective": self.objective.value,
            "grouping_unit": self.grouping_unit.value,
            "fold_protocol": self.fold_protocol.value,
            "seeds": list(self.seeds),
            "screening_fidelity": self.screening_fidelity.value,
            "required_ablations": [value.value for value in self.required_ablations],
            "resource_class": self.resource_class.value,
            "promotion_policy_version": self.promotion_policy_version,
            "rank_fusion": self.rank_fusion.manifest() if self.rank_fusion is not None else None,
        }

    def family_manifest(self) -> dict[str, object]:
        """Return the cross-campaign scientific family key, excluding seed and capacity."""

        return {
            "schema_version": self.schema_version,
            "feature_view_ids": [value.value for value in self.feature_view_ids],
            "strict_past_policy": self.strict_past.manifest(),
            "training_target": self.training_target.value,
            "auxiliary_targets": [value.value for value in self.auxiliary_targets],
            "model_family": self.model_family.value,
            "objective": self.objective.value,
            "grouping_unit": self.grouping_unit.value,
            "fusion_member_count": len(self.rank_fusion.members) if self.rank_fusion else 0,
        }

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "proposal_key": self.proposal_key,
            "metadata": self.metadata.manifest(),
            "semantic": self.semantic_manifest(),
        }

    @property
    def semantic_digest(self) -> str:
        return _digest(b"kuairand-experiment-spec-v2\0", self.semantic_manifest())

    @property
    def family_digest(self) -> str:
        return _digest(b"kuairand-family-v1\0", self.family_manifest())

    @property
    def family_id(self) -> FamilyId:
        """Return the nominal cross-campaign identity without importing identity at module load."""

        from kuairand_agent.domain.identity import FamilyId

        temporal_family = json.dumps(
            self.strict_past.manifest(), sort_keys=True, separators=(",", ":")
        )
        return FamilyId.derive(
            mechanism=f"{self.model_family.value}|{temporal_family}",
            feature_family=",".join(value.value for value in self.feature_view_ids),
            target_family=",".join(
                (
                    self.training_target.value,
                    *(value.value for value in self.auxiliary_targets),
                )
            ),
            objective_family=f"{self.objective.value}|{self.grouping_unit.value}",
        )

    def derive_experiment_id(
        self,
        *,
        data_identities: Mapping[str, str],
        fold_identities: Mapping[str, str],
        code_artifact_sha256: str,
    ) -> ExperimentId:
        """Bind semantic science to exact data, fold, and trusted-code identities."""

        from kuairand_agent.domain.identity import ExperimentId

        return ExperimentId.derive(
            experiment_spec=self.semantic_manifest(),
            data_identities=data_identities,
            fold_identities=fold_identities,
            code_artifact_sha256=code_artifact_sha256,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.manifest())

    def with_metadata(self, metadata: MechanismMetadata) -> Self:
        """Replace prose without changing semantic or family identity."""

        return replace(self, metadata=metadata)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        """Parse strict canonical wire data and reject free-form/unknown executable fields."""

        _exact_fields(
            raw,
            frozenset({"schema_version", "proposal_key", "metadata", "semantic"}),
            "ExperimentSpec",
        )
        if raw["schema_version"] != EXPERIMENT_SPEC_SCHEMA_VERSION:
            raise ExperimentSpecError(
                f"ExperimentSpec.schema_version must be {EXPERIMENT_SPEC_SCHEMA_VERSION}"
            )
        semantic = _mapping(raw["semantic"], "ExperimentSpec.semantic")
        semantic_fields = frozenset(
            {
                "schema_version",
                "parent_experiment_refs",
                "fallback_prediction_ref",
                "feature_view_ids",
                "strict_past",
                "training_target",
                "auxiliary_targets",
                "model_family",
                "hyperparameters",
                "objective",
                "grouping_unit",
                "fold_protocol",
                "seeds",
                "screening_fidelity",
                "required_ablations",
                "resource_class",
                "promotion_policy_version",
                "rank_fusion",
            }
        )
        _exact_fields(semantic, semantic_fields, "ExperimentSpec.semantic")
        if semantic["schema_version"] != EXPERIMENT_SPEC_SCHEMA_VERSION:
            raise ExperimentSpecError("semantic schema_version differs from envelope")
        model_family = _enum_member(ModelFamily, semantic["model_family"], "semantic.model_family")
        hyperparameters_raw = _mapping(semantic["hyperparameters"], "semantic.hyperparameters")
        parameters: dict[str, ParameterValue] = {}
        for name, value in hyperparameters_raw.items():
            if type(value) not in {str, int, float, bool}:
                raise ExperimentSpecError(f"hyperparameter {name} is not a JSON scalar")
            parameters[name] = cast(ParameterValue, value)
        rank_fusion_raw = semantic["rank_fusion"]
        rank_fusion = (
            None
            if rank_fusion_raw is None
            else RankFusionRecipe.from_mapping(_mapping(rank_fusion_raw, "semantic.rank_fusion"))
        )
        return cls(
            schema_version=EXPERIMENT_SPEC_SCHEMA_VERSION,
            proposal_key=_identifier(raw["proposal_key"], "proposal_key"),
            metadata=MechanismMetadata.from_mapping(_mapping(raw["metadata"], "metadata")),
            parent_experiment_refs=tuple(
                _digest_or_identifier(item, f"semantic.parent_experiment_refs[{index}]")
                for index, item in enumerate(
                    _array(semantic["parent_experiment_refs"], "semantic.parent_experiment_refs")
                )
            ),
            fallback_prediction_ref=_digest_or_identifier(
                semantic["fallback_prediction_ref"], "semantic.fallback_prediction_ref"
            ),
            feature_view_ids=tuple(
                _enum_member(FeatureViewId, item, f"semantic.feature_view_ids[{index}]")
                for index, item in enumerate(
                    _array(semantic["feature_view_ids"], "semantic.feature_view_ids")
                )
            ),
            strict_past=StrictPastPolicy.from_mapping(
                _mapping(semantic["strict_past"], "semantic.strict_past")
            ),
            training_target=_enum_member(
                TrainingTarget, semantic["training_target"], "semantic.training_target"
            ),
            auxiliary_targets=tuple(
                _enum_member(AuxiliaryTarget, item, f"semantic.auxiliary_targets[{index}]")
                for index, item in enumerate(
                    _array(semantic["auxiliary_targets"], "semantic.auxiliary_targets")
                )
            ),
            model_family=model_family,
            hyperparameters=parameters,
            objective=_enum_member(Objective, semantic["objective"], "semantic.objective"),
            grouping_unit=_enum_member(
                GroupingUnit, semantic["grouping_unit"], "semantic.grouping_unit"
            ),
            fold_protocol=_enum_member(
                FoldProtocol, semantic["fold_protocol"], "semantic.fold_protocol"
            ),
            seeds=tuple(cast(int, item) for item in _array(semantic["seeds"], "semantic.seeds")),
            screening_fidelity=_enum_member(
                ScreeningFidelity,
                semantic["screening_fidelity"],
                "semantic.screening_fidelity",
            ),
            required_ablations=tuple(
                _enum_member(RequiredAblation, item, f"semantic.required_ablations[{index}]")
                for index, item in enumerate(
                    _array(semantic["required_ablations"], "semantic.required_ablations")
                )
            ),
            resource_class=_enum_member(
                ResourceClass, semantic["resource_class"], "semantic.resource_class"
            ),
            promotion_policy_version=_identifier(
                semantic["promotion_policy_version"], "semantic.promotion_policy_version"
            ),
            rank_fusion=rank_fusion,
        )


def parse_experiment_spec(payload: bytes | str) -> ExperimentSpec:
    """Strictly parse one wire spec, rejecting duplicate JSON keys and non-canonical data."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ExperimentSpecError("ExperimentSpec JSON must be ASCII") from exc
    elif type(payload) is str:
        text = payload
    else:
        raise ExperimentSpecError("ExperimentSpec payload must be bytes or text")

    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ExperimentSpecError(f"duplicate ExperimentSpec key: {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExperimentSpecError(f"non-finite JSON number is forbidden: {value}")
            ),
        )
    except ExperimentSpecError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ExperimentSpecError(f"invalid ExperimentSpec JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ExperimentSpecError("ExperimentSpec JSON must be an object")
    result = ExperimentSpec.from_mapping(cast(Mapping[str, object], decoded))
    if _canonical_json(decoded) != text.encode("ascii"):
        raise ExperimentSpecError("ExperimentSpec JSON is not canonical")
    return result
