"""Pinned organizer and benchmark identities owned by the trusted controller."""

from __future__ import annotations

import hashlib
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Self

from kuairand_agent.domain.identity import (
    ContractId,
    IdentityError,
    canonical_json_bytes,
    canonical_json_sha256,
)

STARTER_FILE_SHA256: Final[Mapping[str, str]] = {
    "README.md": "c7a58e652a1aceea144e651ba9ef7a6a4f7dc13f0916e3c4ed342dce69699861",
    "ablation_features.py": "944ff3003451d82cd4694dd2ac0a7a587e53890956cb098f8daa04537d97b457",
    "baseline.py": "c8f7fc60178413e247e78bb231e7550eeef52101b6493fcf1a4d2b0e5fe18f8a",
    "baseline_scores.json": "950f98181770c030a68bdddab7be3c0abbf060531f54455a6a6f81a4cb003324",
    "data.py": "1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541",
    "evaluate.py": "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de",
    "submit.py": "ab01bb2b970ae2a9f2ead299f5240b71ff4126c2d9bb0e0c4de6c7e245dc148c",
}
STARTER_ZIP_SHA256: Final = "07237e62cc1a9cd8278556dab995dd5388516f10772724f582ef8320ac68b10b"
DATASET_ARCHIVE_MD5: Final = "0820331067a3784d9691136f772b35a7"
DATASET_ARCHIVE_SHA256: Final = "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"


class OrganizerIntegrityError(RuntimeError):
    """Raised when immutable organizer artifacts differ from their pinned manifest."""


@dataclass(frozen=True, slots=True)
class StarterVerification:
    root: Path
    files: Mapping[str, str]
    manifest_sha256: str


class BenchmarkContractError(ValueError):
    """Raised when a benchmark identity differs from the frozen organizer contract."""


class SplitName(StrEnum):
    """Organizer split names in physical loading order."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitContract:
    """Inclusive date interval and outcome-access policy for one organizer split."""

    name: SplitName
    start_date: int
    end_date: int
    development_outcome_access: str

    def manifest(self) -> dict[str, object]:
        """Return this split's stable JSON representation."""

        return {
            "name": self.name.value,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "development_outcome_access": self.development_outcome_access,
        }


@dataclass(frozen=True, slots=True)
class RankingTaskContract:
    """The operative prediction task, independent of stale challenge prose."""

    dataset: str
    target: str
    relevance: str
    ranking_unit: str

    def manifest(self) -> dict[str, str]:
        """Return this task's stable JSON representation."""

        return {
            "dataset": self.dataset,
            "target": self.target,
            "relevance": self.relevance,
            "ranking_unit": self.ranking_unit,
        }


@dataclass(frozen=True, slots=True)
class MetricContract:
    """Exact organizer metric semantics used by the protected scorer."""

    gauc_name: str
    gauc_user_eligibility: str
    gauc_weight: str
    ndcg_name: str
    ndcg_cutoff: int
    ndcg_user_eligibility: str
    ndcg_zero_positive_value: float
    ndcg_gain: str
    primary_formula: str

    def manifest(self) -> dict[str, object]:
        """Return these metric semantics in a stable JSON representation."""

        return {
            "gauc": {
                "name": self.gauc_name,
                "user_eligibility": self.gauc_user_eligibility,
                "weight": self.gauc_weight,
            },
            "ndcg": {
                "name": self.ndcg_name,
                "cutoff": self.ndcg_cutoff,
                "user_eligibility": self.ndcg_user_eligibility,
                "zero_positive_value": self.ndcg_zero_positive_value,
                "gain": self.ndcg_gain,
            },
            "primary": self.primary_formula,
        }


@dataclass(frozen=True, slots=True)
class PublishedScore:
    """A published, four-decimal public-validation score triplet."""

    gauc: float
    ndcg_at_5: float
    primary: float

    def validate(self) -> None:
        """Require finite unit-interval metrics and the organizer primary formula."""

        values = (self.gauc, self.ndcg_at_5, self.primary)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise BenchmarkContractError("published reference metrics must be finite in [0, 1]")
        expected = ((Decimal(str(self.gauc)) + Decimal(str(self.ndcg_at_5))) / Decimal(2)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        if Decimal(str(self.primary)) != expected:
            raise BenchmarkContractError(
                "published primary must be the four-decimal mean of GAUC and nDCG@5"
            )

    def manifest(self) -> dict[str, float]:
        """Return the score names used by the organizer evaluator."""

        return {
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
        }


@dataclass(frozen=True, slots=True)
class ReferenceRung:
    """One published public-validation qualification target."""

    name: str
    validation: PublishedScore
    qualification: str

    def manifest(self) -> dict[str, object]:
        """Return this rung without exposing final-period outcomes."""

        return {
            "name": self.name,
            "split": SplitName.VALID.value,
            "scores": self.validation.manifest(),
            "qualification": self.qualification,
        }


@dataclass(frozen=True, slots=True)
class ConvergenceContract:
    """Frozen scientific-iteration convergence policy."""

    epsilon: float
    patience: int
    comparison: str

    def manifest(self) -> dict[str, object]:
        """Return the exact policy used by the campaign controller."""

        return {
            "epsilon": self.epsilon,
            "patience": self.patience,
            "comparison": self.comparison,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkContract:
    """Complete immutable scientific identity for KuaiRand-Pure development."""

    schema_version: int
    task: RankingTaskContract
    splits: tuple[SplitContract, ...]
    metrics: MetricContract
    reference_rungs: tuple[ReferenceRung, ...]
    submission_header: tuple[str, ...]
    row_identity: str
    convergence: ConvergenceContract

    def validate(self) -> Self:
        """Reject any drift from the hash-pinned operative benchmark."""

        if self.schema_version != 1:
            raise BenchmarkContractError("benchmark contract schema_version must be 1")
        if self.task != _FROZEN_TASK:
            raise BenchmarkContractError("operative task must remain native long_view ranking")
        if self.splits != _FROZEN_SPLITS:
            raise BenchmarkContractError("organizer split dates or outcome policy changed")
        if self.metrics != _FROZEN_METRICS:
            raise BenchmarkContractError("organizer metric semantics changed")
        if self.reference_rungs != _FROZEN_REFERENCE_RUNGS:
            raise BenchmarkContractError("published validation reference rungs changed")
        for rung in self.reference_rungs:
            rung.validation.validate()
        if self.submission_header != ("row_id", "user_id", "video_id", "score"):
            raise BenchmarkContractError("submission header changed")
        if self.row_identity != "zero-based contiguous physical split order":
            raise BenchmarkContractError("row identity convention changed")
        if self.convergence != _FROZEN_CONVERGENCE:
            raise BenchmarkContractError("convergence must use strict delta > 0.002, patience 3")
        return self

    def manifest(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible benchmark and artifact manifest."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "task": self.task.manifest(),
            "splits": [split.manifest() for split in self.splits],
            "metrics": self.metrics.manifest(),
            "reference_rungs": [rung.manifest() for rung in self.reference_rungs],
            "submission": {
                "header": list(self.submission_header),
                "row_identity": self.row_identity,
            },
            "convergence": self.convergence.manifest(),
            "organizer_artifacts": {
                "starter_file_sha256": dict(sorted(STARTER_FILE_SHA256.items())),
                "starter_zip_sha256": STARTER_ZIP_SHA256,
                "dataset_archive_md5": DATASET_ARCHIVE_MD5,
                "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
            },
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 identity of the canonical benchmark manifest."""

        return canonical_json_sha256(self.manifest())


_FROZEN_TASK: Final = RankingTaskContract(
    dataset="KuaiRand-Pure",
    target="long_view",
    relevance="native binary",
    ranking_unit="within-user ranking over logged impressions",
)
_FROZEN_SPLITS: Final = (
    SplitContract(SplitName.TRAIN, 20220408, 20220421, "training"),
    SplitContract(SplitName.VALID, 20220422, 20220428, "trusted_scorer_only"),
    SplitContract(SplitName.TEST, 20220429, 20220508, "forbidden_during_development"),
)
_FROZEN_METRICS: Final = MetricContract(
    gauc_name="GAUC",
    gauc_user_eligibility="both positive and negative impressions",
    gauc_weight="positive count",
    ndcg_name="nDCG@5",
    ndcg_cutoff=5,
    ndcg_user_eligibility="all users",
    ndcg_zero_positive_value=0.0,
    ndcg_gain="2^rel - 1",
    primary_formula="(GAUC + nDCG@5) / 2",
)
_FROZEN_REFERENCE_RUNGS: Final = (
    ReferenceRung(
        "random",
        PublishedScore(gauc=0.4993, ndcg_at_5=0.4675, primary=0.4834),
        "reproduce after four-decimal rounding",
    ),
    ReferenceRung(
        "item_popularity",
        PublishedScore(gauc=0.6387, ndcg_at_5=0.5227, primary=0.5807),
        "reproduce after four-decimal rounding",
    ),
    ReferenceRung(
        "fm_official",
        PublishedScore(gauc=0.6674, ndcg_at_5=0.5357, primary=0.6016),
        "reproduce five-seed public-validation mean after four-decimal rounding",
    ),
)
_FROZEN_CONVERGENCE: Final = ConvergenceContract(
    epsilon=0.002,
    patience=3,
    comparison="eligible outer primary delta strictly greater than epsilon",
)

BENCHMARK_CONTRACT: Final = BenchmarkContract(
    schema_version=1,
    task=_FROZEN_TASK,
    splits=_FROZEN_SPLITS,
    metrics=_FROZEN_METRICS,
    reference_rungs=_FROZEN_REFERENCE_RUNGS,
    submission_header=("row_id", "user_id", "video_id", "score"),
    row_identity="zero-based contiguous physical split order",
    convergence=_FROZEN_CONVERGENCE,
).validate()


def benchmark_manifest() -> dict[str, object]:
    """Return a new JSON-compatible copy of the frozen benchmark manifest."""

    return BENCHMARK_CONTRACT.manifest()


def benchmark_digest() -> str:
    """Return the canonical benchmark and organizer-artifact identity."""

    return BENCHMARK_CONTRACT.digest


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_starter_kit(root: str | Path) -> StarterVerification:
    """Require the organizer directory to contain exactly the seven pinned files."""

    starter_root = Path(root)
    if not starter_root.is_dir():
        raise OrganizerIntegrityError(f"starter directory does not exist: {starter_root}")
    actual_entries = {entry.name for entry in starter_root.iterdir()}
    expected_entries = set(STARTER_FILE_SHA256)
    missing = expected_entries - actual_entries
    unexpected = actual_entries - expected_entries
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)!r}")
        raise OrganizerIntegrityError("organizer starter member set changed: " + ", ".join(details))

    observed: dict[str, str] = {}
    for name, expected_digest in STARTER_FILE_SHA256.items():
        path = starter_root / name
        if path.is_symlink() or not path.is_file():
            raise OrganizerIntegrityError(f"starter member is not a regular file: {name}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise OrganizerIntegrityError(
                f"organizer starter digest mismatch for {name}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        observed[name] = actual_digest

    manifest_payload = "".join(f"{name}\0{observed[name]}\n" for name in sorted(observed))
    manifest_digest = hashlib.sha256(manifest_payload.encode("ascii")).hexdigest()
    return StarterVerification(
        root=starter_root.resolve(), files=observed, manifest_sha256=manifest_digest
    )


class ContractManifestError(BenchmarkContractError):
    """Raised when the complete competition manifest is malformed or drifts at admission."""


def _reject_machine_paths(value: object, location: str = "contract") -> None:
    if type(value) is str:
        windows_absolute = (
            len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in ("/", "\\")
        )
        if value.startswith("/") or windows_absolute:
            raise ContractManifestError(f"{location} must not contain absolute machine paths")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_machine_paths(key, f"{location} key")
            _reject_machine_paths(item, f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_machine_paths(item, f"{location}[{index}]")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ContractManifest:
    """Complete path-independent organizer, dataset, metric, row, and submission contract."""

    schema_version: int
    executable_benchmark_manifest: Mapping[str, object]
    challenge_manifest: Mapping[str, object]
    organizer_file_sha256: Mapping[str, str]
    dataset_sha256: str
    split_identities: Mapping[str, str]
    metric_implementation_sha256: str
    row_identity_policy: Mapping[str, object]
    submission_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractManifestError("contract manifest schema_version must be 1")
        for name in (
            "executable_benchmark_manifest",
            "challenge_manifest",
            "organizer_file_sha256",
            "split_identities",
            "row_identity_policy",
            "submission_schema",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ContractManifestError(f"{name} must be a non-empty object")
        try:
            canonical_json_bytes(self.manifest())
        except (IdentityError, ValueError) as exc:
            raise ContractManifestError("contract manifest must be finite canonical JSON") from exc
        for name, digest in self.organizer_file_sha256.items():
            if name not in STARTER_FILE_SHA256:
                raise ContractManifestError(f"unknown organizer file {name!r}")
            _contract_digest(digest, f"organizer_file_sha256.{name}")
        if set(self.organizer_file_sha256) != set(STARTER_FILE_SHA256):
            raise ContractManifestError("organizer file manifest must contain every pinned file")
        _contract_digest(self.dataset_sha256, "dataset_sha256")
        _contract_digest(self.metric_implementation_sha256, "metric_implementation_sha256")
        if set(self.split_identities) != {split.name.value for split in BENCHMARK_CONTRACT.splits}:
            raise ContractManifestError("split identities must contain train, valid, and test")
        for name, digest in self.split_identities.items():
            _contract_digest(digest, f"split_identities.{name}")
        _reject_machine_paths(self.manifest())

        # Defensive deep freezing keeps content identity stable even if callers retain and mutate
        # the dictionaries used to construct this value object.
        for name in (
            "executable_benchmark_manifest",
            "challenge_manifest",
            "organizer_file_sha256",
            "split_identities",
            "row_identity_policy",
            "submission_schema",
        ):
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))

    def manifest(self) -> dict[str, object]:
        """Return a fresh complete JSON-compatible projection."""

        return {
            "schema_version": self.schema_version,
            "executable_benchmark_manifest": _thaw_json(self.executable_benchmark_manifest),
            "challenge_manifest": _thaw_json(self.challenge_manifest),
            "organizer_file_sha256": _thaw_json(self.organizer_file_sha256),
            "dataset_sha256": self.dataset_sha256,
            "split_identities": _thaw_json(self.split_identities),
            "metric_implementation_sha256": self.metric_implementation_sha256,
            "row_identity_policy": _thaw_json(self.row_identity_policy),
            "submission_schema": _thaw_json(self.submission_schema),
        }

    @property
    def contract_id(self) -> ContractId:
        """Nominal identity of exactly this manifest."""

        return ContractId.from_manifest(self.manifest())


@dataclass(frozen=True, slots=True)
class ContractVerificationReceipt:
    """Startup evidence that admission used exactly the frozen contract lineage."""

    schema_version: int
    contract_id: ContractId
    expected_contract_id: ContractId
    verified: bool

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id.value,
            "expected_contract_id": self.expected_contract_id.value,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class PinnedSplitInput:
    """One explicitly configured physical split file and its external digest pin.

    ``path`` is runtime configuration only.  It is deliberately excluded from every canonical
    manifest and identity emitted by this module.
    """

    split: SplitName
    path: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.split, SplitName):
            raise ContractManifestError("pinned split input must use a SplitName")
        if not isinstance(self.path, Path):
            raise ContractManifestError("pinned split input path must be pathlib.Path")
        _contract_digest(self.expected_sha256, f"pinned split {self.split.value} sha256")


@dataclass(frozen=True, slots=True)
class RepositoryInputManifest:
    """Path-independent evidence derived from the repository inputs read at admission."""

    schema_version: int
    contract_id: ContractId
    organizer_file_sha256: Mapping[str, str]
    organizer_manifest_sha256: str
    starter_zip_sha256: str
    dataset_archive_sha256: str | None
    split_input_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractManifestError("repository input manifest schema_version must be 1")
        if not isinstance(self.contract_id, ContractId):
            raise ContractManifestError("repository input manifest requires a ContractId")
        if set(self.organizer_file_sha256) != set(STARTER_FILE_SHA256):
            raise ContractManifestError(
                "repository input manifest must contain every organizer file"
            )
        for name, digest in self.organizer_file_sha256.items():
            _contract_digest(digest, f"repository organizer file {name}")
        _contract_digest(self.organizer_manifest_sha256, "organizer_manifest_sha256")
        _contract_digest(self.starter_zip_sha256, "starter_zip_sha256")
        if self.dataset_archive_sha256 is not None:
            _contract_digest(self.dataset_archive_sha256, "dataset_archive_sha256")
        for name, digest in self.split_input_sha256.items():
            if name not in {split.value for split in SplitName}:
                raise ContractManifestError(f"unknown configured split input {name!r}")
            _contract_digest(digest, f"split_input_sha256.{name}")
        _reject_machine_paths(self.manifest(), "repository input manifest")
        object.__setattr__(
            self,
            "organizer_file_sha256",
            _freeze_json(self.organizer_file_sha256),
        )
        object.__setattr__(self, "split_input_sha256", _freeze_json(self.split_input_sha256))

    def manifest(self) -> dict[str, object]:
        """Return canonical live-input evidence without any local filesystem paths."""

        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id.value,
            "organizer_file_sha256": _thaw_json(self.organizer_file_sha256),
            "organizer_manifest_sha256": self.organizer_manifest_sha256,
            "starter_zip_sha256": self.starter_zip_sha256,
            "dataset_archive_sha256": self.dataset_archive_sha256,
            "split_input_sha256": _thaw_json(self.split_input_sha256),
        }

    @property
    def digest(self) -> str:
        """Return the canonical identity of exactly the inputs observed at admission."""

        return canonical_json_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class RepositoryContractVerificationReceipt(ContractVerificationReceipt):
    """Contract receipt proving that live repository inputs, not constants, were hashed."""

    repository_inputs: RepositoryInputManifest

    def __post_init__(self) -> None:
        if not self.verified or self.contract_id != self.expected_contract_id:
            raise ContractManifestError("repository verification requires an exact contract match")
        if self.repository_inputs.contract_id != self.contract_id:
            raise ContractManifestError("repository input evidence belongs to another contract")

    def manifest(self) -> dict[str, object]:
        return {
            **ContractVerificationReceipt.manifest(self),
            "repository_inputs": self.repository_inputs.manifest(),
            "repository_inputs_sha256": self.repository_inputs.digest,
        }


def _contract_digest(value: object, location: str) -> str:
    try:
        return ContractId(str(value)).value
    except (IdentityError, ValueError) as exc:
        raise ContractManifestError(f"{location} must be a full lowercase SHA-256") from exc


def _split_contract_identities() -> dict[str, str]:
    return {
        split.name.value: canonical_json_sha256(
            {
                "schema_version": 1,
                "identity_type": "organizer_split_contract",
                "split": split.manifest(),
            }
        )
        for split in BENCHMARK_CONTRACT.splits
    }


def _contract_manifest_from_inputs(
    *,
    organizer_file_sha256: Mapping[str, str],
    dataset_sha256: str,
) -> ContractManifest:
    """Build the semantic contract from caller-supplied physical input identities."""

    from kuairand_agent.challenge_contract import KUAI_PURE_CHALLENGE

    return ContractManifest(
        schema_version=1,
        executable_benchmark_manifest=BENCHMARK_CONTRACT.manifest(),
        challenge_manifest=KUAI_PURE_CHALLENGE.manifest(),
        organizer_file_sha256=organizer_file_sha256,
        dataset_sha256=dataset_sha256,
        split_identities=_split_contract_identities(),
        metric_implementation_sha256=organizer_file_sha256["evaluate.py"],
        row_identity_policy={
            "assignment": "before_sort_join_group_or_feature_generation",
            "definition": BENCHMARK_CONTRACT.row_identity,
            "candidate_visibility": "forbidden",
            "prediction_alignment": "exact_ordered_row_identity",
        },
        submission_schema={
            "columns": [
                {"name": "row_id", "type": "non_negative_integer"},
                {"name": "user_id", "type": "organizer_identity"},
                {"name": "video_id", "type": "organizer_identity"},
                {"name": "score", "type": "finite_float64_round_trip"},
            ],
            "header": list(BENCHMARK_CONTRACT.submission_header),
            "row_count": "exact_canonical_final_split_row_count",
            "ordering": "canonical_final_split_order",
        },
    )


def _frozen_contract_manifest() -> ContractManifest:
    return _contract_manifest_from_inputs(
        organizer_file_sha256=STARTER_FILE_SHA256,
        dataset_sha256=DATASET_ARCHIVE_SHA256,
    )


CONTRACT_MANIFEST: Final = _frozen_contract_manifest()
CONTRACT_ID: Final = ContractId("fedc7599f59c6cc1b3319c542d147dcaf499bdd08ceeba92551787d9c9bf4f93")
if CONTRACT_MANIFEST.contract_id != CONTRACT_ID:  # pragma: no cover - import guard
    raise RuntimeError("frozen ContractId differs from its canonical contract manifest")


def contract_manifest() -> dict[str, object]:
    """Return a fresh copy of the frozen complete contract manifest."""

    return CONTRACT_MANIFEST.manifest()


def verify_contract_manifest(
    candidate: ContractManifest,
    *,
    expected: ContractManifest = CONTRACT_MANIFEST,
) -> ContractVerificationReceipt:
    """Reject contract drift before training and return a durable verification receipt."""

    if not isinstance(candidate, ContractManifest):
        raise ContractManifestError("candidate contract must be a ContractManifest")
    if not isinstance(expected, ContractManifest):
        raise ContractManifestError("expected contract must be a ContractManifest")
    candidate_id = candidate.contract_id
    expected_id = expected.contract_id
    if candidate_id != expected_id:
        raise ContractManifestError(
            f"contract mismatch: expected {expected_id.value}, got {candidate_id.value}"
        )
    return ContractVerificationReceipt(
        schema_version=1,
        contract_id=candidate_id,
        expected_contract_id=expected_id,
        verified=True,
    )


def _hash_regular_file(path: Path, *, label: str) -> str:
    """Hash one stable regular file without following a final-component symlink."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractManifestError(f"{label} is missing or cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractManifestError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if stable_identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ContractManifestError(f"{label} changed while it was being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_pinned_file(path: Path, *, expected_sha256: str, label: str) -> str:
    expected = _contract_digest(expected_sha256, f"expected {label} sha256")
    observed = _hash_regular_file(path, label=label)
    if observed != expected:
        raise ContractManifestError(f"{label} digest mismatch: expected {expected}, got {observed}")
    return observed


def verify_repository_contract_inputs(
    repository_root: str | Path,
    *,
    dataset_archive: Path | None = None,
    split_inputs: Sequence[PinnedSplitInput] = (),
    expected: ContractManifest = CONTRACT_MANIFEST,
) -> RepositoryContractVerificationReceipt:
    """Hash live pinned inputs and reject drift before callers open mutable state.

    The organizer starter directory and its source ZIP are mandatory fixed repository-relative
    inputs.  A dataset archive is verified against the organizer's frozen archive digest only when
    the caller explicitly configures it; this function never guesses a machine-local data path.
    Likewise, physical split files are admitted only through explicit external digest pins.

    The returned receipt contains logical names and content digests, never resolved paths.  The
    caller must invoke this function before creating a state directory or opening SQLite.
    """

    if not isinstance(expected, ContractManifest):
        raise ContractManifestError("expected contract must be a ContractManifest")
    root = Path(repository_root)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ContractManifestError("repository root is missing or unavailable") from exc
    if not resolved_root.is_dir():
        raise ContractManifestError("repository root must be a directory")

    starter_root = resolved_root / "kuairand-starter-kit"
    try:
        starter = verify_starter_kit(starter_root)
    except OrganizerIntegrityError as exc:
        raise ContractManifestError(f"organizer starter verification failed: {exc}") from exc

    starter_zip_sha256 = _verify_pinned_file(
        resolved_root / "kuairand-starter-kit.zip",
        expected_sha256=STARTER_ZIP_SHA256,
        label="organizer starter ZIP",
    )

    dataset_sha256: str | None = None
    if dataset_archive is not None:
        if not isinstance(dataset_archive, Path):
            raise ContractManifestError("dataset_archive must be pathlib.Path when configured")
        dataset_sha256 = _verify_pinned_file(
            dataset_archive,
            expected_sha256=expected.dataset_sha256,
            label="configured dataset archive",
        )

    observed_splits: dict[str, str] = {}
    seen_splits: set[SplitName] = set()
    for split_input in split_inputs:
        if not isinstance(split_input, PinnedSplitInput):
            raise ContractManifestError("split_inputs must contain PinnedSplitInput values")
        if split_input.split in seen_splits:
            raise ContractManifestError(
                f"configured split input {split_input.split.value!r} is duplicated"
            )
        seen_splits.add(split_input.split)
        observed_splits[split_input.split.value] = _verify_pinned_file(
            split_input.path,
            expected_sha256=split_input.expected_sha256,
            label=f"configured {split_input.split.value} split input",
        )

    # Construct a new candidate from the bytes observed above.  The official dataset identity
    # remains a declarative contract field when no archive was configured, and the receipt makes
    # that absence explicit rather than claiming full-data verification.
    candidate = _contract_manifest_from_inputs(
        organizer_file_sha256=starter.files,
        dataset_sha256=dataset_sha256 or expected.dataset_sha256,
    )
    verification = verify_contract_manifest(candidate, expected=expected)
    repository_inputs = RepositoryInputManifest(
        schema_version=1,
        contract_id=verification.contract_id,
        organizer_file_sha256=starter.files,
        organizer_manifest_sha256=starter.manifest_sha256,
        starter_zip_sha256=starter_zip_sha256,
        dataset_archive_sha256=dataset_sha256,
        split_input_sha256=observed_splits,
    )
    return RepositoryContractVerificationReceipt(
        schema_version=verification.schema_version,
        contract_id=verification.contract_id,
        expected_contract_id=verification.expected_contract_id,
        verified=verification.verified,
        repository_inputs=repository_inputs,
    )
