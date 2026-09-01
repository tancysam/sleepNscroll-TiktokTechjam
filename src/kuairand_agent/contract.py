"""Pinned organizer and benchmark identities owned by the trusted controller."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

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

        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


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
    # The organizer README instructs running `python3 submit.py --check` from inside this
    # directory before every submission, and CPython then writes __pycache__ next to the pinned
    # files.  Treating that as tampering halted a live campaign for a directory entry that changes
    # no pinned byte.  hash_source_tree already excludes it (campaign/provenance.py), so exclude it
    # here too rather than leaving the organizer's own documented workflow lethal.  Every pinned
    # file is still digest-verified below, so an actual edit is still caught.
    actual_entries = {
        entry.name for entry in starter_root.iterdir() if entry.name != "__pycache__"
    }
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
