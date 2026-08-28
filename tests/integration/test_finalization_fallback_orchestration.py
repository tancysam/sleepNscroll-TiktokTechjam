from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kuairand_agent.contract import verify_starter_kit
from kuairand_agent.data.canonical import LOG_HEADER, OUTCOME_FIELDS, VIDEO_BASIC_HEADER
from kuairand_agent.data.capabilities import CandidateInputs, DataPhase, build_candidate_inputs
from kuairand_agent.data.fields import STANDARD_LATE_MEMBER, VIDEO_BASIC_MEMBER, FieldKey
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.finalization.finalize import (
    FinalizationCancelledError,
    FinalizationCandidate,
    FinalizationRequest,
    run_finalization,
)
from kuairand_agent.finalization.replay import (
    CleanReplayRequest,
    FrozenReplayIdentity,
    ReplayArtifacts,
    ReplayBackend,
    ReplayCapabilities,
    ReplayEquality,
)
from kuairand_agent.finalization.report import (
    ExperimentNarrative,
    FinalReportContext,
    MetricEvidence,
    ResourceEvidence,
)
from kuairand_agent.finalization.submission_bundle import FinalBundleMetadata, FinalStatus
from kuairand_agent.scoring.submission import AlignmentRow, prediction_digest

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"


def _log_row(user: bytes, video: bytes, date: int, outcome: bytes = b"0") -> bytes:
    cells = [b"0"] * len(LOG_HEADER)
    for name, value in {
        "user_id": user,
        "video_id": video,
        "date": str(date).encode("ascii"),
        "hourmin": b"1200",
        "time_ms": b"1",
        "duration_ms": b"1000",
        "is_rand": b"0",
        "tab": b"1",
    }.items():
        cells[LOG_HEADER.index(name)] = value
    for name in OUTCOME_FIELDS:
        cells[LOG_HEADER.index(name)] = outcome
    return b",".join(cells) + b"\n"


def _data(root: Path) -> Path:
    root.mkdir()
    header = ",".join(LOG_HEADER).encode("ascii") + b"\n"
    (root / "log_standard_4_08_to_4_21_pure.csv").write_bytes(
        header + _log_row(b"train", b"v-train", 20220408)
    )
    (root / "log_standard_4_22_to_5_08_pure.csv").write_bytes(
        header
        + _log_row(b"valid", b"v-valid", 20220422)
        + _log_row(b"test-a", b"v-final-a", 20220429, b"\xff")
        + _log_row(b"test-b", b"v-final-b", 20220508, b"\x80\xfe")
    )
    videos = ("v-train", "v-valid", "v-final-a", "v-final-b")
    rows = []
    for video in videos:
        values = {
            "video_id": video,
            "author_id": f"author-{video}",
            "video_type": "NORMAL",
            "upload_dt": "2022-01-01",
            "upload_type": "1",
            "visible_status": "1",
            "video_duration": "1000",
            "server_width": "720",
            "server_height": "1280",
            "music_id": "music",
            "music_type": "1",
            "tag": "tag",
        }
        rows.append(",".join(values[name] for name in VIDEO_BASIC_HEADER))
    (root / "video_features_basic_pure.csv").write_text(
        ",".join(VIDEO_BASIC_HEADER) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    return root


def _inputs(phase: DataPhase, users: tuple[str, str], videos: tuple[str, str]) -> CandidateInputs:
    return build_candidate_inputs(
        phase,
        {
            FieldKey(STANDARD_LATE_MEMBER, "user_id"): users,
            FieldKey(STANDARD_LATE_MEMBER, "video_id"): videos,
            FieldKey(VIDEO_BASIC_MEMBER, "author_id"): ("a1", "a2"),
            FieldKey(STANDARD_LATE_MEMBER, "tab"): ("0", "1"),
            FieldKey(STANDARD_LATE_MEMBER, "duration_ms"): (100.0, 200.0),
        },
    )


class _GoodBackend:
    def replay_validation(self, *, workspace: object, inputs: object) -> np.ndarray:
        del workspace, inputs
        return np.array([0.1, 0.9], dtype=np.float64)

    def predict_final(self, *, workspace: object, inputs: object) -> np.ndarray:
        del workspace, inputs
        return np.array([0.2, 0.8], dtype=np.float64)


class _CorruptBackend(_GoodBackend):
    def replay_validation(self, *, workspace: object, inputs: object) -> np.ndarray:
        del workspace, inputs
        return np.array([0.9, 0.1], dtype=np.float64)


def _report(
    candidate_id: str,
    *,
    metrics: tuple[float, float, float] = (0.6, 0.5, 0.55),
) -> FinalReportContext:
    gauc, ndcg_at_5, primary = metrics
    return FinalReportContext(
        benchmark_contract={"dataset": "KuaiRand-Pure", "target": "long_view"},
        baselines=(MetricEvidence("official FM", "outer validation", 0.6, 0.5, 0.55),),
        selected=MetricEvidence(candidate_id, "outer validation", gauc, ndcg_at_5, primary),
        experiments=(
            ExperimentNarrative(
                1,
                candidate_id,
                "qualification",
                "Replay the eligible frozen incumbent.",
                "Restore exact source and checkpoint artifacts.",
                ("candidate.py is material executable source",),
                ("organizer starter evaluator",),
                "eligible",
                0.55,
                0.55,
            ),
        ),
        inner_fold_evidence=("Frozen inner-fold evidence retained in experiments.jsonl.",),
        seed_confirmation=("Matched seed evidence retained in the campaign ledger.",),
        failures_and_recoveries=("Incumbent protection remained active.",),
        leakage_controls=("FINAL capability exposes inputs only.",),
        test_evidence=("Untouched organizer checker runs in check-only mode.",),
        selection_rationale="Highest replayable eligible candidate after fallback.",
        resources=ResourceEvidence(1.0, 1024, 2, 0, "No call during replay.", ("CPU",)),
    )


def test_each_candidate_must_survive_closure_replay_report_and_bundle_before_selection(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidate.py").write_text("def score(rows): return rows\n", encoding="utf-8")
    source_ref = store.put_directory(source, kind=ArtifactKind.SOURCE)
    config_ref = store.put_bytes(b"config", kind=ArtifactKind.INPUT)
    features_ref = store.put_bytes(b"features", kind=ArtifactKind.INPUT)
    checkpoint_ref = store.put_bytes(b"checkpoint", kind=ArtifactKind.CHECKPOINT)
    reference = tmp_path / "reference.npy"
    with reference.open("xb") as handle:
        np.save(handle, np.array([0.1, 0.9], dtype=np.float64), allow_pickle=False)
    prediction_ref = store.put_file(reference, kind=ArtifactKind.PREDICTION)
    environment_body: dict[str, object] = {
        "schema_version": 1,
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "platform": {"system": "fixture", "machine": "test"},
        "packages": {
            "lightgbm": "4.6.0",
            "numpy": "2.5.2",
            "psutil": "7.0.0",
            "torch": "2.8.0",
        },
        "uv_lock_sha256": "f" * 64,
    }
    # Campaign provenance uses the domain bytes directly rather than an ordinary JSON wrapper.
    environment_digest = hashlib.sha256(
        b"kuairand-environment-v1\0"
        + json.dumps(
            environment_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    environment = environment_body | {"digest": environment_digest}
    data_sha256 = "d" * 64
    identity = FrozenReplayIdentity(
        source_ref.sha256,
        config_ref.sha256,
        features_ref.sha256,
        checkpoint_ref.sha256,
        prediction_ref.sha256,
        prediction_digest((0.1, 0.9)),
        data_sha256,
        environment_digest,
    )
    artifacts = ReplayArtifacts(
        source_ref, config_ref, features_ref, checkpoint_ref, prediction_ref
    )
    valid_alignment = (
        AlignmentRow(0, "valid-a", "v-valid-a"),
        AlignmentRow(1, "valid-b", "v-valid-b"),
    )
    final_alignment = (
        AlignmentRow(0, "test-a", "v-final-a"),
        AlignmentRow(1, "test-b", "v-final-b"),
    )
    capabilities = ReplayCapabilities(
        data_sha256,
        _inputs(DataPhase.OUTER_VALID, ("valid-a", "valid-b"), ("v-valid-a", "v-valid-b")),
        _inputs(DataPhase.FINAL, ("test-a", "test-b"), ("v-final-a", "v-final-b")),
        valid_alignment,
        final_alignment,
    )
    starter_digest = verify_starter_kit(STARTER).manifest_sha256

    def candidate(
        candidate_id: str,
        lineage: tuple[str, ...],
        backend: ReplayBackend,
        *,
        fallback: bool,
        metrics: tuple[float, float, float] = (0.6, 0.5, 0.55),
        corrupt_closure: bool = False,
        corrupt_bundle: bool = False,
    ) -> FinalizationCandidate:
        gauc, ndcg_at_5, primary = metrics
        scientific_source = "0" * 64 if corrupt_closure else identity.source_sha256
        campaign_totals = {
            "attempt_count": 5,
            "scientific_iteration_count": 1,
            "launch_count": 2,
            "elapsed_seconds": 1.0,
            "manual_intervention_count": 0,
        }
        if corrupt_bundle:
            campaign_totals.pop("launch_count")
        metadata = FinalBundleMetadata(
            benchmark_identity={"name": "KuaiRand-Pure", "digest": "b" * 64},
            starter_identity={"manifest_sha256": starter_digest},
            data_identity={"canonical_digest": data_sha256},
            selected_experiment=candidate_id,
            lineage=lineage,
            status=FinalStatus.BASELINE_REPRODUCED,
            validation_metrics={"GAUC": gauc, "nDCG@5": ndcg_at_5, "primary": primary},
            seed_summary={"seeds": [0], "mean_primary": primary},
            inner_fold_results=({"fold": "A", "primary": primary},),
            scientific_artifact_hashes={
                "source": scientific_source,
                "config": identity.config_sha256,
                "features": identity.features_sha256,
                "checkpoint": identity.checkpoint_sha256,
                "predictions": identity.validation_prediction_artifact_sha256,
            },
            environment_and_resource_usage={
                "environment_sha256": identity.environment_sha256,
                "device": "cpu",
                "runtime_identity": {
                    "schema_version": 1,
                    "project_source_digest": "0" * 64,
                    "environment_digest": identity.environment_sha256,
                    "uv_lock_sha256": environment_body["uv_lock_sha256"],
                    "dependency_groups": ["research-tree", "research-neural"],
                },
            },
            campaign_totals=campaign_totals,
            known_limitations=("Hidden-test improvement is unverified until organizer scoring.",),
        )
        return FinalizationCandidate(
            candidate_id,
            lineage,
            FinalStatus.BASELINE_REPRODUCED,
            CleanReplayRequest(
                candidate_id,
                tmp_path / f"unused-{candidate_id}",
                identity,
                artifacts,
                environment,
                ReplayEquality.EXACT,
            ),
            backend,
            metadata,
            _report(candidate_id, metrics=metrics),
            fallback,
        )

    experiments_jsonl = tmp_path / "experiments.jsonl"
    experiments_jsonl.write_text('{"experiment_id":"official-fm"}\n', encoding="utf-8")
    experiments_csv = tmp_path / "experiments.csv"
    experiments_csv.write_text("experiment_id,status\nofficial-fm,eligible\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    request = FinalizationRequest(
        destination=tmp_path / "final",
        artifact_store=store,
        capabilities=capabilities,
        protected_metric_evaluator=lambda scores: {
            "GAUC": 0.6,
            "nDCG@5": 0.5,
            "primary": 0.55,
        },
        data_dir=_data(tmp_path / "data"),
        starter_dir=STARTER,
        experiments_jsonl=experiments_jsonl,
        experiments_csv=experiments_csv,
        candidates=(
            candidate(
                "bad-selected",
                ("official-fm", "bad-selected"),
                _CorruptBackend(),
                fallback=False,
            ),
            candidate(
                "bad-closure",
                ("official-fm", "bad-closure"),
                _GoodBackend(),
                fallback=False,
                corrupt_closure=True,
            ),
            candidate(
                "bad-report",
                ("official-fm", "bad-report"),
                _GoodBackend(),
                fallback=False,
                metrics=(0.61, 0.5, 0.555),
            ),
            candidate(
                "bad-bundle",
                ("official-fm", "bad-bundle"),
                _GoodBackend(),
                fallback=False,
                corrupt_bundle=True,
            ),
            candidate("official-fm", ("official-fm",), _GoodBackend(), fallback=True),
        ),
        scratch_dir=scratch,
    )
    result = run_finalization(request)

    assert result.selected_candidate_id == "official-fm"
    assert result.fallback_count == 4
    assert tuple(failure.stage for failure in result.failures) == (
        "clean replay",
        "candidate closure",
        "final report",
        "closed bundle",
    )
    assert result.organizer_check.checker_returncode == 0
    assert result.organizer_check.masked_view.final_rows_masked == 2
    assert result.bundle.root == (tmp_path / "final").resolve()
    manifest = json.loads((result.bundle.root / "manifest.json").read_text(encoding="ascii"))
    assert manifest["selection"]["selected_experiment"] == "official-fm"
    report = (result.bundle.root / "report.md").read_text(encoding="utf-8")
    assert "bad-selected failed clean replay" in report
    reproduce = (result.bundle.root / "reproduce.sh").read_text(encoding="ascii")
    assert "uv sync --locked --group research-tree --group research-neural" in reproduce
    verification = json.loads(
        (result.bundle.root / "verification.json").read_text(encoding="ascii")
    )
    assert verification["organizer_check"]["mode"] == "check_only"
    assert tuple(scratch.iterdir()) == ()

    cancellation = threading.Event()
    cancellation.set()
    cancelled_destination = tmp_path / "cancelled-final"
    with pytest.raises(FinalizationCancelledError):
        run_finalization(
            replace(request, destination=cancelled_destination),
            cancel_event=cancellation,
        )
    assert not cancelled_destination.exists()
