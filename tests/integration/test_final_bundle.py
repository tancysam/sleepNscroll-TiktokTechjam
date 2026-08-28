from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

import kuairand_agent.finalization.production as production
import kuairand_agent.finalization.submission_bundle as bundle_module
from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore
from kuairand_agent.finalization.organizer_check import (
    MaskedFileEvidence,
    MaskedViewEvidence,
    OrganizerCheckEvidence,
)
from kuairand_agent.finalization.submission_bundle import (
    REQUIRED_DIRECTORY_PATHS,
    REQUIRED_FILE_PATHS,
    BundleFileEvidence,
    FinalBundleCancelledError,
    FinalBundleError,
    FinalBundleMetadata,
    FinalBundleSources,
    FinalStatus,
    create_final_bundle,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sources(
    tmp_path: Path,
    *,
    generated: bool = False,
    candidate_id: str = "pairwise-confirmed",
) -> FinalBundleSources:
    inputs = tmp_path / "inputs"
    submission = _write(
        inputs / "submission.csv",
        "row_id,user_id,video_id,score\n0,u,v,0.12345678901234566\n",
    )
    directories = {component: inputs / component for component in REQUIRED_DIRECTORY_PATHS}
    source_file = _write(
        directories["source"] / "candidate.py",
        "def predict(rows):\n    return rows\n",
    )
    config_file = _write(directories["config"] / "artifact", '{"recipe":"fixture"}')
    if generated:
        _write(directories["preprocessing"] / "validation.npy", "validation features\n")
        _write(directories["preprocessing"] / "final.npy", "final features\n")
        _write(directories["preprocessing"] / "fm-encoding.npz", "encoding\n")
        _write(directories["model"] / "tree-model.txt", "tree checkpoint\n")
        _write(directories["model"] / "fm-checkpoint.npz", "fm checkpoint\n")
    else:
        feature_file = _write(directories["preprocessing"] / "artifact", "encoded features\n")
        checkpoint_file = _write(directories["model"] / "artifact", "model checkpoint\n")
    reference = _write(
        directories["validation-evidence"] / "reference-validation-predictions.npy",
        "reference prediction bytes\n",
    )
    public_validation = _write(
        directories["validation-evidence"] / "public-validation.csv",
        "row_id,user_id,video_id,score\n0,u,v,0.5\n",
    )
    replayed_validation = _write(
        directories["replay"] / "validation-predictions.npy", "replayed validation\n"
    )
    replayed_final = _write(directories["replay"] / "final-predictions.npy", "replayed final\n")

    environment_body: dict[str, object] = {
        "schema_version": 1,
        "python": {"implementation": "CPython", "version": "3.12.0"},
        "platform": {"system": "fixture", "machine": "test"},
        "packages": {"numpy": "2.5.2"},
        "uv_lock_sha256": "6" * 64,
    }
    canonical_environment = json.dumps(
        environment_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    environment_digest = hashlib.sha256(
        b"kuairand-environment-v1\0" + canonical_environment.encode("ascii")
    ).hexdigest()
    environment_file = _write(
        inputs / "environment.json",
        json.dumps(
            environment_body | {"digest": environment_digest},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )

    artifact_store = ArtifactStore(tmp_path / "fixture-artifacts")
    source_identity = artifact_store.put_directory(
        source_file.parent,
        kind=ArtifactKind.SOURCE,
    ).sha256
    if generated:
        feature_identity = artifact_store.put_directory(
            directories["preprocessing"],
            kind=ArtifactKind.INPUT,
        ).sha256
        checkpoint_identity = artifact_store.put_directory(
            directories["model"],
            kind=ArtifactKind.CHECKPOINT,
        ).sha256
    else:
        feature_identity = hashlib.sha256(feature_file.read_bytes()).hexdigest()
        checkpoint_identity = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    identity = {
        "source_sha256": source_identity,
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "features_sha256": feature_identity,
        "checkpoint_sha256": checkpoint_identity,
        "validation_prediction_artifact_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "validation_prediction_digest": "6" * 64,
        "data_sha256": "f" * 64,
        "environment_sha256": environment_digest,
    }
    replay_evidence = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "identity": identity,
        "equality": {"policy": "exact_same_host_bytes", "absolute_tolerance": 0.0},
        "training_replay": "fixture replay",
        "validation": {
            "reference_prediction_digest": identity["validation_prediction_digest"],
            "replay_prediction_digest": "a" * 64,
            "replay_prediction_file_sha256": hashlib.sha256(
                replayed_validation.read_bytes()
            ).hexdigest(),
            "public_submission_sha256": hashlib.sha256(public_validation.read_bytes()).hexdigest(),
            "public_submission_prediction_digest": "a" * 64,
        },
        "final": {
            "prediction_digest": "b" * 64,
            "prediction_file_sha256": hashlib.sha256(replayed_final.read_bytes()).hexdigest(),
            "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
            "submission_prediction_digest": "b" * 64,
        },
        "capabilities": {},
        "workspace": {
            "fresh_materialization": True,
            "artifact_identities_reverified_after_inference": True,
            "clean_workspace_removed": True,
        },
    }
    (directories["replay"] / "evidence.json").write_text(
        json.dumps(
            replay_evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    return FinalBundleSources(
        submission=submission,
        report=_write(
            inputs / "report.md",
            "# Final report\n\nHidden-test improvement remains unverified.\n",
        ),
        experiments_jsonl=_write(inputs / "experiments.jsonl", '{"experiment":"fm"}\n'),
        experiments_csv=_write(inputs / "experiments.csv", "experiment,status\nfm,eligible\n"),
        environment=environment_file,
        reproduce=_write(
            inputs / "reproduce.sh",
            '#!/bin/sh\nset -eu\n: "${KUAI_DATA_DIR:?supply verified dataset}"\n',
        ),
        config=directories["config"],
        source=directories["source"],
        model=directories["model"],
        preprocessing=directories["preprocessing"],
        validation_evidence=directories["validation-evidence"],
        replay=directories["replay"],
    )


def _organizer_evidence(submission: Path) -> OrganizerCheckEvidence:
    payload = submission.read_bytes()
    masked_file = MaskedFileEvidence(
        relative_path="log_standard_4_22_to_5_08_pure.csv",
        sha256="a" * 64,
        size_bytes=100,
        data_rows=2,
        final_rows_masked=1,
    )
    masked_view = MaskedViewEvidence(
        files=(masked_file,),
        final_rows_masked=1,
        final_outcome_cells_replaced=11,
        digest="b" * 64,
    )
    stdout = "checked one final row\n"
    return OrganizerCheckEvidence(
        starter_manifest_sha256="c" * 64,
        submission_sha256=hashlib.sha256(payload).hexdigest(),
        submission_size_bytes=len(payload),
        masked_view=masked_view,
        checker_command=(
            "python",
            "-B",
            "submit.py",
            "submission.csv",
            "--data_dir",
            "<private-masked-data-view>",
            "--split",
            "test",
            "--check",
        ),
        checker_returncode=0,
        checker_stdout=stdout,
        checker_stderr="",
        checker_stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
        checker_stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )


def _metadata(
    check: OrganizerCheckEvidence,
    sources: FinalBundleSources,
) -> FinalBundleMetadata:
    replay = json.loads((sources.replay / "evidence.json").read_text(encoding="ascii"))
    identity = replay["identity"]
    environment = json.loads(sources.environment.read_text(encoding="ascii"))
    return FinalBundleMetadata(
        benchmark_identity={"name": "KuaiRand-Pure", "digest": "d" * 64},
        starter_identity={"manifest_sha256": check.starter_manifest_sha256},
        data_identity={"archive_sha256": "e" * 64, "canonical_digest": "f" * 64},
        selected_experiment="pairwise-confirmed",
        lineage=("fm-official", "pairwise-screen", "pairwise-confirmed"),
        status=FinalStatus.MATERIALLY_CONFIRMED,
        validation_metrics={"GAUC": 0.61, "nDCG@5": 0.53, "primary": 0.57},
        seed_summary={"seeds": [0, 1, 2], "mean_primary": 0.57},
        inner_fold_results=({"fold": "late-train", "primary": 0.56},),
        scientific_artifact_hashes={
            "source": identity["source_sha256"],
            "config": identity["config_sha256"],
            "features": identity["features_sha256"],
            "checkpoint": identity["checkpoint_sha256"],
            "predictions": identity["validation_prediction_artifact_sha256"],
        },
        environment_and_resource_usage={
            "environment_sha256": identity["environment_sha256"],
            "device": "cpu",
            "peak_rss_bytes": 1024,
            "locked_environment": True,
            "runtime_identity": {
                "schema_version": 1,
                "project_source_digest": "9" * 64,
                "environment_digest": identity["environment_sha256"],
                "uv_lock_sha256": environment["uv_lock_sha256"],
                "dependency_groups": ["research-tree"],
            },
        },
        campaign_totals={
            "attempt_count": 8,
            "scientific_iteration_count": 3,
            "launch_count": 12,
            "elapsed_seconds": 100.5,
            "manual_intervention_count": 0,
        },
        known_limitations=("Hidden-test improvement is unverified until organizer scoring.",),
        unresolved_organizer_questions=(),
    )


def test_final_bundle_is_complete_hashed_and_exclusively_published(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    destination = tmp_path / "final"

    result = create_final_bundle(
        destination,
        sources=sources,
        metadata=_metadata(check, sources),
        organizer_check=check,
    )

    assert result.root == destination.resolve()
    assert result.submission_sha256 == check.submission_sha256
    assert (
        result.manifest_sha256
        == hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()
    )
    for path in ("manifest.json", *REQUIRED_FILE_PATHS, *REQUIRED_DIRECTORY_PATHS):
        assert (destination / path).exists()

    manifest = json.loads((destination / "manifest.json").read_text(encoding="ascii"))
    assert manifest["selection"] == {
        "lineage": ["fm-official", "pairwise-screen", "pairwise-confirmed"],
        "selected_experiment": "pairwise-confirmed",
        "status": "materially_confirmed",
    }
    assert manifest["scientific_artifact_hashes"]["submission"] == check.submission_sha256
    assert manifest["components"]["required_paths"][0] == "manifest.json"
    assert set(manifest["components"]["roots"]) == {
        *REQUIRED_FILE_PATHS,
        *REQUIRED_DIRECTORY_PATHS,
    }
    for entry in manifest["components"]["files"]:
        artifact = destination / entry["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]

    verification = json.loads((destination / "verification.json").read_text(encoding="ascii"))
    assert verification["assertions"]["final_outcomes_masked_before_organizer_load"] is True
    assert verification["organizer_check"]["mode"] == "check_only"
    assert "--score" not in verification["organizer_check"]["command"]
    receipt_path = destination / "prepublication-resource.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    assert receipt["scope"] == "final_bundle_prepublication"
    assert receipt["coverage"]["included_through"] == "scientific_identity_closure"
    assert receipt["coverage"]["excluded_tail"] == [
        "receipt_and_manifest_serialization",
        "closed_file_verification_and_durability_flush",
        "exclusive_publication_and_postpublication_verification",
    ]
    assert receipt["resources"]["wall_seconds"] >= 0.0
    assert receipt["resources"]["cpu_seconds"] >= 0.0
    assert receipt["resources"]["peak_rss_bytes"] > 0
    assert manifest["prepublication_resource_receipt"] == {
        "path": "prepublication-resource.json",
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    assert (destination / "reproduce.sh").stat().st_mode & 0o111


def test_production_verifier_closes_the_exact_bundle_emitted_by_the_producer(
    tmp_path: Path,
) -> None:
    fallback_id = "official-fm-fallback-seed-4"
    sources = _sources(tmp_path, candidate_id=fallback_id)
    check = _organizer_evidence(sources.submission)
    destination = tmp_path / "final"
    metadata = replace(
        _metadata(check, sources),
        selected_experiment=fallback_id,
        lineage=(fallback_id,),
        status=FinalStatus.BASELINE_REPRODUCED,
        seed_summary={
            "schema_version": 1,
            "seeds": [4],
            "representative_seed": 4,
            "matched_confirmation": False,
            "derived_status": FinalStatus.BASELINE_REPRODUCED.value,
            "confirmation_is_controller_derived": True,
        },
        inner_fold_results=({"fold": "official qualification"},),
    )

    emitted = create_final_bundle(
        destination,
        sources=sources,
        metadata=metadata,
        organizer_check=check,
    )
    verified = production._close_bundle_directories(emitted.root)

    assert verified.manifest_sha256 == emitted.manifest_sha256
    assert verified.manifest["prepublication_resource_receipt"] == {
        "path": "prepublication-resource.json",
        "sha256": hashlib.sha256(
            (destination / "prepublication-resource.json").read_bytes()
        ).hexdigest(),
    }
    assert all(
        path.stat().st_mode & 0o222 == 0
        for path in (destination, *destination.rglob("*"))
        if path.is_dir()
    )


def test_generated_bundle_closes_directory_artifact_identities(tmp_path: Path) -> None:
    sources = _sources(tmp_path, generated=True)
    check = _organizer_evidence(sources.submission)
    metadata = _metadata(check, sources)

    result = create_final_bundle(
        tmp_path / "final",
        sources=sources,
        metadata=metadata,
        organizer_check=check,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="ascii"))
    assert manifest["scientific_artifact_hashes"] == {
        **metadata.scientific_artifact_hashes,
        "submission": check.submission_sha256,
    }
    assert {path.name for path in (result.root / "preprocessing").iterdir()} == {
        "validation.npy",
        "final.npy",
        "fm-encoding.npz",
    }
    assert {path.name for path in (result.root / "model").iterdir()} == {
        "tree-model.txt",
        "fm-checkpoint.npz",
    }


def test_bundle_rejects_scientific_hashes_unrelated_to_copied_evidence(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    destination = tmp_path / "final"
    metadata = replace(
        _metadata(check, sources),
        scientific_artifact_hashes={
            "source": "1" * 64,
            "config": "2" * 64,
            "features": "3" * 64,
            "checkpoint": "4" * 64,
            "predictions": "5" * 64,
        },
    )

    with pytest.raises(FinalBundleError, match="scientific"):
        create_final_bundle(
            destination,
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


@pytest.mark.parametrize(
    ("identity_name", "relative_path"),
    (
        ("source", "source/candidate.py"),
        ("config", "config/artifact"),
        ("features", "preprocessing/artifact"),
        ("checkpoint", "model/artifact"),
        (
            "predictions",
            "validation-evidence/reference-validation-predictions.npy",
        ),
    ),
)
def test_fallback_bundle_rejects_scientific_component_byte_tampering(
    tmp_path: Path,
    identity_name: str,
    relative_path: str,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    metadata = _metadata(check, sources)
    copied_source = sources.source.parent / relative_path
    copied_source.write_bytes(copied_source.read_bytes() + b"tampered")

    with pytest.raises(
        FinalBundleError,
        match=rf"scientific_artifact_hashes\.{identity_name}",
    ):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


@pytest.mark.parametrize(
    ("identity_name", "relative_path"),
    (
        ("features", "preprocessing/validation.npy"),
        ("checkpoint", "model/tree-model.txt"),
    ),
)
def test_generated_bundle_rejects_directory_artifact_member_tampering(
    tmp_path: Path,
    identity_name: str,
    relative_path: str,
) -> None:
    sources = _sources(tmp_path, generated=True)
    check = _organizer_evidence(sources.submission)
    metadata = _metadata(check, sources)
    copied_source = sources.source.parent / relative_path
    copied_source.write_bytes(copied_source.read_bytes() + b"tampered")

    with pytest.raises(
        FinalBundleError,
        match=rf"scientific_artifact_hashes\.{identity_name}",
    ):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


@pytest.mark.parametrize(
    "replay_identity_name",
    (
        "source_sha256",
        "config_sha256",
        "features_sha256",
        "checkpoint_sha256",
        "validation_prediction_artifact_sha256",
    ),
)
def test_bundle_rejects_replay_identity_unrelated_to_scientific_manifest(
    tmp_path: Path,
    replay_identity_name: str,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    metadata = _metadata(check, sources)
    replay_path = sources.replay / "evidence.json"
    replay = json.loads(replay_path.read_text(encoding="ascii"))
    replay["identity"][replay_identity_name] = "8" * 64
    replay_path.write_text(
        json.dumps(
            replay,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(FinalBundleError, match=replay_identity_name):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


@pytest.mark.parametrize(
    ("section", "field", "message"),
    (
        ("validation", "reference_prediction_digest", "reference prediction semantics"),
        ("validation", "replay_prediction_file_sha256", "validation prediction file"),
        ("validation", "public_submission_sha256", "public-validation CSV"),
        ("final", "prediction_file_sha256", "final prediction file"),
        ("final", "submission_sha256", "final submission"),
    ),
)
def test_bundle_rejects_replay_claims_unrelated_to_copied_bytes(
    tmp_path: Path,
    section: str,
    field: str,
    message: str,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    metadata = _metadata(check, sources)
    replay_path = sources.replay / "evidence.json"
    replay = json.loads(replay_path.read_text(encoding="ascii"))
    replay[section][field] = "8" * 64
    replay_path.write_text(
        json.dumps(
            replay,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(FinalBundleError, match=message):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


def test_bundle_rejects_replay_data_identity_unrelated_to_declared_dataset(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    metadata = replace(
        _metadata(check, sources),
        data_identity={"canonical_digest": "8" * 64},
    )

    with pytest.raises(FinalBundleError, match="data identity"):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()


def test_bundle_rejects_replay_environment_unrelated_to_declared_environment(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    metadata = replace(
        _metadata(check, sources),
        environment_and_resource_usage={
            "environment_sha256": "8" * 64,
            "device": "cpu",
        },
    )

    with pytest.raises(FinalBundleError, match="environment identity"):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()


def test_bundle_rejects_runtime_identity_unrelated_to_copied_environment(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    metadata = _metadata(check, sources)
    metadata = replace(
        metadata,
        environment_and_resource_usage={
            **metadata.environment_and_resource_usage,
            "runtime_identity": {
                "schema_version": 1,
                "project_source_digest": "9" * 64,
                "environment_digest": "8" * 64,
                "uv_lock_sha256": "6" * 64,
                "dependency_groups": ["research-tree"],
            },
        },
    )

    with pytest.raises(FinalBundleError, match="runtime environment identity"):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()


def test_bundle_rejects_self_consistent_environment_byte_tampering(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    metadata = _metadata(check, sources)
    environment = json.loads(sources.environment.read_text(encoding="ascii"))
    environment["uv_lock_sha256"] = "5" * 64
    environment_body = {key: value for key, value in environment.items() if key != "digest"}
    canonical_body = json.dumps(
        environment_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    environment["digest"] = hashlib.sha256(
        b"kuairand-environment-v1\0" + canonical_body
    ).hexdigest()
    sources.environment.write_text(
        json.dumps(
            environment,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(FinalBundleError, match="runtime environment identity"):
        create_final_bundle(
            tmp_path / "final",
            sources=sources,
            metadata=metadata,
            organizer_check=check,
        )

    assert not (tmp_path / "final").exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


def test_bundle_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    destination = tmp_path / "final"
    destination.mkdir()
    sentinel = _write(destination / "user-owned.txt", "keep me\n")

    with pytest.raises(FinalBundleError, match="already exists"):
        create_final_bundle(
            destination,
            sources=sources,
            metadata=_metadata(check, sources),
            organizer_check=check,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_submission_change_after_check_fails_and_cleans_staging(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    sources.submission.write_text("row_id,user_id,video_id,score\n0,u,v,0.999\n", encoding="utf-8")
    destination = tmp_path / "final"

    with pytest.raises(FinalBundleError, match="changed after"):
        create_final_bundle(
            destination,
            sources=sources,
            metadata=_metadata(check, sources),
            organizer_check=check,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


def test_cancellation_during_bundle_build_prevents_exclusive_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    destination = tmp_path / "final"
    cancellation = threading.Event()
    original_verify = bundle_module._verify_closed_files

    def cancel_after_staging_is_complete(
        root: Path,
        evidence: Sequence[BundleFileEvidence],
    ) -> None:
        original_verify(root, evidence)
        if root.name.startswith(".final.staging-"):
            cancellation.set()

    monkeypatch.setattr(bundle_module, "_verify_closed_files", cancel_after_staging_is_complete)

    with pytest.raises(FinalBundleCancelledError, match="cancelled"):
        create_final_bundle(
            destination,
            sources=sources,
            metadata=_metadata(check, sources),
            organizer_check=check,
            cancel_event=cancellation,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))


def test_symlink_in_required_component_fails_closed_and_cleans_staging(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    check = _organizer_evidence(sources.submission)
    linked = sources.source / "linked.py"
    linked.symlink_to(sources.source / "artifact.txt")
    destination = tmp_path / "final"

    with pytest.raises(FinalBundleError, match="contains a symlink"):
        create_final_bundle(
            destination,
            sources=sources,
            metadata=_metadata(check, sources),
            organizer_check=check,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".final.staging-*"))
