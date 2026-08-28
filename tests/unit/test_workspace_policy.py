from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kuairand_agent.execution.artifacts import ArtifactKind, ArtifactStore, DirectoryArtifactRef
from kuairand_agent.execution.policy import (
    ApprovedInput,
    CandidateInputRole,
    DeclaredOutput,
    OutputDeclaration,
    SourceEntry,
    SourceManifest,
    SplitRole,
    WorkspacePolicy,
    WorkspacePolicyError,
)
from kuairand_agent.execution.workspace import (
    CandidateWorkspace,
    WorkspaceExistsError,
    WorkspaceMaterializer,
    WorkspaceSpec,
)


def _source_snapshot(tmp_path: Path, store: ArtifactStore) -> DirectoryArtifactRef:
    source = tmp_path / "candidate-source"
    source.mkdir()
    (source / "candidate.py").write_text("print('candidate')\n", encoding="utf-8")
    (source / "config.json").write_text('{"model":"fm"}\n', encoding="utf-8")
    return store.put_directory(source, kind=ArtifactKind.SOURCE)


def _workspace(
    tmp_path: Path,
    pytest_request: pytest.FixtureRequest,
    *,
    execution_id: str = "exec-001",
    split_role: SplitRole = SplitRole.TRAIN,
    request: dict[str, object] | None = None,
) -> tuple[ArtifactStore, WorkspacePolicy, CandidateWorkspace, ApprovedInput]:
    store = ArtifactStore(tmp_path / "artifact-store", max_object_bytes=1024 * 1024)
    snapshot = _source_snapshot(tmp_path, store)
    input_ref = store.put_bytes(b"approved capability", kind=ArtifactKind.INPUT)
    approved = ApprovedInput(
        "train_inputs",
        CandidateInputRole.TRAIN_INPUTS,
        input_ref,
    )
    policy = WorkspacePolicy(
        max_input_file_bytes=1024 * 1024,
        max_input_total_bytes=1024 * 1024,
        max_output_file_bytes=1024,
        max_output_total_bytes=2048,
        max_temp_bytes=2048,
    )
    materializer = WorkspaceMaterializer(
        tmp_path / "workspaces", artifact_store=store, policy=policy
    )
    workspace = materializer.materialize(
        WorkspaceSpec(
            execution_id=execution_id,
            split_role=split_role,
            source_snapshot=snapshot,
            approved_inputs=(approved,),
            request_payload=request or {"seed": 0, "mode": "train"},
            output_limit_bytes=2048,
            temp_limit_bytes=2048,
        )
    )
    pytest_request.addfinalizer(lambda: materializer.cleanup(workspace))
    return store, policy, workspace, approved


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/absolute.py",
        "../escape.py",
        "nested/../../escape.py",
        ".hidden.py",
        "nested/.control/config.json",
        "evaluate.py",
        "package/protected.py",
        "kuairand_agent/model.py",
        "candidate.tar.gz",
        "native.so",
        "sitecustomize.py",
    ),
)
def test_source_policy_rejects_unsafe_or_trusted_paths(unsafe_path: str) -> None:
    manifest = SourceManifest(
        entries=(SourceEntry(unsafe_path, "a" * 64, 1),),
        artifact_manifest_sha256="b" * 64,
    )

    with pytest.raises(WorkspacePolicyError):
        WorkspacePolicy().validate_source_manifest(manifest)


def test_source_policy_accepts_exact_bounded_candidate_inventory() -> None:
    manifest = SourceManifest(
        entries=(
            SourceEntry("candidate.py", "a" * 64, 10),
            SourceEntry("lib/pairwise.py", "b" * 64, 20),
        ),
        artifact_manifest_sha256="c" * 64,
    )

    WorkspacePolicy().validate_source_manifest(manifest)
    assert len(manifest.digest) == 64


def test_workspace_has_only_the_approved_private_inventory(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    _, policy, workspace, _ = _workspace(tmp_path, request)

    assert {path.name for path in workspace.root.iterdir()} == {
        "home",
        "inputs",
        "output",
        "process.json",
        "request.json",
        "source",
        "tmp",
        "workspace-manifest.json",
    }
    assert not tuple(workspace.home_dir.iterdir())
    assert not tuple(workspace.output_dir.iterdir())
    assert not tuple(workspace.tmp_dir.iterdir())
    assert workspace.root.stat().st_mode & 0o077 == 0
    assert workspace.source_dir.stat().st_mode & 0o222 == 0
    assert workspace.inputs_dir.stat().st_mode & 0o222 == 0
    assert workspace.output_dir.stat().st_mode & 0o200
    assert not tuple(workspace.root.rglob(".env"))
    assert not tuple(workspace.root.rglob("*credential*"))
    policy.validate_workspace(workspace)


def test_candidate_input_copy_cannot_mutate_the_object_store(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    store, policy, workspace, approved = _workspace(tmp_path, request)
    copied = workspace.root / workspace.input_files[0].relative_path
    object_path = store.verify(approved.artifact)

    assert (copied.stat().st_dev, copied.stat().st_ino) != (
        object_path.stat().st_dev,
        object_path.stat().st_ino,
    )
    copied.chmod(0o600)
    copied.write_bytes(b"candidate mutation")

    assert store.verify(approved.artifact).read_bytes() == b"approved capability"
    with pytest.raises(WorkspacePolicyError, match="immutable copy"):
        policy.validate_workspace(workspace)


def test_request_rejects_credentials_raw_archives_and_alignment() -> None:
    policy = WorkspacePolicy()

    for payload in (
        {"api_key": "secret"},
        {"input": "KuaiRand-Pure.tar.gz"},
        {"row_id": [0, 1]},
        {"endpoint": "https://example.invalid"},
        {"scorer": "evaluate.py"},
    ):
        with pytest.raises(WorkspacePolicyError):
            policy.validate_request_payload(SplitRole.TRAIN, payload)


def test_outer_and_final_requests_reject_protected_outcomes() -> None:
    policy = WorkspacePolicy()

    for split_role in (SplitRole.INNER_VALID, SplitRole.OUTER_VALID, SplitRole.FINAL):
        with pytest.raises(WorkspacePolicyError, match="protected outcomes"):
            policy.validate_request_payload(split_role, {"labels": [0, 1]})


def test_input_role_is_phase_specific() -> None:
    from kuairand_agent.execution.artifacts import ArtifactRef

    train_targets = ApprovedInput(
        "training_capability",
        CandidateInputRole.TRAIN_TARGETS,
        ArtifactRef("a" * 64, 4, ArtifactKind.INPUT),
    )

    WorkspacePolicy().validate_approved_inputs(SplitRole.TRAIN, (train_targets,))
    with pytest.raises(WorkspacePolicyError, match="forbidden for split"):
        WorkspacePolicy().validate_approved_inputs(SplitRole.OUTER_VALID, (train_targets,))


def test_protected_outer_target_handle_is_rejected_even_if_mislabeled() -> None:
    from kuairand_agent.execution.artifacts import ArtifactRef

    disguised = ApprovedInput(
        "outer_valid_targets",
        CandidateInputRole.OUTER_VALID_INPUTS,
        ArtifactRef("a" * 64, 4, ArtifactKind.INPUT),
    )

    with pytest.raises(WorkspacePolicyError, match="protected"):
        WorkspacePolicy().validate_approved_inputs(SplitRole.OUTER_VALID, (disguised,))


def test_materialization_refuses_to_overwrite_existing_workspace(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    store = ArtifactStore(tmp_path / "artifact-store")
    snapshot = _source_snapshot(tmp_path, store)
    input_ref = store.put_bytes(b"inputs", kind=ArtifactKind.INPUT)
    approved = ApprovedInput("train_inputs", CandidateInputRole.TRAIN_INPUTS, input_ref)
    materializer = WorkspaceMaterializer(tmp_path / "workspaces", artifact_store=store)
    spec = WorkspaceSpec(
        "same-execution",
        SplitRole.TRAIN,
        snapshot,
        (approved,),
        {"seed": 0},
        1024,
        1024,
    )
    first = materializer.materialize(spec)
    request.addfinalizer(lambda: materializer.cleanup(first))
    original_manifest = first.manifest_path.read_bytes()

    with pytest.raises(WorkspaceExistsError):
        materializer.materialize(spec)

    assert first.manifest_path.read_bytes() == original_manifest
    assert not tuple((tmp_path / "workspaces").glob(".staging-*"))


def test_workspace_manifests_are_deterministic_across_roots(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    store = ArtifactStore(tmp_path / "artifact-store")
    snapshot = _source_snapshot(tmp_path, store)
    input_ref = store.put_bytes(b"inputs", kind=ArtifactKind.INPUT)
    approved = ApprovedInput("train_inputs", CandidateInputRole.TRAIN_INPUTS, input_ref)
    spec = WorkspaceSpec(
        "exec-deterministic",
        SplitRole.TRAIN,
        snapshot,
        (approved,),
        {"seed": 7, "threads": 1},
        1024,
        1024,
    )

    first_materializer = WorkspaceMaterializer(tmp_path / "left-workspaces", artifact_store=store)
    second_materializer = WorkspaceMaterializer(tmp_path / "right-workspaces", artifact_store=store)
    first = first_materializer.materialize(spec)
    second = second_materializer.materialize(spec)
    request.addfinalizer(lambda: first_materializer.cleanup(first))
    request.addfinalizer(lambda: second_materializer.cleanup(second))

    assert first.request_path.read_bytes() == second.request_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.manifest_digest == second.manifest_digest


def test_declared_outputs_produce_a_deterministic_inventory(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    _, policy, workspace, _ = _workspace(tmp_path, request)
    checkpoint = workspace.output_dir / "checkpoint" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    result = workspace.output_dir / "candidate_result.json"
    result.write_text('{"schema_version":1}\n', encoding="utf-8")
    declaration = OutputDeclaration(
        (
            DeclaredOutput("candidate_result.json", 128),
            DeclaredOutput("checkpoint/model.ckpt", 128),
        )
    )

    first = policy.validate_outputs(workspace, declaration)
    second = policy.validate_outputs(workspace, declaration)

    assert first == second
    assert first.total_size_bytes == len(b"weights") + len(b'{"schema_version":1}\n')
    assert tuple(file.path for file in first.files) == (
        "candidate_result.json",
        "checkpoint/model.ckpt",
    )


def test_undeclared_oversized_linked_and_special_outputs_fail(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    _, policy, workspace, _ = _workspace(tmp_path, request)
    output = workspace.output_dir / "scores.npy"
    output.write_bytes(b"12345")

    with pytest.raises(WorkspacePolicyError, match="differs"):
        policy.validate_outputs(workspace, OutputDeclaration(()))
    with pytest.raises(WorkspacePolicyError, match="exceeds"):
        policy.validate_outputs(
            workspace,
            OutputDeclaration((DeclaredOutput("scores.npy", 4),)),
        )

    output.unlink()
    external = tmp_path / "external.npy"
    external.write_bytes(b"safe")
    output.symlink_to(external)
    with pytest.raises(WorkspacePolicyError, match="symlink"):
        policy.validate_outputs(
            workspace,
            OutputDeclaration((DeclaredOutput("scores.npy", 10),)),
        )

    output.unlink()
    os.mkfifo(output)
    with pytest.raises(WorkspacePolicyError, match="special"):
        policy.validate_outputs(
            workspace,
            OutputDeclaration((DeclaredOutput("scores.npy", 10),)),
        )


def test_request_and_manifest_do_not_contain_absolute_host_paths(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    _, _, workspace, _ = _workspace(tmp_path, request)
    parsed_request = json.loads(workspace.request_path.read_text(encoding="ascii"))
    manifest_text = workspace.manifest_path.read_text(encoding="ascii")

    assert (
        parsed_request["approved_inputs"][0]["workspace_path"] == "inputs/000-train_inputs.artifact"
    )
    assert str(tmp_path) not in manifest_text
    assert str(tmp_path) not in workspace.request_path.read_text(encoding="ascii")


@pytest.mark.parametrize("relative_path", ("request.json", "workspace-manifest.json"))
def test_workspace_policy_detects_trusted_metadata_tampering(
    tmp_path: Path, relative_path: str, request: pytest.FixtureRequest
) -> None:
    _, policy, workspace, _ = _workspace(tmp_path, request)
    path = workspace.root / relative_path
    original = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(original + b" ")
    path.chmod(0o444)

    try:
        with pytest.raises(WorkspacePolicyError, match=r"digest|manifest bytes"):
            policy.validate_workspace(workspace)
    finally:
        path.chmod(0o600)
        path.write_bytes(original)
        path.chmod(0o444)


def test_trusted_cleanup_removes_read_only_tree_without_following_symlinks(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifact-store")
    snapshot = _source_snapshot(tmp_path, store)
    input_ref = store.put_bytes(b"inputs", kind=ArtifactKind.INPUT)
    approved = ApprovedInput("train_inputs", CandidateInputRole.TRAIN_INPUTS, input_ref)
    materializer = WorkspaceMaterializer(tmp_path / "workspaces", artifact_store=store)
    workspace = materializer.materialize(
        WorkspaceSpec(
            "cleanup-exec",
            SplitRole.TRAIN,
            snapshot,
            (approved,),
            {"seed": 0},
            1024,
            1024,
        )
    )
    external = tmp_path / "external.txt"
    external.write_text("do not follow", encoding="utf-8")
    (workspace.tmp_dir / "link").symlink_to(external)
    os.mkfifo(workspace.output_dir / "candidate.fifo")

    materializer.remove(workspace)

    assert not workspace.root.exists()
    assert external.read_text(encoding="utf-8") == "do not follow"
