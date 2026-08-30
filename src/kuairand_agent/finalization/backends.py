"""Trusted provider-free inference backends for allowlisted frozen replay recipes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.artifacts import StarterFMCheckpoint, load_checkpoint
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.organizer import load_verified_organizer
from kuairand_agent.baselines.starter_fm import StarterFMConfig
from kuairand_agent.candidate_api.protocol import (
    PredictionExpectation,
    validate_prediction_outputs,
)
from kuairand_agent.candidates.fusion import fuse_ranked_members, fuse_ranked_predictions
from kuairand_agent.contract import sha256_file, verify_starter_kit
from kuairand_agent.data.capabilities import CandidateInputs, DataPhase
from kuairand_agent.execution.policy import SplitRole
from kuairand_agent.execution.runner import ExecutionOutcome, ExecutionSpec, Runner
from kuairand_agent.finalization.recipe import (
    GeneratedLambdaRankEnsembleReplayRecipe,
    GeneratedLambdaRankReplayRecipe,
    OfficialFMMemberRecipe,
    OfficialFMReplayRecipe,
    ReplayRecipe,
    ReplayRecipeError,
    load_replay_recipe,
)
from kuairand_agent.finalization.replay import (
    FrozenCandidateWorkspace,
    FrozenReplayIdentity,
    ReplayBackend,
    ReplayCancelledError,
    ReplayError,
)

type Float64Vector = npt.NDArray[np.float64]
type Int32Matrix = npt.NDArray[np.int32]

_FM_FIELDS: Final = ("user_id", "video_id", "author_id", "tab", "duration_ms")


def _require_workspace(workspace: FrozenCandidateWorkspace) -> None:
    if not isinstance(workspace, FrozenCandidateWorkspace):
        raise ReplayError("workspace must be a FrozenCandidateWorkspace")
    for name in ("source_dir", "config_path", "features_path", "checkpoint_path"):
        path = getattr(workspace, name)
        if not isinstance(path, Path) or path.is_symlink() or not path.exists():
            raise ReplayError(f"frozen replay {name} is unavailable or unsafe")


def _require_recipe_artifact(workspace: FrozenCandidateWorkspace, recipe: ReplayRecipe) -> None:
    restored = load_replay_recipe(
        workspace.config_path,
        expected_sha256=workspace.identity.config_sha256,
    )
    if restored != recipe or recipe.digest != workspace.identity.config_sha256:
        raise ReplayError("frozen config artifact differs from the replay backend recipe")


def _require_common_identity(
    workspace: FrozenCandidateWorkspace,
    inputs: CandidateInputs,
    *,
    source_artifact_sha256: str,
    feature_artifact_sha256: str,
    checkpoint_artifact_sha256: str,
    data_sha256: str,
    expected_inputs_digest: str,
    phase: DataPhase,
) -> None:
    identity = workspace.identity
    if (
        identity.source_sha256 != source_artifact_sha256
        or identity.features_sha256 != feature_artifact_sha256
        or identity.checkpoint_sha256 != checkpoint_artifact_sha256
        or identity.data_sha256 != data_sha256
    ):
        raise ReplayError("replay recipe artifact identity differs from the frozen candidate")
    if not isinstance(inputs, CandidateInputs) or inputs.phase is not phase:
        raise ReplayError(f"replay backend requires a {phase.value} CandidateInputs capability")
    if inputs.digest != expected_inputs_digest:
        raise ReplayError("candidate input capability differs from the replay recipe")


def _fm_matrix(encoding: StarterEncoding, inputs: CandidateInputs) -> Int32Matrix:
    if tuple(column.name for column in inputs.schema) != _FM_FIELDS:
        raise ReplayError("official FM replay requires exactly the five organizer input fields")
    try:
        durations = np.asarray(inputs.column("duration_ms"), dtype=np.float64)
        raw_fields: tuple[tuple[str, ...], ...] = (
            tuple(str(value) for value in inputs.column("user_id")),
            tuple(str(value) for value in inputs.column("video_id")),
            tuple(str(value) for value in inputs.column("author_id")),
            tuple(str(value) for value in inputs.column("tab")),
            tuple(
                str(int(value)) for value in np.searchsorted(np.asarray(encoding.edges), durations)
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ReplayError("official FM replay could not encode candidate inputs") from exc
    matrix = np.empty((inputs.row_count, len(_FM_FIELDS)), dtype=np.int32)
    for column, (values, vocabulary, offset) in enumerate(
        zip(raw_fields, encoding.vocabs, encoding.offsets, strict=True)
    ):
        lookup = {value: index for index, value in enumerate(vocabulary)}
        unknown = len(vocabulary)
        matrix[:, column] = np.asarray(
            [lookup.get(value, unknown) + offset for value in values], dtype=np.int32
        )
    if matrix.size == 0 or int(matrix.min()) < 0 or int(matrix.max()) >= encoding.total_dim:
        raise ReplayError("official FM replay produced an invalid encoded matrix")
    return cast(Int32Matrix, np.ascontiguousarray(matrix))


def _load_fm(
    *,
    encoding_path: Path,
    checkpoint_path: Path,
    member: OfficialFMMemberRecipe,
) -> tuple[StarterEncoding, StarterFMCheckpoint, StarterFMConfig]:
    try:
        encoding = StarterEncoding.load(
            encoding_path,
            expected_file_sha256=member.encoding_sha256,
        )
        if encoding.digest != member.encoding_digest:
            raise ReplayError("official FM encoding logical digest differs from recipe")
        config = StarterFMConfig(seed=member.seed)
        if config.digest != member.config_digest:
            raise ReplayError("official FM config digest differs from immutable hyperparameters")
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_file_sha256=member.checkpoint_sha256,
            expected_checkpoint_digest=member.checkpoint_digest,
            expected_encoding_digest=member.encoding_digest,
            expected_starter_manifest_digest=member.starter_manifest_sha256,
            expected_config_digest=member.config_digest,
            expected_seed=member.seed,
        )
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError("official FM replay artifacts failed identity validation") from exc
    return encoding, checkpoint, config


def _fm_predictions(
    *,
    source_dir: Path,
    inputs: CandidateInputs,
    encoding: StarterEncoding,
    checkpoint: StarterFMCheckpoint,
    config: StarterFMConfig,
    expected_starter_manifest_sha256: str,
) -> Float64Vector:
    try:
        verification = verify_starter_kit(source_dir)
        if verification.manifest_sha256 != expected_starter_manifest_sha256:
            raise ReplayError("organizer source differs from the official FM checkpoint")
        organizer = load_verified_organizer(source_dir)
        matrix = _fm_matrix(encoding, inputs)
        model = organizer.baseline.FM(
            encoding.total_dim,
            k=config.k,
            lr=config.learning_rate,
            l2=config.l2,
            seed=config.seed,
        )
        model.V = np.array(checkpoint.V, dtype=np.float32, copy=True)
        model.W = np.array(checkpoint.W, dtype=np.float32, copy=True)
        model.b = np.float32(checkpoint.b)
        raw = model.predict(matrix, bs=config.predict_batch_size)
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError("verified organizer FM inference failed") from exc
    scores = np.asarray(raw)
    if scores.shape != (inputs.row_count,) or scores.dtype != np.dtype("float32"):
        raise ReplayError("verified organizer FM returned invalid prediction shape or dtype")
    result = np.ascontiguousarray(scores, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ReplayError("verified organizer FM returned non-finite predictions")
    result.setflags(write=False)
    return result


def _fm_checkpoint_predictions(
    *,
    inputs: CandidateInputs,
    encoding: StarterEncoding,
    checkpoint: StarterFMCheckpoint,
    config: StarterFMConfig,
) -> Float64Vector:
    """Run the pinned organizer FM equation without requiring a second source tree.

    This path is used only for the official-FM member of a generated-candidate fusion.  The
    checkpoint loader has already bound the immutable organizer source manifest and frozen
    hyperparameters; inference mirrors the published NumPy ``FM.logits`` expression exactly.
    """

    matrix = _fm_matrix(encoding, inputs)
    batches: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(matrix), config.predict_batch_size):
        batch = matrix[start : start + config.predict_batch_size]
        embeddings = checkpoint.V[batch]
        summed = embeddings.sum(1)
        interaction = 0.5 * ((summed**2).sum(1) - (embeddings**2).sum((1, 2)))
        batches.append(checkpoint.b + checkpoint.W[batch].sum(1) + interaction)
    raw = np.concatenate(batches)
    if raw.dtype != np.dtype("float32") or raw.shape != (inputs.row_count,):
        raise ReplayError("official FM fusion member returned invalid predictions")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ReplayError("official FM fusion member returned non-finite predictions")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class OfficialFMReplayBackend:
    """Immutable organizer-source/checkpoint replay over a label-free candidate capability."""

    recipe: OfficialFMReplayRecipe

    def _predict(
        self,
        *,
        workspace: FrozenCandidateWorkspace,
        inputs: CandidateInputs,
        phase: DataPhase,
        expected_inputs_digest: str,
    ) -> Float64Vector:
        _require_workspace(workspace)
        _require_recipe_artifact(workspace, self.recipe)
        _require_common_identity(
            workspace,
            inputs,
            source_artifact_sha256=self.recipe.source_artifact_sha256,
            feature_artifact_sha256=self.recipe.feature_artifact_sha256,
            checkpoint_artifact_sha256=self.recipe.checkpoint_artifact_sha256,
            data_sha256=self.recipe.data_sha256,
            expected_inputs_digest=expected_inputs_digest,
            phase=phase,
        )
        if (
            sha256_file(workspace.features_path) != self.recipe.fm_member.encoding_sha256
            or sha256_file(workspace.checkpoint_path) != self.recipe.fm_member.checkpoint_sha256
        ):
            raise ReplayError("official FM member files differ from the replay recipe")
        encoding, checkpoint, config = _load_fm(
            encoding_path=workspace.features_path,
            checkpoint_path=workspace.checkpoint_path,
            member=self.recipe.fm_member,
        )
        return _fm_predictions(
            source_dir=workspace.source_dir,
            inputs=inputs,
            encoding=encoding,
            checkpoint=checkpoint,
            config=config,
            expected_starter_manifest_sha256=self.recipe.fm_member.starter_manifest_sha256,
        )

    def replay_validation(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Float64Vector:
        return self._predict(
            workspace=workspace,
            inputs=inputs,
            phase=DataPhase.OUTER_VALID,
            expected_inputs_digest=self.recipe.validation_inputs_digest,
        )

    def predict_final(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Float64Vector:
        return self._predict(
            workspace=workspace,
            inputs=inputs,
            phase=DataPhase.FINAL,
            expected_inputs_digest=self.recipe.final_inputs_digest,
        )


def _regular_digest(path: Path, expected: str, name: str) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"{name} must be a regular non-symlink file")
    try:
        size = path.stat().st_size
        observed = sha256_file(path)
    except OSError as exc:
        raise ReplayError(f"{name} could not be verified") from exc
    if size <= 0 or observed != expected:
        raise ReplayError(f"{name} differs from the replay recipe")
    return observed, size


def _exact_regular_inventory(root: Path, expected: set[str], name: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ReplayError(f"{name} must be a real directory")
    observed: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink() or not candidate.is_file():
            if candidate.is_dir() and not candidate.is_symlink():
                continue
            raise ReplayError(f"{name} contains an unsafe member")
        observed.add(relative)
    if observed != expected:
        raise ReplayError(f"{name} inventory differs from the replay recipe")


def _feature_shape(path: Path, *, expected_rows: int, expected_features: int) -> None:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ReplayError("generated replay features are not one safe NumPy array") from exc
    try:
        if (
            not isinstance(values, np.ndarray)
            or values.shape != (expected_rows, expected_features)
            or values.dtype.kind not in "iuf"
        ):
            raise ReplayError("generated replay feature shape or dtype differs from recipe")
    finally:
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _copy_capability(source: Path, destination: Path, expected_sha256: str) -> int:
    _, size = _regular_digest(source, expected_sha256, "generated feature capability")
    destination.parent.mkdir(mode=0o700)
    try:
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o444, follow_symlinks=False)
    except OSError as exc:
        raise ReplayError("generated feature capability could not be privately copied") from exc
    _regular_digest(destination, expected_sha256, "copied generated feature capability")
    return size


def _write_request(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ReplayError("generated replay request could not be written") from exc


@dataclass(frozen=True, slots=True)
class GeneratedLambdaRankReplayBackend:
    """Fresh-process generated LambdaRank inference with optional fixed rank fusion."""

    recipe: GeneratedLambdaRankReplayRecipe
    interpreter: Path
    runner: Runner
    cancel_event: threading.Event | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, GeneratedLambdaRankReplayRecipe):
            raise ReplayRecipeError("recipe must be GeneratedLambdaRankReplayRecipe")
        if not isinstance(self.interpreter, Path) or not self.interpreter.is_absolute():
            raise ReplayRecipeError("generated replay interpreter must be an absolute path")
        if not isinstance(self.runner, Runner):
            raise ReplayRecipeError("generated replay runner must be Runner")
        if self.cancel_event is not None and not isinstance(self.cancel_event, threading.Event):
            raise ReplayRecipeError("cancel_event must be threading.Event or None")

    def _paths(
        self,
        workspace: FrozenCandidateWorkspace,
        *,
        phase: DataPhase,
    ) -> tuple[Path, Path, Path | None, Path | None, str]:
        feature_names = {"validation.npy", "final.npy"}
        checkpoint_names = {"tree-model.txt"}
        if self.recipe.fm_member is not None:
            feature_names.add("fm-encoding.npz")
            checkpoint_names.add("fm-checkpoint.npz")
        _exact_regular_inventory(workspace.features_path, feature_names, "preprocessing artifact")
        _exact_regular_inventory(workspace.checkpoint_path, checkpoint_names, "model artifact")
        if phase is DataPhase.OUTER_VALID:
            feature_path = workspace.features_path / "validation.npy"
            feature_digest = self.recipe.validation_features_sha256
        else:
            feature_path = workspace.features_path / "final.npy"
            feature_digest = self.recipe.final_features_sha256
        tree_checkpoint = workspace.checkpoint_path / "tree-model.txt"
        fm_encoding = (
            None if self.recipe.fm_member is None else workspace.features_path / "fm-encoding.npz"
        )
        fm_checkpoint = (
            None
            if self.recipe.fm_member is None
            else workspace.checkpoint_path / "fm-checkpoint.npz"
        )
        return feature_path, tree_checkpoint, fm_encoding, fm_checkpoint, feature_digest

    def _tree_predict(
        self,
        *,
        workspace: FrozenCandidateWorkspace,
        inputs: CandidateInputs,
        phase: DataPhase,
        feature_path: Path,
        feature_digest: str,
        tree_checkpoint: Path,
    ) -> Float64Vector:
        candidate_source = workspace.source_dir / "candidate.py"
        candidate_config = workspace.source_dir / "config.json"
        _regular_digest(
            candidate_source,
            self.recipe.candidate_source_sha256,
            "generated candidate.py",
        )
        _regular_digest(
            candidate_config,
            self.recipe.candidate_config_sha256,
            "generated config.json",
        )
        _regular_digest(
            tree_checkpoint,
            self.recipe.tree_checkpoint_sha256,
            "generated tree checkpoint",
        )
        _regular_digest(feature_path, feature_digest, "generated replay features")
        _feature_shape(
            feature_path,
            expected_rows=inputs.row_count,
            expected_features=self.recipe.feature_count,
        )

        action = Path(tempfile.mkdtemp(prefix=f"generated-{phase.value}-", dir=workspace.root))
        feature_copy = action / "inputs" / "features.npy"
        feature_size = _copy_capability(feature_path, feature_copy, feature_digest)
        split_role = SplitRole.OUTER_VALID if phase is DataPhase.OUTER_VALID else SplitRole.FINAL
        split_token = f"frozen-{phase.value.replace('_', '-')}"
        execution_id = f"replay-{phase.value.replace('_', '-')}-{action.name[-8:]}"
        request = {
            "schema_version": 1,
            "execution_id": execution_id,
            "split_role": split_role.value,
            "source_snapshot_sha256": self.recipe.source_artifact_sha256,
            "budgets": {
                "output_limit_bytes": 2 * 1024**3,
                "temp_limit_bytes": 2 * 1024**3,
            },
            "approved_inputs": [
                {
                    "name": "features",
                    "role": (
                        "outer_valid_inputs" if phase is DataPhase.OUTER_VALID else "final_inputs"
                    ),
                    "workspace_path": "inputs/features.npy",
                    "artifact": {
                        "schema_version": 1,
                        "algorithm": "sha256",
                        "sha256": feature_digest,
                        "size_bytes": feature_size,
                        "kind": "input",
                    },
                }
            ],
            "request": {
                "protocol_schema_version": 1,
                "source_digest": self.recipe.source_artifact_sha256,
                "config_digest": self.recipe.candidate_config_sha256,
                "data_digest": inputs.digest,
                "split_token": split_token,
                "features_handle": "features",
                "expected_count": inputs.row_count,
                "checkpoint_digest": self.recipe.tree_checkpoint_sha256,
            },
        }
        _write_request(action / "request.json", request)
        nonce = hashlib.sha256(
            b"kuairand-frozen-generated-replay-v1\0"
            + execution_id.encode("ascii")
            + inputs.digest.encode("ascii")
        ).hexdigest()[:32]
        spec = ExecutionSpec(
            execution_id=execution_id,
            nonce=nonce,
            interpreter=self.interpreter,
            arguments=(
                str(candidate_source),
                "predict",
                "--request",
                "request.json",
                "--checkpoint",
                str(tree_checkpoint),
                "--output",
                "output",
            ),
            workspace=action.resolve(strict=True),
            control_dir=workspace.root / f"control-{action.name}",
            timeout_seconds=float(self.recipe.timeout_seconds),
            memory_limit_bytes=self.recipe.memory_limit_bytes,
            workspace_disk_limit_bytes=4 * 1024**3,
            stdout_limit_bytes=1024 * 1024,
            stderr_limit_bytes=1024 * 1024,
            threads=self.recipe.threads,
            source_digest=self.recipe.source_artifact_sha256,
            config_digest=self.recipe.candidate_config_sha256,
            data_digest=inputs.digest,
            checkpoint_digest=self.recipe.tree_checkpoint_sha256,
            device="cpu",
            process_limit=64,
            python_hash_seed=0,
            extra_environment=(
                ("KUAIRAND_MODE", "predict"),
                ("KUAIRAND_SEED", "0"),
                ("KUAIRAND_SPLIT_ROLE", split_role.value),
            ),
        )
        result = self.runner.run(
            spec,
            commit_launch=lambda _record: None,
            cancel_event=self.cancel_event,
        )
        if result.outcome is ExecutionOutcome.CANCELLED:
            raise ReplayCancelledError("generated replay subprocess was cancelled")
        if not result.succeeded:
            raise ReplayError(
                f"generated replay subprocess failed with outcome {result.outcome.value}"
            )
        try:
            validated = validate_prediction_outputs(
                (action / "output").resolve(strict=True),
                PredictionExpectation(
                    source_digest=self.recipe.source_artifact_sha256,
                    config_digest=self.recipe.candidate_config_sha256,
                    data_digest=inputs.digest,
                    split_token=split_token,
                    checkpoint_digest=self.recipe.tree_checkpoint_sha256,
                    expected_count=inputs.row_count,
                ),
            )
        except Exception as exc:
            raise ReplayError("generated replay output failed the trusted protocol") from exc
        for path, digest, name in (
            (candidate_source, self.recipe.candidate_source_sha256, "generated candidate.py"),
            (candidate_config, self.recipe.candidate_config_sha256, "generated config.json"),
            (feature_path, feature_digest, "generated replay features"),
            (tree_checkpoint, self.recipe.tree_checkpoint_sha256, "generated tree checkpoint"),
        ):
            _regular_digest(path, digest, name)
        return validated.scores

    def _predict(
        self,
        *,
        workspace: FrozenCandidateWorkspace,
        inputs: CandidateInputs,
        phase: DataPhase,
        expected_inputs_digest: str,
    ) -> Float64Vector:
        _require_workspace(workspace)
        _require_recipe_artifact(workspace, self.recipe)
        _require_common_identity(
            workspace,
            inputs,
            source_artifact_sha256=self.recipe.source_artifact_sha256,
            feature_artifact_sha256=self.recipe.feature_artifact_sha256,
            checkpoint_artifact_sha256=self.recipe.checkpoint_artifact_sha256,
            data_sha256=self.recipe.data_sha256,
            expected_inputs_digest=expected_inputs_digest,
            phase=phase,
        )
        feature_path, tree_checkpoint, fm_encoding_path, fm_checkpoint_path, feature_digest = (
            self._paths(workspace, phase=phase)
        )
        tree_scores = self._tree_predict(
            workspace=workspace,
            inputs=inputs,
            phase=phase,
            feature_path=feature_path,
            feature_digest=feature_digest,
            tree_checkpoint=tree_checkpoint,
        )
        if self.recipe.fm_member is None:
            return tree_scores
        assert fm_encoding_path is not None and fm_checkpoint_path is not None
        member = self.recipe.fm_member
        _regular_digest(fm_encoding_path, member.encoding_sha256, "fusion FM encoding")
        _regular_digest(fm_checkpoint_path, member.checkpoint_sha256, "fusion FM checkpoint")
        encoding, checkpoint, config = _load_fm(
            encoding_path=fm_encoding_path,
            checkpoint_path=fm_checkpoint_path,
            member=member,
        )
        fm_scores = _fm_checkpoint_predictions(
            inputs=inputs,
            encoding=encoding,
            checkpoint=checkpoint,
            config=config,
        )
        assert self.recipe.fusion_weights is not None
        try:
            fused = fuse_ranked_predictions(
                inputs.column("user_id"),
                inputs.column("video_id"),
                tree_scores,
                fm_scores,
                weights=self.recipe.fusion_weights,
                phase=phase,
            )
        except Exception as exc:
            raise ReplayError("frozen label-free rank fusion failed") from exc
        return fused.scores

    def replay_validation(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Float64Vector:
        return self._predict(
            workspace=workspace,
            inputs=inputs,
            phase=DataPhase.OUTER_VALID,
            expected_inputs_digest=self.recipe.validation_inputs_digest,
        )

    def predict_final(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Float64Vector:
        return self._predict(
            workspace=workspace,
            inputs=inputs,
            phase=DataPhase.FINAL,
            expected_inputs_digest=self.recipe.final_inputs_digest,
        )


@dataclass(frozen=True, slots=True)
class GeneratedLambdaRankEnsembleReplayBackend:
    """Replay complete v1 members in order, then perform one frozen n-member fusion.

    The enclosing frozen bundle uses a fixed, non-configurable layout so a recipe cannot select
    paths or executable code::

        source/members/<position>/source, source/members/<position>/recipe.json
        features/members/<position>, checkpoint/members/<position>

    ``position`` is the zero-based tuple position from the recipe.  Each nested recipe is loaded
    again from its own SHA-bound artifact by the unchanged v1 backend.  Scores are retained only
    in memory and the method returns only after every member has replayed and the one fusion call
    has passed its validation bindings.
    """

    recipe: GeneratedLambdaRankEnsembleReplayRecipe
    interpreter: Path
    runner: Runner
    cancel_event: threading.Event | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, GeneratedLambdaRankEnsembleReplayRecipe):
            raise ReplayRecipeError("recipe must be GeneratedLambdaRankEnsembleReplayRecipe")
        if not isinstance(self.interpreter, Path) or not self.interpreter.is_absolute():
            raise ReplayRecipeError("generated replay interpreter must be an absolute path")
        if not isinstance(self.runner, Runner):
            raise ReplayRecipeError("generated replay runner must be Runner")
        if self.cancel_event is not None and not isinstance(self.cancel_event, threading.Event):
            raise ReplayRecipeError("cancel_event must be threading.Event or None")

    def _member_workspace(
        self,
        *,
        workspace: FrozenCandidateWorkspace,
        member: GeneratedLambdaRankReplayRecipe,
        position: int,
        validation_prediction_digest: str,
    ) -> FrozenCandidateWorkspace:
        # Only the fixed ordinal is used to resolve member files.  Digests are verified by the v1
        # backend after this construction; no recipe-controlled path reaches the filesystem.
        member_name = str(position)
        member_identity = FrozenReplayIdentity(
            source_sha256=member.source_artifact_sha256,
            config_sha256=member.digest,
            features_sha256=member.feature_artifact_sha256,
            checkpoint_sha256=member.checkpoint_artifact_sha256,
            validation_prediction_artifact_sha256=validation_prediction_digest,
            validation_prediction_digest=validation_prediction_digest,
            data_sha256=member.data_sha256,
            environment_sha256=workspace.identity.environment_sha256,
        )
        return FrozenCandidateWorkspace(
            root=workspace.root,
            source_dir=workspace.source_dir / "members" / member_name / "source",
            config_path=workspace.source_dir / "members" / member_name / "recipe.json",
            features_path=workspace.features_path / "members" / member_name,
            checkpoint_path=workspace.checkpoint_path / "members" / member_name,
            identity=member_identity,
        )

    def _predict(
        self,
        *,
        workspace: FrozenCandidateWorkspace,
        inputs: CandidateInputs,
        phase: DataPhase,
        expected_inputs_digest: str,
    ) -> Float64Vector:
        _require_workspace(workspace)
        _require_recipe_artifact(workspace, self.recipe)
        if not isinstance(inputs, CandidateInputs) or inputs.phase is not phase:
            raise ReplayError(f"replay backend requires a {phase.value} CandidateInputs capability")
        if inputs.digest != expected_inputs_digest:
            raise ReplayError("candidate input capability differs from the replay recipe")

        scores: list[Float64Vector] = []
        for position, (member, member_digest, validation_digest) in enumerate(
            zip(
                self.recipe.members,
                self.recipe.member_recipe_digests,
                self.recipe.validation_member_prediction_digests,
                strict=True,
            )
        ):
            # Constructor validation makes this redundant for normal callers, but it keeps the
            # fail-closed guarantee local to the executable boundary.
            if member.digest != member_digest:
                raise ReplayError("ensemble member recipe digest differs from ordered binding")
            member_backend = GeneratedLambdaRankReplayBackend(
                member,
                self.interpreter,
                self.runner,
                self.cancel_event,
            )
            member_scores = member_backend._predict(
                workspace=self._member_workspace(
                    workspace=workspace,
                    member=member,
                    position=position,
                    validation_prediction_digest=validation_digest,
                ),
                inputs=inputs,
                phase=phase,
                expected_inputs_digest=expected_inputs_digest,
            )
            scores.append(member_scores)

        try:
            fused = fuse_ranked_members(
                inputs.column("user_id"),
                inputs.column("video_id"),
                tuple(scores),
                weights=self.recipe.fusion_weights,
                phase=phase,
            )
        except Exception as exc:
            raise ReplayError("frozen ordered rank-ensemble fusion failed") from exc
        if phase is DataPhase.OUTER_VALID:
            if fused.member_prediction_digests != self.recipe.validation_member_prediction_digests:
                raise ReplayError("ensemble validation member predictions differ from recipe")
            if fused.fusion_digest != self.recipe.validation_fusion_digest:
                raise ReplayError("ensemble validation fusion digest differs from recipe")
        return fused.scores

    def replay_validation(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Float64Vector:
        return self._predict(
            workspace=workspace,
            inputs=inputs,
            phase=DataPhase.OUTER_VALID,
            expected_inputs_digest=self.recipe.validation_inputs_digest,
        )

    def predict_final(
        self, *, workspace: FrozenCandidateWorkspace, inputs: CandidateInputs
    ) -> Float64Vector:
        return self._predict(
            workspace=workspace,
            inputs=inputs,
            phase=DataPhase.FINAL,
            expected_inputs_digest=self.recipe.final_inputs_digest,
        )


def build_replay_backend(
    recipe: ReplayRecipe,
    *,
    interpreter: Path | None = None,
    cancel_event: threading.Event | None = None,
) -> ReplayBackend:
    """Map one parsed recipe to trusted code without dynamic imports or shell execution."""

    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise ReplayRecipeError("cancel_event must be threading.Event or None")
    if isinstance(recipe, OfficialFMReplayRecipe):
        return OfficialFMReplayBackend(recipe)
    if isinstance(recipe, GeneratedLambdaRankReplayRecipe):
        selected_interpreter = Path(sys.executable) if interpreter is None else interpreter
        return GeneratedLambdaRankReplayBackend(
            recipe,
            selected_interpreter,
            Runner(),
            cancel_event,
        )
    if isinstance(recipe, GeneratedLambdaRankEnsembleReplayRecipe):
        selected_interpreter = Path(sys.executable) if interpreter is None else interpreter
        return GeneratedLambdaRankEnsembleReplayBackend(
            recipe,
            selected_interpreter,
            Runner(),
            cancel_event,
        )
    raise ReplayRecipeError("replay backend is not allowlisted")


def load_replay_backend(
    recipe_path: str | Path,
    *,
    expected_sha256: str,
    interpreter: Path | None = None,
    cancel_event: threading.Event | None = None,
) -> ReplayBackend:
    """Load one SHA-bound recipe artifact and instantiate only its trusted backend.

    This is the schema-independent CLI/campaign seam: callers remain responsible for restoring
    ``ReplayArtifacts`` and label-free ``ReplayCapabilities``, but never choose an import path,
    entry point, shell command, or executable implementation from bundle-controlled text.
    """

    recipe = load_replay_recipe(recipe_path, expected_sha256=expected_sha256)
    return build_replay_backend(
        recipe,
        interpreter=interpreter,
        cancel_event=cancel_event,
    )


__all__ = [
    "GeneratedLambdaRankEnsembleReplayBackend",
    "GeneratedLambdaRankReplayBackend",
    "OfficialFMReplayBackend",
    "build_replay_backend",
    "load_replay_backend",
]
