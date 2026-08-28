from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kuairand_agent.baselines.artifacts import (
    load_checkpoint,
    load_predictions,
    save_checkpoint,
    save_predictions,
)
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.baselines.starter_fm import StarterFMAdapter, StarterFMConfig
from kuairand_agent.data.canonical import CanonicalInputs
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

ROOT = Path(__file__).parents[2]
STARTER = ROOT / "kuairand-starter-kit"


def _inputs(prefix: str, rows: int, *, start_time: int = 0) -> CanonicalInputs:
    return CanonicalInputs(
        user_id=tuple(f"u{index % 5}" for index in range(rows)),
        video_id=tuple(f"{prefix}-v{index % 7}" for index in range(rows)),
        date=tuple(20220408 for _ in range(rows)),
        duration_ms=tuple(float(800 + index * 101) for index in range(rows)),
        tab=tuple(str(index % 3) for index in range(rows)),
        author_id=tuple(f"a{index % 4}" for index in range(rows)),
        time_ms=tuple(start_time + index for index in range(rows)),
    )


@dataclass(frozen=True)
class _Targets:
    primary: npt.NDArray[np.int8]
    training_inputs_digest: str
    digest: str = "7" * 64

    @property
    def row_count(self) -> int:
        return int(self.primary.size)


@dataclass(frozen=True)
class _BoundScorer:
    validation_inputs_digest: str
    callback: Callable[[npt.NDArray[np.float64]], object]

    def __call__(self, scores: npt.NDArray[np.float64]) -> object:
        return self.callback(scores)


def test_saved_checkpoint_and_encoding_replay_exact_same_host_predictions(
    tmp_path: Path,
) -> None:
    train = _inputs("train", 24)
    valid = _inputs("valid", 9, start_time=100)
    final = _inputs("final", 11, start_time=200)
    targets = _Targets(
        np.asarray([int(index % 3 == 0) for index in range(len(train))], dtype=np.int8),
        training_inputs_digest=train.digest,
    )
    encoding = StarterEncoding.fit(train)
    config = StarterFMConfig(seed=2)
    adapter = StarterFMAdapter(starter_dir=STARTER, config=config)

    def constant_scorer(scores: npt.NDArray[np.float64]) -> dict[str, float]:
        assert scores.ndim == 1
        return {"GAUC": 0.6, "nDCG@5": 0.4, "primary": 0.5}

    first = adapter.fit(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=valid,
        validation_scorer=_BoundScorer(valid.digest, constant_scorer),
    )
    clean_retrain = adapter.fit(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=valid,
        validation_scorer=_BoundScorer(valid.digest, constant_scorer),
    )
    encoding_artifact = encoding.save(tmp_path / "encoding.npz")
    checkpoint_artifact = save_checkpoint(tmp_path / "checkpoint.npz", first.checkpoint)
    prediction_artifact = save_predictions(
        tmp_path / "validation_predictions.npy", first.validation_predictions
    )

    restored_encoding = StarterEncoding.load(encoding_artifact.path)
    restored_checkpoint = load_checkpoint(
        checkpoint_artifact.path,
        expected_file_sha256=checkpoint_artifact.file_sha256,
        expected_checkpoint_digest=checkpoint_artifact.checkpoint_digest,
        expected_encoding_digest=restored_encoding.digest,
        expected_starter_manifest_digest=adapter.starter_manifest_digest,
        expected_config_digest=config.digest,
        expected_seed=config.seed,
    )
    restored_predictions = load_predictions(
        prediction_artifact.path,
        expected_file_sha256=prediction_artifact.file_sha256,
        expected_prediction_digest=prediction_artifact.prediction_digest,
        expected_row_count=len(valid),
    )
    replay = adapter.predict(
        checkpoint=restored_checkpoint,
        encoding=restored_encoding,
        inputs=valid,
        expected_prediction_digest=first.prediction_digest,
    )
    final_predictions = adapter.predict(
        checkpoint=restored_checkpoint,
        encoding=restored_encoding,
        inputs=final,
    )

    assert first.logical_digest == clean_retrain.logical_digest
    assert first.checkpoint.digest == clean_retrain.checkpoint.digest
    assert first.prediction_digest == clean_retrain.prediction_digest
    assert replay.scores.tobytes() == first.validation_predictions.scores.tobytes()
    assert restored_predictions.scores.tobytes() == replay.scores.tobytes()
    assert final_predictions.row_count == len(final)
    assert np.isfinite(final_predictions.scores).all()


def test_adapter_metrics_match_untouched_run_fm_on_synthetic_fixture() -> None:
    train = _inputs("train", 12)
    valid = _inputs("valid", 6, start_time=100)
    train_labels = tuple(int(index % 3 == 0) for index in range(len(train)))
    valid_labels = (1, 0, 1, 0, 0, 1)
    targets = _Targets(
        np.asarray(train_labels, dtype=np.int8),
        training_inputs_digest=train.digest,
    )
    encoding = StarterEncoding.fit(train)
    split = SplitIdentity(
        name="inner_valid",
        token="fm-organizer-fixture",
        expected_count=len(valid),
    )
    alignment = Alignment.from_ids(
        split=split,
        user_ids=valid.user_id,
        video_ids=valid.video_id,
    )
    scorer = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)

    def score(scores: npt.NDArray[np.float64]) -> object:
        return scorer.score_with_encoded_labels(
            alignment=alignment,
            split=split,
            labels=valid_labels,
            scores=scores,
        )

    adapted = StarterFMAdapter(
        starter_dir=STARTER,
        config=StarterFMConfig(seed=0),
    ).fit(
        encoding=encoding,
        train_inputs=train,
        train_targets=targets,
        validation_inputs=valid,
        validation_scorer=_BoundScorer(valid.digest, score),
    )

    def raw_rows(inputs: CanonicalInputs, labels: tuple[int, ...]) -> list[list[object]]:
        return [
            [
                inputs.date[index],
                inputs.user_id[index],
                inputs.video_id[index],
                inputs.author_id[index],
                inputs.tab[index],
                inputs.duration_ms[index],
                labels[index],
            ]
            for index in range(len(inputs))
        ]

    fixture = {
        "train": raw_rows(train, train_labels),
        "valid": raw_rows(valid, valid_labels),
        # Untouched run_fm always evaluates test.  This harmless synthetic row
        # is nonempty solely to exercise that immutable fixture path.
        "test": [[20220429, "placeholder-user", "placeholder-video", "UNK", "0", 1000.0, 0]],
    }
    expected_baseline_path = json.dumps(str(STARTER / "baseline.py"))
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "import baseline\n"
        f"assert Path(baseline.__file__).resolve() == Path({expected_baseline_path})\n"
        f"splits = json.loads({json.dumps(json.dumps(fixture))})\n"
        "result = baseline.run_fm(splits, seed=0, verbose=False)['valid']\n"
        "print(json.dumps({name: float(result[name]) for name in "
        "('GAUC', 'nDCG@5', 'primary')}, sort_keys=True))\n"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(STARTER)
    environment["UV_CACHE_DIR"] = str(ROOT / ".uv-cache")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=STARTER,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    direct = json.loads(completed.stdout)

    assert adapted.validation_metrics.gauc == direct["GAUC"]
    assert adapted.validation_metrics.ndcg_at_5 == direct["nDCG@5"]
    assert adapted.validation_metrics.primary == direct["primary"]
