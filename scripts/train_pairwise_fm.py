"""Train the pairwise FM on full data and score it against the organizer evaluator.

The organizers rank "change the loss function" as the most promising untried direction: training
is pointwise log loss while GAUC and nDCG@5 are ranking metrics.  This repository already
implements that direction -- ``candidates/pairwise_fm.py`` shares the official FM's five encoded
fields, 16-factor float32 state, seeded initializer and Adam optimizer, and replaces only the
objective with a GAUC-matched pairwise sampler and logistic loss.

It has never been trained.  The single existing measurement is a CI acceptance gate deliberately
sized for speed -- one epoch, one optimizer step, 8,192 pairs against 1,141,112 training rows --
which scores validation primary 0.5537.  That number says the configuration is undertrained; it
says nothing about whether the objective works.

This script answers the actual question.  It sweeps learning rate and epoch count at full pair
volume and reports the organizer's own metrics for each cell.

Two things it is careful about:

* The learning rate is swept, not assumed.  ``PairwiseFMConfig`` defaults mirror the *pointwise*
  FM (``learning_rate=0.001``), but this repository's separate pairwise reimplementation for
  generated candidates uses ``0.03``.  A pairwise loss over score differences has a different
  gradient scale than a pointwise loss over absolute scores, so inheriting the pointwise step size
  is a plausible reason the model barely moves.  Reporting a single low-rate run as evidence that
  the direction fails would be wrong.
* The encoding is proven identical to the baseline's before anything is claimed.  Without that,
  the comparison against 0.6016 is not apples to apples.

This is a manual experiment.  ``pairwise_fm.py`` is imported by nothing in ``src/``, so no
campaign has ever run it and no result here is an autonomous agent result.

Usage:

    uv run --locked python scripts/train_pairwise_fm.py --data-dir .data/KuaiRand-Pure/data \\
        --learning-rates 0.01 --epochs 2 --out runs/pairwise-stage1.json

    uv run --locked python scripts/train_pairwise_fm.py --data-dir .data/KuaiRand-Pure/data \\
        --learning-rates 0.001 0.01 0.03 --epochs 5 10 20 40 --out runs/pairwise-sweep.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import kuairand_agent.candidates.pairwise_fm as pairwise_fm_module
from kuairand_agent.baselines.encoding import StarterEncoding
from kuairand_agent.candidates.pairwise_fm import (
    EncodedFMInputs,
    PairwiseFMAdapter,
    PairwiseFMConfig,
    PairwiseFMTrainingData,
)
from kuairand_agent.data.canonical import ProtectedTargets, TrainingTargets, load_canonical_dataset
from kuairand_agent.data.capabilities import DataPhase
from kuairand_agent.scoring.protected import Alignment, ProtectedScorer, SplitIdentity

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "kuairand-starter-kit"

# From runs/maki-qualification/manifest.json. The official FM that scores validation primary
# 0.6016 was trained through an encoding with exactly this digest, so matching it is what makes
# any number produced here comparable rather than merely plausible.
BASELINE_ENCODING_DIGEST = "c78447dd67feafe31a273146d8d5e6b4774f80a7ce157d3dc208bde151d8a542"

# Local five-seed mean validation primary for the official FM on this machine
# (runs/maki-qualification). Deliberately not Samuel's arm64 figure: float32 arithmetic differs
# between platforms, so a candidate must be judged against a baseline measured on the same host.
LOCAL_BASELINE_PRIMARY = 0.6015722

# tests/full_data/test_pairwise_fm_acceptance.py:29 -- users with both a positive and a negative.
EXPECTED_ELIGIBLE_TRAIN_ROWS = 1_130_240


@dataclass(frozen=True, slots=True)
class CellResult:
    """One (learning rate, epochs) cell of the sweep."""

    learning_rate: float
    max_epochs: int
    gauc: float
    ndcg_at_5: float
    primary: float
    first_epoch_loss: float
    final_epoch_loss: float
    fit_seconds: float
    sampled_pairs: int

    @property
    def delta(self) -> float:
        return self.primary - LOCAL_BASELINE_PRIMARY

    @property
    def learned(self) -> bool:
        """A flat loss trace means the step size is wrong, not that the objective failed."""

        return self.final_epoch_loss < self.first_epoch_loss

    def to_wire(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
            "primary_delta_vs_local_baseline": self.delta,
            "first_epoch_mean_pairwise_loss": self.first_epoch_loss,
            "final_epoch_mean_pairwise_loss": self.final_epoch_loss,
            "loss_decreased": self.learned,
            "fit_seconds": self.fit_seconds,
            "sampled_pairs": self.sampled_pairs,
        }


def _source_digest() -> str:
    source_path = Path(pairwise_fm_module.__file__).resolve(strict=True)
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON evidence destination")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.01, 0.03])
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--pairs-per-epoch", type=int, default=1_000_000)
    parser.add_argument("--pair-batch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-encoding-mismatch",
        action="store_true",
        help="proceed even if the encoding digest differs from the qualified baseline",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)

    print(f"loading canonical dataset from {arguments.data_dir} ...", flush=True)
    started = time.perf_counter()
    dataset = load_canonical_dataset(arguments.data_dir)
    train, valid = dataset.train, dataset.valid
    if not isinstance(train.targets, TrainingTargets):
        raise SystemExit("training split does not expose training targets")
    if not isinstance(valid.targets, ProtectedTargets):
        raise SystemExit("validation targets are not protected")
    print(f"  train {train.row_count:,}  valid {valid.row_count:,}", flush=True)

    # The expensive part, and the reason the sweep shares one setup: StarterEncoding.fit and
    # transform are per-row Python loops over 1.14M and 124,909 rows.
    print("building the five-field starter encoding (shared by every cell) ...", flush=True)
    encoding = StarterEncoding.fit(train.inputs)
    encoded_train = EncodedFMInputs.from_encoding(encoding, train.inputs, phase=DataPhase.TRAIN)
    encoded_valid = EncodedFMInputs.from_encoding(
        encoding, valid.inputs, phase=DataPhase.OUTER_VALID
    )
    setup_seconds = time.perf_counter() - started

    observed_digest = encoded_train.encoding_digest
    parity = observed_digest == BASELINE_ENCODING_DIGEST
    print(f"  encoding digest {observed_digest}", flush=True)
    print(f"  matches qualified baseline: {parity}", flush=True)
    if not parity and not arguments.allow_encoding_mismatch:
        raise SystemExit(
            "encoding digest differs from the qualified official-FM baseline, so any score here "
            "would not be comparable to 0.6016; rerun with --allow-encoding-mismatch only if you "
            "intend to report an incomparable number"
        )
    print(f"  total dim {encoded_train.total_dim:,}   setup {setup_seconds:.1f}s", flush=True)

    training = PairwiseFMTrainingData(
        inputs=encoded_train,
        labels=np.asarray(train.targets.long_view, dtype=np.int8),
        user_ids=train.inputs.user_id,
        training_targets_digest=train.targets.digest,
        target_inputs_digest=train.inputs.digest,
    )

    split = SplitIdentity(
        name="outer_valid",
        token="pairwise-fm-full-training-sweep-v1",
        expected_count=valid.row_count,
    )
    alignment = Alignment.from_ids(
        split=split,
        row_ids=valid.alignment.row_id,
        user_ids=valid.alignment.user_id,
        video_ids=valid.alignment.video_id,
    )
    scorer = ProtectedScorer(starter_dir=STARTER, trusted_alignment=alignment)
    labels = valid.targets.reveal_for_scorer()
    source_digest = _source_digest()

    results: list[CellResult] = []
    eligible_rows = 0
    grid = [(rate, epochs) for rate in arguments.learning_rates for epochs in arguments.epochs]
    print(
        f"\nsweeping {len(grid)} cells at {arguments.pairs_per_epoch:,} pairs/epoch\n",
        flush=True,
    )

    for index, (rate, epochs) in enumerate(grid, start=1):
        config = PairwiseFMConfig(
            seed=arguments.seed,
            learning_rate=rate,
            pair_batch_size=arguments.pair_batch_size,
            pairs_per_epoch=arguments.pairs_per_epoch,
            max_epochs=epochs,
            device_metadata="pairwise-fm-full-training-sweep-cpu",
        )
        adapter = PairwiseFMAdapter(source_digest=source_digest, config=config)
        fit_started = time.perf_counter()
        run = adapter.fit(training)
        fit_seconds = time.perf_counter() - fit_started

        eligible_rows = run.stored_row_index_count
        if index == 1 and run.stored_row_index_count != EXPECTED_ELIGIBLE_TRAIN_ROWS:
            raise SystemExit(
                f"sampler retained {run.stored_row_index_count} eligible rows, expected "
                f"{EXPECTED_ELIGIBLE_TRAIN_ROWS}; the pair space is not the one the acceptance "
                "gate measured"
            )

        prediction = adapter.predict(run.checkpoint, encoded_valid)
        score = scorer.score_with_encoded_labels(
            alignment=alignment, split=split, labels=labels, scores=prediction.scores
        )
        result = CellResult(
            learning_rate=rate,
            max_epochs=epochs,
            gauc=score.gauc,
            ndcg_at_5=score.ndcg_at_5,
            primary=score.primary,
            first_epoch_loss=run.trace[0].mean_pairwise_loss,
            final_epoch_loss=run.trace[-1].mean_pairwise_loss,
            fit_seconds=fit_seconds,
            sampled_pairs=run.checkpoint.sampled_pairs,
        )
        results.append(result)
        flag = "" if result.learned else "   [loss did not fall]"
        print(
            f"  lr {rate:<7} epochs {epochs:<3} primary {result.primary:.7f} "
            f"delta {result.delta:+.7f}  loss {result.first_epoch_loss:.5f} -> "
            f"{result.final_epoch_loss:.5f}  {fit_seconds:.0f}s{flag}",
            flush=True,
        )

    best = max(results, key=lambda item: item.primary)
    payload = {
        "schema_version": 1,
        "experiment": "pairwise_fm_full_training_sweep",
        "is_autonomous_agent_result": False,
        "note": (
            "Manual experiment. candidates/pairwise_fm.py is not wired into the campaign, so this "
            "measures the direction rather than demonstrating agent behaviour."
        ),
        "data_dir": str(arguments.data_dir),
        "encoding_digest": observed_digest,
        "encoding_matches_qualified_baseline": parity,
        "source_digest": source_digest,
        "total_dim": int(encoded_train.total_dim),
        "train_rows": int(train.row_count),
        "validation_rows": int(valid.row_count),
        "eligible_train_rows": eligible_rows,
        "pairs_per_epoch": arguments.pairs_per_epoch,
        "pair_batch_size": arguments.pair_batch_size,
        "seed": arguments.seed,
        "setup_seconds": setup_seconds,
        "local_baseline_primary": LOCAL_BASELINE_PRIMARY,
        "cells": [item.to_wire() for item in results],
        "best": best.to_wire(),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"\nbest: lr {best.learning_rate} epochs {best.max_epochs} -> primary "
        f"{best.primary:.7f} (GAUC {best.gauc:.7f}, nDCG@5 {best.ndcg_at_5:.7f})"
    )
    print(f"official FM on this machine: {LOCAL_BASELINE_PRIMARY:.7f}")
    print(f"delta: {best.delta:+.7f}")
    if best.delta > 0.002:
        print("clears the organizer convergence threshold of 0.002")
    elif best.delta > 0:
        print("above baseline but inside the 0.002 threshold; not a material improvement")
    else:
        print("does not beat the baseline")
    print(f"\nevidence written to {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
