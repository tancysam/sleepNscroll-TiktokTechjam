"""Extend the official FM seed pool and save each seed's public-validation predictions.

Only five qualified seeds exist (`runs/maki-qualification/fm/seed-{0..4}`), and the five-seed
within-user rank ensemble is what we currently ship at 0.6026034.  `StarterFMConfig` freezes k,
lr, l2 and batch_size with `field(init=False)`, but `seed` is settable: it is the one published
degree of freedom.  This trains additional seeds through the hash-pinned organizer source itself
rather than any reimplementation, replicating `baseline.run_fm` exactly (same Adam, same batch
size, same per-epoch reseeded permutation, same early stopping on validation primary with
patience 4) while additionally saving the prediction vector `run_fm` throws away.

Before training anything it reproduces seed 0 and checks it against the qualified vector already
on disk.  If that does not match bit for bit the loop is not equivalent and nothing else here can
be trusted.

Offline and read-only with respect to the repository: writes only into its output directory.  Lives
at the repository root, outside the slice `hash_source_tree` covers, so it cannot strand a running
campaign.

Run:  python3 -B seed_pool_probe.py --first 5 --last 19
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

STARTER = Path("kuairand-starter-kit").resolve()
QUALIFIED = Path("runs/maki-qualification/fm")
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
IMPROVEMENT = 1e-5


def _starter():
    """Import the pinned organizer modules without shadowing repository packages."""

    sys.path.insert(0, str(STARTER))
    from baseline import FM  # noqa: PLC0415
    from data import load  # noqa: PLC0415
    from data import encode as encode_splits  # noqa: PLC0415
    from evaluate import evaluate  # noqa: PLC0415

    return FM, load, encode_splits, evaluate


def train_seed(FM, evaluate, enc, dim: int, seed: int) -> tuple[np.ndarray, float, int]:
    """Replicate baseline.run_fm for one seed and return its validation predictions."""

    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    model = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        index = rng.permutation(len(ytr))
        for start in range(0, len(index), BATCH):
            chunk = index[start : start + BATCH]
            model.step(Xtr[chunk], ytr[chunk])
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        if primary > best + IMPROVEMENT:
            best, bad, best_epoch = primary, 0, epoch
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    assert best_state is not None
    model.V, model.W, model.b = best_state
    return np.asarray(model.predict(Xva), dtype=np.float64), best, best_epoch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".data/KuaiRand-Pure/data")
    parser.add_argument("--out", default="runs/seed-pool")
    parser.add_argument("--first", type=int, default=5)
    parser.add_argument("--last", type=int, default=19)
    args = parser.parse_args()

    FM, load, encode_splits, evaluate = _starter()
    print(f"loading {args.data_dir} ...", flush=True)
    splits = load(args.data_dir)
    enc, dim = encode_splits(splits)
    print(f"encoded: dim={dim} train={len(enc['train'][1])} valid={len(enc['valid'][1])}\n")

    # Positive control on our own loop before trusting any new seed.
    reference = QUALIFIED / "seed-0" / "validation-predictions.npy"
    started = time.perf_counter()
    scores, primary, epoch = train_seed(FM, evaluate, enc, dim, 0)
    elapsed = time.perf_counter() - started
    known = np.load(reference)
    identical = bool(np.array_equal(scores.astype(known.dtype), known))
    close = bool(np.allclose(scores, known, rtol=0, atol=1e-6))
    print(f"seed 0 replication: primary={primary:.10f} best_epoch={epoch} {elapsed:.1f}s")
    print(f"  bit identical to the qualified vector: {identical}")
    print(f"  within 1e-6 of it                    : {close}")
    if not (identical or close):
        print("\n  MISMATCH. This loop is not equivalent to the qualified path; stopping.")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(args.first, args.last + 1):
        started = time.perf_counter()
        scores, primary, epoch = train_seed(FM, evaluate, enc, dim, seed)
        np.save(out / f"seed-{seed}.npy", scores)
        print(
            f"seed {seed:2d}: primary={primary:.10f} best_epoch={epoch:2d} "
            f"{time.perf_counter() - started:.1f}s -> {out / f'seed-{seed}.npy'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
