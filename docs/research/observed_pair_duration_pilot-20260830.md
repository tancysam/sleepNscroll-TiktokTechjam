# Duration-conditioned observed-pair FM pilot

This is a train-derived pilot, not a promotion or protected-score result.  It compares the
existing GAUC-aligned logged-pair sampler with an equal-budget 50/50 intervention that samples
half of its pairs from the same user and the same frozen duration bucket.  Both arms use the same
five-field FM, factor dimension, optimizer, initialization seed, number of epochs, and total pair
budget.  Only the pair distribution differs.

Command:

```text
UV_CACHE_DIR=/tmp/tiktok-uv-cache uv run python docs/research/evaluate_observed_pair_duration_pilot.py
```

The run used the script's predeclared pilot budget of 8,192 pairs/epoch, two epochs, and seed
20,260,830.  Fold A and Fold B are rolling-origin windows carved from the official training
period.  Metrics were computed by the trusted local fold scorer over those train-derived query
windows only.  No public-validation or final-period outcome was loaded or scored.

| Fold | Arm | GAUC | nDCG@5 | Primary | Train seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Uniform control | 0.6065285802 | 0.5356289148 | 0.5710787773 | 0.1604 |
| A | Duration-conditioned | 0.6099796295 | 0.5359517336 | 0.5729656816 | 0.2890 |
| B | Uniform control | 0.6044757962 | 0.4785201848 | 0.5414980054 | 0.1787 |
| B | Duration-conditioned | 0.6081938148 | 0.4790934920 | 0.5436436534 | 0.3293 |

Treatment-minus-control deltas:

| Fold | ΔGAUC | ΔnDCG@5 | ΔPrimary |
| --- | ---: | ---: | ---: |
| A | +0.0034510493 | +0.0003228188 | +0.0018869042 |
| B | +0.0037180185 | +0.0005733073 | +0.0021456480 |

The treatment was byte-reproducible on a second invocation: checkpoint diagnostics, scores, and
deltas were identical; wall time varied with cache/process state.  The treatment used 8,192
intervention pairs in total (4,096 per epoch), 61,033/66,651 eligible user-duration groups in
Fold A/B, and 16,384 total sampled pairs per arm.

Interpretation: this one-seed, deliberately low-budget pilot is positive on both frozen folds,
with the larger movement in GAUC and a smaller but same-direction nDCG@5 movement.  It is not
enough to claim a reliable improvement: the default protected budget is 250,000 pairs × 5
epochs, only one seed was tested, no uncertainty interval was estimated, and Fold A/B are not a
pristine hidden holdout.  The next safe gate is a preregistered multi-seed train-only replicate
at the full equal budget, followed by the existing fold/component/uncertainty policy before any
protected query is considered.

## Full-budget three-seed replicate

The full-budget command was then run with 250,000 pairs/epoch × 5 epochs for each arm and seeds
0, 1, and 2:

```text
UV_CACHE_DIR=/tmp/tiktok-uv-cache uv run python \
  docs/research/evaluate_observed_pair_duration_pilot.py --full-replicate
```

Per-seed/per-fold results:

| Fold | Seed | Uniform primary | Duration primary | ΔGAUC | ΔnDCG@5 | ΔPrimary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0 | 0.6074241400 | 0.6072412729 | -0.0005676746 | +0.0002020001 | -0.0001828671 |
| A | 1 | 0.6066162586 | 0.6066588163 | +0.0001569986 | -0.0000718832 | +0.0000425577 |
| A | 2 | 0.6061363816 | 0.6074610353 | +0.0010746121 | +0.0015746951 | +0.0013246536 |
| B | 0 | 0.5739845634 | 0.5740506053 | +0.0006499290 | -0.0005178154 | +0.0000660419 |
| B | 1 | 0.5742572546 | 0.5757734776 | +0.0023328662 | +0.0006995797 | +0.0015162230 |
| B | 2 | 0.5736982226 | 0.5753541589 | +0.0027526617 | +0.0005591810 | +0.0016559362 |

Across all six fold/seed cells, the mean treatment-minus-control deltas were:

| Mean ΔGAUC | Mean ΔnDCG@5 | Mean ΔPrimary | Worst ΔGAUC | Worst ΔnDCG@5 | Worst ΔPrimary |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +0.0010665655 | +0.0004076262 | +0.0007370909 | -0.0005676746 | -0.0005178154 | -0.0001828671 |

All twelve trainings completed at the declared budget.  The six duration arms used 625,000
intervention pairs each, while every control/treatment pair used 1,250,000 sampled pairs and 155
optimizer steps.  Model training took approximately 2.16–2.91 seconds per arm; feature/data
preparation dominated total wall time (306.9 seconds for the run).

This is positive average evidence but does not pass a conservative promotion gate: one seed/fold
cell is negative on primary, and the worst-fold primary delta is negative.  These results also
compare the new arm with the uniform-pair FM, not with the already-qualified official-FM control
or its within-user rank-fusion portfolio.  An exact official-FM-relative fusion comparison remains
the unresolved next gate; no public/final score claim follows from this replicate.

## Exact official-FM-relative three-seed comparison

The unresolved comparison was run with the same full-budget arms and seeds.  Official-FM
predictions were produced by the repository-authoritative ``run_fold_fm_control`` helper using
``StarterEncoding`` and the immutable organizer NumPy FM semantics.  Every official prediction
was replayed through the helper and required to be byte-identical before it was used.  The
duration/FM fusion used controller-owned within-user rank normalization.  For each candidate
grid point, the script first fused duration and official FM within each seed, then equal-weighted
the three resulting seed vectors.  Exactly one weight was selected on aggregate Fold B and frozen
unchanged for Fold A.

Command:

```text
UV_CACHE_DIR=/tmp/tiktok-uv-cache uv run python \
  docs/research/evaluate_observed_pair_duration_pilot.py \
  --full-replicate --official-relative --summary-only
```

Direct duration-vs-official-FM results:

| Fold | Seed | Official FM primary | Duration primary | ΔGAUC | ΔnDCG@5 | ΔPrimary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | 0 | 0.5754240155 | 0.5740506053 | -0.0024765730 | -0.0002702773 | -0.0013734102 |
| B | 1 | 0.5741409063 | 0.5757734776 | +0.0022191405 | +0.0010460615 | +0.0016325712 |
| B | 2 | 0.5752792358 | 0.5753541589 | -0.0002925992 | +0.0004424453 | +0.0000749230 |
| A | 0 | 0.6071290374 | 0.6072412729 | -0.0000312924 | +0.0002558231 | +0.0001122355 |
| A | 1 | 0.6077449322 | 0.6066588163 | -0.0013915300 | -0.0007807612 | -0.0010861158 |
| A | 2 | 0.6067887545 | 0.6074610353 | -0.0000863671 | +0.0014309287 | +0.0006722808 |

Fold-B selection chose the single shared weight ``duration=0.50, official-FM=0.50`` from the
predeclared 0.05 grid.  It was not selected separately per seed and it was not retuned on Fold A.
The deployable three-seed result was:

| Fold | Duration-only equal-seed primary | Official-FM-only equal-seed primary | Frozen duration/FM primary | Frozen ΔGAUC | Frozen ΔnDCG@5 | Frozen ΔPrimary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B (selection) | 0.5757069588 | 0.5752887726 | 0.5771089196 | +0.0028111339 | +0.0008292198 | +0.0018201470 |
| A (frozen confirmation) | 0.6074685454 | 0.6083299518 | 0.6090084314 | +0.0009556413 | +0.0004012585 | +0.0006784797 |

The three duration members and three official members had distinct prediction digests on both
folds.  The selected Fold-B fusion digest was
``faf19147f9fa0bdd3669b2529d7ef088f8507c0c92a1b52673dcd86feddc3d55`` and the frozen Fold-A
fusion digest was
``d981c7ae2e03a7890c65a8e934c590a56b0f38df74685c5601cbf852376a9739``.  Official-FM replay
parity was true for all six fold/seed controls.  The full run took approximately 403 seconds,
including data/feature preparation and six official-FM control trainings.

This is the first evidence that the new duration signal can improve the already-qualified
official-FM control after deployment-mirroring fusion: the frozen aggregate primary delta is
positive on both Fold B and the unchanged Fold A confirmation, with positive movement in both
GAUC and nDCG@5.  It is still not a protected-score claim.  Fold B is a tuning fold, Fold A has
been used repeatedly in this research history, no user-cluster bootstrap interval was computed,
and the result needs an integration/replay test in the production candidate path before any
protected query is considered.
