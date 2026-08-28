# Compact neural interaction primitives

Status: **building blocks qualified; full-data branch not enabled**.

## Purpose

This family tests whether bounded nonlinear interactions add signal after the cheaper FM,
pairwise, causal-aggregate tree, and rank-fusion branches. It is not a hidden model zoo and is
not eligible to delay campaign finalization.

The executable mechanisms are:

- a plain same-feature control consisting of first-order terms plus a shallow MLP;
- compact DeepFM, which adds the parameter-free sum of pairwise embedding dot products to the
  identical control backbone;
- compact full-matrix DCNv2 cross layers using
  `x_(l+1) = x_0 * (W_l x_l + b_l) + x_l`, plus the identical control backbone;
- a frozen binary pointwise/pairwise hybrid objective.

DIN, shared-bottom multi-task learning, MMoE, and PLE are deliberately not implemented or
claimed here. They require separate coverage, leakage, same-backbone, and throughput evidence.

## Control and attribution

Control and DeepFM use exactly the same embeddings, first-order terms, dense projection, MLP,
and trainable parameter count. DeepFM's only difference is its parameter-free interaction
equation. DCNv2 retains the same backbone and adds only explicit cross matrices, biases, and a
cross head. Result evidence records both total and shared-backbone parameter counts so a claimed
mechanism cannot be confused with unbounded capacity growth.

## Data and scoring contract

- `NeuralTrainingTargets` validates the phase before converting or inspecting target content;
  only `train` and `inner_train` are accepted.
- `NeuralFeatureBatch` contains categorical and dense inference inputs only. It is valid for the
  label-free final path and contains no outcome field.
- Fitting requires train-derived features and targets with the same phase and row count.
- Pair indices must refer to observed positive and negative training targets and are bounded;
  this module never enumerates a Cartesian pair space.
- Optional model screening accepts only `inner_valid` features and a bound aggregate scorer.
  The callback must return the protected `ScoreResult`; no validation target vector enters the
  candidate API.
- Public checkpoint inference accepts only `inner_valid`, `outer_valid`, or `final` feature
  batches and has no label argument.

All feature, target, pair, configuration, state, checkpoint, and prediction identities are
SHA-256 bound. Serialized state is rehashed after safe loading before replay.

## Determinism and devices

CPU is mandatory and remains the canonical device for qualification, campaign execution, and
final replay. The runtime records seed, PyTorch version, thread count, requested and actual
device, MPS availability, and deterministic-algorithm status. The focused replay gate fits twice
and requires an identical state digest, semantic checkpoint digest, and prediction array.

MPS is opt-in only. Availability alone is insufficient: a saved CPU checkpoint must replay on
MPS within the declared absolute prediction tolerance, with the actual device recorded. On the
current locked local host, the out-of-sandbox MPS checkpoint-prediction parity gate passed: one
test passed in 1.17 seconds. This focused result proves tolerance-bounded replay parity on that
host; it does not establish an acceleration claim or make MPS a canonical execution path.

## Eligibility and automatic closure

Every evidence fit must receive predeclared limits for:

- maximum trainable parameters;
- maximum deterministic nearest-rank p95 epoch time;
- minimum observed examples per second;
- maximum serialized checkpoint bytes.

All four gates are strict and produce stable failure reasons. Passing them means only that a run
is technically eligible for scientific comparison; it does not authorize promotion. The
campaign must close the family when any ceiling fails, predicted completion threatens the
finalization reserve, or the candidate fails to beat the same-feature control on both
train-derived inner folds. Outer validation remains subject to the project-wide promotion
ledger and can never be used for iterative neural tuning.

## Verification

Run the locked focused gates with:

```shell
UV_CACHE_DIR=.uv-cache \
  uv run --locked --group research-neural pytest -q \
  tests/unit/test_neural_primitives.py \
  tests/performance/test_neural_throughput.py
```

The unit suite covers independent worked DeepFM/DCNv2 equations, configuration bounds,
same-backbone shapes and parameter counts, binary/mask/nonfinite failures, train-only target
authorization, aggregate-scorer routing, corrupted-pair rejection, exact CPU checkpoint replay,
label-free final inference, and conditional MPS parity. The performance test executes one tiny
DCNv2 epoch and requires the measured result to pass its declared CPU family ceilings.
