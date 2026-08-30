# Score-improvement implementation and evidence addendum

**Date:** 2026-08-30

**Benchmark:** KuaiRand-Pure logged-impression ranking for native `long_view`

**Primary:** arithmetic mean of organizer GAUC and organizer nDCG@5

**Evidence boundary:** official training-period folds only in the new experiments below. No
public-validation, final-period outcome, or protected final score was loaded or scored.

## Outcome

The repository is now stricter about what can replace the submission artifact and less likely to
stop a productive research lineage prematurely. It also contains two genuinely new experimental
surfaces: schema-v8 input-only strict-past exposure context and an equal-budget
duration-conditioned logged-pair objective. The duration mechanism has positive average
train-derived evidence, but it does not satisfy the frozen robustness gate and is therefore not a
claimed score improvement.

The previous retained generated challenger is no longer sufficient for deployment. Its matched
three-seed primary gain was approximately `+0.00004225`, one seed regressed, the paired user
bootstrap interval crossed zero, and the actual representative seed-0 artifact scored below the
immutable seed-4 FM fallback. The selector and finalization handoff now prevent evidence of that
shape from replacing the deployable fallback.

## Score-moving experiment

The implemented duration intervention changes exactly half of the existing logged same-user
positive/negative training pairs to pairs that also share a frozen duration bucket. The uniform
arm delegates to the existing reference sampler and has byte-exact model-state parity at an equal
seed and budget. Architecture, factor dimension, initialization, optimizer, batch size, total pair
count, epochs, and inference are otherwise unchanged.

The frozen duration buckets are `[0,5)`, `[5,10)`, `[10,18)`, `[18,30)`, `[30,60)`, and
`[60,+inf)` seconds. The treatment is train-only; prediction accepts neither labels nor user
groups.

The bounded pilot was positive on both rolling folds:

| Fold | Delta GAUC | Delta nDCG@5 | Delta primary |
| --- | ---: | ---: | ---: |
| A | +0.0034510493 | +0.0003228188 | +0.0018869042 |
| B | +0.0037180185 | +0.0005733073 | +0.0021456480 |

The preregistered full-budget replicate used 250,000 pairs per epoch, five epochs, and seeds 0, 1,
and 2 on both folds. Across all six matched cells:

| Mean delta GAUC | Mean delta nDCG@5 | Mean delta primary | Worst primary cell |
| ---: | ---: | ---: | ---: |
| +0.0010665655 | +0.0004076262 | +0.0007370909 | -0.0001828671 |

This falsifies a simple promotion claim: average movement is positive, but one seed/fold cell
regresses and the mean effect is well below the `0.002` materiality threshold. The implementation
is retained only as a possible complementary specialist. Exact per-seed values and the runnable
command are in [the duration diagnostic](observed_pair_duration_pilot-20260830.md).

## Implemented score-integrity changes

### Exact deployable-artifact gate

Scientific lineage selection and artifact deployment are now distinct decisions. A generated
candidate can remain a useful research parent, but its finalization plan is not adopted unless:

- the scientific result is materially confirmed;
- the retained seed-0 record and selector/incumbent identities agree;
- the immutable fallback is the verified seed-4 official FM; and
- the exact seed-0 candidate primary exceeds the exact seed-4 fallback primary by strictly more
  than `0.002`.

An exact `+0.002` is rejected. Invalid or incomplete identity evidence fails closed to the current
deployable selection.

### Material promotion gate

The trusted selector no longer replaces the incumbent merely because the matched-seed mean is
slightly positive. A mean outer delta in `(0, 0.002]` is retained as research evidence but does not
promote. This closes the path that treated the earlier `+0.00004225` mean as an incumbent change.

### Cumulative convergence

The convergence state now tracks the score at the beginning of its patience window. Three
successive improvements of roughly `+0.0007` no longer trigger convergence after cumulatively
moving more than `0.002`. An exactly `0.002` cumulative movement remains non-material. Schema-v1
state is explicitly and conservatively migrated to schema v2, with restart and fault-injection
coverage in the campaign store.

### Exact protected-score receipts

Protected-score reuse is now keyed by the exact benchmark, dataset, scorer, split role, and
canonical prediction digest. Operational campaign IDs and execution metadata are excluded. The
historical project ledger remains append-only: its six completed operational rows project to three
unique per-seed score receipts without mutation. An exact hit reuses trusted metrics without a new
reservation or scorer call; a miss durably reserves before scoring and still fails closed when the
query budget is exhausted.

### Truthful scientific iterations and reflection

Execution, policy, serialization, and budget failures no longer masquerade as completed scientific
iterations or inherit fallback metrics in reflection. Only typed completed scientific observations
advance convergence. Failed/unscored experiments report null candidate metrics and cannot poison
family-level research decisions.

## New legal information surface

Feature schema v8 contains 95 columns. Positions 0 through 81 preserve the historical surface,
position 82 is the prefix-fitted video-type code, and positions 83 through 94 add label-free
strict-past exposure context for user, video, author, and user-video scopes:

- strictly earlier exposure count;
- first-seen indicator; and
- `log1p` time since the previous exposure.

Date precedes clock time; equal timestamps are atomic; earlier query inputs may warm later query
rows; and no query outcome is accepted by the interface. Schema versions 1 through 7 retain their
existing replay identities.

## Fusion and replay boundary

The rank-fusion primitive now supports two or more ordered members with one normalization per
member, finite non-negative simplex weights, immutable float64 output, and ordered member/fusion
digests. A versioned homogeneous generated-LambdaRank ensemble recipe and backend can replay every
member in a fixed nested layout and fail closed on any member or fusion mismatch.

The production builder can materialize this ensemble and reconstruct an exact fused validation
artifact. It is deliberately opt-in. The current scientific campaign scores three seeds
individually and selects on their mean; it has never scored the nonlinear fused vector. Deploying
that fusion by default would therefore create a new, untested transformation. A future ensemble
promotion requires a versioned controller-owned outer evaluator, a distinct ensemble score receipt,
and byte-exact handoff of the protected-scored fused artifact to finalization.

## Research conclusion

The plateau is primarily an information/objective/selection problem, not a shortage of model
classes. GAUC positive-count weights mixed users and rewards all within-user pairs; nDCG@5 gives
each user one vote and concentrates on the top five. Loss variants over the same representation
can move one component without materially moving their mean. Strictly monotone recalibration alone
cannot change either rank metric.

The randomized log and generic IPS are not a valid shortcut under the current contract. The
released rows do not provide presentation rank, a full candidate set, or logging/examination
propensities; the randomized period also overlaps public/final dates. The next high-information
experiment is a versioned, train-only watch-progress auxiliary target with native `long_view` as
the only inference head. Scenario-gated residuals and short-horizon causal context follow if that
same-capacity control shows complementary fold signal.

## Verification

At this addendum's verification point:

- full repository: `1,548 passed, 21 skipped`;
- Ruff over source, candidate seed, tests, and the duration diagnostic: passed;
- mypy over all 91 package source files: passed; and
- `git diff --check`: passed.

The skips are explicit environment gates for completed-run artifacts, full-data acceptance jobs,
optional Torch tests, an API key, or archive paths. They are not test failures. No new protected
score or long autonomous campaign was run.
