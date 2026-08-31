# KuaiRand autonomous competition laboratory: implementation handoff

- Status: authoritative implementation plan
- Plan version: 1.0
- Date: 2026-08-31
- Repository root: the directory containing this file (`<repo-root>` below)
- Historical planning record: `docs/history/autonomous-agent-plan-history-2026-08-31.md`
- Companion context: `keylearnings.md`

## 1. Purpose and success boundary

Build one main KuaiRand program that autonomously proposes, trains, evaluates, selects, replays,
and finalizes ranking candidates during the six-hour competition window. There must not be a
separate “competition mode” whose scientific standards are weaker than normal operation. The main
program is the competition program; resource profiles may change execution hardware, but they must
not change leakage controls, metric semantics, promotion standards, replay requirements, or fallback
safety.

The system cannot honestly guarantee that an unknown model family will beat the official FM on an
unseen evaluation set. It can guarantee the controllable part:

1. never knowingly submit a candidate that fails the frozen selection policy;
2. retain the qualified official FM when no challenger passes frozen submission eligibility;
3. spend protected validation queries only on pre-registered, replayable finalists;
4. make every autonomous decision durable and recoverable;
5. separate infrastructure failures from scientific non-improvements;
6. preserve the exact selected prediction vector through replay and submission validation; and
7. produce evidence sufficient to decide whether the score improved.

“Works” therefore means a completed, replayable, resource-bounded campaign with a valid submission
and an auditable decision. “Validation improved” means the selected exact prediction vector is
strictly better on protected primary and passes frozen submission eligibility against the official
fallback. “Materially confirmed” is the stronger uncertainty decision defined below. A successful
run status alone is not improvement.

## 2. Authority order

When sources disagree, use this order:

1. hash-pinned organizer artifacts and the executable benchmark contract in
   `src/kuairand_agent/contract.py`;
2. the executable challenge contract in `src/kuairand_agent/challenge_contract.py`;
3. this plan;
4. `keylearnings.md` and archived planning documents;
5. stale prose, old run reports, or inferred behavior.

Do not weaken an executable organizer rule to match this document. If the contract itself is wrong,
stop and document the discrepancy before changing either the contract or this plan.

## 3. Scope, non-goals, and authorization boundary

In scope:

- the native `long_view` ranking task;
- exact organizer GAUC and nDCG@5 semantics;
- the primary `(GAUC + nDCG@5) / 2` objective;
- official FM qualification and fallback;
- a deterministic, typed experiment ladder;
- GPU-first training with a qualified CPU portability path;
- optional OpenRouter/OpenAI-compatible proposal generation;
- durable recovery, exact finalization, and evidence generation;
- a single six-hour campaign with at most 50 scientific iterations and at most six protected
  validation evaluations.

Out of scope for the first accepted implementation:

- hidden test-label access;
- external training data;
- manual steering of individual experiments;
- a second competition-only orchestration path;
- unconstrained code generation as the primary search mechanism;
- neural ranking as a required MVP dependency;
- distributed workers, Kubernetes, or a remote event store;
- LangChain, LangGraph, MLflow, or Optuna on the correctness-critical path;
- claiming an unseen competition score before the organizer evaluates the final submission.

This plan authorizes implementation and local deterministic tests. It does not by itself authorize a
paid provider call, a protected validation query, or a live six-hour campaign. Those require an
explicit run request and locally available credentials. Never write secrets to configuration, logs,
SQLite, reports, or artifacts.

## 4. Current checkout: treat as a migration, not a blank rewrite

The checkout already contains substantial, partially overlapping implementations. File existence is
not proof that a target module satisfies this plan. The first implementer must classify every relevant
surface as `REUSE`, `ADAPT`, `REPLACE`, or `DELETE_AFTER_CUTOVER` before adding another layer.

Current high-value surfaces include:

| Concern | Existing surface | Initial disposition |
| --- | --- | --- |
| Benchmark identity | `contract.py`, `challenge_contract.py` | REUSE; add contract hash and startup receipt |
| Official FM | `baselines/starter_fm.py`, qualification modules | REUSE after parity qualification |
| Typed experiments | `research/experiment_spec.py` | ADAPT to canonical identity and trainer requests |
| Research proposals | `research/*` | ADAPT behind optional `ProposalAdapter` |
| Candidate execution | `execution/*`, `candidate_api/*` | ADAPT into process isolation below trainer seam |
| Scientific control | `campaign/full_campaign_runtime.py`, controller/scientific modules | REPLACE incrementally behind one facade |
| Durable state | `campaign/store.py`, scientific store, journals, ledgers, checkpoints | CONSOLIDATE into one SQLite authority |
| Tree ranker | `candidates/tree_ranker.py`, generated templates | ADAPT into qualified LightGBM trainer |
| FM variants | `candidates/pairwise_fm.py`, related modules | Retain only after deterministic qualification |
| Rank composition | `finalization/rank_graph.py`, fusion modules | REUSE after exact-vector replay tests |
| Finalization | `finalization/*`, `scoring/*` | ADAPT to one selection and bundle transaction |
| Existing tests | `tests/unit`, `integration`, `fault_injection`, `replay`, `full_data` | REUSE as regression evidence; add missing target tests |

Known migration hazards to test explicitly:

- campaign state, scientific state, protected-query accounting, rejection journals, and filesystem
  checkpoints currently have more than one writable authority;
- a crash can leave a later challenge iteration beside an earlier coarse scientific checkpoint;
- not every rejection path is guaranteed to use the same durable journal;
- current `full-pure.toml` is CPU-only and has a 16 GiB memory limit, so it is not the target GPU-first
  profile;
- the checkout is dirty and contains user work. Never discard or overwrite unrelated changes.

Before editing source:

1. record `git status --short`, current branch, and `git diff --stat` in a local migration note;
2. run the existing fast deterministic test suite and record failures without “fixing” unrelated work;
3. inventory every writer of campaign state, outer-query accounting, selection, and finalization;
4. capture golden fixtures for official FM predictions, metric results, rank-graph output, and a small
   completed scripted campaign;
5. map each old write path to its future authoritative transaction; and
6. avoid deleting an old path until its replacement passes recovery and replay tests.

## 5. Immutable competition contract

Admission must fail before any proposal or training when any of these values drift:

| Field | Required value |
| --- | --- |
| Dataset | KuaiRand-Pure, hash-verified by the pinned contract |
| Target | `long_view` |
| Ranking unit | exact organizer contract value |
| Metrics | `GAUC`, `nDCG@5` |
| Primary | `(GAUC + nDCG@5) / 2` |
| nDCG cutoff | 5 |
| Scientific iterations | maximum 50 |
| Wall clock | 21,600 seconds |
| Convergence | epsilon 0.002, patience 3, exact executable comparison |
| Selection | validation-best subject to this plan’s promotion and fallback gates |
| Protected evaluations | maximum 6 per `ContractId` evidence lineage |
| External training data | forbidden |
| Hidden test outcomes | unrepresentable during development and finalization |
| Prediction context | strict past only |

Create a canonical `ContractManifest` containing the executable benchmark manifest, challenge
manifest, organizer file hashes, dataset hash, split identities, metric implementation hash, row
identity policy, and submission schema. Canonical JSON plus SHA-256 is `ContractId`.

Every campaign, experiment, trial, score, prediction vector, selection decision, replay, and bundle
must reference the same `ContractId`. Resume must reject a different contract rather than silently
starting from mixed evidence.

## 6. External interface: one deep module

Expose one small public module. The implementation behind it owns admission, budgets, search,
durability, training, evaluation, selection, replay, and finalization.

Python interface:

```python
lab = AutonomousExperimentLab.open(
    repository_root=repo,
    state_root=repo / ".kuairand",
    run_root=run_root,
    profile="gpu",  # or "cpu"
)

result = lab.compete(
    options=CampaignOptions(config_path=config_path),
    idempotency_key=campaign_key,
)
```

CLI interface:

```text
kuairand-agent compete --config configs/competition-gpu.toml --state-root .kuairand --run-root runs/<id>
kuairand-agent compete --config configs/competition-cpu.toml --state-root .kuairand --run-root runs/<id>
kuairand-agent inspect --state-root .kuairand --campaign-id <id> --json
kuairand-agent replay --state-root .kuairand --campaign-id <id> --grade scoring-exact
kuairand-agent validate-bundle --bundle runs/<id>/final/submission-bundle
```

The CPU and GPU commands enter the same `compete` implementation. `profile` is a resource and
trainer-selection input, not a scientific-policy mode. Do not add `competition_mode`,
`fast_competition`, or alternate selection logic.

The returned `CampaignResult` is terminal and typed:

```text
CampaignResult
  campaign_id
  contract_id
  terminal_state
  submission_disposition
  scientific_disposition
  selected_prediction_id
  fallback_prediction_id
  exact_metrics
  bundle_path
  bundle_sha256
  replay_grade
  resource_receipt_id
  protected_query_count
  evidence_manifest_path
```

## 7. Target architecture and seams

Use module depth deliberately: the public facade is small; complexity remains behind cohesive
interfaces. Create or converge toward this ownership map:

```text
src/kuairand_agent/
  lab.py                       # AutonomousExperimentLab public facade
  contract.py                  # immutable organizer and metric identity
  challenge_contract.py        # six-hour competition rules
  domain/
    identity.py                # canonical hashes and typed IDs
    decisions.py               # states and dispositions
    experiment.py              # semantic ExperimentSpec
  state/
    schema.py                  # SQLite schema and migrations
    repository.py              # sole durable write API
    projections.py             # rebuildable JSON/report views
  proposal/
    protocol.py                # ProposalAdapter
    deterministic.py           # mandatory built-in ladder
    openai_compatible.py       # optional OpenRouter/OpenAI adapter
  training/
    protocol.py                # QualifiedTrainer
    process_executor.py        # isolated child execution and limits
    official_fm.py
    lightgbm_cpu.py
    lightgbm_gpu.py
    qualification.py
  evaluation/
    inner.py                   # fold and cluster-resampling evidence
    protected.py               # budgeted quarantined validation service
    promotion.py               # frozen PromotionPolicy
  search/
    tournament.py              # deterministic archive/beam ownership
    family_ledger.py           # cross-campaign evidence firewall
  finalization/
    rank_graph.py              # exact declarative vector composition
    selection.py               # fallback/challenger decision
    replay.py
    bundle.py
  observability/
    receipts.py
    report.py
```

Existing modules may remain in their current paths when moving them would add risk, but ownership and
imports must follow this graph. Old campaign code may call the new facade during migration; the new
facade must never call two competing authorities for the same decision.

### 7.1 Internal seams

These are controlled, stable protocols:

- `StateRepository`: sole writer for campaign truth and the protected-query ledger;
- `QualifiedTrainer`: converts a typed trial request into a prediction artifact and receipts;
- `ProcessExecutor`: runs an admitted attempt with process-tree limits and cancellation;
- `InnerEvaluator`: calculates leakage-safe evidence from immutable predictions;
- `ProtectedEvaluator`: spends one authorized query and quarantines its outcome;
- `PromotionPolicy`: turns frozen evidence into a typed decision;
- `RankGraphEvaluator`: builds one exact candidate prediction vector;
- `BundleFinalizer`: atomically selects, replays, validates, and seals the bundle.

### 7.2 External adapters

External adapters may fail without corrupting the laboratory:

- `ProposalAdapter` for OpenRouter/OpenAI-compatible structured proposals;
- optional observer adapter for MLflow or another dashboard;
- filesystem and process platform adapters;
- optional future scheduler adapter.

LangChain may be used inside `ProposalAdapter` if it reduces provider boilerplate. LangGraph must not
own campaign truth, budgets, retries, or replay. Provider outages produce typed proposal failures;
the deterministic proposal ladder remains sufficient to finish a valid campaign.

### 7.3 Research-quality boundary

The proposal layer may research ranking methods, objectives, leakage risks, and implementation
constraints, but research is advisory evidence rather than training data or campaign truth.

- Prefer original papers, organizer code, and official library documentation over summaries.
- Record source title, canonical URL or local path, retrieval date, content digest where available,
  and the specific mechanism claim supported.
- Separate retrieved prose from executable `ExperimentSpec` fields.
- Never import external user/item interactions, labels, embeddings, or benchmark outcomes into model
  training or feature construction.
- Require every research-derived proposal to state a falsifiable mechanism, matched control,
  expected metric effect, leakage argument, and rejection criterion.
- Treat source freshness and provider confidence as metadata, never as promotion evidence.
- Keep the deterministic offline ladder capable of finishing when web research or OpenRouter is
  unavailable.

## 8. Canonical identities

Canonicalize all manifests as UTF-8 JSON with sorted keys, normalized numeric representations, no
absolute machine paths, and content hashes for referenced inputs.

| Identity | Includes | Excludes |
| --- | --- | --- |
| `ContractId` | organizer, data, split, metric, row and submission contract | machine and provider state |
| `CampaignId` | `ContractId`, campaign config, start nonce | transient PID |
| `FamilyId` | scientific mechanism, feature/target/objective family | seed, device, attempt |
| `ExperimentId` | semantic `ExperimentSpec`, data/fold identities, code artifact | device, retry count |
| `TrialId` | `ExperimentId`, trainer ID/version, backend, precision, dependency lock, seed/fold/fidelity | PID, hostname |
| `AttemptId` | `TrialId`, monotonically increasing infrastructure attempt | scientific semantics |
| `PredictionId` | ordered row IDs, exact prediction bytes, producing `TrialId` or rank graph | report labels |
| `DecisionId` | policy hash plus exact evidence IDs | prose explanation |
| `BundleId` | selected prediction, replay outputs, submission and manifest hashes | mutable log paths |

A GPU-to-CPU switch keeps the same `ExperimentId` but creates a new `TrialId`. An OOM retry with the
same backend and admitted settings creates a new `AttemptId`. Changing batch size only keeps the same
`TrialId` when qualification proves batch size cannot change prediction semantics; otherwise it is a
new `TrialId`.

## 9. Typed experiment and trainer contracts

`ExperimentSpec` must be the only scientific proposal language accepted by the controller. It must
be canonical, schema-versioned, prose-insensitive, and declarative. At minimum it contains:

```text
mechanism and falsifiable hypothesis
parent/fallback references
feature view IDs and strict-past lookback policy
training target and allowed auxiliary targets
model family and bounded hyperparameters
objective and grouping unit
fold protocol and seed set
screening fidelity
required ablations
expected resource class
promotion policy version
rank-graph recipe, if composition is requested
```

Reject free-form source code as a substitute for missing fields. Any generated implementation must
be materialized from an allowlisted template and content-hashed before admission.

`QualifiedTrainer` is the principal hardware seam:

```python
class QualifiedTrainer(Protocol):
    @property
    def identity(self) -> TrainerIdentity: ...

    def preflight(self, request: TrialRequest) -> QualificationReceipt: ...

    def fit_predict(self, request: TrialRequest) -> TrialResult: ...
```

`TrialResult` must contain exact predictions plus data, feature, model, environment, resource, and
timing receipts. Trainer errors are typed as `UNSUPPORTED`, `ADMISSION_REJECTED`, `TIMEOUT`, `OOM`,
`CANCELLED`, `DEPENDENCY_ERROR`, `NUMERICAL_ERROR`, or `INTERNAL_ERROR`.

The process executor is deliberately lower-level and model-agnostic. It owns child process groups,
deadlines, stdout/stderr capture, signal escalation, peak RSS/disk accounting, and stale-process
cleanup. It does not choose a model, retry science, spend protected queries, or decide promotion.

## 10. GPU-first, CPU-reversible profiles

Implement two explicit configurations with identical scientific gates:

- `configs/competition-gpu.toml`: preferred profile;
- `configs/competition-cpu.toml`: degraded-throughput portability profile.

Common immutable policy:

- one candidate process at a time initially;
- four CPU threads per candidate unless qualification proves another fixed value;
- deterministic seeds and stable row ordering;
- strict-past features only;
- identical folds, metric code, materiality margins, multiple-comparison correction, fallback policy,
  replay grades, and bundle checks;
- no training launch after minute 300;
- 60 minutes reserved for replay, finalization, and contingency;
- candidate host target of 12 GiB RSS and process-tree hard cap of 14 GiB;
- 20 GiB candidate disk cap unless measured evidence justifies less;
- no device fallback inside a trainer attempt.

GPU profile:

- LightGBM GPU trainer preferred where qualified;
- up to eight full scientific screens, subject to measured p95 duration;
- up to two frozen finalists;
- all three confirmation seeds `[0, 1, 2]` for every protected-eligible finalist; if the remaining
  budget cannot admit them, that candidate is not protected-eligible;
- at most two protected evaluations planned, while the global ledger limit remains six;
- CPU official FM remains available as fallback and parity anchor.

CPU-degraded profile:

- no import, initialization, or probing of GPU-only libraries during normal execution;
- at least four full scientific screens;
- at least one strict-past ablation;
- at least one frozen finalist when a candidate passes inner gates;
- screening may use the frozen low-fidelity seed policy, but every protected-eligible finalist must
  complete confirmation seeds `[0, 1, 2]`; an unadmitted or incomplete seed set forces fallback;
- at most one planned protected challenger evaluation;
- official FM qualification, exact replay, and final bundle validation remain mandatory.

The CPU profile may reduce breadth and screening fidelity only through predeclared rungs. It may not
lower quality, leakage, materiality, reproducibility, or finalization gates.

### 10.1 Qualification and fallback

Before campaign admission:

1. qualify official FM on the fixture and full-data parity rung;
2. qualify LightGBM CPU on deterministic fixture predictions;
3. if GPU profile is requested, qualify GPU availability, backend support, memory headroom, and a
   same-backend deterministic replay;
4. run cross-backend portability on the fixture and compare metrics and ranking order under a frozen
   tolerance;
5. measure p50/p95 wall time and peak resources for each admitted rung; and
6. freeze the resulting capability receipt into campaign configuration.

If GPU preflight fails before campaign admission, select the CPU profile and create the campaign only
after recomputing its budget. If GPU fails after a GPU trial is admitted, close that attempt with a
typed infrastructure failure, create a new CPU `TrialId`, and re-run only if the remaining-budget
admission test passes. Never relabel CPU output as the GPU trial.

## 11. One durable authority and recovery model

Use one SQLite database at `<state-root>/authority.sqlite3` as the sole mutable authority for the
laboratory, all of its campaigns, and the cross-campaign evidence lineage. The state root is outside
individual run directories, is gitignored, and is never reset merely because `runs/` is cleaned.
Require the state root explicitly in production; the CLI examples use `<repo-root>/.kuairand` as the
local default. Enable WAL, foreign keys, busy timeout, explicit schema migrations, and
transactionally allocated sequence numbers. Run-directory JSON, Markdown, filesystem status files,
and sealed SQLite exports are projections that can be deleted and rebuilt.

Minimum tables:

```text
contracts
campaigns
campaign_events
families
experiments
trials
attempts
artifacts
predictions
inner_evaluations
protected_query_reservations
protected_evaluations
promotion_decisions
rank_graphs
selection_decisions
replays
bundles
resource_receipts
provider_operations
failures
```

Every state transition is compare-and-swap on the expected prior state and appends an immutable event
in the same transaction. Artifact files use temporary names, fsync where required, content-hash
verification, and atomic rename before the database commits their availability.

Protected queries use a reservation transaction:

1. check `ContractId` lineage budget and family eligibility;
2. reserve the next query ordinal against an exact `PredictionId`;
3. call the scorer once with an idempotency key;
4. journal either the exact result or typed unknown outcome;
5. never automatically retry an unknown outcome;
6. expose the outcome to the selection service, not proposal/training code.

Recovery rules:

- terminal records are immutable;
- `RUNNING` attempts whose process identity is absent become typed `INTERRUPTED` failures;
- committed artifacts are verified by hash before reuse;
- uncommitted temporary artifacts are quarantined, not trusted;
- projections are rebuilt from SQLite;
- resume continues the same `CampaignId`, clock basis, budgets, and protected-query ledger;
- the controller reconciles campaign iteration, scientific checkpoint, selection, and finalization in
  one transaction boundary;
- stale or duplicate workers cannot acquire a completed lease;
- rejection reasons from every admission path use the same durable event path.

## 12. Evidence firewall

There are three evidence zones:

1. **Inner scientific evidence**: folds, seeds, ablations, resource results. It may inform proposals,
   search, and parent selection.
2. **Protected validation evidence**: scarce organizer-equivalent outcomes. It may inform only
   promotion, final selection, and reports.
3. **Final-period data**: features may be used only as permitted by the organizer for final
   prediction; outcomes remain inaccessible and unrepresentable.

Protected scores must never be included in proposal prompts, `ExperimentSpec`, feature selection,
hyperparameter search, parent-selection fitness, or training data. Enforce this with separate types,
module imports, database views, and tests—not prompt wording.

The cross-campaign family ledger is keyed by `ContractId` and `FamilyId`. It survives campaign
restarts, source changes, provider changes, and run-directory changes. A new run directory must not
reset protected-query eligibility. Only a genuinely new `ContractId` creates a new evidence lineage,
and contract changes require explicit migration review.

Once the first protected result is committed, freeze research permanently for that campaign: no new
proposal, feature, parent, hyperparameter, seed, weight, or fusion decision may be created from any
protected outcome. Remaining pre-registered finalists may be scored in their frozen query order;
otherwise proceed directly to selection and finalization.

## 13. Search program

Use a deterministic ladder as the mandatory search engine. A provider may propose bounded parameter
values or choose among allowed next experiments, but provider availability is not required.

Priority order:

1. hash-qualified official FM fallback;
2. LightGBM LambdaRank using organizer ranking groups and leak-safe base features;
3. strict-past user/video/exposure recency and frequency features;
4. target/objective ablations, including allowed auxiliary targets without changing the native final
   target;
5. one pointwise tree control where it falsifies the ranking-objective hypothesis;
6. qualified pairwise/FM variants only when existing evidence supports materiality;
7. exact rank-level fusion of diverse, frozen candidates.

Do not make neural rankers part of MVP acceptance. Admit them later only after CPU portability,
resource predictability, deterministic replay, and incremental value over the tree/FM ladder are
demonstrated.

Tournament rules:

- candidates compete against matched parents and the official fallback;
- screening uses cheap, predeclared rungs; no post-hoc favorable subset;
- archive slots preserve performance and mechanism diversity;
- non-improving scientific experiments are valid negative evidence, not infrastructure failures;
- failed infrastructure attempts do not count as scientific losses;
- parent selection sees inner evidence only;
- no protected query is spent on a submaterial inner delta;
- composition is evaluated as one exact fused vector before any promotion claim.

## 14. Promotion policy

Implement a versioned `PromotionPolicy` as executable configuration. Freeze it before the first
scientific trial. Required fields:

```text
primary metric formula
component-metric non-regression margins
practical primary-improvement margin
cluster unit for resampling
resample count
bootstrap/random seed
confidence level
one-sided or two-sided comparison
family of simultaneous hypotheses
Holm correction procedure and alpha
tie-breaking policy
missing/invalid cluster handling
minimum eligible users/clusters
minimum completed folds and seeds
protected-query eligibility rule
```

The version-1 policy is exact and must not be tuned after seeing campaign outcomes:

- cluster unit: organizer user identity, never individual rows;
- comparison: paired candidate and fallback predictions on identical ordered rows;
- resamples: 10,000 user-cluster bootstrap replicates;
- bootstrap seed: `20260831`;
- confidence procedure: one-sided 95% lower confidence bound;
- simultaneous family: all frozen finalists in the protected batch, maximum two;
- multiplicity: Holm correction at family-wise alpha `0.05`;
- required inner evidence: both configured temporal folds and all confirmation seeds `[0, 1, 2]`;
- required cluster population: 100% of users eligible under the organizer metric after exact paired
  row alignment; any unexplained loss, duplication, or eligibility mismatch is a hard failure;
- inner protected-batch eligibility: mean primary delta `>= +0.002`, worst temporal-fold primary
  delta `>= -0.002`, and each mean component delta `>= -0.001`;
- submission challenger eligibility: protected primary delta strictly `> 0`, each protected
  component delta `>= -0.001`, valid resource receipts, and required replay grades;
- material scientific confirmation: Holm-adjusted primary lower bound strictly `> +0.002`, protected
  primary point delta `>= +0.002`, and both protected component deltas `>= -0.001`;
- ties: retain the simpler, cheaper, already qualified fallback;
- missing, duplicated, reordered, or invalid clusters: hard evidence failure, never denominator
  adjustment;
- arithmetic: full-precision internal metrics; published four-decimal values are display-only.

Keep two separate terminal decisions:

- `SubmissionDisposition`: `CHALLENGER_SELECTED` or `OFFICIAL_FM_RETAINED`;
- `ScientificDisposition`: `MATERIALLY_CONFIRMED`, `NOT_CONFIRMED`, or
  `INSUFFICIENT_VALID_EVIDENCE`.

A challenger may be selected only when the exact final vector passes submission eligibility. It is
`MATERIALLY_CONFIRMED` only when it also passes the stronger uncertainty rule. This intentionally
allows a strictly better protected validation vector to be selected while labeling weak uncertainty
honestly. If no challenger passes submission eligibility, retain official FM. If no challenger
passes scientific confirmation, report that no material improvement was confirmed even when a
strictly higher eligible challenger was selected.

## 15. Exact composition and replay grades

Rank composition must be declarative. A `RankGraph` contains only content-addressed prediction
inputs, deterministic transforms, fixed weights, tie policy, normalization scope, and ordered row
identity. The graph evaluation creates a new `PredictionId`; member scores are not the ensemble score.

Define replay grades explicitly:

| Grade | Requirement |
| --- | --- |
| `SCORING_EXACT` | identical stored prediction bytes produce identical exact metrics |
| `EXPERIMENT_SAME_BACKEND` | same trial manifest/backend reproduces predictions or frozen exact tolerance |
| `EXPERIMENT_TOLERANT` | approved environment drift stays inside metric and rank tolerances |
| `CROSS_BACKEND_PORTABILITY` | CPU/GPU implementations satisfy frozen portability thresholds |
| `BUNDLE_EXACT` | clean finalization regenerates identical submission bytes and bundle hash |

Do not use the word “replayable” without naming the achieved grade. Final acceptance requires
`SCORING_EXACT` and `BUNDLE_EXACT`. GPU campaign acceptance additionally requires same-backend GPU
replay; CPU fallback readiness requires same-backend CPU replay. Cross-backend portability is a
separate receipt and must not be confused with bitwise identity.

## 16. Six-hour schedule and admission control

The clock begins only after contract, data, fallback, environment, and requested-profile admission
succeed. Persist monotonic elapsed time so a process restart cannot reset the budget.

GPU target schedule:

| Elapsed time | Work |
| --- | --- |
| 0–45 min | admission receipts, cache/materialization, official FM and GPU qualification |
| 45–195 min | up to eight deterministic scientific screens |
| 195–255 min | full-fidelity confirmation and required ablations for up to two finalists |
| 255–300 min | frozen batch comparison and authorized protected evaluations |
| 300–360 min | selection, exact replay, bundle validation, contingency |

CPU target schedule:

| Elapsed time | Work |
| --- | --- |
| 0–60 min | admission, cache/materialization, official FM and CPU trainer qualification |
| 60–210 min | at least four scientific screens plus strict-past ablation |
| 210–270 min | full-fidelity confirmation of the best eligible finalist |
| 270–300 min | frozen comparison and at most one planned protected challenger evaluation |
| 300–360 min | selection, exact replay, bundle validation, contingency |

Before every launch calculate:

```text
remaining_wall_time
- finalization_reserve
- p95_duration_for_requested_rung
- cancellation_and_persistence_margin
```

Reject the launch if the result is negative. At minute 300, cancel or close all non-finalization work
according to the frozen policy. Never trade away final bundle validity for one more experiment.

## 17. Resource and provider receipts

Every attempt receipt records:

- trainer/backend/device identity and availability;
- CPU model, logical threads, thread environment, and process-tree peak RSS;
- GPU model/UUID, driver/runtime/library versions, precision, peak allocated and reserved memory,
  utilization sample summary, and GPU-active time;
- disk bytes written and artifact sizes;
- wall/CPU time, deadline, cancellation sequence, and exit status;
- dependency lock hash, source artifact hash, data/feature/prediction hashes;
- whether the result came from preferred backend, declared fallback, or no backend.

OOM, timeout, dependency, provider, and process failures are infrastructure evidence. They are not
negative model evidence. Provider operations have a 420-second total request deadline unless the
frozen config is made stricter. A timeout must journal a retry or typed terminal failure; no request
may remain indefinitely “running.”

## 18. Dependency profiles

Keep optional dependencies outside the core runtime:

```text
core
  numpy, psutil, standard-library SQLite, official FM requirements

tree-cpu
  LightGBM CPU and pinned numeric dependencies

tree-gpu
  a verified GPU-enabled LightGBM build and pinned runtime requirements

proposal-openai-compatible
  minimal HTTP/schema adapter; optional LangChain wrapper

observer-mlflow
  optional projection only

optimizer-optuna
  optional bounded proposer only
```

The CPU installation must function without GPU libraries. The core must complete a scripted campaign
without provider, LangChain, MLflow, or Optuna. Lock files and backend build capabilities are part of
`TrialId`; an installed package version alone is not proof that GPU support exists.

## 19. Ordered implementation phases

Each phase ends with a recorded decision: `GO`, `REWORK`, or `STOP`. Do not begin the next phase on a
red gate.

### Phase -1: migration inventory and golden evidence

Deliver:

- dirty-tree-safe inventory;
- writer/reader map for all current state surfaces;
- `REUSE/ADAPT/REPLACE/DELETE_AFTER_CUTOVER` file matrix;
- golden official FM, scoring, rank graph, scripted campaign, and final bundle fixtures;
- baseline test report distinguishing pre-existing from introduced failures.

Gate:

- every current authoritative write path is mapped;
- golden fixtures are content-hashed;
- no user change has been discarded;
- no new architecture code has been added before the map is reviewable.

### Phase 0: contract and identity kernel

Deliver:

- canonical manifests and typed IDs;
- startup contract verification and receipt;
- hidden-outcome-unrepresentable data types;
- profile-independent `PromotionPolicy` schema;
- tests for canonical hashing and contract drift.

Gate:

- semantically equal manifests hash identically;
- meaningful contract/backend changes alter the correct identity;
- admission fails before training on any contract mismatch.

### Phase 1: trainer and process seams

Deliver:

- `QualifiedTrainer` protocol;
- official FM, LightGBM CPU, and LightGBM GPU adapters;
- isolated process executor and process-tree accounting;
- scripted trainer for deterministic tests;
- `configs/competition-cpu.toml` and `configs/competition-gpu.toml` with shared scientific-policy
  validation;
- pinned `tree-cpu` and `tree-gpu` dependency profiles; and
- qualification and resource receipts.

Gate:

- CPU path imports no GPU runtime;
- absent GPU selects CPU before campaign creation;
- post-admission GPU failure creates a distinct CPU `TrialId`;
- timeout/OOM/cancellation leaves no descendant process;
- same-backend fixture replay passes.

### Phase 2: single state authority and deep facade

Deliver:

- migrated SQLite schema and repository;
- evented compare-and-swap transitions;
- protected-query reservations;
- projection rebuild command;
- `AutonomousExperimentLab` facade and `compete/inspect/replay` CLI;
- compatibility wrapper from old campaign entrypoint to the new facade.

Gate:

- crash injection after every transition recovers deterministically;
- deleting all projections loses no truth;
- duplicate/stale workers cannot double-commit;
- challenge and scientific iteration cannot diverge;
- every rejection path is durable;
- no dual writes remain for migrated decisions.

### Phase 3: offline end-to-end fallback path

Deliver:

- data admission and strict-past feature materialization;
- official FM qualification;
- exact inner scoring;
- fallback selection;
- clean replay and sealed submission bundle;
- both CPU and GPU profile fixture runs without a provider.

Gate:

- official FM predictions match the frozen qualification target;
- no final-period outcomes are opened;
- both profiles produce a valid fallback bundle;
- `SCORING_EXACT` and `BUNDLE_EXACT` pass.

### Phase 4: deterministic scientific ladder

Deliver:

- typed built-in proposals for LambdaRank, strict-past features, objective controls, and pointwise
  control;
- matched-control tournament and archive;
- inner evidence persistence and family ledger;
- optional provider adapter constrained to the same schema.

Gate:

- core campaign finishes with provider disabled;
- provider malformed output and timeout become typed failures without corrupting state;
- protected evidence cannot reach proposal or parent-selection code;
- ordinary non-improvements remain scientific outcomes.

### Phase 5: frozen confirmation and promotion

Deliver:

- user-cluster paired resampling;
- component and primary deltas;
- Holm correction and tie policy;
- separate scientific and submission dispositions;
- exact rank-graph finalist construction;
- transactional protected-query flow.

Gate:

- promotion is invariant to report prose and proposal source;
- exact fused vector, not its members, is evaluated and selected;
- submaterial candidates cannot spend protected budget;
- unknown protected outcomes cannot be retried automatically;
- official FM wins every tie or invalid-evidence case.

### Phase 6: full acceptance and cutover

Deliver:

- measured GPU and CPU p95 resource budgets;
- one bounded GPU rehearsal and one bounded CPU rehearsal on full data without protected scoring;
- recovery, replay, and finalization evidence;
- deletion of obsolete dual-write paths after all compatibility consumers move;
- operator-facing inspect report and implementation handoff.

Gate:

- both profiles meet their minimum search program inside five hours;
- one hour remains for finalization;
- resource caps are enforced on the whole process tree;
- final bundle validates against the organizer loader;
- legacy and new authorities cannot both mutate campaign truth;
- all claim-language tests pass.

### Phase 7: optional extensions

Only after Phase 6 is green, consider bounded Optuna proposals, MLflow projection, LangChain provider
convenience, neural candidates, or parallel workers. Each new adapter needs two real implementations
or one real implementation plus a proven imminent use; do not introduce speculative interfaces.

## 20. Required test matrix

### Contract and leakage

- organizer file, dataset, split, row identity, metric, and submission hash drift;
- exact GAUC and nDCG@5 fixture parity;
- strict-past boundary at every fold and final-period boundary;
- forbidden external data and hidden labels cannot enter `ExperimentSpec`;
- protected result types are not importable by proposal/training modules.

### Identity and artifacts

- prose-only proposal changes do not change `ExperimentId`;
- semantic feature/objective changes do;
- device/backend/precision changes alter `TrialId` but not `ExperimentId`;
- retries alter only `AttemptId`;
- row reorder changes or invalidates `PredictionId`;
- corrupt or partially written artifacts are rejected.

### Trainer portability

- CPU-only environment and dependency installation;
- missing GPU before admission;
- unsupported LightGBM GPU build;
- GPU OOM and GPU loss after admission;
- CPU fallback creates new trial identity;
- same-backend deterministic replay;
- cross-backend rank/metric tolerance and explicit failure outside tolerance.

### State and failure injection

- kill after each transaction boundary;
- kill after artifact rename but before database commit and the reverse order;
- stale launcher/worker PID and PID reuse;
- duplicate resume and duplicate protected-query request;
- provider deadline, malformed schema, transport retry, and typed terminal failure;
- cancellation escalation and zero leftover descendants;
- challenge iteration/checkpoint mismatch regression;
- pre-admission and post-proposal rejection durability;
- projection deletion and complete rebuild.

### Scientific decisions

- paired user-cluster resampling fixture;
- fixed bootstrap seed;
- Holm edge cases and missing clusters;
- component regression despite primary gain;
- submaterial primary delta;
- tied candidate/fallback;
- distinct submission and scientific dispositions;
- one exact composed vector selected and persisted.

### Finalization

- exact selected `PredictionId` reaches the submission rows;
- organizer header, row count, row identity, finite values, and ordering;
- clean-environment scoring and bundle replay;
- immutable selection evidence and fallback evidence;
- bundle and submission SHA-256;
- outer/protected query accounting matches journal exactly;
- zero final-period outcome access;
- resource receipt completeness.

## 21. Live campaign operating and monitoring protocol

When a live six-hour campaign is explicitly authorized, launch it through the same `compete` entry
point and do not manually choose hypotheses, implementations, training trials, parents, finalists,
or iterations.

Monitoring cadence:

1. for the first 10 minutes, inspect durable state, launcher/worker liveness, resource receipts,
   provider deadlines, and newly journaled failures at intervals no longer than one minute;
2. after minute 10, inspect at 30-minute intervals until the campaign reaches a terminal state or the
   six-hour deadline;
3. report only meaningful state, infrastructure, budget, selection, or finalization changes;
4. treat ordinary non-improving or rejected scientific experiments as expected evidence, not defects;
5. never expose protected outcomes to research or use monitoring to steer the campaign.

A genuine runtime defect is one of: unexpected campaign-process exit, missing or divergent durable
transition, orphaned process tree, process/resource cap enforcement failure, artifact hash failure,
provider operation beyond its total deadline without a journaled retry or typed failure, unauthorized
protected query, query-ledger mismatch, replay mismatch, or finalization contract failure.

On a genuine defect:

1. stop only processes belonging to that campaign and verify that no descendants remain;
2. preserve the run directory, state authority, logs, artifacts, and exact failure receipt;
3. attempt same-campaign resume only when recovery validation proves the source, contract, trial
   identities, query ledger, and clock are unchanged;
4. if a source or configuration repair is required, terminally close the defective campaign,
   implement and qualify the repair, and create a new `CampaignId`—never resume mixed code under the
   old identity; and
5. do not manually alter a scientific decision to compensate for infrastructure failure.

At terminal completion, verify exact fallback and selected metrics, selection and scientific
dispositions, selected `PredictionId`, clean replay grades, submission schema and SHA-256, bundle
hash, resource receipts, provider accounting, protected-query ordinals, and zero final-period outcome
access. A terminal status without this evidence is not a completed campaign claim.

## 22. Acceptance commands and evidence bundle

The implementer may choose exact command names while the facade is being introduced, but the final
handoff must provide one command for each of these gates:

```text
install core + test dependencies
run fast deterministic unit tests
run integration and fault-injection tests
run CPU-only acceptance in an environment without GPU packages
run GPU qualification when compatible hardware exists
run full-data no-protected-score rehearsal for each available profile
run clean replay
validate the final bundle with organizer code
inspect a campaign as durable JSON
```

Each accepted rehearsal emits:

```text
contract-manifest.json
campaign-manifest.json
campaign-state-snapshot.sqlite3
event-export.jsonl
selection-evidence.json
scientific-decision.json
submission-decision.json
replay-receipt.json
resource-receipts.jsonl
protected-query-accounting.json
provider-accounting.json
failure-summary.json
submission.csv
bundle-manifest.json
bundle.sha256
report.md
```

`<state-root>/authority.sqlite3` remains authoritative; the campaign-scoped SQLite snapshot and all
other exported files are sealed projections for review. The snapshot must include the campaign’s
query ordinals and family-lineage references but cannot become a writable resume source.

## 23. Claim taxonomy

Reports and CLI output must use these exact distinctions:

- **Implemented**: source exists and deterministic tests pass.
- **Qualified**: an adapter/backend passed its declared qualification receipt.
- **Campaign completed**: the state machine reached a valid terminal state.
- **Replay verified**: name the achieved replay grade.
- **Validation improved**: exact protected candidate vector has primary delta strictly `> 0` and
  passes frozen submission eligibility against the official fallback; include exact GAUC, nDCG@5,
  primary, deltas, and query ordinal.
- **Materially confirmed**: the same exact vector additionally passes the frozen Holm-adjusted
  uncertainty and `+0.002` materiality rules.
- **Submission selected**: challenger or official FM was transactionally selected and bundled.
- **Competition score improved**: use only after the organizer evaluates that exact submission.

Never infer score improvement from `COMPLETED`, inner-fold uplift, a positive single seed, an
individual ensemble member, fallback-only output, or successful bundle generation.

## 24. First working session checklist

The next implementer should do this in order:

1. read this plan, `keylearnings.md`, `contract.py`, and `challenge_contract.py`;
2. inspect dirty status without changing it;
3. complete Phase -1’s writer map and golden fixtures;
4. run existing fast tests and classify pre-existing failures;
5. write failing tests for the iteration/checkpoint mismatch and rejection-journal gaps;
6. introduce canonical IDs and the `QualifiedTrainer` protocol without moving unrelated files;
7. build the CPU scripted vertical slice first;
8. add GPU qualification as a second real trainer implementation;
9. migrate one state transition at a time into the single repository;
10. keep a compatibility wrapper until the old entrypoint has no independent writes;
11. stop at the first red phase gate and repair it before expanding scope; and
12. do not start a paid/provider/protected/full six-hour run without explicit authorization.

The CPU scripted slice in step 7 is an implementation-order test of modularity, not a change to the
production preference: an admitted live campaign still chooses the qualified GPU profile first and
uses CPU only through the declared portability policy.

## 25. Definition of done

This reconstruction is done only when all of the following are simultaneously true:

- there is one main `compete` path and no weaker competition-only mode;
- CPU and GPU are trainer/resource profiles behind the same scientific controller;
- the CPU profile works in a genuinely GPU-free environment;
- official FM is hash-qualified and always available as safe fallback;
- deterministic tree/feature experiments can run without an LLM provider;
- optional provider proposals cannot bypass typed specs or scientific gates;
- one SQLite authority owns state, query budgets, decisions, and recovery;
- protected evidence is firewalled from adaptive research;
- every protected query is durable, bounded, and tied to an exact prediction vector;
- promotion statistics, materiality, correction, tie, and missing-data rules are executable and
  frozen;
- exact rank composition is scored, selected, replayed, and bundled as one vector;
- recovery tests cover every state transition and leave no live descendants;
- GPU and CPU resource receipts prove the declared caps and timing assumptions;
- both profiles pass fallback finalization, `SCORING_EXACT`, and `BUNDLE_EXACT`;
- a full-data no-protected-score rehearsal completes inside the measured budget;
- the final report distinguishes completion, validation evidence, selection, and unseen organizer
  score; and
- old dual-write paths are removed only after the replacement passes all gates.

At that point the laboratory is ready for an explicitly authorized six-hour campaign. Its design
maximizes the probability of finding and retaining a real improvement while preserving a valid,
replayable official fallback when the evidence does not support one.
