# Key learnings for the next KuaiRand agent

Last consolidated: 2026-08-31

Purpose: give the next agent a durable, evidence-conscious handoff across the related
TiktokTechJam tasks without requiring it to reconstruct the project from task titles, stale run
statuses, or historical campaign directories.

This document is guidance, not an executable contract. The checked-out starter/evaluator,
current source, current configuration, and newly produced durable receipts are authoritative.
Historical scores and paths below describe prior evidence and may be stale or absent because the
user intentionally requested execution-history cleanup.

## 1. Read this first

Before editing or launching anything:

1. Read the authoritative implementation handoff in `plan.md`.
2. Inspect `git status`, the full diff, branch, source digest, and untracked files.
3. Preserve the dirty worktree. Do not reset, discard, overwrite, or reinterpret existing changes.
4. Re-read the executable organizer contract and untouched evaluator. Do not rely on prose from an
   old attachment when it conflicts with executable starter code.
5. Verify whether data, qualification evidence, dependencies, GPU hardware, provider credentials,
   run directories, and protected-query receipts currently exist. Do not infer their presence from
   an older task.
6. Check for a live campaign and monitor before changing trusted source. A source change during a
   campaign changes the source digest and must make resume fail closed.
7. Do not start a paid provider qualification, protected evaluation, or long campaign without the
   authority that applies to that action in the current task.

Current repository facts at the time this handoff was written:

- The worktree is heavily modified and contains many untracked scientific and test files.
- The root `plan.md` is now one authoritative GPU-first, CPU-reversible implementation handoff.
- The complete older planning record is preserved at
  `docs/history/autonomous-agent-plan-history-2026-08-31.md` and is non-authoritative.
- The GPU-first reconstruction is a plan, not an implemented or score-validated result.
- No source implementation was performed while creating this handoff.
- No current live campaign status was asserted while creating this handoff.

## 2. Operative competition contract

The executable KuaiRand-Pure contract—not stale planning prose—has historically required:

- Target: `long_view`.
- Ranking unit: logged impressions grouped within user.
- Metrics: organizer GAUC and organizer nDCG@5.
- Primary metric: arithmetic mean of GAUC and nDCG@5.
- No external training data.
- No final-period/hidden outcome access during research, training, prediction, selection, or
  reporting.
- Exact row identity and ordering through prediction, scoring, replay, and submission.
- A bounded campaign with a hard finalization reserve.
- A qualified official sparse-interaction FM as the immutable fallback.

Revalidate every digest and row count from the current checkout before using historical values.
The most useful historical reference values were:

| Reference | Historical primary |
| --- | ---: |
| Official FM five-seed mean | `0.6015721678733825` |
| Strongest qualified fallback, seed 4 | `0.6020370721817017` |
| Published rounded reference | `0.6016` |

These are comparison references, not proof that a new campaign improved the model.

The historical practical-materiality threshold is a primary delta strictly above `+0.002`.
Search convergence epsilon, competition submission selection, and material scientific
confirmation are different decisions and must not be conflated.

## 3. User intent and action boundaries

These instructions have recurred across tasks and should be treated as durable user intent unless
the user explicitly changes them:

- The whole program is for the competition. Do not create a separate user-selectable
  `competition_mode`; competition constraints belong to the main program.
- “It MUST be the agent flow” means the campaign owns hypotheses, proposals, trial choices,
  training, parent selection, and iterations. Codex may repair infrastructure only when authorized;
  it must not manually steer individual scientific experiments.
- “Halt everything” or “Stop ALL tasks right now” means stop the active campaign process group,
  stop/delete monitors, verify no descendants remain, launch no replacement, and make no further
  campaign changes.
- “Erase execution history only” means remove run history and disposable caches while preserving
  source changes, `.env.local`, `.data`, `.venv`, local configuration, and user work.
- OpenRouter connection was authorized, but that does not automatically authorize unbounded spend,
  a long live campaign, protected-query use, or external submission.
- A terminal request to finish or monitor requires persistence toward the outcome but does not
  expand authority to unrelated mutations or manual scientific steering.
- The user values exact outcomes over optimistic status language. State what was verified, what was
  not, and the decision gate that remains.

## 4. Claim taxonomy: never collapse these

| Claim | Minimum evidence |
| --- | --- |
| Engineering ready | Relevant static, unit, integration, leakage, recovery, and replay gates pass |
| Provider reachable | Authenticated request completed; this says nothing about schema reliability or candidate admission |
| Candidate admitted | Generated/typed experiment passed policy, materiality, source, dependency, and runtime admission |
| Candidate trained | A scientific trial—not a qualification or fold-control launch—completed with durable receipts |
| Validation improved | The exact frozen challenger scored above the exact reference under the declared validation decision |
| Materially confirmed | Adjusted cluster-aware evidence, component guardrails, exact identity, replay, and protected delta satisfy the frozen policy |
| Submission ready | Exact selected artifact was replayed, serialized, organizer-validated, hashed, and closed |
| Performance ready | A genuine autonomous challenger satisfied the declared improvement gate; engineering completion alone is insufficient |
| Hidden-test improvement | Only organizer scoring can establish this; never infer it from local validation |

`RUNNING`, `COMPLETED`, a green smoke test, elapsed time, provider HTTP 200, a positive single seed,
or a finalized fallback bundle is not performance evidence.

## 5. Core architecture that survived the reviews

### One deep competition module

The normal caller should learn one primary operation:

```python
lab = AutonomousExperimentLab.open(run_root, profile=profile_name)
result = lab.compete(options, key=idempotency_key)
```

Normal command:

```bash
kuairand-agent compete
```

The module owns preflight, exact resume, qualification, search, confirmation, protected batch,
selection, finalization, replay, validation, and publication. `status`, `cancel`, and `replay` may be
administrative views/commands, but they must use the same authority and cannot derive another
scientific decision.

### One SQL scientific authority

Use one authoritative SQL database for:

- commands and events;
- campaign state and budgets;
- experiments, trials, attempts, and rejections;
- leases and fencing tokens;
- artifact bindings;
- finalist batches;
- protected-query reservations/outcomes;
- selection and scientific decisions;
- finalization and publication; and
- observer outbox records.

Files, JSON status, progress checkpoints, MLflow runs, Optuna studies, and LangGraph checkpoints are
immutable blobs, projections, or caches. Deleting a projection must not change the authoritative
decision.

### Typed science, trusted execution

The research model may submit an immutable, canonical `ExperimentSpec` describing:

- one falsifiable mechanism and one primary change;
- parent/control identity;
- approved feature views;
- semantic model family;
- bounded parameters/fidelity/seeds;
- matched controls and ablations;
- resource envelope;
- expected metric/slice effects; and
- falsification/novelty identity.

The model must not generate or repair trusted training, evaluator, selector, artifact, finalization,
or replay implementation. Trusted adapters own executable behavior. The older broad
`propose/implement/repair/reflect` source-generation seam contradicts this target architecture.

### Exact artifacts

The same `RankGraphDigest` must identify what was:

- confirmed;
- protected-scored;
- selected;
- replayed;
- finalized; and
- submitted.

Do not score seed members and then finalize an unscored blend. Construct, persist, score, select,
replay, and finalize the exact fused prediction vector.

## 6. Scientific and submission decisions must remain separate

Return two explicit dispositions:

```text
submission_disposition:
    ChallengerSelected | OfficialFMRetained

scientific_disposition:
    MateriallyConfirmed | NotScientificallyConfirmed
```

Recommended sequence:

1. Search only on inner/search data.
2. Select and freeze at most two finalists before any protected result.
3. Freeze backend, trainer, parameters, features, seeds, fusion, dependencies, rows, evaluator,
   statistical family, component margins, and query order.
4. Use seeds to construct one predeclared stable artifact; do not treat seeds as independent
   evaluation samples.
5. Estimate uncertainty over matched users or predeclared user/time blocks.
6. Apply a predeclared multiplicity correction across the frozen finalists.
7. Score each exact frozen protected vector at most once.
8. Never reopen search after a protected result.

A challenger may be competition-selected while remaining scientifically unconfirmed. It must be
reported exactly that way. Tiny positive means must not be called material improvement.

## 7. GPU-first, CPU-reversible compute seam

The current plan requires GPU-first execution without GPU coupling.

The hardware seam is private:

```text
ExperimentSpec
    -> semantic trainer adapter
        -> TrainingExecution interface
            -> GPU adapter, preferred
            -> CPU adapter, qualified
            -> scripted adapter, tests
```

Rules:

- Both GPU and CPU adapters must be real in the first qualified release.
- Public `compete(...)` and scientific `ExperimentSpec` remain hardware-neutral.
- CPU reversion is a profile/configuration change, not a source rewrite.
- The CPU profile imports/initializes no GPU-only dependency.
- GPU and CPU variants share a semantic `ExperimentId` but receive different `TrialId` values when
  backend/precision can change numerical results.
- A retry of one backend gets a new `AttemptId`; switching backend creates another `TrialId`.
- Never switch device silently inside a trial.
- Never retrain a protected-scored GPU artifact on CPU and assume it is the same candidate.
- Exact scoring replay uses frozen prediction bytes and must not require the training device.
- GPU-trained artifacts intended for CPU inference require portability qualification.
- Record device, runtime, dependency, precision, accelerator memory, host memory, and process-tree
  resource receipts.
- The official FM fallback must remain finalizable if the accelerator disappears.

The first ladder should remain narrow:

1. Official sparse-interaction FM.
2. GPU-preferred/CPU-qualified LightGBM LambdaRank over approved causal features.
3. Matched strict-past recency ablation.
4. Optional pointwise tree only when reserve permits.
5. Exact FM/tree within-user rank fusion.

Neural trainers are later, separately qualified profiles—not part of the first six-hour critical
path.

## 8. Historical scientific results worth preserving

These were historical matched diagnostics, not current-live claims. Revalidate data, source, and
evaluator identities before reuse.

| Mechanism | Historical result | Durable decision |
| --- | --- | --- |
| Five-seed FM rank ensemble vs seed 4 | Positive on both train-derived folds; `+0.0004556` Fold B and `+0.0015146` Fold A | Useful variance reduction, below `+0.002` |
| Five-seed ensemble vs best individual seed | `+0.0000591` Fold B; `+0.0006772` Fold A | Positive but submaterial |
| Three-day pseudo-slates | `-0.016343` Fold B; `-0.013606` Fold A | Strongly rejected |
| Three-day pseudo-slates after FM fusion | Negative on both folds | Rejected |
| LambdaRank truncation 5 vs 8 | `-0.0010788` Fold B | Truncation 5 rejected; retain the qualified truncation-near-8 lane |
| Raw watch-progress auxiliary | Approximately zero/negative vs zero-loss control | Rejected |
| Duration-adjusted ECDF watch-progress | Real target underperformed matched shuffle | Failed causal control; apparent gain was regularization/noise |
| Corrected top-5 LambdaLoss vs older custom objective | Positive | Mathematical correction valid |
| Corrected top-5 LambdaLoss vs built-in LambdaRank | Small Fold-B gain, `-0.0019832` Fold A | Fails frozen confirmation |
| Corrected LambdaLoss in common FM portfolio | Negative | Rejected for deployment |
| Deterministic exact rank fusion | Only mechanism with repeatable positive movement | Preserve as first-class exact rank graph; still needs material evidence |

Important interpretation:

- No tested treatment in that research pass justified protected/public promotion.
- Do not spend protected evaluation on a submaterial inner hypothesis.
- Correct theory is not sufficient; it must beat the strongest incumbent under frozen controls.
- Matched shuffle/zero-loss/primary-only controls are mandatory for auxiliary-target mechanisms.
- The exact Python LambdaLoss callback was historically far slower than built-in LightGBM. Profile
  before including it in a broad search.

## 9. What historical campaigns actually established

### Strong engineering evidence

The repository historically demonstrated:

- verified KuaiRand loading and data audit;
- official-FM qualification;
- leakage-safe capabilities and protected scoring;
- temporal-fold evaluation;
- candidate subprocess/resource controls;
- artifact hashing and closure;
- final-period-outcome-free inference;
- organizer-format submission validation; and
- provider-free exact replay of a closed bundle.

These are valuable, but every gate must be rerun against the current dirty checkout.

### Weak performance evidence

The strongest old scripted candidate was `generated-causal-lambdarank-v1`. Its historical matched
three-seed mean primary delta was only about `+0.00004225`, below the official five-seed mean and
far below `+0.002`; its diagnostic user-cluster interval crossed zero. It was engineering/submission
evidence, not a materially confirmed autonomous improvement.

The completed campaign used a local scripted provider and one scientific iteration. It proved the
execution seam, not that a live model autonomously performed repeated research.

### Baseline-only long runs

Several prolonged campaigns finalized `official-fm-fallback-seed-4` with
`baseline_reproduced`. Some had zero accepted proposals, zero candidate admission, zero candidate
training, or zero scientific iterations even though qualification/fold-control launches were
charged.

Do not call a campaign successful because it ran for hours or finalized a bundle. Verify:

- accepted proposals/specs;
- admitted candidates;
- scientific trial launches distinct from qualification/fold controls;
- completed scientific iterations;
- exact finalist identities;
- protected scorer receipts;
- selection decision;
- replay;
- submission validation/hash; and
- zero final-outcome access.

## 10. Candidate-generation failure lessons

One historical live campaign spent many proposal/implementation/repair cycles without admitting a
candidate. The terminal provider `429` was secondary. The primary failure chain was:

1. Proposals expected a forbidden generated filename such as `baseline.py` and sometimes an
   unsupported output such as `submission.csv`.
2. Prompt/request policy did not expose all local materialization rules.
3. Local materialization correctly rejected the package.
4. Pre-materialization repair lost the rejected generated package and repaired from the original
   seed instead.
5. Repair could not preserve the proposed mechanism and often returned no material executable
   change.
6. Root failure fingerprints were not fed back durably, so equivalent proposal families repeated.

Durable lessons:

- One immutable candidate-source policy must generate prompt facts, typed request manifests, and
  local validation.
- Local enforcement remains authoritative even when the model ignores the prompt.
- Repair must receive a bounded digest-verified snapshot of the causative rejected package,
  separately from the trusted parent.
- Persist root and terminal rejection fingerprints immediately.
- Circuit-break repeated root failures and proposal families before another expensive model call.
- An HTTP/provider error after many local rejections is not necessarily the original cause.

## 11. Provider lessons

- HTTP 200 proves connectivity/inference, not strict schema compliance.
- A tiny probe can pass while the campaign’s large prompt/schema is too slow or malformed.
- Markdown-fenced JSON may be human-readable but fail the strict local parser.
- Too-small output ceilings on reasoning models may yield `finish_reason=length` with no visible
  content.
- A fallback may be faster and more schema-compliant than the main route, but must be qualified
  against the exact operation schemas.
- Evaluate providers by cost and elapsed time per admitted runnable candidate, not generic model
  reputation or raw endpoint latency.
- Qualify transport, schema, embedded artifact JSON, Python compilation, source materiality,
  latency, reasoning-token share, and candidate admission before a long run.
- Keep retries bounded and deadline-aware. Avoid nested campaign + framework + SDK retries.
- Account separately for network latency, retry wait, attempts, token use, and failover.
- Provider/model availability and specifications change. Re-verify current model IDs, limits,
  structured-output behavior, and pricing from authoritative sources before use.
- Never copy credentials or `.env.local` contents into logs, tasks, reports, or this file.

## 12. Durability and monitoring lessons

- A manifest or SQL `RUNNING` status can be stale after process death.
- Verify both durable state and the actual launcher/worker process tree.
- Ordinary weak/non-improving experiments are scientific outcomes, not infrastructure defects.
- Provider operation timeouts, missing typed failures, corrupted state, stuck leases, duplicate
  protected queries, and orphaned descendants are infrastructure defects.
- Do not edit trusted source during a live campaign; source identity must remain stable.
- Do not launch against a moving tree while implementation workers are still editing.
- Candidate/trial failure must not count as completed non-improving science for convergence.
- `TrialId` identifies immutable science; `AttemptId` identifies an infrastructure execution.
- Suggestions/parameters commit before training. Retry must not ask the optimizer again.
- Every lease needs a monotonically increasing fencing token; stale workers cannot commit.
- Persist every rejection and iteration transition. Do not rely on a coarse end-of-phase checkpoint.
- Protected evaluation must use an idempotent identity. If dispatch may have succeeded but the
  receipt is missing, enter `OUTCOME_UNKNOWN`; never spend a blind replacement query.
- Cancellation/timeout/OOM must terminate and verify the full descendant tree.
- Monitoring must observe infrastructure and durable evidence, not steer proposals or selection.

## 13. LangChain, LangGraph, Optuna, and MLflow

Do not make these frameworks the scientific authority.

- LangChain may later be a proposal-only provider adapter after schema/retry/deadline/transcript
  parity tests.
- Keep trusted prompts/schemas, local validation, source policy, budgets, authority, evaluator,
  selector, finalizer, and replay outside LangChain.
- Disable or account for framework/SDK retries so they cannot exceed the campaign policy.
- LangGraph checkpoint replay is not exact experiment or scoring replay and can re-execute model
  nodes. Do not adopt it without a concrete unmet requirement.
- Optuna may later sit behind a planner seam. Canonical trial history and committed suggestions
  remain authoritative in the lab.
- MLflow may be an outbox-fed evidence projection. Registry aliases must not decide the champion.
- Deleting any optional framework state must not alter selection, query count, replay, or bundle.

The deterministic six-hour MVP should use the project-owned static planner and no critical-path
LangChain, LangGraph, Optuna, MLflow, or neural dependency.

## 14. Privacy and collaboration boundaries

Never publish or copy the working folder wholesale.

Keep local:

- `.env.local` and all credentials;
- `.data/` and dataset-derived behavior records;
- `runs/`, SQLite/WAL/SHM files, predictions, checkpoints, logs, and local provenance;
- `.venv/`, `.uv-cache/`, `.cache/`, and tool caches;
- machine-specific paths and `.DS_Store`.

`.gitignore` protects ordinary Git staging; it does not protect ZIP, Drive, AirDrop, or direct
folder uploads. Use a release allowlist and audit staged files before publication. If a credential
may have left the machine, rotate it rather than relying on deletion.

Do not preserve API keys, raw provider transcripts, protected outcomes, or dataset rows in a task
handoff or memory document.

## 15. Useful repository map

Reinspect these paths because the checkout is dirty and may have changed:

| Area | Typical responsibility |
| --- | --- |
| `kuairand-starter-kit/` | Immutable organizer reference/evaluator |
| `src/kuairand_agent/contract.py` and `challenge_contract.py` | Benchmark and competition identities/policies |
| `src/kuairand_agent/data/` and `data_v2/` | Canonical data, capabilities, strict-past views, groups, sequences |
| `src/kuairand_agent/baselines/` | Official FM qualification and fallback |
| `src/kuairand_agent/research/` | Safe context, typed experiments, proposal provider |
| `src/kuairand_agent/execution/` | Candidate/compute execution, process supervision, receipts |
| `src/kuairand_agent/campaign/` | Authority, budgets, lifecycle, evidence, search, selection |
| `src/kuairand_agent/candidates/` | Trusted model families, grouping, fusion, diagnostics |
| `src/kuairand_agent/scoring/` | Protected organizer-compatible scoring |
| `src/kuairand_agent/finalization/` | Rank graph, replay, final inference, bundle and submission |
| `candidate_seed/` | Candidate-owned starting surface; do not hide a hand-authored winner here |
| `configs/` | Locked profiles and resource/search policies |
| `tests/` | Unit, integration, leakage, replay, fault, full-data, and performance gates |
| `docs/research/` | Historical scientific analyses; useful evidence, not current authority |
| `plan.md` | Current GPU-first reconstruction plan followed by preserved history |

## 16. Related task review index

The handoff reviewed the actual recent histories of these substantive tasks. Titles/summaries were
used only to locate histories; durable lessons came from the task content.

| Task | Durable lesson |
| --- | --- |
| `01a043be-0e1c-74f2-b96f-623881fb9504` — `Summarize autonomous ML agents` | Start from the executable long_view/GAUC/nDCG@5 contract; GPU is permitted but CPU portability remains necessary |
| `01a04427-9d5f-7763-96ba-fda9f6f091bb` — `Implement KuaiRand ML Research Agent` | Strong engineering/submission evidence existed, but scripted one-iteration evidence and `+0.00004225` matched gain were not Performance Ready |
| `01a0463c-8354-7e21-8806-7c61857c7400` — `Evaluate project aim fulfillment` | An automated predetermined pipeline is not a repeated autonomous research agent; convergence and material improvement remained unfulfilled |
| `01a0471a-6948-7290-b095-9a0145fa1fe8` — `Scan repo for personal data` | Never share credentials, data, run history, caches, or machine paths; use a curated source allowlist |
| `01a048cf-1955-7df0-9678-8a05eb319f10` — `Verify model and run campaign` | Leakage safety is architectural: phase-scoped capabilities, protected scorer, strict-past features, and metamorphic tests |
| `01a048da-7c1f-73a0-8fcc-6e2f4b129beb` — `Run evaluation against baseline` | A live process generating rejected branches is not scientific progress; monitor without steering and verify admission/iterations |
| `01a04cb7-d900-7f11-9ad0-29790c3148f1` — `Check final report selection status` | Baseline-only final reports can charge qualification/control launches while training zero generated candidates |
| `01a04ce1-530d-7e42-a318-5df29b473fb6` — `Assess LangChain conversion` | LangChain can simplify provider integration, but cannot replace trusted campaign authority, leakage gates, exact evidence, or replay |
| `01a04cde-b05e-7ca3-b5be-5a4795228725` — `Inspect run provenance and lineage` | Verify platform/source provenance and actual lineage artifacts; reports may omit or flatten real iteration history |
| `01a04d6c-4368-7f70-a106-4f113a63c9b4` — `Complete stable 6-hour full run (2)` | Direct provider probes separated connectivity from schema/prompt latency; HTTP 200 alone was insufficient |
| `01a04c44-f7cb-7a70-8b4b-c2c5bbe5286c` — `Complete stable 6-hour full run` | Qualify model/provider combinations by schema-valid admitted candidates before another expensive campaign |
| `01a0503c-ce70-7fe0-993e-0c140a97838c` — `Investigate 6h run thread drift` | Seven hours of repairs/restarts improved infrastructure but produced no material model gain; many final bundles retained FM |
| `01a050b2-bc8d-7633-9ac6-6b6c5794799c` — `Map Python codebase architecture` | The trusted flow is CLI/contract -> data/baseline -> research/execution -> campaign/scoring -> finalization/replay; generated code must not reach protected labels |
| `01a05044-d90c-7a62-a57c-f3be1657cbc9` — the long score-stagnation research task | The plateau was systemic; exact fusion helped submaterially while pseudo-slates, truncation 5, watch-progress auxiliaries, and custom top-5 loss failed controls/confirmation |
| `01a0527b-9f50-75a3-945d-bc83a14b37c7` — `Add competition-aligned execution (2)` | Competition policy belongs in the main program; separate submission best, research parent, and specialist archive—but later reviews rejected broad 50-query/50-iteration defaults for the six-hour MVP |
| `01a052e1-1b49-7e91-b1a9-22052ebe5fc2` — `Run staged model experiments` | This task was interrupted before meaningful execution; intention and planned subagents are not implementation evidence |
| `01a05305-7532-7dd3-b1da-4be99705fc66` — `Implement staged ensemble search` | Preserve dirty work, split disjoint implementation ownership, qualify the baseline first, and never launch against moving source; the requested one-hour run was not completed in the visible history |

The presentation and standalone image-generation tasks were excluded because they did not contain
durable model/campaign implementation lessons. No related archived TiktokTechJam task appeared in
the archived-task listing at review time.

## 17. First-session checklist for the next agent

### Read-only orientation

1. Inspect `git status`, diff stat, untracked files, branch, remotes, and active processes.
2. Read the new section of `plan.md`, then this file, then the executable contract/evaluator.
3. Identify which planned modules already exist and whether they match the current plan or an older
   architecture.
4. Check current qualification/data/provider/GPU availability without printing secrets.
5. Locate authoritative tests and current failures before changing implementation.

### Re-establish trusted gates

1. Reproduce or verify the exact official FM fallback.
2. Verify final-outcome capability exclusion and strict-past metamorphic tests.
3. Verify canonical row alignment and exact scoring replay.
4. Reproduce known current bugs before fixing them, especially recovery/rejection persistence and
   selection semantics.
5. Verify process-tree cleanup and resource receipts on the actual target hardware.

### Implement in this order

1. Hardware-neutral contracts and exact identities.
2. Real GPU, CPU, and scripted compute adapters behind one private interface.
3. One SQL authority with idempotency, attempts, fencing, rejections, outbox, and protected
   outcome reconciliation.
4. One-command offline FM competition path through finalization/replay/validation.
5. Deterministic FM/LightGBM/strict-past/fusion ladder.
6. Frozen finalist batch, cluster-aware confirmation, component guardrails, and dual dispositions.
7. Full GPU-first and CPU-reversion acceptance runs.
8. Optional provider/LangChain/Optuna/MLflow/neural adapters only after the deterministic core
   qualifies.

### Before any long live run

- Freeze the source digest and stop all implementation edits.
- Confirm no other campaign/monitor is active.
- Pass provider schema/admission qualification if a provider is enabled.
- Confirm exact data/evaluator/profile identities.
- Confirm baseline fallback and finalization reserve.
- Confirm project-wide protected-query accounting.
- Confirm the complete batch/no-feedback policy.
- Confirm cancellation and full descendant cleanup.
- Record user authority for provider spend and campaign duration.

## 18. Anti-patterns to reject

- A separate competition mode or alternate competition pipeline.
- “Guaranteed future score improvement.”
- Any-positive-mean being called material promotion.
- Three seeds being treated as three evaluation samples.
- Protected scoring of individual seed members before fusing them.
- Search or fusion changes after a protected result.
- Generated implementation/repair source controlling trusted training or scoring.
- Multiple writable scientific authorities.
- Blind retry of an uncertain protected query.
- Counting failed execution as scientific convergence.
- Adding frameworks before the baseline, authority, compute seam, and replay are qualified.
- Launching a six-hour campaign because unit tests pass while candidate admission is unqualified.
- Editing trusted source while a campaign is live.
- Reporting a fallback bundle as a challenger win.
- Treating an old run directory, model recommendation, price, or provider status as current without
  verification.
- Resetting or deleting the dirty worktree to make implementation easier.

## 19. Open decisions that require current verification

- Actual GPU type and availability in the target environment.
- GPU runtime/library choice and CPU inference portability for each trainer.
- Which dirty/untracked changes belong to the final reconstruction versus superseded experiments.
- Whether the current source already contains two conflicting authority stores or coarse progress
  recovery paths.
- Current protected-query budget and whether old reservations still exist after history cleanup.
- Current provider models, schema behavior, latency, pricing, and credentials.
- Current data and qualification artifact identities.
- Whether the CPU profile can meet six hours with the same trial fidelity or needs a frozen reduced
  profile.
- Exact cluster unit, component margins, confidence construction, and family correction frozen in
  the next version of the competition contract.

Resolve these with current evidence. Do not inherit answers merely because an older task once
reported them.

## 20. Definition of a successful handoff continuation

The next agent has used this handoff correctly when it can state, with evidence:

- which current files and changes it preserved;
- which plan phase it is implementing;
- which executable contract and identities are frozen;
- whether GPU and CPU are real qualified adapters;
- whether one SQL authority owns the campaign;
- whether a provider is optional and proposal-only;
- whether the official FM can always finalize;
- whether a candidate was genuinely admitted, trained, confirmed, and exact-vector scored;
- whether the result is competition-selected, materially confirmed, both, or neither;
- whether replay/submission validation passed;
- how many protected queries were consumed;
- whether final-period outcome access remained zero; and
- what remains unverified.

Anything less should be reported as partial progress, not completion or guaranteed improvement.
