# Archived planning history: autonomous KuaiRand competition laboratory

Archive date: 2026-08-31
Purpose: preserve the complete pre-handoff planning record. This file is historical context,
not the active implementation plan. The authoritative handoff is `/plan.md` at the repository root.

---

# GPU-first autonomous competition laboratory reconstruction plan

Status: proposed target architecture; implementation has not started under this plan
Date: 2026-08-31
Scope: recreate the main KuaiRand competition program as a GPU-first, CPU-reversible,
fully autonomous experiment laboratory while preserving scientific validity, exact replay,
fallback safety, and the six-hour competition contract

This section is the current reconstruction plan. The earlier remediation and run-history plan is
retained below for provenance; it must not be interpreted as authorizing a live campaign or as
evidence that the reconstruction has already been implemented or improved the score.

## 1. Outcome and honest guarantee

Build one main competition program that:

- starts with a qualified GPU execution path;
- can run on CPU through a configuration/profile change rather than a source rewrite;
- keeps model semantics, experiment specifications, feature views, evaluator behavior, selection,
  replay, and finalization independent from the hardware implementation;
- autonomously searches, trains, confirms, evaluates, selects, replays, and finalizes;
- always retains a qualified official-FM fallback;
- never labels weak, incomplete, or corrupted evidence as an improvement; and
- never exposes final-period outcomes or protected results to proposal generation or further
  search.

No architecture can guarantee that every future experiment improves an unseen competition score.
This program will instead guarantee monotonic certified selection: it cannot replace the certified
incumbent unless the frozen candidate satisfies the declared evidence, identity, resource, replay,
and protected-evaluation gates. It may separately select a validation-leading competition artifact
without mislabeling it as statistically confirmed.

## 2. Design decision: GPU-first, not GPU-coupled

GPU-first means the default competition profile prefers a qualified accelerator adapter. It does
not mean that CUDA, Metal, device names, mixed precision, GPU memory management, or framework
objects appear in the public `compete` interface or in the scientific `ExperimentSpec`.

The hardware seam sits below semantic trainer selection and above concrete framework execution:

```text
kuairand-agent compete
        |
        v
AutonomousExperimentLab.compete(options)
        |
        v
private durable campaign kernel
        |
        +-- ExperimentSpec and trusted model-family adapter
        |         |
        |         v
        |   TrainingExecution interface
        |         |
        |         +-- GPU training adapter    <- preferred profile
        |         +-- CPU training adapter    <- qualified fallback/profile
        |
        +-- frozen evaluator, selector, replay, and finalizer
```

This seam provides leverage to the caller and locality to maintainers. Switching hardware changes
one adapter/profile selection, while all scientific control remains in the same deep module.

Two adapters must exist from the first qualified release. A GPU-only implementation with a
hypothetical future CPU interface does not satisfy this plan.

## 3. Public module and interface

The public interface remains hardware-neutral:

```python
lab = AutonomousExperimentLab.open(
    run_root=Path("runs"),
    profile="kuairand-competition-gpu-first-v1",
)

result = lab.compete(
    CompetitionOptions.from_config(Path("configs/full-pure.toml")),
    key="competition-final-attempt-01",
)
```

Normal command:

```bash
kuairand-agent compete
```

CPU reversion must require no source edit:

```bash
kuairand-agent compete \
  --profile kuairand-competition-cpu-v1 \
  --resume auto
```

The caller does not choose a concrete CUDA, Metal, LightGBM, or PyTorch class. A locked profile
declares a compute policy; the laboratory qualifies and resolves the matching adapter.

`compete` owns:

```text
preflight
-> exact create-or-resume
-> backend qualification
-> official-FM qualification
-> immutable feature construction
-> deterministic inner search
-> frozen confirmation
-> complete protected-batch freeze
-> protected evaluation
-> submission and scientific decisions
-> fallback-safe finalization
-> exact scoring replay
-> submission validation
-> atomic publication
```

Administrative `status`, `cancel`, and `replay` operations must delegate to the same authoritative
module. They cannot independently select, promote, or reconstruct another scientific outcome.

## 4. Private compute interface

Use one internal `TrainingExecution` interface:

```python
class TrainingExecution(Protocol):
    def descriptor(self) -> ComputeDescriptor:
        ...

    def qualify(
        self,
        request: QualificationRequest,
    ) -> QualificationReceipt:
        ...

    def train(
        self,
        request: FrozenTrainingRequest,
        lease: FencedLease,
    ) -> TrainingReceipt:
        ...

    def predict(
        self,
        request: FrozenPredictionRequest,
        lease: FencedLease,
    ) -> PredictionReceipt:
        ...
```

The interface includes these invariants:

- Requests contain only canonical scientific inputs and immutable artifact references.
- The adapter cannot access hidden/final-period outcomes or protected scores.
- The adapter returns immutable model, prediction, resource, environment, and provenance receipts.
- The adapter cannot perform selection, protected evaluation, promotion, or publication.
- Every output is staged, hashed, verified, and bound by the authority after the adapter returns.
- The adapter must expose supported model families, training devices, inference targets,
  determinism grade, precision modes, dependency lock, and resource limits.
- Device-native framework objects never cross the seam.

Concrete adapters in the first release:

```text
GpuTrainingExecution
CpuTrainingExecution
ScriptedTrainingExecution  # deterministic tests only
```

Model-family adapters remain separate from compute adapters. For example, `LambdaRankTrainer`
defines the semantic model and parameter contract; `GpuTrainingExecution` or
`CpuTrainingExecution` performs the qualified execution. If a model family has meaningfully
different GPU and CPU implementations, expose two qualified trainer implementations behind the
same semantic family rather than pretending they are numerically identical.

## 5. Compute profiles

### GPU-first competition profile

```toml
[compute]
preference = ["gpu", "cpu"]
training_backend = "gpu_preferred"
inference_backend = "cpu_compatible"
allow_pre_trial_fallback = true
allow_mid_trial_device_switch = false
require_cpu_export_qualification = true
max_active_candidates = 1
cpu_threads = 4
gpu_memory_fraction = 0.80
process_tree_memory_mb = 14336
candidate_host_memory_mb = 12288
```

### CPU profile

```toml
[compute]
preference = ["cpu"]
training_backend = "cpu_required"
inference_backend = "cpu_required"
allow_pre_trial_fallback = false
allow_mid_trial_device_switch = false
max_active_candidates = 1
cpu_threads = 4
process_tree_memory_mb = 14336
candidate_host_memory_mb = 12288
```

The exact GPU family, driver/runtime requirements, precision, and memory limits belong to a
qualified deployment-specific profile. The domain contract uses semantic capabilities such as
`accelerator`, `cpu`, `deterministic_training`, and `cpu_portable_artifact`; it does not hard-code a
vendor into the laboratory interface.

## 6. Backend identity and fallback rules

GPU and CPU execution can produce numerically different models even when the semantic parameters
match. Therefore hardware reversion must be convenient but never scientifically invisible.

Identity rules:

- `ExperimentId` identifies the scientific mechanism independent of hardware.
- `TrialId` includes the qualified trainer descriptor, compute adapter descriptor, precision mode,
  dependency lock, parameters, folds, fidelity, and seed policy.
- `AttemptId` identifies an infrastructure retry of exactly one `TrialId`.
- GPU and CPU variants of the same experiment share an `ExperimentId` but have different
  `TrialId` values unless qualification proves a stronger equivalence contract.
- `RankGraphDigest` and `PredictionId` always bind the actual backend-derived artifacts and exact
  prediction bytes.

Fallback rules:

1. If GPU qualification fails before a trial is admitted, the profile may admit the corresponding
   CPU trial variant without consuming a scientific result.
2. If a GPU attempt fails after its trial is admitted, retrying the same GPU trial creates a new
   `AttemptId` under the same `TrialId`.
3. Switching that failed trial to CPU creates a new CPU `TrialId` under the same `ExperimentId`.
   It is not recorded as an infrastructure retry or as the same scientific observation.
4. Never switch GPU to CPU silently inside a running attempt.
5. Never retrain a protected-scored GPU candidate on CPU and assume it remains the same candidate.
   A changed rank graph must be requalified, reconfirmed, and—if eligible—protected-scored under
   its own identity.
6. Exact scoring replay uses frozen prediction bytes and the frozen evaluator. It must not require
   the original training device.
7. If deployment or final inference must run on CPU, the GPU-trained artifact is eligible only
   after a CPU-inference portability qualification proves the declared prediction tolerance and
   the same decision on frozen fixtures.

## 7. Qualification contract

Before an adapter can participate, it must produce a durable `QualificationReceipt` proving:

- the expected backend and device were actually used;
- dependency, driver/runtime, trainer, and precision identities are complete;
- row alignment, feature schema, grouping, and target semantics match the contract;
- training and prediction respect the deadline and process-tree resource envelope;
- repeated same-backend runs satisfy the declared determinism grade;
- model serialization and loading succeed;
- GPU-trained artifacts can perform qualified CPU inference when the profile requires it;
- prediction drift remains within a predeclared tolerance and does not change the fixture-level
  selection decision;
- final-period outcomes are inaccessible; and
- all descendants terminate on success, timeout, cancellation, memory breach, and adapter failure.

Qualification does not claim GPU and CPU models are byte-identical. It states exactly what is
portable: configuration compatibility, artifact loadability, prediction tolerance, decision
parity, or exactness where actually demonstrated.

## 8. GPU-first model ladder

The initial GPU-first ladder remains deliberately narrow:

1. Qualified official sparse-interaction FM fallback.
2. GPU-preferred LightGBM LambdaRank on approved causal features.
3. Matched strict-past recency ablation using the same model family and search budget.
4. Optional qualified pointwise tree challenger when measured reserve permits.
5. Exact FM/tree within-user rank fusion over frozen cross-fitted predictions.

The CPU profile runs the same semantic ladder through CPU-qualified trial variants. A GPU-only
neural ladder is not part of the first release; it would make CPU reversion incomplete and consume
the six-hour budget before the shared scientific and recovery contracts are qualified.

Neural trainers may be added later only when both of these are true:

- the trainer satisfies the same `TrainingExecution` interface and authority/replay gates; and
- a CPU adapter or an explicitly documented CPU-degraded profile exists, with measured time and
  memory consequences.

## 9. Search and decision policy

Hardware must not change selection semantics.

### Inner screening

- Use only inner/search evidence.
- Predeclare up to eight deterministic trials under the active compute profile.
- Require positive primary delta and frozen component non-inferiority margins.
- Record GPU/CPU resource receipts but do not reward a backend merely for being faster.
- Select at most two candidates before any protected result is available.

### Frozen confirmation

- Freeze model, parameters, backend descriptor, precision, features, seeds, fusion, rows, folds,
  dependencies, and rank graph.
- Treat seeds as producers of one predeclared ensemble/fused vector, not independent samples.
- Use matched users or predeclared user/time blocks for uncertainty.
- Apply a one-sided cluster-aware confidence procedure and Holm adjustment across the frozen
  finalists.
- Require primary point delta of at least `+0.002` for protected-batch eligibility.
- Freeze recommended component non-inferiority margins at `-0.001` before the run.

### Protected competition evaluation

- Transactionally freeze zero, one, or two exact prediction identities and their query order.
- Score each exact fused vector at most once.
- Never protected-score individual seed members.
- Never reopen search after any protected result.
- Select the highest protected primary score that remains strictly above FM and satisfies both
  component guardrails, exact replay, resource, and policy gates.
- Exact ties retain FM or the earlier certified incumbent.

Return two separate dispositions:

```text
submission_disposition:
    ChallengerSelected | OfficialFMRetained

scientific_disposition:
    MateriallyConfirmed | NotScientificallyConfirmed
```

`MateriallyConfirmed` additionally requires the adjusted cluster lower bound above `+0.002` and a
protected primary delta of at least `+0.002`. A competition-leading but statistically weak result
must be labeled `ChallengerSelected / NotScientificallyConfirmed`.

## 10. Authoritative state and recovery

Use one SQL authority for:

- commands and event sequence;
- campaign lifecycle;
- experiments, trials, attempts, and rejections;
- compute and trainer qualification receipts;
- work leases and fencing tokens;
- resource budgets and receipts;
- artifact bindings;
- finalist batches;
- protected-query reservations and outcomes;
- submission and scientific decisions;
- finalization and bundle publication; and
- transactional observer outbox entries.

Files, status JSON, model files, feature caches, MLflow runs, Optuna studies, and LangGraph
checkpoints are immutable blobs, projections, or caches. They cannot be independently writable
scientific authorities.

Protected evaluation states:

```text
RESERVED -> DISPATCHED -> SUCCEEDED
                       -> FAILED
                       -> OUTCOME_UNKNOWN
```

A crash after dispatch retries only through the same protected-evaluation identity and external
idempotency key. If the scorer cannot reconcile whether the query was consumed, enter
`OUTCOME_UNKNOWN`; never spend a replacement query. Finalize the FM fallback unless the original
receipt can be recovered.

## 11. Six-hour GPU-first budget

The GPU profile may finish individual trials faster, but it must not use that speed to borrow from
the finalization reserve or expand protected-query use.

| Time | Work |
| --- | --- |
| 0–45 minutes | Preflight, GPU and CPU qualification/reuse, FM replay, immutable feature cache |
| 45–195 minutes | Up to eight deterministic GPU-preferred inner screens |
| 195–255 minutes | Frozen confirmation for at most two candidates; use CPU variant only under explicit fallback identity |
| 255–300 minutes | Final seed production, exact fusion, batch freeze, at most two protected vectors |
| 300–360 minutes | Hard provider-free finalization, exact replay, submission validation, hashes, and atomic publication |

No scientific work launches after minute 300. The budget planner must stop earlier whenever the
measured worst-case remaining confirmation, protected evaluation, cleanup, replay, and finalization
would enter the reserve.

GPU acceleration may be used to increase fidelity within the eight predeclared screens only after
the reference acceptance run proves the complete six-hour path. It must not silently increase the
statistical family, number of protected finalists, or adaptive exposure to validation data.

## 12. Dependencies and packaging

Keep the core laboratory hardware-neutral:

```text
core:
    SQL authority, contracts, identities, evaluator, selector, replay, artifacts,
    resource policy, deterministic planner, FM fallback

competition-tree-cpu:
    CPU-qualified tree trainer and predictor

competition-tree-gpu:
    GPU-qualified tree trainer and predictor

proposal-openrouter:
    optional typed proposal adapter

proposal-langchain:
    optional LangChain typed proposal adapter

observer-mlflow:
    optional post-commit evidence projection

optimizer-optuna:
    optional later planner adapter
```

The CPU profile must not import or initialize GPU-only dependencies. The GPU profile may depend on
the CPU package when CPU inference or fallback qualification is required.

LangChain is not part of the compute module or the MVP authority. It may later wrap a proposal-only
provider after schema, timeout, retry, transcript, usage, and failure-parity tests pass. LangGraph
is not scheduled for adoption; reconsider it only for a concrete workflow requirement not met by
the SQL command kernel.

## 13. Implementation phases and acceptance gates

### Phase 0: freeze benchmark and compute contracts

Implement or preserve:

- immutable benchmark, evaluator, split, row-order, and feature identities;
- hardware-neutral `ExperimentSpec`;
- `ComputeDescriptor`, qualification receipt, resource receipt, and portability policy;
- official-FM exact replay and fallback;
- exact scoring replay over frozen prediction bytes.

Gate:

- Future-label changes cannot alter earlier features or predictions.
- Outcome columns can be absent during prediction.
- FM replay is exact and provider-free.
- Changing the compute descriptor changes `TrialId` but not the semantic `ExperimentId`.

### Phase 1: build the private compute seam with two real adapters

Implement:

- `TrainingExecution` interface;
- scripted deterministic test adapter;
- CPU adapter;
- GPU adapter;
- backend resolution from a locked profile;
- GPU-training/CPU-inference portability qualification;
- process-tree and device-memory receipts.

Gate:

- The same semantic experiment can be instantiated as explicit GPU and CPU trial variants.
- CPU-only mode imports and initializes no GPU runtime.
- GPU absence before admission selects the CPU variant when policy allows.
- GPU failure after admission never masquerades as a CPU retry.
- Mid-trial device switching is impossible.
- Timeout, OOM, cancellation, and adapter crash leave no descendants.

### Phase 2: build the single authority and deep competition module

Implement:

- `AutonomousExperimentLab.open(...).compete(...)`;
- one SQL command/event authority;
- exact command idempotency;
- experiment/trial/attempt identities;
- fencing leases;
- immediate rejection persistence;
- artifact staging and binding;
- transactional outbox;
- protected-evaluation reservations and `OUTCOME_UNKNOWN`;
- automatic exact resume.

Gate:

- Kill after every durable transition and resume to the same evidence root.
- Duplicate commands, trials, metrics, and protected queries are impossible.
- Stale workers cannot commit.
- Deleting projections does not change the decision.
- Ambiguous resume fails closed.

### Phase 3: deliver CPU and GPU fallback campaigns

Implement:

- `kuairand-agent compete`;
- GPU-first and CPU-only profiles;
- preflight and qualification reuse;
- official-FM finalization through both profiles;
- replay, bundle generation, and organizer validation.

Gate:

- Both profiles produce a verified FM bundle with no provider.
- Switching profiles requires configuration only.
- Provider outage cannot block fallback finalization.
- Interruption at every lifecycle stage resumes exactly.

### Phase 4: add the trusted score ladder

Implement:

- GPU- and CPU-qualified LambdaRank execution;
- static eight-trial planner;
- strict-past recency ablation;
- optional pointwise adapter;
- cross-fitted prediction store;
- deterministic FM/tree rank fusion;
- complete-chain deadline admission;
- lazy screen evidence and exact finalist replay.

Gate:

- Same-profile repeated runs reproduce trial ordering and declared determinism behavior.
- CPU and GPU results are never conflated.
- Feature views are built once per immutable identity.
- The CPU profile remains within six hours at its declared reduced fidelity or trial ceiling.
- The GPU profile reaches finalization reserve on time without expanding protected exposure.

### Phase 5: add frozen confirmation and dual decisions

Implement:

- complete finalist-batch freeze;
- cluster-aware uncertainty;
- Holm adjustment;
- component non-inferiority;
- exact seed-vector fusion;
- protected vector idempotency;
- separate competition and scientific dispositions;
- no-feedback enforcement.

Gate:

- Tiny positive means cannot be called material improvements.
- A component regression blocks material confirmation.
- Null simulations satisfy the declared false-promotion rate.
- All finalists and query order are frozen before protected evaluation.
- No protected result can alter search.
- The selected vector is byte-identical to the scored, replayed, and submitted vector.

### Phase 6: full acceptance runs

Run one uncached GPU-first reference acceptance and one CPU profile acceptance.

GPU-first gate:

- Complete within six hours.
- Finalization starts by minute 300.
- One active candidate at a time.
- Host and device memory remain within the qualified profile.
- At most two exact protected vectors are scored.
- Final bundle and replay validate exactly.

CPU-reversion gate:

- Requires no source change from the accepted GPU build.
- Uses only the CPU profile and CPU dependency set.
- Completes within its locked budget and reduced search/fidelity policy.
- Produces the same evidence schema, decision taxonomy, finalization contract, and organizer-valid
  bundle.
- Does not claim numerical equality with the GPU campaign unless qualification proves it.

Fault gate for both profiles:

- Provider outage, device/backend absence, trial timeout, OOM, memory breach, weak candidates,
  no-improvement, cancellation, and crash all retain or recover the valid FM path.
- No descendant remains after cancellation or terminal failure.
- Final-period outcome access remains zero.

### Phase 7: optional intelligence and observability

Only after both compute profiles pass:

1. Add an optional OpenRouter proposal adapter.
2. Add LangChain only as an interchangeable proposal adapter.
3. Add Optuna only if measured throughput justifies adaptive planning.
4. Add MLflow only as an outbox-fed projection.
5. Add neural trainers as separately qualified GPU and CPU/degraded profiles.

For every optional adapter, disabling or deleting it must not change authoritative state,
selection, protected-query accounting, replay, or finalization.

## 14. Required source changes

1. Add the deep `AutonomousExperimentLab` module and route `src/kuairand_agent/cli.py` through
   `compete(...)`.
2. Add the private compute interface, descriptors, qualification receipts, GPU adapter, CPU
   adapter, and scripted test adapter under `src/kuairand_agent/execution/`.
3. Keep semantic model-family adapters separate from hardware execution; qualify FM and
   LambdaRank for both profiles.
4. Expand `src/kuairand_agent/challenge_contract.py` and configuration with compute policy,
   portability, statistical family, deployment seed, exact query unit, resource, and reserve
   identities.
5. Consolidate commands, events, trials, attempts, leases, rejections, protected evaluations,
   decisions, and finalization into one SQL authority beginning in
   `src/kuairand_agent/campaign/store.py`.
6. Replace generated implement/repair authority in the research interface with proposal-only typed
   `ExperimentSpec` generation.
7. Replace any-positive-mean selection in `src/kuairand_agent/campaign/selector.py` with frozen
   finalist batches, dual dispositions, materiality, component, identity, and replay gates.
8. Decompose coarse recovery in `src/kuairand_agent/campaign/full_campaign_runtime.py` behind the
   lab kernel and persist every accepted proposal, rejection, trial, result, and decision event.
9. Bind the same `RankGraphDigest` through confirmation, protected scoring, replay, finalization,
   and submission.
10. Add GPU-first and CPU-only qualified configurations; never let either profile modify the
    benchmark, evaluator, protected-query ceiling, final-label policy, or submission contract.

## 15. Definition of done

The reconstruction is complete only when:

- `kuairand-agent compete` owns the entire competition lifecycle;
- GPU-first and CPU-only profiles both pass through the same deep module;
- reverting to CPU requires configuration only;
- GPU and CPU scientific identities remain explicit and cannot be conflated;
- the LLM has proposal-only authority and the default run requires no provider;
- one SQL authority can reconstruct the complete decision after any crash;
- the official FM is always qualified, replayable, and finalizable;
- no protected result can influence another experiment;
- the exact selected prediction vector is scored, replayed, and submitted;
- a valid no-gain campaign returns `OfficialFMRetained` successfully;
- a competition-selected but statistically weak challenger is labeled unconfirmed;
- only the materiality and uncertainty gates can produce `MateriallyConfirmed`;
- both compute profiles satisfy their time, memory, cleanup, replay, and organizer-validation
  acceptance suites; and
- optional LangChain, OpenRouter, Optuna, MLflow, neural, or remote-execution adapters can be
  disabled without changing the authoritative outcome.

---

# Remediation plan: Fulfil the Autonomous ML Research Agent aim

Status: previous autonomous run admitted no generated candidate; corrective P0/P1 code is
implemented and verified locally; live P2 gates remain pending explicit authorization
Date: 2026-08-29
Required benchmark: KuaiRand-Pure
Current verdict: the production code launched an autonomous live-provider campaign, but the run
never admitted generated candidate source to smoke or training. It retained the official FM
fallback with validation primary `0.6020370721817017`; therefore it did not beat the baseline and
did not test any agent-generated scientific improvement. The run's immediate blocker was an
inconsistent candidate-generation contract across prompts, typed requests, materialization, and
repair—not an evaluation result showing that the proposed model ideas were worse. The local
corrective code now addresses that contract failure; a live admission canary has not yet verified
the correction against a paid provider.

## Previous-run failure investigation and corrective plan (2026-08-29)

This section is the authoritative remediation plan for `runs/full-pure-20260829-05`. It supersedes
older statements below that the live repair path had not yet been exercised. No guardrail,
benchmark feature, provider capability, resource limit, evaluator protection, or recovery feature
may be removed to make the run pass. The repair must make the agent-facing contract agree with the
existing trusted enforcement.

### A. What actually failed

The final HTTP `429` was real, but it was the last and secondary failure. The campaign had already
spent 23 complete propose/implement/repair cycles without admitting one candidate to execution.
The immutable diagnostic artifact records 74 provider-attempt transcripts: 24 proposals,
26 implementation attempts, and 24 repair attempts. The campaign then finalized safely with the
baseline after `12,531.681` seconds of wall time, below the six-hour cap.

The deterministic failure chain was:

1. Every accepted proposal expected a generated file named `baseline.py`. The first proposal also
   expected `submission.csv`, although generated source allows only `.py`, `.json`, and `.md`.
2. The proposal request and proposal system prompt contained no generated-file policy. The
   implementation request exposed allowed suffixes, file count, and byte limits, but omitted the
   forbidden basenames enforced by materialization, including `baseline.py`.
3. All 23 accepted implementation packages therefore reached the local materializer before being
   rejected with `reserved candidate filename is forbidden: 'baseline.py'`. The filename guardrail
   worked correctly and must remain in force.
4. A failure before materialization leaves no `attempted_candidate`. The repair controller then
   supplied the original FM seed as `failed_child`; all 23 repair requests contained exactly the
   same seed digest and only `README.md`, `candidate.py`, and `config.json`. They did not contain
   the rejected implementation package in which the agent had written its proposed algorithm.
5. The repair system prompt said to preserve the principal scientific claim, but the request made
   that impossible at source level. Offline replay of all 23 accepted repairs found zero executable
   algorithm changes: 18 left `candidate.py` byte-identical, while the other five changed only
   module-docstring whitespace, which the material-change guard correctly ignores.
6. Each branch's campaign record retained only the terminal material-symbol diagnostic, not the
   initial recurring `baseline.py` violation. Later proposal requests therefore did not receive the
   shared root cause. With no normalized failure fingerprint or repetition circuit breaker, all
   24 proposals continued to expect `baseline.py`; 20 were variants of the same pairwise/BPR
   family. No training metric ever became available to guide the scientific loop.
7. The 24th implementation call ended in three consecutive HTTP `429` responses. This stopped the
   repeated invalid-generation loop, but provider availability was not the reason the preceding
   23 candidates failed admission. Provider failover alone cannot fix this run pattern.

### B. Root-cause verdict

| Rank | Hypothesis | Evidence from the captured run | Verdict |
| --- | --- | --- | --- |
| 1 | The agent instructions and typed request did not disclose the actual candidate-file contract | Proposal had no policy; implementation exposed suffixes but not forbidden basenames; all 24 proposals named `baseline.py` | Confirmed primary cause |
| 2 | Repair lost the rejected implementation it was supposed to preserve | All 23 repair requests supplied the same original seed digest; none supplied the rejected generated package | Confirmed primary cause |
| 3 | Repeated failures were not fed back or stopped at the root-cause level | Records retained only the final material-symbol error; `baseline.py` recurred 23 times with no circuit breaker | Confirmed amplifier |
| 4 | Prompt wording alone caused the material-symbol failures | Repair wording correctly prohibited unchanged symbols, but the model lacked the rejected source needed to change them | Contributing, not sufficient |
| 5 | Provider throttling prevented a competitive candidate from training | The `429` occurred only after 23 candidates had already failed local admission | Confirmed terminal/secondary cause |

The correction is deliberately layered. System prompts must be explicit enough for the research
agent to act correctly, while typed validation must reject an invalid plan before an expensive
implementation call. A safety property must never depend on prompt compliance alone.

### C. Local implementation status and evidence (2026-08-29)

The corrective code is now implemented locally without removing the materializer, static-policy,
materiality, evaluator, provider, budget, or fallback protections. The rows below record what is
implemented; they are not evidence that a live agent-generated candidate has trained or beaten the
official FM baseline.

| Work slice | Implemented behavior | Focused local evidence |
| --- | --- | --- |
| P0: captured regression and one source policy | One immutable candidate-source policy now owns the required `candidate.py` entrypoint, suffixes, forbidden basenames/roots/imports/calls, and byte/file limits. Materialization and all three typed agent requests use the same digest-checked policy. The captured `baseline.py`/unsupported-`.csv` failure shape is covered without weakening enforcement. | Source-policy, materializer, request-schema, prompt, provider, and compatibility checks passed; the complete unit suite reported `1,207 passed, 12 skipped`. The skips were existing environment-dependent data, optional-device, and live-key gates. |
| P0: delivered instructions and early proposal correction | Propose, implement, and repair instructions are generated from the exact request policy. A policy-invalid proposal is rejected or corrected before `implement`; provider parsing uses the existing bounded malformed-response correction path, and controller admission independently remains authoritative. | Provider/factory/schema checks reported `58 passed, 1 skipped`; prompt snapshot checks reported `7 passed`; the live-key test remained intentionally skipped. |
| P0: rejected-package repair fidelity | Every generated-package repair carries an inert, bounded, digest-verified snapshot of the causative rejected response, separately from the trusted parent. Each repair is applied freshly over that parent, and the final package must still pass path, static, reachability, and executable-materiality checks. Rehydration verifies the exact causal chain without another model call. | Production-lineage repair checks reported `18 passed`; related runtime integrations reported `19 passed`; the legacy research-loop repair/admission integrations reported `9 passed`. |
| P0: root feedback and circuit breaker | Rejected initial-admission branches retain typed root and terminal observations, normalized fingerprints, proposal family/signature, and bounded diagnostics in an immutable per-iteration journal. Resume reconstructs and verifies its hash chain and fails closed on tampering. The live runtime blocks a proposal family after its second pre-admission rejection and closes on the third occurrence of an identical root fingerprint using `repeated_pre_admission_failure`, while preserving the incumbent and rejection evidence across provider failure. | Runtime and resilience tests cover journal reconstruction/tamper rejection, exact third-identical-root closure, second-same-family blocking, provider-failure preservation, bounded rejection summaries, and strict nine-key stage counts. The focused campaign-control suite reported `23 passed`; after the Chat Completions migration, the repository-wide suite reported `1,400 passed, 21 skipped`. |
| P1: proposal novelty | Proposal signatures normalize the trusted parent/evidence cursor, mechanism/objective family, required fields, enabled inputs, and legal manifest. Prose-only changes do not evade duplication, and no more than two pre-admission attempts from one family are allowed until trusted training/evaluation evidence advances the cursor. | Duplicate-proposal, family-cap, and positive-control evaluation tests are included in the `9 passed` research-loop slice and the production-lineage/runtime suites above. |
| P1: Chat Completions and deadline-aware provider throttling | Both main and fallback now dispatch through `POST {base_url}/chat/completions`, with strict `json_schema` output by default and explicit per-profile compatibility controls for `json_object`, `max_tokens`, and omission of `reasoning_effort`. Typed local validation and every downstream guardrail remain unchanged. Provider transport retains bounded `Retry-After` evidence, applies bounded server-directed or exponential-jitter waits only within the remaining research time before the finalization reserve, caps every transport timeout to that same remaining window, records wait time separately from network latency, and ends the current provider route after the second consecutive `429` so the main/fallback chain can switch. An exhausted shared deadline prevents both a new main request and spurious fallback activation. | Provider tests cover the Chat Completions request/envelope/usage contract, strict and portable JSON modes, delta-seconds and HTTP-date waits, fallback after the second `429`, deterministic backoff, denied waits at the research boundary, transport-timeout capping, zero-call deadline exhaustion, sticky fallback, and usage accounting. The historical Responses-named Python classes remain compatibility aliases and cannot dispatch to `/responses`. |
| P1: truthful no-admission reporting | Research evidence now records nine distinct progress stages, the durable root/terminal rejection summary and examples, provider-chain switches, retry waits, and actual provider usage. Finalization distinguishes provider response acceptance, candidate admission, training, inner evaluation, and outer evaluation, and identifies the official FM fallback as non-agent-generated. | Finalization adapter checks reported `34 passed`; adapter plus runtime resilience/fallback orchestration reported `48 passed`; repository-wide Ruff, Ruff format, strict mypy, and `git diff --check` passed. |

P2 remains intentionally pending. No live admission canary, paid autonomous campaign, matched-seed
baseline comparison, hidden-test evaluation, or organizer submission has been run or claimed from
these code changes. The consolidated offline suite and deterministic fake-provider vertical slices
now pass. The next authorized action is one fresh live admission canary. A full six-hour campaign
remains separately gated on that canary and explicit API-spend/runtime authorization.

### D. Corrective work packages and acceptance gates

#### P0. Freeze the captured failure as a deterministic regression

Before changing behavior, add a secret-free fixture distilled from the immutable transcript. It
must reproduce this exact sequence without network access:

- a proposal whose `files_expected` includes `baseline.py` and an unsupported `.csv` file;
- an implementation package containing a scientifically material module under the forbidden name;
- a pre-materialization filename rejection;
- a repair attempt that must preserve the material implementation while converting it to a legal
  candidate package.

Acceptance gates:

- The regression exercises the captured contract mismatch and fails if the corrected admission or
  repair behavior regresses.
- The fixture contains no API key, provider response ID, encrypted reasoning, dataset row, or
  protected outcome.
- Test runtime remains short enough for the default unit suite.

#### P0. Create one candidate-source policy and expose it everywhere

Define one immutable `CandidateSourcePolicy` owned by the trusted controller. It must be the single
source of truth for:

- required entrypoint (`candidate.py`);
- allowed suffixes;
- forbidden basenames, including `baseline.py`;
- forbidden import roots and calls;
- maximum changed-file count and total UTF-8 bytes;
- complete-package/replacement semantics; and
- the rule that candidate packages may not emit evaluator or submission artifacts such as
  `submission.csv`.

Materialization must continue enforcing the policy. Proposal, implementation, and repair requests
must carry a value-only policy manifest plus a canonical digest derived from the same object. Do
not copy an independently maintained filename list into each layer.

Add controller-side proposal admission before `implement`:

- validate every `files_expected` path against the policy;
- require `candidate.py` in the proposed output contract;
- reject forbidden basenames, unsupported suffixes, absolute/non-canonical paths, duplicates, and
  reserved output artifacts;
- return a bounded, typed proposal-correction request instead of spending an implementation call;
- record the rejected proposal and normalized policy fingerprint durably.

Acceptance gates:

- A proposal containing `baseline.py` or `submission.csv` is corrected or rejected before any
  implementation call.
- A legal helper such as `pairwise_fm.py` remains allowed; no existing safe extension point is
  removed.
- Request and policy digests round-trip and reject tampering.
- The materializer remains authoritative even if a provider ignores the prompt.

#### P0. Rewrite the delivered system instructions from that policy

Update the actual propose, implement, and repair system instructions, not only documentation.
Generate the constraint block from `CandidateSourcePolicy` so prompt text cannot drift from local
enforcement.

The delivered instructions must state, in direct terms:

- `candidate.py` is the required entrypoint;
- `baseline.py` and every other reserved basename are forbidden generated filenames;
- only the listed suffixes are allowed and `.csv` outputs are forbidden;
- responses contain complete replacement candidate-owned files, never patches;
- changed scientific code may live in legal helper modules imported by `candidate.py`;
- `material_symbols` must name reachable top-level Python identifiers actually changed relative to
  the trusted parent; filenames, documentation, docstrings, and unchanged symbols do not count;
- on repair, preserve the rejected package's principal mechanism while resolving the stated local
  failure; and
- provider JSON acceptance is not candidate admission—local policy and materiality checks still
  decide admission.

Include one compact valid manifest example and one invalid example (`baseline.py`) without giving
the agent a hand-authored scientific solution.

Acceptance gates:

- Captured outbound requests for all three operations contain the same policy digest and the exact
  required/forbidden file facts.
- Prompt snapshot tests fail if policy enforcement changes without regenerated instructions.
- A fake model can follow the prompt to produce an allowed multi-file package without privileged
  controller or evaluator access.

#### P0. Preserve rejected source in the repair contract

Add a bounded `RejectedPackageSnapshot` to `RepairRequest`. When validation fails before a
candidate directory is materialized, the snapshot must contain the provider-created file paths,
contents, content digests, package digest, and declared material symbols, subject to the same
file-count and byte limits. Keep the trusted parent snapshot separately so the repair can compare
against it. Never publish the invalid package as an executable lineage node.

Repair semantics must be explicit: return a complete legal replacement package for the trusted
parent while preserving the rejected package's principal scientific mechanism. After repair,
materiality is still measured against the trusted parent, and static policy is rerun from scratch.

Acceptance gates:

- The captured `baseline.py` regression is repaired into legal candidate-owned source, for example
  `candidate.py` plus an allowed helper module, without controller-authored algorithm substitution.
- The repaired package passes path validation, static validation, executable reachability, and
  material-change validation.
- A repair that only edits README text, a docstring, whitespace, or a declared-but-unchanged symbol
  remains rejected.
- Rehydration reproduces the same repair request and package without duplicate provider calls.

#### P0. Add normalized failure feedback and a circuit breaker

Persist two diagnostics for every rejected branch:

- the initial/root admission failure; and
- the final failure after bounded repair.

Normalize them into stable fingerprints such as
`candidate_path_policy:forbidden_basename:baseline.py` and
`materiality:declared_symbol_unchanged:main`. Include counts and the most recent bounded examples in
the next safe research context.

Use this exact repetition policy:

1. First policy violation: issue one bounded proposal/package correction.
2. Second identical root fingerprint in consecutive attempts: reject that manifest/mechanism
   family and require a structurally different legal manifest before implementation.
3. Third identical root fingerprint in the campaign: stop initial portfolio preparation with
   `repeated_pre_admission_failure`, preserve the incumbent, and finalize. Do not spend the
   remaining six-hour or iteration budget repeating the same invalid request.

Acceptance gates:

- The captured recurring failure can consume at most three bounded corrective attempts, not
  23 propose/implement/repair cycles.
- Root and terminal failures both survive interruption, resume, provider failure, and finalization.
- A different actionable failure is not collapsed into the same fingerprint.

#### P1. Enforce proposal novelty before implementation

Create a normalized proposal signature from the parent digest, principal mechanism, objective/loss
family, enabled inputs, required fields, and legal file manifest. Near-duplicate proposals should
be rejected or reframed before source generation when no new training evidence justifies them.

Until one member of a scientific family reaches training, allow at most two pre-admission attempts
from that family. This prevents twenty paraphrased pairwise/BPR plans from consuming the campaign
without evidence while still allowing a measured revision after an actual result.

Acceptance gates:

- Semantic duplicates with different prose produce the same novelty family.
- Pairwise, listwise, feature, calibration, and optimization changes remain distinguishable.
- Once a candidate trains, its metrics and reflection—not a hard-coded scientific preference—decide
  whether another family member is warranted.

#### P1. Make rate-limit handling provider-aware and deadline-aware

Retain the main/fallback inference configuration. For HTTP `429`:

- preserve the first provider attempt and usage evidence;
- honor a bounded `Retry-After` value when present and when campaign/finalization time permits;
- otherwise use exponential backoff with jitter;
- switch to the fallback provider after the second consecutive rate limit, rather than sending
  three near-immediate requests to the same throttled endpoint; and
- keep request identity, schema validation, transcript durability, and total retry limits unchanged
  across providers.

Acceptance gates:

- A fake main provider returning repeated `429` responses fails over to the fallback without
  duplicating an accepted logical operation.
- An unavailable fallback still preserves the incumbent and complete failure ledger.
- Retry waits cannot cross the six-hour cap or consume finalization reserve.

#### P1. Report what happened even when no lineage is admitted

Persist the rejected-branch ledger independently of successful portfolio construction. The final
report must include provider calls/attempts, token and cost evidence, root and terminal failure
counts, affected manifests, repair counts, provider switches, elapsed time, and whether any
candidate reached smoke, training, inner evaluation, or outer evaluation.

Acceptance gates:

- A provider failure during initial portfolio preparation no longer produces a misleading
  `Research-model calls=0` report when durable transcripts exist.
- The report distinguishes `provider_response_accepted`, `candidate_admitted`, `training_started`,
  and `evaluation_completed`.
- Baseline fallback remains explicit and cannot be presented as an agent-generated result.

#### P2. Stage the rerun through objective gates

Do not start another six-hour campaign immediately after the code changes. Advance through these
gates in order:

1. Run the captured secret-free regression and focused unit tests.
2. Run the full static/type/default test gates.
3. Run a deterministic fake-provider vertical slice through proposal, implementation, material
   admission, smoke, repair, resume, and evidence reporting.
4. Run one live admission canary with a fresh no-overwrite run ID. The agent flow—not a human or
   controller-authored candidate—must independently propose and generate one package that passes
   source-policy and materiality checks. Stop before a full campaign if it repeats a normalized
   pre-admission failure.
5. Only after the canary passes, launch the autonomous official campaign with both providers
   configured and the existing monitoring/deadline policy.

The full rerun is ready only when all of the following are true:

- zero proposal manifests violate the disclosed source policy;
- at least one agent-generated candidate reaches smoke and training without manual code repair;
- repair preserves rejected scientific source when invoked;
- no identical root failure exceeds the circuit-breaker threshold;
- provider usage and rejection evidence remain correct after failover or interruption;
- the campaign respects every existing leakage, resource, promotion, convergence, replay, and
  finalization guardrail; and
- baseline comparison is based on completed official evaluation, not provider acceptance or local
  materialization alone.

## Historical pre-run implementation snapshot (2026-08-28)

Completed locally before `runs/full-pure-20260829-05`:

- Schema-v2 run identity distinguishes `autonomous`, `demo`, and `test`; autonomous requires the
  live OpenAI provider, while a full-data scripted demo requires explicit acknowledgement.
- `configs/full-pure.toml` freezes `gpt-5.6-sol`, `xhigh` reasoning, bounded structured-output and
  transport retries, response size/output limits, credential-variable name, and token pricing.
- `kuairand-agent provider preflight --config configs/full-pure.toml` verifies configuration,
  adapter construction, and credential availability without sending an API request or printing a
  credential value. The saved local key passes this preflight.
- Live proposal and implementation responses are validated through the typed `ResearchModel`
  interface, persisted before source publication, rehydrated without duplicate provider calls,
  materialized as immutable source, statically checked, and bound to source/diff/transcript
  artifacts.
- The production driver continues proposal, implementation, protected evaluation, reflection,
  and revision across scientific iterations. It carries incumbent, convergence, launch, elapsed
  time, and outer-promotion cursors; `CANDIDATES_EXHAUSTED` is not a live terminal.
- The safe research context now includes value-free capability manifests, train-only target
  aggregates, train/input-only EDA, prediction-period input-only aggregates, and a feature method
  card. It contains no public-validation or prediction-period outcomes.
- OpenAI usage evidence includes input, cached-input, output, reasoning, and total tokens;
  estimated cost; bounded unaccounted attempts; transcript count; and provider wall time.
- The locked tree-only setup explicitly excludes the optional neural group so an inherited
  `torch` installation cannot silently change campaign identity. Before source edits, the
  archived closed bundle replay passed after this exact environment was restored; the historical
  bundle itself was not changed.
- The README and launch scripts now describe the live configuration, local ignored key loading,
  preflight, exact dependency groups, autonomous terminal conditions, resume/finalize/replay, and
  the fact that a full run can consume API credits and up to six hours.

Verification completed after implementation:

- Full default suite: `1330 passed, 21 skipped`.
- Focused changed-path suite: `151 passed, 1 skipped`.
- Live typed provider vertical slice with fake transport: passed.
- Live outer-loop convergence integration: passed through three iterations.
- Verified-data three-iteration failure/repair/recovery campaign: passed.
- Official six-launch FM qualification and clean replay: passed in `170.76s`.
- Archived campaign resource-receipt acceptance: passed.
- Ruff check/format, mypy over `src` and `tests`, and launcher shell syntax: passed.

At that pre-run point, the following was still required before claiming that the project aim was
fulfilled. The 2026-08-29 investigation above is authoritative where later runtime evidence has
superseded an item:

1. Wire the bounded `ResearchModel.repair` path into the full live production driver for
   same-iteration static/execution failures. The provider-independent research loop already owns
   and tests this repair policy, but the full official live lineage currently closes the failed
   child and relies on the next proposal instead of issuing an in-iteration repair.
2. Run the controlled interruption/recovery drill against the live production path and verify
   that a post-response restart neither duplicates provider calls nor resets any budget.
3. Obtain explicit authorization for API spend/runtime, then launch a new no-overwrite
   autonomous pilot and official campaign. Key creation and preflight are not authorization to
   spend credits.
4. Demonstrate matched-seed mean validation-primary delta strictly greater than `0.002`, pass all
   temporal safeguards, finalize the validation-best eligible checkpoint, and replay its closed
   bundle in a newly synchronized environment.
5. Complete the public repository and Devpost artifacts only from that frozen evidence. Hidden
   test improvement must remain an organizer-only claim.

## 0. Outcome this plan must produce

Produce a new, immutable KuaiRand-Pure campaign in which a live research model, with no
pre-recorded scientific response, does all of the following through the typed research interface:

1. Reads the benchmark contract and trusted EDA.
2. Proposes a falsifiable improvement to any allowed pipeline stage.
3. Creates or materially edits executable candidate source.
4. Runs smoke, temporal-fold, and protected outer-validation evaluation.
5. Repairs ordinary failures within a declared repair budget.
6. Reflects on metrics, resource use, and failures.
7. Uses that reflection to choose and implement the next experiment.
8. Repeats until the organizer convergence rule, 50-scientific-iteration cap, 50-charged-launch
   cap, six-public-promotion cap, finalization reserve, or six-hour cap is reached.
9. Selects the validation-best eligible checkpoint at that terminal condition.
10. Replays it in a clean locked environment and emits a structurally valid, label-free final
    submission and complete judge-facing evidence.

The target readiness claim is `materially_confirmed`: matched-seed mean validation primary must
improve by strictly more than `0.002`, temporal-fold safeguards must pass, and the clean replay
must be exact. Hidden-test improvement remains an organizer-only claim.

## 1. Historical gaps and required corrections

The table below preserves the evidence that motivated remediation. The implementation snapshot
above is authoritative for current local status; its remaining items must not be reported as
complete merely because the corresponding design exists below.

| Gap in the completed campaign | Evidence | Required correction | Acceptance gate |
| --- | --- | --- | --- |
| Scientific decisions were predetermined | `configs/full-pure.toml` selects `provider = "scripted"`; `research/production.py` supplies a fully recorded LambdaRank proposal and package | Make the judged configuration use a live provider. Keep scripted responses only in fixtures, deterministic integration tests, and demos explicitly labelled non-autonomous | A full-data run records `provider = "openai"`, `live_provider_used = true`, non-zero token counts, and no production scripted transcript |
| Candidate code existed before the run | Production reads `candidate_templates/lambdarank` and returns it through `ScriptedResearchModel` | Give the model only the candidate protocol, safe context, parent source, constraints, and method references. Candidate algorithm source must be created or edited in the live response | Winning lineage contains provider response digests and material source diffs not equal to a repository-shipped complete model template |
| Only one scientific iteration ran | Final status reports one scientific iteration and 49 remaining | Continue proposal, implementation, evaluation, reflection, and revision after every completed experiment | At least three genuine scientific iterations, unless a hard cap terminates earlier; every reflection is followed by a next proposal until a legal terminal condition |
| Campaign stopped on a one-candidate portfolio cap | Runtime hard-codes `portfolio_cap = 1` and `CANDIDATES_EXHAUSTED` | Remove the production one-candidate cap. Candidate exhaustion must request another proposal while budget remains | Terminal reason is an exact convergence, iteration, charged-launch, outer-promotion, reserve, deadline, or trusted-failure boundary; `CANDIDATES_EXHAUSTED` is non-terminal for a live campaign |
| Improvement was negligible and unstable | Matched-seed mean delta was `+0.00004225`; one of three seeds regressed; confidence interval crossed zero | Search multiple feature/model/training/fusion hypotheses and retain only reproducible improvements | Matched-seed mean primary delta `> 0.002`, worst temporal-fold delta `>= -0.002`, and no replay mismatch |
| Trusted EDA was absent from the scientific context | Recorded `train_eda` and `validation_input_eda` were empty | Materialize bounded, leakage-safe EDA before the first proposal and expose it through `SafeResearchContext` | Run log contains non-empty train EDA and label-free validation-input EDA with schema/digest provenance |
| Current exact replay fails | Opt-in replay acceptance raises `locked environment differs from campaign creation` | Diagnose the identity mismatch, make environment recreation deterministic, then publish a new campaign rather than altering the closed old bundle | Clean checkout/environment replay reproduces validation metrics, predictions, checkpoint identity, and final submission SHA-256 |
| Robustness was mainly synthetic | Fault-injection tests exist, but the official run recorded no recovery event | Retain synthetic gates and run one controlled pre-production recovery drill; do not inject failure into the final scored run | Interrupted/failed candidate resumes or repairs without losing the incumbent, resetting budgets, or corrupting logs |
| Required resource reporting is incomplete | Calls are counted, but input/output tokens are not reported | Capture per-call and total usage, latency, retry count, and estimated cost when supplied by the provider | Final report has total input tokens, output tokens, cached/reasoning tokens when available, calls, wall time, iterations, CPU/GPU hours, and manual interventions |
| Submission documents are incomplete | No verified public Git repository or Devpost artifact; README lacks several requested sections | Add self-contained submission documents and publish only after secret/data/artifact audit | Public-repo checklist and Devpost checklist pass; all claims link to reproducible evidence |

## 2. Non-negotiable design decisions

### 2.1 Separate trusted control from scientific authority

The trusted controller continues to own:

- Dataset identity, split boundaries, leakage policy, and final-outcome isolation.
- Resource, launch, repair, promotion, and wall-clock budgets.
- Candidate sandboxing and static/dynamic safety gates.
- Organizer-compatible scoring and submission alignment.
- Incumbent protection, convergence calculation, logging, recovery, and finalization.

The live research model owns:

- Interpreting trusted EDA and prior experiment evidence.
- Selecting what pipeline stage to improve next.
- Forming the hypothesis and explaining its expected metric effect.
- Writing or materially revising candidate code and configuration.
- Reflecting on failures and metrics and choosing the next change.

The controller may reject unsafe, invalid, leaky, over-budget, or statistically ineligible work.
It must not silently replace a live proposal with a hand-authored scientific candidate.

### 2.2 Scripted mode is test-only

Retain `ScriptedResearchModel` because deterministic tests need it. Enforce all of the following:

- Move the shipped scripted campaign to `configs/scripted-demo.toml`.
- Set `configs/full-pure.toml` to the live provider.
- Require an explicit `allow_scripted_demo = true` flag for full-data scripted execution.
- Add `run_kind = "test" | "demo" | "autonomous"` to the immutable campaign identity.
- Refuse `materially_confirmed`, `autonomous`, or submission-ready claims for a scripted run.
- Exclude scripted runs from the manual-intervention and research-token comparison used in the
  final submission.

### 2.3 Templates may define a protocol, not a solution

Repository-shipped candidate material may contain:

- The typed `train`, `predict`, checkpoint, and result schemas.
- Safe data-access examples using small synthetic inputs.
- Resource and determinism requirements.
- An intentionally trivial baseline parent.

It must not contain the complete scientific implementation selected as the winning candidate.
Method cards may describe published approaches and citations, but executable algorithm choices
must be materialized by the research response during the campaign.

### 2.4 Use the organizer convergence rule exactly

After each completed eligible scientific iteration, append its validation primary to the durable
convergence series. A campaign terminates only when one of these is true:

1. Over the last three consecutive completed iterations, validation primary has not improved by
   more than `0.002` under the frozen organizer interpretation.
2. Fifty charged scientific iterations have been consumed.
3. Six hours of durable campaign wall time have elapsed, preserving finalization reserve rules.
4. Fifty charged training launches have been consumed.
5. Six protected outer promotions have been consumed.
6. The finalization reserve boundary prevents another safe scientific launch.
7. A trusted-system error is unrecoverable; this is a failed campaign, not a converged one.

Candidate rejection, provider exhaustion, branch exhaustion, or a reflection recommending a new
idea is not convergence. Failed attempts still consume budget according to the existing
conservative accounting rules but are recorded separately from the completed-score series.

### 2.5 Preserve old evidence

Do not edit or republish `runs/wp8-official-campaign-07`. Treat it as historical evidence. All
remediation runs use new no-overwrite paths, for example:

```text
runs/remediation-smoke-01
runs/autonomous-pilot-01
runs/official-live-01
```

## 3. Ordered implementation work

### Phase 0: Freeze evidence and restore reproducibility

Objective: understand and eliminate the current environment mismatch before creating new result
claims.

Work:

1. Capture the current `uv.lock`, `.python-version`, Python build, platform, dependency groups,
   installed distributions, relevant CPU/native-library identity, and source-tree digest.
2. Compare that identity field-by-field with
   `runs/wp8-official-campaign-07/final/environment.json` and the campaign create request.
3. Determine whether the drift is caused by Python patch version, optional dependency groups,
   mutable virtual-environment state, native LightGBM/OpenMP identity, or overly broad hashing.
4. Make `capture_environment_identity` represent only inputs that can be recreated from the
   documented locked setup. Do not weaken checks on model-affecting native or Python packages.
5. Add a human-readable identity diff to replay failures while keeping credentials and sensitive
   paths out of logs.
6. From a newly synchronized environment, rerun the old bundle replay. If the old environment can
   no longer be recreated, record that limitation and require the new official campaign to pass
   two fresh-environment replays.

Primary files:

- `src/kuairand_agent/finalization/production.py`
- Environment-identity implementation located through its current source call graph
- `tests/unit/test_portable_environment_identity.py`
- `tests/replay/test_clean_replay_orchestration.py`
- `tests/full_data/test_completed_campaign_replay_acceptance.py`
- `README.md`

Exit gate:

```bash
UV_CACHE_DIR=.uv-cache uv sync --locked --group research-tree
KUAIRAND_COMPLETED_RUN_DIR=runs/wp8-official-campaign-07 \
KUAIRAND_FINAL_BUNDLE_DIR=runs/wp8-official-campaign-07/final \
KUAIRAND_PURE_DATA_DIR=.data/KuaiRand-Pure/data \
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree \
pytest -q tests/full_data/test_completed_campaign_replay_acceptance.py
```

The gate either passes exactly or produces a signed, precise historical-environment limitation.
No old artifact is rewritten to manufacture a pass.

### Phase 1: Make the live research provider production-complete

Objective: make live propose, implement, repair, and reflect operations durable, bounded, and
fully accounted.

Work:

1. Change `configs/full-pure.toml` from `scripted` to `openai` and define the model, reasoning
   effort, request timeout, maximum response bytes, and named credential environment variable.
2. Move the old scripted configuration to `configs/scripted-demo.toml`.
3. Ensure every provider operation validates the exact typed schema and request/campaign digest.
4. Persist response ID, model identity, operation, request/response digests, latency, retry count,
   and provider usage fields without persisting credentials.
5. Aggregate input, output, cached, and reasoning tokens when returned. Preserve `unknown` rather
   than inventing zero when a field is unavailable.
6. Implement bounded retry for transport errors and rate limits. Schema or policy errors go
   through the candidate repair budget, not infinite provider retries.
7. Verify that resume never reissues a completed provider call and never double-counts usage.
8. Add a preflight command that validates configuration and credential availability without
   consuming a research call.

Primary files:

- `configs/full-pure.toml`
- `configs/scripted-demo.toml`
- `src/kuairand_agent/research/provider.py`
- `src/kuairand_agent/research/factory.py`
- `src/kuairand_agent/research/schemas.py`
- `src/kuairand_agent/campaign/store.py`
- `src/kuairand_agent/campaign/full_campaign_runtime.py`
- `src/kuairand_agent/finalization/report.py`
- Provider unit, vertical-slice, resume, and failure-injection tests

Exit gates:

- A fake-transport vertical slice proves all four operations and exact usage aggregation.
- A live synthetic-data smoke campaign completes at least two research turns without full data.
- Killing the process after a provider response but before publication resumes without a duplicate
  call or duplicate token charge.
- No secret value appears in SQLite, JSON, logs, reports, or test snapshots.

### Phase 2: Replace the bounded scripted branch with a genuine iterative campaign

Objective: make reflection causally affect the next experiment and remove the one-candidate stop.

Work:

1. Refactor `run_provider_free_scientific_campaign` and the full runtime so provider choice does
   not determine the scientific state machine.
2. At iteration `n`, build `ProposalRequest` from the incumbent, safe EDA, eligible history,
   previous reflection, remaining budgets, and a bounded method-card index.
3. Materialize the returned package in a fresh child workspace and require a material executable
   diff against its declared parent.
4. Run static validation, smoke training, inner folds, optional repair, outer promotion, and seed
   confirmation under the existing protected boundaries.
5. Record an honest result for rejected, failed, repaired, promoted, and incumbent-replacing
   candidates.
6. Call `reflect` with metrics, paired deltas, runtime, errors, source-diff summary, and remaining
   budget.
7. Feed the reflection into iteration `n + 1`; assert this linkage in the durable ledger.
8. Continue until the exact convergence function or a hard budget terminates the campaign.
9. Remove production constants and assertions that force portfolio count/cap to one.
10. Keep the official FM fallback immutable and finalizable throughout every iteration.

Primary files:

- `src/kuairand_agent/campaign/full_campaign_runtime.py`
- `src/kuairand_agent/campaign/scientific.py`
- `src/kuairand_agent/campaign/convergence.py`
- `src/kuairand_agent/research/loop.py`
- `src/kuairand_agent/research/prompts.py`
- `src/kuairand_agent/research/production.py` (demote to demo/fixture use)
- `src/kuairand_agent/finalization/production.py`
- Campaign controller, resume, convergence, and full-data acceptance tests

Exit gates:

- A deterministic fake-provider campaign performs at least four iterations with proposal IDs,
  implementation diffs, evaluations, reflections, and parent lineage all linked.
- One candidate fails and is repaired; one is rejected; one replaces the incumbent.
- `CANDIDATES_EXHAUSTED` requests another live proposal while budget remains.
- Tests prove all three legal success terminal reasons and that no other reason is labelled
  converged/completed.
- Resume in the middle of iterations two and three produces the same final ledger as uninterrupted
  execution.

### Phase 3: Add trusted, useful EDA and research context

Objective: ensure the research model can actually inspect the problem without receiving protected
validation labels or final outcomes.

Work:

1. Materialize bounded train EDA covering row/user/item counts, label prevalence, per-user
   impression/positive distributions, categorical cardinalities, missingness, time drift, signal
   correlations, duration/watch behavior, randomized-exposure prevalence, and cold-start rates.
2. Materialize validation-input EDA using label-free fields only: cardinalities, overlap/cold-start,
   feature drift, group sizes, and time distribution.
3. Include feature capability inventory, official baseline metrics, inner-fold metrics, prior
   experiment summaries, and remaining budgets.
4. Add schema versions, exact field provenance, truncation rules, and digests.
5. Limit context size with deterministic summaries rather than raw data or arbitrary sampled rows.
6. Test that validation targets and final outcomes cannot be serialized into the research context.

Primary files:

- `src/kuairand_agent/data/audit.py`
- `src/kuairand_agent/data/capabilities.py`
- `src/kuairand_agent/research/context.py`
- `src/kuairand_agent/research/prompts.py`
- `src/kuairand_agent/campaign/full_campaign_runtime.py`
- Data, leakage, research-context, and full-data tests

Exit gates:

- `train_eda` and `validation_input_eda` are non-empty in the first live proposal request.
- Context-size and deterministic-digest tests pass.
- Sentinel tests prove protected validation labels and all final-period outcomes are absent.
- The final report lists the EDA facts that materially influenced each hypothesis.

### Phase 4: Run a bounded but genuinely adaptive performance program

Objective: obtain a meaningful and repeatable validation improvement without hard-coding the
winning response.

The controller supplies method cards and capability descriptions, not executable answers. The
research model may choose, combine, revise, or abandon them. The initial portfolio should make the
following high-value directions discoverable:

1. Stronger leakage-safe causal history: smoothed user/item/author/tag long-view rates, recency,
   counts, cold-start fallbacks, and time-of-day/context crosses.
2. Metric-aligned grouped tree ranking: LambdaRank objectives, leaves/depth/min-data/learning-rate
   tuning, deterministic early stopping on train-derived folds, and calibrated CPU budgets.
3. Pairwise or listwise FM improvements: negative pairing within user, field-aware crosses,
   regularization, dimension, optimizer, and seed stability.
4. Rank fusion: fold-selected rank or logit fusion among independent candidates without selecting
   weights on public-validation labels.
5. Compact neural interaction models only when expected benefit justifies runtime and memory.
6. Auxiliary-feedback or watch-time objectives only with explicit current-row leakage rules and
   train-only labels.

Promotion discipline:

- Cheap smoke and temporal-fold screens precede public outer validation.
- Maximum six distinct public promotions remains enforced.
- A candidate must beat its declared parent on mean inner primary and respect the worst-fold guard.
- Public validation is used for controlled promotion, not unrestricted hyperparameter search.
- Seed confirmation uses identical seeds and paired evidence against the official FM.
- The incumbent is selected by eligible validation evidence; `materially_confirmed` requires the
  matched-seed mean delta to be strictly greater than `0.002`.

Exit gates:

- At least three materially different executable approaches are attempted unless convergence or a
  hard cap legally fires first.
- Experiment logs explain why each next approach followed from the preceding result.
- The selected candidate passes all leakage, fold, seed, replay, runtime, and alignment gates.
- If no candidate clears `0.002`, the run is reported honestly as `validation_improved`,
  `baseline_reproduced`, or `below_baseline`; it is not submission-ready under this plan.

### Phase 5: Prove recovery without weakening the final run

Objective: demonstrate that long autonomous execution survives ordinary failures.

Controlled pre-production drills:

1. Provider timeout followed by bounded retry.
2. Invalid generated schema followed by one model repair.
3. Candidate syntax or import failure followed by repair or branch rejection.
4. Candidate training timeout followed by clean process-tree termination.
5. Controller interruption after durable subprocess completion followed by reconciliation.
6. Interruption during finalization followed by provider-free completion.

For every drill, verify:

- The attempt is charged exactly once.
- The original deadline is preserved.
- The incumbent and FM fallback remain intact.
- The failure and recovery appear in the judge-facing iteration log.
- Resume does not duplicate provider calls, outer queries, or training launches.
- An unrecoverable trusted-system error fails closed and is not labelled convergence.

Primary tests:

- `tests/fault_injection/`
- `tests/integration/test_campaign_resume.py`
- `tests/integration/test_full_campaign_runtime_resilience.py`
- `tests/integration/test_finalization_fallback_orchestration.py`
- New live-provider ledger/retry tests using a deterministic fake transport

### Phase 6: Execute and freeze the official autonomous campaign

Preconditions:

- Phases 0–5 pass.
- Full developer suite passes.
- Official five-seed baseline qualification passes from verified data.
- Live provider preflight passes.
- No generated candidate, provider transcript, dataset, secret, or old run is accidentally staged
  for publication.

Execution:

```bash
scripts/run_full_campaign.sh \
  configs/full-pure.toml \
  runs/wp3-official-qualification \
  runs/official-live-01
```

Operational rules:

- Do not edit source, configuration, data, or the run directory after campaign creation.
- Do not manually choose candidates during the run.
- A human action required to unblock execution is recorded immediately as an intervention with
  timestamp, reason, and effect.
- Use `status` for observation and `resume` only for genuine interruption.
- Do not reset the wall-clock or launch budgets.
- Never score or inspect final-period outcomes.

Required terminal evidence:

- Legal convergence/cap reason.
- Full experiment and candidate lineage.
- Every hypothesis, source diff, metric, error, repair, reflection, and next decision.
- Validation-best eligible checkpoint at the terminal condition.
- Matched-seed primary delta and temporal-fold summary.
- Token/resource/intervention totals.
- Label-free final predictions and organizer-valid submission.
- Provider-free clean replay from a freshly synchronized environment.
- A second clean replay in a new temporary workspace with identical submission SHA-256.

### Phase 7: Complete the public submission package

Repository deliverables:

1. Initialize and review a Git repository if this directory is still not version-controlled.
2. Add a public-safe `.gitignore` and secret/data/artifact scanning gate.
3. Update `README.md` with:
   - Project overview and challenge mapping.
   - Architecture and autonomous loop.
   - Setup and installation.
   - Exact commands to reproduce baseline, campaign, replay, and submission validation.
   - Development tools, APIs, libraries, frameworks, datasets, and assets.
   - Results table and delta over the official baseline.
   - Resource and intervention totals.
   - Limitations and improvements with more time.
   - Team contributions.
4. Add `docs/devpost.md` containing the complete written project description.
5. Add `docs/results.md` with machine-linked evidence and no hidden-test claim before scoring.
6. Add `docs/interventions.md`, even when the count is zero, explaining the counting boundary.
7. Add `docs/reproducibility.md` describing environment identity and replay guarantees.
8. Include or link the required iteration log and final artifact manifest without publishing raw
   data, credentials, unsafe provider transcripts, or oversized ignored artifacts.

Publication requires user-controlled authentication and an explicit public-repository action; the
implementation may prepare and audit the repository but must not claim publication before a URL
is verified.

Exit gates:

- A fresh reader can reproduce baseline qualification and validate the frozen submission from the
  README alone.
- Every numeric claim resolves to a bundle or run-log artifact.
- Secret scanning, ignored-file audit, test suite, format, lint, typing, replay, and submission
  validation all pass.
- Devpost text includes all four required deliverable groups.

## 4. Verification matrix

| Claim | Required evidence | Failure meaning |
| --- | --- | --- |
| Baseline reproduced | Fresh five-seed full-data qualification within frozen tolerance | Campaign cannot start |
| Agent is autonomous | Live-provider proposal/implementation/reflection chain, material code diffs, no scripted production response | Run is a demo, not the challenge submission |
| Agent iterates | At least three linked iterations or a legal hard-cap terminal condition | Full research loop not demonstrated |
| Agent converged | Exact epsilon/patience series or an allowed iteration, launch, promotion, reserve, or wall-time terminal receipt | `COMPLETED` label is invalid |
| Candidate improved | Eligible validation primary above official baseline | Retain/report FM fallback |
| Improvement is material | Matched-seed mean delta `> 0.002` and fold guards pass | Do not claim strong or confirmed improvement |
| No leakage | Capability/sentinel tests and final-outcome access counters remain zero | Reject run and submission |
| Robustness | Fault drills and durable recovery ledger | Recovery criterion unproven |
| Reproducible | Two fresh-environment clean replays with exact artifact identities | Not submission-ready |
| Submission valid | Organizer checker plus independent canonical-alignment validation | Do not upload |
| Resource report complete | Provider usage totals, wall time, iterations, CPU/GPU hours, interventions | Deliverable incomplete |
| Hidden test improved | Organizer result above hidden baseline | Never infer locally |

## 5. Mandatory commands before declaring readiness

Developer gates:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked pytest -q
UV_CACHE_DIR=.uv-cache uv run --locked ruff check src tests
UV_CACHE_DIR=.uv-cache uv run --locked ruff format --check src tests
UV_CACHE_DIR=.uv-cache uv run --locked mypy src tests
sh -n scripts/run_full_campaign.sh
```

Full-data gates:

```bash
KUAIRAND_PURE_DATA_DIR=.data/KuaiRand-Pure/data \
UV_CACHE_DIR=.uv-cache uv run --locked \
pytest -q tests/full_data/test_official_baseline_qualification.py

KUAIRAND_COMPLETED_RUN_DIR=runs/official-live-01 \
KUAIRAND_FINAL_BUNDLE_DIR=runs/official-live-01/final \
KUAIRAND_PURE_DATA_DIR=.data/KuaiRand-Pure/data \
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree \
pytest -q tests/full_data/test_completed_campaign_replay_acceptance.py

KUAIRAND_COMPLETED_RUN_DIR=runs/official-live-01 \
KUAIRAND_FINAL_BUNDLE_DIR=runs/official-live-01/final \
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree \
pytest -q tests/performance/test_completed_run_resource_receipts.py

UV_CACHE_DIR=.uv-cache uv run --locked kuairand-agent validate-submission \
  --split test \
  --data-dir .data/KuaiRand-Pure/data \
  runs/official-live-01/final/submission.csv
```

Readiness audit:

- Confirm the terminal reason is legal.
- Confirm `scientific_iterations >= 3` unless an allowed hard cap explains otherwise.
- Confirm the provider is live and usage totals are present.
- Confirm no scripted complete candidate supplied the winning lineage.
- Confirm matched-seed mean primary delta is strictly greater than `0.002`.
- Confirm no validation or final-label leakage.
- Confirm two fresh replay results match exactly.
- Confirm the final CSV is finite and canonically aligned.
- Confirm manual interventions are counted using a documented boundary.
- Confirm public documents and claim-to-evidence links are complete.

## 6. Definition of fixed

The project is fixed only when all of the following are true:

1. The official campaign is driven by a live research model, not a scripted response.
2. The agent writes or materially revises candidate code during the run.
3. Evaluation and reflection cause subsequent scientific decisions.
4. The run terminates under the organizer convergence rule or an allowed hard cap.
5. The selected candidate is materially confirmed above the official validation baseline.
6. Final outcomes remain completely inaccessible and unscored.
7. Recovery, budget integrity, and incumbent preservation are proven.
8. Exact clean replay succeeds twice from newly synchronized environments.
9. The final submission passes organizer and independent alignment checks.
10. Token, wall-time, iteration, device, and intervention totals are complete.
11. README, Devpost, results, limitations, and team-contribution deliverables are ready.
12. Hidden-test performance is described as unknown until the organizer returns it.

Until then, use precise partial labels such as `baseline_reproduced` or
`validation_improved`; do not describe the historical scripted one-iteration campaign—or the
new unexecuted live implementation—as fulfilling the autonomous research-agent aim.

---

# Original architecture plan: Autonomous KuaiRand-Pure ML Research Agent

Status: implementation-ready planning document
Primary benchmark: KuaiRand-Pure
Reference execution path: local CPU
Implementation scope: the autonomous agent, its experiment harness, a strong recommender search program, and reproducible final artifacts

## 1. Decision on the supplied agent plan

The supplied HTML plan has the correct high-level idea: keep the ML pipeline being optimized separate from the LLM-driven loop that proposes, writes, runs, evaluates, repairs, and revises that pipeline. Its emphasis on experiment memory, error recovery, and run logs also matches the judging criteria.

It is not safe to implement as written, however. This plan replaces it rather than treating it as the implementation specification.

The material corrections are:

1. The HTML targets click with NDCG@10 and Recall@50. The supplied executable starter kit targets `long_view` and scores GAUC plus nDCG@5, with primary equal to their mean. The starter kit and organizer evaluator are the implementation source of truth.
2. The HTML suggests a broad stack before proving the benchmark contract. The implementation starts with a small controller, the immutable organizer scorer, NumPy, and only then adds PyTorch and LightGBM where an experiment requires them.
3. The HTML does not define a safe boundary between current-row outcomes and inference features. This plan makes leakage prevention executable and testable.
4. The HTML does not protect the public validation period from unlimited adaptive tuning. This plan uses train-derived temporal folds and a small outer-promotion budget.
5. The HTML says the agent writes code, but also suggests a largely prebuilt/config-selected model library. This plan requires material generated source changes in the winning lineage.
6. The HTML does not define conservative iteration accounting, durable recovery, incumbent protection, or exact final replay. This plan does.
7. The HTML gives disproportionate attention to the optional large datasets. KuaiRand-Pure is completed and evidenced first; bonus datasets are explicitly deferred.

The result should be an autonomous ML research agent, not merely a hyperparameter tuner and not a hand-authored winning model wrapped in an LLM narrative.

## 2. Product outcome

Implement a command-line system that can perform the following sequence with minimal human intervention:

1. Verify and prepare the official KuaiRand-Pure data.
2. Reproduce the official validation rungs and five-seed FM baseline.
3. Inspect train data and label-free validation inputs.
4. Ask a research model for a falsifiable improvement hypothesis.
5. Ask it to create or modify executable candidate source code.
6. Validate the generated package before consuming a full experiment.
7. Train under resource and time limits.
8. Score aligned validation predictions only through the trusted organizer-compatible scorer.
9. Record the hypothesis, source diff, metrics, runtime, resources, errors, repairs, and intervention count.
10. Promote or reject the candidate using a deterministic policy.
11. Reflect on the result and choose the next experiment.
12. Stop on convergence, the 50-iteration cap, the six-hour ceiling, or an unrecoverable trusted-system error.
13. Restore the best confirmed eligible candidate, run clean inference, validate its submission, and produce a replayable final bundle.

The implementation must preserve a valid FM fallback throughout the campaign. A failed research branch may spend budget, but it must never destroy the incumbent or prevent finalization.

## 3. Exact benchmark contract

### 3.1 Source-of-truth order

When sources conflict, use this order:

1. The hash-pinned organizer `evaluate.py`, `data.py`, `baseline.py`, `submit.py`, and `baseline_scores.json` in `kuairand-starter-kit/`.
2. Any later organizer clarification explicitly applicable to this starter-kit version.
3. The challenge prose supplied by the user.
4. The attached HTML plan.
5. Papers, framework defaults, or public examples.

Do not silently reconcile a contradiction. Record the chosen interpretation in the run manifest.

### 3.2 Operative task

- Training dates: 2022-04-08 through 2022-04-21.
- Public validation dates: 2022-04-22 through 2022-04-28.
- Final evaluation dates: 2022-04-29 through 2022-05-08.
- Relevance target: native binary `long_view`.
- Ranking unit: each user's logged impressions in the evaluation split.
- GAUC: only users with both positive and negative impressions; each user AUC is weighted by the user's positive count.
- nDCG@5: every user is included; users with zero positives receive zero; gain is `2^rel - 1`.
- Primary: `(GAUC + nDCG@5) / 2`.
- Final CSV header: `row_id,user_id,video_id,score`.
- `row_id`: zero-based contiguous physical split order as produced by the organizer loading convention.
- Scores: finite real values; only within-user ordering affects the metrics.

The implementation must not substitute a framework's AUC, grouped AUC, or nDCG for the organizer evaluator. Framework metrics may be diagnostics only.

### 3.3 Published reference rungs

The full-data qualification command must reproduce, within declared tolerance:

| Reference | Validation result |
| --- | ---: |
| Random scoring | Reproduce the hash-pinned starter reference after rounding. |
| Item popularity | Reproduce the hash-pinned starter reference after rounding. |
| Official FM | GAUC `0.6674`, nDCG@5 `0.5357`, primary `0.6016`. |

The supplied starter currently yields five validation FM primaries near:

```text
0.6014695
0.6017609
0.6010903
0.6015031
0.6020371
```

Their mean reproduces `0.6016` after rounding. Implementation must rerun these values from the verified data and store its own evidence rather than trusting this planning observation.

### 3.4 Immutable organizer files

Treat the following as read-only references:

- `kuairand-starter-kit/data.py`
- `kuairand-starter-kit/evaluate.py`
- `kuairand-starter-kit/baseline.py`
- `kuairand-starter-kit/submit.py`
- `kuairand-starter-kit/baseline_scores.json`
- `kuairand-starter-kit/README.md`

The project may import, wrap, and test them. It must not edit them, copy an altered version over them, or permit generated code to shadow their module names.

### 3.5 Dataset identity

The data preparation command should accept either an already downloaded archive or an explicit download flag. Network access must not be required after acquisition.

Pinned planning identity:

- Artifact: `KuaiRand-Pure.tar.gz`
- Official record: Zenodo record `10439422`
- Publisher MD5: `0820331067a3784d9691136f772b35a7`
- Planning-time SHA-256: `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0`

The implementation independently verifies the archive and writes a member manifest before extraction. It must expect exactly eight regular files plus the two archive directory entries, reject links and special files, reject absolute or parent-traversing paths, and extract into a new data directory.

### 3.6 Claims policy

Use these result labels precisely:

- `baseline_reproduced`: the qualification gates pass.
- `validation_improved`: an eligible candidate scores above the official validation baseline.
- `materially_confirmed`: the matched-seed mean improvement is greater than `0.002` and the temporal-fold safeguards pass.
- `hidden_test_improved`: only after the organizer scores the frozen final submission above its reference.

No plan or local result can guarantee the last claim.

## 4. Scope and non-goals

### 4.1 Required scope

- KuaiRand-Pure only.
- Complete autonomous propose, implement, execute, evaluate, reflect, repair, select, and finalize loop.
- Baseline reproduction.
- Leakage-safe feature engineering.
- At least one metric-aligned loss branch.
- At least one strong non-neural ranking branch.
- Bounded neural, history, multi-task, and ensemble branches when evidence and time permit.
- Structured logs and human-readable report.
- Resume and failure recovery.
- Clean replay and valid final submission.

### 4.2 Deferred scope

- KuaiRand-1k and KuaiRand-27k.
- Full-catalog retrieval.
- Distributed training.
- Production serving.
- A web dashboard.
- A generic autonomous-science platform.
- Automatic deployment to a remote service.

The architecture should not prevent later dataset adapters, but no required milestone depends on them.

### 4.3 Execution assumptions

- A local CPU path is mandatory and is the reference for reproducibility.
- Accelerator support may be added later, but no core test, baseline, candidate family, replay, or finalization step may require it.
- The default campaign should fit a conventional development machine with configurable thread, memory, process, and disk limits.
- External training data is forbidden.
- Candidate execution is local and must not depend on an external sandbox service.

## 5. Selected architecture

### 5.1 Trusted and generated planes

Keep two explicit planes:

```text
Trusted controller
  data contract -> safe capabilities -> experiment policy -> protected scorer
        |                    |                    ^
        v                    v                    |
Generated candidate workspace -> train/infer -> score vector only
        ^                                           |
        |                                           v
research model <- safe context, aggregate results, errors, remaining budget
```

The trusted controller owns:

- archive verification and data preparation;
- split definitions and physical row order;
- feature/label capability construction;
- the evaluator wrapper;
- attempt and wall-clock accounting;
- workspace creation;
- process supervision;
- score alignment and validation;
- experiment records;
- promotion, convergence, and final selection;
- final CSV creation and verification.

Generated candidate code owns:

- candidate-specific feature transformations over approved inputs;
- model architecture;
- objective and sampling logic;
- training loop and checkpoint contents;
- prediction logic;
- declared diagnostics.

Generated code may use trusted, narrow primitives, but it must not edit the controller, data policy, scorer, split logic, budget policy, or finalizer.

### 5.2 Components

Implement these cohesive modules:

| Component | Responsibility |
| --- | --- |
| `Contract` | Loads the pinned benchmark configuration and organizer artifact hashes. |
| `DataManager` | Verifies the archive, preserves row identity, creates safe phase-specific data packages, and caches causal features. |
| `ProtectedScorer` | Performs structural checks, calls the immutable evaluator, and returns organizer metrics. |
| `BaselineQualifier` | Reproduces random, popularity, and FM validation rungs and creates a replayable FM fallback. |
| `ResearchModel` | Typed propose, implement, reflect, and repair interface with a scripted test implementation and a real provider adapter. |
| `CandidateWorkspace` | Materializes an immutable parent snapshot plus one generated child package. |
| `Runner` | Starts a subprocess, enforces resource/time limits, captures outputs, and kills the complete process tree on failure. |
| `ExperimentStore` | Persists campaigns, experiments, executions, metrics, artifacts, source diffs, failures, and interventions. |
| `Selector` | Applies inner-screen, outer-promotion, confirmation, convergence, and incumbent rules. |
| `Finalizer` | Restores the selected candidate, replays inference, writes high-precision CSV, validates it, and packages evidence. |
| `Reporter` | Produces the experiment table, trajectory, failure/recovery summary, resource usage, and final rationale. |

The controller should be ordinary Python with explicit domain types. Do not hide core campaign behavior in framework callbacks.

### 5.3 Public CLI

Provide one discoverable CLI, for example `kuairand-agent`, with these commands:

```text
kuairand-agent data prepare --archive PATH --data-dir PATH
kuairand-agent data audit --data-dir PATH
kuairand-agent qualify --data-dir PATH --run-dir PATH
kuairand-agent run --config PATH
kuairand-agent resume --run-dir PATH
kuairand-agent status --run-dir PATH [--json]
kuairand-agent replay --bundle PATH --data-dir PATH
kuairand-agent validate-submission --split valid|test FILE
```

Rules:

- `run` either creates a new campaign or refuses to overwrite an existing one.
- `resume` continues the same campaign and remaining budgets.
- `status --json` emits one stable JSON object without progress noise on stdout.
- commands return non-zero on invalid data, failed qualification, corrupt state, invalid submission, or incomplete replay.
- progress logs go to stderr and files.
- every command supports `--help` and clear error messages.

## 6. Repository shape

Implement toward this structure:

```text
.
├── plan.md
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── default.toml
│   ├── smoke.toml
│   └── full-pure.toml
├── kuairand-starter-kit/          # organizer-owned, unchanged
├── src/kuairand_agent/
│   ├── __init__.py
│   ├── cli.py
│   ├── contract.py
│   ├── config.py
│   ├── data/
│   │   ├── acquire.py
│   │   ├── audit.py
│   │   ├── canonical.py
│   │   ├── fields.py
│   │   ├── capabilities.py
│   │   ├── folds.py
│   │   └── causal_features.py
│   ├── scoring/
│   │   ├── protected.py
│   │   ├── baseline.py
│   │   └── submission.py
│   ├── research/
│   │   ├── interface.py
│   │   ├── schemas.py
│   │   ├── scripted.py
│   │   ├── provider.py
│   │   ├── method_cards.py
│   │   └── prompts.py
│   ├── execution/
│   │   ├── workspace.py
│   │   ├── policy.py
│   │   ├── runner.py
│   │   └── artifacts.py
│   ├── campaign/
│   │   ├── models.py
│   │   ├── store.py
│   │   ├── controller.py
│   │   ├── selector.py
│   │   ├── convergence.py
│   │   └── recovery.py
│   ├── finalization/
│   │   ├── finalize.py
│   │   ├── replay.py
│   │   └── report.py
│   └── candidate_api/
│       ├── protocol.py
│       └── primitives.py
├── candidate_seed/
│   ├── candidate.py
│   ├── config.json
│   └── README.md
├── tests/
│   ├── fixtures/
│   ├── unit/
│   ├── integration/
│   ├── full_data/
│   ├── fault_injection/
│   └── replay/
├── docs/
│   ├── architecture.md
│   ├── benchmark-contract.md
│   ├── data-policy.md
│   ├── experiment-policy.md
│   └── research/
└── scripts/
    ├── smoke.sh
    ├── qualify_full_data.sh
    └── run_full_campaign.sh
```

Runtime data belongs under ignored directories such as `.data/`, `.cache/`, and `runs/`. Do not commit the extracted dataset, checkpoints, predictions, credentials, or provider transcripts that contain secrets.

## 7. Environment and dependencies

### 7.1 Python and locking

- Target Python 3.11 or 3.12; choose one during bootstrap and lock it in project metadata.
- Use `uv` with a committed lock file.
- The base install contains the controller, NumPy, and testing dependencies.
- Put PyTorch and LightGBM in an explicit research dependency group if lock/platform constraints require separation.
- Do not add AIDE, RecBole, FuxiCTR, TorchRec, Optuna, Ray, or MLflow to the core path unless a measured requirement justifies the cost.

### 7.2 Reproducibility controls

Every execution records:

- source revision and generated-source digest;
- organizer artifact and dataset digests;
- Python, NumPy, PyTorch, and LightGBM versions when present;
- OS and architecture;
- device and numeric precision;
- thread counts;
- Python, NumPy, framework, sampler, and model seeds;
- feature-cache, split, config, checkpoint, and prediction digests;
- command, start/end time, wall time, peak RSS, and exit status.

The canonical confirmation path uses fixed thread counts and CPU. Cross-platform bit identity is not promised for every model; metric and within-user ordering tolerances must be stated and tested.

### 7.3 Configuration

Use a versioned TOML schema. At minimum it contains:

```toml
schema_version = 1

[benchmark]
name = "kuairand-pure"
data_dir = ".data/KuaiRand-Pure/data"
starter_dir = "kuairand-starter-kit"
target = "long_view"
max_iterations = 50
wall_clock_seconds = 21600
epsilon = 0.002
convergence_patience = 3

[validation]
outer_promotion_limit = 6
confirmation_seeds = [0, 1, 2]
inner_folds = ["20220416:20220418", "20220419:20220421"]

[runner]
max_processes = 1
threads = 4
memory_mb = 16384
disk_mb = 20480
default_timeout_seconds = 1800
finalization_reserve_seconds = 3600

[research]
provider = "scripted"
max_repairs_per_experiment = 2
```

Validate unknown fields, types, ranges, and mutually inconsistent budgets before a run begins. Store the normalized effective config with the campaign.

## 8. Leakage-safe data implementation

### 8.1 Canonical row identity

Assign row identity before sorting, joining, grouping, or feature generation:

- Read the two standard-log files in the organizer order.
- Filter by the date interval.
- Preserve physical file order among retained rows.
- Assign `row_id = 0..N-1` independently within train, validation, and final splits.
- Keep row identity in the trusted data layer only.
- Candidate inputs are fixed-order arrays/tables and do not expose `row_id`, source ordinal, provenance hash, or alignment hash as feature columns.
- Candidate inference returns exactly one finite score per input row in the same order.
- The trusted finalizer attaches `row_id`, `user_id`, and `video_id` afterward.

Repeated `(user_id, video_id)` pairs must remain distinct rows. No join may use that pair as a unique key.

### 8.2 Phase-specific capabilities

Do not expose one monolithic raw table and rely on generated code to ignore columns. Build separate data packages:

| Capability | Contents | Consumer |
| --- | --- | --- |
| `train_inputs` | Approved pre-impression fields in canonical order | Candidate feature and model code |
| `train_targets` | `long_view` plus approved auxiliary outcomes | Candidate training only |
| `inner_valid_inputs` | Approved fields for a train-derived fold | Candidate inference |
| `inner_valid_targets` | Fold targets | Trusted inner scorer; candidate receives aggregate results only |
| `outer_valid_inputs` | Public-validation approved fields | Candidate inference |
| `outer_valid_targets` | Public-validation `long_view` | Protected scorer only |
| `final_inputs` | Final-period approved fields, no outcome values | Candidate inference and finalizer |
| `alignment` | Trusted row/user/video identity and split digest | Scorer/finalizer only |

The candidate should never receive public-validation or final-period outcome columns. The research model receives schemas and aggregate scores, not row-level targets, predictions, residuals, or examples selected using target error.

### 8.3 Field registry

Classify every field with one of these roles:

- `inference_input`
- `training_primary_target`
- `training_auxiliary_target`
- `strict_past_history_source`
- `trusted_alignment_only`
- `blocked`

Initial inference inputs:

- `user_id`
- `video_id`
- `author_id`
- `tab`
- `duration_ms`
- reviewed date/time fields
- reviewed static item metadata whose timestamp/provenance is safe
- causal aggregates derived only from permitted past training rows

Current-row outcome fields are never inference inputs:

- `long_view`
- `is_click`
- `is_like`
- `is_follow`
- `is_comment`
- `is_forward`
- `hate`
- `play_time_ms`
- profile/comment dwell outcomes
- any response-derived field from the same impression

They may be training targets where semantics are verified. In particular, `is_click` is a plausible auxiliary target for the primary `long_view` task; it is not the benchmark label and is never a same-row feature.

Block by default:

- `video_features_statistic.csv`, because it includes outcome aggregates over a period that overlaps evaluation dates and lacks a per-row as-of cutoff;
- randomized-exposure logs, because permitted use and cutoff semantics are not pinned by the required benchmark contract;
- mutable user/item snapshots with unknown extraction date;
- external embeddings or data not explicitly allowed by the organizer.

An organizer clarification may change a blocked field only through a reviewed contract/config update and new leakage tests.

### 8.4 Causal feature rules

All response-derived aggregates and histories obey:

1. A row may use only events strictly earlier than its timestamp.
2. Rows with equal timestamps are simultaneous; none may update features for another in the same timestamp bucket.
3. Train features use expanding or out-of-fold construction so the current target cannot leak into its own feature.
4. Each inner fold rebuilds histories using only its training prefix.
5. Public-validation features freeze outcome history at the end of official training. Do not roll validation outcomes forward within the validation week.
6. Final-period features freeze outcome history at the end of official training unless the organizer explicitly supplies and allows an observable, label-free online update protocol.
7. Smoothing priors are fit from the permitted prefix only.
8. Cache keys include data, split, field-policy, builder-source, and parameter hashes.

Required metamorphic tests:

- changing a future outcome does not alter an earlier feature;
- changing the current outcome does not alter the current feature vector;
- permuting rows within an equal-timestamp bucket does not alter features;
- changing any skipped public-validation or final outcome value does not alter candidate inputs, predictions, selection context, or final CSV;
- rebuilding twice produces identical logical feature artifacts.

### 8.5 Data audit

`data audit` streams the verified archive and emits JSON plus a readable report containing:

- archive and member hashes;
- exact headers, sizes, and row counts;
- split row counts and date ranges;
- train and validation target domain/rate only;
- user, video, and author cardinalities;
- per-user slate-size distribution;
- cold user/video/author counts relative to training;
- repeated-pair counts and maximum multiplicity;
- train-only associations useful for auxiliary-target planning;
- a field-policy table;
- a statement and machine-checkable trace showing that final-period outcome values were skipped rather than converted, validated, aggregated, logged, or scored.

Never report a final-period label rate, domain, correlation, metric, or model score during development.

## 9. Scorer and baseline qualification

### 9.1 Protected scorer

Before calling the immutable organizer evaluator, validate:

- equal, non-zero lengths for users, labels, and scores;
- exact expected row count and split identity;
- binary labels for train-derived and public-validation scoring;
- finite numeric scores;
- no missing or duplicate positional outputs;
- unchanged trusted alignment;
- no framework metric is being submitted as the official result.

Return GAUC, nDCG@5, primary, user count, row count, scorer digest, prediction digest, and runtime. Candidate code receives only the permitted aggregate result.

Golden fixtures must cover:

- zero-positive, all-positive, single-row, and mixed users;
- interleaved user rows;
- tied scores, including ties around rank five;
- duplicate user/video pairs;
- short, long, empty, non-numeric, NaN, and infinite score vectors;
- monotonic score transforms that preserve order;
- organizer-wrapper equality on valid fixtures.

### 9.2 Replayable official FM

The untouched `run_fm` function returns only aggregate metric dictionaries and always tries to evaluate a `test` split. It cannot by itself provide the checkpoint and prediction artifacts needed for fallback and replay.

Implement a trusted `StarterFMAdapter` that:

- imports the immutable organizer `FM`, `encode`, and evaluator behavior;
- mirrors the organizer epoch order, RNG, batches, pointwise objective, early stopping, and best-state restore;
- never consumes final-period targets;
- emits the restored `V`, `W`, `b`, encoding metadata, validation predictions, metrics, and training trace;
- can infer on label-free final inputs.

Prove behavior rather than assuming it:

1. On fixtures, compare the adapter against untouched `run_fm` with a harmless nonempty placeholder test split.
2. On full data, compare validation aggregates for seeds 0–4 against both untouched organizer logic and the published reference.
3. Replay each stored checkpoint and require identical validation predictions on the same host.
4. Retrain seed 0 from source/config/data and require the declared reproducibility tolerance and identical within-user ordering.

The best confirmed official-FM seed becomes the immutable fallback lineage. A candidate may warm-start from a copy, never mutate the stored fallback.

### 9.3 Qualification sequence

The `qualify` command performs, in order:

1. Organizer file hash check.
2. Dataset preparation/audit check.
3. Evaluator golden tests.
4. Deterministic random rungs for fixed seeds.
5. Exact item-popularity rung.
6. FM seeds 0–4.
7. FM checkpoint replay.
8. High-precision validation submission round-trip.
9. Evidence manifest creation.

Stop immediately on a contract or scorer mismatch. Model research cannot compensate for a broken benchmark harness.

## 10. Autonomous research protocol

### 10.1 Research-model interface

Use a narrow typed protocol:

```python
class ResearchModel(Protocol):
    def propose(self, request: ProposalRequest) -> Proposal: ...
    def implement(self, request: ImplementationRequest) -> GeneratedPackage: ...
    def reflect(self, request: ReflectionRequest) -> Reflection: ...
    def repair(self, request: RepairRequest) -> GeneratedPackage: ...
```

Provide:

- `ScriptedResearchModel` for deterministic integration and fault tests;
- one real provider adapter configured by environment/config;
- strict JSON-schema parsing;
- request timeout, response-size, and token/cost accounting;
- redacted transcript retention;
- bounded retries for malformed responses.

Provider failure must not corrupt campaign state. If the provider is unavailable, the campaign stops cleanly with the FM fallback still finalizable.

### 10.2 Proposal schema

Every proposal states:

- unique hypothesis and mechanism;
- expected effect on GAUC, nDCG@5, or both;
- parent candidate;
- one principal scientific change;
- files it expects to create or modify;
- required input fields and their roles;
- objective, sampling, grouping, and weighting;
- causal cutoff semantics;
- estimated runtime and memory;
- smoke and inner-fold plan;
- falsification and promotion criteria;
- maximum repair count;
- rollback parent;
- primary-source or method-card attribution where applicable.

Reject proposals that:

- request protected labels, final outcomes, raw archive access, network access, or forbidden fields;
- modify trusted modules;
- exceed remaining time/attempt budgets;
- duplicate a prior mechanism/config fingerprint without new evidence;
- bundle multiple scientific changes that cannot be attributed;
- claim guaranteed improvement.

### 10.3 Generated code contract

The implementation response contains a bounded mapping of allowlisted relative file paths to complete UTF-8 file contents. This is more reproducible than applying an underspecified patch.

The controller:

1. Gives the model exact contents and hashes of the parent candidate files relevant to the accepted proposal.
2. Limits new/changed files, total response bytes, and permitted suffixes.
3. Rejects absolute paths, `..`, symlinks, binaries, hidden control files, and trusted-module paths.
4. Writes the files only into a new disposable candidate directory.
5. Computes and logs the unified diff itself.
6. Runs format, import, policy, fixture, and smoke gates before a full experiment.

Repair requests include the exact failed child source, bounded stderr/traceback, failure category, and remaining repair budget. The model has no implicit repository or shell access through this protocol.

A proposal claiming a new model, loss, sampler, feature transform, or fusion mechanism must produce a material executable-source change. A config-only toggle cannot satisfy that claim. The final report and manifest identify the material generated changes in the winner's lineage.

### 10.4 Safe research context

The research model may receive:

- benchmark contract and immutable hashes;
- approved field schema and role registry;
- train EDA and label-free validation-input EDA;
- method cards and primary-source summaries;
- exact inner-fold aggregate metrics;
- outer aggregate metrics rounded to the predeclared reporting precision;
- current incumbent, candidate tree, source diffs, runtime, memory, and failure summaries;
- prediction correlation or rank-diversity aggregates computed without exposing labels;
- remaining attempts and wall time;
- intervention count and prior failure fingerprints.

It must not receive:

- final-period outcome values or scores;
- public-validation row-level labels, predictions paired with labels, residuals, or worst-user examples;
- secrets or unrelated filesystem content;
- authority to change the scorer, split, field policy, convergence rule, limits, or final selection rules.

### 10.5 Loop state machine

For each scientific iteration:

```text
PROPOSED
  -> POLICY_REJECTED
  -> MATERIALIZED
       -> STATIC_REJECTED
       -> SMOKE_REJECTED -> REPAIRING -> MATERIALIZED
       -> INNER_RUNNING
            -> FAILED -> REPAIRING or CLOSED
            -> INNER_SCORED
                 -> INNER_REJECTED
                 -> OUTER_ELIGIBLE
                      -> OUTER_SCORED
                           -> PROMOTED or REJECTED
                           -> REFLECTED
```

Persist every transition before launching the next external action. On restart, reconcile any execution left in `RUNNING` with its PID/process record and artifacts; mark it interrupted if it cannot be resumed safely.

## 11. Candidate execution contract

### 11.1 Workspace

Each candidate directory contains only:

- immutable copied parent source;
- generated child source/config;
- a manifest of approved input artifacts;
- a writable output directory;
- a bounded temp directory.

It does not contain:

- raw archives;
- public-validation targets;
- final-period outcomes;
- the protected scorer;
- other candidates' workspaces;
- provider credentials;
- unrelated home/project files.

Use local process isolation and resource controls honestly: it is robustness containment, not a claim of hostile-code security unless the implementation can prove a stronger boundary. Generated code is expected to cooperate with the contract, while static and runtime checks catch accidental violations.

### 11.2 Candidate entry point

Define one stable command contract, for example:

```text
python candidate.py train \
  --request request.json \
  --output output/

python candidate.py predict \
  --request request.json \
  --checkpoint output/checkpoint/ \
  --output prediction/
```

`request.json` contains opaque artifact handles, approved schemas, split role, seeds, and budgets. Candidate-visible tables do not include trusted alignment columns.

Successful training emits:

- `candidate_result.json` with schema version, diagnostics, and declared artifacts;
- checkpoint/preprocessor files;
- optional training curves;
- no official validation metric computed inside the candidate.

Successful inference emits:

- `scores.npy` with shape `(N,)` and a fixed numeric dtype;
- `prediction_result.json` with schema version, expected count, split token, checkpoint digest, and runtime diagnostics.

The trusted controller checks count, dtype, finiteness, split token, and checkpoint/source/config identity. It then attaches alignment and calls the protected scorer.

### 11.3 Supervisor

The runner must:

- launch without a shell where possible;
- set a sanitized environment;
- set deterministic thread variables;
- create a new process group;
- capture bounded stdout/stderr to files;
- monitor wall time and peak memory;
- kill the full process tree on timeout, cancellation, or memory violation;
- reject undeclared/oversized output;
- persist termination reason and resource measurements;
- never interpret a metric printed by candidate stdout as official.

Before launching, estimate whether the job's conservative runtime plus finalization reserve fits the remaining campaign time. If not, reject it without consuming a training launch.

## 12. Validation and selection policy

### 12.1 Train-derived temporal folds

Freeze two rolling-origin inner folds before research:

- Fold A: train on 2022-04-08 through 2022-04-15; validate on 2022-04-16 through 2022-04-18.
- Fold B: train on 2022-04-08 through 2022-04-18; validate on 2022-04-19 through 2022-04-21.

The data audit must confirm that each has enough rows, mixed-label users, positives, and slate depth. If a fold is structurally unusable, the implementation may move its boundary using a deterministic minimum-support rule defined before viewing candidate results; it must record the change in the contract manifest.

Tier policy:

1. Fixture gate: syntax/import/shapes on tiny synthetic data.
2. Inner screen: one fixed seed on Fold B.
3. Temporal confirmation: Fold A and Fold B for candidates that clear the screen threshold.
4. Outer promotion: retrain on all official training data and score once on public validation.
5. Seed confirmation: matched incumbent/challenger seeds 0, 1, and 2 for a material promotion.

Inner metrics guide search. Only public validation determines the final local winner.

### 12.2 Outer-promotion discipline

- At most six distinct scientific candidates per campaign receive a public-validation score.
- Store every outer query in an append-only project log so restarting or changing a campaign fingerprint does not erase the development history.
- The research model normally receives the aggregate GAUC, nDCG@5, and primary rounded to the predeclared reporting precision, plus pass/fail and component direction; exact values remain in the trusted log and final report.
- Do not tune an epoch on public validation. Use inner folds for early stopping; outer validation is one terminal evaluation per promoted training run.
- Do not run per-user error mining on public-validation labels.
- A blend chosen on inner folds consumes one outer-promotion slot when evaluated publicly.

### 12.3 Promotion gates

A candidate is eligible for outer promotion only if:

- all policy, import, smoke, and output-contract gates pass;
- it completes within its predicted resource envelope;
- its Fold B primary exceeds its parent by the configured screen margin or it is a preallocated diversity root;
- its mean inner primary is positive relative to the relevant baseline/incumbent;
- neither temporal fold degrades by more than `0.002` relative to the parent unless the experiment is explicitly a metric-specialist retained for blending;
- its result is not explained only by score serialization or alignment changes;
- an outer slot and sufficient finalization time remain.

A public challenger replaces the incumbent only after all structural gates and the configured seed-confirmation rule pass. Until then, it may be retained as an unconfirmed specialist but cannot replace the final fallback.

### 12.4 Matched-seed confirmation

For incumbent and challenger seeds 0, 1, and 2, report:

- GAUC, nDCG@5, and primary per seed;
- paired deltas per seed;
- mean, median, minimum, and standard deviation of primary delta;
- a user-cluster bootstrap confidence diagnostic on aligned predictions;
- inner-fold mean and worst-fold delta;
- runtime and memory deltas.

Call the result materially confirmed only when:

- mean paired public-validation primary delta is greater than `0.002`;
- mean inner-fold primary delta is positive;
- worst inner-fold delta is no worse than `-0.002`;
- no policy, replay, or serialization gate fails.

The controller may still select the highest eligible public-validation mean as the local final candidate if no challenger clears the material threshold, but the report must label it `unconfirmed` and retain the official FM fallback.

### 12.5 Convergence

Use one explicit implementation of `epsilon = 0.002`, `N = 3`:

1. Maintain the best eligible public-validation primary before each completed scientific iteration.
2. If the iteration produces an eligible outer result that improves the previous best by strictly more than `0.002`, reset the non-material streak to zero.
3. Otherwise increment the streak, including failed or policy-valid but non-promoted scientific iterations that consumed an iteration.
4. Stop when the streak reaches three, unless a required confirmation/finalization step is already reserved and must complete.

Golden-test exact epsilon, just-above epsilon, regressions, failures, repairs, and restart behavior. If the organizer publishes a different executable rule, update the contract and tests before a scored run.

## 13. Attempt and time budgets

### 13.1 What consumes an iteration

Conservatively count every launched full train/evaluate execution in the research campaign, including:

- official baseline seed runs;
- candidate inner screens;
- additional inner folds;
- outer retraining/evaluation;
- seed confirmations;
- model blends that require training;
- failed, timed-out, or out-of-memory launches;
- repair children that launch training;
- final from-scratch training replay.

Static checks and tiny synthetic smoke tests are logged but do not consume a scientific iteration. A retry that reuses the same completed checkpoint only for deterministic inference/replay is logged separately and does not masquerade as a new experiment.

### 13.2 Frozen maximum allocation

Use this ceiling, not a quota:

| Purpose | Maximum launches |
| --- | ---: |
| Baseline qualification and one replay | 6 |
| Diverse inner screens | 20 |
| Additional temporal-fold confirmations | 6 |
| Distinct outer promotions | 6 |
| Matched-seed confirmation | 5 |
| Blend/fusion experiments | 3 |
| Final training/replay | 2 |
| Recovery reserve | 2 |
| **Total** | **50** |

Unused branch capacity remains available only within the same total and without taking the finalization reserve. The controller records both the original category and any approved reallocation.

### 13.3 Six-hour wall clock

- Start one monotonic deadline when the campaign begins.
- Include model training, evaluation, provider calls, repairs, and finalization.
- Reserve at least 60 minutes for confirmation, replay, submission checks, and reporting.
- Stop launching research jobs once only the reserve remains.
- Kill work at the hard ceiling and finalize the best already eligible candidate if there is still time.
- If finalization cannot complete, report the campaign incomplete rather than publishing an unverified submission.

Maintain rolling p50/p95 runtime estimates by candidate family. Reject a proposed job whose p95 plus cleanup/finalization buffer exceeds remaining time.

## 14. Performance research program

This is a prioritized hypothesis portfolio, not a hand-authored ladder whose result is assumed in advance. The autonomous proposer selects among eligible method cards based on evidence, novelty, cost, and remaining budget.

### 14.1 Stage 0: exact FM and cheap correctness ablations

Goals:

- establish the replayable official FM;
- measure seed variation;
- profile loading, encoding, training, scoring, and serialization;
- test deterministic cold-user/item handling as a single controlled change;
- confirm that merely increasing FM embedding size or adding known redundant static buckets is not worth repeating unless new evidence exists.

Useful low-cost controls:

- mask or zero the never-trained UNK user embedding at inference;
- train-time ID dropout with a deterministic generic embedding;
- exact duration threshold feature around 18 seconds;
- time-of-day/date inputs, with one-change ablations.

### 14.2 Stage 1: metric-aligned pairwise FM

This is the first priority research branch.

For an eligible user `u` with `P_u` positives and `N_u` negatives, organizer GAUC weights the user's AUC by `P_u`. Since AUC averages over `P_u * N_u` pairs, each positive-negative pair has effective weight proportional to `1 / N_u`.

An unbiased sampler is:

1. Sample an eligible user with probability proportional to `P_u`.
2. Sample one of that user's positives uniformly.
3. Sample one of that user's logged negatives uniformly.
4. Optimize `-log(sigmoid(score_positive - score_negative))`.

This yields the desired pair weighting in expectation. Tests must compare sampler frequencies and a small exact pair-loss calculation against brute-force enumeration.

Experiments:

- identical FM features/initialization with only pointwise versus pairwise objective changed;
- pairwise plus a small pointwise auxiliary term for users with all-one/all-zero labels;
- pair sampling count, regularization, learning rate, and early stopping chosen on inner folds;
- deterministic user grouping and logged-impression negatives only.

Do not use full-catalog negative sampling; the benchmark ranks logged impressions.

### 14.3 Stage 2: causal aggregates and grouped tree rankers

Build one reusable causal feature cache with train-prefix features such as:

- smoothed item, author, tab, and duration-bucket exposure/`long_view` rates;
- recency-weighted counts and rates;
- user-author, user-duration, user-tab, and user-item-history affinities;
- days since first/last permitted interaction;
- item/author freshness and popularity trend;
- exact duration, log duration, the 18-second regime indicator, and safe interactions;
- date, day-of-week, and time-of-day after semantic validation;
- cold-entity indicators;
- repeated-pair context features that do not use row order as a predictor.

Then implement a benchmark-specific LightGBM adapter:

- sort a private training view into contiguous user groups without changing canonical output order;
- use `lambdarank` or a controlled pairwise ranking objective;
- set deterministic CPU parameters and fixed thread count;
- test group sizes, inverse permutation, and score scatter-back;
- use the protected organizer scorer for every official comparison;
- treat LightGBM's own NDCG as a diagnostic only because its all-negative-query convention may differ.

Tree rankers are prioritized before expensive neural capacity changes because they are fast, strong on aggregate/context features, and complementary to the ID-heavy FM.

### 14.4 Stage 3: low-cost fusion

Retain candidates with complementary GAUC/nDCG behavior or low prediction correlation.

Fusion rules:

- normalize each member to deterministic within-user ranks or percentiles before blending unlike score scales;
- choose weights from a small predeclared grid on inner folds;
- use stable deterministic tie handling that does not turn `row_id` into a learned signal;
- spend one outer slot on the selected blend, not one per weight;
- serialize and read back before accepting the gain.

Likely useful combinations include pairwise FM plus LambdaRank, or an identity model plus a causal aggregate model.

### 14.5 Stage 4: compact neural interactions

Only enter this branch if simpler models leave evidence of nonlinear feature interactions and enough time remains.

Eligible mechanisms:

- compact DeepFM;
- small DCNv2 cross network;
- shallow MLP over embeddings and dense causal features;
- calibrated pointwise/pairwise hybrid loss.

Use small embeddings and explicit parameter/runtime ceilings. A model-family change must be generated as executable candidate code, not selected from a complete hidden model zoo.

Close the branch if it cannot beat the same feature table with FM/tree models on both inner folds or its p95 runtime threatens the reserve.

### 14.6 Stage 5: aggregate history and sequence attention

Pure contains incomplete candidate-pool-filtered sequences, so sequence models are hypotheses, not guaranteed wins.

Start with cheap aggregate history. Attempt DIN-style attention only after measuring:

- fraction of rows with at least 1, 5, 10, and 20 usable prior events;
- sequence truncation effects;
- feature-build memory and time;
- one-epoch throughput;
- incremental gain over aggregate history on both inner folds.

History semantics:

- strict-past only;
- equal timestamps simultaneous;
- validation/final histories freeze at the training cutoff;
- pad/mask deterministically;
- keep compact event arrays and offsets rather than materializing large dense histories for every candidate.

Abandon attention if coverage is poor, throughput is too slow, or aggregate history performs equivalently.

### 14.7 Stage 6: controlled multi-task learning

The primary task remains `long_view`. Same-row click, like, follow, comment, forward, and watch-time signals may be auxiliary training targets only.

Gating sequence:

1. Add click as an auxiliary head to an otherwise identical backbone.
2. Require positive inner evidence relative to the single-task control.
3. Try a simple shared-bottom model before MMoE.
4. Try MMoE only if shared-bottom shows negative transfer or task conflict.
5. Try PLE only if MMoE leaves clear task-specific conflict and time remains.
6. Consider an isolated censored-watch-time auxiliary loss only after reproducing its equation and verifying target semantics; do not import an old research stack wholesale.

Tune loss weights on inner folds. Do not expose auxiliary outcomes at inference. Record every head, mask, transform, and loss weight.

### 14.8 Research routing rules

Encode these as proposer guidance, not hard-coded outcome claims:

- GAUC improves while nDCG falls: test a complementary LambdaRank/listwise component or rank blend.
- nDCG improves while GAUC falls: blend with the GAUC-aligned pairwise model.
- causal aggregates help both folds: refine smoothing, decay, and interactions before increasing architecture depth.
- a complex model loses to a simple model on the identical feature table: close the complex branch.
- auxiliary click does not beat the single-task control: stop the MTL branch before MMoE/PLE.
- history coverage or throughput fails its gate: retain aggregates and stop attention work.
- two models are nearly rank-identical and neither has component complementarity: do not spend a blend attempt.
- predictions contain excessive repeated-pair ties: add legal context or a real secondary model, never row order as a feature.
- a branch p95 threatens finalization: prune it regardless of theoretical appeal.

## 15. Experiment persistence and logs

### 15.1 Store

Use SQLite in WAL mode for controller metadata plus content-addressed files for source, checkpoints, predictions, and logs.

Minimum tables:

- `campaigns`
- `experiments`
- `executions`
- `proposals`
- `source_snapshots`
- `metrics`
- `artifacts`
- `failures`
- `interventions`
- `validation_queries`

Transactions must make state transitions and their referenced artifact metadata atomic. Write artifacts to a temporary same-filesystem path, validate and hash them, then rename them into their final content-addressed path before committing the referencing state.

### 15.2 Required iteration record

Every scientific iteration records:

- iteration number and launch count;
- hypothesis, mechanism, parent, and method attribution;
- full proposal request/response digest;
- generated source snapshot and controller-computed diff;
- data capability, feature, config, and environment identities;
- seed and exact command;
- inner/outer tier;
- GAUC, nDCG@5, primary, and deltas when scored;
- training diagnostics;
- wall time, peak memory, disk, and provider usage;
- promotion/rejection reason;
- error category, traceback digest, repair, and recovery outcome;
- human intervention category and count;
- remaining attempts/time and convergence state.

Export judge-readable `experiments.jsonl` and `experiments.csv` at finalization.

### 15.3 Incumbent rules

- The official FM fallback is immutable after qualification.
- A new candidate becomes eligible only after source, data, output, scorer, and replay checks pass.
- Promotion creates a new immutable incumbent reference; it does not overwrite prior artifacts.
- Failure, timeout, invalid output, or provider error cannot demote the existing incumbent.
- Finalization uses the best candidate allowed by the frozen selection policy, not the last attempted candidate and not an LLM's unverified choice.

## 16. Failure recovery

Classify failures and use bounded policy:

| Failure | Default action |
| --- | --- |
| Malformed model response | Reparse once, then one schema-focused provider retry; no training launch. |
| Syntax/import/static-policy failure | Send bounded diagnostics for repair, maximum two child repairs. |
| Deterministic data/shape bug | Repair child if budget remains; identical fingerprint is not retried indefinitely. |
| Non-finite scores | Reject output; one repair may add numerical stabilization. |
| Timeout | Kill process tree; do not rerun identical config; proposer must reduce cost or close branch. |
| Out of memory | Kill process tree; allow one explicitly smaller batch/model repair if predicted to fit. |
| Interrupted process | Mark interrupted; resume only from a checkpoint whose source/config/data identities match. |
| Candidate-produced invalid metric | Ignore it; only trusted scoring counts. |
| Protected scorer/contract failure | Stop research and preserve state; do not ask generated code to repair trusted code. |
| Provider unavailable | Retry within provider policy, then stop cleanly and finalize the best eligible candidate if possible. |
| Final-candidate replay mismatch | Fall back to the previous replayable incumbent, ultimately the official FM. |

Fault injection belongs in pre-campaign tests. Do not manufacture failures during the scored run merely to create a robustness story.

On SIGINT or SIGTERM, stop launching work, terminate the current process group safely, flush state, and leave the campaign resumable when time policy permits.

## 17. Finalization and submission

### 17.1 Selection

At convergence or budget stop:

1. Freeze the best eligible candidate under the predeclared policy.
2. Record why it beat or replaced its parent.
3. Ensure confirmation requirements and status labels are accurate.
4. If the candidate cannot replay, walk back through eligible ancestors to the official FM.

The research model may provide a written reflection, but it cannot override deterministic eligibility or selection.

### 17.2 Clean replay

From a clean candidate workspace:

1. Restore the exact selected source, config, preprocessing metadata, and checkpoint.
2. Rebuild or verify feature artifacts from the verified dataset.
3. Reproduce public-validation predictions.
4. Require exact same-host prediction bytes where the model supports it; otherwise require the declared numeric tolerance, identical per-user top-5 ordering, and metric parity.
5. Run label-free final-period inference.
6. Confirm one finite score per canonical row.

Also run at least one from-scratch selected-candidate training replay if the campaign budget reserves it. If only checkpoint replay is feasible, report that limitation explicitly.

### 17.3 CSV safety

Do not use the starter writer's six-significant-digit formatting for the final artifact because it can create new ties.

The trusted writer must:

- write the exact required header;
- use contiguous trusted `row_id`;
- attach trusted `user_id` and `video_id` in canonical order;
- serialize enough digits for round-trip float64 identity or proven rank equivalence;
- reject NaN and infinity;
- read the file back;
- verify row count and alignment;
- verify that within-user ordering and public-validation metrics are unchanged by serialization;
- run the untouched `submit.py --check` against the target split.

Only the organizer performs hidden-test scoring. The local finalizer performs structural checking, not final-label evaluation.

### 17.4 Final bundle

Produce a self-contained bundle with paths relative to its root:

```text
final/
├── manifest.json
├── report.md
├── submission.csv
├── experiments.jsonl
├── experiments.csv
├── config/
├── source/
├── model/
├── preprocessing/
├── validation-evidence/
├── replay/
├── environment.json
├── reproduce.sh
└── verification.json
```

`manifest.json` includes:

- benchmark, starter, and data identities;
- selected experiment and complete lineage;
- status label (`baseline_reproduced`, `validation_improved`, or `materially_confirmed`);
- exact validation metrics and seed summary;
- inner-fold results;
- source, config, feature, checkpoint, prediction, and submission hashes;
- environment and resource usage;
- attempt, time, and intervention totals;
- relative paths and hashes for every required component;
- known limitations and unresolved organizer questions.

`reproduce.sh` must use only repository code, the locked environment, an explicitly supplied verified dataset path, and declared credentials only if a fresh research campaign is requested. Replaying the frozen final model must not call an LLM.

### 17.5 Report

The final report contains:

- benchmark contract and baseline parity table;
- experiment trajectory and candidate tree;
- hypotheses, material code changes, and source attributions;
- inner, outer, and seed-confirmation evidence;
- per-metric tradeoffs;
- failures, repairs, recoveries, and interventions;
- runtime, peak memory, provider usage, and optional-device usage if any;
- leakage controls and tests;
- selected-candidate rationale;
- replay and submission verification;
- honest statement that hidden-test improvement is unverified until organizer scoring.

## 18. Verification strategy

### 18.1 Fast unit tests

- Config parsing, defaults, validation, and normalization.
- Hash and artifact utilities.
- Metric wrapper golden fixtures and invalid input rejection.
- Convergence truth table.
- Budget accounting and reallocation.
- Proposal and generated-package schemas.
- Path allowlist and generated-file limits.
- Candidate output validation.
- Selector tie-breaking and incumbent protection.
- Failure fingerprinting and retry limits.
- Submission high-precision round-trip.
- Pairwise sampler weighting versus brute force.
- User-group permutation and inverse scatter.
- Within-user rank normalization and blend determinism.

### 18.2 Data and leakage tests

- Secure archive-member validation.
- Canonical split counts and row ordering.
- Duplicate-pair preservation.
- Candidate capabilities contain only registered fields.
- Final inputs contain no outcomes.
- Current or future outcome mutation does not affect current or past features.
- Equal-timestamp permutation invariance.
- Inner-fold histories use only their training prefix.
- Public-validation history does not consume public-validation outcomes.
- Blocked side tables cannot enter a cache or candidate request.
- Row identity cannot appear in candidate arrays or declared feature lists.

### 18.3 Integration tests

- Scripted proposal to generated source to smoke to train to inference to protected score to reflection.
- Material generated code change is present in lineage.
- Invalid generated path is rejected without changing trusted source.
- Exact parent source context reproduces the generated child and diff.
- Baseline adapter checkpoint and inference replay.
- LightGBM group sorting and scatter-back on fixtures.
- Runner timeout kills descendants.
- Resume reconciles interrupted execution.
- Failed child leaves incumbent intact.
- Outer-promotion limit survives restart.
- Provider transcript redaction.
- Finalizer falls back when selected replay is corrupted.
- Untouched submission checker accepts the final CSV.

### 18.4 Fault-injection tests

Deliberately inject:

- malformed JSON response;
- syntax error;
- import error;
- forbidden field or path request;
- process child that outlives its parent;
- timeout;
- memory exhaustion fixture;
- non-finite, truncated, or wrong-length predictions;
- killed controller between artifact write and database transition;
- corrupted checkpoint;
- scorer exception;
- disk-full simulation where practical;
- interrupt during final inference.

Verify bounded recovery, persistent diagnostics, accurate iteration charging, no incumbent loss, and clean resume or fallback.

### 18.5 Full-data gates

These are slower and explicitly marked:

1. Data audit reproduces the expected train and validation facts without final-outcome inspection.
2. Random, popularity, and FM seeds reproduce references.
3. FM checkpoint replay passes.
4. Pairwise-FM objective and sampler run end-to-end.
5. One causal-feature tree candidate runs end-to-end.
6. A scripted autonomous campaign completes at least three scientific iterations including a failed child and recovery.
7. A real-provider vertical slice proposes and materially implements one safe change.
8. A bounded full campaign completes or converges within the caps.
9. Clean final replay and submission validation pass.

### 18.6 Performance tests

- Peak memory for audit, feature build, FM, pairwise sampler, tree ranker, and selected final model.
- p50 and p95 elapsed time by family.
- Feature-cache cold versus warm time.
- One-epoch throughput and predicted completion error.
- Controller overhead relative to model time.
- Finalization reserve sufficiency.
- No accidental quadratic grouping or pair enumeration on full data.

Performance regressions fail when they break configured ceilings or make the six-hour plan infeasible, even if functional tests pass.

## 19. Ordered implementation work packages

Complete these in order. A later package cannot waive an earlier gate.

### WP0: project bootstrap

Deliver:

- `pyproject.toml`, lock file, package skeleton, CLI shell, and config schema;
- lint, type, and test commands;
- ignored data, cache, and run paths;
- concise developer README.

Acceptance:

- clean install on the supported Python version;
- `--help` works;
- unit-test skeleton runs offline;
- organizer files are unchanged and protected by hash tests.

### WP1: benchmark contract and protected evaluator

Deliver:

- typed benchmark contract;
- organizer artifact hash manifest;
- protected scorer wrapper;
- metric and convergence fixtures;
- high-precision submission writer and reader.

Acceptance:

- all evaluator golden tests pass against immutable `evaluate.py`;
- malformed score vectors are rejected before organizer evaluation;
- starter files remain byte-identical.

### WP2: acquisition, canonical data, and leakage boundary

Deliver:

- verified acquisition and extraction;
- streaming audit;
- canonical split and order implementation;
- field registry and phase capabilities;
- temporal folds;
- initial causal-feature builders and cache.

Acceptance:

- expected counts, order, and duplicates are reproduced;
- final-period outcomes are skipped;
- capability and metamorphic leakage tests pass;
- two builds have identical logical manifests.

### WP3: official baseline reproduction

Deliver:

- random and popularity adapters;
- replayable `StarterFMAdapter`;
- seed orchestration;
- checkpoint, prediction, and parity evidence;
- immutable fallback bundle.

Acceptance:

- seeds 0 through 4 reproduce official validation values within tolerance;
- checkpoint and clean seed-0 replay pass;
- no final-period target is read or scored;
- fallback inference produces a valid aligned CSV.

### WP4: durable campaign engine and runner

Deliver:

- experiment database and artifact store;
- workspace materialization;
- subprocess supervisor;
- attempt and time accounting;
- selector, convergence, resume, and incumbent protection;
- failure taxonomy.

Acceptance:

- scripted executions survive restart;
- timeouts kill process trees;
- failed or invalid children cannot replace or corrupt the incumbent;
- 50-iteration and six-hour policies are deterministic.

### WP5: autonomous code-generation vertical slice

Deliver:

- research-model schemas and interface;
- scripted provider;
- real provider adapter;
- safe context builder;
- generated-package materializer and diff logger;
- propose, implement, reflect, and repair controller loop.

Acceptance:

- a scripted model makes a material candidate-source change and completes the loop;
- a deliberate syntax error is repaired without human editing;
- exact request and response plus parent source reproduce the child package;
- a real-provider smoke run completes when credentials are supplied;
- protected fields and trusted files remain inaccessible.

### WP6: metric-aligned and causal high-value branches

Deliver:

- GAUC-aligned sampler and loss primitives;
- pairwise FM candidate seed and reference tests;
- causal aggregate features;
- LightGBM grouping adapter;
- rank-fusion utilities;
- method cards and routing policy.

Acceptance:

- sampler matches brute-force weighting;
- all group and order tests pass;
- autonomous generated candidates can use the primitives while still materially owning their mechanism code;
- inner-fold screens remain within time and memory budgets.

### WP7: bounded advanced branches

Deliver as evidence permits:

- compact DeepFM and DCNv2 primitives;
- compact aggregate-history representation;
- DIN-style candidate support;
- shared-bottom and MMoE or PLE building blocks;
- auxiliary-target masks and losses;
- family-specific performance gates.

Acceptance:

- each enabled branch has a same-feature or same-backbone control;
- no branch is enabled without leakage and throughput tests;
- generated source implements the claimed mechanism;
- branches close automatically when evidence or budget fails.

WP7 is not allowed to delay a strong pairwise, tree, or blend campaign or finalization.

### WP8: full autonomous Pure campaign

Deliver:

- frozen full-run config;
- completed qualification;
- autonomous campaign under the hard caps;
- exact experiment ledger;
- selected incumbent and seed confirmation;
- measured performance profile.

Acceptance:

- the run ends through convergence or a declared cap;
- manual interventions are zero or explicitly counted and justified;
- a valid replayable incumbent exists;
- validation status is reported honestly;
- any performance claim is backed by retained predictions, logs, and rerunnable commands.

If the score is not yet improved, diagnose from evidence and make bounded implementation or scientific corrections without weakening evaluation, leakage, or replay gates. Rerun only the affected tests and then one fresh capped campaign; do not tune indefinitely on public validation.

### WP9: finalization, replay, and expert-readiness audit

Deliver:

- clean selected-candidate replay;
- final-period label-free inference;
- checked submission;
- closed final bundle;
- human-readable report;
- clean-install reproduction instructions.

Acceptance:

- a fresh environment can install from the lock;
- the verified dataset rebuilds the required artifacts;
- final replay meets declared equality or tolerance rules;
- public-validation CSV serialization preserves metrics and order;
- untouched `submit.py --check` accepts the submission;
- the report clearly separates baseline, local improvement, confirmation, and unverified hidden-test status.

## 20. Definition of done

### 20.1 Engineering complete

All of the following are true:

- The benchmark contract is executable and hash-pinned.
- Data acquisition, audit, and all leakage metamorphic tests pass.
- The official five-seed validation baseline is reproduced.
- The FM fallback is checkpoint- and source-replayable.
- The research model materially writes candidate source.
- The full propose, implement, run, evaluate, and reflect loop operates without manual source editing.
- Failures are bounded, recorded, and recovered or closed correctly.
- Attempt and wall-clock caps cannot be bypassed by resume or retry.
- Incumbent selection is deterministic and protected.
- Final inference and CSV generation never access final outcomes.
- The final bundle and report are complete and replayable.
- Unit, integration, leakage, fault, full-data, performance, and replay gates pass at the required level.

### 20.2 Performance ready

All engineering-complete requirements pass, and:

- a generated candidate lineage exceeds the official public-validation FM reference;
- the matched-seed mean exceeds the matched FM mean;
- the mean temporal-fold delta is positive;
- the worst temporal-fold delta is no worse than `-0.002`;
- the gain survives high-precision CSV round-trip;
- replay reproduces the declared predictions, order, and metrics;
- the full campaign and finalization fit the limits;
- no external data, forbidden outcomes, evaluator change, or row-identity feature explains the gain.

The preferred target is a confirmed public-validation primary improvement greater than `0.002`, not merely a lucky best seed. If that target is not achieved, the implementation is still engineering-complete only if it reports the shortfall accurately; it must not relabel an unconfirmed result.

### 20.3 Submission ready

- One immutable final submission is designated.
- `row_id`, `user_id`, `video_id`, row count, score finiteness, and canonical alignment pass.
- The untouched organizer checker passes.
- The model, source, config, data, and environment identities are recorded.
- The report contains run logs and intervention and resource totals.
- The hidden test has not been locally inspected or scored.
- The organizer may now perform its one hidden-test scoring run.

## 21. Implementation discipline for the next Codex task

The implementing task should use this file as the source of truth and follow its work-package order, while retaining authority to make small evidence-backed implementation choices that do not change the benchmark, leakage, budget, or acceptance contracts.

Expected working behavior:

- inspect the existing starter before writing code;
- preserve user-owned files and organizer references;
- implement a thin vertical slice before model breadth;
- use tests first for scorer, data causality, alignment, budgets, and recovery;
- use primary sources when a model equation or framework behavior matters;
- profile before optimizing;
- continue through ordinary test failures and performance regressions;
- record discovered deviations in the relevant docs and tests;
- do not stop after scaffolding or a single green smoke test;
- do not claim a score that was not reproduced in the current implementation;
- end with exact commands, test results, measured metrics, remaining limitations, and final artifact paths.

The implementation should prefer the smallest dependency and mechanism that passes the next scientific gate. Complexity is justified by measured transferable gain, not by the prestige of a model family.
