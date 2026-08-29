# Autonomous ML Research Agent — MVP Build Brief

> # ⚠️ SUPERSEDED — HISTORICAL PLANNING DOCUMENT
>
> **Do not implement from this document. It describes neither the task we solved nor the system
> we built.** It is retained only as a record of early planning.
>
> **Its task definition is wrong.** It was written before the organizers' starter kit was
> available, from prose that has since been corrected. The authoritative contract is
> `kuairand-starter-kit/` — specifically `evaluate.py`:
>
> | | This document says | Actual contract |
> |---|---|---|
> | Label | `click` | **`long_view`** |
> | Metrics | NDCG@10 / Recall@50 | **GAUC + nDCG@5**, primary = mean |
> | Task shape | ranked candidate list | **within-user ranking of logged impressions** |
> | Baseline | unspecified | **primary 0.5946** hidden test / 0.6016 validation |
>
> **Its architecture is not what was built.** This document proposes FuxiCTR config-editing,
> LiteLLM routing, MLflow, Optuna, and an `agent/` + `pipeline/` layout. The delivered system
> instead uses a trusted-controller / generated-plane split with a typed four-method research
> seam (`propose` / `implement` / `repair` / `reflect`), a byte-protected organizer scorer,
> phase-scoped leakage-safe data capabilities, and direct OpenAI Responses API calls. See
> `README.md` and `plan.md`.
>
> **Its open questions are resolved.** Every ⚠️ CONFIRM below was answered by the starter kit:
> ε = 0.002 and N = 3; scoring is `evaluate.py` verbatim; the submission schema is
> `row_id,user_id,video_id,score`; the splits are train 2022-04-08→04-21, validation
> 04-22→04-28, hidden test 04-29→05-08.
>
> Sections that remain broadly sound regardless: §9 (robustness requirements) and §12 (scoring
> levers). Everything else is stale.

**Challenge:** Autonomous Machine Learning Research Agent for Recommender Systems (KuaiRand)
**Required benchmark:** KuaiRand-Pure · ~~**Task:** predict `click` · **Metrics:** NDCG@10 / Recall@50~~ (see banner)
**Audience:** the engineer building the MVP
**Status of this doc:** superseded planning artefact — see banner above.

---

## 0. TL;DR

Build an LLM-driven agent that runs the ML-engineering loop on KuaiRand-Pure on its own: it reproduces the official baseline, then repeatedly proposes a change, writes the code for it, trains, evaluates on validation, and reflects — driving the validation score above the baseline with minimal human help. Every iteration is logged. The thing being graded is the **agent and its loop**, not a hand-tuned model.

The MVP target is a single run that (a) is fully autonomous end-to-end, (b) beats the official baseline on validation, (c) recovers from at least the common failure modes without a human, and (d) emits the required per-iteration logs.

---

## 1. Scope

### In scope (MVP)
- **KuaiRand-Pure only** (1.4M interactions, 27K users × 7.6K items). This determines 100% of the primary score.
- **Single eval fidelity** — train on the full Pure training split each iteration (it's small enough).
- The full autonomous loop: propose → code → execute → evaluate → reflect → repeat, with memory, convergence detection, and error recovery.
- Per-iteration run logs + a final results table + resource accounting (tokens, GPU-hours).

### Out of scope (defer past MVP)
- KuaiRand-1k / 27k scaling, multi-GPU/DDP, sharded embeddings, the proxy-fidelity funnel.
- Off-policy / counterfactual debiasing using the randomized-exposure data.
- Elaborate tree search — a linear "greedy with memory" loop is fine for the MVP; upgrade to tree search only if time allows.

### Definition of done (MVP)
- [ ] One command starts the agent and it runs to convergence with **zero manual intervention** on a clean path.
- [ ] Final validation NDCG@10 and Recall@50 both exceed the official baseline.
- [ ] A `runs/<id>/iterations.jsonl` log exists with one valid record per iteration (schema in §7).
- [ ] The agent survives at least: a code error in generated code, and a training timeout — recovering from each, with the recovery visible in the log.
- [ ] A results table + token/GPU-hour totals are produced.

---

## 2. Blockers to resolve first ⚠️ CONFIRM

Do these in M0; the rest of the build depends on them.

| Item | Where | Why it blocks |
|---|---|---|
| Official baseline pipeline + its published val/test scores | CWM / starter kit | This is the reference you must reproduce and beat. Not a baseline you invent. |
| Convergence rule: `ε` and `N` | Starter kit | The evaluator's stop condition. Score is the *converged* result, not the peak. |
| Exact evaluation script (NDCG@K / Recall@K) | Starter kit / CWM | You must score identically to the judges. Do not re-implement from memory. |
| Submission output schema + example submission | Starter kit | Final deliverable format. |
| Compute budget | "TBD" in brief | Sets the wall-clock/GPU stop and how many iterations you can afford. |
| Exact split files | Given | `log_standard_4_08_to_4_21_*` → train; `4_22_to_5_08` first 50% → val; last 50% → test. Confirm the split indices. |

Until the eval script and baseline are in hand, work against a *local placeholder* eval so the loop can be built, then swap in the official one.

---

## 3. The task the agent optimizes

### Data
- **KuaiRand-Pure**, from https://kuairand.com. Short-video feed logs with 12 feedback signals (`click`, `like`, `follow`, `comment`, `forward`, `long_view`, `play_time`, …) plus a randomized-exposure flag.
- **Label:** `click = 1` is the positive. Only `click` is scored.
- **Splits (date-based, fixed):** train = 4/08–4/21 standard; validation = first 50% of 4/22–5/08 standard; test = last 50%. **Develop on train + validation only. The test set is hidden — never read it during development.**

### Metrics
- **NDCG@10** and **Recall@50**, computed per user over a ranked candidate list, then averaged. Use the organizer's script for exact definitions (candidate set construction, tie-breaking, and averaging all matter).

### The loop the agent runs (from the challenge's Figure 1)
1. Read the problem — dataset + target metrics.
2. Inspect data — EDA on distributions.
3. Engineer features — build/select inputs.
4. Train + tune — model, loss, hyperparameters.
5. Evaluate — read metrics, check overfitting, compare to baseline.
→ Reflect + revise → next iteration.

Stages 3 and 4 are code the agent writes and rewrites. That is the automation target.

---

## 4. Architecture

Two layers. The **agent loop** (what you build) writes and rewrites the **recommender pipeline** (what gets optimized).

### Components

| Component | Type | Responsibility | Input → Output |
|---|---|---|---|
| **Proposer** | LLM (reasoning) | Pick the next change, grounded in a named RS method + a hypothesis for why it should help. | run history + current best → `{hypothesis, target_stage, change_spec}` |
| **Coder** | LLM (execution) | Turn the change_spec into a concrete edit (config diff or code diff) to the pipeline. | change_spec + current pipeline → patch |
| **Executor** | Tool (execution) | Apply patch, run training + eval in a subprocess/sandbox with a hard timeout; capture stdout, metrics, and tracebacks. | patch → `{metrics | error, logs, artifacts}` |
| **Evaluator** | Logic + LLM (reasoning) | Parse metrics, update the solution store, decide continue vs converged (`ε`/`N` or budget). | exec result → `{val_scores, decision, best_so_far}` |
| **Debugger** | LLM (error handling) | On failure, read the traceback and produce a fix patch; retry up to K times, else route around (revert to last good + try a different idea). | error + patch → fix patch |
| **Solution store / memory** | Storage | Persist every attempt: spec, patch, metrics, status. Feed the proposer. MVP: a list/tree in a JSON file. | — |
| **Run logger** | Storage | Write one structured record per iteration (§7). This is a deliverable and scored. | — |

### Control flow (one iteration)
```
proposer(history, best) ──► change_spec
        │
        ▼
coder(change_spec, pipeline) ──► patch
        │
        ▼
executor(patch) ──► result
        │
   ┌────┴─────┐
 error       metrics
   │            │
debugger     evaluator ──► {scores, decision}
 (retry K)      │
   │       update store + log
   └──────► loop or STOP(converged/budget)
```

### Suggested repo layout
```
agent/
  loop.py            # orchestrator: the while-loop in §6
  proposer.py        # LLM call → change_spec
  coder.py           # LLM call → patch
  executor.py        # subprocess run + timeout + capture
  evaluator.py       # metric parse + convergence check
  debugger.py        # error → fix patch
  memory.py          # solution store (JSON-backed)
  logger.py          # JSONL run logs
  llm.py             # LiteLLM wrapper + token accounting
pipeline/
  train.py           # the recommender training+eval entrypoint the agent edits
  config.yaml        # config surface the agent mutates (FuxiCTR-style)
  eval_official.py   # organizer's scoring script (drop-in)
data/                # parquet splits (git-ignored)
runs/<run_id>/       # logs, patches, checkpoints, best submission
README.md
```

---

## 5. Tech stack (MVP)

| Role | Pick | Notes |
|---|---|---|
| Agent harness | **Thin custom loop** (recommended for MVP) or **fork AIDE** | A custom loop is faster to get end-to-end and easier to instrument for the autonomy/robustness scores. Adopt AIDE's tree search later if time allows. |
| Recommender models | **FuxiCTR** | Config-driven (DeepFM, DCNv2, DIN, MMoE, PLE). The agent edits YAML — cheaper and more robust than freeform code edits. Verify exact config keys against FuxiCTR docs. |
| LLM access | **LiteLLM** | One interface, model routing. Route Coder/Debugger to a cheap model; Proposer can use a stronger one. Meter tokens here. |
| HPO | **Optuna** (optional in MVP) | Skip for the first loop; add pruning once the loop works. |
| Tracking | **MLflow** | Params/metrics/artifacts per run; complements the JSONL log. |
| Data | **pandas** (fine at 1.4M) | Polars/DuckDB only needed for the bonus datasets. |
| Foundation | PyTorch (+ Lightning optional) | Standard. |

**Decision to make early:** custom loop vs. AIDE fork. Recommendation: **build the custom loop for the MVP** — you control the logging and recovery, which are exactly what's scored. Keep AIDE as a fast-follow if the custom loop's proposer underperforms.

---

## 6. The loop, in detail

### Shared state
```python
@dataclass
class Attempt:
    id: int
    parent_id: int | None        # for tree search later; MVP: previous id
    hypothesis: str
    target_stage: str            # "features" | "model" | "training" | "eval"
    change_spec: dict
    patch: str                   # unified diff or config delta
    status: str                  # "ok" | "error" | "recovered" | "reverted"
    val_scores: dict | None      # {"ndcg@10": ..., "recall@50": ...}
    error: str | None
    tokens: dict                 # {"in": int, "out": int}
    gpu_seconds: float
```

### Orchestrator (pseudocode)
```python
def run(budget):
    memory = Memory()                      # loads/creates solution store
    best = reproduce_baseline()            # M0/M1: confirm we match official baseline
    memory.add(best)
    stale = 0
    while not budget.exhausted() and stale < N:
        spec  = proposer(memory.history(), best)          # what to try + why
        patch = coder(spec, memory.current_pipeline())    # how to do it
        result, retries = execute_with_recovery(patch)    # executor + debugger
        attempt = evaluator.assess(spec, patch, result)   # scores + status
        memory.add(attempt); logger.write(attempt)
        if attempt.status in ("ok", "recovered") and improves(attempt, best, eps=EPS):
            best = attempt; stale = 0
        else:
            stale += 1
            memory.revert_to(best)         # route around: don't build on a regression
    finalize(best)                         # write submission + results table + totals
```

### `execute_with_recovery`
```python
def execute_with_recovery(patch, K=3):
    for i in range(K):
        r = executor.run(patch, timeout=TIMEOUT)
        if r.ok: return r, i
        patch = debugger.fix(patch, r.error)   # feed traceback back to LLM
    return ExecResult(ok=False, error="unrecovered after K retries"), K
```

### Prompt sketches (adapt, don't copy verbatim)
- **Proposer:** *"You are improving a KuaiRand-Pure click-ranking pipeline. Current best: NDCG@10=…, Recall@50=…. History of tried ideas and their deltas: [...]. Propose ONE next change grounded in a named recommender-systems method. Output JSON: {hypothesis, target_stage, change_spec}. Prefer ideas that target a different stage than the last two failures."*
- **Coder:** *"Apply this change_spec to the pipeline. Return only a unified diff / config delta. Do not change the eval script or touch the test split."*
- **Debugger:** *"This patch failed with the traceback below. Return a corrected diff. If the error is OOM, reduce batch size. If it's a missing column, inspect the schema in `data/schema.json`."*

### Convergence (Evaluator)
Stop when validation best has not improved by more than **`ε`** over the last **`N`** iterations (⚠️ CONFIRM both), or the compute/wall-clock budget is hit — whichever comes first. The submission is the **validation-best checkpoint at that point**. Do not submit a lucky mid-run peak.

---

## 7. Run-log schema (deliverable + scored)

Write `runs/<run_id>/iterations.jsonl` — one JSON object per line per iteration. This is how judges assess **autonomy** and **robustness**, so make errors and recoveries first-class, not hidden.

```json
{
  "iteration": 7,
  "timestamp": "2026-08-27T10:32:11Z",
  "hypothesis": "Add a PLE multi-task head predicting like + long_view alongside click; shared experts should lift click ranking.",
  "target_stage": "model",
  "change": {
    "type": "config_diff",
    "diff": "model: DeepFM -> PLE\nnum_experts: 4\naux_tasks: [like, long_view]"
  },
  "result": {
    "status": "recovered",
    "val": {"ndcg@10": 0.0421, "recall@50": 0.1897},
    "delta_vs_baseline": {"ndcg@10": 0.0037, "recall@50": 0.0112},
    "is_new_best": true
  },
  "events": [
    {"type": "error", "kind": "CUDA_OOM", "detail": "batch 4096"},
    {"type": "recovery", "action": "batch_size 4096->2048", "retry": 1, "outcome": "ok"}
  ],
  "cost": {"tokens_in": 3120, "tokens_out": 640, "gpu_seconds": 148.2},
  "interventions": 0
}
```

Also maintain a run-level summary: total iterations, total manual interventions, total tokens (in+out), total GPU-hours, final val scores + deltas.

---

## 8. Improvement ladder (seed / sanity-check for the Proposer)

The proposer should discover changes like these on its own, but use this list to (a) seed few-shot examples and (b) verify the agent is finding sensible moves. Ordered roughly cheap→involved:

1. **Reproduce baseline** — match the official score before changing anything.
2. **Feature engineering** — user-history sequence features, recency/temporal features (splits are date-based), user×category cross features.
3. **Architecture** — DeepFM → DIN (attention over history) → DCNv2 (explicit interactions).
4. **Multi-task (the KuaiRand insight)** — MMoE / PLE head also predicting `like`, `long_view`, `follow`. Only `click` is scored, but joint learning shares representation and usually lifts click. PLE targets the seesaw problem.
5. **Training strategy** — negative sampling scheme, loss choice, LR schedule, early stopping.
6. **(Stretch)** debiasing via randomized-exposure data.

Improvements need not be monotonic — expect the trajectory to fluctuate; what matters is sustained progress over the baseline.

---

## 9. Robustness requirements

The agent must handle difficulty without a human. At minimum:
- **Sandbox execution** — run generated code in a subprocess with a hard **timeout** and captured stdout/stderr; never `exec()` in-process.
- **Error → fix loop** — feed tracebacks back to the Debugger; retry up to K times.
- **OOM handling** — catch and halve batch size, retry (cheap, common, and a clean logged win).
- **Route-around** — if an idea can't be made to work or regresses, revert to the last good pipeline and try a different idea; never build on a broken/regressed state.
- **Never crash the whole run** — the orchestrator catches everything and logs it; a failed iteration is a logged event, not a process death.
- **Self-stop** — convergence + budget checks so no human decides when to end.

Robustness is judged on *recovery*, not on never failing. Surface and log the recoveries.

---

## 10. Build plan (milestones)

**M0 — Foundations (unblock).** Environment, data downloaded + converted to parquet, official eval script + baseline in hand (§2). Manually run the baseline pipeline and confirm you match its reported validation score. *Exit: baseline reproduced by hand.*

**M1 — One autonomous iteration.** Wire proposer → coder → executor → evaluator for a single pass, logged. No memory, no recovery yet. *Exit: agent proposes one change, runs it, logs one valid record.*

**M2 — Full loop.** Add memory/solution store, the while-loop, convergence detection, and the debugger retry/route-around. Inject a deliberate code error and a timeout to prove recovery. *Exit: agent runs to convergence unattended and beats baseline on val; recoveries visible in log.*

**M3 — Cost + polish.** Add LiteLLM model routing and token metering, MLflow, optional Optuna pruning. Produce the results table + token/GPU totals. Write the README + Devpost description. *Exit: all §1 done-criteria and §11 deliverables met.*

**M4 — Stretch (only if time).** Tree search over attempts; the proxy-fidelity path toward the 1k/27k bonus.

---

## 11. Deliverables checklist (maps to the challenge)

- [ ] **Devpost write-up:** how the solution addresses the problem; dev tools; APIs; libraries/frameworks; datasets.
- [ ] **Public GitHub repo:** commented code for all components; README with overview, setup, reproduce-steps, a limitations/future-work reflection, and team contributions.
- [ ] **Run & iteration logs:** the `iterations.jsonl` (§7) — hypothesis, code diff, metrics, error/recovery per iteration — plus a summary reporting the **number of manual interventions**.
- [ ] **Final submission + results summary:** final KuaiRand-Pure model output/checkpoint in the organizer schema; a results table with val-best NDCG@10 / Recall@50 and the **absolute delta over the official baseline**; total **token consumption** (in+out) and total **GPU-hours**.

---

## 12. Scoring levers (keep these in view while building)

| Criterion | Weight | MVP implication |
|---|---|---|
| Technical Execution | 35% | Beat the baseline **and** recover from failures gracefully (log the recoveries). |
| Innovation & Insight | 20% | Proposer grounds ideas in named methods; the multi-task angle is the headline insight. |
| Impact & Relevance (autonomy) | 20% | Minimize manual interventions; count and report them. |
| Feasibility (tokens + GPU) | 15% | Cheap model for routine coding; early-stop bad runs; don't over-train. |
| Presentation | 10% (final only) | The logs + a solution-tree view double as the story. |

---

## 13. Open questions / risks

- **Eval parity:** any mismatch with the official scoring script invalidates comparisons — adopt it verbatim, don't reimplement.
- **Config surface vs. freeform code:** editing FuxiCTR YAML is more robust but caps what the agent can invent; leave an escape hatch for freeform code edits on the training script when a change_spec doesn't map to a config key.
- **Proposer quality:** a weak proposer produces naive tweaks (hurts Innovation). If the custom loop's proposals are shallow, that's the trigger to adopt AIDE.
- **Budget unknown:** ⚠️ CONFIRM the compute budget early — it decides how many iterations and whether any subsampling is needed even on Pure.
- **Overfitting to validation:** the agent optimizes val; guard against val-overfit (the hidden test is the real score). Reasonable early stopping and not over-iterating on tiny deltas help.

---

*This brief covers the required KuaiRand-Pure MVP. Scaling to the 1k/27k bonus (proxy-fidelity funnel, DDP, cached columnar data, OOM-aware executor) is a separate phase — see the companion plan.*
