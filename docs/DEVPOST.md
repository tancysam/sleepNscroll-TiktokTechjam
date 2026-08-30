# Devpost submission — Autonomous ML Research Agent for KuaiRand-Pure

Track 2 · Team **sleepNscroll** · Benchmark: KuaiRand-Pure (required)

---

## What we built

An autonomous ML research agent that runs the machine-learning engineering loop on KuaiRand-Pure
by itself: read the problem, inspect the data, form a hypothesis, write the code, train, evaluate,
reflect, and iterate — until the organizers' own convergence rule says stop.

The thing being graded is the *agent and its process*, not a trained checkpoint. So the design
question we cared about was not "which model scores best" but **how much authority you can safely
give an LLM inside a scientific loop, and how you prove it did not cheat.**

## How it addresses the problem statement

The challenge asks for an agent that reproduces an official baseline, iterates on it autonomously
using only train and validation data, and drives the score up. Our answer separates the system
into two planes with one narrow typed wire between them.

**The trusted controller** owns everything that could compromise a result: dataset identity, split
boundaries, the byte-protected organizer scorer, the execution sandbox, every budget, convergence
arithmetic, and the durable campaign record.

**The research model** reaches it through exactly four typed methods — `propose`, `implement`,
`repair`, `reflect`. It has no filesystem, shell, network, evaluator, credential, or label
authority. It cannot run code. It cannot see a label. It cannot compute or report a metric.

One iteration: the model receives a leakage-safe context (benchmark contract, train-only EDA,
label-free validation-input EDA, method cards, prior outcomes, remaining budget) and returns a
falsifiable hypothesis naming one principal change. It then returns *complete candidate source
files* — never patches, so there is nothing ambiguous to apply. The controller materializes them
into a disposable directory, computes the diff itself, runs static gates, trains the candidate in
a sandbox, and scores it with the untouched organizer evaluator through a three-tier gate: a smoke
run, then inner temporal folds, then a rationed outer validation query. Promotion is decided by
the measured number. The model then reflects on the trusted result and chooses what to try next.

**The model proposes; code decides.** No metric in this system is ever self-reported. That single
rule is what makes the run log trustworthy evidence rather than a transcript of claims.

## What we think is interesting about it

**Our agent was being lied to by its own evaluation harness, and we caught it by auditing the
harness instead of the agent.**

For three campaigns the records showed generated candidates landing at almost exactly the
baseline's score. Several came back *bit-identical* to it. We wrote that up as a finding about the
task — with a median slate of four impressions and only 63.7% of users GAUC-eligible, there are
genuinely few distinct orderings available, so different models plausibly induce the same one. It
was a tidy, and completely wrong, explanation.

The controller rank-fuses every candidate's predictions with the official FM baseline over a fixed
five-point weight grid, picks the best blend on an inner screen, and freezes it. The score written
into the record is the *blend's*. When the selector picks weight `0.0` for the model, it discards
the model and scores the baseline's own prediction vector — so identical metrics are a certainty,
not a coincidence. The proof is a digest: `scored_prediction_digest` is the same value
`41629b9c856e1921…` in every such case, and never equals the raw generated prediction.

Reading the grid point the selector *didn't* choose gave us each candidate's own score for the
first time. Every generated candidate we had ever run was **4σ to 12σ below the baseline
standalone**. We had not been narrowly missing an improvement for three campaigns; we had never
been in contention, and fusion had been substituting the baseline's score for ours the whole time.

Two causes, both structural:

- **No user identity at prediction time.** Candidates received user grouping during training and a
  bare feature matrix at prediction. They could not evaluate a user-conditioned term when it
  mattered, which rules out the user×video and user×author crosses that are where a factorization
  machine's ranking power actually lives. We were asking the agent to beat a model while
  withholding that model's main mechanism.
- **The agent was never told fusion existed.** The word appeared nowhere in its briefing, and the
  primary returned to its reflection step was the blend's. One iteration recorded in its own
  reasoning that a direction *"successfully measured 0.5754240304, matching official_fm_fold_B"*
  and dropped it — reading "my model was thrown away" as "my model tied the baseline." It then
  reasoned correctly from a false premise, which is the failure mode you cannot debug by reading
  the model's output.

The lesson generalises past this competition. We built elaborate guarantees that the agent could
not tamper with its evaluator — hash-pinned scorer, no filesystem, no label access — and every one
of them held. What we did not check was whether the evaluator was telling the agent the truth. A
feedback channel that silently floors a bad result at the baseline's score is not an integrity
violation; nothing is compromised, every digest verifies, and the agent's reasoning is impeccable
on the numbers it is given. It is simply wrong, and it is invisible from inside the loop. **An
agent's feedback signal deserves the same adversarial scrutiny as its permissions.**

Both are fixed, both are one commit each, and the diagnostic that found it ships in the repo as
`fusion_audit.py` so any reviewer can rerun it against our recorded campaigns.

**We deliberately did not use an agent framework.** No LangChain, no LiteLLM, no orchestration
library, not even the `openai` SDK. Provider calls are `urllib.request` from the standard library
against an OpenAI-compatible Chat Completions endpoint with strict JSON-schema structured
outputs. The two runtime dependencies are `numpy` and `psutil`. This was a considered choice: AIDE's own published benchmarking found that plain
tree search over code outperformed a LangChain-based agent on this class of task, and every
framework layer we skipped is a layer that cannot silently take authority we did not intend to
grant.

**We taught the agent the organizers' negative results.** The organizers published which
directions they had already measured as dead — adding all 13 static feature domains scored 0.5940
against 0.5950 for the 5-field baseline, and embedding dimension 8/16/32 barely moved the score —
along with the structural fact that purely user-side first-order features contribute *exactly
zero*, because ranking happens within a user and a term constant across that user's rows cannot
reorder them. Rather than build an ablation loop to rediscover this empirically, we put it in the
agent's briefing, together with the real ceiling (a perfect oracle scores 0.8645, not 1.0) and the
ranked list of untried directions. Target selection is 20% of the grade and is judged on what the
agent chose to target and why. Spending GPU-hours to re-derive published negative results is the
opposite of insight.

**The most instructive bug was in our own instruction surface.** Our static gates enforce 44
constants on generated code — forbidden import roots, reserved basenames, trusted path prefixes,
forbidden calls, size limits — but the implementation request transmitted exactly three of them.
Every model we tried (gpt-4o, DeepSeek, gpt-5.4-mini) failed the contract, and we initially read
that as a model-capability problem. It was not: the models were being rejected for rules they were
never told. The subtlest was a material-change check that follows imports from `candidate.py`,
strips docstrings, and compares only top-level function and class ASTs — so a model that carefully
tuned a module-level constant and declared it as its change was rejected for "claiming changes it
did not make," which was true but unhelpfully phrased. The prompt now renders its constraint text
*directly from the enforcement constants*, so instruction and enforcement cannot drift. In an
agent system the prompt is part of the interface and deserves the same rigour as a schema.

**Evaluation integrity is structural, not promised.** Current research on ML-engineering agents
identifies two compromise vectors: evaluator tampering and train/test leakage. Our agent is
incapable of both by construction. The scorer is hash-pinned and its filenames are in the
generated-code forbidden set; generated code has no filesystem or subprocess authority; candidates
receive phase-scoped numeric capabilities rather than the raw archive, and no final-period label
artifact is ever materialized. Metamorphic tests assert the causal properties directly — changing
a future outcome must not alter an earlier feature.

**Zero GPU-hours.** Not an approximation. Every configuration is CPU-only, and the locked
environment explicitly excludes the optional neural dependency group so an inherited `torch`
install cannot silently change the execution profile.

## Honest status

**Six live-provider autonomous campaigns completed end to end**, every one terminating on
`converged` under the organizers' own frozen rule (ε = 0.002, patience 3) with **zero manual
interventions**. Five published organizer-valid submission bundles: 170,588 rows, checker return
code 0, replay verified, byte-identical SHA-256 `e12746ae…` across all five.

**We do not claim an improvement over the baseline, and the reason is the most useful result we
have.** One run promoted a generated candidate at 0.6017118 against an incumbent of 0.6014403 —
+0.0002715, or 0.34 of one seed standard deviation. We reran the same method rather than reporting
it. It landed at 0.6014124, on the other side of zero. That is what a non-significant margin does
when you replicate it, and it is why the number is not in our headline.

Reported honestly, best first:

| Result | Public validation | vs organizer 0.6016 | Provenance |
|---|---:|---:|---|
| Five-seed rank ensemble of the organizer FM | 0.6026034 | +0.0010 | controller-side, **not agent-generated** |
| Shipped fallback `official-fm-fallback-seed-4` | 0.6020371 | +0.0004 | best-of-five **selected on this split** |
| Best agent candidate (run 13, fused) | 0.6017118 | +0.0001 | within noise, 75% organizer FM by weight |

Every caveat in that table is one we went looking for. The ensemble is real and reproducible but
it is an engineering property of *their* baseline, not our agent doing research, and it is below
our own materiality threshold. The shipped fallback's margin is a selection effect. And the agent's
own number is a blend.

Hidden-test performance is organizer-computed and is neither known nor claimed. Full detail,
including a section on what is *not* demonstrated, is in [`docs/RESULTS.md`](RESULTS.md).

We would rather report a small honest number than a large unverifiable one.

## Development tools

| | |
|---|---|
| Editors / assistants | VS Code; OpenAI Codex and Anthropic Claude Code as coding assistants |
| Environment & packaging | `uv` (locked, hash-pinned resolution), Hatchling, CPython 3.12.13 |
| Quality gates | pytest 9.1.1, mypy 1.20.2 (strict, `warn_unreachable`), ruff 0.15.22, pytest-cov |
| Version control | Git / GitHub |
| Platforms | macOS and Linux (the runtime is POSIX-only; WSL2 on Windows) |

## APIs used

| | |
|---|---|
| **OpenAI-compatible Chat Completions**, two profiles | A frozen main/fallback provider chain over `POST {base_url}/chat/completions` with strict JSON-schema structured outputs. We run `deepseek/deepseek-v4-pro-0813`, served by OpenRouter as the main profile and TokenRouter as the fallback; a terminal typed failure switches to the fallback exactly once and stays there, so a known-bad endpoint is not retried for every later operation. Called directly over `urllib.request` — no vendor SDK and no agent framework. Each profile carries its own dedicated credential variable and its own pinned pricing block, and per-call input, cached-input, output, reasoning and total tokens are recorded. Choosing an open-weights model behind a provider chain, rather than a frontier model behind one endpoint, was a deliberate cost and availability decision. |
| Zenodo | One-time hash-verified dataset acquisition (record 10439422). Not called at runtime. |

No other network access exists anywhere in the system. Generated candidate code cannot reach the
network at all — `socket`, `urllib`, `http`, `requests`, `httpx`, `aiohttp` and `ftplib` are
rejected statically before execution.

## Libraries and frameworks

**Runtime (all of them):** `numpy` 2.5.2, `psutil` 7.2.2.

**Optional research backends:** `lightgbm` 4.7.0 (`research-tree`, used for the LambdaRank
branch); `torch` 2.13.0 (`research-neural`, explicitly excluded from campaign runs so the
environment identity cannot change silently).

**Deliberately not used:** LangChain, LiteLLM, the OpenAI SDK, FuxiCTR, RecBole, TorchRec, MLflow,
Optuna, pandas, scikit-learn. Several were evaluated and rejected — for example, FuxiCTR's `gAUC`
weights each eligible user's AUC by group row count, whereas the organizer evaluator weights by
positive count, so adopting it would have silently optimized a different quantity than the one
being scored.

## Datasets and assets

| | |
|---|---|
| **KuaiRand-Pure** (Kuaishou) | The only training data used. 1,141,112 train rows (2022-04-08→04-21), 124,909 validation rows (04-22→04-28), 170,588 hidden-test rows (04-29→05-08). Acquired from Zenodo record 10439422 and verified by hash before extraction. |
| Organizer starter kit | `evaluate.py`, `data.py`, `baseline.py`, `submit.py`, `baseline_scores.json`, `ablation_features.py` — retained byte-for-byte as immutable, SHA-256-pinned references. `evaluate.py` is the sole scoring authority and is never modified, shadowed, or reimplemented. |
| Blocked by our own policy | `video_features_statistic.csv` (contains outcome aggregates over a period overlapping evaluation dates, with no per-row as-of cutoff) and `log_random_4_22_to_5_08_pure.csv` (randomized-exposure log; permitted-use and cutoff semantics are not pinned by the benchmark contract). Both are blocked by default in the field-role registry. |

**No external training data was used.** No pretrained weights of any kind were used. Nothing was
trained on, joined with, or pre-trained against any dataset outside KuaiRand-Pure.

## Repository

<https://github.com/tancysam/sleepNscroll-TiktokTechjam>

- [`README.md`](../README.md) — overview, setup, reproduction, limitations, contributions
- [`docs/RESULTS.md`](RESULTS.md) — results, run logs, resource accounting, integrity analysis
- [`plan.md`](../plan.md) — full architecture and design rationale
- [`docs/research/`](research/) — primary-source verification and implementation-readiness research
