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

It is also not sufficient, which is the next section. Every metric here was computed by trusted
code and every digest verified — and the number the agent was handed still did not mean what the
agent thought it meant.

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

**What happened next is the part we are most pleased with, and it is one discovery, one confirmed
fix, and two honest negatives.**

*The fix changed how the agent reasons.* Same iteration slot, consecutive runs. Before: *"recency
weighting successfully measured 0.5754240304, matching official_fm_fold_B; recency weighting is
therefore excluded."* After: *"the parent scores 0.5713044 standalone against the 0.575424 Fold B
control."* The first is a model reasoning correctly from a false premise — the failure you cannot
debug by reading its output, because the output is impeccable. The second is the same model with a
truthful signal.

*Then the model got substantially better.* Every candidate to that point crossed all 38 feature
columns in one factorization-machine interaction, so 33 continuous aggregates shared a latent space
with the identity codes; the organizer FM crosses only categorical fields. Restricting the
interaction to the identity codes and keeping the aggregates as additive first-order terms moved
the deficit from **−4σ…−12σ to −1.12σ**. Not a bigger model — a better-specified one.

*Then we killed two of our own hypotheses.* Giving candidates user identity at prediction time was
structurally correct and **did not close the gap**. Letting a candidate ensemble itself came out
**flat**. We measured why: averaging the five official FM seeds on raw scores is worth +0.0000772,
on within-user rank percentiles +0.0005664. **86% of the ensembling gain needs an operation our
candidates cannot perform**, because prediction receives no user grouping.

**That is the finding we would carry to another project: an agent's capability boundary is a design
parameter, not an implementation detail.** Ours gives a candidate a feature matrix and nothing else
at prediction time — drawn that way for good isolation reasons, and it silently capped the
achievable score in three separate ways: no user-conditioned terms, no within-user normalisation,
and no early stopping on the scored split, which the baseline itself *does* get
(`starter_fm.py:701-708` keeps the best of 40 epochs measured on the split it then reports; the
published 0.6016 has the same property). We could not have found any of it from the agent's
transcripts. We found it by auditing what the agent was allowed to see.

**A code path that only runs on success accumulates defects invisibly.** Our promotion path went
unexecuted for twelve campaigns. Each time it finally ran it exposed a latent bug — most recently a
comparison of the organizer's float32 primary against our own float64 recomputation at
`abs_tol=1e-15`, **2.98e-8 apart, a single float32 unit in the last place**, which stranded the
first campaign ever to both promote a candidate and publish a bundle. Fixed and regression-tested.

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

**Twenty-five campaigns, eighteen of them producing a final bundle under the organizers' own
frozen rule (ε = 0.002, patience 3), with zero manual interventions in every single one.** Five
consecutive campaigns executed every generated candidate they produced — 22/22, 21/21, 22/22, 15/15
and 15/15 subprocess executions with empty stderr, and the final campaign admitted and trained all
six candidates it wrote with zero repairs and zero pre-execution rejections. Bundles are
organizer-valid: 170,588 rows, checker return code 0, replay verified.

**We do not claim an improvement over the baseline, and the reason is the most useful result we
have.** A candidate was promoted at +0.0002715, or 0.34 of one seed standard deviation. We reran
the same method rather than reporting it. Three measurements: **+0.34σ, −0.03σ, +0.36σ** — mean
+0.00018 against an ε of 0.002. Had we shipped the first as a win, our own next run would have
contradicted it four hours later.

Reported honestly, best first:

| Result | Public validation | vs organizer 0.6016 | Provenance |
|---|---:|---:|---|
| **Five-seed rank ensemble of the organizer FM** (submitted) | **0.6026034** | **+0.0010** | controller-side, **not agent-generated** |
| Shipped fallback `official-fm-fallback-seed-4` | 0.6020371 | +0.0004 | best-of-five **selected on this split** |
| Best agent candidate, fused | 0.6017246 | +0.0001 | within noise, 75% organizer FM by weight |
| Best agent candidate, **scored alone** | 0.5746261 vs control 0.5754240 | **−1.00σ** | our model, honestly measured |

Every caveat in that table is one we went looking for. The submitted ensemble is real and
reproducible — `python3 build_ensemble_submission.py` regenerates it from already-qualified
checkpoints with no new training, and the organizers' own checker passes it — but it is an
engineering property of *their* baseline, not our agent doing research, and it is below our own
materiality threshold. The shipped fallback's margin is a selection effect. The agent's headline
number is a blend. **And our agent, scored on its own, still does not beat the baseline** — though
it is now within about one standard deviation of a reference that gets to early-stop on the split
it is scored on, which our candidates structurally cannot.

We would rather hand a judge four rows with their provenance than one row without it.

**Resource accounting.** **6,284,739 LLM tokens** across 247 provider calls that returned a
usage block, at an estimated **$60.61**, covering every campaign in the series. **0.00
GPU-hours**, by construction rather than by approximation. **Zero manual interventions** in
every campaign.
The per-run breakdown, including the two accounting caveats that make the table reconcile, is
in [`docs/RESULTS.md`](RESULTS.md) section 4.

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
| **OpenAI-compatible Chat Completions**, two profiles | A frozen main/fallback provider chain over `POST {base_url}/chat/completions` with strict JSON-schema structured outputs, called directly over `urllib.request` — no vendor SDK and no agent framework. Early campaigns (runs 01–07) ran `deepseek/deepseek-v4-pro-0813`; the scored campaigns (runs 09–14) ran `openai/gpt-5.6-sol` as the main profile with `openai/gpt-5.6-terra` in the fallback slot. A terminal typed failure switches to the fallback exactly once and stays there, so a known-bad endpoint is not retried for every later operation. Each profile carries its own dedicated credential variable and its own pinned pricing block, and per-call input, cached-input, output, reasoning and total tokens are recorded. |
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
- [`docs/agent-memory-experiment.md`](agent-memory-experiment.md) — the cross-run memory experiment and its negative result
- [`docs/run-logs/`](run-logs/README.md) — rendered per-iteration run log
- [`docs/research/`](research/) — primary-source verification and implementation-readiness research
