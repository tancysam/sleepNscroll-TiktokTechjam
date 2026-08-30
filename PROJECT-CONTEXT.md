# KuaiRand Autonomous Research Agent — engineering brief

**Read this first. It is written to be the only context you need.**
Operational detail — launch commands, monitoring, how to grade a finished run — is in
`NEXT-SESSION.md` in the repo root.

*Written 2026-08-30 12:15 SGT. Supersedes the 2026-08-29 version, which is stale in three places
noted inline.*

---

## Your role

You are a **senior ML research engineer** joining a four-person team 48 hours before a competition
deadline. The system is built, the tests pass, and it runs end to end. It is not yet doing the one
thing it exists to do.

You are not here to redesign it. You are here to make the existing loop produce a working
candidate, and to report honestly on whether it did.

Three things about how this team works, learned the hard way:

1. **The starter kit is always right.** Where any document disagrees with
   `kuairand-starter-kit/evaluate.py`, the document is wrong. Two planning documents encoded the
   wrong task for days before anyone checked.
2. **Verify before relying.** Multiple confident assertions in this project turned out to be false
   on inspection — that `PROMPT_VERSION` could be bumped (four tests pin `_v1`), that
   `run_kind = "test"` was legal for a live config (`config.py` forbids it), that
   `tools`/`tool_choice` caused a provider 404 (measured: they don't). Read the enforcing code.
3. **Report the number honestly.** A fabricated improvement is worse than a small one. Hidden-test
   performance is organizer-only and cannot be claimed.

---

## Your environment — read this before you touch a file

**The user is on Windows 11. This project cannot run on Windows.** That is not a preference, it is
a hard blocker: `campaign/full_campaign_runtime.py` does an unguarded `import resource`, and
`execution/runner.py` uses `os.killpg`, `signal.SIGKILL` and `os.setsid`. All of those are
POSIX-only, and the platform guards in the code branch on `darwin` and `linux` only. There is no
Windows path through the runtime and adding one is not in scope.

The fix was **WSL2 with Ubuntu**, which is installed and working. Everything real happens in the
Ubuntu terminal, not in PowerShell and not in a Windows shell.

### There are two copies of this repo. Only one of them works.

| | Path | State |
|---|---|---|
| **WSL — the live one** | `~/sleepNscroll-TiktokTechjam` | current, has the 195 MB dataset at `.data/`, has the virtualenv, runs campaigns |
| Windows — stale | `C:\Users\Makendra Prasad\OneDrive - Singapore Management University\Desktop\tiktok jam\sleepNscroll-TiktokTechjam` | on `Maki` at `ba8fd55`, **18 commits behind**, **no dataset**, cannot run anything |

**This is the single easiest way to waste an hour.** If you read code and it does not match what
this brief describes, check which copy you are in. Everything you do belongs in the WSL copy.

The Windows Desktop folder is still useful for one thing: it holds the reference material and the
docs the user reads outside the terminal — `PROJECT-CONTEXT.md` and `NEXT-SESSION.md` are mirrored
there, alongside `WSL-SETUP.md`, `WHATS-GOING-ON.md`, the MLE-STAR paper, and cloned reference
repos (`aideml`, `openevolve`, `adk-samples`, `RD-Agent`, `IR-Benchmark`).

### If you are Claude Code running on the Windows side

You reach the working repo through `wsl.exe`. Two things will bite you:

1. **Quoting.** `wsl.exe -e bash -lc '...'` breaks constantly on nested quotes — a Python triple
   single quote or a plain apostrophe inside the command is enough to produce
   `unexpected EOF while looking for matching`. For anything beyond a one-liner, write a Python
   patch script to a file, `cp` it into WSL, and run it there. Anchor every substitution and assert
   the match count is exactly 1.
2. **`uv` is not on the non-login PATH.** `wsl.exe -e bash -s` with a heredoc will fail with
   `uv: command not found`. Use `wsl.exe -e bash -lc` (note the `-l`) for anything that runs `uv`.

Windows paths are visible from inside WSL under `/mnt/c/...`, which is how files move between the
two sides.

### Running things

Dependencies and tests go through `uv`, always from the WSL copy:

```sh
cd ~/sleepNscroll-TiktokTechjam
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural pytest -q
UV_CACHE_DIR=.uv-cache uv run --locked ruff check src tests
UV_CACHE_DIR=.uv-cache uv run --locked mypy src tests
```

Two failures are **pre-existing and not yours**: `test_starter_fm.py::
test_first_update_matches_untouched_organizer_float32_golden` (an arm64 float32 golden; we are on
x86), and a mypy `Statement is unreachable` at `execution/runner.py:909` (inside a
`sys.platform != "darwin"` branch, so it is unreachable on Linux and reachable on the macOS the
code also targets). Expect **1479 passed, 1 failed**.

---

## The problem

**TikTok TechJam 2026, Track 2: Autonomous ML Research Agent for Recommender Systems.**
**Deadline: Tuesday 2026-09-01, 12:00 SGT.** Organizers are strict on it.

Build an agent that autonomously runs the ML engineering loop — read the problem, inspect the data,
engineer features, train, evaluate, reflect, repeat — on KuaiRand-Pure. **The agent and its process
are graded, not the trained model.**

### How it is scored

| Criterion | Weight | What it actually rewards |
|---|---|---|
| Technical Execution | 35% | Delta over baseline on the **converged** result, plus robustness and recovery |
| Innovation & Problem Insight | 20% | What the agent chose to target and **why** — reasoning, not implementation |
| Impact & Relevance (Autonomy) | 20% | Fewer manual interventions is better |
| Feasibility & Practicality | 15% | Total LLM tokens plus GPU hours (ours is CPU-only, so GPU hours = 0) |
| Presentation | 10% | Final event only |

Judges score the **converged** result. Terminating on `ITERATION_CAP` or `HARD_DEADLINE` rather
than `CONVERGED` is worth measurably less.

### Required deliverables

Devpost writeup; public repo with README (overview, setup, reproduce steps, limitations, team
contributions); **per-iteration run logs** (hypothesis, code diff, metrics, error and recovery
events); manual intervention count; final submission CSV; results table with absolute delta vs
baseline; total token consumption and GPU hours.

---

## The task spec — authoritative

From the hash-pinned `kuairand-starter-kit/README.md`. **This supersedes the prose problem
statement**, which describes a different task.

| | |
|---|---|
| Target label | **`long_view`** (binary). **NOT `click`** — stale briefs say `click`, they are wrong |
| Metrics | **GAUC** and **nDCG@5**; primary = arithmetic mean |
| Task shape | **Within-user ranking** of each user's own logged impressions. Not catalogue retrieval |
| Baseline to beat | primary **0.6016 validation**, 0.5946 hidden test |
| Convergence | ε = 0.002, N = 3 consecutive non-material iterations |
| Oracle ceiling | primary **0.8645**, not 1.0 |
| Splits | train 2022-04-08→04-21 (1,141,112 rows) · validation 04-22→04-28 (124,909) · hidden test 04-29→05-08 (170,588) |

**Always compare against 0.6016.** That is the validation primary, which is the split every
promotion decision in this campaign is measured on. 0.5946 is the held-out test number that only
the organizers can compute.

27.1% of test users have zero positives (nDCG permanently 0, still counted) and 9.2% are all
positive (permanently 1), so only 63.7% are GAUC-eligible. The baseline has already captured 30.7%
of the reachable range — remaining headroom is 0.27, not 0.41. FM seed-to-seed std is 0.0008, which
is why ε = 0.002 is roughly 2.5σ.

**Hard rule:** no external training data, no pretrained weights trained on these benchmarks' test
labels. Libraries, papers and public solutions are all allowed.

---

## Architecture

Two planes with a narrow typed wire between them.

**The trusted controller** (~63k lines src, ~35k tests) owns dataset identity, split boundaries,
the byte-protected organizer scorer, the execution sandbox, all budgets, convergence arithmetic,
the durable campaign record, finalization and replay.

**The research model** reaches it through exactly four typed methods in
`src/kuairand_agent/research/interface.py`: `propose`, `implement`, `repair`, `reflect`. It has no
filesystem, shell, network, evaluator, credential or label authority. It returns complete file
contents, never patches. It never computes or reports a metric.

This separation is the project's strongest asset. **Do not erode it to make something pass.**

Three-tier evaluation gate: smoke → inner temporal folds → rationed outer promotion (capped at 6),
with matched-seed confirmation across seeds 0/1/2.

Provider calls go over `urllib.request` directly with strict JSON-schema structured outputs. **No
LangChain, no LiteLLM, no OpenAI SDK, no agent framework.** Runtime dependencies are `numpy` and
`psutil`; `lightgbm` optional, `torch` explicitly excluded.

### The candidate contract — this is what keeps failing

- `candidate.py` is a **protected wrapper**. The model may not return it. It owns argument parsing,
  capability loading, checkpoint I/O and score writing.
- The model writes **`model_impl.py`**, defining `validate_config`, `train_model`, `predict_scores`
  and `training_diagnostics`.
- `files_expected` is the **final tree manifest** and **must include `candidate.py`**
  (`provider._parse` validates this). The **response** is an **overlay** and **must not** contain
  `candidate.py`. Two different things. Confusing them cost a live campaign four proposals.
- Helper modules import **by plain name** (`import pairwise_sampler`), never relatively. A relative
  import is invisible to the reachability walk that decides materiality, *and* the candidate runs
  as a script so `from . import helper` fails at execution too.
- `require_material_executable_change` compares **only top-level `def`/`async def`/`class` AST
  nodes** in the reachable file set, after stripping docstrings. Changing a module-level constant,
  a docstring, or code under `if __name__` counts as **no change at all**. `material_symbols` must
  be bare names (`fit_scores`), never qualified.

---

## Where things actually stand

### Proven

- **The live autonomous loop runs and finalizes.** *(This corrects the 2026-08-29 brief, which said
  no live provider campaign had ever completed.)* Runs `maki-overnight-02`, `03` and `06` all
  finalized with real LLM calls. Run 06 made 10 provider calls including **2 accepted repairs**, so
  propose → implement → repair → reflect works end to end against a live model.
- **Submissions are valid.** 170,588 rows, organizer checker rc 0, replay verified.
- **GPU hours = 0.** CPU only, by design.
- **The survivability work landed** (commit `ce1c692`, branch `Maki`). See §"What was fixed" below.

### Not proven — this is the gap

**No generated candidate has ever trained and been scored.** Every run so far ends one of two ways:
the agent produces nothing and the official-FM fallback ships, or it produces code that is rejected
by the static gates before it ever executes.

Best evidence, from `maki-overnight-07` (pre-fix, 1h14m, killed):

```
propose    accepted 3
implement  accepted 2, failed 3, malformed 1

candidate-01  declared_symbol_unchanged  repairs=0
              declared material symbol(s) did not change executable source: ['confirmno']
candidate-02  invalid_python             repairs=0
              invalid Python in 'model_impl.py' at line 1: invalid syntax
```

Two signals. **`repairs=0` on both** — the model was emitting `maximum_repairs: 0`, so the first
static-gate miss ended each branch with no chance to self-correct. The prompt now explicitly tells
it to set 2. And **`['confirmno']` and `invalid syntax at line 1`** are truncation artifacts:
output quality is the binding constraint.

### Also measured, and honest about it

The pairwise FM direction was trained properly and swept (`runs/pairwise-sweep.json`). Best primary
**0.6015003** against a local baseline of **0.6015722** — a **marginal regression**, not a win.
Caveats that must appear alongside it: hyperparameters were selected on the reported split, one
seed, best-of-12 against a five-seed mean. And `candidates/pairwise_fm.py` is imported by nothing
in `src/`, so this measures a direction rather than demonstrating agent behaviour.

---

## What was fixed on 2026-08-29/30

Seven campaigns had failed. Each time the proximate cause was fixed and the run relaunched. The
audit found the real problem was structural:

1. The scientific clock advances `elapsed_seconds` **only by candidate subprocess wall time**.
   Provider latency, materialization and reflection are invisible to it. `overnight-05` alone had
   **4011 seconds** of unaccounted provider latency — more than the entire 3600s finalization
   reserve, in a partial run.
2. So the loop believed it had slack it did not have and called past the reserve, where the
   provider raises a deadline error **by design**.
3. **Nothing caught it.** The followup loop caught only `LiveResearchBranchRejected`; reflection was
   unguarded entirely.
4. `cli.py` runs research and finalization as sequential statements in one `try`, so anything
   escaping research meant **finalization never ran**. Five hours of work, no bundle.

Fixed: the loop consults the real engine clock every iteration; provider errors are caught,
recorded to the durable rejection ledger and returned normally so finalization runs; reflection
degrades to a named substitute. Three separate ways the *fallback path itself* could lose a
submission are closed. Prompt corrections: the baseline number, removal of stale I/O guidance the
wrapper owns, the relative-import fix, and an explanation of `maximum_repairs`. 14 new tests.

---

## Your mission, in priority order

### 1. Get a candidate to train

This is the whole job. Launch a campaign (see `NEXT-SESSION.md` §3), then check the rejection
journal:

```sh
for f in runs/maki-overnight-09/production/generated-source/controller-rejection-journal/*.json; do
  python3 -c "
import json,sys
v=json.load(open(sys.argv[1]))['record']['values']
print(v.get('candidate_id'), '|', v.get('root_failure_code'), '| repairs=', v.get('repairs_attempted'))
print('   ', str(v.get('root_failure_diagnostic'))[:160])
" "$f"
done
```

**Check `repairs_attempted` first.** If it is still 0, the model is ignoring the instruction and the
next move is to stop trusting it with that field — clamp it controller-side, or drop it from the
schema and use the configured `max_repairs_per_experiment` (currently dead code).

If repairs are happening but candidates still fail, read the diagnostics. Each one is a prompt fix,
not a code fix, unless it names a genuine contract bug.

### 2. Write up whatever happens

`docs/RESULTS.md` §3.2 still needs the pairwise sweep result with the caveats above. The
per-iteration run log is a **required deliverable** and is generated from
`runs/<id>/production/generated-source/iteration-NN-lineage.json` — verify it renders.

### 3. Push

`Maki` has ~11 unpushed commits. **Do not push to `main`.**

---

## Constraints and gotchas

- **Everything runs in WSL2 Ubuntu, never Windows, and only in the WSL copy of the repo.** See
  "Your environment" above. *(The 2026-08-29 brief lists WSL as not yet installed — it is now set
  up and working.)*
- **`convergence_patience = 3` and `epsilon = 0.002` are frozen benchmark constants**, not tuning
  knobs. They live beside `target = "long_view"` in `config.py` as `FROZEN_*` values, the validator
  rejects anything else, and they are stamped into `benchmark_digest`. Changing them means you are
  no longer running the stated benchmark.
- **Run directories are never reused.** Every attempt needs a fresh `--run-dir`.
- **Only one campaign at a time.** Two campaigns write the same `runs/outer-query-ledger.sqlite3`
  and fight for the CPU. This has already happened once.
- `run_kind` and `provider` are strictly paired: `autonomous` requires `openai`, `demo`/`test`
  require `scripted`. **There is no legal way to label a live run as a test** — separate a probe
  from the scored campaign by run directory, and count its tokens in the total.
- `finalization_reserve_seconds` has a hard floor of 3600 and must be less than
  `wall_clock_seconds`; also `default_timeout + reserve <= wall_clock`.
- **Never paste API keys into chat.** They live in `.env.local`, gitignored and `chmod 600`.
  `.env.example` is tracked and its key fields must stay blank.
- **The user launches campaigns themselves.** Give them the command; do not run it for them, and
  never `nohup` a long job on their behalf — a detached process survives a permission denial.

### Provider facts that were measured. Do not re-litigate.

- **`tools: []` and `tool_choice: "none"` do NOT cause the OpenRouter 404.** Measured 2026-08-30: a
  strict `json_schema` request returns 200 with them, without them, with either alone, and with
  `require_parameters` off. The 404 was an account-level data policy on an old key.
- **Provider dialects differ.** OpenRouter honours `reasoning: {effort, exclude}`. TokenRouter
  honours only `thinking: {type: enabled|disabled}` and ignores `budget_tokens` at realistic sizes
  — a 4096 budget still returned 11k–26k reasoning tokens.
- **Do not disable thinking.** With it off the model stopped writing code and returned 18-character
  stubs, with `import numpy as np` written into `config.json`. Reasoning stays on; the cost is paid
  in output budget, which is why `max_output_tokens` is 65536 and the timeout is 600s.
- **Never probe provider behaviour with a trivial prompt.** A budget that appears to work on a toy
  request is ignored at real payload size.

---

## Where the score can actually come from

The organizers published their own measured dead ends and open directions. This is already in the
prompt.

**Dead ends, measured by them:** all 13 static feature domains scored 0.5940 against 0.5950 for the
5-field baseline — worse. Embedding dim k = 8/16/32 scored 0.5895/0.5902/0.5887 — flat. Their
conclusion: the bottleneck is neither features nor capacity. And **purely user-side first-order
terms contribute exactly zero**, because ranking is within-user, so any term constant across a
user's rows cannot reorder them.

**Open, in their priority order:** (1) **the loss function** — training is pointwise logloss while
GAUC and nDCG are ranking metrics, and this is the single largest known opportunity; (2) user
behaviour sequences; (3) multi-task on the other feedback signals; (4) censored watch-time
regression; (5) architecture swaps (deprioritised); (6) temporal drift; (7) the randomized exposure
log as an unbiased **validation** signal only.

`docs/research/implementation-readiness-research.md` §4.1 derives the metric-matched pair sampler
from the GAUC formula: sample a positive uniformly from GAUC-eligible users, then a negative
uniformly from that same user's logged negatives, and optimise `softplus(-(s_pos - s_neg))`. It is
implementation-ready.

---

## Team

| Person | GitHub | Machine | Owns |
|---|---|---|---|
| Samuel Tan | `tancysam` | macOS | Built most of the system. Has the working env and dataset |
| Sean Koh | `TerrorByter` | Mac/Linux | Architecture review, live provider testing, the repair loop |
| Makendra (the user) | `makilover3000` | Windows 11 + WSL2 | Prompts, benchmark contract, deliverables, writeup |

Makendra is new to recommender systems. **Lead with plain English, give an opinion rather than a
survey of options, and ask before committing.**

---

## Repo orientation

```
src/kuairand_agent/
  campaign/full_campaign_runtime.py   the autonomous research loop
  campaign/scientific.py              the inner campaign and its clock
  campaign/convergence.py             the epsilon/patience arithmetic
  research/prompts.py                 every instruction the model sees
  research/provider.py                HTTP, payload shape, retries, provider dialects
  research/production.py              lineage construction, branch rejection
  finalization/production.py          bundle, report, the official-FM fallback
  finalization/iteration_evidence.py  recovers the per-iteration trajectory
config.py                             FROZEN_* benchmark constants live here
configs/full-pure.toml                6h budget, 1h reserve, provider profiles
scripts/run_full_campaign.sh          the launcher
docs/RESULTS.md                       the deliverable writeup
runs/maki-qualification               the qualified baseline every run depends on
NEXT-SESSION.md                       launch, monitor and grade a run
```
