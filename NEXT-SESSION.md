# Start here — session handoff

**Written 2026-08-30 ~12:00 SGT. Submission deadline: Tue 2026-09-01 12:00. About 48 hours left.**

If you are a fresh Claude Code session, read this file top to bottom before touching anything.
It is written to be the only context you need.

---

## 1. What this project is

TikTok TechJam 2026, **Track 2**. We are not building a recommender by hand. We are building an
**autonomous ML research agent** that proposes hypotheses, writes model code, runs experiments,
reads its own results, and iterates — on the KuaiRand-Pure dataset.

- **Task:** within-user ranking. Label is **`long_view`**. *(Some older briefs in the repo say
  `click`. They are stale. It is `long_view`.)*
- **Metrics:** GAUC and nDCG@5. The **primary** score is the mean of the two.
- **The number to beat:** the official FM baseline scores **0.6016 on validation**. That is the
  split every promotion decision is measured on. The same baseline scores **0.5946 on the
  organizers' held-out test period**, which only they can measure. When comparing anything, use
  **0.6016**.
- **Ceiling:** an oracle reaches 0.8645, so there is real headroom.
- **Convergence:** epsilon = 0.002, patience N = 3.

### The architecture in one paragraph

Two planes. A **trusted controller** (our code, deterministic, audited) and an **LLM** that is
reached through exactly four typed methods: `propose`, `implement`, `repair`, `reflect`. The LLM
never touches the filesystem, never parses a protocol, never picks a metric. It returns typed JSON.
The controller materializes it, runs static gates, executes it in a sandbox, scores it, and decides.
This separation is the project's best asset — do not erode it to make something pass.

### The candidate contract (this trips up every model, including you)

- `candidate.py` is a **protected wrapper**. The LLM may not return it. It owns argument parsing,
  capability loading, checkpoint I/O, and writing scores.
- The LLM writes **`model_impl.py`**, which defines four functions: `validate_config`,
  `train_model`, `predict_scores`, `training_diagnostics`.
- `files_expected` in a proposal is the **final tree manifest** and **must include `candidate.py`**
  (`provider._parse` validates this). The **response** is an **overlay** and **must not** contain
  `candidate.py`. These are two different things. Confusing them cost a live campaign four
  proposals — do not "simplify" this.
- Helper modules must be imported **by plain name** (`import pairwise_sampler`), never relatively.
  A relative import is invisible to the reachability walk that decides materiality, *and* the
  candidate runs as a script so `from . import helper` also fails at execution.

---

## 2. Where things stand right now

**Branch: `Maki`. 10 commits ahead of `origin/Maki` and unpushed.** Do not push to `main`.

Everything is committed and green:

```
suite   1479 passed, 21 skipped, 1 failed
        the 1 failure is pre-existing: tests/unit/test_starter_fm.py::
        test_first_update_matches_untouched_organizer_float32_golden
        (an arm64 float32 golden; we are on x86 WSL). Not ours, do not chase it.
ruff    check clean, format clean
mypy    1 error in src/kuairand_agent/execution/runner.py:909 "Statement is unreachable"
        pre-existing and platform-dependent: the line is inside a `sys.platform != "darwin"`
        branch, so it is unreachable on Linux and reachable on the macOS the code also targets.
        Not ours.
```

> **A campaign is running right now.** `runs/maki-overnight-09` was launched at 12:29 SGT on
> 2026-08-30 on the corrected code, with both provider keys verified live. Do not launch another
> one — only one campaign may run at a time. Check on it with
> `ls runs/maki-overnight-09/final/` (empty means still going) and grade it with the ladder in
> section 3. If it is no longer in `ps -eo pid,args | grep "[.]venv/bin/kuairand-agent"` and there
> is no `final/` directory, it died: read `logs/overnight-09.log`.

Two earlier campaigns were stopped at ~11:52 today:

| Run dir | Code | Status |
|---|---|---|
| `runs/maki-overnight-07` | pre-fix | ran 1h14m, stopped. Evidence kept — see §6. |
| `runs/maki-overnight-08` | post-fix | ran 3 minutes, stopped. Partial, ignore it. |

**Both run dirs are left on disk deliberately as evidence. Start a new run in a new directory
(`maki-overnight-09`) rather than reusing either.**

**API keys are configured and both were verified live at 11:47 today** — main and fallback both
return HTTP 200 on a strict `json_schema` request. They live in `.env.local`, which is gitignored
and `chmod 600`. **Never print a key into the chat.** `.env.example` is tracked in git and its key
fields must stay blank.

---

## 3. How to run the campaign

Everything runs inside **WSL2 Ubuntu**, not Windows. The runtime uses `resource`, `os.killpg`,
`SIGKILL`, and `os.getpgid`, which are POSIX-only. That is the whole reason WSL exists here.

Open the Ubuntu terminal and run:

```sh
cd ~/sleepNscroll-TiktokTechjam
mkdir -p logs
nohup sh scripts/run_full_campaign.sh \
  configs/full-pure.toml \
  "$HOME/sleepNscroll-TiktokTechjam/runs/maki-qualification" \
  "$HOME/sleepNscroll-TiktokTechjam/runs/maki-overnight-09" \
  > logs/overnight-09.log 2>&1 &
```

That is it. `nohup ... &` detaches it, so you can close the terminal and it keeps running. The
script sources `.env.local` itself and syncs dependencies before starting.

**It runs for 6 hours** (`wall_clock_seconds = 21600`), of which the **last hour is reserved for
finalization** (`finalization_reserve_seconds = 3600`). So roughly 5 hours of research, then it
writes the submission bundle. Leave the laptop on and plugged in. It will use up to 4 threads and
16 GB.

**Only ever run one campaign at a time.** Two campaigns write the same
`runs/outer-query-ledger.sqlite3` and fight over the CPU. That is exactly what happened today.

### Checking on it

```sh
cd ~/sleepNscroll-TiktokTechjam

# is it alive?
ps -eo pid,args | grep "[.]venv/bin/kuairand-agent"

# recent log
tail -40 logs/overnight-09.log

# what the agent has actually produced
ls runs/maki-overnight-09/production/generated-source/

# every provider call, with outcome
python3 - <<'PY'
import json, glob, collections
c = collections.Counter()
for f in glob.glob("runs/maki-overnight-09/production/provider-attempt-journal/*.json"):
    d = json.load(open(f))
    c[(d.get("operation"), d.get("outcome"))] += 1
for k, v in sorted(c.items()):
    print(k, v)
PY

# why any branch was rejected
for f in runs/maki-overnight-09/production/generated-source/controller-rejection-journal/*.json; do
  python3 -c "
import json,sys
v=json.load(open(sys.argv[1]))['record']['values']
print(v.get('candidate_id'), '|', v.get('root_failure_code'), '| repairs=', v.get('repairs_attempted'))
print('   ', str(v.get('root_failure_diagnostic'))[:160])
" "$f"
done
```

### Stopping it

```sh
pkill -TERM -f "bin/kuairand-agent"
```

### What good looks like in the first 20 minutes

1. `runs/.../generated-source/iteration-01-lineage.json` appears. That means propose → implement
   → materialize succeeded end to end.
2. A malformed or failed provider response **costs a branch, not the campaign**. Historically we
   saw ~43% non-accepted responses, so one will happen early. The run must keep going.
3. `repairs_attempted` on any rejection should be **greater than 0**. If it is still 0, the model
   is emitting `maximum_repairs: 0` and the prompt fix did not land — see §6.

### What "done" looks like, and how to grade it

Nothing is required of the operator during the run. Launch it and walk away.

A finished run writes:

```
runs/maki-overnight-09/final/
  report.md          the deliverable writeup
  submission.csv     the predictions
  experiments.csv    the trajectory in tabular form
  manifest.json      digests for every artifact
  reproduce.sh       the replay entrypoint
```

**If `final/` exists at all, the run finalized.** That alone is a pass on the failure this project
kept hitting: runs 04, 05 and 07 never got there.

Grade the outcome from one table in `report.md`, **Experiment trajectory and candidate tree**.
Every completed run so far:

| Run | Trajectory row | Meaning |
|---|---|---|
| 02, 03 | `official-fm-fallback-seed-4 / baseline_reproduced` | no candidate was ever admitted |
| 06 | `candidate-01-...-repair-2 / rejected_before_execution` | code was written, rejected before it ran |

The ladder, worst to best:

1. Only `official-fm-fallback-seed-4`. The agent produced nothing usable; the safety net shipped.
2. A `candidate-NN-...` row appears. The agent wrote code that cleared the static gates. Run 06.
3. **A number in `Inner primary`.** The candidate actually trained and was scored. **No run has
   ever reached this.** This is the first real milestone.
4. A number in `Outer primary`. It cleared inner folds and was promoted to matched-seed outer
   validation.
5. Multiple iterations, 1, 2, 3... The agent proposed, failed, learned and retried. This is the
   Track 2 deliverable — judges grade the loop, not only the score.

**Did it beat the baseline?** Check the **Baseline parity** table. If the selected row is
`official-fm-fallback-seed-4` at tier `qualified fallback`, the fallback shipped. If it is a
`candidate-NN-...` at tier `matched-seed outer validation`, the agent's own model won and its
primary is the score. The bar is **0.6016**, the official FM validation primary. For reference the
fallback's own seed-4 row reads 0.6020370721817017 and the five-seed mean reads 0.6015721678733825.

**A run that ends with no `final/` directory is a bug, not a bad result** — that was the entire
point of the work in section 5.

---

## 4. Cost

Every campaign spends real money on OpenRouter API calls. A 6-hour run makes on the order of tens
of provider calls. It is not free, but it is not large either. Reasoning tokens are billed as
completion tokens, which is why `max_output_tokens` is set to 65536 rather than something smaller
(see §7).

---

## 5. What was fixed in the last session, and why

Seven campaigns had failed. Each time the proximate cause was fixed and the run relaunched. The
audit found that was the wrong loop — the real problem was structural.

**The chain that killed every run:**

1. The scientific clock advances `elapsed_seconds` **only by candidate subprocess wall time**.
   Provider latency, materialization, feature builds and reflection are invisible to it. Summing
   our own provider journals: `overnight-05` alone had **4011 seconds** of unaccounted provider
   latency — more than the entire 3600s finalization reserve, in a partial run.
2. So the loop believed it had slack it did not have and kept calling past the real reserve
   boundary, where the provider raises a deadline error **by design**.
3. **Nothing caught it.** The autonomous followup loop caught only `LiveResearchBranchRejected`,
   and the reflection call was unguarded entirely.
4. `cli.py` runs research and finalization as sequential statements in one `try`. Anything escaping
   research means **finalization never runs**. Five hours of work, no bundle.

**What changed (commit `ce1c692`):**

- The followup loop now consults the engine's real clock every iteration and breaks on the
  finalization reserve or a hard expiry.
- It catches `ResearchModelError` and `ProductionResearchError`, records a closure, and **returns
  normally** so finalization runs. A provider deadline is reported as `FINALIZATION_RESERVE` (a
  scheduled stop), anything else as `CANDIDATES_EXHAUSTED`.
- The closure record also reaches the **durable rejection ledger** — the local list died with the
  frame, so the report would have shown research stopping with no stated cause.
- Reflection degrades to a named substitute (`_unavailable_reflection`) instead of propagating.
- Three separate ways the *fallback* could still lose a submission are closed: `_report_context`
  degrades instead of raising (it could fail three different ways, one caught and then re-raised);
  the official-FM fallback builder is now wrapped like the generated one; and model-authored text
  is collapsed to one line, because `schemas._text` permits newlines that `report._text` rejects
  and lineage records are written read-only.
- The selected candidate carries its measured primary into the trajectory table, which had been
  rendering a dash for the one row whose score is actually known.
- Prompt fixes: the baseline number (0.5946 → 0.6016), removal of stale file-I/O and
  request-parsing guidance the wrapper owns, the relative-import correction, removal of a vacuous
  condition about `candidate.py`, and an explanation of `maximum_repairs`.

14 new tests cover all of it.

---

## 6. What to do next, in order

### First — read the evidence from run 07

It is the best diagnostic we have, and it directly tests whether the prompt fixes work. On the
**old** code it produced:

```
propose    accepted 3
implement  accepted 2, failed 3, malformed 1

candidate-01  declared_symbol_unchanged  repairs=0
              declared material symbol(s) did not change executable source: ['confirmno']
candidate-02  invalid_python             repairs=0
              invalid Python in 'model_impl.py' at line 1: invalid syntax
```

Two things stand out. **`repairs=0` on both** — the model was emitting `maximum_repairs: 0`, so
the very first static-gate miss ended each branch with no chance to fix itself. The new prompt
explicitly tells it to set 2. **And `['confirmno']` and `invalid syntax at line 1`** are
truncation/garbage artifacts, meaning output quality is still the binding constraint.

**So: launch a fresh run and check whether `repairs_attempted > 0`.** If it is still 0, the model
is ignoring the instruction and the next move is to stop trusting the model with that field —
either clamp it controller-side or drop it from the schema and use the configured
`max_repairs_per_experiment` (which is currently dead code).

### Then — carry-over work

1. **Write the pairwise FM result into `docs/RESULTS.md` §3.2.** The sweep is at
   `runs/pairwise-sweep.json`. Report it honestly: best primary **0.6015003** against a local
   baseline of **0.6015722**, i.e. **−0.00007 — a marginal regression, not a win**. And state the
   caveats: hyperparameters were selected on the reported split, one seed, best-of-12 against a
   five-seed mean. The file itself already carries the right note: *"Manual experiment.
   candidates/pairwise_fm.py is not wired into the campaign, so this measures the direction rather
   than demonstrating agent behaviour."* Keep that framing.
2. **Push `Maki`.** 10 commits are unpushed. Do not push to `main`.
3. **Deliberately deferred, with reasons** (do not treat these as oversights):
   - The hard-deadline absorbing state (`production.py` discards a published bundle) is real but
     only reachable if we overshoot, which the clock fix prevents.
   - Bit-exact replay has no reachable tolerance path — a genuine single point of failure, but
     same-host and same-lockfile makes it unlikely, and changing it risks the integrity guarantee
     that is the project's best asset.
   - Prompt token bloat (~9,200 tokens/iteration, with the benchmark block paid on IMPLEMENT where
     it cannot influence direction) is an optimisation, not a survival fix.

---

## 7. Facts that were measured. Do not re-litigate them.

Each of these cost real time to establish. They are recorded in code comments too.

- **`tools: []` and `tool_choice: "none"` are NOT the cause of the OpenRouter 404.** Measured
  2026-08-30 against the live endpoint: a strict `json_schema` request returns 200 with them,
  without them, with either one alone, and with `require_parameters` off. The 404 was an
  **account-level data policy** on the old key. Do not remove those fields on suspicion.
- **Provider dialects differ.** OpenRouter honours `reasoning: {effort, exclude}`. TokenRouter
  honours only `thinking: {type: enabled|disabled}` and **ignores `budget_tokens` at realistic
  request sizes** — a 4096 budget still returned 11k–26k reasoning tokens.
- **Do not disable thinking.** With it off, the model stopped writing code and returned 18-character
  stubs — every file identical, with `import numpy as np` written into `config.json`. Reasoning
  stays on; the cost is paid in output budget instead, which is why `max_output_tokens` is 65536
  and the timeout is 600s.
- **Never probe provider behaviour with a trivial prompt.** A budget that appears to work on a toy
  request is ignored at real payload size. Bracket empirically at realistic size.
- **Samuel's screenshot comparison (+0.00047) is invalid.** Seed 4 is inside the five-seed mean it
  is being compared against.
- **Every previous campaign ended at `official-fm-fallback-seed-4`** because the fallback ships
  when all candidates fail. That is the system working, not the system winning.

---

## 8. Working preferences

- **The user runs the campaign themselves.** Do not launch it for them. Give them the command.
- Lead with plain English. The user is new to RecSys — give an opinion, not a survey of options.
- Ask before committing.
- Commit messages: no hyphens or dashes as punctuation, no quote marks.
- Never paste API keys into the chat.
- Stay on `Maki`. Do not push to `main`.
- `wsl.exe -e bash -lc '...'` breaks on nested quotes constantly. For any multi-line edit, write a
  Python patch script to a file and `cp` it into WSL, then run it. Anchor every substitution and
  assert the match count is exactly 1.

---

## 9. Repo orientation

```
src/kuairand_agent/
  campaign/full_campaign_runtime.py   the autonomous research loop (the survivability fixes)
  campaign/scientific.py              the inner scientific campaign and its clock
  research/prompts.py                 every instruction the LLM sees
  research/provider.py                HTTP, payload shape, retries, provider dialects
  research/production.py              lineage construction, branch rejection
  finalization/production.py          bundle, report, the official-FM fallback path
  finalization/iteration_evidence.py  recovers the per-iteration trajectory for the report
configs/full-pure.toml                6h budget, 1h reserve, epsilon, provider profiles
scripts/run_full_campaign.sh          the launcher
docs/RESULTS.md                       the deliverable writeup
runs/maki-qualification               the qualified baseline every run depends on
```
