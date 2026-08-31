# Autonomous KuaiRand-Pure ML Research Agent

This repository provides a local implementation of the autonomous research campaign in
[`plan.md`](plan.md). The operative benchmark is within-user `long_view` ranking over
logged KuaiRand-Pure impressions. The untouched organizer evaluator reports GAUC and nDCG@5;
their arithmetic mean is the primary metric.

The files in `kuairand-starter-kit/` are immutable, hash-pinned organizer references. Runtime
data, campaign databases, caches, checkpoints, predictions, and final bundles stay in ignored
local directories. Development never reads, aggregates, exposes, or scores final-period outcomes.

## Install the locked environment

The lock targets CPython 3.12.13. A full LambdaRank campaign requires the pinned `research-tree`
dependency group; the CPU path remains the reproducible reference path.

```bash
UV_CACHE_DIR=.uv-cache uv sync --locked --group research-tree --no-group research-neural
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  kuairand-agent --help
```

`configs/full-pure.toml` is the production autonomous configuration. It uses two OpenAI-compatible
Chat Completions profiles with strict structured output by default. Copy `.env.example` to the ignored
`.env.local` and set all six production values:

```dotenv
INFERENCE_MAIN_API_KEY=
INFERENCE_MAIN_BASE_URL=https://openrouter.ai/api/v1
INFERENCE_MAIN_MODEL=deepseek/deepseek-v4-pro-0813
INFERENCE_FALLBACK_API_KEY=
INFERENCE_FALLBACK_BASE_URL=https://api.tokenrouter.com/v1
INFERENCE_FALLBACK_MODEL=deepseek/deepseek-v4-pro-0813
```

Both base URLs must implement `POST {base_url}/chat/completions`. Both dedicated credentials are
checked before a campaign starts. Each provider
keeps bounded HTTP/transport and malformed-response retries. For HTTP `429`, the adapter honors a
bounded `Retry-After` value when it fits within the remaining research time before the finalization
reserve, otherwise uses bounded exponential backoff with jitter, and records retry-wait time
separately from network latency. Every transport timeout is capped to that same remaining research
window, and an exhausted shared deadline starts neither a new main call nor a fallback call. A
second consecutive `429` ends that provider route so the same logical inference operation can use
the fallback instead of immediately sending a third request to a throttled endpoint. If the main
provider otherwise ends with a typed provider error, the operation is also attempted on fallback;
fallback then remains active for later operations. If both profiles fail, the campaign reports
provider-chain exhaustion rather than silently switching to scripted research. Configure each
profile's actual token prices in its corresponding `pricing` table in `full-pure.toml`.

Each profile also declares its non-secret Chat Completions dialect in `full-pure.toml`. Keep
`response_format = "json_schema"`, `max_tokens_parameter = "max_completion_tokens"`, and
`send_reasoning_effort = true` for providers/models implementing those current fields. For a
provider that supports Chat Completions but only portable JSON mode, select `json_object`,
`max_tokens`, and `false`; the exact operation schema is then included in the system message and
the same strict local parser, malformed-response retry, source policy, and admission checks remain
in force. Main and fallback can use different dialects.

Provider calls generate only typed proposal, complete-file implementation/repair, and reflection
records; they never receive shell, filesystem, evaluator, public-label, or final-outcome authority.
Schema-v2 single-provider and schema-v3 two-provider configurations remain supported for replay
and backward compatibility, but the production schema-v4 config never silently reuses
`OPENAI_API_KEY`.

## Candidate-source admission and repair contract

The trusted controller exposes one digest-bound candidate-source policy to every proposal,
implementation, and repair request and derives the delivered agent instructions from that exact
policy. It requires the final candidate tree to contain `candidate.py`, allows only the declared
candidate-owned suffixes and bounded helper files, and preserves every existing forbidden
basename, path root, import, call, and size guardrail. In particular, generated `baseline.py` and
evaluator/submission artifacts such as `submission.csv` are forbidden. Responses contain complete
replacement contents for every returned file rather than patches; documentation, filenames,
docstrings, whitespace, and unchanged symbols do not count as an executable scientific change.

Proposed file manifests are checked before implementation. An invalid manifest enters a bounded
typed correction path and cannot consume candidate execution or training. Provider JSON acceptance
alone is never candidate admission: generated packages still pass local path, static-policy,
reachable-symbol, and executable-materiality validation before they can become a lineage node.

When an implementation or repair package fails before materialization, the next repair request
contains a bounded, inert, digest-verified snapshot of the exact rejected provider package, kept
separate from the trusted parent. The research agent can therefore preserve its own scientific
mechanism while moving it into legal candidate-owned files. The rejected package is never executed
or published, every repair is applied freshly over the trusted parent, and all local admission
checks rerun from scratch. Resume verifies the persisted request/response causal chain and does not
repeat an already accepted provider call.

Rejected initial-admission branches persist both the root and terminal failure fingerprints in an
immutable per-iteration hash-chained journal. Resume reconstructs that journal and fails closed if
an entry or link was changed; provider failure cannot erase already journaled initial-admission
evidence. The first failure receives one bounded correction, a scientific family is blocked after
its second pre-admission rejection, and the third occurrence of one identical root fingerprint
closes initial admission as `repeated_pre_admission_failure`. The controller retains the official FM
incumbent and finalizes rather than spending the remaining deadline repeating the same invalid
request. Proposal novelty signatures also block prose-only duplicates and cap one untrained
scientific family at two pre-admission attempts until trusted training/evaluation evidence advances.

`configs/scripted-demo.toml` is a deterministic demo configuration and requires the explicit
`allow_scripted_demo = true` acknowledgement. `configs/default.toml` and `configs/smoke.toml` are
deterministic test configurations. They require no network or API credential and cannot be
reported as autonomous runs. The optional neural dependency group is excluded explicitly from
tree-campaign commands so a previously installed `torch` cannot silently change the environment
identity.

## Command surface

Every successful automation-facing lifecycle command writes one stable JSON object to stdout.
Progress and errors use stderr, and invalid, corrupt, or incomplete work exits nonzero.

```text
kuairand-agent data prepare --archive PATH --data-dir PATH
kuairand-agent data prepare --download --data-dir PATH
kuairand-agent data audit --data-dir PATH [--output-dir PATH] [--json]
kuairand-agent qualify --data-dir PATH --run-dir PATH
kuairand-agent run --config PATH [--qualification-run-dir PATH] [--run-dir PATH]
kuairand-agent resume --run-dir PATH
kuairand-agent status --run-dir PATH [--json]
kuairand-agent finalize --run-dir PATH
kuairand-agent replay --project-root PATH --bundle PATH --data-dir PATH --expected-data-sha256 SHA256
kuairand-agent validate-submission --split valid|test --data-dir PATH FILE
kuairand-agent config validate FILE
kuairand-agent provider preflight --config FILE
kuairand-agent contract verify-starter [--starter-dir PATH]
```

`run` is a new-campaign operation: it verifies the configuration, canonical data, starter kit,
and official-FM qualification, creates a no-overwrite durable campaign, runs the bounded
research policy selected by the explicit run kind, and finalizes the retained eligible lineage.
An autonomous run repeatedly proposes, implements, evaluates, and reflects until a frozen
convergence, iteration, launch, outer-promotion, repeated-pre-admission, reserve, or deadline
terminal condition. It never silently resumes or replaces an existing run directory.

`resume` verifies the original immutable campaign identity, reconciles unfinished subprocesses,
continues only while the original launch/time policies permit, and then finalizes. It does not
reset budgets or start a second deadline.

`finalize` is the provider-free recovery entry point for a campaign whose research outcome was
already retained. It deterministically tries the selected generated candidate and walks back
through replayable ancestors to the immutable organizer-FM fallback. An exact retry verifies and
returns the already published closed bundle; it does not overwrite it.

`replay` accepts only a closed final bundle, verified canonical data, and the exact frozen data
SHA-256. The trusted loader verifies the closed member inventory and SHA-bound allowlisted replay
recipe before rebuilding label-free capabilities. Bundle-controlled import paths, shell commands,
or arbitrary Python backends are never executed. An exact retry must reproduce the same immutable
bundle and submission identities.

`status` is read-only and never reconciles or mutates a campaign.

## Prepare and audit the official data

Prepare a locally supplied official archive into a new destination:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked kuairand-agent data prepare \
  --archive /absolute/path/KuaiRand-Pure.tar.gz \
  --data-dir .data
```

Use `--download` instead of `--archive` only when explicit acquisition from the hash-pinned
organizer source is wanted. After verification, the example produces
`.data/KuaiRand-Pure/data`, which matches the frozen configs.

Audit the canonical data without materializing final outcomes and retain paired evidence:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked kuairand-agent data audit \
  --data-dir .data/KuaiRand-Pure/data \
  --output-dir runs/wp2-evidence \
  --json
```

Data extraction and audit-evidence destinations must not already exist. The commands fail closed
instead of merging with or overwriting prior evidence.

## Reproduce the official FM qualification

Run the six charged launches—five official FM seeds plus the clean seed-0 replay—into a new
qualification directory:

```bash
scripts/qualify_full_data.sh \
  .data/KuaiRand-Pure/data \
  runs/wp3-official-qualification
```

The qualification preserves the exact five-seed public metrics, replayable checkpoints and
predictions, and the immutable seed-4 fallback. The run directory must be new. If verified
qualification evidence already exists, reuse it; do not rerun qualification merely to recreate a
directory name.

## Run the full autonomous campaign

The convenience launcher synchronizes the exact tree-only environment, loads an ignored local API
key file when present, and defaults to the frozen live-provider full-data configuration and
official qualification directory. A full run can consume API credits and up to six hours, so
launch it only when that spend and runtime are intended:

First validate the frozen provider settings and credential without sending an API request:

```bash
set -a
. ./.env.local
set +a
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  kuairand-agent provider preflight --config configs/full-pure.toml
```

Successful output explicitly records `"api_request_sent": false`; it names the credential
environment variable but never prints its value.

Then, when API spend and the bounded full campaign are intended:

```bash
scripts/run_full_campaign.sh
```

Its complete interface is:

```text
scripts/run_full_campaign.sh [CONFIG_PATH [QUALIFICATION_RUN_DIR [RUN_DIR]]]
```

For an explicit new campaign path:

```bash
scripts/run_full_campaign.sh \
  configs/full-pure.toml \
  runs/wp3-official-qualification \
  runs/full-pure
```

The equivalent direct command is:

```bash
set -a
. ./.env.local
set +a
UV_CACHE_DIR=.uv-cache uv sync --locked --group research-tree --no-group research-neural
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  kuairand-agent run \
  --config configs/full-pure.toml \
  --qualification-run-dir runs/wp3-official-qualification \
  --run-dir runs/full-pure
```

Configuration paths are interpreted relative to the invoking working directory. Run from the
repository root unless intentionally supplying another complete project root. The default
configuration enforces one process, four CPU threads, a 16 GiB memory ceiling, a 20 GiB disk
ceiling, 50 total training launches, at most six distinct public promotions, a six-hour hard
deadline, and a protected one-hour finalization reserve.

If the process is interrupted, inspect without mutation and then resume the same campaign:

`SIGINT` and `SIGTERM` request cooperative cancellation. The runner stops at a durable stage or
launch-admission boundary, preserves the original budget and deadline evidence, and exits
nonzero; it does not publish a partial success result.

```bash
UV_CACHE_DIR=.uv-cache uv run --locked kuairand-agent status \
  --run-dir runs/full-pure \
  --json

UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  kuairand-agent resume \
  --run-dir runs/full-pure
```

If research closed durably but automatic finalization was interrupted, finalize from the retained
outcome without invoking a research provider:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  kuairand-agent finalize \
  --run-dir runs/full-pure
```

The final JSON result reports the selected candidate, fallback count, closed bundle root, bundle
manifest SHA-256, submission SHA-256, organizer-check evidence, and replay evidence. Research
evidence separately counts attempted branches, accepted proposal/implementation/repair responses,
pre-execution rejections, candidate admissions, training starts, and completed inner/outer
evaluations. It also preserves bounded root/terminal rejection summaries, provider-route switches,
and retry-wait evidence, so an accepted provider response cannot be described as trained or
evaluated. A baseline result is reported as `baseline_reproduced` and explicitly remains the
protected official FM fallback, not an agent-generated improvement.

## Replay a closed final bundle

Run the bundle's generated `reproduce.sh`, or use the equivalent command below with the exact
data digest recorded by that script. The repository, bundle, data, and digest inputs are explicit;
replay re-proves the trusted source and locked environment and never downloads data or consults a
research provider.

```bash
UV_CACHE_DIR=.uv-cache uv sync --locked --group research-tree --no-group research-neural
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  kuairand-agent replay \
  --project-root /absolute/path/to/repository \
  --bundle /absolute/path/reported/by/finalization \
  --data-dir .data/KuaiRand-Pure/data \
  --expected-data-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Replace the illustrative digest with the exact 64-character value in `reproduce.sh`. A data,
bundle-member, recipe, capability, prediction, metric, alignment, or submission mismatch exits
nonzero and emits no success JSON.

### Each bundle replays only at the commit that produced it

Replay re-proves the trusted source tree, so `--project-root` must be checked out at the commit
whose `hash_source_tree` digest matches the one recorded in that run's
`controller/create-request.json`. Checking out any other commit exits nonzero.

| Bundle | Recorded `source_digest` | Check out |
|---|---|---|
| `runs/maki-overnight-09/final` | `6bb8f2b48e9079ae1ebd5afe…` | `0978e46` |
| `runs/maki-overnight-10/final` | `2573f28b4ac1aaf6d5bf0a8d…` | `d82f15a` |
| `runs/maki-overnight-11/final` | `b7ef85773ef03cb2e38b2614…` | `784ae49` |
| `runs/maki-overnight-12/final` | `59714d3bc49ad73a1c8effd0…` | `7b7d83e` |
| `runs/maki-overnight-14/final` | `3a6d21e5b220595acc773497…` | `e758693` or `702e548` |
| `runs/maki-overnight-15/final` | `8eee43fa4f27…` | **unusable — see below** |
| `runs/maki-overnight-16/final` | `258e67f72773…` | `8978b4b` |
| `runs/maki-overnight-17/final` | `160f902c43ba…` | `d8f0baa` |

Verified by computing `hash_source_tree` in a detached worktree at each commit and comparing
against the recorded digest; every one matches exactly. Run 14's two commits share a digest because
`702e548` changes documentation only, and documentation is outside the hashed source slice.

**Runs 16 and 17 are the only bundles that replay against the repository as submitted.** Every
earlier bundle requires checking out its own older commit first.

`runs/maki-overnight-15/final` was for a long time the **only agent-generated submission** the
project had produced (SHA-256 `c98d7cd6…`, organizer checker rc=0), and it cannot be replayed at
any commit. `replay_final_bundle` calls `_verify_closed_bundle` before the current-source check, so
at a fixed commit it fails source identity, and at its own commit `8124607` the float32 defect it
was stranded by is still present and the bundle check fails. Both doors are shut, exactly as for
run 13.

That is no longer the only one. The sol6 and sol7 campaigns each finalized an agent-generated
bundle at the current commit — `runs/sol7-20260831T124700Z/final` and
`runs/sol7-20260831T131245Z/final`, both 170,588 rows with organizer checker rc=0 — and both
replay against the repository as submitted rather than needing an older checkout. Their scores are
in the table above; neither clears epsilon, and the second is one of two campaigns run under
`CONTINUE_ON_SUCCESS=1`, so it must be read as one of N rather than as a single measurement.

`runs/maki-overnight-13` has no bundle. Its campaign ran at `f66e364`
(`cca9cc26a6e57fdb36fc6dfb…`) and promoted a generated candidate for the first time, which exposed
a latent defect in ledger export that aborted finalization. Because finalization verifies the
working tree still hashes to the digest recorded at campaign creation, the only source that can
finalize run 13 is the source containing that bug. The run's scores survive in
`runs/maki-overnight-13/production/`; its bundle is permanently unrecoverable, and the alternative
— deleting the losing candidates' evidence to satisfy the old check — would fabricate the audit
trail.

To validate a retained CSV independently against canonical alignment:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked kuairand-agent validate-submission \
  --split test \
  --data-dir .data/KuaiRand-Pure/data \
  /absolute/path/to/submission.csv
```

This is a structural check only. Hidden-test labels and scores remain organizer-only.

## Developer checks

Offline checks after the locked artifacts are present:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural pytest -q
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  ruff check src tests
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  ruff format --check src tests
UV_CACHE_DIR=.uv-cache uv run --locked --group research-tree --no-group research-neural \
  mypy src tests
sh -n scripts/run_full_campaign.sh
```

Full-data and optional performance/replay gates require the verified local archive or their
documented optional dependency groups. They never require external training data.

## How the agent works

Two planes with a narrow typed wire between them.

The **trusted controller** owns everything that could compromise the result: the dataset and its
split boundaries, the byte-protected organizer scorer, the sandbox, every budget, and the durable
campaign record. The **research model** reaches it through exactly four typed methods —
`propose`, `implement`, `repair`, `reflect` (`src/kuairand_agent/research/interface.py`) — and has
no filesystem, shell, network, evaluator, credential, or label authority.

One scientific iteration:

1. The model receives a leakage-safe context: benchmark contract, train-only EDA, label-free
   validation-input EDA, method cards, prior iteration outcomes, and remaining budget.
2. It returns a falsifiable **hypothesis** naming one principal change and its expected effect on
   GAUC or nDCG@5.
3. It returns **complete candidate source files** — never patches. The controller writes them into
   a fresh disposable directory and computes the diff itself.
4. Static gates run: path policy, forbidden imports, forbidden calls, and a material-change check
   that rejects a package claiming a modification it did not make.
5. The candidate trains in a sandbox and is scored by the untouched organizer evaluator through a
   three-tier gate — smoke, then inner temporal folds, then a rationed outer validation query.
6. Promotion or rejection is decided by the measured score, never by the model's own report.
7. The model reflects on the trusted result and chooses what to try next.

The campaign ends on the organizer convergence rule (ε = 0.002 over N = 3 iterations) or a frozen
iteration, launch, promotion, reserve, or wall-clock boundary — whichever comes first. The
validation-best eligible checkpoint is then replayed in a clean locked environment and emitted as
a label-free submission.

**The model proposes; code decides.** No metric in this system is ever self-reported.

## Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full table, run logs, resource accounting, and
evaluation-integrity analysis.

Headline, stated conservatively:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| FM — official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| FM — official baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| Oracle ceiling (test) | 1.0000 | 0.7289 | 0.8645 |
| Best agent-generated candidate (validation, matched outer seeds 0/1/2) | — | — | 0.6017246 |
| **Five-seed rank ensemble of the official FM (validation, shipped)** | 0.6688063 | 0.5364006 | **0.6026034** |

**No generated candidate has beaten the baseline by a material margin, and we do not claim an
improvement.** Nine live autonomous campaigns ran with **zero manual interventions**; seventeen
campaigns in total across the project. Seven selected the protected official-FM fallback; two
promoted a generated candidate, neither clearing ε = 0.002.

The best generated result, `maki-overnight-15` `candidate-01`, scored 0.6017246 against an
incumbent of 0.6014403 — **+0.0002844, or 0.36 of one seed standard deviation.** The same method
was run three times: **+0.34σ, −0.03σ, +0.36σ**, mean +0.00018 against ε = 0.002. Run 13's original
+0.0002715 was replicated rather than reported, and run 14 landed on the other side of zero. That
replication is the reason the number is not our headline.

Three caveats we went looking for and are reporting against ourselves:

- **The agent's number is a blend, not its model.** Every candidate prediction is rank-fused with
  the official FM over a fixed five-point weight grid, and the recorded metric is the blend's.
  0.6017246 is a 25/75 blend, so roughly three quarters of it is the organizer's ordering. Scored
  alone, no candidate we have run has beaten the baseline — best is 0.5745312 against 0.5754240,
  or −1.12σ. Reproduce with `python3 fusion_audit.py <run-id>`; the analysis is in
  [`docs/RESULTS.md`](docs/RESULTS.md) §3.3a.
- **The five-seed ensemble is not an agent result.** 0.6026034 beats the 0.6016 baseline by
  +0.0010 with no selection effect. But it ensembles the organizer's own baseline with itself, it
  is controller-side rather than agent-generated, and it is below our own ε = 0.002 materiality
  threshold.
- **The shipped fallback's margin is a selection effect.** `official-fm-fallback-seed-4` scores
  0.6020371, but seed 4 was chosen as the best of five on the same split it is reported on
  (`qualification.py:1080`). That is not a clean margin over the baseline.

What the runs do demonstrate is that the loop runs end to end without human intervention,
converges under the frozen rule, and emits an organizer-valid submission every time.

### The trusted parent, and what the earlier runs were actually measuring

Every campaign above started from a fixed-step logistic scorer, and none improved on it. The three
candidates produced after the instruction-surface fixes scored **0.5693, 0.5674 and 0.5724**
standalone on the Fold B screen, bracketing that parent's own **0.5698** — they were spending their
single scientific iteration rebuilding the base rather than testing their own hypothesis. So
"candidates land nowhere near the control" was a finding about the seed, not about the task.

The trusted parent is now the identity-code factorization machine that the records already showed
was strongest: a second-order interaction over the identity codes only, causal aggregates kept as
additive first-order terms, Adam over shuffled minibatches, five deterministically seeded members
averaged on within-user percentiles.

| model | Fold B standalone | vs control |
|---|---|---|
| logistic parent (previous seed) | 0.5698351 | −17.98σ |
| **identity-FM parent (current seed)** | **0.5746869** | **−2.33σ** |
| run 16, the best prior measurement | 0.5745312 | −2.83σ |
| official FM control | 0.5754240 | — |

It was rebuilt from a *configuration* recorded in the cross-run ledger, because the source of the
run that produced it had been deleted with its run directory. `parent_acceptance_probe.py` gates
the promotion and refused four configurations before accepting this one at +0.49σ above run 16 —
without that gate every later `parent_fold_b_primary` would have meant "versus an unverified
reimplementation". Details, including the three ways the first rebuild failed, are in
[`docs/RESULTS.md`](docs/RESULTS.md) §3.4a.

This is still **not** a result that clears ε, and the parent remains 2.33σ below the control. What
changed is that the next campaign measures the task instead of the seed.

### The submitted artifact

```sh
python3 build_ensemble_submission.py            # ~12 min; writes ensemble-submission/
cd kuairand-starter-kit && python submit.py ../ensemble-submission/submission.csv \
    --data_dir ../.data/KuaiRand-Pure/data --split test --check      # returns 0
```

A five-seed within-user rank ensemble of the official FM, scoring **0.6026034355** on public
validation. Built and organizer-checked at the current commit: 170,588 rows, `submit.py --check`
returns 0, submission SHA-256 `7e6dc8b1f21d5a08…`, starter manifest `91f8f98098c40caf…`. It performs **no new training**: every member is an already-qualified organizer FM run
from `runs/maki-qualification`, each checkpoint is verified against its qualification digest, the
shared encoding is verified across all five rather than assumed, and inference runs through the
hash-pinned organizer source. The number reproduced exactly through two independent code paths.
`ensemble-submission/provenance.json` records that it is controller-side and not agent-generated,
and that it does not clear ε.

Members are combined on within-user rank percentiles, not raw scores. That choice is worth
+0.0004891 (`python3 ensemble_mode_probe.py`), and it carries most of the ensembling effect:
averaging raw scores is worth +0.0000772 against +0.0005664 for percentiles.

**A candidate can do this too, and the claim that it could not was wrong.** `user_groups` is a
training-only capability, but `user_id_code` is a *column of the feature matrix*, so it reaches
`predict_scores` like any other feature. Grouping by that code instead of the true user id scores
0.6025848 — **+0.0005478 over the best single seed, 96.2% of the ceiling above** — with the
unknown-identity slot (1.59% of validation rows) costing −0.0000186. The briefing told candidates
the opposite for four campaigns; the trusted parent now performs it.

GPU-hours consumed: **0.00** — every configuration is CPU-only. Total provider spend across every
run to date: **$29.27** for **2,996,874 tokens** over **141 attempts** (130 returned a usage
block). Regenerate with `python3 token_accounting.py`.

## Limitations, and what we would do with more time

**What is not demonstrated.** We have not shown a validation-primary improvement above ε = 0.002,
and no generated candidate has beaten the baseline scored on its own. The pairwise direction was
separately swept to convergence over twelve configurations without beating the baseline in any of
them. Hidden-test performance is unknown and unclaimed; only the organizers can measure it.

**The candidate seam capped what the agent could achieve, and we can show it.** A candidate's
prediction request carries the feature matrix and the checkpoint and nothing else
(`candidate_api/runtime_contract.py`, `prediction_request_fields`). Three consequences, each
measured rather than argued:

1. **No user identity at prediction time**, so no user×item crosses — the term the baseline FM
   draws most of its ordering power from. We added `user_id_code` (bundle 37 → 38 columns, user
   vocabulary 26,211). **The hypothesis was falsified:** the gap stayed at −5σ.
2. **No `user_groups` at prediction time**, so no within-user rank normalisation — **RETRACTED.**
   The measurement stands: on the five official FM seeds, raw-score averaging is worth +0.0000772
   while rank averaging is worth +0.0005664, so 86% of the ensembling gain is the normalisation.
   The conclusion did not. The absent capability is `user_groups`; `user_id_code` is a *column of
   the feature matrix* and reaches `predict_scores` like any other feature. Grouping by it scores
   0.6025848 — 96.2% of the ceiling — with the unknown slot costing −0.0000186. We told candidates
   this was impossible for four campaigns and they obeyed. Reproduce both with
   `python3 ensemble_mode_probe.py`.
3. **No early stopping on the scored split** — which the baseline *does* get.
   `baselines/starter_fm.py:701-708` keeps the best of up to 40 epochs measured on the split it
   then reports, and the published 0.6016 has the same property (`baseline.py:88`). Our candidate
   gets one shot, so the residual ~1σ is measured against an advantaged reference.

The transferable point is that **an agent's capability boundary is a design parameter, not a
detail.** Ours was drawn for good isolation reasons and silently capped the achievable score. What
finally moved the number was not a better model but a better-specified interaction structure:
restricting the FM interaction to the identity codes and keeping the causal aggregates additive
took the gap from −4σ…−12σ to **−1.12σ**.

**Our agent's feedback signal was wrong for three campaigns, and we found it late.** Candidate
predictions are rank-fused with the official FM baseline, and the score recorded for a candidate is
the *blend's*, not the model's. Where the selector weighted a model out entirely, the recorded
metrics were the baseline's own — which we initially wrote up as a finding about distinct models
inducing identical orderings. It was an artifact. Read at the standalone grid point, every
candidate we have run sits 4σ to 12σ below the baseline; we were never in contention, and the blend
was substituting the baseline's score for ours. Two causes, both now fixed: candidates received no
user identity at prediction time, so they could not represent the user×item crosses the baseline
depends on; and the briefing never mentioned fusion, so the agent reasoned correctly from a false
premise. `fusion_audit.py` reproduces the audit against any recorded run. The general lesson is
that an agent's *feedback channel* deserves the same adversarial scrutiny as its permissions — no
integrity guarantee was violated here, and every digest verified.

**Candidate execution was the prior bottleneck, and it is now solved.** Across the three campaigns
after identity-code features were added, only one of nine generated candidates executed at all,
failing in three recurring classes: within-user pair-sampling index arithmetic, mixing an `(N,)`
accumulator with an `(N, rank)` one inside hand-written factorization-machine interaction maths,
and a non-scalar checkpoint entry read by the candidate's own `training_diagnostics`.
Implementation size predicted this almost perfectly: candidates at or under roughly 260 lines
executed 3 for 3, while none of the eight written at over 580 lines executed. After tested helpers
for those three classes were added to the candidate seed, the last five campaigns executed
**3 of 3 every time** — 22/22, 21/21, 22/22, 15/15 and 15/15 subprocess executions with empty
stderr throughout. A crash inside `train_model` still ends a branch outright, because the repair
loop covers pre-execution static gates only — the clearest structural gap remaining in the loop.

**A code path that runs only on success accumulates defects invisibly.** The promotion path went
unexecuted for twelve campaigns, and each time it finally ran it exposed a latent bug: a ledger
export that misattributed runs (run 13), then a comparison of the float32 organizer primary against
its own float64 recomputation at `abs_tol=1e-15` — **2.98e-8 apart, one float32 ulp** — which
stranded run 15 at `FINALIZING` after it became the first campaign to promote a candidate *and*
publish a bundle. Both are fixed and regression-tested. Both were found by reading the whole run
rather than the failing traceback.

**The constraint-transmission asymmetry.** The most instructive defect we found was in our own
design. `materialize.py` enforces 44 distinct constants on generated code — forbidden import
roots, reserved basenames, trusted path prefixes, forbidden calls, size limits — but the
implementation request transmitted only three of them. The model was being rejected for rules it
was never told. Every model tried (gpt-4o, DeepSeek, gpt-5.4-mini) failed the contract, and the
cause was our instruction surface, not their capability. `research/prompts.py` now renders its
constraint text *from those enforcement constants directly*, so the two cannot drift apart. It is
a reminder that in an agent system the prompt is part of the interface, and deserves the same
rigour as a schema.

**A record-ordering defect that made failure permanent.** The live lineage record was written
immediately after the model responded but before static validation, using create-once read-only
semantics. A single contract-violating package therefore persisted an unwritable record, killed
the campaign, and made `resume` fail identically forever. Fixed by holding the response in memory
and writing only after every check passes. The general lesson — persist a decision after
validating it, not before — is one we would apply earlier next time.

**With more time, in priority order:**

1. **Optimize the ranking objective properly.** Training uses pointwise log loss while GAUC and
   nDCG are ranking metrics. This is the organizers' own top-ranked open direction and our
   pairwise implementation exists but is trained only to a tiny acceptance configuration.
2. **Per-operation model routing.** Reasoning effort and model choice are currently uniform across
   all four research operations, so deep reasoning is paid on code generation as well as on
   research. The adapter seam already supports a router; only configuration plumbing is missing.
3. **Inspiration-based crossover and a quality-diversity archive.** The search is greedy against a
   single incumbent. Because GAUC (mixed-label users only) and nDCG@5 (all users) genuinely
   diverge, keeping the best candidate *per objective* and letting the model merge two parents'
   mechanisms is a natural fit. We designed this and deliberately did not ship it: it requires a
   new wire field in the subsystem that guarantees replay integrity, which is not a change to make
   days before a deadline.
4. **Unbiased validation via the randomized-exposure log.** `log_random_4_22_to_5_08_pure.csv`
   (1.18M rows) is blocked as training data by our field policy, correctly. Used as a *second,
   unbiased validation signal*, it would reveal whether a promoted candidate only wins on biased
   traffic — a genuine improvement to the selection loop rather than the model.
5. **Windows support.** The runtime is POSIX-only (`resource`, `os.killpg`, `SIGKILL`), which cost
   the team parallel capacity when one member could not execute anything locally.

## Team contributions

- **Samuel Tan** ([@tancysam](https://github.com/tancysam)) — system architecture and the bulk of
  the implementation: trusted controller, leakage-safe data layer, campaign store and runtime,
  execution sandbox, finalization and replay. Environment and dataset qualification; execution of
  the full-data campaigns; the lineage record-ordering fix and its regression test.
- **Sean Koh** ([@TerrorByter](https://github.com/TerrorByter)) — architecture review and failure
  analysis; live-provider testing across multiple models that surfaced the code-generation
  contract failures; design and implementation of the bounded repair path and campaign-level
  failure recovery.
- **Makendra Prasad** ([@makilover3000](https://github.com/makilover3000)) — benchmark contract
  verification against the starter kit (identifying that the planning documents encoded the wrong
  label and metrics); the research-model instruction surface, including constraint transmission
  and the domain briefing; the rank-fusion audit that retracted the identical-ordering finding and
  established that no candidate had been competitive standalone, and the identity-code feature work
  it motivated; results, resource accounting, and submission documentation.
