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
| Best agent-generated candidate (validation, matched outer seeds 0/1/2) | — | — | 0.6012030 |

**No generated candidate has beaten the baseline, and we do not claim an improvement.** Four live
autonomous campaigns converged with zero manual interventions on 2026-08-30, and each selected the
protected official-FM fallback rather than a generated candidate. The best generated result,
`maki-overnight-09` `candidate-01`, scored 0.6012030 against an incumbent of 0.6014403 on the same
matched outer seeds — a difference well inside the baseline's own 0.0008 seed-to-seed standard
deviation, and therefore statistically indistinguishable from it.

The shipped submission is the official FM at seed 4. Its 0.6020370722 is **not** an improvement
over the 0.6015721679 five-seed mean: seed 4 is one of the five seeds in that mean, so comparing
the two compares the baseline against itself.

What the runs do demonstrate is that the loop runs end to end without human intervention,
converges under the frozen rule, and emits an organizer-valid submission every time.

GPU-hours consumed: **0.00**. Total provider spend across every run to date: **$20.84** for
1,965,299 tokens over 92 calls.

GPU-hours consumed: **0.00**. Every configuration is CPU-only.

## Limitations, and what we would do with more time

**What is not demonstrated.** We have not shown a validation-primary improvement above ε = 0.002.
Four live campaigns converged and every one selected the official-FM fallback, and the pairwise
direction was separately swept to convergence over twelve configurations without beating the
baseline in any of them. Hidden-test performance is unknown and unclaimed; only the organizers can
measure it.

**The current bottleneck is candidate execution, not modelling.** Across the three campaigns after
identity-code features were added, only one of nine generated candidates executed at all. The rest
raised before evaluation, in three recurring classes: within-user pair-sampling index arithmetic,
mixing an `(N,)` accumulator with an `(N, rank)` one inside hand-written factorization-machine
interaction maths, and a non-scalar checkpoint entry read by the candidate's own
`training_diagnostics`. Implementation size predicted this almost perfectly: candidates at or under
roughly 260 lines executed 3 for 3, while none of the eight written at over 580 lines executed.
A crash inside `train_model` currently ends a branch outright, because the repair loop covers
pre-execution static gates only — the clearest structural gap we know of in the loop.

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
  and the domain briefing; results, resource accounting, and submission documentation.
