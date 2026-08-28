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

`configs/full-pure.toml` is the production autonomous configuration. It uses the OpenAI Responses
API with strict structured outputs and requires `OPENAI_API_KEY`. The convenience launcher loads
that variable from an ignored `.env.local` when present. Provider calls generate only typed
proposal, complete-file implementation/repair, and reflection records; they never receive shell,
filesystem, evaluator, public-label, or final-outcome authority. The frozen model and pricing
fields are explicit campaign identity rather than values inferred at runtime.

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
convergence, iteration, launch, outer-promotion, reserve, or deadline terminal condition. It never silently resumes or
replaces an existing run directory.

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
manifest SHA-256, submission SHA-256, organizer-check evidence, and replay evidence. A baseline
result is reported as `baseline_reproduced`; the agent never fabricates an improvement label.

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
