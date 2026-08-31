# Autonomous laboratory migration inventory

- Decision: `GO` for the deterministic source kernel and offline scripted vertical slice;
  hardware/full-data qualification and production cutover remain gated.
- Recorded: 2026-08-31, Asia/Singapore.
- Authoritative plan: repository-root `plan.md`.
- Authoritative plan SHA-256: `3b20a3fa1df1bbaa2499e3e0b95ee9a9cb0d05174e1a968c82e9dcd5adcc0fb5`.
- Repository: `https://github.com/tancysam/sleepNscroll-TiktokTechjam.git`.
- Branch: `newRevision`.
- Migration base: `bc074aaa9419c5466ae472609b7dd4de66fde254` (`origin/codex/candidate-unstable`).
- Pre-edit status: clean; `git diff --stat` was empty.
- Active campaign audit: no KuaiRand campaign, launcher, or monitor process was running.
- Available local inputs: the clean checkout contains no `.data`, `runs`, or `.kuairand` state. A separate historical checkout contains ignored data and run directories; they are not copied, mutated, or treated as current evidence.

## Baseline verification

The accidental older-base audit on `origin/main` recorded `1330 passed, 21 skipped`. After correcting the branch to the plan-compatible committed base, the authoritative pre-edit run recorded:

```text
1549 passed, 21 skipped in 28.09s
```

Skipped gates require official data, retained production evidence, a provider credential, or the optional neural dependency. They are not failures and are not claimed as verified.

## Mutable authority map

| Truth currently written | Current surface | Target transaction | Disposition |
| --- | --- | --- | --- |
| Campaign lifecycle, experiments, executions, metrics, failures, incumbent | `campaign/store.py`, run-local `campaign.sqlite3` | lab-scoped `state/repository.py` transaction plus event | `ADAPT`, then replace the old writer |
| Cross-run public/protected query count | `OuterQueryLedger`, sibling SQLite database | same authority transaction as reservation/evaluation | `REPLACE`; `DELETE_AFTER_CUTOVER` |
| Deadline and reconciliation cursor | controller JSON/hash chain | campaign event and monotonic clock rows | `ADAPT`; projection only after cutover |
| Coarse scientific stage and outcome | `FullCampaignProgressLedger`, `FullCampaignOutcomeRepository` | campaign event, selection, and finalization rows | `DELETE_AFTER_CUTOVER` as authority |
| Generated scientific record | `FileScientificRunEvidenceRepository` JSON | experiments, trials, attempts, artifacts, predictions, inner evaluations | `DELETE_AFTER_CUTOVER` as authority |
| Provider operation and lineage | live lineage JSON | idempotent `provider_operations` rows | `REPLACE` |
| Selection and finalization | in-memory choice, outcome JSON, bundle, then campaign completion | finalization intent plus one authoritative decision/completion transaction | `REPLACE` decision writer; retain sealed bundle |
| Content-addressed artifact bytes | `execution/artifacts.py` | immutable bytes plus verified artifact registration | `REUSE` and `ADAPT` registration |

The existing global-first/local-second protected-query writes are deliberately conservative but not atomic. The new path must never dual-write one decision to the old and new authorities.

## Source disposition

| Surface | Disposition | Migration rule |
| --- | --- | --- |
| `contract.py`, organizer hashes, data/folds/capabilities | `REUSE` / `ADAPT` | Build the complete path-independent `ContractManifest` above the pinned primitives; retain final-outcome exclusion. |
| `baselines/*` | `REUSE` / `ADAPT` | Keep official FM arithmetic and qualification; wrap it behind `QualifiedTrainer`. |
| `candidates/tree_ranker.py` | `ADAPT` | Keep deterministic lazy CPU LightGBM; add distinct CPU/GPU trainer adapters rather than device fallback in one trial. |
| `execution/runner.py`, artifacts/workspace/signals | `REUSE` | Keep process-tree supervision and immutable artifacts below the trainer seam. |
| `campaign/store.py`, controller, full runtime | `REPLACE INCREMENTALLY` | Reuse CAS/event idioms only. The new facade must not call both authorities. |
| `research/*` | `ADAPT`, selected old loops `DELETE_AFTER_CUTOVER` | Make canonical `ExperimentSpec` the only controller proposal language; provider becomes optional. |
| `scoring/protected.py`, `scoring/submission.py` | `REUSE` / `ADAPT` | Keep exact organizer scoring/alignment; move reservation and protected result types behind the trusted evaluation seam. |
| `candidates/fusion.py` | `ADAPT` | Preserve label-free arithmetic as private declarative RankGraph transforms. |
| `candidates/bootstrap.py`, `campaign/selector.py` | `REPLACE` for decisions | Implement frozen 10,000-cluster-bootstrap, Holm, component guards, exact-vector selection, and dual dispositions. |
| `finalization/replay.py`, bundle publication, organizer check | `REUSE` / `ADAPT` | Preserve exact replay and atomic no-overwrite publication; add named grades and deterministic `BundleId`. |
| `finalization/production.py` | `DELETE_AFTER_CUTOVER` | Replace monolithic reconstruction and multi-authority writes with transactional selection/replay/bundle modules. |
| historical dirty runtime/config changes | `DO NOT TRANSPLANT WHOLESALE` | They include an unreserved protected scorer call, protected feedback into search, non-atomic challenge state, a contradictory 50-query path, and idle-duration padding. |

## Pre-cutover goldens

These small fixtures freeze semantic behavior without embedding dataset rows, protected run outcomes, machine paths, or secrets:

| Fixture | Purpose | SHA-256 |
| --- | --- | --- |
| `official_fm_golden.json` | first-update organizer FM state and exact predictions | `38c0feece48ff90e37d63a2e644e90d8fe1940b759b567753c86bba7be75f761` |
| `organizer_scoring_golden.json` | exact GAUC/nDCG@5 fixture semantics | `ff69468bb7beb57d5e1d6260a4b3c5959c02f3dbe7bc7f57a61ba67dc092d5fc` |
| `rank_fusion_golden.json` | exact rank normalization/composition vector | `5936e2bf3a8ece5e6c0cc3bc253f5cb31d6bf5ab5e40e650aa68c401b8a52edd` |
| `scripted_campaign_golden.json` | three attempts with two completed scientific iterations | `de6100d4a84b8909482636a041a28eb4156f8e386e142d11619faa55ea808939` |

The new tests must read these fixtures and prove equivalent output at the new deep interfaces. Content hashes are recorded after file creation and must change only through an explicit fixture review.

## Phase gates after implementation

- Phase -1: `GO`; the writer map is recorded, fixture hashes are frozen, and the committed-base
  baseline was green.
- Phase 0: `GO`; canonical identity, contract drift, phase-specific final data, and frozen-policy
  tests pass.
- Phase 1: `GO` for implemented local seams; scripted, official-FM fixture, and LightGBM CPU
  same-backend replay pass. The current stock LightGBM build truthfully reports GPU
  `UNSUPPORTED`, so no GPU qualification claim is made.
- Phase 2: `GO` for the new facade's offline slice; it has one `authority.sqlite3`, fenced
  mutations, atomic finalization, idempotent retry, read-only inspection, and deterministic
  projection recovery. The legacy `run`, `resume`, and `finalize` parser shapes remain available
  only as fail-closed migration guards; public campaign mutation cannot enter the old authority.
- Phase 3: `REWORK`; the provider-free scripted fallback produces exact replay and a sealed bundle,
  but official full-data fallback qualification and a real GPU-profile fixture run were not
  possible without the local archive and compatible GPU build.
- Phase 4: `IMPLEMENTED, NOT CAMPAIGN-QUALIFIED`; canonical deterministic proposals, matched-control
  tournament/archive, and the family ledger pass isolated tests but are not yet driven by the
  production `compete` slice.
- Phase 5: `IMPLEMENTED, NOT PROTECTED-QUALIFIED`; frozen user-cluster resampling, Holm correction,
  exact rank graphs, selection rules, and transactional protected-query primitives pass
  deterministic tests. No protected query was made.
- Phase 6: `STOP`; no full-data CPU/GPU rehearsal, measured GPU p95, production cutover, paid
  provider call, protected evaluation, or live six-hour campaign was authorized or performed.
