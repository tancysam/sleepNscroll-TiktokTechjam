# Does giving an ML research agent memory make it explore?

A negative result, the mechanism we built in response, and the architectural
constraint that turned up when we tried to enforce it.

This document covers what we chose to target and why. The measured campaign results are in
[`RESULTS.md`](RESULTS.md); this is the reasoning behind them.

## 1. The problem we actually observed

Our agent converged honestly and reproduced the official baseline, run after run. It did not fail.
It just kept doing the same thing.

Every campaign starts from the same fixed FM parent, the same frozen benchmark contract, and the
same 33-feature causal bundle. The research model proposes a hypothesis, the trusted controller
implements and evaluates it, and the campaign converges under the organizer's frozen rule
(ε = 0.002, N = 3). Nothing carried across campaign boundaries. A campaign that had already spent
real training compute characterizing an idea left no trace a later campaign could read, so the
next run rediscovered the same idea from scratch.

The obvious fix is memory. We built it, and then measured whether it worked. It did not work the
way we expected, and that is the interesting part.

## 2. What we built

`ResearchLineageLedger` (see [`store.py`](../src/kuairand_agent/campaign/store.py)) is a
project-wide, append-only SQLite ledger of generated-candidate outcomes. Every rejection and every
admission is recorded durably. Admissions carry the real achieved metrics — inner-fold GAUC,
nDCG@5, and primary — against the candidate's *actual parent*, not against the fixed baseline. So a
later campaign can see `pairwise already scored 0.6074 on Fold A, parent 0.6071, promoted: false`
as a concrete data point rather than a bare category flag.

It is scoped by **benchmark digest + starter digest + trusted-controller source digest** together.
All three matter. Scoping on the source digest means that changing our own controller starts a
clean slate, so stale evidence produced by a since-fixed bug can never permanently block a
corrected agent. That decision is deliberate, and §5 describes what it cost us.

Fold A and Fold B are recorded independently. A candidate that fails the cheap Fold B screen never
reaches Fold A, and that is the most common outcome by far; discarding its real Fold B result would
have thrown away most of the evidence the ledger exists to collect.

Alongside it we added `research.proposal_breadth` (K independent proposals per round, set to 2),
on the theory that breadth plus memory would widen the search.

## 3. The negative result

Memory was delivered to the model correctly. It did not change the model's behaviour.

Across the **8 live campaigns** run while memory was advisory-only — before the circuit breaker of
§4 existed — we recorded **24 admission events**. The `pairwise`/BPR family accounts for **17 of
24**, and it was proposed as `candidate-01` in *every single campaign*.

The cleanest window is the five-campaign auto-retry batch, which ran under one unchanged source
digest (`ffd7a64f`) so memory accumulated across all five exactly as designed:

| Proposal family | Admissions |
|---|---|
| `pairwise` | 10 |
| `listwise` | 2 |
| slate-softmax cross-entropy | 1 |
| recency-weighted exponential loss | 1 |
| pointwise binary cross-entropy | 1 |
| **Total** | **15** |

Ten of fifteen, in a run configured for K = 2 breadth, with the contradicting evidence sitting in
the prompt.

**We verified it was actually in the prompt.** This is the step that makes the finding a result
rather than a guess. We read the raw provider-attempt journal for a `propose` call in
`runs/auto-retry-20260830T075125Z` and confirmed that the record `prior_campaign_lesson_0004` —
showing the pairwise family's tied result and `promoted: false` in full detail — was present in the
exact payload sent to the model. There was no plumbing bug. The model received the evidence,
and proposed pairwise anyway.

Two secondary observations point the same way. `proposal_breadth = 2` did not help: two
independently generated proposals per round still clustered on the same family, which is what you
would expect from two draws against the same strong prior rather than genuine diversification. And
the non-pairwise families in the table above are mostly *reformulations of the same pointwise or
ranking-loss idea* under different names, not structurally different directions.

### What we conclude, and what we do not

We conclude that **advisory memory is not sufficient** to change proposal behaviour in this setup.
Making evidence available to a model is not the same as making it act on that evidence.

We are not claiming a general result about LLM agents. This is one model, one benchmark, one
prompt design, n = 8 campaigns, and it is observational rather than controlled — we did not run a
memory-off arm alongside a memory-on arm. A stronger claim would need that comparison. What we can
say is that within this system, on the evidence we gathered, the mechanism we built to change the
agent's behaviour demonstrably did not change it.

There is also a legitimate competing explanation we cannot fully rule out and should not hide:
**pairwise may simply be the correct first move.** The training objective is pointwise log loss
while the scored metrics (GAUC, nDCG@5) are ranking metrics, and closing that gap is the
organizers' own top-ranked open direction. A model that keeps proposing it may be exercising good
judgement, not blind repetition. The evidence against that reading is narrower than it looks: the
model kept proposing it *after being shown its own flat result*, and never proposed anything
outside the loss-function axis — not features, not sampling, not calibration.

## 4. The response: enforcement, not persuasion

This project already holds a rule for cases like this: **a safety or quality property must never
depend on prompt compliance alone.** The outer-query budget is not a request to the model; it is a
ledger the controller enforces. Leakage controls are not instructions; they are gates.

So we moved the constraint out of the prompt and into the trusted controller. `_proposal_family_is_blocked`
(see [`research/production.py`](../src/kuairand_agent/research/production.py)) now
deterministically refuses a family whose cross-run evidence shows it already reached a full
inner-fold evaluation and lost, whether that evidence came from this campaign or an earlier one.

The threshold is deliberately asymmetric:

- A **cheap pre-admission rejection** (static policy, materiality) needs two strikes.
- A **full evaluation that lost** needs one. That comparison already cost a real training run, and
  the organizer's own metric already returned its verdict.
- A **screen-rejected** candidate records no `promoted` value at all, so a single cheap-fold miss
  never exiles a family.
- A family that **was promoted** stays eligible.

## 5. The constraint we hit: memory that resets cannot enforce

The fix did not fire on the run that introduced it, and could not have.

Changing `production.py` to add the block changed the trusted-controller source digest. The ledger
is scoped by that digest. So the campaign that introduced the enforcement mechanism saw an empty
ledger, had no evidence to enforce against, and admitted a pairwise candidate immediately.

This is not a bug in either mechanism. It is a real tension between two properties we want at once:

- **Clean-slate-on-source-change** exists so a corrected agent is never held hostage by evidence
  its own bug produced.
- **Cross-run enforcement** requires evidence that survives long enough to be enforced against.

They conflict exactly when the code change *is* the enforcement mechanism. The block needs two
consecutive campaigns on an unchanged tree before it can act, and our development loop kept
changing the tree. Our ledger shows the cost plainly: **4 distinct source scopes across 8
campaigns** — memory reset three times, each time for a legitimate reason.

We do not think the answer is to weaken the clean-slate rule, which protects something important.
The more promising direction is to separate *evidence about the benchmark* (which survives our code
changes, because a flat Fold A result is a fact about the data and the method, not about our
controller) from *evidence about our own controller's failures* (which should reset). That
distinction did not exist in the schema when we built it, and adding it is the change we would make
next.

Notably, the codebase already contains the right pattern. `OuterQueryLedger` — which rations the
scarcest resource in the project — is scoped by benchmark, dataset, and scorer digest, *not* by our
source, and deliberately survives code changes. We simply did not apply that scoping to the
lineage ledger.

## 6. We tested enforcement. It worked, and it did not help.

To get around §5 we froze the tree and ran two campaigns back to back, so the second inherited the
first's evidence. The source digest was `79ab6cbf` before, between, and after both runs, so the
test is valid.

**The first campaign** proposed `pairwise`, trained it, and scored Fold A at 0.6071290 against a
parent of 0.6071290 — flat to seven decimal places — and recorded `promoted: false`. That is the
breaker's trigger condition, written durably.

**The second campaign** inherited it and ran 19 candidates:

| Outcome | Count | Detail |
|---|---|---|
| `pairwise` → **blocked** | **14** | `novelty_policy/proposal_family_blocked`, refused on inherited evidence |
| non-pairwise → admitted | 3 | Fold B `0.5754240304231644` vs parent `0.5754240304231644` |

Three findings, and only the first is good news.

**1. Cross-run enforcement works.** The breaker fired 14 times against evidence from a different
campaign, and never deadlocked the run. The mechanism does what it was built to do.

**2. It did not produce exploration.** The model proposed the blocked family *fourteen times*. Its
non-blocked alternatives scored bit-identical to the parent — not close, identical to the last
digit — and one of them, candidate-19, had the objective "pointwise binary logistic loss on
long_view with L2 regularization", which is a restatement of the parent's own objective. Blocked
from its preferred move, the model largely proposed the baseline back at itself.

The honest conclusion is that blocking did not unlock a better idea; it **revealed that there was
no better idea waiting**. We had assumed the model was ignoring evidence it understood. The
stronger reading now is that the proposal step has no mechanism for generating a genuinely
different hypothesis once its preferred one is unavailable. That relocates the bottleneck: not
memory, not enforcement, but hypothesis generation.

**3. The family classifier can be evaded by rewording.** `_proposal_family` matches four hardcoded
markers and otherwise falls back to slugifying the objective text, so those three admissions were
recorded under families like `the-exact-parent-s-pointwise-binary-logistic-loss-changed-only-b`.
Every near-duplicate proposal earns a *unique* slug, so the breaker can never block it. Enforcement
is therefore only as good as the classifier, and ours is a substring match over four families.

**And enforcement broke finalization.** The second campaign died emitting its report:
`failures_and_recoveries entries must be unique`. All 14 blocks rendered byte-identical text,
because the line format discarded the iteration and candidate fields it had already validated.
Adding cross-run enforcement is precisely what made this reachable — before it, the block needed
two strikes inside a single campaign and effectively never fired twice. Fixed, with a regression
test confirmed to fail without it. It is the fourth defect this project found by running the system
live rather than by reasoning about it.

## 7. What this cost, and what it bought

The honest accounting: neither cross-run memory nor its enforcement produced a better model. No
generated candidate has beaten the baseline by more than ε. The measurable improvement in this
project came from elsewhere — four production defects found by running the system live for real,
each root-caused from an actual failure and fixed with a regression test confirmed to fail without
it (see [`RESULTS.md`](RESULTS.md)).

What memory bought is the evidence base this document rests on. Without the ledger we would have
had an anecdote — "it seems to keep trying pairwise". With it we have 46 typed, durable records
with real metrics attached across 10 campaigns, which is what let us state the problem precisely,
build a mechanism against it, test that mechanism, and then discover the mechanism was aimed at
the wrong target.

That last step is the one worth keeping. We believed the agent was ignoring evidence it
understood; the enforcement experiment showed instead that it had nothing else to offer. We would
not have found that by reasoning about the design, and we would not have found it without durable
cross-run records to test against. The most valuable thing the memory produced was a corrected
belief about our own agent.

**What we would build next, in order:**

1. **Split the ledger's scope** so benchmark facts survive our code changes while controller
   failures reset — using the digests each scientific run already records (`data_digest`,
   `scorer_digest`, `training_policy_digest`), the same way `OuterQueryLedger` is already scoped.
   This removes the two-campaigns-on-a-frozen-tree requirement entirely.
2. **Mandate distinct hypothesis lanes per breadth slot** — loss function, feature interaction,
   negative sampling, calibration — drawn from ledger-unexhausted lanes. This is the direct
   response to finding 2: the proposer needs somewhere to go, not just somewhere it cannot go.
3. **Replace the substring family classifier** with something a reworded objective cannot evade,
   since enforcement is only ever as good as its classifier.

That is the loop the challenge is actually asking an agent to run, applied to the agent itself.
