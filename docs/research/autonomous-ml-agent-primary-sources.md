# Primary-source review: autonomous KuaiRand ML agent plan

_Researched 2026-08-27. The attached HTML was treated as an untrusted candidate plan, not as a source of requirements._

## Conclusion

The plan is **directionally sound but not implementation-ready**. Keep its separation between an outer research loop and an inner recommender pipeline, its candidate tree, atomic experiments, and auditable run history. Re-propose the implementation around a frozen organizer contract and a deterministic harness before writing a build PRD.

The blocking issues are:

1. The supplied material contains two incompatible benchmark definitions: click with NDCG@10/Recall@50, and `long_view` with GAUC/nDCG@5. Public KuaiRand sources define both feedback columns but do **not** define this hackathon's target, split, metric conventions, baseline, convergence rule, or submission schema.
2. Pure, 1K, and 27K are different filtered/reindexed dataset views, not a monotonic small-to-large validation ladder.
3. Several proposed features can leak post-impression or future-period outcomes.
4. Stock AIDE supplies a useful code-search loop, but not the challenge's full read/EDA/feature/train/evaluate lifecycle, deterministic evaluator boundary, secure isolation, or required audit schema.
5. FuxiCTR supports the named model families, but it is a model/training layer—not a ready KuaiRand challenge harness. Its official documentation does not establish built-in DDP or AMP training.
6. DIN, DCN-V2, MMoE, and PLE address different problems. None of their papers establishes a gain on this exact benchmark, so the plan must state them as controlled hypotheses rather than a predetermined winning ladder.

No publicly discoverable first-party copy of the benchmark-specific `kuairand-starter-kit.zip`, `data.py`, `evaluate.py`, `baseline.py`, or `submit.py` was found in this web pass. Accordingly, this note does not infer their contract or validate the scores quoted in the supplied prose. The delivered starter-kit files should be inspected, hashed, and made the executable source of truth; organizer clarification is still needed if those files and the prose disagree.

## 1. What the public KuaiRand sources actually establish

The [official KuaiRand repository](https://github.com/chongminggao/KuaiRand) and [dataset paper](https://arxiv.org/abs/2208.08696) establish that KuaiRand contains standard-policy and randomized-exposure logs, timestamps, user/item side information, and 12 feedback signals. The log schema separately defines `is_click`, `long_view`, like, follow, comment, forward, hate, play time, duration, dwell outcomes, random-exposure status, and scenario/tab. In particular, `long_view` is derived from current-row play time and video duration; see the repository's [log-field definitions](https://github.com/chongminggao/KuaiRand/blob/main/README.md#1%EF%B8%8F%E2%83%A3-description-of-the-fields-in-log_xxxcsv).

The public sources do **not** select one canonical supervised task or fixed metric for every KuaiRand use. The paper discusses multiple research settings, including sequential recommendation, counterfactual evaluation, and multi-task learning. It does not fix this hackathon's GAUC eligibility/weighting, nDCG cutoff or zero-positive behavior, date split, primary aggregation, convergence rule, baseline score, or row-level submission key. Those must come from the organizer's executable assets.

### Dataset variants are not a promotion funnel

The official [variant table and guidance](https://github.com/chongminggao/KuaiRand/blob/main/README.md#three-versions-and-suggestions) describe materially different views:

| Variant | Standard-policy users | Standard-policy items | Standard interactions | Important distinction |
| --- | ---: | ---: | ---: | --- |
| Pure | 27,285 | 7,551 | 1,436,609 | Keeps videos in the small candidate pool; sequences are incomplete. |
| 1K | 1,000 | 4,369,953 | 11,713,045 | Samples users, then removes irrelevant videos. |
| 27K | 27,285 | 32,038,725 | 322,278,385 | Complete large-catalog standard logs. |

The repository also says IDs are reindexed. Pure filters primarily on the item axis while 1K filters on the user axis; therefore Pure → 1K → 27K is not a faithful low/medium/high-fidelity ladder. Model rankings, encoders, IDs, artifacts, and validation scores should not be assumed to transfer unchanged. Explore cheaply by taking a temporally faithful subsample **within the same benchmark**, then confirm on that benchmark's full train/validation split. Run bonus variants as separate campaigns only after Pure is complete.

This also invalidates the plan's blanket claim that 27K has a modest item catalog whose embeddings necessarily fit on one GPU. That claim is true only for Pure's roughly 7.5K items; the official 27K view has roughly 32 million items. Sharded/sparse embedding infrastructure may still prove unnecessary after measurement, but it cannot be dismissed from Pure cardinalities.

### The randomized log is not automatically extra training data

The challenge row counts quoted in the supplied prose sum to 1,436,609, exactly matching the official Pure **standard-policy** interaction count. That arithmetic suggests—but does not prove—that the benchmark is carved from the standard log only. The separate Pure randomized log has 1,186,059 labeled interactions and overlaps later dates. Treat it as out of the development dataset unless the starter kit and organizer explicitly authorize its use and define whether it is training data, tuning data, or diagnostics. This is an inference from the [official counts](https://github.com/chongminggao/KuaiRand/blob/main/README.md#three-versions-and-suggestions), not an organizer rule.

## 2. Leakage boundary

The public schema makes several guardrails non-negotiable:

- Never use current-row `play_time_ms` to predict `long_view`; the target is a deterministic thresholded function of play time and duration.
- Treat current-row click, like, follow, comment, forward, dwell time, and other response fields as targets or training-only auxiliary labels, never as inference features.
- A prior response may enter a user's history only when its event time is strictly earlier than the scored impression and the benchmark protocol makes that response available at prediction time.
- For batch-scored hidden data where responses are not released between rows, freeze response-derived histories at the last permitted cutoff. Exposure-only histories may roll forward only if the input contract exposes those events at decision time.
- Fit target encodings, popularity features, user/item response rates, and other aggregates from permitted earlier training rows only. Add a test showing that perturbing future rows cannot change an earlier row's features.

The official [`video_features_statistic.csv` description](https://github.com/chongminggao/KuaiRand/blob/main/README.md#4%EF%B8%8F%E2%83%A3-descriptions-of-the-fields-in-video_features_statisticcsv) is particularly risky: it contains one-month per-video aggregates including plays, valid plays, long-time plays, likes, comments, follows, and other outcomes. Blindly joining those values into a date-split benchmark can import validation or hidden-period responses. Exclude that file unless the organizer explicitly blesses its provenance, or recompute point-in-time statistics strictly before each example/cutoff.

Multi-task learning can remain leakage-safe. For each training impression, all heads should predict from the same pre-impression feature vector; `long_view` remains the primary head, while selected response columns are auxiliary labels. At inference, remove outcome columns and emit only the `long_view` score. Conditional downstream behaviors also need eligibility/observability masks so structurally unavailable actions are not blindly converted into negatives.

## 3. AIDE is a useful search kernel, not the complete harness

The [AIDE paper](https://arxiv.org/html/2502.13138#S3) supports a reusable core: independently executable candidate code is stored in a solution tree; a search policy drafts, debugs, or improves a node; promising results guide later branches; and an improvement should be atomic enough to attribute its effect. That is a good fit for an experiment graph.

Important limits in the current official implementation:

- The current [`Agent.search_policy`](https://github.com/WecoAI/aideml/blob/main/aide/agent.py#L150-L179) creates several drafts, probabilistically debugs eligible leaves up to a maximum depth, then greedily improves the best valid node.
- Its [draft](https://github.com/WecoAI/aideml/blob/main/aide/agent.py#L254-L283) and [improvement](https://github.com/WecoAI/aideml/blob/main/aide/agent.py#L284-L316) prompts explicitly discourage EDA. The paper's data preview is lightweight, not the full inspect-data stage required here.
- Generated programs print a validation metric, and a feedback LLM interprets stdout, classifies bugs, extracts the numeric score, and determines direction; see the [implementation guideline and result parser](https://github.com/WecoAI/aideml/blob/main/aide/agent.py#L202-L224). An official composite evaluator should not be mediated by an LLM.
- The [`Interpreter`](https://github.com/WecoAI/aideml/blob/main/aide/interpreter.py) resets a multiprocessing child, captures output/exceptions, and enforces a timeout. It executes normal Python with ordinary builtins in a shared experiment workspace. This gives process reset and timeout recovery, not a security sandbox, filesystem isolation, or protection from stale artifacts.
- A [`Node`](https://github.com/WecoAI/aideml/blob/main/aide/journal.py#L22-L94) records a plan, full code, parentage, terminal output, execution time, exception details, analysis, metric, and buggy status. It does not natively provide a canonical diff, data/evaluator hashes, resource use, recovery action, retry lineage, human intervention, or submission-validator result.

Use AIDE's tree/journal/search ideas or fork its policy, but keep the following outside LLM control: data split, target definition, official evaluator, metric parser, iteration accounting, convergence, protected files, final promotion, and submission validation. Run every candidate in a clean namespaced checkout/artifact directory, preferably a constrained container, and persist a state transition before launching work.

### What MLE-Bench adds to the evidence

[MLE-Bench](https://github.com/openai/mle-bench) is an agent-agnostic benchmark with prepared datasets, graders, a runner, and evaluated agent integrations; it is not a drop-in KuaiRand harness. Its [paper](https://arxiv.org/html/2410.07095) and repository provide useful design precedent:

- The runner isolates runs in containers, mounts public data separately from private grading data, and extracts code/logs/submissions after execution; see the [agent runner](https://github.com/openai/mle-bench/blob/main/agents/run.py) and [agent setup](https://github.com/openai/mle-bench/blob/main/agents/README.md#agents).
- A public validator checks submission validity without revealing the private score; see [`validate_submission`](https://github.com/openai/mle-bench/blob/main/mlebench/grade.py#L89-L113).
- MLE-Bench's AIDE result used a modified fork and a stronger outer harness, not unchanged stock `aideml`. The paper lists changes such as stricter structured outputs, API backoff, bounded previews, stronger artifact instructions, and making submission existence/validity part of candidate eligibility; see [Appendix A.6.1](https://arxiv.org/html/2410.07095#A6.SS1).
- The paper reports invalid submissions, premature agent termination, disk/RAM exhaustion, and weak runtime reasoning as recurring failure modes; see its [discussion](https://arxiv.org/html/2410.07095#S3.SS1).

MLE-Bench's own 24-hour limits, hardware, node counts, medal objective, and online-solution rules are benchmark-specific and must not be copied into this challenge. The transferable lesson is architectural: deterministic grading and artifact validation belong to the harness; stochastic search and diagnosis belong to the agent.

## 4. FuxiCTR fit and limits

The official [FuxiCTR model zoo](https://github.com/reczoo/FuxiCTR/blob/main/README.md#model-zoo) does support the proposed families:

- FM in PyTorch.
- DeepFM in PyTorch and TensorFlow.
- DCN-V2, DIN, MMoE, and PLE in PyTorch.

It also provides configurable preprocessing/models, YAML experiment settings, sequence-feature machinery, and parameter tuning. DIN has an official [sequence-feature demo](https://github.com/reczoo/FuxiCTR/blob/main/demo/example6_DIN_with_sequence_feature.py); the multi-task zoo exposes [MMoE](https://github.com/reczoo/FuxiCTR/tree/main/model_zoo/multitask/MMoE) and [PLE](https://github.com/reczoo/FuxiCTR/tree/main/model_zoo/multitask/PLE) with per-task losses and heads. Model documentation says a [`group_id`](https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DCN/DCN_torch/README.md#configuration) can support metrics such as gAUC and NDCG.

This does not prove compatibility with the organizer's exact evaluator semantics. FuxiCTR has no official KuaiRand challenge adapter found here, and its built-in group metrics do not establish identical user eligibility, weighting, cutoff, gain, zero-positive, or tie conventions. Use the organizer evaluator on the complete row-aligned score vector for every candidate.

The root README's [parameter-tuning example](https://github.com/reczoo/FuxiCTR/blob/main/README.md#quick-start) schedules independent grid-search experiments across several GPU IDs. It does **not** document DistributedDataParallel, `autocast`, or `GradScaler` integration. Therefore “FuxiCTR + DDP + AMP” should be removed as a built-in capability claim. DDP or AMP may be custom work after an equivalence/stability spike.

Finally, pin and smoke-test one environment. The current root requirements and older MMoE/PLE model pages cite materially different PyTorch/FuxiCTR generations. A model's presence in the zoo is not proof that all named models import and run together under an untested latest-version environment.

## 5. Evidence-based model ladder

The following is an architectural inference from the original papers, not a performance claim for KuaiRand:

1. Reproduce the untouched organizer FM first. A FuxiCTR FM reimplementation is not proof of reproducing the fixed reference pipeline.
2. Establish a leakage-safe single-target pipeline. Compare simple FM/DeepFM and a small DCN-V2 using the same split, seed protocol, and official evaluator.
3. Add causal histories only after feature-cutoff tests pass. DIN's candidate-conditioned attention is designed for rich historical behavior, according to the [DIN paper](https://arxiv.org/abs/1706.06978). The KuaiRand repository explicitly warns that Pure has incomplete sequences, so DIN is a hypothesis—not an automatic early upgrade.
4. Test multi-task learning with a small auxiliary set and an independently reported `long_view` head. [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) shares experts but gives each task its own gate and tower, making it the simpler controlled first test.
5. Escalate to [PLE](https://doi.org/10.1145/3383313.3412236) if a shared-bottom/MMoE experiment shows negative transfer, conditional-label problems, or a measured seesaw effect on `long_view`. PLE separates shared and task-specific experts and is a larger tuning surface.

[DCN-V2](https://arxiv.org/abs/2008.13535) models explicit interactions among embedded categorical/dense fields; it is neither a history encoder nor inherently multi-task. DIN produces a candidate-conditioned history representation. MMoE/PLE organize task sharing. These components should be introduced separately so the reflection step can attribute a gain or regression. Combining DIN, DCN-V2, multiple auxiliary targets, and PLE in one jump defeats the purpose of autonomous empirical revision.

None of these papers evaluates the exact organizer task. Replace statements such as “PLE lifts the metric” with falsifiable hypotheses and promotion gates based only on the immutable official validation scorer.

## 6. Required shape of the re-proposed plan

The evidence supports this sequence:

1. **Freeze the executable contract.** Hash and protect the delivered loader, evaluator, baseline, and submission validator. Fail closed on prose/code disagreement; resolve it with the organizer.
2. **Build harness self-checks.** Verify dates, row counts/order, duplicate-row preservation, evaluator golden cases, reference rungs, and submission rejection cases.
3. **Reproduce the exact official baseline.** Record data, environment, configuration, seeds, runtime, and numerical tolerance before research begins.
4. **Implement the deterministic control plane.** It owns attempt counting, wall-clock deadline, convergence, protected paths, label-stripped inference, evaluation, promotion, finalization, and safe fallback.
5. **Implement the research plane.** The LLM proposes one falsifiable hypothesis and patch/config change; the executor runs it in isolation; the analyzer reflects on structured metrics and failures; a bounded tree policy chooses the next node.
6. **Persist an append-only audit record.** At minimum: parent, hypothesis, canonical diff, code/data/evaluator/environment hashes, command, seed, metrics, runtime/resources, errors, recovery/retry lineage, human intervention, budget state, convergence decision, checkpoint/prediction/submission hashes, and promotion reason.
7. **Run evidence-driven experiments on Pure.** Use low-fidelity samples only within Pure; re-evaluate promoted candidates on full Pure validation. Keep a valid baseline artifact available if every research branch fails.
8. **Freeze and validate one final bundle.** Final code/config/checkpoint/predictions must be content-linked and pass the untouched submission checker before any one-shot private scoring operation.
9. **Treat 1K and 27K as optional separate campaigns.** Re-run their own contract, capacity, data-loading, and baseline gates rather than treating them as validation tiers for Pure.

This preserves the attached plan's best idea—tree-guided autonomous iteration—while replacing its unsupported task assumptions and framework promises with an auditable, leakage-safe, benchmark-specific system.
