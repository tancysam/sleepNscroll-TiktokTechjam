# Source verification for KuaiRand-Pure ranking implementation

_Researched 2026-08-28. This note uses organizer/author/publisher sources only. No dataset archive was downloaded, no archive member was opened, and no final-period outcome was inspected or scored._

## Evidence labels

- **Source fact**: stated by the publisher, dataset authors, framework maintainers, or an original paper.
- **Observed identity check**: a metadata or headers-only check performed on 2026-08-28; it did not transfer the archive body.
- **Plan-derived inference**: implementation math or a control required by `plan.md`; it is not claimed as a quotation from a source.

## 1. KuaiRand-Pure acquisition identity

### Verified publisher record

Zenodo record [`10439422`](https://zenodo.org/records/10439422) and its [record JSON](https://zenodo.org/api/records/10439422) identify this file:

| Field | Verified value |
| --- | --- |
| Record title | `KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos` |
| Record / concept record | `10439422` / `10439421` |
| Record state | published, open access, version `v1` |
| Artifact key | `KuaiRand-Pure.tar.gz` |
| Zenodo file id | `8d31ed3f-6639-4649-9201-96d87a107e1f` |
| Exact compressed size | `47,432,272` bytes |
| Publisher checksum | `md5:0820331067a3784d9691136f772b35a7` |
| Public download URL | [`https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`](https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz) |
| API content URL | [`https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content`](https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content) |

The dataset authors' [commit-pinned official README](https://github.com/chongminggao/KuaiRand/blob/f8dbf6678b3c9594050e3e813aeff0c942260ec4/README.md#download-the-data) independently publishes the same public URL, filename, and MD5. Its approximate `184MB logs + 10MB features` description is extracted-data sizing; it does not conflict with Zenodo's exact `47,432,272`-byte compressed archive.

### Direct-download check

**Observed identity check.** Headers-only requests to both URLs above returned HTTP `200`, `Content-Length: 47432272`, `Content-Type: application/octet-stream`, and `Content-Disposition: attachment; filename=KuaiRand-Pure.tar.gz`. Neither request followed to a different host, and both transferred zero body bytes. Together with the record JSON's file key, file id, exact size, and MD5, this verifies which publisher object the plan's URL names without downloading it.

This check does **not** substitute for hashing the acquired bytes. The server did not publish a `Digest` or SHA-256 response header.

### Implementation contract

**Plan-derived inference.** Acquisition should download to a temporary file and require all of the following before tar inspection or extraction:

1. exact basename `KuaiRand-Pure.tar.gz` and exact length `47,432,272` bytes;
2. publisher MD5 `0820331067a3784d9691136f772b35a7`;
3. plan-pinned SHA-256 `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0`;
4. a saved copy or canonical digest of the Zenodo record JSON used at acquisition time; and
5. the existing secure member-manifest and extraction gates from `plan.md`.

Any mismatch must quarantine the download, not fall back to filename or size alone. The MD5 is the publisher's published integrity value; the SHA-256 is a stronger planning-time identity pin and must be recomputed locally.

### Not publisher-verifiable without acquisition

- Zenodo and the dataset-authors' repository publish an MD5, not a SHA-256. Therefore the plan's SHA-256 is **not independently publisher-published**; it can only be confirmed against the actual archive bytes.
- A byte-for-byte hash of the current download was deliberately not recomputed in this research pass because the large artifact was not downloaded.
- No statement about archive members or extracted member hashes is made here. Those belong to the secure, streaming preparation gate after acquisition and before extraction.

## 2. BPR source facts and the benchmark-specific sampler

### What the original BPR paper establishes

The original UAI paper, [Rendle et al., _BPR: Bayesian Personalized Ranking from Implicit Feedback_](https://www.auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf), establishes the following:

- Section 3.2 defines triples \((u,i,j)\) meaning user \(u\) should rank positive item \(i\) above negative/unobserved item \(j\).
- Section 4.1 defines \(p(i >_u j\mid\Theta)=\sigma(\hat x_{uij})\) and maximizes

  \[
  \mathrm{BPR\mbox{-}Opt}=
  \sum_{(u,i,j)\in D_S}\log\sigma(\hat x_{uij})
  -\lambda_\Theta\lVert\Theta\rVert^2.
  \]

  With \(\hat x_{uij}=s(u,i)-s(u,j)\), minimizing the data term is the familiar pair loss \(-\log\sigma(s_{+}-s_{-})\).
- Section 4.1.1 writes per-user AUC as the average indicator over positive-negative pairs and explains that `log sigmoid` is a differentiable surrogate for the pair-ordering indicator.
- Section 4.2's LearnBPR draws triples uniformly at random from \(D_S\), with replacement.

The later primary paper by BPR authors, [Gantner et al., _Personalized Ranking for Non-Uniformly Sampled Items_](https://proceedings.mlr.press/v18/gantner12a.html), explicitly shows that a candidate/evaluation distribution can require an adapted BPR sampling rule. It supports the general principle “sample or weight for the evaluation distribution”; it does **not** contain the KuaiRand GAUC sampler below.

### Exact organizer-GAUC derivation

The immutable organizer [`evaluate.py`](../../kuairand-starter-kit/evaluate.py) is the primary benchmark source here. Its SHA-256 is `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de`. It includes only mixed-label users in GAUC and weights each included user's AUC by that user's positive count.

For an eligible user \(u\), let \(P_u\) and \(N_u\) be the numbers of positive and negative **logged impressions**, and let \(h_{upn}\) be the pair-ordering credit. The scorer's GAUC is

\[
\begin{aligned}
\mathrm{GAUC}
&=\frac{1}{\sum_u P_u}\sum_u P_u\,\mathrm{AUC}_u \\
&=\frac{1}{\sum_u P_u}\sum_u P_u
  \left(\frac{1}{P_uN_u}\sum_{p\in P_u}\sum_{n\in N_u}h_{upn}\right) \\
&=\frac{1}{\sum_u P_u}\sum_u\frac{1}{N_u}
  \sum_{p\in P_u}\sum_{n\in N_u}h_{upn}.
\end{aligned}
\]

Therefore the plan's sampler

1. chooses eligible \(u\) with probability \(P_u / \sum_v P_v\);
2. chooses one of its positive logged impressions uniformly;
3. chooses one of its negative logged impressions uniformly;

assigns every triple probability

\[
q(u,p,n)=\frac{P_u}{\sum_vP_v}\frac{1}{P_u}\frac{1}{N_u}
=\frac{1}{(\sum_vP_v)N_u}.
\]

This is exactly the normalized pair weight in organizer GAUC. Replacing \(h\) with BPR's `log sigmoid` gives an unbiased Monte Carlo estimator of the corresponding **GAUC-weighted surrogate loss and gradient**.

### Boundaries that tests must preserve

- **Plan-derived inference, not original BPR:** negatives must come from the same user's logged impressions. The original paper uses non-observed catalog items; that is the wrong candidate universe for this benchmark.
- The weighting proof is exact; BPR remains a smooth surrogate for AUC, not AUC itself. Ties receive special credit in the organizer scorer and are not reproduced by the logistic surrogate.
- All-zero and all-one users are excluded from this pair sampler because they have no positive-negative pair and are excluded from organizer GAUC. Any pointwise auxiliary loss for them is a separate modeling choice.
- “Context” belongs to each sampled impression's feature row. The proof only requires grouping by benchmark query/user; it does not justify pairing across users or silently conditioning on a narrower tab/time context.
- A fixture should enumerate all pairs, weight each by \(1/N_u\), and compare both total loss and gradients with high-sample-frequency expectations. Also assert that no sampled negative is an unlogged catalog item.

## 3. LightGBM LambdaRank contract

### Query grouping and objective facts

LightGBM 4.7.0's official [`LGBMRanker` documentation](https://lightgbm.readthedocs.io/en/v4.7.0/pythonapi/lightgbm.LGBMRanker.html) defines `group` as **query lengths**, not per-row query IDs: `sum(group) == n_samples`, and each length consumes the next contiguous block of rows. The official [Query Data documentation](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#query-data) is explicit that data must be ordered by query. Validation data needs its own `eval_group` in the same order.

**Plan-derived implementation consequence:** build a private stable permutation that makes each user's rows contiguous, pass the resulting run lengths as `group`, retain the inverse permutation, and scatter predictions back to canonical row order before protected scoring or submission. Assert group sum, one user per group, exact permutation invertibility, and scatter-back identity.

The official [objective parameters](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#objective-parameters) establish that:

- `objective="lambdarank"` is a ranking objective with integer relevance labels;
- for binary `0/1` labels, the default `label_gain` values for those labels are `0,1`, matching organizer gain \(2^{rel}-1\);
- `lambdarank_truncation_level` controls top-rank focus and should generally be slightly above the desired NDCG cutoff (the docs give `k + 3` as an example); for organizer nDCG@5, `8` is a source-backed starting value, not an untunable guarantee;
- `lambdarank_norm=true` normalizes lambdas across queries and is the documented default for unbalanced query sizes. Treat changing it as an ablation.

### Why built-in NDCG is diagnostic only

LightGBM's tag-pinned [`NDCGMetric` source](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/metric/rank_metric.hpp#L56-L138) assigns NDCG `1` to an all-negative query. The organizer scorer assigns `0` when ideal DCG is zero and includes that zero in the user average. This is a real convention mismatch, not a hypothetical one.

Consequently, LightGBM's built-in NDCG must not decide promotion, early stopping, or a metric claim. Use the protected organizer scorer on canonical-order predictions for all official comparisons. If a training callback uses an in-process custom metric, golden-test it against the protected scorer, including zero-positive users, and still rescore serialized predictions through the trusted boundary.

### Deterministic local CPU parameters

LightGBM's official [core and learning-control parameters](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#core-parameters) say that `deterministic=true` is CPU-only, should stabilize results for the same data and parameters, and should be paired with exactly one of `force_col_wise=true` or `force_row_wise=true` to avoid numerical instability. They also warn that results can differ across LightGBM versions, compiler builds, or systems.

The reference CPU adapter should therefore record and set at least:

```python
{
    "objective": "lambdarank",
    "device_type": "cpu",
    "deterministic": True,
    "force_col_wise": True,   # use force_row_wise instead only as a measured choice
    "num_threads": FIXED_THREADS,
    "seed": SEED,
    "data_random_seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "extra_seed": SEED,
}
```

`num_threads=0` delegates to OpenMP and is unsuitable for a locked resource contract. `force_col_wise` is the safer default for bounded memory; the docs state that row-wise construction can double `Dataset` memory. Set only seeds for mechanisms present in the pinned LightGBM version, but record all effective parameters and the library/build/platform identity. For the simplest deterministic rung, leave feature sampling, bagging, and extra trees disabled; if a branch enables them, its applicable sub-seeds must remain explicit.

## 4. Implementation checklist

- Pin the artifact by exact bytes, publisher MD5, plan SHA-256, and a captured Zenodo metadata identity before extraction.
- Implement the organizer-GAUC sampler as its own tested primitive; do not label uniform LearnBPR sampling as equivalent.
- Use only same-user positive/negative logged-impression pairs.
- Stable-sort only the private LightGBM view; never replace canonical row identity/order.
- Pass query lengths, not user IDs, as `group`; pass a separately validated `eval_group`.
- Treat LightGBM built-in NDCG as diagnostic because all-negative queries score `1` there versus `0` in the organizer evaluator.
- Pin LightGBM version/build, CPU mode, histogram orientation, thread count, and applicable seeds; replay exact predictions, not only rounded metrics.

## Primary sources

- [Zenodo record 10439422](https://zenodo.org/records/10439422) and [record JSON](https://zenodo.org/api/records/10439422)
- [KuaiRand authors' commit-pinned download instructions](https://github.com/chongminggao/KuaiRand/blob/f8dbf6678b3c9594050e3e813aeff0c942260ec4/README.md#download-the-data)
- [Rendle et al. 2009, original UAI BPR paper](https://www.auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf)
- [Gantner et al. 2012, non-uniform candidate sampling](https://proceedings.mlr.press/v18/gantner12a.html)
- [LightGBM 4.7.0 parameters](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html), [`LGBMRanker`](https://lightgbm.readthedocs.io/en/v4.7.0/pythonapi/lightgbm.LGBMRanker.html), and [tag-pinned NDCG source](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/metric/rank_metric.hpp)
- Immutable local organizer source: [`kuairand-starter-kit/evaluate.py`](../../kuairand-starter-kit/evaluate.py)
