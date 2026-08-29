# Deterministic candidate seed

This package is the smallest generated-plane implementation of the stable candidate command
contract from `plan.md`:

```text
python candidate.py train --request request.json --output output/
python candidate.py predict --request request.json --checkpoint output/checkpoint/model.txt \
  --output prediction/
```

`candidate.py` owns the executable scientific mechanism: finite numeric capability loading,
training-only standardization, a full-batch logistic objective with fixed deterministic updates,
checkpoint serialization, and label-free prediction. It imports neither the protected scorer nor
trusted controller modules and never computes or declares GAUC, nDCG@5, or the primary benchmark
metric.

The trusted workspace request supplies opaque approved-input handles. The nested `request` object
uses these exact fields for training, and the key set is checked exactly — a missing or extra key
is a hard failure:

```json
{
  "protocol_schema_version": 1,
  "source_digest": "<sha256>",
  "config_digest": "<sha256 of config.json bytes>",
  "data_digest": "<trusted capability identity>",
  "split_token": "<opaque split identity>",
  "seed": 0,
  "features_handle": "features",
  "targets_handle": "targets",
  "user_groups_handle": "user_groups"
}
```

Prediction replaces `targets_handle`, `user_groups_handle`, and `seed` with `expected_count` and
`checkpoint_digest`. There is no target or grouping input at prediction time.

Feature, target, and user-group capabilities are `.npy` arrays. Features have shape `(N, D)`,
training targets have shape `(N,)` and are binary, and user groups have shape `(N,)`. Prediction
writes fixed little-endian float64 `scores.npy` bytes.

## The user-group capability

`user_groups` is the trusted per-row user identity for the training split, in canonical row
order. The benchmark ranks **within a user**: each user's own logged impressions are ordered
against each other, and rows belonging to different users are never compared. Any objective that
compares two rows — pairwise, listwise, or grouped — must therefore group by this array, never by
a feature column, and must not construct pairs that span two users.

This seed's objective is pointwise, so it validates the array's shape and otherwise does not
consume it. That is a property of this particular objective, not of the protocol.

## The seed value

`seed` is a validated unsigned 32-bit integer supplied per request by the controller. The same
source and config are executed across inner folds, matched confirmation seeds, and final replay,
so a stochastic mechanism must draw all of its randomness from this value and from nothing else.
This seed's updates are deterministic, so it records the value in the checkpoint and in
diagnostics for replay identity rather than drawing from it.

## Fixed output paths

The training checkpoint must be written to `checkpoint/model.txt`. That path is pinned by the
trusted protocol and verified after the process exits; it is not a free choice, and the file
extension does not constrain the bytes (this seed writes a NumPy archive there). Prediction must
write `scores.npy`.

The candidate declarations are not authority. The parent process must call
`validate_train_outputs` or `validate_prediction_outputs` from
`kuairand_agent.candidate_api.protocol` before retaining artifacts or sending scores to the
protected evaluator.
