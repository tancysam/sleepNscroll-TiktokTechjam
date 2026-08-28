# Deterministic candidate seed

This package is the smallest generated-plane implementation of the stable candidate command
contract from `plan.md`:

```text
python candidate.py train --request request.json --output output/
python candidate.py predict --request request.json --checkpoint output/checkpoint/model.npz \
  --output prediction/
```

`candidate.py` owns the executable scientific mechanism: finite numeric capability loading,
training-only standardization, a full-batch logistic objective with fixed deterministic updates,
checkpoint serialization, and label-free prediction. It imports neither the protected scorer nor
trusted controller modules and never computes or declares GAUC, nDCG@5, or the primary benchmark
metric.

The trusted workspace request supplies opaque approved-input handles. The nested `request` object
uses these exact fields for training:

```json
{
  "protocol_schema_version": 1,
  "source_digest": "<sha256>",
  "config_digest": "<sha256 of config.json bytes>",
  "data_digest": "<trusted capability identity>",
  "split_token": "<opaque split identity>",
  "features_handle": "features",
  "targets_handle": "targets"
}
```

Prediction replaces `targets_handle` with `expected_count` and `checkpoint_digest`. Feature and
target capabilities are `.npy` arrays; features have shape `(N, D)`, training targets have shape
`(N,)`, and targets are binary. Prediction writes fixed little-endian float64 `scores.npy` bytes.

The candidate declarations are not authority. The parent process must call
`validate_train_outputs` or `validate_prediction_outputs` from
`kuairand_agent.candidate_api.protocol` before retaining artifacts or sending scores to the
protected evaluator.
