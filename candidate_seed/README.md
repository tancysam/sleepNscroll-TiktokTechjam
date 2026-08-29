# Deterministic candidate seed

This package is the smallest generated-plane implementation of the stable candidate command
contract from `plan.md`:

```text
python candidate.py train --request request.json --output output/
python candidate.py predict --request request.json --checkpoint output/checkpoint/model.txt \
  --output prediction/
```

`candidate.py` is the stable protocol entrypoint. It owns finite numeric capability loading,
request validation, checkpoint serialization, result writing, and the command-line interface.
`model_impl.py` is the deliberately small mutable scientific surface: it owns the objective,
optimization, model-specific checkpoint arrays, diagnostics, and label-free prediction.
Autonomous implementations may replace `model_impl.py`, `config.json`, and reachable helper
modules. They may not replace `candidate.py`; admission enforces that rule rather than relying on
prompt compliance. Keeping the entrypoint immutable avoids repeatedly regenerating risky protocol
and NumPy I/O plumbing. Neither file imports the protected scorer or trusted controller
modules, and neither computes or declares GAUC, nDCG@5, or the primary benchmark metric.

The trusted workspace request supplies opaque approved-input handles. The nested `request` object
uses these exact fields for training:

```json
{
  "protocol_schema_version": 1,
  "source_digest": "<sha256>",
  "config_digest": "<sha256 of config.json bytes>",
  "data_digest": "<trusted capability identity>",
  "split_token": "<opaque split identity>",
  "seed": 7,
  "features_handle": "features",
  "targets_handle": "targets",
  "user_groups_handle": "user_groups"
}
```

Prediction replaces `targets_handle` with `expected_count` and `checkpoint_digest`. Feature and
target capabilities are `.npy` arrays. Features are a finite float64 `(N, D)` matrix engineered by
the controller in the exact order named by the safe-context
`controller_causal_feature_bundle.feature_names_csv` method card; source-capability column lists
must not be interpreted as runtime feature positions. Training targets and user groups have shape
`(N,)`, and targets are binary. The controller-provided seed is an unsigned 32-bit integer.
`train_model` returns between 1 and 64 named finite numeric NumPy arrays, allowing each model family
to own its state without changing the wrapper. The fixed `checkpoint/model.txt` path contains that
safe NumPy archive despite its protocol-owned filename. `predict_scores` returns one finite score
per row, and the wrapper writes fixed little-endian float64 `scores.npy` bytes.

The candidate declarations are not authority. The parent process must call
`validate_train_outputs` or `validate_prediction_outputs` from
`kuairand_agent.candidate_api.protocol` before retaining artifacts or sending scores to the
protected evaluator.
