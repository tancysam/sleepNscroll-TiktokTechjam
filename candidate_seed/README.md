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
`reference_pairwise_fm.py` is the protected five-field pairwise-FM control. Its training function
is `train_reference_pairwise_fm(features, targets, user_groups, *, seed)`. It uses positive-ticket
GAUC-aligned sampling, the organizer codes at positions 51–55, float32 FM reductions, and the
frozen dense-Adam recipe. New pairwise work must reuse `sample_reference_logged_pairs` instead of
rebuilding group offsets or row-index maps.

`reference_categorical_ranker.py` is the native-categorical LambdaRank specialist. Its training
function is `train_reference_categorical_ranker(features, targets, user_groups, *, seed)`. It
consumes the first 83 columns and treats positions 51–55 and 82 (`video_type_code`) as categorical.
`reference_listnet_ranker.py` and `reference_pointwise_ranker.py` preserve their historical
first-82 correction surfaces; they supply a neutral column 82 only to their nested categorical
backbone. Their training functions also require the controller seed as the keyword-only `seed`
argument. Mutable candidates may compose these specialists, but cannot replace, retune, or omit
their checkpoint arrays.

`reference_observed_pair_objectives.py` and `reference_observed_pair_fm.py` provide a predeclared
equal-budget ablation of pair selection. The control is byte-exact to the uniform reference FM;
the treatment changes exactly half of its logged pairs to same-user, same-duration-bucket
positive/negative comparisons. Use `train_reference_uniform_pairwise_fm(..., *, seed)` and
`train_reference_duration_pairwise_fm(..., *, seed)` as paired arms. Inference remains
label- and group-free through `reference_observed_pair_fm_scores(features, checkpoint)`.
The full-budget three-seed train-fold replicate is recorded in
`docs/research/observed_pair_duration_pilot-20260830.md`: its mean primary delta versus the
uniform-pair control was `+0.0007370909`, but the worst fold/seed cell was `-0.0001828671`.
Accordingly, this remains an experimental specialist and is not evidence for deployment by itself.

The current controller matrix has 95 columns. Positions 0–81 preserve the historical feature
surface, position 82 is `video_type_code`, and positions 83–94 are input-only strict-past exposure,
first-seen, and time-since-last-exposure features for user, video, author, and user-video scopes.
These final 12 columns may advance from earlier query inputs but never accept query outcomes.
Mutable code may use them only in the exact order declared by the current
`controller_causal_feature_bundle.feature_names_csv` method card.
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
