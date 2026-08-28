# Generated deterministic LambdaRank candidate

This directory is a self-contained generated-plane candidate template. It uses the optional,
locked `lightgbm==4.7.0` dependency from the `research-tree` group and always runs deterministic
CPU training. It imports no trusted controller, evaluator, organizer starter file, provider, or
raw-data loader.

Run it only in a fresh candidate subprocess so a neural worker's OpenMP runtime cannot share the
same interpreter:

```text
python candidate.py train --request request.json --output train-output/
python candidate.py predict --request request.json \
  --checkpoint train-output/checkpoint/model.txt --output prediction-output/
```

The nested training `request` object has these exact keys:

```json
{
  "protocol_schema_version": 1,
  "source_digest": "<trusted source snapshot digest>",
  "config_digest": "<sha256 of config.json bytes>",
  "data_digest": "<trusted training capability identity>",
  "split_token": "<opaque train or inner-train split identity>",
  "seed": 0,
  "features_handle": "features",
  "targets_handle": "targets",
  "user_groups_handle": "user_groups"
}
```

The referenced approved capabilities must be exactly one finite numeric `(N, D)` feature array,
one binary `(N,)` target array, and one finite numeric `(N,)` user-group array. The feature and
group capabilities use role `train_inputs`; the target capability uses `train_targets`. Training
is rejected for validation or final workspace roles.

The nested prediction request has these exact keys:

```json
{
  "protocol_schema_version": 1,
  "source_digest": "<trusted source snapshot digest>",
  "config_digest": "<sha256 of config.json bytes>",
  "data_digest": "<trusted prediction capability identity>",
  "split_token": "<opaque prediction split identity>",
  "features_handle": "features",
  "expected_count": 123,
  "checkpoint_digest": "<trusted checkpoint/model.txt digest>"
}
```

Prediction accepts exactly one phase-appropriate numeric feature capability and a checkpoint
whose bytes match `checkpoint_digest`; there is no target or group argument. Training constructs
a private stable view ordered by each user's first appearance while retaining within-user row
order. Prediction scores the canonical feature matrix directly, so no inverse permutation or
alignment column exists in generated code.

`config.json` has an exact bounded schema. Its `num_boost_round` is a frozen tree count selected
only by the controller's train-derived inner-fold policy. This executable performs no early
stopping or tree-count choice on public validation. The objective, CPU device, deterministic
flags, seed fan-out, sampling-off controls, binary gain, truncation level, and L2 value are fixed
in source. The immutable config binds `seed_policy` to `controller_request_uint32`; each training
request supplies the actual validated uint32 seed, which is recorded in diagnostics and the model
checkpoint. Thus Fold B, Fold A, outer confirmation seeds, and final replay retain one unchanged
source/config digest. The remaining exposed controls are the frozen round count, leaves, minimum
leaf rows, learning rate, and fixed thread count.

Training emits only `checkpoint/model.txt` and `candidate_result.json`. Prediction emits only a
little-endian float64 `scores.npy` and `prediction_result.json`. Each result declares exact byte
digests and bounded non-authoritative diagnostics. The parent process must still call
`validate_train_outputs` or `validate_prediction_outputs`; candidate declarations never establish
artifact meaning or benchmark performance.
