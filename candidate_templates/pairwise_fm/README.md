# Deterministic generated pairwise FM

This standalone generated candidate trains only on controller-approved logged impressions. For
each sampled pair it first selects an eligible user in proportion to that user's positive count,
then selects one logged positive and one logged negative uniformly within the same user. It never
constructs or samples negatives from a full item catalog.

The candidate owns its NumPy factorization-machine objective, optimizer, checkpoint, and replay
logic. Training is accepted only for `train` and `inner_train` workspaces. Prediction accepts one
feature capability in `inner_valid`, `outer_valid`, or `final`; targets and user groups are not
prediction capabilities. The controller remains solely responsible for protected metrics.

Commands:

```text
python candidate.py train --request request.json --output output
python candidate.py predict --request request.json --checkpoint checkpoint/model.txt --output output
```
