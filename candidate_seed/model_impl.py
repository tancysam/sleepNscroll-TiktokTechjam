"""Small mutable model surface for autonomous candidate generation.

The stable ``candidate.py`` entrypoint owns capabilities, protocol validation, checkpoint I/O,
and result writing. Research implementations should normally replace only this module and
``config.json``.
"""

from __future__ import annotations

from typing import cast

import numpy as np

SCORES_DTYPE = "<f8"
CONFIG_KEYS = {
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "batch_size",
    "candidate_family",
    "category_count",
    "dense_l2",
    "embedding_init",
    "embedding_l2",
    "epochs",
    "learning_rate",
    "logit_clip",
    "max_members",
    "rank",
    "schema_version",
}


class CandidateModelError(ValueError):
    """The candidate-owned model configuration or numeric state is invalid."""


def _config_epochs(config: dict[str, object]) -> int:
    value = config["epochs"]
    if type(value) is not int or not 1 <= value <= 10_000:
        raise CandidateModelError("epochs is invalid")
    return value


def _config_float(config: dict[str, object], name: str) -> float:
    value = config[name]
    if type(value) not in {int, float}:
        raise CandidateModelError(f"{name} is invalid")
    numeric = float(cast(int | float, value))
    if not 0 < numeric < 1_000:
        raise CandidateModelError(f"{name} is invalid")
    return numeric


def _config_int(config: dict[str, object], name: str, low: int = 1, high: int = 1_000_000) -> int:
    value = config[name]
    if type(value) is not int or not low <= value <= high:
        raise CandidateModelError(f"{name} is invalid")
    return value


def validate_config(config: dict[str, object]) -> None:
    """Validate the model-owned configuration before training or prediction.

    Every knob a campaign might want to tune stays here rather than being hard-coded, so a
    candidate can move the parent's rank, regularization, epochs or ensemble size without
    replacing its structure.
    """

    if set(config) != CONFIG_KEYS:
        raise CandidateModelError("model config keys do not match the implementation")
    if config["schema_version"] != 1:
        raise CandidateModelError("config schema_version must be 1")
    if config["candidate_family"] != "identity_fm_seed_ensemble":
        raise CandidateModelError("candidate_family is invalid")
    _config_epochs(config)
    for name in ("learning_rate", "logit_clip", "dense_l2", "embedding_l2", "embedding_init"):
        _config_float(config, name)
    for name in ("adam_beta1", "adam_beta2", "adam_epsilon"):
        _config_float(config, name)
    _config_int(config, "batch_size", 1024, 1_000_000)
    _config_int(config, "rank", 1, 64)
    _config_int(config, "category_count", 1, 64)
    _config_int(config, "max_members", 1, 16)


def _sigmoid(logits: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(logits, -clip, clip)
    result = np.reciprocal(1.0 + np.exp(-bounded), dtype=np.float64)
    return cast(np.ndarray, result)


# ---------------------------------------------------------------------------------------------
# Provided, verified building blocks.
#
# The controller appends categorical identity codes to the feature matrix, and writing the
# interaction maths and the within-user pair sampler from scratch has been the dominant source of
# candidate failure in this project. Both helpers below are tested; prefer calling them over
# reimplementing them. ``train_model`` here does not use them, so a candidate that stays pointwise
# and linear is unaffected.
# ---------------------------------------------------------------------------------------------


def categorical_codes(features: np.ndarray, category_count: int) -> list[np.ndarray]:
    """Return the trailing ``category_count`` identity code columns as int64 vectors."""

    if features.ndim != 2 or features.shape[1] <= category_count:
        raise CandidateModelError("features must have more columns than categorical codes")
    codes: list[np.ndarray] = []
    for column in range(features.shape[1] - category_count, features.shape[1]):
        raw = features[:, column]
        rounded = np.rint(raw)
        if not bool(np.equal(raw, rounded).all()) or float(rounded.min()) < 0.0:
            raise CandidateModelError("categorical code columns must be nonnegative integers")
        codes.append(np.ascontiguousarray(rounded, dtype=np.int64))
    return codes


def embedding_table_size(code: np.ndarray) -> int:
    """Rows needed for ``code``, with one spare row for identities unseen during training.

    Each fold fits its own vocabulary, so size tables from the training matrix you are handed and
    never from a constant.
    """

    return int(code.max()) + 2


def fm_interaction_scores(embeddings: list[np.ndarray], codes: list[np.ndarray]) -> np.ndarray:
    """Second-order factorization-machine interaction term, shape ``(N,)``.

    The two accumulators deliberately have DIFFERENT shapes: ``pair_sum`` is ``(N, rank)`` and
    ``square_sum`` is ``(N,)``. Mixing them raises a broadcast error, which is the most frequent
    defect in this codebase's history. Codes at or beyond a table's last row are clamped onto the
    spare row, so an identity unseen in training scores rather than raising.
    """

    if not embeddings or len(embeddings) != len(codes):
        raise CandidateModelError("one embedding table is required per code column")
    rows = int(codes[0].shape[0])
    rank = int(embeddings[0].shape[1])
    pair_sum = np.zeros((rows, rank), dtype=np.float64)
    square_sum = np.zeros(rows, dtype=np.float64)
    for table, code in zip(embeddings, codes, strict=True):
        if table.ndim != 2 or table.shape[1] != rank:
            raise CandidateModelError("embedding tables must share one rank")
        if code.shape != (rows,):
            raise CandidateModelError("code columns must align with the feature rows")
        part = table[np.minimum(code, table.shape[0] - 1)]
        pair_sum += part
        square_sum += np.sum(part * part, axis=1)
    return 0.5 * (np.sum(pair_sum * pair_sum, axis=1) - square_sum)


def within_user_pairs(
    targets: np.ndarray,
    user_groups: np.ndarray,
    generator: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample GAUC-matched ``(positive row, negative row)`` pairs from mixed-label users.

    Drawing the positive uniformly from the pooled positives weights each user by its positive
    count, which is what GAUC does; the negative is then uniform inside that same user's rows.

    Every lookup below uses the COMPACT position of a group in the eligible layout, never a raw
    ``user_groups`` value, and every offset is drawn against that group's own negative count.
    Confusing either is what raises IndexError.
    """

    if targets.shape != user_groups.shape or targets.ndim != 1:
        raise CandidateModelError("targets and user_groups must be aligned one-dimensional")
    if type(count) is not int or count <= 0:
        raise CandidateModelError("pair count must be a positive integer")

    order = np.argsort(user_groups, kind="stable")
    sorted_groups = user_groups[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), boundaries.astype(np.int64)))
    ends = np.concatenate((boundaries.astype(np.int64), np.asarray([sorted_groups.size])))

    positive_blocks: list[np.ndarray] = []
    negative_blocks: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        rows = order[start:end]
        labels = targets[rows]
        positives, negatives = rows[labels == 1.0], rows[labels == 0.0]
        if positives.size and negatives.size:
            positive_blocks.append(positives)
            negative_blocks.append(negatives)
    if not positive_blocks:
        raise CandidateModelError("no GAUC eligible user has both a positive and a negative")

    positive_rows = np.concatenate(positive_blocks)
    negative_rows = np.concatenate(negative_blocks)
    positive_counts = np.asarray([block.size for block in positive_blocks], dtype=np.int64)
    negative_counts = np.asarray([block.size for block in negative_blocks], dtype=np.int64)
    group_of_positive = np.repeat(np.arange(positive_counts.size), positive_counts)
    negative_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(negative_counts)[:-1]))

    picked = generator.integers(0, positive_rows.size, size=count)
    groups = group_of_positive[picked]
    offsets = generator.integers(0, negative_counts[groups])
    return positive_rows[picked], negative_rows[negative_starts[groups] + offsets]


def _effective_category_count(features: np.ndarray, config: dict[str, object]) -> int:
    """Configured identity-column count, or zero when the matrix is too narrow to hold one."""

    configured = _config_int(config, "category_count", 1, 64)
    return configured if features.ndim == 2 and features.shape[1] > configured else 0


class _Adam:
    """Per-parameter adaptive steps.

    This is the point of the recorded recipe. A rare identity row appears in ~43 of 1.1M training
    rows, so under a fixed step its gradient is four orders of magnitude smaller than a dense
    weight's and the embedding never leaves its initialization. Adam normalizes by each
    parameter's own second moment, which gives that row a unit-scale step without any hand-tuned
    per-row reweighting.
    """

    def __init__(self, shape: tuple[int, ...], beta1: float, beta2: float, epsilon: float) -> None:
        self.first = np.zeros(shape, dtype=np.float64)
        self.second = np.zeros(shape, dtype=np.float64)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.step = 0

    def update(self, parameter: np.ndarray, gradient: np.ndarray, rate: float) -> np.ndarray:
        self.step += 1
        self.first = self.beta1 * self.first + (1.0 - self.beta1) * gradient
        self.second = self.beta2 * self.second + (1.0 - self.beta2) * gradient * gradient
        corrected_first = self.first / (1.0 - self.beta1**self.step)
        corrected_second = self.second / (1.0 - self.beta2**self.step)
        return parameter - rate * corrected_first / (np.sqrt(corrected_second) + self.epsilon)


def _train_member(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit one identity-code factorization machine with Adam over shuffled minibatches."""

    # A matrix with no room for the identity block still has to train: the protocol's own smoke
    # and contract paths hand this model narrow synthetic matrices, and a parent that raises on
    # them cannot be verified. With no identity columns there is no interaction to learn, so the
    # model degrades to its dense term rather than refusing.
    category_count = _effective_category_count(features, config)
    rank = _config_int(config, "rank", 1, 64)
    batch_size = _config_int(config, "batch_size", 1024, 1_000_000)
    epochs = _config_epochs(config)
    rate = _config_float(config, "learning_rate")
    clip = _config_float(config, "logit_clip")
    dense_l2 = _config_float(config, "dense_l2")
    embedding_l2 = _config_float(config, "embedding_l2")
    beta1 = _config_float(config, "adam_beta1")
    beta2 = _config_float(config, "adam_beta2")
    epsilon = _config_float(config, "adam_epsilon")

    codes = categorical_codes(features, category_count) if category_count else []
    dense_count = features.shape[1] - category_count
    dense = features[:, :dense_count]
    mean = dense.mean(axis=0, dtype=np.float64)
    scale = dense.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)

    generator = np.random.default_rng(seed)
    sizes = [embedding_table_size(code) for code in codes]
    tables = [
        np.ascontiguousarray(
            generator.normal(0.0, _config_float(config, "embedding_init"), (size, rank)),
            dtype=np.float64,
        )
        for size in sizes
    ]
    weights = np.zeros(dense_count, dtype=np.float64)
    bias = np.zeros(1, dtype=np.float64)

    weight_adam = _Adam(weights.shape, beta1, beta2, epsilon)
    bias_adam = _Adam(bias.shape, beta1, beta2, epsilon)
    table_adams = [_Adam(table.shape, beta1, beta2, epsilon) for table in tables]

    order = np.arange(features.shape[0])
    for _ in range(epochs):
        generator.shuffle(order)
        for start in range(0, order.size, batch_size):
            batch = order[start : start + batch_size]
            count = float(batch.size)
            block = np.ascontiguousarray((dense[batch] - mean) / scale, dtype=np.float64)
            batch_codes = [
                np.minimum(code[batch], size - 1) for code, size in zip(codes, sizes, strict=True)
            ]
            interaction = (
                fm_interaction_scores(tables, batch_codes)
                if tables
                else np.zeros(batch.size, dtype=np.float64)
            )
            probabilities = _sigmoid(block @ weights + bias[0] + interaction, clip)
            error = probabilities - targets[batch]

            weights = weight_adam.update(
                weights, (block.T @ error) / count + dense_l2 * weights, rate
            )
            bias = bias_adam.update(bias, np.array([float(np.mean(error))]), rate)

            pair_sum = np.zeros((batch.size, rank), dtype=np.float64)
            for table, code in zip(tables, batch_codes, strict=True):
                pair_sum += table[code]
            for index, (table, code) in enumerate(zip(tables, batch_codes, strict=True)):
                rows = error[:, None] * (pair_sum - table[code])
                gradient = np.zeros_like(table)
                for dimension in range(rank):
                    gradient[:, dimension] = np.bincount(
                        code, weights=rows[:, dimension], minlength=sizes[index]
                    )
                tables[index] = table_adams[index].update(
                    table, gradient / count + embedding_l2 * table, rate
                )

    return {
        "bias": np.asarray(float(bias[0]), dtype=np.float64),
        "feature_mean": np.ascontiguousarray(mean, dtype=np.float64),
        "feature_scale": np.ascontiguousarray(scale, dtype=np.float64),
        "weights": np.ascontiguousarray(weights, dtype=np.float64),
        "category_count": np.asarray(float(category_count), dtype=np.float64),
        "logit_clip": np.asarray(clip, dtype=np.float64),
        **{
            f"embedding_{index}": np.ascontiguousarray(table, dtype=np.float64)
            for index, table in enumerate(tables)
        },
    }


def _predict_member(features: np.ndarray, checkpoint: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the dense term and the identity interaction from a verified checkpoint."""

    category_count = int(checkpoint["category_count"].reshape(()))
    dense_count = features.shape[1] - category_count
    codes = categorical_codes(features, category_count) if category_count else []
    tables = [checkpoint[f"embedding_{index}"] for index in range(category_count)]
    block = (features[:, :dense_count] - checkpoint["feature_mean"]) / checkpoint["feature_scale"]
    interaction = (
        fm_interaction_scores(tables, codes)
        if tables
        else np.zeros(features.shape[0], dtype=np.float64)
    )
    logits = block @ checkpoint["weights"] + float(checkpoint["bias"].reshape(())) + interaction
    return np.ascontiguousarray(
        _sigmoid(logits, float(checkpoint["logit_clip"].reshape(()))), dtype=np.dtype(SCORES_DTYPE)
    )


def training_diagnostics(
    config: dict[str, object], checkpoint: dict[str, np.ndarray]
) -> dict[str, int | float]:
    """Return bounded JSON diagnostics owned by the model implementation.

    No key may contain the token gauc, ndcg or primary, and none may equal auc, metric, metrics
    or a *_score name: the evaluator alone may name a score, and the refusal happens after
    training has already succeeded.
    """

    return {
        "selected_members": int(checkpoint["members"].reshape(())),
        "rank": _config_int(config, "rank", 1, 64),
        "epochs": _config_epochs(config),
    }


def _percentiles_by_user(user_codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Rank ``values`` into ``[0, 1]`` inside each user's own rows, ties sharing an average rank.

    ``user_id_code`` is a column of the prediction matrix, so a candidate can do this even though
    ``user_groups`` is training-only. It matters: averaging five members on raw scores is worth
    +0.0000772 on this benchmark and averaging on within-user percentiles is worth +0.0005664.
    """

    out = np.empty(values.size, dtype=np.float64)
    order = np.argsort(user_codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(user_codes[order])) + 1
    for block in np.split(order, boundaries):
        if block.size == 1:
            out[block[0]] = 0.5
            continue
        inner = np.argsort(values[block], kind="stable")
        ranks = np.empty(block.size, dtype=np.float64)
        ordered = values[block][inner]
        start = 0
        for index in range(1, block.size + 1):
            if index == block.size or ordered[index] != ordered[start]:
                ranks[inner[start:index]] = (start + index + 1) / 2.0
                start = index
        out[block] = (ranks - 1.0) / (block.size - 1.0)
    return out


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    user_groups: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> dict[str, np.ndarray]:
    """Train a deterministic ensemble of independently seeded members.

    The recorded recipe carries ``max_members: 5``; a single member is only its base. Child seeds
    are derived from the supplied seed so replay stays byte exact.
    """

    members = int(cast(int, config.get("max_members", 1)))
    checkpoint: dict[str, np.ndarray] = {"members": np.asarray(float(members), dtype=np.float64)}
    for member in range(members):
        child = _train_member(features, targets, user_groups, config, seed * 1000 + member)
        for name, array in child.items():
            checkpoint[f"m{member}__{name}"] = array
    return checkpoint


def predict_scores(features: np.ndarray, checkpoint: dict[str, np.ndarray]) -> np.ndarray:
    """Average the members' WITHIN-USER percentiles, not their raw scores."""

    members = int(checkpoint["members"].reshape(()))
    category_count = int(checkpoint["m0__category_count"].reshape(()))
    # user_id_code leads the identity block, so it is the first of the trailing code columns.
    # Without an identity block there is no user to group by, and averaging raw scores is the
    # only thing left. That path is worth far less -- +0.0000772 against +0.0005664 for the
    # within-user version on this benchmark -- so it is a fallback, never the intent.
    user_codes = (
        np.rint(features[:, features.shape[1] - category_count]).astype(np.int64)
        if category_count
        else None
    )
    total = np.zeros(features.shape[0], dtype=np.float64)
    for member in range(members):
        single = {
            name[len(f"m{member}__") :]: array
            for name, array in checkpoint.items()
            if name.startswith(f"m{member}__")
        }
        scores = _predict_member(features, single)
        total += scores if user_codes is None else _percentiles_by_user(user_codes, scores)
    return np.ascontiguousarray(total / float(members), dtype=np.dtype(SCORES_DTYPE))
