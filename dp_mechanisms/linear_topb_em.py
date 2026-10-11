"""Linear Top-B exponential mechanism for binary label release."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


LINEAR_TOPB_EM_NOISE_MODES = {
    "linear_topb_em",
    "linear-topb-em",
    "linear_top_b_em",
    "linear-top-b-em",
}


def _validate_ranking(ranking: Sequence[int] | np.ndarray) -> np.ndarray:
    order = np.asarray(ranking, dtype=np.int64)
    if order.ndim != 1:
        raise ValueError("ranking must be a one-dimensional permutation.")
    m = int(order.shape[0])
    if m <= 0:
        raise ValueError("ranking must contain at least one item.")
    if np.array_equal(np.sort(order), np.arange(m, dtype=np.int64)):
        return order
    raise ValueError("ranking must be a permutation of 0..m-1.")


def _sample_index(probabilities: np.ndarray, rng=None) -> int:
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    if hasattr(rng, "choice"):
        return int(rng.choice(probabilities.shape[0], p=probabilities))
    if hasattr(rng, "random"):
        draw = float(rng.random())
        return min(
            int(np.searchsorted(np.cumsum(probabilities), draw, side="right")),
            int(probabilities.shape[0] - 1),
        )
    raise TypeError("rng must be None, an integer seed, or a random generator with choice/random.")


def linear_topb_exponential_mechanism(
    ranking: Sequence[int] | np.ndarray,
    epsilon: float,
    B: int,
    score_function: str = "linear",
    rng=None,
) -> np.ndarray:
    """Sample privatized size-B positive labels from a clean ranking.

    ``ranking`` is a permutation from best to worst, so ``ranking[0]`` has
    rank 1. Each size-B binary label vector ``z`` receives utility
    ``U(z | ranking) = sum_i z_i * w_i``.

    Supported score functions and item-level DP calibration:

    * ``score_function='linear'``:
      ``w_i = B + 0.5 - rank_i``, ``Delta = max(B, m - B)``.
    * ``score_function='sign'``:
      ``w_i = +1`` for ``rank_i <= B`` and ``-1`` otherwise, ``Delta = 2``.

    In both cases ``eta = epsilon / Delta`` and probabilities are proportional
    to ``exp(eta * U(z | ranking))`` over all size-B subsets.

    This implementation enumerates all ``itertools.combinations(range(m), B)``.
    That is simple and exact for the current small-m training setting
    (for example, m=8 and B=4 has only 70 subsets). For much larger m, this
    should be replaced by a sampler that avoids enumerating all subsets.
    """

    order = _validate_ranking(ranking)
    m = int(order.shape[0])
    b_value = int(B)
    if b_value < 0 or b_value > m:
        raise ValueError("B must satisfy 0 <= B <= len(ranking).")

    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value < 0.0:
        raise ValueError("epsilon must be a finite non-negative value.")
    resolved_score_function = str(score_function).lower().replace("-", "_")
    if resolved_score_function not in {"linear", "sign"}:
        raise ValueError("score_function must be 'linear' or 'sign'.")

    labels = np.zeros(m, dtype=np.int64)
    if b_value == 0:
        return labels
    if b_value == m:
        labels.fill(1)
        return labels

    rank_positions = np.empty(m, dtype=np.float64)
    rank_positions[order] = np.arange(1, m + 1, dtype=np.float64)
    if resolved_score_function == "linear":
        weights = float(b_value) + 0.5 - rank_positions
        delta = float(max(b_value, m - b_value))
    elif resolved_score_function == "sign":
        weights = np.where(rank_positions <= float(b_value), 1.0, -1.0)
        delta = 2.0
    eta = epsilon_value / delta
    subsets = tuple(combinations(range(m), b_value))
    utilities = np.fromiter(
        (weights[np.asarray(subset, dtype=np.int64)].sum() for subset in subsets),
        dtype=np.float64,
        count=len(subsets),
    )
    logits = eta * utilities
    shifted_logits = logits - np.max(logits)
    unnormalized = np.exp(shifted_logits)
    probabilities = unnormalized / unnormalized.sum()

    selected = subsets[_sample_index(probabilities, rng=rng)]
    labels[np.asarray(selected, dtype=np.int64)] = 1
    return labels
