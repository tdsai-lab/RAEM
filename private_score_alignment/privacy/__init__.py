"""Privacy primitives required by the RAEM+KTO main-table experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dp_mechanisms.linear_topb_em import (
    LINEAR_TOPB_EM_NOISE_MODES,
    linear_topb_exponential_mechanism,
)


def clean_ranking_from_scores(scores: Sequence[float] | torch.Tensor) -> list[int]:
    """Return item ids sorted by descending score with stable tie-breaking."""

    if isinstance(scores, torch.Tensor):
        values = [float(value) for value in scores.detach().cpu().reshape(-1).tolist()]
    else:
        values = [float(value) for value in scores]
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def ranking_to_rank_positions(ranking: Sequence[int] | torch.Tensor) -> list[int]:
    """Convert a best-to-worst permutation to zero-based item ranks."""

    if isinstance(ranking, torch.Tensor):
        order = [int(value) for value in ranking.detach().cpu().reshape(-1).tolist()]
    else:
        order = [int(value) for value in ranking]
    if sorted(order) != list(range(len(order))):
        raise ValueError("ranking must be a permutation of 0..m-1.")
    ranks = [0] * len(order)
    for rank, item_index in enumerate(order):
        ranks[item_index] = rank
    return ranks


def resolve_topb_boundary(m: int, boundary_rank: int | None = None) -> int:
    """Resolve the Top-B boundary."""

    m_value = int(m)
    if m_value <= 0:
        raise ValueError("m must be positive.")
    return m_value // 2 if boundary_rank is None else min(max(int(boundary_rank), 0), m_value)


__all__ = [
    "LINEAR_TOPB_EM_NOISE_MODES",
    "clean_ranking_from_scores",
    "linear_topb_exponential_mechanism",
    "ranking_to_rank_positions",
    "resolve_topb_boundary",
]
