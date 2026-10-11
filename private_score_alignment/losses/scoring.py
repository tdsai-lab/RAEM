"""Scalar candidate scoring helpers used by preference losses."""

from __future__ import annotations

import torch


def compute_relative_policy_scores(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    beta: float,
    response_token_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``s_theta(x, y) = beta * (log pi_theta - log pi_ref)``.

    Uses raw sum-of-log-probs (no length normalization).
    ``response_token_counts`` is accepted for call-site compatibility but
    deliberately ignored — length normalization biases ranks toward
    shorter responses under listwise preference training.
    """

    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy and reference log-prob tensors must share the same shape.")
    del response_token_counts  # intentionally unused
    return beta * (policy_log_probs - reference_log_probs)
