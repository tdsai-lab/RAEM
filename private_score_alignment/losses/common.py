"""Shared dataclasses and tensor helpers for alignment losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AlignmentLossOutputs:
    """Loss bundle returned by the trainer."""

    loss_total: torch.Tensor
    loss_main: torch.Tensor
    loss_dist: torch.Tensor
    loss_vmean: torch.Tensor
    cl_ipo_loss: torch.Tensor
    clla_align_loss: torch.Tensor
    r_lipo_loss: torch.Tensor
    v_centered_norm: torch.Tensor
    v_mean: torch.Tensor
    v_mean_abs: torch.Tensor
    z_hat_entropy: torch.Tensor
    bar_z_entropy: torch.Tensor
    num_active_pairs: torch.Tensor
    sum_pair_weight: torch.Tensor
    mean_pair_margin: torch.Tensor
    mean_pair_gap_bar_z: torch.Tensor
    mean_pair_weight: torch.Tensor
    pair_agreement_rate: torch.Tensor
    power_p: torch.Tensor
    reward_mean: torch.Tensor | None = None
    reward_std: torch.Tensor | None = None
    raw_reward_mean: torch.Tensor | None = None
    raw_reward_std: torch.Tensor | None = None
    kto_reference_point: torch.Tensor | None = None
    kto_centered_reward_mean: torch.Tensor | None = None
    kto_centered_reward_std: torch.Tensor | None = None
    kto_positive_branch_mean: torch.Tensor | None = None
    kto_negative_branch_mean: torch.Tensor | None = None
    kto_positive_loss_mean: torch.Tensor | None = None
    kto_negative_loss_mean: torch.Tensor | None = None
    target_mean: torch.Tensor | None = None
    pred_utility_mean: torch.Tensor | None = None
    avg_alpha_pos: torch.Tensor | None = None
    avg_alpha_neg: torch.Tensor | None = None
    dpo_chosen_reward_mean: torch.Tensor | None = None
    dpo_rejected_reward_mean: torch.Tensor | None = None
    dpo_reward_margin_mean: torch.Tensor | None = None
    dpo_pair_accuracy: torch.Tensor | None = None
    dpo_predicted_preference_mean: torch.Tensor | None = None


def _masked_row_mean(values: torch.Tensor, item_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked per-row means and valid counts."""

    mask = item_mask.to(dtype=values.dtype)
    counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=-1, keepdim=True) / counts
    return mean, counts


def _masked_simplex_normalize(
    values: torch.Tensor,
    item_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize non-negative masked values into a simplex on valid positions."""

    valid_mask = item_mask.to(dtype=torch.bool)
    safe_values = torch.where(valid_mask, values.clamp_min(0.0), torch.zeros_like(values))
    normalizer = safe_values.sum(dim=-1, keepdim=True)
    return torch.where(
        normalizer > 0.0,
        safe_values / normalizer.clamp_min(eps),
        torch.zeros_like(safe_values),
    )


def _smooth_private_targets(
    z_hat: torch.Tensor,
    item_mask: torch.Tensor,
    temperature: float,
    delta: float,
    eps: float,
) -> torch.Tensor:
    """Apply target-side simplex smoothing to privatized targets."""

    if temperature <= 0.0:
        raise ValueError("target smoothing temperature must be positive.")
    if delta <= 0.0:
        raise ValueError("simplex_delta must be positive.")
    valid_mask = item_mask.to(dtype=torch.bool)
    values = z_hat.to(dtype=torch.float32).clamp_min(0.0)
    stabilized = torch.where(valid_mask, values + float(delta), torch.zeros_like(values))
    smoothed = stabilized.pow(1.0 / float(temperature))
    return _masked_simplex_normalize(smoothed, valid_mask, eps=eps)


def _distribution_entropy(
    distribution: torch.Tensor,
    item_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the mean entropy of masked categorical distributions."""

    if distribution.shape != item_mask.shape:
        raise ValueError("distribution and item_mask must share shape [B, m].")
    valid_mask = item_mask.to(dtype=torch.bool)
    safe_distribution = torch.where(
        valid_mask,
        distribution.clamp_min(eps),
        torch.ones_like(distribution),
    )
    entropy = -(
        torch.where(valid_mask, distribution, torch.zeros_like(distribution))
        * safe_distribution.log()
    ).sum(dim=-1)
    valid_rows = valid_mask.any(dim=-1)
    return entropy[valid_rows].mean() if valid_rows.any() else entropy.new_tensor(0.0)


def _masked_softmax_distribution(
    values: torch.Tensor,
    item_mask: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    """Return a masked softmax distribution over valid candidates."""

    if temperature <= 0.0:
        raise ValueError("softmax temperature must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if values.shape != item_mask.shape:
        raise ValueError("values and item_mask must share shape [B, m].")

    valid_mask = item_mask.to(dtype=torch.bool)
    logits = values.to(dtype=torch.float32) / float(temperature)
    logits = torch.where(valid_mask, logits, torch.zeros_like(logits))
    row_max = torch.where(valid_mask, logits, torch.zeros_like(logits)).max(dim=-1, keepdim=True).values
    exp_logits = torch.where(valid_mask, (logits - row_max).exp(), torch.zeros_like(logits))
    normalizer = exp_logits.sum(dim=-1, keepdim=True)
    return torch.where(
        normalizer.gt(0.0),
        exp_logits / normalizer.clamp_min(eps),
        torch.zeros_like(exp_logits),
    )
