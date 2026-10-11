"""Reference-centered KTO loss for RAEM's released Top-B labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RpwKtoLossComponents:
    loss: torch.Tensor
    rewards: torch.Tensor
    reference_point: torch.Tensor
    centered_rewards: torch.Tensor
    logits: torch.Tensor
    pred_utilities: torch.Tensor
    targets: torch.Tensor
    positive_weights: torch.Tensor
    negative_weights: torch.Tensor
    item_losses: torch.Tensor
    per_prompt_loss: torch.Tensor
    reward_mean: torch.Tensor
    reward_std: torch.Tensor
    raw_reward_mean: torch.Tensor
    raw_reward_std: torch.Tensor
    centered_reward_mean: torch.Tensor
    centered_reward_std: torch.Tensor
    positive_branch_mean: torch.Tensor
    negative_branch_mean: torch.Tensor
    positive_loss_mean: torch.Tensor
    negative_loss_mean: torch.Tensor
    target_mean: torch.Tensor
    pred_utility_mean: torch.Tensor


def masked_forward_kl_reference_point(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch KTO reference point: mean token-level KL(pi || pi_ref)."""

    if policy_logits.ndim != 3 or policy_logits.shape != reference_logits.shape:
        raise ValueError("policy_logits and reference_logits must share shape [N, L, V].")
    if response_mask.shape != policy_logits.shape[:2]:
        raise ValueError("response_mask must share the first two logits dimensions.")
    shifted_mask = response_mask[:, 1:].bool()
    if attention_mask is not None:
        shifted_mask = shifted_mask & attention_mask[:, 1:].bool()
    if not torch.any(shifted_mask):
        return policy_logits.new_zeros((), dtype=torch.float32).detach()

    policy = policy_logits[:, :-1, :]
    reference = reference_logits[:, :-1, :]
    positions = shifted_mask.nonzero(as_tuple=False)
    chunk_size = max(1, 4_000_000 // max(int(policy_logits.shape[-1]), 1))
    total_kl = policy_logits.new_zeros((), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, positions.shape[0], chunk_size):
            chunk = positions[start : start + chunk_size]
            batch_indices, token_indices = chunk[:, 0], chunk[:, 1]
            policy_chunk = policy[batch_indices, token_indices, :].float()
            reference_chunk = reference[batch_indices, token_indices, :].float()
            logp = F.log_softmax(policy_chunk, dim=-1)
            logq = F.log_softmax(reference_chunk, dim=-1)
            total_kl = total_kl + (logp.exp() * (logp - logq)).sum(dim=-1).sum()
    return (total_kl / float(positions.shape[0])).detach()


def prepare_private_score_targets(
    private_scores: torch.Tensor,
    item_mask: torch.Tensor | None = None,
    **_unused: object,
) -> torch.Tensor:
    """Convert released RAEM labels to float targets and clear padding."""

    targets = private_scores.to(dtype=torch.float32)
    if item_mask is None:
        return targets
    if item_mask.shape != targets.shape:
        raise ValueError("item_mask must share shape [B, m] with private_scores.")
    return torch.where(item_mask.bool(), targets, torch.zeros_like(targets))


def rpw_kto_components_from_rewards(
    rewards: torch.Tensor,
    targets: torch.Tensor,
    item_mask: torch.Tensor | None = None,
    *,
    reference_point: torch.Tensor | float,
    beta: float = 0.1,
    lambda_d: float = 1.0,
    lambda_u: float = 1.0,
    positive_branch_weights: torch.Tensor,
    negative_branch_weights: torch.Tensor,
    **_unused: object,
) -> RpwKtoLossComponents:
    """Compute RPW-KTO using precomputed hard branches from the RAEM release."""

    beta_value = float(beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive.")
    if float(lambda_d) < 0.0 or float(lambda_u) < 0.0:
        raise ValueError("lambda_d and lambda_u must be non-negative.")
    if rewards.ndim != 2 or rewards.shape != targets.shape:
        raise ValueError("rewards and targets must share shape [B, m].")
    valid_mask = torch.ones_like(rewards, dtype=torch.bool) if item_mask is None else item_mask.bool()
    if valid_mask.shape != rewards.shape:
        raise ValueError("item_mask must share shape [B, m] with rewards.")
    if positive_branch_weights.shape != rewards.shape or negative_branch_weights.shape != rewards.shape:
        raise ValueError("RAEM branch weights must share shape [B, m] with rewards.")

    rewards_f = torch.where(valid_mask, rewards.float(), torch.zeros_like(rewards, dtype=torch.float32))
    reference_point_f = torch.as_tensor(reference_point, device=rewards.device, dtype=torch.float32).detach()
    if reference_point_f.numel() != 1:
        raise ValueError("reference_point must be scalar.")
    reference_point_f = reference_point_f.reshape(())
    centered = torch.where(valid_mask, rewards_f - reference_point_f, torch.zeros_like(rewards_f))
    logits = beta_value * centered
    positive_branch = 1.0 - torch.sigmoid(logits)
    negative_branch = 1.0 - torch.sigmoid(-logits)
    positive_weights = torch.where(
        valid_mask,
        positive_branch_weights.to(device=rewards.device, dtype=torch.float32),
        torch.zeros_like(rewards_f),
    )
    negative_weights = torch.where(
        valid_mask,
        negative_branch_weights.to(device=rewards.device, dtype=torch.float32),
        torch.zeros_like(rewards_f),
    )
    positive_losses = positive_weights * float(lambda_d) * positive_branch
    negative_losses = negative_weights * float(lambda_u) * negative_branch
    item_losses = torch.where(valid_mask, positive_losses + negative_losses, torch.zeros_like(rewards_f))
    counts = valid_mask.float().sum(dim=-1).clamp_min(1.0)
    per_prompt_loss = item_losses.sum(dim=-1) / counts
    loss = item_losses[valid_mask].mean() if valid_mask.any() else rewards_f.new_zeros(())
    pred_utilities = torch.where(valid_mask, torch.sigmoid(logits), torch.zeros_like(logits))

    if valid_mask.any():
        valid_raw = rewards_f[valid_mask]
        valid_centered = centered[valid_mask]
        valid_logits = logits[valid_mask]
        reward_mean = valid_logits.mean()
        reward_std = valid_logits.std(unbiased=False)
        raw_reward_mean = valid_raw.mean()
        raw_reward_std = valid_raw.std(unbiased=False)
        centered_reward_mean = valid_centered.mean()
        centered_reward_std = valid_centered.std(unbiased=False)
        positive_branch_mean = positive_branch[valid_mask].mean()
        negative_branch_mean = negative_branch[valid_mask].mean()
        positive_loss_mean = positive_losses[valid_mask].mean()
        negative_loss_mean = negative_losses[valid_mask].mean()
        target_mean = (positive_weights - negative_weights)[valid_mask].mean()
        pred_utility_mean = pred_utilities[valid_mask].mean()
    else:
        zero = rewards_f.new_zeros(())
        reward_mean = reward_std = raw_reward_mean = raw_reward_std = zero
        centered_reward_mean = centered_reward_std = zero
        positive_branch_mean = negative_branch_mean = zero
        positive_loss_mean = negative_loss_mean = target_mean = pred_utility_mean = zero

    return RpwKtoLossComponents(
        loss=loss,
        rewards=rewards_f,
        reference_point=reference_point_f,
        centered_rewards=centered,
        logits=logits,
        pred_utilities=pred_utilities,
        targets=positive_weights - negative_weights,
        positive_weights=positive_weights,
        negative_weights=negative_weights,
        item_losses=item_losses,
        per_prompt_loss=per_prompt_loss,
        reward_mean=reward_mean,
        reward_std=reward_std,
        raw_reward_mean=raw_reward_mean,
        raw_reward_std=raw_reward_std,
        centered_reward_mean=centered_reward_mean,
        centered_reward_std=centered_reward_std,
        positive_branch_mean=positive_branch_mean,
        negative_branch_mean=negative_branch_mean,
        positive_loss_mean=positive_loss_mean,
        negative_loss_mean=negative_loss_mean,
        target_mean=target_mean,
        pred_utility_mean=pred_utility_mean,
    )


def rpw_kto_components(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    targets: torch.Tensor,
    item_mask: torch.Tensor | None = None,
    **kwargs,
) -> RpwKtoLossComponents:
    if policy_logps.shape != ref_logps.shape or policy_logps.shape != targets.shape:
        raise ValueError("policy_logps, ref_logps, and targets must share shape [B, m].")
    rewards = policy_logps.float() - ref_logps.to(device=policy_logps.device, dtype=torch.float32)
    return rpw_kto_components_from_rewards(
        rewards=rewards,
        targets=targets,
        item_mask=item_mask,
        **kwargs,
    )


def rpw_kto_loss(*args, **kwargs) -> torch.Tensor:
    return rpw_kto_components(*args, **kwargs).loss


class RpwKtoLoss(torch.nn.Module):
    def __init__(self, beta: float = 0.1, lambda_d: float = 1.0, lambda_u: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.lambda_d = float(lambda_d)
        self.lambda_u = float(lambda_u)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return rpw_kto_loss(
            *args,
            beta=self.beta,
            lambda_d=self.lambda_d,
            lambda_u=self.lambda_u,
            **kwargs,
        )
