"""RAEM+KTO alignment loss used by the released main-table pipeline."""

from __future__ import annotations

import torch

from private_score_alignment.losses.common import (
    AlignmentLossOutputs,
    _distribution_entropy,
    _masked_simplex_normalize,
)
from private_score_alignment.losses.rpw_kto_loss import (
    masked_forward_kl_reference_point,
    rpw_kto_components,
)


def total_alignment_loss(
    relative_scores: torch.Tensor,
    private_targets: torch.Tensor,
    item_mask: torch.Tensor,
    loss_type: str = "rpw_kto",
    beta: float = 0.1,
    gamma: float | None = None,
    kto_lambda_d: float = 1.0,
    kto_lambda_u: float = 1.0,
    policy_logps: torch.Tensor | None = None,
    reference_logps: torch.Tensor | None = None,
    policy_logits: torch.Tensor | None = None,
    reference_logits: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    kto_reference_point: torch.Tensor | None = None,
    ldp_target_mode: str = "linear_topb_em_hard_labels",
    posterior_boundary_rank: int | None = None,
    rank_branch_mode: str = "hard_noisy_rank",
    posterior_positive_weights: torch.Tensor | None = None,
    posterior_negative_weights: torch.Tensor | None = None,
    power_p: float = 2.0,
    eps: float = 1e-8,
    **_unused: object,
) -> AlignmentLossOutputs:
    """Compute RPW-KTO from RAEM's released Top-B labels."""

    if str(loss_type).lower() != "rpw_kto":
        raise ValueError("This release supports only privacy.loss_type='rpw_kto'.")
    if str(ldp_target_mode).lower() != "linear_topb_em_hard_labels":
        raise ValueError("This release supports only Linear Top-B EM hard-label targets.")
    if relative_scores.ndim != 2:
        raise ValueError("relative_scores must have shape [B, m].")
    if relative_scores.shape != private_targets.shape or relative_scores.shape != item_mask.shape:
        raise ValueError("relative_scores, private_targets, and item_mask must share shape [B, m].")
    if policy_logps is None or reference_logps is None:
        raise ValueError("policy_logps and reference_logps are required for RPW-KTO.")
    if posterior_positive_weights is None or posterior_negative_weights is None:
        raise ValueError("RAEM+KTO requires released positive and negative branch weights.")

    valid_mask = item_mask.to(dtype=torch.bool)
    targets = torch.where(
        valid_mask,
        private_targets.detach().to(dtype=torch.float32),
        torch.zeros_like(private_targets, dtype=torch.float32),
    )
    if kto_reference_point is None:
        if policy_logits is None or reference_logits is None or response_mask is None:
            raise ValueError("policy/reference logits and response_mask are required to compute KTO z0.")
        kto_reference_point = masked_forward_kl_reference_point(
            policy_logits=policy_logits,
            reference_logits=reference_logits,
            response_mask=response_mask,
            attention_mask=attention_mask,
        )

    components = rpw_kto_components(
        policy_logps=policy_logps,
        ref_logps=reference_logps,
        targets=targets,
        item_mask=valid_mask,
        reference_point=kto_reference_point,
        beta=float(beta if gamma is None else gamma),
        lambda_d=float(kto_lambda_d),
        lambda_u=float(kto_lambda_u),
        posterior_epsilon=None,
        posterior_boundary_rank=posterior_boundary_rank,
        branch_weight_mode=rank_branch_mode,
        positive_branch_weights=posterior_positive_weights,
        negative_branch_weights=posterior_negative_weights,
    )
    target_distribution = _masked_simplex_normalize(targets.abs(), valid_mask, eps=eps)
    target_entropy = _distribution_entropy(target_distribution, valid_mask)
    zero = components.loss.new_zeros(())
    avg_alpha_pos = components.positive_weights[valid_mask].mean() if valid_mask.any() else zero
    avg_alpha_neg = components.negative_weights[valid_mask].mean() if valid_mask.any() else zero
    agreement = (
        (components.pred_utilities[valid_mask].gt(0.5) == components.targets[valid_mask].gt(0.0))
        .to(dtype=torch.float32)
        .mean()
        if valid_mask.any()
        else zero
    )
    return AlignmentLossOutputs(
        loss_total=components.loss,
        loss_main=components.loss,
        loss_dist=components.loss,
        loss_vmean=zero,
        cl_ipo_loss=components.loss,
        clla_align_loss=components.loss,
        r_lipo_loss=components.loss,
        v_centered_norm=components.centered_rewards.norm(dim=-1).mean(),
        v_mean=components.reward_mean,
        v_mean_abs=components.reward_mean.abs(),
        z_hat_entropy=target_entropy,
        bar_z_entropy=target_entropy,
        num_active_pairs=valid_mask.to(dtype=torch.float32).sum(),
        sum_pair_weight=zero,
        mean_pair_margin=zero,
        mean_pair_gap_bar_z=components.target_mean,
        mean_pair_weight=zero,
        pair_agreement_rate=agreement,
        power_p=components.loss.new_tensor(float(power_p)),
        reward_mean=components.reward_mean,
        reward_std=components.reward_std,
        raw_reward_mean=components.raw_reward_mean,
        raw_reward_std=components.raw_reward_std,
        kto_reference_point=components.reference_point,
        kto_centered_reward_mean=components.centered_reward_mean,
        kto_centered_reward_std=components.centered_reward_std,
        kto_positive_branch_mean=components.positive_branch_mean,
        kto_negative_branch_mean=components.negative_branch_mean,
        kto_positive_loss_mean=components.positive_loss_mean,
        kto_negative_loss_mean=components.negative_loss_mean,
        target_mean=components.target_mean,
        pred_utility_mean=components.pred_utility_mean,
        avg_alpha_pos=avg_alpha_pos,
        avg_alpha_neg=avg_alpha_neg,
    )
