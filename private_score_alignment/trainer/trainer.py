"""Hugging Face Trainer for the released RAEM+KTO experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Optional

import numpy as np
import torch
from transformers import Trainer

from private_score_alignment.config.schemas import PrivacyConfig
from private_score_alignment.losses.alignment import total_alignment_loss
from private_score_alignment.losses.rpw_kto_loss import prepare_private_score_targets
from private_score_alignment.models.scoring import masked_sequence_log_probs, reshape_candidate_scores
from private_score_alignment.privacy import (
    LINEAR_TOPB_EM_NOISE_MODES,
    clean_ranking_from_scores,
    linear_topb_exponential_mechanism,
    ranking_to_rank_positions,
    resolve_topb_boundary,
)


@dataclass
class _PrivateTargetBatch:
    z: torch.Tensor
    z_tilde: torch.Tensor
    z_hat: torch.Tensor
    U_tilde: torch.Tensor
    alpha_pos: torch.Tensor
    alpha_neg: torch.Tensor
    clean_rankings: torch.Tensor
    epsilon: float
    boundary_rank: int | None
    linear_topb_delta: torch.Tensor
    linear_topb_eta: torch.Tensor


class PrivateScoreAlignmentTrainer(Trainer):
    """Align a policy with a frozen reference using RAEM labels and RPW-KTO."""

    def __init__(
        self,
        *args,
        reference_model: torch.nn.Module,
        privacy_config: PrivacyConfig,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.privacy_config = privacy_config
        self._reference_device: Optional[torch.device] = None
        self._latest_breakdown: dict[str, Any] = {}
        self._breakdown_sums: dict[str, float] = {}
        self._breakdown_count = 0
        self._logged_ldp_config = False

    def _ensure_reference_device(self, device: torch.device) -> None:
        if self._reference_device == device:
            return
        self.reference_model.to(device)
        self.reference_model.eval()
        self._reference_device = device

    def _record_training_breakdown(self, breakdown: dict[str, Any]) -> None:
        self._latest_breakdown = dict(breakdown)
        self._breakdown_count += 1
        for key, value in breakdown.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            self._breakdown_sums[key] = self._breakdown_sums.get(key, 0.0) + numeric_value

    def _consume_logged_breakdown(self) -> dict[str, float]:
        if self._breakdown_count <= 0:
            return dict(self._latest_breakdown)
        averaged = {
            key: value / float(self._breakdown_count) for key, value in self._breakdown_sums.items()
        }
        self._breakdown_sums = {}
        self._breakdown_count = 0
        return averaged

    def _stable_sample_noise_seed(self, cache_key: str) -> int:
        base_seed = 0 if self.privacy_config.noise_seed is None else int(self.privacy_config.noise_seed)
        digest = hashlib.sha256(f"{base_seed}:{cache_key}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)

    @staticmethod
    def _private_target_cache_key(
        sample_id: object,
        dataset_name: object,
        scores: torch.Tensor,
        valid_count: int,
    ) -> str:
        dataset_part = "unknown" if dataset_name is None else str(dataset_name)
        score_part = ",".join(
            f"{float(value):.8g}" for value in scores[:valid_count].detach().cpu().tolist()
        )
        sample_part = f"{sample_id}:scores:{score_part}" if sample_id is not None else f"scores:{score_part}"
        return f"{dataset_part}:{sample_part}:{valid_count}"

    def _linear_topb_em_batch(
        self,
        *,
        raw_scores: torch.Tensor,
        item_mask: torch.Tensor,
        sample_ids: list[object] | None,
        dataset_names: list[object] | None,
    ) -> _PrivateTargetBatch:
        """Release exactly-B positive labels with the Linear Top-B EM."""

        z = torch.zeros_like(raw_scores, dtype=torch.float32)
        z_tilde = torch.zeros_like(raw_scores, dtype=torch.float32)
        z_hat = torch.zeros_like(raw_scores, dtype=torch.float32)
        utility = torch.zeros_like(raw_scores, dtype=torch.float32)
        alpha_pos = torch.zeros_like(raw_scores, dtype=torch.float32)
        alpha_neg = torch.zeros_like(raw_scores, dtype=torch.float32)
        clean_rankings = torch.full_like(raw_scores, -1.0, dtype=torch.float32)
        deltas = torch.zeros(raw_scores.size(0), 1, device=raw_scores.device, dtype=torch.float32)
        etas = torch.zeros(raw_scores.size(0), 1, device=raw_scores.device, dtype=torch.float32)

        noise_mode = str(self.privacy_config.noise_mode).lower().replace("-", "_")
        add_noise = noise_mode != "none"
        if add_noise and noise_mode not in LINEAR_TOPB_EM_NOISE_MODES:
            raise ValueError(f"Unsupported Linear Top-B EM noise_mode: {self.privacy_config.noise_mode}")
        epsilon = float(self.privacy_config.epsilon)
        score_function = str(self.privacy_config.dp_score_function).lower().replace("-", "_")
        if score_function not in {"linear", "sign"}:
            raise ValueError("privacy.dp_score_function must be 'linear' or 'sign'.")

        for row_index in range(raw_scores.size(0)):
            valid_count = int(item_mask[row_index].sum().item())
            if valid_count <= 0:
                continue
            sample_id = sample_ids[row_index] if sample_ids is not None else None
            dataset_name = dataset_names[row_index] if dataset_names is not None else None
            cache_key = self._private_target_cache_key(
                sample_id,
                dataset_name,
                raw_scores[row_index],
                valid_count,
            )
            clean_phi = clean_ranking_from_scores(raw_scores[row_index, :valid_count])
            clean_ranks = torch.tensor(
                ranking_to_rank_positions(clean_phi),
                device=raw_scores.device,
                dtype=torch.long,
            )
            clean_rankings[row_index, :valid_count] = torch.tensor(
                clean_phi,
                device=raw_scores.device,
                dtype=torch.float32,
            )
            boundary = resolve_topb_boundary(valid_count, self.privacy_config.boundary_rank)
            clean_positive = clean_ranks.lt(boundary)
            if add_noise:
                rng = np.random.default_rng(
                    self._stable_sample_noise_seed(f"linear_topb_em:{cache_key}")
                )
                released_np = linear_topb_exponential_mechanism(
                    clean_phi,
                    epsilon=epsilon,
                    B=boundary,
                    score_function=score_function,
                    rng=rng,
                )
                released_positive = torch.as_tensor(
                    released_np.astype(np.bool_),
                    device=raw_scores.device,
                    dtype=torch.bool,
                )
            else:
                released_positive = clean_positive

            clean_labels = clean_positive.to(dtype=torch.float32)
            released_labels = released_positive.to(dtype=torch.float32)
            z[row_index, :valid_count] = clean_labels
            z_tilde[row_index, :valid_count] = released_labels
            z_hat[row_index, :valid_count] = released_labels
            utility[row_index, :valid_count] = released_labels
            alpha_pos[row_index, :valid_count] = released_labels
            alpha_neg[row_index, :valid_count] = (~released_positive).to(dtype=torch.float32)
            delta = 2.0 if score_function == "sign" else float(max(boundary, valid_count - boundary))
            deltas[row_index, 0] = delta
            etas[row_index, 0] = epsilon / delta if delta > 0.0 else 0.0

        return _PrivateTargetBatch(
            z=z,
            z_tilde=z_tilde,
            z_hat=z_hat,
            U_tilde=utility,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            clean_rankings=clean_rankings,
            epsilon=epsilon,
            boundary_rank=self.privacy_config.boundary_rank,
            linear_topb_delta=deltas,
            linear_topb_eta=etas,
        )

    def _maybe_log_ldp_config(self) -> None:
        if self._logged_ldp_config:
            return
        self._logged_ldp_config = True
        try:
            should_log = self.is_world_process_zero()
        except Exception:
            should_log = True
        if should_log:
            print(
                "[ldp] target_source=linear_topb_em_top_b_labels "
                "dp_mechanism=linear_topb_em "
                f"noise_mode={self.privacy_config.noise_mode} "
                f"score_function={self.privacy_config.dp_score_function} "
                f"epsilon={float(self.privacy_config.epsilon):.6g} "
                "sensitivity=max(B,m-B)_or_2 "
                f"boundary_rank={self.privacy_config.boundary_rank} "
                "rank_branch_mode=hard_noisy_rank",
                flush=True,
            )

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """Compute the RAEM+KTO training objective."""

        del num_items_in_batch
        if str(self.privacy_config.loss_type).lower() != "rpw_kto":
            raise ValueError("This release supports only privacy.loss_type='rpw_kto'.")
        mechanism = str(self.privacy_config.dp_mechanism).lower().replace("-", "_")
        if mechanism not in LINEAR_TOPB_EM_NOISE_MODES:
            raise ValueError("This release supports only privacy.dp_mechanism='linear_topb_em'.")

        batch = dict(inputs)
        raw_scores = batch.pop("raw_scores").to(dtype=torch.float32)
        item_mask = batch.pop("item_mask")
        batch.pop("score_mins")
        batch.pop("score_maxs")
        group_sizes = batch.pop("group_sizes")
        response_mask = batch.pop("response_mask")
        dataset_names = batch.pop("dataset_names", None)
        sample_ids = batch.pop("sample_ids", None)
        batch.pop("prompts", None)
        batch.pop("responses", None)

        policy_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            use_cache=False,
        )
        self._ensure_reference_device(policy_outputs.logits.device)
        with torch.inference_mode():
            reference_outputs = self.reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
                use_cache=False,
            )
        policy_log_probs = masked_sequence_log_probs(
            logits=policy_outputs.logits,
            input_ids=batch["input_ids"],
            response_mask=response_mask,
            attention_mask=batch["attention_mask"],
        )
        reference_log_probs = masked_sequence_log_probs(
            logits=reference_outputs.logits,
            input_ids=batch["input_ids"],
            response_mask=response_mask,
            attention_mask=batch["attention_mask"],
        )
        relative_scores = reshape_candidate_scores(
            policy_log_probs.to(dtype=torch.float32) - reference_log_probs.to(dtype=torch.float32),
            group_sizes=group_sizes,
            max_candidates=item_mask.size(1),
        )
        policy_candidate_logps = reshape_candidate_scores(
            policy_log_probs,
            group_sizes=group_sizes,
            max_candidates=item_mask.size(1),
        )
        reference_candidate_logps = reshape_candidate_scores(
            reference_log_probs,
            group_sizes=group_sizes,
            max_candidates=item_mask.size(1),
        )

        private_targets = self._linear_topb_em_batch(
            raw_scores=raw_scores,
            item_mask=item_mask,
            sample_ids=sample_ids,
            dataset_names=dataset_names,
        )
        self._maybe_log_ldp_config()
        targets = prepare_private_score_targets(
            private_targets.z_hat,
            item_mask=item_mask,
            label_smoothing=self.privacy_config.label_smoothing,
            target_kind="raem_labels",
        )
        loss_outputs = total_alignment_loss(
            relative_scores=relative_scores,
            private_targets=targets,
            item_mask=item_mask,
            loss_type="rpw_kto",
            beta=self.privacy_config.beta,
            kto_lambda_d=self.privacy_config.kto_lambda_d,
            kto_lambda_u=self.privacy_config.kto_lambda_u,
            policy_logps=policy_candidate_logps,
            reference_logps=reference_candidate_logps,
            policy_logits=policy_outputs.logits,
            reference_logits=reference_outputs.logits,
            response_mask=response_mask,
            attention_mask=batch["attention_mask"],
            ldp_target_mode="linear_topb_em_hard_labels",
            posterior_boundary_rank=private_targets.boundary_rank,
            rank_branch_mode="hard_noisy_rank",
            posterior_positive_weights=private_targets.alpha_pos,
            posterior_negative_weights=private_targets.alpha_neg,
            power_p=self.privacy_config.power_p,
        )

        valid_counts = item_mask.to(dtype=torch.long).sum(dim=-1)
        valid_counts = valid_counts[valid_counts.gt(0)]
        boundary = (
            float((valid_counts // 2).to(dtype=torch.float32).mean().item())
            if self.privacy_config.boundary_rank is None and valid_counts.numel()
            else float(self.privacy_config.boundary_rank or 0)
        )
        valid_deltas = private_targets.linear_topb_delta.reshape(-1)
        valid_deltas = valid_deltas[valid_deltas.gt(0.0)]
        valid_etas = private_targets.linear_topb_eta.reshape(-1)
        valid_etas = valid_etas[valid_etas.gt(0.0)]
        breakdown: dict[str, Any] = {
            "loss_main": float(loss_outputs.loss_main.detach().item()),
            "dp_mechanism": "linear_topb_em",
            "noise_mode": self.privacy_config.noise_mode,
            "epsilon": private_targets.epsilon,
            "boundary_rank": boundary,
            "rank_branch_mode": "hard_noisy_rank",
            "dp_score_function": self.privacy_config.dp_score_function,
            "dp_sensitivity": float(valid_deltas.mean().item()) if valid_deltas.numel() else 0.0,
            "linear_topb_eta": float(valid_etas.mean().item()) if valid_etas.numel() else 0.0,
        }
        for metric_name in (
            "raw_reward_mean",
            "raw_reward_std",
            "kto_reference_point",
            "pred_utility_mean",
            "kto_positive_loss_mean",
            "kto_negative_loss_mean",
            "avg_alpha_pos",
            "avg_alpha_neg",
        ):
            value = getattr(loss_outputs, metric_name, None)
            if value is not None:
                breakdown[metric_name] = float(value.detach().to(dtype=torch.float32).mean().item())
        if model.training:
            self._record_training_breakdown(breakdown)
        else:
            self._latest_breakdown = breakdown

        if return_outputs:
            return loss_outputs.loss_total, {
                "relative_scores": relative_scores.detach(),
                "z": private_targets.z.detach(),
                "z_tilde": private_targets.z_tilde.detach(),
                "z_hat": private_targets.z_hat.detach(),
                "targets": targets.detach(),
                "alpha_pos": private_targets.alpha_pos.detach(),
                "alpha_neg": private_targets.alpha_neg.detach(),
                "linear_topb_delta": private_targets.linear_topb_delta.detach(),
                "linear_topb_eta": private_targets.linear_topb_eta.detach(),
            }
        return loss_outputs.loss_total

    def prediction_step(
        self,
        model,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.inference_mode():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        return loss.detach(), outputs["relative_scores"], None

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        merged = dict(logs)
        if "loss" in merged and "eval_loss" not in merged:
            for key, value in self._consume_logged_breakdown().items():
                merged.setdefault(f"train_{key}", value)
        super().log(merged, *args, **kwargs)
