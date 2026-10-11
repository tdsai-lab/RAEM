"""Utilities for scoring candidate responses with causal LMs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def masked_sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence log-probabilities over response tokens only.

    Args:
        logits: Shape ``[N, L, V]`` from a causal LM.
        input_ids: Shape ``[N, L]``.
        response_mask: Shape ``[N, L]`` with ones on response tokens.
        attention_mask: Shape ``[N, L]``.

    Returns:
        Tensor of shape ``[N]`` with log ``pi(y|x)``.
    """

    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    shifted_response_mask = response_mask[:, 1:].bool()
    shifted_attention_mask = attention_mask[:, 1:].bool()
    valid_mask = shifted_response_mask & shifted_attention_mask
    if not torch.any(valid_mask):
        return logits.new_zeros(logits.size(0), dtype=torch.float32)

    # Score only the response-token positions. This avoids materializing
    # vocabulary-sized tensors for prompt tokens that do not contribute to the
    # objective, which is especially important during preference-stage eval.
    selected_logits = shifted_logits[valid_mask]
    selected_labels = shifted_labels[valid_mask]
    token_negative_log_probs = F.cross_entropy(
        selected_logits,
        selected_labels,
        reduction="none",
    )
    sequence_indices = valid_mask.nonzero(as_tuple=False)[:, 0]
    sequence_log_probs = token_negative_log_probs.new_zeros(logits.size(0))
    sequence_log_probs.index_add_(0, sequence_indices, -token_negative_log_probs)
    return sequence_log_probs


def masked_response_token_counts(
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Count valid response tokens after the causal shift."""

    shifted_response_mask = response_mask[:, 1:]
    shifted_attention_mask = attention_mask[:, 1:].to(dtype=shifted_response_mask.dtype)
    return (shifted_response_mask * shifted_attention_mask).sum(dim=-1)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/DeepSpeed containers without depending on either package."""

    unwrapped = module
    seen: set[int] = set()
    while hasattr(unwrapped, "module") and id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        candidate = getattr(unwrapped, "module")
        if not isinstance(candidate, torch.nn.Module):
            break
        unwrapped = candidate
    return unwrapped


def qwen2_sparse_masked_sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    lm_head_chunk_size: int = 4096,
    checkpoint_lm_head: bool = True,
) -> torch.Tensor:
    """Score Qwen2 responses without materializing prompt/padding logits.

    Qwen2-0.5B has a large vocabulary, so full ``[N, L, V]`` logits dominate
    memory. This path runs the transformer once, selects only hidden states that
    predict response tokens, and applies the LM head in token chunks. During
    training, checkpointing each LM-head chunk avoids retaining vocabulary-sized
    logits until backward.
    """

    if lm_head_chunk_size <= 0:
        raise ValueError("lm_head_chunk_size must be positive.")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [N, L].")
    if response_mask.shape != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("response_mask and attention_mask must share input_ids shape.")

    causal_lm = _unwrap_module(model)
    model_type = str(getattr(getattr(causal_lm, "config", None), "model_type", "")).lower()
    if model_type != "qwen2":
        raise ValueError(
            "Sparse private DPO scoring is Qwen2-specific; "
            f"received model_type={model_type or 'unknown'!r}."
        )
    backbone = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if not isinstance(backbone, torch.nn.Module) or not isinstance(lm_head, torch.nn.Module):
        raise TypeError("Expected a Qwen2ForCausalLM-style model with .model and .lm_head modules.")

    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = outputs.last_hidden_state
    shifted_valid_mask = response_mask[:, 1:].bool() & attention_mask[:, 1:].bool()
    if not torch.any(shifted_valid_mask):
        return hidden_states.sum(dim=(-2, -1)) * 0.0

    selected_hidden_states = hidden_states[:, :-1, :][shifted_valid_mask]
    selected_labels = input_ids[:, 1:][shifted_valid_mask]
    sequence_indices = shifted_valid_mask.nonzero(as_tuple=False)[:, 0]
    sequence_log_probs = hidden_states.new_zeros(input_ids.size(0), dtype=torch.float32)

    def token_negative_log_probs(
        hidden_chunk: torch.Tensor,
        label_chunk: torch.Tensor,
    ) -> torch.Tensor:
        logits = lm_head(hidden_chunk)
        return F.cross_entropy(logits, label_chunk, reduction="none").to(dtype=torch.float32)

    for start in range(0, selected_hidden_states.size(0), int(lm_head_chunk_size)):
        end = start + int(lm_head_chunk_size)
        hidden_chunk = selected_hidden_states[start:end]
        label_chunk = selected_labels[start:end]
        should_checkpoint = (
            bool(checkpoint_lm_head)
            and torch.is_grad_enabled()
            and hidden_chunk.requires_grad
        )
        if should_checkpoint:
            token_nll = checkpoint(
                token_negative_log_probs,
                hidden_chunk,
                label_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            token_nll = token_negative_log_probs(hidden_chunk, label_chunk)
        sequence_log_probs = sequence_log_probs.index_add(
            0,
            sequence_indices[start:end],
            -token_nll,
        )

    return sequence_log_probs


def reshape_candidate_scores(
    flat_scores: torch.Tensor,
    group_sizes: torch.Tensor,
    max_candidates: int,
) -> torch.Tensor:
    """Reshape flattened candidate scores back to ``[B, m]``."""

    batch_size = int(group_sizes.numel())
    padded = flat_scores.new_zeros(batch_size, max_candidates)
    cursor = 0
    for batch_index, count in enumerate(group_sizes.tolist()):
        padded[batch_index, :count] = flat_scores[cursor : cursor + count]
        cursor += count
    return padded
