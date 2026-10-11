"""Greedy generation helper used by the two local proxy-model evaluations."""

from __future__ import annotations

from dataclasses import replace

import torch


def generate_greedy_model_answers(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    model_config,
    eval_config,
    device: torch.device,
    max_new_tokens: int | None = None,
    truncate_prompts: bool = True,
) -> list[str]:
    if not prompts:
        return []

    from private_score_alignment.eval.external_common import GenerationExample, generate_completions
    from private_score_alignment.models.factory import PolicyBundle

    greedy_eval_config = replace(
        eval_config,
        generation_temperature=0.0,
        generation_top_p=1.0,
        generation_do_sample=False,
    )
    return generate_completions(
        bundle=PolicyBundle(policy_model=model, tokenizer=tokenizer),
        model_config=model_config,
        eval_config=greedy_eval_config,
        examples=[GenerationExample(prompt=str(prompt)) for prompt in prompts],
        device=device,
        max_new_tokens=max_new_tokens,
        truncate_prompts=truncate_prompts,
    )


__all__ = ["generate_greedy_model_answers"]
