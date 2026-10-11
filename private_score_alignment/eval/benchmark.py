"""Proxy-model evaluation for the RAEM+KTO main-table columns."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from private_score_alignment.config.schemas import ExperimentConfig
from private_score_alignment.data.datasets import infer_dataset_m, load_prepared_dataset
from private_score_alignment.eval.distributed import get_distributed_state, reduce_sum
from private_score_alignment.eval.external_common import resolve_generation_max_new_tokens
from private_score_alignment.eval.local_judges import (
    JudgeComparison,
    build_judge_comparisons,
    evaluate_local_judge_metrics,
    release_local_judge_models,
)
from private_score_alignment.losses.generation import generate_greedy_model_answers
from private_score_alignment.models.factory import load_policy_from_checkpoint


def _log_main_process(message: str) -> None:
    if get_distributed_state().is_main_process:
        tqdm.write(message)


def _generation_limit(config: ExperimentConfig) -> int:
    configured = resolve_generation_max_new_tokens(config.evaluation, "benchmark")
    training_limit = int(config.model.max_response_length)
    if training_limit <= 0:
        raise ValueError("model.max_response_length must be positive.")
    return min(configured, training_limit)


def _build_comparisons(
    *,
    config: ExperimentConfig,
    checkpoint_path: str,
    dataset_name: str,
) -> tuple[int, int, list[JudgeComparison], object]:
    state = get_distributed_state()
    dataset = load_prepared_dataset(config.data, dataset_name, "benchmark")
    resolved_m = infer_dataset_m(dataset)
    local_samples = dataset.samples[state.rank :: state.world_size]
    bundle = load_policy_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config=config.model,
        training_config=config.training,
    )
    bundle.policy_model.eval().to(state.device)
    bundle.reference_model.eval().to(state.device)

    comparisons: list[JudgeComparison] = []
    batch_size = max(1, int(config.training.per_device_eval_batch_size))
    max_new_tokens = _generation_limit(config)
    progress = tqdm(
        range(0, len(local_samples), batch_size),
        desc=f"proxy:{dataset_name}",
        dynamic_ncols=True,
        disable=not state.is_main_process,
    )
    for start in progress:
        samples = local_samples[start : start + batch_size]
        prompts = [sample.prompt for sample in samples]
        policy_answers = generate_greedy_model_answers(
            model=bundle.policy_model,
            tokenizer=bundle.tokenizer,
            prompts=prompts,
            model_config=config.model,
            eval_config=config.evaluation,
            device=state.device,
            max_new_tokens=max_new_tokens,
        )
        reference_answers = generate_greedy_model_answers(
            model=bundle.reference_model,
            tokenizer=bundle.tokenizer,
            prompts=prompts,
            model_config=config.model,
            eval_config=config.evaluation,
            device=state.device,
            max_new_tokens=max_new_tokens,
        )
        comparisons.extend(
            build_judge_comparisons(
                sample_ids=[sample.sample_id for sample in samples],
                prompts=prompts,
                policy_answers=policy_answers,
                sft_answers=reference_answers,
            )
        )
    return len(dataset), resolved_m, comparisons, bundle


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str,
    dataset_names: list[str],
    output_path: str | None = None,
    base_benchmark_path: str | None = None,
) -> dict[str, object]:
    """Evaluate policy-vs-reference answers with the two local proxy judges."""

    if base_benchmark_path:
        raise ValueError("Base-prediction ranking metrics are outside this minimal release.")
    if not dataset_names:
        dataset_names = ["list_ultrafeedback"]
    unsupported = sorted(set(dataset_names) - {"list_ultrafeedback"})
    if unsupported:
        raise ValueError(f"Unsupported proxy dataset(s): {unsupported}")

    state = get_distributed_state()
    results: dict[str, object] = {}
    for dataset_name in dataset_names:
        _log_main_process(f"proxy:{dataset_name}: generating policy/reference answers")
        total_examples, resolved_m, comparisons, bundle = _build_comparisons(
            config=config,
            checkpoint_path=checkpoint_path,
            dataset_name=dataset_name,
        )
        del bundle.policy_model
        del bundle.reference_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            metrics = evaluate_local_judge_metrics(
                comparisons=comparisons,
                eval_config=config.evaluation,
                device=state.device,
                shared_model_root=config.paths.pretrained_root,
            )
        finally:
            release_local_judge_models()
        reduced_examples = int(reduce_sum([float(len(comparisons))])[0])
        results[dataset_name] = {
            "m": resolved_m,
            "num_examples": total_examples,
            "num_examples_evaluated": reduced_examples,
            **metrics,
        }

    if output_path is not None and state.is_main_process:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
    return results
