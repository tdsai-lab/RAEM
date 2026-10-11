"""Dispatch the proxy and general benchmarks reported in the main table."""

from __future__ import annotations

from private_score_alignment.config.schemas import ExperimentConfig
from private_score_alignment.eval.alpaca_eval import run_alpaca_eval
from private_score_alignment.eval.benchmark import evaluate_checkpoint
from private_score_alignment.eval.distributed import get_distributed_state
from private_score_alignment.eval.external_common import (
    infer_model_id,
    load_policy_bundle_for_generation,
    resolve_project_path,
    validate_external_eval_prerequisites,
    write_json,
)
from private_score_alignment.eval.mt_bench import run_mt_bench


SUPPORTED_EVAL_FRAMEWORKS = {"local", "alpaca_eval", "mt_bench"}


def run_evaluations(
    config: ExperimentConfig,
    checkpoint_path: str,
    frameworks: list[str],
    dataset_names: list[str],
    output_path: str = "",
    model_id: str = "",
    base_benchmark_path: str = "",
) -> dict[str, object]:
    """Run the selected main-table evaluation frameworks."""

    state = get_distributed_state()
    unknown = sorted(set(frameworks) - SUPPORTED_EVAL_FRAMEWORKS)
    if unknown:
        raise ValueError(f"Unsupported evaluation framework(s): {unknown}")

    resolved_model_id = infer_model_id(checkpoint_path, explicit_model_id=model_id)
    results: dict[str, object] = {}
    if "local" in frameworks:
        local_output_path = output_path or str(
            resolve_project_path(config.paths.output_root)
            / config.evaluation.metrics_output_subdir
            / f"{resolved_model_id}.json"
        )
        results["local"] = evaluate_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            dataset_names=dataset_names,
            output_path=local_output_path,
            base_benchmark_path=base_benchmark_path or None,
        )

    external_frameworks = [name for name in frameworks if name != "local"]
    if external_frameworks:
        validate_external_eval_prerequisites(config=config, frameworks=external_frameworks)
        bundle = load_policy_bundle_for_generation(checkpoint_path=checkpoint_path, config=config)
        if "alpaca_eval" in external_frameworks:
            results["alpaca_eval"] = run_alpaca_eval(
                config=config,
                checkpoint_path=checkpoint_path,
                model_id=resolved_model_id,
                bundle=bundle,
            )
        if "mt_bench" in external_frameworks:
            results["mt_bench"] = run_mt_bench(
                config=config,
                checkpoint_path=checkpoint_path,
                model_id=resolved_model_id,
                bundle=bundle,
            )

    if external_frameworks and state.is_main_process:
        summary_path = (
            resolve_project_path(config.paths.output_root)
            / config.evaluation.external_output_subdir
            / "summary"
            / f"{resolved_model_id}.json"
        )
        write_json(summary_path, results)
        results["summary_path"] = str(summary_path)
    return results
