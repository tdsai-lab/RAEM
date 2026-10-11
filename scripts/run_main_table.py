#!/usr/bin/env python3
"""Train and evaluate the RAEM+KTO rows in the paper's main table.

This entry point intentionally excludes the randomized-response baseline. Run it
once for Qwen and once for Mistral; by default each run covers epsilon 0.5, 1,
and 3 and evaluates both local proxy models.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from private_score_alignment.cache import configure_shared_cache_env


DEFAULT_EPSILONS = (0.5, 1.0, 3.0)


def _configure_environment() -> None:
    configure_shared_cache_env(force=True)
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _running_under_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def _parse_num_gpus_for_launcher() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-gpus", type=int, required=True)
    args, _ = parser.parse_known_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be a positive integer")
    return args.num_gpus


def _maybe_reexec_with_torchrun(num_gpus: int) -> None:
    if _running_under_torchrun():
        return

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(num_gpus),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(
        f"[launcher] torchrun nproc_per_node={num_gpus}",
        flush=True,
    )
    os.execvpe(sys.executable, command, os.environ.copy())


_configure_environment()
if "-h" not in sys.argv and "--help" not in sys.argv:
    _maybe_reexec_with_torchrun(_parse_num_gpus_for_launcher())

from private_score_alignment.config import load_config
from private_score_alignment.config.model_presets import MODEL_PRESETS, apply_model_preset
from private_score_alignment.eval.distributed import (
    barrier,
    finalize_distributed_evaluation,
    get_distributed_state,
    init_distributed_evaluation,
)
from private_score_alignment.eval.external import run_evaluations
from private_score_alignment.trainer.runner import ensure_prepared_data, run_training


def _parse_epsilons(value: str) -> list[float]:
    epsilons = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not epsilons:
        raise ValueError("At least one epsilon is required.")
    if any(epsilon <= 0 for epsilon in epsilons):
        raise ValueError("All epsilon values must be positive.")
    unsupported = sorted(set(epsilons) - set(DEFAULT_EPSILONS))
    if unsupported:
        raise ValueError(f"Main-table epsilon must be one of {DEFAULT_EPSILONS}; got {unsupported}.")
    return epsilons


def _epsilon_tag(epsilon: float) -> str:
    return f"{float(epsilon):g}".replace(".", "p")


def _resolve_checkpoint_dir(artifacts: dict[str, Any]) -> str:
    if "checkpoint_dir" in artifacts:
        return str(artifacts["checkpoint_dir"])
    if "preference" in artifacts:
        return str(artifacts["preference"]["checkpoint_dir"])
    raise KeyError(f"Cannot find checkpoint_dir in artifacts: {artifacts}")


def _cleanup_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce RAEM+KTO for one model at epsilon 0.5, 1, and 3."
    )
    parser.add_argument("--config", default="config/main.yaml")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of local GPU processes to launch; must be a positive integer.",
    )
    parser.add_argument(
        "--epsilons",
        default=",".join(str(value) for value in DEFAULT_EPSILONS),
        help="Comma-separated privacy budgets (default: 0.5,1,3).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Metrics directory (default: outputs/main_table/<model>).",
    )
    parser.add_argument(
        "--skip-proxy-eval",
        action="store_true",
        help="Train only; proxy-model evaluation can be run later with scripts/eval.py.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable dotted config override, for example data.max_train_samples=8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_distributed_evaluation()
    state = get_distributed_state()
    try:
        base_config = load_config(args.config, overrides=args.override)
        apply_model_preset(base_config, args.model)
        base_output_name = base_config.training.output_name
        base_run_name = base_config.training.run_name

        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else PROJECT_ROOT / "outputs" / "main_table" / args.model
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary: list[dict[str, Any]] = []

        for epsilon in _parse_epsilons(args.epsilons):
            epsilon_tag = _epsilon_tag(epsilon)
            tag = f"raem-kto-list-ultrafeedback-eps{epsilon_tag}"
            if state.is_main_process:
                print(
                    f"\n{'=' * 72}\n"
                    f"RAEM+KTO | model={args.model} | epsilon={epsilon:g}\n"
                    f"{'=' * 72}",
                    flush=True,
                )

            config = deepcopy(base_config)
            config.data.dataset_mode = "list_ultrafeedback"
            config.privacy.dp_mechanism = "linear_topb_em"
            config.privacy.noise_mode = "linear_topb_em"
            config.privacy.dp_score_function = "linear"
            config.privacy.rank_branch_mode = "hard_noisy_rank"
            config.privacy.loss_type = "rpw_kto"
            config.privacy.epsilon = float(epsilon)
            config.training.output_name = f"{base_output_name}-{tag}"
            config.training.run_name = f"{base_run_name}-{tag}"

            artifacts = run_training(config)
            checkpoint_dir = _resolve_checkpoint_dir(artifacts)
            results: dict[str, Any] = {}

            if not args.skip_proxy_eval:
                if state.is_main_process:
                    ensure_prepared_data(config)
                barrier()
                results = run_evaluations(
                    config=config,
                    checkpoint_path=checkpoint_dir,
                    frameworks=["local"],
                    dataset_names=["list_ultrafeedback"],
                    output_path=str(output_dir / f"eps{epsilon_tag}.json"),
                )

            summary.append(
                {
                    "model": args.model,
                    "method": "RAEM+KTO",
                    "epsilon": float(epsilon),
                    "checkpoint_dir": checkpoint_dir,
                    "proxy_results": results,
                }
            )
            _cleanup_cuda_cache()

        if state.is_main_process:
            summary_path = output_dir / "summary.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            print(f"Summary written to {summary_path}", flush=True)
    finally:
        finalize_distributed_evaluation()


if __name__ == "__main__":
    main()
