#!/usr/bin/env python3
"""Evaluate a checkpoint on Proxy Models, AlpacaEval, or MT-Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from private_score_alignment.cache import configure_shared_cache_env

configure_shared_cache_env(force=True)

from private_score_alignment.config import load_config
from private_score_alignment.config.model_presets import (
    MODEL_PRESETS,
    apply_model_preset,
    infer_model_preset_from_checkpoint,
)
from private_score_alignment.eval.distributed import (
    barrier,
    finalize_distributed_evaluation,
    init_distributed_evaluation,
)
from private_score_alignment.eval.external import run_evaluations
from private_score_alignment.trainer.runner import ensure_prepared_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/main.yaml")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="")
    parser.add_argument(
        "--framework",
        action="append",
        choices=["local", "alpaca_eval", "mt_bench"],
        default=[],
    )
    parser.add_argument("--output-path", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    state = init_distributed_evaluation()
    args = parse_args()
    try:
        config = load_config(args.config, overrides=args.override)
        preset = args.model or infer_model_preset_from_checkpoint(args.checkpoint_path)
        if preset:
            apply_model_preset(config, preset)
        frameworks = args.framework or ["local"]
        if "local" in frameworks:
            if state.is_main_process:
                ensure_prepared_data(config)
            barrier()
        results = run_evaluations(
            config=config,
            checkpoint_path=args.checkpoint_path,
            frameworks=frameworks,
            dataset_names=["list_ultrafeedback"],
            output_path=args.output_path,
            model_id=args.model_id,
        )
        if state.is_main_process:
            print(json.dumps(results, indent=2, ensure_ascii=False))
    finally:
        finalize_distributed_evaluation()


if __name__ == "__main__":
    main()
