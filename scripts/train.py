#!/usr/bin/env python3
"""Train one RAEM+KTO main-table setting from a public SFT checkpoint."""

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
from private_score_alignment.config.model_presets import MODEL_PRESETS, apply_model_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one RAEM+KTO main-table setting.")
    parser.add_argument("--config", default="config/main.yaml")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument("--epsilon", type=float, choices=[0.5, 1.0, 3.0], default=1.0)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = [
        *args.override,
        "data.dataset_mode=list_ultrafeedback",
        "privacy.dp_mechanism=linear_topb_em",
        "privacy.noise_mode=linear_topb_em",
        "privacy.dp_score_function=linear",
        "privacy.rank_branch_mode=hard_noisy_rank",
        "privacy.loss_type=rpw_kto",
        f"privacy.epsilon={args.epsilon}",
    ]

    from private_score_alignment.trainer.runner import run_training

    config = load_config(args.config, overrides=overrides)
    apply_model_preset(config, args.model)
    if args.output_suffix:
        suffix = args.output_suffix.strip("-_")
        config.training.output_name = f"{config.training.output_name}-{suffix}"
        config.training.run_name = f"{config.training.run_name}-{suffix}"
    print(json.dumps(run_training(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
