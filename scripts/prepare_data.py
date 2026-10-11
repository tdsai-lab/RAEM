#!/usr/bin/env python3
"""Prepare the deterministic ListUltraFeedback train/benchmark splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from private_score_alignment.config import load_config
from private_score_alignment.data.converters import prepare_named_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/main.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.override)
    print(prepare_named_dataset("list_ultrafeedback", config.data))


if __name__ == "__main__":
    main()
