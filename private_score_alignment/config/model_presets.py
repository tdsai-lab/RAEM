"""Named model presets shared by training and evaluation entrypoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

from private_score_alignment.config.schemas import ExperimentConfig


MODEL_PRESETS = {
    "qwen": {
        "policy_model_name": "hZzy/qwen2-0.5b-sft",
        "reference_model_name": "hZzy/qwen2-0.5b-sft (frozen)",
        "policy_model_path": os.environ.get("PSA_QWEN_MODEL_PATH", "hZzy/qwen2-0.5b-sft"),
        "reference_model_path": os.environ.get("PSA_QWEN_MODEL_PATH", "hZzy/qwen2-0.5b-sft"),
        "tokenizer_path": os.environ.get("PSA_QWEN_MODEL_PATH", "hZzy/qwen2-0.5b-sft"),
        "model_family": "qwen2",
        "training_output_name": "qwen2_0p5b_instruct_private_score_alignment",
        "training_run_name": "qwen2-0p5b-instruct-private-score-alignment",
    },
    "mistral": {
        "policy_model_name": "Mistral-7B-v0.1-SFT-ultrachat",
        "reference_model_name": "Mistral-7B-v0.1-SFT-ultrachat (frozen)",
        "policy_model_path": os.environ.get(
            "PSA_MISTRAL_MODEL_PATH", "kaitchup/Mistral-7B-v0.1-SFT-ultrachat"
        ),
        "reference_model_path": os.environ.get(
            "PSA_MISTRAL_MODEL_PATH", "kaitchup/Mistral-7B-v0.1-SFT-ultrachat"
        ),
        "tokenizer_path": os.environ.get(
            "PSA_MISTRAL_MODEL_PATH", "kaitchup/Mistral-7B-v0.1-SFT-ultrachat"
        ),
        "model_family": "mistral",
        "training_output_name": "mistral_7b_v0p1_sft_ultrachat_private_score_alignment",
        "training_run_name": "mistral-7b-v0.1-sft-ultrachat-private-score-alignment",
    },
}

_MODEL_TYPE_TO_PRESET = {
    "mistral": "mistral",
    "qwen2": "qwen",
}


def apply_model_preset(config: ExperimentConfig, model_name: str) -> None:
    """Apply a named SFT checkpoint preset to preference training."""

    preset = MODEL_PRESETS[model_name]
    config.model.policy_model_name = preset["policy_model_name"]
    config.model.reference_model_name = preset["reference_model_name"]
    config.model.policy_model_path = preset["policy_model_path"]
    config.model.reference_model_path = preset["reference_model_path"]
    config.model.tokenizer_path = preset["tokenizer_path"]
    config.model.model_family = preset["model_family"]

    config.training.output_name = preset["training_output_name"]
    config.training.run_name = preset["training_run_name"]


def infer_model_preset_from_checkpoint(checkpoint_path: str) -> str | None:
    """Infer a named model preset from a Hub id or local checkpoint directory."""

    normalized = checkpoint_path.strip()
    for preset_name, preset in MODEL_PRESETS.items():
        candidate_paths = {
            preset["policy_model_path"],
            preset["reference_model_path"],
            preset["tokenizer_path"],
        }
        if normalized in candidate_paths:
            return preset_name

    local_path = Path(normalized).expanduser()
    if not local_path.exists():
        return None

    model_config_path = local_path / "config.json"
    if model_config_path.exists():
        with model_config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model_type = str(payload.get("model_type", "")).strip().lower()
        return _MODEL_TYPE_TO_PRESET.get(model_type)

    return None
