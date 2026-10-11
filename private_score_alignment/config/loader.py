"""YAML configuration loading helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from private_score_alignment.config.schemas import ExperimentConfig, build_dataclass


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if not isinstance(parsed, str):
        return parsed

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(config_dict: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted-key overrides like ``privacy.epsilon=2``."""

    if not overrides:
        return config_dict

    updated = deepcopy(config_dict)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must look like section.key=value, got: {override}")
        key_path, raw_value = override.split("=", 1)
        target = updated
        keys = key_path.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = _coerce_scalar(raw_value)
    return updated


def _override_keys(overrides: list[str] | None) -> set[str]:
    """Return the dotted override keys provided on the command line."""

    if not overrides:
        return set()
    return {item.split("=", 1)[0] for item in overrides if "=" in item}


def apply_dataset_specific_defaults(
    config_dict: dict[str, Any],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Keep the released ListUltraFeedback candidate count at eight by default."""

    updated = deepcopy(config_dict)
    data = updated.setdefault("data", {})
    dataset_mode = data.get("dataset_mode")
    explicit_overrides = _override_keys(overrides)

    if dataset_mode == "list_ultrafeedback":
        if "data.num_candidates" not in explicit_overrides and data.get("num_candidates", 4) == 4:
            data["num_candidates"] = 8
        if "data.min_candidates" not in explicit_overrides:
            current_min = data.get("min_candidates", data.get("num_candidates", 8))
            data["min_candidates"] = max(int(current_min), int(data.get("num_candidates", 8)))

    return updated


def apply_privacy_hyperparameter_aliases(
    config_dict: dict[str, Any],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Return a copy before constructing the typed privacy configuration."""

    updated = deepcopy(config_dict)
    privacy = updated.get("privacy")
    if not isinstance(privacy, dict):
        return updated

    return updated


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML with optional overrides."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload = apply_overrides(payload, overrides)
    payload = apply_privacy_hyperparameter_aliases(payload, overrides)
    payload = apply_dataset_specific_defaults(payload, overrides)
    return build_dataclass(ExperimentConfig, payload)


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    """Save a config object as YAML."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
