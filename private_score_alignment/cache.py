"""Shared cache locations for Hugging Face, Torch, and dynamic modules."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SHARED_CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"
DEFAULT_HF_CACHE_HOME = DEFAULT_SHARED_CACHE_ROOT / ".cache" / "huggingface"


def _resolve_shared_cache_root(cache_root: str | Path | None = None) -> Path:
    if cache_root is not None:
        return Path(cache_root).expanduser().resolve()
    configured = os.environ.get("PSA_SHARED_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_SHARED_CACHE_ROOT


def _is_home_cache(value: str | None) -> bool:
    if not value:
        return True
    try:
        path = Path(value).expanduser().resolve()
    except OSError:
        return False
    home = Path.home().resolve()
    return path == home / ".cache" or (home / ".cache") in path.parents


def shared_cache_paths(cache_root: str | Path | None = None) -> dict[str, Path]:
    """Return the project-wide shared cache directories."""

    root = _resolve_shared_cache_root(cache_root)
    hf_home = root / ".cache" / "huggingface"
    return {
        "PSA_SHARED_CACHE_ROOT": root,
        "XDG_CACHE_HOME": root / ".cache",
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "TRANSFORMERS_CACHE": hf_home / "transformers",
        "HF_DATASETS_CACHE": hf_home / "datasets",
        "HF_MODULES_CACHE": hf_home / "modules",
        "TORCH_HOME": root / ".cache" / "torch",
        "TMPDIR": root / "tmp",
    }


def configure_shared_cache_env(
    cache_root: str | Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Set cache env vars to project-local or explicitly configured storage.

    Existing non-home paths are preserved unless ``force=True`` is passed. Set
    ``PSA_SHARED_CACHE_ROOT`` to place caches on a larger shared filesystem.
    """

    paths = shared_cache_paths(cache_root)
    for key, path in paths.items():
        if key == "PSA_SHARED_CACHE_ROOT":
            os.environ.setdefault(key, str(path))
            continue
        current = os.environ.get(key)
        if force or _is_home_cache(current):
            os.environ[key] = str(path)
        Path(os.environ[key]).expanduser().mkdir(parents=True, exist_ok=True)
    return paths


def hf_hub_cache_dir() -> str:
    """Return the configured shared Hub cache directory as a string."""

    configure_shared_cache_env()
    configured = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    )
    cache_dir = Path(configured).expanduser().resolve() if configured else DEFAULT_HF_CACHE_HOME / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)
