"""Private Score-to-Generation Alignment package."""

from private_score_alignment.cache import configure_shared_cache_env


def _patch_accelerate_utils_compat() -> None:
    try:
        import accelerate.utils as accelerate_utils
    except Exception:
        return

    if hasattr(accelerate_utils, "is_mlu_available"):
        return

    def _is_mlu_available(*args, **kwargs) -> bool:
        del args, kwargs
        return False

    accelerate_utils.is_mlu_available = _is_mlu_available


_patch_accelerate_utils_compat()
configure_shared_cache_env()

from private_score_alignment.config.loader import load_config

__all__ = ["load_config"]
