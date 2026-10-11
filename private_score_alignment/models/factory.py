"""Factories for policy/reference/tokenizer construction."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from private_score_alignment.cache import hf_hub_cache_dir
from private_score_alignment.config.schemas import ModelConfig, TrainingConfig


_FLASH_ATTN_FALLBACK_WARNED = False


@dataclass
class ModelBundle:
    """Bundle of trainable policy model, frozen reference model, and tokenizer."""

    policy_model: torch.nn.Module
    reference_model: torch.nn.Module
    tokenizer: object


@dataclass
class PolicyBundle:
    """Bundle of inference-time policy model and tokenizer."""

    policy_model: torch.nn.Module
    tokenizer: object


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a configured dtype name."""

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[key]


def resolve_attn_implementation(requested: str | None) -> str | None:
    """Resolve attention backend with a safe fallback.

    If ``flash_attention_2`` is requested but ``flash_attn`` is unavailable in the
    current environment, automatically fall back to ``sdpa``.
    """

    if requested is None:
        return None
    if requested != "flash_attention_2":
        return requested
    if importlib.util.find_spec("flash_attn") is not None:
        return requested
    global _FLASH_ATTN_FALLBACK_WARNED
    if not _FLASH_ATTN_FALLBACK_WARNED:
        warnings.warn(
            "model.attn_implementation=flash_attention_2 was requested, but "
            "flash_attn is not importable. Falling back to sdpa.",
            RuntimeWarning,
            stacklevel=2,
        )
        _FLASH_ATTN_FALLBACK_WARNED = True
    return "sdpa"


def resolve_hf_cache_dir() -> str:
    """Return the shared cache dir used for Hub-backed model/tokenizer downloads."""

    return hf_hub_cache_dir()


def build_tokenizer(model_config: ModelConfig):
    """Load the tokenizer for the configured model family."""

    tokenizer_path = resolve_pretrained_path(
        model_config.tokenizer_path or model_config.policy_model_path,
        purpose="tokenizer",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=resolve_hf_cache_dir(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    _ensure_qwen2_chat_template(tokenizer, model_config)
    return tokenizer


# ChatML template matching Qwen2-Instruct. Injected onto tokenizers that do not
# ship with a chat template so preference training uses consistent formatting.
_QWEN2_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def _ensure_qwen2_chat_template(tokenizer, model_config: ModelConfig) -> None:
    """Attach the Qwen2 ChatML template + align eos/pad with the template.

    Qwen2-Instruct ships with ``chat_template`` set. Some Qwen2 tokenizers keep
    ``eos_token = <|endoftext|>``, which does not match the chat template's
    end-of-turn marker ``<|im_end|>``.

    For Qwen2 we therefore:
      * inject the ChatML template if missing;
      * realign eos (and pad) to ``<|im_end|>`` whenever the tokenizer exposes
        that token — so the model is trained to close turns with ``<|im_end|>``.
    """
    if getattr(model_config, "model_family", "").lower() != "qwen2":
        return
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = _QWEN2_CHATML_TEMPLATE
        for special in ("<|im_start|>", "<|im_end|>"):
            if tokenizer.convert_tokens_to_ids(special) == tokenizer.unk_token_id:
                tokenizer.add_special_tokens({"additional_special_tokens": [special]})

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        return
    if tokenizer.eos_token_id != im_end_id:
        tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Keep pad != eos when possible so label==-100 on pad is unambiguous.
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_id is not None and endoftext_id != tokenizer.unk_token_id:
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token


def configure_training_model(
    model: torch.nn.Module,
    training_config: TrainingConfig,
) -> torch.nn.Module:
    """Apply training-time settings like gradient checkpointing."""

    if training_config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    return model


def build_real_model(
    model_path: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer=None,
) -> torch.nn.Module:
    """Load a pretrained causal LM from disk or the Hugging Face Hub."""

    torch_dtype = resolve_torch_dtype(model_config.torch_dtype)
    resolved_model_path = resolve_pretrained_path(model_path, purpose="model")
    resolved_attn = resolve_attn_implementation(model_config.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=resolved_attn,
        cache_dir=resolve_hf_cache_dir(),
    )
    if tokenizer is not None and getattr(model_config, "model_family", "").lower() == "qwen2":
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.eos_token_id = tokenizer.eos_token_id
                if tokenizer.pad_token_id is not None:
                    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def resolve_pretrained_path(identifier: str, purpose: str) -> str:
    """Resolve a local pretrained path or pass through a Hub id.

    Absolute paths, ``./relative`` paths, and ``~/`` paths are treated as filesystem paths.
    If such a path does not exist, raise a clear ``FileNotFoundError`` instead of letting
    Transformers misinterpret it as a Hub repo id.
    """

    if not identifier:
        raise ValueError(f"Missing {purpose} identifier.")

    looks_like_local_path = (
        identifier.startswith("/")
        or identifier.startswith("./")
        or identifier.startswith("../")
        or identifier.startswith("~/")
    )
    if not looks_like_local_path:
        return identifier

    path = Path(identifier).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{purpose.capitalize()} path does not exist: {path}. "
            "Update the configured path or pass a valid local checkpoint/model directory."
        )
    return str(path.resolve())


def resolve_local_path_if_exists(identifier: str | None) -> Path | None:
    """Return a resolved local path only when the identifier is truly filesystem-backed."""

    if not identifier:
        return None
    looks_like_local_path = (
        identifier.startswith("/")
        or identifier.startswith("./")
        or identifier.startswith("../")
        or identifier.startswith("~/")
    )
    if not looks_like_local_path:
        return None
    path = Path(identifier).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local path does not exist: {path}")
    return path.resolve()


def build_model_bundle(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    policy_init_checkpoint: str | None = None,
    reference_model_override: str | None = None,
) -> ModelBundle:
    """Build policy model, frozen reference model, and tokenizer."""

    tokenizer = build_tokenizer(model_config)

    policy_source = policy_init_checkpoint or model_config.policy_model_path
    reference_source = reference_model_override or model_config.reference_model_path
    policy_model = build_real_model(
        model_path=policy_source,
        model_config=model_config,
        training_config=training_config,
    )
    reference_model = build_real_model(
        model_path=reference_source,
        model_config=model_config,
        training_config=training_config,
    )

    policy_model = configure_training_model(policy_model, training_config)

    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    reference_model.eval()
    if hasattr(reference_model, "config"):
        reference_model.config.use_cache = False

    return ModelBundle(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
    )


def build_eval_model_bundle(
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> ModelBundle:
    """Build evaluation-time policy/reference models from full checkpoints."""

    tokenizer = build_tokenizer(model_config)
    policy_model = build_real_model(
        model_path=model_config.policy_model_path,
        model_config=model_config,
        training_config=training_config,
    )
    reference_model = build_real_model(
        model_path=model_config.reference_model_path,
        model_config=model_config,
        training_config=training_config,
    )

    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    reference_model.eval()
    if hasattr(reference_model, "config"):
        reference_model.config.use_cache = False

    return ModelBundle(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
    )


def load_policy_from_checkpoint(
    checkpoint_path: str | None,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> ModelBundle:
    """Load a policy model for evaluation from a full checkpoint."""

    if not checkpoint_path:
        return build_model_bundle(model_config, training_config)

    checkpoint_source = resolve_pretrained_path(checkpoint_path, purpose="checkpoint")
    bundle = build_eval_model_bundle(model_config, training_config)
    bundle.policy_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_source,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=resolve_torch_dtype(model_config.torch_dtype),
        attn_implementation=resolve_attn_implementation(model_config.attn_implementation),
        cache_dir=resolve_hf_cache_dir(),
    )
    bundle.policy_model.eval()
    return bundle


def load_policy_for_inference(
    checkpoint_path: str | None,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> PolicyBundle:
    """Load only the policy model and tokenizer for generation/evaluation."""

    checkpoint_source = (
        resolve_pretrained_path(checkpoint_path, purpose="checkpoint") if checkpoint_path else None
    )
    checkpoint_dir = resolve_local_path_if_exists(checkpoint_source)
    tokenizer_source = model_config.tokenizer_path or model_config.policy_model_path
    if checkpoint_dir and (
        (checkpoint_dir / "tokenizer.json").exists()
        or (checkpoint_dir / "tokenizer_config.json").exists()
    ):
        tokenizer_source = str(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_pretrained_path(tokenizer_source, purpose="tokenizer"),
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=resolve_hf_cache_dir(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    if checkpoint_source:
        policy_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_source,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=resolve_torch_dtype(model_config.torch_dtype),
            attn_implementation=resolve_attn_implementation(model_config.attn_implementation),
            cache_dir=resolve_hf_cache_dir(),
        )
    else:
        policy_model = build_real_model(
            model_path=model_config.policy_model_path,
            model_config=model_config,
            training_config=training_config,
        )

    policy_model.eval()
    if hasattr(policy_model, "config"):
        policy_model.config.use_cache = True
    return PolicyBundle(policy_model=policy_model, tokenizer=tokenizer)
