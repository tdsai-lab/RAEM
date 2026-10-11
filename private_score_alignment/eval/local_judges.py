"""Local reward-judge integrations for benchmark win-rate metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
from typing import Callable

from huggingface_hub import snapshot_download
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from private_score_alignment.cache import DEFAULT_SHARED_CACHE_ROOT, configure_shared_cache_env
from private_score_alignment.config.schemas import EvalConfig
from private_score_alignment.eval.distributed import barrier, get_distributed_state, reduce_sum


PAIR_PREFERENCE_MODEL_NAME = "pair_preference"
SCORING_MODEL_NAME = "scoring"
SCALAR_JUDGE_TIE_EPSILON = 1e-4
PAIR_PREFERENCE_CONTEXT_TEMPLATE = "\n\n\n<turn> user\n {prompt}\n\n\n"


@dataclass(frozen=True)
class JudgeComparison:
    """A single prompt-level policy-vs-SFT comparison for local judges."""

    sample_id: str
    prompt: str
    model_answer: str
    baseline_answer: str


@dataclass(frozen=True)
class LocalJudgeSpec:
    """Registered local-judge evaluator and output namespace."""

    name: str
    evaluator: Callable[..., dict[str, float | int | str | None]]


@dataclass
class PairPreferenceRuntime:
    """Runtime bundle for the RLHFlow pair-preference judge."""

    model: torch.nn.Module
    tokenizer: object
    option_a_token_id: int
    option_b_token_id: int
    supports_logits_to_keep: bool


@dataclass
class ScoringRuntime:
    """Runtime bundle for scalar reward-model judges."""

    model: torch.nn.Module
    tokenizer: object


_PAIR_RUNTIME_CACHE: dict[tuple[str, str], PairPreferenceRuntime] = {}
_SCORING_RUNTIME_CACHE: dict[tuple[str, str], ScoringRuntime] = {}


def _maybe_int(value: float) -> float | int:
    """Preserve integer-looking counters while allowing float reductions."""

    rounded = int(value)
    return rounded if float(rounded) == float(value) else float(value)


def _judge_summary(
    *,
    prefix: str,
    model_name: str,
    wins: float,
    losses: float,
    ties: int,
    invalids: int,
    total_examples: int,
) -> dict[str, float | int | str | None]:
    """Format a local-judge metric block without changing the external schema shape."""

    non_tie_examples = float(wins) + float(losses)
    win_rate = float(wins / non_tie_examples) if non_tie_examples else None
    tie_rate = float(ties / total_examples) if total_examples else None
    net_win_rate = float((wins - losses) / total_examples) if total_examples else None
    return {
        f"{prefix}_judge_model": model_name,
        f"{prefix}_win_rate": win_rate,
        f"{prefix}_adjusted_win_rate": win_rate,
        f"{prefix}_tie_rate": tie_rate,
        f"{prefix}_net_win_rate": net_win_rate,
        f"{prefix}_wins": _maybe_int(wins),
        f"{prefix}_losses": _maybe_int(losses),
        f"{prefix}_ties": ties,
        f"{prefix}_invalids": invalids,
        f"{prefix}_num_examples_evaluated": _maybe_int(non_tie_examples),
        f"{prefix}_total": total_examples,
    }


def _scalar_judge_outcome_counts(
    model_scores: torch.Tensor,
    baseline_scores: torch.Tensor,
    tie_epsilon: float = SCALAR_JUDGE_TIE_EPSILON,
) -> tuple[int, int, int]:
    """Count scalar reward-model wins, losses, and ties with a small tie band."""

    margins = model_scores.float() - baseline_scores.float()
    wins = int((margins > float(tie_epsilon)).sum().item())
    losses = int((margins < -float(tie_epsilon)).sum().item())
    ties = int((margins.abs() <= float(tie_epsilon)).sum().item())
    return wins, losses, ties


def _device_cache_key(device: torch.device) -> str:
    """Convert a device into a stable runtime-cache key."""

    if device.type == "cuda":
        return f"cuda:{device.index}"
    return device.type


def _judge_dtype(device: torch.device) -> torch.dtype:
    """Use bf16 on GPU judge inference and float32 elsewhere."""

    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _judge_progress(total: int, desc: str):
    """Create a main-process-only progress bar for local-judge evaluation."""

    return tqdm(
        total=total,
        desc=desc,
        dynamic_ncols=True,
        disable=not get_distributed_state().is_main_process or total <= 0,
    )


def _local_judge_tokenizer_kwargs(eval_config: EvalConfig) -> dict[str, object]:
    """Build tokenizer kwargs for local-judge batches.

    The benchmark path can disable truncation by passing ``local_judge_max_length=0``.
    """

    max_length = int(getattr(eval_config, "local_judge_max_length", 0) or 0)
    kwargs: dict[str, object] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_length > 0,
    }
    if max_length > 0:
        kwargs["max_length"] = max_length
    return kwargs


def _rendered_text_tokenizer_kwargs(eval_config: EvalConfig) -> dict[str, object]:
    """Build tokenizer kwargs for text that already includes model-specific formatting."""

    kwargs = _local_judge_tokenizer_kwargs(eval_config)
    kwargs["add_special_tokens"] = False
    return kwargs


def _empty_judge_summary(prefix: str, model_name: str) -> dict[str, float | int | str | None]:
    """Return an empty metric block for a local judge."""

    return _judge_summary(
        prefix=prefix,
        model_name=model_name,
        wins=0.0,
        losses=0.0,
        ties=0,
        invalids=0,
        total_examples=0,
    )


def _local_judge_model_name(name: str, eval_config: EvalConfig) -> str:
    """Return the configured model id for a registered local judge."""

    if name == PAIR_PREFERENCE_MODEL_NAME:
        return eval_config.local_pair_preference_judge_model
    if name == SCORING_MODEL_NAME:
        return eval_config.local_scoring_judge_model
    raise ValueError(f"Unknown local judge name: {name}")


def empty_local_judge_metrics(eval_config: EvalConfig) -> dict[str, float | int | str | None]:
    """Return empty metric blocks for all registered local judges."""

    summary: dict[str, float | int | str | None] = {}
    for spec in LOCAL_JUDGE_REGISTRY:
        summary.update(_empty_judge_summary(spec.name, _local_judge_model_name(spec.name, eval_config)))
    return summary


def _is_cuda_oom(error: BaseException) -> bool:
    """Return whether an exception came from CUDA memory exhaustion."""

    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _configure_dynamic_module_cache(modules_root: Path) -> None:
    """Point Transformers' dynamic module cache at shared storage instead of ``~/.cache``."""

    modules_root.mkdir(parents=True, exist_ok=True)
    init_file = modules_root / "__init__.py"
    init_file.touch(exist_ok=True)

    cache_path = str(modules_root)
    os.environ["HF_MODULES_CACHE"] = cache_path

    import transformers.dynamic_module_utils as dynamic_module_utils
    from transformers import utils as transformers_utils

    dynamic_module_utils.HF_MODULES_CACHE = cache_path
    transformers_utils.HF_MODULES_CACHE = cache_path
    if cache_path not in sys.path:
        sys.path.append(cache_path)
    importlib.invalidate_caches()
    dynamic_module_utils.init_hf_modules()


def _prepare_shared_hf_cache(shared_model_root: str | None) -> tuple[Path, Path] | None:
    """Configure model/cache directories for judge downloads and remote code."""

    environment_root = os.environ.get("PSA_SHARED_CACHE_ROOT", "").strip()
    if environment_root:
        root = Path(environment_root).expanduser().resolve()
    elif shared_model_root:
        root = Path(shared_model_root).expanduser().resolve()
    else:
        root = DEFAULT_SHARED_CACHE_ROOT
    cache_paths = configure_shared_cache_env(root, force=True)
    model_root = root / "judge_models"
    cache_root = cache_paths["HF_HUB_CACHE"]
    modules_root = cache_paths["HF_MODULES_CACHE"]

    model_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    _configure_dynamic_module_cache(modules_root)
    return model_root, cache_root


def _ensure_armorm_transformers_compatibility() -> None:
    """Backfill symbols older ArmoRM remote code expects from Transformers."""

    from transformers.models.llama import modeling_llama

    if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
        modeling_llama.LLAMA_INPUTS_DOCSTRING = ""


def _materialize_model_locally(model_id: str, shared_model_root: str | None = None) -> str:
    """Download a judge model into shared storage once and reuse the local path across ranks."""

    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    prepared = _prepare_shared_hf_cache(shared_model_root)
    if prepared is None:
        return model_id

    model_root, cache_root = prepared
    local_dir = model_root / model_id.replace("/", "__")
    ready_marker = local_dir / ".download_complete"
    state = get_distributed_state()

    if state.is_main_process and not ready_marker.exists():
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            cache_dir=str(cache_root),
            resume_download=True,
        )
        ready_marker.write_text("ready\n", encoding="utf-8")

    barrier()

    if not local_dir.exists():
        raise FileNotFoundError(
            f"Judge model '{model_id}' was expected at {local_dir}, but it was not materialized."
        )
    return str(local_dir.resolve())


def _ensure_tokenizer_padding(tokenizer: object, *, padding_side: str = "right") -> None:
    """Ensure chat tokenizers have a valid pad token for batched inference."""

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = padding_side


def _add_sequence_classifier_pad_token(tokenizer: object, model: torch.nn.Module) -> None:
    """Add a real pad special token for sequence-classification reward models."""

    add_special_tokens = getattr(tokenizer, "add_special_tokens", None)
    if not callable(add_special_tokens):
        return
    try:
        added = int(add_special_tokens({"pad_token": "<|pad|>"}))
    except (TypeError, ValueError):
        return
    if added > 0 and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))


def _ensure_sequence_classifier_padding(tokenizer: object, model: torch.nn.Module) -> None:
    """Avoid EOS-as-padding for sequence-classification reward heads."""

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None and eos_token_id is not None and int(pad_token_id) == int(eos_token_id):
        _add_sequence_classifier_pad_token(tokenizer, model)

    config = getattr(model, "config", None)
    if config is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        config.pad_token_id = int(tokenizer.pad_token_id)


def _single_option_token_id(tokenizer: object, label: str) -> int:
    """Resolve the single-token id for the pair-preference judge outputs."""

    for candidate in (label, f" {label}"):
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1:
            return int(token_ids[0])
    raise ValueError(
        f"Could not find a single-token representation for judge label '{label}'."
    )


def _supports_logits_to_keep(model: torch.nn.Module) -> bool:
    """Return whether the model forward path supports keeping only final-step logits."""

    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    return "logits_to_keep" in signature.parameters


def _can_use_pair_preference_logits_to_keep(runtime: PairPreferenceRuntime) -> bool:
    """Use final-logit slicing only when padding preserves the final content token."""

    return bool(runtime.supports_logits_to_keep and getattr(runtime.tokenizer, "padding_side", None) == "left")


def _load_pair_preference_runtime(
    model_id: str,
    device: torch.device,
    shared_model_root: str | None = None,
) -> PairPreferenceRuntime:
    """Load and cache the RLHFlow pair-preference judge on the current rank/device."""

    local_model_path = _materialize_model_locally(model_id, shared_model_root=shared_model_root)
    cache_key = (local_model_path, _device_cache_key(device))
    if cache_key in _PAIR_RUNTIME_CACHE:
        return _PAIR_RUNTIME_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)
    _ensure_tokenizer_padding(tokenizer, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=_judge_dtype(device),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    runtime = PairPreferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        option_a_token_id=_single_option_token_id(tokenizer, "A"),
        option_b_token_id=_single_option_token_id(tokenizer, "B"),
        supports_logits_to_keep=_supports_logits_to_keep(model),
    )
    _PAIR_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _load_scoring_runtime(
    model_id: str,
    device: torch.device,
    shared_model_root: str | None = None,
) -> ScoringRuntime:
    """Load and cache the RLHFlow ArmoRM judge on the current rank/device."""

    local_model_path = _materialize_model_locally(model_id, shared_model_root=shared_model_root)
    cache_key = (local_model_path, _device_cache_key(device))
    if cache_key in _SCORING_RUNTIME_CACHE:
        return _SCORING_RUNTIME_CACHE[cache_key]

    _prepare_shared_hf_cache(shared_model_root)
    _ensure_armorm_transformers_compatibility()

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True, trust_remote_code=True)
    _ensure_tokenizer_padding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=_judge_dtype(device),
        low_cpu_mem_usage=True,
    )
    _ensure_sequence_classifier_padding(tokenizer, model)
    model.to(device)
    model.eval()

    runtime = ScoringRuntime(model=model, tokenizer=tokenizer)
    _SCORING_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _pair_preference_prompt(prompt: str, answer_a: str, answer_b: str) -> str:
    """Construct the RLHFlow pair-preference training template."""

    return (
        f"[CONTEXT] {prompt} "
        f"[RESPONSE A] {answer_a} "
        f"[RESPONSE B] {answer_b} \n"
    )


def _pair_preference_context_text(prompt: str) -> str:
    """Render the single-turn benchmark prompt with RLHFlow's plain context template."""

    return PAIR_PREFERENCE_CONTEXT_TEMPLATE.format(prompt=prompt)


def _pair_preference_prompt_text(
    tokenizer: object,
    prompt: str,
    answer_a: str,
    answer_b: str,
) -> str:
    """Render the full RLHFlow pair-preference prompt for next-token A/B scoring."""

    slic_prompt = _pair_preference_prompt(
        _pair_preference_context_text(prompt),
        answer_a,
        answer_b,
    )
    return _messages_to_text(
        tokenizer,
        [{"role": "user", "content": slic_prompt}],
        add_generation_prompt=True,
    )


def _policy_answer_is_option_a(sample_id: str) -> bool:
    """Counterbalance pairwise judge answer order to avoid systematic A/B bias."""

    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return bool(digest[0] % 2 == 0)


def _messages_to_text(tokenizer: object, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Render chat messages into a model-specific prompt string."""

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _finite_positive_model_length(value: object) -> int | None:
    """Return finite model/tokenizer length limits while ignoring HF sentinels."""

    try:
        length = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if 0 < length < 1_000_000_000:
        return length
    return None


def _reward_model_max_length(runtime: ScoringRuntime) -> int | None:
    """Return the tightest declared max length for a sequence-classification RM."""

    limits = [
        _finite_positive_model_length(getattr(runtime.tokenizer, "model_max_length", None)),
        _finite_positive_model_length(getattr(getattr(runtime.model, "config", None), "max_position_embeddings", None)),
    ]
    finite_limits = [limit for limit in limits if limit is not None]
    return min(finite_limits) if finite_limits else None


def _scoring_tokenizer_kwargs(
    eval_config: EvalConfig,
    runtime: ScoringRuntime,
) -> dict[str, object]:
    """Build tokenizer kwargs for ArmoRM without exceeding its position limit."""

    kwargs = _rendered_text_tokenizer_kwargs(eval_config)
    max_length = _reward_model_max_length(runtime)
    if max_length is not None:
        configured_max_length = int(kwargs.get("max_length", max_length))
        kwargs["truncation"] = True
        kwargs["max_length"] = min(configured_max_length, max_length)
    return kwargs


def _scalar_scores_from_outputs(outputs: object) -> torch.Tensor:
    """Extract one scalar reward score per prompt-response example."""

    if hasattr(outputs, "score"):
        scores = outputs.score
    else:
        scores = outputs.logits
    scores = scores.float()
    if scores.dim() == 0:
        return scores.reshape(1)
    if scores.dim() == 1:
        return scores
    if scores.size(-1) == 1:
        return scores.squeeze(-1)
    raise ValueError(f"Expected scalar reward scores, got tensor shape {tuple(scores.shape)}.")


def build_judge_comparisons(
    sample_ids: list[str],
    prompts: list[str],
    policy_answers: list[str],
    sft_answers: list[str],
) -> list[JudgeComparison]:
    """Build prompt-level policy-vs-SFT answer comparisons for judge win-rate metrics."""

    if not (
        len(sample_ids) == len(prompts) == len(policy_answers) == len(sft_answers)
    ):
        raise ValueError("Judge comparison inputs must share the same batch length.")

    comparisons: list[JudgeComparison] = []
    for sample_id, prompt, policy_answer, sft_answer in zip(
        sample_ids,
        prompts,
        policy_answers,
        sft_answers,
    ):
        comparisons.append(
            JudgeComparison(
                sample_id=str(sample_id),
                prompt=str(prompt),
                model_answer=str(policy_answer),
                baseline_answer=str(sft_answer),
            )
        )
    return comparisons


@torch.inference_mode()
def evaluate_pair_preference_judge(
    comparisons: list[JudgeComparison],
    eval_config: EvalConfig,
    device: torch.device,
    shared_model_root: str | None = None,
) -> dict[str, float | int | str | None]:
    """Compute policy-vs-SFT win rate using the RLHFlow pair-preference judge."""

    runtime = _load_pair_preference_runtime(
        eval_config.local_pair_preference_judge_model,
        device=device,
        shared_model_root=shared_model_root,
    )
    wins = 0.0
    losses = 0.0
    ties = 0
    invalids = 0
    batch_size = max(1, int(eval_config.local_judge_batch_size))
    progress = _judge_progress(len(comparisons), "judge:pair_preference")

    start = 0
    try:
        while start < len(comparisons):
            current_batch_size = min(batch_size, len(comparisons) - start)
            while True:
                chunk = comparisons[start : start + current_batch_size]
                policy_is_option_a = [_policy_answer_is_option_a(item.sample_id) for item in chunk]
                prompt_texts = [
                    _pair_preference_prompt_text(
                        runtime.tokenizer,
                        item.prompt,
                        item.model_answer if policy_first else item.baseline_answer,
                        item.baseline_answer if policy_first else item.model_answer,
                    )
                    for item, policy_first in zip(chunk, policy_is_option_a)
                ]
                encoded = runtime.tokenizer(prompt_texts, **_rendered_text_tokenizer_kwargs(eval_config))
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model_kwargs = {
                    **encoded,
                    "use_cache": False,
                    "return_dict": True,
                }
                if _can_use_pair_preference_logits_to_keep(runtime):
                    model_kwargs["logits_to_keep"] = 1

                try:
                    outputs = runtime.model(**model_kwargs)
                    logits = outputs.logits
                    if logits.size(1) == 1:
                        next_token_logits = logits[:, 0, :]
                    elif getattr(runtime.tokenizer, "padding_side", None) == "left":
                        next_token_logits = logits[:, -1, :]
                    else:
                        last_positions = encoded["attention_mask"].sum(dim=1) - 1
                        row_indices = torch.arange(logits.size(0), device=device)
                        next_token_logits = logits[row_indices, last_positions]
                    next_token_log_probs = F.log_softmax(next_token_logits.float(), dim=-1)
                except RuntimeError as error:
                    if not _is_cuda_oom(error) or current_batch_size == 1:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    continue
                break

            option_a_log_probs = next_token_log_probs[:, runtime.option_a_token_id]
            option_b_log_probs = next_token_log_probs[:, runtime.option_b_token_id]
            policy_is_option_a_tensor = torch.tensor(policy_is_option_a, device=device, dtype=torch.bool)
            policy_log_probs = torch.where(policy_is_option_a_tensor, option_a_log_probs, option_b_log_probs)
            baseline_log_probs = torch.where(policy_is_option_a_tensor, option_b_log_probs, option_a_log_probs)
            valid_mask = torch.isfinite(policy_log_probs) & torch.isfinite(baseline_log_probs)
            invalid_count = int((~valid_mask).sum().item())
            invalids += invalid_count
            if invalid_count:
                raise ValueError(
                    "Local judge produced invalid outcome(s); inspect the judge outputs before continuing."
                )

            valid_policy_log_probs = policy_log_probs[valid_mask]
            valid_baseline_log_probs = baseline_log_probs[valid_mask]
            margins = valid_policy_log_probs - valid_baseline_log_probs
            wins += float((margins > 0).sum().item())
            losses += float((margins < 0).sum().item())
            ties += int((margins == 0).sum().item())
            start += current_batch_size
            progress.update(current_batch_size)
    finally:
        progress.close()

    reduced = reduce_sum([wins, losses, float(ties), float(invalids), float(len(comparisons))])
    return _judge_summary(
        prefix=PAIR_PREFERENCE_MODEL_NAME,
        model_name=eval_config.local_pair_preference_judge_model,
        wins=float(reduced[0]),
        losses=float(reduced[1]),
        ties=int(reduced[2]),
        invalids=int(reduced[3]),
        total_examples=int(reduced[4]),
    )


@torch.inference_mode()
def evaluate_scoring_judge(
    comparisons: list[JudgeComparison],
    eval_config: EvalConfig,
    device: torch.device,
    shared_model_root: str | None = None,
) -> dict[str, float | int | str | None]:
    """Compute policy-vs-SFT win rate using the RLHFlow ArmoRM scoring judge."""

    runtime = _load_scoring_runtime(
        eval_config.local_scoring_judge_model,
        device=device,
        shared_model_root=shared_model_root,
    )
    wins = 0.0
    losses = 0.0
    ties = 0
    invalids = 0
    batch_size = max(1, int(eval_config.local_judge_batch_size))
    progress = _judge_progress(len(comparisons), "judge:scoring")

    message_texts: list[str] = []
    for item in comparisons:
        message_texts.append(
            _messages_to_text(
                runtime.tokenizer,
                [
                    {"role": "user", "content": item.prompt},
                    {"role": "assistant", "content": item.model_answer},
                ],
                add_generation_prompt=False,
            )
        )
        message_texts.append(
            _messages_to_text(
                runtime.tokenizer,
                [
                    {"role": "user", "content": item.prompt},
                    {"role": "assistant", "content": item.baseline_answer},
                ],
                add_generation_prompt=False,
            )
        )

    start = 0
    try:
        while start < len(message_texts):
            current_batch_size = min(batch_size, (len(message_texts) - start) // 2)
            current_batch_size = max(1, current_batch_size)
            while True:
                chunk_texts = message_texts[start : start + 2 * current_batch_size]
                if not chunk_texts:
                    break
                encoded = runtime.tokenizer(chunk_texts, **_scoring_tokenizer_kwargs(eval_config, runtime))
                encoded = {key: value.to(device) for key, value in encoded.items()}
                try:
                    outputs = runtime.model(**encoded, return_dict=True)
                    scores = _scalar_scores_from_outputs(outputs)
                except RuntimeError as error:
                    if not _is_cuda_oom(error) or current_batch_size == 1:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    continue
                break
            if not chunk_texts:
                continue
            model_scores = scores[0::2]
            baseline_scores = scores[1::2]
            valid_mask = torch.isfinite(model_scores) & torch.isfinite(baseline_scores)
            invalid_count = int((~valid_mask).sum().item())
            invalids += invalid_count
            if invalid_count:
                raise ValueError(
                    "Local judge produced invalid outcome(s); inspect the judge outputs before continuing."
                )

            valid_model_scores = model_scores[valid_mask]
            valid_baseline_scores = baseline_scores[valid_mask]
            win_count, loss_count, tie_count = _scalar_judge_outcome_counts(
                valid_model_scores,
                valid_baseline_scores,
            )
            wins += float(win_count)
            losses += float(loss_count)
            ties += tie_count
            start += 2 * current_batch_size
            progress.update(current_batch_size)
    finally:
        progress.close()

    reduced = reduce_sum([wins, losses, float(ties), float(invalids), float(len(comparisons))])
    return _judge_summary(
        prefix=SCORING_MODEL_NAME,
        model_name=eval_config.local_scoring_judge_model,
        wins=float(reduced[0]),
        losses=float(reduced[1]),
        ties=int(reduced[2]),
        invalids=int(reduced[3]),
        total_examples=int(reduced[4]),
    )


def evaluate_local_judge_metrics(
    comparisons: list[JudgeComparison],
    eval_config: EvalConfig,
    device: torch.device,
    shared_model_root: str | None = None,
) -> dict[str, float | int | str | None]:
    """Evaluate all registered local judges and merge their metric blocks."""

    summary: dict[str, float | int | str | None] = {}
    for spec in LOCAL_JUDGE_REGISTRY:
        if not comparisons and not get_distributed_state().enabled:
            summary.update(_empty_judge_summary(spec.name, _local_judge_model_name(spec.name, eval_config)))
            continue
        summary.update(
            spec.evaluator(
                comparisons=comparisons,
                eval_config=eval_config,
                device=device,
                shared_model_root=shared_model_root,
            )
        )
    return summary


def release_local_judge_models() -> None:
    """Release cached judge runtimes between benchmark runs."""

    for runtime in list(_PAIR_RUNTIME_CACHE.values()):
        del runtime.model
        del runtime.tokenizer
    for runtime in list(_SCORING_RUNTIME_CACHE.values()):
        del runtime.model
        del runtime.tokenizer
    _PAIR_RUNTIME_CACHE.clear()
    _SCORING_RUNTIME_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


LOCAL_JUDGE_REGISTRY = (
    LocalJudgeSpec(name=PAIR_PREFERENCE_MODEL_NAME, evaluator=evaluate_pair_preference_judge),
    LocalJudgeSpec(name=SCORING_MODEL_NAME, evaluator=evaluate_scoring_judge),
)
