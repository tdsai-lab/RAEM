"""Shared helpers for external generation-based evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Iterable
import urllib.request

import torch

from private_score_alignment.config.schemas import EvalConfig, ExperimentConfig, ModelConfig
from private_score_alignment.eval.distributed import barrier, gather_objects, get_distributed_state
from private_score_alignment.models.factory import PolicyBundle, load_policy_for_inference


DEFAULT_USER_ROLE = "user"
DEFAULT_ASSISTANT_ROLE = "assistant"
DEFAULT_SYSTEM_ROLE = "system"
DEFAULT_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OPENAI_COMPAT_SITE_CUSTOMIZE = """import os


compat_key = os.environ.get("PSA_OPENAI_COMPAT_API_KEY", "")
compat_base = os.environ.get("PSA_OPENAI_COMPAT_BASE_URL", "")
compat_model = os.environ.get("PSA_OPENAI_COMPAT_MODEL", "")

if compat_key and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = compat_key
if compat_base and not os.environ.get("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = compat_base

try:
    import openai

    if compat_key:
        openai.api_key = compat_key
    if compat_base:
        openai.api_base = compat_base.rstrip("/")
except Exception:
    pass

if compat_model:
    try:
        import fastchat.model.model_adapter as model_adapter

        if compat_model not in model_adapter.OPENAI_MODEL_LIST:
            model_adapter.OPENAI_MODEL_LIST = tuple(model_adapter.OPENAI_MODEL_LIST) + (
                compat_model,
            )
    except Exception:
        pass

try:
    import tiktoken

    _original_encoding_for_model = tiktoken.encoding_for_model

    def _encoding_for_model(model_name):
        try:
            return _original_encoding_for_model(model_name)
        except KeyError:
            if str(model_name).startswith("gemini-"):
                return tiktoken.get_encoding("cl100k_base")
            raise

    tiktoken.encoding_for_model = _encoding_for_model
except Exception:
    pass
"""


@dataclass
class GenerationExample:
    """Single prompt or multi-turn generation request."""

    prompt: str
    history: list[tuple[str, str]] | None = None
    metadata: dict[str, object] | None = None


def resolve_project_path(configured_root: str) -> Path:
    """Resolve an absolute or project-relative path."""

    path = Path(configured_root).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / path).resolve()


def sanitize_model_id(raw: str) -> str:
    """Normalize a model/checkpoint name for filesystem and benchmark IDs."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    normalized = normalized.strip("-._")
    return normalized or "private-score-alignment-model"


def infer_model_id(checkpoint_path: str, explicit_model_id: str = "") -> str:
    """Infer a stable model id from CLI input or the checkpoint directory name."""

    if explicit_model_id:
        return sanitize_model_id(explicit_model_id)
    checkpoint_path_obj = Path(checkpoint_path).expanduser()
    if checkpoint_path.startswith(("/", "./", "../", "~/")):
        checkpoint_dir = checkpoint_path_obj.resolve()
        checkpoint_name = checkpoint_dir.name
        if re.fullmatch(r"checkpoint-\d+", checkpoint_name) and checkpoint_dir.parent.name:
            checkpoint_name = checkpoint_dir.parent.name
    else:
        checkpoint_name = checkpoint_path_obj.name or checkpoint_path
    return sanitize_model_id(checkpoint_name)


def infer_reference_model_id(config: ExperimentConfig) -> str:
    """Infer a stable external-eval id for the configured reference model."""

    return infer_model_id(config.model.reference_model_path)


def is_gemini_model_name(model_name: str) -> bool:
    """Return whether a judge/config string names a Gemini API model."""

    return model_name.strip().lower().startswith("gemini-")


def ensure_local_asset(
    source: str,
    destination: str | Path,
    force_refresh: bool = False,
) -> Path:
    """Ensure a local asset exists, downloading it when given a URL source."""

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    state = get_distributed_state()
    if state.is_main_process and (force_refresh or not destination_path.exists()):
        if source.startswith("http://") or source.startswith("https://"):
            with urllib.request.urlopen(source) as response:
                payload = response.read()
            destination_path.write_bytes(payload)
        else:
            source_path = Path(source).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"Missing evaluation asset: {source_path}")
            destination_path.write_bytes(source_path.read_bytes())
    barrier()
    return destination_path


def load_records(path: str | Path) -> list[dict[str, object]]:
    """Load a JSON or JSONL file into a list of records."""

    source = Path(path).expanduser().resolve()
    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected a list payload in {source}, found {type(payload)!r}.")


def write_json(path: str | Path, payload: object) -> Path:
    """Write a JSON payload to disk."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return destination


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    """Write line-delimited JSON rows to disk."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination


def ensure_external_eval_root(
    config: ExperimentConfig,
    framework_name: str,
    model_id: str,
) -> Path:
    """Return the per-framework output directory under the local output root."""

    output_root = resolve_project_path(config.paths.output_root)
    destination = (
        output_root
        / config.evaluation.external_output_subdir
        / framework_name
        / sanitize_model_id(model_id)
    )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _rank0_status_paths(run_dir: Path, framework_name: str) -> tuple[Path, Path]:
    """Return the success and failure marker paths for a rank0-only judge stage."""

    success = run_dir / f".{framework_name}.rank0.success"
    failure = run_dir / f".{framework_name}.rank0.failure"
    return success, failure


def prepare_rank0_status_files(run_dir: Path, framework_name: str) -> tuple[Path, Path]:
    """Remove stale rank0 completion markers before a new distributed eval run."""

    success, failure = _rank0_status_paths(run_dir, framework_name)
    state = get_distributed_state()
    if state.is_main_process:
        success.unlink(missing_ok=True)
        failure.unlink(missing_ok=True)
    barrier()
    return success, failure


def mark_rank0_eval_success(run_dir: Path, framework_name: str) -> Path:
    """Write a success marker after the rank0-only judge stage completes."""

    success, _ = _rank0_status_paths(run_dir, framework_name)
    success.write_text("ok\n", encoding="utf-8")
    return success


def mark_rank0_eval_failure(run_dir: Path, framework_name: str, message: str) -> Path:
    """Write a failure marker so non-main ranks can exit instead of hanging."""

    _, failure = _rank0_status_paths(run_dir, framework_name)
    failure.write_text(message, encoding="utf-8")
    return failure


def wait_for_rank0_eval_completion(
    run_dir: Path,
    framework_name: str,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Block non-main ranks until rank0 completes or fails its single-process judge step."""

    success, failure = _rank0_status_paths(run_dir, framework_name)
    while True:
        if failure.exists():
            message = failure.read_text(encoding="utf-8").strip()
            raise RuntimeError(
                f"{framework_name} rank0 stage failed on the main process.\n{message}"
            )
        if success.exists():
            return
        time.sleep(poll_interval_seconds)


def require_openai_api_key(reason: str) -> None:
    """Fail fast when an OpenAI-backed external judge is configured without credentials."""

    if os.environ.get("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        f"{reason} requires OPENAI_API_KEY in the environment. "
        "Export it before launching `torchrun` or `nohup`."
    )


def resolve_gemini_api_key() -> str | None:
    """Return a configured Gemini API key from the supported environment variables."""

    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def require_gemini_api_key(reason: str) -> None:
    """Fail fast when a Gemini-backed external judge is configured without credentials."""

    if resolve_gemini_api_key():
        return
    raise RuntimeError(
        f"{reason} requires GEMINI_API_KEY in the environment. "
        "Export GEMINI_API_KEY (or GOOGLE_API_KEY) before launching `torchrun` or `nohup`."
    )


def build_gemini_openai_compatible_env(
    *,
    run_dir: Path,
    judge_model: str,
) -> dict[str, str]:
    """Build subprocess env vars for Gemini's OpenAI-compatible endpoint."""

    api_key = resolve_gemini_api_key()
    if api_key is None:
        raise RuntimeError(
            f"Gemini judge_model='{judge_model}' requires GEMINI_API_KEY in the environment."
        )

    compat_dir = (run_dir / ".python_startup").resolve()
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(
        _OPENAI_COMPAT_SITE_CUSTOMIZE,
        encoding="utf-8",
    )

    env = os.environ.copy()
    base_url = env.get("GEMINI_OPENAI_BASE_URL", DEFAULT_GEMINI_OPENAI_BASE_URL)
    env["GEMINI_API_KEY"] = api_key
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_API_BASE"] = base_url
    env["PSA_OPENAI_COMPAT_API_KEY"] = api_key
    env["PSA_OPENAI_COMPAT_BASE_URL"] = base_url
    env["PSA_OPENAI_COMPAT_MODEL"] = judge_model
    pythonpath_entries = [str(compat_dir)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def validate_external_eval_prerequisites(
    config: ExperimentConfig,
    frameworks: Iterable[str],
    skip_api_check: bool = True,
) -> None:
    """Validate framework prerequisites before loading large model weights."""

    if skip_api_check:
        return

    selected = set(frameworks)
    if (
        "alpaca_eval" in selected
        and not getattr(config.evaluation, "alpaca_eval_prepare_only", False)
        and is_gemini_model_name(config.evaluation.alpaca_eval_annotators_config)
    ):
        require_gemini_api_key(
            f"AlpacaEval annotators_config='{config.evaluation.alpaca_eval_annotators_config}'"
        )
    if (
        "alpaca_eval" in selected
        and not getattr(config.evaluation, "alpaca_eval_prepare_only", False)
        and "gpt" in config.evaluation.alpaca_eval_annotators_config.lower()
    ):
        require_openai_api_key(
            f"AlpacaEval annotators_config='{config.evaluation.alpaca_eval_annotators_config}'"
        )
    if (
        "mt_bench" in selected
        and is_gemini_model_name(config.evaluation.mt_bench_judge_model)
    ):
        require_gemini_api_key(
            f"MT-Bench judge_model='{config.evaluation.mt_bench_judge_model}'"
        )
    if (
        "mt_bench" in selected
        and config.evaluation.mt_bench_judge_model.lower().startswith("gpt-")
    ):
        require_openai_api_key(
            f"MT-Bench judge_model='{config.evaluation.mt_bench_judge_model}'"
        )


def resolve_generation_max_new_tokens(
    eval_config: EvalConfig,
    framework_name: str,
) -> int:
    """Resolve the generation length cap for a specific external evaluation framework."""

    override_fields = {
        "alpaca_eval": "alpaca_eval_generation_max_new_tokens",
        "mt_bench": "mt_bench_generation_max_new_tokens",
    }
    field_name = override_fields.get(framework_name)
    if field_name is not None:
        override_value = int(getattr(eval_config, field_name, 0) or 0)
        if override_value > 0:
            return override_value
    return int(eval_config.generation_max_new_tokens)


def build_messages(
    prompt: str,
    model_config: ModelConfig,
    history: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat-style messages for a prompt with optional turn history."""

    messages: list[dict[str, str]] = []
    if model_config.system_prompt:
        messages.append({"role": DEFAULT_SYSTEM_ROLE, "content": model_config.system_prompt})
    for user_turn, assistant_turn in history or []:
        messages.append({"role": DEFAULT_USER_ROLE, "content": user_turn})
        messages.append({"role": DEFAULT_ASSISTANT_ROLE, "content": assistant_turn})
    messages.append({"role": DEFAULT_USER_ROLE, "content": prompt})
    return messages


def render_generation_prompt(
    tokenizer,
    model_config: ModelConfig,
    prompt: str,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, bool]:
    """Render a prompt string suitable for tokenizer encoding."""

    messages = build_messages(prompt=prompt, model_config=model_config, history=history)
    if (
        model_config.use_chat_template
        and getattr(tokenizer, "chat_template", None)
        and hasattr(tokenizer, "apply_chat_template")
    ):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text, False

    parts: list[str] = []
    if model_config.system_prompt:
        parts.append(f"System: {model_config.system_prompt}")
    for message in messages:
        role = message["role"].capitalize()
        parts.append(f"{role}: {message['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts), True


def load_policy_bundle_for_generation(
    checkpoint_path: str,
    config: ExperimentConfig,
) -> PolicyBundle:
    """Load a policy-only bundle for external generation-based evaluation."""

    return load_policy_for_inference(
        checkpoint_path=checkpoint_path,
        model_config=config.model,
        training_config=config.training,
    )


def offload_policy_bundle(bundle: PolicyBundle | None) -> None:
    """Move an inference bundle off accelerator memory between evaluation stages."""

    if bundle is None:
        return
    policy_model = getattr(bundle, "policy_model", None)
    if policy_model is None:
        return
    policy_model.to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_batch_encoding(
    tokenizer,
    prompts: list[str],
    add_special_tokens: bool,
    model_config: ModelConfig,
    truncate_prompts: bool = True,
) -> dict[str, torch.Tensor]:
    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": truncate_prompts,
        "add_special_tokens": add_special_tokens,
    }
    if truncate_prompts:
        tokenizer_kwargs["max_length"] = model_config.max_prompt_length
        if original_truncation_side is not None:
            # Generation prompts are rendered before tokenization.  Left
            # truncation preserves the final assistant marker instead of
            # leaving the model to continue a half-cut user prompt.
            tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(prompts, **tokenizer_kwargs)
    finally:
        if original_truncation_side is not None:
            tokenizer.truncation_side = original_truncation_side
    return encoded


@torch.inference_mode()
def generate_completions(
    bundle: PolicyBundle,
    model_config: ModelConfig,
    eval_config: EvalConfig,
    examples: list[GenerationExample],
    device: torch.device | None = None,
    max_new_tokens: int | None = None,
    truncate_prompts: bool = True,
) -> list[str]:
    """Generate completions for prompts or multi-turn chat states."""

    if not examples:
        return []

    policy_model = bundle.policy_model
    tokenizer = bundle.tokenizer
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(target_device)
    policy_model.eval()

    responses: list[str] = []
    batch_size = max(1, int(eval_config.generation_batch_size))
    do_sample = bool(eval_config.generation_do_sample or eval_config.generation_temperature > 0)
    resolved_max_new_tokens = (
        int(max_new_tokens) if max_new_tokens is not None else int(eval_config.generation_max_new_tokens)
    )
    generation_kwargs = {
        "max_new_tokens": resolved_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "use_cache": True,
        "synced_gpus": False,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(eval_config.generation_temperature)
        generation_kwargs["top_p"] = float(eval_config.generation_top_p)

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            rendered = [
                render_generation_prompt(
                    tokenizer=tokenizer,
                    model_config=model_config,
                    prompt=example.prompt,
                    history=example.history,
                )
                for example in chunk
            ]
            prompt_texts = [item[0] for item in rendered]
            add_special_tokens = rendered[0][1]
            encoded = _prepare_batch_encoding(
                tokenizer=tokenizer,
                prompts=prompt_texts,
                add_special_tokens=add_special_tokens,
                model_config=model_config,
                truncate_prompts=truncate_prompts,
            )
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            output_ids = policy_model.generate(**encoded, **generation_kwargs)
            prompt_width = int(encoded["input_ids"].shape[1])
            for sequence in output_ids:
                # The tokenizer is left-padded for batched decoding, so slicing
                # by each row's non-pad token count would leak the prompt tail
                # back into the decoded completion for shorter rows.
                generated_ids = sequence[prompt_width:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                responses.append(text)
    finally:
        tokenizer.padding_side = original_padding_side

    return responses


def generate_completions_distributed(
    bundle: PolicyBundle,
    model_config: ModelConfig,
    eval_config: EvalConfig,
    examples: list[GenerationExample],
    max_new_tokens: int | None = None,
) -> list[str]:
    """Generate completions in parallel across ranks and merge them on rank 0."""

    state = get_distributed_state()
    indexed_examples = list(enumerate(examples))
    local_examples = indexed_examples[state.rank :: state.world_size]
    local_outputs = generate_completions(
        bundle=bundle,
        model_config=model_config,
        eval_config=eval_config,
        examples=[example for _, example in local_examples],
        device=state.device,
        max_new_tokens=max_new_tokens,
    )
    local_rows = [
        {"index": index, "output": output}
        for (index, _), output in zip(local_examples, local_outputs)
    ]
    gathered_rows = gather_objects(local_rows)
    merged = {}
    for rows in gathered_rows:
        for row in rows:
            merged[int(row["index"])] = str(row["output"])
    ordered_outputs = [merged[index] for index in range(len(examples))]
    return ordered_outputs
