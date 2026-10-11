"""AlpacaEval integration for Private Score Alignment checkpoints."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from private_score_alignment.config.schemas import ExperimentConfig
from private_score_alignment.eval.distributed import get_distributed_state
from private_score_alignment.eval.external_common import (
    GenerationExample,
    ensure_external_eval_root,
    ensure_local_asset,
    infer_reference_model_id,
    generate_completions_distributed,
    infer_model_id,
    is_gemini_model_name,
    load_policy_bundle_for_generation,
    load_records,
    mark_rank0_eval_failure,
    mark_rank0_eval_success,
    offload_policy_bundle,
    prepare_rank0_status_files,
    resolve_generation_max_new_tokens,
    wait_for_rank0_eval_completion,
    write_json,
)
from private_score_alignment.models.factory import PolicyBundle

_ALPACA_EVAL_INVALID_JSON_SIGNATURE = "We could not parse the JSON body of your request."
_ALPACA_EVAL_MODEL_NOT_FOUND_SIGNATURES = (
    "does not exist or you do not have access to it",
    "model_not_found",
)
_ALPACA_EVAL_ANNOTATOR_FALLBACKS = {
    "weighted_alpaca_eval_gpt4_turbo": "weighted_alpaca_eval_gpt4_turbo_new",
}
_GEMINI_ALPACA_EVAL_PROMPT = '''<|im_start|>system
You are a careful evaluator that ranks model answers by likely human preference.
<|im_end|>
<|im_start|>user
You will receive one instruction and two candidate answers.
Rank the answers by which one humans would prefer overall for helpfulness, correctness, completeness, and clarity.

Return only a valid Python list in exactly this format:
[
    {'model': 'model_1', 'rank': 1},
    {'model': 'model_2', 'rank': 2}
]
Use rank 1 for the better answer and rank 2 for the worse answer.

Instruction:
{
    "instruction": """{instruction}""",
}

Candidate answers:
[
    {
        "model": "model_1",
        "answer": """{output_1}"""
    },
    {
        "model": "model_2",
        "answer": """{output_2}"""
    }
]

Do not add any explanation or extra text outside the Python list.
<|im_end|>
'''


def _extract_instruction(record: dict[str, Any]) -> str:
    """Extract a single instruction string from an AlpacaEval example."""

    if isinstance(record.get("instruction"), str) and record["instruction"].strip():
        instruction = record["instruction"].strip()
        input_text = str(record.get("input", "") or "").strip()
        if input_text:
            return f"{instruction}\n\n{input_text}"
        return instruction
    if isinstance(record.get("prompt"), str) and record["prompt"].strip():
        return record["prompt"].strip()
    raise ValueError(f"Cannot extract instruction from record keys: {sorted(record.keys())}")


def build_alpaca_eval_model_outputs(
    records: list[dict[str, Any]],
    outputs: list[str],
    model_id: str,
) -> list[dict[str, Any]]:
    """Convert generated responses into AlpacaEval's model_outputs format."""

    if len(records) != len(outputs):
        raise ValueError(
            f"AlpacaEval record/output mismatch: {len(records)} records vs {len(outputs)} outputs."
        )

    rows: list[dict[str, Any]] = []
    for index, (record, output_text) in enumerate(zip(records, outputs)):
        rows.append(
            {
                "instruction": _extract_instruction(record),
                "output": output_text,
                "generator": model_id,
                "dataset": str(record.get("dataset", "alpaca_eval")),
                "example_id": record.get("example_id", record.get("id", index)),
            }
        )
    return rows


def _infer_asset_filename(source: str, fallback_name: str) -> str:
    """Infer a stable local filename for a local path or URL asset."""

    parsed = urlparse(source)
    if parsed.scheme and parsed.path:
        candidate = Path(parsed.path).name
    else:
        candidate = Path(source).name
    return candidate or fallback_name


def _coerce_row_values(row: dict[str, str]) -> dict[str, Any]:
    """Convert numeric CSV fields when possible."""

    converted: dict[str, Any] = {}
    for key, value in row.items():
        text = value.strip()
        if not text:
            converted[key] = text
            continue
        try:
            if "." in text or "e" in text.lower():
                converted[key] = float(text)
            else:
                converted[key] = int(text)
        except ValueError:
            converted[key] = text
    return converted


def _parse_alpaca_eval_leaderboard(run_dir: Path, model_id: str) -> dict[str, Any]:
    """Parse the structured leaderboard output saved by AlpacaEval."""

    csv_candidates = sorted(run_dir.rglob("leaderboard*.csv"))
    for candidate in csv_candidates:
        with candidate.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        for row in rows:
            normalized = {key.lower(): str(value) for key, value in row.items()}
            if model_id in normalized.values():
                parsed = _coerce_row_values(row)
                parsed["leaderboard_path"] = str(candidate)
                return parsed
        parsed = _coerce_row_values(rows[0])
        parsed["leaderboard_path"] = str(candidate)
        return parsed

    json_candidates = sorted(run_dir.rglob("leaderboard*.json"))
    for candidate in json_candidates:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list) and payload:
            row = payload[0]
            if isinstance(row, dict):
                row = dict(row)
                row["leaderboard_path"] = str(candidate)
                return row
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["leaderboard_path"] = str(candidate)
            return payload
    return {}


def _resolve_retry_annotators_config(
    annotators_config: str,
    *,
    stdout: str,
    stderr: str,
) -> str | None:
    """Choose a fallback annotator config for known API compatibility failures."""

    fallback = _ALPACA_EVAL_ANNOTATOR_FALLBACKS.get(annotators_config)
    if fallback is None:
        return None
    combined = f"{stdout}\n{stderr}"
    if _ALPACA_EVAL_INVALID_JSON_SIGNATURE in combined:
        return fallback
    if any(signature in combined for signature in _ALPACA_EVAL_MODEL_NOT_FOUND_SIGNATURES):
        return fallback
    return None


def _materialize_gemini_alpaca_eval_config(
    *,
    run_dir: Path,
    judge_model: str,
) -> Path:
    """Write a local AlpacaEval annotator config for a Gemini judge model."""

    config_dir = (run_dir / "alpaca_eval_gemini_judge").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = config_dir / "prompt.txt"
    prompt_path.write_text(_GEMINI_ALPACA_EVAL_PROMPT, encoding="utf-8")
    config_path = write_json(
        config_dir / "configs.json",
        {
            "gemini_judge": {
                "prompt_template": str(prompt_path),
                "fn_completions": "openai_completions",
                "completions_kwargs": {
                    "model_name": judge_model,
                    "max_tokens": 120,
                    "temperature": 0,
                    "num_procs": 1,
                },
                "fn_completion_parser": "ranking_parser",
                "batch_size": 1,
            }
        },
    )
    return config_path


def _resolve_alpaca_eval_annotators_config(
    *,
    requested_annotators_config: str,
    run_dir: Path,
) -> tuple[str, str | None]:
    """Resolve the CLI annotator config path and the effective judge model."""

    if is_gemini_model_name(requested_annotators_config):
        return (
            str(
                _materialize_gemini_alpaca_eval_config(
                    run_dir=run_dir,
                    judge_model=requested_annotators_config,
                )
            ),
            requested_annotators_config,
        )
    return requested_annotators_config, None


def _prepare_reference_outputs(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    dataset_records: list[dict[str, Any]],
    generation_examples: list[GenerationExample],
    candidate_bundle: PolicyBundle,
    generation_max_new_tokens: int,
) -> tuple[Path, str]:
    """Resolve AlpacaEval reference outputs, preferring the configured base model."""

    assets_dir = run_dir / "assets"
    if config.evaluation.alpaca_eval_reference_outputs:
        reference_outputs_path = ensure_local_asset(
            config.evaluation.alpaca_eval_reference_outputs,
            assets_dir
            / _infer_asset_filename(
                config.evaluation.alpaca_eval_reference_outputs,
                "alpaca_eval_reference_outputs.json",
            ),
        )
        reference_model_id = infer_model_id(
            str(reference_outputs_path),
            explicit_model_id=Path(reference_outputs_path).stem,
        )
        return reference_outputs_path, reference_model_id

    cached_reference_outputs_path = run_dir / "reference_model_outputs.json"
    if cached_reference_outputs_path.exists():
        reference_model_id = infer_reference_model_id(config)
        cached_rows = load_records(cached_reference_outputs_path)
        if cached_rows:
            cached_generator = str(cached_rows[0].get("generator", "") or "").strip()
            if cached_generator:
                reference_model_id = cached_generator
        return cached_reference_outputs_path, reference_model_id

    reference_model_id = infer_reference_model_id(config)
    offload_policy_bundle(candidate_bundle)
    reference_bundle = load_policy_bundle_for_generation(
        checkpoint_path=config.model.reference_model_path,
        config=config,
    )
    try:
        reference_outputs = generate_completions_distributed(
            bundle=reference_bundle,
            model_config=config.model,
            eval_config=config.evaluation,
            examples=generation_examples,
            max_new_tokens=generation_max_new_tokens,
        )
    finally:
        offload_policy_bundle(reference_bundle)
    reference_rows = build_alpaca_eval_model_outputs(
        records=dataset_records,
        outputs=reference_outputs,
        model_id=reference_model_id,
    )
    reference_outputs_path = write_json(run_dir / "reference_model_outputs.json", reference_rows)
    return reference_outputs_path, reference_model_id


def _load_cached_output_metadata(
    output_path: Path,
    fallback_model_id: str,
) -> tuple[Path, str, int]:
    """Return a cached output path, generator id, and row count."""

    rows = load_records(output_path)
    resolved_model_id = fallback_model_id
    if rows:
        cached_generator = str(rows[0].get("generator", "") or "").strip()
        if cached_generator:
            resolved_model_id = cached_generator
    return output_path, resolved_model_id, len(rows)


def _run_alpaca_eval_request(
    *,
    run_dir: Path,
    model_outputs_path: Path,
    reference_outputs_path: Path,
    annotators_config: str,
    model_id: str,
) -> dict[str, Any]:
    """Run AlpacaEval's pairwise judge on prepared outputs."""

    from alpaca_eval import evaluate

    requested_annotators_config = annotators_config
    resolved_annotators_config, effective_judge_model = _resolve_alpaca_eval_annotators_config(
        requested_annotators_config=requested_annotators_config,
        run_dir=run_dir,
    )
    annotations = None

    try:
        _, annotations = evaluate(
            model_outputs=str(model_outputs_path),
            reference_outputs=str(reference_outputs_path),
            annotators_config=resolved_annotators_config,
            name=model_id,
            output_path=run_dir,
            precomputed_leaderboard=None,
            is_overwrite_leaderboard=True,
            leaderboard_mode_to_print="minimal",
            current_leaderboard_mode="community",
            is_return_instead_of_print=True,
        )
    except Exception as error:
        fallback_annotators_config = _resolve_retry_annotators_config(
            requested_annotators_config,
            stdout="",
            stderr=str(error),
        )
        if fallback_annotators_config is None:
            raise
        logging.warning(
            "AlpacaEval failed with annotators_config=%s; retrying with %s.",
            requested_annotators_config,
            fallback_annotators_config,
        )
        resolved_annotators_config, effective_judge_model = _resolve_alpaca_eval_annotators_config(
            requested_annotators_config=fallback_annotators_config,
            run_dir=run_dir,
        )
        _, annotations = evaluate(
            model_outputs=str(model_outputs_path),
            reference_outputs=str(reference_outputs_path),
            annotators_config=resolved_annotators_config,
            name=model_id,
            output_path=run_dir,
            precomputed_leaderboard=None,
            is_overwrite_leaderboard=True,
            leaderboard_mode_to_print="minimal",
            current_leaderboard_mode="community",
            is_return_instead_of_print=True,
        )

    result = {
        "framework": "alpaca_eval",
        "mode": "evaluate",
        "model_id": model_id,
        "annotators_config": resolved_annotators_config,
        "judge_model": effective_judge_model or requested_annotators_config,
        "model_outputs_path": str(model_outputs_path),
        "reference_outputs_path": str(reference_outputs_path),
        "run_dir": str(run_dir),
    }
    if annotations is not None:
        result["num_annotations"] = int(len(annotations))
    result.update(_parse_alpaca_eval_leaderboard(run_dir=run_dir, model_id=model_id))
    return result


def run_alpaca_eval(
    config: ExperimentConfig,
    checkpoint_path: str,
    model_id: str = "",
    bundle: PolicyBundle | None = None,
) -> dict[str, Any]:
    """Run AlpacaEval using cached or freshly generated model/reference outputs."""

    state = get_distributed_state()
    resolved_model_id = infer_model_id(checkpoint_path, explicit_model_id=model_id)
    run_dir = ensure_external_eval_root(config, "alpaca_eval", resolved_model_id)
    prepare_rank0_status_files(run_dir, "alpaca_eval")
    prepare_only = bool(getattr(config.evaluation, "alpaca_eval_prepare_only", False))
    cached_model_outputs_path = run_dir / "model_outputs.json"
    cached_reference_outputs_path = run_dir / "reference_model_outputs.json"
    if cached_model_outputs_path.exists() and cached_reference_outputs_path.exists():
        model_outputs_path, cached_model_id, num_examples = _load_cached_output_metadata(
            cached_model_outputs_path,
            fallback_model_id=resolved_model_id,
        )
        reference_outputs_path, reference_model_id, _ = _load_cached_output_metadata(
            cached_reference_outputs_path,
            fallback_model_id=infer_reference_model_id(config),
        )
        if not state.is_main_process and not prepare_only:
            wait_for_rank0_eval_completion(run_dir, "alpaca_eval")
            return {
                "framework": "alpaca_eval",
                "model_id": cached_model_id,
                "worker_rank": state.rank,
                "mode": "evaluate",
                "cache_hit": True,
            }
        try:
            if prepare_only:
                result = {
                    "framework": "alpaca_eval",
                    "mode": "prepare_only",
                    "cache_hit": True,
                    "model_id": cached_model_id,
                    "reference_model_id": reference_model_id,
                    "num_examples": num_examples,
                    "model_outputs_path": str(model_outputs_path),
                    "reference_outputs_path": str(reference_outputs_path),
                    "run_dir": str(run_dir),
                }
            else:
                result = _run_alpaca_eval_request(
                    run_dir=run_dir,
                    model_outputs_path=model_outputs_path,
                    reference_outputs_path=reference_outputs_path,
                    annotators_config=config.evaluation.alpaca_eval_annotators_config,
                    model_id=cached_model_id,
                )
                result.update(
                    {
                        "cache_hit": True,
                        "reference_model_id": reference_model_id,
                        "num_examples": num_examples,
                    }
                )
            if state.is_main_process:
                write_json(run_dir / "summary.json", result)
                mark_rank0_eval_success(run_dir, "alpaca_eval")
            return result
        except Exception as error:
            if state.is_main_process:
                mark_rank0_eval_failure(run_dir, "alpaca_eval", str(error))
            raise
    assets_dir = run_dir / "assets"
    dataset_path = ensure_local_asset(
        config.evaluation.alpaca_eval_dataset_path,
        assets_dir / _infer_asset_filename(
            config.evaluation.alpaca_eval_dataset_path,
            "alpaca_eval.json",
        ),
    )
    dataset_records = load_records(dataset_path)
    generation_examples = [
        GenerationExample(prompt=_extract_instruction(record), metadata={"index": index})
        for index, record in enumerate(dataset_records)
    ]
    generation_max_new_tokens = resolve_generation_max_new_tokens(
        config.evaluation,
        "alpaca_eval",
    )
    policy_bundle = bundle or load_policy_bundle_for_generation(
        checkpoint_path=checkpoint_path,
        config=config,
    )
    reference_outputs_path, reference_model_id = _prepare_reference_outputs(
        config=config,
        run_dir=run_dir,
        dataset_records=dataset_records,
        generation_examples=generation_examples,
        candidate_bundle=policy_bundle,
        generation_max_new_tokens=generation_max_new_tokens,
    )
    outputs = generate_completions_distributed(
        bundle=policy_bundle,
        model_config=config.model,
        eval_config=config.evaluation,
        examples=generation_examples,
        max_new_tokens=generation_max_new_tokens,
    )
    if not state.is_main_process:
        wait_for_rank0_eval_completion(run_dir, "alpaca_eval")
        return {
            "framework": "alpaca_eval",
            "model_id": resolved_model_id,
            "worker_rank": state.rank,
            "mode": "prepare_only" if prepare_only else "evaluate",
        }
    try:
        model_outputs = build_alpaca_eval_model_outputs(
            records=dataset_records,
            outputs=outputs,
            model_id=resolved_model_id,
        )
        model_outputs_path = write_json(run_dir / "model_outputs.json", model_outputs)
        if prepare_only:
            result = {
                "framework": "alpaca_eval",
                "mode": "prepare_only",
                "model_id": resolved_model_id,
                "reference_model_id": reference_model_id,
                "num_examples": len(model_outputs),
                "model_outputs_path": str(model_outputs_path),
                "reference_outputs_path": str(reference_outputs_path),
                "run_dir": str(run_dir),
            }
        else:
            result = _run_alpaca_eval_request(
                run_dir=run_dir,
                model_outputs_path=model_outputs_path,
                reference_outputs_path=reference_outputs_path,
                annotators_config=config.evaluation.alpaca_eval_annotators_config,
                model_id=resolved_model_id,
            )
            result.update(
                {
                    "reference_model_id": reference_model_id,
                    "num_examples": len(model_outputs),
                }
            )
        write_json(run_dir / "summary.json", result)
        mark_rank0_eval_success(run_dir, "alpaca_eval")
        return result
    except Exception as error:
        mark_rank0_eval_failure(run_dir, "alpaca_eval", str(error))
        raise
