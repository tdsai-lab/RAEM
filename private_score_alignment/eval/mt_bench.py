"""MT-Bench integration built around FastChat's official judge."""

from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

from private_score_alignment.config.schemas import ExperimentConfig
from private_score_alignment.eval.distributed import barrier, gather_objects, get_distributed_state
from private_score_alignment.eval.external_common import (
    GenerationExample,
    build_gemini_openai_compatible_env,
    ensure_external_eval_root,
    ensure_local_asset,
    infer_reference_model_id,
    generate_completions,
    infer_model_id,
    is_gemini_model_name,
    load_policy_bundle_for_generation,
    load_records,
    mark_rank0_eval_failure,
    mark_rank0_eval_success,
    offload_policy_bundle,
    prepare_rank0_status_files,
    require_gemini_api_key,
    require_openai_api_key,
    resolve_generation_max_new_tokens,
    wait_for_rank0_eval_completion,
    write_json,
    write_jsonl,
)
from private_score_alignment.models.factory import PolicyBundle


def _ensure_valid_records_asset(
    source: str,
    destination: str | Path,
    label: str,
) -> Path:
    """Ensure a cached JSON/JSONL asset can actually be parsed, refreshing when needed."""

    path = ensure_local_asset(source, destination)
    try:
        load_records(path)
        return path
    except (json.JSONDecodeError, ValueError) as first_error:
        state = get_distributed_state()
        if state.is_main_process and path.exists():
            path.unlink()
        barrier()
        path = ensure_local_asset(source, destination, force_refresh=True)
        try:
            load_records(path)
            return path
        except (json.JSONDecodeError, ValueError) as second_error:
            raise RuntimeError(
                f"MT-Bench {label} asset at {path} is invalid even after refresh."
            ) from second_error


def build_mt_bench_answer_records(
    questions: list[dict[str, Any]],
    outputs_by_question_id: dict[str, list[str]],
    model_id: str,
) -> list[dict[str, Any]]:
    """Convert generated turns into FastChat's model_answer JSONL format."""

    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question["question_id"])
        turns = outputs_by_question_id[question_id]
        rows.append(
            {
                "question_id": question["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": [{"index": 0, "turns": turns}],
                "tstamp": time.time(),
            }
        )
    return rows


def summarize_mt_bench_single_scores(judgment_rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    """Aggregate single-answer MT-Bench judgments into summary metrics."""

    selected = [
        row for row in judgment_rows if row.get("model") == model_id and float(row.get("score", -1)) >= 0
    ]
    if not selected:
        return {
            "model_id": model_id,
            "num_judgments": 0,
            "first_turn_score": 0.0,
            "second_turn_score": 0.0,
            "average_score": 0.0,
        }

    by_turn: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        by_turn[int(row["turn"])].append(float(row["score"]))
    first_turn = by_turn.get(1, [])
    second_turn = by_turn.get(2, [])
    all_scores = [float(row["score"]) for row in selected]
    return {
        "model_id": model_id,
        "num_judgments": len(selected),
        "first_turn_score": sum(first_turn) / max(len(first_turn), 1),
        "second_turn_score": sum(second_turn) / max(len(second_turn), 1),
        "average_score": sum(all_scores) / max(len(all_scores), 1),
    }


def summarize_mt_bench_pairwise(judgment_rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    """Aggregate pairwise MT-Bench judgments into adjusted win-rate metrics."""

    wins = 0.0
    losses = 0.0
    ties = 0.0
    matches = 0
    for row in judgment_rows:
        if row.get("g1_winner") == "error" or row.get("g2_winner") == "error":
            continue
        if model_id not in {row.get("model_1"), row.get("model_2")}:
            continue
        matches += 1
        if row.get("g1_winner") == "tie" or row.get("g1_winner") != row.get("g2_winner"):
            ties += 1.0
            continue
        winner = row.get("model_1") if row.get("g1_winner") == "model_1" else row.get("model_2")
        if winner == model_id:
            wins += 1.0
        else:
            losses += 1.0

    total = wins + losses + ties
    return {
        "model_id": model_id,
        "num_matches": matches,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_adjusted": (wins + 0.5 * ties) / max(total, 1.0),
    }


def _expected_mt_bench_turn_count(questions: list[dict[str, Any]]) -> int:
    """Return the total number of judged turns expected for MT-Bench single mode."""

    return sum(len(question.get("turns", [])) for question in questions)


def _latest_mt_bench_single_rows(
    judgment_rows: list[dict[str, Any]],
    model_id: str,
) -> list[dict[str, Any]]:
    """Select the latest judgment row for each ``(question_id, turn)`` pair."""

    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in judgment_rows:
        if row.get("model") != model_id:
            continue
        key = (int(row["question_id"]), int(row["turn"]))
        previous = latest.get(key)
        current_tstamp = float(row.get("tstamp", 0.0) or 0.0)
        previous_tstamp = float(previous.get("tstamp", 0.0) or 0.0) if previous else float("-inf")
        if previous is None or current_tstamp >= previous_tstamp:
            latest[key] = row
    return [latest[key] for key in sorted(latest)]


def _validate_mt_bench_single_rows(
    judgment_rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    model_id: str,
    judgment_file: Path,
) -> list[dict[str, Any]]:
    """Validate MT-Bench single-mode coverage and fail fast on judge errors."""

    selected_rows = _latest_mt_bench_single_rows(judgment_rows, model_id=model_id)
    expected_turns = _expected_mt_bench_turn_count(questions)
    if len(selected_rows) != expected_turns:
        raise RuntimeError(
            f"MT-Bench judgment coverage mismatch for model '{model_id}': "
            f"expected {expected_turns} judged turns but found {len(selected_rows)} latest rows "
            f"in {judgment_file}."
        )
    error_rows = [
        row
        for row in selected_rows
        if float(row.get("score", -1)) < 0 or row.get("judgment") == "$ERROR$"
    ]
    if error_rows:
        sample = error_rows[0]
        raise RuntimeError(
            f"MT-Bench produced {len(error_rows)} error judgments for model '{model_id}'. "
            f"First error: question_id={sample['question_id']}, turn={sample['turn']}, "
            f"score={sample.get('score')}, judgment={sample.get('judgment')!r}. "
            f"See {judgment_file}."
        )
    return selected_rows


def _clear_mt_bench_previous_outputs(
    run_dir: Path,
    mt_bench_dir: Path,
    judge_model: str,
    judge_mode: str,
) -> None:
    """Remove stale MT-Bench outputs before re-running the judge in the same run directory."""

    suffix = "single" if judge_mode == "single" else "pair"
    stale_paths = [
        mt_bench_dir / "model_judgment" / f"{judge_model}_{suffix}.jsonl",
        run_dir / "summary.json",
    ]
    for path in stale_paths:
        path.unlink(missing_ok=True)


def _resolve_mt_bench_baseline_model_id(config: ExperimentConfig) -> str:
    """Choose the FastChat baseline model id for pairwise-baseline judgments."""

    if config.evaluation.mt_bench_baseline_answer_path:
        if config.evaluation.mt_bench_baseline_model:
            return infer_model_id(
                config.evaluation.mt_bench_baseline_model,
                explicit_model_id=config.evaluation.mt_bench_baseline_model,
            )
        return infer_model_id(
            config.evaluation.mt_bench_baseline_answer_path,
            explicit_model_id=Path(config.evaluation.mt_bench_baseline_answer_path).stem,
        )
    return infer_reference_model_id(config)


def _generate_local_mt_bench_answer_rows(
    *,
    questions: list[dict[str, Any]],
    bundle: PolicyBundle,
    config: ExperimentConfig,
    model_id: str,
    rank: int,
    world_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate FastChat-format answer rows for one model across the local rank shard."""

    local_rows: list[dict[str, Any]] = []
    for question in questions[rank::world_size]:
        history: list[tuple[str, str]] = []
        generated_turns: list[str] = []
        for turn in question["turns"]:
            response = generate_completions(
                bundle=bundle,
                model_config=config.model,
                eval_config=config.evaluation,
                examples=[GenerationExample(prompt=str(turn), history=list(history))],
                max_new_tokens=max_new_tokens,
            )[0]
            generated_turns.append(response)
            history.append((str(turn), response))
        local_rows.extend(
            build_mt_bench_answer_records(
                questions=[question],
                outputs_by_question_id={str(question["question_id"]): generated_turns},
                model_id=model_id,
            )
        )
    return local_rows


def _prepare_fastchat_workspace(
    config: ExperimentConfig,
    checkpoint_path: str,
    model_id: str,
) -> tuple[Path, Path, Path]:
    """Prepare the workspace layout expected by FastChat's judge."""

    run_dir = ensure_external_eval_root(config, "mt_bench", model_id)
    data_dir = run_dir / "data"
    mt_bench_dir = data_dir / "mt_bench"
    reference_dir = mt_bench_dir / "reference_answer"
    question_path = _ensure_valid_records_asset(
        config.evaluation.mt_bench_question_path,
        mt_bench_dir / "question.jsonl",
        label="question",
    )
    _ensure_valid_records_asset(
        config.evaluation.mt_bench_judge_prompts_path,
        data_dir / "judge_prompts.jsonl",
        label="judge prompt",
    )
    reference_filename = f"{config.evaluation.mt_bench_judge_model}.jsonl"
    _ensure_valid_records_asset(
        config.evaluation.mt_bench_reference_answer_path,
        reference_dir / reference_filename,
        label="reference answer",
    )
    if config.evaluation.mt_bench_mode == "pairwise-baseline":
        baseline_model_id = _resolve_mt_bench_baseline_model_id(config)
        if config.evaluation.mt_bench_baseline_answer_path:
            ensure_local_asset(
                config.evaluation.mt_bench_baseline_answer_path,
                mt_bench_dir / "model_answer" / f"{baseline_model_id}.jsonl",
            )
    if config.evaluation.mt_bench_mode == "pairwise-all":
        raise ValueError(
            "MT-Bench pairwise-all is not supported by this single-model wrapper. "
            "Use single mode or provide a custom multi-model answer directory."
        )
    return run_dir, mt_bench_dir, question_path


def run_mt_bench(
    config: ExperimentConfig,
    checkpoint_path: str,
    model_id: str = "",
    bundle: PolicyBundle | None = None,
) -> dict[str, Any]:
    """Generate answers for MT-Bench and score them with FastChat's judge."""

    state = get_distributed_state()
    if importlib.util.find_spec("fastchat") is None:
        raise ImportError(
            "FastChat is not installed. Install it with `python -m pip install fschat openai pandas`."
        )
    if is_gemini_model_name(config.evaluation.mt_bench_judge_model):
        require_gemini_api_key(
            f"MT-Bench judge_model='{config.evaluation.mt_bench_judge_model}'"
        )
    elif config.evaluation.mt_bench_judge_model.lower().startswith("gpt-"):
        require_openai_api_key(
            f"MT-Bench judge_model='{config.evaluation.mt_bench_judge_model}'"
        )

    resolved_model_id = infer_model_id(checkpoint_path, explicit_model_id=model_id)
    run_dir, mt_bench_dir, question_path = _prepare_fastchat_workspace(
        config=config,
        checkpoint_path=checkpoint_path,
        model_id=resolved_model_id,
    )
    prepare_rank0_status_files(run_dir, "mt_bench")
    questions = load_records(question_path)
    generation_max_new_tokens = resolve_generation_max_new_tokens(
        config.evaluation,
        "mt_bench",
    )
    policy_bundle = bundle or load_policy_bundle_for_generation(checkpoint_path=checkpoint_path, config=config)
    local_answer_rows = _generate_local_mt_bench_answer_rows(
        questions=questions,
        bundle=policy_bundle,
        config=config,
        model_id=resolved_model_id,
        rank=state.rank,
        world_size=state.world_size,
        max_new_tokens=generation_max_new_tokens,
    )
    local_baseline_rows: list[dict[str, Any]] = []
    baseline_model_id = _resolve_mt_bench_baseline_model_id(config)
    if (
        config.evaluation.mt_bench_mode == "pairwise-baseline"
        and not config.evaluation.mt_bench_baseline_answer_path
    ):
        offload_policy_bundle(policy_bundle)
        baseline_bundle = load_policy_bundle_for_generation(
            checkpoint_path=config.model.reference_model_path,
            config=config,
        )
        try:
            local_baseline_rows = _generate_local_mt_bench_answer_rows(
                questions=questions,
                bundle=baseline_bundle,
                config=config,
                model_id=baseline_model_id,
                rank=state.rank,
                world_size=state.world_size,
                max_new_tokens=generation_max_new_tokens,
            )
        finally:
            offload_policy_bundle(baseline_bundle)

    gathered_rows = gather_objects(local_answer_rows)
    gathered_baseline_rows = gather_objects(local_baseline_rows)
    if not state.is_main_process:
        wait_for_rank0_eval_completion(run_dir, "mt_bench")
        return {
            "framework": "mt_bench",
            "model_id": resolved_model_id,
            "worker_rank": state.rank,
        }
    try:
        _clear_mt_bench_previous_outputs(
            run_dir=run_dir,
            mt_bench_dir=mt_bench_dir,
            judge_model=config.evaluation.mt_bench_judge_model,
            judge_mode=config.evaluation.mt_bench_mode,
        )
        answer_records = []
        for rows in gathered_rows:
            answer_records.extend(rows)
        answer_records.sort(key=lambda row: int(row["question_id"]))
        answer_file = write_jsonl(
            mt_bench_dir / "model_answer" / f"{resolved_model_id}.jsonl",
            answer_records,
        )
        baseline_answer_file: Path | None = None
        if config.evaluation.mt_bench_mode == "pairwise-baseline":
            baseline_answer_file = (
                mt_bench_dir / "model_answer" / f"{baseline_model_id}.jsonl"
            ).resolve()
            if not config.evaluation.mt_bench_baseline_answer_path:
                baseline_records: list[dict[str, Any]] = []
                for rows in gathered_baseline_rows:
                    baseline_records.extend(rows)
                baseline_records.sort(key=lambda row: int(row["question_id"]))
                write_jsonl(baseline_answer_file, baseline_records)

        judge_command = [
            sys.executable,
            "-m",
            "private_score_alignment.eval._fastchat_judge_runner",
            "--bench-name",
            "mt_bench",
            "--judge-file",
            "data/judge_prompts.jsonl",
            "--judge-model",
            config.evaluation.mt_bench_judge_model,
            "--mode",
            config.evaluation.mt_bench_mode,
            "--parallel",
            str(config.evaluation.mt_bench_parallel),
            "--model-list",
            resolved_model_id,
        ]
        if config.evaluation.mt_bench_mode == "pairwise-baseline":
            judge_command.extend(["--baseline-model", baseline_model_id])

        subprocess_env = (
            build_gemini_openai_compatible_env(
                run_dir=run_dir,
                judge_model=config.evaluation.mt_bench_judge_model,
            )
            if is_gemini_model_name(config.evaluation.mt_bench_judge_model)
            else os.environ.copy()
        )
        # The judge subprocess runs with cwd=run_dir (FastChat resolves data/ relative to cwd).
        # Make sure the PSA package is still importable from there so the wrapper module loads.
        project_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = subprocess_env.get("PYTHONPATH", "").strip()
        pythonpath_entries = [project_root]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        subprocess_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        completed = subprocess.run(
            judge_command,
            cwd=run_dir,
            check=False,
            capture_output=True,
            env=subprocess_env,
            text=True,
            input="\n",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "MT-Bench judgment failed.\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )

        suffix = "single" if config.evaluation.mt_bench_mode == "single" else "pair"
        judgment_file = run_dir / "data" / "mt_bench" / "model_judgment" / (
            f"{config.evaluation.mt_bench_judge_model}_{suffix}.jsonl"
        )
        judgment_rows = load_records(judgment_file)
        if config.evaluation.mt_bench_mode == "single":
            summary = summarize_mt_bench_single_scores(
                _validate_mt_bench_single_rows(
                    judgment_rows=judgment_rows,
                    questions=questions,
                    model_id=resolved_model_id,
                    judgment_file=judgment_file,
                ),
                model_id=resolved_model_id,
            )
        else:
            summary = summarize_mt_bench_pairwise(judgment_rows, model_id=resolved_model_id)

        summary.update(
            {
                "framework": "mt_bench",
                "judge_model": config.evaluation.mt_bench_judge_model,
                "judge_mode": config.evaluation.mt_bench_mode,
                "baseline_model_id": baseline_model_id if baseline_answer_file is not None else None,
                "run_dir": str(run_dir),
                "question_file": str(question_path),
                "model_answer_path": str(answer_file),
                "baseline_answer_path": str(baseline_answer_file) if baseline_answer_file else None,
                "judgment_path": str(judgment_file),
                "stdout": completed.stdout.strip(),
            }
        )
        write_json(run_dir / "summary.json", summary)
        mark_rank0_eval_success(run_dir, "mt_bench")
        return summary
    except Exception as error:
        mark_rank0_eval_failure(run_dir, "mt_bench", str(error))
        raise
