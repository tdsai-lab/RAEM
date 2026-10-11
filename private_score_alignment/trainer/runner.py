"""Training entry point for RAEM+KTO preference optimization."""

from __future__ import annotations

import json
import os
from inspect import signature
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed

from private_score_alignment.cache import DEFAULT_HF_CACHE_HOME, configure_shared_cache_env
from private_score_alignment.config.loader import save_config
from private_score_alignment.config.schemas import ExperimentConfig, TrainingConfig
from private_score_alignment.data.collator import (
    PromptResponseCollator,
    build_dataset_score_bounds,
)
from private_score_alignment.data.converters import prepare_named_dataset
from private_score_alignment.data.datasets import infer_dataset_m, load_dataset_mode
from private_score_alignment.models.factory import build_model_bundle
from private_score_alignment.privacy import LINEAR_TOPB_EM_NOISE_MODES
from private_score_alignment.trainer.trainer import PrivateScoreAlignmentTrainer


def resolve_output_root(project_root: Path, configured_root: str) -> Path:
    """Resolve an output root while preserving absolute shared-storage paths."""

    path = Path(configured_root).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def ensure_prepared_data(config: ExperimentConfig) -> None:
    """Prepare real preference datasets on demand if they are missing."""

    if config.data.dataset_mode != "list_ultrafeedback":
        raise ValueError("This release supports only data.dataset_mode='list_ultrafeedback'.")
    prepare_named_dataset("list_ultrafeedback", config.data)


def build_stage_training_arguments(
    stage_config: TrainingConfig,
    output_dir: Path,
    logging_dir: Path,
) -> TrainingArguments:
    """Translate a stage config into Hugging Face ``TrainingArguments``."""

    do_eval = bool(getattr(stage_config, "do_eval", True))
    eval_strategy = "steps" if do_eval else "no"
    report_to: list[str] | str | None
    if not stage_config.report_to:
        report_to = "none"
    else:
        report_to = stage_config.report_to
    kwargs = dict(
        output_dir=str(output_dir),
        run_name=stage_config.run_name,
        do_train=True,
        do_eval=do_eval,
        optim=stage_config.optim,
        per_device_train_batch_size=stage_config.per_device_train_batch_size,
        per_device_eval_batch_size=stage_config.per_device_eval_batch_size,
        gradient_accumulation_steps=stage_config.gradient_accumulation_steps,
        learning_rate=stage_config.learning_rate,
        weight_decay=stage_config.weight_decay,
        num_train_epochs=stage_config.num_train_epochs,
        warmup_ratio=stage_config.warmup_ratio,
        lr_scheduler_type=stage_config.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=stage_config.logging_steps,
        save_strategy="steps",
        save_steps=stage_config.save_steps,
        save_total_limit=stage_config.save_total_limit,
        bf16=stage_config.bf16,
        fp16=stage_config.fp16,
        tf32=stage_config.tf32,
        gradient_checkpointing=stage_config.gradient_checkpointing,
        remove_unused_columns=stage_config.remove_unused_columns,
        dataloader_num_workers=stage_config.dataloader_num_workers,
        report_to=report_to,
        deepspeed=(
            stage_config.deepspeed_config
            if stage_config.deepspeed_config
            and Path(stage_config.deepspeed_config).expanduser().exists()
            else None
        ),
        ddp_find_unused_parameters=stage_config.ddp_find_unused_parameters,
        max_grad_norm=stage_config.max_grad_norm,
        seed=stage_config.seed,
    )
    if do_eval:
        kwargs["eval_steps"] = stage_config.eval_steps
    training_arguments_signature = signature(TrainingArguments.__init__)
    if "logging_dir" in training_arguments_signature.parameters:
        kwargs["logging_dir"] = str(logging_dir)
    if "evaluation_strategy" in training_arguments_signature.parameters:
        kwargs["evaluation_strategy"] = eval_strategy
    if "eval_strategy" in training_arguments_signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    return TrainingArguments(**kwargs)


def build_training_arguments(
    config: ExperimentConfig,
    output_dir: Path,
    logging_dir: Path,
) -> TrainingArguments:
    """Backward-compatible wrapper for preference-stage training args."""

    return build_stage_training_arguments(config.training, output_dir=output_dir, logging_dir=logging_dir)


def _prepare_roots(config: ExperimentConfig) -> tuple[Path, Path, Path]:
    """Resolve and create checkpoint/run/log roots."""

    project_root = Path(__file__).resolve().parents[2]
    ckpt_root = resolve_output_root(project_root, config.paths.ckpt_root)
    run_root = resolve_output_root(project_root, config.paths.run_root)
    log_root = resolve_output_root(project_root, config.paths.log_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    return ckpt_root, run_root, log_root


def _configure_tracking_env(run_dir: Path, log_dir: Path) -> None:
    """Point TensorBoard runtime files to the project-local output tree."""

    del run_dir
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(log_dir)


def _configure_shared_hf_cache(cache_home: str | None = None) -> None:
    """Route Hugging Face caches away from ``~/.cache/huggingface``."""

    if cache_home is None:
        configure_shared_cache_env(force=True)
        return

    hf_root = Path(cache_home).expanduser().resolve()
    hub_cache = hf_root / "hub"
    transformers_cache = hf_root / "transformers"
    datasets_cache = hf_root / "datasets"
    modules_cache = hf_root / "modules"

    for path in (hf_root, hub_cache, transformers_cache, datasets_cache, modules_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)


def _normalize_rpw_kto_privacy_config(config: ExperimentConfig) -> None:
    """Validate and canonicalize the sole released method: RAEM+KTO."""

    if str(config.privacy.loss_type).lower() != "rpw_kto":
        raise ValueError("This release supports only privacy.loss_type='rpw_kto'.")
    mechanism = str(config.privacy.dp_mechanism).lower().replace("-", "_")
    if mechanism not in LINEAR_TOPB_EM_NOISE_MODES:
        raise ValueError("This release supports only privacy.dp_mechanism='linear_topb_em'.")
    config.privacy.dp_mechanism = "linear_topb_em"
    if str(config.privacy.noise_mode).lower() != "none":
        config.privacy.noise_mode = "linear_topb_em"
    config.privacy.rank_branch_mode = "hard_noisy_rank"


def _save_artifact_summary(output_dir: Path, artifact_paths: dict[str, str]) -> None:
    """Persist a small artifact summary next to the checkpoint."""

    with (output_dir / "artifact_paths.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact_paths, handle, indent=2, ensure_ascii=False)


def _stage_dirs(
    config: ExperimentConfig,
    stage_output_name: str,
) -> tuple[Path, Path, Path]:
    """Resolve the output/run/log directories for a stage."""

    ckpt_root, run_root, log_root = _prepare_roots(config)
    output_dir = ckpt_root / stage_output_name
    run_dir = run_root / stage_output_name
    log_dir = log_root / stage_output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_dir, log_dir


def run_preference_stage(config: ExperimentConfig) -> dict[str, str]:
    """Run noisy preference optimization with the configured listwise loss."""

    _normalize_rpw_kto_privacy_config(config)

    if config.training.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ensure_prepared_data(config)
    set_seed(config.training.seed)

    output_dir, run_dir, log_dir = _stage_dirs(config, config.training.output_name)
    _configure_tracking_env(run_dir, log_dir)

    bundle = build_model_bundle(
        config.model,
        config.training,
    )
    train_dataset = load_dataset_mode(config.data, "train")
    should_load_eval_dataset = config.training.do_eval or config.training.do_final_eval
    eval_dataset = load_dataset_mode(config.data, "benchmark") if should_load_eval_dataset else None
    resolved_m = infer_dataset_m(train_dataset)
    config.privacy.m = resolved_m
    config.evaluation.ndcg_k = resolved_m
    collator = PromptResponseCollator(
        bundle.tokenizer,
        config.model,
        score_bounds=build_dataset_score_bounds(config.data),
    )
    training_args = build_training_arguments(config, output_dir, log_dir)

    trainer_kwargs = dict(
        model=bundle.policy_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        reference_model=bundle.reference_model,
        privacy_config=config.privacy,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer_signature = signature(PrivateScoreAlignmentTrainer.__mro__[1].__init__)
    if hasattr(bundle.tokenizer, "save_pretrained"):
        if "tokenizer" in trainer_signature.parameters:
            trainer_kwargs["tokenizer"] = bundle.tokenizer
        if "processing_class" in trainer_signature.parameters:
            trainer_kwargs["processing_class"] = bundle.tokenizer
    trainer = PrivateScoreAlignmentTrainer(**trainer_kwargs)
    train_result = trainer.train(
        resume_from_checkpoint=config.training.resume_from_checkpoint,
    )
    trainer.save_model(str(output_dir))
    if hasattr(bundle.tokenizer, "save_pretrained"):
        bundle.tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if config.training.do_final_eval and eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    save_config(config, output_dir / "resolved_config.yaml")
    artifact_paths = {
        "checkpoint_dir": str(output_dir),
        "run_root": str(run_dir),
        "log_root": str(log_dir),
        "resolved_config": str(output_dir / "resolved_config.yaml"),
    }
    score_function = str(config.privacy.dp_score_function).lower().replace("-", "_")
    artifact_paths.update(
        {
            "target_source": "linear_topb_em_top_b_labels",
            "loss_type": "rpw_kto",
            "dp_mechanism": "linear_topb_em",
            "noise_mode": str(config.privacy.noise_mode),
            "epsilon": str(config.privacy.epsilon),
            "dp_score_function": score_function,
            "dp_sensitivity": "2" if score_function == "sign" else "max(B,m-B)",
            "eta": "epsilon/dp_sensitivity",
            "boundary_rank": str(config.privacy.boundary_rank),
            "rank_branch_mode": "hard_noisy_rank",
        }
    )
    _save_artifact_summary(output_dir, artifact_paths)
    return artifact_paths


def run_training(config: ExperimentConfig) -> dict[str, str]:
    """Run RAEM+KTO preference training from the configured SFT checkpoint."""

    _configure_shared_hf_cache()
    artifact_paths = run_preference_stage(config)
    artifact_paths["stage"] = "pref"
    return artifact_paths
