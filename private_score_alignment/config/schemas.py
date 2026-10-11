"""Dataclass configuration for the released main-table experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PathConfig:
    data_root: str = "./data"
    model_root: str = "./models"
    pretrained_root: str = "./.cache"
    checkpoint_source_root: str = "./checkpoints"
    output_root: str = "./outputs"
    log_root: str = "./logs"
    run_root: str = "./runs"
    ckpt_root: str = "./checkpoints"


@dataclass
class ModelConfig:
    policy_model_name: str = "hZzy/qwen2-0.5b-sft"
    reference_model_name: str = "hZzy/qwen2-0.5b-sft (frozen)"
    policy_model_path: str = "hZzy/qwen2-0.5b-sft"
    reference_model_path: str = "hZzy/qwen2-0.5b-sft"
    tokenizer_path: str = "hZzy/qwen2-0.5b-sft"
    model_family: str = "qwen2"
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    use_chat_template: bool = True
    max_prompt_length: int = 512
    max_response_length: int = 1024
    prompt_template: str = "User: {prompt}\n\nAssistant:"
    system_prompt: str = ""


@dataclass
class PrivacyConfig:
    dp_mechanism: str = "linear_topb_em"
    noise_mode: str = "linear_topb_em"
    epsilon: float = 1.0
    boundary_rank: Optional[int] = None
    rank_branch_mode: str = "hard_noisy_rank"
    dp_score_function: str = "linear"
    noise_seed: int = 42
    loss_type: str = "rpw_kto"
    beta: float = 0.1
    kto_lambda_d: float = 1.0
    kto_lambda_u: float = 1.0
    label_smoothing: float = 0.0
    power_p: float = 2.0
    m: Optional[int] = None


@dataclass
class DataConfig:
    dataset_mode: str = "list_ultrafeedback"
    list_ultrafeedback_name: str = "NDCG-alignment/ListUltraFeedback"
    list_ultrafeedback_train_split: str = "train"
    list_ultrafeedback_eval_split: str = "test"
    prepared_root: str = "./data/prepared"
    split_seed: int = 42
    num_candidates: int = 8
    min_candidates: int = 8
    list_ultrafeedback_score_min: float = -0.07257080078125
    list_ultrafeedback_score_max: float = 0.94775390625
    max_train_samples: int = 0
    max_eval_samples: int = 0
    num_proc: int = 1
    overwrite_prepared: bool = False


@dataclass
class TrainingConfig:
    seed: int = 42
    output_name: str = "qwen2_0p5b_instruct_private_score_alignment"
    run_name: str = "qwen2-0p5b-instruct-private-score-alignment"
    optim: str = "adamw_torch"
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    do_eval: bool = True
    do_final_eval: bool = True
    save_total_limit: int = 2
    dataloader_num_workers: int = 2
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    deepspeed_config: Optional[str] = "./config/deepspeed/zero2.json"
    resume_from_checkpoint: Optional[str] = None
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False


@dataclass
class EvalConfig:
    ndcg_k: Optional[int] = None
    metrics_output_subdir: str = "benchmark"
    external_output_subdir: str = "external_eval"
    generation_max_new_tokens: int = 512
    generation_temperature: float = 0.0
    generation_top_p: float = 1.0
    generation_do_sample: bool = False
    generation_batch_size: int = 1
    alpaca_eval_generation_max_new_tokens: int = 256
    mt_bench_generation_max_new_tokens: int = 384
    local_pair_preference_judge_model: str = "RLHFlow/pair-preference-model-LLaMA3-8B"
    local_scoring_judge_model: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    local_judge_batch_size: int = 4
    local_judge_max_length: int = 4096
    alpaca_eval_dataset_path: str = (
        "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    )
    alpaca_eval_annotators_config: str = "weighted_alpaca_eval_gpt4_turbo_new"
    alpaca_eval_prepare_only: bool = False
    alpaca_eval_reference_outputs: str = ""
    mt_bench_question_path: str = (
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    )
    mt_bench_judge_prompts_path: str = (
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/judge_prompts.jsonl"
    )
    mt_bench_reference_answer_path: str = (
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"
    )
    mt_bench_judge_model: str = "gpt-4-turbo"
    mt_bench_mode: str = "pairwise-baseline"
    mt_bench_parallel: int = 2
    mt_bench_baseline_model: str = ""
    mt_bench_baseline_answer_path: str = ""


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataclass(cls: type[Any], payload: Optional[dict[str, Any]]) -> Any:
    if payload is None:
        return cls()
    kwargs = {}
    defaults = cls()
    for field_name, field_definition in cls.__dataclass_fields__.items():
        has_value = field_name in payload
        value = payload.get(field_name)
        default_value = getattr(defaults, field_name)
        if hasattr(default_value, "__dataclass_fields__"):
            kwargs[field_name] = build_dataclass(type(default_value), value)
        else:
            kwargs[field_name] = value if has_value else default_value
    return cls(**kwargs)
