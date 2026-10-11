from pathlib import Path

from private_score_alignment.config import load_config
from private_score_alignment.config.model_presets import apply_model_preset
from private_score_alignment.eval.local_judges import _prepare_shared_hf_cache


def test_main_config_is_portable_and_main_method_only() -> None:
    config = load_config("config/main.yaml")

    for value in (
        config.paths.data_root,
        config.paths.model_root,
        config.paths.pretrained_root,
        config.paths.checkpoint_source_root,
        config.paths.output_root,
        config.paths.log_root,
        config.paths.run_root,
        config.paths.ckpt_root,
        config.data.prepared_root,
    ):
        assert not Path(value).expanduser().is_absolute()

    assert config.data.dataset_mode == "list_ultrafeedback"
    assert config.data.num_candidates == 8
    assert config.data.min_candidates == 8
    assert config.privacy.dp_mechanism == "linear_topb_em"
    assert config.privacy.dp_score_function == "linear"
    assert config.privacy.loss_type == "rpw_kto"
    assert config.privacy.rank_branch_mode == "hard_noisy_rank"
    assert config.training.report_to == ["tensorboard"]
    assert config.model.policy_model_path == "hZzy/qwen2-0.5b-sft"
    assert config.model.reference_model_path == "hZzy/qwen2-0.5b-sft"


def test_main_table_presets_use_public_sft_checkpoints() -> None:
    config = load_config("config/main.yaml")

    apply_model_preset(config, "qwen")
    assert config.model.policy_model_path == "hZzy/qwen2-0.5b-sft"
    assert config.model.reference_model_path == "hZzy/qwen2-0.5b-sft"

    apply_model_preset(config, "mistral")
    expected = "kaitchup/Mistral-7B-v0.1-SFT-ultrachat"
    assert config.model.policy_model_path == expected
    assert config.model.reference_model_path == expected


def test_shared_cache_environment_overrides_config_path(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "shared-cache"
    monkeypatch.setenv("PSA_SHARED_CACHE_ROOT", str(cache_root))

    model_root, hub_cache = _prepare_shared_hf_cache("./ignored-config-path")

    assert model_root == cache_root / "judge_models"
    assert hub_cache == cache_root / ".cache" / "huggingface" / "hub"
