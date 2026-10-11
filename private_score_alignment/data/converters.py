"""Prepare ListUltraFeedback for the RAEM+KTO main-table experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from private_score_alignment.config.schemas import DataConfig
from private_score_alignment.data.io import save_samples_jsonl
from private_score_alignment.data.schemas import UnifiedSample


PREPARED_DATA_VERSION = 4


def dataset_slug(name: str) -> str:
    return name.lower().replace("/", "__").replace("-", "_")


def select_candidates(
    responses: Iterable[str],
    scores: Iterable[float],
    num_candidates: int,
) -> tuple[list[str], list[float]]:
    paired = sorted(
        zip((str(response) for response in responses), (float(score) for score in scores)),
        key=lambda item: (-item[1], item[0]),
    )[:num_candidates]
    return [response for response, _ in paired], [score for _, score in paired]


def load_hf_dataset_split(dataset_name_or_path: str, split: str) -> Dataset:
    if os.path.isdir(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in {dataset_name_or_path}.")
            return dataset[split]
        return dataset
    return load_dataset(dataset_name_or_path, split=split)


def convert_list_ultrafeedback(
    dataset: Dataset,
    num_candidates: int,
    min_candidates: int,
) -> list[UnifiedSample]:
    samples: list[UnifiedSample] = []
    for row in dataset:
        responses = [str(value) for value in row["all_responses"]]
        scores = [float(value) for value in row["overall_scores"]]
        if len(responses) < min_candidates or len(scores) < min_candidates:
            continue
        kept_responses, kept_scores = select_candidates(responses, scores, num_candidates)
        samples.append(
            UnifiedSample(
                prompt=str(row["prompt"]),
                responses=kept_responses,
                scores=kept_scores,
                sample_id=str(row["prompt_id"]),
                dataset_name="NDCG-alignment/ListUltraFeedback",
            )
        )
    return samples


def prepared_dataset_dir(
    prepared_root: str | Path,
    dataset_name: str,
    num_candidates: int,
    seed: int,
) -> Path:
    return (
        Path(prepared_root).expanduser().resolve()
        / dataset_slug(dataset_name)
        / f"m{num_candidates}"
        / f"seed{seed}"
    )


def _prepared_data_is_current(output_dir: Path, config: DataConfig) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        metadata.get("prepared_data_version") == PREPARED_DATA_VERSION
        and metadata.get("num_candidates") == config.num_candidates
        and metadata.get("min_candidates") == config.min_candidates
        and metadata.get("source_train_split") == config.list_ultrafeedback_train_split
        and metadata.get("source_benchmark_split") == config.list_ultrafeedback_eval_split
    )


def prepare_named_dataset(
    dataset_name: str,
    dataset_config: DataConfig,
    overwrite: bool | None = None,
) -> Path:
    if dataset_name != "list_ultrafeedback":
        raise ValueError("This release prepares only 'list_ultrafeedback'.")
    overwrite = dataset_config.overwrite_prepared if overwrite is None else overwrite
    output_dir = prepared_dataset_dir(
        dataset_config.prepared_root,
        dataset_name,
        dataset_config.num_candidates,
        dataset_config.split_seed,
    )
    if output_dir.exists() and not overwrite and _prepared_data_is_current(output_dir, dataset_config):
        return output_dir

    minimum = max(dataset_config.min_candidates, dataset_config.num_candidates)
    train_samples = convert_list_ultrafeedback(
        load_hf_dataset_split(
            dataset_config.list_ultrafeedback_name,
            dataset_config.list_ultrafeedback_train_split,
        ),
        dataset_config.num_candidates,
        minimum,
    )
    benchmark_samples = convert_list_ultrafeedback(
        load_hf_dataset_split(
            dataset_config.list_ultrafeedback_name,
            dataset_config.list_ultrafeedback_eval_split,
        ),
        dataset_config.num_candidates,
        minimum,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_samples_jsonl(train_samples, output_dir / "train.jsonl")
    save_samples_jsonl(benchmark_samples, output_dir / "benchmark.jsonl")
    metadata = {
        "dataset": dataset_config.list_ultrafeedback_name,
        "prepared_dataset_name": dataset_name,
        "prepared_data_version": PREPARED_DATA_VERSION,
        "num_candidates": dataset_config.num_candidates,
        "min_candidates": dataset_config.min_candidates,
        "split_seed": dataset_config.split_seed,
        "num_train_samples": len(train_samples),
        "num_benchmark_samples": len(benchmark_samples),
        "source_train_split": dataset_config.list_ultrafeedback_train_split,
        "source_benchmark_split": dataset_config.list_ultrafeedback_eval_split,
        "configured_score_range": [
            dataset_config.list_ultrafeedback_score_min,
            dataset_config.list_ultrafeedback_score_max,
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir
