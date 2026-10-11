"""Prepared dataset loading for training and benchmark evaluation."""

from __future__ import annotations

from torch.utils.data import Dataset

from private_score_alignment.config.schemas import DataConfig
from private_score_alignment.data.converters import prepared_dataset_dir
from private_score_alignment.data.io import load_samples_jsonl
from private_score_alignment.data.schemas import UnifiedSample


class UnifiedListwiseDataset(Dataset):
    """Simple torch dataset backed by :class:`UnifiedSample` records."""

    def __init__(self, samples: list[UnifiedSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> UnifiedSample:
        return self.samples[index]


def infer_dataset_m(dataset: UnifiedListwiseDataset) -> int:
    """Infer the effective candidate count ``m`` from a loaded dataset."""

    if len(dataset) == 0:
        raise ValueError("Cannot infer m from an empty dataset.")
    candidate_counts = [len(sample.responses) for sample in dataset.samples]
    return max(candidate_counts)


def load_prepared_dataset(
    data_config: DataConfig,
    dataset_name: str,
    split: str,
) -> UnifiedListwiseDataset:
    """Load a prepared real dataset split from JSONL."""

    directory = prepared_dataset_dir(
        prepared_root=data_config.prepared_root,
        dataset_name=dataset_name,
        num_candidates=data_config.num_candidates,
        seed=data_config.split_seed,
    )
    path = directory / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {path}. Run scripts/prepare_data.py first."
        )
    samples = load_samples_jsonl(path)
    if split == "train" and data_config.max_train_samples > 0:
        samples = samples[: data_config.max_train_samples]
    if split != "train" and data_config.max_eval_samples > 0:
        samples = samples[: data_config.max_eval_samples]
    return UnifiedListwiseDataset(samples)


def load_dataset_mode(
    data_config: DataConfig,
    split: str,
) -> UnifiedListwiseDataset:
    """Load dataset(s) based on the configured mode."""

    if data_config.dataset_mode != "list_ultrafeedback":
        raise ValueError("This release supports only data.dataset_mode='list_ultrafeedback'.")
    return load_prepared_dataset(data_config, "list_ultrafeedback", split)
