"""JSONL serialization helpers for prepared datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from private_score_alignment.data.schemas import UnifiedSample


def save_samples_jsonl(samples: Iterable[UnifiedSample], path: str | Path) -> None:
    """Save unified samples to JSONL."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "prompt": sample.prompt,
                "responses": sample.responses,
                "scores": sample.scores,
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "metadata": sample.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_samples_jsonl(path: str | Path) -> list[UnifiedSample]:
    """Load unified samples from JSONL."""

    source = Path(path).expanduser().resolve()
    samples: list[UnifiedSample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            samples.append(
                UnifiedSample(
                    prompt=str(payload["prompt"]),
                    responses=[str(item) for item in payload["responses"]],
                    scores=[float(item) for item in payload["scores"]],
                    sample_id=str(payload["sample_id"]),
                    dataset_name=str(payload["dataset_name"]),
                    metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
                )
            )
    return samples
