"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnifiedSample:
    """Unified prompt-level sample used across datasets."""

    prompt: str
    responses: list[str]
    scores: list[float]
    sample_id: str
    dataset_name: str
    metadata: dict[str, str] = field(default_factory=dict)
