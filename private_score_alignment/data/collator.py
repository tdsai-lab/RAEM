"""Tokenizer-aware collator for prompt/response candidate batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from private_score_alignment.config.schemas import DataConfig, ModelConfig
from private_score_alignment.data.schemas import UnifiedSample


@dataclass
class EncodedCandidate:
    """Tokenized prompt/response pair."""

    input_ids: list[int]
    attention_mask: list[int]
    response_mask: list[int]


def build_dataset_score_bounds(data_config: DataConfig) -> dict[str, tuple[float, float]]:
    """Return per-dataset score bounds used for privacy calibration."""

    return {
        "list_ultrafeedback": (
            data_config.list_ultrafeedback_score_min,
            data_config.list_ultrafeedback_score_max,
        ),
        data_config.list_ultrafeedback_name: (
            data_config.list_ultrafeedback_score_min,
            data_config.list_ultrafeedback_score_max,
        ),
    }


class PromptResponseCollator:
    """Collate a batch of listwise prompt/response groups."""

    def __init__(
        self,
        tokenizer,
        model_config: ModelConfig,
        score_bounds: dict[str, tuple[float, float]] | None = None,
        truncate_prompt: bool = True,
        truncate_response: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.score_bounds = score_bounds or {}
        self.truncate_prompt = truncate_prompt
        self.truncate_response = truncate_response

    def _extract_input_ids(self, encoded: object) -> list[int]:
        """Normalize tokenizer outputs into a plain ``list[int]``.

        Fast tokenizers may return ``BatchEncoding`` or low-level
        ``tokenizers.Encoding`` objects instead of a raw Python list. The
        collator only needs token ids, so normalize all supported shapes here.
        """

        input_ids = encoded
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            input_ids = getattr(encoded, "input_ids")

        if hasattr(input_ids, "ids"):
            input_ids = getattr(input_ids, "ids")

        if not isinstance(input_ids, list):
            try:
                input_ids = list(input_ids)
            except TypeError as error:
                raise TypeError(
                    f"Unsupported tokenizer output type for input ids: {type(encoded)!r}"
                ) from error

        if input_ids and hasattr(input_ids[0], "ids"):
            input_ids = list(getattr(input_ids[0], "ids"))

        return [int(token_id) for token_id in input_ids]

    def _tokenize_text(self, text: str, add_special_tokens: bool) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=add_special_tokens)
        return self._extract_input_ids(encoded)

    def _encode_prefix(self, prompt: str) -> list[int]:
        if (
            self.model_config.use_chat_template
            and getattr(self.tokenizer, "chat_template", None)
            and hasattr(self.tokenizer, "apply_chat_template")
        ):
            messages = []
            if self.model_config.system_prompt:
                messages.append({"role": "system", "content": self.model_config.system_prompt})
            messages.append({"role": "user", "content": prompt})
            prefix_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            prefix_ids = self._extract_input_ids(prefix_ids)
        else:
            prefix_text = self.model_config.prompt_template.format(prompt=prompt)
            if self.model_config.system_prompt:
                prefix_text = f"System: {self.model_config.system_prompt}\n\n{prefix_text}"
            prefix_ids = self._tokenize_text(prefix_text, add_special_tokens=True)
        if self.truncate_prompt:
            return prefix_ids[-self.model_config.max_prompt_length :]
        return prefix_ids

    def _encode_candidate(self, prompt: str, response: str) -> EncodedCandidate:
        prefix_ids = self._encode_prefix(prompt)
        response_ids = self._tokenize_text(response, add_special_tokens=False)
        if self.truncate_response:
            response_ids = response_ids[: self.model_config.max_response_length]

        input_ids = list(prefix_ids)
        response_mask = [0] * len(prefix_ids)
        input_ids.extend(response_ids)
        response_mask.extend([1] * len(response_ids))

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and (not input_ids or input_ids[-1] != eos_token_id):
            input_ids.append(int(eos_token_id))
            response_mask.append(1)

        attention_mask = [1] * len(input_ids)
        return EncodedCandidate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
        )

    def __call__(self, samples: list[UnifiedSample]) -> dict[str, torch.Tensor | list[str]]:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")

        group_sizes = [len(sample.responses) for sample in samples]
        max_candidates = max(group_sizes)
        encoded_candidates: list[EncodedCandidate] = []
        for sample in samples:
            for response in sample.responses:
                encoded_candidates.append(self._encode_candidate(sample.prompt, response))

        max_length = max(len(candidate.input_ids) for candidate in encoded_candidates)
        pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

        input_ids = []
        attention_mask = []
        response_mask = []
        for candidate in encoded_candidates:
            pad_length = max_length - len(candidate.input_ids)
            input_ids.append(candidate.input_ids + [pad_token_id] * pad_length)
            attention_mask.append(candidate.attention_mask + [0] * pad_length)
            response_mask.append(candidate.response_mask + [0] * pad_length)

        raw_scores = torch.zeros(len(samples), max_candidates, dtype=torch.float32)
        item_mask = torch.zeros(len(samples), max_candidates, dtype=torch.bool)
        score_mins = torch.zeros(len(samples), dtype=torch.float32)
        score_maxs = torch.zeros(len(samples), dtype=torch.float32)
        dataset_names: list[str] = []
        sample_ids: list[str] = []
        prompts: list[str] = []
        responses: list[list[str]] = []
        for batch_index, sample in enumerate(samples):
            count = len(sample.responses)
            raw_scores[batch_index, :count] = torch.tensor(sample.scores, dtype=torch.float32)
            item_mask[batch_index, :count] = True
            dataset_names.append(sample.dataset_name)
            if sample.dataset_name not in self.score_bounds:
                raise KeyError(
                    f"Missing score bounds for dataset '{sample.dataset_name}'. "
                    "Pass a score-bounds mapping when constructing the collator."
                )
            score_min, score_max = self.score_bounds[sample.dataset_name]
            score_mins[batch_index] = float(score_min)
            score_maxs[batch_index] = float(score_max)
            sample_ids.append(sample.sample_id)
            prompts.append(sample.prompt)
            responses.append(sample.responses)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.float32),
            "raw_scores": raw_scores,
            "item_mask": item_mask,
            "score_mins": score_mins,
            "score_maxs": score_maxs,
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
            "dataset_names": dataset_names,
            "sample_ids": sample_ids,
            "prompts": prompts,
            "responses": responses,
        }
        return batch
