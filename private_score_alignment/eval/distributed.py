"""Lightweight distributed helpers for multi-GPU evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Iterable, TypeVar

import torch
import torch.distributed as dist


T = TypeVar("T")
DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 7200
TIMEOUT_ENV_VAR = "PSA_DISTRIBUTED_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class DistributedEvalState:
    """Runtime state for torchrun-launched evaluation."""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


_STATE: DistributedEvalState | None = None


def _distributed_timeout() -> timedelta:
    """Return a long eval timeout for imbalanced generation shards."""

    raw_value = os.environ.get(TIMEOUT_ENV_VAR, "").strip()
    if not raw_value:
        return timedelta(seconds=DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS)
    timeout_seconds = int(raw_value)
    if timeout_seconds <= 0:
        raise ValueError(f"{TIMEOUT_ENV_VAR} must be positive when set.")
    return timedelta(seconds=timeout_seconds)


def init_distributed_evaluation() -> DistributedEvalState:
    """Initialize torch.distributed when launched with torchrun."""

    global _STATE
    if _STATE is not None:
        return _STATE

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=_distributed_timeout(),
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    _STATE = DistributedEvalState(
        enabled=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )
    return _STATE


def get_distributed_state() -> DistributedEvalState:
    """Return the current distributed evaluation state."""

    return init_distributed_evaluation()


def barrier() -> None:
    """Synchronize all ranks when running distributed evaluation."""

    state = get_distributed_state()
    if state.enabled and dist.is_initialized():
        if state.device.type == "cuda":
            dist.barrier(device_ids=[state.local_rank])
        else:
            dist.barrier()


def finalize_distributed_evaluation() -> None:
    """Tear down the torch.distributed process group when initialized here."""

    global _STATE
    if dist.is_initialized():
        barrier()
        dist.destroy_process_group()
    _STATE = None


def shard_sequence(items: list[T]) -> list[T]:
    """Take a deterministic strided shard of a list for the current rank."""

    state = get_distributed_state()
    if not state.enabled:
        return items
    return items[state.rank :: state.world_size]


def shard_indices(length: int) -> list[int]:
    """Return the strided index shard for the current rank."""

    return list(range(get_distributed_state().rank, length, get_distributed_state().world_size))


def gather_objects(local_object: T) -> list[T]:
    """Gather arbitrary Python objects from all ranks."""

    state = get_distributed_state()
    if not state.enabled:
        return [local_object]
    gathered: list[T | None] = [None for _ in range(state.world_size)]
    dist.all_gather_object(gathered, local_object)
    return list(gathered)


def reduce_sum(values: Iterable[float]) -> list[float]:
    """All-reduce a list of scalar sums across ranks."""

    state = get_distributed_state()
    tensor = torch.tensor(list(values), dtype=torch.float64, device=state.device)
    if state.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()
