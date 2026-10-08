"""Caching-allocator observations of one device.

The worker charges graph pools and binding workspace against device
budgets; PyTorch's caching allocator owns weights, activations and graph
pools, so the worker reads its state here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "memory.stats": stats,
        "memory.release_cached": release_cached,
    }


def stats(device: int) -> dict[str, Any]:
    """Reserved and allocated bytes, and every segment's pool and size.

    ``segments`` lists ``[pool_a, pool_b, total_size]`` for each caching
    allocator segment of ``device``; the pool id pair names the graph pool
    a segment belongs to (``[0, 0]`` for the default pool).
    """
    device = int(device)
    segments = [
        [
            int(segment["segment_pool_id"][0]),
            int(segment["segment_pool_id"][1]),
            int(segment["total_size"]),
        ]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == device
    ]
    return {
        "reserved": int(torch.cuda.memory_reserved(device)),
        "allocated": int(torch.cuda.memory_allocated(device)),
        "segments": segments,
    }


def release_cached(device: int) -> None:
    """Return the allocator's unused cached blocks to the device.

    The device is synchronized first so blocks freed by completed work are
    released too; process memory measured afterwards excludes reusable cache.
    """
    torch.cuda.synchronize(int(device))
    torch.cuda.empty_cache()
