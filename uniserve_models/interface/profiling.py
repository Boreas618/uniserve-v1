"""Torch profiler windows the worker opens and closes.

The worker owns capture windows, step counting and trace naming; these
functions start the installed PyTorch profiler and export its results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "profiling.start": start,
        "profiling.stop": stop,
    }


def start(activities: Sequence[str], with_stack: bool, record_shapes: bool):
    """Start a profiler window.

    Args:
        activities: What to record: ``"CPU"``, ``"GPU"`` or both.
        with_stack: Record the source location of each operator.
        record_shapes: Record the operators' input shapes.

    Returns:
        The started profiler, or ``None`` when no requested activity is
        available.
    """
    requested = {str(name) for name in activities}
    selected = []
    if "CPU" in requested:
        selected.append(torch.profiler.ProfilerActivity.CPU)
    if "GPU" in requested:
        if torch.cuda.is_available():
            selected.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            logger.warning(
                "UNISERVE_PROFILE_ACTIVITIES requested GPU but CUDA is "
                "unavailable"
            )
    if not selected:
        return None
    profiler = torch.profiler.profile(
        activities=selected,
        with_stack=bool(with_stack),
        record_shapes=bool(record_shapes),
        acc_events=True,
    )
    profiler.start()
    return profiler


def stop(profiler, prefix: str, sort_device: bool) -> None:
    """Stop a window and export ``{prefix}.trace.json.gz`` and a summary.

    The summary (``{prefix}.summary.txt``) tables the 120 most expensive
    operators by self device time, or self CPU time without a device.
    """
    profiler.stop()
    prefix = str(prefix)
    profiler.export_chrome_trace(f"{prefix}.trace.json.gz")
    summary = profiler.key_averages().table(
        sort_by="self_device_time_total"
        if sort_device
        else "self_cpu_time_total",
        row_limit=120,
    )
    with open(f"{prefix}.summary.txt", "w", encoding="utf-8") as file:
        file.write(summary)
