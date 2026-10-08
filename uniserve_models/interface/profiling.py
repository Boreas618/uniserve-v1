"""Torch profiler windows the worker opens and closes.

The worker owns capture windows, step counting and trace naming; these
functions start the installed PyTorch profiler and export its results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch

logger = logging.getLogger(__name__)


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "profiling.start": start,
        "profiling.stop": stop,
    }


def start(options: Mapping[str, Any]):
    """Start a profiler window.

    ``options`` carries ``activities`` (``"CPU"``, ``"GPU"``),
    ``with_stack`` and ``record_shapes``. Returns the started profiler, or
    ``None`` when no requested activity is available.
    """
    activities = {str(name) for name in options["activities"]}
    selected = []
    if "CPU" in activities:
        selected.append(torch.profiler.ProfilerActivity.CPU)
    if "GPU" in activities:
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
        with_stack=bool(options["with_stack"]),
        record_shapes=bool(options["record_shapes"]),
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
