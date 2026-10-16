"""Profiling of the numerical library: spans, profiler windows, kernel table.

- :func:`profile_range` emits optional PyTorch and NVTX spans around
  numerical and infrastructure calls; :func:`ranges_active` tells a caller
  whose span name is costly to build whether a span would be emitted.
  NVTX spans follow ``UNISERVE_NVTX`` (``1``, ``true``, ``yes`` or ``on``),
  read once when this module is imported, as the native executor reads it
  once per process.
- :func:`start_profiler` and :func:`stop_profiler` open and close a torch
  profiler window. The worker owns capture windows, step counting and trace
  naming; these functions start the installed PyTorch profiler and export
  its results.
- :func:`kernel_table` renders the kernel provider choices of a rank's
  runners. Every runner reports one record per prepared operator: its layer
  ``path``, its ``op``, the ``provider`` serving it, for some operators the
  provider of each input class (``inputs``), and fields particular to the
  operator. The rank collects the records as JSON object texts it does not
  read and hands them back to :func:`kernel_table`, which renders the table
  it logs.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)

torch: Any | None
try:
    import torch as _torch_module
except ImportError:
    torch = None
else:
    torch = _torch_module

# Whether NVTX spans are requested for this process. The switch is the one
# the native executor reads (``crates/worker/src/profiling``), with the same
# accepted values.
_NVTX_RANGES = os.environ.get("UNISERVE_NVTX", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Whether a torch profiler window is open; windows open and close at run
# time, so this is asked per span.
_profiler_enabled: Callable[[], bool] = (
    torch.autograd._profiler_enabled if torch is not None else lambda: False
)

_NULL_CONTEXT = nullcontext()


def ranges_active() -> bool:
    """Whether :func:`profile_range` currently emits any span."""
    return (_NVTX_RANGES and _nvtx_available()) or _profiler_enabled()


def profile_range(debug_name: str):
    """Emit a torch-profiler span and/or NVTX range when profiling is active."""
    record = _profiler_enabled()
    nvtx = _NVTX_RANGES and _nvtx_available()
    if not record and not nvtx:
        return _NULL_CONTEXT
    return _profile_range_impl(debug_name, record=record, nvtx=nvtx)


@functools.cache
def _nvtx_available() -> bool:
    """Whether this torch build and host emit NVTX ranges (CUDA devices).

    Asked once, and only once spans are requested, so importing this module
    never initializes the CUDA driver.
    """
    return torch is not None and torch.cuda.is_available()


@contextmanager
def _profile_range_impl(
    debug_name: str, *, record: bool, nvtx: bool
) -> Iterator[None]:
    """Enter configured record-function and NVTX ranges.

    Around one code region.
    """
    assert torch is not None
    with ExitStack() as stack:
        if record:
            stack.enter_context(torch.profiler.record_function(debug_name))
        if nvtx:
            torch.cuda.nvtx.range_push(debug_name)
            stack.callback(torch.cuda.nvtx.range_pop)
        yield


def start_profiler(
    activities: Sequence[str], with_stack: bool, record_shapes: bool
):
    """Start a profiler window.

    Args:
        activities: What to record: ``"CPU"``, ``"GPU"`` or both.
        with_stack: Record the source location of each operator.
        record_shapes: Record the operators' input shapes.

    Returns:
        The started profiler, or ``None`` when no requested activity is
        available.
    """
    # A profiler window needs torch, which spans only use when present.
    import torch

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


def stop_profiler(profiler, prefix: str, sort_device: bool) -> None:
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


def layer_paths(paths: Sequence[str]) -> list[str]:
    """Compress numbered layer paths into ranges: ``layers.{0-3,5}.mlp``.

    Paths that differ in one numbered part share a template; paths that
    differ in several stay as they are.
    """
    templates: dict[tuple[str, ...], list[list[str]]] = {}
    for path in paths:
        parts = path.split(".")
        key = tuple("*" if part.isdecimal() else part for part in parts)
        templates.setdefault(key, []).append(parts)

    result: list[str] = []
    for key, members in templates.items():
        varying = [
            position
            for position, part in enumerate(key)
            if part == "*"
            and any(
                member[position] != members[0][position] for member in members
            )
        ]
        if len(varying) > 1:
            result.extend(".".join(parts) for parts in members)
            continue
        parts = list(members[0])
        if varying:
            (position,) = varying
            numbers = sorted({int(member[position]) for member in members})
            runs: list[list[int]] = []
            for number in numbers:
                if runs and runs[-1][1] + 1 == number:
                    runs[-1][1] = number
                else:
                    runs.append([number, number])
            ranges = ",".join(
                str(start) if start == stop else f"{start}-{stop}"
                for start, stop in runs
            )
            parts[position] = "{" + ranges + "}"
        result.append(".".join(parts))
    return result


def _call_sites(
    records: Sequence[str],
) -> list[tuple[dict[str, Any], list[str]]]:
    """Group a runner's records into call sites with their layer paths."""
    # Each layer's distinct records, in the order they were reported.
    layers: dict[str, list[dict[str, Any]]] = {}
    for text in records:
        record = json.loads(text)
        values = layers.setdefault(str(record.pop("path")), [])
        if record not in values:
            values.append(record)

    sites: list[tuple[dict[str, Any], list[str]]] = []
    for path, values in layers.items():
        # An attention layer with a prepared provider reports its
        # unprepared default too; the prepared one is the call site.
        prepared = any(
            record.get("op") == "attention"
            and record.get("provider") is not None
            for record in values
        )
        for record in values:
            if (
                prepared
                and record.get("op") == "attention"
                and record.get("provider") is None
            ):
                continue
            for site, paths in sites:
                if site == record:
                    paths.append(path)
                    break
            else:
                sites.append((record, [path]))
    return sites


def kernel_table(
    stage: str,
    rank: int,
    device: str,
    labels: Sequence[str],
    devices: Sequence[str],
    counts: Sequence[int],
    records: Sequence[str],
) -> str:
    """Render the kernel table of a rank's runners as one JSON object.

    Args:
        stage: ``"startup"`` or ``"serving"``.
        rank: The reporting rank.
        device: The rank's device name.
        labels: Each runner's label.
        devices: Each runner's device name.
        counts: How many of ``records`` each runner reported.
        records: Every runner's kernel records, back to back, as the JSON
            object texts the runners returned.

    Returns:
        ``{"stage", "rank", "device", "runners", "portable"}``: each runner
        with its call sites (equal records of several layers share one,
        with compressed ``layers``), and the call sites of CUDA runners
        that portable torch code serves, with the inputs it serves.
    """
    runners = []
    portable = []
    first = 0
    for label, placed, count in zip(labels, devices, counts, strict=True):
        reported = records[first : first + int(count)]
        first += int(count)
        calls = []
        for record, paths in _call_sites(reported):
            compressed = layer_paths(paths)
            calls.append({**record, "layers": compressed})
            provider = record.get("provider")
            if provider == "torch":
                served = ["every input"]
            elif provider is None:
                served = []
            else:
                served = [
                    name
                    for name, serving in (record.get("inputs") or {}).items()
                    if serving == "torch"
                ]
            if served and (placed == "cuda" or placed.startswith("cuda:")):
                portable.append(
                    {
                        "runner": label,
                        "op": record.get("op"),
                        "layers": compressed,
                        "inputs": served,
                    }
                )
        runners.append(
            {"runner": str(label), "device": str(placed), "call_sites": calls}
        )
    return json.dumps(
        {
            "stage": str(stage),
            "rank": int(rank),
            "device": str(device),
            "runners": runners,
            "portable": portable,
        },
        separators=(",", ":"),
    )
