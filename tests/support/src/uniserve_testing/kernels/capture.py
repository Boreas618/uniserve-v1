"""A native module's entry called on a stream, directly or while it captures.

A native entry is safe to call while its stream captures (``ops::kernel``'s
native entry ABI): the engine captures every bucket it serves, so an entry is
called during a capture far more often than on a stream that runs it at once.
A capture rejects some host calls on the capturing stream, such as asking the
runtime which device the stream belongs to, and a rejected call invalidates the
capture, so an entry that makes one fails only when it is captured. A check of
a native module therefore runs its case in both modes (:data:`MODES`) through
:func:`call`, which captures the call as the engine's worker does (in the
relaxed mode, on a stream of its own) and replays the graph once; the case's
assertions then read what the replay wrote.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

import pytest
import torch

#: The modes a check calls an entry in, as test parameters with readable ids:
#: on the device's current stream, and captured into a graph that is replayed.
MODES = [
    pytest.param(False, id="direct"),
    pytest.param(True, id="captured"),
]


def call(
    entry: Callable[..., int],
    structure: ctypes.Structure,
    device: torch.device,
    captured: bool,
) -> None:
    """Call ``entry`` with ``structure`` on a stream of ``device`` and wait
    for the work it issued.

    ``captured`` captures the call into a graph on a stream of its own and
    replays the graph; otherwise the call runs on the device's current stream.
    Either way the device is synchronized before this returns.

    Raises:
        AssertionError: When the entry returns a nonzero status, with the
            message it wrote into the structure's ``error`` field. Under a
            capture the refusal is raised inside the capture, whose end then
            reports the invalidated capture as well.
    """  # noqa: D205
    if not captured:
        stream = torch.cuda.current_stream(device)
        _require_success(entry, structure, stream)
        torch.cuda.synchronize(device)
        return

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device)
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="relaxed"):
        _require_success(entry, structure, stream)
    graph.replay()
    torch.cuda.synchronize(device)


def _require_success(
    entry: Callable[..., int], structure: Any, stream: torch.cuda.Stream
) -> None:
    """Call ``entry`` on ``stream`` and assert that it accepted the call."""
    status = entry(ctypes.byref(structure), ctypes.c_void_p(stream.cuda_stream))
    message = structure.error.decode("utf-8", "replace").rstrip("\x00")
    assert status == 0, f"the module refused the call ({status}): {message}"
