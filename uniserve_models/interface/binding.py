"""Component bindings over Rust-owned workspace.

A binding is one module tree prepared for one stream and size. The worker
first asks for its requirements (``component.prepare``), allocates every
requested workspace field itself, then binds the allocations
(``component.bind``). Capability runtimes build their step functions over
the bound context (``prepared.context``) with their own ``<area>.bind``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from uniserve_models.bindings.context import (
    BoundWorkspace,
    DeclaredWorkspace,
    ExecutionContext,
)
from uniserve_models.model import TextSize


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "component.prepare": prepare,
        "component.bind": bind,
    }


@dataclass
class Prepared:
    """A binding's module, options and, once bound, its context."""

    component: str
    module: nn.Module
    size: TextSize | None
    options: dict[str, Any]
    context: ExecutionContext | None = None
    workspaces: dict[str, torch.Tensor] = field(default_factory=dict)


def _size(value) -> TextSize | None:
    if value is None:
        return None
    return TextSize(int(value["tokens"]), int(value["rows"]))


def prepare(loaded, request: Mapping[str, Any]) -> list[Any]:
    """Declare one binding's workspace requirements.

    Returns:
        ``[prepared, {"workspaces": [{"name", "role", "dtype", "shape",
        "host", "symmetric", "device"}]}]``. ``role`` names a shared
        scratch area (one backing per stream sized for every binding
        prepared on it); ``None`` marks storage owned by this binding alone.
        ``symmetric`` lists the process ranks, in member order, across which
        a field must be peer-mapped (``None`` for local storage); its peer
        views are bound as ``{name}.peer{index}``. ``device`` is the CUDA
        ordinal, ``-1`` for CPU.
    """
    path = str(request["path"])
    module = loaded.module.get_submodule(path) if path else loaded.module
    stream = int(request["stream"])
    options = {
        "stream": stream if stream else None,
        "cache": request.get("cache"),
        "attention": str(request["attention"])
        if str(request["attention"]) != "auto"
        else loaded.attention,
        "vsa": str(request["vsa"]),
        "matmul": str(request["matmul"]),
        "moe": str(request["moe"]),
        "derive_host_lengths": bool(request["derive_host_lengths"]),
        "experts": request.get("experts"),
        "weights": request.get("weights"),
    }
    prepared = Prepared(
        str(request["component"]), module, _size(request.get("size")), options
    )
    declared = DeclaredWorkspace()
    context = ExecutionContext(module, workspace=declared, **options)
    try:
        # Without a text size the call sites bind without a row bound, as
        # encoders, decoders and media denoisers serve their sizes.
        context.prepare(prepared.size)
        requirements = [
            {
                "name": item.name,
                "role": item.role,
                "dtype": str(item.dtype).removeprefix("torch."),
                "shape": list(item.shape),
                "host": item.host,
                "symmetric": None
                if item.symmetric is None
                else list(item.symmetric),
                "device": -1
                if item.device.type != "cuda"
                else (item.device.index or 0),
            }
            for item in declared.requirements
        ]
    finally:
        context.close()
    return [prepared, {"workspaces": requirements}]


def bind(prepared: Prepared, resources: Mapping[str, Any]) -> dict:
    """Bind the declared workspace to tensors the worker allocated.

    ``resources["workspaces"]`` maps every declared name to a DLPack tensor
    whose storage the worker owns; operators and plans are created now.

    Returns:
        ``{"kernels", "close"}``: the selected kernels by module path, and
        release of the binding (``close(aborted)``; an aborted close keeps
        every Python reference until the process exits).
    """
    tensors = {
        str(name): torch.from_dlpack(tensor)
        for name, tensor in dict(resources.get("workspaces") or {}).items()
    }
    prepared.workspaces = tensors
    context = ExecutionContext(
        prepared.module, workspace=BoundWorkspace(tensors), **prepared.options
    )
    context.prepare(prepared.size)
    prepared.context = context

    def kernels() -> list[dict]:
        return context.kernels()

    def close(aborted: bool) -> None:
        context.close(aborted=bool(aborted))
        if not aborted:
            prepared.context = None
            prepared.workspaces = {}

    return {"kernels": kernels, "close": close}
