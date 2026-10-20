"""Layer weights a rank keeps in host memory and streams to its device.

A component placed with ``"weights": "host"`` keeps the dense projection
weight matrices of its repeated layer stack in one page-locked host region
the worker runtime owns. The device holds ``slots`` layer-sized buffers, and
layer ``i``'s projection parameters are fixed views of buffer ``i % slots``,
so operators bound to those parameters read whichever layer was last copied
there. The runtime copies the layers on its own stream and orders them by
events against the computing stream: before a layer's first projection call
it makes the layer readable and copies the following layers ahead into
buffers no live layer reads, and after every projection call it records that
the layer's buffer was read.

This module states what is numerical about that: which modules form the
stack (:func:`layer_stack`), how one layer's projection matrices lie in a
layer buffer (:func:`layout`), and the rebinding of the parameters and the
hooks that bracket each projection call (:func:`bind`). Biases, norms and
every other parameter stay resident.

The host region is memory the host's Storage Server retains. A worker that
loaded the stack copies the matrices into it (their places in it are
:func:`placements`); a restarted worker finds them there, bound to views of
the region in place of their device tensors, and :func:`bind` rebinds them
to the device slots without copying.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from uniserve.quantization import QuantizedTensor

__all__ = [
    "address",
    "bind",
    "layer_stack",
    "layers",
    "layout",
    "placements",
    "streamed",
]

# Byte alignment of every storage placed in a layer buffer.
_ALIGNMENT = 256


def _module(loaded, path: str) -> nn.Module:
    """The module at ``path`` of the loaded model (``""`` for its root)."""
    path = str(path)
    return loaded.module.get_submodule(path) if path else loaded.module


def layer_stack(module: nn.Module) -> tuple[nn.Module, ...]:
    """Return the repeated layer stack of ``module``.

    The stack is the ``ModuleList`` or ``ModuleDict`` under ``module`` whose
    children, at least four, share one class and hold the most parameter
    bytes among such containers.

    Raises:
        ValueError: No container holds four children of one class.
    """
    best, most = None, -1
    for container in module.modules():
        if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            continue
        children = tuple(container.children())
        if len(children) < 4 or len({type(child) for child in children}) != 1:
            continue
        size = sum(
            parameter.numel() * parameter.element_size()
            for parameter in container.parameters()
        )
        if size > most:
            best, most = children, size
    if best is None:
        raise ValueError(
            f"{type(module).__name__} has no stack of four layers of one class"
        )
    return best


def _projections(layer: nn.Module) -> list[nn.Module]:
    """Return a layer's outermost projection modules in module order.

    A projection is a ``Linear`` or a merged column projection; a merged
    projection evaluates its branches itself, so its branches are not
    projections of their own.
    """
    from uniserve.nn.linear import Linear, MergedColumnParallelLinear

    found: list[nn.Module] = []

    def visit(module: nn.Module) -> None:
        if isinstance(module, (Linear, MergedColumnParallelLinear)):
            found.append(module)
            return
        for child in module.children():
            visit(child)

    visit(layer)
    return found


def _place(modules: Sequence[nn.Module]):
    """Lay out the storages of the modules' projection weight matrices.

    Returns ``(parameters, storages, size)``: every weight matrix with its
    byte offset in a layer buffer, every distinct backing storage with its
    offset, and the buffer bytes. Matrices sharing one storage keep their
    relative placement, so merged projection views stay merged. Vectors
    (biases) stay resident.

    Raises:
        ValueError: A projection weight is an encoded tensor.
    """
    parameters, storages, offsets = [], [], {}
    size = 0
    for module in modules:
        for parameter in module.parameters():
            if isinstance(parameter.data, QuantizedTensor):
                raise ValueError(
                    "streamed projection weights must be dense tensors"
                )
            if parameter.ndim < 2:
                continue
            storage = parameter.untyped_storage()
            if storage.data_ptr() not in offsets:
                offsets[storage.data_ptr()] = size
                storages.append((storage, size))
                size += -(-storage.nbytes() // _ALIGNMENT) * _ALIGNMENT
            offset = (
                offsets[storage.data_ptr()]
                + parameter.storage_offset() * parameter.element_size()
            )
            parameters.append((parameter, offset))
    return parameters, storages, size


def _stack(loaded, path: str):
    """The stack under ``path``, each layer's projections and placement.

    Raises:
        ValueError: Fewer than four layers, no dense projection weights,
            projections of differing layout, an encoded projection weight,
            or weights not on one CUDA device.
    """
    layers = layer_stack(_module(loaded, path))
    projections = [_projections(layer) for layer in layers]
    placements = [_place(modules) for modules in projections]
    shapes = {
        (
            size,
            tuple((storage.nbytes(), offset) for storage, offset in stores),
            tuple(
                (tuple(p.shape), tuple(p.stride()), p.dtype, offset)
                for p, offset in parameters
            ),
        )
        for parameters, stores, size in placements
    }
    if len(shapes) != 1 or not placements[0][0]:
        raise ValueError(
            "streamed layers need one common projection weight layout"
        )
    devices = {
        parameter.device
        for parameters, _, _ in placements
        for parameter, _ in parameters
    }
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("streamed weights must share one CUDA device")
    return projections, placements


def layers(loaded, path: str) -> int:
    """The number of layers of the stack under ``path``."""
    return len(layer_stack(_module(loaded, path)))


def address(loaded, path: str) -> int:
    """The lowest address of the streamed matrices under ``path``.

    After binding a worker's retained weights, the matrices are views of
    the stack's retained host region, which this address lies in.
    """
    return min(parameter.data_ptr() for parameter in streamed(loaded, path))


def streamed(loaded, path: str) -> list[nn.Parameter]:
    """The projection weight matrices the stack under ``path`` streams."""
    return [
        parameter
        for layer in layer_stack(_module(loaded, path))
        for module in _projections(layer)
        for parameter in module.parameters()
        if parameter.ndim >= 2
    ]


def placements(loaded, path: str) -> list[tuple[nn.Parameter, int]]:
    """Each streamed matrix with its byte offset in the host region.

    Layer ``i``'s matrices lie in bytes ``[i * layer_bytes, (i + 1) *
    layer_bytes)``, at their offsets in a layer buffer. Call before
    :func:`bind`, which rebinds the matrices to the slots.
    """
    projections, placements = _stack(loaded, path)
    layer_bytes = placements[0][2]
    return [
        (parameter, index * layer_bytes + offset)
        for index, (parameters, _, _) in enumerate(placements)
        for parameter, offset in parameters
    ]


def layout(loaded, path: str) -> list[int]:
    """The streamed stack under ``path``: ``[layers, layer_bytes]``.

    ``layer_bytes`` is the bytes of one layer buffer, which holds every
    projection weight matrix of a layer at offsets aligned to 256 bytes.

    Raises:
        ValueError: The module under ``path`` cannot stream its layers (see
            ``layer_stack``; the layers' projections must share one dense
            layout on one CUDA device).
    """
    projections, placements = _stack(loaded, path)
    return [len(projections), int(placements[0][2])]


@torch.no_grad()
def bind(
    loaded,
    path: str,
    before: Callable[..., Any],
    after: Callable[..., Any],
    filled: bool,
    host: torch.Tensor,
    *buffers: torch.Tensor,
) -> None:
    """Move the stack under ``path`` into host memory and stream it.

    ``host`` is the runtime's page-locked byte region of ``layers *
    layer_bytes`` bytes (:func:`layout`) and ``buffers`` its device layer
    buffers of ``layer_bytes`` bytes each. Unless ``filled``, every layer's
    projection matrices are copied into its share of ``host`` and the device
    storage they held is released. A ``filled`` region already holds them:
    the matrices are views of it at their :func:`placements`, as binding a
    worker's retained weights leaves them. Each matrix is then rebound in
    place to a view of buffer ``layer % len(buffers)``: the parameter keeps
    its identity, which operator bindings key on. ``before(layer, stream)``
    and ``after(layer, stream)`` bracket every projection call of a layer on
    the calling CUDA stream handle.

    Call this after loading and before any operator binds the layers'
    parameters: operators prepared earlier keep views of the released
    storage.

    Raises:
        ValueError: The region or the buffers disagree with the layout, or
            a matrix of a filled region lies outside its layer's share.
    """
    if filled:
        projections = [
            _projections(layer) for layer in layer_stack(_module(loaded, path))
        ]
        layer_bytes = host.numel() // len(projections)
        base = host.data_ptr()
        placements = []
        for index, modules in enumerate(projections):
            parameters = []
            for module in modules:
                for parameter in module.parameters():
                    if parameter.ndim < 2:
                        continue
                    offset = parameter.data_ptr() - base - index * layer_bytes
                    if not 0 <= offset < layer_bytes:
                        raise ValueError(
                            "a retained streamed matrix lies outside its "
                            "layer's share of the host region"
                        )
                    parameters.append((parameter, offset))
            placements.append((parameters, [], layer_bytes))
        device = buffers[0].device
    else:
        projections, placements = _stack(loaded, path)
        layer_bytes = placements[0][2]
        device = placements[0][0][0][0].device
    if host.numel() != len(projections) * layer_bytes or any(
        buffer.numel() != layer_bytes for buffer in buffers
    ):
        raise ValueError("host weight storage disagrees with the layout")

    for index, (modules, (parameters, stores, _)) in enumerate(
        zip(projections, placements, strict=True)
    ):
        share = host.narrow(0, index * layer_bytes, layer_bytes)
        for storage, offset in stores:
            source = torch.empty(0, dtype=torch.uint8, device=device).set_(
                storage
            )
            share[offset : offset + source.numel()].copy_(source)
        buffer = buffers[index % len(buffers)].untyped_storage()
        for parameter, offset in parameters:
            parameter.data = torch.empty(
                0, dtype=parameter.dtype, device=device
            ).set_(
                buffer,
                offset // parameter.element_size(),
                parameter.shape,
                parameter.stride(),
            )
        for module in modules:
            module.register_forward_pre_hook(_hook(before, index, device))
            module.register_forward_hook(_hook(after, index, device))


def _hook(call: Callable[..., Any], layer: int, device: torch.device):
    """A module hook calling ``call(layer, stream)`` on the current stream."""

    def hook(module, *arguments):
        call(layer, torch.cuda.current_stream(device).cuda_stream)

    return hook
