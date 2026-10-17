"""A runner's execution context and the helpers its forwards run in.

:func:`execution_context` prepares a runner's module on its stream. A
runner's forwards run on its stream with its prepared operators
(:func:`scope`) and publish results into product tensors (:func:`write`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from uniserve.bindings.context import ExecutionContext
from uniserve.bindings.workspace import Scratch


def execution_context(
    module,
    stream: int,
    attention,
    scratch: Scratch,
    *,
    device: int | None = None,
    derive_host_lengths: bool = False,
    cache=None,
    experts=None,
    weights=None,
    collectives: Sequence[Any] | None = None,
) -> ExecutionContext:
    """An execution context of ``module`` on a caller's stream.

    Args:
        module: The module whose operators the context prepares.
        stream: The CUDA stream handle the context enqueues on (0 for the
            current stream).
        attention: The attention backend the model loaded with: its name or
            the resolved backend.
        scratch: The transient work areas of the stream, which every
            context serialized on it borrows. Contexts are prepared largest
            first, so captured graphs keep addressing one backing.
        device: The CUDA ordinal ``stream`` belongs to, -1 for the host;
            ``None`` takes the device of the module's parameters. A module
            placed on a generation tower names the stream's device here and
            computes on the tower between the stream's steps.
        derive_host_lengths: Whether attention derives the host lengths of
            its rows itself (encoders and decoders) instead of receiving
            them exactly with each forward.
        cache: The prefix cache view object attention layers read.
        experts: The expert exchange routed experts communicate through.
        weights: The distributed expert weights' prefetch object.
        collectives: The ``NativeCollectives`` of each of the module's
            communicators on ``stream``; ``None`` when the module
            communicates over no multi-member group.
    """
    bound = None
    if collectives is not None:
        from uniserve.distributed import communicators
        from uniserve.distributed.collectives import group_collectives

        bound = group_collectives(communicators(module), collectives)
    return ExecutionContext(
        module,
        device=None
        if device is None or int(device) < 0
        else torch.device("cuda", int(device)),
        stream=int(stream) if stream else None,
        cache=cache,
        attention=attention,
        vsa="auto",
        matmul="auto",
        moe="auto",
        derive_host_lengths=bool(derive_host_lengths),
        experts=experts,
        weights=weights,
        scratch=scratch,
        collectives=bound,
    )


def held_module(root: Any, path: str) -> Any | None:
    """The module at ``path`` below ``root``, or ``None`` when absent here.

    A rank's share of a partitioned component keeps a submodule it does not
    hold as ``None``: a pipeline stage without the denoiser's conditioner.
    A path naming an attribute the model does not declare raises.
    """
    module = root
    for name in str(path).split(".") if path else ():
        module = getattr(module, name)
        if module is None:
            return None
    return module


def kernel_records(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Kernel records as a runner reports them to the rank.

    Each record describes one prepared operator: its layer ``path``, its
    ``op``, the ``provider`` serving it and fields particular to the
    operator. The rank only groups and logs them, so each crosses as the
    text of one JSON object.
    """
    return [json.dumps(record) for record in records]


@contextmanager
def scope(context) -> Iterator[None]:
    """Run one forward on the runner's stream with its prepared operators."""
    with torch.inference_mode(), context.activate():
        yield


def write(target: torch.Tensor, value: torch.Tensor) -> list[int]:
    """Copy ``value`` into the leading elements of a product region.

    Converts to the product's dtype on the current stream and returns the
    value's shape, which the worker publishes as the product's shape.

    Raises:
        ValueError: The value exceeds the region.
    """
    shape = list(value.shape)
    if value.numel() > target.numel():
        raise ValueError("a result exceeds its product bound")
    target.reshape(-1)[: value.numel()].view(value.shape).copy_(
        local(value.detach(), target.device), non_blocking=True
    )
    return shape


def local(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``value`` on ``device``, for a copy into worker storage there.

    A value computed on another device (a generation tower) moves into
    allocator storage of ``device`` first: the worker's storage is mapped
    for its own device only, and a cross-device copy runs on the source
    device.
    """
    return value.to(device, non_blocking=True)
