"""The execution context a runner constructs for its module."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from uniserve_models.bindings.context import ExecutionContext, Scratch


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
        from uniserve_models.distributed import communicators
        from uniserve_models.interface.distributed import stream_collectives

        bound = stream_collectives(communicators(module), collectives)
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


def kernel_records(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Kernel records as a runner reports them to the rank.

    Each record describes one prepared operator: its layer ``path``, its
    ``op``, the ``provider`` serving it and fields particular to the
    operator. The rank only groups and logs them, so each crosses as the
    text of one JSON object.
    """
    return [json.dumps(record) for record in records]
