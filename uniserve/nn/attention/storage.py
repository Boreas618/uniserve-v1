"""Bounded attention communication storage: declarations and borrowed views.

Context-parallel and Ulysses attention exchange rows through fixed storage
that must exist before graph capture. This module declares that storage per
sharing key (``output_requirements``, ``context_requirements``) and builds
the views layers borrow over tensors the caller allocated for those
declarations (``bind_output_storage``, ``bind_context_storage``). It
allocates nothing: the serving worker allocates every declared field as a
bind-time workspace, and symmetric fields as peer-mapped storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.nn.attention._parallel import (
    AttentionBuffers,
    OutputBuffers,
    ParallelAttention,
)
from uniserve.tensors import BufferConfig

__all__ = [
    "StorageRequirement",
    "bind_context_storage",
    "bind_output_storage",
    "context_requirements",
    "output_requirements",
]


@dataclass(frozen=True)
class StorageRequirement:
    """One shared storage allocation and the layers borrowing it.

    ``fields`` declares each tensor; ``symmetric`` names the fields that
    must be peer-mapped across ``group`` (every member's allocation of the
    field visible to every other member, in member order).
    """

    group: Communicator
    fields: Mapping[str, BufferConfig]
    symmetric: tuple[str, ...]
    layers: tuple[ParallelAttention, ...]


def output_requirements(
    layers: Iterable[ParallelAttention],
    *,
    rows: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> dict[tuple[Communicator, bool], StorageRequirement]:
    """Declare peer outputs shared only by one caller's serialized layers.

    The dimensions describe query rows and heads after Ulysses exchange.
    Each peer receives its sequence rows with every head in that logical
    group. Layers sharing a Ulysses group and output composition share one
    allocation, keyed by ``(group, composes_peer_rows)``.

    Raises:
        ValueError: An extent is not positive or rows do not divide the
            Ulysses membership.
    """
    if min(rows, heads, head_dim) < 1:
        raise ValueError("attention output extents must be positive")

    members: dict[tuple[Communicator, bool], list[ParallelAttention]] = {}
    for layer in layers:
        group = layer.ulysses_group
        if rows % group.size:
            raise ValueError(
                "attention output rows must divide Ulysses membership"
            )
        # Only a layer with a context partition composes every owner's rows
        # and writes each share into that owner's storage, which is what
        # symmetric storage provides and what requires peer access across
        # the whole group. A plain Ulysses layer computes its own rows and
        # exchanges them afterwards, so demanding that storage for it would
        # refuse the layout on a device whose peer access does not span the
        # group without giving the layer anything it uses.
        members.setdefault((group, layer.context_group.size > 1), []).append(
            layer
        )

    result = {}
    for (group, composes), owners in members.items():
        rows_shape = (rows // group.size, heads * group.size, head_dim)
        result[(group, composes)] = StorageRequirement(
            group,
            {
                "output": BufferConfig(rows_shape, dtype),
                "receive": BufferConfig(rows_shape, dtype),
                "sync_input": BufferConfig((1,), torch.int32),
                "sync_output": BufferConfig((group.size,), torch.int32),
            },
            ("output", "receive") if composes else (),
            tuple(owners),
        )
    return result


def bind_output_storage(
    requirements: Mapping[tuple[Communicator, bool], StorageRequirement],
    tensors: Mapping[tuple[Communicator, bool], Mapping[str, torch.Tensor]],
    peers: Mapping[tuple[Communicator, bool], tuple[torch.Tensor, ...]],
) -> Mapping[ParallelAttention, OutputBuffers]:
    """Bind each layer's output views over caller-allocated storage.

    Args:
        requirements: ``output_requirements`` result.
        tensors: Per requirement key, the local tensor of every field.
        peers: Per composing key, every member's ``output`` storage in member
            order (the peer-mapped views of the symmetric field).

    Seeds each rank's own synchronization slot with its rank for the peer
    rendezvous, on the current stream.
    """
    bindings = {}
    for key, requirement in requirements.items():
        views = _checked(requirement, tensors[key])
        views["sync_input"].fill_(requirement.group.rank)
        composes = bool(requirement.symmetric)
        outputs = OutputBuffers(
            tuple(peers[key]) if composes else (),
            views["output"],
            views["receive"],
            views["sync_input"],
            views["sync_output"],
        )
        bindings.update(dict.fromkeys(requirement.layers, outputs))
    return MappingProxyType(bindings)


def context_requirements(
    layers: Iterable[ParallelAttention],
    *,
    rows: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> dict[tuple[Communicator, int], StorageRequirement]:
    """Declare the compact context K/V domain shared by serialized layers.

    Every rank holds the whole context's rows: its own window of ``rows``
    is where it publishes, and the gather fills the rest. Layers without a
    context partition need none. Layers sharing a key group share one
    allocation, keyed by ``(group, rows)``.

    Raises:
        ValueError: An extent is not positive.
    """
    if min(rows, heads, head_dim) < 1:
        raise ValueError("attention context extents must be positive")

    members: dict[tuple[Communicator, int], list[ParallelAttention]] = {}
    for layer in layers:
        if layer.context_group.size > 1:
            members.setdefault((layer.key_group, rows), []).append(layer)

    result = {}
    for (group, _), owners in members.items():
        domain = rows * group.size
        result[(group, rows)] = StorageRequirement(
            group,
            {
                "key": BufferConfig((domain, heads, head_dim), dtype),
                "value": BufferConfig((domain, heads, head_dim), dtype),
            },
            (),
            tuple(owners),
        )
    return result


def bind_context_storage(
    requirements: Mapping[tuple[Communicator, int], StorageRequirement],
    tensors: Mapping[tuple[Communicator, int], Mapping[str, torch.Tensor]],
) -> Mapping[ParallelAttention, AttentionBuffers]:
    """Bind each context-parallel layer's views over caller storage.

    Each rank's own window of the replicated domain starts at
    ``rank * rows``.
    """
    bindings = {}
    for (group, rows), requirement in requirements.items():
        views = _checked(requirement, tensors[(group, rows)])
        begin = group.rank * rows
        buffers = AttentionBuffers(
            views["key"],
            views["value"],
            views["key"][begin : begin + rows],
            views["value"][begin : begin + rows],
        )
        bindings.update(dict.fromkeys(requirement.layers, buffers))
    return MappingProxyType(bindings)


def _checked(
    requirement: StorageRequirement, tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return the declared fields, rejecting a tensor of another layout."""
    views = {}
    for name, config in requirement.fields.items():
        tensor = tensors.get(name)
        if (
            tensor is None
            or tuple(tensor.shape) != config.shape
            or tensor.dtype != config.dtype
        ):
            raise ValueError(
                f"attention storage field {name!r} disagrees with its "
                "declaration"
            )
        views[name] = tensor
    return views
