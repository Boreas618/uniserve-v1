"""Bounded attention communication storage: declarations and borrowed views.

Context-parallel attention gathers every owner's K/V into fixed storage that
must exist before graph capture. This module declares that storage per
sharing key (``context_requirements``) and builds the views layers borrow
over tensors the caller allocated for those declarations
(``bind_context_storage``). It allocates nothing: the serving worker
allocates every declared field as a bind-time workspace, and symmetric
fields as peer-mapped storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.nn.attention._parallel import (
    AttentionBuffers,
    ParallelAttention,
)
from uniserve.tensors import BufferConfig

__all__ = [
    "StorageRequirement",
    "bind_context_storage",
    "context_requirements",
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
    context partition need none. Layers sharing a context group share one
    allocation, keyed by ``(group, rows)``.

    Raises:
        ValueError: An extent is not positive.
    """
    if min(rows, heads, head_dim) < 1:
        raise ValueError("attention context extents must be positive")

    members: dict[tuple[Communicator, int], list[ParallelAttention]] = {}
    for layer in layers:
        if layer.context_group.size > 1:
            members.setdefault((layer.context_group, rows), []).append(layer)

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
