"""Peer-mapped storage for direct library use.

The serving worker allocates symmetric workspaces itself and binds every
member's mapping (``{name}.peer{index}``). A library caller without the
worker maps them through ``torch.distributed`` symmetric memory over the
communicator's NCCL process group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .mesh import Communicator


@dataclass
class SymmetricStorage:
    """One member's allocation and every member's view of it.

    ``peers`` is in the communicator's logical member order; ``handle``
    retains the backend rendezvous, which keeps the peer mappings alive.
    """

    local: torch.Tensor
    peers: tuple[torch.Tensor, ...]
    handle: Any = None


def allocate_symmetric_storage(
    group: Communicator, shape: tuple[int, ...], dtype: torch.dtype
) -> SymmetricStorage:
    """Allocate ``shape`` on every member of ``group`` and map them all.

    Collective over ``group``: every member calls it at the same point.

    Raises:
        RuntimeError: ``group`` spans several members without an NCCL
            process group.
    """
    if group.size == 1:
        local = torch.empty(shape, dtype=dtype, device=group.device)
        return SymmetricStorage(local, (local,))
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symmetric

    backend = group._require()
    if "cuda:nccl" not in dist.get_backend_config(backend).split(","):
        raise RuntimeError("symmetric peer storage requires the NCCL backend")
    if symmetric.get_backend(group.device) != "NCCL":
        symmetric.set_backend("NCCL")
    # The NCCL rendezvous uses the group's communicator, which torch creates
    # at its first collective.
    dist.all_reduce(torch.zeros(1, device=group.device), group=backend)
    local = symmetric.empty(shape, dtype=dtype, device=group.device)
    handle = symmetric.rendezvous(local, backend)
    # Backend positions follow ascending global rank; logical member `i` is
    # the backend position of `group.ranks[i]`.
    positions = sorted(group.ranks)
    peers = tuple(
        handle.get_buffer(positions.index(rank), shape, dtype)
        for rank in group.ranks
    )
    return SymmetricStorage(local, peers, handle)
