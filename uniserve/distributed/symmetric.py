"""Peer-mapped storage over ``torch.distributed`` symmetric memory.

Execution contexts back every symmetric workspace field with this storage,
in serving and in direct library use alike: the Ulysses output storage of
attention with a context partition. Torch allocates it through NCCL and maps
every member's allocation over the communicator's NCCL process group, which
registers it as a window of that group's communicator. Members store into
one another's mappings directly; no NCCL collective takes this storage as an
operand.
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
