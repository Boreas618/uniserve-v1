"""One rank's torch process world: groups created on the worker's plan.

Model code communicates through torch process groups behind
``DeviceMesh`` communicators. Creating a group is a collective over the
whole process world, so the worker fixes the creation order (every rank
creates every group, members or not) and the functions here execute it,
called by the worker by import path. Host
collectives (agreement, barriers, bootstrap bytes) never use torch: the
worker runs them over its own host group.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist

from uniserve.bindings.workspace import close_resources
from uniserve.distributed import Communicator, DeviceMesh


@dataclass
class World:
    """One rank's torch process world.

    ``instance`` spans this worker's ranks; ``experts`` spans the expert
    union (a singleton communicator without one). ``groups`` holds the
    groups this rank owns, in creation order.
    """

    rank: int
    size: int
    device: torch.device
    backend: str
    instance: Communicator
    experts: Communicator
    groups: list[Any] = field(default_factory=list)

    def physical(self, rank: int) -> int:
        """The torch world rank of instance rank ``rank``."""
        return self.instance.ranks[rank]


def _local_device(name: str, local_rank: int) -> torch.device:
    """Select the rank's device; a bare CUDA device follows ``local_rank``."""
    device = torch.device(name)
    if device.type != "cuda":
        return device
    index = local_rank if device.index is None else device.index
    device = torch.device("cuda", index)
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or index >= count:
        raise ValueError(
            f"cuda device {device} is outside the {count} visible CUDA "
            "device(s)"
        )
    torch.cuda.set_device(device)
    return device


def _options(backend: str, device: torch.device) -> dict[str, Any]:
    """Process-group options shared by the world and every subgroup."""
    options: dict[str, Any] = {"backend": backend}
    if backend == "nccl":
        pg = dist.ProcessGroupNCCL.Options()
        pg.use_pg_for_symm_mem_rendezvous = True
        options["pg_options"] = pg
    if device.type == "cuda" and (
        backend == "nccl" or "cuda:nccl" in backend.split(",")
    ):
        options["device_id"] = device
    return options


def _store(
    host: str,
    port: int,
    listen_fd: int | None,
    rank: int,
    size: int,
    backend: str,
):
    """Join the rendezvous store; rank 0 serves it on the handed-over fd.

    The worker bound the listening socket and transfers its ownership here.
    """
    timeout = (
        dist.constants.default_pg_nccl_timeout if backend == "nccl" else None
    ) or dist.constants.default_pg_timeout
    return dist.TCPStore(
        str(host),
        int(port),
        size,
        is_master=rank == 0,
        timeout=timeout if isinstance(timeout, timedelta) else None,
        master_listen_fd=None if listen_fd is None else int(listen_fd),
    )


def initialize(
    rank: int,
    local_rank: int,
    world_size: int,
    device: str,
    backend: str | None,
    store_host: str | None,
    store_port: int | None,
    listen_fd: int | None,
    union_rank: int | None,
    union_size: int | None,
) -> World:
    """Join this worker's process world, or a replica in the expert union.

    Args:
        rank: This rank among the worker's ranks.
        local_rank: This rank among the ranks of its node, which selects a
            bare CUDA device.
        world_size: The worker's rank count.
        device: The rank's device name.
        backend: The process-group backend; ``None`` selects NCCL on CUDA
            and Gloo otherwise.
        store_host, store_port: The rendezvous store of the torch world
            this rank joins: the expert union's when replicas share
            experts, else the worker's. ``None`` for a one-rank worker
            outside a union, which needs no process group.
        listen_fd: The store's listening socket on the rank that serves it
            (rank 0 of that world); ``None`` elsewhere.
        union_rank, union_size: This rank among the expert union's ranks
            and their count; ``None`` when replicas share no experts.

    Raises:
        ValueError: The rank is outside its world or union, a multi-rank
            world has no store, or the device is not visible.
    """
    rank = int(rank)
    size = int(world_size)
    local_rank = int(local_rank)
    if size < 1 or not 0 <= rank < size or local_rank < 0:
        raise ValueError(
            "launch rank must satisfy 0 <= rank < positive world_size"
        )
    placed = _local_device(str(device), local_rank)
    backend = str(backend or ("nccl" if placed.type == "cuda" else "gloo"))

    if union_rank is not None and union_size is not None:
        union_rank, union_size = int(union_rank), int(union_size)
        if store_host is None or store_port is None:
            raise ValueError("an expert union requires its rendezvous store")
        start = union_rank - rank
        if start < 0 or start + size > union_size:
            raise ValueError("worker ranks must fit inside the expert union")
        # Leave WORLD's device_id unbound: ncclCommSplit would force every
        # union rank to participate in each independent replica's binding.
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            store=_store(
                store_host,
                store_port,
                listen_fd,
                union_rank,
                union_size,
                "nccl",
            ),
            rank=union_rank,
            world_size=union_size,
        )
        world_group = dist.group.WORLD
        if placed.type == "cuda":
            dist.all_reduce(torch.zeros((), device=placed))
        experts = Communicator(
            tuple(range(union_size)), union_rank, "experts", placed, world_group
        )
        members = tuple(range(start, start + size))
        subgroup = dist.new_group(
            ranks=sorted(members),
            use_local_synchronization=True,
            **_options(backend, placed),
        )
        instance = Communicator(members, rank, "instance", placed, subgroup)
        return World(
            rank,
            size,
            placed,
            backend,
            instance,
            experts,
            [world_group, subgroup],
        )

    groups = []
    if size > 1:
        if store_host is None or store_port is None:
            raise ValueError("a multi-rank world requires its rendezvous store")
        dist.init_process_group(
            store=_store(
                store_host, store_port, listen_fd, rank, size, backend
            ),
            rank=rank,
            world_size=size,
            **_options(backend, placed),
        )
        groups.append(dist.group.WORLD)
    instance = Communicator(
        tuple(range(size)),
        rank,
        "instance",
        placed,
        dist.group.WORLD if groups else None,
    )
    return World(rank, size, placed, backend, instance, Communicator(), groups)


# The backend name of a world that plans a rank's partition
# (:func:`planning`): its groups are never created.
PLANNING = "planning"


def planning(
    rank: int,
    local_rank: int,
    world_size: int,
    device: str,
    union_rank: int | None,
    union_size: int | None,
) -> World:
    """A rank's world for planning its partition, without process groups.

    Its communicators name the ranks of the worker and of its expert union
    and this rank among them, which is all partitioning reads; no group is
    created (:func:`create_groups` creates none for it), so nothing
    communicates and no other rank need exist: a collective over one of its
    communicators raises. A daemon plans the weights each rank of a world
    holds with it before the world starts. The rank's device is selected as
    :func:`initialize` selects it.

    Raises:
        ValueError: The rank is outside its world or union, or the device
            is not visible.
    """
    rank, size, local_rank = int(rank), int(world_size), int(local_rank)
    if size < 1 or not 0 <= rank < size or local_rank < 0:
        raise ValueError(
            "launch rank must satisfy 0 <= rank < positive world_size"
        )
    placed = _local_device(str(device), local_rank)
    experts = Communicator()
    members = tuple(range(size))
    if union_rank is not None and union_size is not None:
        union_rank, union_size = int(union_rank), int(union_size)
        start = union_rank - rank
        if start < 0 or start + size > union_size:
            raise ValueError("worker ranks must fit inside the expert union")
        experts = Communicator(
            tuple(range(union_size)), union_rank, "experts", placed
        )
        members = tuple(range(start, start + size))
    instance = Communicator(members, rank, "instance", placed)
    return World(rank, size, placed, PLANNING, instance, experts)


def _runs(counts: Sequence[int], values: Sequence[Any]) -> list[list[Any]]:
    """Split back-to-back ``values`` into runs of ``counts`` items."""
    runs, first = [], 0
    for count in counts:
        runs.append(list(values[first : first + int(count)]))
        first += int(count)
    return runs


def create_groups(
    world: World,
    member_counts: Sequence[int],
    members: Sequence[int],
    local_sync: Sequence[bool],
    keep: Sequence[bool],
) -> list:
    """Create every planned group in order; return the ones this rank keeps.

    Every rank executes every entry, members or not, so creation stays a
    matching sequence of world collectives.

    Args:
        world: The world :func:`initialize` returned.
        member_counts: How many of ``members`` each planned group has.
        members: Every group's torch world ranks, back to back.
        local_sync: Whether each group is created by its members alone.
        keep: Whether this rank keeps each group.

    Returns:
        One entry per planned group: the group this rank keeps, or
        ``None``.
    """
    handles = []
    for ranks, local, kept in zip(
        _runs(member_counts, members), local_sync, keep, strict=True
    ):
        if world.backend == PLANNING:
            # A planning world creates no group: its communicators name
            # their members alone.
            handles.append(None)
            continue
        group = dist.new_group(
            ranks=[int(member) for member in ranks],
            use_local_synchronization=bool(local),
            **_options(world.backend, world.device),
        )
        if kept:
            world.groups.append(group)
            handles.append(group)
        else:
            handles.append(None)
    return handles


def mesh(
    world: World,
    ranks: Sequence[int],
    shape: Sequence[int],
    axes: Sequence[str],
    rank: int,
    names: Sequence[str],
    member_counts: Sequence[int],
    members: Sequence[int],
    groups: Sequence[Any],
) -> DeviceMesh:
    """A ``DeviceMesh`` whose communicators borrow created groups.

    Args:
        world: The world :func:`initialize` returned.
        ranks, shape, axes, rank: The mesh topology and this rank in it
            (``DeviceMesh``).
        names: Each communicator's name: its mesh axes joined by ``.``, or
            ``local``.
        member_counts: How many of ``members`` each communicator has.
        members: Every communicator's torch world ranks in logical order,
            back to back.
        groups: Each communicator's created group; ``None`` for a
            communicator of one member.
    """
    rank = int(rank)
    result = DeviceMesh(
        ranks=tuple(int(value) for value in ranks),
        shape=tuple(int(value) for value in shape),
        axes=tuple(str(value) for value in axes),
        rank=rank,
    )
    physical = world.physical(rank)
    bound = {}
    for name, ranks_of, group in zip(
        names, _runs(member_counts, members), groups, strict=True
    ):
        name = str(name)
        logical = tuple(int(member) for member in ranks_of)
        selection = () if name == "local" else tuple(name.split("."))
        bound[selection] = Communicator(
            logical,
            logical.index(physical),
            name,
            world.device,
            group,
        )
    # Mesh topology remains a numerical value; its bound communicators are
    # attached after every group exists.
    object.__setattr__(result, "_device", world.device)
    object.__setattr__(result, "_groups", MappingProxyType(bound))
    return result


def _nccl_backends(world: World) -> list[Any]:
    """The NCCL backend of every group the world owns on its device."""
    if world.device.type != "cuda":
        return []
    backends = []
    for group in world.groups:
        try:
            backend = group._get_backend(world.device)
        except RuntimeError:
            # A group without a backend for the device: a Gloo-only group.
            continue
        if isinstance(backend, dist.ProcessGroupNCCL):
            backends.append(backend)
    return backends


def suspend(world: World) -> None:
    """Release the device memory of the world's NCCL communicators.

    A parked standby calls this after its device drained; nothing runs on
    the communicators, nor a graph that captured one of their collectives,
    until :func:`resume`. Local to this rank (``ncclCommSuspend``).
    """
    if world.device.type == "cuda":
        torch.cuda.synchronize(world.device)
    for backend in _nccl_backends(world):
        backend.suspend()


def resume(world: World) -> None:
    """Allocate again what :func:`suspend` released.

    Called before the communicators, or the graphs that captured their
    collectives, run again.
    """
    for backend in _nccl_backends(world):
        backend.resume()


def destroy(world: World) -> None:
    """Retire owned groups, newest first, after the device drained."""
    groups, world.groups = world.groups, []
    actions = []
    if world.device.type == "cuda" and groups:
        actions.append(functools.partial(torch.cuda.synchronize, world.device))
    actions += [
        functools.partial(dist.destroy_process_group, group)
        for group in reversed(groups)
    ]
    close_resources(*actions)
