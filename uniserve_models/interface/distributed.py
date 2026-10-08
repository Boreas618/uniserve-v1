"""Torch process groups for numerical collectives, created on Rust's plan.

Model code communicates through torch process groups behind
``DeviceMesh`` communicators. Creating a group is a collective over the
whole process world, so the worker fixes the creation order (every rank
creates every group, members or not) and these functions execute it. Host
collectives (agreement, barriers, bootstrap bytes) never use torch: the
worker runs them over its own host group.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist

from uniserve_models.bindings.context import close_resources
from uniserve_models.distributed import Communicator, DeviceMesh


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "distributed.initialize": initialize,
        "distributed.create_groups": create_groups,
        "distributed.mesh": mesh,
        "distributed.destroy": destroy,
    }


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
        # NCCL chooses an eligible algorithm for each operation and topology;
        # zero-CTA is a preference, not a claim about a mesh axis.
        pg.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        options["pg_options"] = pg
    if device.type == "cuda" and (
        backend == "nccl" or "cuda:nccl" in backend.split(",")
    ):
        options["device_id"] = device
    return options


def _store(store: Mapping[str, Any], rank: int, size: int, backend: str):
    """Join the rendezvous store; rank 0 serves it on the handed-over fd.

    The worker bound the listening socket and transfers its ownership here.
    """
    timeout = (
        dist.constants.default_pg_nccl_timeout if backend == "nccl" else None
    ) or dist.constants.default_pg_timeout
    listen_fd = store.get("listen_fd")
    return dist.TCPStore(
        str(store["host"]),
        int(store["port"]),
        size,
        is_master=rank == 0,
        timeout=timeout if isinstance(timeout, timedelta) else None,
        master_listen_fd=None if listen_fd is None else int(listen_fd),
    )


def initialize(request: Mapping[str, Any]) -> World:
    """Join this worker's process world, or a replica in the expert union.

    Request: ``rank``, ``local_rank``, ``world_size``, ``device``,
    ``backend`` (``None`` selects NCCL on CUDA, Gloo otherwise), ``store``
    (``{host, port, listen_fd}`` for a multi-rank world) and ``union``
    (``{rank, size, store}`` when replicas share experts).

    Raises:
        ValueError: The rank is outside its world or union, or the device
            is not visible.
    """
    rank = int(request["rank"])
    size = int(request["world_size"])
    local_rank = int(request["local_rank"])
    if size < 1 or not 0 <= rank < size or local_rank < 0:
        raise ValueError(
            "launch rank must satisfy 0 <= rank < positive world_size"
        )
    device = _local_device(str(request["device"]), local_rank)
    backend = request.get("backend") or (
        "nccl" if device.type == "cuda" else "gloo"
    )
    backend = str(backend)
    union = request.get("union")

    if union is not None:
        union_rank, union_size = int(union["rank"]), int(union["size"])
        start = union_rank - rank
        if start < 0 or start + size > union_size:
            raise ValueError("worker ranks must fit inside the expert union")
        # Leave WORLD's device_id unbound: ncclCommSplit would force every
        # union rank to participate in each independent replica's binding.
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            store=_store(union["store"], union_rank, union_size, "nccl"),
            rank=union_rank,
            world_size=union_size,
        )
        world_group = dist.group.WORLD
        if device.type == "cuda":
            dist.all_reduce(torch.zeros((), device=device))
        experts = Communicator(
            tuple(range(union_size)), union_rank, "experts", device, world_group
        )
        members = tuple(range(start, start + size))
        subgroup = dist.new_group(
            ranks=sorted(members),
            use_local_synchronization=True,
            **_options(backend, device),
        )
        instance = Communicator(members, rank, "instance", device, subgroup)
        return World(
            rank,
            size,
            device,
            backend,
            instance,
            experts,
            [world_group, subgroup],
        )

    groups = []
    if size > 1:
        store = request.get("store")
        if store is None:
            raise ValueError("a multi-rank world requires its rendezvous store")
        dist.init_process_group(
            store=_store(store, rank, size, backend),
            rank=rank,
            world_size=size,
            **_options(backend, device),
        )
        groups.append(dist.group.WORLD)
    instance = Communicator(
        tuple(range(size)),
        rank,
        "instance",
        device,
        dist.group.WORLD if groups else None,
    )
    return World(rank, size, device, backend, instance, Communicator(), groups)


def create_groups(world: World, plan: Sequence[Mapping[str, Any]]) -> list:
    """Create every planned group in order; return the ones this rank keeps.

    Every rank executes every entry, members or not, so creation stays a
    matching sequence of world collectives.
    """
    handles = []
    for entry in plan:
        group = dist.new_group(
            ranks=[int(member) for member in entry["members"]],
            use_local_synchronization=bool(entry["local_sync"]),
            **_options(world.backend, world.device),
        )
        if entry["keep"]:
            world.groups.append(group)
            handles.append(group)
        else:
            handles.append(None)
    return handles


def mesh(world: World, spec: Mapping[str, Any]) -> DeviceMesh:
    """A ``DeviceMesh`` whose communicators borrow created groups.

    ``spec["groups"]`` maps a communicator name (axes joined by ``.``, or
    ``local``) to its physical members in logical order and its group.
    """
    rank = int(spec["rank"])
    result = DeviceMesh(
        ranks=tuple(int(value) for value in spec["ranks"]),
        shape=tuple(int(value) for value in spec["shape"]),
        axes=tuple(str(value) for value in spec["axes"]),
        rank=rank,
    )
    physical = world.physical(rank)
    groups = {}
    for name, value in dict(spec["groups"]).items():
        name = str(name)
        members = tuple(int(member) for member in value["members"])
        axes = () if name == "local" else tuple(name.split("."))
        groups[axes] = Communicator(
            members,
            members.index(physical),
            name,
            world.device,
            dict(value).get("group"),
        )
    # Mesh topology remains a numerical value; its bound communicators are
    # attached after every group exists.
    object.__setattr__(result, "_device", world.device)
    object.__setattr__(result, "_groups", MappingProxyType(groups))
    return result


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
