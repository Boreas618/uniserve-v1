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
from collections.abc import Callable, Sequence
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
        "distributed.bind_collectives": bind_collectives,
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
        # torch's stub declares neither the policy field nor its constant.
        pg.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO  # type: ignore[attr-defined]
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


class _Transfer:
    """A started native collective; its consumer joins it on its stream."""

    def __init__(self, wait: Callable[..., Any], token: int) -> None:
        self._wait = wait
        # Cleared once the consumer has joined the collective.
        self._token: int | None = token

    def block_current_stream(self) -> None:
        token, self._token = self._token, None
        if token is not None:
            self._wait(token, torch.cuda.current_stream().cuda_stream)


class NativeCollectives:
    """One group's collectives on the worker's NCCL communicator.

    The worker binds one communicator per stream and group and constructs
    this object over its functions; every call enqueues on that stream
    (asynchronous ones on its transfer fork, joined by the consumer).
    Members are ordered by ascending world rank, the backend order
    ``Communicator`` assumes; roots and peers are world ranks.

    Args:
        name: The communicator's name: its mesh axes joined by ``.``.
        all_reduce: ``(value, op)`` with ``op`` 0 for sum, 2 for max and 3
            for min.
        all_gather, reduce_scatter: ``(output, value)``.
        start_all_gather, start_all_to_all: ``(output, value)``, returning
            the token ``wait`` joins.
        wait: ``(token, stream)`` joins a started collective on the CUDA
            stream handle ``stream``.
        all_to_all: ``(output, value, output_splits, input_splits)``.
        gather: ``(outputs, value, root)``.
        broadcast: ``(value, root)``.
        send, recv: ``(value, peer)``.
        send_recv: ``(output, value, destination, source)``.
    """

    _REDUCTIONS = {"sum": 0, "max": 2, "min": 3}

    def __init__(
        self,
        name: str,
        all_reduce: Callable[..., Any],
        all_gather: Callable[..., Any],
        start_all_gather: Callable[..., Any],
        start_all_to_all: Callable[..., Any],
        wait: Callable[..., Any],
        all_to_all: Callable[..., Any],
        gather: Callable[..., Any],
        broadcast: Callable[..., Any],
        reduce_scatter: Callable[..., Any],
        send: Callable[..., Any],
        recv: Callable[..., Any],
        send_recv: Callable[..., Any],
    ) -> None:
        self.name = str(name)
        self._all_reduce = all_reduce
        self._all_gather = all_gather
        self._start_all_gather = start_all_gather
        self._start_all_to_all = start_all_to_all
        self._wait = wait
        self._all_to_all = all_to_all
        self._gather = gather
        self._broadcast = broadcast
        self._reduce_scatter = reduce_scatter
        self._send = send
        self._recv = recv
        self._send_recv = send_recv

    def all_reduce(self, value: torch.Tensor, op: str = "sum") -> None:
        self._all_reduce(value, self._REDUCTIONS[op])

    def all_gather(self, output: torch.Tensor, value: torch.Tensor) -> None:
        self._all_gather(output, value)

    def start_all_gather(self, output, value) -> _Transfer:
        return _Transfer(self._wait, self._start_all_gather(output, value))

    def start_all_to_all(self, output, value) -> _Transfer:
        return _Transfer(self._wait, self._start_all_to_all(output, value))

    def all_to_all(self, output, value, output_splits, input_splits) -> None:
        self._all_to_all(output, value, list(output_splits), list(input_splits))

    def gather(self, outputs, value, root: int) -> None:
        self._gather(list(outputs or ()), value, int(root))

    def broadcast(self, value: torch.Tensor, root: int) -> None:
        self._broadcast(value, int(root))

    def reduce_scatter(self, output, value) -> None:
        self._reduce_scatter(output, value)

    def send(self, value: torch.Tensor, peer: int) -> None:
        self._send(value, int(peer))

    def recv(self, value: torch.Tensor, peer: int) -> None:
        self._recv(value, int(peer))

    def send_recv(self, output, value, dst: int, src: int) -> None:
        self._send_recv(output, value, int(dst), int(src))

    def close(self) -> None:
        """The worker destroys the communicator."""


def bind_collectives(
    prepared, collectives: Sequence[NativeCollectives]
) -> None:
    """Route a binding's collectives to the worker's stream communicators.

    ``collectives`` holds one stream's collectives of each communicator.
    Every multi-member communicator the binding uses must have one: within
    an activation, a collective never falls back to another stream.

    Raises:
        ValueError: A used communicator has no stream collectives.
    """
    context = prepared.context
    if context is None:
        raise ValueError("collectives bind to a bound component")
    context.collectives = stream_collectives(context.groups, collectives)


def stream_collectives(
    groups, collectives: Sequence[NativeCollectives]
) -> dict:
    """The stream collectives of ``groups``, for an execution context.

    ``collectives`` holds one stream's collectives of each communicator.
    Every multi-member communicator of ``groups`` must have one: within an
    activation, a collective never falls back to another stream.

    Raises:
        ValueError: A communicator has no stream collectives.
    """
    named = {native.name: native for native in collectives}
    bound = {}
    for group in groups:
        if group.size == 1:
            continue
        native = named.get(group.name)
        if native is None:
            raise ValueError(
                f"communicator {group.name!r} has no stream collectives"
            )
        bound[group._require().group_name] = native
    return bound
