"""Torch process worlds and bound meshes for multi-rank library tests.

Model code communicates through ``DeviceMesh`` communicators that borrow
torch process groups. The serving worker plans which groups exist; a test
rank creates only the groups of the mesh it exercises, through the same
``uniserve.interface.distributed`` functions the worker calls.
"""

import socket
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

from uniserve.distributed import DeviceMesh
from uniserve.interface import distributed


@contextmanager
def rendezvous() -> Iterator[socket.socket]:
    """Bind the loopback socket a test world's first rank serves its store on.

    The test passes the socket to every rank it spawns. Handing over the
    bound socket, as the engine does for a worker, leaves no window in which
    another process could take a port the test only probed.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        yield listener


@contextmanager
def process_world(
    rank: int,
    world_size: int,
    listener: socket.socket,
    device: torch.device | str = "cpu",
) -> Iterator[distributed.World]:
    """Join the test's torch world and retire its groups after the body.

    ``listener`` is this rank's copy of the :func:`rendezvous` socket; rank
    0 transfers it to the store and the other ranks only read its address.
    A failing body leaves the groups to process exit: its peers may never
    reach the collectives an orderly teardown would wait for.
    """
    host, port = listener.getsockname()
    if rank == 0:
        listen_fd = listener.detach()
    else:
        listen_fd = None
        listener.close()

    world = distributed.initialize(
        rank,
        rank,
        world_size,
        str(device),
        None,
        host,
        port,
        listen_fd,
        None,
        None,
    )
    yield world
    distributed.destroy(world)


def bind_mesh(
    world: distributed.World,
    mesh: DeviceMesh,
    axes: Iterable[tuple[str, ...]] | None = None,
) -> DeviceMesh:
    """Bind ``mesh`` communicators over groups created in ``world``.

    Every rank of the world calls this with the same topology, members or
    not: creating a torch group is a collective over the whole world.

    Args:
        world: The world every mesh rank belongs to.
        mesh: The topology, with this process's rank.
        axes: Axis selections to bind; defaults to each single axis. Every
            fiber of a selection gets one group, shared by selections with
            the same members.

    Returns:
        The mesh with a communicator for each selection this rank is in.
    """
    selections = sorted(
        ((axis,) for axis in mesh.axes) if axes is None else axes
    )
    # Each planned group's sorted members and whether this rank keeps it.
    plan: list[tuple[tuple[int, ...], bool]] = []
    entries: dict[tuple[int, ...], int] = {}
    bound = {}
    for selection in selections:
        for members in mesh.members(selection):
            key = tuple(sorted(members))
            if len(members) > 1 and key not in entries:
                entries[key] = len(plan)
                plan.append((key, mesh.rank in members))
            if mesh.rank in members:
                bound[".".join(selection) or "local"] = (members, key)

    handles = distributed.create_groups(
        world,
        [len(key) for key, _ in plan],
        [world.physical(rank) for key, _ in plan for rank in key],
        [False] * len(plan),
        [keep for _, keep in plan],
    )
    return distributed.mesh(
        world,
        mesh.ranks,
        mesh.shape,
        mesh.axes,
        mesh.rank,
        list(bound),
        [len(members) for members, _ in bound.values()],
        [
            world.physical(rank)
            for members, _ in bound.values()
            for rank in members
        ],
        [
            handles[entries[key]] if key in entries else None
            for _, key in bound.values()
        ],
    )
