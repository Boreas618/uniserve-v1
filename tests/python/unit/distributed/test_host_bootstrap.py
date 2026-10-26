"""Host collectives over the worker's host functions.

The worker hands the library its host group as one tuple: ``rank, size,
all_gather, broadcast, barrier, split, all_gather_fixed``, and ``split``
returns that same tuple for a subgroup
(``crates/worker/src/parallel/host.rs``). These tests play one member of
such a group, whose members all contribute the bytes this member holds.
"""

import pytest

from uniserve.backends.experts import HostBootstrap

pytestmark = pytest.mark.unit


def _host(rank: int, size: int) -> tuple:
    """The host functions of a group whose members all hold this one's bytes."""

    def all_gather(value: bytes) -> list[bytes]:
        return [value] * size

    def broadcast(value: bytes | None, root: int) -> bytes:
        assert value is not None
        return value

    def barrier() -> None:
        return None

    def split(color: int, key: int) -> tuple:
        # The subgroup of this member alone.
        return _host(0, 1)

    def all_gather_fixed(record: bytes) -> bytes:
        return record * size

    return (
        rank,
        size,
        all_gather,
        broadcast,
        barrier,
        split,
        all_gather_fixed,
    )


def test_a_bootstrap_takes_the_worker_host_functions_and_splits_into_them():
    bootstrap = HostBootstrap(*_host(1, 2))
    assert (bootstrap.Get_rank(), bootstrap.Get_size()) == (1, 2)
    assert bootstrap.allgather({"step": 3}) == ({"step": 3}, {"step": 3})

    subgroup = bootstrap.Split(color=1, key=0)
    assert (subgroup.Get_rank(), subgroup.Get_size()) == (0, 1)
    assert subgroup.bcast([7], root=0) == [7]
