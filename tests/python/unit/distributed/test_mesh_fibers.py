"""Fiber order shared by the worker's group plan and ``DeviceMesh``.

The worker plans torch group creation from mesh fibers it enumerates in
Rust (``parallel::topology::Mesh::members``); numerical code selects the
same fibers through ``DeviceMesh.members``. Both must agree, or ranks would
bind communicators to groups created for other fibers. The expected values
here are the ones the Rust plan tests assert.
"""

import pytest

from uniserve.distributed import DeviceMesh

pytestmark = pytest.mark.unit


def test_fibers_follow_topology_order_and_selection_order():
    mesh = DeviceMesh(
        ranks=(0, 1, 2, 3), shape=(2, 2), axes=("pp", "tp"), rank=0
    )

    assert mesh.members(("tp",)) == ((0, 1), (2, 3))
    assert mesh.members(("pp",)) == ((0, 2), (1, 3))
    assert mesh.members(("tp", "pp")) == ((0, 2, 1, 3),)
    assert mesh.members(()) == ((0,), (1,), (2,), (3,))


def test_sequence_parallel_fibers_match_the_worker_plan():
    # [pp, cp, tp, ulysses] = [1, 1, 2, 2], the dimensions of a component
    # with two tensor and two Ulysses ranks.
    mesh = DeviceMesh(
        ranks=(0, 1, 2, 3),
        shape=(1, 1, 2, 2),
        axes=("pp", "cp", "tp", "ulysses"),
        rank=2,
    )

    assert mesh.members(("tp",)) == ((0, 2), (1, 3))
    assert mesh.members(("tp", "ulysses")) == ((0, 1, 2, 3),)
    assert mesh.members(("ulysses",)) == ((0, 1), (2, 3))
