"""The host turn of microbatch calls
(``uniserve.executor.experts.microbatches.Rotation``).
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

import pytest

from uniserve.executor.experts import MicrobatchAborted, Rotation

pytestmark = pytest.mark.unit

T = TypeVar("T")


def start(
    pool: ThreadPoolExecutor,
    rotation: Rotation,
    index: int,
    call: Callable[[], T],
) -> Future[T]:
    """Run ``index``'s call of ``rotation`` on a thread of its own."""
    return pool.submit(rotation.run, index, call)


def test_uneven_calls_keep_collective_submission_order():
    rotation = Rotation(3)
    dispatches = []
    rotation.begin()

    def call(index: int, layers: int) -> Callable[[], int]:
        def layered() -> int:
            for layer in range(layers):
                dispatches.append((index, layer))
                rotation.yield_turn()
            return layers

        return layered

    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [
            start(pool, rotation, index, call(index, layers))
            for index, layers in enumerate([3, 1, 2])
        ]
        outputs = [task.result(timeout=30) for task in tasks]
    rotation.end()

    assert outputs == [3, 1, 2]
    # Collective peers must observe the same order even when individual
    # calls finish before the others; a finished call has no further turn.
    assert dispatches == [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2)]


def test_unwinding_wakes_peers_and_allows_a_later_invocation():
    rotation = Rotation(2)
    rotation.begin()

    def fail() -> None:
        raise RuntimeError("numerical callback failed")

    with ThreadPoolExecutor(max_workers=2) as pool:
        peer = start(pool, rotation, 0, rotation.yield_turn)
        failed = start(pool, rotation, 1, fail)
        with pytest.raises(RuntimeError, match="numerical callback failed"):
            failed.result(timeout=30)
        with pytest.raises(MicrobatchAborted):
            peer.result(timeout=30)

    assert rotation.failure == 1
    rotation.end()

    rotation.begin()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = start(pool, rotation, 0, lambda: 7)
        second = start(pool, rotation, 1, lambda: 11)
        assert (first.result(timeout=30), second.result(timeout=30)) == (7, 11)
    rotation.end()
    rotation.close()


def test_yielding_outside_a_call_does_nothing():
    rotation = Rotation(2)
    rotation.yield_turn()
    rotation.begin()
    rotation.yield_turn()
    rotation.end()


def test_invocations_admit_one_call_per_microbatch_until_closed():
    with pytest.raises(ValueError):
        Rotation(0)

    rotation = Rotation(2)
    rotation.begin()
    with pytest.raises(RuntimeError, match="already running"):
        rotation.begin()
    with pytest.raises(RuntimeError, match="running"):
        rotation.close()

    rotation.end()
    rotation.close()
    with pytest.raises(RuntimeError, match="closed"):
        rotation.begin()
