"""Microbatch threads and the rotation of their calls
(``uniserve.executor.experts.microbatches.MicrobatchThreads``).
"""  # noqa: D205

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator

import pytest

from uniserve.executor.experts import MicrobatchAborted, MicrobatchThreads

pytestmark = pytest.mark.unit


def test_microbatches_take_turns_at_each_yield_on_their_own_threads():
    threads = MicrobatchThreads(3)
    order = []
    runners = []

    def call(round_: int, index: int, layers: int) -> Callable[[], int]:
        def layered() -> int:
            runners.append((index, threading.current_thread()))
            for layer in range(layers):
                order.append((round_, index, layer))
                threads.rotation.yield_turn()
            return index * 10

        return layered

    for round_ in range(2):
        calls = [
            call(round_, index, layers)
            for index, layers in enumerate([3, 1, 2])
        ]
        assert threads.run(calls) == [0, 10, 20]
    threads.close()

    # Every invocation submits its dispatches in the same cyclic order; a
    # finished microbatch takes no further turn.
    assert order == [
        (round_, index, layer)
        for round_ in range(2)
        for index, layer in [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2)]
    ]
    # Each microbatch runs on its own persistent daemon thread, which close
    # joins.
    assert {(index, thread.name) for index, thread in runners} == {
        (0, "microbatch-0"),
        (1, "microbatch-1"),
        (2, "microbatch-2"),
    }
    assert len({thread for _, thread in runners}) == 3
    assert all(thread.daemon for _, thread in runners)
    assert not any(thread.is_alive() for _, thread in runners)


def test_the_first_failure_wakes_suspended_peers_and_is_reported():
    threads = MicrobatchThreads(2)
    interrupted = []

    def suspended() -> None:
        # Suspended at its yield when its peer fails.
        try:
            threads.rotation.yield_turn()
            threads.rotation.yield_turn()
        except MicrobatchAborted as error:
            interrupted.append(error)
            raise

    def failing() -> None:
        raise ValueError("expert layer failed")

    with pytest.raises(ValueError, match="expert layer failed"):
        threads.run([suspended, failing])
    assert len(interrupted) == 1

    # The rotation is reusable after a failure, then closes its threads.
    assert threads.run([lambda: 1, lambda: 2]) == [1, 2]
    threads.close()
    with pytest.raises(RuntimeError, match="closed"):
        threads.run([lambda: 1, lambda: 2])


def test_each_call_runs_in_its_microbatch_context():
    local = threading.local()

    @contextlib.contextmanager
    def stream(index: int) -> Iterator[None]:
        local.stream = f"stream-{index}"
        try:
            yield
        finally:
            del local.stream

    threads = MicrobatchThreads(2, context=stream)
    assert threads.run([lambda: local.stream, lambda: local.stream]) == [
        "stream-0",
        "stream-1",
    ]

    prepared = {}

    def prepare(index: int) -> None:
        prepared[index] = (threading.current_thread().name, local.stream)

    threads.each(prepare)
    assert prepared == {
        0: ("microbatch-0", "stream-0"),
        1: ("microbatch-1", "stream-1"),
    }
    threads.close()


def test_thread_preparation_reports_its_failure_and_requires_open_threads():
    threads = MicrobatchThreads(2)

    def prepare(index: int) -> None:
        if index == 1:
            raise RuntimeError("no cuBLAS handle")

    with pytest.raises(RuntimeError, match="no cuBLAS handle"):
        threads.each(prepare)
    with pytest.raises(ValueError, match="one call"):
        threads.run([lambda: 1])

    threads.close()
    with pytest.raises(RuntimeError, match="closed"):
        threads.each(prepare)
