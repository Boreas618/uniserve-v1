"""Captured expert joins per step capacity
(``uniserve.executor.experts.joins``).
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Callable

import pytest

from uniserve.executor.experts import JoinGraphs, capture_order

pytestmark = pytest.mark.unit


class _Graph:
    """A captured join that logs its replays and its release."""

    def __init__(self, capacity: int, log: list[tuple[str, int]]) -> None:
        self.capacity = capacity
        self.log = log

    def replay(self) -> None:
        self.log.append(("replay", self.capacity))

    def release(self) -> None:
        self.log.append(("release", self.capacity))


class _Recorder:
    """Captures a join body into a :class:`_Graph` of the capacity it
    joins, logging the body's join while it is captured.
    """  # noqa: D205

    def __init__(self) -> None:
        self.log: list[tuple[str, int]] = []
        self.capturing = False

    def join(self, capacity: int) -> None:
        self.log.append(("capture" if self.capturing else "warm", capacity))

    def record(self, body: Callable[[], object]) -> _Graph:
        self.capturing = True
        try:
            body()
        finally:
            self.capturing = False
        return _Graph(self.log[-1][1], self.log)


def test_capture_order_is_distinct_capacities_largest_first():
    assert capture_order([1, 8, 2, 8, 192]) == (192, 8, 2, 1)
    assert capture_order([]) == ()


def test_joins_warm_and_capture_every_capacity_largest_first():
    recorder = _Recorder()
    joins = JoinGraphs.capture(
        [1, 2, 4, 8, 192, 8], recorder.join, recorder.record, warm=True
    )
    assert recorder.log == [
        (event, capacity)
        for capacity in (192, 8, 4, 2, 1)
        for event in ("warm", "capture")
    ]
    assert len(joins) == 5

    recorder.log.clear()
    for capacity in (8, 192, 1, 2, 4):
        joins.replay(capacity)
    assert recorder.log == [
        ("replay", capacity) for capacity in (8, 192, 1, 2, 4)
    ]
    with pytest.raises(ValueError, match="capacity 16"):
        joins.replay(16)


def test_joins_captured_without_warming_join_only_inside_the_capture():
    recorder = _Recorder()
    JoinGraphs.capture([4, 8], recorder.join, recorder.record, warm=False)
    assert recorder.log == [("capture", 8), ("capture", 4)]


def test_a_failed_capture_releases_the_joins_captured_before():
    recorder = _Recorder()

    def join(capacity: int) -> None:
        recorder.join(capacity)
        if capacity == 4 and recorder.capturing:
            raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        JoinGraphs.capture([1, 4, 8, 16], join, recorder.record, warm=True)
    released = {
        capacity for event, capacity in recorder.log if event == "release"
    }
    assert released == {8, 16}
    # The smaller capacities are never warmed.
    assert ("warm", 1) not in recorder.log


def test_release_releases_every_graph_and_reports_the_first_failure():
    log: list[tuple[str, int]] = []

    class _Failing(_Graph):
        def release(self) -> None:
            super().release()
            raise RuntimeError(f"release of {self.capacity} failed")

    joins = JoinGraphs(
        {
            1: _Graph(1, log),
            8: _Failing(8, log),
            4: _Failing(4, log),
        }
    )
    with pytest.raises(RuntimeError, match="release of"):
        joins.release()
    assert sorted(log) == [("release", 1), ("release", 4), ("release", 8)]
    assert len(joins) == 0
    with pytest.raises(ValueError):
        joins.replay(1)
