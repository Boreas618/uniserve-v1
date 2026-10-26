"""Captured graphs of a text computation (``uniserve.executor.text.graphs``)."""

from __future__ import annotations

import pytest

from uniserve.executor.text.graphs import GraphTable

pytestmark = pytest.mark.unit


class _Graph:
    """A captured graph whose release records its value."""

    def __init__(self, value: int, released: list[int]) -> None:
        self.value = value
        self._released = released

    def release(self) -> None:
        self._released.append(self.value)


def test_sealed_tables_replay_captured_keys_and_refuse_the_rest():
    released: list[int] = []
    table = GraphTable("text")
    assert table.lookup(8, None) is None
    first = _Graph(1, released)
    table.insert(8, None, first)
    table.insert(8, 64, _Graph(2, released))
    assert table.lookup(8, None) is first

    # A second capture of one key is refused and its graph released.
    with pytest.raises(ValueError):
        table.insert(8, None, _Graph(3, released))
    assert released == [3]

    table.seal()
    assert table.lookup(8, 64).value == 2
    with pytest.raises(ValueError):
        table.lookup(4, None)
    with pytest.raises(ValueError):
        table.insert(4, None, _Graph(4, released))
    assert released == [3, 4]


def test_close_releases_graphs_newest_first_unless_aborted():
    released: list[int] = []
    table = GraphTable("text")
    for key, value in [(16, 1), (8, 2), (4, 3)]:
        table.insert(key, None, _Graph(value, released))
    table.close(aborted=False)
    assert released == [3, 2, 1]

    released = []
    aborted = GraphTable("text")
    aborted.insert(1, None, _Graph(9, released))
    aborted.close(aborted=True)
    assert released == []


def test_a_reopened_table_releases_its_graphs_and_captures_again():
    # A sealed table that reopens releases its graphs, captures every key
    # again, and seals again.
    released: list[int] = []
    table = GraphTable("text")
    table.insert(8, None, _Graph(1, released))
    table.insert(4, None, _Graph(2, released))
    table.seal()

    table.reopen()
    assert released == [2, 1]
    assert table.lookup(8, None) is None
    table.insert(8, None, _Graph(3, released))
    assert table.lookup(8, None).value == 3

    table.seal()
    with pytest.raises(ValueError):
        table.lookup(4, None)


def test_close_attempts_every_release_before_raising_the_first_failure():
    released: list[int] = []

    class _Failing(_Graph):
        def release(self) -> None:
            super().release()
            raise RuntimeError(f"graph {self.value} failed")

    table = GraphTable("text")
    table.insert(1, None, _Failing(1, released))
    table.insert(2, None, _Failing(2, released))
    table.insert(3, None, _Graph(3, released))
    with pytest.raises(RuntimeError, match="graph 2 failed"):
        table.close(aborted=False)
    assert released == [3, 2, 1]
