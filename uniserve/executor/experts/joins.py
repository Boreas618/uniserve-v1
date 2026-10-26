"""Captured expert participation without tokens.

A rank that is not part of an agreed expert step still executes every
expert layer of it with no tokens of its own (a join), so its peers' tokens
routed to its experts are computed and returned. :class:`JoinGraphs` holds
one CUDA graph of that layer sequence per step capacity; a join replays the
graph of the agreed capacity instead of running the layers eagerly. All
ranks warm and capture the same capacities in the same order, largest
first (:func:`capture_order`), because each warmup is a collective expert
step.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterable
from typing import Protocol

__all__ = ["JoinGraph", "JoinGraphs", "capture_order"]


class JoinGraph(Protocol):
    """One captured join, such as a ``uniserve.cuda_graph`` graph."""

    def replay(self) -> object:
        """Submit the join on the calling thread's current stream."""

    def release(self) -> None:
        """Drop the graph after every stream it ran on drained."""


def capture_order(capacities: Iterable[int]) -> tuple[int, ...]:
    """The distinct ``capacities``, largest first.

    Every rank of the union warms and captures its joins in this order, so
    their collective warmup steps match.
    """
    return tuple(sorted(set(capacities), reverse=True))


class JoinGraphs:
    """One captured join per step capacity.

    Args:
        graphs: The captured join of each capacity.
    """

    def __init__(self, graphs: dict[int, JoinGraph]) -> None:
        self._graphs = dict(graphs)

    @classmethod
    def capture(
        cls,
        capacities: Iterable[int],
        join: Callable[[int], object],
        record: Callable[[Callable[[], object]], JoinGraph],
        warm: bool,
    ) -> JoinGraphs:
        """Capture ``join(capacity)`` for every capacity, largest first.

        With ``warm``, each capacity first joins eagerly: that is a
        collective step, so every rank of the union warms the same
        capacities at the same point of its startup. The capture itself
        issues no communication.

        Args:
            capacities: The step capacities to capture.
            join: Runs one join of a capacity on the capturing stream.
            record: Captures its argument, a body enqueueing work on the
                capturing stream, into a graph; for example
                ``functools.partial(uniserve.cuda_graph.capture,
                stream=stream, pool=pool)``.
            warm: Join each capacity eagerly before its capture.

        Raises:
            Exception: A join or capture failed; the joins captured before
                were released.
        """
        graphs: dict[int, JoinGraph] = {}
        try:
            for capacity in capture_order(capacities):
                if warm:
                    join(capacity)

                graphs[capacity] = record(functools.partial(join, capacity))
        except BaseException:
            # The failure is the error to report; a release failure of an
            # earlier graph adds nothing to it.
            for graph in graphs.values():
                with contextlib.suppress(Exception):
                    graph.release()
            raise

        return cls(graphs)

    def __len__(self) -> int:
        return len(self._graphs)

    def replay(self, capacity: int) -> None:
        """Join one step of ``capacity`` tokens per rank.

        Submits the captured join on the calling thread's current stream,
        which the caller sets to the stream the join was captured for: the
        primary stream for the union's joins, a microbatch's own for its
        joins.

        Raises:
            ValueError: No captured join serves ``capacity``.
        """
        graph = self._graphs.get(capacity)
        if graph is None:
            raise ValueError(
                f"no captured expert join serves step capacity {capacity}"
            )

        graph.replay()

    def release(self) -> None:
        """Release every graph after every stream it ran on drained.

        Every graph is released even when one fails; the container is
        empty afterwards. A rank that aborts keeps its graphs instead:
        device work may still address their storage.

        Raises:
            Exception: The first graph's release failure.
        """
        graphs, self._graphs = self._graphs, {}
        failure: Exception | None = None
        for capacity in capture_order(graphs):
            try:
                graphs[capacity].release()
            except Exception as error:  # noqa: BLE001 - release every graph
                failure = failure or error

        if failure is not None:
            raise failure
