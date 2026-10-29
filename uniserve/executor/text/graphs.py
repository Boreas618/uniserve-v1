"""The captured graphs of a text computation and the keys that name them.

A key names every input property a captured step depends on besides the
computation's fixed tensors: its method and that method's scalar arguments
(shape bucket, numerical variant, sampler). Each key holds one graph per
routed expert token capacity (``None`` for a computation without
disaggregated experts).

A table is filled at startup and sealed; afterwards a configured shape
without a resident graph is an error, never a silent eager fallback,
because every rank of a group must make the same collective calls in the
same order.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from .shapes import PrefillVariant

__all__ = [
    "CommitKey",
    "DecodeKey",
    "DraftKey",
    "Graph",
    "GraphKey",
    "GraphTable",
    "PrefillKey",
    "PromptKey",
    "ScoreKey",
    "SelectKey",
    "VerifiedKey",
    "VerifyKey",
]


@dataclass(frozen=True, slots=True)
class DecodeKey:
    """``decode`` of a row bucket, greedy or ``sampled`` from top-k rows."""

    rows: int
    sampled: bool


@dataclass(frozen=True, slots=True)
class PrefillKey:
    """``prefill`` of a row and token bucket for one numerical variant.

    ``outputs`` is false for a graph that only writes the prefix cache.
    """

    rows: int
    tokens: int
    outputs: bool
    variant: PrefillVariant


@dataclass(frozen=True, slots=True)
class VerifyKey:
    """A verification of a row bucket: ``prefill`` of its rows and ``verify``
    of their tasks in one graph, including recurrent state acceptance.

    ``rows`` is the verification rows the bucket holds (it evaluates one
    padding row more); ``sorts`` whether the general sampler sorts whole
    vocabulary rows.
    """  # noqa: D205

    rows: int
    sorts: bool


@dataclass(frozen=True, slots=True)
class DraftKey:
    """A drafter pass over a verification row bucket."""

    rows: int


@dataclass(frozen=True, slots=True)
class VerifiedKey:
    """``score_verified`` of a bucket of a verification's reported rows and
    a top width.
    """  # noqa: D205

    rows: int
    top: int


@dataclass(frozen=True, slots=True)
class CommitKey:
    """``commit`` of a row bucket."""

    rows: int


@dataclass(frozen=True, slots=True)
class SelectKey:
    """``select`` of a task bucket with the samplers its tasks take.

    ``top_k`` when some task takes the top-k sampler; ``categorical`` is
    ``None`` when no task takes the general sampler, else whether it sorts
    whole vocabulary rows.
    """

    tasks: int
    top_k: bool
    categorical: bool | None


@dataclass(frozen=True, slots=True)
class ScoreKey:
    """``score`` of a row bucket of reporting tasks and a top width."""

    rows: int
    top: int


@dataclass(frozen=True, slots=True)
class PromptKey:
    """``score_prompt`` of a block of prompt rows reporting ``top`` each."""

    rows: int
    top: int


GraphKey = (
    DecodeKey
    | PrefillKey
    | VerifyKey
    | DraftKey
    | CommitKey
    | SelectKey
    | ScoreKey
    | PromptKey
    | VerifiedKey
)


class Graph(Protocol):
    """A captured graph the table owns until it releases it."""

    def replay(self) -> None:
        """Launch the captured work on the stream it was captured on."""

    def release(self) -> None:
        """Release the graph's executables and the step outputs it keeps."""


# Graphs of tables closed after an abort: kept referenced, with the inputs
# and pools they address, until the process exits.
_ABANDONED: list[dict[Any, Graph]] = []


class GraphTable:
    """The captured graphs of one text computation, in capture order.

    Args:
        owner: Names the computation in errors.
    """

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self._graphs: dict[tuple[Hashable, int | None], Graph] = {}
        self.sealed = False

    def lookup(
        self, key: Hashable, capacity: int | None = None
    ) -> Graph | None:
        """Resolve a configured graph shape.

        Returns:
            The resident graph to replay, or ``None`` when startup has not
            captured the key yet: the caller captures it now.

        Raises:
            ValueError: The table is sealed and holds no such graph; startup
                captured every shape serving can select.
        """
        graph = self._graphs.get((key, capacity))
        if graph is None and self.sealed:
            raise ValueError(
                f"{self.owner}: configured graph bucket {key!r} (capacity "
                f"{capacity!r}) is not resident"
            )
        return graph

    def insert(self, key: Hashable, capacity: int | None, graph: Graph) -> None:
        """Record a captured graph.

        Raises:
            ValueError: The table is sealed, or the key was captured before;
                the refused graph is released.
        """
        if self.sealed:
            reason = "graphs are captured only before the computation is sealed"
        elif (key, capacity) in self._graphs:
            reason = f"graph bucket {key!r} was already captured"
        else:
            self._graphs[(key, capacity)] = graph
            return
        graph.release()
        raise ValueError(f"{self.owner}: {reason}")

    def keys(self) -> Iterator[tuple[Hashable, int | None]]:
        """Return the ``(key, capacity)`` pairs in capture order."""
        return iter(list(self._graphs))

    def __len__(self) -> int:
        return len(self._graphs)

    def seal(self) -> None:
        """Refuse further captures; serving may only replay."""
        self.sealed = True

    def reopen(self) -> None:
        """Release every graph, newest first, and accept captures again.

        The workspaces the graphs addressed were created anew, and warmup
        captures every key again. The caller drained the streams that
        replay them.

        Raises:
            Exception: The first release failure, after every release was
                attempted.
        """
        try:
            self.close(aborted=False)
        finally:
            self.sealed = False

    def close(self, aborted: bool) -> None:
        """Release every graph, newest first, and seal the table.

        Every release is attempted before the first failure is raised. The
        caller has drained the streams that replay the graphs. An aborted
        close releases nothing: it keeps the graphs (and the inputs and
        pools they address) referenced until the process exits.
        """
        graphs, self._graphs = self._graphs, {}
        self.sealed = True
        if aborted:
            _ABANDONED.append(graphs)
            return
        failure: BaseException | None = None
        for graph in reversed(list(graphs.values())):
            try:
                graph.release()
            except Exception as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
