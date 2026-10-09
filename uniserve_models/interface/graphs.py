"""CUDA graph capture, replay and release shared by every binding.

Rust owns graph policy: shapes, buckets, capture order, allocator pools and
the decision to capture or replay. A binding's ``capture`` function records
one bound step with :func:`capture` in the pool Rust passes and returns the
resulting :class:`Graph` to Rust as an opaque handle; the binding's
``replay(graph, ...)`` calls :meth:`Graph.replay` and its
``release(graph)`` calls :meth:`Graph.release`. Graph inputs are the
binding's fixed buffers, so a replay needs no copies.

The interface entry ``graphs.pool`` creates the allocator pools Rust
assigns to graph owners. ``graphs.begin_capture`` and ``graphs.end_capture``
let Rust capture work it submits itself between them, possibly from several
host threads on streams forked from the capturing one (expert joins of all
microbatches); ``graphs.replay`` and ``graphs.release`` operate on such
graphs. Importing this module does not initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import (
    AbstractContextManager,
    ExitStack,
    contextmanager,
    nullcontext,
)
from typing import Any

import torch

__all__ = ["Capture", "Graph", "capture", "functions", "pool"]


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "graphs.pool": pool,
        "graphs.begin_capture": begin_capture,
        "graphs.end_capture": end_capture,
        "graphs.replay": replay,
        "graphs.release": release,
    }


def pool(device: int) -> list[Any]:
    """Create one CUDA graph allocator pool on ``device``.

    Returns:
        ``[pool, id_a, id_b, total_bytes]``: the pool (an opaque handle Rust
        passes back to captures), its allocator id pair (which names the
        pool's segments in ``memory.stats``) and the device's total memory,
        from which Rust derives the default graph budget.
    """
    device = int(device)
    with torch.cuda.device(device):
        handle = torch.cuda.MemPool()
    first, second = handle.id
    total = torch.cuda.get_device_properties(device).total_memory
    return [handle, int(first), int(second), int(total)]


class Graph:
    """One captured step: graph segments in submission order and its output.

    A step is usually one CUDA graph. Work CUDA cannot record (an eager copy
    submission between segments, see :func:`capture`) splits it into several
    graphs with the eager calls between them; replay submits them in capture
    order. The graph keeps its private capture stream: PyTorch frees a
    capture stream's workspaces when the graph resets, so a stream shared
    with other graphs could invalidate their inputs.
    """

    def __init__(
        self,
        graphs: Sequence[torch.cuda.CUDAGraph],
        steps: Sequence[Callable[[], object]],
        output: Any,
        capture_stream: torch.cuda.Stream,
    ) -> None:
        self._graphs = tuple(graphs)
        self._steps = tuple(steps)
        self._capture_stream = capture_stream
        self.output = output

    def replay(self) -> Any:
        """Submit every segment on the stream that captured the step.

        Replay also advances the philox offsets of every generator the
        capture registered, so random draws continue as an eager sequence
        would. Returns the captured output (views into fixed buffers or the
        graph's pool, overwritten by the next replay).
        """
        for step in self._steps:
            step()
        return self.output

    def release(self) -> None:
        """Drop the graph executables and the captured output.

        The caller has drained every stream that replays this graph.
        """
        graphs, self._graphs, self._steps = self._graphs, (), ()
        self.output = None
        failure = None
        for graph in graphs:
            try:
                graph.reset()
            except Exception as error:  # noqa: BLE001 - reset every graph
                failure = failure or error
        if failure is not None:
            raise failure


class _Segments:
    """Capture segments split at eager submissions.

    ``begin`` starts a graph on the private capture stream and joins the
    computation stream into the capture; ``end`` joins it back and ends the
    graph. ``submit`` (handed to a segmenting scope) ends the current
    segment, records an eager call and begins the next segment in the same
    pool, so allocations stay shared across the step.
    """

    def __init__(
        self,
        capture_stream: torch.cuda.Stream,
        computation: torch.cuda.Stream,
        pool_handle,
        generators: Sequence[torch.Generator],
    ) -> None:
        self.capture_stream = capture_stream
        self.computation = computation
        self.pool = pool_handle
        self.generators = tuple(generators)
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.steps: list[Callable[[], object]] = []
        self._active: torch.cuda.CUDAGraph | None = None

    def begin(self) -> None:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        for generator in self.generators:
            graph.register_generator_state(generator)
        self.graphs.append(graph)
        with torch.cuda.stream(self.capture_stream):
            graph.capture_begin(pool=self.pool)
            self._active = graph
            self.computation.wait_stream(self.capture_stream)

    def end(self) -> None:
        graph, self._active = self._active, None
        if graph is None:
            return
        with torch.cuda.stream(self.capture_stream):
            self.capture_stream.wait_stream(self.computation)
            graph.capture_end()
        if self.pool is None:
            # Later segments share the first segment's private pool.
            self.pool = graph.pool()
        self.steps.append(graph.replay)

    def submit(self, call: Callable[[], object]) -> None:
        self.end()
        self.steps.append(call)
        self.begin()

    def reset(self) -> None:
        if self._active is not None:
            # A failed capture leaves the graph capturing on its stream.
            try:
                with torch.cuda.stream(self.capture_stream):
                    self._active.capture_end()
            except Exception:  # noqa: BLE001 - the capture already failed
                pass
            self._active = None
        for graph in self.graphs:
            graph.reset()
        self.graphs.clear()
        self.steps.clear()


@contextmanager
def _prepared(
    device: torch.device,
    computation: torch.cuda.Stream,
    peers: Sequence[tuple[torch.device, torch.cuda.MemPool]],
) -> Iterator[None]:
    """Retire warmup allocations and order the capture after its caller.

    Capturing waits for all prior work so the allocator can return warmup
    blocks, and creates this thread's cuBLAS handle outside the capture
    (handles are thread-local and may not exist on a thread that only
    warmed up elsewhere). Allocations on each peer device go to its pool
    for the capture's duration. The caller's stream waits on the
    computation afterwards, whether or not the capture succeeded.
    """
    for target in (device, *(peer for peer, _ in peers)):
        torch.cuda.synchronize(target)
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    with torch.cuda.stream(computation):
        torch.cuda.current_blas_handle()
    current = torch.cuda.current_stream(device)
    try:
        with ExitStack() as pools:
            for peer, pool in peers:
                pools.enter_context(torch.cuda.use_mem_pool(pool, peer))
            yield
    finally:
        current.wait_stream(computation)


class Capture:
    """One graph capture in progress on a computation stream.

    :meth:`begin` starts capturing work enqueued on ``stream`` (and on
    streams that wait on it and are joined back before :meth:`end`);
    :meth:`end` instantiates the :class:`Graph`. Both run on the same host
    thread; work between them may be submitted from other threads.
    ``peers`` pairs each other device the work allocates on with the pool
    receiving those allocations.
    """

    def __init__(
        self,
        stream: torch.cuda.Stream,
        pool: torch.cuda.MemPool | None = None,
        generators: Iterable[torch.Generator] = (),
        peers: Sequence[tuple[torch.device, torch.cuda.MemPool]] = (),
    ) -> None:
        self.device = stream.device
        self.stream = stream
        self.peers = tuple(peers)
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.segments = _Segments(
            self.capture_stream,
            stream,
            None if pool is None else pool.id,
            generators,
        )
        self._scopes = ExitStack()

    def begin(self) -> None:
        try:
            for scope in (
                torch.inference_mode(),
                torch.cuda.device(self.device),
                _prepared(self.device, self.stream, self.peers),
            ):
                self._scopes.enter_context(scope)
            self.segments.begin()
        except BaseException:
            self._scopes.close()
            raise

    def end(self, output: Any = None, failure: BaseException | None = None):
        """End the capture; with ``failure`` discard it and raise.

        Raises:
            RuntimeError: Capture failed; every partial graph was reset and
                the original error is the cause.
        """
        try:
            try:
                self.segments.end()
            finally:
                self._scopes.close()
            if failure is not None:
                raise failure
            for graph in self.segments.graphs:
                graph.instantiate()
        except Exception as error:
            self.segments.reset()
            raise RuntimeError(
                f"CUDA graph capture failed on {self.device}: {error}"
            ) from error
        return Graph(
            self.segments.graphs,
            self.segments.steps,
            output,
            self.capture_stream,
        )


def capture(
    body: Callable[[], Any],
    *,
    stream: torch.cuda.Stream,
    pool: torch.cuda.MemPool | None = None,
    segments: Callable[
        [Callable[[Callable[[], object]], None]], AbstractContextManager
    ]
    | None = None,
    generators: Iterable[torch.Generator] = (),
    peers: Sequence[Sequence[Any]] = (),
) -> Graph:
    """Capture ``body`` as it runs on ``stream``.

    The body enqueues its work on ``stream`` (the binding's computation
    stream) exactly as an eager call would; the capture runs on a private
    sibling stream that the computation stream joins. The caller warmed up
    the body eagerly first, so lazy initialization (kernel compilation,
    plan creation) does not occur inside the capture.

    Args:
        body: The step; its return value is the graph's ``output``.
        stream: The binding's computation stream.
        pool: The allocator pool Rust assigned to this graph's owner;
            ``None`` creates a private pool.
        segments: For steps containing work a graph cannot record (an
            expert weight prefetch submitting copies between layers): a
            factory of a scope entered around the body that receives a
            ``submit(call)`` function. Each submission ends the current
            graph, runs ``call`` eagerly at replay, and begins the next.
        generators: Generators the body draws from besides the device's
            default generator, which every capture registers. Replay
            advances each registered generator's offset.
        peers: ``[device, pool]`` pairs: the allocator pool (from
            :func:`pool`) receiving the step's allocations on each other
            device its modules run on (a generation tower reached through
            transfers). Each pool must outlive the graph.

    Returns:
        The instantiated :class:`Graph`.

    Raises:
        RuntimeError: Capture failed; every partial graph was reset and the
            original error is the cause.
    """
    placed = tuple(
        (torch.device("cuda", int(ordinal)), handle)
        for ordinal, handle in peers
    )
    recording = Capture(stream, pool, generators, placed)
    scope = (
        nullcontext()
        if segments is None
        else segments(recording.segments.submit)
    )
    recording.begin()
    try:
        with scope, torch.cuda.stream(stream):
            output = body()
    except Exception as error:
        recording.end(failure=error)
        raise
    return recording.end(output)


def begin_capture(pool, stream: int, device: int) -> Capture:
    """Begin capturing work Rust submits on ``stream`` into ``pool``."""
    computation = torch.cuda.ExternalStream(
        int(stream), device=torch.device("cuda", int(device))
    )
    recording = Capture(computation, pool)
    recording.begin()
    return recording


def end_capture(recording: Capture) -> Graph:
    """End a capture begun by :func:`begin_capture` and return its graph."""
    return recording.end()


def replay(graph: Graph, stream: int, device: int) -> None:
    """Replay a graph on ``stream``."""
    computation = torch.cuda.ExternalStream(
        int(stream), device=torch.device("cuda", int(device))
    )
    with torch.cuda.device(computation.device), torch.cuda.stream(computation):
        graph.replay()


def release(graph: Graph) -> None:
    """Release a graph after every stream replaying it drained."""
    graph.release()
