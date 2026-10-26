"""CUDA graph capture, replay and release shared by every binding.

The executor captures every graph it serves with :func:`capture` and
replays it with :meth:`BreakableCUDAGraph.replay`; a Python caller outside
a worker does the same. Where a captured step reaches work a graph cannot
record (a distributed expert layer's weight copy), :func:`break_graph`
ends the segment, begins the next and records the work, which every replay
performs between the two segments.

:func:`pool` creates the allocator pools graphs capture into: in a worker,
a pool allocates from the device memory Rust reserved for it, through the
worker library's allocator entry points for the pool's slot, so the memory
keeps its addresses for as long as a graph captured in it lives. Importing
this module does not initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

import torch

__all__ = [
    "Capture",
    "BreakableCUDAGraph",
    "break_graph",
    "capture",
    "pool",
    "pool_bytes",
]

# The capture in progress (:func:`capture`), which :func:`break_graph`
# reaches.
_capturing: Capture | None = None

# Caching-allocator option that lets a capture reuse a freed block another
# stream used, once the capture's dependencies show that use complete.
_CROSS_STREAM_REUSE = "graph_capture_record_stream_reuse"


def _reuse_joined_cross_stream_blocks() -> None:
    """Let captures reuse freed blocks whose other-stream uses are joined.

    PyTorch's NCCL point-to-point calls, which a Python caller's
    process-group transport issues for a projection exchange, record the
    NCCL stream on every tensor they send or receive. By default the caching
    allocator never reuses such a block inside a capture, even after the
    consumer stream has joined the NCCL work, so a captured forward that
    receives into a fresh output each layer keeps every layer's output
    reserved until the capture ends. With this option the allocator reuses
    the block once the capture's dependency graph orders the new allocation
    after those uses. The worker runtime's own collectives record no stream on
    any tensor, so serving captures allocate as before.

    The option is added to the process's current allocator settings, which
    are otherwise re-parsed with their defaults; an explicit setting of it is
    left as configured.
    """
    settings = torch._C._accelerator_getAllocatorSettings()
    if _CROSS_STREAM_REUSE not in settings:
        torch._C._accelerator_setAllocatorSettings(
            ",".join(filter(None, (settings, f"{_CROSS_STREAM_REUSE}:True")))
        )


# The pluggable allocator of each slot's pool, kept for the process's life:
# torch's allocator calls into the library it loaded.
_allocators: dict[tuple[str, int], Any] = {}


def pool(device: int, slot: int | None = None, library: str = "") -> list[Any]:
    """Create one CUDA graph allocator pool on ``device``.

    In a worker, Rust passes the ``slot`` of the device memory it reserved
    for the pool and the path of the worker ``library``: the pool allocates
    its segments through the library's entry points of that slot
    (``uniserve_graph_malloc_<slot>`` and ``uniserve_graph_free_<slot>``).
    Without a slot, the pool allocates from torch's own device memory.

    Returns:
        ``[pool, id_a, id_b, total_bytes]``: the pool (an opaque handle Rust
        passes to the executor's captures), its allocator id pair (which
        names the pool in :func:`pool_bytes`) and the device's total memory,
        from which Rust derives the default graph budget.
    """
    device = int(device)
    allocator = None
    if slot is not None:
        key = (str(library), int(slot))
        allocator = _allocators.get(key)
        if allocator is None:
            allocator = torch.cuda.memory.CUDAPluggableAllocator(
                str(library),
                f"uniserve_graph_malloc_{int(slot)}",
                f"uniserve_graph_free_{int(slot)}",
            )
            _allocators[key] = allocator
    with torch.cuda.device(device):
        handle = (
            torch.cuda.MemPool()
            if allocator is None
            else torch.cuda.MemPool(allocator=allocator.allocator())
        )
    first, second = handle.id
    total = torch.cuda.get_device_properties(device).total_memory
    return [handle, int(first), int(second), int(total)]


def pool_bytes(device: int) -> list[int]:
    """Bytes the caching allocator holds in each allocator pool of ``device``.

    Returns:
        A flat list ``[id_a, id_b, bytes, ...]``: one triple per pool with
        segments on the device. The id pair is the one :func:`pool`
        returned for a graph pool, ``0, 0`` for the default pool.
    """
    device = int(device)
    sizes: dict[tuple[int, int], int] = {}
    for segment in torch.cuda.memory_snapshot():
        if segment["device"] == device:
            first, second = segment["segment_pool_id"]
            key = (int(first), int(second))
            sizes[key] = sizes.get(key, 0) + int(segment["total_size"])
    return [value for key, size in sizes.items() for value in (*key, size)]


class BreakableCUDAGraph:
    """One captured step: graph segments in submission order and its output.

    A step is usually one CUDA graph. Work a graph cannot record splits it
    into several graphs, its segments, with a graph break between two of
    them (:func:`break_graph`). Each break carries the work performed there
    at every launch, a callable that :meth:`replay` invokes between the two
    segments. The graph keeps its private capture stream: PyTorch frees a
    capture stream's workspaces when the graph resets, so a stream shared
    with other graphs could invalidate their inputs.
    """

    def __init__(
        self,
        graphs: Sequence[torch.cuda.CUDAGraph],
        output: Any,
        capture_stream: torch.cuda.Stream,
        breaks: Sequence[Callable[[], Any]] = (),
    ) -> None:
        if len(graphs) != len(breaks) + 1:
            raise ValueError(
                "a captured step has one break between each two segments"
            )
        self._graphs = tuple(graphs)
        self._breaks = tuple(breaks)
        self._capture_stream = capture_stream
        self.output = output

    @property
    def segments(self) -> int:
        """The step's segments: one more than its graph breaks."""
        return len(self._graphs)

    def replay(self) -> Any:
        """Submit the graph on the stream that captured the step.

        Segments launch in submission order with each break's work invoked
        between its two segments, so the work runs on the host after the
        segment before it is enqueued and before the segment after it; the
        caller keeps the captured stream current. Replay also advances the
        philox offsets of every generator the capture registered, so random
        draws continue as an eager sequence would. Returns the captured
        output (views into fixed buffers or the graph's pool, overwritten by
        the next replay).
        """
        first, *rest = self._graphs
        first.replay()
        for work, graph in zip(self._breaks, rest, strict=True):
            work()
            graph.replay()
        return self.output

    def release(self) -> None:
        """Drop the captured graphs and the captured output.

        The caller has drained every stream this graph ran on.
        """
        graphs, self._graphs = self._graphs, ()
        # Break work may hold worker storage (weight copies); it goes with
        # the graphs.
        self._breaks = ()
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
    """Capture segments split at graph breaks.

    ``begin`` starts a graph on the private capture stream and joins the
    computation stream into the capture; ``end`` joins it back and ends the
    graph. ``break_graph`` ends the current segment and begins the next in
    the same pool, so allocations stay shared across the step, and records
    the work performed between the two (``breaks[i]`` follows
    ``graphs[i]``).
    """

    def __init__(
        self,
        capture_stream: torch.cuda.Stream,
        computation: torch.cuda.Stream,
        pool_handle,
        generators: Iterable[torch.Generator],
    ) -> None:
        self.capture_stream = capture_stream
        self.computation = computation
        self.pool = pool_handle
        self.generators = tuple(generators)
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.breaks: list[Callable[[], Any]] = []
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

    def break_graph(self, work: Callable[[], Any]) -> None:
        """End the current segment, record ``work``, begin the next."""
        self.end()
        self.breaks.append(work)
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
        self.breaks.clear()


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
    _reuse_joined_cross_stream_blocks()
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

    :meth:`begin` starts capturing work enqueued on ``stream`` (and on streams
    that wait on it and are joined back before :meth:`end`); :meth:`end`
    instantiates the :class:`BreakableCUDAGraph`. Both run on the same host
    thread; work between them may be submitted from other threads (expert
    microbatches). ``peers`` pairs each other device the work allocates on
    with the pool receiving those allocations.
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

        Every segment is instantiated for :meth:`BreakableCUDAGraph.replay`.

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
        return BreakableCUDAGraph(
            self.segments.graphs,
            output,
            self.capture_stream,
            self.segments.breaks,
        )


def capture(
    body: Callable[[], Any],
    *,
    stream: torch.cuda.Stream,
    pool: torch.cuda.MemPool | None = None,
    generators: Iterable[torch.Generator] = (),
    peers: Sequence[Sequence[Any]] = (),
) -> BreakableCUDAGraph:
    """Capture ``body`` as it runs on ``stream``.

    The body enqueues its work on ``stream`` (the binding's computation
    stream) exactly as an eager call would; the capture runs on a private
    sibling stream that the computation stream joins. The caller warmed up
    the body eagerly first, so lazy initialization (kernel compilation,
    plan creation) does not occur inside the capture.

    Args:
        body: The step; its return value is the graph's ``output``.
        stream: The binding's computation stream.
        pool: The allocator pool of the graph's owner (in a worker, one
            Rust reserved); ``None`` creates a private pool.
        generators: Generators the body draws from besides the device's
            default generator, which every capture registers. Replay
            advances each registered generator's offset.
        peers: ``[device, pool]`` pairs: the allocator pool (from
            :func:`pool`) receiving the step's allocations on each other
            device its modules run on (a generation tower reached through
            transfers). Each pool must outlive the graph.

    Returns:
        The instantiated :class:`BreakableCUDAGraph`.

    Raises:
        RuntimeError: Capture failed; every partial graph was reset and the
            original error is the cause.
    """
    placed = tuple(
        (torch.device("cuda", int(ordinal)), handle)
        for ordinal, handle in peers
    )
    global _capturing
    if _capturing is not None:
        raise RuntimeError("a graph capture is already in progress")
    recording = Capture(stream, pool, generators, placed)
    recording.begin()
    # The body reaches :func:`break_graph` through the module's capture in
    # progress: process-wide rather than per thread, because a body may
    # submit from microbatch threads while this thread waits.
    _capturing = recording
    try:
        with torch.cuda.stream(stream):
            output = body()
    except Exception as error:
        _capturing = None
        recording.end(failure=error)
        raise
    _capturing = None
    return recording.end(output)


def break_graph(work: Callable[[], Any]) -> None:
    """End the capturing graph's segment and begin its next.

    Called where the step being captured by :func:`capture` reaches work a
    graph cannot record. ``work`` performs it:
    :meth:`BreakableCUDAGraph.replay` calls it between the two segments at
    every launch, on the host, with the captured stream current (a
    distributed expert layer's weight copy).

    Raises:
        RuntimeError: No capture is in progress.
    """
    if _capturing is None:
        raise RuntimeError("break_graph is called while a graph is captured")
    _capturing.segments.break_graph(work)
