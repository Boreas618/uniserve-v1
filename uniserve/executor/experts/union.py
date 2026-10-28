"""This rank's member of an expert union, driven by the executor.

Ranks of an expert union execute every expert step together: the ranks
whose forward the step runs send their tokens, every other rank joins each
expert layer with no tokens of its own. :class:`ExpertUnion` holds this
rank's side of that protocol:

- the step agreement (:class:`~uniserve.executor.experts.ExpertAgreement`)
  over the worker's host group and, during startup, the warmup
  announcements dedicated expert ranks follow;
- one expert exchange per microbatch (``uniserve.distributed.experts``),
  each with its own stream, and the microbatch threads that rotate eager
  forwards at their expert dispatch points;
- joins: per microbatch the owner of an execution context whose expert
  layers a join runs with no tokens, and their captured graphs.

Every call runs on the executor's thread (the worker's builder thread),
which makes every agreement: each one advances the union's ready-kind
rotation, so steps and idle joins share one :class:`ExpertAgreement`.

Construction is collective and happens at the same point of every member's
startup, after loading and before capacity planning, so the exchanges'
device workspaces are resident when the rank measures its free storage.
Warmup reaches the same collectives in the same order on every member: the
runners capture their graphs (each capture's eager warmup an expert step the
sources announce), then :meth:`ExpertUnion.finish_warmup` lets dedicated
expert ranks follow the announcements until the sources announce zero and
captures every rank's joins. Steps are agreed from then on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch

from uniserve import cuda_graph
from uniserve.backends.experts import HostBootstrap
from uniserve.distributed import experts as library

from .agreement import ExpertAgreement, HostGroup
from .joins import JoinGraphs
from .microbatches import MicrobatchThreads

_logger = logging.getLogger(__name__)

# One microbatch's forward at an agreed capacity: called with the capacity on
# the microbatch's stream, it enqueues the forward (and its selection). Its
# result is ignored.
Forward = Callable[[int], object]


class ExpertUnion:
    """This rank's member of an expert union.

    Construct it with :meth:`configure` and :meth:`connect`. Runners hand
    each microbatch's context owner to :meth:`bind_joins` (an expert-only
    rank calls :meth:`bind_experts`), expert-step forwards go through
    :meth:`step`, and a rank without a forward of its own joins the union's
    next step through :meth:`idle`.
    """

    def __init__(
        self,
        agreement: ExpertAgreement,
        *,
        model,
        transport: str,
        bootstrap: HostBootstrap,
        dedicated: bool,
        graphs: bool,
    ) -> None:
        self.agreement = agreement
        self.dedicated = dedicated
        self._model = model
        self._transport = transport
        self._bootstrap = bootstrap
        self._graphs = graphs
        # Set by :meth:`connect`: each microbatch's exchange and stream (the
        # first the device's primary stream), the microbatch threads with two
        # or more, and split MegaMoE's peer-mapped storage, borrowed from the
        # worker and referenced until every exchange closed.
        self.exchanges: tuple[Any, ...] = ()
        self.streams: tuple[torch.cuda.Stream, ...] = ()
        self._threads: MicrobatchThreads | None = None
        self._storages: tuple[torch.Tensor | None, ...] = ()
        # Per microbatch: the owner of the execution context its joins run
        # (a runner or an expert-only rank's ExpertLayers), and its graph
        # pool for captured joins.
        self._owners: list[Any] = []
        self._pools: list[Any] = []
        self._layers: list[library.ExpertLayers] = []
        # Joins of every microbatch, replayed on the first stream outside
        # this rank's steps; with microbatches, also each microbatch's join
        # alone, replayed by an empty microbatch of a populated step.
        self._joins: JoinGraphs | None = None
        self._microbatch_joins: list[JoinGraphs] = []
        # The captured joins and microbatch joins an eager union set aside
        # (:meth:`run_eager`); their pools' storage is given back meanwhile.
        self._eager_joins: tuple[JoinGraphs, list[JoinGraphs]] | None = None
        self._warming = True
        self._closed = False

    @classmethod
    def configure(
        cls,
        model,
        settings: Mapping[str, Any],
        host: Sequence[Any],
        *,
        max_tokens: int,
        graphs: bool,
    ) -> ExpertUnion:
        """Configure this rank's step agreement, collectively over the union.

        The exchanges follow in :meth:`connect`, once the worker mapped the
        peer storage :meth:`storage_bytes` states.

        Args:
            model: The loaded model.
            settings: The executor settings' ``experts`` section, of
                which the agreement reads ``transport`` (``alltoall``,
                ``deepep`` or ``megamoe``), ``attention_ranks``
                (dedicated-expert placement; zero when every rank holds its
                experts), ``sources`` (the union members that execute this
                rank's forwards together; empty on an expert-only rank) and
                ``dedicated`` (this rank holds only routed experts).
            host: The worker's host functions over the union, in order:
                ``rank, size, all_gather, broadcast, barrier, split,
                all_gather_fixed``.
            max_tokens: Tokens one of this rank's forwards sends per step at
                most, graph padding included; dedicated expert ranks take
                the sources' bound instead.
            graphs: Whether joins are captured.
        """
        rank, size, all_gather, broadcast, barrier, split, gather_fixed = host
        # A waiting host collective must not hold the interpreter lock: other
        # microbatch threads run numerical code meanwhile.
        for function in (all_gather, broadcast, barrier, split, gather_fixed):
            function.release_gil = True
        transport = str(settings["transport"])
        agreement = ExpertAgreement.configure(
            HostGroup(
                rank=int(rank),
                size=int(size),
                all_gather=all_gather,
                broadcast=broadcast,
                all_gather_fixed=gather_fixed,
            ),
            int(settings["attention_ranks"]),
            [int(member) for member in settings["sources"]],
            max(int(max_tokens), 1),
            fused=transport == "megamoe",
        )
        return cls(
            agreement,
            model=model,
            transport=transport,
            bootstrap=HostBootstrap(
                rank, size, all_gather, broadcast, barrier, split, gather_fixed
            ),
            dedicated=bool(settings["dedicated"]),
            graphs=graphs,
        )

    def storage_bytes(self) -> int:
        """Bytes of peer-mapped storage each microbatch's exchange needs.

        Split MegaMoE maps one region per microbatch across the union's
        members; every other transport needs none (zero).
        """
        return int(
            library.symmetric_bytes(
                self._model,
                self._transport,
                self.agreement.size,
                self.agreement.attention_ranks,
                self.agreement.max_tokens,
            )
        )

    def connect(
        self,
        microbatches: Sequence[Mapping[str, Any]],
        storages: Sequence[torch.Tensor | None],
    ) -> None:
        """Create every microbatch's exchange, collectively over the union.

        ``microbatches`` holds each microbatch's ``stream`` (CUDA stream
        handle; the first is the primary stream), ``device`` (CUDA ordinal) and
        ``pointers`` (every member's address of its peer storage; empty
        without it); ``storages`` each microbatch's peer-mapped storage as
        an int8 tensor, or ``None``. Microbatch threads, with two or more
        microbatches, prepare their thread-local numerical state first.
        """
        if self.exchanges:
            raise RuntimeError("the expert union is connected")
        if len(storages) != len(microbatches):
            raise ValueError("every microbatch has one storage entry")
        device = torch.device("cuda", int(microbatches[0]["device"]))
        handles = [int(entry["stream"]) for entry in microbatches]
        streams = [
            torch.cuda.ExternalStream(handle, device=device)
            for handle in handles
        ]
        threads = None
        if len(streams) > 1:
            threads = MicrobatchThreads(
                len(streams),
                context=partial(_microbatch_scope, device, streams),
            )
            # Thread-local numerical state (the cuBLAS handle) belongs to the
            # thread an eager forward runs on, so each thread prepares its
            # own before any capture.
            threads.each(lambda _: library.warm_thread(device.index))
        exchanges = [
            library.create_exchange(
                self._model,
                self._transport,
                self.agreement.rank,
                self.agreement.size,
                self.agreement.attention_ranks,
                self._bootstrap,
                self.agreement.max_tokens,
                device.index,
                handle,
                None if threads is None else threads.rotation.yield_turn,
                storage,
                [int(pointer) for pointer in entry["pointers"]],
            )
            for handle, storage, entry in zip(
                handles, storages, microbatches, strict=True
            )
        ]
        self.exchanges = tuple(exchanges)
        self.streams = tuple(streams)
        self._threads = threads
        self._storages = tuple(storages)
        _logger.info(
            "expert union member %d of %d: %s transport, %d microbatches, "
            "%d tokens per rank and step",
            self.agreement.rank,
            self.agreement.size,
            self._transport,
            len(streams),
            self.agreement.max_tokens,
        )

    @property
    def microbatches(self) -> int:
        """The microbatches each step runs: one exchange and stream each."""
        return len(self.exchanges)

    def replays(self) -> bool:
        """Whether this rank's forwards may replay their captured graphs.

        A replayed forward submits its whole step in one launch without
        passing the host turn, and its thread then waits in the driver once
        the stream holds about a thousand unfinished launches. A rank that
        joins eagerly submits its microbatches in rotation behind kernels
        that wait for this rank's other microbatches, reaches the same limit
        and keeps its turn: neither rank would submit what the other waits
        for. A replayed join waits on no host turn, so forwards replay only
        once the union's joins are captured; until then they run eagerly, in
        the same rotation as every rank's joins. Without microbatch threads
        no turn is held.
        """
        return self._threads is None or self._joins is not None

    def bind_joins(self, owners: Sequence[Any], pools: Sequence[Any]) -> None:
        """Take each microbatch's context owner and join graph pool.

        Owners (in microbatch order) hold the execution context whose expert
        layers a join runs; pools receive the captured joins. Then binds the
        exchanges' shared microbatch plan.
        """
        if len(owners) != self.microbatches or len(pools) != self.microbatches:
            raise ValueError(
                "every microbatch joins through one context and pool"
            )
        self._owners = list(owners)
        self._pools = list(pools)
        library.bind_microbatches(self.exchanges)

    def bind_experts(
        self, model, scratches: Sequence[Any], pools: Sequence[Any]
    ) -> None:
        """Construct an expert-only rank's layers once per microbatch.

        Each ``ExpertLayers`` runs on its microbatch's stream with its
        exchange and the stream's scratch, prepared for the sources' token
        bound; they become the joins' context owners.
        """
        self._layers = [
            library.ExpertLayers(
                model,
                stream.cuda_stream,
                scratch,
                exchange,
                self.agreement.max_tokens,
            )
            for stream, scratch, exchange in zip(
                self.streams, scratches, self.exchanges, strict=True
            )
        ]
        self.bind_joins(self._layers, pools)

    def capacities_for(
        self, tokens: int, microbatches: int = 1
    ) -> tuple[int, ...]:
        """The capacities a forward of ``tokens`` tokens can run at.

        ``microbatches`` is how many forwards the step it belongs to holds
        (``ExpertAgreement.capacities_for``): warmup captures each forward
        at these, ascending.
        """
        return self.agreement.capacities_for(tokens, microbatches)

    def step(
        self,
        forwards: Sequence[Forward | None],
        tokens: int,
        kind: int,
        forced: int | None = None,
    ) -> int:
        """Run one expert step of this rank's forwards; return its capacity.

        ``forwards`` holds one entry per microbatch: the forward of a
        populated microbatch, or ``None`` for an empty one, which joins.
        ``tokens`` is the most tokens any of them sends (its graph's padded
        extent when it replays) and ``kind`` the call kind it has ready.

        While serving, the ranks agree on the capacity; until this rank's
        step is selected, it joins every other selected step. During startup
        warmup the step runs at ``forced`` or the smallest capacity holding
        its tokens, announced to dedicated expert ranks.

        Raises:
            ValueError: ``forced`` outside warmup, or a forward per
                microbatch missing.
            RuntimeError: The step or a join failed.
        """
        if len(forwards) != self.microbatches:
            raise ValueError("an expert step has one entry per microbatch")
        if self._warming:
            capacity = (
                int(forced)
                if forced is not None
                else self.agreement.capacity_for(int(tokens))
            )
            capacity = self.agreement.warmup(capacity)
        else:
            if forced is not None:
                raise ValueError(
                    "a forced expert capacity runs only during warmup"
                )
            while True:
                agreement = self.agreement.agree(int(tokens), int(kind), False)
                if agreement.capacity == 0:
                    # No complete source group is ready yet; ask again.
                    continue
                if agreement.active:
                    capacity = agreement.capacity
                    break
                self._join(agreement.capacity)
        self._run(
            [
                (partial(forward, capacity) if forward is not None else None)
                for forward in forwards
            ],
            capacity,
        )
        return capacity

    def idle(self, leaving: bool) -> tuple[bool, bool]:
        """Join the union's next step without a forward of this rank's own.

        Returns whether a step ran and whether every rank has left the
        union.
        """
        agreement = self.agreement.agree(0, 0, bool(leaving))
        if agreement.capacity == 0:
            return False, agreement.released
        self._join(agreement.capacity)
        return True, agreement.released

    def _join(self, capacity: int) -> None:
        """Join one step of ``capacity`` with every microbatch.

        Replays the captured joins once they exist, else joins eagerly in
        rotation.
        """
        if self._joins is not None:
            with torch.cuda.stream(self.streams[0]):
                self._joins.replay(capacity)
            return
        self._join_eager(capacity)

    def finish_warmup(self) -> None:
        """End the union's warmup on this rank.

        Follows the announced warmup steps, starts serving and captures the
        joins every rank replays outside its own steps.
        """
        self._follow()
        self._warming = False
        if self._graphs:
            self._capture_joins()

    # Standby: collective over the union, while no step runs.

    def run_eager(self, eager: bool) -> None:
        """Join eagerly in rotation (``eager``), as before the joins were
        captured, or replay the captured joins again.

        With the joins set aside, this rank's forwards run eagerly too
        (:meth:`replays`), so every rank of the union submits in the same
        rotation; the worker gives the joins' graph storage back meanwhile.
        Every rank of the union switches between the same two steps.
        """  # noqa: D205
        # Both kinds of captured join live in the join pools the worker
        # gives back: an empty microbatch of a populated step must join
        # eagerly as well (:meth:`_microbatch_join`).
        if eager and self._joins is not None:
            self._eager_joins = (self._joins, self._microbatch_joins)
            self._joins, self._microbatch_joins = None, []
        elif not eager and self._eager_joins is not None:
            self._joins, self._microbatch_joins = self._eager_joins
            self._eager_joins = None

    def suspend(self) -> bool:
        """Release workspaces that keep their addresses; whether any did."""
        # Every exchange suspends (a collective), whatever the others did.
        released = False
        for exchange in self.exchanges:
            released |= bool(exchange.suspend())
        return released

    def restore(self) -> None:
        """Map the workspaces :meth:`suspend` released back in place."""
        for exchange in self.exchanges:
            exchange.restore()

    def release(self) -> bool:
        """Free workspaces that cannot keep their addresses; whether any did.

        The graphs that run expert steps address them, so the promotion
        rebuilds them (:meth:`rebuild`) and captures those graphs again.
        """
        # Every exchange releases (a collective), whatever the others did.
        released = False
        for exchange in self.exchanges:
            released |= bool(exchange.release())
        return released

    def rebuild(self) -> None:
        """Create the freed workspaces again and drop the joins over them.

        The warmup that follows announces its steps again and captures the
        joins anew.
        """
        for exchange in self.exchanges:
            exchange.rebuild()
        self._release_joins()
        self._warming = True

    def close(self, aborted: bool) -> None:
        """Release joins, threads, layers and exchanges.

        After every graph and binding using them retired. Exchange close is
        collective over the union; an aborted close never waits on peers and
        keeps every resource referenced until the process exits, because
        device work may still address it.
        """
        if self._closed:
            return
        self._closed = True
        if aborted:
            # A microbatch thread may be blocked on a failed peer, and device
            # work may still address the joins' pools and the workspaces.
            _abandoned.append(self)
            return
        closes: list[Callable[[], None]] = [self._release_joins]
        if self._threads is not None:
            closes.append(self._threads.close)
        closes += [layers.close for layers in self._layers]
        closes += [exchange.close for exchange in self.exchanges]
        failure: BaseException | None = None
        for close in closes:
            try:
                close()
            except Exception as error:  # noqa: BLE001 - close every part
                failure = failure or error
        # Peers no longer address the storage once every exchange passed
        # its closing barrier.
        self._storages = ()
        if failure is not None:
            raise failure

    def _follow(self) -> None:
        """Dedicated expert ranks execute every announced warmup step.

        Sources end their announcements with zero.
        """
        if self.agreement.attention_ranks == 0:
            return
        if not self.dedicated:
            self.agreement.warmup(0)
            return
        steps = 0
        while capacity := self.agreement.warmup(0):
            self._join_eager(capacity)
            steps += 1
        _logger.info("joined %d announced expert warmup steps", steps)

    def _capture_joins(self) -> None:
        """Capture every capacity's joins, largest first.

        The first stream's graphs join every microbatch, each capacity
        warmed eagerly first (a collective step on every rank). Source ranks
        with microbatches also capture each microbatch's join alone in its
        own pool, which an empty microbatch of a populated step replays.
        """
        if len(self._pools) != self.microbatches:
            raise RuntimeError(
                "expert joins need one graph pool per microbatch"
            )
        capacities = self.agreement.capacities
        self._joins = JoinGraphs.capture(
            capacities,
            self._join_eager,
            partial(_record, self.streams[0], self._pools[0]),
            warm=True,
        )
        if self.microbatches > 1 and not self.dedicated:
            self._microbatch_joins = [
                JoinGraphs.capture(
                    capacities,
                    partial(self._join_one, index),
                    partial(_record, self.streams[index], self._pools[index]),
                    warm=False,
                )
                for index in range(self.microbatches)
            ]
        captured = len(self._joins) + sum(map(len, self._microbatch_joins))
        _logger.info(
            "captured %d expert join graphs for capacities %s",
            captured,
            list(capacities),
        )

    def _release_joins(self) -> None:
        joins, self._joins = self._joins, None
        microbatch_joins, self._microbatch_joins = self._microbatch_joins, []
        # Joins an eager union set aside are released with the others.
        aside: list[JoinGraphs] = []
        if self._eager_joins is not None:
            aside = [self._eager_joins[0], *self._eager_joins[1]]
            self._eager_joins = None
        failure: BaseException | None = None
        for graphs in (joins, *microbatch_joins, *aside):
            if graphs is None:
                continue
            try:
                graphs.release()
            except Exception as error:  # noqa: BLE001 - release every table
                failure = failure or error
        if failure is not None:
            raise failure

    def _join_one(self, index: int, capacity: int) -> None:
        """Microbatch ``index`` joins a step of ``capacity`` with no tokens."""
        owner = self._owners[index]
        with torch.cuda.stream(self.streams[index]):
            library.join(owner, capacity)

    def _join_eager(self, capacity: int) -> None:
        """Every microbatch joins a step of ``capacity`` eagerly."""
        self._run(
            [
                partial(self._join_one, index, capacity)
                for index in range(self.microbatches)
            ],
            capacity,
        )

    def _microbatch_join(self, index: int, capacity: int) -> Callable[[], None]:
        """The join of an empty microbatch within a populated step.

        Its captured graph once the joins exist (no host turn), else its
        eager join (which passes the turn at each dispatch).
        """
        if self._microbatch_joins:
            graphs = self._microbatch_joins[index]

            def replay() -> None:
                with torch.cuda.stream(self.streams[index]):
                    graphs.replay(capacity)

            return replay

        return partial(self._join_one, index, capacity)

    def _run(
        self,
        calls: Sequence[Callable[[], Any] | None],
        capacity: int,
    ) -> None:
        """Run one step's microbatch calls between a fork and a join.

        Every microbatch stream beyond the first waits on the primary
        stream's work before the step and the primary stream waits on every
        microbatch's work after it, so the caller orders later work (the
        batch's seal) behind the whole step. An empty microbatch of a
        populated step joins. With microbatch threads, every call runs on its
        thread in rotation; without, the single call runs here.
        """
        resolved = [
            call if call is not None else self._microbatch_join(index, capacity)
            for index, call in enumerate(calls)
        ]
        with self._forked():
            if self._threads is None:
                resolved[0]()
            else:
                self._threads.run(resolved)

    @contextmanager
    def _forked(self) -> Iterator[None]:
        primary = self.streams[0]
        if len(self.streams) == 1:
            yield
            return
        fork = primary.record_event()
        for stream in self.streams[1:]:
            stream.wait_event(fork)
        try:
            yield
        finally:
            for stream in self.streams[1:]:
                primary.wait_event(stream.record_event())


# Unions whose aborted close left their resources to the process's exit.
_abandoned: list[ExpertUnion] = []


def _record(stream: torch.cuda.Stream, pool, body: Callable[[], Any]):
    """Capture ``body`` on ``stream`` into the worker's graph ``pool``."""
    return cuda_graph.capture(body, stream=stream, pool=pool)


@contextmanager
def _microbatch_scope(
    device: torch.device, streams: Sequence[torch.cuda.Stream], index: int
) -> Iterator[None]:
    """A microbatch thread's numerical scope: its device, stream, inference."""
    with (
        torch.cuda.device(device),
        torch.cuda.stream(streams[index]),
        torch.inference_mode(),
    ):
        yield
