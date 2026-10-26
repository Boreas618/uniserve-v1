"""A diffusion LM computation: one diffusion language model component's
token-denoising batches.

The worker hands a computation each token-denoising batch it carries
(``Executor.execute``): host columns of its calls and of the scheduler's
forward rows with their attention pages, the calls' flat tokens (each
readout's answer slots and candidates appended), a small JSON locating the
readouts' lists, the record column and the output words it lends. A batch's
active calls are readouts or denoising steps, never both
(:func:`~.calls.batch_form`). The computation
(:meth:`DiffusionLMComputation.execute`):

1. checks and packs the active calls on the host (:func:`~.calls.pack_readouts`,
   :func:`~.calls.pack_steps`) and writes their completion records
   (:func:`~.calls.record_readouts`, :func:`~.calls.record_steps`), before
   any device work or collective, so every rank of a group refuses the same
   calls;
2. pads the forward to its graph bucket (or keeps its live shape without
   graphs), stages its host columns and gathers its attention metadata from
   the slots' installed tables on the device;
3. runs it as one expert step (with an expert union): a readout's
   ``attend``, completed by its ``finish`` when the final layer exchanges
   tokens, or a denoising step's ``denoise`` over the state rows gathered
   from the resident banks; then a local ``finish`` and one ``readout`` per
   slot chunk and candidate window;
4. copies the candidate values or the steps' results into the lent words
   and relays each step's continuation into the worker's completion-flag
   staging (row 1).

Each method replays its bucket's captured graph (after planning the
attention launches on the host when an operator plans there), or runs
eagerly without a graph pool. Before the computation is sealed, a bucket
without a graph runs eagerly and is captured right after, which records
without executing. Once sealed, every graph a batch selects is looked up
before its expert step agrees, so a missing graph never strands the peers.

The banks advance inside ``denoise``: a step commits its rows' stepped
state to their slots on the device. The computation commits nothing past
the worker's commit point.
"""  # noqa: D205

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
import torch

from uniserve import cuda_graph
from uniserve.executor.text.buffers import Stage, Staging
from uniserve.executor.text.graphs import GraphTable
from uniserve.executor.text.protocol import (
    CALL_FIELDS,
    GraphMode,
    RecordWriter,
    forward_fields,
    page_count,
    page_start,
)
from uniserve.executor.text.rows import graph_widths, pack_rows, page_width
from uniserve.runner.attention import AttentionMetadata
from uniserve.runner.diffusion_lm import DiffusionLMRunner

from .bounds import Bounds, Facts, ServedSampling
from .buffers import Banks, Columns
from .calls import (
    ReadoutPass,
    Seeds,
    Steps,
    Window,
    batch_form,
    pack_readouts,
    pack_steps,
    readout_spans,
    record_readouts,
    record_steps,
)
from .shapes import (
    CANDIDATES,
    AttendKey,
    DenoiseKey,
    FinishKey,
    ReadoutKey,
    attend_padding,
)
from .warmup import (
    Alone,
    Readout,
    WarmStep,
    layout,
    scratch_slots,
    synthetic_readout,
    synthetic_steps,
)

__all__ = ["DiffusionLMComputation", "Executed"]

_logger = logging.getLogger(__name__)

# The completion-flag row of the worker's relay staging.
_COMPLETIONS = 1

# The expert call kind of diffusion LM computation ``g`` is this plus ``g``:
# above the text computations' kinds (their group indices), so the union's
# ready kinds take turns between the two.
_EXPERT_KINDS = 64

# The attention metadata tensors in the runner's parameter order: query
# lengths and offsets, prefix lengths and offsets, block tables and start
# pages.
_Metadata = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True, slots=True)
class Executed:
    """How a batch's forward ran.

    Attributes:
        mode: The mode of its ``attend`` or ``denoise``.
        live_tokens: Its own tokens.
        padded_tokens: The tokens its graph bucket evaluated beside them.
        rows: Its own sequences.
        capacity: The expert capacity its step agreed; 0 without one.
    """

    mode: GraphMode
    live_tokens: int
    padded_tokens: int
    rows: int
    capacity: int


# What a batch without an active call reports.
_IDLE = Executed(GraphMode.NONE, 0, 0, 0, 0)


def _positions(position: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return each row's positions ``position..position + query``, packed."""
    query = np.asarray(query, dtype=np.int64)
    offsets = np.cumsum(query) - query
    owner = np.repeat(np.arange(len(query), dtype=np.int64), query)
    within = np.arange(int(query.sum()), dtype=np.int64) - offsets[owner]
    return np.asarray(position, dtype=np.int64)[owner] + within


def _padded(values: np.ndarray, count: int) -> np.ndarray:
    """Append ``count`` zeros along the last axis."""
    values = np.asarray(values, dtype=np.int64)
    zeros = np.zeros((*values.shape[:-1], count), dtype=np.int64)
    return np.concatenate([values, zeros], axis=-1)


class DiffusionLMComputation:
    """The token-denoising batches of one diffusion LM component on a rank.

    Args:
        index: The computation's index among the executor's computations.
        group: The computation's index among the rank's diffusion LM
            computations, which names its expert steps.
        component: The component the computation serves.
        label: The kernel-table label of the computation's runner.
        runner: The numerical runner on the primary stream.
        bounds: The computation's bounds over the fitted pool.
        facts: The model's facts.
        sampling: The block sampling a generating computation steps with;
            ``None`` on a computation that serves readouts only, which
            refuses steps.
        seeds: The requests the rank's slots hold.
        columns: The computation's columns.
        metadata: The attention metadata the forwards read.
        banks: A generating computation's resident banks, which ``runner``
            commits to; ``None`` otherwise.
        stream: The primary stream.
        pool: The primary stream's graph pool; ``None`` runs every forward
            eagerly at its live shape.
        relay_flags: ``uint8 [2, calls]`` the worker's staging of
            transition decisions (row 0) and completion flags (row 1), by
            call.
        union: The rank's expert union member, or ``None``.
        table_groups: ``int64 [tables]`` the cache group of each numerical
            block table.
    """

    def __init__(
        self,
        *,
        index: int,
        group: int,
        component: str,
        label: str,
        runner: DiffusionLMRunner,
        bounds: Bounds,
        facts: Facts,
        sampling: ServedSampling | None,
        seeds: Seeds,
        columns: Columns,
        metadata: AttentionMetadata,
        banks: Banks | None,
        stream: torch.cuda.Stream,
        pool: Any | None,
        relay_flags: torch.Tensor,
        union: Any | None,
        table_groups: np.ndarray,
    ) -> None:
        self.index = index
        self.kind = _EXPERT_KINDS + group
        self.label = label
        self.runner = runner
        self.bounds = bounds
        self.shapes = bounds.shapes()
        self.facts = facts
        self.sampling = sampling
        self.seeds = seeds
        self.columns = columns
        self.metadata = metadata
        self.banks = banks
        self.stream = stream
        self.pool = pool
        self.relay_flags = relay_flags
        self.union = union
        self.table_groups = table_groups
        # The cache groups whose pages the forward columns carry.
        self.groups = int(table_groups.max(initial=-1)) + 1
        self.graphs = GraphTable(f"diffusion LM computation of {component}")
        self.staging = Staging(columns.arena.device)
        # An attention operator plans each launch on the host, so a replay of
        # ``attend`` or ``denoise`` follows ``plan_attention``. Known once
        # every shape has run; until then every replay plans.
        self.plans_attention = True
        # Segments of the captured graph with the most graph breaks.
        self.segments = 0

    # A batch.

    def execute(
        self,
        step: Mapping[str, Any],
        calls: torch.Tensor,
        forward: torch.Tensor,
        tokens: torch.Tensor,
        records: torch.Tensor,
        words: torch.Tensor,
    ) -> int:
        """Run one token-denoising batch; return the record words written.

        Args:
            step: The batch's JSON: ``calls`` holds, by call index, each
                readout's ``readout`` lists (:class:`~.calls.ReadoutSpans`).
            calls: ``int64 [CALL_FIELDS, C]`` host call columns.
            forward: ``int64 [FORWARD_FIELDS + 2G, R]`` host forward rows
                with pages.
            tokens: ``int64 [N]`` host flat tokens and readout lists.
            records: ``int64 [K]`` host record column, written.
            words: ``int64 [W]`` page-locked output words, written on the
                primary stream.

        Raises:
            ValueError: A call or row is malformed or disagrees with the
                requests, the forward rows or the computation's bounds; or,
                once sealed, the batch selects a graph startup did not
                capture.
            AssertionError: Records or output words exceed their columns.
        """
        call_columns = calls.numpy()
        forward_columns = forward.numpy()
        if call_columns.shape[0] != CALL_FIELDS:
            raise ValueError(
                "a token batch's call columns have the wrong fields"
            )
        if forward_columns.shape[0] != forward_fields(self.groups):
            raise ValueError(
                "a token batch's forward columns have the wrong groups"
            )
        writer = RecordWriter(records.numpy(), call_columns.shape[1])
        lent = int(words.numel())
        active, stepping = batch_form(call_columns)
        if len(active) == 0:
            written, executed = 0, _IDLE
        elif stepping:
            steps = pack_steps(
                call_columns, forward_columns, active, self.seeds, self.sampling
            )
            written = record_steps(
                writer,
                call_columns,
                steps,
                length=self.bounds.sequence_length,
                lent=lent,
            )
            starts, counts = self._row_pages(forward_columns, steps.rows)
            executed = self.denoise(
                steps,
                steps.slot,
                starts,
                counts,
                words=words,
                relay=steps.calls,
            )
        else:
            readout = pack_readouts(
                call_columns,
                forward_columns,
                tokens.numpy(),
                active,
                readout_spans(step["calls"]),
                vocab=self.facts.vocab_size,
                capacity=self.bounds.tokens,
            )
            written = record_readouts(writer, call_columns, readout, lent=lent)
            starts, counts = self._row_pages(forward_columns, readout.rows)
            executed = self.readout(readout, starts, counts, words=words)
        return writer.finish(
            words_written=written,
            lent=lent,
            mode=executed.mode,
            live_tokens=executed.live_tokens,
            padded_tokens=executed.padded_tokens,
            rows=executed.rows,
            draft_tokens=0,
            expert_capacity=executed.capacity,
        )

    def _row_pages(
        self, forward: np.ndarray, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the first selected page and page count of ``rows`` in
        every numerical table, each ``int64 [tables, rows]``.
        """  # noqa: D205
        starts = np.stack(
            [forward[page_start(group), rows] for group in range(self.groups)]
        )
        counts = np.stack(
            [forward[page_count(group), rows] for group in range(self.groups)]
        )
        return self._tables_of(starts), self._tables_of(counts)

    def _tables_of(self, groups: np.ndarray) -> np.ndarray:
        """Expand per-group page columns ``[groups, n]`` to every table."""
        return np.asarray(groups, dtype=np.int64)[self.table_groups]

    def _widths(self, counts: np.ndarray) -> list[int]:
        # A graph addresses the configured widths; an eager forward its rows'.
        return graph_widths(
            [page_width(table) for table in counts],
            list(self.bounds.widths) if self.pool is not None else None,
            decode=False,
            capacity=self.columns.dimensions.width,
        )

    def _metadata(self) -> _Metadata:
        """The attention metadata tensors, in the runner's parameter order."""
        metadata = self.metadata
        return (
            metadata.query_lengths,
            metadata.query_offsets,
            metadata.cache_lengths,
            metadata.prefix_offsets,
            metadata.block_tables,
            metadata.start_pages,
        )

    def _require(self, keys: Iterable[tuple[Hashable, int | None]]) -> None:
        """Look up every graph a sealed computation's batch selects.

        Called before the batch's expert step agrees, so a missing graph
        fails this rank alone and never strands its peers mid-step.

        Raises:
            ValueError: A graph is not resident.
        """
        for key, capacity in keys:
            self.graphs.lookup(key, capacity)

    def _capacities(self, extent: int) -> list[int | None]:
        """Every capacity an expert step sending ``extent`` tokens can be
        agreed at, ascending; ``[None]`` without an expert union.

        Raises:
            ValueError: The tokens exceed the exchange's tokens per rank.
        """  # noqa: D205
        if self.union is None:
            return [None]
        capacities = self.union.capacities_for(extent)
        if not capacities:
            raise ValueError(
                f"{extent} tokens exceed the expert exchange's tokens per rank"
            )
        return list(capacities)

    def _launch(
        self,
        key: Hashable,
        capacity: int | None,
        call: Callable[[], None],
        plan: Callable[[], None] | None,
    ) -> GraphMode:
        """Run ``call``, the method of graph ``key`` at ``capacity``.

        Replays the resident graph (behind ``plan``, the host planning of
        the attention launches it reads, when operators plan there), or
        calls the method eagerly without a pool. Before the computation is
        sealed, a graph not yet captured is called eagerly, which produces
        this forward's results and plans its attention, then captured.
        """
        stream = self.stream
        with torch.cuda.stream(stream):
            if self.pool is None:
                call()
                return GraphMode.EAGER
            graph = self.graphs.lookup(key, capacity)
            if graph is not None:
                if plan is not None and self.plans_attention:
                    plan()
                cast(cuda_graph.BreakableCUDAGraph, graph).replay()
                return GraphMode.REPLAY
            call()
        captured = cuda_graph.capture(call, stream=stream, pool=self.pool)
        self.graphs.insert(key, capacity, captured)
        self.segments = max(self.segments, captured.segments)
        return GraphMode.CAPTURE

    def _step(
        self,
        forward: Callable[[int | None], None],
        extent: int,
        forced: int | None,
    ) -> int:
        """Run ``forward`` as an expert step; return the step's capacity.

        ``extent`` is the tokens the step sends to each expert layer,
        padding included. Without an expert union the forward runs at no
        capacity. During warmup the step runs at ``forced``, or at the
        smallest capacity holding the extent.
        """
        if self.union is None:
            forward(None)
            return 0
        return self.union.step([forward], extent, self.kind, forced)

    # Readouts.

    def readout(
        self,
        readout: ReadoutPass,
        starts: np.ndarray,
        counts: np.ndarray,
        *,
        words: torch.Tensor | None = None,
        forced: int | None = None,
    ) -> Executed:
        """Evaluate a readout and read every candidate's log-probability.

        Pads the readout to the ``attend`` bucket holding its blocks (or
        keeps its live shape without graphs), stages its tokens, positions
        and rows, gathers its attention metadata, and runs ``attend`` (and
        an exchanging ``finish``) as one expert step, a local ``finish``
        behind it, then one ``readout`` per slot chunk and window of its
        candidates.

        Args:
            readout: The packed readout.
            starts: ``int64 [tables, rows]`` each row's first selected page.
            counts: ``int64 [tables, rows]`` each row's selected pages.
            words: The lent output words: candidate ``i``'s FP32 value
                goes to word ``i`` as its 32-bit pattern. ``None`` keeps
                no value (warmup).
            forced: The capacity a warmup step runs at.

        Raises:
            ValueError: The readout exceeds the computation's bounds or a
                table's width, or, once sealed, a graph it selects is not
                resident.
        """
        columns = self.columns
        dims = columns.dimensions
        graphed = self.pool is not None
        length = self.bounds.sequence_length
        live_rows = len(readout.query)
        live_tokens = len(readout.tokens)
        if graphed:
            rows, total = self.shapes.attend(
                live_rows, live_tokens, readout.longest
            )
            padding = np.asarray(
                attend_padding(live_rows, live_tokens, rows, total, length),
                dtype=np.int64,
            )
            # A graph's launches are sized for sequences of a whole block.
            visible = length
        else:
            rows, total = live_rows, live_tokens
            padding = np.zeros(0, dtype=np.int64)
            visible = readout.longest
        sequences = live_rows + len(padding)
        if total > dims.tokens or sequences > dims.rows:
            raise ValueError(
                f"a readout of {sequences} sequences and {total} tokens "
                f"exceeds the computation's {dims.rows} sequences and "
                f"{dims.tokens} tokens"
            )
        widths = self._widths(counts)
        slots = len(readout.slots)
        finished_slots = self.shapes.finish(slots) if graphed else slots

        # The final layer's exchange makes ``finish`` complete the expert
        # step ``attend`` began; the step then also carries the slot rows,
        # padding included. A local ``finish`` follows the step.
        last, local = self.facts.last_stage, self.facts.local_tail
        exchanging = last and not local
        extent = total if local else max(total, finished_slots)
        attend_key = AttendKey(rows, total)
        finish_key = FinishKey(finished_slots)
        windows = list(readout.windows())
        if graphed and self.graphs.sealed:
            capacities = self._capacities(extent)
            required: list[tuple[Hashable, int | None]] = [
                (attend_key, capacity) for capacity in capacities
            ]
            if exchanging:
                required += [(finish_key, capacity) for capacity in capacities]
            elif last:
                required.append((finish_key, None))
            required += [
                (ReadoutKey(window.bucket), None) for window in windows
            ]
            self._require(required)

        # Padding sequences read no prefix and select no page.
        pad = len(padding)
        queries = np.concatenate([readout.query, padding])
        prefixes = _padded(readout.prefix, pad)
        stage = Stage()
        stage.put(columns["input_ids"][:live_tokens], readout.tokens)
        stage.put(
            columns["positions"][:live_tokens],
            _positions(readout.position, readout.query),
        )
        row_columns = pack_rows(
            _padded(readout.slot, pad),
            prefixes,
            queries,
            np.zeros(sequences, dtype=np.int64),
            list(counts),
            dims.rows + 1,
            total,
        )
        stage.put(columns["row_columns"][: len(row_columns)], row_columns)
        if last:
            stage.put(columns["slot_rows"][:slots], readout.slots)
        stream = self.stream
        with torch.cuda.stream(stream):
            self.staging.submit(stage, stream)
            columns["input_ids"][live_tokens:total].zero_()
            columns["positions"][live_tokens:total].zero_()
            if last:
                # Padding slots complete token row zero.
                columns["slot_rows"][slots:finished_slots].zero_()
            self.metadata.gather(
                columns["row_columns"], sequences, total, widths
            )

        attention = (
            sequences,
            visible,
            widths,
            queries.tolist(),
            prefixes.tolist(),
            _padded(starts, pad).reshape(-1).tolist(),
        )
        metadata = self._metadata()
        state = columns.attend_state
        # The final stage completes and projects the slot rows; the others
        # receive the candidate values it broadcasts.
        finished = columns["finished"] if last else None

        def plan() -> None:
            self.runner.plan_attention(*attention, *metadata)

        def attend(capacity: int) -> None:
            self.runner.attend(
                sequences,
                total,
                *attention[1:],
                capacity,
                columns["input_ids"],
                columns["positions"],
                *metadata,
                *state,
            )

        def finish(capacity: int) -> None:
            # Called on the final stage alone.
            self.runner.finish(
                finished_slots,
                capacity,
                columns["slot_rows"],
                columns["finished"],
                *state,
            )

        mode = GraphMode.NONE

        def forward(capacity: int | None) -> None:
            # Eager attend, its capture, then the eager finish completing
            # the step and its capture: the order every rank communicates in.
            nonlocal mode
            agreed = int(capacity or 0)
            mode = self._launch(
                attend_key, capacity, partial(attend, agreed), plan
            )
            if exchanging:
                self._launch(
                    finish_key, capacity, partial(finish, agreed), None
                )

        capacity = self._step(forward, extent, forced)
        if last and local:
            self._launch(finish_key, None, partial(finish, 0), None)
        for window in windows:
            self._read(readout, window, finished, words)
        return Executed(
            mode, live_tokens, total - live_tokens, live_rows, capacity
        )

    def _read(
        self,
        readout: ReadoutPass,
        window: Window,
        finished: torch.Tensor | None,
        words: torch.Tensor | None,
    ) -> None:
        """Read one slot chunk's window of candidates into the lent words.

        The final stage projects the chunk's finished rows, and every stage
        receives the candidate values the final stage broadcasts.
        """
        columns = self.columns
        graphed = self.pool is not None
        count = window.end - window.first
        slots = window.bucket if graphed else window.live
        candidates = CANDIDATES if graphed else count
        stage = Stage()
        stage.put(
            columns["finished_rows"][: window.live],
            np.arange(window.start, window.start + window.live, dtype=np.int64),
        )
        stage.put(
            columns["candidate_index"][:count],
            readout.index[window.first : window.end],
        )
        stream = self.stream
        with torch.cuda.stream(stream):
            self.staging.submit(stage, stream)
            # Padding rows project finished row zero and unused candidates
            # select its first log-probability: earlier windows' indices
            # could exceed this bucket's rows.
            columns["finished_rows"][window.live : slots].zero_()
            columns["candidate_index"][count:candidates].zero_()
        call = partial(
            self.runner.readout,
            slots,
            candidates,
            columns["finished_rows"],
            finished,
            columns["candidate_index"],
            columns["candidate_logprobs"],
        )
        self._launch(ReadoutKey(window.bucket), None, call, None)
        if words is None:
            return
        with torch.cuda.stream(stream):
            # FP32 values travel as their 32-bit patterns, which the worker
            # reads from each word's low half.
            values = columns["candidate_words"][:count]
            values.copy_(
                columns["candidate_logprobs"][:count].view(torch.int32)
            )
            words[window.first : window.end].copy_(values, non_blocking=True)

    # Denoising steps.

    def denoise(
        self,
        steps: Steps,
        table_slots: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
        *,
        words: torch.Tensor | None = None,
        relay: np.ndarray | None = None,
        forced: int | None = None,
    ) -> Executed:
        """Denoise the resident blocks of ``steps`` by one step.

        Pads the step to its row bucket (or keeps its live rows without
        graphs) with rows of the sentinel slot zero, stages its columns,
        positions and rows, gathers its attention metadata and its rows'
        state from the banks, and runs ``denoise`` as one expert step, which
        commits each continuing row's stepped state to its slot.

        Args:
            steps: The step's rows.
            table_slots: The slot whose installed tables each row's
                attention reads: the row's own slot when serving.
            starts: ``int64 [tables, rows]`` each row's first selected page.
            counts: ``int64 [tables, rows]`` each row's selected pages.
            words: The lent output words: row ``i``'s outcome and block
                tokens go to words ``i * (1 + C)..``. ``None`` keeps none
                (warmup).
            relay: ``int64 [rows]`` the worker's completion-flag staging
                column of each row's call; ``None`` relays nothing
                (warmup).
            forced: The capacity a warmup step runs at.

        Raises:
            ValueError: The step exceeds the denoising rows or a table's
                width, or, once sealed, its graph is not resident.
        """
        # Only a generating computation, which keeps banks, packs or warms
        # steps.
        denoising, banks = self.bounds.denoise, self.banks
        assert denoising is not None and banks is not None
        columns = self.columns
        dims = columns.dimensions
        graphed = self.pool is not None
        length = self.bounds.sequence_length
        live = len(steps.slot)
        padded = self.shapes.denoise(live) if graphed else live
        if padded > denoising.rows:
            raise ValueError(
                f"a denoising step of {live} rows exceeds its "
                f"{denoising.rows} rows"
            )
        widths = self._widths(counts)
        total = padded * length
        key = DenoiseKey(padded, steps.first)
        if graphed and self.graphs.sealed:
            self._require(
                (key, capacity) for capacity in self._capacities(total)
            )

        # A padding row starts a block of seed and block zero in the
        # sentinel slot zero, which its commit skips; it reads no prefix.
        # The whole column is staged, which zeroes every padding row.
        pad = padded - live
        queries = np.full(padded, length, dtype=np.int64)
        prefixes = _padded(steps.prefix, pad)
        stage = Stage()
        stage.put(
            columns["denoise.columns"],
            _padded(steps.columns(), denoising.rows - live),
        )
        stage.put(
            columns["positions"][: live * length],
            _positions(steps.position, queries[:live]),
        )
        row_columns = pack_rows(
            _padded(table_slots, pad),
            prefixes,
            queries,
            np.zeros(padded, dtype=np.int64),
            list(counts),
            dims.rows + 1,
            total,
        )
        stage.put(columns["row_columns"][: len(row_columns)], row_columns)
        if relay is not None:
            stage.put(columns["denoise.relays"][:live], relay)
        stream = self.stream
        with torch.cuda.stream(stream):
            self.staging.submit(stage, stream)
            columns["positions"][live * length : total].zero_()
            self.metadata.gather(columns["row_columns"], padded, total, widths)
            banks.gather(
                columns["denoise.columns"][0],
                padded,
                columns["denoise.tokens"],
                columns["denoise.history"],
                columns["denoise.self_conditioning"],
            )

        attention = (
            padded,
            length,
            widths,
            queries.tolist(),
            prefixes.tolist(),
            _padded(starts, pad).reshape(-1).tolist(),
        )
        metadata = self._metadata()

        def plan() -> None:
            self.runner.plan_attention(*attention, *metadata)

        def denoise(capacity: int) -> None:
            self.runner.denoise(
                padded,
                steps.first,
                *attention[2:],
                capacity,
                columns["denoise.columns"],
                columns["positions"],
                *metadata,
                columns["denoise.tokens"],
                columns["denoise.history"],
                columns["denoise.self_conditioning"],
                columns["denoise.results"],
                columns["denoise.continued"],
            )

        mode = GraphMode.NONE

        def forward(capacity: int | None) -> None:
            nonlocal mode
            mode = self._launch(
                key, capacity, partial(denoise, int(capacity or 0)), plan
            )

        capacity = self._step(forward, total, forced)
        with torch.cuda.stream(stream):
            if words is not None:
                # One copy of every row's outcome and block tokens.
                words[: live * (1 + length)].view(live, 1 + length).copy_(
                    columns["denoise.results"][:live], non_blocking=True
                )
            if relay is not None:
                # Each row's continuation is its call's completion flag.
                self.relay_flags[_COMPLETIONS].index_copy_(
                    0,
                    columns["denoise.relays"][:live],
                    columns["denoise.continued"][:live],
                )
        return Executed(mode, live * length, pad * length, live, capacity)

    # Startup.

    def warmup(self) -> list[WarmStep]:
        """Return the warmup forwards (:func:`~.warmup.layout`)."""
        denoising = self.bounds.denoise
        return layout(
            self.shapes,
            pooled=self.pool is not None,
            generating=denoising is not None,
            local=self.facts.local_tail,
            last=self.facts.last_stage,
            chunk_rows=0 if denoising is None else denoising.chunk_rows,
            # Every capacity a forward can be agreed at, largest first.
            capacities=lambda tokens: self._capacities(tokens)[::-1],
        )

    def warm(
        self, steps: Sequence[WarmStep], pages: Sequence[np.ndarray]
    ) -> None:
        """Run the warmup forwards ``steps`` over the scratch tables.

        ``pages[k]`` is the host ``int64 [2 * G, rows]`` of the ``k``-th
        forward that attends: per cache group, each row's first selected
        page and page count. Returns once the computation's device work
        completed, because the worker releases the scratch tables right
        after; then learns whether operators plan their launches on the
        host.

        Raises:
            ValueError: The page columns do not match the attending
                forwards.
        """
        attending = [step for step in steps if not isinstance(step, Alone)]
        if len(pages) != len(attending):
            raise ValueError(
                f"a warmup phase of {len(attending)} attending forwards was "
                f"given {len(pages)} page columns"
            )
        installed = self.shapes.capacity()
        columns = iter(pages)
        try:
            for step in steps:
                if isinstance(step, Alone):
                    self._alone(step)
                    continue
                selected = np.asarray(next(columns), dtype=np.int64)
                starts = self._tables_of(selected[0::2])
                counts = self._tables_of(selected[1::2])
                if isinstance(step, Readout):
                    self.readout(
                        synthetic_readout(
                            step, installed, vocab=self.facts.vocab_size
                        ),
                        starts,
                        counts,
                        forced=step.capacity,
                    )
                else:
                    self.denoise(
                        synthetic_steps(step.rows, step.first),
                        scratch_slots(step.rows, installed),
                        starts,
                        counts,
                        forced=step.capacity,
                    )
        finally:
            self.stream.synchronize()
        self.plans_attention = bool(self.runner.plans_attention())
        if self.pool is not None:
            families = Counter(type(key) for key, _ in self.graphs.keys())
            _logger.info(
                "%s: captured %d graphs of up to %d segments: %d attend, %d "
                "finish, %d readout and %d denoise buckets; attention "
                "launches are planned on the %s",
                self.graphs.owner,
                len(self.graphs),
                self.segments,
                families[AttendKey],
                families[FinishKey],
                families[ReadoutKey],
                families[DenoiseKey],
                "host" if self.plans_attention else "device",
            )

    def _alone(self, step: Alone) -> None:
        """Capture a slot bucket's ``readout`` or ``finish`` graph alone.

        Its inputs are padding: every row projects finished row zero and
        reads candidate zero, or completes token row zero. Their values do
        not choose kernels, and a slot bucket may hold more rows than a
        forward's tokens.
        """
        columns = self.columns
        slots = step.slots
        last = self.facts.last_stage
        stream = self.stream
        if step.method == "readout":
            with torch.cuda.stream(stream):
                columns["finished_rows"].zero_()
                columns["candidate_index"].zero_()
            call = partial(
                self.runner.readout,
                slots,
                CANDIDATES,
                columns["finished_rows"],
                columns["finished"] if last else None,
                columns["candidate_index"],
                columns["candidate_logprobs"],
            )
            self._launch(ReadoutKey(slots), None, call, None)
            return
        with torch.cuda.stream(stream):
            columns["slot_rows"][:slots].zero_()
        call = partial(
            self.runner.finish,
            slots,
            0,
            columns["slot_rows"],
            columns["finished"],
            *columns.attend_state,
        )
        self._launch(FinishKey(slots), None, call, None)

    def kernels(self) -> list[tuple[str, list[str]]]:
        """The runner's kernel-table label and selected kernels."""
        return [(self.label, self.runner.kernels())]

    def seal(self) -> None:
        """Refuse further captures; serving may only replay."""
        self.graphs.seal()

    def reopen(self) -> None:
        """Release every graph and accept captures again.

        A promoted standby whose expert workspaces were rebuilt captures its
        forwards anew: the released graphs addressed the freed workspaces.
        The caller drained the primary stream.
        """
        try:
            self.graphs.reopen()
        finally:
            self.plans_attention = True
            self.segments = 0

    def close(self, aborted: bool) -> None:
        """Release the graphs, then the execution context, once the stream
        drained. An aborted close keeps both: device work may still address
        them.
        """  # noqa: D205
        try:
            self.graphs.close(aborted)
        finally:
            if not aborted:
                self.runner.close()
                self.staging.close()
