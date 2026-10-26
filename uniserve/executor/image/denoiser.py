"""An image model's denoiser on row buckets: trajectories, steps, conditions.

The denoiser computation of an image model (BAGEL, SenseNova) runs three
kinds of work on its stream:

- a latent preparation opens a request's trajectory and writes its seeded
  initial latent into the worker's latent ledger;
- a denoising step advances every stepping trajectory of a batch by one
  solver step: its guidance branches' rows are packed into one forward per
  bucket (:class:`RowFamily`), whose predictions the solver combines with
  each trajectory's guidance;
- a condition write (the text computation's latent conditioning,
  :meth:`Denoiser.condition_latents`) writes the keys and values of latent
  images inserted into a prompt.

A *row* is one image's framed sequence in a packed forward
(:mod:`.rows`): one guidance branch of a stepping image, or one latent image
written as conditioning. Each family keeps its own execution context, its
own attention metadata and device columns, and one graph per row bucket
``(T, R)`` (:mod:`.shapes`) on the computation's graph pool, so any rows that
fit a bucket replay its graph; the rows enter only as device values. The
worker owns the latent ledger's two banks of pages: a step gathers each
trajectory's committed pages into the denoiser's workspace, and scatters
its successor into the other bank's pages (:class:`LatentBanks`).

Guidance branches attend to cache prefixes the scheduler placed: the
request's own context, or prefixes in other slots which the text
computation prefills the first time a branch reads them
(``TextComputation.prefill_cache``), ordered against the denoiser's stream
by events.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from uniserve import cuda_graph
from uniserve.bindings import capturing
from uniserve.diffusion import advance_, normal_noise
from uniserve.executor.text.context import ConditionRow
from uniserve.executor.text.graphs import GraphTable
from uniserve.executor.text.protocol import GraphMode
from uniserve.media import image
from uniserve.models import LatentInput, TextSize
from uniserve.nn.attention.gather import row_columns_size
from uniserve.nn.rng import flow_noise_seed
from uniserve.processing import BranchSource, load_tokenizer, resolve_prefix
from uniserve.runner.attention import AttentionMetadata
from uniserve.runner.context import execution_context, kernel_records

from .layout import ImageLayout
from .rows import (
    ImageRow,
    RowColumns,
    RowField,
    RowValue,
    expand_rows,
    pack_attention,
)
from .shapes import RowLimits, RowShape, RowShapes
from .trajectory import (
    BranchRow,
    BranchSet,
    Coordinates,
    ForwardSlot,
    GuidancePlan,
    ImageParams,
    ImageTrajectory,
    Prefix,
    PrefixFill,
    Trajectories,
    admitted_prefix,
)

__all__ = [
    "CacheTables",
    "Denoiser",
    "LatentBanks",
    "LatentBinding",
    "PrefixComputation",
    "RowFamily",
    "StepCall",
    "WarmForward",
    "framed_prefix",
    "split_rows",
]

# Fields of the worker's forward columns before the page columns: call
# index, slot, sequence length, query length and cache write.
_FORWARD_FIELDS = 5

# Visible prefix tokens of the scratch slots a warmup step's rows read.
WARM_PREFIX = 16


@functools.cache
def _tokenizer(path: Path):
    return load_tokenizer(path)


def framed_prefix(loaded, source: BranchSource) -> tuple[int, ...]:
    """The tokens the model frames for an unconditional branch's source.

    The model's prompt framing and tokenizer frame an empty negative prompt
    for ``NEGATIVE_OR_START`` and empty text for ``START``
    (``uniserve.processing.resolve_prefix``); a source no admission fixes
    (``START``) reads these tokens for every request.
    """

    def tokenizer():
        return (
            None if loaded.tokenizer is None else _tokenizer(loaded.tokenizer)
        )

    return tuple(
        int(token)
        for token in resolve_prefix(
            loaded.flow_prompt, tokenizer, source.value, negative_prompt=""
        )
    )


@dataclass(frozen=True, slots=True)
class CacheTables:
    """The prefix cache's numerical block tables a denoiser attends through.

    Attributes:
        pool: The cache's K/V pool object attention layers read.
        unit_tables: ``int32 [tables, S, W]`` each slot's installed units
            from its installed start page on.
        start_pages: ``int32 [groups, S]`` each slot's installed start page.
        table_shapes: ``int32 [tables, 3]`` ``(group, block size, window or
            -1)``.
        groups: Each table's cache group.
        block_sizes: Each table's tokens per page.
        windowed: Whether each table retires pages behind a window.
        widths: The most pages a row selects from each table.
    """

    pool: Any
    unit_tables: torch.Tensor
    start_pages: torch.Tensor
    table_shapes: torch.Tensor
    groups: np.ndarray
    block_sizes: tuple[int, ...]
    windowed: tuple[bool, ...]
    widths: tuple[int, ...]

    @classmethod
    def of(
        cls,
        pool: Any,
        tables: Mapping[str, torch.Tensor],
        *,
        max_sequence: int,
        max_query: int,
    ) -> CacheTables:
        """The tables of the worker's block table views (``tables``).

        A row reads at most ``max_sequence`` tokens of its slot, its
        queries included, and holds at most ``max_query`` queries: a
        full-attention table's width covers the sequence, a windowed one's
        its window and queries plus the partial page they start in, each
        within the pages a slot's table holds.
        """
        shapes = tables["table_shapes"].cpu().numpy().astype(np.int64)
        unit_tables = tables["unit_tables"]
        held = int(unit_tables.shape[2])
        widths = []
        for _, block, window in shapes:
            resident = -(-max(int(max_sequence), 1) // int(block))
            if window >= 0:
                reach = int(window) + max(int(max_query), 1)
                resident = min(resident, -(-reach // int(block)) + 1)
            widths.append(max(1, min(resident, held)))
        return cls(
            pool=pool,
            unit_tables=unit_tables,
            start_pages=tables["start_pages"],
            table_shapes=tables["table_shapes"],
            groups=shapes[:, 0].copy(),
            block_sizes=tuple(int(block) for _, block, _ in shapes),
            windowed=tuple(bool(window >= 0) for _, _, window in shapes),
            widths=tuple(widths),
        )

    @property
    def tables(self) -> int:
        """The numerical tables."""
        return len(self.block_sizes)

    def tables_of(self, groups: np.ndarray) -> np.ndarray:
        """Expand per-group columns ``[groups, n]`` to every table."""
        return np.asarray(groups, dtype=np.int64)[self.groups]


@dataclass(frozen=True, slots=True)
class RowPages:
    """The pages each live row of a forward selects, per numerical table.

    Attributes:
        counts: ``int64 [tables, rows]`` each row's selected page count.
        starts: ``int64 [tables, rows]`` each row's first selected page,
            which windowed tables read.
    """

    counts: np.ndarray
    starts: np.ndarray

    @classmethod
    def of(cls, tables: CacheTables, columns: np.ndarray) -> RowPages:
        """The pages of rows given per cache group.

        ``columns`` is ``int64 [2G, rows]``: group ``g``'s first selected
        pages in row ``2g``, its page counts in row ``2g + 1``.
        """
        columns = np.asarray(columns, dtype=np.int64)
        return cls(
            counts=tables.tables_of(columns[1::2]),
            starts=tables.tables_of(columns[0::2]),
        )

    @classmethod
    def join(cls, tables: CacheTables, parts: Sequence[RowPages]) -> RowPages:
        """The pages of several rows' pages, in order."""
        if not parts:
            empty = np.zeros((tables.tables, 0), np.int64)
            return cls(empty, empty.copy())
        return cls(
            counts=np.concatenate([part.counts for part in parts], axis=1),
            starts=np.concatenate([part.starts for part in parts], axis=1),
        )


def _unbucketed(rows: int, tokens: int, multiple: int) -> RowShape:
    """The shape a forward of ``rows`` live rows and ``tokens`` takes unpadded.

    Its own shape; under sequence parallelism (``multiple`` above one), one
    padding row more and the least multiple of the degree above its tokens,
    so the degree divides the forward's tokens.
    """
    if multiple == 1:
        return RowShape(tokens, rows)
    return RowShape(-(-(tokens + 1) // multiple) * multiple, rows + 1)


def _capacity(
    limits: RowLimits, shapes: RowShapes | None, multiple: int
) -> RowShape:
    """The tokens and rows a family's buffers and context hold.

    Its largest bucket's extents, or without buckets its largest admitted
    forward's shape.
    """
    if shapes is None:
        return _unbucketed(limits.max_rows, limits.max_tokens, multiple)
    return RowShape(
        max(shape.tokens for shape in shapes.shapes),
        max(shape.rows for shape in shapes.shapes),
    )


@contextlib.contextmanager
def _ordered(
    stream: torch.cuda.Stream | None, caller: torch.cuda.Stream | None
) -> Iterator[None]:
    """Order the block's work on ``stream`` between ``caller``'s work.

    ``stream`` waits for what ``caller`` enqueued before the block, and
    ``caller`` for what ``stream`` enqueued in it; nothing is ordered on the
    host or when both are one stream.
    """
    if stream is None or caller is None or stream == caller:
        yield
        return
    stream.wait_stream(caller)
    yield
    caller.wait_stream(stream)


class RowFamily:
    """One kind of packed forward of an image denoiser on row buckets.

    A family evaluates its rows (:class:`~.rows.RowColumns`) in one forward
    of the network: it stages the rows' columns and attention row column
    into its device buffers, then replays the graph of the rows' bucket, or
    runs the forward eagerly: without buckets (graphs off), or as the warm
    run of a bucket whose graph it then captures into the computation's
    pool. The family's own execution context keeps the host attention plans
    of its forwards, which an operator that plans on the host builds from
    the rows' exact lengths before each replay.

    Args:
        owner: Names the family in errors.
        network: The image denoiser.
        context: The family's execution context.
        tables: The block tables its rows attend through.
        limits: The forwards the family admits; their longest row bounds
            every row's queries in a captured forward.
        capacity: The most tokens and rows of one forward, which the
            buffers and the context hold (:func:`_capacity`).
        source: The latent rows the family's latent tokens read.
        writes: Whether the rows write their keys and values to their slots.
        predictions: ``[capacity.tokens + 1, width]`` predictions the rows'
            latent tokens write (the last row is the dump row of every other
            token), or ``None`` for a family whose outputs are discarded.
        shapes: The buckets the family replays, ``None`` without graphs.
        stream: The primary stream.
        pool: The primary stream's graph pool, ``None`` without graphs.
        peers: ``(device ordinal, pool)`` of each other device the network
            computes on (a generation tower).
        multiple: The sequence-parallel degree, which divides every
            forward's tokens.
    """

    def __init__(
        self,
        *,
        owner: str,
        network,
        context,
        tables: CacheTables,
        limits: RowLimits,
        capacity: RowShape,
        source: torch.Tensor,
        writes: bool,
        predictions: torch.Tensor | None,
        shapes: RowShapes | None,
        stream: torch.cuda.Stream | None,
        pool: Any | None,
        peers: Sequence[tuple[int, Any]] = (),
        multiple: int = 1,
    ) -> None:
        self.owner = owner
        self.multiple = int(multiple)
        self.network = network
        self.context = context
        self.tables = tables
        self.limits = limits
        self.capacity = capacity
        self.source = source
        self.writes = bool(writes)
        self.predictions = predictions
        self.shapes = shapes
        self.stream = stream
        self.pool = pool
        self.peers = tuple(peers)
        self.device = source.device
        self.graphs = GraphTable(owner)
        mesh = network.mesh
        self.pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
        # Captured graphs address these tensors, so they live as long as
        # the family. Rows past a forward's own hold stale values, which no
        # forward reads.
        with torch.cuda.stream(stream):
            self.metadata = AttentionMetadata(
                tables.block_sizes,
                tables.windowed,
                tables.unit_tables,
                tables.start_pages,
                tables.table_shapes,
                capacity.rows,
                capacity.tokens,
                max(tables.widths),
                self.device,
            )
            self.row_columns = torch.zeros(
                row_columns_size(capacity.rows, tables.tables, capacity.tokens),
                dtype=torch.int64,
                device=self.device,
            )
            self.fields = torch.zeros(
                (len(RowField), capacity.rows),
                dtype=torch.int64,
                device=self.device,
            )
            self.values = torch.zeros(
                (len(RowValue), capacity.rows),
                dtype=torch.float32,
                device=self.device,
            )

    @property
    def row_tokens(self) -> int:
        """The longest framed sequence one row holds, padding rows too."""
        return self.limits.max_row_tokens

    @property
    def live_rows(self) -> int:
        """The most live rows one forward holds."""
        if self.shapes is None:
            return self.limits.max_rows
        return self.shapes.max_rows

    @property
    def live_tokens(self) -> int:
        """The most live tokens one forward holds."""
        if self.shapes is None:
            return self.limits.max_tokens
        return self.shapes.max_tokens

    def select(self, rows: int, tokens: int) -> RowShape | None:
        """The bucket a forward of ``rows`` live rows and ``tokens`` takes.

        Without buckets, its unpadded shape (:func:`_unbucketed`); ``None``
        when no bucket holds it.
        """
        if self.shapes is not None:
            return self.shapes.select(rows, tokens)
        shape = _unbucketed(rows, tokens, self.multiple)
        if (
            shape.rows > self.capacity.rows
            or shape.tokens > self.capacity.tokens
        ):
            return None
        return shape

    def fit(self, live: RowColumns) -> RowColumns:
        """The live rows padded to the shape they select (:meth:`select`).

        Without buckets the rows stay unchanged, but for the padding
        sequence parallelism divides.

        Raises:
            ValueError: No bucket holds the rows.
        """
        shape = self.select(live.live, live.live_tokens)
        if shape is None:
            raise ValueError(
                f"{self.owner}: no row bucket holds {live.live} rows of "
                f"{live.live_tokens} tokens"
            )
        if shape == RowShape(live.live_tokens, live.live):
            return live
        return live.padded(shape.tokens, shape.rows, self.row_tokens)

    def run(self, columns: RowColumns, pages: RowPages) -> GraphMode:
        """Evaluate ``columns``' rows (fitted to a bucket) in one forward.

        ``pages`` holds each live row's selected pages. Returns how the
        forward ran.

        Raises:
            ValueError: The family is sealed and holds no graph of the rows'
                bucket, or the rows exceed the family's capacity.
        """
        rows, tokens = columns.rows, columns.tokens
        if rows > self.capacity.rows or tokens > self.capacity.tokens:
            raise ValueError(
                f"{self.owner}: {rows} rows of {tokens} tokens exceed the "
                f"family's {self.capacity.rows} rows of "
                f"{self.capacity.tokens} tokens"
            )
        words = pack_attention(
            columns,
            list(pages.counts),
            self.capacity.rows + 1,
            self.capacity.tokens,
        )
        queries = tuple(int(value) for value in columns.column(RowField.TOKENS))
        prefixes = tuple(
            int(value) for value in columns.column(RowField.PREFIX)
        )
        # Each table's first pages, ``rows`` per table; windowed tables read
        # them, rows past the live ones select none.
        firsts = np.zeros((self.tables.tables, rows), np.int64)
        firsts[:, : pages.starts.shape[1]] = pages.starts
        starts = tuple(int(value) for value in firsts.reshape(-1))
        # The columns are pageable host memory: each copy reads its source
        # before it returns, so the host values may change behind it.
        fields, values = columns.tensors()
        with torch.cuda.stream(self.stream):
            self.row_columns[: words.size].copy_(
                torch.from_numpy(words), non_blocking=True
            )
            self.fields[:, :rows].copy_(fields, non_blocking=True)
            self.values[:, :rows].copy_(values, non_blocking=True)

        # Rows of a bucket replay its graph, or capture it on the
        # computation's stream and pool; any other forward runs eagerly.
        key = RowShape(tokens, rows)
        capture = (
            self.stream
            if self.shapes is not None and self.pool is not None
            else None
        )
        graph = None if capture is None else self.graphs.lookup(key)
        if graph is not None:
            if self.context.plans_attention():
                self._plan(rows, tokens, queries, prefixes, starts)
            with torch.cuda.stream(self.stream):
                graph.replay()
            return GraphMode.REPLAY

        # An eager forward reads the rows' host regions (an eager per-image
        # region of the network); a captured one cannot.
        regions = columns.regions() if capture is None else None
        self._forward(rows, tokens, queries, prefixes, starts, regions)
        if capture is None:
            return GraphMode.EAGER
        # The host plans a capture reads are built outside it.
        self._plan(rows, tokens, queries, prefixes, starts)
        captured = cuda_graph.capture(
            functools.partial(
                self._forward, rows, tokens, queries, prefixes, starts, None
            ),
            stream=capture,
            pool=self.pool,
            peers=self.peers,
        )
        self.graphs.insert(key, None, captured)
        return GraphMode.CAPTURE

    def _batch(
        self,
        rows: int,
        tokens: int,
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
    ):
        """The rows' attention batch, gathered from the slot tables.

        The metadata is gathered on the device and viewed as the attention
        batch the network reads. Every row's queries are bounded by the
        forward's tokens and the longest framed sequence, a bound that does
        not depend on the rows, so a captured forward serves rows of any
        lengths.
        """
        widths = self.tables.widths
        self.metadata.gather(self.row_columns, rows, tokens, widths)
        return self.metadata.batch(
            rows,
            tokens,
            widths,
            queries,
            prefixes,
            starts,
            (False,) * rows,
            writes=self.writes,
            query_bound=min(tokens, self.row_tokens),
        )

    def _plan(self, rows, tokens, queries, prefixes, starts) -> None:
        """Bind the host attention plans of the next replay's rows."""
        with torch.inference_mode(), self.context.activate():
            self.context.bind_attention(
                self._batch(rows, tokens, queries, prefixes, starts)
            )

    def _forward(
        self, rows, tokens, queries, prefixes, starts, regions
    ) -> None:
        """The forward a bucket's graph captures: gather, expand, predict."""
        network, context = self.network, self.context
        predictions = self.predictions
        with torch.inference_mode(), context.activate():
            attention = self._batch(rows, tokens, queries, prefixes, starts)
            if not capturing(self.device):
                context.bind_attention(attention)
            expanded = expand_rows(
                self.fields[:, :rows],
                self.values[:, :rows],
                tokens=tokens,
                dump=self.capacity.tokens,
                regions=regions,
            )
            typed = network.bind_packed(
                source=self.source, rows=expanded.rows, attention=attention
            )
            (prediction,) = network(
                typed,
                state={},
                constants=context.constants,
                workspace=context.workspace_views,
            )["image"]
            if predictions is None:
                return
            # Only the last pipeline stage predicts; its broadcast gives
            # every stage the same values. A mesh without ``pp`` broadcasts
            # locally. A latent token's prediction row lies below the
            # forward's tokens, so the broadcast covers the bucket.
            if prediction is not None:
                predictions.index_copy_(
                    0,
                    expanded.prediction_index,
                    prediction.tensor.to(
                        device=predictions.device,
                        dtype=predictions.dtype,
                        non_blocking=True,
                    ),
                )
            self.pipeline.broadcast(
                predictions[:tokens], src=self.pipeline.size - 1
            )

    def seal(self) -> None:
        self.graphs.seal()

    def close(self, aborted: bool) -> None:
        self.graphs.close(aborted)


@dataclass(frozen=True, slots=True)
class LatentBinding:
    """A trajectory call's binding in the worker's latent ledger.

    Attributes:
        read: The bank holding the committed latent.
        write: The bank the call writes.
        step: The committed step the call evaluates.
        pages: The trajectory's pages, in logical order.
        units: The trajectory's latent rows.
        height: Image height in pixels.
        width: Image width in pixels.
    """

    read: int
    write: int
    step: int
    pages: tuple[int, ...]
    units: int
    height: int
    width: int

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> LatentBinding:
        return cls(
            read=int(value["read"]),
            write=int(value["write"]),
            step=int(value["step"]),
            pages=tuple(int(page) for page in value["pages"]),
            units=int(value["units"]),
            height=int(value["height"]),
            width=int(value["width"]),
        )


class LatentBanks:
    """The worker's two banks of latent pages, which trajectories live in.

    A bank is ``[pages, page_units, width]``; a trajectory owns whole pages,
    its latent rows in logical page order, zero past its last row.

    Args:
        banks: The two banks.
    """

    def __init__(self, banks: tuple[torch.Tensor, torch.Tensor]) -> None:
        first, second = banks
        if first.shape != second.shape or first.ndim != 3:
            raise ValueError("latent banks are two [pages, units, width] views")
        self.banks = (first, second)
        self.page_units = int(first.shape[1])
        self.width = int(first.shape[2])
        self.dtype = first.dtype
        self.device = first.device

    @property
    def units(self) -> int:
        """Latent rows the banks hold, the padding page excluded."""
        return (int(self.banks[0].shape[0]) - 1) * self.page_units

    def padded(self, pages: Sequence[int]) -> int:
        """Rows ``pages`` span."""
        return len(pages) * self.page_units

    def _index(self, pages: Sequence[int]) -> torch.Tensor:
        return torch.tensor(list(pages), dtype=torch.int64).to(
            self.device, non_blocking=True
        )

    def gather(
        self, bank: int, pages: Sequence[int], out: torch.Tensor
    ) -> None:
        """Copy ``bank``'s ``pages`` into ``out``'s leading padded rows."""
        rows = out[: self.padded(pages)].view(
            len(pages), self.page_units, self.width
        )
        torch.index_select(self.banks[bank], 0, self._index(pages), out=rows)

    def scatter(
        self, bank: int, pages: Sequence[int], rows: torch.Tensor
    ) -> None:
        """Copy ``rows``' leading padded rows into ``bank``'s ``pages``."""
        self.banks[bank].index_copy_(
            0,
            self._index(pages),
            rows[: self.padded(pages)].view(
                len(pages), self.page_units, self.width
            ),
        )


class PrefixComputation(Protocol):
    """The text computation a denoiser's guidance prefixes are prefilled on."""

    @property
    def stream(self) -> torch.cuda.Stream: ...

    def prefill_cache(
        self, prefixes: Sequence[tuple[int, Sequence[int]]], pages: np.ndarray
    ) -> Any: ...


@dataclass(slots=True)
class StepCall:
    """One active denoising call of a batch.

    Attributes:
        index: The call's index in its batch.
        slot: Its request slot.
        key: Its request's identity.
        params: Its request's image parameters.
        negative: Its request's admitted negative tokens.
        latent: Its trajectory's binding in the ledger.
        logical: Its request's logical position.
        visible: Its request's visible cache tokens (its conditioning).
        allocated: The allocated tokens of each slot its rows address.
    """

    index: int
    slot: int
    key: Any
    params: ImageParams
    negative: tuple[int, ...]
    latent: LatentBinding
    logical: int
    visible: int
    allocated: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class WarmForward:
    """A synthetic forward warmup runs to warm or capture a bucket.

    Attributes:
        family: ``"step"`` or ``"condition"``.
        shape: The bucket, or the forward's exact shape without buckets.
        rows: Each live row's ``(slot, prefix, tokens)``.
    """

    family: str
    shape: RowShape
    rows: tuple[tuple[int, int, int], ...]

    def attention(self) -> list[list[int]]:
        """The rows' attention rows ``[slot, prefix, query]``."""
        return [[slot, prefix, tokens] for slot, prefix, tokens in self.rows]


def split_rows(
    shape: RowShape, limits: RowLimits, shapes: RowShapes | None
) -> list[int] | None:
    """Row lengths of a forward that selects bucket ``shape`` of ``shapes``.

    A forward selects the least bucket of more rows than it holds, then
    within it the least bucket of more tokens (``RowShapes.select``): so it
    holds at least as many rows as the next smaller row bucket and at least
    as many tokens as the next smaller token bucket of its rows, fewer than
    ``shape``'s, rows of ``limits``' lengths, and leaves the padding row at
    most a row's longest sequence. The most tokens such a forward holds are
    split evenly over the fewest rows. Without buckets (``shapes`` ``None``)
    any forward of fewer rows and tokens qualifies. ``None`` when no such
    forward exists.
    """
    rows = tokens = 0
    if shapes is not None:
        rows = max(
            (
                bucket.rows
                for bucket in shapes.shapes
                if bucket.rows < shape.rows
            ),
            default=0,
        )
        tokens = max(
            (
                bucket.tokens
                for bucket in shapes.shapes
                if bucket.rows == shape.rows and bucket.tokens < shape.tokens
            ),
            default=0,
        )
    for count in range(max(rows, 1), shape.rows):
        low = max(
            shape.tokens - limits.max_row_tokens,
            count * limits.min_row_tokens,
            tokens,
            1,
        )
        high = min(
            shape.tokens - 1,
            count * limits.max_row_tokens,
            limits.max_tokens,
        )
        if low <= high:
            base, extra = divmod(high, count)
            return [base + int(index < extra) for index in range(count)]
    return None


@dataclass(slots=True)
class _Stepping:
    """One stepping image of a denoising batch.

    Attributes:
        call: Its call.
        trajectory: Its open trajectory.
        branches: The guidance branches its step evaluates.
        time: The model timestep of its step.
        pages: Each branch row's selected pages, in evaluation order.
        rows: Each branch row's prefix coordinates and temporal position,
            resolved once its prefixes are prefilled.
    """

    call: StepCall
    trajectory: ImageTrajectory
    branches: BranchSet
    time: float
    pages: list[RowPages] = field(default_factory=list)
    rows: list[BranchRow] = field(default_factory=list)

    @property
    def units(self) -> int:
        return self.call.latent.units


class Denoiser:
    """An image model's denoiser, which its denoiser computation runs.

    Args:
        network: The loaded ``ImageDenoiser``.
        owner: The denoiser's component and path, naming it.
        layout: The denoiser's latent layout.
        stream: The primary stream; ``None`` on the host.
        device: The CUDA ordinal ``stream`` belongs to, -1 on the host.
        scratch: The stream's shared work areas.
        attention: The attention backend the model loaded with.
        tables: The prefix cache's block tables.
        banks: The latent ledger's banks.
        slots: The request slots.
        step_limits: The denoising steps the denoiser admits.
        condition_limits: The condition writes it admits, ``None`` for a
            denoiser that does not frame images.
        step_shapes: The denoising steps' buckets, ``None`` without graphs.
        condition_shapes: The condition writes' buckets, ``None`` without
            graphs.
        pool: The primary stream's graph pool, ``None`` without graphs.
        peers: ``(ordinal, pool)`` of each other device the network computes
            on.
        collectives: The process group of each of the network's
            multi-member communicators on the primary stream, by name.
        frame: The model's framing of an unconditional branch's prefix
            (:func:`framed_prefix`).
        multiple: The sequence-parallel degree, which divides every
            forward's tokens.
    """

    def __init__(
        self,
        *,
        network,
        owner: str,
        layout: ImageLayout,
        stream: torch.cuda.Stream | None,
        device: int,
        scratch,
        attention,
        tables: CacheTables,
        banks: LatentBanks,
        slots: int,
        step_limits: RowLimits,
        condition_limits: RowLimits | None,
        step_shapes: RowShapes | None,
        condition_shapes: RowShapes | None,
        pool: Any | None,
        peers: Sequence[tuple[int, Any]],
        collectives: Mapping[str, Any] | None,
        frame: Callable[[BranchSource], tuple[int, ...]],
        multiple: int = 1,
    ) -> None:
        self.network = network
        self.owner = owner
        self.layout = layout
        self.stream = stream
        self.tables = tables
        self.banks = banks
        self._frame = frame
        self._framed: dict[BranchSource, tuple[int, ...]] = {}
        self.trajectories = Trajectories()
        self.text: PrefixComputation | None = None
        target = banks.device

        step_capacity = _capacity(step_limits, step_shapes, multiple)
        condition_capacity = (
            None
            if condition_limits is None
            else _capacity(condition_limits, condition_shapes, multiple)
        )

        def context(size: RowShape):
            prepared = execution_context(
                network,
                0 if stream is None else stream.cuda_stream,
                attention,
                scratch,
                device=int(device),
                cache=tables.pool,
                collectives=collectives,
            )
            prepared.prepare(TextSize(size.tokens, size.rows))
            return prepared

        # Each family keeps its own context and attention plans, so a
        # condition write between two steps never disturbs a step's plans.
        # Contexts on one stream are prepared largest first: the captured
        # graphs keep addressing one backing of the shared work areas.
        sizes = [("step", step_capacity)]
        if condition_capacity is not None:
            sizes.append(("condition", condition_capacity))
        sizes.sort(key=lambda item: item[1].tokens * item[1].rows, reverse=True)
        contexts = {name: context(size) for name, size in sizes}
        self.contexts = [contexts[name] for name, _ in sizes]

        width = layout.width
        with torch.inference_mode(), torch.cuda.stream(stream):
            # The step's source rows: every stepping image's committed
            # latent, page-padded, at most one page past each row's latents.
            self.workspace = torch.zeros(
                (
                    step_capacity.tokens
                    + step_capacity.rows * banks.page_units,
                    width,
                ),
                dtype=banks.dtype,
                device=target,
            )
            self.predictions = torch.zeros(
                (step_capacity.tokens + 1, width),
                dtype=layout.prediction_dtype,
                device=target,
            )
            # Each slot's reference time of the solver update.
            self.slot_timesteps = torch.zeros(
                (slots + 1, 1), dtype=torch.float32, device=target
            )
            # One initial noise draw of the largest image, in the samples'
            # dtype, which the network's patching reads.
            self.noise = torch.zeros(
                layout.max_sequence_tokens * width,
                dtype=banks.dtype,
                device=target,
            )
            self.conditions = None
            if condition_capacity is not None:
                self.conditions = torch.zeros(
                    (condition_capacity.tokens, width),
                    dtype=banks.dtype,
                    device=target,
                )
        self.steps = RowFamily(
            owner=f"{owner} steps",
            network=network,
            context=contexts["step"],
            tables=tables,
            limits=step_limits,
            capacity=step_capacity,
            source=self.workspace,
            writes=False,
            predictions=self.predictions,
            shapes=step_shapes,
            stream=stream,
            pool=pool,
            peers=peers,
            multiple=multiple,
        )
        self.conditioning = None
        if condition_limits is not None:
            assert condition_capacity is not None
            assert self.conditions is not None
            self.conditioning = RowFamily(
                owner=f"{owner} conditions",
                network=network,
                context=contexts["condition"],
                tables=tables,
                limits=condition_limits,
                capacity=condition_capacity,
                source=self.conditions,
                writes=True,
                predictions=None,
                shapes=condition_shapes,
                stream=stream,
                pool=pool,
                peers=peers,
                multiple=multiple,
            )

    # The text computation's questions (``uniserve.executor.text.context``).

    def sequence_length(self, height: int, width: int) -> int:
        """Tokens of an image's framed latent sequence."""
        return self.layout.sequence_length(height, width)

    def conditioning_positions(
        self, height: int, width: int, start: int
    ) -> np.ndarray:
        """Positions of a latent image inserted as conditioning at ``start``."""
        return self.layout.conditioning_positions(height, width, start)

    def condition_latents(
        self,
        rows: Sequence[ConditionRow],
        pages: np.ndarray,
        stream: torch.cuda.Stream | None,
    ) -> None:
        """Write the keys and values of latent images into their slots.

        ``rows`` are latent images inserted into prompts, each written after
        its slot's visible ``prefix`` at its conditioning ``position``;
        ``pages`` is ``int64 [2G, len(rows)]`` each row's first selected
        page and page count per cache group. The writes run on the
        denoiser's stream after what ``stream`` (the text computation's)
        enqueued before, and ``stream`` orders after them.

        Raises:
            ValueError: The denoiser does not frame images, or a latent
                image's rows disagree with its size.
        """
        family, conditions = self.conditioning, self.conditions
        if family is None or conditions is None:
            raise ValueError(
                "latent feature export requires framed image conditioning"
            )
        if not rows:
            return
        layout = self.layout
        pages = np.asarray(pages, dtype=np.int64)
        lengths = [
            layout.sequence_length(row.height, row.width) for row in rows
        ]
        chunks = _chunks(
            lengths, max_rows=family.live_rows, max_tokens=family.live_tokens
        )
        with (
            _ordered(self.stream, stream),
            torch.inference_mode(),
            torch.cuda.stream(self.stream),
        ):
            for chunk in chunks:
                image_rows = []
                offset = 0
                for position, index in enumerate(chunk):
                    row = rows[index]
                    units = layout.latent_rows(row.height, row.width)
                    features = row.features
                    if features.ndim != 2 or features.shape[0] < units:
                        raise ValueError(
                            "a latent image's features hold fewer rows than "
                            "its latent grid"
                        )
                    conditions[offset : offset + units].copy_(
                        features[:units].to(
                            device=conditions.device,
                            dtype=conditions.dtype,
                            non_blocking=True,
                        )
                    )
                    image_rows.append(
                        ImageRow(
                            image=position,
                            sample=offset,
                            height=row.height,
                            width=row.width,
                            positions=(
                                row.position,
                                row.position + 1,
                                row.position + layout.rope_advance,
                            ),
                            timestep=0.0,
                            noise_scale=self._noise_scale(units),
                            slot=row.slot,
                            prefix=row.prefix,
                            write=True,
                        )
                    )
                    offset += units
                columns = family.fit(RowColumns.of(layout, image_rows))
                family.run(columns, RowPages.of(self.tables, pages[:, chunk]))

    def _noise_scale(self, units: int) -> float:
        return float(self.network.noise_scale.scale(units))

    # Trajectories.

    def attach_text(self, computation: PrefixComputation | None) -> None:
        """Take the text computation that prefills the guidance prefixes."""
        self.text = computation

    def _prefix(self, source: BranchSource) -> Prefix:
        tokens = self._framed.get(source)
        if tokens is None:
            tokens = self._framed[source] = self._frame(source)
        return Prefix(tokens=tokens, reuses_conditioning=False)

    def trajectory(
        self,
        slot: int,
        key: Any,
        params: ImageParams,
        negative: Sequence[int],
    ) -> ImageTrajectory:
        """The open trajectory of request ``key`` at ``slot``.

        A slot opens a trajectory the first time one of its calls computes
        on it, and opens it again when its size changed; opening is
        idempotent, so a batch that fails leaves nothing to roll back. The
        conditioning source reads the request's own context, the negative
        source its admitted negative tokens, and the model frames the
        others.
        """
        size = (params.height, params.width)
        open_ = self.trajectories.get(slot, key)
        if open_ is not None and open_.size == size:
            return open_
        plan = GuidancePlan.make(self.network, params)
        prefixes = {}
        for source in (
            BranchSource.CONDITIONING,
            BranchSource.NEGATIVE_OR_START,
            self.layout.image_unconditional,
        ):
            given = admitted_prefix(source, negative)
            prefixes[source] = (
                given if given is not None else self._prefix(source)
            )
        return self.trajectories.open(
            slot, ImageTrajectory(key, size, plan, prefixes)
        )

    def release(self, slots: Sequence[int]) -> None:
        """Forget the trajectories of retired or restarted slots."""
        self.trajectories.release(slots)

    def prepare_latents(
        self,
        slot: int,
        key: Any,
        params: ImageParams,
        negative: Sequence[int],
        image_index: int,
        latent: LatentBinding,
    ) -> GraphMode:
        """Open a trajectory and write its seeded initial latent.

        The image's noise comes from the request's seed and ``image_index``,
        the index the engine assigned the image; the initial latent fills
        the trajectory's pages of the bank the ledger bound, zero past its
        rows.

        Raises:
            ValueError: The ledger's binding disagrees with the image.
        """
        network, layout = self.network, self.layout
        self.trajectory(slot, key, params, negative)
        size = image.Config(latent.height, latent.width)
        units = layout.latent_rows(latent.height, latent.width)
        padded = self.banks.padded(latent.pages)
        if units != latent.units or padded < units:
            raise ValueError(
                "image trajectory storage must have its canonical patch shape"
            )
        shape = (1, *network.noise_shape("image", size))
        elements = int(np.prod(shape))
        if elements > self.noise.numel():
            raise ValueError("an image's noise exceeds the largest draw")
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            noise = self.noise[:elements].view(shape)
            normal_noise(
                (flow_noise_seed(params.draw_seed, int(image_index)),),
                out=(noise,),
            )
            out = self.workspace[:padded]
            # Physical page padding past the trajectory's rows stays zero.
            out.zero_()
            network.prepare_latents(
                (size,),
                noise={"image": noise},
                state={"image": out[:units].unsqueeze(0)},
                constants={},
                workspace={},
            )
            self.banks.scatter(latent.write, latent.pages, out)
        return GraphMode.EAGER

    def denoise(
        self, calls: Sequence[StepCall], forward: np.ndarray
    ) -> list[GraphMode]:
        """Advance every call's trajectory by one solver step.

        ``forward`` is the worker's ``int64 [5 + 2G, R]`` attention rows of
        the calls (``crates/worker/src/executor/image.rs``): per call, its
        scheduler rows (guidance prefix prefills, then one row per guidance
        branch), then its conditioning row; each with its call index, slot,
        sequence and query lengths, cache write and selected pages per cache
        group. Each call's guidance prefixes not yet evaluated are prefilled
        on the text computation first; then the stepping images' branch rows
        are evaluated in as few packed forwards as their buckets allow, and
        each image's guided prediction advances its committed latent into
        the bank the ledger bound. Returns each call's graph mode.

        Raises:
            ValueError: A call continues past its schedule, its rows
                disagree with its guidance branches, or a prefix exceeds its
                slot.
        """
        forward = np.asarray(forward, dtype=np.int64)
        stepping: list[_Stepping] = []
        fills: list[tuple[_Stepping, PrefixFill, np.ndarray]] = []
        for call in calls:
            entry, prefills = self._resolve(call, forward)
            stepping.append(entry)
            fills.extend((entry, fill, pages) for fill, pages in prefills)

        # Prefixes the steps read for the first time are written before the
        # rows that read them.
        if fills:
            self._prefill(fills)
        for entry in stepping:
            entry.rows = entry.trajectory.branch_rows(
                entry.branches, entry.call.logical
            )

        modes: dict[int, GraphMode] = {}
        chunks = _chunks(
            [
                len(entry.branches) * (entry.units + self.layout.framing_tokens)
                for entry in stepping
            ],
            max_rows=self.steps.live_rows,
            max_tokens=self.steps.live_tokens,
            rows=[len(entry.branches) for entry in stepping],
        )
        for chunk in chunks:
            mode = self._step([stepping[index] for index in chunk])
            for index in chunk:
                modes[stepping[index].call.index] = mode
        return [modes[call.index] for call in calls]

    def _resolve(
        self, call: StepCall, forward: np.ndarray
    ) -> tuple[_Stepping, list[tuple[PrefixFill, np.ndarray]]]:
        """Open a call's step and resolve its branches' prefixes and pages.

        ``forward`` holds the batch's attention rows (:meth:`denoise`).
        Returns the call's stepping image, its branch rows' pages selected,
        and each prefill its step needs first with the prefill row's page
        columns.
        """
        rows = np.flatnonzero(forward[0] == call.index)
        if len(rows) < 1:
            raise ValueError("a denoising call has no conditioning row")
        descriptors, conditioning = rows[:-1], rows[-1]
        trajectory = self.trajectory(
            call.slot, call.key, call.params, call.negative
        )
        branches, time = trajectory.plan.step(call.latent.step)

        def coordinates(slot: int, visible: int) -> Coordinates:
            return Coordinates(
                slot=slot,
                visible=visible,
                capacity=int(call.allocated.get(slot, 0)),
            )

        trajectory.begin_call(coordinates(call.slot, call.visible))
        slots = [
            ForwardSlot(
                slot=int(forward[1, row]),
                seq_len=int(forward[2, row]),
                query_len=int(forward[3, row]),
            )
            for row in descriptors
        ]
        image_unconditional = self.layout.image_unconditional
        prefills = []
        for fill in trajectory.resolve_branches(
            branches, image_unconditional, slots, coordinates
        ):
            # The prefill's own row: a cache write of exactly its tokens
            # into its slot.
            row = next(
                (
                    int(row)
                    for row in descriptors
                    if forward[4, row]
                    and int(forward[1, row]) == fill.slot
                    and int(forward[3, row]) == len(fill.tokens)
                ),
                None,
            )
            if row is None:
                raise ValueError(
                    "a guidance prefix's prefill has no forward row"
                )
            prefills.append((fill, forward[_FORWARD_FIELDS:, row]))

        # A branch reading the request's own context reads the conditioning
        # row's pages; any other its descriptor's, which follow the call's
        # prefills in evaluation order.
        entry = _Stepping(call, trajectory, branches, time)
        split = len(descriptors) - len(branches)
        for position, branch in enumerate(branches):
            row = (
                conditioning
                if trajectory.reuses_conditioning(branch, image_unconditional)
                else descriptors[split + position]
            )
            entry.pages.append(
                RowPages.of(
                    self.tables, forward[_FORWARD_FIELDS:, row : row + 1]
                )
            )
        return entry, prefills

    def _prefill(
        self, fills: Sequence[tuple[_Stepping, PrefixFill, np.ndarray]]
    ) -> None:
        """Prefill guidance prefixes on the text computation.

        The prefill is ordered against the denoiser's stream both ways.
        """
        text = self.text
        if text is None:
            raise NotImplementedError(
                "an image step whose guidance prefixes were not prefilled "
                "requires a text computation to prefill them"
            )
        prefixes = [(fill.slot, fill.tokens) for _, fill, _ in fills]
        pages = np.stack([columns for _, _, columns in fills], axis=1)
        with _ordered(text.stream, self.stream):
            text.prefill_cache(prefixes, pages)
        for entry, fill, _ in fills:
            entry.trajectory.prefilled(fill.slot, len(fill.tokens))

    def _step(self, stepping: Sequence[_Stepping]) -> GraphMode:
        """Evaluate one packed forward of ``stepping``'s branch rows.

        Each image's solver update then advances its samples and scatters
        its successor.
        """
        layout, banks = self.layout, self.banks
        rows: list[ImageRow] = []
        parts: list[RowPages] = []
        placed: list[tuple[_Stepping, int, int]] = []
        sample = prediction = 0
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for position, entry in enumerate(stepping):
                latent = entry.call.latent
                banks.gather(latent.read, latent.pages, self.workspace[sample:])
                placed.append((entry, sample, prediction))
                scale = self._noise_scale(latent.units)
                for (_, coordinates, temporal), pages in zip(
                    entry.rows, entry.pages, strict=True
                ):
                    rows.append(
                        ImageRow(
                            image=position,
                            sample=sample,
                            height=latent.height,
                            width=latent.width,
                            positions=(temporal,) * 3,
                            timestep=entry.time,
                            noise_scale=scale,
                            prediction=prediction,
                            slot=coordinates.slot,
                            prefix=coordinates.visible,
                        )
                    )
                    parts.append(pages)
                    prediction += latent.units
                sample += banks.padded(latent.pages)
            columns = self.steps.fit(RowColumns.of(layout, rows))
            mode = self.steps.run(columns, RowPages.join(self.tables, parts))
            for entry, offset, first in placed:
                self._integrate(entry, offset, first)
        return mode

    def _integrate(self, entry: _Stepping, sample: int, first: int) -> None:
        """Advance one image's samples by its step and scatter them."""
        latent = entry.call.latent
        plan = entry.trajectory.plan
        if plan.schedules is None or plan.guidance is None:
            raise ValueError("an image trajectory lost its schedules")
        schedule = plan.schedules["image"]
        guidance = plan.guidance
        units, step = latent.units, latent.step
        values = self.workspace[sample : sample + units]
        outputs = {
            branch: self.predictions[
                first + position * units : first + (position + 1) * units
            ]
            for position, branch in enumerate(entry.branches)
        }
        guided = guidance.combine(outputs, schedule, step)
        timestep = self.slot_timesteps[entry.call.slot]
        timestep.fill_(entry.time)
        advance_(
            self.network,
            {"image": (LatentInput(values, timestep),)},
            {"image": (guided,)},
            plan.schedules,
            schedule.step(step),
        )
        self.banks.scatter(latent.write, latent.pages, self.workspace[sample:])

    # Warmup.

    def warm_forwards(
        self, slots: int
    ) -> tuple[list[WarmForward], list[WarmForward]]:
        """The synthetic forwards warmup runs: the steps', then the writes'.

        A denoiser that writes no latent conditioning has no writes. Under
        graphs, one forward per bucket in capture order (by ``T *
        R``, then ``T``, descending), which warms and captures it; without,
        one forward of the shortest row, which selects the families'
        kernels. A step's rows read ``WARM_PREFIX`` tokens of scratch slots
        ``1..slots``; a condition write's rows write their tokens from
        position zero, each on a scratch slot of its own where the slots
        allow.
        """
        families = [("step", self.steps, False)]
        if self.conditioning is not None:
            families.append(("condition", self.conditioning, True))
        result: list[list[WarmForward]] = [[], []]
        for position, (name, family, writes) in enumerate(families):
            limits = family.limits
            if family.shapes is None:
                buckets = [RowShape(limits.min_row_tokens, 1)]
                lengths: list[list[int] | None] = [[limits.min_row_tokens]]
            else:
                buckets = family.shapes.capture_order()
                lengths = [
                    split_rows(shape, limits, family.shapes)
                    for shape in buckets
                ]
            forwards = []
            for shape, split in zip(buckets, lengths, strict=True):
                if split is None:
                    raise ValueError(
                        f"{family.owner}: no forward selects bucket {shape}"
                    )
                forwards.append(
                    WarmForward(
                        name,
                        shape,
                        tuple(
                            (
                                1 + index % max(slots, 1),
                                0 if writes else WARM_PREFIX,
                                tokens,
                            )
                            for index, tokens in enumerate(split)
                        ),
                    )
                )
            result[position] = forwards
        steps, conditions = result
        return steps, conditions

    def warm(self, forward: WarmForward, pages: np.ndarray) -> GraphMode:
        """Run warmup forward ``forward`` over its rows' scratch pages.

        ``pages`` is ``int64 [2G, rows]`` each row's selected pages. A
        step's rows read zero latents and predict into the predictions; a
        condition write's rows write their keys and values into the
        scratch slots.
        """
        layout = self.layout
        writes = forward.family == "condition"
        family = self.conditioning if writes else self.steps
        if family is None:
            raise ValueError("the denoiser writes no conditions")
        rows = []
        sample = 0
        for index, (slot, prefix, tokens) in enumerate(forward.rows):
            units = tokens - layout.framing_tokens
            rows.append(
                ImageRow(
                    image=index,
                    sample=sample,
                    height=layout.downsample,
                    width=layout.downsample * units,
                    positions=(prefix,) * 3,
                    timestep=0.0 if writes else 0.5,
                    noise_scale=self._noise_scale(units),
                    prediction=sample,
                    slot=slot,
                    prefix=prefix,
                    write=writes,
                )
            )
            sample += units
        columns = RowColumns.of(layout, rows)
        if family.shapes is None:
            columns = family.fit(columns)
        else:
            columns = columns.padded(
                forward.shape.tokens, forward.shape.rows, family.row_tokens
            )
        return family.run(columns, RowPages.of(self.tables, pages))

    def seal(self) -> None:
        self.steps.seal()
        if self.conditioning is not None:
            self.conditioning.seal()

    def kernels(self) -> list[str]:
        """The kernels both families' operators selected, as JSON objects."""
        return [
            record
            for context in self.contexts
            for record in kernel_records(context.kernels())
        ]

    def close(self, aborted: bool) -> None:
        """Release the graphs, then the contexts; every release is tried."""
        failure: BaseException | None = None
        families = [self.steps, self.conditioning]
        for family in families:
            if family is None:
                continue
            try:
                family.close(aborted)
            except Exception as error:  # noqa: BLE001 - close every part
                failure = failure or error
        for context in self.contexts:
            try:
                context.close(aborted=bool(aborted))
            except Exception as error:  # noqa: BLE001 - close every part
                failure = failure or error
        if failure is not None:
            raise failure


def _chunks(
    tokens: Sequence[int],
    *,
    max_rows: int,
    max_tokens: int,
    rows: Sequence[int] | None = None,
) -> list[list[int]]:
    """Consecutive groups of items within a forward's rows and tokens.

    Item ``i`` holds ``tokens[i]`` tokens in ``rows[i]`` rows (one each by
    default).

    Raises:
        ValueError: An item alone exceeds a forward.
    """
    counts = [1] * len(tokens) if rows is None else list(rows)
    chunks: list[list[int]] = []
    used_rows = used_tokens = 0
    for index, (count, length) in enumerate(zip(counts, tokens, strict=True)):
        if count > max_rows or length > max_tokens:
            raise ValueError(
                f"an image of {count} rows and {length} tokens exceeds a "
                f"forward's {max_rows} rows and {max_tokens} tokens"
            )
        if (
            chunks
            and used_rows + count <= max_rows
            and used_tokens + length <= max_tokens
        ):
            chunks[-1].append(index)
            used_rows += count
            used_tokens += length
        else:
            chunks.append([index])
            used_rows, used_tokens = count, length
    return chunks
