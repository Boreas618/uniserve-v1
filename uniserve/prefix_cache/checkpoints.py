"""Checkpoints of the cache layers kept per request slot.

A cache layer whose type is indexed by request slot
(``StateConfig.indexing == "states"``, such as a recurrent state) keeps a
request's state in the request's slot (:mod:`uniserve.prefix_cache.slots`)
from one forward to the next. The unit pool's state group keeps the same
state at token prefixes, so that a request whose prefix is reused resumes
from it instead of computing the prefix again: the group's page ``k`` holds
the state after token ``(k + 1) * C - 1``, its checkpoint at the boundary
``(k + 1) * C``, where ``C`` is the group's block size (``interval``), a
multiple of every K/V block size.

The engine decides which pages hold checkpoints. A row's block tables of
the state group name a page's units for a page that holds a checkpoint or
receives one in the forward, and the padding sentinel, unit 0, for a page
that holds none. Within a forward of rows with prefix lengths ``p`` and
query lengths ``q``:

- a row whose tables name the page ending at its prefix (``p > 0`` a
  multiple of ``C``) starts from that page's state (:meth:`Checkpoints.
  restore`); any other row continues its slot's state, or starts from
  zero when it has no prefix (:func:`resume`). A request admitted with a
  reused prefix therefore resumes from the checkpoint at the reused
  boundary, whatever its slot held;
- a row writes the state it reaches at its last boundary ``b = (p + q) //
  C * C > p`` into the page ending at ``b`` when its tables name units for
  it (:meth:`Checkpoints.boundary`, :meth:`Checkpoints.write`), and its
  final state into its slot (:func:`keep`).

A checkpoint copies every slot field of the layer's type (the values and
the initialization flags), so restoring one leaves the slot as the forward
that wrote the checkpoint left it at the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from uniserve.nn.attention import AttentionBatch

    from . import CacheBinding
    from .units import CheckpointLayout


def _lengths(
    attention: AttentionBatch, table: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's prefix and query lengths, int64 ``[rows]`` on the device."""
    entry = attention.entry(table)
    prefixes = getattr(entry, "prefixes", None)
    if prefixes is None:
        raise ValueError(
            "a cache layer kept per request slot reads paged attention rows"
        )
    return (
        prefixes.values.to(torch.int64),
        entry.queries.values.to(torch.int64),
    )


class Lookups:
    """The pages the rows of a forward restore from and checkpoint into.

    Every layer of the state group reads the same tables, so the layers of
    one pool share one ``Lookups``: the first layer of a forward to ask
    computes them, and the others of the same forward (the same attention
    batch) reuse them.
    """

    def __init__(self) -> None:
        self._batch: AttentionBatch | None = None
        self._values: dict[str, tuple[torch.Tensor, ...]] = {}

    def get(self, attention: AttentionBatch, kind: str, compute):
        """The values ``compute()`` returns for ``attention``, computed
        once per forward and kind.
        """  # noqa: D205
        if self._batch is not attention:
            self._batch, self._values = attention, {}
        if kind not in self._values:
            self._values[kind] = compute()
        return self._values[kind]


class Checkpoints:
    """One layer's checkpoints in the pages of the pool's state group.

    Args:
        planes: The pool's key and value planes, each uint8 ``[columns,
            num_units, plane_bytes]``.
        layout: Where the layer's checkpoint lies in a page.
        tables: The state group's block tables: table ``tables[j]`` names
            unit ``j`` of every page.
        interval: Tokens between two page boundaries, the group's block
            size.
        lookups: The lookups the group's layers share; the layer's own
            without one.
    """

    def __init__(
        self,
        planes: tuple[torch.Tensor, torch.Tensor],
        layout: CheckpointLayout,
        *,
        tables: range,
        interval: int,
        lookups: Lookups | None = None,
    ) -> None:
        self._planes = planes
        self.layout = layout
        self.tables = tables
        self.interval = int(interval)
        self._lookups = Lookups() if lookups is None else lookups
        columns, _, plane_bytes = planes[0].shape
        self._plane_bytes = int(plane_bytes)

        # The run's planes in the page's plane order: key planes of every
        # unit and column, then value planes. For each field half, the run
        # positions it holds, their units' positions in the page and their
        # columns, as index vectors on the planes' device.
        half = len(tables) * int(columns)
        runs: list[list[tuple[int, int, int]]] = [[], []]
        for index in range(layout.planes):
            field, slot = divmod(layout.first_plane + index, half)
            position, column = divmod(slot, int(columns))
            runs[field].append((index, position, column))
        device = planes[0].device
        self._runs = tuple(
            tuple(
                torch.tensor(
                    [entry[axis] for entry in run],
                    dtype=torch.int64,
                    device=device,
                )
                for axis in range(3)
            )
            for run in runs
        )

    def units(
        self, attention: AttentionBatch, pages: torch.Tensor
    ) -> torch.Tensor:
        """Each row's units of logical page ``pages[row]``.

        Returns int64 ``[rows, units_per_page]``; a row whose page is
        negative or outside its tables receives the padding sentinel, as a
        page its tables name without a checkpoint does. The group's tables
        share their start pages.
        """
        # [units_per_page, rows, width]: every table's page columns.
        indices = torch.stack(
            [
                attention.entry(table).block_table.indices
                for table in self.tables
            ]
        )
        start_page = attention.entry(self.tables[0]).block_table.start_page
        start = (
            torch.zeros_like(pages)
            if start_page is None
            else start_page.to(torch.int64)
        )
        width = indices.shape[2]
        column = pages - start
        held = (pages >= 0) & (column >= 0) & (column < width)
        selected = column.clamp(0, max(width - 1, 0)).view(1, -1, 1)
        unit = indices.gather(2, selected.expand(indices.shape[0], -1, 1))
        return torch.where(held, unit.squeeze(2).to(torch.int64), 0).T

    def _gather(self, units: torch.Tensor) -> torch.Tensor:
        """The run's planes of each row's page, uint8 ``[rows, planes,
        plane_bytes]``.
        """  # noqa: D205
        rows = units.shape[0]
        values = torch.empty(
            (rows, self.layout.planes, self._plane_bytes),
            dtype=torch.uint8,
            device=units.device,
        )
        for planes, (index, position, column) in zip(
            self._planes, self._runs, strict=True
        ):
            if not index.numel():
                continue
            selected = units.index_select(1, position)
            values[:, index] = planes[column.expand_as(selected), selected]
        return values

    def _fields(self, run: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each field of ``[rows, planes, plane_bytes]`` checkpoint bytes,
        viewed in place as ``[rows, *extent]``.
        """  # noqa: D205
        flat = run.view(run.shape[0], -1)
        fields = {}
        for name, offset, shape, dtype in self.layout.fields:
            size = torch.empty((), dtype=dtype).element_size()
            count = 1
            for extent in shape:
                count *= extent
            fields[name] = (
                flat[:, offset : offset + count * size]
                .view(dtype)
                .view(run.shape[0], *shape)
            )
        return fields

    def read(self, units: torch.Tensor) -> dict[str, torch.Tensor]:
        """The checkpoint each row's page holds, by slot field: ``[rows,
        *extent]`` tensors the call allocates.
        """  # noqa: D205
        return self._fields(self._gather(units))

    def write(
        self,
        units: torch.Tensor,
        values: Mapping[str, torch.Tensor],
        written: torch.Tensor,
    ) -> None:
        """Write each row's checkpoint ``values`` into its page.

        ``values`` holds every slot field as ``[rows, *extent]``; a row
        that ``written`` (bool ``[rows]``) clears writes nothing. Its
        writes address the padding sentinel with the sentinel's own bytes,
        so the sentinel keeps them and every write has one shape whatever
        the rows; a written row's page belongs to its request alone.
        """
        rows = units.shape[0]
        run = torch.zeros(
            (rows, self.layout.planes, self._plane_bytes),
            dtype=torch.uint8,
            device=units.device,
        )
        for name, field in self._fields(run).items():
            field.copy_(values[name])
        sentinel = self._gather(torch.zeros_like(units[:1]))
        run = torch.where(written.view(rows, 1, 1), run, sentinel)
        units = torch.where(written.view(rows, 1), units, 0)
        for planes, (index, position, column) in zip(
            self._planes, self._runs, strict=True
        ):
            if not index.numel():
                continue
            selected = units.index_select(1, position)
            planes[column.expand_as(selected), selected] = run[:, index]

    def restore(
        self, attention: AttentionBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Rows starting at a checkpoint, and the checkpoints they start at.

        Returns bool ``[rows]``, whether the row's tables name the page
        ending at its prefix, and that page's fields (``[rows, *extent]``,
        unspecified for the other rows).
        """

        def lookup():
            prefixes, _ = _lengths(attention, self.tables[0])
            pages = torch.where(
                (prefixes > 0) & (prefixes % self.interval == 0),
                prefixes // self.interval - 1,
                -1,
            )
            units = self.units(attention, pages)
            return units[:, 0] != 0, units

        restored, units = self._lookups.get(attention, "restore", lookup)
        return restored, self.read(units)

    def boundary(
        self, attention: AttentionBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where each row writes a checkpoint, and into which units.

        Returns int64 ``[rows]``, the row's last boundary within its query
        whose page its tables name, or -1 when it writes none, and the
        units of that page (``[rows, units_per_page]``).
        """

        def lookup():
            prefixes, queries = _lengths(attention, self.tables[0])
            boundary = (prefixes + queries) // self.interval * self.interval
            pages = torch.where(
                boundary > prefixes, boundary // self.interval - 1, -1
            )
            units = self.units(attention, pages)
            return torch.where(units[:, 0] != 0, boundary, -1), units

        return self._lookups.get(attention, "boundary", lookup)


def chains(
    rows: torch.Tensor, prefixes: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """The first row of each row's chain, int64 ``[rows]``.

    Consecutive rows of one request in one forward (a prompt's text and the
    blocks between it, or warmup's rows) continue each other: a row of the
    same request slot as the row before it, other than the padding slot,
    at the position after it continues it. A chain of such rows is one
    sequence: it starts from its first row's state (:func:`resume`) and
    leaves its last row's in the slot. ``rows`` are the forward's request
    slots, ``prefixes`` and ``lengths`` each row's prefix and query
    lengths.
    """
    previous = torch.cat((rows.new_zeros(1), rows[:-1]))
    after = torch.cat((prefixes.new_full((1,), -1), (prefixes + lengths)[:-1]))
    continues = (rows == previous) & (rows != 0) & (prefixes == after)
    position = torch.arange(rows.shape[0], device=rows.device)
    return torch.cummax(torch.where(continues, 0, position), 0).values


def resume(
    binding: CacheBinding,
    attention: AttentionBatch,
    rows: torch.Tensor,
    *,
    restore: bool = True,
) -> dict[str, torch.Tensor]:
    """Each row's state at its prefix, by slot field: ``[rows, *extent]``.

    The state is the checkpoint its tables name at its prefix, else its
    slot's state, and zero for a row without a prefix (its request's first
    forward). ``rows`` are the forward's request slots. A caller that knows
    no row's prefix ends at a page boundary passes ``restore=False`` to
    skip the checkpoint lookup.
    """
    banks = binding.state.buffers()
    prefixes, _ = _lengths(
        attention,
        binding.checkpoints.tables[0]
        if binding.checkpoints is not None
        else next(iter(attention.entries)),
    )
    started = prefixes > 0
    values = {}
    for name, bank in banks.items():
        value = bank.index_select(0, rows)
        mask = started.view(-1, *(1,) * (value.ndim - 1))
        values[name] = torch.where(mask, value, torch.zeros_like(value))
    if binding.checkpoints is not None and restore:
        restored, checkpoint = binding.checkpoints.restore(attention)
        for name, value in values.items():
            mask = restored.view(-1, *(1,) * (value.ndim - 1))
            values[name] = torch.where(mask, checkpoint[name], value)
    return values


def keep(
    binding: CacheBinding,
    rows: torch.Tensor,
    values: Mapping[str, torch.Tensor],
) -> None:
    """Store each row's final state ``values`` into its request slot.

    Padding rows address slot 0, the padding slot, whose state no request
    reads.
    """
    for name, bank in binding.state.buffers().items():
        bank.index_copy_(0, rows, values[name].to(bank.dtype))
