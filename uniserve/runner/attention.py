"""A runner's attention metadata over the prefix cache.

A runner that attends to the prefix cache through block tables (a causal
LM, an image denoiser) owns one set of device metadata for ``R`` rows and
``T`` tokens over ``Tb`` numerical block tables of width ``W``:

| Tensor | dtype | Shape |
| --- | --- | --- |
| ``block_tables`` | int32 | ``[Tb, R, W]`` |
| ``start_pages`` | int32 | ``[Tb, R]`` |
| ``cache_lengths``, ``query_lengths`` | int32 | ``[R]`` |
| ``query_offsets``, ``prefix_offsets`` | int32 | ``[R + 1]`` |
| ``slot_mapping`` | int64 | ``[Tb, T]`` |

Per forward the executor packs the rows' slots, lengths and selected page
counts into one host column (``row_columns``,
:func:`uniserve.executor.text.rows.pack_rows`);
:meth:`AttentionMetadata.gather` derives the metadata from it and the
request slots' installed tables (``unit_tables``, ``start_pages``,
``table_shapes``) in one launch, and :meth:`AttentionMetadata.batch` views
it as the ``AttentionBatch`` the model's attention layers read. A decode
gathers its rows from the slot state into the same tensors instead
(:func:`~uniserve.nn.attention.gather.gather_request_decode_inputs`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from uniserve.executor.arena import ALLOCATION_ROUNDING, aligned
from uniserve.nn.attention import (
    AttentionBatch,
    AttentionInput,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.nn.attention.gather import gather_request_rows

__all__ = ["AttentionMetadata"]


def _tensors(
    rows: int, tokens: int, tables: int, width: int
) -> tuple[tuple[str, torch.dtype, tuple[int, ...], int], ...]:
    """``(name, dtype, shape, initial value)`` of every metadata tensor.

    The one statement of the layout the module's table describes, which
    both the allocation and :meth:`AttentionMetadata.device_bytes` read. The
    slot mapping starts at -1, so rows no gather wrote write no cache entry.
    """
    return (
        ("block_tables", torch.int32, (tables, rows, width), 0),
        ("start_pages", torch.int32, (tables, rows), 0),
        ("cache_lengths", torch.int32, (rows,), 0),
        ("query_lengths", torch.int32, (rows,), 0),
        ("query_offsets", torch.int32, (rows + 1,), 0),
        ("prefix_offsets", torch.int32, (rows + 1,), 0),
        ("slot_mapping", torch.int64, (tables, tokens), -1),
    )


class AttentionMetadata:
    """The attention metadata of one runner's forwards.

    Captured graphs address these tensors, so they live as long as the
    runner. Rows no gather wrote are inert: they select no page and write
    no cache entry. The tensors are allocated on the current stream.

    Args:
        block_size: Each numerical table's tokens per page.
        windowed: Whether each table reads from a first selected page (1)
            or from its first page (0).
        unit_tables: The request slots' installed unit tables.
        start_pages: The request slots' first pages of windowed tables.
        table_shapes: The request slots' table shapes.
        rows: Rows of the largest forward.
        tokens: Tokens of the largest forward.
        width: The widest page width a row selects from one table.
        device: The device the forwards compute on.
    """

    def __init__(
        self,
        block_size: Sequence[int],
        windowed: Sequence[int],
        unit_tables: torch.Tensor,
        start_pages: torch.Tensor,
        table_shapes: torch.Tensor,
        rows: int,
        tokens: int,
        width: int,
        device: torch.device,
    ) -> None:
        self.block_size = tuple(int(value) for value in block_size)
        self.windowed = tuple(bool(value) for value in windowed)
        self.unit_tables = unit_tables
        self.slot_start_pages = start_pages
        self.table_shapes = table_shapes

        tensors = {
            name: torch.full(shape, value, dtype=dtype, device=device)
            for name, dtype, shape, value in _tensors(
                int(rows), int(tokens), len(self.block_size), int(width)
            )
        }
        # Tables the model's forwards do not read: the draft cache's, whose
        # pages the drafter reads through a batch of its own.
        self.omitted: frozenset[int] = frozenset()
        self.block_tables = tensors["block_tables"]
        self.start_pages = tensors["start_pages"]
        self.cache_lengths = tensors["cache_lengths"]
        self.query_lengths = tensors["query_lengths"]
        self.query_offsets = tensors["query_offsets"]
        self.prefix_offsets = tensors["prefix_offsets"]
        self.slot_mapping = tensors["slot_mapping"]

    @staticmethod
    def device_bytes(rows: int, tokens: int, tables: int, width: int) -> int:
        """Device bytes the metadata of a forward's bounds allocates.

        ``rows`` and ``tokens`` bound the largest forward, ``tables`` counts
        the numerical block tables and ``width`` is the widest page width a
        row selects, as the constructor takes them. Each tensor is a
        separate allocation, which torch's caching allocator rounds up to
        ``ALLOCATION_ROUNDING`` bytes (:mod:`uniserve.executor.arena`);
        capacity planning reserves these bytes beside an executor's other
        buffers.
        """
        return sum(
            aligned(
                max(math.prod(shape) * dtype.itemsize, 1), ALLOCATION_ROUNDING
            )
            for _, dtype, shape, _ in _tensors(
                int(rows), int(tokens), int(tables), int(width)
            )
        )

    def gather(
        self,
        row_columns: torch.Tensor,
        rows: int,
        tokens: int,
        widths: Sequence[int],
    ) -> None:
        """Gather ``rows`` rows of ``tokens`` tokens from the slot tables.

        Fills the block tables, start pages, lengths, offsets and write
        indices on the current stream from the packed ``row_columns``.
        """
        gather_request_rows(
            columns=row_columns,
            rows=int(rows),
            tokens=int(tokens),
            width=int(max(widths)),
            request_unit_tables=self.unit_tables,
            request_start_pages=self.slot_start_pages,
            table_shapes=self.table_shapes,
            block_tables=self.block_tables,
            start_pages=self.start_pages,
            cache_lengths=self.cache_lengths,
            query_lengths=self.query_lengths,
            query_offsets=self.query_offsets,
            prefix_offsets=self.prefix_offsets,
            slot_mapping=self.slot_mapping,
        )

    def batch(
        self,
        rows: int,
        tokens: int,
        widths: Sequence[int],
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: Sequence[int],
        causal_values: torch.Tensor | None = None,
        writes: bool = True,
        query_bound: int | None = None,
    ) -> AttentionBatch:
        """The attention batch of ``rows`` rows over the gathered metadata.

        ``queries`` and ``prefixes`` are the rows' host lengths, ``starts``
        each windowed table's first pages (``rows`` per table), ``causal``
        each row's causality and ``widths`` each table's page width.
        ``causal_values`` is the device column operators read each row's
        causality from, which the caller staged when the forward reads it
        there (its rows differ, or its captured graph serves rows of either
        causality); ``None`` applies ``causal`` to the rows. Rows that write the
        cache read paged inputs with a slot mapping; read-only rows
        (``writes`` false) read their segments without writing.
        ``query_bound`` bounds every row's queries independently of these
        rows (``SequenceLengths.bound``), so a captured batch serves rows
        of other lengths; read-only rows' visibility spans it. ``None``
        bounds them by the longest of these rows.
        """
        rows, tokens = int(rows), int(tokens)
        query = SequenceLengths(
            values=self.query_lengths[:rows],
            offsets=self.query_offsets[: rows + 1],
            host=tuple(int(value) for value in queries),
            bound=None if query_bound is None else int(query_bound),
        )
        prefix = SequenceLengths(
            values=self.cache_lengths[:rows],
            offsets=self.prefix_offsets[: rows + 1],
            host=tuple(int(value) for value in prefixes),
        )
        mode = tuple(bool(value) for value in causal)
        flags = None if causal_values is None else causal_values[:rows]
        entries: dict[int, AttentionInput] = {}
        for table, width in enumerate(widths):
            if table in self.omitted:
                continue
            first = None
            host_first = None
            if self.windowed[table]:
                first = self.start_pages[table, :rows]
                host_first = tuple(
                    int(value)
                    for value in starts[table * rows : (table + 1) * rows]
                )
            blocks = BlockTable(
                self.block_tables[table, :rows, : int(width)],
                self.block_size[table],
                first,
                host_first,
            )
            if writes:
                entries[table] = PagedInput(
                    query,
                    prefix,
                    blocks,
                    self.slot_mapping[table, :tokens],
                    mode,
                    flags,
                )
            else:
                entries[table] = SegmentedInput(
                    query,
                    prefix,
                    blocks,
                    None,
                    query.values[:, None].expand(
                        -1, max(query.maximum or 1, 1)
                    ),
                    True,
                )
        return AttentionBatch(entries, query)
