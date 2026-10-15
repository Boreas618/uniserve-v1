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
| ``write_indices`` | int64 | ``[Tb, T]`` |

Per forward the rank packs the rows' slots, lengths and selected page
counts into one host column (``row_columns``, the Rust half's
``AttentionMetadata``); :meth:`AttentionMetadata.gather` derives the
metadata from it and the request slots' installed tables (``unit_tables``,
``start_pages``, ``table_shapes``) in one launch, and
:meth:`AttentionMetadata.batch` views it as the ``AttentionBatch`` the
model's attention layers read. A decode gathers its rows from the slot
state into the same tensors instead
(:func:`~uniserve.nn.attention.gather.gather_request_decode_inputs`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

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


class AttentionMetadata:
    """The attention metadata of one runner's forwards.

    Captured graphs address these tensors, so they live as long as the
    runner. Rows no gather wrote are inert: they select no page and write
    no cache entry. The tensors are allocated on the current stream.

    Args:
        page_tokens: Each numerical table's tokens per page.
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
        page_tokens: Sequence[int],
        windowed: Sequence[int],
        unit_tables: torch.Tensor,
        start_pages: torch.Tensor,
        table_shapes: torch.Tensor,
        rows: int,
        tokens: int,
        width: int,
        device: torch.device,
    ) -> None:
        self.page_tokens = tuple(int(value) for value in page_tokens)
        self.windowed = tuple(bool(value) for value in windowed)
        self.unit_tables = unit_tables
        self.slot_start_pages = start_pages
        self.table_shapes = table_shapes
        rows, tokens, tables = int(rows), int(tokens), len(self.page_tokens)

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int32, device=device)

        self.block_tables = zeros(tables, rows, int(width))
        self.start_pages = zeros(tables, rows)
        self.cache_lengths = zeros(rows)
        self.query_lengths = zeros(rows)
        self.query_offsets = zeros(rows + 1)
        self.prefix_offsets = zeros(rows + 1)
        self.write_indices = torch.full(
            (tables, tokens), -1, dtype=torch.int64, device=device
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
            write_indices=self.write_indices,
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
    ) -> AttentionBatch:
        """The attention batch of ``rows`` rows over the gathered metadata.

        ``queries`` and ``prefixes`` are the rows' host lengths, ``starts``
        each windowed table's first pages (``rows`` per table), ``causal``
        each row's causality and ``widths`` each table's page width. When
        the rows' causality differs, the caller staged it in
        ``causal_values`` for operators that consume the device column
        (``None`` for a forward whose rows agree). Rows that write the
        cache read paged inputs with write indices; read-only rows
        (``writes`` false) read their segments without writing.
        """
        rows, tokens = int(rows), int(tokens)
        query = SequenceLengths(
            values=self.query_lengths[:rows],
            offsets=self.query_offsets[: rows + 1],
            host=tuple(int(value) for value in queries),
        )
        prefix = SequenceLengths(
            values=self.cache_lengths[:rows],
            offsets=self.prefix_offsets[: rows + 1],
            host=tuple(int(value) for value in prefixes),
        )
        mode = tuple(bool(value) for value in causal)
        flags = (
            None
            if causal_values is None or len(set(mode)) == 1
            else causal_values[:rows]
        )
        entries: dict[int, AttentionInput] = {}
        for table, width in enumerate(widths):
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
                self.page_tokens[table],
                first,
                host_first,
            )
            if writes:
                entries[table] = PagedInput(
                    query,
                    prefix,
                    blocks,
                    self.write_indices[table, :tokens],
                    mode,
                    flags,
                )
            else:
                entries[table] = SegmentedInput(
                    query,
                    prefix,
                    blocks,
                    None,
                    query.values[:, None].expand(-1, max(query.host or (1,))),
                    True,
                )
        return AttentionBatch(entries, query)
