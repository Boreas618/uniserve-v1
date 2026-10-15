"""Paged attention metadata of a binding's fixed attention columns.

A binding that attends to the prefix cache (a causal LM, an image denoiser)
receives one set of attention columns from the worker at bind time, with
``R`` rows and ``T`` tokens of capacity over ``Tb`` numerical block tables
of the widest graph width ``W``:

| Column | dtype | Shape |
| --- | --- | --- |
| ``row_columns`` | int64 | host-packed rows (``gather_request_rows``) |
| ``block_tables`` | int32 | ``[Tb, R, W]`` |
| ``start_pages`` | int32 | ``[Tb, R]`` |
| ``cache_lengths``, ``query_lengths`` | int32 | ``[R]`` |
| ``causal_values`` | int32 | ``[R]`` |
| ``query_offsets``, ``prefix_offsets`` | int32 | ``[R + 1]`` |
| ``write_indices`` | int64 | ``[Tb, T]`` |

with the request slots' installed tables (``slot.unit_tables``,
``slot.start_pages``, ``slot.table_shapes``). Per call the worker packs the
rows' slots, lengths and selected page counts into ``row_columns``;
:meth:`AttentionColumns.gather` derives the device metadata from the slot
tables in one launch, and :meth:`AttentionColumns.batch` views the columns
as the ``AttentionBatch`` the model's attention layers read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

__all__ = ["AttentionColumns"]


class AttentionColumns:
    """Views of one binding's attention columns.

    Args:
        columns: The bound tensors by column name (see the module table).
        layout: ``{"page_tokens": [...], "windowed": [...]}`` per numerical
            block table.
    """

    def __init__(
        self, columns: Mapping[str, torch.Tensor], layout: Mapping
    ) -> None:
        self.columns = columns
        self.page_tokens = tuple(int(value) for value in layout["page_tokens"])
        self.windowed = tuple(bool(value) for value in layout["windowed"])

    def gather(self, rows: int, tokens: int, widths: Sequence[int]) -> None:
        """Gather ``rows`` rows of ``tokens`` tokens from the slot tables.

        Fills the block tables, start pages, lengths, offsets and write
        indices on the current stream from the packed ``row_columns``.
        """
        b = self.columns
        gather_request_rows(
            columns=b["row_columns"],
            rows=int(rows),
            tokens=int(tokens),
            width=int(max(widths)),
            request_unit_tables=b["slot.unit_tables"],
            request_start_pages=b["slot.start_pages"],
            table_shapes=b["slot.table_shapes"],
            block_tables=b["block_tables"],
            start_pages=b["start_pages"],
            cache_lengths=b["cache_lengths"],
            query_lengths=b["query_lengths"],
            query_offsets=b["query_offsets"],
            prefix_offsets=b["prefix_offsets"],
            write_indices=b["write_indices"],
        )

    def batch(
        self,
        rows: int,
        tokens: int,
        queries: Sequence[int],
        prefixes: Sequence[int],
        starts: Sequence[int],
        causal: Sequence[int],
        writes: bool,
        widths: Sequence[int],
    ) -> AttentionBatch:
        """The attention batch of ``rows`` rows over the gathered columns.

        ``queries`` and ``prefixes`` are the rows' host lengths, ``starts``
        each windowed table's first pages (``rows`` per table), ``causal``
        each row's causality (staged in ``causal_values`` when the rows
        differ) and ``widths`` each table's page width. Rows that write cache
        read paged inputs with write indices; read-only rows read their
        segments without writing.
        """
        b = self.columns
        rows, tokens = int(rows), int(tokens)
        query = SequenceLengths(
            values=b["query_lengths"][:rows],
            offsets=b["query_offsets"][: rows + 1],
            host=tuple(int(value) for value in queries),
        )
        prefix = SequenceLengths(
            values=b["cache_lengths"][:rows],
            offsets=b["prefix_offsets"][: rows + 1],
            host=tuple(int(value) for value in prefixes),
        )
        mode = tuple(bool(value) for value in causal)
        flags = b["causal_values"][:rows] if len(set(mode)) > 1 else None
        entries: dict[int, AttentionInput] = {}
        for table, width in enumerate(widths):
            first = None
            host_first = None
            if self.windowed[table]:
                first = b["start_pages"][table, :rows]
                host_first = tuple(
                    int(value)
                    for value in starts[table * rows : (table + 1) * rows]
                )
            blocks = BlockTable(
                b["block_tables"][table, :rows, : int(width)],
                self.page_tokens[table],
                first,
                host_first,
            )
            if writes:
                entries[table] = PagedInput(
                    query,
                    prefix,
                    blocks,
                    b["write_indices"][table, :tokens],
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
