"""Attention metadata for decode rows: the Triton source of the
``gather_decode_inputs`` operator.

One launch fills a decode call's attention columns from resident request state:
each request's input token, position, cache length, query length and its block
table's entries, and every token's cache slot. Padding rows (slot 0) are
initialized too, so the launch can be captured in a decode graph of any row
bucket.

The kernel body is the one UniServe gathered with
(``uniserve/nn/attention/gather.py``), unchanged except that it imports
nothing but Triton: the capacity checks and the tensor-operation path on
devices without Triton stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["max_rows", "columns"])
def _gather_request_decode_inputs_kernel(
    request_pool_indices,
    request_unit_tables,
    request_start_pages,
    table_shapes,
    request_cache_lengths,
    request_tokens,
    request_positions,
    input_ids,
    positions,
    block_tables,
    start_pages,
    cache_lengths,
    query_lengths,
    query_offsets,
    prefix_offsets,
    slot_mapping,
    unit_table_stride: tl.constexpr,
    unit_row_stride: tl.constexpr,
    request_width: tl.constexpr,
    start_group_stride: tl.constexpr,
    request_token_stride: tl.constexpr,
    request_position_stride: tl.constexpr,
    position_axis_stride: tl.constexpr,
    position_axes: tl.constexpr,
    block_table_stride: tl.constexpr,
    block_table_row_stride: tl.constexpr,
    start_page_stride: tl.constexpr,
    write_stride: tl.constexpr,
    max_rows,
    row_block: tl.constexpr,
    columns,
    block: tl.constexpr,
):
    """Gather one table's cells and per-row decode scalars.

    Axis 1 of the grid selects the numerical table. On it, program 0
    writes the table's start pages and slot mappings, and on table 0
    also every shared per-row scalar column; the other programs copy the
    table's cells. Their outputs are disjoint and no program stores to
    an input, so no global barrier is needed. A row is live when its
    slot is not the padding slot 0.
    """
    table = tl.program_id(1)

    # [group, block size, window or -1] of this table.
    group = tl.load(table_shapes + table * 3)
    block_size = tl.load(table_shapes + table * 3 + 1)
    window = tl.load(table_shapes + table * 3 + 2)
    units = request_unit_tables + table * unit_table_stride

    if tl.program_id(0) == 0:
        offsets = tl.arange(0, row_block)
        scalar_mask = offsets < max_rows
        slots = tl.load(
            request_pool_indices + offsets, mask=scalar_mask, other=0
        )
        live_rows = scalar_mask & (slots != 0)
        cache = tl.load(
            request_cache_lengths + slots, mask=live_rows, other=0
        )
        installed = tl.load(
            request_start_pages + group * start_group_stride + slots,
            mask=live_rows,
            other=0,
        )

        # A windowed table selects pages from the first one the decode
        # query's window reaches; a full table from page zero.
        first = tl.where(
            window >= 0, tl.maximum(cache - window, 0) // block_size, 0
        )
        tl.store(
            start_pages + table * start_page_stride + offsets,
            tl.where(live_rows, first, 0),
            mask=scalar_mask,
        )

        # Cache length identifies the append page and its token offset.
        # Inactive rows and unavailable pages keep the public -1 sentinel.
        column = cache // block_size - installed
        write_units = tl.load(
            units + slots * unit_row_stride + column,
            mask=live_rows & (column >= 0) & (column < request_width),
            other=-1,
        )
        writes = tl.where(
            write_units >= 0,
            write_units.to(tl.int64) * block_size + cache % block_size,
            -1,
        )
        tl.store(
            slot_mapping + table * write_stride + offsets,
            writes,
            mask=scalar_mask,
        )

        if table == 0:
            tokens = tl.load(
                request_tokens + slots * request_token_stride,
                mask=live_rows,
                other=1,
            )
            token_positions = tl.load(
                request_positions + slots * request_position_stride,
                mask=live_rows,
                other=0,
            )
            tl.store(input_ids + offsets, tokens, mask=scalar_mask)

            # Only the first axis carries the token position; the
            # spatial axes of three-axis multimodal positions stay zero
            # for text tokens.
            for axis in tl.static_range(position_axes):
                tl.store(
                    positions + axis * position_axis_stride + offsets,
                    token_positions if axis == 0 else 0,
                    mask=scalar_mask,
                )
            tl.store(cache_lengths + offsets, cache, mask=scalar_mask)
            tl.store(query_lengths + offsets, 1, mask=scalar_mask)

            # Exclusive prefix sums over the full capacity: every row,
            # including padding, has one query token, and padded rows add
            # no cached tokens.
            tl.store(query_offsets, 0)
            tl.store(prefix_offsets, 0)
            tl.store(
                query_offsets + offsets + 1, offsets + 1, mask=scalar_mask
            )
            tl.store(
                prefix_offsets + offsets + 1,
                tl.cumsum(cache),
                mask=scalar_mask,
            )
    else:
        table_offsets = (tl.program_id(0) - 1) * block + tl.arange(0, block)

        # Map active output rows through the request pool into the
        # table's installed units, from the row's first selected page.
        # Stores span the full row capacity, zeroing inactive rows.
        table_elements = max_rows * columns
        table_mask = table_offsets < table_elements
        table_rows = table_offsets // columns
        cells = table_offsets - table_rows * columns
        # Names differ from the scalar branch's: Triton requires a name
        # bound in both branches of a runtime condition to keep one type.
        table_slots = tl.load(
            request_pool_indices + table_rows,
            mask=table_mask,
            other=0,
        )
        live_table = table_mask & (table_slots != 0)
        cell_cache = tl.load(
            request_cache_lengths + table_slots, mask=live_table, other=0
        )
        cell_installed = tl.load(
            request_start_pages + group * start_group_stride + table_slots,
            mask=live_table,
            other=0,
        )
        cell_first = tl.where(
            window >= 0,
            tl.maximum(cell_cache - window, 0) // block_size,
            0,
        )
        source = cell_first - cell_installed + cells
        table_values = tl.load(
            units + table_slots * unit_row_stride + source,
            mask=live_table & (source >= 0) & (source < request_width),
            other=0,
        )
        tl.store(
            block_tables
            + table * block_table_stride
            + table_rows * block_table_row_stride
            + cells,
            table_values,
            mask=table_mask,
        )
