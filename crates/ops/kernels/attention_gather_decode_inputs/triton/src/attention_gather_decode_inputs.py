"""The decode step's attention inputs gathered on the device: the Triton program.

One launch writes a decode batch's per-row inputs from the request pool: each table's
block table and first page and append address, and, once per call, the token ids, the
positions, the cache and query lengths and the two exclusive offsets. Program 0 of each
table writes that table's scalars over a power-of-two row block; the programs after it
copy `block` block-table cells each, over rows × columns of the caller's table. Every
program writes a disjoint part of the outputs and nothing stores to an input, so no
program reads another's result and no barrier is needed.

A row is live when its request slot is not 0: slot 0 is the caller's padding slot, and
a padded row contributes a token id of 1, a position of 0 and no cached tokens, while
its query length is still 1 — every row of a decode batch is one query token.

The requests are addressed through the pool rather than copied first, which is what
this operator is for: the two gather strides into the request rows
(`request_token_stride`, `request_position_stride`) and the positions' axis stride are
words of the launch, because they are the caller's layout, not this program's
arithmetic.

The arithmetic is the model's (`uniserve/nn/attention/gather.py`): a row's append page
is its cache length divided by the table's block size, offset by the slot's installed
start page, and its append address is the unit's page times the block size plus its
offset inside the page; a table's block table starts at the first page the window
reaches, `max(0, cache - window) // block_size`, and continues unit by unit; a page the
slot never installed reads as 0, and a row that cannot append addresses no slot (`-1`).
"""

import triton
import triton.language as tl


@triton.jit
def _attention_gather_decode_inputs_kernel(
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
    max_rows,
    columns,
    unit_table_stride,
    request_token_stride,
    request_position_stride,
    position_axis_stride,
    block_table_stride,
    start_page_stride,
    write_stride,
    UNIT_ROW_STRIDE: tl.constexpr,  # noqa: N803
    REQUEST_WIDTH: tl.constexpr,  # noqa: N803
    START_GROUP_STRIDE: tl.constexpr,  # noqa: N803
    POSITION_AXES: tl.constexpr,  # noqa: N803
    BLOCK_TABLE_ROW_STRIDE: tl.constexpr,  # noqa: N803
    ROW_BLOCK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Gather one table's scalars, or its cells, by program."""
    table = tl.program_id(1)
    program = tl.program_id(0)

    # `[group, block size, window or -1]` of this table, and the slot's units.
    shape = table_shapes + table * 3
    group = tl.load(shape)
    block_size = tl.load(shape + 1)
    window = tl.load(shape + 2)
    units = request_unit_tables + table * unit_table_stride

    if program == 0:
        row = tl.arange(0, ROW_BLOCK)
        row_mask = row < max_rows
        slots = tl.load(request_pool_indices + row, mask=row_mask, other=0)
        live = row_mask & (slots != 0)
        # A padded row reads slot 0's state and then discards it, so the reads stay
        # uniform over the row block and only the select differs per lane.
        cache = tl.where(
            live,
            tl.load(request_cache_lengths + slots, mask=row_mask, other=0),
            0,
        )
        installed = tl.where(
            live,
            tl.load(
                request_start_pages + group * START_GROUP_STRIDE + slots,
                mask=row_mask,
                other=0,
            ),
            0,
        )

        # A windowed table selects pages from the first one the decode query's
        # window reaches; a full table from page zero.
        first = tl.where(
            window >= 0, tl.maximum(cache - window, 0) // block_size, 0
        )
        tl.store(
            start_pages + table * start_page_stride + row,
            tl.where(live, first, 0),
            mask=row_mask,
        )

        # The cache length identifies the page an append lands in and the offset
        # inside it. A row that cannot append, or a unit the slot never installed,
        # keeps the public -1.
        column = cache // block_size - installed
        appendable = live & (column >= 0) & (column < REQUEST_WIDTH)
        unit = tl.load(
            units + slots * UNIT_ROW_STRIDE + column,
            mask=appendable,
            other=-1,
        )
        addresses = tl.where(
            unit >= 0,
            unit.to(tl.int64) * block_size + cache % block_size,
            -1,
        )
        tl.store(
            slot_mapping + table * write_stride + row,
            addresses,
            mask=row_mask,
        )

        if table == 0:
            tokens = tl.where(
                live,
                tl.load(
                    request_tokens + slots * request_token_stride,
                    mask=row_mask,
                    other=1,
                ),
                1,
            )
            token_positions = tl.where(
                live,
                tl.load(
                    request_positions + slots * request_position_stride,
                    mask=row_mask,
                    other=0,
                ),
                0,
            )
            tl.store(input_ids + row, tokens, mask=row_mask)

            # Only the first axis carries a text token's position; the spatial
            # axes of a three-axis multimodal position stay zero.
            for axis in tl.static_range(POSITION_AXES):
                tl.store(
                    positions + axis * position_axis_stride + row,
                    tl.where(axis == 0, token_positions, 0),
                    mask=row_mask,
                )
            tl.store(cache_lengths + row, cache, mask=row_mask)
            tl.store(query_lengths + row, 1, mask=row_mask)

            # Exclusive offsets over the full capacity of rows: every row has one
            # query token, and a padded row adds no cached tokens.
            tl.store(query_offsets, 0)
            tl.store(prefix_offsets, 0)
            tl.store(query_offsets + row + 1, row + 1, mask=row_mask)
            tl.store(prefix_offsets + row + 1, tl.cumsum(cache), mask=row_mask)
    else:
        # One program per `block` cells of rows × columns, from the pool row's
        # first selected page. Stores cover the whole row capacity, so a row the
        # call does not select reads as zero rather than as stale storage.
        cell_offsets = (program - 1) * BLOCK + tl.arange(0, BLOCK)
        table_elements = max_rows * columns
        store_mask = cell_offsets < table_elements
        cell_rows = cell_offsets // columns
        cell_columns = cell_offsets - cell_rows * columns
        cell_slots = tl.load(
            request_pool_indices + cell_rows, mask=store_mask, other=0
        )
        cell_live = store_mask & (cell_slots != 0)
        cell_cache = tl.load(
            request_cache_lengths + cell_slots, mask=cell_live, other=0
        )
        cell_installed = tl.load(
            request_start_pages + group * START_GROUP_STRIDE + cell_slots,
            mask=cell_live,
            other=0,
        )
        cell_first = tl.where(
            window >= 0, tl.maximum(cell_cache - window, 0) // block_size, 0
        )
        source = cell_first - cell_installed + cell_columns
        page = tl.load(
            units + cell_slots * UNIT_ROW_STRIDE + source,
            mask=cell_live & (source >= 0) & (source < REQUEST_WIDTH),
            other=0,
        )
        tl.store(
            block_tables
            + table * block_table_stride
            + cell_rows * BLOCK_TABLE_ROW_STRIDE
            + cell_columns,
            page,
            mask=store_mask,
        )
