"""The request rows' attention metadata gathered on the device: the Triton program.

One launch writes every table's columns of a call's rows, from the host column array
and the request slots' resident tables. Program 0 of each table writes the table's
first selected page per row and, for table 0, the lengths and offsets every table
shares; programs `1..cell_programs` copy `block` block-table cells each; the rest
write `block` tokens' cache addresses. Each program writes a disjoint part of the
outputs, so no program reads another's result and no barrier is needed.

The host column array is `ROW_SECTIONS` sections of `row_stride` elements — the slot,
the prefix length, the query length, whether the row writes the cache, then the
exclusive query and prefix offsets, each one element per row plus the array's end —
followed by one section per table of its selected page count per row, and then each
of the `tokens` tokens' row. That layout is the operator: a program that assumed a
different order would read a plausible number from the wrong section, which is why
the sections are named here once and used by name below.

The arithmetic is the model's (`uniserve/nn/attention/gather.py`), which is what the
check scripts compare against: a token of a row sits at position `prefix + token -
start`, appends to the unit that position falls in, and addresses that unit's page
times the block size plus its offset inside the page. A row that does not write the
cache addresses no slot (`-1`), and so does a unit the slot has not installed.
"""

import triton
import triton.language as tl

#: The sections of the host column array every table shares: the slot, the prefix and
#: query lengths, the write flag, and the exclusive query and prefix offsets. A
#: `tl.constexpr` because a kernel may only read globals declared as one.
ROW_SECTIONS = tl.constexpr(6)


@triton.jit
def _attention_gather_rows_kernel(
    columns,
    request_unit_tables,
    request_start_pages,
    table_shapes,
    block_tables,
    start_pages,
    cache_lengths,
    query_lengths,
    query_offsets,
    prefix_offsets,
    slot_mapping,
    rows,
    tokens,
    cell_programs,
    width,
    ROW_STRIDE: tl.constexpr,  # noqa: N803
    TOKEN_SECTION: tl.constexpr,  # noqa: N803
    UNIT_TABLE_STRIDE: tl.constexpr,  # noqa: N803
    UNIT_ROW_STRIDE: tl.constexpr,  # noqa: N803
    REQUEST_WIDTH: tl.constexpr,  # noqa: N803
    START_GROUP_STRIDE: tl.constexpr,  # noqa: N803
    BLOCK_TABLE_STRIDE: tl.constexpr,  # noqa: N803
    BLOCK_TABLE_ROW_STRIDE: tl.constexpr,  # noqa: N803
    START_PAGE_STRIDE: tl.constexpr,  # noqa: N803
    WRITE_STRIDE: tl.constexpr,  # noqa: N803
    ROW_BLOCK: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Gather one table's rows, cells or tokens, by program.

    Axis 1 selects the numerical table.
    """
    table = tl.program_id(1)
    program = tl.program_id(0)

    # `[group, block size, window or -1]` of this table, and the slot's units.
    shape = table_shapes + table * 3
    group = tl.load(shape)
    block_size = tl.load(shape + 1)
    window = tl.load(shape + 2)
    units = request_unit_tables + table * UNIT_TABLE_STRIDE

    # The shared sections, then this table's selected-page counts.
    slots = columns
    prefixes = columns + ROW_STRIDE
    queries = columns + 2 * ROW_STRIDE
    writes = columns + 3 * ROW_STRIDE
    query_starts = columns + 4 * ROW_STRIDE
    prefix_starts = columns + 5 * ROW_STRIDE
    selected = columns + (ROW_SECTIONS + table) * ROW_STRIDE

    if program == 0:
        # The row variables carry the `row_` prefix because Triton unifies a name
        # assigned in two branches of one `if`, and these lanes are `ROW_BLOCK` wide
        # while the other two branches' are `BLOCK` wide.
        row_offsets = tl.arange(0, ROW_BLOCK)
        row_live = row_offsets < rows
        row_prefix = tl.load(prefixes + row_offsets, mask=row_live, other=0)
        # A windowed table keeps only the pages its query can reach back to; a full
        # table starts at page zero.
        row_first = tl.where(
            window >= 0,
            tl.maximum(row_prefix - window, 0) // block_size,
            0,
        )
        tl.store(
            start_pages + table * START_PAGE_STRIDE + row_offsets,
            row_first,
            mask=row_live,
        )
        if table == 0:
            # The lengths and offsets are one row's answer for every table.
            row_query = tl.load(queries + row_offsets, mask=row_live, other=0)
            tl.store(cache_lengths + row_offsets, row_prefix, mask=row_live)
            tl.store(query_lengths + row_offsets, row_query, mask=row_live)
            row_bounds = row_offsets < rows + 1
            tl.store(
                query_offsets + row_offsets,
                tl.load(query_starts + row_offsets, mask=row_bounds, other=0),
                mask=row_bounds,
            )
            tl.store(
                prefix_offsets + row_offsets,
                tl.load(prefix_starts + row_offsets, mask=row_bounds, other=0),
                mask=row_bounds,
            )
    elif program <= cell_programs:
        cells = (program - 1) * BLOCK + tl.arange(0, BLOCK)
        row = cells // width
        cell = cells - row * width
        live = row < rows
        slot = tl.load(slots + row, mask=live, other=0).to(tl.int64)
        prefix = tl.load(prefixes + row, mask=live, other=0).to(tl.int64)
        pages = tl.load(selected + row, mask=live, other=0).to(tl.int64)
        installed = tl.load(
            request_start_pages
            + group.to(tl.int64) * START_GROUP_STRIDE
            + slot,
            mask=live,
            other=0,
        ).to(tl.int64)
        first = tl.where(
            window >= 0,
            tl.maximum(prefix - window.to(tl.int64), 0)
            // block_size.to(tl.int64),
            0,
        )
        # The slot's row of the resident table holds the units of pages
        # `installed..`; cells past the row's selected pages are zero.
        source = first - installed + cell.to(tl.int64)
        inside = live & (cell < pages)
        value = tl.load(
            units + slot * UNIT_ROW_STRIDE + source,
            mask=inside & (source >= 0) & (source < REQUEST_WIDTH),
            other=0,
        )
        tl.store(
            block_tables
            + table.to(tl.int64) * BLOCK_TABLE_STRIDE
            + row * BLOCK_TABLE_ROW_STRIDE
            + cell.to(tl.int64),
            value,
            mask=live,
        )
    else:
        token = (program - 1 - cell_programs) * BLOCK + tl.arange(0, BLOCK)
        live = token < tokens
        row = tl.load(columns + TOKEN_SECTION + token, mask=live, other=0).to(
            tl.int64
        )
        slot = tl.load(slots + row, mask=live, other=0).to(tl.int64)
        prefix = tl.load(prefixes + row, mask=live, other=0).to(tl.int64)
        start = tl.load(query_starts + row, mask=live, other=0).to(tl.int64)
        writable = tl.load(writes + row, mask=live, other=0)
        installed = tl.load(
            request_start_pages
            + group.to(tl.int64) * START_GROUP_STRIDE
            + slot,
            mask=live,
            other=0,
        ).to(tl.int64)
        # The token's position in its row, the unit that position falls in, and the
        # address inside that unit. A read-only row's tokens address nothing.
        position = prefix + token.to(tl.int64) - start
        column = position // block_size.to(tl.int64) - installed
        unit = tl.load(
            units + slot * UNIT_ROW_STRIDE + column,
            mask=live
            & (writable != 0)
            & (column >= 0)
            & (column < REQUEST_WIDTH),
            other=-1,
        )
        address = tl.where(
            (writable != 0) & (unit >= 0),
            unit.to(tl.int64) * block_size.to(tl.int64)
            + position % block_size.to(tl.int64),
            -1,
        )
        tl.store(
            slot_mapping + table.to(tl.int64) * WRITE_STRIDE + token,
            address.to(slot_mapping.dtype.element_ty),
            mask=live,
        )
