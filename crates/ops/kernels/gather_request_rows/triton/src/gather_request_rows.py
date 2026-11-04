"""Attention metadata for rows of any length: the Triton source of the
``gather_request_rows`` operator.

One launch fills a call's fixed-capacity attention columns from the request
slots' installed block tables: each table's selected start pages, the block
tables themselves, and every token's cache slot, plus the shared lengths and
prefix/query offsets. The host supplies only per-row scalars, each table's
selected page count per row and each token's row, packed in one int64 column
array.

The kernel body is the one UniServe gathered with
(``uniserve/nn/attention/gather.py``), unchanged except that it imports
nothing but Triton: the column layout, the capacity checks and the
tensor-operation path on devices without Triton stay with the operator's Rust
side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "tokens", "cell_programs", "width"])
def _gather_request_rows_kernel(
    columns_ptr,
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
    row_stride: tl.constexpr,
    token_section: tl.constexpr,
    width,
    unit_table_stride: tl.constexpr,
    unit_row_stride: tl.constexpr,
    request_width: tl.constexpr,
    start_group_stride: tl.constexpr,
    block_table_stride: tl.constexpr,
    block_table_row_stride: tl.constexpr,
    start_page_stride: tl.constexpr,
    write_stride: tl.constexpr,
    row_block: tl.constexpr,
    block: tl.constexpr,
):
    """Gather one table's rows, cells or tokens, by program.

    Axis 1 selects the numerical table. Program 0 writes the table's
    first selected pages and, on table 0, the shared lengths and offsets;
    programs ``1..cell_programs`` copy ``block`` table cells each; the
    rest write ``block`` tokens' cache addresses each. Every program
    reads only the host columns and the resident request tables, and
    their outputs are disjoint.
    """
    table = tl.program_id(1)
    program = tl.program_id(0)

    # [group, block size, window or -1] of this table.
    group = tl.load(table_shapes + table * 3)
    block_size = tl.load(table_shapes + table * 3 + 1)
    window = tl.load(table_shapes + table * 3 + 2)
    units = request_unit_tables + table * unit_table_stride
    slots_ptr = columns_ptr
    prefixes_ptr = columns_ptr + row_stride
    queries_ptr = columns_ptr + 2 * row_stride
    writes_ptr = columns_ptr + 3 * row_stride
    query_offsets_ptr = columns_ptr + 4 * row_stride
    prefix_offsets_ptr = columns_ptr + 5 * row_stride
    pages_ptr = columns_ptr + (6 + table) * row_stride

    if program == 0:
        offsets = tl.arange(0, row_block)
        row_live = offsets < rows
        row_prefix = tl.load(prefixes_ptr + offsets, mask=row_live, other=0)
        # A windowed table selects from the first page the row's first
        # query reaches back to; a full table from page zero.
        row_first = tl.where(
            window >= 0,
            tl.maximum(row_prefix - window, 0) // block_size,
            0,
        )
        tl.store(
            start_pages + table * start_page_stride + offsets,
            row_first,
            mask=row_live,
        )
        if table == 0:
            row_query = tl.load(
                queries_ptr + offsets, mask=row_live, other=0
            )
            tl.store(cache_lengths + offsets, row_prefix, mask=row_live)
            tl.store(query_lengths + offsets, row_query, mask=row_live)
            bounds = offsets < rows + 1
            tl.store(
                query_offsets + offsets,
                tl.load(query_offsets_ptr + offsets, mask=bounds, other=0),
                mask=bounds,
            )
            tl.store(
                prefix_offsets + offsets,
                tl.load(prefix_offsets_ptr + offsets, mask=bounds, other=0),
                mask=bounds,
            )
    elif program <= cell_programs:
        cells = (program - 1) * block + tl.arange(0, block)
        cell_row = cells // width
        cell = cells - cell_row * width
        live = cell_row < rows
        slot = tl.load(slots_ptr + cell_row, mask=live, other=0)
        prefix = tl.load(prefixes_ptr + cell_row, mask=live, other=0)
        pages = tl.load(pages_ptr + cell_row, mask=live, other=0)
        installed = tl.load(
            request_start_pages + group * start_group_stride + slot,
            mask=live,
            other=0,
        )
        first = tl.where(
            window >= 0,
            tl.maximum(prefix - window, 0) // block_size,
            0,
        )
        # Row ``slot`` of the resident table holds the units of pages
        # ``installed..``; cells past the row's selected pages are zero.
        source = first - installed + cell
        selected = live & (cell < pages)
        values = tl.load(
            units + slot * unit_row_stride + source,
            mask=selected & (source >= 0) & (source < request_width),
            other=0,
        )
        tl.store(
            block_tables
            + table * block_table_stride
            + cell_row * block_table_row_stride
            + cell,
            values,
            mask=live,
        )
    else:
        token = (program - 1 - cell_programs) * block + tl.arange(0, block)
        live = token < tokens
        token_row = tl.load(
            columns_ptr + token_section + token, mask=live, other=0
        )
        slot = tl.load(slots_ptr + token_row, mask=live, other=0)
        prefix = tl.load(prefixes_ptr + token_row, mask=live, other=0)
        start = tl.load(query_offsets_ptr + token_row, mask=live, other=0)
        writes = tl.load(writes_ptr + token_row, mask=live, other=0)
        installed = tl.load(
            request_start_pages + group * start_group_stride + slot,
            mask=live,
            other=0,
        )
        # Token ``token`` of row ``token_row`` sits at position prefix +
        # token - start and appends to that page's unit; a read-only row's
        # tokens address no cache slot (-1).
        position = prefix + token - start
        column = position // block_size - installed
        unit = tl.load(
            units + slot * unit_row_stride + column,
            mask=live
            & (writes != 0)
            & (column >= 0)
            & (column < request_width),
            other=-1,
        )
        address = tl.where(
            (writes != 0) & (unit >= 0),
            unit.to(tl.int64) * block_size + position % block_size,
            -1,
        )
        tl.store(
            slot_mapping + table * write_stride + token,
            address,
            mask=live,
        )
