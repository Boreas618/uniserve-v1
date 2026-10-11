"""Attention metadata gathered on the device from resident block tables.

Two kernels fill a call's fixed-capacity attention columns from the request
slots' installed block tables (``unit_tables``, ``start_pages``,
``table_shapes``), the way vLLM derives its slot mapping on the device from
its block table, positions and query starts
(``vllm/v1/worker/block_table.py``, ``compute_slot_mapping``):

- :func:`gather_request_rows` serves rows of any length (prefill chunks,
  verification, padded graph buckets). The host supplies only per-row
  scalars, each table's selected page count per row and each token's row,
  packed in one int64 column array (:func:`row_columns_size`).
- :func:`gather_request_decode_inputs` serves decode rows, whose token,
  position and cache length are resident request state: it reads them by
  request slot and initializes padding rows (slot 0), so it can be captured
  in a decode graph of any row bucket.

Both write disjoint outputs in one launch per call and have a
tensor-operation path for devices where Triton cannot launch.
"""

from __future__ import annotations

import torch
from uniserve_kernels.triton import launchable

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

# Sections of the int64 host column array, each ``row_columns_stride`` long:
# slot, prefix, query, writes, query offsets and prefix offsets (``rows + 1``
# entries each). One section of selected page counts per table follows, then
# each token's row.
ROW_SECTIONS = 6

if triton is not None:
    # Row, token, cell counts and the output width change every call;
    # keeping them as unspecialized runtime values means a new count loads
    # no other kernel variant.
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
        write_indices,
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

        # [group, page tokens, window or -1] of this table.
        group = tl.load(table_shapes + table * 3)
        page_tokens = tl.load(table_shapes + table * 3 + 1)
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
                tl.maximum(row_prefix - window, 0) // page_tokens,
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
                tl.maximum(prefix - window, 0) // page_tokens,
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
            column = position // page_tokens - installed
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
                unit.to(tl.int64) * page_tokens + position % page_tokens,
                -1,
            )
            tl.store(
                write_indices + table * write_stride + token,
                address,
                mask=live,
            )


def gather_request_rows(
    *,
    columns: torch.Tensor,
    rows: int,
    tokens: int,
    width: int,
    request_unit_tables: torch.Tensor,
    request_start_pages: torch.Tensor,
    table_shapes: torch.Tensor,
    block_tables: torch.Tensor,
    start_pages: torch.Tensor,
    cache_lengths: torch.Tensor,
    query_lengths: torch.Tensor,
    query_offsets: torch.Tensor,
    prefix_offsets: torch.Tensor,
    write_indices: torch.Tensor,
) -> None:
    """Gather every table's columns of ``rows`` request rows.

    ``columns`` is the device copy of the host column array (sections of
    ``row_columns_stride`` elements, see ``ROW_SECTIONS``): per row its
    request slot, prefix length,
    query length and whether it writes the cache; the exclusive query and
    prefix offsets (``rows + 1`` each); each table's selected page count per
    row; and each of the ``tokens`` tokens' row. For table ``t``, described
    by ``table_shapes[t] = (group, page_tokens, window or -1)``, the launch
    writes rows ``0..rows`` of ``block_tables[t]`` (``width`` cells, zero
    past a row's selected pages) from the first page each row selects (page
    zero, or for a windowed table ``max(0, prefix - window) //
    page_tokens``) into ``start_pages[t]``, and each token's append address
    into ``write_indices[t]`` (-1 for rows that do not write); and, shared by
    all tables, the lengths and offsets. ``request_unit_tables[t, slot]``
    holds the slot's units from its installed start page
    ``request_start_pages[group, slot]`` on.

    Every tensor must reside on one device with unit innermost stride; the
    caller validates the rows against the installed tables
    (``attention.prepare_attention``). On a device where Triton launches, one
    kernel gathers every table; elsewhere equivalent tensor operations do.

    Raises:
        ValueError: When the tensors are not on one device or the counts are
            outside the columns' capacity.
    """
    tensors = (
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
        write_indices,
    )
    device = columns.device
    if any(value.device != device for value in tensors):
        raise ValueError("request row preparation requires one device")

    count, token_count = int(rows), int(tokens)
    tables = int(request_unit_tables.shape[0])
    stride = row_columns_stride(block_tables.shape[1])
    if (
        count < 1
        or count > int(block_tables.shape[1])
        or token_count < 1
        or int(width) < 1
        or int(width) > int(block_tables.shape[2])
        or token_count > int(write_indices.shape[1])
        or int(columns.numel())
        < row_columns_size(
            int(block_tables.shape[1]), tables, int(write_indices.shape[1])
        )
    ):
        raise ValueError("request rows exceed their buffer capacity")

    if triton is None or device.type != "cuda" or not launchable(device):
        _gather_rows(
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
            write_indices,
            rows=count,
            tokens=token_count,
            width=int(width),
            stride=stride,
        )
        return

    block = 256
    cell_programs = triton.cdiv(count * int(width), block)
    grid = (1 + cell_programs + triton.cdiv(token_count, block), tables)
    _gather_request_rows_kernel[grid](
        *tensors,
        rows=count,
        tokens=token_count,
        cell_programs=cell_programs,
        row_stride=stride,
        token_section=(ROW_SECTIONS + tables) * stride,
        width=int(width),
        unit_table_stride=int(request_unit_tables.stride(0)),
        unit_row_stride=int(request_unit_tables.stride(1)),
        request_width=int(request_unit_tables.shape[2]),
        start_group_stride=int(request_start_pages.stride(0)),
        block_table_stride=int(block_tables.stride(0)),
        block_table_row_stride=int(block_tables.stride(1)),
        start_page_stride=int(start_pages.stride(0)),
        write_stride=int(write_indices.stride(0)),
        row_block=triton.next_power_of_2(stride),
        block=block,
        num_warps=4,
    )


def _gather_rows(
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
    write_indices,
    *,
    rows,
    tokens,
    width,
    stride,
):
    """Compute the kernel's outputs with tensor operations, table by table."""
    tables = int(request_unit_tables.shape[0])
    request_width = int(request_unit_tables.shape[2])
    slots = columns[:rows]
    prefix = columns[stride : stride + rows]
    query = columns[2 * stride : 2 * stride + rows]
    writes = columns[3 * stride : 3 * stride + rows] != 0
    token_section = (ROW_SECTIONS + tables) * stride
    owner = columns[token_section : token_section + tokens]
    starts = columns[4 * stride : 4 * stride + rows + 1]
    position = prefix[owner] + torch.arange(tokens, device=columns.device)
    position -= starts[owner]
    cells = torch.arange(width, device=columns.device)

    for table, (group, page_tokens, window) in enumerate(table_shapes.tolist()):
        first = (
            (prefix - window).clamp(min=0) // page_tokens
            if window >= 0
            else torch.zeros_like(prefix)
        )
        start_pages[table, :rows] = first
        installed = request_start_pages[group, slots].to(torch.int64)
        pages = columns[(ROW_SECTIONS + table) * stride :][:rows]

        source = (first - installed)[:, None] + cells[None]
        selected = (
            (cells[None] < pages[:, None])
            & (source >= 0)
            & (source < request_width)
        )
        units = request_unit_tables[
            table, slots[:, None], source.clamp(0, request_width - 1)
        ]
        block_tables[table, :rows, :width] = torch.where(
            selected, units, torch.zeros_like(units)
        )

        column = position // page_tokens - installed[owner]
        valid = writes[owner] & (column >= 0) & (column < request_width)
        unit = request_unit_tables[
            table, slots[owner], column.clamp(0, request_width - 1)
        ].to(torch.int64)
        write_indices[table, :tokens] = torch.where(
            valid & (unit >= 0),
            unit * page_tokens + position % page_tokens,
            torch.full_like(unit, -1),
        )

    cache_lengths[:rows] = prefix
    query_lengths[:rows] = query
    query_offsets[: rows + 1] = starts
    prefix_offsets[: rows + 1] = columns[5 * stride : 5 * stride + rows + 1]


def row_columns_stride(max_rows: int) -> int:
    """Return the length of one row section: every row and its end offset."""
    return int(max_rows) + 1


def row_columns_size(max_rows: int, tables: int, max_tokens: int) -> int:
    """Return the int64 elements the host column array of a call needs."""
    return (ROW_SECTIONS + int(tables)) * row_columns_stride(max_rows) + int(
        max_tokens
    )


if triton is not None:
    # Row counts and input widths change between eager forwards. Keeping
    # them unspecialized runtime values means another count does not load
    # another kernel variant; only the power-of-two row block does.
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
        write_indices,
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
        writes the table's start pages and write addresses, and on table 0
        also every shared per-row scalar column; the other programs copy the
        table's cells. Their outputs are disjoint and no program stores to
        an input, so no global barrier is needed. A row is live when its
        slot is not the padding slot 0.
        """
        table = tl.program_id(1)

        # [group, page tokens, window or -1] of this table.
        group = tl.load(table_shapes + table * 3)
        page_tokens = tl.load(table_shapes + table * 3 + 1)
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
                window >= 0, tl.maximum(cache - window, 0) // page_tokens, 0
            )
            tl.store(
                start_pages + table * start_page_stride + offsets,
                tl.where(live_rows, first, 0),
                mask=scalar_mask,
            )

            # Cache length identifies the append page and its token offset.
            # Inactive rows and unavailable pages keep the public -1 sentinel.
            column = cache // page_tokens - installed
            write_units = tl.load(
                units + slots * unit_row_stride + column,
                mask=live_rows & (column >= 0) & (column < request_width),
                other=-1,
            )
            writes = tl.where(
                write_units >= 0,
                write_units.to(tl.int64) * page_tokens + cache % page_tokens,
                -1,
            )
            tl.store(
                write_indices + table * write_stride + offsets,
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
                tl.maximum(cell_cache - window, 0) // page_tokens,
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


def _gather_decode_inputs(
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
    write_indices,
    *,
    columns,
):
    """Compute the decode kernel's outputs with tensor operations."""
    slots = request_pool_indices
    rows = int(slots.numel())
    live = slots != 0
    request_width = int(request_unit_tables.shape[2])
    cache = torch.where(live, request_cache_lengths[slots].to(torch.int64), 0)
    cells = torch.arange(columns, device=slots.device)

    for table, (group, page_tokens, window) in enumerate(table_shapes.tolist()):
        installed = torch.where(
            live, request_start_pages[group, slots].to(torch.int64), 0
        )
        first = (
            (cache - window).clamp(min=0) // page_tokens
            if window >= 0
            else torch.zeros_like(cache)
        )
        start_pages[table, :rows] = first

        # Cache length identifies the append page and its token offset.
        column = cache // page_tokens - installed
        appendable = live & (column >= 0) & (column < request_width)
        unit = request_unit_tables[
            table, slots, column.clamp(0, request_width - 1)
        ].to(torch.int64)
        write_indices[table, :rows] = torch.where(
            appendable & (unit >= 0),
            unit * page_tokens + cache % page_tokens,
            -1,
        )

        source = (first - installed)[:, None] + cells[None]
        selected = live[:, None] & (source >= 0) & (source < request_width)
        units = request_unit_tables[
            table, slots[:, None], source.clamp(0, request_width - 1)
        ]
        block_tables[table, :rows, :columns] = torch.where(selected, units, 0)

    input_ids[:rows] = torch.where(live, request_tokens[slots], 1)
    positions[:, :rows] = 0
    positions[0, :rows] = torch.where(live, request_positions[slots], 0)
    cache_lengths[:rows] = cache
    query_lengths[:rows] = 1
    query_offsets[: rows + 1] = torch.arange(rows + 1, device=slots.device)
    prefix_offsets[0] = 0
    prefix_offsets[1 : rows + 1] = torch.cumsum(cache, dim=0)


def gather_request_decode_inputs(
    *,
    request_pool_indices: torch.Tensor,
    request_unit_tables: torch.Tensor,
    request_start_pages: torch.Tensor,
    table_shapes: torch.Tensor,
    request_cache_lengths: torch.Tensor,
    request_tokens: torch.Tensor,
    request_positions: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    start_pages: torch.Tensor,
    cache_lengths: torch.Tensor,
    query_lengths: torch.Tensor,
    query_offsets: torch.Tensor,
    prefix_offsets: torch.Tensor,
    write_indices: torch.Tensor,
    columns: int,
) -> None:
    """Gather request state into fixed-capacity decode buffers in place.

    Each element of ``request_pool_indices`` is one decode row's
    scheduler-owned request slot; slot 0 is the padding slot, and a row
    naming it is padding. Every value the function reads is the contents of
    a tensor, so a captured graph of it serves any rows of its shape.

    The function materializes, for every numerical table ``t`` described by
    ``table_shapes[t] = (group, page_tokens, window or -1)``, the slots'
    units from their first selected page into ``block_tables[t]``, that page
    into ``start_pages[t]`` and the append address into
    ``write_indices[t]``; and, shared by all tables, next-token ids and
    position axes, cache/query lengths and offsets. A row's first selected
    page of a windowed table is the first page its window reaches,
    ``max(0, cache - window) // page_tokens``; of a full table, page zero.
    ``request_unit_tables[t, slot]`` holds the slot's units from its
    installed start page ``request_start_pages[group, slot]`` on. Each table
    writes ``columns`` cells per row. Padding rows are initialized for a
    fixed-shape forward: token id 1, zero positions, cache length and start
    page, one query token, a zeroed table row, and the write-index sentinel
    -1.

    Every tensor must reside on the same device. The kernel honors the
    table and slot strides of ``request_unit_tables``, the group stride of
    ``request_start_pages``, the first-axis strides of ``request_tokens`` and
    ``request_positions``, the table and row strides of ``block_tables``, the
    table strides of ``start_pages`` and ``write_indices``, and the axis
    stride of ``positions``; every other dimension is indexed with unit
    stride, which is checked for the columns of ``positions`` and the unit
    tables.

    Raises:
        ValueError: When the tensors are not on one device, there is no
            row, a table has the wrong rank, count or capacity, ``columns``
            is not positive, ``positions`` is not one or three unit-stride
            axes, or an output buffer is smaller than the rows require.
    """
    # The fused kernel dereferences every tensor directly, so all must share
    # one device. All validation precedes the launch, so a rejected call
    # writes nothing.
    tensors = (
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
        write_indices,
    )
    device = request_pool_indices.device
    if any(value.device != device for value in tensors):
        raise ValueError("request-indexed decode requires one device")

    # ``request_pool_indices`` defines the rows every output holds.
    max_rows = int(request_pool_indices.numel())
    if max_rows < 1:
        raise ValueError("request-indexed decode requires a row")
    if (
        request_unit_tables.ndim != 3
        or request_start_pages.ndim != 2
        or block_tables.ndim != 3
        or start_pages.ndim != 2
        or write_indices.ndim != 2
    ):
        raise ValueError("request-indexed decode tables have the wrong rank")

    # Every input table must hold every output row and ``columns`` cells.
    tables = int(request_unit_tables.shape[0])
    if (
        table_shapes.shape != (tables, 3)
        or table_shapes.dtype != torch.int32
        or not table_shapes.is_contiguous()
        or request_unit_tables.stride(2) != 1
        or tuple(block_tables.shape[:2]) != (tables, max_rows)
        or int(block_tables.shape[2]) < int(columns)
        or block_tables.stride(2) != 1
        or int(start_pages.shape[0]) != tables
        or int(start_pages.shape[1]) < max_rows
        or int(write_indices.shape[0]) != tables
        or int(write_indices.shape[1]) < max_rows
        or int(columns) < 1
    ):
        raise ValueError(
            "request-indexed decode table capacity or shape is invalid"
        )
    if (
        positions.ndim != 2
        or positions.shape[0] not in (1, 3)
        or positions.stride(1) != 1
    ):
        raise ValueError(
            "decode positions require one or three contiguous token axes"
        )
    scalar_outputs = (
        input_ids,
        positions[0],
        cache_lengths,
        query_lengths,
    )
    if any(int(value.numel()) < max_rows for value in scalar_outputs):
        raise ValueError("request-indexed decode scalar buffers are undersized")
    if min(query_offsets.numel(), prefix_offsets.numel()) < max_rows + 1:
        raise ValueError("request-indexed decode offset buffers are undersized")

    if triton is None or device.type != "cuda" or not launchable(device):
        _gather_decode_inputs(*tensors, columns=int(columns))
        return

    # Per table, one program produces the scalar columns over a power-of-two
    # row block (``tl.arange`` extents must be powers of two) masked to
    # ``max_rows``; the remaining programs copy ``block`` table cells each.
    block = 256
    grid = (1 + triton.cdiv(max_rows * int(columns), block), tables)
    _gather_request_decode_inputs_kernel[grid](
        *tensors,
        unit_table_stride=int(request_unit_tables.stride(0)),
        unit_row_stride=int(request_unit_tables.stride(1)),
        request_width=int(request_unit_tables.shape[2]),
        start_group_stride=int(request_start_pages.stride(0)),
        request_token_stride=int(request_tokens.stride(0)),
        request_position_stride=int(request_positions.stride(0)),
        position_axis_stride=int(positions.stride(0)),
        position_axes=int(positions.shape[0]),
        block_table_stride=int(block_tables.stride(0)),
        block_table_row_stride=int(block_tables.stride(1)),
        start_page_stride=int(start_pages.stride(0)),
        write_stride=int(write_indices.stride(0)),
        max_rows=max_rows,
        row_block=triton.next_power_of_2(max_rows),
        columns=int(columns),
        block=block,
        num_warps=4,
    )


__all__ = [
    "ROW_SECTIONS",
    "gather_request_decode_inputs",
    "gather_request_rows",
    "row_columns_size",
    "row_columns_stride",
]
