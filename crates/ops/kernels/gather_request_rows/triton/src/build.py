"""The gather_request_rows package's compile script: one descriptor -> one module.

A descriptor is the number of numerical tables, the width of a slot's resident
table row (pages per slot), the slot capacity the row strides come from, the
call's row capacity (the host column array's section length is that plus one),
the program's cell block and the call's token capacity. The row block is the
next power of two at least the row capacity, so it is part of the descriptor
too: a captured graph of another row bucket asks for its own module.
"""

import _compile
import gather_request_rows as kernels

#: Sections of the int64 host column array before the per-table page counts.
ROW_SECTIONS = 6


def spec(fields, parameters, target):
    """The compilation of one gather_request_rows descriptor."""
    request_width = int(fields["request_width"])
    slots = int(fields["slots"])
    tables = int(fields["tables"])
    rows = int(fields["rows"]) + 1
    return {
        "kernel": kernels._gather_request_rows_kernel,
        "signature": {
            "columns_ptr": "*i64",
            "request_unit_tables": "*i32",
            "request_start_pages": "*i32",
            "table_shapes": "*i64",
            "block_tables": "*i32",
            "start_pages": "*i32",
            "cache_lengths": "*i32",
            "query_lengths": "*i32",
            "query_offsets": "*i32",
            "prefix_offsets": "*i32",
            "slot_mapping": "*i64",
            "rows": "i32",
            "tokens": "i32",
            "cell_programs": "i32",
            "row_stride": "constexpr",
            "token_section": "constexpr",
            "width": "i32",
            "unit_table_stride": "constexpr",
            "unit_row_stride": "constexpr",
            "request_width": "constexpr",
            "start_group_stride": "constexpr",
            "block_table_stride": "constexpr",
            "block_table_row_stride": "constexpr",
            "start_page_stride": "constexpr",
            "write_stride": "constexpr",
            "row_block": "constexpr",
            "block": "constexpr",
        },
        "constexprs": {
            # One column-array section holds every row and its end offset.
            "row_stride": rows,
            # The token rows follow the sections and the per-table page counts.
            "token_section": (ROW_SECTIONS + tables) * rows,
            "unit_table_stride": slots * request_width,
            "unit_row_stride": request_width,
            "request_width": request_width,
            "start_group_stride": slots,
            "block_table_stride": rows * request_width,
            "block_table_row_stride": request_width,
            "start_page_stride": rows,
            "write_stride": int(parameters["tokens"]),
            # `tl.arange` extents are powers of two, so the row block is the
            # row capacity rounded up.
            "row_block": 1 << max(0, (rows - 1).bit_length()),
            "block": int(parameters["block"]),
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
