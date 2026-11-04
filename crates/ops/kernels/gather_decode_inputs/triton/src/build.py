"""The gather_decode_inputs package's compile script: one descriptor -> one module.

A descriptor is the number of numerical tables, the width of a slot's resident
table row (pages per slot), the slot capacity the row stride comes from, the
captured row bucket, the program's cell block and the number of token-axis rows
a decode position buffer has. All of them are build-config values of one epoch,
so one epoch asks for one module per bucket.
"""

import _compile
import gather_decode_inputs as kernels


def spec(fields, parameters, target):
    """The compilation of one gather_decode_inputs descriptor."""
    request_width = int(fields["request_width"])
    slots = int(fields["slots"])
    return {
        "kernel": kernels._gather_request_decode_inputs_kernel,
        "signature": {
            # The request slot state: the row's slot, the installed page
            # tables, the installed first pages, the table shapes and the
            # resident per-slot scalars.
            "request_pool_indices": "*i32",
            "request_unit_tables": "*i32",
            "request_start_pages": "*i32",
            "table_shapes": "*i64",
            "request_cache_lengths": "*i32",
            "request_tokens": "*i64",
            "request_positions": "*i64",
            # The call's metadata outputs.
            "input_ids": "*i64",
            "positions": "*i64",
            "block_tables": "*i32",
            "start_pages": "*i32",
            "cache_lengths": "*i32",
            "query_lengths": "*i32",
            "query_offsets": "*i32",
            "prefix_offsets": "*i32",
            "slot_mapping": "*i64",
            "unit_table_stride": "constexpr",
            "unit_row_stride": "constexpr",
            "request_width": "constexpr",
            "start_group_stride": "constexpr",
            "request_token_stride": "constexpr",
            "request_position_stride": "constexpr",
            "position_axis_stride": "constexpr",
            "position_axes": "constexpr",
            "block_table_stride": "constexpr",
            "block_table_row_stride": "constexpr",
            "start_page_stride": "constexpr",
            "write_stride": "constexpr",
            "max_rows": "i32",
            "row_block": "constexpr",
            "columns": "i32",
            "block": "constexpr",
        },
        "constexprs": {
            # The resident tables are [tables, slots, request_width] int32 rows
            # and the start pages [groups, slots] int32; one group in this
            # epoch's layout.
            "unit_table_stride": slots * request_width,
            "unit_row_stride": request_width,
            "request_width": request_width,
            "start_group_stride": slots,
            "request_token_stride": int(parameters["request_fields"]),
            "request_position_stride": int(parameters["request_fields"]),
            "position_axis_stride": int(fields["rows"]),
            "position_axes": 1,
            # The call's metadata: [tables, rows, width] and [tables, rows] in
            # int32, [tables, tokens] int64.
            "block_table_stride": int(fields["rows"]) * request_width,
            "block_table_row_stride": request_width,
            "start_page_stride": int(fields["rows"]),
            "write_stride": int(parameters["tokens"]),
            "row_block": int(fields["rows"]),
            "block": int(parameters["block"]),
        },
        "options": {"num_warps": 4},
    }


if __name__ == "__main__":
    _compile.main(spec)
