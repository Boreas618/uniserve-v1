"""The attention_gather_decode_inputs package's compile script: one descriptor, one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor dimensions give, and the compilation options.

The values that place the caller's storage — the seven capacity and request strides —
and the counts that change per call (`max_rows`, `columns`) are runtime arguments, so
one module serves any storage and any batch capacity. A stride the program's own
arithmetic needs stays a constexpr, because the compiler folds it into the index
expressions; the row block is one too, because `tl.arange` needs a power-of-two extent
and the row capacity is not in the key.
"""

import _compile
import attention_gather_decode_inputs as kernels

#: Triton's element type for each element type the operator accepts, as the
#: signature spells a pointer to it.
POINTERS = {"i32": "*i32", "i64": "*i64"}


def warps_of(block):
    """The warp count one program of `block` cells needs."""
    return max(1, min(8, block // 128))


def spec(fields, parameters, target):
    """The compilation of one attention_gather_decode_inputs descriptor."""
    block = int(fields["block"])
    return {
        "kernel": kernels._attention_gather_decode_inputs_kernel,
        "signature": {
            "request_pool_indices": POINTERS[fields["index"]],
            "request_unit_tables": POINTERS[fields["index"]],
            "request_start_pages": POINTERS[fields["index"]],
            "table_shapes": POINTERS[fields["index"]],
            "request_cache_lengths": POINTERS[fields["index"]],
            "request_tokens": POINTERS[fields["index"]],
            "request_positions": POINTERS[fields["index"]],
            "input_ids": POINTERS[fields["index"]],
            "positions": POINTERS[fields["index"]],
            "block_tables": POINTERS[fields["index"]],
            "start_pages": POINTERS[fields["index"]],
            "cache_lengths": POINTERS[fields["index"]],
            "query_lengths": POINTERS[fields["index"]],
            "query_offsets": POINTERS[fields["index"]],
            "prefix_offsets": POINTERS[fields["index"]],
            "slot_mapping": POINTERS[fields["index"]],
            "max_rows": "i64",
            "columns": "i64",
            "unit_table_stride": "i64",
            "request_token_stride": "i64",
            "request_position_stride": "i64",
            "position_axis_stride": "i64",
            "block_table_stride": "i64",
            "start_page_stride": "i64",
            "write_stride": "i64",
            "UNIT_ROW_STRIDE": "constexpr",
            "REQUEST_WIDTH": "constexpr",
            "START_GROUP_STRIDE": "constexpr",
            "POSITION_AXES": "constexpr",
            "BLOCK_TABLE_ROW_STRIDE": "constexpr",
            "ROW_BLOCK": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "UNIT_ROW_STRIDE": int(fields["unit_row_stride"]),
            "REQUEST_WIDTH": int(fields["request_width"]),
            "START_GROUP_STRIDE": int(fields["start_group_stride"]),
            "POSITION_AXES": int(fields["position_axes"]),
            "BLOCK_TABLE_ROW_STRIDE": int(fields["block_table_row_stride"]),
            "ROW_BLOCK": int(fields["row_block"]),
            "BLOCK": block,
        },
        "options": {"num_warps": warps_of(block)},
    }


if __name__ == "__main__":
    _compile.main(spec)
