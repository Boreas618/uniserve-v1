"""The attention_gather_rows package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor dimensions give, and the compilation options.

Every stride in the key is a constexpr here, because each one decides the index
arithmetic the program performs; the counts that change per call — the rows, the
tokens, the cell programs and the selected width — stay runtime values, so a new
batch size loads no other module. The table count is in the key because the store
keeps one module per call shape, and it is the grid's second axis, not a constant
the program reads.
"""

import _compile
import attention_gather_rows as kernels

#: Triton's element type for each element type the operator accepts, as the
#: signature spells a pointer to it.
POINTERS = {"i32": "*i32", "i64": "*i64"}


def warps_of(block):
    """The warp count one program of `block` cells or tokens needs."""
    return max(1, min(8, block // 128))


def spec(fields, parameters, target):
    """The compilation of one attention_gather_rows descriptor."""
    block = int(fields["block"])
    return {
        "kernel": kernels._attention_gather_rows_kernel,
        "signature": {
            "columns": POINTERS[fields["column_dtype"]],
            "request_unit_tables": POINTERS[fields["index"]],
            "request_start_pages": POINTERS[fields["index"]],
            "table_shapes": POINTERS[fields["index"]],
            "block_tables": POINTERS[fields["index"]],
            "start_pages": POINTERS[fields["index"]],
            "cache_lengths": POINTERS[fields["index"]],
            "query_lengths": POINTERS[fields["index"]],
            "query_offsets": POINTERS[fields["index"]],
            "prefix_offsets": POINTERS[fields["index"]],
            "slot_mapping": POINTERS[fields["index"]],
            "rows": "i64",
            "tokens": "i64",
            "cell_programs": "i64",
            "width": "i64",
            "ROW_STRIDE": "constexpr",
            "TOKEN_SECTION": "constexpr",
            "UNIT_TABLE_STRIDE": "constexpr",
            "UNIT_ROW_STRIDE": "constexpr",
            "REQUEST_WIDTH": "constexpr",
            "START_GROUP_STRIDE": "constexpr",
            "BLOCK_TABLE_STRIDE": "constexpr",
            "BLOCK_TABLE_ROW_STRIDE": "constexpr",
            "START_PAGE_STRIDE": "constexpr",
            "WRITE_STRIDE": "constexpr",
            "ROW_BLOCK": "constexpr",
            "BLOCK": "constexpr",
        },
        "constexprs": {
            "ROW_STRIDE": int(fields["row_stride"]),
            "TOKEN_SECTION": int(fields["token_section"]),
            "UNIT_TABLE_STRIDE": int(fields["unit_table_stride"]),
            "UNIT_ROW_STRIDE": int(fields["unit_row_stride"]),
            "REQUEST_WIDTH": int(fields["request_width"]),
            "START_GROUP_STRIDE": int(fields["start_group_stride"]),
            "BLOCK_TABLE_STRIDE": int(fields["block_table_stride"]),
            "BLOCK_TABLE_ROW_STRIDE": int(fields["block_table_row_stride"]),
            "START_PAGE_STRIDE": int(fields["start_page_stride"]),
            "WRITE_STRIDE": int(fields["write_stride"]),
            "ROW_BLOCK": int(fields["row_block"]),
            "BLOCK": block,
        },
        "options": {"num_warps": warps_of(block)},
    }


if __name__ == "__main__":
    _compile.main(spec)
