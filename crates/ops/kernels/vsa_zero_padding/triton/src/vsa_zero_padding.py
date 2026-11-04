"""VSA tile padding: the Triton source of the ``vsa_zero_padding`` operator.

Programs store zeros to one tile's row block where the tile's valid size ends, so
a plain per-tile sum of the rows equals the sum of the masked rows. Only padding
rows are stored, and the launch's shape depends on the tile count alone.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/attention/vsa_segments.py``), unchanged
except that it imports nothing but Triton: the launch-time wrapper, its operand
checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["stride_row", "stride_head"])
def _zero_tile_padding_kernel(
    values,
    variable_block_sizes,
    stride_row,
    stride_head,
    heads: tl.constexpr,
    tile_rows: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
):
    """Store zeros to one tile's row block past the tile's valid size."""
    tile = tl.program_id(0)
    block = tl.program_id(1)
    valid = tl.load(variable_block_sizes + tile)
    # A row block wholly inside the valid rows stores nothing.
    if (block + 1) * block_rows > valid:
        local = block * block_rows + tl.arange(0, block_rows)
        rows = (tile * tile_rows + local).to(tl.int64)
        columns = tl.arange(0, width)
        padding = (local >= valid)[:, None]
        zeros = tl.zeros((block_rows, width), dtype=values.dtype.element_ty)
        for head in tl.static_range(heads):
            offsets = (
                rows[:, None] * stride_row
                + head * stride_head
                + columns[None, :]
            )
            tl.store(values + offsets, zeros, mask=padding)
