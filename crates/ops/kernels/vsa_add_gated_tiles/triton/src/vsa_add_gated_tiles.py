"""VSA gated tiles: the Triton source of the ``vsa_add_gated_tiles`` operator.

Programs add one tile's gated compression to its rows, ``out + compressed[tile] *
gate``, with the product and the sum each rounded to the storage dtype as two
eager element-wise operations round them: the package compiles with
``enable_fp_fusion = false``, so the compiler does not contract the two into one
fused multiply-add.

The kernel body imports nothing but Triton: the launch-time wrapper, its operand
checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "output_stride_row",
        "output_stride_head",
        "gate_stride_row",
        "gate_stride_head",
        "compressed_stride_head",
        "compressed_stride_tile",
    ]
)
def _add_gated_tiles_kernel(
    output,
    compressed,
    gate,
    output_stride_row,
    output_stride_head,
    gate_stride_row,
    gate_stride_head,
    compressed_stride_head,
    compressed_stride_tile,
    tile_rows: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
):
    """Add one row block's gated tile compression for one head."""
    block = tl.program_id(0)
    head = tl.program_id(1)
    rows = (block * block_rows + tl.arange(0, block_rows)).to(tl.int64)
    columns = tl.arange(0, width)
    # A row block lies within one tile.
    tile = (block * block_rows) // tile_rows
    tile_values = tl.load(
        compressed
        + head * compressed_stride_head
        + tile * compressed_stride_tile
        + columns
    ).to(tl.float32)
    gate_values = tl.load(
        gate
        + rows[:, None] * gate_stride_row
        + head * gate_stride_head
        + columns[None, :]
    ).to(tl.float32)
    destination = (
        output
        + rows[:, None] * output_stride_row
        + head * output_stride_head
        + columns[None, :]
    )
    # The product rounds to the storage dtype before the sum, as the two eager
    # operations store it in between.
    product = tile_values[None, :] * gate_values
    product = product.to(
        output.dtype.element_ty, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    total = tl.load(destination).to(tl.float32) + product
    tl.store(
        destination,
        total.to(output.dtype.element_ty, fp_downcast_rounding="rtne"),
    )
