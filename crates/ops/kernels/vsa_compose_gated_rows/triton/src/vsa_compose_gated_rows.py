"""VSA gated row composition: the Triton source of the ``vsa_compose_gated_rows`` operator.

One program writes one row block of one member's interval for one head: row ``i``
of member ``m``'s interval is row ``m · owner_rows + start + i`` of ``fine`` and
``gate``, and its gated sum lands in row ``i`` of the member's slot of the
contiguous ``[members, count, heads, width]`` payload, with the arithmetic of
``vsa_add_gated_tiles``.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/attention/vsa_segments.py``), unchanged
except that it imports nothing but Triton: the launch-time wrapper, its operand
checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _add_rn(a, b):
    # Round-to-nearest FP32 sum the compiler cannot contract with a neighbouring
    # multiply.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(
    do_not_specialize=[
        "fine_stride_row",
        "fine_stride_head",
        "gate_stride_row",
        "gate_stride_head",
        "compressed_stride_head",
        "compressed_stride_tile",
        "owner_rows",
        "start",
        "count",
    ]
)
def _compose_gated_rows_kernel(
    output,
    fine,
    compressed,
    gate,
    fine_stride_row,
    fine_stride_head,
    gate_stride_row,
    gate_stride_head,
    compressed_stride_head,
    compressed_stride_tile,
    owner_rows,
    start,
    count,
    heads: tl.constexpr,
    tile_rows: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
    slots: tl.constexpr,
):
    """Compose one row block of one member's interval for one head."""
    block = tl.program_id(0)
    head = tl.program_id(1)
    member = tl.program_id(2).to(tl.int64)
    local = (block * block_rows + tl.arange(0, block_rows)).to(tl.int64)
    first = member * owner_rows + start + block * block_rows
    rows = first + tl.arange(0, block_rows)
    columns = tl.arange(0, width)
    # Owner rows, interval starts and counts are whole row blocks, so a row
    # block lies within one tile.
    tile = first // tile_rows
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
    fine_values = tl.load(
        fine
        + rows[:, None] * fine_stride_row
        + head * fine_stride_head
        + columns[None, :]
    ).to(tl.float32)
    product = _mul_rn(tile_values[None, :], gate_values)
    product = product.to(
        output.dtype.element_ty, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    total = _add_rn(fine_values, product)

    slot = member
    for index in tl.static_range(len(slots)):
        slot = tl.where(member == index, slots[index], slot)
    destination = (
        output
        + ((slot * count + local)[:, None] * heads + head) * width
        + columns[None, :]
    )
    tl.store(
        destination,
        total.to(output.dtype.element_ty, fp_downcast_rounding="rtne"),
    )
