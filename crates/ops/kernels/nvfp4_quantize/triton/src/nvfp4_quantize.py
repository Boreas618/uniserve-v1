"""NVFP4 activation encoding: the Triton source of the ``nvfp4_quantize`` operator.

A calibrated NVFP4 checkpoint stores a matrix as packed E2M1 values, two to a
byte, with one E4M3 block scale per sixteen elements and one FP32 tensor scale
for the matrix. This kernel encodes a floating matrix into that form, block by
block, with the activation's tensor scale given: the bytes it writes are the
bytes FlashInfer's ``fp4_quantize`` (TensorRT-LLM ``cvt_warp_fp16_to_fp4``)
writes for the same input.

Three pieces are the hardware's own rounding, so they are inline assembly rather
than composed arithmetic:

* ``rcp.approx.ftz.f32`` is the reciprocal the vendor encoder uses, and the
  quotient of a block's amax by six must be that approximation, not a division;
* ``cvt.rn.satfinite.e4m3x2.f32`` rounds to the E4M3 block scale, saturating;
* ``cvt.rn.satfinite.e2m1x2.f32`` rounds a pair of values to E2M1 codes packed
  into one byte, lower column in the lower nibble.

Encoding is two-level micro-scaling. A row is first brought into the block format's range by a
factor the row itself determines -- ``token_scale = amax(row) * global_scale`` -- and each
sixteen-element block then carries an E4M3 scale relative to that factor. The product that reads
the encoding multiplies both, so this kernel writes the factor beside the values and the scales.
That is why one program owns a whole row: the factor is a reduction over the row, and every block
of the row is scaled against it, so a program owning a fragment of a row could not state it. A
program walks its row twice -- once for the row's amax, once to encode -- for the same reason.

A block whose E4M3 scale rounds to zero encodes to zeros: the reciprocal of a
zero scale is not a number, and the vendor encoder multiplies by zero there.

The block scales are written in one of the two layouts a product reads: the
linear matrix, whose rows are the matrix's own, or the 128x4 swizzle, which pads
the rows to a multiple of 128 and the scale columns to a multiple of 4 and
interleaves them. The padded entries are part of the tensor the operator hands
over and the reference encoder writes them as zeros, so this kernel writes them
too — inside the same launch, since the operator's grid is the matrix's elements.
"""

import triton
import triton.language as tl

#: The row block the swizzled layout interleaves: 128 rows, in four groups of 32.
SWIZZLE_ROWS = tl.constexpr(128)
SWIZZLE_GROUPS = tl.constexpr(4)
SWIZZLE_ROWS_PER_GROUP = tl.constexpr(32)
#: The scale columns the swizzled layout groups.
SWIZZLE_COLUMNS = tl.constexpr(4)


@triton.jit
def _rcp_approx(x):
    """``rcp.approx.ftz.f32``: the reciprocal the NVFP4 encoder uses."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e4m3_byte(x):
    """The saturating round-to-nearest E4M3 byte of FP32 ``x``.

    ``cvt.rn.satfinite.e4m3x2.f32`` stores its first operand in the upper byte;
    the lower byte encodes ``x``.
    """
    pair = tl.inline_asm_elementwise(
        "{ .reg .b16 pair; "
        "cvt.rn.satfinite.e4m3x2.f32 pair, $2, $1; "
        "cvt.u32.u16 $0, pair; }",
        "=r,f,f",
        [x, tl.zeros_like(x)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return (pair & 0xFF).to(tl.uint8)


@triton.jit
def _e2m1_pair(even, odd):
    """One byte of two saturating round-to-nearest E2M1 codes.

    The lower nibble holds ``even`` (the lower column), as the packed NVFP4
    values store them.
    """
    pair = tl.inline_asm_elementwise(
        "{ .reg .b8 pair; .reg .b16 wide; "
        "cvt.rn.satfinite.e2m1x2.f32 pair, $2, $1; "
        "mov.b16 wide, {pair, 0}; cvt.u32.u16 $0, wide; }",
        "=r,f,f",
        [even, odd],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return pair.to(tl.uint8)


@triton.jit
def _swizzled_offset(rows, columns, PADDED_COLUMNS: tl.constexpr):
    """Flat offsets of the 128x4 swizzle for rows and scale columns ``rows``/``columns``.

    The layout groups the scale matrix into blocks of 128 rows and 4 scale
    columns; within a block the 128 rows split into four groups of 32 and the
    four columns are interleaved fastest.
    """
    block_row = rows // SWIZZLE_ROWS
    within = rows % SWIZZLE_ROWS
    group = within // SWIZZLE_ROWS_PER_GROUP
    lane = within % SWIZZLE_ROWS_PER_GROUP
    return (
        block_row * (PADDED_COLUMNS * SWIZZLE_ROWS)
        + (columns // SWIZZLE_COLUMNS) * (SWIZZLE_ROWS * SWIZZLE_COLUMNS)
        + lane * (SWIZZLE_GROUPS * SWIZZLE_COLUMNS)
        + group * SWIZZLE_COLUMNS
        + (columns % SWIZZLE_COLUMNS)
    )


@triton.jit
def _nvfp4_quantize_kernel(
    input_ptr,
    values_ptr,
    scales_ptr,
    rows,
    global_scale,
    token_scale_ptr,
    COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE_COLUMNS: tl.constexpr,
    SWIZZLED: tl.constexpr,
    PADDED_COLUMNS: tl.constexpr,
    PADDED_TILE: tl.constexpr,
):
    """Encode one row: its per-token factor, its packed values and its E4M3 block scales.

    ``input`` is ``[rows, COLUMNS]`` with contiguous rows, ``values``
    ``[rows, COLUMNS / 2]`` bytes, ``token_scale`` ``[rows]`` f32 and ``scales`` bytes in the
    descriptor's layout -- ``[rows, SCALE_COLUMNS]`` linear, or ``[padded_rows, PADDED_COLUMNS]``
    swizzled. ``global_scale`` is the address of a one-element FP32 tensor holding the reciprocal
    of the matrix's FP32 tensor scale.
    """
    # The tensor scale is a device operand: the loader publishes a calibrated
    # checkpoint's value as one FP32 word, and every program reads it from there,
    # so the number the kernel scales by is the checkpoint's and not a host copy.
    global_scale = tl.load(global_scale).to(tl.float32)

    chunks = COLUMNS // BLOCK
    row = tl.program_id(0)

    # Pass one: the row's amax. The per-token factor is the row's own dequantization
    # factor, `amax * global_scale`, which is what the reference encoder publishes;
    # the encode multiplier is its approximate reciprocal, as the reference's
    # `reciprocal_approximate_ftz` is, so the bytes agree with it.
    # One-element vectors rather than scalars: the value is loop-carried, and a Triton loop's
    # carried variable needs a shape it can unify across iterations.
    row_amax = tl.zeros([1], tl.float32)
    for chunk in range(chunks):
        columns = chunk * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(input_ptr + row * COLUMNS + columns).to(tl.float32)
        row_amax = tl.maximum(row_amax, tl.max(tl.abs(values).reshape([BLOCK])))

    token_scale = row_amax * global_scale
    tl.store(token_scale_ptr + row + tl.arange(0, 1), token_scale)
    ## The encode multiplier is the approximate reciprocal of the row's factor, as the reference
    ## encoder's `reciprocal_approximate_ftz` is: the bytes agree with it, not merely the values.
    encode = _rcp_approx(token_scale)
    inverse_encode = _rcp_approx(encode)

    # Pass two: each block's E4M3 scale relative to the row's factor, and its codes.
    sixth = _rcp_approx(tl.full([1], 6.0, tl.float32))
    for chunk in range(chunks):
        columns = chunk * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(input_ptr + row * COLUMNS + columns).to(tl.float32)

        blocks = tl.reshape(values, [BLOCK // 16, 16])
        amax = tl.max(tl.abs(blocks), axis=1)

        scale_bytes = _e4m3_byte(encode * (amax * sixth))
        decoded = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        output_scale = tl.where(
            decoded != 0.0, _rcp_approx(decoded * inverse_encode), 0.0
        )

        scaled = blocks * output_scale[:, None]
        even, odd = tl.split(tl.reshape(scaled, [BLOCK // 2, 2]))
        packed = _e2m1_pair(even, odd)

        value_columns = chunk * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
        tl.store(values_ptr + row * (COLUMNS // 2) + value_columns, packed)

        scale_columns = chunk * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
        destination = row * SCALE_COLUMNS + scale_columns
        if SWIZZLED:
            destination = _swizzled_offset(row, scale_columns, PADDED_COLUMNS)
        tl.store(scales_ptr + destination, scale_bytes)

    if SWIZZLED:
        # The tensor's padding is zeros, as the reference encoder writes it. Each
        # program owns one row block of it: the last chunk of a row writes that
        # row's padding columns, and the programs write the padding rows, a whole
        # padded row each.
        padded_rows = tl.cdiv(rows, SWIZZLE_ROWS) * SWIZZLE_ROWS
        padding_columns = tl.arange(0, SWIZZLE_COLUMNS)
        extra = SCALE_COLUMNS + padding_columns
        keep = extra < PADDED_COLUMNS
        tl.store(
            scales_ptr + _swizzled_offset(row, extra, PADDED_COLUMNS),
            tl.zeros([SWIZZLE_COLUMNS], tl.uint8),
            mask=keep,
        )
        padded = tl.arange(0, PADDED_TILE)
        keep_column = padded < PADDED_COLUMNS
        padding_row = rows + row
        programs = tl.num_programs(0)
        while padding_row < padded_rows:
            tl.store(
                scales_ptr + _swizzled_offset(padding_row, padded, PADDED_COLUMNS),
                tl.zeros([PADDED_TILE], tl.uint8),
                mask=keep_column,
            )
            padding_row += programs
