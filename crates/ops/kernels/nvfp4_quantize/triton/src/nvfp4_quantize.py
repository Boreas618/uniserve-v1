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

A block whose E4M3 scale rounds to zero encodes to zeros: the reciprocal of a
zero scale is not a number, and the vendor encoder multiplies by zero there.
"""

import triton
import triton.language as tl


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
def _nvfp4_quantize_kernel(
    input_ptr,
    values_ptr,
    scales_ptr,
    rows,
    global_scale,
    COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE_COLUMNS: tl.constexpr,
):
    """Encode one program's whole K16 blocks of one row.

    ``input`` is ``[rows, COLUMNS]`` with contiguous rows, ``values``
    ``[rows, COLUMNS / 2]`` bytes and ``scales`` ``[rows, SCALE_COLUMNS]``
    bytes in the linear layout — ``SCALE_COLUMNS`` is ``COLUMNS / 16``.
    ``global_scale`` is the reciprocal of the matrix's FP32 tensor scale.
    """
    # Each program owns one row's run of BLOCK elements, a whole number of K16
    # blocks, so a block never crosses a row.
    chunks = COLUMNS // BLOCK
    program = tl.program_id(0)
    row = (program // chunks).to(tl.int64)
    chunk = program % chunks

    columns = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + row * COLUMNS + columns).to(tl.float32)

    blocks = tl.reshape(values, [BLOCK // 16, 16])
    amax = tl.max(tl.abs(blocks), axis=1)

    # The block scale: amax / 6, in the global scale's domain, rounded to E4M3.
    sixth = _rcp_approx(tl.full([BLOCK // 16], 6.0, tl.float32))
    scale_bytes = _e4m3_byte(global_scale * (amax * sixth))
    decoded = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    inverse_global = _rcp_approx(tl.zeros_like(amax) + global_scale)
    output_scale = tl.where(
        decoded != 0.0, _rcp_approx(decoded * inverse_global), 0.0
    )

    scaled = blocks * output_scale[:, None]
    even, odd = tl.split(tl.reshape(scaled, [BLOCK // 2, 2]))
    packed = _e2m1_pair(even, odd)

    value_columns = chunk * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(values_ptr + row * (COLUMNS // 2) + value_columns, packed)
    scale_columns = chunk * (BLOCK // 16) + tl.arange(0, BLOCK // 16)
    tl.store(scales_ptr + row * SCALE_COLUMNS + scale_columns, scale_bytes)
