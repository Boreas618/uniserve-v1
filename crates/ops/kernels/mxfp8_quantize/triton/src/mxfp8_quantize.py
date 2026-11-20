"""MXFP8 activation encoding: the ``mxfp8_quantize`` operator's Triton source.

A row is encoded in blocks of 32 elements, each block into E4M3 codes under one
UE8M0 scale, a power of two stored as its biased exponent byte (``ops::mx``).
The arithmetic is the reference encoder's (DeepSeek's ``act_quant`` with
``scale_fmt = "ue8m0"``), step for step:

* ``amax = max(max |x|, 1e-4)`` over the block, in FP32;
* the scale is the least power of two at or above ``amax * fl(1/448)``: the
  product's exponent field, plus one unless its mantissa is zero, read from its
  bits rather than through a logarithm;
* ``code = e4m3(clamp(x / scale, ±448))``, round to nearest even. A division
  by a power of two is exact, so the one rounding is the cast.

The scales are written as the row's byte table, ``[rows, COLUMNS / 32]``
(``PACKED`` false), or as DeepGEMM's MN-major packed words (``PACKED`` true):
the four scales of a 128-element word of a row in one ``u32``, block ``4g + i``
in byte ``i``, word ``(g, row)`` at ``g * padded_rows + row`` with the rows
padded to four. The padding words are never read for a row that exists, and
the kernel leaves them as they are.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch

#: Elements one scale covers.
BLOCK_ELEMENTS = tl.constexpr(32)
#: Scales one packed word holds, and the elements it covers.
WORD_SCALES = tl.constexpr(4)
WORD_ELEMENTS = tl.constexpr(128)
#: The rows a packed scale column is padded to.
WORD_ROWS = tl.constexpr(4)
#: The largest finite E4M3 magnitude.
E4M3_MAX = tl.constexpr(448.0)
#: The smallest block magnitude the encoding takes.
AMAX_FLOOR = tl.constexpr(1e-4)


@triton.jit
def _mxfp8_quantize_kernel(
    input_ptr,
    codes_ptr,
    scales_ptr,
    rows,
    input_row_stride,
    COLUMNS: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    PACKED: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Encode ``BLOCK`` elements of one row: their codes and their scales.

    ``input`` is ``[rows, COLUMNS]`` with rows ``input_row_stride`` elements
    apart, ``codes`` ``[rows, COLUMNS]`` E4M3, contiguous, and ``scales`` the
    layout ``PACKED`` names. Program ``(row, chunk)`` encodes elements
    ``chunk * BLOCK`` onwards of ``row``.
    """
    dependent_launch.prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    columns = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + row * input_row_stride + columns).to(
        tl.float32
    )

    # One scale per 32 elements: the block's magnitude under the floor, and the
    # least power of two its codes fit under, from the bits of amax * fl(1/448).
    blocks = tl.reshape(values, [BLOCK // BLOCK_ELEMENTS, BLOCK_ELEMENTS])
    amax = tl.maximum(tl.max(tl.abs(blocks), axis=1), AMAX_FLOOR)
    ratio = (amax * (1.0 / E4M3_MAX)).to(tl.uint32, bitcast=True)
    exponent = ((ratio >> 23) & 0xFF).to(tl.int32) - 127
    exponent += ((ratio & 0x7FFFFF) != 0).to(tl.int32)
    biased = exponent + 127
    scale = (biased << 23).to(tl.float32, bitcast=True)

    scaled = tl.clamp(blocks / scale[:, None], -E4M3_MAX, E4M3_MAX)
    codes = tl.reshape(scaled, [BLOCK]).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    tl.store(codes_ptr + row * COLUMNS + columns, codes)

    if PACKED:
        # Four consecutive scales of the row, lowest block in the lowest byte.
        bytes_ = tl.reshape(
            biased.to(tl.uint32), [BLOCK // WORD_ELEMENTS, WORD_SCALES]
        )
        shifts = tl.arange(0, WORD_SCALES).to(tl.uint32) * 8
        words = tl.sum(bytes_ << shifts[None, :], axis=1)
        padded_rows = tl.cdiv(rows, WORD_ROWS) * WORD_ROWS
        groups = chunk * (BLOCK // WORD_ELEMENTS) + tl.arange(
            0, BLOCK // WORD_ELEMENTS
        )
        tl.store(scales_ptr + groups * padded_rows + row, words)
    else:
        blocks_per_row = COLUMNS // BLOCK_ELEMENTS
        scale_columns = chunk * (BLOCK // BLOCK_ELEMENTS) + tl.arange(
            0, BLOCK // BLOCK_ELEMENTS
        )
        tl.store(
            scales_ptr + row * blocks_per_row + scale_columns,
            biased.to(tl.uint8),
        )
