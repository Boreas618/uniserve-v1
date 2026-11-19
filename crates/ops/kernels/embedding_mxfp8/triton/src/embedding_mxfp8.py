"""MXFP8 table rows: the Triton source of the ``embedding_mxfp8`` operator.

A row of the table is ``HIDDEN`` E4M3 values and one UE8M0 scale, an
exponent byte ``e`` meaning ``2^(e - 127)``, for each ``GROUP`` consecutive
values. One program dequantizes the rows of ``BLOCK`` ids::

    row       = id_i - first
    out[i, c] = round(e4m3(values[row, c]) * 2^(scales[row, c // GROUP] - 127))

the product in FP32, rounded once into the output type, or zeros for an id
outside the shard's rows ``[first, first + rows)``. The byte 255 is not a
number and 0 is ``2^-127``, as the format states them.

The kernel body imports nothing but Triton: the launch-time wrapper, its
operand checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count", "first", "rows"])
def _embedding_mxfp8_kernel(
    ids_ptr,
    values_ptr,
    scales_ptr,
    out_ptr,
    count,
    first,
    rows,
    HIDDEN: tl.constexpr,  # noqa: N803
    GROUP: tl.constexpr,  # noqa: N803
    TILE: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """``BLOCK`` ids; grid ``(cdiv(count, BLOCK),)``.

    ``ids`` is ``[count]``, ``values`` ``[rows, HIDDEN]`` E4M3, ``scales``
    ``[rows, HIDDEN / GROUP]`` UE8M0 bytes and ``out`` ``[count, HIDDEN]``.
    """
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = index < count
    local = tl.load(ids_ptr + index, mask=live, other=0).to(tl.int64) - first
    held = live & (local >= 0) & (local < rows)
    columns = tl.arange(0, TILE)
    inside = columns < HIDDEN
    mask = held[:, None] & inside[None, :]

    values = tl.load(
        values_ptr + local[:, None] * HIDDEN + columns[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    exponents = tl.load(
        scales_ptr
        + local[:, None] * (HIDDEN // GROUP)
        + (columns // GROUP)[None, :],
        mask=mask,
        other=127,
    ).to(tl.int32)
    # 2^(e - 127) as the FP32 of exponent field e; e = 0 is the subnormal 2^-127
    # and e = 255 not a number.
    scale = (exponents << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponents == 0, 5.877471754111438e-39, scale)
    scale = tl.where(exponents == 255, float("nan"), scale)

    tl.store(
        out_ptr + index[:, None] * HIDDEN + columns[None, :],
        tl.where(mask, values * scale, 0.0),
        mask=live[:, None] & inside[None, :],
    )


def derive(fields, parameters, target):
    """The lanes of a program's rows: the row width, rounded up to a power
    of two.
    """  # noqa: D205
    hidden = int(fields["hidden"])
    return {"tile": 1 << (hidden - 1).bit_length()}
