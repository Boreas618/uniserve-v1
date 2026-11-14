"""Per-tensor E4M3 encoding: the Triton source of the ``fp8_quantize`` operator.

``q[m, k] = e4m3(clamp(x[m, k] * encode, -448, 448))``, the product formed in
FP32 and rounded to nearest even into E4M3, as the reference's
``scaled_fp8_quant`` encodes. ``encode`` is one FP32 word in device memory, the
reciprocal of the activation's scale: a calibrated export's constant, or the
factor ``dynamic_scale`` measured.

The kernel body imports nothing but Triton.
"""

import triton
import triton.language as tl

#: The largest finite E4M3 magnitude.
E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    k,
    x_row_stride,
    encode_ptr,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Encode ``BLOCK`` elements of one row; grid ``(m, cdiv(k, BLOCK))``.

    ``x`` is ``[m, k]`` with rows ``x_row_stride`` apart and ``out`` is
    ``[m, k]`` E4M3 contiguous.
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = columns < k

    encode = tl.load(encode_ptr)
    x = tl.load(x_ptr + row * x_row_stride + columns, mask=live, other=0.0)
    scaled = tl.clamp(x.to(tl.float32) * encode, -E4M3_MAX, E4M3_MAX)
    tl.store(
        out_ptr + row * k + columns,
        scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
        mask=live,
    )
