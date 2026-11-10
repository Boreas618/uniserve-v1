"""Per-tensor FP8 product: the Triton source of the ``fp8_gemm`` operator.

One program computes a ``BLOCK_M x BLOCK_N`` tile of ``out = (e4m3(x / s_x) @ w^T) * alpha``. The
16-bit activation tile is loaded, scaled by the reciprocal of its static scale in FP32, clamped to
the E4M3 range and rounded to nearest even into E4M3 as the reference's
``static_scaled_fp8_quant`` does; the E4M3 product accumulates in FP32 and the accumulator is
scaled once by ``alpha = s_x * s_w`` before the one rounding into the output type.

The kernel body imports nothing but Triton: the launch-time wrapper, its operand checks and its
torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl

#: The largest finite E4M3 magnitude.
E4M3_MAX = tl.constexpr(448.0)


@triton.jit(do_not_specialize=["m", "n", "k", "x_row_stride"])
def _fp8_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    m,
    n,
    k,
    x_row_stride,
    inverse_scale,
    alpha,
    BLOCK_M: tl.constexpr,  # noqa: N803
    BLOCK_N: tl.constexpr,  # noqa: N803
    BLOCK_K: tl.constexpr,  # noqa: N803
    BIAS: tl.constexpr,  # noqa: N803
):
    """One output tile; grid ``(cdiv(m, BLOCK_M), cdiv(n, BLOCK_N))``.

    ``x`` is ``[m, k]`` with rows ``x_row_stride`` apart, ``w`` is ``[n, k]`` E4M3 contiguous,
    ``out`` is ``[m, n]`` contiguous and ``bias`` is ``[n]`` when ``BIAS``.
    """
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    row_mask = rows < m
    column_mask = columns < n

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, k, BLOCK_K):
        reach = start + depth
        depth_mask = reach < k
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + reach[None, :],
            mask=row_mask[:, None] & depth_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        encoded = tl.clamp(x * inverse_scale, -E4M3_MAX, E4M3_MAX).to(
            tl.float8e4nv, fp_downcast_rounding="rtne"
        )
        w = tl.load(
            w_ptr + columns[None, :] * k + reach[:, None],
            mask=column_mask[None, :] & depth_mask[:, None],
            other=0.0,
        )
        accumulator = tl.dot(encoded, w, accumulator, out_dtype=tl.float32)

    result = accumulator * alpha
    if BIAS:
        bias = tl.load(bias_ptr + columns, mask=column_mask, other=0.0).to(tl.float32)
        result = result + bias[None, :]
    tl.store(
        out_ptr + rows[:, None] * n + columns[None, :],
        result,
        mask=row_mask[:, None] & column_mask[None, :],
    )
