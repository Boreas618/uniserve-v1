"""Per-tensor FP8 product: the Triton source of the ``fp8_gemm`` operator.

One program computes a ``BLOCK_M x BLOCK_N`` tile of ``out = (q @ w^T) * alpha``: ``q`` is the
activation ``fp8_quantize`` encoded (``e4m3(x / s_x)``, the reference's ``scaled_fp8_quant``) and
``w`` the E4M3 weight, both loaded as stored codes, which the tensor-core product reads; the
product accumulates in FP32 and the accumulator is scaled once by ``alpha = s_x * s_w`` before the
one rounding into the output type.

``alpha`` is one FP32 word in device memory: a calibrated export's constant, or the factor a
launch before measured.

The kernel body imports nothing but Triton: the launch-time wrapper, its operand checks and its
torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["m", "n"])
def _fp8_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    m,
    n,
    k,
    x_row_stride,
    alpha_ptr,
    BLOCK_M: tl.constexpr,  # noqa: N803
    BLOCK_N: tl.constexpr,  # noqa: N803
    BLOCK_K: tl.constexpr,  # noqa: N803
    BIAS: tl.constexpr,  # noqa: N803
):
    """One output tile; grid ``(cdiv(m, BLOCK_M), cdiv(n, BLOCK_N))``.

    ``x`` is ``[m, k]`` E4M3 with rows ``x_row_stride`` apart, ``w`` is ``[n, k]`` E4M3
    contiguous, ``out`` is ``[m, n]`` contiguous and ``bias`` is ``[n]`` when ``BIAS``.
    ``alpha_ptr`` holds the product's factor.
    """
    alpha = tl.load(alpha_ptr)

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
        )
        w = tl.load(
            w_ptr + columns[None, :] * k + reach[:, None],
            mask=column_mask[None, :] & depth_mask[:, None],
            other=0.0,
        )
        accumulator = tl.dot(x, w, accumulator, out_dtype=tl.float32)

    result = accumulator * alpha
    if BIAS:
        bias = tl.load(bias_ptr + columns, mask=column_mask, other=0.0).to(tl.float32)
        result = result + bias[None, :]
    tl.store(
        out_ptr + rows[:, None] * n + columns[None, :],
        result,
        mask=row_mask[:, None] & column_mask[None, :],
    )


def derive(fields, parameters, target):
    """The elements of one output tile, ``tile``, whose size sets the warp count."""
    return {"tile": fields["block_m"] * fields["block_n"]}
