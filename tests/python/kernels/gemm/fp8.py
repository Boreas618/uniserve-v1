"""The per-tensor FP8 product's two launches, as the GEMM checks run them.

`fp8_quantize` encodes an activation to E4M3 by its scale (`x · encode`,
clamped to ±448, rounded to nearest even) and `fp8_gemm` multiplies the codes
by an E4M3 weight with an FP32 accumulator and dequantizes by `alpha`; both
read their factor from an FP32 device word. The measured factors' check
(`test_dynamic_scale.py`) and the product's own (`test_fp8_gemm.py`) launch
them alike, so each launch is written here once.
"""

import triton

#: The elements of a row one encode program covers.
ENCODE_BLOCK = 1024

#: The largest output tile the product runs on four warps; a larger one takes
#: eight.
FOUR_WARP_TILE = 64 * 64

#: The software-pipelining stages of the product.
STAGES = 3


def quantize(kernels, x, codes, encode):
    """Encode the `[m, k]` rows of `x` into the E4M3 `codes`, one row a grid
    row and `ENCODE_BLOCK` elements a program across.
    """  # noqa: D205
    m, k = codes.shape
    kernels._fp8_quantize_kernel[(m, triton.cdiv(k, ENCODE_BLOCK))](
        x, codes, k, x.stride(0), encode, BLOCK=ENCODE_BLOCK, num_warps=4
    )


def multiply(kernels, codes, weight, bias, out, alpha, blocks):
    """`out = alpha · codes · weightᵀ (+ bias)` over `blocks`, the output tile
    `(BLOCK_M, BLOCK_N)` and the contraction step `BLOCK_K`.

    `codes` is `[m, k]` E4M3, `weight` `[n, k]` E4M3 and `out` `[m, n]`; a
    `None` bias launches the program without one.
    """  # noqa: D205
    m, k = codes.shape
    n = weight.shape[0]
    block_m, block_n, block_k = blocks
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    kernels._fp8_gemm_kernel[grid](
        codes,
        weight,
        out if bias is None else bias,
        out,
        m,
        n,
        k,
        codes.stride(0),
        alpha,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BIAS=bias is not None,
        num_warps=4 if block_m * block_n <= FOUR_WARP_TILE else 8,
        num_stages=STAGES,
    )
