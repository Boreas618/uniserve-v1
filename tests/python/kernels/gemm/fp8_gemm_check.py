#!/usr/bin/env python3
"""Run the fp8_quantize and fp8_gemm programs on a GPU and check every output against torch and FP64.

    python fp8_gemm_check.py                 # bf16 activations, the model's projections

The operators are the per-tensor FP8 product: `fp8_quantize` encodes the activation to E4M3 by
its scale (`x / s_x`, clamped to ±448, rounded to nearest even), and `fp8_gemm` multiplies the
codes by the E4M3 weight with an FP32 accumulator and dequantizes by `alpha = s_x · s_w`. The
programs read `1 / s_x` and `alpha` from FP32 device words, as a calibrated export's constants or
a measured scale reach them. The codes are compared with torch's encoding of the same activation
byte for byte; the product with the FP64 sum of the codes' products.

The bound follows from that arithmetic: the codes are equal, so the two differ by the FP32
accumulation of `k` products of exact values — at most `γ_k = k·u_f32/(1 − k·u_f32)` of the sum of
the products' magnitudes, scaled by `alpha` — and by the one rounding of the output, `u_out` of
the reference value. The interpreter does not store E4M3 as the device does, so the check runs on
a device only.
"""

import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The packages this check covers.
sys.path.insert(0, str(KERNELS / "fp8_gemm" / "triton" / "src"))
sys.path.insert(0, str(KERNELS / "fp8_quantize" / "triton" / "src"))

import fp8_gemm as kernels  # noqa: E402
import fp8_quantize as encoder  # noqa: E402
import torch  # noqa: E402

UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
F32_ROUNDOFF = 2.0**-24


def check(
    label, m, n, k, blocks, bias, seed, dtype=torch.bfloat16, row_stride=None
):
    torch.manual_seed(seed)
    row_stride = row_stride or k
    storage = torch.randn(m, row_stride, dtype=torch.float32)
    # Activations of a few hundred in magnitude reach the E4M3 range: some saturate.
    storage[0, :4] = torch.tensor([3000.0, -3000.0, 1e-6, 0.0])
    x = storage.to(dtype)[:, :k].cuda()
    weight_float = torch.randn(n, k) * 0.05
    weight_scale = float(weight_float.abs().max() / 448.0)
    weight = (
        (weight_float / weight_scale)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .cuda()
    )
    input_scale = float(storage.abs().max() / 448.0 / 4.0)
    alpha = input_scale * weight_scale
    b = (torch.randn(n) * 0.1).to(dtype).cuda() if bias else None
    out = torch.empty(m, n, dtype=dtype, device="cuda")
    # The two factors are device words of FP32, as the operator passes them.
    encode = torch.tensor(
        [1.0 / input_scale], dtype=torch.float32, device="cuda"
    )
    factor = torch.tensor([alpha], dtype=torch.float32, device="cuda")
    # The encode: one row a grid row, blocks of 1024 elements across.
    codes = torch.empty(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    encoder._fp8_quantize_kernel[(m, (k + 1023) // 1024)](
        x, codes, k, x.stride(0), encode, BLOCK=1024, num_warps=4
    )
    block_m, block_n, block_k = blocks
    grid = ((m + block_m - 1) // block_m, (n + block_n - 1) // block_n)
    kernels._fp8_gemm_kernel[grid](
        codes,
        weight,
        b if bias else out,
        out,
        m,
        n,
        k,
        codes.stride(0),
        factor,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BIAS=bias,
        num_warps=4 if block_m * block_n <= 64 * 64 else 8,
        num_stages=3,
    )
    torch.cuda.synchronize()
    inverse = encode.cpu()[0]
    wanted = (
        (x.float().cpu() * inverse).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    differing = int(
        (codes.cpu().view(torch.uint8) != wanted.view(torch.uint8)).sum()
    )
    codes = wanted.double()
    products = codes @ weight.cpu().double().T
    alpha = float(factor.cpu()[0])
    expected = products * alpha
    magnitudes = (codes.abs() @ weight.cpu().double().abs().T) * alpha
    if bias:
        expected = expected + b.cpu().double()
        magnitudes = magnitudes + b.cpu().double().abs()
    gamma = k * F32_ROUNDOFF / (1 - k * F32_ROUNDOFF)
    bound = gamma * magnitudes + UNIT_ROUNDOFF[dtype] * expected.abs() + 1e-30
    error = (out.cpu().double() - expected).abs()
    outside = int((error > bound).sum()) + differing
    relative = float((error / (magnitudes + 1e-30)).max())
    print(
        f"{'ok' if outside == 0 else 'FAIL'}: {label}, {m} x {n} x {k} in {blocks}"
        f"{' with a bias' if bias else ''} — max abs {float(error.max()):.3e}, max error over "
        f"the products' magnitude {relative:.3e}, {differing} codes differ, {outside} outside"
    )
    return outside


def main():
    if not torch.cuda.is_available():
        print("fp8_gemm_check needs a CUDA device")
        return 1
    outside = 0
    # The model's FP8 projections (Qwen3.6-35B-A3B): in_proj_qkv, in_proj_z, out_proj, q, k/v, o.
    for label, m, n, k in [
        ("in_proj_qkv, one decode row", 1, 8192, 2048),
        ("in_proj_z, a prompt", 200, 4096, 2048),
        ("out_proj", 37, 2048, 4096),
        ("q_proj", 64, 8192, 2048),
        ("k_proj", 5, 512, 2048),
    ]:
        outside += check(label, m, n, k, (128, 128, 128), False, m + n)
    outside += check(
        "a masked tail and a bias", 19, 100, 96, (16, 32, 32), True, 7
    )
    outside += check(
        "rows wider than the contraction",
        9,
        256,
        512,
        (32, 64, 64),
        False,
        8,
        row_stride=1024,
    )
    outside += check(
        "f16", 33, 512, 1024, (64, 128, 128), False, 9, dtype=torch.float16
    )
    print(
        f"{'ok' if outside == 0 else 'FAIL'}: {outside} values outside the bound"
    )
    return 0 if outside == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
