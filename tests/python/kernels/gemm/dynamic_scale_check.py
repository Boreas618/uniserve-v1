#!/usr/bin/env python3
"""Run the dynamic_scale program and check its factors, then the FP8 product they feed.

    python dynamic_scale_check.py              # on a device
    python dynamic_scale_check.py --cpu-only   # the factors alone, on Triton's interpreter

`dynamic_scale` folds an activation's partial largest magnitudes into `amax = max(partials,
floor)` and writes `encode = range / amax` and `alpha = (amax / range) * (weight_amax / range)`,
each division rounded to nearest: the factors are compared with torch's IEEE FP32 arithmetic bit
for bit, for every partial type, counts below, at and past a tile, and an activation of zeros.

On a device the check then runs the measured FP8 projection a converted weight serves: the
weight encoded as the loader encodes it (`e4m3(w * 448 / amax_w)`), each row's largest magnitude,
the factors, the activation's codes (`fp8_quantize`) and `fp8_gemm`, reading them from device
memory. Each output is held to the bound the
formats give against the exact FP64 product of the 16-bit operands:

    e(v)  = 2^-4 |v| + 2^-10 s                       an E4M3 element of scale s
    bound = sum_k (e_x |w| + |x| e_w + e_x e_w)
            + (k + 8) u_f32 sum_k (|x| + e_x)(|w| + e_w)  the accumulation and the factors
            + u_out (|y| + the two terms above)          the output's rounding
"""

import os
import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
sys.path.insert(0, str(KERNELS / "dynamic_scale" / "triton" / "src"))
sys.path.insert(0, str(KERNELS / "fp8_gemm" / "triton" / "src"))
sys.path.insert(0, str(KERNELS / "fp8_quantize" / "triton" / "src"))

CPU_ONLY = "--cpu-only" in sys.argv
if CPU_ONLY:
    os.environ["TRITON_INTERPRET"] = "1"

import dynamic_scale as scale_kernels  # noqa: E402
import torch  # noqa: E402

#: `ops::dynamic_scale`'s ranges and floor.
FP8_RANGE = 448.0
NVFP4_RANGE = 6.0 * 448.0
FLOOR = 1e-12

UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
F32_ROUNDOFF = 2.0**-24

DEVICE = "cpu" if CPU_ONLY else "cuda"


def factors(partials, weight_amax, range_, tile=1024):
    """Launch `dynamic_scale` over `partials` and return `(encode, alpha)` as FP32 tensors."""
    weight = torch.tensor([weight_amax], dtype=torch.float32, device=DEVICE)
    encode = torch.empty(1, dtype=torch.float32, device=DEVICE)
    alpha = torch.empty(1, dtype=torch.float32, device=DEVICE)
    scale_kernels._dynamic_scale_kernel[(1,)](
        partials,
        weight,
        encode,
        alpha,
        partials.numel(),
        range_,
        FLOOR,
        TILE=tile,
        PDL=False,
    )
    if not CPU_ONLY:
        torch.cuda.synchronize()
    return encode.cpu(), alpha.cpu()


def expected_factors(partials, weight_amax, range_):
    """The formula in torch's FP32, whose division and product round to nearest."""
    f32 = torch.float32
    amax = torch.maximum(
        partials.float().cpu().max(), torch.tensor(FLOOR, dtype=f32)
    )
    range_ = torch.tensor(range_, dtype=f32)
    weight = torch.tensor(weight_amax, dtype=f32)
    return (range_ / amax).reshape(1), (
        (amax / range_) * (weight / range_)
    ).reshape(1)


def check_factors():
    """Every partial type and count, both ranges, and zeros: the factors bit for bit."""
    failures = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for count in (1, 37, 1024, 1025, 30_000):
            for range_ in (FP8_RANGE, NVFP4_RANGE):
                generator = torch.Generator().manual_seed(count)
                partials = (torch.rand(count, generator=generator) * 300.0).to(
                    dtype
                )
                if count > 1:
                    partials[count // 2] = 1234.5
                partials = partials.to(DEVICE)
                encode, alpha = factors(partials, 0.37, range_)
                wanted = expected_factors(partials, 0.37, range_)
                same = torch.equal(encode, wanted[0]) and torch.equal(
                    alpha, wanted[1]
                )
                failures += not same
                if not same:
                    print(
                        f"FAIL: {dtype} x {count} at range {range_}: "
                        f"{encode.item()!r}, {alpha.item()!r} against "
                        f"{wanted[0].item()!r}, {wanted[1].item()!r}"
                    )
        zeros = torch.zeros(9, dtype=dtype, device=DEVICE)
        encode, alpha = factors(zeros, 2.0, NVFP4_RANGE)
        wanted = expected_factors(zeros, 2.0, NVFP4_RANGE)
        if not (
            torch.equal(encode, wanted[0])
            and torch.isfinite(encode).all()
            and torch.equal(alpha, wanted[1])
        ):
            failures += 1
            print(f"FAIL: {dtype} zeros: {encode.item()!r}, {alpha.item()!r}")
    print(
        f"{'ok' if failures == 0 else 'FAIL'}: dynamic_scale factors, {failures} wrong"
    )
    return failures


def encoding_bound(values, scale):
    """The E4M3 element bound `2^-4 |v| + 2^-10 s` of every element of `values`, in FP64."""
    return values.abs() / 16.0 + scale / 1024.0


def check_product(label, m, n, k, dtype, seed):
    """The measured FP8 projection of a converted weight against the exact product."""
    import fp8_gemm as gemm_kernels
    import fp8_quantize as encoder

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=generator)
    x[0, 3] = 40.0
    x = x.to(dtype)
    w = (torch.randn(n, k, generator=generator) * 0.05).to(dtype)

    # The loader's encoding: one scale of the whole tensor's largest magnitude.
    weight_amax = w.float().abs().max()
    codes = (
        (w.float() * (torch.tensor(FP8_RANGE) / weight_amax))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )

    # Each row's largest magnitude, as `reduce`'s `absmax` folds it, then the factors.
    x_device = x.cuda()
    partials = x_device.abs().amax(dim=1)
    encode, alpha = factors(partials, weight_amax.item(), FP8_RANGE)

    activation = torch.empty(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    encoder._fp8_quantize_kernel[(m, (k + 1023) // 1024)](
        x_device, activation, k, x_device.stride(0), encode.cuda(), BLOCK=1024
    )
    out = torch.empty(m, n, dtype=dtype, device="cuda")
    block_m, block_n, block_k = 128, 128, 128
    grid = ((m + block_m - 1) // block_m, (n + block_n - 1) // block_n)
    gemm_kernels._fp8_gemm_kernel[grid](
        activation,
        codes.cuda(),
        out,
        out,
        m,
        n,
        k,
        activation.stride(0),
        alpha.cuda(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BIAS=False,
        num_warps=8,
        num_stages=3,
    )
    torch.cuda.synchronize()

    xs, ws = x.double(), w.double()
    e_x = encoding_bound(xs, xs.abs().max() / FP8_RANGE)
    e_w = encoding_bound(ws, ws.abs().max() / FP8_RANGE)
    exact = xs @ ws.T
    encoding = e_x @ ws.abs().T + xs.abs() @ e_w.T + e_x @ e_w.T
    magnitude = (xs.abs() + e_x) @ (ws.abs() + e_w).T
    accumulation = (k + 8) * F32_ROUNDOFF * magnitude
    bound = (
        encoding
        + accumulation
        + UNIT_ROUNDOFF[dtype] * (exact.abs() + encoding + accumulation)
    )

    error = (out.cpu().double() - exact).abs()
    # `~(error <= bound)`: an output that is not a number counts as outside.
    outside = int((~(error <= bound)).sum())
    print(
        f"{'ok' if outside == 0 else 'FAIL'}: {label}, {m} x {n} x {k} {dtype} — max error "
        f"over the bound {float((error / bound).max()):.3f}, relative to the exact product "
        f"{float(error.norm() / exact.norm()):.3e}, {outside} outside"
    )
    return outside


def main():
    if not CPU_ONLY and not torch.cuda.is_available():
        print("dynamic_scale_check needs a CUDA device, or --cpu-only")
        return 1
    failures = check_factors()
    if not CPU_ONLY:
        for label, m, n, k in [
            ("a denoiser feed-forward's rows", 300, 1024, 5376),
            ("one row", 1, 512, 2048),
            ("a masked tail", 77, 200, 96),
        ]:
            failures += check_product(label, m, n, k, torch.bfloat16, m + n)
        failures += check_product(
            "a video decoder's rows", 129, 256, 2048, torch.float16, 5
        )
    print(f"{'ok' if failures == 0 else 'FAIL'}: {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
