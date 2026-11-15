"""The `dynamic_scale` factors bit for bit, then the FP8 product they feed.

`dynamic_scale` folds an activation's partial largest magnitudes into
`amax = max(partials, floor)` and writes `encode = range / amax` and
`alpha = (amax / range) · (weight_amax / range)`, each division rounded to
nearest: the factors are compared with torch's IEEE FP32 arithmetic bit for
bit, for every partial type, counts below, at and past a tile, both ranges,
and an activation of zeros.

On a device the check then runs the measured FP8 projection a converted weight
serves: the weight encoded as the loader encodes it (`e4m3(w · 448 / amax_w)`),
each row's largest magnitude, the factors, the activation's codes
(`fp8_quantize`) and `fp8_gemm`, reading them from device memory. Each output
is held to the bound the formats give against the exact FP64 product of the
16-bit operands::

    e(v)  = 2^-4 |v| + 2^-10 s                      an E4M3 element of scale s
    bound = sum_k (e_x |w| + |x| e_w + e_x e_w)
            + (k + 8) u_f32 sum_k (|x| + e_x)(|w| + e_w)   the accumulation
                                                           and the factors
            + u_out (|y| + the two terms above)    the output's rounding

The interpreter does not store E4M3 as a device does, so the product needs a
device.
"""

import pytest
import torch

from tests.python.kernels.gemm import fp8

pytestmark = pytest.mark.unit

#: `ops::dynamic_scale`'s ranges and floor.
FP8_RANGE = 448.0
NVFP4_RANGE = 6.0 * 448.0
FLOOR = 1e-12

#: The factors' fold tile.
TILE = 1024

UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
F32_ROUNDOFF = 2.0**-24

#: The weight's largest magnitude the factor cases pass.
WEIGHT_AMAX = 0.37

#: The measured projection's output tile and contraction step.
PRODUCT_BLOCKS = (128, 128, 128)


def factors(kernels, partials, weight_amax, range_):
    """Launch `dynamic_scale` over `partials`; answers `(encode, alpha)` as
    FP32 host tensors.
    """  # noqa: D205
    device = partials.device
    weight = torch.tensor([weight_amax], dtype=torch.float32, device=device)
    encode = torch.empty(1, dtype=torch.float32, device=device)
    alpha = torch.empty(1, dtype=torch.float32, device=device)
    kernels._dynamic_scale_kernel[(1,)](
        partials,
        weight,
        encode,
        alpha,
        partials.numel(),
        range_,
        FLOOR,
        TILE=TILE,
        PDL=False,
    )
    return encode.cpu(), alpha.cpu()


def expected_factors(partials, weight_amax, range_):
    """The formula in torch's FP32, whose division and product round to
    nearest.
    """  # noqa: D205
    f32 = torch.float32
    amax = torch.maximum(
        partials.float().cpu().max(), torch.tensor(FLOOR, dtype=f32)
    )
    range_ = torch.tensor(range_, dtype=f32)
    weight = torch.tensor(weight_amax, dtype=f32)
    return (range_ / amax).reshape(1), (
        (amax / range_) * (weight / range_)
    ).reshape(1)


DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


@pytest.mark.parametrize(
    "range_", [FP8_RANGE, NVFP4_RANGE], ids=["fp8 range", "nvfp4 range"]
)
@pytest.mark.parametrize("count", [1, 37, 1024, 1025, 30_000])
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_factors_are_the_formulas_bit_for_bit(
    dtype, count, range_, torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(count)
    partials = (torch.rand(count, generator=generator) * 300.0).to(
        DTYPES[dtype]
    )
    if count > 1:
        partials[count // 2] = 1234.5
    partials = partials.to(torch_device)

    encode, alpha = factors(
        kernel_source("dynamic_scale"), partials, WEIGHT_AMAX, range_
    )

    wanted = expected_factors(partials, WEIGHT_AMAX, range_)
    assert torch.equal(encode, wanted[0]) and torch.equal(alpha, wanted[1]), (
        f"{dtype} x {count} at range {range_}: {encode.item()!r}, "
        f"{alpha.item()!r} against {wanted[0].item()!r}, {wanted[1].item()!r}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_activation_of_zeros_takes_the_floor(
    dtype, torch_device, kernel_source
):
    zeros = torch.zeros(9, dtype=DTYPES[dtype], device=torch_device)

    encode, alpha = factors(
        kernel_source("dynamic_scale"), zeros, 2.0, NVFP4_RANGE
    )

    wanted = expected_factors(zeros, 2.0, NVFP4_RANGE)
    assert (
        torch.equal(encode, wanted[0])
        and bool(torch.isfinite(encode).all())
        and torch.equal(alpha, wanted[1])
    ), (
        f"{dtype} zeros: {encode.item()!r}, {alpha.item()!r} against "
        f"{wanted[0].item()!r}, {wanted[1].item()!r}"
    )


def encoding_bound(values, scale):
    """The E4M3 element bound `2^-4 |v| + 2^-10 s` of every element of
    `values`, in FP64.
    """  # noqa: D205
    return values.abs() / 16.0 + scale / 1024.0


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("m", "n", "k", "dtype", "seed"),
    [
        pytest.param(
            300, 1024, 5376, torch.bfloat16, 1324,
            id="a denoiser feed-forward's rows",
        ),
        pytest.param(1, 512, 2048, torch.bfloat16, 513, id="one row"),
        pytest.param(77, 200, 96, torch.bfloat16, 277, id="a masked tail"),
        pytest.param(
            129, 256, 2048, torch.float16, 5, id="a video decoder's rows"
        ),
    ],
)  # fmt: skip
def test_a_measured_fp8_projection_is_within_its_formats_bound(
    m, n, k, dtype, seed, torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=generator)
    x[0, 3] = 40.0
    x = x.to(dtype)
    w = (torch.randn(n, k, generator=generator) * 0.05).to(dtype)

    # The loader's encoding: one scale of the whole tensor's largest magnitude.
    weight_amax = w.float().abs().max()
    codes = (
        (w.float() * (torch.tensor(FP8_RANGE) / weight_amax))
        .clamp(-FP8_RANGE, FP8_RANGE)
        .to(torch.float8_e4m3fn)
    )

    # Each row's largest magnitude, as `reduce`'s `absmax` folds it, then the
    # factors.
    x_device = x.to(torch_device)
    partials = x_device.abs().amax(dim=1)
    encode, alpha = factors(
        kernel_source("dynamic_scale"), partials, weight_amax.item(), FP8_RANGE
    )
    activation = torch.empty(
        m, k, dtype=torch.float8_e4m3fn, device=torch_device
    )
    fp8.quantize(
        kernel_source("fp8_quantize"),
        x_device,
        activation,
        encode.to(torch_device),
    )
    out = torch.empty(m, n, dtype=dtype, device=torch_device)
    fp8.multiply(
        kernel_source("fp8_gemm"),
        activation,
        codes.to(torch_device),
        None,
        out,
        alpha.to(torch_device),
        PRODUCT_BLOCKS,
    )

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
    assert outside == 0, (
        f"{m} x {n} x {k} {dtype}: {outside} outside the formats' bound; max "
        f"error over the bound {float((error / bound).max()):.3f}, relative "
        f"to the exact product {float(error.norm() / exact.norm()):.3e}"
    )
