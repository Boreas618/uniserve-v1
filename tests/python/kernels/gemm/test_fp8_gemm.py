"""The per-tensor FP8 product: its codes against torch, its sum against FP64.

The operators are the per-tensor FP8 product: `fp8_quantize` encodes the
activation to E4M3 by its scale (`x / s_x`, clamped to ±448, rounded to nearest
even), and `fp8_gemm` multiplies the codes by the E4M3 weight with an FP32
accumulator and dequantizes by `alpha = s_x · s_w`. The programs read `1 / s_x`
and `alpha` from FP32 device words, as a calibrated export's constants or a
measured scale reach them. The codes are compared with torch's encoding of
the same activation byte for byte; the product with the FP64 sum of the codes'
products.

The bound follows from that arithmetic: the codes are equal, so the two differ
by the FP32 accumulation of `k` products of exact values, at most
`γ_k = k·u_f32/(1 − k·u_f32)` of the sum of the products' magnitudes scaled by
`alpha`, and by the one rounding of the output, `u_out` of the reference
value. The interpreter does not store E4M3 as a device does, so every case
needs a device.
"""

import pytest
import torch
from uniserve_testing.kernels.gemm import fp8

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
F32_ROUNDOFF = 2.0**-24

#: The largest E4M3 magnitude.
E4M3_MAX = 448.0

#: The model's projections run in the product's widest tile.
WIDE = (128, 128, 128)

#: The cases: the extents, the output tile and contraction step, a bias, the
#: seed, the activation's type and the elements between its rows (its
#: contraction when `None`).
CASES = [
    # FP8 projections of a hybrid-attention MoE model: in_proj_qkv, in_proj_z,
    # out_proj, q, k/v.
    pytest.param(
        1,
        8192,
        2048,
        WIDE,
        False,
        8193,
        torch.bfloat16,
        None,
        id="in_proj_qkv, one decode row",
    ),
    pytest.param(
        200,
        4096,
        2048,
        WIDE,
        False,
        4296,
        torch.bfloat16,
        None,
        id="in_proj_z, a prompt",
    ),
    pytest.param(
        37, 2048, 4096, WIDE, False, 2085, torch.bfloat16, None, id="out_proj"
    ),
    pytest.param(
        64, 8192, 2048, WIDE, False, 8256, torch.bfloat16, None, id="q_proj"
    ),
    pytest.param(
        5, 512, 2048, WIDE, False, 517, torch.bfloat16, None, id="k_proj"
    ),
    pytest.param(
        19,
        100,
        96,
        (16, 32, 32),
        True,
        7,
        torch.bfloat16,
        None,
        id="a masked tail and a bias",
    ),
    pytest.param(
        9,
        256,
        512,
        (32, 64, 64),
        False,
        8,
        torch.bfloat16,
        1024,
        id="rows wider than the contraction",
    ),
    pytest.param(
        33, 512, 1024, (64, 128, 128), False, 9, torch.float16, None, id="f16"
    ),
]


@pytest.mark.parametrize(
    ("m", "n", "k", "blocks", "bias", "seed", "dtype", "row_stride"), CASES
)
def test_the_codes_are_torchs_and_the_product_is_within_the_accumulation(
    m, n, k, blocks, bias, seed, dtype, row_stride, torch_device, kernel_source
):
    torch.manual_seed(seed)
    storage = torch.randn(m, row_stride or k, dtype=torch.float32)
    # Activations of a few hundred in magnitude reach the E4M3 range: some
    # saturate.
    storage[0, :4] = torch.tensor([3000.0, -3000.0, 1e-6, 0.0])
    x = storage.to(dtype)[:, :k].to(torch_device)
    weight_float = torch.randn(n, k) * 0.05
    weight_scale = float(weight_float.abs().max() / E4M3_MAX)
    weight = (
        (weight_float / weight_scale)
        .clamp(-E4M3_MAX, E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch_device)
    )
    input_scale = float(storage.abs().max() / E4M3_MAX / 4.0)
    b = (torch.randn(n) * 0.1).to(dtype).to(torch_device) if bias else None
    out = torch.empty(m, n, dtype=dtype, device=torch_device)
    # The two factors are device words of FP32, as the operator passes them.
    encode = torch.tensor(
        [1.0 / input_scale], dtype=torch.float32, device=torch_device
    )
    factor = torch.tensor(
        [input_scale * weight_scale], dtype=torch.float32, device=torch_device
    )
    codes = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=torch_device)

    fp8.quantize(kernel_source("fp8_quantize"), x, codes, encode)
    fp8.multiply(
        kernel_source("fp8_gemm"), codes, weight, b, out, factor, blocks
    )

    wanted = (
        (x.float().cpu() * encode.cpu()[0])
        .clamp(-E4M3_MAX, E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    differing = int(
        (codes.cpu().view(torch.uint8) != wanted.view(torch.uint8)).sum()
    )
    assert differing == 0, (
        f"{m} x {k}: {differing} codes differ from torch's encoding"
    )

    exact_codes = wanted.double()
    weights = weight.cpu().double()
    alpha = float(factor.cpu()[0])
    expected = (exact_codes @ weights.T) * alpha
    magnitudes = (exact_codes.abs() @ weights.abs().T) * alpha
    if bias:
        expected = expected + b.cpu().double()
        magnitudes = magnitudes + b.cpu().double().abs()
    gamma = k * F32_ROUNDOFF / (1 - k * F32_ROUNDOFF)
    bound = gamma * magnitudes + UNIT_ROUNDOFF[dtype] * expected.abs() + 1e-30
    error = (out.cpu().double() - expected).abs()
    # `~(error <= bound)`: an output that is not a number counts as outside.
    outside = int((~(error <= bound)).sum())
    relative = float((error / (magnitudes + 1e-30)).max())
    assert outside == 0, (
        f"{m} x {n} x {k} in {blocks}{' with a bias' if bias else ''}: "
        f"{outside} values outside γ_k · Σ|products| + u_out · |y|; max abs "
        f"{float(error.max()):.3e}, max error over the products' magnitude "
        f"{relative:.3e}"
    )
