"""The gated_rms_norm program against FP64, every row.

The operator is a gated delta network's output norm::

    x * rsqrt(mean(x^2) + eps) * w * act(z)

normalized before it is gated (FLA's ``RMSNormGated``,
``norm_before_gate=True``), with the SiLU or the sigmoid
(``output_gate_type: "sigmoid"``) as ``act``. The cases are what a masked row
reduction and a gate can get wrong: the head width the model serves, a row far
below its tile, a row of zeros (whose output is exactly zero), and gates
spanning the SiLU's saturation.

The reference is FP64 and the bound the operator's
(`GatedRmsNorm::tolerance`): the sum of squares of ``columns`` terms in FP32
(``2 gamma_n``), the scale, the SiLU's and the products' few FP32 ulps
(``3 u_f32``), and one rounding into the output type, relative to each
element. A narrow store on Triton's interpreter truncates where a device
rounds, so the interpreter is allowed the one unit in the last place that
costs.
"""

import pytest
import torch
from uniserve_testing.kernels.norm.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    gamma,
    outside,
)

pytestmark = pytest.mark.unit

#: The cases: the rows, the row width, the program's tile and the seed.
CASES = {
    "the model's head rows": (32 * 7, 128, 128, 1),
    "a row far below its tile": (5, 100, 256, 2),
    "a wider row": (3, 2048, 2048, 3),
}

EPS = 1e-6


def bound(columns, dtype, interpreted):
    """`GatedRmsNorm::tolerance`: `rounding(columns, f32, out)` plus three
    FP32 ulps, with the interpreter's truncating store worth a whole unit of
    the narrow output where a device's rounding is worth half of one.
    """  # noqa: D205
    store = UNIT_ROUNDOFF[dtype] * (2 if interpreted and dtype != "f32" else 1)
    return 2.0 * gamma(columns) + store + 3.0 * UNIT_ROUNDOFF["f32"]


def launch(kernels, x, gate, weight, out, columns, tile, sigmoid):
    """One launch, one program per row."""
    kernels._gated_rms_norm_kernel[(x.shape[0],)](
        x,
        gate,
        weight,
        out,
        columns,
        EPS,
        BLOCK=tile,
        SIGMOID=sigmoid,
        num_warps=4,
    )


@pytest.mark.parametrize("gate", ["silu", "sigmoid"])
@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("case", CASES)
def test_every_row_matches_fp64_within_the_operators_bound(
    case, dtype, gate, torch_device, interpreted, kernel_source
):
    rows, columns, tile, seed = CASES[case]
    element = TORCH_TYPES[dtype]
    torch.manual_seed(seed)
    x = torch.randn(rows, columns).to(element)
    x[0] = 0.0
    z = (torch.randn(rows, columns) * 6.0).to(element)
    weight = (torch.rand(columns) + 0.5).to(element)
    out = torch.full(
        (rows, columns), float("nan"), dtype=element, device=torch_device
    )

    launch(
        kernel_source("gated_rms_norm"),
        x.to(torch_device),
        z.to(torch_device),
        weight.to(torch_device),
        out,
        columns,
        tile,
        gate == "sigmoid",
    )

    exact = x.double()
    normalized = exact * torch.rsqrt(
        (exact * exact).mean(-1, keepdim=True) + EPS
    )
    opened = (
        torch.sigmoid(z.double())
        if gate == "sigmoid"
        else z.double() * torch.sigmoid(z.double())
    )
    expected = normalized * weight.double() * opened
    error = (out.cpu().double() - expected).abs()
    relative = bound(columns, dtype, interpreted)
    missed = outside(error, relative * expected.abs())
    assert missed == 0, (
        f"{case}, {rows} x {columns} of tile {tile}, {dtype}, {gate} gate: "
        f"{missed} values outside the bound, max |error| "
        f"{float(error.max()):.3e}, rtol {relative:.3e}"
    )
