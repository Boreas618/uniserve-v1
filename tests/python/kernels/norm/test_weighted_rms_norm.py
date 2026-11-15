"""The weighted_rms_norm programs against the operator's own formula.

One program normalizes one contiguous row and applies its learned scale, with
the mean square reduced in FP32 and one rounding into the output type::

    inverse  = rsqrt(mean(h^2) + eps)     the statistic over the whole row
    out[r,k] = h[r,k] inverse w[k]        one weight per channel, every row

With ``absmax`` the program also stores, in ``partials[r]``, the largest
magnitude of the row **it stored**, measured after the rounding, over the
whole row and nothing outside it. The caller reduces those per-row values to
one scalar in a separate launch, so the check holds the program to the per-row
magnitudes, which is what the reduction reads.

**What is compared.** The reference is the formula in FP64 and the bound the
operator's (`WeightedRmsNorm::tolerance`)::

    |got - reference| <= (2 gamma(columns, u_f32) + u_out) |reference|

the FP32 reduction of ``columns`` terms plus the single rounding into the
output type. Each case is launched **twice**: with an FP32 output, where
nothing narrows and the comparison is the arithmetic at the bound, and with
the case's own output type, where the narrow store must be **bit for bit** the
store of the wide result: its rounding to nearest on a device, its truncation
on Triton's interpreter, which narrows a store by truncation (a whole unit in
the last place, twice the operator's ``u_out`` at the bottom of a binade).
Neither half hides the other: a program that computes the wrong value fails
the first, and one that stores something other than its own value fails the
second.

The per-row magnitude is held **bit for bit** as well, because a maximum of
stored values is exact: anything else means the program measured a different
quantity (before the rounding, without the weight, or across a lane it does
not store).

The kernel reads at most ``BLOCK`` lanes of a row, so ``tile >= columns`` is
the operator's precondition, and one case puts the row below the tile so the
masking of the lanes past ``columns`` is exercised.
"""

import pytest
import torch

from tests.python.kernels.norm.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    gamma,
    largest_relative,
    outside,
    store,
)

pytestmark = pytest.mark.unit

#: The cases, each named for the mistake it catches: the row width, the
#: program's tile, the rows, the element type, the epsilon, the seed, and what
#: the case adds to random rows (`absmax`, a zero row and a tiny one, a weight
#: of mixed signs with one zero).
CASES = {
    "the fixture's cell, bf16 both ways, 5376 columns in 8192 lanes": {
        "columns": 5376,
        "tile": 8192,
        "rows": 2,
        "dtype": "bf16",
        "eps": 1e-6,
        "seed": 1,
    },
    "an f32 store, the arithmetic with no rounding to hide behind": {
        "columns": 512,
        "tile": 1024,
        "rows": 2,
        "dtype": "f32",
        "eps": 1e-6,
        "seed": 2,
    },
    "a row narrower than the tile, whose lanes past it are masked": {
        "columns": 100,
        "tile": 256,
        "rows": 3,
        "dtype": "bf16",
        "eps": 1e-6,
        "seed": 3,
    },
    "the absmax program, whose magnitudes describe what it stored": {
        "columns": 96,
        "tile": 256,
        "rows": 3,
        "dtype": "bf16",
        "eps": 1e-6,
        "seed": 4,
        "absmax": True,
    },
    "a zero row and a tiny one, whose reciprocal is the epsilon's": {
        "columns": 64,
        "tile": 64,
        "rows": 2,
        "dtype": "f32",
        "eps": 1e-2,
        "seed": 5,
        "zeros": True,
        "tiny": True,
    },
    "a weight of mixed signs and one zero, which multiplies": {
        "columns": 64,
        "tile": 64,
        "rows": 2,
        "dtype": "bf16",
        "eps": 1e-6,
        "seed": 6,
        "signed_weight": True,
    },
}


def bound(columns, dtype):
    """`WeightedRmsNorm::tolerance`: the FP32 reduction, then one rounding
    into `dtype`.
    """  # noqa: D205
    return 2.0 * gamma(columns) + UNIT_ROUNDOFF[dtype]


def launch(kernels, hidden, weight, out, partials, eps, tile):
    """One launch, one program per row: the absmax program when `partials`
    is given, which receives one FP32 magnitude per row.
    """  # noqa: D205
    rows, columns = hidden.shape
    if partials is None:
        kernels._weighted_rms_norm_kernel[(rows,)](
            hidden, weight, out, eps, WIDTH=columns, BLOCK=tile
        )
    else:
        kernels._weighted_rms_norm_absmax_kernel[(rows,)](
            hidden, weight, out, partials, eps, WIDTH=columns, BLOCK=tile
        )


@pytest.mark.parametrize("case", CASES)
def test_a_row_and_its_store_match_the_formula(
    case, torch_device, interpreted, kernel_source
):
    spec = CASES[case]
    columns, tile, rows = spec["columns"], spec["tile"], spec["rows"]
    dtype, eps = spec["dtype"], spec["eps"]
    absmax = spec.get("absmax", False)
    element = TORCH_TYPES[dtype]
    generator = torch.Generator().manual_seed(spec["seed"])
    hidden = torch.randn(rows, columns, generator=generator).to(element)
    weight = torch.randn(columns, generator=generator).to(element)
    if spec.get("signed_weight"):
        weight[::2] = -weight[::2].abs()
        weight[columns // 2] = 0.0
    if spec.get("zeros"):
        hidden[0, :] = 0.0
    if spec.get("tiny"):
        hidden[-1, :] *= 1e-30
    kernels = kernel_source("weighted_rms_norm")

    # One magnitude buffer per launch: the magnitudes describe what that
    # launch stored, so the comparison reads the one the narrow launch wrote.
    outputs = {}
    for name, stored in (("wide", torch.float32), ("narrow", element)):
        out = torch.full(
            (rows, columns), float("nan"), dtype=stored, device=torch_device
        )
        partials = (
            torch.full((rows,), float("nan"), device=torch_device)
            if absmax
            else None
        )
        launch(
            kernels,
            hidden.to(torch_device),
            weight.to(torch_device),
            out,
            partials,
            eps,
            tile,
        )
        outputs[name] = (out.cpu(), partials)
    wide, _ = outputs["wide"]
    narrow, partials = outputs["narrow"]

    exact = hidden.double()
    inverse = torch.rsqrt((exact * exact).mean(dim=1, keepdim=True) + eps)
    expected = exact * inverse * weight.double()
    difference = (wide.double() - expected).abs()
    relative = bound(columns, "f32")
    missed = outside(difference, relative * expected.abs())
    assert missed == 0, (
        f"{case}, {rows} x {columns}, tile {tile}, eps {eps:g}, f32 out: "
        f"{missed} of {expected.numel()} outside the bound, max relative "
        f"error {largest_relative(difference, expected):.3e}, rtol "
        f"{relative:.3e}"
    )

    wanted = store(wide.double(), dtype, interpreted).to(element)
    differing = int((narrow != wanted).sum())
    assert torch.equal(narrow, wanted), (
        f"{case}, {dtype} out: {differing} of {narrow.numel()} stored values "
        f"are not the {'truncation' if interpreted else 'rounding'} of the "
        "wide result"
    )

    if absmax:
        magnitudes = narrow.float().abs().amax(dim=1)
        differing = int((partials.cpu() != magnitudes).sum())
        assert torch.equal(partials.cpu(), magnitudes), (
            f"{case}: {differing} of {rows} per-row magnitudes differ from "
            "the largest magnitude of the row the program stored"
        )
