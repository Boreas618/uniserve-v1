"""The act_and_mul program against torch's own arithmetic, both halves strided.

The operator is ``act(gate) * value`` over a row of ``width`` elements, with
**each half's rows a stride apart**: ``gate_stride`` and ``value_stride`` are
separate, because the two halves are views of a fused projection and nothing
requires them to be adjacent. The program therefore reads both strides; a
check that packed the halves contiguously would pass whether or not the
program honoured them, so the cases build the two strides different as well as
equal, and one case interleaves nothing at all.

The reference is torch in FP64: ``silu`` as ``g * sigmoid(g)``, the logistic
``sigmoid`` of an attention's output gate, and the exact GELU as ``0.5 g (1 +
erf(g / sqrt(2)))``. The bound is the operator's: no reduction, so
``roundings * u_out`` with three roundings for ``silu`` and ``sigmoid`` and
four for the GELU, a relative comparison with no tolerance of the check's own
choosing. The element type is the fixture's, ``bf16``.

The exact GELU needs ``libdevice.erf``, which Triton's interpreter does not
provide: it runs on a device only.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit

#: The relative bound's factor per activation, from `activation.rs::tolerance`.
ROUNDINGS = {"silu": 3.0, "gelu": 4.0, "sigmoid": 3.0}

#: The fixture's descriptor: its element type and the output's unit roundoff
#: (`ops::tolerance::unit_roundoff`), its width and the block one program
#: covers.
DTYPE, UNIT_ROUNDOFF = torch.bfloat16, 2.0**-8
WIDTH, BLOCK = 25600, 1024

#: The cases, cheapest and most diverse first: the rows, their width, each
#: half's row stride, and whether the first row's gates are zero.
CASES = {
    "the fixture's width at the stride a fused projection leaves": {
        "rows": 2,
        "width": WIDTH,
        "gate_stride": WIDTH,
        "value_stride": WIDTH,
        "zero_row": False,
        "seed": 1,
    },
    "the two halves at different strides: nothing is adjacent": {
        "rows": 2,
        "width": WIDTH,
        "gate_stride": WIDTH + 512,
        "value_stride": WIDTH + 1024,
        "zero_row": True,
        "seed": 2,
    },
    "a row narrower than one block: the tail is masked": {
        "rows": 3,
        "width": 100,
        "gate_stride": 100,
        "value_stride": 100,
        "zero_row": False,
        "seed": 3,
    },
}


def activate(gate, name):
    """The reference activation, evaluated in FP64."""
    if name == "sigmoid":
        return torch.sigmoid(gate)
    if name == "silu":
        return gate * torch.sigmoid(gate)
    return 0.5 * gate * (1.0 + torch.erf(gate * math.sqrt(0.5)))


def launch(kernels, gate, value, out, case, activation):
    """One launch over the flattened `[rows, width]` output."""
    elements = case["rows"] * case["width"]
    kernels._act_and_mul_kernel[(math.ceil(elements / BLOCK),)](
        gate,
        value,
        out,
        elements,
        case["gate_stride"],
        case["value_stride"],
        n_cols=case["width"],
        ACTIVATION=activation,
        BLOCK=BLOCK,
        WIDE=False,
        PDL=False,
        stepwise=False,
        num_warps=8,
    )


@pytest.mark.parametrize(
    "activation",
    ["silu", "sigmoid", pytest.param("gelu", marks=pytest.mark.gpu)],
)
@pytest.mark.parametrize("case", CASES)
def test_both_halves_match_torch_within_the_operators_bound(
    case, activation, torch_device, kernel_source
):
    shape = CASES[case]
    generator = torch.Generator().manual_seed(shape["seed"])
    rows, width = shape["rows"], shape["width"]
    # Gates over a wide range, so a saturating activation is exercised too,
    # and values of both signs so a product cannot pass by being zero.
    gate_flat = (
        torch.randn(rows * shape["gate_stride"], generator=generator) * 4.0
    ).to(DTYPE)
    value_flat = torch.randn(
        rows * shape["value_stride"], generator=generator
    ).to(DTYPE)
    # The `[rows, width]` views whose rows are each half's stride apart.
    gate = gate_flat.to(torch_device).as_strided(
        (rows, width), (shape["gate_stride"], 1)
    )
    value = value_flat.to(torch_device).as_strided(
        (rows, width), (shape["value_stride"], 1)
    )
    if shape["zero_row"]:
        gate[0, :] = 0.0
    out = torch.full(
        (rows, width), float("nan"), dtype=DTYPE, device=torch_device
    )

    launch(
        kernel_source("act_and_mul"),
        gate,
        value,
        out,
        shape,
        activation,
    )

    expected = activate(gate.cpu().double(), activation) * value.cpu().double()
    got = out.cpu().double()
    bound = ROUNDINGS[activation] * UNIT_ROUNDOFF
    difference = (got - expected).abs()
    # A difference that is not a number is never inside a bound: `>` says
    # otherwise.
    outside = int((~(difference <= bound * expected.abs())).sum())
    largest = (difference / expected.abs().clamp(min=1e-30)).max().item()
    assert outside == 0, (
        f"{rows} x {width}, {activation}, gate_stride "
        f"{shape['gate_stride']}, value_stride {shape['value_stride']}: "
        f"{outside} of {expected.numel()} outside the bound {bound:.3e}, max "
        f"relative error {largest:.3e}"
    )
