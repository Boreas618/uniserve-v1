"""The sigmoid_gated_add program against FP64, every row.

The operator joins a shared expert to the routed ones::

    target[r] += sigmoid(gate[r]) * update[r]

with one logit per row. The cases are a row width the blocks divide and one
they do not (a masked tail), with logits spanning the sigmoid's saturation, so
a row whose gate is exactly zero or one in FP32 is in the batch.

The reference is FP64 and the bound the operator's
(`SigmoidGatedAdd::tolerance`): the sigmoid's few FP32 ulps, one product and
one sum, rounded once into the target type, ``2 u_out + 4 u_f32`` (the device
exponential is an approximation of a few FP32 ulps), relative to the sum of
the two terms' magnitudes, because the sum may cancel.
"""

import pytest
import torch

from tests.python.kernels.norm.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    outside,
)

pytestmark = pytest.mark.unit

#: The cases: the rows, the row width, the program's block and the seed.
CASES = {
    "the model's hidden rows": (17, 2048, 1024, 1),
    "a masked tail": (4, 1000, 256, 2),
}

#: The logits of the first and the last row, where the sigmoid is exactly one
#: and exactly zero in FP32.
SATURATED = (40.0, -40.0)


def launch(kernels, target, update, gate, columns, block):
    """One launch over every row, `ceil(columns / block)` programs per row."""
    rows = target.shape[0]
    kernels._sigmoid_gated_add_kernel[((columns + block - 1) // block, rows)](
        target, update, gate, columns, BLOCK=block, num_warps=4
    )


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("case", CASES)
def test_every_row_matches_fp64_within_the_operators_bound(
    case, dtype, torch_device, kernel_source
):
    rows, columns, block, seed = CASES[case]
    element = TORCH_TYPES[dtype]
    torch.manual_seed(seed)
    target = torch.randn(rows, columns).to(element)
    update = torch.randn(rows, columns).to(element)
    gate = (torch.randn(rows, 1) * 8.0).to(element)
    gate[0, 0], gate[-1, 0] = SATURATED
    # The program adds in place: the operand is kept apart from the target it
    # launches on, which on the host is the same tensor otherwise.
    moved = target.clone().to(torch_device)

    launch(
        kernel_source("sigmoid_gated_add"),
        moved,
        update.to(torch_device),
        gate.to(torch_device),
        columns,
        block,
    )

    weight = torch.sigmoid(gate.double())
    expected = target.double() + weight * update.double()
    terms = target.double().abs() + (weight * update.double()).abs()
    error = (moved.cpu().double() - expected).abs()
    relative = 2.0 * UNIT_ROUNDOFF[dtype] + 4.0 * UNIT_ROUNDOFF["f32"]
    missed = outside(error, relative * terms)
    assert missed == 0, (
        f"{case}, {rows} x {columns} in blocks of {block}, {dtype}: {missed} "
        f"values outside the bound, max |error| {float(error.max()):.3e}, "
        f"rtol {relative:.3e} of the terms' magnitudes"
    )
