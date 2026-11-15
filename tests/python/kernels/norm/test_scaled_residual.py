"""The scaled_residual program against FP64, the whole stream, in place.

The operator is ``hidden += (update [+ bias]) * scale``, where ``scale`` is a
per-channel vector or a single factor for the whole stream, and it writes
**in place**: the residual tensor it is given is the one that comes back. The
element count is a runtime extent while the scale's width and the flags are
compiled in, so the cases are what a flat program over a row-major stream can
get wrong: a row width the block does not divide (a block that straddles two
rows, where the channel index must come from the element's own position and
not from the program's), a width that is a multiple of the block, and a single
broadcast factor rather than a vector.

The reference is torch in FP64 and the bound the operator's
(`ScaledResidual::tolerance`): three roundings in the output type (the biased
update, the product, the sum) and no reduction, a relative comparison at
``3 u_out`` with ``atol`` zero.

That bound is about the terms, not about the sum, and a residual add cancels:
an element whose sum is near zero has to agree relatively, which an FP32
stream does not at the 1152-wide case (149 of 4608 elements differ by up to
``6.8e-5`` relative in any implementation, because ``hidden`` and
``update * scale`` are each larger than their sum). A failure states both
counts, the declared bound's and the one on the terms' magnitudes
(``3 u_out (|hidden| + |update * scale|)``); the declared one decides, since
the tolerance is the operator's to change.
"""

import pytest
import torch
from uniserve_testing.kernels.norm.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    interpreter_store,
    largest_relative,
    outside,
)

pytestmark = pytest.mark.unit

#: `ScaledResidual::tolerance`'s count of output roundings.
ROUNDINGS = 3.0

#: The cases, cheapest and most diverse first: the rows, the row width, the
#: program's block, whether the update carries a bias, whether the scale is
#: per channel, and the seed.
CASES = {
    "the fixture's width, 1152 in blocks of 1024, blocks straddling rows": (
        4,
        1152,
        1024,
        False,
        True,
        1,
    ),
    "the fixture's width with an update bias": (4, 1152, 1024, True, True, 1),
    "one factor for the whole stream": (3, 1152, 1024, False, False, 2),
    "a width that is a whole number of blocks": (2, 512, 512, True, True, 3),
}


def launch(kernels, hidden, update, bias, scale, block):
    """One launch over the flattened stream, `ceil(elements / block)`
    programs; a call without a bias passes the update's address for it.
    """  # noqa: D205
    elements = hidden.numel()
    kernels._scaled_residual_kernel[(-(-elements // block),)](
        hidden,
        update,
        bias if bias is not None else update,
        scale,
        elements,
        HAS_UPDATE_BIAS=bias is not None,
        WIDTH=scale.numel(),
        BLOCK=block,
    )


@pytest.mark.parametrize(
    "dtype", ["bf16", pytest.param("f16", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("case", CASES)
def test_the_stream_matches_fp64_within_the_operators_bound(
    case, dtype, torch_device, interpreted, kernel_source
):
    rows, columns, block, update_bias, per_channel, seed = CASES[case]
    element = TORCH_TYPES[dtype]
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, columns, generator=generator).to(element)
    update = torch.randn(rows, columns, generator=generator).to(element)
    bias = (
        torch.randn(columns, generator=generator).to(element)
        if update_bias
        else None
    )
    scale = torch.randn(columns if per_channel else 1, generator=generator).to(
        element
    )
    before = hidden.double()
    moved = hidden.to(torch_device)

    launch(
        kernel_source("scaled_residual"),
        moved,
        update.to(torch_device),
        bias.to(torch_device) if bias is not None else None,
        scale.to(torch_device),
        block,
    )

    exact_update = update.double()
    if bias is not None:
        exact_update = exact_update + bias.double()
    product = exact_update * scale.double()
    expected = interpreter_store(before + product, dtype, interpreted)
    rounding = ROUNDINGS * UNIT_ROUNDOFF[dtype]
    difference = (moved.cpu().double() - expected).abs()
    declared = outside(difference, rounding * expected.abs())
    on_terms = outside(difference, rounding * (before.abs() + product.abs()))
    assert declared == 0, (
        f"{case}, {rows} x {columns} of block {block}, {dtype}: {declared} "
        f"elements outside the declared bound ({on_terms} on the terms' "
        f"magnitudes), max absolute "
        f"{difference.nan_to_num(nan=float('inf')).max().item():.3e}, max "
        f"relative {largest_relative(difference, expected):.3e}, rtol "
        f"{rounding:.3e}"
    )
