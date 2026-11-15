"""The silu program against the operator's formula, and launched dependently.

The activation is one scalar formula per element, not a tensor-core product::

    out = x / (1 + exp(-x))

The reference is computed in FP32 with torch, and the comparison is the
operator's: one rounding from FP32 into the output's type plus an absolute
part covering the exponential's own error. The check uses FP32 buffers so the
comparison isolates the formula; a narrowed output's rounding is the store,
which the operator's tolerance states.

A module compiled for a dependent launch reads only what the grid before it
wrote: the operator launched behind a producer still writing its input
(:mod:`tests.python.kernels.chain`) must give its launch after a
synchronization, bit for bit. That launch needs a device.
"""

import pytest
import torch
import triton

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: The operator's value envelope and its absolute tolerance, from
#: `crates/ops/src/activation.rs`.
VALUE_ENVELOPE = 8.0
ATOL = 4.0 * VALUE_ENVELOPE * torch.finfo(torch.float32).eps

#: The buffers the formula is checked over: rows, columns and the program's
#: block.
SHAPES = [(1, 64, 64), (3, 100, 128), (4, 5376, 1024)]


def launch(kernels, x, out, columns, block, pdl=False):
    """One launch over every element of `x`, compiled for a dependent launch
    when `pdl` is.
    """  # noqa: D205
    kernels._silu_kernel[(triton.cdiv(x.numel(), block),)](
        x,
        out,
        x.numel(),
        columns,
        BLOCK=block,
        PDL=pdl,
        launch_pdl=pdl,
        num_warps=4,
    )


@pytest.mark.parametrize(("rows", "columns", "block"), SHAPES)
def test_every_element_matches_the_formula(
    rows, columns, block, torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(rows * columns)
    x = (
        torch.rand(rows, columns, dtype=torch.float32, generator=generator)
        * 2.0
        - 1.0
    ) * VALUE_ENVELOPE
    x = x.to(torch_device)
    out = torch.full_like(x, float("nan"))

    launch(kernel_source("silu"), x, out, columns, block)

    expected = x / (1.0 + torch.exp(-x))
    error = (out - expected).abs().max().item()
    tolerance = (
        torch.finfo(torch.float32).eps * expected.abs().max().item() + ATOL
    )
    assert error <= tolerance, (
        f"rows={rows} columns={columns} block={block}: max |error| = "
        f"{error:.3e}, tolerance {tolerance:.3e}"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("silu")
    rows, columns, block = 8, 1024, 1024

    def launched(inputs, outputs, pdl):
        launch(kernels, inputs["x"], outputs["out"], columns, block, pdl)
        return [outputs["out"]]

    case = chain.Case(
        "silu",
        {"x": torch.randn(rows, columns).to(torch.bfloat16)},
        {"out": torch.zeros(rows, columns, dtype=torch.bfloat16)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"silu launched as a dependent: {failures} of {chain.ROUNDS} rounds "
        "differ from its launch after a synchronization"
    )
