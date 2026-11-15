"""The adaptive_pool program against the model's two reductions.

The audio encoder's last reduction averages the attention projection over its
heads and pools the width adaptively; the program does both in one FP32 pass.
The reference is the model's own two steps in FP64: the mean over heads, then
``torch.nn.functional.adaptive_avg_pool1d`` along the width, which reads a
two-dimensional input as (channels, length), exactly the row and width here.

The head count, the width and the pooled width are compile-time values of the
launched kernel, and the longest window is the package's own derivation
(``derive``), so a case may choose a geometry no descriptor declares. The
geometries are chosen for what they break:

- the checkpoint's own (8 heads of 256 pooled to 32) and a width the output
  divides (256 to 64), where every window has one length;
- the degenerate pool (8 pooled to 16): with more outputs than inputs,
  ``floor(i·width/output)`` and ``ceil((i+1)·width/output)`` give an empty
  window at every position, and the pool takes one input position per output,
  repeating it;
- widths the output does not divide (24 to 7, 17 to 5), where a window reaches
  one position past ``ceil(width/output)``, the case that makes the loop bound
  ``ceil(width/output) + 1`` and that a width which divides never shows.

The bound is the operator's, ``rounding(heads · window, f32, dtype)`` (one FP32
accumulation rounded once into the output) with the absolute envelope
``64 · u_out`` an element near zero needs when a window's terms cancel. Over
the geometries it is held against the output's largest magnitude in FP32; over
every element type it is held element by element. A BF16 store on Triton's
interpreter truncates where a device rounds, a whole unit in the last place
instead of half of one, which the element-type comparison allows there.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: `ops::tolerance::unit_roundoff` of each element type.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

#: (heads, width, output, rows, block): the geometries above, in the order in
#: which they break things.
GEOMETRIES = [
    (8, 256, 32, 2, 64),
    (8, 256, 64, 3, 64),
    (8, 8, 16, 2, 64),
    (16, 24, 7, 2, 16),
    (8, 24, 7, 1, 64),
    (4, 17, 5, 3, 8),
]

#: (heads, width, output) of each element type's comparison: the checkpoint's,
#: the degenerate pool and a width the output divides, over three rows in
#: blocks of 1024.
POOLS = [(8, 256, 32), (8, 8, 16), (8, 256, 64)]
ROWS = 3
BLOCK = 1024


def launch(kernels, x, out, heads, width, output, block, pdl=False):
    """One launch over every output element of `out`, compiled for a
    dependent launch when `pdl` is.
    """  # noqa: D205
    rows = x.shape[0]
    # The longest window, as the package derives it for its launch table.
    derived = kernels.derive({"width": width, "output": output}, None, None)
    kernels._adaptive_pool_kernel[(-(-(rows * output) // block),)](
        x,
        out,
        rows,
        heads,
        width,
        output,
        HEADS=heads,
        WIDTH=width,
        OUTPUT=output,
        WINDOW=derived["window"],
        BLOCK=block,
        PDL=pdl,
        num_warps=max(1, min(8, block // 256)),
        launch_pdl=pdl,
    )


def windows(width, output):
    """The lengths of the pool's windows: ``floor`` to ``ceil``, never empty."""
    lengths = set()
    for position in range(output):
        start = (position * width) // output
        end = -((-(position + 1) * width) // output)
        lengths.add(max(end, start + 1) - start)
    return sorted(lengths)


def relative_bound(heads, window, dtype):
    """``Tolerance::rounding(heads · window, f32, dtype)``'s relative part."""
    terms = heads * window * UNIT_ROUNDOFF["f32"]
    return 2.0 * terms / (1.0 - terms) + UNIT_ROUNDOFF[dtype]


def two_reductions(x, output):
    """The model's two steps in FP64: the head mean, then the width pool."""
    return functional.adaptive_avg_pool1d(x.double().cpu().mean(dim=1), output)


@pytest.mark.parametrize(
    ("heads", "width", "output", "rows", "block"),
    GEOMETRIES,
    ids=[f"{h}x{w}to{o}" for h, w, o, _, _ in GEOMETRIES],
)
def test_every_geometry_pools_within_the_operators_bound(
    heads, width, output, rows, block, torch_device, kernel_source
):
    generator = np.random.default_rng(heads * 1000 + width)
    x = torch.from_numpy(
        generator.normal(0.0, 1.0, (rows, heads, width)).astype(np.float32)
    ).to(torch_device)
    out = torch.zeros((rows, output), dtype=torch.float32, device=torch_device)

    launch(kernel_source("adaptive_pool"), x, out, heads, width, output, block)

    expected = two_reductions(x, output)
    difference = (out.double().cpu() - expected).abs().max().item()
    longest = windows(width, output)[-1]
    allowed = (
        relative_bound(heads, longest, "f32") * expected.abs().max().item()
        + 64.0 * UNIT_ROUNDOFF["f32"]
    )
    # A difference that is not a number is never inside a bound.
    assert difference <= allowed, (
        f"{heads} heads of {width} to {output}, {rows} rows, windows "
        f"{windows(width, output)}: max |out - exact| = {difference:.3e}, "
        f"allowed {allowed:.3e}"
    )


@pytest.mark.parametrize("dtype", list(TORCH_TYPES))
@pytest.mark.parametrize(
    ("heads", "width", "output"),
    POOLS,
    ids=[f"{h}x{w}to{o}" for h, w, o in POOLS],
)
def test_every_element_is_within_the_operators_bound(
    heads, width, output, dtype, torch_device, interpreted, kernel_source
):
    generator = torch.Generator().manual_seed(0)
    # The attended values of a projection are bounded in practice; a plain
    # normal keeps the comparison about the accumulation rather than about a
    # designed distribution.
    x = torch.randn(ROWS, heads, width, generator=generator).to(
        device=torch_device, dtype=TORCH_TYPES[dtype]
    )
    out = torch.empty(
        (ROWS, output), dtype=TORCH_TYPES[dtype], device=torch_device
    )

    launch(kernel_source("adaptive_pool"), x, out, heads, width, output, BLOCK)

    expected = two_reductions(x, output)
    window = max(-(-width // output), 1)
    relative = relative_bound(heads, window, dtype)
    if interpreted and dtype == "bf16":
        relative += UNIT_ROUNDOFF["bf16"]
    allowed = 64.0 * UNIT_ROUNDOFF[dtype] + relative * expected.abs()
    got = out.double().cpu()
    difference = (got - expected).abs()
    # A difference that is not a number is never inside a bound: `>` says
    # otherwise.
    wrong = ~(difference <= allowed)
    if wrong.any():
        row, position = (int(value) for value in torch.argwhere(wrong)[0])
        pytest.fail(
            f"{dtype}, {heads} heads of {width} pooled to {output} (window "
            f"{window}), {ROWS} rows: {int(wrong.sum())} element(s) outside "
            f"the bound, max |out - exact| = {difference.max().item():.6g}; "
            f"at [row {row}, output {position}]: got "
            f"{got[row, position].item():.9g}, exact "
            f"{expected[row, position].item():.9g}, allowed "
            f"{allowed[row, position].item():.6g}"
        )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("adaptive_pool")
    rows, heads, width, output, block = 16, 4, 64, 8, 64

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["x"],
            outputs["out"],
            heads,
            width,
            output,
            block,
            pdl,
        )
        return [outputs["out"]]

    case = chain.Case(
        "adaptive_pool",
        {"x": torch.randn(rows, heads, width).to(torch.bfloat16)},
        {"out": torch.zeros(rows, output, dtype=torch.bfloat16)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"adaptive_pool launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
