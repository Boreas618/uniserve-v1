"""The vsa_scores program against the operator's product.

The kernel is a per-head contraction (``VsaScores`` in
``crates/ops/src/vsa.rs``): one dot product per (head, query tile, key tile)
over the pooled rows, accumulated in FP32 and scaled, with no softmax, over
operands in the layout the pooling writes:

```text
scores[head, i, j] = Σ_w query[i, head, w] · key[j, head, w] / sqrt(width)
```

The reference is the same product in FP64 rounded once into FP32, so it is
exact and the bound is the operator's own,
``Tolerance::split_dot(width, tf32x3(width), f32, f32)``: the bound of a
``width``-term contraction with the three-pass product's excess counted. A
dot product's error scales with the sum of its terms' magnitudes, which is
what the bound is held against, so a score that cancels to near zero is
measured on its terms' scale.

Triton's interpreter multiplies ``tl.dot`` in FP32 whatever the
``input_precision``, so it proves the indexing, the masks and the layout; the
three TF32 passes of ``tf32x3`` multiply on a device's tensor cores, which
compile the same cases. The cases: a key-tile count that is not a multiple of
the kernel's key step, so the step's mask is exercised; a contraction narrower
than the block of lanes, so the lane mask is; query-tile counts that fill no
program's sixteen; and the destination's layout, read back as the contiguous
``[heads, query tiles, key tiles]`` matrix the selection reads, so a
transposed or padded store fails, and a score left unwritten (its NaN fill)
is outside every bound.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`tests.python.kernels.chain`); that launch needs a device.
"""

import pytest
import torch

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: Unit roundoff of FP32, the accumulator the operator names.
UNIT_ROUNDOFF = 2.0**-24

#: The query tiles one program holds.
QUERIES = 16

#: (heads, width, block, query tiles, key tiles, key tiles a step): the
#: fixture's cell, a key count that is not a multiple of a 16-tile step and a
#: contraction narrower than its lanes; then the launch table's step of 32 over
#: tile counts that fill no step, with a contraction under its lanes.
CASES = [
    pytest.param(2, 128, 128, 3, 7, 16, id="the fixture's cell"),
    pytest.param(3, 64, 128, 2, 5, 16, id="width under its lanes"),
    pytest.param(1, 32, 128, 4, 20, 16, id="keys past one step"),
    pytest.param(2, 128, 128, 37, 70, 32, id="the module's step"),
    pytest.param(3, 96, 128, 20, 45, 32, id="the module's step, narrow"),
]

#: The seed every case draws its operands from.
SEED = 23


def launch(kernels, query, key, out, block, step, pdl=False):
    """One launch scoring every (head, query tile, key tile) into `out`;
    compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    query_tiles, heads, width = query.shape
    kernels._vsa_scores_kernel[(-(-query_tiles // QUERIES), heads, 1)](
        query,
        key,
        out,
        query_tiles,
        key.shape[0],
        int(pdl),
        WIDTH=width,
        BLOCK=block,
        KEY_TILES=step,
        QUERIES=QUERIES,
        PDL=pdl,
        num_warps=8 if block >= 1024 else 4,
        launch_pdl=pdl,
    )


def gamma(reduction, unit=UNIT_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def tf32x3(reduction):
    """``ops::tolerance::tf32x3``: the roundings the three-pass product costs
    beyond FP32's.
    """  # noqa: D205
    return 21 + -(-3 * reduction // 1024)


def declared_bound(width):
    """``VsaScores::tolerance``: ``split_dot(width, tf32x3(width), f32,
    f32)``'s ``rtol``.
    """  # noqa: D205
    return 2.0 * gamma(width + tf32x3(width)) * width**0.5 + UNIT_ROUNDOFF


@pytest.mark.parametrize(
    ("heads", "width", "block", "query_tiles", "key_tiles", "step"), CASES
)
def test_every_score_is_within_the_operators_bound(
    heads,
    width,
    block,
    query_tiles,
    key_tiles,
    step,
    torch_device,
    kernel_source,
):
    torch.manual_seed(SEED)
    # Both operands are FP32: the type `vsa_prepare_tiles` pools a tile's
    # per-head means in, and the only one the operator takes.
    query = torch.randn(query_tiles, heads, width)
    key = torch.randn(key_tiles, heads, width)
    out = torch.full(
        (heads, query_tiles, key_tiles), float("nan"), device=torch_device
    )

    launch(
        kernel_source("vsa_scores"),
        query.to(torch_device),
        key.to(torch_device),
        out,
        block,
        step,
    )

    got = out.cpu()
    expected = (
        torch.einsum("ihw,jhw->hij", query.double(), key.double()) / width**0.5
    ).to(torch.float32)
    magnitude = (
        torch.einsum("ihw,jhw->hij", query.abs().double(), key.abs().double())
        / width**0.5
    )
    bound = declared_bound(width)
    allowed = bound * magnitude.to(torch.float32)
    # `~(error <= allowed)` counts a NaN difference as outside, which covers
    # the scores the kernel never wrote.
    outside = ~((got - expected).abs() <= allowed)
    if outside.any():
        position = int(outside.reshape(-1).nonzero()[0])
        pytest.fail(
            f"heads {heads} width {width} block {block} {query_tiles}x"
            f"{key_tiles} tiles, step {step}: {int(outside.sum())} of "
            f"{got.numel()} scores outside the bound "
            f"({int(torch.isnan(got).sum())} not finite); first at {position}: "
            f"{float(got.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}, rtol {bound:.3e}"
        )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("vsa_scores")
    heads, width, block, query_tiles, key_tiles = 4, 64, 64, 32, 48

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["query"],
            inputs["key"],
            outputs["out"],
            block,
            32,
            pdl,
        )
        return [outputs["out"]]

    case = chain.Case(
        "vsa_scores",
        {
            "query": torch.randn(query_tiles, heads, width),
            "key": torch.randn(key_tiles, heads, width),
        },
        {"out": torch.zeros(heads, query_tiles, key_tiles)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"vsa_scores launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
