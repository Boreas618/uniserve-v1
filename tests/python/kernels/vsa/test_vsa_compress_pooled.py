"""The vsa_compress_pooled program against the operator's composition.

The operator is one softmax over the key tiles, weighted into the pooled
per-head means, narrowed once into the attention's row type:

```text
weights[head, i, j] = softmax_j(scores[head, i, j])    tiles with live keys only
out[head, i, w]     = Σ_j weights[head, i, j] · values[j, head, w]
```

The reference is that composition in torch's FP32, before the store's
rounding, and each output element is held against its own magnitude at the
operator's bound, ``Tolerance::split_dot(n, tf32x3(n), f32, out)`` over the
longer of the softmax's tiles and the weighted sum's width.

On Triton's interpreter, whose ``tl.dot`` is an FP32 product whatever the
``input_precision``, the cases are the properties a wrong program breaks, over
a few tiles and a width that is not a power of two:

- **every tile live**, the scores an asymmetric ramp, so a program that swapped
  the head and the query tile in its indexing fails;
- **an empty tile** (``sizes[j] == 0``), first, last and every one, which the
  softmax excludes before it renormalizes the rest;
- **a partial last tile**, whose size changes no weight (the pooling already
  averaged its live keys): the program must not renormalize by key count;
- **``width < block``**, the lane mask;
- **BF16 rows**, narrowed once by the store.

Those extents (four key tiles a step, eight lanes) are the interpreter's: a
device's ``tl.dot`` takes at least sixteen. On a device the weighted sum is
three TF32 passes on the tensor cores, and the module's own extents run (the
launch table's 64 key tiles a step over a tile count that fills no step, some
tiles empty, a width under its lanes, query tiles past one program's sixteen),
with positive values, so no output cancels below the scale its own magnitude
measures.

A BF16 store moves a value by half a unit in the last place on a device and
by a whole one on Triton's interpreter, which truncates where a device rounds;
the bound's output rounding is that step.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

#: The query tiles one program holds.
QUERIES = 16

#: The interpreter's extents: a few tiles, a width that is not a power of two.
INTERPRETED = {
    "heads": 2,
    "width": 6,
    "query_tiles": 2,
    "tiles": 3,
    "max_tiles": 4,
    "keys": 4,
    "block": 8,
}

#: A device's extents: the module's 64 key tiles a step over a tile count that
#: fills no step, a width under its lanes, and query tiles that fill one
#: program's sixteen and part of another's.
ON_DEVICE = {
    "heads": 2,
    "width": 96,
    "query_tiles": 20,
    "tiles": 150,
    "max_tiles": 1024,
    "keys": 64,
    "block": 128,
}

#: The interpreter's cases: the valid size of each of the three tiles.
SIZES = {
    "every tile live": [4, 4, 4],
    "an empty tile": [4, 0, 4],
    "the last tile empty": [4, 4, 0],
    "every tile empty": [0, 0, 0],
    "a partial last tile": [4, 4, 2],
    "width under block": [4, 4, 4],
}

#: The unit roundoff of the accumulation's type.
UNIT_ROUNDOFF = 2.0**-24

#: The step one store moves an output by, relative, per row type and mode: a
#: BF16 value just above a power of two steps by 2^-7 of it, half of which is
#: a round-to-nearest store's on a device and all of which the interpreter's
#: truncation may take.
OUTPUT_ROUNDOFF = {
    (torch.float32, False): UNIT_ROUNDOFF,
    (torch.float32, True): UNIT_ROUNDOFF,
    (torch.bfloat16, False): 2.0**-8,
    (torch.bfloat16, True): 2.0**-7,
}

#: The seed every case draws its operands from.
SEED = 7


def launch(kernels, scores, values, sizes, out, shape, pdl=False):
    """One launch compressing every (head, query tile) row into `out`;
    compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    heads, query_tiles = shape["heads"], shape["query_tiles"]
    kernels._vsa_compress_pooled_kernel[(-(-query_tiles // QUERIES), heads, 1)](
        scores,
        values,
        sizes,
        out,
        query_tiles,
        shape["tiles"],
        HEADS=heads,
        WIDTH=shape["width"],
        BLOCK=shape["block"],
        QUERIES=QUERIES,
        KEYS=shape["keys"],
        PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )


def operands(shape, positive):
    """The scores and the pooled values in the layouts the operator states.

    Positive operands spread the weight over many tiles and sum without
    cancelling; the others are an asymmetric ramp of scores, so a head/query
    swap changes the answer, over values of both signs.
    """
    heads, query_tiles, tiles, width = (
        shape[name] for name in ("heads", "query_tiles", "tiles", "width")
    )
    generator = torch.Generator().manual_seed(SEED)
    if positive:
        scores = torch.rand((heads, query_tiles, tiles), generator=generator)
        values = torch.rand((tiles, heads, width), generator=generator) + 0.5
        return scores * 4.0, values
    scores = torch.arange(heads * query_tiles * tiles, dtype=torch.float32)
    scores = (scores.reshape(heads, query_tiles, tiles) - 5.0) / 3.0
    scores = scores + torch.rand(scores.shape, generator=generator) * 0.25
    values = torch.rand((tiles, heads, width), generator=generator) * 2.0 - 1.0
    return scores, values


def reference(scores, values, sizes):
    """The composition the operator states, in torch's FP32."""
    weights = scores.clone()
    weights[:, :, sizes == 0] = float("-inf")
    peak = weights.max(dim=-1, keepdim=True).values
    peak = torch.where(torch.isinf(peak), torch.zeros_like(peak), peak)
    exponent = torch.exp(weights - peak)
    total = exponent.sum(dim=-1, keepdim=True)
    total = torch.where(total == 0.0, torch.ones_like(total), total)
    return torch.einsum("hqt,thw->hqw", exponent / total, values)


def gamma(reduction, unit=UNIT_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def tf32x3(reduction):
    """``ops::tolerance::tf32x3``: the roundings the three-pass product costs
    beyond FP32's.
    """  # noqa: D205
    return 21 + -(-3 * reduction // 1024)


def declared_bound(reduction, output_roundoff):
    """``VsaCompressPooled::tolerance``: ``split_dot(n, tf32x3(n), f32,
    output)``'s ``rtol``.
    """  # noqa: D205
    return (
        2.0 * gamma(reduction + tf32x3(reduction)) * reduction**0.5
        + output_roundoff
    )


def compress(name, shape, sizes, element, device, interpreted, kernels):
    """Launch one compression and hold every output element to the reference
    within the operator's bound.
    """  # noqa: D205
    scores, values = operands(shape, positive=not interpreted)
    sizes = torch.tensor(sizes, dtype=torch.int32)
    # NaN until written: an element the program leaves is outside every bound,
    # an all-empty row's zeros included.
    out = torch.full(
        (shape["heads"], shape["query_tiles"], shape["width"]),
        float("nan"),
        dtype=element,
        device=device,
    )

    launch(
        kernels,
        scores.to(device),
        values.to(device),
        sizes.to(device),
        out,
        shape,
    )

    expected = reference(scores, values, sizes)
    got = out.cpu().to(torch.float32)
    # The operator's reduction length is the longer of its two: the tile
    # softmax's (the compiled tile block it reduces in) and the weighted sum's
    # width.
    bound = declared_bound(
        max(shape["max_tiles"], shape["width"]),
        OUTPUT_ROUNDOFF[(element, interpreted)],
    )
    close = (got - expected).abs() <= bound * expected.abs()
    if not close.all():
        differing = (~close).nonzero()
        head, query, feature = differing[0].tolist()
        pytest.fail(
            f"{name}: {len(differing)} of {got.numel()} values outside rtol "
            f"{bound:.3e}; first at head {head}, query {query}, feature "
            f"{feature}: got {got[head, query, feature].item():.9g}, expected "
            f"{expected[head, query, feature].item():.9g}"
        )


@pytest.mark.parametrize(
    ("case", "dtype"),
    [(case, "f32") for case in SIZES] + [("an empty tile", "bf16")],
)
def test_the_composition_on_the_interpreters_extents(
    case, dtype, torch_device, interpreted, kernel_source
):
    if not interpreted:
        pytest.skip(
            "the interpreter's extents: a device's tl.dot takes at least "
            "sixteen keys and lanes, and its product is held by the module's "
            "extents"
        )
    element = {"f32": torch.float32, "bf16": torch.bfloat16}[dtype]
    compress(
        f"{case}, {dtype} rows",
        INTERPRETED,
        SIZES[case],
        element,
        torch_device,
        interpreted,
        kernel_source("vsa_compress_pooled"),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_the_tensor_core_product_meets_its_bound(
    dtype, torch_device, interpreted, kernel_source
):
    # Every seventh tile empty, the last one partial.
    tiles = ON_DEVICE["tiles"]
    sizes = [0 if j % 7 == 3 else 4 for j in range(tiles - 1)] + [2]
    element = {"f32": torch.float32, "bf16": torch.bfloat16}[dtype]
    compress(
        f"many tiles, {dtype} rows",
        ON_DEVICE,
        sizes,
        element,
        torch_device,
        interpreted,
        kernel_source("vsa_compress_pooled"),
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("vsa_compress_pooled")
    shape = {
        "heads": 4,
        "width": 64,
        "query_tiles": 16,
        "tiles": 80,
        "keys": 64,
        "block": 64,
    }

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["scores"],
            inputs["values"],
            inputs["sizes"],
            outputs["out"],
            shape,
            pdl,
        )
        return [outputs["out"]]

    heads, query_tiles, tiles, width = (
        shape[name] for name in ("heads", "query_tiles", "tiles", "width")
    )
    case = chain.Case(
        "vsa_compress_pooled",
        {
            "scores": torch.randn(heads, query_tiles, tiles),
            "values": torch.randn(tiles, heads, width),
            "sizes": torch.randint(1, 64, (tiles,), dtype=torch.int32),
        },
        {"out": torch.zeros(heads, query_tiles, width, dtype=torch.bfloat16)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"vsa_compress_pooled launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a synchronization"
    )
