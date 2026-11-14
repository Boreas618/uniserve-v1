"""An activation's measured quantization factors: the ``dynamic_scale`` operator's source.

```text
amax   = max(max_i partials[i], floor)
encode = range / amax
alpha  = (amax / range) * (weight_amax / range)
```

``partials`` hold the largest magnitudes of the activation's parts (its rows),
in the activation's element type or FP32; every magnitude widens to FP32
exactly. The divisions are ``div_rn`` and the product an FP32 multiply, each
rounding to nearest, so the factors are the IEEE results of the formula: the
weight's scale ``weight_amax / range`` is the number its encoding at load was
divided by.

The kernel body imports nothing but Triton.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):  # noqa: N803
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _dynamic_scale_kernel(
    partials,
    weight_amax,
    encode,
    alpha,
    count,
    range_,
    floor,
    TILE: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Fold ``count`` partials and write ``encode`` and ``alpha``; grid ``(1,)``.

    ``partials`` is ``[count]`` contiguous; ``weight_amax``, ``encode`` and
    ``alpha`` are one FP32 word each.
    """
    _pdl_prologue(PDL)

    # Magnitudes are never below zero, so zero is the fold's neutral value and
    # a lane past the count reads it.
    best = tl.zeros([TILE], tl.float32)
    for start in range(0, count, TILE):
        index = start + tl.arange(0, TILE).to(tl.int64)
        values = tl.load(partials + index, mask=index < count, other=0.0)
        best = tl.maximum(best, values.to(tl.float32))

    amax = tl.maximum(tl.max(best, axis=0), floor)
    weight = tl.load(weight_amax)

    tl.store(encode, tl.math.div_rn(range_, amax))
    tl.store(
        alpha, tl.math.div_rn(amax, range_) * tl.math.div_rn(weight, range_)
    )
