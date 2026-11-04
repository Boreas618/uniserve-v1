"""The H3 audio decoder's periodic activation: the Triton source of the ``snake`` operator.

```text
out = x + sin²(α · x) / (α + 1e-9)
```

over a row-major ``[rows, channels, width]`` signal with one linear frequency ``α`` per channel. One
element of ``out`` per thread; the channel comes from the linear index, so each thread reads one
``α``.

The formula is the decoder's own (``uniserve/models/minimax_h3/audio_vae.py``, ``Snake.forward``):
``values + (alpha + 1e-9).reciprocal() * sin(alpha * values).pow(2)``. The epsilon keeps the
reciprocal finite for a zero frequency and is part of the formula, not a guard. The arithmetic is
FP32 and the stored value rounds once into the output dtype, which the operator's tolerance covers
along with one unit of the sine's own last place.
"""

import triton
import triton.language as tl

#: The reciprocal's offset, as the decoder's activation evaluates it, instantiated so a
#: kernel may read it.
_EPSILON = tl.constexpr(1e-9)


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _snake_kernel(
    x_ptr,
    alpha_ptr,
    out_ptr,
    rows,
    channels,
    width,
    pdl,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of the activated signal.

    Grid: ``ceil(rows · channels · width / BLOCK)`` programs. ``width`` is the row width the
    descriptor was compiled for and ``channels`` the parameter count; a block covers consecutive
    elements of the row-major signal.
    """
    _pdl_prologue(PDL)

    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    width_of = width.to(tl.int64)
    channels = channels.to(tl.int64)
    valid = offsets < rows * channels * width_of
    channel = (offsets // width_of) % channels

    alpha = tl.load(alpha_ptr + channel, mask=valid).to(tl.float32)
    values = tl.load(x_ptr + offsets, mask=valid).to(tl.float32)
    sine = tl.sin(alpha * values)
    result = values + sine * sine / (alpha + _EPSILON)
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=valid,
    )
