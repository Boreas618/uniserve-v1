"""The audio decoder's periodic activation with a separate gain: the Triton source of the
``snake_beta`` operator.

```text
out = x + sin²(exp(α) · x) / (exp(β) + 1e-9)
```

over a row-major ``[rows, channels, width]`` signal, with ``α`` and ``β`` stored in log space as one
pair per channel, so a channel's frequency and its reciprocal gain are learned independently. One
element of ``out`` per thread; the channel comes from the linear index, so each thread reads one
``α`` and one ``β``.

The formula is the one BigVGAN-style blocks evaluate
(``diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py``), which UniServe's audio
decoder follows. It is **not** ``snake``: H3's audio VAE stores one linear parameter that serves as
both frequency and gain, so the two operators read different checkpoints and neither is a flag of
the other. The epsilon keeps the reciprocal finite at ``β = -∞``. The arithmetic is FP32 and the
stored value rounds once into the output dtype.
"""

import triton
import triton.language as tl

#: The reciprocal's offset, as the activation evaluates it, instantiated so a kernel may
#: read it.
_EPSILON = tl.constexpr(1e-9)


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _snake_beta_kernel(
    x_ptr,
    alpha_ptr,
    beta_ptr,
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
    beta = tl.load(beta_ptr + channel, mask=valid).to(tl.float32)
    values = tl.load(x_ptr + offsets, mask=valid).to(tl.float32)
    sine = tl.sin(tl.exp(alpha) * values)
    result = values + sine * sine / (tl.exp(beta) + _EPSILON)
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=valid,
    )
