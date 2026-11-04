"""Per-channel bias addition with an optional residual: the Triton source of the
``bias_add`` operator.

The reference adds a convolution's bias in a separate pass rather than inside the
convolution:

```text
out = (residual + residual_bias) + (values + bias)
```

with ``values`` and ``residual`` of ``[batch, channels, *spatial]`` and the two
biases one value per channel.

One element of ``out`` per thread: a flat index gives the batch, channel and
position of the element, so a program handles any storage whose spatial axes
merge into one strided axis with unit position stride — contiguous channel-first
and channels-last are both that shape read with different strides — and the two
strides travel as words. ``out`` is written contiguous, channel-first.

The additions are the reference's (``uniserve_kernels/src/uniserve_kernels/
bias.py``): each is one IEEE round-to-nearest FP32 addition, which the compiler
cannot contract or flush, and the store rounds once into the output dtype.
"""

import triton
import triton.language as tl


@triton.jit
def _add_rn(a, b):
    """One round-to-nearest FP32 addition, as a tensor addition computes it."""
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _bias_add_kernel(
    values,
    bias,
    residual,
    residual_bias,
    out,
    channels,
    batch,
    positions,
    stride_batch,
    stride_channel,
    residual_batch,
    residual_channel,
    pdl,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,  # noqa: N803
    HAS_RESIDUAL_BIAS: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of the bias addition of one ``[batch, channels, positions]``
    value.

    Grid: ``ceil(batch · channels · positions / BLOCK)`` programs over the
    contiguous output. An absent residual carries the values' address and, because
    the views agree, the values' strides, so one argument list serves both variants;
    ``HAS_RESIDUAL`` and ``HAS_RESIDUAL_BIAS`` say which additions the module makes.
    """
    _pdl_prologue(PDL)

    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    channels = channels.to(tl.int64)
    valid = offsets < batch * channels * positions
    position = offsets % positions
    rest = offsets // positions
    channel = rest % channels
    sample = rest // channels

    hidden = tl.load(
        values + sample * stride_batch + channel * stride_channel + position,
        mask=valid,
    )
    hidden = _add_rn(hidden, tl.load(bias + channel, mask=valid))
    if HAS_RESIDUAL:
        shortcut = tl.load(
            residual
            + sample * residual_batch
            + channel * residual_channel
            + position,
            mask=valid,
        )
        if HAS_RESIDUAL_BIAS:
            shortcut = _add_rn(shortcut, tl.load(residual_bias + channel, mask=valid))
        # The shortcut is the left operand, as in `shortcut + hidden`.
        hidden = _add_rn(shortcut, hidden)
    tl.store(out + offsets, hidden, mask=valid)
