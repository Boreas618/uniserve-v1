"""The per-channel affine map: the Triton source of the ``channel_affine`` operator.

```text
out[b, c, p] = x[b, c, p] · scale[c] + shift[c]
```

over a contiguous ``[batch, channels, positions]`` view; ``scale`` and ``shift`` are one value per
channel. A latent restore (``latents * std + mean``) is this map with the channel on any axis: the
caller folds the axes before the channel into ``batch`` and the ones after it into ``positions``.

**Two roundings, never one.** The reference multiplies and then adds as two tensor operations, so
the product rounds to FP32 before the shift is added. Triton turns ``a * b + c`` into a fused
multiply-add by default, which rounds once and differs in the last bit; the product and the sum
here are each one round-to-nearest ``mul.rn.f32`` and ``add.rn.f32``: the package compiles with
``enable_fp_fusion = false``, so Triton does not contract them, and the store rounds once more
into the output's element type.

One element of ``out`` per thread; a flat index gives the element's channel.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch


@triton.jit
def _channel_affine_kernel(
    x,
    scale,
    shift,
    out,
    channels,
    positions,
    elements,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of ``x · scale + shift`` over a contiguous ``[batch, channels, positions]``.

    Grid: ``ceil(elements / BLOCK)`` programs, ``elements = batch · channels · positions``.
    """
    dependent_launch.prologue(PDL)

    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    channel = (offsets // positions.to(tl.int64)) % channels.to(tl.int64)

    value = tl.load(x + offsets, mask=valid).to(tl.float32)
    factor = tl.load(scale + channel, mask=valid).to(tl.float32)
    offset = tl.load(shift + channel, mask=valid).to(tl.float32)
    restored = value * factor + offset
    tl.store(out + offsets, restored.to(out.dtype.element_ty), mask=valid)
