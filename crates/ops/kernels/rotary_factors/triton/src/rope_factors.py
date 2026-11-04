"""Rotary factor tables: the Triton source of the ``rope_factors`` operator.

One program stores a block of scaled cosine and sine factors for strided
positions: ``cos/sin(position * frequency) * scale`` with full-range FP32
trigonometry. Tables are compact, one ``[tokens, rotated / 2]`` row per rotary
axis, shared by every head of a token.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/rope.py``), unchanged except that it
imports nothing but Triton: the shape and alias checks, the output allocation
and the torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["total"])
def _rotary_factors_kernel(
    positions,
    frequencies,
    cosine,
    sine,
    total: tl.int64,
    width: tl.constexpr,
    shape: tl.constexpr,
    strides: tl.constexpr,
    frequency_stride: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store scaled cosine and sine factors for strided positions.

    ``offsets`` flatten the contiguous ``[*positions.shape, width]`` outputs.
    ``shape`` holds the trailing position extents and ``strides`` every
    position stride, both as compile-time tuples.
    """
    _pdl_prologue(PDL)

    offsets = tl.program_id(0) * block + tl.arange(0, block)
    rows = offsets // width
    position_offsets = tl.full((block,), 0, tl.int64)

    # The leading extent only bounds the launch through ``total``. The trailing
    # extents and strides are compile-time constants, so a new leading extent
    # with unchanged strides reuses the compiled kernel.
    for axis in tl.static_range(len(shape) - 1, -1, -1):
        position_offsets += (rows % shape[axis]) * strides[axis + 1]
        rows //= shape[axis]
    position_offsets += rows * strides[0] if len(strides) else 0

    position = tl.load(positions + position_offsets, offsets < total, other=0).to(
        tl.float32
    )
    frequency = tl.load(frequencies + (offsets % width) * frequency_stride).to(
        tl.float32
    )

    # Match the public FP32 phase domain, including range reduction for large or
    # negative positions. Approximate sin/cos instructions do not provide that
    # domain; libdevice uses the CUDA numerical functions.
    phase = position * frequency
    factor = tl.full((), scale, tl.float32)
    tl.store(cosine + offsets, libdevice.cos(phase) * factor, offsets < total)
    tl.store(sine + offsets, libdevice.sin(phase) * factor, offsets < total)
