"""Rotary factor tables: the Triton source of the ``rotary_factors`` operator.

One program stores a block of scaled cosine and sine factors:
``cos(position * frequency) * scale`` with full-range FP32 trigonometry. Tables
are compact, one ``[tokens, columns]`` row per rotary axis, shared by every head
of a token.

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


@triton.jit(do_not_specialize=["total", "columns"])
def _rotary_factors_kernel(
    positions,
    frequencies,
    cosine,
    sine,
    total,
    columns,
    scale,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store scaled cosine and sine factors for one block of table elements.

    ``positions`` holds one position per output element, ``frequencies`` the
    ``columns`` angles per row, ``cosine``/``sine`` the ``[tokens, columns]``
    tables. Grid: ``ceil(total / BLOCK)``.
    """
    _pdl_prologue(PDL)

    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    position = tl.load(positions + offsets, mask=mask, other=0).to(tl.float32)
    frequency = tl.load(frequencies + offsets % columns).to(tl.float32)

    # Match the public FP32 phase domain, including range reduction for large
    # or negative positions. Approximate sin/cos instructions do not provide
    # that domain; libdevice uses the CUDA numerical functions.
    phase = position * frequency
    tl.store(cosine + offsets, libdevice.cos(phase) * scale, mask=mask)
    tl.store(sine + offsets, libdevice.sin(phase) * scale, mask=mask)
