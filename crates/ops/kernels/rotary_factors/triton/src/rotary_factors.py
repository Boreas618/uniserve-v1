"""Rotary factor tables: the Triton source of the ``rotary_factors`` operator.

One program stores a block of scaled cosine and sine factors, one
``[tokens, columns]`` table per output: ``cos(position * frequency) * scale``
with full-range FP32 trigonometry. Tables are compact — one row per token, one
column per frequency — and shared by every head of a token.

A multimodal transformer reads several position axes: each axis owns a section
of the columns, and a column's angle comes from the coordinate of the axis its
section belongs to (M-RoPE). Two assignments of columns to axes exist, and both
are compile-time facts of the descriptor's sections:

- the contiguous cut gives axis ``a`` the next ``sections[a] / 2`` frequencies
  in axis order;
- the interleaved form cycles the axes over the whole spectrum: frequency ``j``
  reads axis ``j % n`` while ``j < sections[axis] · n`` and axis 0 otherwise.

The positions of a sectioned call are ``[axes, tokens]``, contiguous, so axis
``a``'s coordinate of a token starts at ``a * tokens``. A one-dimensional call's
positions are ``[tokens]``, and its key (and so its arguments) are what they
always were.

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


@triton.jit
def _factors(
    positions,
    frequencies,
    cosine,
    sine,
    total,
    columns,
    scale,
    AXES: tl.constexpr,
    HALVES: tl.constexpr,
    CUTS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store scaled cosine and sine factors for one block of table elements.

    ``positions`` holds one coordinate per slot: one per token for a
    one-dimensional call, ``[AXES, tokens]`` for a sectioned one. ``frequencies``
    holds the ``columns`` angles per row, ``cosine``/``sine`` the
    ``[tokens, columns]`` tables. Grid: ``ceil(total / BLOCK)``.
    """
    _pdl_prologue(PDL)

    elements = total.to(tl.int64)
    width = columns.to(tl.int64)
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    tokens = elements // width
    token = offsets // width
    column = (offsets % width).to(tl.int32)
    frequency = tl.load(frequencies + column, mask=mask, other=0).to(tl.float32)

    if AXES == 1:
        position = tl.load(positions + token, mask=mask, other=0).to(tl.float32)
    else:
        # The axis a column reads: a threshold count for the contiguous cut, a
        # cycle for the interleaved one.
        if INTERLEAVED:
            cycled = column % AXES
            half = tl.full((BLOCK,), HALVES[0], tl.int32)
            for axis in tl.static_range(1, AXES):
                half = tl.where(cycled == axis, HALVES[axis], half)
            axis = tl.where(column < half * AXES, cycled, 0)
        else:
            axis = tl.zeros((BLOCK,), tl.int32)
            for cut in tl.static_range(1, AXES):
                axis += (column >= CUTS[cut]).to(tl.int32)
        # One coordinate per axis: the axis's own row of the position table.
        position = tl.zeros((BLOCK,), tl.float32)
        for candidate in tl.static_range(AXES):
            taken = axis == candidate
            row = positions + candidate * tokens + token
            position = tl.where(
                taken,
                tl.load(row, mask=mask & taken, other=0).to(tl.float32),
                position,
            )

    # Match the public FP32 phase domain, including range reduction for large
    # or negative positions. Approximate sin/cos instructions do not provide
    # that domain; libdevice uses the CUDA numerical functions.
    phase = position * frequency
    tl.store(cosine + offsets, libdevice.cos(phase) * scale, mask=mask)
    tl.store(sine + offsets, libdevice.sin(phase) * scale, mask=mask)


# The two programs a call's arguments can name: a one-dimensional rotation's
# call ends at `scale`, and a sectioned one carries the axes and the width each
# one owns. A Triton program's parameters are its call's words in order, so the
# two forms are two kernels over one body.
@triton.jit(do_not_specialize=["total", "columns"])
def _rotary_factors_kernel(
    positions,
    frequencies,
    cosine,
    sine,
    total,
    columns,
    scale,
    AXES: tl.constexpr,
    HALVES: tl.constexpr,
    CUTS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """The one-dimensional program: one position per token."""
    _factors(
        positions,
        frequencies,
        cosine,
        sine,
        total,
        columns,
        scale,
        AXES,
        HALVES,
        CUTS,
        INTERLEAVED,
        BLOCK,
        PDL,
    )


@triton.jit(do_not_specialize=["total", "columns"])
def _rotary_factors_sectioned_kernel(
    positions,
    frequencies,
    cosine,
    sine,
    total,
    columns,
    scale,
    axes,
    section_0,
    section_1,
    section_2,
    section_3,
    AXES: tl.constexpr,
    HALVES: tl.constexpr,
    CUTS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """The sectioned program: one coordinate per axis per token.

    The call's `axes` and section widths are the words the operator always
    passes; the module compiled its own from the key, so they are taken for the
    call's order and not read.
    """
    _factors(
        positions,
        frequencies,
        cosine,
        sine,
        total,
        columns,
        scale,
        AXES,
        HALVES,
        CUTS,
        INTERLEAVED,
        BLOCK,
        PDL,
    )
