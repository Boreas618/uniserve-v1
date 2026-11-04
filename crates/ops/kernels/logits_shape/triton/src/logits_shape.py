"""Penalties and temperature: the Triton source of the ``logits_shape``
operator.

One program penalizes, and optionally temperature-scales, the ``block`` columns
of one row's logits: repetition, frequency and presence penalties over the
slot's committed token counts plus the row's extra tokens, then division by the
row's temperature. Every FP32 operation is a separately rounded intrinsic, so the
result equals the tensor composition this kernel replaces bit for bit.

A ``block`` that is larger than a program can hold in registers is walked in
``STEP`` wide tiles, so one module serves the descriptor's block whatever its
size.

The kernel body is the one UniServe launched before this package existed
(``uniserve/sampling/top_k.py``), unchanged except that it imports nothing but
Triton: the parameter table, the torch types and the allocation stay with the
operator's Rust side.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["rows", "vocabulary"])
def _logits_shape_kernel(
    logits_ptr,
    counts_ptr,
    slots_ptr,
    parameters_ptr,
    extras_ptr,
    out_ptr,
    rows,
    vocabulary,
    divide,
    BLOCK: tl.constexpr,
    STEP: tl.constexpr,
    EXTRAS: tl.constexpr,
    DIVIDE: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Penalize, and with ``DIVIDE`` scale, one block of one row's logits.

    The program of row ``r`` reads the counts of slot ``slots[r]``, plus one per
    occurrence of a column among the row's ``EXTRAS`` tokens (``-1`` counts
    nothing). Grid: ``[rows, ceil(vocabulary / BLOCK), 1]``.
    """
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    parameters = parameters_ptr + row * 8
    temperature = tl.load(parameters + 0)
    repetition = tl.load(parameters + 4)
    frequency = tl.load(parameters + 5)
    presence = tl.load(parameters + 6)

    first = tl.program_id(1).to(tl.int64) * BLOCK
    last = tl.minimum(first + BLOCK, vocabulary)
    for start in range(first, last, STEP):
        columns = start + tl.arange(0, STEP)
        mask = columns < vocabulary

        current = tl.load(
            logits_ptr + row * vocabulary + columns, mask=mask, other=0.0
        ).to(tl.float32)
        counts = tl.load(
            counts_ptr + slot * vocabulary + columns, mask=mask, other=0
        ).to(tl.float32)
        for index in tl.static_range(EXTRAS):
            token = tl.load(extras_ptr + row * EXTRAS + index)
            counts += (columns == token).to(tl.float32)

        repeated = tl.where(
            current > 0.0,
            libdevice.div_rn(current, repetition),
            libdevice.mul_rn(current, repetition),
        )
        adjusted = libdevice.sub_rn(
            libdevice.sub_rn(repeated, libdevice.mul_rn(frequency, counts)),
            presence,
        )
        apply = (counts > 0.0) & (current != float("-inf"))
        penalized = tl.where(apply, adjusted, current)
        if DIVIDE:
            divisor = tl.where(temperature > 0.0, temperature, 1.0)
            penalized = libdevice.div_rn(penalized, divisor)
        tl.store(out_ptr + row * vocabulary + columns, penalized, mask=mask)
