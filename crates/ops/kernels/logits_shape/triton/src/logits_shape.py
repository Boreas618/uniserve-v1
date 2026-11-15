"""Logit adjustments, penalties and temperature: the Triton source of the
``logits_shape`` operator.

One program adjusts, penalizes, and optionally temperature-scales, the ``block``
columns of one row's logits, in vLLM's order: first the row's adjustments - its
slot's bias row (the request's logit bias, ``-inf`` outside its allowed set)
when the row's flag names it, then ``-inf`` at every token its ban mask sets
(bad words, the ``min_tokens`` floor) - then the repetition, frequency and
presence penalties over the slot's committed token counts plus the row's extra
tokens, then division by the row's temperature. Every FP32 operation is a
separately rounded intrinsic, so the result equals the tensor composition this
kernel replaces bit for bit; a row with no adjustment selects its own logits
unchanged, so it is shaped exactly as it would be without them.

A ``block`` that is larger than a program can hold in registers is walked in
``STEP`` wide tiles, so one module serves the descriptor's block whatever its
size.

The kernel body imports nothing but Triton: the parameter table, the torch
types and the allocation stay with the operator's Rust side.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from uniserve_kernels import dependent_launch


@triton.jit(do_not_specialize=["rows", "vocabulary"])
def _logits_shape_kernel(
    logits_ptr,
    counts_ptr,
    slots_ptr,
    parameters_ptr,
    extras_ptr,
    bias_ptr,
    adjustments_ptr,
    bans_ptr,
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
    nothing). ``adjustments[r]`` is ``(biased, mask)``: a nonzero ``biased``
    adds row ``slots[r]`` of the ``[slots, vocabulary]`` bias table, and a mask
    index of zero or more bans every token whose bit is set in that row of the
    ``[masks, ceil(vocabulary / 32)]`` ban table. Grid:
    ``[rows, ceil(vocabulary / BLOCK), 1]``.
    """
    dependent_launch.prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    parameters = parameters_ptr + row * 8
    temperature = tl.load(parameters + 0)
    repetition = tl.load(parameters + 4)
    frequency = tl.load(parameters + 5)
    presence = tl.load(parameters + 6)
    # A padded row names slot -1 and reads no bias row, whatever its flag says.
    biased = (tl.load(adjustments_ptr + row * 2) != 0) & (slot >= 0)
    mask_row = tl.load(adjustments_ptr + row * 2 + 1).to(tl.int64)
    banning = mask_row >= 0
    words = (vocabulary + 31) // 32

    first = tl.program_id(1).to(tl.int64) * BLOCK
    last = tl.minimum(first + BLOCK, vocabulary)
    for start in range(first, last, STEP):
        columns = start + tl.arange(0, STEP)
        mask = columns < vocabulary

        current = tl.load(
            logits_ptr + row * vocabulary + columns, mask=mask, other=0.0
        ).to(tl.float32)
        # The adjustments select, never round, for a row without them: the
        # loads are masked off and the row keeps its own logits.
        bias = tl.load(
            bias_ptr + slot * vocabulary + columns,
            mask=mask & biased,
            other=0.0,
        )
        current = tl.where(biased, libdevice.add_rn(current, bias), current)
        word = tl.load(
            bans_ptr + mask_row * words + columns // 32,
            mask=mask & banning,
            other=0,
        )
        banned = ((word >> (columns % 32).to(tl.int32)) & 1) != 0
        current = tl.where(banned, float("-inf"), current)
        # A padded row names slot -1, which addresses memory before the table: it reads no
        # counts, and its logits are discarded.
        counts = tl.load(
            counts_ptr + slot * vocabulary + columns,
            mask=mask & (slot >= 0),
            other=0,
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


#: The most lanes a program holds in registers in one step: a wider block is
#: walked in more steps, not in a wider tile.
LANES = 4096


def derive(fields, parameters, target):
    """The lanes of one step through the block, for the launch table.

    ``step`` is the descriptor's block up to :data:`LANES`.
    """
    return {"step": min(fields["block"], LANES)}
