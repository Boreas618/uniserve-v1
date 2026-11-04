"""Penalties and temperature: the Triton source of the ``sampler_shape``
operator.

One program penalizes, and optionally temperature-scales, one block of one
row's logits: repetition, frequency and presence penalties over the slot's
committed token counts plus the row's extra tokens, then division by the row's
temperature. Every FP32 operation is a separately rounded intrinsic, so the
result equals the tensor composition this kernel replaces bit for bit.

The kernel body is the one UniServe launched before this package existed
(``uniserve/sampling/top_k.py``), unchanged except that it imports nothing but
Triton: the parameter table, the torch types and the allocation stay with the
operator's Rust side.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(
    do_not_specialize=[
        "vocab",
        "logits_stride",
        "counts_stride",
        "params_stride",
        "extra_stride",
        "out_stride",
    ]
)
def _shape_kernel(
    logits_ptr,
    counts_ptr,
    slots_ptr,
    params_ptr,
    extra_ptr,
    out_ptr,
    vocab,
    logits_stride,
    counts_stride,
    params_stride,
    extra_stride,
    out_stride,
    block: tl.constexpr,
    extras: tl.constexpr,
    divide: tl.constexpr,
):
    """Penalize, and with ``divide`` scale, one block of one row's logits.

    The program of row ``r`` reads the counts of slot ``slots[r]``, plus one
    per occurrence of a column among the row's ``extras`` tokens (``-1``
    counts nothing).
    """
    row = tl.program_id(0)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    mask = columns < vocab

    current = tl.load(logits_ptr + row * logits_stride + columns, mask=mask).to(
        tl.float32
    )
    slot = tl.load(slots_ptr + row).to(tl.int64)
    counts = tl.load(
        counts_ptr + slot * counts_stride + columns, mask=mask, other=0
    ).to(tl.float32)
    for index in tl.static_range(extras):
        token = tl.load(extra_ptr + row * extra_stride + index)
        counts += (columns == token).to(tl.float32)
    parameters = params_ptr + row * params_stride
    temperature = tl.load(parameters + 0)
    repetition = tl.load(parameters + 4)
    frequency = tl.load(parameters + 5)
    presence = tl.load(parameters + 6)

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
    if divide:
        divisor = tl.where(temperature > 0.0, temperature, 1.0)
        penalized = libdevice.div_rn(penalized, divisor)
    tl.store(out_ptr + row * out_stride + columns, penalized, mask=mask)
