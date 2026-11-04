"""Indexed RMS modulation: the Triton source of the ``modulated_rms_norm`` operator.

One program normalizes one contiguous activation row with FP32 statistics and rounds every
intermediate to the activation dtype, as the eager expression does. The descriptor bakes the row
width, the lane block (the width rounded up to a power of two), the epsilon, and whether the block
folds a gated update in, writes an E4M3 output with a per-row scale, and retains the normalized
source; the row count comes from the grid. The three modulation row strides are launch arguments,
because one parameter table serves many activation rows and its layout is the model's.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/norm/modulation.py``), unchanged except that it imports
nothing but Triton: the launch-time wrapper, its operand checks and its torch types stay with the
operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "gate_row_stride",
        "shift_row_stride",
        "scale_row_stride",
    ]
)
def _modulated_rms_kernel(
    hidden_ptr,
    update_ptr,
    gate_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    row_indices_ptr,
    output_ptr,
    output_scale_ptr,
    retain_ptr,
    gate_row_stride,
    shift_row_stride,
    scale_row_stride,
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    EPS: tl.constexpr,  # noqa: N803
    HAS_UPDATE: tl.constexpr,  # noqa: N803
    FP8_OUTPUT: tl.constexpr,  # noqa: N803
    RETAIN: tl.constexpr,  # noqa: N803
):
    """Normalize, modulate and store one activation row per program.

    ``BLOCK`` is ``WIDTH`` rounded up to a power of two, and masked lanes load zeros so they add
    nothing to the mean square. ``modulation_row`` selects this row's shift, scale and gate from
    rows ``*_row_stride`` elements apart. Every pointer is an operand of the launch; the ones whose
    flag is off are never dereferenced.
    """
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    offsets = row * WIDTH + columns
    modulation_row = tl.load(row_indices_ptr + row)
    # The activation dtype, which every intermediate rounds to.
    activation = hidden_ptr.dtype.element_ty

    # value: one [WIDTH] activation row, accumulated in FP32.
    value = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if HAS_UPDATE:
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = tl.load(
            gate_ptr + modulation_row * gate_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # The gated update rounds, then the residual sum rounds, and the
        # normalization reads that rounded residual, which ``update``
        # receives.
        gated = (gate * update).to(activation).to(tl.float32)
        value = (value + gated).to(activation).to(tl.float32)
        tl.store(update_ptr + offsets, value, mask=mask)
    if RETAIN:
        # Copy the normalized row's source (the gated sum when HAS_UPDATE)
        # for the caller in the same pass that reads it.
        tl.store(retain_ptr + offsets, value, mask=mask)

    mean_square = tl.sum(value * value, axis=0) / WIDTH
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(
        shift_ptr + modulation_row * shift_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + modulation_row * scale_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    # Eager order: the weighted normalization rounds once, then
    # ``1 + scale``, its product and the shifted sum each round.
    normalized = value * tl.rsqrt(mean_square + EPS) * weight
    normalized = normalized.to(activation).to(tl.float32)
    factor = (1.0 + scale).to(activation).to(tl.float32)
    output = (normalized * factor).to(activation).to(tl.float32)
    output = (output + shift).to(activation).to(tl.float32)

    if FP8_OUTPUT:
        # Per-row E4M3 dequantization scale from the row's absolute max,
        # clamped away from zero so the division stays well-defined. 448
        # is the largest finite E4M3 magnitude.
        output = tl.where(mask, output, 0.0)
        output_scale = (
            tl.maximum(tl.max(tl.abs(output), axis=0), 1.0e-12) / 448.0
        )
        output = tl.minimum(
            tl.maximum(output / output_scale, -448.0), 448.0
        )
        tl.store(output_scale_ptr + row, output_scale)

    tl.store(output_ptr + offsets, output, mask=mask)
