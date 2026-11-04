"""Packed SwiGLU: the Triton source of the ``value_first_swiglu`` operator.

Programs tile the flattened contiguous ``[rows, width]`` output in ``block``
elements; input rows start ``input_row_stride`` elements apart and hold the value
half first, then the gate half. The optional bias is packed ``[value_bias,
gate_bias]`` like the input. With ``absmax``, each program also stores the
absolute maximum of its outputs, after rounding to the output dtype.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/activation.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements", "input_row_stride"])
def _value_first_swiglu_kernel(
    value_gate_ptr,
    bias_ptr,
    output_ptr,
    partials_ptr,
    elements,
    input_row_stride,
    width: tl.constexpr,
    HAS_BIAS: tl.constexpr,  # noqa: N803
    RETURN_ABSMAX: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Evaluate ``value * silu(gate)`` over packed ``[value, gate]`` rows.

    The bias and the magnitude buffer are operands of the launch; the ones whose
    flag is off are never dereferenced.
    """
    # Flattened projections can exceed int32 offsets, so index in int64.
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offsets = offsets.to(tl.int64)
    mask = offsets < elements
    row = offsets // width
    column = offsets - row * width

    value = tl.load(
        value_gate_ptr + row * input_row_stride + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        value_gate_ptr + row * input_row_stride + width + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(bias_ptr + column, mask=mask, other=0.0).to(
            tl.float32
        )
        gate += tl.load(bias_ptr + width + column, mask=mask, other=0.0).to(
            tl.float32
        )

    output = (value * gate / (1.0 + tl.exp(-gate))).to(
        output_ptr.dtype.element_ty
    )
    tl.store(output_ptr + offsets, output, mask=mask)

    if RETURN_ABSMAX:
        partial = tl.max(
            tl.where(mask, tl.abs(output.to(tl.float32)), 0.0), axis=0
        )
        tl.store(partials_ptr + tl.program_id(0), partial)
