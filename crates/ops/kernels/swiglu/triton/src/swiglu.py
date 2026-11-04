"""SwiGLU: the Triton source of the ``swiglu`` operator.

Each input is a channel view whose rows are one stride apart and whose last axis
is unit-strided; each half has its own bias; the output is contiguous. Programs
tile the output and, on request, each records the largest magnitude of its
outputs after rounding to the output dtype.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/activation.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def _swiglu_kernel(
    value_ptr,
    gate_ptr,
    value_bias_ptr,
    gate_bias_ptr,
    output_ptr,
    partials_ptr,
    elements,
    width: tl.constexpr,
    value_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    HAS_VALUE_BIAS: tl.constexpr,  # noqa: N803
    HAS_GATE_BIAS: tl.constexpr,  # noqa: N803
    RETURN_ABSMAX: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Evaluate SwiGLU from separate value and gate channel views.

    The biases and the magnitude buffer are operands of the launch; the ones
    whose flag is off are never dereferenced.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offsets = offsets.to(tl.int64)
    mask = offsets < elements
    row = offsets // width
    column = offsets - row * width

    value = tl.load(
        value_ptr + row * value_row_stride + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + row * gate_row_stride + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_VALUE_BIAS:
        value += tl.load(value_bias_ptr + column, mask=mask, other=0.0).to(
            tl.float32
        )
    if HAS_GATE_BIAS:
        gate += tl.load(gate_bias_ptr + column, mask=mask, other=0.0).to(
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
