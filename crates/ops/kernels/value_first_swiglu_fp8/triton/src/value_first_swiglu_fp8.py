"""Row-scaled E4M3 SwiGLU: the Triton source of the ``value_first_swiglu_fp8`` operator.

One program owns one complete packed ``[value, gate]`` row, so the row's maximum
is a single block reduction and every row carries one E4M3 dequantization scale.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/activation.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl

# Largest finite float8_e4m3fn magnitude, and the epsilon that keeps the scale of
# an all-zero row positive so the division stays finite.
FP8_MAX = tl.constexpr(448.0)
FP8_SCALE_EPS = tl.constexpr(1.0e-12)


@triton.jit(do_not_specialize=["input_row_stride"])
def _value_first_swiglu_fp8_kernel(
    value_gate_ptr,
    output_ptr,
    output_scale_ptr,
    input_row_stride,
    width: tl.constexpr,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Evaluate one packed ``[value, gate]`` row and quantize it to E4M3.

    One program owns one complete row (``BLOCK >= width``) and stores its
    dequantization scale to ``output_scale_ptr[row]``.
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    mask = columns < width
    value = tl.load(
        value_gate_ptr + row * input_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        value_gate_ptr + row * input_row_stride + width + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activated_gate = gate / (1.0 + tl.exp(-gate))
    output = value * activated_gate

    output_fp32 = tl.where(mask, output.to(tl.float32), 0.0)
    output_scale = tl.maximum(
        tl.max(tl.abs(output_fp32), axis=0), FP8_SCALE_EPS
    )
    output_scale /= FP8_MAX
    quantized = tl.maximum(
        tl.minimum(output_fp32 / output_scale, FP8_MAX),
        -FP8_MAX,
    )
    tl.store(output_ptr + row * width + columns, quantized, mask=mask)
    tl.store(output_scale_ptr + row, output_scale)
