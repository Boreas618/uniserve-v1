"""Row-scaled E4M3 gating: the Triton source of the ``act_and_mul_fp8`` operator.

One program owns one complete gate and value row, so the row's maximum is a
single block reduction and every row carries one E4M3 dequantization scale. The
activation evaluates in FP32; ``stepwise`` rounds the activated gate and the
product to the input type before encoding, which is the eager expression's
rounding rather than the kernel's default evaluation.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/activation.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Largest finite float8_e4m3fn magnitude, and the epsilon that keeps the scale of
# an all-zero row positive so the division stays finite.
FP8_MAX = tl.constexpr(448.0)
FP8_SCALE_EPS = tl.constexpr(1.0e-12)
# sqrt(8 / pi) and sqrt(1 / 2): constants of the two GELU forms.
_SQRT_8_OVER_PI = tl.constexpr(1.5957691216057308)
_SQRT_HALF = tl.constexpr(0.7071067811865476)


@triton.jit
def _activate(gate, ACTIVATION: tl.constexpr):  # noqa: N803
    """Apply the named gate activation to an FP32 tile."""
    if ACTIVATION == "silu":
        return gate / (1.0 + tl.exp(-gate))
    elif ACTIVATION == "gelu_tanh":
        # 0.5 * x * (1 + tanh(u)) equals x * sigmoid(2u) for
        # u = sqrt(2 / pi) * (x + 0.044715 x^3); the sigmoid form needs
        # one exponential and saturates to exact 0 and x at both ends.
        inner = _SQRT_8_OVER_PI * (gate + 0.044715 * gate * gate * gate)
        return gate / (1.0 + tl.exp(-inner))
    else:
        return 0.5 * gate * (1.0 + libdevice.erf(gate * _SQRT_HALF))


@triton.jit(do_not_specialize=["gate_row_stride", "value_row_stride"])
def _act_and_mul_fp8_kernel(
    gate_ptr,
    value_ptr,
    out_ptr,
    scale_ptr,
    gate_row_stride,
    value_row_stride,
    n_cols: tl.constexpr,  # noqa: N803
    ACTIVATION: tl.constexpr,  # noqa: N803
    block: tl.constexpr,
    stepwise: tl.constexpr,
):
    """Apply gating and emit its row-scaled E4M3 output.

    One program owns one complete gate and value row, with ``block`` at least
    ``n_cols``, so the row maximum is a single block reduction. Row offsets are
    int64, so one launch covers every row of inputs past ``2**31`` elements.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block)
    mask = cols < n_cols

    gate = tl.load(
        gate_ptr + row * gate_row_stride + cols, mask=mask, other=0.0
    ).to(tl.float32)
    value = tl.load(
        value_ptr + row * value_row_stride + cols, mask=mask, other=0.0
    ).to(tl.float32)
    activated = _activate(gate, ACTIVATION)
    if stepwise:
        dtype = gate_ptr.dtype.element_ty
        activated = activated.to(dtype).to(tl.float32)
        output = (activated * value).to(dtype).to(tl.float32)
    else:
        output = activated * value

    # One E4M3 dequantization scale per row from the products under the selected
    # rounding recipe; stored codes decode as code * scale.
    output_fp32 = tl.where(mask, output, 0.0)
    scale = tl.maximum(tl.max(tl.abs(output_fp32), axis=0), FP8_SCALE_EPS)
    scale = scale / FP8_MAX
    quantized = tl.maximum(
        tl.minimum(output_fp32 / scale, FP8_MAX),
        -FP8_MAX,
    )
    tl.store(out_ptr + row * n_cols + cols, quantized, mask=mask)
    tl.store(scale_ptr + row, scale)
