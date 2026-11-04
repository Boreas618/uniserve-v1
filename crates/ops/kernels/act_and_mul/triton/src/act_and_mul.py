"""Gated activation: the Triton source of the ``act_and_mul`` operator.

Programs tile the flattened contiguous output, so no row width leaves a
mostly-masked tail program. ``gate`` and ``value`` are arbitrary views whose
rows are each one stride apart and unit-strided per channel; the activation
evaluates in FP32 and the product rounds to the output type.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/activation.py``), unchanged except that
it imports nothing but Triton: the launch-time wrapper, its operand checks and
its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

#: sqrt(8 / pi) and sqrt(1 / 2): constants of the two GELU forms.
_SQRT_8_OVER_PI = tl.constexpr(1.5957691216057308)
_SQRT_HALF = tl.constexpr(0.7071067811865476)


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):  # noqa: N803
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


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


@triton.jit(
    do_not_specialize=["elements", "gate_row_stride", "value_row_stride"]
)
def _act_and_mul_kernel(
    gate_ptr,
    value_ptr,
    out_ptr,
    elements,
    gate_row_stride,
    value_row_stride,
    n_cols: tl.constexpr,  # noqa: N803
    ACTIVATION: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    WIDE: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
    stepwise: tl.constexpr,
):
    """Store ``act(gate) * value`` for one block of the flattened output.

    Grid: ``ceil(elements / BLOCK)`` programs over the contiguous
    ``[rows, n_cols]`` output. Gate rows start ``gate_row_stride`` and value
    rows ``value_row_stride`` elements apart. ``WIDE`` computes offsets in
    int64, which launches whose input or output spans ``2**31`` elements
    require. ``PDL`` launches run the dependent-launch prologue before the
    first load.
    """
    _pdl_prologue(PDL)

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        offsets = offsets.to(tl.int64)
    mask = offsets < elements
    row = offsets // n_cols
    column = offsets - row * n_cols

    gate = tl.load(
        gate_ptr + row * gate_row_stride + column, mask=mask, other=0.0
    ).to(tl.float32)
    value = tl.load(
        value_ptr + row * value_row_stride + column, mask=mask, other=0.0
    ).to(tl.float32)
    activated = _activate(gate, ACTIVATION)
    if stepwise:
        activated = activated.to(gate_ptr.dtype.element_ty).to(tl.float32)
    out = activated * value
    tl.store(out_ptr + offsets, out, mask=mask)
