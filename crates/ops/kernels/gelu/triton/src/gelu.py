"""The Gaussian error linear unit: the Triton source of the ``gelu`` operator.

PyTorch's two GELU forms, one program per block of the flattened row-major
``[rows, columns]`` input:

- ``exact``: ``0.5·x·(1 + erf(x/√2))``.
- ``tanh``: ``x·sigmooid(√(8/π)·(x + 0.044715·x³))``, written as one logistic
  rather than ``0.5·x·(1 + tanh(u))`` because the logistic needs one exponential
  and saturates exactly at both ends.

The formulas are the reference's (``uniserve_kernels/src/uniserve_kernels/
activation.py``, ``_activate``), which evaluates them in FP32 and rounds once
into the output: the exact form's error function and the tanh form's exponential
are each accurate to about one unit in the last place of their own result, and
the operator's tolerance covers that unit over ``|x| ≤ 8``.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

#: ``sqrt(8 / pi)`` and ``sqrt(1 / 2)``: the two forms' constants, instantiated so a
#: kernel may read them.
_SQRT_8_OVER_PI = tl.constexpr(1.5957691216057308)
_SQRT_HALF = tl.constexpr(0.7071067811865476)


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    elements,
    columns,
    pdl,
    APPROXIMATE: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one block of the activation of the contiguous row-major input.

    Grid: ``ceil(elements / BLOCK)`` programs. ``columns`` is the row width the
    descriptor was compiled for; a block covers consecutive elements of the
    flattened rows, so it is not read.
    """
    _pdl_prologue(PDL)

    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if APPROXIMATE == "tanh":
        inner = _SQRT_8_OVER_PI * (values + 0.044715 * values * values * values)
        result = values / (1.0 + tl.exp(-inner))
    else:
        result = 0.5 * values * (1.0 + libdevice.erf(values * _SQRT_HALF))
    tl.store(
        out_ptr + offsets,
        result.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask,
    )
