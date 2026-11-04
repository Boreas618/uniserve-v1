"""The plain sigmoid linear unit: the ``silu`` operator's Triton source.

One formula, one program: ``out = x / (1 + exp(−x))``. Unlike ``act_and_mul`` this operator
reads a single buffer and writes a single one — no gate to multiply by — and unlike ``gelu``
there is no approximation to choose between, so the descriptor carries the geometry and the
launch flag and nothing else.

The exponential and the division are evaluated in FP32 and the result is stored once into the
output's element type, which is what the operator's tolerance states: one rounding from FP32,
plus an absolute part covering the sigmoid's own error over the operator's value envelope.
"""

import triton
import triton.language as tl


@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    elements,
    columns,
    pdl,
    BLOCK: tl.constexpr,
):
    """Store one block of ``x / (1 + exp(−x))``.

    The grid covers the flattened tensor, and ``columns`` is the row width the operator was
    compiled for. ``pdl`` is the launch's flag and the module's metadata carries the decision.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = offsets < elements
    x = tl.load(x_ptr + offsets, mask=live, other=0.0).to(tl.float32)
    # A single rounding into the output's element type, rounding to nearest even.
    tl.store(
        out_ptr + offsets,
        (x / (1.0 + tl.exp(-x))).to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=live,
    )
