"""Softmax over the last axis: the Triton source of the ``softmax`` operator.

One program normalizes one row, loading its whole row before storing it, so the output may alias the
input:

```text
out = exp(x − max(x)) / Σ exp(x − max(x))
```

The maximum, the exponentials and the denominator are FP32 and the stored value is rounded once into
the output dtype. Subtracting the row's maximum keeps every exponential at or below one, so a row of
large values neither overflows nor loses the small terms.

This is the arithmetic of `torch.softmax`, whose reduction order this row-per-program form does not
reproduce: the operator's tolerance is the denominator's own rounding over the row (`columns` FP32
roundings), not exact equality with a copied kernel.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["columns"])
def _softmax_kernel(
    x_ptr,
    out_ptr,
    columns,
    pdl,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one normalized row.

    Grid: one program per row. ``BLOCK`` covers the row width; the lanes past it hold the row's
    smallest value, so they are neither the maximum nor a term of the denominator.
    """
    _pdl_prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    columns = columns.to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    live = lanes < columns
    offsets = row * columns + lanes

    values = tl.load(x_ptr + offsets, mask=live, other=-float("inf")).to(
        tl.float32
    )
    shifted = values - tl.max(values, axis=0)
    exponentials = tl.exp(shifted)
    total = tl.sum(exponentials, axis=0)
    tl.store(
        out_ptr + offsets,
        (exponentials / total).to(
            out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"
        ),
        mask=live,
    )
