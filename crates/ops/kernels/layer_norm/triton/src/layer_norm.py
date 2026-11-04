"""Row-wise layer normalization: the Triton source of the ``layer_norm`` operator.

One program normalizes one row, loading its whole row before storing it, so the output may alias the
input:

```text
out = (x − mean(x)) · rsqrt(var(x) + eps) · weight + bias
```

with the mean and the variance reduced over the row in FP32 and the stored value rounded once into
the output dtype. The variance is the mean square of the centered row, so the two reductions are
over the same lanes and the normalization needs no second pass over memory.

This is the arithmetic of `torch.nn.functional.layer_norm`, whose reduction order this row-per-program
form does not reproduce: the operator's tolerance is the sum's own rounding over both reductions
(`2 · columns` FP32 roundings) rather than the exact equality a copied kernel would carry.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["columns", "eps"])
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    columns,
    eps,
    has_bias,
    pdl,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,  # noqa: N803
    PDL: tl.constexpr,  # noqa: N803
):
    """Store one normalized row.

    Grid: one program per row. ``BLOCK`` covers the row width; the lanes past it are masked and
    center to zero, so they contribute nothing to either reduction. ``bias_ptr`` carries the
    weight's address when the descriptor declares no bias.
    """
    _pdl_prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    columns = columns.to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    live = lanes < columns
    offsets = row * columns + lanes

    values = tl.load(x_ptr + offsets, mask=live, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(live, values - mean, 0.0)
    inverse = tl.rsqrt(tl.sum(centered * centered, axis=0) / columns + eps)
    weight = tl.load(weight_ptr + lanes, mask=live, other=0.0).to(tl.float32)
    normalized = centered * inverse * weight
    if HAS_BIAS:
        normalized += tl.load(bias_ptr + lanes, mask=live, other=0.0).to(
            tl.float32
        )
    tl.store(
        out_ptr + offsets,
        normalized.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=live,
    )
