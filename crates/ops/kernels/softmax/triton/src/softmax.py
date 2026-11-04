"""Softmax over the last axis: the Triton source of the ``softmax`` operator.

One program normalizes one row, walking it in tiles of ``BLOCK`` lanes and keeping only a running
maximum and a running denominator, so a row may be far wider than one block:

```text
out = exp(x − max(x)) / Σ exp(x − max(x))
```

The first pass over the tiles finds the row's maximum and its denominator together - a tile that
raises the maximum rescales the terms already summed - and the second pass stores
``exp(x − max) / denominator`` tile by tile. The row is walked in tiles and never held whole, which
is what makes its width unbounded: the alternative, one block covering the whole row, asks for a
register per element, and a vocabulary-wide row of 151936 elements is a megabyte per program.

The maximum, the exponentials and the denominator are FP32 and the stored value is rounded once into
the output dtype. Subtracting the row's maximum keeps every exponential at or below one, so a row of
large values neither overflows nor loses the small terms. The output may not alias the input: the
second pass would then read values the first pass had not yet overwritten but the tile order would
make the result depend on it.

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
    """Normalize one row of ``columns`` elements.

    Grid: one program per row. ``BLOCK`` is the width of one tile of the row, not the width of the
    row: a row of ``columns`` elements is ``ceil(columns / BLOCK)`` tiles. The lanes of the last tile
    past ``columns`` hold the row's smallest value, so they are neither the maximum nor a term of the
    denominator.
    """
    _pdl_prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    columns = columns.to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    start = row * columns
    tiles = tl.cdiv(columns, BLOCK)

    # Pass one: the row's maximum and, at the same time, its denominator. The two are accumulated
    # together because a tile that raises the maximum rescales the terms already summed - the
    # running-total form of one reduction, which reads the row once instead of twice:
    #
    #     m' = max(m, tile's maximum)
    #     s' = s * exp(m - m') + Σ_tile exp(x - m')
    maximum = -float("inf")
    total = 0.0
    for tile in range(0, tiles):
        offsets = tile * BLOCK + lanes
        live = offsets < columns
        values = tl.load(x_ptr + start + offsets, mask=live, other=-float("inf")).to(
            tl.float32
        )
        raised = tl.maximum(maximum, tl.max(values, axis=0))
        total = total * tl.exp(maximum - raised) + tl.sum(
            tl.where(live, tl.exp(values - raised), 0.0), axis=0
        )
        maximum = raised

    # Pass two: the normalized value, rounded once into the output's type.
    for tile in range(0, tiles):
        offsets = tile * BLOCK + lanes
        live = offsets < columns
        values = tl.load(x_ptr + start + offsets, mask=live, other=-float("inf")).to(
            tl.float32
        )
        tl.store(
            out_ptr + start + offsets,
            (tl.exp(values - maximum) / total).to(
                out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"
            ),
            mask=live,
        )
