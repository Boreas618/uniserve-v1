"""Last-row gathering: the Triton source of the ``logits_gather`` operator.

One program copies a block of elements of the selected rows of the hidden state
into a dense ``[count, columns]`` matrix, which the vocabulary head's GEMM then
multiplies. The gather is one launch, so a captured decode step compacts its
rows on the device without a host plan.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):  # noqa: N803
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["count", "columns"])
def _logits_gather_kernel(
    hidden_ptr,
    rows_ptr,
    out_ptr,
    count,
    columns,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Copy the selected rows of ``hidden`` into dense rows of ``out``.

    ``rows`` holds one row index per output row (each request's last position),
    ``hidden`` is ``[tokens, columns]`` and ``out`` ``[count, columns]``. Grid:
    ``ceil(count * columns / BLOCK)``.
    """
    _pdl_prologue(PDL)

    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count * columns
    row = offsets // columns
    column = offsets % columns

    # A negative index is a row that names nothing: it reads the first row, which is padding in
    # every table gathered by a request's slot, and a hidden state's indices are never negative.
    source = tl.maximum(tl.load(rows_ptr + row, mask=mask, other=0).to(tl.int64), 0)
    value = tl.load(hidden_ptr + source * columns + column, mask=mask)
    tl.store(out_ptr + offsets, value, mask=mask)
