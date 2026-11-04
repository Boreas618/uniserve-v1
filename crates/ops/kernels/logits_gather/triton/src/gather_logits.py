"""Last-row gathering: the Triton source of the ``gather_logits`` operator.

One program copies one block of one selected row of the hidden state into a
dense ``[requests, width]`` matrix, which the vocabulary head's GEMM then
multiplies. UniServe selected each request's last row with ATen indexing
(``uniserve/models/logits.py:62-104``); the gather is one launch here, so a
captured decode step compacts its rows on the device without a host plan.
"""

import triton
import triton.language as tl


@triton.jit
def _gather_logits_kernel(
    hidden_ptr,
    rows_ptr,
    out_ptr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    """Copy the selected rows of ``hidden`` into dense rows of ``out``.

    ``rows`` holds one row index per output row (each request's last position),
    ``hidden`` is ``[tokens, width]`` and ``out`` ``[rows, width]``. Grid:
    (rows, ceil(width / block)).
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    mask = columns < width

    source = tl.load(rows_ptr + row).to(tl.int64)
    value = tl.load(hidden_ptr + source * width + columns, mask=mask)
    tl.store(out_ptr + row * width + columns, value, mask=mask)
