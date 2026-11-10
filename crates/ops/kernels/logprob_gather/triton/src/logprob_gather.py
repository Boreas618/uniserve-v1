"""The log-probability and the rank of one chosen token per row: the Triton
source of the ``logprob_gather`` operator.

One program reads one row of log-probabilities and stores the selected token's
entry and its one-based competition rank in the row: one plus the number of
entries strictly larger than the token's, so tied entries share a rank (the
rank vLLM reports, ``batched_count_greater_than``). The count walks the row in
``STEP`` wide tiles, so one module serves any row width.

A row whose token lies outside the row - a negative token names no position, as
a prompt row the batch does not score does - stores ``0.0`` and rank ``0``,
which no scored token reports.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["rows", "columns"])
def _logprob_gather_kernel(
    logprobabilities_ptr,
    tokens_ptr,
    out_ptr,
    ranks_ptr,
    rows,
    columns,
    STEP: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Store the selected token's log-probability and rank of one row per program.

    ``logprobabilities`` is ``[rows, columns]`` FP32, ``tokens`` the per-row
    selected token in the descriptor's id dtype, ``out`` ``[rows]`` FP32 and
    ``ranks`` ``[rows]`` i32. Grid: (rows,).
    """
    _pdl_prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    base = row * columns
    token = tl.load(tokens_ptr + row).to(tl.int64)
    inside = (token >= 0) & (token < columns)
    value = tl.load(logprobabilities_ptr + base + token, mask=inside, other=0.0)

    # The rank counts the entries strictly above the token's, so a NaN entry, which compares
    # false, counts nothing.
    above = tl.zeros((), dtype=tl.int32)
    for start in range(0, columns, STEP):
        offsets = start + tl.arange(0, STEP)
        entries = tl.load(
            logprobabilities_ptr + base + offsets,
            mask=offsets < columns,
            other=-float("inf"),
        )
        above += tl.sum((entries > value).to(tl.int32), axis=0)

    tl.store(out_ptr + row, tl.where(inside, value, 0.0))
    tl.store(ranks_ptr + row, tl.where(inside, above + 1, 0))
