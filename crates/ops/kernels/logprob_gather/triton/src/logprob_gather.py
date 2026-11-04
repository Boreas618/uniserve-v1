"""The log-probability of one chosen token per row: the Triton source of the
``logprob_gather`` operator.

One program reads one row of log-probabilities and stores the selected token's
entry. It moves one float per row, so the row width only bounds the index it
reads.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "columns"])
def _logprob_gather_kernel(
    logprobabilities_ptr,
    tokens_ptr,
    out_ptr,
    rows,
    columns,
):
    """Store the selected token's log-probability of one row per program.

    ``logprobabilities`` is ``[rows, columns]`` FP32, ``tokens`` the per-row
    selected token in the descriptor's id dtype and ``out`` ``[rows]`` FP32.
    Grid: (rows,).
    """
    row = tl.program_id(0).to(tl.int64)
    token = tl.load(tokens_ptr + row).to(tl.int64)
    value = tl.load(logprobabilities_ptr + row * columns + token)
    tl.store(out_ptr + row, value)
