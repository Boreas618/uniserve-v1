"""Log-softmax over a row: the Triton source of the ``log_softmax`` operator.

One program scores one row in two streaming passes: the row's maximum with the
log-sum-exp of ``exp(x - max)``, then ``x - logsumexp`` for every element. The
output is FP32 whatever the input dtype, which is what the sampling path reads.

The numbers are the ones UniServe's tensor composition defined
(``uniserve/sampling/logprobs.py``): ``log_softmax`` of the logits row.
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
def _log_softmax_kernel(
    x_ptr,
    out_ptr,
    columns,
    pdl,
    STEP: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Write the log-softmax of one row per program.

    ``x`` is ``[rows, columns]`` in the descriptor's dtype, ``out`` the same
    shape in FP32, ``columns`` the row width and ``pdl`` the launch's copy of the
    descriptor's flag. The row is walked in ``STEP`` wide tiles, so a descriptor
    tile larger than a program's registers costs more steps, not a bigger block.
    Grid: (rows,).
    """
    _pdl_prologue(PDL)

    row = tl.program_id(0).to(tl.int64)
    base = row * columns

    # Pass one: the row's maximum, which keeps the exponentials in range.
    running_max = tl.full((), -float("inf"), tl.float32)
    for start in range(0, columns, STEP):
        offsets = start + tl.arange(0, STEP)
        values = tl.load(x_ptr + base + offsets, mask=offsets < columns, other=-float("inf"))
        running_max = tl.maximum(running_max, tl.max(values.to(tl.float32), axis=0))

    # Pass two: the log-sum-exp normalizer, accumulated in FP32.
    total = tl.zeros((), dtype=tl.float32)
    for start in range(0, columns, STEP):
        offsets = start + tl.arange(0, STEP)
        values = tl.load(x_ptr + base + offsets, mask=offsets < columns, other=-float("inf"))
        total += tl.sum(tl.exp(values.to(tl.float32) - running_max), axis=0)
    logsumexp = running_max + tl.log(total)

    # Pass three: the log-probabilities.
    for start in range(0, columns, STEP):
        offsets = start + tl.arange(0, STEP)
        mask = offsets < columns
        values = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + base + offsets, values.to(tl.float32) - logsumexp, mask=mask)
