"""Selected-token log probabilities and ranks: the Triton source of the
``sampler_logprobs`` operator.

One program scores one output row: it reads the logits row that produced the
row's token, takes the row's maximum and the log-sum-exp of
``exp(logit - max)``, and then stores the selected token's log probability
(``logit - logsumexp``) and its competition rank (one plus the number of
strictly larger scores, so equal scores share a rank). Three passes over one
row, each a streaming reduction, so the vocabulary may be any size.

The numbers are the ones UniServe's tensor composition defined
(``uniserve/sampling/logprobs.py``): ``log_softmax`` of the logits, the
selected token's score, and competition ranks computed over the whole row. The
top-k logprob fields of that composition need a device top-k selection this
operator does not have yet (the same selection the stochastic sampler needs), so
this kernel reports the selected token only.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["vocab", "logits_stride", "rows_stride", "tokens_stride"])
def _logprobs_kernel(
    logits_ptr,
    rows_ptr,
    tokens_ptr,
    scores_ptr,
    ranks_ptr,
    vocab,
    logits_stride,
    rows_stride,
    tokens_stride,
    block: tl.constexpr,
):
    """Score one output row's selected token.

    ``logits`` is ``[rows, vocab]`` float32, ``rows`` the per-output-row logits
    row, ``tokens`` the per-output-row selected token, and ``scores``/``ranks``
    the per-output-row FP32 log probability and int32 competition rank. Grid:
    (output rows,).
    """
    output = tl.program_id(0).to(tl.int64)
    source = tl.load(rows_ptr + output * rows_stride).to(tl.int64)
    token = tl.load(tokens_ptr + output * tokens_stride).to(tl.int64)
    row = logits_ptr + source * logits_stride

    # Pass one: the row's maximum, which keeps the exponentials in range.
    running_max = tl.full((), -float("inf"), tl.float32)
    for start in range(0, vocab, block):
        columns = start + tl.arange(0, block)
        values = tl.load(row + columns, mask=columns < vocab, other=-float("inf"))
        running_max = tl.maximum(running_max, tl.max(values.to(tl.float32), axis=0))

    # Pass two: the log-sum-exp normalizer.
    total = tl.zeros((), dtype=tl.float32)
    for start in range(0, vocab, block):
        columns = start + tl.arange(0, block)
        values = tl.load(row + columns, mask=columns < vocab, other=-float("inf"))
        total += tl.sum(tl.exp(values.to(tl.float32) - running_max), axis=0)
    logsumexp = running_max + tl.log(total)

    # Pass three: the selected score and its competition rank over the row.
    selected = tl.load(row + token).to(tl.float32)
    larger = tl.zeros((), dtype=tl.int32)
    for start in range(0, vocab, block):
        columns = start + tl.arange(0, block)
        values = tl.load(row + columns, mask=columns < vocab, other=-float("inf"))
        larger += tl.sum((values.to(tl.float32) > selected).to(tl.int32), axis=0)

    tl.store(scores_ptr + output, selected - logsumexp)
    tl.store(ranks_ptr + output, larger + 1)
