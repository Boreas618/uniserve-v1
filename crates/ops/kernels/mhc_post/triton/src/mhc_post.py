"""An output written back into an mHC residual: the source of ``mhc_post``.

For each token, each of the ``STREAMS`` streams ``k`` becomes::

    h'_k = round(post_k y + sum_j comb[j, k] h_j)

with the sum over the read streams ``j`` formed first and the post term
added to it, in FP32, rounded once into the streams' type, as the reference
evaluates it. One program writes ``BLOCK`` columns of every stream of one
token.

The kernel body imports nothing but Triton: the launch-time wrapper, its
operand checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _mhc_post_kernel(
    streams_ptr,
    output_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    STREAMS: tl.constexpr,  # noqa: N803
    HIDDEN: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """One token's columns; grid ``(tokens, cdiv(HIDDEN, BLOCK))``.

    ``streams`` and ``out`` are ``[tokens, STREAMS, HIDDEN]``, ``output``
    ``[tokens, HIDDEN]``, ``post`` ``[tokens, STREAMS]`` and ``comb``
    ``[tokens, STREAMS, STREAMS]``, row ``j`` the stream read and column ``k``
    the stream written, both FP32.
    """
    width: tl.constexpr = STREAMS * HIDDEN
    token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = columns < HIDDEN
    streams = tl.arange(0, STREAMS)

    output = tl.load(
        output_ptr + token * HIDDEN + columns, mask=inside, other=0.0
    ).to(tl.float32)
    gains = tl.load(post_ptr + token * STREAMS + streams)

    mixed = tl.zeros((STREAMS, BLOCK), dtype=tl.float32)
    for j in tl.static_range(STREAMS):
        read = tl.load(
            streams_ptr + token * width + j * HIDDEN + columns,
            mask=inside,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            comb_ptr + token * (STREAMS * STREAMS) + j * STREAMS + streams
        )
        mixed += weights[:, None] * read[None, :]

    tl.store(
        out_ptr + token * width + streams[:, None] * HIDDEN + columns[None, :],
        gains[:, None] * output[None, :] + mixed,
        mask=inside[None, :],
    )
