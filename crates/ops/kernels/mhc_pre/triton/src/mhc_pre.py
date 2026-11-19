"""An mHC residual read into one normalized row: the source of ``mhc_pre``.

For each token::

    x   = round(sum_j pre_j h_j)                      FP32 sum, rounded once
    out = x rsqrt(mean(x^2) + eps) w                  FP32, rounded once more

the collapse rounded into the streams' type before the normalization reads
it, as the reference stores it between the two. One program reads one
token's streams, a row of ``TILE`` lanes at a time.

The kernel body imports nothing but Triton: the launch-time wrapper, its
operand checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def _mhc_pre_kernel(
    streams_ptr,
    pre_ptr,
    norm_ptr,
    out_ptr,
    eps,
    STREAMS: tl.constexpr,  # noqa: N803
    HIDDEN: tl.constexpr,  # noqa: N803
    TILE: tl.constexpr,  # noqa: N803
):
    """One token; grid ``(tokens,)``.

    ``streams`` is ``[tokens, STREAMS, HIDDEN]``, ``pre`` ``[tokens, STREAMS]``
    FP32, ``norm`` ``[HIDDEN]`` and ``out`` ``[tokens, HIDDEN]``.
    """
    token = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, TILE)
    inside = columns < HIDDEN

    collapsed = tl.zeros((TILE,), dtype=tl.float32)
    for j in tl.static_range(STREAMS):
        weight = tl.load(pre_ptr + token * STREAMS + j)
        read = tl.load(
            streams_ptr + (token * STREAMS + j) * HIDDEN + columns,
            mask=inside,
            other=0.0,
        ).to(tl.float32)
        collapsed += weight * read
    collapsed = collapsed.to(out_ptr.dtype.element_ty).to(tl.float32)

    inverse = tl.rsqrt(tl.sum(collapsed * collapsed, axis=0) / HIDDEN + eps)
    gain = tl.load(norm_ptr + columns, mask=inside, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + token * HIDDEN + columns,
        collapsed * inverse * gain,
        mask=inside,
    )
