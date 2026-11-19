"""The gate of a hashed n-gram memory: the Triton source of ``ngram_gate``.

For one stream ``k`` of one token, with ``key_k`` and ``value`` the token's
projected rows and ``q_k``, ``kappa_k`` the gate's weights::

    dot  = sum_c h[c] (q[c] kappa[c]) key[c]
           * rsqrt(mean(h^2) + eps) rsqrt(mean(key^2) + eps) * scale
    gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
    h'   = round(h + gate value)

every value FP32 until the store, the weights' product formed first, as the
reference forms it. One program holds one stream of one token, a row of
``TILE`` lanes.

The kernel body imports nothing but Triton: the launch-time wrapper, its
operand checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["projected_stride", "eps", "scale"])
def _ngram_gate_kernel(
    streams_ptr,
    projected_ptr,
    query_ptr,
    key_ptr,
    out_ptr,
    projected_stride,
    eps,
    scale,
    STREAMS: tl.constexpr,  # noqa: N803
    HIDDEN: tl.constexpr,  # noqa: N803
    TILE: tl.constexpr,  # noqa: N803
):
    """One stream of one token; grid ``(tokens, STREAMS)``.

    ``streams`` and ``out`` are ``[tokens, STREAMS, HIDDEN]``; ``projected``
    ``[tokens, (STREAMS + 1) HIDDEN]`` with rows ``projected_stride`` apart,
    the key of stream ``k`` at ``k HIDDEN`` and the value after the keys;
    ``query`` and ``key`` ``[STREAMS, HIDDEN]``.
    """
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1)
    columns = tl.arange(0, TILE)
    inside = columns < HIDDEN
    row = (token * STREAMS + stream) * HIDDEN
    projected = projected_ptr + token * projected_stride

    h = tl.load(streams_ptr + row + columns, mask=inside, other=0.0).to(
        tl.float32
    )
    key = tl.load(
        projected + stream * HIDDEN + columns, mask=inside, other=0.0
    ).to(tl.float32)
    value = tl.load(
        projected + STREAMS * HIDDEN + columns, mask=inside, other=0.0
    ).to(tl.float32)
    weight = tl.load(
        query_ptr + stream * HIDDEN + columns, mask=inside, other=0.0
    ).to(tl.float32) * tl.load(
        key_ptr + stream * HIDDEN + columns, mask=inside, other=0.0
    ).to(tl.float32)

    inverse = tl.rsqrt(tl.sum(h * h, axis=0) / HIDDEN + eps) * tl.rsqrt(
        tl.sum(key * key, axis=0) / HIDDEN + eps
    )
    dot = tl.sum(h * weight * key, axis=0) * inverse * scale
    # The root takes the agreement's sign bit, as `copysign` does: a
    # negative zero gives the negative root.
    root = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    negative = dot.to(tl.int32, bitcast=True) < 0
    gate = tl.sigmoid(tl.where(negative, -root, root))

    tl.store(out_ptr + row + columns, h + gate * value, mask=inside)
