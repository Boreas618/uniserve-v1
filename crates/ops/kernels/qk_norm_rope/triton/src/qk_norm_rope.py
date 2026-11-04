"""Per-head RMS normalization and rotation of queries and keys: the Triton
source of the ``qk_norm_rope`` operator.

One program normalizes and rotates the query heads and the key heads of a block
of ``ROWS`` tokens. Each head's row is normalized by its own RMS with FP32
statistics and the head's weight, then rotated by the token's cosine and sine
factors: split-half rotation ("half") or adjacent-pair rotation
("interleaved"). The result is rounded once into the output dtype.

The numbers are the ones UniServe's kernel computed
(``uniserve_kernels/src/uniserve_kernels/rope.py``, the single-axis path): FP32
sum of squares over the head, ``x · rsqrt(mean + eps) · weight``, one product and
one sum per rotated element, one rounding to the output.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _qk_norm_rope_kernel(
    query_ptr,
    key_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cosine_ptr,
    sine_ptr,
    q_out_ptr,
    k_out_ptr,
    tokens,
    epsilon,
    pdl,
    Q_HEADS: tl.constexpr,  # noqa: N803
    K_HEADS: tl.constexpr,  # noqa: N803
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Normalize and rotate the queries and keys of ``ROWS`` tokens.

    ``query`` is ``[tokens, Q_HEADS, DIM]``, ``key`` ``[tokens, K_HEADS, DIM]``,
    the weights ``[DIM]``, the factors ``[tokens, DIM/2]`` FP32, and the outputs
    the same shapes as ``query`` and ``key``. Grid: ``ceil(tokens / ROWS)``.
    """
    _pdl_prologue(PDL)

    program = tl.program_id(0).to(tl.int64)
    token = program * ROWS + tl.arange(0, ROWS)
    live = token < tokens
    # One lane pair per rotated feature: the two elements each factor rotates.
    pair = tl.arange(0, DIM // 2)
    if INTERLEAVED:
        first_column = 2 * pair
        second_column = 2 * pair + 1
    else:
        first_column = pair
        second_column = pair + DIM // 2
    cosine = tl.load(
        cosine_ptr + token[:, None] * (DIM // 2) + pair[None, :],
        mask=live[:, None],
        other=0.0,
    )
    sine = tl.load(
        sine_ptr + token[:, None] * (DIM // 2) + pair[None, :],
        mask=live[:, None],
        other=0.0,
    )

    for head in tl.static_range(Q_HEADS):
        base = (token * Q_HEADS + head) * DIM
        first = tl.load(
            query_ptr + base[:, None] + first_column[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            query_ptr + base[:, None] + second_column[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        first_weight = tl.load(q_weight_ptr + first_column).to(tl.float32)
        second_weight = tl.load(q_weight_ptr + second_column).to(tl.float32)

        squares = tl.sum(first * first + second * second, axis=1) / DIM
        scale = tl.rsqrt(squares + epsilon)
        normalized_first = first * scale[:, None] * first_weight[None, :]
        normalized_second = second * scale[:, None] * second_weight[None, :]

        direct = normalized_first * cosine
        crossed = normalized_second * sine
        tl.store(
            q_out_ptr + base[:, None] + first_column[None, :],
            direct - crossed,
            mask=live[:, None],
        )
        tl.store(
            q_out_ptr + base[:, None] + second_column[None, :],
            normalized_second * cosine + normalized_first * sine,
            mask=live[:, None],
        )

    for head in tl.static_range(K_HEADS):
        base = (token * K_HEADS + head) * DIM
        first = tl.load(
            key_ptr + base[:, None] + first_column[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        second = tl.load(
            key_ptr + base[:, None] + second_column[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        first_weight = tl.load(k_weight_ptr + first_column).to(tl.float32)
        second_weight = tl.load(k_weight_ptr + second_column).to(tl.float32)

        squares = tl.sum(first * first + second * second, axis=1) / DIM
        scale = tl.rsqrt(squares + epsilon)
        normalized_first = first * scale[:, None] * first_weight[None, :]
        normalized_second = second * scale[:, None] * second_weight[None, :]

        direct = normalized_first * cosine
        crossed = normalized_second * sine
        tl.store(
            k_out_ptr + base[:, None] + first_column[None, :],
            direct - crossed,
            mask=live[:, None],
        )
        tl.store(
            k_out_ptr + base[:, None] + second_column[None, :],
            normalized_second * cosine + normalized_first * sine,
            mask=live[:, None],
        )
