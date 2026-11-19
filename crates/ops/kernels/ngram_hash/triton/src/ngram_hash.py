"""Hashed n-gram rows: the Triton source of the ``ngram_hash`` operator.

For each token, its context is the ``ORDER - 1`` vocabulary ids before it,
oldest first, then its own. With ``t_i`` the normalized id ``i`` tokens
back::

    rolling = t_0 mu_0
    rolling = rolling xor (t_i mu_i)              i = 1 .. ORDER - 1
    row[(i - 1) HEADS + h] = rolling mod P[i - 1, h] + O[i - 1, h]

in 64-bit integers: the multipliers are bounded so that no product
overflows, the exclusive-or of non-negative values stays non-negative, and
the remainder is the non-negative one. One program hashes ``BLOCK`` tokens;
an id at or past the normalization table reads its last entry.

The kernel body imports nothing but Triton: the launch-time wrapper, its
operand checks and its torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["tokens", "vocabulary"])
def _ngram_hash_kernel(
    context_ptr,
    normalize_ptr,
    multipliers_ptr,
    moduli_ptr,
    offsets_ptr,
    out_ptr,
    tokens,
    vocabulary,
    ORDER: tl.constexpr,  # noqa: N803
    HEADS: tl.constexpr,  # noqa: N803
    HEAD_TILE: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """``BLOCK`` tokens; grid ``(cdiv(tokens, BLOCK),)``.

    ``context`` is ``[tokens, ORDER]``, ``normalize`` ``[vocabulary]`` I32,
    ``multipliers`` ``[ORDER]`` I64, ``moduli`` and ``offsets`` ``[ORDER - 1,
    HEADS]`` I64 and ``out`` ``[tokens, (ORDER - 1) HEADS]`` I64.
    """
    columns: tl.constexpr = (ORDER - 1) * HEADS
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = rows < tokens
    heads = tl.arange(0, HEAD_TILE)
    head_mask = heads < HEADS

    rolling = tl.zeros((BLOCK,), dtype=tl.int64)
    for back in tl.static_range(ORDER):
        identity = tl.load(
            context_ptr + rows * ORDER + (ORDER - 1 - back), mask=live, other=0
        ).to(tl.int64)
        identity = tl.minimum(identity, vocabulary - 1)
        normalized = tl.load(normalize_ptr + identity, mask=live, other=0).to(
            tl.int64
        )
        term = normalized * tl.load(multipliers_ptr + back)

        if back == 0:
            rolling = term
        else:
            rolling = rolling ^ term
            moduli = tl.load(
                moduli_ptr + (back - 1) * HEADS + heads, mask=head_mask, other=1
            )
            offsets = tl.load(
                offsets_ptr + (back - 1) * HEADS + heads,
                mask=head_mask,
                other=0,
            )
            tl.store(
                out_ptr
                + rows[:, None] * columns
                + (back - 1) * HEADS
                + heads[None, :],
                rolling[:, None] % moduli[None, :] + offsets[None, :],
                mask=live[:, None] & head_mask[None, :],
            )


def derive(fields, parameters, target):
    """The lanes of a program's heads: their count, rounded up to a power of
    two.
    """  # noqa: D205
    heads = int(fields["heads"])
    return {"head_tile": 1 << (heads - 1).bit_length()}
