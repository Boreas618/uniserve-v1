"""The rotary rotation alone: the Triton source of the ``rope_apply`` operator.

One program rotates ``ROWS`` tokens' heads. A head's leading ``ROTATED``
channels turn by the token's own cosine and sine factors — split-half pairing
("half": channel ``j`` with ``j + ROTATED/2``) or adjacent-pair ("interleaved":
``2j`` with ``2j + 1``) — and the rest pass through: a multimodal transformer
rotates part of a head, and a head need not be a power of two (H3's audio path
rotates all 72 channels of its heads). The arithmetic is FP32 and the store
rounds once into the output type.

The tables arrive already evaluated, ``[tokens, ROTATED/2]`` FP32 each, and the
unit-modulus scale is in them, so no frequency or position is read here.

Two programs, because the operator's arguments differ: a full rotation's launch
ends at the dependent-launch flag, a partial one names how many channels rotate.
The count is also in the key, so a module refuses a call of another rotation by
construction rather than by a branch.
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
def _turned_token(
    x,
    cosine,
    sine,
    out,
    token,
    tokens,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    PAIRS_TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    NARROWED: tl.constexpr,
):
    """Rotate one token's heads; the token's row is ``HEADS * HEAD_DIM`` wide."""
    live = token < tokens
    index = tl.arange(0, PAIRS_TILE)
    turning = index < PAIRS
    cos = tl.load(cosine + token * PAIRS + index, mask=live & turning, other=1.0)
    sin = tl.load(sine + token * PAIRS + index, mask=live & turning, other=0.0)

    head = tl.arange(0, HEADS_TILE)
    head_mask = (head < HEADS)[:, None] & live
    offsets = token * (HEADS * HEAD_DIM) + head[:, None] * HEAD_DIM
    column_mask = turning[None, :] & head_mask

    if INTERLEAVED:
        low_column = 2 * index
        high_column = 2 * index + 1
    else:
        low_column = index
        high_column = index + PAIRS

    low = tl.load(x + offsets + low_column[None, :], mask=column_mask, other=0.0).to(tl.float32)
    high = tl.load(x + offsets + high_column[None, :], mask=column_mask, other=0.0).to(tl.float32)
    turned_low = low * cos[None, :] - high * sin[None, :]
    turned_high = high * cos[None, :] + low * sin[None, :]
    tl.store(
        out + offsets + low_column[None, :], turned_low.to(out.dtype.element_ty), mask=column_mask
    )
    tl.store(
        out + offsets + high_column[None, :], turned_high.to(out.dtype.element_ty), mask=column_mask
    )

    if NARROWED:
        # The channels past the rotation are copied: a partial rotation leaves
        # the rest of the head as it arrived.
        rest = tl.arange(0, HEAD_DIM)
        rest_mask = (rest[None, :] >= ROTATED) & head_mask
        values = tl.load(x + offsets + rest[None, :], mask=rest_mask, other=0.0)
        tl.store(out + offsets + rest[None, :], values, mask=rest_mask)


@triton.jit(do_not_specialize=["tokens"])
def _rope_apply_full_kernel(
    x,
    cosine,
    sine,
    out,
    tokens,
    pdl,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    PAIRS_TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    PDL: tl.constexpr,
):
    """Every channel of a head rotates: the launch's last word is ``pdl``."""
    _pdl_prologue(PDL)
    base = tl.program_id(0) * ROWS
    for row in tl.static_range(ROWS):
        _turned_token(
            x,
            cosine,
            sine,
            out,
            base + row,
            tokens,
            ROTATED,
            PAIRS,
            PAIRS_TILE,
            HEAD_DIM,
            HEADS,
            HEADS_TILE,
            INTERLEAVED,
            False,
        )


@triton.jit(do_not_specialize=["tokens", "rotated"])
def _rope_apply_narrowed_kernel(
    x,
    cosine,
    sine,
    out,
    tokens,
    pdl,
    rotated,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    PAIRS_TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    PDL: tl.constexpr,
):
    """``rotated`` channels of a head turn and the rest are copied."""
    _pdl_prologue(PDL)
    base = tl.program_id(0) * ROWS
    for row in tl.static_range(ROWS):
        _turned_token(
            x,
            cosine,
            sine,
            out,
            base + row,
            tokens,
            ROTATED,
            PAIRS,
            PAIRS_TILE,
            HEAD_DIM,
            HEADS,
            HEADS_TILE,
            INTERLEAVED,
            True,
        )
