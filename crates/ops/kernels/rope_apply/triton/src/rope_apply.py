"""The rotary rotation alone: the Triton source of the ``rope_apply`` operator.

One program rotates ``ROWS`` tokens' heads. A head's ``ROTATED`` channels from
channel ``FIRST`` on turn by the token's own cosine and sine factors —
split-half pairing ("half": channel ``FIRST + j`` with ``FIRST + j +
ROTATED/2``) or adjacent-pair ("interleaved": ``FIRST + 2j`` with ``FIRST + 2j
+ 1``) — and the rest pass through: a multimodal transformer rotates part of a
head, a latent attention its trailing channels, and a head need not be a
power of two (a head of 72 channels rotated whole). ``INVERSE`` turns by the
negated angles, which takes the rotation by the same tables back out. The
arithmetic is FP32 and the store rounds once into the output type.

The tables arrive already evaluated, ``[tokens, ROTATED/2]`` FP32 each, and the
unit-modulus scale is in them, so no frequency or position is read here.

Two programs, because the operator's arguments differ: a full rotation's launch
ends at the dependent-launch flag, a partial one names how many channels rotate.
The count is also in the key, so a module refuses a call of another rotation by
construction rather than by a branch. A partial rotation into another tensor
copies the channels it does not turn (``COPY``); one in place leaves them where
they are.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch


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
    FIRST: tl.constexpr,
    INVERSE: tl.constexpr,
    COPY: tl.constexpr,
):
    """Rotate one token's heads; the token's row is ``HEADS * HEAD_DIM`` wide."""
    live = token < tokens
    index = tl.arange(0, PAIRS_TILE)
    turning = index < PAIRS
    cos = tl.load(cosine + token * PAIRS + index, mask=live & turning, other=1.0)
    sin = tl.load(sine + token * PAIRS + index, mask=live & turning, other=0.0)
    if INVERSE:
        # The rotation by the negated angle: cos(-a) = cos(a), sin(-a) = -sin(a).
        sin = -sin

    head = tl.arange(0, HEADS_TILE)
    head_mask = (head < HEADS)[:, None] & live
    offsets = token * (HEADS * HEAD_DIM) + head[:, None] * HEAD_DIM
    column_mask = turning[None, :] & head_mask

    if INTERLEAVED:
        low_column = FIRST + 2 * index
        high_column = FIRST + 2 * index + 1
    else:
        low_column = FIRST + index
        high_column = FIRST + index + PAIRS

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

    if COPY:
        # The channels outside the rotation are copied: a partial rotation into
        # another tensor leaves the rest of the head as it arrived.
        rest = tl.arange(0, HEAD_DIM)
        outside = (rest < FIRST) | (rest >= FIRST + ROTATED)
        rest_mask = outside[None, :] & head_mask
        values = tl.load(x + offsets + rest[None, :], mask=rest_mask, other=0.0)
        tl.store(out + offsets + rest[None, :], values, mask=rest_mask)


@triton.jit(do_not_specialize=["tokens"])
def _rope_apply_full_kernel(
    x,
    cosine,
    sine,
    out,
    tokens,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    PAIRS_TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    INVERSE: tl.constexpr,
    PDL: tl.constexpr,
):
    """Every channel of a head rotates: the launch's last word is ``tokens``."""
    dependent_launch.prologue(PDL)
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
            0,
            INVERSE,
            False,
        )


@triton.jit(do_not_specialize=["tokens", "rotated"])
def _rope_apply_narrowed_kernel(
    x,
    cosine,
    sine,
    out,
    tokens,
    rotated,
    ROTATED: tl.constexpr,
    PAIRS: tl.constexpr,
    PAIRS_TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS_TILE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    FIRST: tl.constexpr,
    INVERSE: tl.constexpr,
    COPY: tl.constexpr,
    PDL: tl.constexpr,
):
    """``rotated`` channels of a head from ``FIRST`` on turn; the rest are
    copied, unless the output is the input.
    """  # noqa: D205
    dependent_launch.prologue(PDL)
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
            FIRST,
            INVERSE,
            COPY,
        )


def derive(fields, parameters, target):
    """The rotation's width, placement and direction, its pairs, and the
    program its launch names.

    A key names ``rotated`` only for a rotation narrower than the head, so a
    key without it turns all ``head_dim`` channels; it names ``first``,
    ``inverse`` and ``inplace`` only when they are set, so a key without them
    turns the leading channels forward into another tensor. The turned pairs
    are walked in a power-of-two tile. A rotation as wide as the head takes
    the full program; a narrower one the program whose call names its width,
    which copies the channels it does not turn unless it rotates in place.

    Raises:
        SystemExit: When the rotation leaves the head, or the head tile
            ``block`` is narrower than the head count.
    """  # noqa: D205
    head_dim = fields["head_dim"]
    rotated = fields.get("rotated", head_dim)
    first = fields.get("first", 0)
    if first + rotated > head_dim:
        raise SystemExit(
            f"a rotation of {rotated} channels from channel {first} exceeds a "
            f"{head_dim} channel head"
        )
    heads, block = fields["heads"], fields["block"]
    if heads > block:
        raise SystemExit(f"{heads} heads do not fit a {block} lane head tile")

    pairs = rotated // 2
    narrowed = rotated != head_dim
    inplace = fields.get("inplace", False)
    return {
        "rotated": rotated,
        "pairs": pairs,
        "pairs_tile": triton.next_power_of_2(pairs),
        "narrowed": narrowed,
        "first": first,
        "inverse": fields.get("inverse", False),
        "copy": narrowed and not inplace,
    }
