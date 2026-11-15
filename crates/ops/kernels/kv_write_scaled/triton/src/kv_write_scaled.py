"""The quantizing page write: the Triton source of ``kv_write_scaled``.

A page's writes are made by one program, the program of the first token of the
launch that writes the page. It derives the page's scale from the values this
launch brings, re-encodes the page when that scale grows, stores the incoming
rows at the new scale, and only then commits the scale and the flag.

The five steps are the operator's rule and their order is part of it: a
program that stores before committing the scale, or that grows the scale without
re-encoding the page, leaves a cache whose shapes, keys and flags all agree and
whose numbers do not.

Why one owner per page: the page's scale is the maximum over *every* column of
*every* row the launch brings to it, and the re-encode is over the rows already
stored, so two programs on one page would race on both. The grid is one program
per token, so the work follows the tokens the launch writes and not the pages
the cache has: a token whose page an earlier token of the launch writes returns
at once (the test against the token before it settles every token of a run but
its first, and the first of a run scans the slots before it a block at a time),
and the owner finds the last token of its page by scanning the slots after it,
so it walks only the tokens between its first and its last. The tokens of one
page need not be consecutive, and the owner stores them in the launch's order,
so a slot written twice holds the later row.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch

#: The largest finite E4M3 magnitude, the divisor of a page's scale.
E4M3_MAX = tl.constexpr(448.0)

#: The floor under the page's maximum, so an all-zero write does not produce a
#: zero scale.
SCALE_FLOOR = tl.constexpr(1e-12)

#: The slots one step of an ownership or extent scan reads.
SCAN = 1024


@triton.jit
def _encode(x):
    """Clamp to the largest finite E4M3 magnitude, in the page's scaled units.

    The store rounds into E4M3; this only keeps a value the format cannot hold
    from becoming an infinity on the way.
    """
    return tl.minimum(tl.maximum(x, -E4M3_MAX), E4M3_MAX)


@triton.jit
def _page_hits(slots_ptr, start, end, page, page_of, SCAN: tl.constexpr):
    """The tokens of `[start, start + SCAN)` below `end` in page `page_of`.

    Answers their mask and their indices; a skipped slot (`-1`) is in no page.
    """
    index = start + tl.arange(0, SCAN)
    slot = tl.load(slots_ptr + index, mask=index < end, other=-1).to(tl.int64)
    return (slot >= 0) & (slot // page == page_of), index


@triton.jit(do_not_specialize=["tokens", "page", "heads", "head_dim"])
def _kv_write_scaled_kernel(
    source_ptr,
    slots_ptr,
    scales_ptr,
    cache_ptr,
    initialized_ptr,
    tokens,
    page,
    heads,
    head_dim,
    pdl,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
    SCAN: tl.constexpr,  # noqa: N803
):
    """Write this launch's rows into their pages, deriving the pages' scales.

    ``source`` is ``[tokens, heads · head_dim]`` in full precision, the cache
    ``[pages, page, heads, head_dim]`` E4M3, ``scales`` and ``initialized`` the
    page's f32 scale and u8 flag at ``[pages, 1, 1, 1]``, and ``slots`` the flat
    ``page · page_size + offset`` of each token (``-1`` skips it). A row is
    walked ``BLOCK`` columns at a time and the slots ``SCAN`` at a time.
    Grid: ``[max(tokens, 1), 1, 1]``, one program per token.
    """
    dependent_launch.prologue(PDL)

    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + token, mask=token < tokens, other=-1).to(
        tl.int64
    )
    page_of = slot // page

    # 0. Whether this token's program writes its page: the token is written, and
    # no earlier token of the launch writes the same page. The token before it
    # is tested first, which settles every token of a run but its first; the
    # first of a run scans the slots before it.
    previous = tl.load(slots_ptr + token - 1, mask=token > 0, other=-1).to(
        tl.int64
    )
    owner = (slot >= 0) & ((previous < 0) | (previous // page != page_of))
    if owner:
        for scanned in range(0, token, SCAN):
            hits, index = _page_hits(
                slots_ptr, scanned, token, page, page_of, SCAN
            )
            if tl.max(hits.to(tl.int32), axis=0) != 0:
                owner = False
    if not owner:
        return

    # The page's last token in the launch: the owner walks only the tokens up to
    # it.
    last = token
    for scanned in range(token, tokens, SCAN):
        hits, index = _page_hits(
            slots_ptr, scanned, tokens, page, page_of, SCAN
        )
        last = tl.maximum(last, tl.max(tl.where(hits, index, -1), axis=0))

    row = heads * head_dim
    base = page_of * page * row

    # 1. The page's proposed scale: the maximum |v| over the values this launch
    # brings to it, floored so an all-zero write still has a scale to divide by.
    proposal = 0.0
    for walked in range(token, last + 1):
        other = tl.load(slots_ptr + walked).to(tl.int64)
        if (other >= 0) & (other // page == page_of):
            for start in range(0, row, BLOCK):
                columns = start + tl.arange(0, BLOCK)
                mask = columns < row
                value = tl.load(
                    source_ptr + walked * row + columns, mask=mask, other=0.0
                ).to(tl.float32)
                proposal = tl.maximum(proposal, tl.max(tl.abs(value), axis=0))

    proposed = tl.maximum(proposal, SCALE_FLOOR) / E4M3_MAX

    # 2. The page's scale: the proposal on a fresh page, and never below the
    # stored one, because the rows already written were divided by it and
    # lowering it would change their decoded values.
    flag = tl.load(initialized_ptr + page_of)
    stored = tl.load(scales_ptr + page_of).to(tl.float32)
    scale = tl.where(flag != 0, tl.maximum(stored, proposed), proposed)

    # 3. When the scale grew, every row already stored is re-encoded at it: the
    # bytes it holds are decoded with the scale it was written at, then encoded
    # at the new one.
    if (flag != 0) & (scale > stored):
        for offset in range(0, page):
            for start in range(0, row, BLOCK):
                columns = start + tl.arange(0, BLOCK)
                mask = columns < row
                index = base + offset * row + columns
                decoded = (
                    tl.load(cache_ptr + index, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    * stored
                )
                tl.store(cache_ptr + index, _encode(decoded / scale), mask=mask)

    # 4. The rows this launch brings, at the page's scale, in the launch's
    # order.
    for walked in range(token, last + 1):
        other = tl.load(slots_ptr + walked).to(tl.int64)
        if (other >= 0) & (other // page == page_of):
            offset = other % page
            for start in range(0, row, BLOCK):
                columns = start + tl.arange(0, BLOCK)
                mask = columns < row
                value = tl.load(
                    source_ptr + walked * row + columns, mask=mask, other=0.0
                ).to(tl.float32)
                tl.store(
                    cache_ptr + base + offset * row + columns,
                    _encode(value / scale),
                    mask=mask,
                )

    # 5. The scale, then the flag: a reader that sees the flag set sees a scale
    # that covers every byte it reads.
    tl.store(scales_ptr + page_of, scale)
    tl.store(initialized_ptr + page_of, 1)


def derive(fields, parameters, target):
    """The slots one step of a scan reads (``scan``), this module's ``SCAN``."""
    return {"scan": SCAN}
