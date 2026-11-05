"""The quantizing page write: the Triton source of the ``kv_write_scaled`` operator.

One program owns one page of an E4M3 cache. It derives the page's scale from the
values this launch brings, re-encodes the page when that scale grows, stores the
incoming rows at the new scale, and only then commits the scale and the flag.

The five steps are the operator's contract and their order is part of it: a
program that stores before committing the scale, or that grows the scale without
re-encoding the page, leaves a cache whose shapes, keys and flags all agree and
whose numbers do not.

Why one program per page and not one per tile: the page's scale is the maximum
over *every* column of *every* row the launch brings, and the re-encode is over
the rows already stored. A program that saw only one tile of the row could
compute neither, and two programs on one page would race on the re-encode, so
the operator's second grid axis (``row / block``) names the width of the passes
this program makes and the programs past the first return.
"""

import triton
import triton.language as tl

#: The largest finite E4M3 magnitude, the divisor of a page's scale.
E4M3_MAX = tl.constexpr(448.0)

#: The floor under the page's maximum, so an all-zero write does not produce a zero scale.
SCALE_FLOOR = tl.constexpr(1e-12)


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _encode(x):
    """Clamp to the largest finite E4M3 magnitude, in the page's scaled units.

    The store rounds into E4M3; this only keeps a value the format cannot hold from
    becoming an infinity on the way.
    """
    return tl.minimum(tl.maximum(x, -E4M3_MAX), E4M3_MAX)


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
):
    """Write this launch's rows into their pages, deriving and committing the pages' scales.

    ``source`` is ``[tokens, heads · head_dim]`` in full precision, the cache
    ``[pages, page, heads, head_dim]`` E4M3, ``scales`` and ``initialized`` the
    page's f32 scale and i32 flag at ``[pages, 1, 1, 1]``, and ``slots`` the flat
    ``page · page_size + offset`` of each token (``-1`` skips it).
    Grid: ``[pages, ceil(heads · head_dim / BLOCK), 1]``.
    """
    _pdl_prologue(PDL)

    page_of = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) != 0:
        return

    row = heads * head_dim
    base = page_of * page * row

    # 1. The page's proposed scale: the maximum |v| over the values this launch brings to it,
    #    floored so an all-zero write still has a scale to divide by.
    proposal = 0.0
    touched = 0
    for token in range(0, tokens):
        slot = tl.load(slots_ptr + token).to(tl.int64)
        if (slot >= 0) & (slot // page == page_of):
            touched += 1
            for start in range(0, row, BLOCK):
                columns = start + tl.arange(0, BLOCK)
                mask = columns < row
                value = tl.load(
                    source_ptr + token * row + columns, mask=mask, other=0.0
                ).to(tl.float32)
                proposal = tl.maximum(proposal, tl.max(tl.abs(value), axis=0))

    # A page no token of this launch touches keeps its bytes, its scale and its flag: the
    # grow-and-re-encode case is driven by the incoming values, so an untouched page never grows.
    if touched == 0:
        return

    proposed = tl.maximum(proposal, SCALE_FLOOR) / E4M3_MAX

    # 2. The page's scale: the proposal on a fresh page, and never below the stored one, because the
    #    rows already written were divided by it and lowering it would change their decoded values.
    flag = tl.load(initialized_ptr + page_of)
    stored = tl.load(scales_ptr + page_of).to(tl.float32)
    scale = tl.where(flag != 0, tl.maximum(stored, proposed), proposed)

    # 3. When the scale grew, every row already stored is re-encoded at it: the bytes it holds are
    #    decoded with the scale it was written at, then encoded at the new one.
    if (flag != 0) & (scale > stored):
        for offset in range(0, page):
            for start in range(0, row, BLOCK):
                columns = start + tl.arange(0, BLOCK)
                mask = columns < row
                index = base + offset * row + columns
                decoded = tl.load(cache_ptr + index, mask=mask, other=0.0).to(tl.float32) * stored
                tl.store(cache_ptr + index, _encode(decoded / scale), mask=mask)

    # 4. The rows this launch brings, at the page's scale.
    for token in range(0, tokens):
        slot = tl.load(slots_ptr + token).to(tl.int64)
        if (slot >= 0) & (slot // page == page_of):
            offset = slot % page
            for start in range(0, row, BLOCK):
                columns = start + tl.arange(0, BLOCK)
                mask = columns < row
                value = tl.load(
                    source_ptr + token * row + columns, mask=mask, other=0.0
                ).to(tl.float32)
                tl.store(
                    cache_ptr + base + offset * row + columns,
                    _encode(value / scale),
                    mask=mask,
                )

    # 5. The scale, then the flag: a reader that sees the flag set sees a scale that covers every
    #    byte it reads.
    tl.store(scales_ptr + page_of, scale)
    tl.store(initialized_ptr + page_of, 1)
