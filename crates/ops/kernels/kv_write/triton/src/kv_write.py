"""Paged K/V scatter: the Triton source of the ``kv_write`` operator.

One program writes one tile of one token's cache row into the page the scheduler
assigned it, and raises the page's initialization flag on the tile that starts
the row. A slot of ``-1`` writes nothing and initializes nothing.

The kernel body is the one UniServe launched before this package existed
(``uniserve_kernels/src/uniserve_kernels/cache.py``), unchanged except that it
imports nothing but Triton: the operand checks, the slot computation and the
torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _pdl_prologue(PDL: tl.constexpr):
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["tokens", "page", "heads", "head_dim"])
def _kv_write_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slots_ptr,
    key_initialized_ptr,
    value_initialized_ptr,
    tokens,
    page,
    heads,
    head_dim,
    pdl,
    TILE: tl.constexpr,
    PDL: tl.constexpr,  # noqa: N803
):
    """Scatter one tile of one token's keys and values into its page.

    ``key`` and ``value`` are ``[tokens, heads, head_dim]``, the caches
    ``[pages, page, heads, head_dim]``, ``slots`` the flat
    ``page · page_size + offset`` of each token (``-1`` skips it) and the flags
    ``[pages, 1, 1, 1]`` i32. Grid:
    ``[tokens, ceil(heads · head_dim / TILE), 1]``.
    """
    _pdl_prologue(PDL)

    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + token).to(tl.int64)
    if slot < 0:
        return

    row = heads * head_dim
    page_of = slot // page
    offset_of = slot % page

    tile = tl.program_id(1).to(tl.int64)
    columns = tile * TILE + tl.arange(0, TILE)
    mask = columns < row

    source = token * row + columns
    # The page is a [page, heads, head_dim] block of the cache; the token's
    # offset inside it picks the row.
    target = (page_of * page + offset_of) * row + columns
    tl.store(
        key_cache_ptr + target,
        tl.load(key_ptr + source, mask=mask),
        mask=mask,
    )
    tl.store(
        value_cache_ptr + target,
        tl.load(value_ptr + source, mask=mask),
        mask=mask,
    )

    # The first tile of a row raises both flags: the owner of a page zeroes only
    # the storage it created, so a reader waits for this mark.
    if tile == 0:
        tl.store(key_initialized_ptr + page_of, 1)
        tl.store(value_initialized_ptr + page_of, 1)
