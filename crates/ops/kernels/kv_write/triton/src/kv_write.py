"""Paged K/V scatter: the Triton source of the ``kv_write`` operator.

One program writes one tile of one token's cache row into the page the scheduler
assigned it, and raises the page's initialization flag on the tile that starts
the row. A slot of ``-1`` writes nothing and initializes nothing.

The cache may hold another dense type than the keys (BF16 keys into an FP16
cache, FP32 keys into a BF16 one): each element is rounded once into the
cache's type, to nearest even, in the store itself, so a cache of the keys' own
type receives them unchanged and a narrower one costs no pass of its own.

The kernel body imports nothing but Triton: the operand checks, the slot
computation and the torch types stay with the operator's Rust side.
"""

import triton
import triton.language as tl
from uniserve_kernels import dependent_launch


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
    ``[pages, 1, 1, 1]`` u8, one byte per page. Grid:
    ``[tokens, ceil(heads · head_dim / TILE), 1]``.
    """
    dependent_launch.prologue(PDL)

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
    # One rounding into the cache's element type, Triton's default for a
    # narrowing cast: to nearest even. A cast between two 16-bit types passes
    # through FP32, which holds both exactly, so the rounding is of the source
    # value itself. A cache of the keys' own type is a copy.
    tl.store(
        key_cache_ptr + target,
        tl.load(key_ptr + source, mask=mask).to(key_cache_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        value_cache_ptr + target,
        tl.load(value_ptr + source, mask=mask).to(
            value_cache_ptr.dtype.element_ty
        ),
        mask=mask,
    )

    # The first tile of a row raises both flags: the owner of a page zeroes only
    # the storage it created, so a reader waits for this mark.
    if tile == 0:
        tl.store(key_initialized_ptr + page_of, 1)
        tl.store(value_initialized_ptr + page_of, 1)
