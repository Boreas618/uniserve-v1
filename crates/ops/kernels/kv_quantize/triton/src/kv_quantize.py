"""Quantizing a token's key or value row into an FP8 cache page: the Triton
source of the ``kv_quantize`` operator.

An FP8 cache holds ``key / scale`` per page, with one static FP32 scale per
page: the read side multiplies by the same tensor, so the write is
``round_e4m3(source / scale[page])`` and nothing here computes a scale. One
program writes one element block of one token's row — the row is
``heads · head_dim`` flattening, exactly the cache's page layout — and derives
its page from its slot, so no second pass initializes anything.

A slot of ``-1`` writes neither the values nor the page's flag: the token is
padding in this layer's page table, and the cache keeps what it held.

The operator is **lossy by contract**, so its tolerance is the E4M3 encoding's
own bound per unit of the supplied scale, not an accumulation bound.
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
def _kv_quantize_kernel(
    source,  # *dtype: [tokens, heads · head_dim], the layer's rotated rows
    slots,  # *i32: [tokens] the flat slot `page · page_size + offset`, or -1
    scales,  # *scale: [pages, 1, 1, 1] each page's static scale
    out,  # *fp8e4nv: [pages, page_size, heads, head_dim], the cache
    initialized,  # *i32: [pages, 1, 1, 1] the pages this launch touched
    tokens,  # the step's rows
    page,  # tokens per cache page
    heads,  # key/value heads this rank holds
    head_dim,  # elements per head
    pdl,  # the dependent-launch flag the call carries; the program's copy is PDL
    BLOCK: tl.constexpr,  # elements of a row one program writes
    PDL: tl.constexpr,  # noqa: N803
):
    """Write one block of one token's row into its page.

    Grid: ``[tokens, ceil(heads · head_dim / BLOCK)]``.
    """
    _pdl_prologue(PDL)

    token = tl.program_id(0).to(tl.int64)
    row = (heads * head_dim).to(tl.int64)
    slot = tl.load(slots + token).to(tl.int64)

    # The page's scale is the deployment's, and the cache stores the quotient;
    # a padding slot reads none of it and writes nothing.
    live = slot >= 0
    page_size = page.to(tl.int64)
    page_index = slot // page_size
    scale = tl.load(scales + page_index, mask=live, other=1.0).to(tl.float32)

    elements = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = live & (elements < row)
    values = tl.load(source + token * row + elements, mask=mask, other=0.0).to(tl.float32)
    # The cache's flat element index is the slot's row: `slot · heads · head_dim`.
    encoded = (values / scale).to(tl.float8e4nv)
    tl.store(out + slot * row + elements, encoded, mask=mask)
    tl.store(initialized + page_index, 1, mask=live)
