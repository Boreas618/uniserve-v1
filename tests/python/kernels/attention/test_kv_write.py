"""The `kv_write` scatter into the paged cache, position for position.

The operator scatters one row per token into the page slot the scheduler
assigned it::

    key, value               [tokens, heads, head_dim]       bf16, f16 or f32
    key_cache, value_cache   [pages, page, heads, head_dim]  bf16, f16 or f32
    slots                    [tokens] i32, the flat `page · page_size + offset`,
                             or -1 to skip
    initialized              [pages, 1, 1, 1] u8, one byte per page, raised to 1
                             by the write

The cache may be another of the three types than the keys (a BF16 model's FP16
cache), and each element is then rounded once into it, to nearest even: the
reference is torch's own rounding of the exact source value. The operator's
tolerance is `exact()`: one rounding of one value has one answer, so two
implementations agree bit for bit and every comparison here is equality, not a
bound.

What a wrong kernel gets wrong, and the case that catches each:

- **the slot read as the token** (the page and offset computed from the token
  instead of `slots`): the slots are not in token order and one sends a later
  token backwards, so a kernel that follows the token order lands every row
  somewhere else;
- **text past the write**: the `-1` row must write nothing and raise no flag,
  so a token the scheduler did not persist leaves its page as the owner zeroed
  it; the caches and the flags are pre-filled with sentinels and compared at
  every position no token names;
- **the flags**: raised to exactly 1 for the pages that received a token and
  left at 0 for the rest, with one page in a case that receives nothing, and
  no byte past the last page written;
- **the row split across programs**: `tile` below, at and above
  `heads · head_dim`, so a row is written by several programs, one, or one with
  a mask, and each token's row is distinct in every element;
- **the conversion into the cache's type**: a kernel that stores the source's
  bits, rounds twice or truncates. The converting cases spread their
  magnitudes over the narrow type's whole range, its subnormals included, and
  a narrowing case in which no element rounds is refused, since it could not
  tell a truncating kernel from a correct one.

Triton's interpreter casts into F16 and F32 to nearest even but truncates a
cast into BF16 (about a third of F32 values differ from the nearest-even
BF16), so the cases that narrow into a BF16 cache need a device.
"""

import pytest
import torch
import triton

pytestmark = pytest.mark.unit

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: Bits of each type's significand, the leading one included: a cast into a
#: type of fewer bits narrows, and a cast into one of at least as many, over a
#: range it covers, is exact.
PRECISION = {"bf16": 8, "f16": 11, "f32": 24}

#: What the owner of a page writes into fresh storage, and what no write may
#: leave behind.
CACHE_SENTINEL = -7.0
FLAG_SENTINEL = 0

#: The bytes the flag buffers hold past the last page, where a four-byte store
#: at a page count not divisible by four would write.
FLAG_TAIL = 4

#: The cases: the case's id, then its source type, the cache's type (the
#: source's own when `None`), the extents, the program's tile, the slots and
#: whether the magnitudes spread over the narrow type's range.
CASES = [
    # Slots out of token order, one -1, and page 2 receives nothing at all.
    # `tile` equals a row, so each token is one program.
    pytest.param(
        "f32", None, 6, 4, 4, 2, 8, 16, [5, 0, 7, -1, 2, 3], False,
        id="ordered slots, one skipped, one page empty",
    ),
    # A row split across four programs, and the flag raised by only the first
    # of them.
    pytest.param(
        "f32", None, 4, 5, 2, 4, 8, 8, [3, 6, -1, 0], False,
        id="a row split across four programs",
    ),
    # `tile` wider than a row: one program per token, the mask covering the
    # rest of it.
    pytest.param(
        "bf16", None, 5, 3, 4, 2, 4, 64, [11, 4, 0, 8, -1], False,
        id="a tile wider than the row",
    ),
    # Every token skipped: nothing written, no flag raised.
    pytest.param(
        "f16", None, 3, 2, 2, 1, 8, 8, [-1, -1, -1], False,
        id="no token persisted",
    ),
    # One token per page exactly, and a page size of one.
    pytest.param(
        "bf16", None, 4, 4, 1, 2, 16, 32, [3, 1, 0, 2], False,
        id="one token per page, page size one",
    ),
    pytest.param(
        "f16", None, 3, 2, 3, 3, 5, 4, [4, 1, 2], False,
        id="f16 row split with a partially filled tail",
    ),
    # A BF16 model's FP16 cache: BF16 values round only where FP16 is coarser,
    # which the spread reaches in FP16's subnormals. A row split across
    # programs with a masked tail, and a skipped token.
    pytest.param(
        "bf16", "f16", 8, 3, 4, 4, 24, 32, [9, 2, -1, 0, 11, 4, 1, 6], True,
        id="bf16 keys into an f16 cache",
    ),
    # FP32 values into FP16, where nearly every element rounds.
    pytest.param(
        "f32", "f16", 5, 2, 4, 2, 16, 16, [3, 6, 0, -1, 5], True,
        id="f32 keys into an f16 cache",
    ),
    # A widening cache holds every value exactly.
    pytest.param(
        "bf16", "f32", 4, 2, 2, 2, 8, 8, [1, 3, 0, 2], True,
        id="bf16 keys into an f32 cache",
    ),
    # The roundings into BF16: FP16's three extra significand bits, and FP32's
    # sixteen. The interpreter truncates a cast into BF16.
    pytest.param(
        "f16", "bf16", 6, 3, 2, 2, 32, 32, [5, -1, 0, 3, 1, 4], True,
        id="f16 keys into a bf16 cache",
        marks=pytest.mark.gpu,
    ),
    pytest.param(
        "f32", "bf16", 4, 2, 4, 3, 8, 16, [7, 2, 4, 0], True,
        id="f32 keys into a bf16 cache",
        marks=pytest.mark.gpu,
    ),
]  # fmt: skip


def source_rows(generator, tokens, heads, head_dim, element, spread):
    """`[tokens, heads, head_dim]` values of `element`, distinct in every
    element.

    A `spread` case scales each element by its own power of two between
    `2^-22` and `2^7`: a cast into F16 then meets its subnormals, where even a
    BF16 value rounds, as well as its normal range, and stays below its largest
    finite value, so the case checks the rounding and not an overflow.
    """  # noqa: D205
    values = torch.randn(
        tokens, heads, head_dim, dtype=torch.float32, generator=generator
    )
    if spread:
        exponents = torch.randint(-22, 8, values.shape, generator=generator)
        values = values * torch.exp2(exponents.to(torch.float32))
    return values.to(element)


def launch(
    kernels,
    key,
    value,
    key_cache,
    value_cache,
    slots,
    key_flags,
    value_flags,
    page,
    tile,
):
    """One scatter, with the grid the operator computes:
    `[tokens, ceil(heads · head_dim / tile)]`.

    The flag buffers hold `FLAG_TAIL` bytes past the last page; the kernel is
    handed the `[pages, 1, 1, 1]` view of the pages' own bytes.
    """  # noqa: D205
    tokens, heads, head_dim = key.shape
    pages = key_cache.shape[0]
    row = heads * head_dim
    kernels._kv_write_kernel[(tokens, triton.cdiv(row, tile), 1)](
        key,
        value,
        key_cache,
        value_cache,
        slots,
        key_flags[:pages].view(pages, 1, 1, 1),
        value_flags[:pages].view(pages, 1, 1, 1),
        tokens,
        page,
        heads,
        head_dim,
        TILE=tile,
        PDL=False,
    )


def rounded(rows, kv):
    """`rows` rounded once into the cache's type `kv`, to nearest even.

    This is torch's cast of the exact source value, which an F32 intermediate
    holds for every source type here.
    """
    return rows.to(torch.float32).to(DTYPES[kv])


@pytest.mark.parametrize(
    (
        "dtype",
        "kv",
        "tokens",
        "pages",
        "page",
        "heads",
        "head_dim",
        "tile",
        "slots",
        "spread",
    ),
    CASES,
)
def test_every_row_lands_on_its_slot_rounded_once_and_nothing_else_moves(
    dtype,
    kv,
    tokens,
    pages,
    page,
    heads,
    head_dim,
    tile,
    slots,
    spread,
    torch_device,
    kernel_source,
):
    kv = kv or dtype
    generator = torch.Generator().manual_seed(7)
    # Distinct in every element: a row written at the wrong place cannot be
    # mistaken for this one.
    key = source_rows(generator, tokens, heads, head_dim, DTYPES[dtype], spread)
    value = source_rows(
        generator, tokens, heads, head_dim, DTYPES[dtype], spread
    )
    cache_shape = (pages, page, heads, head_dim)
    key_cache = torch.full(cache_shape, CACHE_SENTINEL, dtype=DTYPES[kv])
    value_cache = torch.full(cache_shape, CACHE_SENTINEL, dtype=DTYPES[kv])
    # One byte per page, and the buffer is longer than the view so the bytes
    # after the last page are visible.
    key_flags = torch.full(
        (pages + FLAG_TAIL,), FLAG_SENTINEL, dtype=torch.uint8
    )
    value_flags = torch.full(
        (pages + FLAG_TAIL,), FLAG_SENTINEL, dtype=torch.uint8
    )
    slot_tensor = torch.tensor(slots, dtype=torch.int32)

    placed = [
        tensor.to(torch_device)
        for tensor in (
            key,
            value,
            key_cache,
            value_cache,
            slot_tensor,
            key_flags,
            value_flags,
        )
    ]
    launch(kernel_source("kv_write"), *placed, page=page, tile=tile)
    key_cache, value_cache, key_flags, value_flags = (
        tensor.cpu() for tensor in (placed[2], placed[3], placed[5], placed[6])
    )
    where = (
        f"{dtype} into {kv}, tokens {tokens} pages {pages} page {page} heads "
        f"{heads} head_dim {head_dim} tile {tile} slots {slots}"
    )

    # Every position: the writes land on the slots, rounded into the cache's
    # type, and nothing else moves.
    stored_key = rounded(key, kv)
    stored_value = rounded(value, kv)
    expected_key = torch.full_like(key_cache, CACHE_SENTINEL)
    expected_value = torch.full_like(value_cache, CACHE_SENTINEL)
    expected_flags = torch.zeros((pages, 1, 1, 1), dtype=torch.uint8)
    written = 0
    for token, slot in enumerate(slots):
        if slot < 0:
            continue
        page_of, offset_of = divmod(slot, page)
        assert page_of < pages, f"{where}: slot {slot} names page {page_of}"
        expected_key[page_of, offset_of] = stored_key[token]
        expected_value[page_of, offset_of] = stored_value[token]
        expected_flags[page_of] = 1
        written += 1

    # A narrowing case must round some element it writes, or a kernel that
    # truncated or stored the source's bits unrounded could pass it.
    persisted = torch.tensor([slot >= 0 for slot in slots])
    rounding = int(
        (stored_key.to(torch.float32) != key.to(torch.float32))[persisted].sum()
    )
    assert not (PRECISION[kv] < PRECISION[dtype] and written) or rounding, (
        f"{where}: no element of the narrowing case rounds; it cannot tell a "
        "rounding"
    )
    # The counts, so a case that wrote rows and flagged nothing cannot pass.
    assert not written or int(expected_flags.sum()), (
        f"{where}: the case writes {written} tokens and flags no page"
    )

    for name, got, want in (
        ("key cache", key_cache, expected_key),
        ("value cache", value_cache, expected_value),
        ("key flags", key_flags[:pages].view(pages, 1, 1, 1), expected_flags),
        (
            "value flags",
            value_flags[:pages].view(pages, 1, 1, 1),
            expected_flags,
        ),
    ):
        differing = (got != want).reshape(-1).nonzero()
        first = int(differing[0]) if len(differing) else 0
        assert torch.equal(got, want), (
            f"{where}: the {name} differ at {len(differing)} of {got.numel()} "
            f"positions; first at {first}: {float(got.reshape(-1)[first])} vs "
            f"{float(want.reshape(-1)[first])}"
        )
    # The bytes past the last page: a flag stored as four bytes at a page count
    # not divisible by four clobbers them.
    for name, flags in (("key flags", key_flags), ("value flags", value_flags)):
        tail = flags[pages:]
        assert torch.equal(tail, torch.full_like(tail, FLAG_SENTINEL)), (
            f"{where}: the {name} wrote past the last page: {tail.tolist()} "
            f"after the {pages}-page plane"
        )
