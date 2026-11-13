#!/usr/bin/env python3
"""Run `kv_write`'s kernel and check the paged-cache scatter, on the interpreter or a device.

    .venv/bin/python kv_write_check.py [--package DIR]             # Triton's interpreter
    .venv/bin/python kv_write_check.py --device [--package DIR]    # a device

The operator scatters one row per token into the page slot the scheduler assigned it:

```text
key, value        [tokens, heads, head_dim]           bf16, f16 or f32
key_cache, value_cache   [pages, page, heads, head_dim]   bf16, f16 or f32
slots             [tokens] i32, the flat `page · page_size + offset`, or -1 to skip
initialized       [pages, 1, 1, 1] u8, one byte per page, raised to 1 by the write
```

The cache may be another of the three types than the keys (a BF16 model's FP16 cache), and each
element is then rounded once into it, to nearest even: the reference is torch's own rounding of the
exact source value. The operator's tolerance is `exact()`: one rounding of one value has one answer,
so two implementations agree bit for bit and every assertion here is equality, not a bound.

What a wrong kernel gets wrong, and the case here that catches each:

- **the slot read as the token** (`page_of`/`offset_of` computed from `token` instead of `slots`):
  the slots are deliberately **not** in token order and one sends a later token backwards, so a
  kernel that follows the token order lands every row somewhere else;
- **text past the write** — the `-1` row must write nothing *and* raise no flag, so a token the
  scheduler did not persist leaves its page as the owner zeroed it: the caches and the flags are
  pre-filled with sentinels and asserted unchanged at every position no token names;
- **the flags**: raised to exactly 1 for the pages that received a token and left at 0 for the rest,
  with one page in the case that receives nothing;
- **the row split across programs**: `tile` below, at and above `heads · head_dim`, so a row is
  written by several programs, one, or one with a mask — and each token's row is distinct in every
  element, so a program writing at the wrong column, or a program of the next token overlapping this
  one, shows;
- **the conversion into the cache's type**: a kernel that stores the source's bits, rounds twice or
  truncates. The converting cases spread their magnitudes over the narrow type's whole range, its
  subnormals included, so a narrowing case rounds many elements, and the check refuses a narrowing
  case in which no element rounds, since it could not tell a truncating kernel from a correct one.

Where each case runs: Triton's interpreter casts into F16 and F32 to nearest even (numpy's own
casts) but **truncates** a cast into BF16 (measured: about a third of F32 values differ from the
nearest-even BF16), so the cases that narrow into a BF16 cache run with `--device` only, and every
other case runs on both.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "kv_write" / "triton"

#: Triton reads `TRITON_INTERPRET` when it is imported, so the mode is chosen from the command line
#: before the import below: `--device` compiles for the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402  (after TRITON_INTERPRET, which Triton reads at import)

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: Bits of each type's significand, the leading one included: a cast into a type of fewer bits
#: narrows, and a cast into one of at least as many, over a range it covers, is exact.
PRECISION = {"bf16": 8, "f16": 11, "f32": 24}

#: What the owner of a page writes into fresh storage, and what no write may leave behind.
CACHE_SENTINEL = -7.0
FLAG_SENTINEL = 0


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "kv_write.py"
    spec = importlib.util.spec_from_file_location("kv_write_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows(tokens, heads, head_dim, element, spread):
    """`[tokens, heads, head_dim]` values of `element`, distinct in every element.

    A `spread` case scales each element by its own power of two between `2^-22` and `2^7`: a cast
    into F16 then meets its subnormals, where even a BF16 value rounds, as well as its normal range,
    and stays below its largest finite value, so the case checks the rounding and not an overflow.
    """
    values = torch.randn(tokens, heads, head_dim, dtype=torch.float32)
    if spread:
        exponents = torch.randint(-22, 8, values.shape)
        values = values * torch.exp2(exponents.to(torch.float32))
    return values.to(element)


def launch(
    module,
    *,
    tokens,
    pages,
    page,
    heads,
    head_dim,
    tile,
    dtype,
    kv,
    slots,
    spread,
    seed=7,
):
    """One scatter, with the grid the operator computes: `[tokens, ceil(row / tile)]`.

    Returns the source rows and what the launch left in the caches and the flag buffers, on the host.
    """
    element = DTYPES[dtype]
    cache_element = DTYPES[kv]
    # A compiled launch reads device memory; the interpreter reads host tensors.
    device = "cuda" if DEVICE else "cpu"
    torch.manual_seed(seed)
    row = heads * head_dim
    # Distinct in every element: a row written at the wrong place cannot be mistaken for this one.
    key = source_rows(tokens, heads, head_dim, element, spread)
    value = source_rows(tokens, heads, head_dim, element, spread)
    key_cache = torch.full(
        (pages, page, heads, head_dim), CACHE_SENTINEL, dtype=cache_element
    )
    value_cache = torch.full(
        (pages, page, heads, head_dim), CACHE_SENTINEL, dtype=cache_element
    )
    # The flags are one byte per page, and the buffer is longer than the view so the bytes after the
    # last page are visible: a four-byte store at an odd page count writes into them.
    key_flags = torch.full((pages + 4,), FLAG_SENTINEL, dtype=torch.uint8)
    value_flags = torch.full((pages + 4,), FLAG_SENTINEL, dtype=torch.uint8)
    slots = torch.tensor(slots, dtype=torch.int32)

    (
        key_on,
        value_on,
        key_cache_on,
        value_cache_on,
        key_flags_on,
        value_flags_on,
        slots_on,
    ) = (
        tensor.to(device)
        for tensor in (
            key,
            value,
            key_cache,
            value_cache,
            key_flags,
            value_flags,
            slots,
        )
    )

    module._kv_write_kernel[(tokens, (row + tile - 1) // tile, 1)](
        key_on,
        value_on,
        key_cache_on,
        value_cache_on,
        slots_on,
        key_flags_on[:pages].view(pages, 1, 1, 1),
        value_flags_on[:pages].view(pages, 1, 1, 1),
        tokens,
        page,
        heads,
        head_dim,
        0,
        TILE=tile,
        PDL=False,
    )
    key_cache, value_cache, key_flags, value_flags = (
        tensor.cpu()
        for tensor in (
            key_cache_on,
            value_cache_on,
            key_flags_on,
            value_flags_on,
        )
    )
    return (
        key,
        value,
        key_cache,
        value_cache,
        key_flags[:pages].view(pages, 1, 1, 1),
        value_flags[:pages].view(pages, 1, 1, 1),
        key_flags,
        value_flags,
    )


def rounded(rows, kv):
    """`rows` rounded once into the cache's type `kv`, to nearest even.

    This is torch's cast of the exact source value, which an F32 intermediate holds for every
    source type here.
    """
    return rows.to(torch.float32).to(DTYPES[kv])


def check(
    what,
    *,
    dtype,
    tokens,
    pages,
    page,
    heads,
    head_dim,
    tile,
    slots,
    kv=None,
    spread=False,
):
    """One scatter against the layout the operator documents, element for element.

    `kv` is the cache's type, the source's own unless stated.
    """
    kv = kv or dtype
    (
        key,
        value,
        key_cache,
        value_cache,
        key_initialized,
        value_initialized,
        key_flags,
        value_flags,
    ) = launch(
        MODULE,
        tokens=tokens,
        pages=pages,
        page=page,
        heads=heads,
        head_dim=head_dim,
        tile=tile,
        dtype=dtype,
        kv=kv,
        slots=slots,
        spread=spread,
    )
    where = (
        f"{what}: {dtype} into {kv}, tokens {tokens} pages {pages} page {page} heads {heads} "
        f"head_dim {head_dim} tile {tile} slots {slots}"
    )

    # Every position, before and after: the writes land on the slots, rounded into the cache's
    # type, and nothing else moves.
    stored_key = rounded(key, kv)
    stored_value = rounded(value, kv)
    expected_key = torch.full_like(key_cache, CACHE_SENTINEL)
    expected_value = torch.full_like(value_cache, CACHE_SENTINEL)
    expected_flags = torch.zeros_like(key_initialized)
    written = 0
    for token, slot in enumerate(slots):
        if slot < 0:
            continue
        page_of, offset_of = divmod(slot, page)
        if page_of >= pages:
            raise SystemExit(
                f"{where}: slot {slot} names page {page_of} of {pages}"
            )
        expected_key[page_of, offset_of] = stored_key[token]
        expected_value[page_of, offset_of] = stored_value[token]
        expected_flags[page_of] = 1
        written += 1

    # A narrowing case must round some element it writes, or a kernel that truncated or stored
    # the source's bits unrounded could pass it.
    persisted = torch.tensor([slot >= 0 for slot in slots])
    rounding = int(
        (stored_key.to(torch.float32) != key.to(torch.float32))[persisted].sum()
    )
    if PRECISION[kv] < PRECISION[dtype] and written and rounding == 0:
        raise SystemExit(
            f"{where}: no element of the narrowing case rounds; it cannot tell a rounding"
        )

    for name, got, want in (
        ("key cache", key_cache, expected_key),
        ("value cache", value_cache, expected_value),
        ("key flags", key_initialized, expected_flags),
        ("value flags", value_initialized, expected_flags),
    ):
        if not torch.equal(got, want):
            positions = (got != want).reshape(-1).nonzero()
            first = int(positions[0])
            raise SystemExit(
                f"{where}: the {name} differ at {len(positions)} of {got.numel()} positions; "
                f"first at {first}: {float(got.reshape(-1)[first])} vs "
                f"{float(want.reshape(-1)[first])}"
            )
    # The bytes past the last page: a flag stored as four bytes at a page count not divisible by
    # four clobbers them, and nothing else in the case would show it.
    for name, flags in (("key flags", key_flags), ("value flags", value_flags)):
        tail = flags[len(flags) - 4 :]
        if not torch.equal(tail, torch.full_like(tail, FLAG_SENTINEL)):
            raise SystemExit(
                f"{where}: the {name} wrote past the last page: {tail.tolist()} after "
                f"{len(flags)} bytes of a {len(flags) - 4}-page plane"
            )

    # The counts, so a case that wrote nothing or flagged nothing cannot pass quietly.
    if written and int(expected_flags.sum()) == 0:
        raise SystemExit(
            f"{where}: the case writes {written} tokens and flags no page"
        )
    return written, int(expected_flags.sum()), tokens - written, rounding


def main():
    global MODULE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=PACKAGE,
        help="the package directory holding src/kv_write.py",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="compile for the device and run every case, the ones that round into a BF16 cache "
        "included",
    )
    options = parser.parse_args()
    if options.device and not torch.cuda.is_available():
        print(
            "skipped: needs a device - the device half runs every case, the roundings into a "
            "BF16 cache included; without --device the interpreter runs the cases it can answer"
        )
        return 0
    MODULE = load_kernel(options.package)

    cases = [
        # Slots out of token order, one -1, and page 2 receives nothing at all. `tile` equals a row,
        # so each token is one program.
        (
            "ordered slots, one skipped, one page empty",
            dict(
                dtype="f32",
                tokens=6,
                pages=4,
                page=4,
                heads=2,
                head_dim=8,
                tile=16,
                slots=[5, 0, 7, -1, 2, 3],
            ),
        ),
        # A row split across four programs, and the flag raised by only the first of them.
        (
            "a row split across four programs",
            dict(
                dtype="f32",
                tokens=4,
                pages=5,
                page=2,
                heads=4,
                head_dim=8,
                tile=8,
                slots=[3, 6, -1, 0],
            ),
        ),
        # `tile` wider than a row: one program per token, the mask covering the rest of it.
        (
            "a tile wider than the row",
            dict(
                dtype="bf16",
                tokens=5,
                pages=3,
                page=4,
                heads=2,
                head_dim=4,
                tile=64,
                slots=[11, 4, 0, 8, -1],
            ),
        ),
        # Every token skipped: nothing written, no flag raised.
        (
            "no token persisted",
            dict(
                dtype="f16",
                tokens=3,
                pages=2,
                page=2,
                heads=1,
                head_dim=8,
                tile=8,
                slots=[-1, -1, -1],
            ),
        ),
        # One token per page exactly, the fixture's narrow types, and a page size of one.
        (
            "one token per page, page size one",
            dict(
                dtype="bf16",
                tokens=4,
                pages=4,
                page=1,
                heads=2,
                head_dim=16,
                tile=32,
                slots=[3, 1, 0, 2],
            ),
        ),
        (
            "f16 row split with a partially filled tail",
            dict(
                dtype="f16",
                tokens=3,
                pages=2,
                page=3,
                heads=3,
                head_dim=5,
                tile=4,
                slots=[4, 1, 2],
            ),
        ),
        # A BF16 model's FP16 cache: BF16 values round only where FP16 is coarser, which the
        # spread reaches in FP16's subnormals. A row split across programs with a masked tail, and
        # a skipped token.
        (
            "bf16 keys into an f16 cache",
            dict(
                dtype="bf16",
                kv="f16",
                tokens=8,
                pages=3,
                page=4,
                heads=4,
                head_dim=24,
                tile=32,
                slots=[9, 2, -1, 0, 11, 4, 1, 6],
                spread=True,
            ),
        ),
        # FP32 values into FP16, where nearly every element rounds.
        (
            "f32 keys into an f16 cache",
            dict(
                dtype="f32",
                kv="f16",
                tokens=5,
                pages=2,
                page=4,
                heads=2,
                head_dim=16,
                tile=16,
                slots=[3, 6, 0, -1, 5],
                spread=True,
            ),
        ),
        # A widening cache holds every value exactly.
        (
            "bf16 keys into an f32 cache",
            dict(
                dtype="bf16",
                kv="f32",
                tokens=4,
                pages=2,
                page=2,
                heads=2,
                head_dim=8,
                tile=8,
                slots=[1, 3, 0, 2],
                spread=True,
            ),
        ),
        # The roundings into BF16: FP16's three extra significand bits, and FP32's sixteen.
        (
            "f16 keys into a bf16 cache",
            dict(
                dtype="f16",
                kv="bf16",
                tokens=6,
                pages=3,
                page=2,
                heads=2,
                head_dim=32,
                tile=32,
                slots=[5, -1, 0, 3, 1, 4],
                spread=True,
            ),
        ),
        (
            "f32 keys into a bf16 cache",
            dict(
                dtype="f32",
                kv="bf16",
                tokens=4,
                pages=2,
                page=4,
                heads=3,
                head_dim=8,
                tile=16,
                slots=[7, 2, 4, 0],
                spread=True,
            ),
        ),
    ]
    # The interpreter truncates a cast into BF16, so a case that narrows into a BF16 cache is the
    # device's alone.
    left = [
        what
        for what, case in cases
        if case.get("kv") == "bf16" and case["dtype"] != "bf16"
    ]
    if not options.device:
        cases = [(what, case) for what, case in cases if what not in left]

    total_written = total_flagged = total_skipped = total_rounded = 0
    for what, case in cases:
        written, flagged, skipped, rounded_elements = check(what, **case)
        total_written += written
        total_flagged += flagged
        total_skipped += skipped
        total_rounded += rounded_elements

    where = "the device" if options.device else "Triton's interpreter"
    print(
        f"ok on {where}: {len(cases)} scatters, {total_written} rows written and bit for bit in "
        f"place, {total_rounded} key elements rounded to nearest even into a narrower cache, "
        f"{total_flagged} pages flagged, {total_skipped} skipped slots that wrote nothing, "
        f"no other position moved, and no flag written past its page's byte"
    )
    if not options.device:
        print(f"{len(left)} cases are left for --device: {', '.join(left)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
