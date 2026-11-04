#!/usr/bin/env python3
"""Run `kv_write`'s kernel and check the paged-cache scatter, on the CPU.

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python kv_write_check.py [--package DIR]

The operator scatters one row per token into the page slot the scheduler assigned it:

```text
key, value        [tokens, heads, head_dim]
key_cache, value_cache   [pages, page, heads, head_dim]
slots             [tokens] i32, the flat `page · page_size + offset`, or -1 to skip
initialized       [pages, 1, 1, 1] i32, raised to 1 by the write
```

and the operator's tolerance is `exact()`: the kernel **moves** values, so two implementations agree
bit for bit and every assertion here is equality, not a bound.

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
  one, shows.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

# This is an interpreter check: the arithmetic is compared against a CPU reference, so the kernel runs
# on Triton's interpreter even where a device exists. The flag is set before any kernel module is
# imported, because Triton reads it when it launches.
if os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    print("running on Triton's interpreter (no device needed)")

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

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


def launch(module, *, tokens, pages, page, heads, head_dim, tile, dtype, slots, seed=7):
    """One scatter, with the grid the operator computes: `[tokens, ceil(row / tile)]`.

    Returns `(key, value, key_cache, value_cache, key_initialized, value_initialized)`.
    """
    element = DTYPES[dtype]
    torch.manual_seed(seed)
    row = heads * head_dim
    # Distinct in every element: a row written at the wrong place cannot be mistaken for this one.
    key = torch.randn(tokens, heads, head_dim, dtype=torch.float32).to(element)
    value = torch.randn(tokens, heads, head_dim, dtype=torch.float32).to(element)
    key_cache = torch.full((pages, page, heads, head_dim), CACHE_SENTINEL, dtype=element)
    value_cache = torch.full((pages, page, heads, head_dim), CACHE_SENTINEL, dtype=element)
    key_initialized = torch.full((pages, 1, 1, 1), FLAG_SENTINEL, dtype=torch.int32)
    value_initialized = torch.full((pages, 1, 1, 1), FLAG_SENTINEL, dtype=torch.int32)
    slots = torch.tensor(slots, dtype=torch.int32)

    module._kv_write_kernel[(tokens, (row + tile - 1) // tile, 1)](
        key, value, key_cache, value_cache, slots,
        key_initialized, value_initialized,
        tokens, page, heads, head_dim, 0,
        TILE=tile, PDL=False,
    )
    return key, value, key_cache, value_cache, key_initialized, value_initialized


def check(what, *, dtype, tokens, pages, page, heads, head_dim, tile, slots):
    """One scatter against the layout the operator documents, element for element."""
    (key, value, key_cache, value_cache,
     key_initialized, value_initialized) = launch(
        MODULE, tokens=tokens, pages=pages, page=page, heads=heads, head_dim=head_dim,
        tile=tile, dtype=dtype, slots=slots,
    )
    where = f"{what}: {dtype} tokens {tokens} pages {pages} page {page} heads {heads} " \
            f"head_dim {head_dim} tile {tile} slots {slots}"

    # Every position, before and after: the writes land on the slots, and nothing else moves.
    expected_key = torch.full_like(key_cache, CACHE_SENTINEL)
    expected_value = torch.full_like(value_cache, CACHE_SENTINEL)
    expected_flags = torch.zeros_like(key_initialized)
    written = 0
    for token, slot in enumerate(slots):
        if slot < 0:
            continue
        page_of, offset_of = divmod(slot, page)
        if page_of >= pages:
            raise SystemExit(f"{where}: slot {slot} names page {page_of} of {pages}")
        expected_key[page_of, offset_of] = key[token]
        expected_value[page_of, offset_of] = value[token]
        expected_flags[page_of] = 1
        written += 1

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
    # The counts, so a case that wrote nothing or flagged nothing cannot pass quietly.
    if written and int(expected_flags.sum()) == 0:
        raise SystemExit(f"{where}: the case writes {written} tokens and flags no page")
    return written, int(expected_flags.sum()), tokens - written


def main():
    global MODULE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/kv_write.py",
    )
    options = parser.parse_args()
    MODULE = load_kernel(options.package)

    cases = [
        # Slots out of token order, one -1, and page 2 receives nothing at all. `tile` equals a row,
        # so each token is one program.
        ("ordered slots, one skipped, one page empty", dict(
            dtype="f32", tokens=6, pages=4, page=4, heads=2, head_dim=8, tile=16,
            slots=[5, 0, 7, -1, 2, 3])),
        # A row split across four programs, and the flag raised by only the first of them.
        ("a row split across four programs", dict(
            dtype="f32", tokens=4, pages=5, page=2, heads=4, head_dim=8, tile=8,
            slots=[3, 6, -1, 0])),
        # `tile` wider than a row: one program per token, the mask covering the rest of it.
        ("a tile wider than the row", dict(
            dtype="bf16", tokens=5, pages=3, page=4, heads=2, head_dim=4, tile=64,
            slots=[11, 4, 0, 8, -1])),
        # Every token skipped: nothing written, no flag raised.
        ("no token persisted", dict(
            dtype="f16", tokens=3, pages=2, page=2, heads=1, head_dim=8, tile=8,
            slots=[-1, -1, -1])),
        # One token per page exactly, the fixture's narrow types, and a page size of one.
        ("one token per page, page size one", dict(
            dtype="bf16", tokens=4, pages=4, page=1, heads=2, head_dim=16, tile=32,
            slots=[3, 1, 0, 2])),
        ("f16 row split with a partially filled tail", dict(
            dtype="f16", tokens=3, pages=2, page=3, heads=3, head_dim=5, tile=4,
            slots=[4, 1, 2])),
    ]
    total_written = total_flagged = total_skipped = 0
    for what, case in cases:
        written, flagged, skipped = check(what, **case)
        total_written += written
        total_flagged += flagged
        total_skipped += skipped

    print(
        f"ok: {len(cases)} scatters, {total_written} rows written and bit for bit in place, "
        f"{total_flagged} pages flagged, {total_skipped} skipped slots that wrote nothing, "
        f"and no other position moved"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
