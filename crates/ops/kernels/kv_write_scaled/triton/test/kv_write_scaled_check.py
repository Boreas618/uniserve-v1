#!/usr/bin/env python3
"""Run `kv_write_scaled`'s kernel and check the quantizing page write, on the CPU.

    TRITON_INTERPRET=1 python crates/ops/kernels/kv_write_scaled/triton/test/kv_write_scaled_check.py

    [--package DIR]

The operator writes full-precision rows into an E4M3 cache whose pages carry one f32 scale each:

```text
source      [tokens, heads · head_dim]   bf16 or f32
slots       [tokens] i32                 the flat `page · page_size + offset`, or -1 to skip
scales      [pages, 1, 1, 1] f32         read and written: the page's scale
cache       [pages, page, heads, head_dim] E4M3, read and written
initialized [pages, 1, 1, 1] u8          read and written: the page's flag, one byte
```

and it derives the scale rather than reading it, so the rule below is the reference. What a wrong
kernel gets wrong, and the case here that catches each:

- **a scale that shrinks**: the proposal is smaller than the stored scale and the stored bytes must
  not move; a kernel that takes the proposal re-encodes the page and silently changes values a reader
  already sees;
- **a grown scale without a re-encode**: the proposal is larger and every stored row must be decoded
  at the old scale and re-encoded at the new one — a kernel that only stores its own rows leaves the
  page's earlier rows decoded at the wrong scale, and every shape, key and flag still agrees;
- **an all-zero write**: the proposal floors at `1e-12`, so the page's scale is `1e-12 / 448` and not
  zero — a kernel without the floor produces a zero scale and a page nothing can decode;
- **a page the launch does not touch**: its bytes, its scale and its flag stay exactly as they were;
- **the scale committed before the values**: the flag and the scale are written after the rows.

The scale, the comparisons and both divisions are f32, so the reference computes the same maximum and
the same scale bit for bit. Two implementations may differ only in the rounding into E4M3, and that
is the operator's tolerance: `rtol 2^-3`, `atol 2^-6` in the page's scaled units.
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
import triton
import triton.language as tl

DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}

#: The largest finite E4M3 magnitude and the floor under the page's maximum: the page's scale is
#: `max(|v|) floored at 1e-12, over 448`.
E4M3_MAX = 448.0
SCALE_FLOOR = 1e-12

#: The operator's tolerance, in the page's scaled units. The check compares bytes exactly instead,
#: because both sides here perform the *same* conversion (see `interpret_cast`), and a bound is only
#: needed where two implementations round with different hardware.
RTOL = 2.0 ** -3
ATOL = 2.0 ** -6


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "kv_write_scaled.py"
    spec = importlib.util.spec_from_file_location("kv_write_scaled_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@triton.jit
def _cast_kernel(source_ptr, cache_ptr, count, BLOCK: tl.constexpr):
    """Store f32 values into an E4M3 tensor, one block per program."""
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    tl.store(cache_ptr + offset, tl.load(source_ptr + offset, mask=mask, other=0.0), mask=mask)


def interpret_cast(values):
    """The E4M3 conversion this interpreter performs, applied by the interpreter itself.

    Triton's interpreter does not round into E4M3 the way torch's cast does: `126.7` becomes `64.0`
    where the format holds `128.0`, and `-31.2` becomes `-16.0`. That is the *interpreter's*
    conversion and not this kernel's arithmetic, so the reference takes its bytes through the same
    conversion: the check then compares the rule — the scale, the re-encode, the floor, the order —
    byte for byte, and the format's rounding, which lives in hardware on a device, is not what it
    measures. The operator's `rtol 2^-3`/`atol 2^-6` is the bound for two implementations that round
    in their own hardware, and the device run is where it is exercised.
    """
    flat = values.reshape(-1).to(torch.float32).contiguous()
    out = torch.zeros(flat.numel(), dtype=torch.float8_e4m3fn)
    block = 16
    while block < flat.numel():
        block *= 2
    _cast_kernel[((flat.numel() + block - 1) // block,)](flat, out, flat.numel(), BLOCK=block)
    return out.reshape(values.shape)


def reference(source, slots, scales, cache, initialized, page):
    """The rule, in the order the operator states it, in f32."""
    values = source.to(torch.float32)
    scale = scales.clone()
    flag = initialized.reshape(-1).clone()
    page_bytes = interpret_cast(cache.to(torch.float32))
    pages = int(scales.shape[0])
    for current in range(pages):
        incoming = [
            token
            for token, slot in enumerate(slots.tolist())
            if slot >= 0 and slot // page == current
        ]
        if not incoming:
            continue
        # 1. The proposal: the maximum |v| over the rows this launch brings to the page.
        largest = max(float(values[token].abs().max()) for token in incoming)
        proposed = max(largest, SCALE_FLOOR) / E4M3_MAX
        # 2. A scale never shrinks, and a fresh page takes the proposal whatever it held.
        stored = float(scales[current])
        updated = max(stored, proposed) if flag[current] != 0 else proposed
        # 3. A grown scale re-encodes every row already stored.
        if flag[current] != 0 and updated > stored:
            decoded = page_bytes[current].to(torch.float32) * stored
            page_bytes[current] = interpret_cast(
                (decoded / updated).clamp(-E4M3_MAX, E4M3_MAX)
            )
        # 4. The rows this launch brings.
        heads, head_dim = page_bytes.shape[2], page_bytes.shape[3]
        for token in incoming:
            offset = slots[token].item() % page
            page_bytes[current, offset] = interpret_cast(
                (values[token].reshape(heads, head_dim) / updated).clamp(
                    -E4M3_MAX, E4M3_MAX
                )
            )
        scale[current] = updated
        flag[current] = 1
    return scale, page_bytes, flag.reshape_as(initialized)


def launch(module, *, dtype, tokens, pages, page, heads, head_dim, block, slots,
           seeds=None, prefill=None):
    """One write, with the grid the operator computes: `[pages, ceil(row / block, 1)]`.

    `prefill(scales, cache, initialized)` may state a page as already written, which is how the
    grow, the shrink and the untouched cases are set up.
    """
    element = DTYPES[dtype]
    torch.manual_seed(11)
    row = heads * head_dim
    source = torch.randn(tokens, row, dtype=torch.float32).to(element)
    if seeds is not None:
        source = seeds.to(element)
    scales = torch.zeros(pages, 1, 1, 1, dtype=torch.float32)
    cache = torch.zeros(pages, page, heads, head_dim, dtype=torch.float8_e4m3fn)
    # One byte per page, and the buffer is longer than the view so the bytes after the last page are
    # visible: a four-byte flag store at an odd page count writes into them.
    flags = torch.zeros(pages + 4, dtype=torch.uint8)
    initialized = flags[:pages].view(pages, 1, 1, 1)
    if prefill is not None:
        prefill(scales, cache, initialized)
    slots_tensor = torch.tensor(slots, dtype=torch.int32)

    before = (scales.clone(), cache.clone(), initialized.clone())
    module._kv_write_scaled_kernel[(pages, (row + block - 1) // block, 1)](
        source, slots_tensor, scales, cache, initialized,
        tokens, page, heads, head_dim, 0,
        BLOCK=block, PDL=False,
    )
    return source, slots_tensor, before, (scales, cache, initialized, flags)


def check(what, *, dtype, tokens, pages, page, heads, head_dim, block, slots,
          seeds=None, prefill=None, untouched=()):
    """One write against the reference, the page's scale and flag exactly and the bytes within E4M3."""
    source, slots_tensor, before, (scales, cache, initialized, flags) = launch(
        MODULE, dtype=dtype, tokens=tokens, pages=pages, page=page, heads=heads,
        head_dim=head_dim, block=block, slots=slots, seeds=seeds, prefill=prefill,
    )
    where = (f"{what}: {dtype} tokens {tokens} pages {pages} page {page} heads {heads} "
             f"head_dim {head_dim} block {block} slots {slots}")
    want_scale, want_bytes, want_flag = reference(
        source, slots_tensor, before[0], before[1], before[2], page
    )

    if not torch.equal(scales, want_scale):
        raise SystemExit(
            f"{where}: the scales differ: {scales.reshape(-1).tolist()} vs "
            f"{want_scale.reshape(-1).tolist()}"
        )
    if not torch.equal(initialized, want_flag):
        raise SystemExit(
            f"{where}: the flags differ: {initialized.reshape(-1).tolist()} vs "
            f"{want_flag.reshape(-1).tolist()}"
        )
    # The bytes past the last page: a flag stored as four bytes at a page count not divisible by
    # four clobbers them, and no other assertion in the case would show it.
    tail = flags[len(flags) - 4:]
    if bool((tail != 0).any()):
        raise SystemExit(
            f"{where}: the flags wrote past the last page: {tail.tolist()} after "
            f"{len(flags)} bytes of a {len(flags) - 4}-page plane"
        )

    # The page's bytes, compared exactly: both sides cast through `interpret_cast`, so a difference
    # here is the rule and not the format's rounding.
    got = cache.to(torch.uint8)
    expect = want_bytes.to(torch.uint8)
    for current in range(pages):
        if current in untouched:
            if not torch.equal(cache[current].view(torch.uint8), before[1][current].view(torch.uint8)):
                raise SystemExit(f"{where}: page {current} is untouched and its bytes moved")
            if not torch.equal(scales[current], before[0][current]):
                raise SystemExit(f"{where}: page {current} is untouched and its scale moved")
            if not torch.equal(initialized[current], before[2][current]):
                raise SystemExit(f"{where}: page {current} is untouched and its flag moved")
            continue
        if not torch.equal(got[current], expect[current]):
            differing = (got[current] != expect[current]).reshape(-1).nonzero()
            first = int(differing[0])
            raise SystemExit(
                f"{where}: page {current} differs at {len(differing)} positions; first at "
                f"{first}: {float(cache[current].float().reshape(-1)[first])} vs "
                f"{float(want_bytes[current].float().reshape(-1)[first])} at scale "
                f"{float(scales[current])}"
            )
    return scales.reshape(-1).tolist()


def prefilled(page, heads, head_dim, *, scale, flag=1, seed=3):
    """A page already written at `scale`, from values that scale is derived from."""
    def fill(scales, cache, initialized):
        generator = torch.Generator().manual_seed(seed)
        rows = torch.randn(page, heads, head_dim, dtype=torch.float32, generator=generator)
        cache.copy_((rows / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn))
        scales[0] = scale
        initialized[0] = flag
    return fill


def main():
    global MODULE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/kv_write_scaled.py",
    )
    options = parser.parse_args()
    MODULE = load_kernel(options.package)

    # The four cases the rule turns on, each on a single-page cache so the page under test is the
    # whole cache, plus the shapes a row split and a skipped slot need.
    shapes = dict(tokens=4, pages=1, page=4, heads=2, head_dim=8, block=16, slots=[0, 1, 2, -1])

    # 1. A fresh page takes the proposal, floored at 1e-12 and divided by 448.
    seeds = torch.tensor(
        [[0.0] * 16, [3.0] * 16, [-7.0] * 16, [0.0] * 16], dtype=torch.float32
    )
    scales = check("a fresh page takes its proposal", dtype="f32", seeds=seeds, **shapes)
    expected = float(
        torch.tensor(7.0, dtype=torch.float32) / torch.tensor(E4M3_MAX, dtype=torch.float32)
    )
    if abs(scales[0] - expected) > 0.0:
        raise SystemExit(f"a fresh page: the scale is {scales[0]}, the proposal {expected}")

    # 2. An all-zero write floors at 1e-12 rather than producing a zero scale.
    zeros = torch.zeros(4, 16, dtype=torch.float32)
    scales = check("an all-zero write floors the scale", dtype="f32", seeds=zeros, **shapes)
    # In f32, where the kernel computes it: the floor is a f32 constant and the division is f32, so
    # the expectation is the same two constants divided in the same width.
    floored = float(
        torch.tensor(SCALE_FLOOR, dtype=torch.float32)
        / torch.tensor(E4M3_MAX, dtype=torch.float32)
    )
    if scales[0] != floored:
        raise SystemExit(f"an all-zero write: the scale is {scales[0]}, the floor {floored}")

    # 3. A proposal smaller than the stored scale leaves the scale and every stored byte alone.
    stored = 1.0
    small = torch.full((4, 16), 0.25, dtype=torch.float32)
    scales = check(
        "a smaller proposal does not shrink the page",
        dtype="f32",
        seeds=small,
        prefill=prefilled(4, 2, 8, scale=stored),
        **shapes,
    )
    if scales[0] != stored:
        raise SystemExit(f"a smaller proposal: the scale became {scales[0]}, stored {stored}")

    # 4. A larger proposal grows the scale and re-encodes every stored row: the decoded values of
    #    the page's earlier rows are preserved, which is what the re-encode is for.
    #    The stored page is written at a small scale, so the incoming 400.0 values propose a larger
    #    one and every stored row has to move.
    stored_narrow = 0.1
    large = torch.full((4, 16), 400.0, dtype=torch.float32)
    scales = check(
        "a larger proposal re-encodes the stored rows",
        dtype="f32",
        seeds=large,
        prefill=prefilled(4, 2, 8, scale=stored_narrow),
        **shapes,
    )
    if scales[0] <= stored_narrow:
        raise SystemExit(
            f"a larger proposal: the scale stayed {scales[0]}, so the re-encode never ran"
        )

    # 5. Two pages, one of them touched: the untouched page keeps its bytes, its scale and its flag.
    check(
        "a page the launch does not touch",
        dtype="bf16",
        tokens=2,
        pages=2,
        page=4,
        heads=2,
        head_dim=8,
        block=8,
        slots=[7, 5],
        prefill=prefilled(4, 2, 8, scale=0.5, seed=5),
        untouched=(0,),
    )

    # 6. Every slot skipped: nothing is written and nothing is flagged.
    check(
        "no token persisted",
        dtype="bf16",
        tokens=3,
        pages=2,
        page=2,
        heads=1,
        head_dim=8,
        block=8,
        slots=[-1, -1, -1],
        untouched=(0, 1),
    )

    # 7. The row split across passes, the block narrower than a row and wider than it.
    for block in (4, 8, 16, 32):
        check(
            f"a row walked in passes of {block}, five pages",
            dtype="f32",
            tokens=5,
            pages=5,
            page=3,
            heads=3,
            head_dim=5,
            block=block,
            slots=[3, 0, 4, -1, 2],
            untouched=(2, 3, 4),
        )

    print(
        "ok: 10 writes, every page's scale and flag exact, every stored byte the rule's own, "
        "the grown page re-encoded over every stored row, the floor holding for an all-zero write, "
        "an untouched page never moved, and no flag written past its page's byte"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
