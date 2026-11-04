#!/usr/bin/env python3
"""Check `kv_quantize`: the FP8 cache page a token's row is written into, and the rows nothing writes.

    crates/program-server/python/.venv/bin/python kv_check.py            # interpreter: what must not move
    crates/program-server/python/.venv/bin/python kv_check.py --device   # a device: the codes themselves

The operator is **lossy by contract**: it stores `round_e4m3(source / scale)` and the read side
multiplies the same scale back, so nothing downstream can tell a rounded-wrongly value from a
correctly rounded one. Two different questions follow, and they are answerable in different places.

## What the interpreter answers: the rows and pages nothing writes

A padding slot (`-1`) must write **nothing** — not its page, not its row — because a padding token
that zeroed its page would corrupt another step's tokens that share it; and `initialized` must name
only the pages this launch actually touched, because a reader uses it to tell a written page from a
stale one. Both are statements about *absence*, and the interpreter keeps them exactly: the cache is
pre-filled with a sentinel and every byte of a padding slot's row, and every flag of an untouched
page, must still hold it. What the interpreter cannot answer is the *value* of a live write: it
truncates every narrowing float cast (measured: `0.1` becomes the bf16 `0.099609375`, and `1.9375`
becomes `1.0` instead of `2.0`), so those bytes are not compared here.

## What the device answers: every E4M3 code, and why all of them

The quotient's code is the whole content of the operator, and no tolerance bounds it. E4M3 has
exactly 256 codes, two of which are NaN patterns that no arithmetic produces, so the device half
enumerates the **254 finite ones**: a source whose quotient by each page's scale is exactly each
representable value, written through the real path and compared byte for byte.
Every rounding, every subnormal and saturation is covered by construction, and a wrong rounding
(truncation instead of round-to-nearest-even) shows up as a byte rather than as a small difference.

Four properties ride along with the grid, all of them observable in bytes:

* **the page comes from the slot**, not from the token — two tokens of one page land in that page's
  rows at their own offsets, and the scale applied is the page's;
* **a row wider than one block** is written by several programs, so a dropped or doubled block shows;
* **`initialized` names only the pages this launch touched**;
* **a padding slot writes nothing**, the same assertion the interpreter half makes, kept beside the
  grid so one device run covers both.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

#: Triton reads `TRITON_INTERPRET` when it is imported, and the kernel's import is the one that
#: counts: `--device` compiles for the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch

#: The sentinel a padding slot must leave in the cache. Two values are used across cases, so a
#: program that happened to write one of them cannot hide a write behind the other.
SENTINELS = (0x7E, 0x3C)

#: The page scales the grid runs through; two are not powers of two, so a scale applied to the wrong
#: page is visible in the quotient.
SCALES = (1.0, 0.5, 2.0, 3.7)

#: Every code of E4M3 except the two NaN patterns (`0x7F`, `0xFF`), which no arithmetic produces: 254.
FINITE_CODES = [code for code in range(256) if code not in (0x7F, 0xFF)]


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "kv_quantize.py"
    spec = importlib.util.spec_from_file_location("kv_quantize_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_values():
    """The value of each finite E4M3 code, in FP32, in code order.

    Read from the format itself through torch's own decoder rather than written down, so a table
    that disagreed with the format could not become the reference.
    """
    codes = torch.tensor(FINITE_CODES, dtype=torch.uint8).view(torch.float8_e4m3fn)
    return codes.to(torch.float32)


def encode(values):
    """The E4M3 bytes of FP32 values, through torch's cast (round to nearest even)."""
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def case_tensors(case, device):
    """The source rows, slots, scales and the pre-filled cache of one case."""
    rows, page, heads, head_dim = case["tokens"], case["page"], case["heads"], case["head_dim"]
    width = heads * head_dim
    sentinel = case["sentinel"]
    slots = torch.tensor(case["slots"], dtype=torch.int32, device=device)
    # One scale per page, held as the token's own factor: the kernel reads `scales[page]`, and the
    # slot decides the page, so a case that gave two tokens of one page different factors would be
    # describing a call the operator cannot express. `per_token` is what the reference divides by.
    per_token = torch.ones((rows,), dtype=torch.float32, device=device)
    for token, slot in enumerate(case["slots"]):
        if slot < 0:
            continue
        per_token[token] = case["slot_scales"][slot]

    source = case["source"](rows, width, per_token, device)
    cache_bytes = torch.full(
        (case["pages"] * page * width,), sentinel, dtype=torch.uint8, device=device
    )
    flags = torch.zeros((case["pages"],), dtype=torch.int32, device=device)
    return source, slots, per_token, cache_bytes, flags, width


def write_reference(case, source, per_token, cache_bytes, flags, width):
    """What the cache and the flags must hold after the launch, as bytes, in torch."""
    page, width = case["page"], width
    sentinel = case["sentinel"]
    expected = torch.full_like(cache_bytes, sentinel)
    expected_flags = torch.zeros_like(flags)
    quotients = source.to(torch.float32) / per_token[:, None]
    codes = encode(quotients)
    for token, slot in enumerate(case["slots"]):
        if slot < 0:
            continue
        start = slot * width
        expected[start : start + width] = codes[token]
        expected_flags[slot // page] = 1
    return expected, expected_flags


def run(module, case, compare_codes):
    """One launch, and the assertions the case states.

    `compare_codes` is what the two halves differ by: the interpreter gauges only the bytes that must
    not move, the device every byte of the cache.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else None
    source, slots, per_token, cache_bytes, flags, width = case_tensors(case, device)
    before = cache_bytes.clone()
    out = cache_bytes.view(torch.float8_e4m3fn)
    blocks = (width + case["block"] - 1) // case["block"]
    module._kv_quantize_kernel[(case["tokens"], blocks)](
        source,
        slots,
        per_token,
        out,
        flags,
        case["tokens"],
        case["page"],
        case["heads"],
        case["head_dim"],
        # The call carries the dependent-launch flag as an integer; Triton's interpreter cannot
        # infer a type for a Python bool and stops on it, and the kernel's own copy is `PDL`.
        int(case["pdl"]),
        BLOCK=case["block"],
        PDL=int(case["pdl"]),
    )

    where = (
        f"{case['what']}: {case['tokens']} tokens, page {case['page']}, "
        f"{case['heads']}x{case['head_dim']} heads, block {case['block']}, "
        f"{sum(1 for one in case['slots'] if one < 0)} padding slots"
    )
    expected, expected_flags = write_reference(
        case, source, per_token, cache_bytes, flags, width
    )
    if compare_codes:
        apart = int((cache_bytes != expected).sum())
        if apart:
            position = int((cache_bytes != expected).nonzero()[0])
            raise SystemExit(
                f"{where}: {apart} of {cache_bytes.numel()} cache bytes differ from the reference; "
                f"first at {position}: {int(cache_bytes[position])} vs {int(expected[position])}"
            )
    else:
        # Only the bytes no live slot names: a padding slot's whole row, and every byte of a page
        # whose every slot is a padding slot.
        padding = [slot for slot in case["slots"] if slot < 0]
        if not padding:
            raise SystemExit(
                f"{where}: the interpreter half checks what a padding slot leaves alone, and this "
                f"case names none"
            )
        untouched = torch.isin(
            torch.arange(cache_bytes.numel(), device=cache_bytes.device) // width,
            torch.tensor(padding, dtype=torch.int64, device=cache_bytes.device),
        )
        moved = int((cache_bytes[untouched] != before[untouched]).sum())
        if moved:
            raise SystemExit(
                f"{where}: {moved} bytes of the rows no live slot names changed; a padding slot "
                f"writes nothing"
            )
    flag_difference = int((flags != expected_flags).sum())
    if flag_difference:
        position = int((flags != expected_flags).nonzero()[0])
        raise SystemExit(
            f"{where}: {flag_difference} of {flags.numel()} page flags differ; page {position} is "
            f"{int(flags[position])} and must be {int(expected_flags[position])}"
        )
    written = int((cache_bytes != case["sentinel"]).sum())
    return written


def grid_source(rows, width, per_token, device):
    """A source whose quotient by its token's scale is exactly each finite E4M3 value.

    Every row holds one turn of the 255 finite values, and a row's scale is the one its slot's page
    carries, so the quotient the kernel forms is the code's own value and the cast has exactly one
    right answer.
    """
    codes = finite_values().to(device)
    repeats = (width + codes.numel() - 1) // codes.numel()
    row = codes.repeat(repeats)[:width]
    return (row[None, :] * per_token[:, None]).to(torch.float32)


def pattern_source(rows, width, per_token, device):
    """A source of ordinary values, signed and of mixed magnitude, for the page and padding cases."""
    generator = torch.Generator(device="cpu").manual_seed(7)
    values = torch.rand((rows, width), generator=generator, dtype=torch.float32) * 4.0 - 2.0
    return (values.to(device) * per_token[:, None]).to(torch.float32)


def padding_case(what, sentinel, slots=(0, 16, -1, 32, 33), slot_scales=None):
    """A span of three pages with a padding slot in the middle: what must not move, and why.

    The default slot list puts two tokens of the third page at their own offsets — so the page is
    the slot's and not the token's — and leaves the second page's other rows alone; a case that
    names no slot of page one leaves that page's flag at zero, which is how "only the pages this
    launch touched" is observable at all.
    """
    page, heads, head_dim = 16, 8, 128
    width = heads * head_dim
    return dict(
        what=what,
        sentinel=sentinel,
        page=page,
        pages=3,
        heads=heads,
        head_dim=head_dim,
        block=256,
        pdl=False,
        tokens=len(slots),
        slots=slots,
        # Every page has its own factor, so a kernel that took the token's page instead of the
        # slot's would divide by another page's scale.
        slot_scales=slot_scales or {0: 1.0, 16: 0.5, 32: 2.0, 33: 2.0},
        source=pattern_source,
    )


def grid_case(scale, block, pdl):
    """One turn of the exhaustive code grid through a page whose scale is `scale`."""
    page, heads, head_dim = 16, 8, 128
    width = heads * head_dim
    slots = [0, 16]
    return dict(
        what=f"every finite E4M3 code, page scale {scale}, block {block}"
        + (", dependent launch" if pdl else ""),
        sentinel=SENTINELS[0],
        page=page,
        pages=2,
        heads=heads,
        head_dim=head_dim,
        block=block,
        pdl=pdl,
        tokens=len(slots),
        slots=slots,
        slot_scales={0: scale, 16: scale},
        source=grid_source,
    )


#: Cases the interpreter can answer: what must be left alone.
INTERPRETER_CASES = [
    padding_case("a padding slot between two live ones", SENTINELS[0]),
    padding_case("a padding slot with another sentinel byte", SENTINELS[1]),
    # A page no slot names: its whole width must stay the sentinel and its flag must stay zero, so a
    # kernel that flagged every page would be caught here and nowhere else.
    padding_case(
        "a page no slot names",
        SENTINELS[0],
        slots=(0, -1, -1, 32, 33),
        slot_scales={0: 1.0, 32: 2.0, 33: 2.0},
    ),
]

#: Cases only a device can answer: the codes, and the properties that ride with them.
DEVICE_CASES = [
    grid_case(scale, 256, False) for scale in SCALES
] + [
    grid_case(SCALES[0], 1024, False),
    grid_case(SCALES[1], 128, True),
    padding_case("a padding slot with the codes compared", SENTINELS[0]),
    padding_case(
        "a page no slot names, with the codes compared",
        SENTINELS[1],
        slots=(0, -1, -1, 32, 33),
        slot_scales={0: 1.0, 32: 2.0, 33: 2.0},
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/kv_quantize.py",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="run the cases that need a device: the exhaustive E4M3 grid and the exact comparison",
    )
    options = parser.parse_args()
    if options.device and not torch.cuda.is_available():
        print(
            "skipped: needs a device — the code grid compares stored E4M3 bytes and the interpreter "
            "truncates every narrow cast, so its bytes would be the tool's and not the kernel's; "
            "without --device the script runs the half that needs no device"
        )
        return 0
    module = load_kernel(options.package)
    cases = DEVICE_CASES if options.device else INTERPRETER_CASES
    written = 0
    for case in cases:
        written += run(module, case, compare_codes=options.device)
    where = "the device" if options.device else "Triton's interpreter"
    print(
        f"ok on {where}: {len(cases)} cases, {written} cache bytes written where they must be, "
        f"nothing written where a padding slot says so, and every page flag as the slots make it"
    )
    if not options.device:
        print(
            f"{len(DEVICE_CASES)} cases are left for --device: they compare the stored E4M3 codes "
            f"against all {len(FINITE_CODES)} finite values of the format, and the interpreter truncates a narrow "
            f"store, so a green from it would be a statement about the tool"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
