#!/usr/bin/env python3
"""Check `page_conversion`'s kernel: its scale rows on Triton's interpreter, its decode on a device.

    crates/program-server/python/.venv/bin/python page_conversion_check.py            # interpreter
    crates/program-server/python/.venv/bin/python page_conversion_check.py --device   # a device

The script has two halves because the two things it checks lie in different places.

An imported rank's cache arrives as raw keys and values in the source's own element type, with one
FP32 scale per source page and head group when the source stores FP8. This program decodes one field
of a span into the FP32 scratch block `cache_copy_region` then writes pages from. Two things in it can
be wrong without any build noticing:

* **which scale row a token takes.** The caller packs one row per page the span *touches*, starting
  with the page `source_offset` falls in, so a span that starts mid-page must not skip that many rows
  (`page − source_offset // block_size`). Reading the wrong row scales a whole page of keys by another
  page's factor: the attention comes out wrong and every number still looks like a number. The page
  cases below start the span inside a page, cross a page boundary, and start in a later page.

* **the rounding the decoded value takes.** A source page holds `key / scale`; the value times the
  page's scale is the value the source rank wrote, and it must then round through the source's
  *logical* type so an imported value equals a locally written one instead of being one rounding
  better. The check requires the FP32 block to equal a rounding model **exactly**, and proves the case
  separates the models before trusting the answer: BF16 truncates a mantissa in the interpreter and F16
  rounds to nearest — measured, not assumed — and each narrow case is built so the two models differ.

The scratch is `20·E + 8·S` bytes, and this check asserts every byte it does not name is untouched:
the FP32 block past the span, the raw areas, and the scale rows.

**The halves.** The scale-row arithmetic is integer and exact in FP32, so the interpreter runs it and
the answer is a statement about the kernel. The decode is not: Triton's interpreter garbles an E4M3
store (measured: `-126.4941` reads back `-64`, where the format says `-128`), so a green on an FP8
source would be a statement about the tool, and the cases that decode E4M3 or round to a narrow
logical type run only under `--device`. Without a device the script runs its half and says in one
sentence which cases it left, and why.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import struct
import sys

#: Triton reads `TRITON_INTERPRET` when it is imported, so the mode is chosen from the command line
#: before the import below: `--device` compiles for the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402  (after TRITON_INTERPRET, which Triton reads at import)
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

FIELD_KEYS = 0
FIELD_VALUES = 1

#: The stored element types the descriptor names, with Triton's spelling and the bytes one takes.
SOURCES = {
    "f8e4m3": (tl.float8e4nv, 1, torch.float8_e4m3fn),
    "bf16": (tl.bfloat16, 2, torch.bfloat16),
}

#: The source's logical types, which decide the rounding the decoded value takes.
COMPUTES = {
    "bf16": (tl.bfloat16, torch.bfloat16),
    "f16": (tl.float16, torch.float16),
    "f32": (tl.float32, torch.float32),
}

#: The sentinel the FP32 block is primed with, so a program that writes where it should not shows.
SENTINEL = -1234.5


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "page_conversion.py"
    spec = importlib.util.spec_from_file_location("page_conversion_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_scratch(case):
    """The workspace bytes a case starts from, and the tensors it holds the source in."""
    span = case["count"] * case["layers"] * case["heads"] * case["head_dim"]
    groups = 1 if not case["scaled"] else (case["heads"] + case["scale_heads"] - 1) // case["scale_heads"]
    scale_entries = 0 if not case["scaled"] else case["pages"] * case["layers"] * groups
    elements = span
    total = 20 * elements + 8 * scale_entries
    scratch = torch.zeros(total, dtype=torch.uint8)

    generator = torch.Generator().manual_seed(case["seed"])
    shape = (case["count"], case["layers"], case["heads"], case["head_dim"])
    # The stored elements, in the source's own type: the source wrote `value / scale`.
    source = SOURCES[case["source"]][2]
    magnitude = {"f8e4m3": 32.0, "bf16": 512.0}[case["source"]]
    keys = (torch.rand(shape, generator=generator, dtype=torch.float32) * magnitude - magnitude / 2)
    values = (torch.rand(shape, generator=generator, dtype=torch.float32) * magnitude - magnitude / 2)
    keys = keys.to(source)
    values = values.to(source)
    # Each raw area is eight E bytes, which is more than one field's span needs in any source type:
    # the span is written at its start and the rest of the area stays zero, which the run then checks.
    keys_bytes = keys.contiguous().view(torch.uint8).reshape(-1)
    values_bytes = values.contiguous().view(torch.uint8).reshape(-1)
    scratch[0:keys_bytes.numel()] = keys_bytes
    scratch[8 * elements:8 * elements + values_bytes.numel()] = values_bytes

    # The FP32 block starts at 16E, and the scale rows past 20E, every one of them a distinct factor
    # so a program that reads the wrong row cannot pass.
    block = scratch[16 * elements:20 * elements].view(torch.float32)
    block.fill_(SENTINEL)
    scales = None
    if case["scaled"]:
        # Scales the way a real FP8 cache carries them — a factor near 1/100 that no mantissa of a
        # narrow logical type covers — so the product is a value that *must* round, and every row is
        # its own factor so a program reading the wrong row cannot pass.
        scales = (torch.rand((case["pages"], case["layers"], groups), generator=generator,
                             dtype=torch.float32) * 0.02)
        scratch[20 * elements:20 * elements + 4 * scale_entries] = (
            scales.contiguous().view(torch.uint8).reshape(-1))
    return scratch, keys, values, scales, elements, scale_entries


def scale_row(case, token):
    """The scale row a token of the span takes, as the caller packs the rows."""
    page = (case["source_offset"] + token) // case["source_block_size"]
    return page - case["source_offset"] // case["source_block_size"]


def reference(case, keys, values, scales):
    """The FP32 block the kernel must write, in FP32: the product, then the logical type's rounding."""
    stored = (keys if case["field"] == FIELD_KEYS else values).to(torch.float32)
    decoded = stored.clone()
    if case["scaled"]:
        groups = (case["heads"] + case["scale_heads"] - 1) // case["scale_heads"]
        for token in range(case["count"]):
            row = scales[scale_row(case, token)]
            for head in range(case["heads"]):
                decoded[token, :, head, :] *= row[:, head // case["scale_heads"]][:, None]
    if case["compute"] == "f32":
        return decoded
    return decoded.to(COMPUTES[case["compute"]][1]).to(torch.float32)


def truncation_model(values, compute):
    """The same values rounded toward zero to the logical type: the other model a cast can follow."""
    if compute == "bf16":
        bits = values.contiguous().view(torch.int32) & ~0xFFFF
        return bits.view(torch.float32)
    narrow = values.to(torch.float16).view(torch.int16)
    # Clearing the low thirteen bits of a half's pattern lowers its magnitude to the next
    # representable value at or below it, which is truncation toward zero for a normal half.
    return (narrow & ~0x1FFF).view(torch.float16).to(torch.float32)


def run(module, case, models=("nearest", "truncated")):
    """One launch, and what the kernel wrote against the reference.

    `models` is the roundings the run may show: the operator's rule is round-to-nearest through the
    logical type, and only the interpreter is allowed the truncating model as well, because a narrow
    store through it truncates.
    """
    scratch, keys, values, scales, elements, scale_entries = case_scratch(case)
    original = scratch.clone()
    source, source_bytes, _ = SOURCES[case["source"]]
    compute = COMPUTES[case["compute"]][0]
    tiles = (case["heads"] + case["head_dim"] - 1) // case["head_dim"]
    module._page_conversion_kernel[(case["count"] * case["layers"], tiles)](
        scratch, elements, scale_entries, case["count"], case["source_offset"],
        case["source_block_size"], case["scale_heads"] if case["scaled"] else 0, case["pages"],
        FIELD=case["field"], SOURCE=source, COMPUTE=compute, HEAD_DIM=case["head_dim"],
        HEADS=case["heads"], LAYERS=case["layers"], SCALED=case["scaled"],
        SOURCE_BYTES=source_bytes,
    )

    where = (f"{case['what']}: {case['source']} -> {case['compute']}, "
             f"{'values' if case['field'] else 'keys'}, "
             f"{'scaled' if case['scaled'] else 'dense'}, "
             f"{case['count']} tokens from offset {case['source_offset']} of "
             f"{case['source_block_size']}-token pages")

    # Nothing outside the FP32 block may move: the raw areas hold the source's bytes, and the scale
    # rows are the next launch's input.
    block = slice(16 * elements, 20 * elements)
    if not torch.equal(scratch[:16 * elements], original[:16 * elements]):
        raise SystemExit(f"{where}: the raw areas changed; this program writes only the FP32 block")
    if not torch.equal(scratch[20 * elements:], original[20 * elements:]):
        raise SystemExit(f"{where}: the scale bytes changed; this program writes only the FP32 block")

    expected = reference(case, keys, values, scales)
    got = scratch[block].view(torch.float32).reshape(expected.shape)
    if case["compute"] == "f32":
        if not torch.equal(got, expected):
            apart = (got != expected).reshape(-1)
            position = int(apart.nonzero()[0])
            raise SystemExit(
                f"{where}: {int(apart.sum())} of {got.numel()} decoded values differ from the "
                f"reference; first at {position}: {float(got.reshape(-1)[position])!r} vs "
                f"{float(expected.reshape(-1)[position])!r}")
        return got.numel(), "f32"

    exact = reference_exact(case, keys, values, scales)
    if torch.equal(exact.to(COMPUTES[case["compute"]][1]).to(torch.float32), exact):
        # The source type has no more mantissa than the logical type, so the cast cannot round and the
        # answer is exact rather than one of two models.
        if not torch.equal(got, exact):
            apart = (got != exact).reshape(-1)
            position = int(apart.nonzero()[0])
            raise SystemExit(
                f"{where}: {int(apart.sum())} of {got.numel()} decoded values differ from the "
                f"reference although the logical type holds every one exactly; first at "
                f"{position}: {float(got.reshape(-1)[position])!r} vs "
                f"{float(exact.reshape(-1)[position])!r}")
        return got.numel(), "exact"
    nearest = expected
    truncated = truncation_model(exact, case["compute"])
    named = None
    for name, candidate in (("nearest", nearest), ("truncated", truncated)):
        if name not in models:
            continue
        if torch.equal(got, candidate):
            named = name
            break
    if named is None:
        apart = (got != nearest).reshape(-1) & (got != truncated).reshape(-1)
        position = int(apart.nonzero()[0])
        raise SystemExit(
            f"{where}: the FP32 block is neither the nearest-rounding nor the truncating model — "
            f"first at {position}: {float(got.reshape(-1)[position])!r} vs "
            f"{float(nearest.reshape(-1)[position])!r} (nearest) or "
            f"{float(truncated.reshape(-1)[position])!r} (truncated)")
    if len(models) > 1 and torch.equal(nearest, truncated):
        raise SystemExit(
            f"{where}: the nearest and truncating models come out identical, so this case cannot "
            f"tell which rounding the kernel takes and must be rebuilt with a value that sits "
            f"between two representable {case['compute']} numbers")
    return got.numel(), named


def reference_exact(case, keys, values, scales):
    """The decoded product before the logical type's rounding: what the cast is applied to."""
    stored = (keys if case["field"] == FIELD_KEYS else values).to(torch.float32)
    decoded = stored.clone()
    if case["scaled"]:
        for token in range(case["count"]):
            row = scales[scale_row(case, token)]
            for head in range(case["heads"]):
                decoded[token, :, head, :] *= row[:, head // case["scale_heads"]][:, None]
    return decoded


#: Every case states `layers = 32` because the descriptor compiles that depth; `count`, the span's
#: token offset and the page size are launch arguments, so a case may vary them.
CASES = [
    # The served path: an FP8 source, decoded to FP32, its span starting inside a later page — which
    # is what makes the row's `− source_offset // block_size` decide the scale a token takes.
    dict(what="fp8 keys decoded to f32 from inside a later page", field=FIELD_KEYS, source="f8e4m3",
         compute="f32", scaled=True, head_dim=64, heads=4, layers=32, count=2,
         source_block_size=16, source_offset=36, scale_heads=4, pages=1, seed=11),
    # The same span of the values: the second raw area is eight E bytes along, and a program that
    # decoded keys twice would look right on every key case above.
    dict(what="fp8 values decoded to f32 from inside a later page", field=FIELD_VALUES,
         source="f8e4m3", compute="f32", scaled=True, head_dim=64, heads=4, layers=32, count=2,
         source_block_size=16, source_offset=36, scale_heads=4, pages=1, seed=12),
    # One page per token, two head groups: a token that took the other token's row lands on a
    # different factor, and the bf16 rounding model is measured here.
    dict(what="fp8 keys to bf16 across two page rows", field=FIELD_KEYS, source="f8e4m3",
         compute="bf16", scaled=True, head_dim=64, heads=4, layers=32, count=2,
         source_block_size=1, source_offset=5, scale_heads=2, pages=2, seed=13),
    dict(what="bf16 keys to bf16 from page zero", field=FIELD_KEYS, source="bf16", compute="bf16",
         scaled=True, head_dim=64, heads=4, layers=32, count=2, source_block_size=16,
         source_offset=0, scale_heads=4, pages=1, seed=14),
    # Dense sources carry no scales: the value is widened and rounded, and the span starts on the
    # last token of a page so it crosses into the next.
    dict(what="dense bf16 keys to f16 across a page boundary", field=FIELD_KEYS, source="bf16",
         compute="f16", scaled=False, head_dim=64, heads=4, layers=32, count=2,
         source_block_size=16, source_offset=15, scale_heads=0, pages=2, seed=15),
    dict(what="dense bf16 keys to f32", field=FIELD_KEYS, source="bf16", compute="f32",
         scaled=False, head_dim=64, heads=4, layers=32, count=2, source_block_size=16,
         source_offset=0, scale_heads=0, pages=1, seed=16),
    # The scale-row rule in exact arithmetic: a bf16 source multiplied by an FP32 factor is an FP32
    # product with no rounding to argue about, so the interpreter's answer is the kernel's.
    dict(what="bf16 keys to f32 from inside a later page", field=FIELD_KEYS, source="bf16",
         compute="f32", scaled=True, head_dim=64, heads=4, layers=32, count=3,
         source_block_size=16, source_offset=36, scale_heads=4, pages=1, seed=17),
    dict(what="bf16 values to f32 across two page rows", field=FIELD_VALUES, source="bf16",
         compute="f32", scaled=True, head_dim=64, heads=4, layers=32, count=2,
         source_block_size=1, source_offset=5, scale_heads=2, pages=2, seed=18),
]

#: The cases the interpreter can answer: no E4M3 store, and no rounding into a narrow type.
INTERPRETER_CASES = [
    case for case in CASES if case["source"] == "bf16" and case["compute"] == "f32"
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/page_conversion.py",
    )
    parser.add_argument(
        "--device",
        action="store_true",
        help="compile for the device and run every case, including the ones that decode E4M3 or "
        "round into a narrow logical type",
    )
    options = parser.parse_args()
    if options.device and not torch.cuda.is_available():
        raise SystemExit(
            "the device half decodes E4M3 and rounds into a narrow type, which needs the device the "
            "module was built for; run this without --device for the scale-row half"
        )
    module = load_kernel(options.package)
    cases = CASES if options.device else INTERPRETER_CASES
    models = ("nearest",) if options.device else ("nearest", "truncated")
    checked = 0
    seen = {}
    for case in cases:
        count, model = run(module, case, models)
        checked += count
        seen[model] = seen.get(model, 0) + 1
    named = ", ".join(f"{model}: {count}" for model, count in sorted(seen.items()))
    where = "the device" if options.device else "Triton's interpreter"
    print(f"ok on {where}: {checked} decoded values over {len(cases)} cases, every one the "
          f"reference and every byte outside the FP32 block untouched — {named}")
    if not options.device:
        left = len(CASES) - len(cases)
        print(
            f"{left} cases are left for --device: they decode E4M3 or round into a narrow logical "
            f"type, and Triton's interpreter garbles an E4M3 store, so a green from it would be a "
            f"statement about the tool and not about the kernel"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
