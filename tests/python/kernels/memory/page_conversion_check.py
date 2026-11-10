#!/usr/bin/env python3
"""Check `page_conversion`'s kernel: its scale rows on Triton's interpreter, its decode on a device.

    crates/program-server/python/.venv/bin/python page_conversion_check.py            # interpreter
    crates/program-server/python/.venv/bin/python page_conversion_check.py --device   # a device

The script has two halves because the two things it checks lie in different places.

A page stored in another representation is staged in the conversion scratch in the host page
encoding: layer-major keys `[layers, block_size, heads, head_dim]`, values in the same shape, and for
an FP8 page FP32 scales `[2, layers, groups]` (one group per source pool's head range; the host page's
own tail is `groups = 1`). This program decodes one field of the page into the FP32 scratch block
`cache_copy_region` then writes units from. Two things in it can be wrong without any build noticing:

* **which scale a row takes.** A row of layer `l` of field `f` takes the scale
  `[f, l, head // scale_heads]`. Reading another layer's or the other field's entry scales a layer of
  keys by a foreign factor: the attention comes out wrong and every number still looks like a number.
  Every scale entry of a case is a distinct factor, and the cases cover keys and values, one and two
  scale groups, and more than one layer.

* **the rounding the decoded value takes.** A scaled page holds `key / scale`; the value times its
  scale is the value the source rank wrote, and it must then round through the source's *logical*
  type so a loaded value equals a locally written one instead of being one rounding better. The check
  requires the FP32 block to equal a rounding model **exactly**, and proves the case separates the
  models before trusting the answer: BF16 truncates a mantissa in the interpreter and F16 rounds to
  nearest — measured, not assumed — and each narrow case is built so the two models differ.

The scratch is `20·E + 8·S` bytes, and this check asserts every byte it does not name is untouched:
the raw areas past the page, and the scale entries.

**The interpreter's E4M3 store is wrong and its E4M3 load is not.** Both measured on all 256 patterns
of the format with one two-line kernel apiece:

* a **store** of `-126.4941` writes the byte for `-64` where the format says `-128`, which is the
  garbling that keeps every FP8-*writing* check off the interpreter;
* a **load** widens **254 of 256** patterns to torch's own value — the two that differ are the two NaN
  patterns (`0x7f`/`0xff`, which read back as `±480`), and **all 254 finite patterns agree**.

This kernel only ever *loads* the source's elements and *stores* FP32 into the block, and no case
carries a NaN pattern, so every case below is a statement about the kernel and runs without a device.
The one interpreter quirk in play is the cast into a narrow logical type, which the two-model rule
above measures instead of assuming. `--device` runs the same cases where a device exists, and there
the rounding must be the operator's `nearest`.
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
PACKAGE = KERNELS / "page_conversion" / "triton"

#: Triton reads `TRITON_INTERPRET` when it is imported, so the mode is chosen from the command line
#: before the import below: `--device` compiles for the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402  (after TRITON_INTERPRET, which Triton reads at import)
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
    spec = importlib.util.spec_from_file_location(
        "page_conversion_kernel", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def groups_of(case):
    """The scale groups of a case's page: one per source pool's head range."""
    if not case["scaled"]:
        return 1
    return (case["heads"] + case["scale_heads"] - 1) // case["scale_heads"]


def case_scratch(case):
    """The workspace bytes a case starts from, and the tensors it holds the staged page in."""
    shape = (
        case["layers"],
        case["block_size"],
        case["heads"],
        case["head_dim"],
    )
    elements = shape[0] * shape[1] * shape[2] * shape[3]
    groups = groups_of(case)
    scale_entries = 2 * case["layers"] * groups if case["scaled"] else 0
    total = 20 * elements + 8 * scale_entries
    scratch = torch.zeros(total, dtype=torch.uint8)

    generator = torch.Generator().manual_seed(case["seed"])
    # The stored elements, in the source's own type: the source wrote `value / scale`.
    source = SOURCES[case["source"]][2]
    magnitude = {"f8e4m3": 32.0, "bf16": 512.0}[case["source"]]
    keys = (
        torch.rand(shape, generator=generator, dtype=torch.float32) * magnitude
        - magnitude / 2
    )
    values = (
        torch.rand(shape, generator=generator, dtype=torch.float32) * magnitude
        - magnitude / 2
    )
    keys = keys.to(source)
    values = values.to(source)
    # Each raw area is eight E bytes, which is more than one field of the page needs in any source
    # type: the page's field lands at its start, as the staging copy puts it, and the rest of the area
    # stays zero, which the run then checks.
    keys_bytes = keys.contiguous().view(torch.uint8).reshape(-1)
    values_bytes = values.contiguous().view(torch.uint8).reshape(-1)
    scratch[0 : keys_bytes.numel()] = keys_bytes
    scratch[8 * elements : 8 * elements + values_bytes.numel()] = values_bytes

    # The FP32 block starts at 16E, and the scale entries past 20E.
    block = scratch[16 * elements : 20 * elements].view(torch.float32)
    block.fill_(SENTINEL)
    scales = None
    if case["scaled"]:
        # Scales the way a real FP8 cache carries them — a factor near 1/100 that no mantissa of a
        # narrow logical type covers — so the product is a value that *must* round, and every entry
        # is its own factor so a program reading another layer's, field's or group's cannot pass.
        scales = (
            torch.rand(
                (2, case["layers"], groups),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.02
        )
        scratch[20 * elements : 20 * elements + 4 * scale_entries] = (
            scales.contiguous().view(torch.uint8).reshape(-1)
        )
    return scratch, keys, values, scales, elements, scale_entries


def reference_exact(case, keys, values, scales):
    """The decoded product before the logical type's rounding: what the cast is applied to."""
    field = case["field"]
    stored = (keys if field == FIELD_KEYS else values).to(torch.float32)
    decoded = stored.clone()
    if case["scaled"]:
        for head in range(case["heads"]):
            # `[layers]` factors of this field and head's group, broadcast over tokens and features.
            factor = scales[field, :, head // case["scale_heads"]]
            decoded[:, :, head, :] *= factor[:, None, None]
    return decoded


def reference(case, keys, values, scales):
    """The FP32 block the kernel must write, in FP32: the product, then the logical type's rounding."""
    decoded = reference_exact(case, keys, values, scales)
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
    # A compiled launch reads device memory; the interpreter reads the host tensor itself. The bytes
    # come back to the host for the comparison either way.
    launched = scratch.cuda() if DEVICE else scratch
    module._page_conversion_kernel[(case["layers"] * case["block_size"], 1)](
        launched,
        elements,
        scale_entries,
        case["block_size"],
        case["scale_heads"] if case["scaled"] else 0,
        FIELD=case["field"],
        SOURCE=source,
        COMPUTE=compute,
        HEAD_DIM=case["head_dim"],
        HEADS=case["heads"],
        LAYERS=case["layers"],
        SCALED=case["scaled"],
        SOURCE_BYTES=source_bytes,
    )
    scratch = launched.cpu()

    where = (
        f"{case['what']}: {case['source']} -> {case['compute']}, "
        f"{'values' if case['field'] else 'keys'}, "
        f"{'scaled' if case['scaled'] else 'dense'}, "
        f"{case['layers']} layers of a {case['block_size']}-token page, "
        f"{groups_of(case)} scale groups"
    )

    # Nothing outside the FP32 block may move: the raw areas hold the source's bytes, and the scale
    # rows are the next launch's input.
    block = slice(16 * elements, 20 * elements)
    if not torch.equal(scratch[: 16 * elements], original[: 16 * elements]):
        raise SystemExit(
            f"{where}: the raw areas changed; this program writes only the FP32 block"
        )
    if not torch.equal(scratch[20 * elements :], original[20 * elements :]):
        raise SystemExit(
            f"{where}: the scale bytes changed; this program writes only the FP32 block"
        )

    expected = reference(case, keys, values, scales)
    got = scratch[block].view(torch.float32).reshape(expected.shape)
    if case["compute"] == "f32":
        if not torch.equal(got, expected):
            apart = (got != expected).reshape(-1)
            position = int(apart.nonzero()[0])
            raise SystemExit(
                f"{where}: {int(apart.sum())} of {got.numel()} decoded values differ from the "
                f"reference; first at {position}: {float(got.reshape(-1)[position])!r} vs "
                f"{float(expected.reshape(-1)[position])!r}"
            )
        return got.numel(), "f32"

    exact = reference_exact(case, keys, values, scales)
    if torch.equal(
        exact.to(COMPUTES[case["compute"]][1]).to(torch.float32), exact
    ):
        # The source type has no more mantissa than the logical type, so the cast cannot round and the
        # answer is exact rather than one of two models.
        if not torch.equal(got, exact):
            apart = (got != exact).reshape(-1)
            position = int(apart.nonzero()[0])
            raise SystemExit(
                f"{where}: {int(apart.sum())} of {got.numel()} decoded values differ from the "
                f"reference although the logical type holds every one exactly; first at "
                f"{position}: {float(got.reshape(-1)[position])!r} vs "
                f"{float(exact.reshape(-1)[position])!r}"
            )
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
            f"{float(truncated.reshape(-1)[position])!r} (truncated)"
        )
    if len(models) > 1 and torch.equal(nearest, truncated):
        raise SystemExit(
            f"{where}: the nearest and truncating models come out identical, so this case cannot "
            f"tell which rounding the kernel takes and must be rebuilt with a value that sits "
            f"between two representable {case['compute']} numbers"
        )
    return got.numel(), named


#: `layers` and `heads` are compiled into a module; the page's block size and the scale group's heads
#: are launch arguments, so a case may vary them.
CASES = [
    # The served path's widest source: an FP8 page decoded to FP32, one scale group per field and
    # layer — the host page's own `[2, layers]` tail.
    dict(
        what="fp8 keys decoded to f32 from one source pool",
        field=FIELD_KEYS,
        source="f8e4m3",
        compute="f32",
        scaled=True,
        head_dim=64,
        heads=4,
        layers=4,
        block_size=16,
        scale_heads=4,
        seed=11,
    ),
    # The same page's values: the second raw area is eight E bytes along and the values' scales are
    # the tail's second row, so a program that decoded keys twice would look right on every key case.
    dict(
        what="fp8 values decoded to f32 from one source pool",
        field=FIELD_VALUES,
        source="f8e4m3",
        compute="f32",
        scaled=True,
        head_dim=64,
        heads=4,
        layers=4,
        block_size=16,
        scale_heads=4,
        seed=12,
    ),
    # Two source pools of two heads each: a head that took the other group's scale lands on a
    # different factor, and the bf16 rounding model is measured here.
    dict(
        what="fp8 keys to bf16 from two source pools",
        field=FIELD_KEYS,
        source="f8e4m3",
        compute="bf16",
        scaled=True,
        head_dim=64,
        heads=4,
        layers=3,
        block_size=8,
        scale_heads=2,
        seed=13,
    ),
    dict(
        what="bf16 keys to bf16 with scales",
        field=FIELD_KEYS,
        source="bf16",
        compute="bf16",
        scaled=True,
        head_dim=64,
        heads=4,
        layers=2,
        block_size=16,
        scale_heads=4,
        seed=14,
    ),
    # Dense pages carry no scales: the value is widened and rounded.
    dict(
        what="dense bf16 keys to f16",
        field=FIELD_KEYS,
        source="bf16",
        compute="f16",
        scaled=False,
        head_dim=64,
        heads=4,
        layers=3,
        block_size=16,
        scale_heads=0,
        seed=15,
    ),
    dict(
        what="dense bf16 values to f32",
        field=FIELD_VALUES,
        source="bf16",
        compute="f32",
        scaled=False,
        head_dim=64,
        heads=4,
        layers=2,
        block_size=16,
        scale_heads=0,
        seed=16,
    ),
    # The scale rule in exact arithmetic: a bf16 source multiplied by an FP32 factor is an FP32
    # product with no rounding to argue about, so the interpreter's answer is the kernel's.
    dict(
        what="bf16 keys to f32 with per-layer scales",
        field=FIELD_KEYS,
        source="bf16",
        compute="f32",
        scaled=True,
        head_dim=64,
        heads=4,
        layers=4,
        block_size=8,
        scale_heads=4,
        seed=17,
    ),
    dict(
        what="bf16 values to f32 from two source pools",
        field=FIELD_VALUES,
        source="bf16",
        compute="f32",
        scaled=True,
        head_dim=64,
        heads=4,
        layers=3,
        block_size=4,
        scale_heads=2,
        seed=18,
    ),
]

#: The interpreter answers every case: each one loads the source's elements and stores FP32 into the
#: block, and an E4M3 *load* is faithful for every finite pattern. A case that stored E4M3 would have
#: to move to `--device`, and none does.
INTERPRETER_CASES = list(CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=PACKAGE,
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
        print(
            "skipped: needs a device — the device half runs the same cases where the rounding into "
            "a narrow logical type must be the operator's `nearest` and nothing may truncate; "
            "without --device the interpreter runs the cases it can answer"
        )
        return 0
    module = load_kernel(options.package)
    cases = CASES if options.device else INTERPRETER_CASES
    models = ("nearest",) if options.device else ("nearest", "truncated")
    checked = 0
    seen = {}
    for case in cases:
        count, model = run(module, case, models)
        checked += count
        seen[model] = seen.get(model, 0) + 1
    named = ", ".join(
        f"{model}: {count}" for model, count in sorted(seen.items())
    )
    where = "the device" if options.device else "Triton's interpreter"
    print(
        f"ok on {where}: {checked} decoded values over {len(cases)} cases, every one the "
        f"reference and every byte outside the FP32 block untouched — {named}"
    )
    if not options.device:
        left = len(CASES) - len(cases)
        print(
            f"{left} cases are left for --device; the interpreter's E4M3 load is faithful for all 254 "
            f"finite patterns and its store is wrong, so a case that stored E4M3 would be the only "
            f"reason to leave one"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
