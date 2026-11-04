#!/usr/bin/env python3
"""Run cache_copy_region's kernel under Triton's interpreter and compare what it stored.

Four cases, because the operator has two halves with different arithmetic and the FP8 half has two
branches of its own:

* **dense BF16 and F16** — the stored value is the reference cast's own, the span lands at its offset,
  and the flag is raised for exactly the cells the span touches (the fixture's unit vector is a
  permutation ``[2, 0, 1]``, so a program that addressed ``slot`` instead of ``units[slot]`` writes
  the wrong plane and the comparison says where);
* **FP8 into a fresh unit** (`initialized = 0`) — the scale is the span's own
  ``max(abs(span), 1e-12) / 448`` with nothing resident, and every stored byte decodes to the
  reference's ``clamp(value / scale, ±448)`` cast;
* **FP8 into a unit that already holds larger values** (`initialized = 1`) — the scale must grow to
  ``max(old, proposed)``, every resident byte must be rescaled by ``old/new`` so it decodes to the same
  number it did before, and the span must be encoded with the new scale.

The reference is `uniserve/prefix_cache/mha.py:257-304`, and the numbers below are that code's
expression, in that order.

    TRITON_INTERPRET=1 python copy_region_check.py [--package .]
"""

import argparse
import importlib.util
import pathlib

import torch

COLUMNS = 4
UNITS_PER_PAGE = 3
HEADS = 4
HEAD_DIM = 8
ROW = HEADS * HEAD_DIM
BLOCK_TOKENS = 6
UNITS_TOTAL = 8
#: The destination page's units, deliberately not `[0, 1, 2]`: the unit ids are a permutation.
UNITS = [2, 0, 1]
#: The reference's constants.
FP8_MAX = 448.0
SCALE_FLOOR = 1e-12


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    del package  # the check's own location is the package's, wherever it is run from
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "cache_copy_region.py"
    spec = importlib.util.spec_from_file_location("copy_region_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._cache_copy_region_kernel


def source_span(count):
    """The span the conversion step decodes: values a BF16 and an E4M3 round differently."""
    span = torch.arange(count * COLUMNS * ROW, dtype=torch.float32).reshape(count, COLUMNS, ROW)
    return (span - 37.0) / 7.0


def launch(kernel, count, offset, element, initialized_units=None, previous_scale=1.0):
    """Launch one span into a pool of `element`, answering what it wrote and the buffers it used."""
    element_bytes = 1 if element == "f8e4m3" else 2
    plane_elements = BLOCK_TOKENS * ROW
    if element == "bf16":
        values = torch.zeros((COLUMNS, UNITS_TOTAL, plane_elements), dtype=torch.bfloat16)
    elif element == "f16":
        values = torch.zeros((COLUMNS, UNITS_TOTAL, plane_elements), dtype=torch.float16)
    else:
        values = torch.zeros((COLUMNS, UNITS_TOTAL, plane_elements), dtype=torch.uint8)
    scales = torch.zeros((COLUMNS, UNITS_TOTAL), dtype=torch.float32)
    flags = torch.zeros((COLUMNS, UNITS_TOTAL), dtype=torch.uint8)
    resident = {}
    for column in range(COLUMNS):
        for unit in UNITS:
            if initialized_units and (column, unit) in initialized_units:
                flags[column, unit] = 1
                scales[column, unit] = previous_scale
                # A resident plane worth rescaling: values a reader decoded with `previous_scale`.
                block = (torch.arange(plane_elements, dtype=torch.float32) % 97.0) - 48.0
                decoded = block / 512.0
                if element == "f8e4m3":
                    stored = (decoded / previous_scale).clamp(-FP8_MAX, FP8_MAX).to(
                        torch.float8_e4m3fn
                    )
                    values[column, unit] = stored.view(torch.uint8)
                else:
                    values[column, unit] = decoded.to(
                        torch.bfloat16 if element == "bf16" else torch.float16
                    )
                resident[(column, unit)] = values[column, unit].clone()
    source = source_span(count)
    units = torch.tensor(UNITS, dtype=torch.int32)
    row_pow2 = 1
    while row_pow2 < ROW:
        row_pow2 *= 2
    kernel[(COLUMNS, UNITS_PER_PAGE, 1)](
        values,
        scales if element == "f8e4m3" else flags,
        flags,
        source,
        units,
        offset,
        count,
        UNITS_TOTAL,
        ROW_SIZE=ROW,
        ROW_POW2=row_pow2,
        PLANE_ELEMENTS=plane_elements,
        COLUMNS=COLUMNS,
        BF16=1 if element == "bf16" else 0,
        SCALED=1 if element == "f8e4m3" else 0,
        BLOCK=row_pow2,
        PDL=False,
    )
    return source, values, scales, flags, resident


def check_dense(kernel, element, offset, count):
    """Every stored value the reference cast's own, every flag raised for the span's cells."""
    source, values, _, flags, _ = launch(kernel, count, offset, element)
    stored = torch.bfloat16 if element == "bf16" else torch.float16
    problems = []
    for column in range(COLUMNS):
        for slot, unit in enumerate(UNITS):
            expected = torch.zeros((BLOCK_TOKENS * ROW,), dtype=stored)
            for token in range(count):
                rows = slice((offset + token) * ROW, (offset + token + 1) * ROW)
                expected[rows] = source[token, column].to(stored)
            got = values[column, unit]
            if not torch.equal(got, expected):
                differing = (got != expected).nonzero().flatten().tolist()
                first = differing[0]
                problems.append(
                    f"{element} column {column} slot {slot} (unit {unit}) element {first}: "
                    f"{got[first].item()!r} where the cast gives {expected[first].item()!r}"
                )
            if int(flags[column, unit].item()) != 1:
                problems.append(f"{element} column {column} unit {unit}: flag not raised")
    touched = {(column, unit) for column in range(COLUMNS) for unit in UNITS}
    for column in range(COLUMNS):
        for unit in range(UNITS_TOTAL):
            if (column, unit) not in touched and int(flags[column, unit].item()) != 0:
                problems.append(f"{element} column {column} unit {unit}: flagged outside the span")
    return problems


def check_fp8(kernel, offset, count, fresh):
    """The FP8 branch: the scale's growth, the residents' rescale, the span's encoding."""
    initialised = None if fresh else {(column, unit) for column in range(COLUMNS) for unit in UNITS}
    previous = 1.0
    source, values, scales, flags, resident = launch(
        kernel, count, offset, "f8e4m3", initialised, previous
    )
    problems = []
    name = "fresh unit" if fresh else "unit already holding values"
    for column in range(COLUMNS):
        for slot, unit in enumerate(UNITS):
            span = source[:, column]
            # The reference's expression, in the *store's* precision: the kernel divides float32 by
            # float32, so a float64 quotient here would differ in the last bit and report a failure the
            # kernel did not commit.
            proposed = float(span.abs().max().clamp_min(SCALE_FLOOR).div(FP8_MAX).item())
            if fresh:
                expected_scale = proposed
            else:
                expected_scale = float(max(torch.tensor(previous).item(), proposed))
            got_scale = float(scales[column, unit].item())
            if got_scale != expected_scale:
                problems.append(
                    f"fp8 {name} column {column} unit {unit}: scale {got_scale!r}, "
                    f"the reference gives {expected_scale!r}"
                )
            expected = torch.zeros((BLOCK_TOKENS * ROW,), dtype=torch.float8_e4m3fn)
            if not fresh:
                before = resident[(column, unit)].view(torch.float8_e4m3fn).float()
                rescaled = before * (previous / expected_scale)
                expected[:] = rescaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            for token in range(count):
                rows = slice((offset + token) * ROW, (offset + token + 1) * ROW)
                expected[rows] = (span[token] / expected_scale).clamp(-FP8_MAX, FP8_MAX).to(
                    torch.float8_e4m3fn
                )
            got = values[column, unit].view(torch.float8_e4m3fn)
            if not torch.equal(got, expected):
                differing = (got.float() != expected.float()).nonzero().flatten().tolist()
                first = differing[0]
                problems.append(
                    f"fp8 {name} column {column} unit {unit} element {first}: "
                    f"{got[first].float().item()!r} where the reference gives "
                    f"{expected[first].float().item()!r}"
                )
            if int(flags[column, unit].item()) != 1:
                problems.append(f"fp8 {name} column {column} unit {unit}: flag not raised")
    return problems


#: Values that sit on the format's own edges: just under a power of two, exactly on a subnormal
#: quantum, the largest finite value, and one past it.
BOUNDARY = [
    7.943263,
    -7.943263,
    0.0009765625,
    0.0014648438,
    0.013671875,
    0.015625,
    448.0,
    449.0,
    -448.0,
    -449.0,
    0.0,
    1.9999999,
]


def check_boundaries(kernel):
    """The FP8 encoder's own edges, against torch's cast of the same values.

    An instrument that converts FP8 in software is exactly where a boundary value is most likely to be
    converted wrongly -- Triton's interpreter maps `7.943263` to `4.0` -- so the values here are chosen
    to be the ones a plausible implementation gets wrong rather than the ones a span happens to hold.
    """
    count = 1
    values = torch.zeros((COLUMNS, UNITS_TOTAL, BLOCK_TOKENS * ROW), dtype=torch.uint8)
    scales = torch.zeros((COLUMNS, UNITS_TOTAL), dtype=torch.float32)
    flags = torch.zeros((COLUMNS, UNITS_TOTAL), dtype=torch.uint8)
    source = torch.zeros((count, COLUMNS, ROW), dtype=torch.float32)
    # The largest value is at the format's ceiling, so the span's own scale is 1.0 and each crafted
    # number is encoded as itself rather than as a quotient that would blur the edge being tested.
    source[0, 0, : len(BOUNDARY)] = torch.tensor(BOUNDARY, dtype=torch.float32)
    source[0, 0, len(BOUNDARY)] = 448.0
    units = torch.tensor(UNITS, dtype=torch.int32)
    row_pow2 = 1
    while row_pow2 < ROW:
        row_pow2 *= 2
    kernel[(COLUMNS, UNITS_PER_PAGE, 1)](
        values,
        scales,
        flags,
        source,
        units,
        0,
        count,
        UNITS_TOTAL,
        ROW_SIZE=ROW,
        ROW_POW2=row_pow2,
        PLANE_ELEMENTS=BLOCK_TOKENS * ROW,
        COLUMNS=COLUMNS,
        BF16=0,
        SCALED=1,
        BLOCK=row_pow2,
        PDL=False,
    )
    scale = float(scales[0, UNITS[0]].item())
    got = values[0, UNITS[0]].view(torch.float8_e4m3fn).float()
    # The span is one token's row; the plane is `BLOCK_TOKENS` rows and the rest stays zero.
    expected = torch.zeros_like(got)
    expected[:ROW] = (source[0, 0] / scale).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    ).float()
    # The span's own scale is `449 / 448` here, because the largest crafted value is one past the
    # format's ceiling: every comparison below is a quotient by it, which is what a span does.
    problems = []
    differing = (got != expected).nonzero().flatten().tolist()
    for index in differing[:6]:
        problems.append(
            f"fp8 boundaries element {index}: {source[0, 0, index].item()!r} encoded as "
            f"{got[index].item()!r}, torch gives {expected[index].item()!r}"
        )
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=".")
    arguments = parser.parse_args()
    kernel = load_kernel(arguments.package)
    failures = 0
    cases = [
        ("dense bf16", lambda: check_dense(kernel, "bf16", 0, BLOCK_TOKENS)),
        ("dense f16 at an offset", lambda: check_dense(kernel, "f16", 2, 3)),
        ("fp8 fresh unit", lambda: check_fp8(kernel, 2, 3, fresh=True)),
        ("fp8 grown unit", lambda: check_fp8(kernel, 4, 2, fresh=False)),
        ("fp8 boundary values", lambda: check_boundaries(kernel)),
    ]
    for name, case in cases:
        problems = case()
        if problems:
            failures += len(problems)
            for problem in problems:
                print(f"FAIL {problem}")
        else:
            print(f"ok   {name}: {COLUMNS * UNITS_PER_PAGE} cells agree with the reference")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
