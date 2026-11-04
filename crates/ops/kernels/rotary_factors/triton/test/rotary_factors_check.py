#!/usr/bin/env python3
"""Run `rotary_factors`'s kernel and check the tables a rotary embedding reads, **on a device**.

    crates/program-server/python/.venv/bin/python rotary_factors_check.py [--self-test]

```text
cosine[t, j] = cos(position[axis(j), t] · frequency[j]) · scale
sine[t, j]   = sin(position[axis(j), t] · frequency[j]) · scale
```

one `[tokens, columns]` table per output, `frequency[j]` the column's angle and `axis(j)` the position
axis the column reads. The operator's bound is `rounding(2, f32, dtype)` with `2·ε` of absolute room
for the factor where its cosine or sine passes through zero.

**This check cannot run on Triton's interpreter.** The kernel calls `libdevice.cos` and
`libdevice.sin`, as it must: the phase of a multimodal position reaches millions of radians and only
the CUDA numerical functions do the range reduction. The interpreter has no libdevice at all — it
raises `Did you forget to add @triton.jit?` from `_semantic` before any arithmetic — which `--self-test`
measures here and reports. So the kernel's trigonometry is unproven until a device runs this script;
that is the gap, stated in one line.

**What `--self-test` does prove is the case list itself**: the same cases run through a NumPy model of
the kernel's formula, once with the axis mapping the descriptor names and once with each of three
plausible wrong ones (a contiguous cut where the descriptor interleaves, every column on axis 0, and
the axes cycled without the fall-back). A case list that accepts the correct mapping and refuses all
three is a list that discriminates; one that accepts a wrong mapping is a case to fix.

The cases:

- **the fixture's cell** — `f32`, 48 columns, three axes, sections `16-16-16`, interleaved, a scale of
  10⁴, and per-axis positions `t`, `100+t`, `1000+t` so a column reading the wrong axis is not a
  rounding difference;
- **one axis** with positions in the millions, where a cosine computed without range reduction is
  wrong by a whole turn, and **negative positions** with a scale of 10⁶;
- **uneven sections** under both assignments, `16-8-24`, including the frequencies an interleaved
  narrow section cannot claim, which fall to the first axis.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import struct
import sys

import numpy as np
import torch

DTYPES = {"f32": torch.float32}
MAX_AXES = 4


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(dtype):
    """`Tolerance::rounding(2, f32, dtype).with_atol(2·ε_f32)`."""
    unit = {torch.float32: 2.0**-24}[dtype]
    return 2.0 * gamma(2, 2.0**-24) + unit, 2.0 * 2.0**-24


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "rotary_factors.py"
    spec = importlib.util.spec_from_file_location("rotary_factors_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def axis_of_column(columns, sections, interleaved):
    """The axis each column reads — the kernel's own rule, transcribed (`rope.rs` states it too).

    `sections` are rotated *elements* per axis, so an axis owns half as many frequencies. A column
    past the last section reads the last axis under the contiguous cut and the first axis under the
    interleaved one, which is what the kernel's two branches compute.
    """
    widths = list(sections)
    if len(widths) <= 1:
        return [0] * columns
    count = len(widths)
    halves = [width // 2 for width in widths]
    if not interleaved:
        cuts, running = [], 0
        for half in halves:
            running += half
            cuts.append(running)
        axes = []
        for column in range(columns):
            axis = 0
            for cut in cuts[1:]:
                axis += int(column >= cut)
            axes.append(axis)
        return axes
    axes = []
    for column in range(columns):
        cycled = column % count
        half = halves[0]
        for axis in range(1, count):
            if cycled == axis:
                half = halves[axis]
        axes.append(cycled if column < half * count else 0)
    return axes


def reference(case):
    """The two tables in FP32 step by step: the phase, the exact cosine, the scale."""
    tokens, columns = case["tokens"], case["columns"]
    axes = axis_of_column(columns, case["sections"], case["interleaved"])
    positions = case["positions"].astype(np.float32).reshape(case["axes"], tokens)
    frequencies = case["frequencies"].astype(np.float32).reshape(columns)
    phase = np.empty((tokens, columns), dtype=np.float32)
    for column in range(columns):
        phase[:, column] = positions[axes[column]] * frequencies[column]
    cosine = np.cos(phase.astype(np.float64)).astype(np.float32) * np.float32(case["scale"])
    sine = np.sin(phase.astype(np.float64)).astype(np.float32) * np.float32(case["scale"])
    return cosine.astype(np.float32), sine.astype(np.float32), axes


def axes_of(case, mapping="correct"):
    """The axes a mapping names for a case."""
    columns, sections = case["columns"], case["sections"]
    if mapping == "correct":
        return axis_of_column(columns, sections, case["interleaved"])
    if mapping == "contiguous":
        return axis_of_column(columns, sections, False)
    if mapping == "first":
        return [0] * columns
    count = len(sections)
    return [column % count for column in range(columns)]


def model(case, mapping="correct"):
    """A NumPy stand-in for the kernel, with the axis mapping under test."""
    tokens, columns = case["tokens"], case["columns"]
    if mapping == "correct":
        axes = axis_of_column(columns, case["sections"], case["interleaved"])
    elif mapping == "contiguous":
        axes = axis_of_column(columns, case["sections"], False)
    elif mapping == "first":
        axes = [0] * columns
    else:
        count = len(case["sections"])
        axes = [column % count for column in range(columns)]
    positions = case["positions"].astype(np.float32).reshape(case["axes"], tokens)
    frequencies = case["frequencies"].astype(np.float32).reshape(columns)
    phase = np.empty((tokens, columns), dtype=np.float32)
    for column in range(columns):
        phase[:, column] = positions[axes[column]] * frequencies[column]
    return (np.cos(phase.astype(np.float64)).astype(np.float32) * np.float32(case["scale"]),
            np.sin(phase.astype(np.float64)).astype(np.float32) * np.float32(case["scale"]))


def compare(what, got, expected, dtype):
    """Two tables against the operator's bound, counting the violations."""
    relative, absolute = bound(dtype)
    values = got.astype(np.float64)
    allowed = absolute + relative * np.abs(expected.astype(np.float64))
    error = np.abs(values - expected.astype(np.float64))
    violations = int(np.count_nonzero(~(error <= allowed)))
    if violations:
        position = int(np.flatnonzero(~(error <= allowed))[0])
        raise SystemExit(
            f"{what}: {violations} of {values.size} elements outside the bound; first at "
            f"{position}: {values.flat[position]} vs {expected.flat[position]}, "
            f"rtol {relative:.3e} atol {absolute:.3e}"
        )


def cases(seed=31):
    """The case list: shapes, per-axis positions, frequencies and scales to exercise."""
    random = np.random.default_rng(seed)
    built = []
    # The fixture's cell: three axes, sections 16-16-16, interleaved, a scale of 10^4.
    built.append(dict(
        name="the fixture's cell", dtype="f32", positions_dtype=np.int32, columns=48, block=256,
        axes=3, sections=[16, 16, 16], interleaved=True,
        scale=struct.unpack("<f", struct.pack("<I", 0x461C4000))[0], tokens=5,
        positions=np.array([[t for t in range(5)], [100 + t for t in range(5)],
                            [1000 + t for t in range(5)]]),
        frequencies=random.uniform(0.5, 3.0, size=48),
    ))
    # One axis, positions in the millions: without range reduction the cosine is wrong by turns.
    built.append(dict(
        name="one axis, positions in the millions", dtype="f32", positions_dtype=np.int64, columns=64,
        block=256, axes=1, sections=[0], interleaved=False, scale=1.0, tokens=3,
        positions=np.array([[1234567 + 97 * t for t in range(3)]]),
        frequencies=random.uniform(0.01, 0.1, size=64),
    ))
    # Negative positions, and the widest scale a descriptor declares.
    built.append(dict(
        name="negative positions at a scale of a million", dtype="f32", positions_dtype=np.int32,
        columns=32, block=256, axes=1, sections=[0], interleaved=False,
        scale=struct.unpack("<f", struct.pack("<I", 0x49742400))[0], tokens=4,
        positions=np.array([[-500 - 13 * t for t in range(4)]]),
        frequencies=random.uniform(0.2, 1.5, size=32),
    ))
    # Uneven sections under both assignments: the cut, then the interleaved fall-back to axis 0.
    for interleaved in (False, True):
        built.append(dict(
            name=f"uneven sections, {'interleaved' if interleaved else 'cut'}", dtype="f32",
            positions_dtype=np.int32, columns=48, block=256, axes=3, sections=[16, 8, 24],
            interleaved=interleaved, scale=struct.unpack("<f", struct.pack("<I", 0x3F800000))[0],
            tokens=4,
            positions=np.array([[7 * t for t in range(4)], [11 + 3 * t for t in range(4)],
                                [900 - 5 * t for t in range(4)]]),
            frequencies=random.uniform(0.05, 2.0, size=48),
        ))
    return built


def self_test(arguments, module):
    """Run the cases through the NumPy model, and prove the list refuses three wrong mappings."""
    print("no device needed for --self-test: the case list runs through a NumPy model of the formula")
    accepted = refused = 0
    for case in cases():
        dtype = DTYPES[case["dtype"]]
        expected_cosine, expected_sine, axes = reference(case)
        for name, got in (("cosine", expected_cosine), ("sine", expected_sine)):
            compare(f"the model, {case['name']}, {name}", got, expected_cosine if name == "cosine"
                    else expected_sine, dtype)
        accepted += 1
        for mapping in ("contiguous", "first", "cycled"):
            # A mapping that names the same axes as the descriptor's is not a wrong mapping for this
            # case; only a mapping that differs can be refused, and only then is refusing evidence.
            if axes_of(case, mapping) == axes:
                continue
            wrong_cosine, _ = model(case, mapping)
            try:
                compare(f"{case['name']} with the {mapping} mapping", wrong_cosine,
                        expected_cosine, dtype)
            except SystemExit:
                refused += 1
            else:
                raise SystemExit(
                    f"{case['name']}: a {mapping} axis mapping passes the bound, so the case does "
                    f"not discriminate — the columns' axes are {axes[:12]}…"
                )
    print(f"ok: {accepted} cases accepted with the descriptor's mapping, {refused} wrong mappings "
          f"refused by the same cases")


def run_on_device(arguments, module):
    """Run the kernel on the device this host has, comparing both tables."""
    import torch as device

    checked = 0
    for case in cases():
        dtype = DTYPES[case["dtype"]]
        tokens, columns = case["tokens"], case["columns"]
        positions = device.tensor(case["positions"], dtype=device.from_numpy(
            case["positions_dtype"]).dtype)
        frequencies = device.tensor(case["frequencies"], dtype=device.float32)
        cosine = device.full((tokens, columns), float("nan"), dtype=dtype)
        sine = device.full((tokens, columns), float("nan"), dtype=dtype)
        total = tokens * columns
        sectioned = case["axes"] > 1
        entry = (module._rotary_factors_sectioned_kernel if sectioned
                 else module._rotary_factors_kernel)
        words = [positions, frequencies, cosine, sine, total, columns, case["scale"]]
        if sectioned:
            words += [case["axes"]] + list(case["sections"]) + [0] * (MAX_AXES - case["axes"])
        entry[((total + case["block"] - 1) // case["block"], 1, 1)](
            *words,
            AXES=case["axes"], HALVES=tuple(width // 2 for width in case["sections"]),
            CUTS=tuple(sum(width // 2 for width in case["sections"][: index + 1])
                       for index in range(case["axes"])),
            INTERLEAVED=case["interleaved"], BLOCK=case["block"], PDL=False,
        )
        expected_cosine, expected_sine, _ = reference(case)
        compare(f"{case['name']}, cosine", cosine.cpu().numpy(), expected_cosine, dtype)
        compare(f"{case['name']}, sine", sine.cpu().numpy(), expected_sine, dtype)
        checked += cosine.numel() + sine.numel()
    print(f"ok: {checked} factors over {len(cases())} cells, within rounding(2)'s bound")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/rotary_factors.py",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run the case list through a NumPy model instead of the kernel")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test(arguments, None) or 0
    if not torch.cuda.is_available():
        # The gap, measured rather than assumed: the kernel's own call fails before any arithmetic.
        os.environ["TRITON_INTERPRET"] = "1"
        module = load_kernel(arguments.package)
        case = cases()[0]
        try:
            module._rotary_factors_sectioned_kernel[(1, 1, 1)](
                torch.zeros(case["axes"], case["tokens"], dtype=torch.int32),
                torch.zeros(case["columns"]), torch.zeros(case["tokens"], case["columns"]),
                torch.zeros(case["tokens"], case["columns"]), case["tokens"] * case["columns"],
                case["columns"], 1.0, 3, 16, 16, 16, 0,
                AXES=3, HALVES=(8, 8, 8), CUTS=(8, 16, 24), INTERLEAVED=True, BLOCK=256,
                PDL=False,
            )
        except Exception as error:  # noqa: BLE001 - the tool's words are the evidence
            print(f"rotary_factors needs a device: the kernel calls libdevice.cos and "
                  f"libdevice.sin for the phase's range reduction, and the interpreter has no "
                  f"libdevice — {type(error).__name__}: {str(error).splitlines()[0]}",
                  file=sys.stderr)
        print("run this with a device, or `--self-test` for the case list alone", file=sys.stderr)
        return 1
    return run_on_device(arguments, load_kernel(arguments.package)) or 0


if __name__ == "__main__":
    sys.exit(main())
