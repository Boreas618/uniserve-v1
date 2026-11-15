"""The rotary_factors program against the tables a rotary embedding reads.

::

    cosine[t, j] = cos(position[axis(j), t] frequency[j]) scale
    sine[t, j]   = sin(position[axis(j), t] frequency[j]) scale

one ``[tokens, columns]`` table per output, ``frequency[j]`` the column's
angle and ``axis(j)`` the position axis the column reads. The operator's bound
is ``rounding(2, f32, dtype)`` with ``2 eps`` of absolute room for the factor
where its cosine or sine passes through zero (`RotaryFactors::tolerance`).
The reference is FP32 step by step: the phase, its cosine and sine exact in
FP64, the scale.

The kernel calls ``libdevice.cos`` and ``libdevice.sin``, as it must: the
phase of a multimodal position reaches millions of radians and only the CUDA
numerical functions do the range reduction. Triton's interpreter has no
libdevice, so the kernel runs on a device only.

The cases:

- **the fixture's cell**: FP32, 48 columns, three axes, sections 16-16-16,
  interleaved, a scale of 10^4, and per-axis positions ``t``, ``100 + t``,
  ``1000 + t``, so a column reading the wrong axis is not a rounding
  difference;
- **one axis** with positions in the millions, where a cosine computed without
  range reduction is wrong by a whole turn, and **negative positions** with a
  scale of 10^6;
- **uneven sections** under both assignments, 16-8-24, including the
  frequencies an interleaved narrow section cannot claim, which fall to the
  first axis.

The cases are themselves checked on the host: with the reference's axis
mapping replaced by each of three plausible wrong ones (a contiguous cut where
the descriptor interleaves, every column on axis 0, the axes cycled without
the fall-back), every case whose columns that mapping moves is outside the
bound, so the list discriminates the mapping the kernel must compute.
"""

import itertools
import struct

import numpy as np
import pytest
import torch
from uniserve_testing.kernels.rope.rounding import gamma

pytestmark = pytest.mark.unit

#: The most position axes a sectioned launch carries.
MAX_AXES = 4

#: The lanes one program covers.
BLOCK = 256

#: The seed the frequencies are drawn from, in the cases' order.
SEED = 31


def scale_of(bits):
    """A descriptor's FP32 scale from its bits."""
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def build_cases():
    """The cases: the positions (one row per axis) and their integer type,
    the axes' rotated sections, the assignment, the scale and the columns'
    frequencies.
    """  # noqa: D205
    random = np.random.default_rng(SEED)
    cases = {
        "the fixture's cell, three axes interleaved at a scale of 10^4": {
            "positions": np.array(
                [
                    list(range(5)),
                    [100 + t for t in range(5)],
                    [1000 + t for t in range(5)],
                ]
            ),
            "positions_dtype": torch.int32,
            "sections": [16, 16, 16],
            "interleaved": True,
            "scale": scale_of(0x461C4000),
            "frequencies": random.uniform(0.5, 3.0, size=48),
        },
        "one axis, positions in the millions": {
            "positions": np.array([[1234567 + 97 * t for t in range(3)]]),
            "positions_dtype": torch.int64,
            "sections": [0],
            "interleaved": False,
            "scale": 1.0,
            "frequencies": random.uniform(0.01, 0.1, size=64),
        },
        "negative positions at a scale of a million": {
            "positions": np.array([[-500 - 13 * t for t in range(4)]]),
            "positions_dtype": torch.int32,
            "sections": [0],
            "interleaved": False,
            "scale": scale_of(0x49742400),
            "frequencies": random.uniform(0.2, 1.5, size=32),
        },
    }
    for interleaved in (False, True):
        name = f"uneven sections, {'interleaved' if interleaved else 'cut'}"
        cases[name] = {
            "positions": np.array(
                [
                    [7 * t for t in range(4)],
                    [11 + 3 * t for t in range(4)],
                    [900 - 5 * t for t in range(4)],
                ]
            ),
            "positions_dtype": torch.int32,
            "sections": [16, 8, 24],
            "interleaved": interleaved,
            "scale": scale_of(0x3F800000),
            "frequencies": random.uniform(0.05, 2.0, size=48),
        }
    return cases


CASES = build_cases()


def bound():
    """`Tolerance::rounding(2, f32, f32).with_atol(2 eps_f32)`."""
    return 2.0 * gamma(2) + 2.0**-24, 2.0 * 2.0**-24


def axis_of_column(columns, sections, interleaved):
    """The axis each column reads, as `rope.rs` states it (`section_axes`,
    `interleaved_axes`).

    `sections` are rotated *elements* per axis, so axis `a` owns
    `sections[a] / 2` frequencies. The contiguous cut gives each axis the next
    frequencies it owns, in axis order; the interleaved form gives column `j`
    axis `j % n` while `j < sections[axis] / 2 * n`. A column past the last
    section reads the last axis under the cut and the first axis interleaved.
    """  # noqa: D205
    if len(sections) <= 1:
        return [0] * columns
    halves = [width // 2 for width in sections]
    count = len(sections)
    if interleaved:
        return [
            column % count if column < halves[column % count] * count else 0
            for column in range(columns)
        ]
    starts = list(itertools.accumulate(halves[:-1]))
    return [
        sum(column >= start for start in starts) for column in range(columns)
    ]


#: The wrong axis mappings the cases must refuse, each of a case's columns,
#: sections and assignment.
WRONG_MAPPINGS = {
    "contiguous": lambda columns, sections: axis_of_column(
        columns, sections, False
    ),
    "first": lambda columns, sections: [0] * columns,
    "cycled": lambda columns, sections: [
        column % len(sections) for column in range(columns)
    ],
}


def mappings(spec):
    """A case's axes under the reference's mapping, and under each wrong
    mapping that moves any of its columns.
    """  # noqa: D205
    columns, sections = len(spec["frequencies"]), spec["sections"]
    correct = axis_of_column(columns, sections, spec["interleaved"])
    wrong = {
        name: mapping(columns, sections)
        for name, mapping in WRONG_MAPPINGS.items()
    }
    return correct, {
        name: axes for name, axes in wrong.items() if axes != correct
    }


#: Each case with each wrong mapping that moves its columns.
DISCRIMINATED = [
    pytest.param(case, name, id=f"{case}, {name}")
    for case, spec in CASES.items()
    for name in mappings(spec)[1]
]


def tables(case, axes):
    """The two tables in FP32 step by step, each column reading `axes`'s
    axis: the phase, the exact cosine and sine, the scale.
    """  # noqa: D205
    positions = case["positions"].astype(np.float32)
    frequencies = case["frequencies"].astype(np.float32)
    phase = positions[axes].T * frequencies
    scale = np.float32(case["scale"])
    cosine = np.cos(phase.astype(np.float64)).astype(np.float32) * scale
    sine = np.sin(phase.astype(np.float64)).astype(np.float32) * scale
    return cosine, sine


def outside(got, expected):
    """The elements of `got` outside the bound around `expected`."""
    relative, absolute = bound()
    error = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    return ~(error <= absolute + relative * np.abs(expected.astype(np.float64)))


def launch(kernels, positions, frequencies, cosine, sine, case):
    """One launch over every factor; a case of several axes takes the
    sectioned program, with its axes and sections as words.
    """  # noqa: D205
    tokens, columns = cosine.shape
    total = tokens * columns
    sections, axes = case["sections"], len(case["positions"])
    words = [positions, frequencies, cosine, sine, total, columns]
    words.append(case["scale"])
    if axes > 1:
        kernel = kernels._rotary_factors_sectioned_kernel
        words += [axes, *sections] + [0] * (MAX_AXES - axes)
    else:
        kernel = kernels._rotary_factors_kernel
    # The compile-time values the package derives from a key (`derive`): each
    # axis's frequencies, and the cut's thresholds, axis `a`'s being the
    # frequencies the axes before it own.
    halves = tuple(width // 2 for width in sections)
    kernel[((total + BLOCK - 1) // BLOCK, 1, 1)](
        *words,
        AXES=axes,
        HALVES=halves,
        CUTS=tuple(itertools.accumulate(halves[:-1], initial=0)),
        INTERLEAVED=case["interleaved"],
        BLOCK=BLOCK,
        PDL=False,
    )


@pytest.mark.parametrize(("case", "mapping"), DISCRIMINATED)
def test_the_cases_refuse_a_wrong_axis_mapping(case, mapping):
    spec = CASES[case]
    correct, wrong = mappings(spec)

    expected, _ = tables(spec, correct)
    moved, _ = tables(spec, wrong[mapping])

    refused = outside(moved, expected)
    assert refused.any(), (
        f"{case}: the {mapping} axis mapping passes the bound, so the case "
        f"does not discriminate; the columns' axes are {correct[:12]}..."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES)
def test_both_tables_match_the_reference_within_the_operators_bound(
    case, torch_device, kernel_source
):
    spec = CASES[case]
    tokens, columns = spec["positions"].shape[1], len(spec["frequencies"])
    positions = torch.tensor(
        spec["positions"], dtype=spec["positions_dtype"], device=torch_device
    )
    frequencies = torch.tensor(
        spec["frequencies"], dtype=torch.float32, device=torch_device
    )
    cosine = torch.full((tokens, columns), float("nan"), device=torch_device)
    sine = torch.full_like(cosine, float("nan"))

    launch(
        kernel_source("rotary_factors"),
        positions,
        frequencies,
        cosine,
        sine,
        spec,
    )

    axes = axis_of_column(columns, spec["sections"], spec["interleaved"])
    relative, absolute = bound()
    for name, got, expected in zip(
        ("cosine", "sine"), (cosine, sine), tables(spec, axes)
    ):
        values = got.cpu().numpy()
        missed = outside(values, expected)
        if missed.any():
            position = int(np.flatnonzero(missed)[0])
            pytest.fail(
                f"{case}, {name}: {int(missed.sum())} of {values.size} "
                f"elements outside the bound; first at {position}: "
                f"{values.flat[position]} vs {expected.flat[position]}, rtol "
                f"{relative:.3e} atol {absolute:.3e}"
            )
