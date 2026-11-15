"""The cache_copy_region program against the quantizing write's expression.

The operator writes an FP32 span, layer-major like a host page, into a page's
units: unit slot ``s`` of the page takes the layers ``s * columns + column``.
Its two halves have different arithmetic, and the FP8 half two branches of
its own, so the cases are:

- **dense BF16 and F16**: every stored value is the reference cast's own, the
  span lands at its offset, the flag is raised for exactly the cells the span
  touches, and each unit slot takes its own layers. The page's units are a
  permutation ``[2, 0, 1]``, so a program that addressed ``slot`` instead of
  ``units[slot]``, or read one slot's layers for every slot, writes the wrong
  plane and the comparison says where;
- **FP8 into a fresh unit** (``initialized = 0``): the scale is the span's own
  ``max(|span|, 1e-12) / 448`` with nothing resident, and every stored byte
  decodes to the reference's ``clamp(value / scale, +-448)`` cast;
- **FP8 into a unit already holding larger values** (``initialized = 1``): the
  scale grows to ``max(old, proposed)``, every resident byte is rescaled by
  ``old / new`` so it decodes to the number it did before, and the span is
  encoded with the new scale;
- **the E4M3 encoder's edges**: values just under a power of two, on a
  subnormal quantum, at the largest finite value and one past it, against
  torch's cast of the same quotients.

Everything is compared exactly: the operator states both conversions in the
bit pattern (``crates/ops/kernels/cache_copy_region``), so the interpreter
and a device store the same values. The expected numbers are the quantizing
write's expression evaluated in the order above.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

COLUMNS = 4
UNITS_PER_PAGE = 3
HEADS = 4
HEAD_DIM = 8
ROW = HEADS * HEAD_DIM
BLOCK_TOKENS = 6
PLANE = BLOCK_TOKENS * ROW
UNITS_TOTAL = 8
#: The page's layers: layer `k` lies in column `k % COLUMNS` of the page's
#: unit slot `k // COLUMNS`.
LAYERS = UNITS_PER_PAGE * COLUMNS
#: The destination page's units, deliberately not `[0, 1, 2]`: the unit ids
#: are a permutation.
UNITS = [2, 0, 1]
#: The quantizing write's constants: the largest finite E4M3 value and the
#: floor that keeps an all-zero span from a zero scale.
FP8_MAX = 448.0
SCALE_FLOOR = 1e-12

#: The planes' storage per destination element type.
STORAGE = {"bf16": torch.bfloat16, "f16": torch.float16, "f8e4m3": torch.uint8}


def launch(
    kernels,
    values,
    scales,
    flags,
    source,
    units,
    offset,
    count,
    *,
    element,
    columns=COLUMNS,
    row=ROW,
    pdl=False,
):
    """One launch over a page's (column, unit slot) cells, compiled for a
    dependent launch when `pdl` is.

    ``values`` is the pool's ``[columns, units_total, plane]`` planes,
    ``scales`` its FP32 scales and ``flags`` its initialized flags (each
    ``[columns, units_total]``), ``source`` the span ``[layers, count, row]``
    and ``units`` the page's unit of each slot.
    """  # noqa: D205
    row_pow2 = 1 << (row - 1).bit_length()
    scaled = element == "f8e4m3"
    kernels._cache_copy_region_kernel[(columns, units.numel(), 1)](
        values,
        # A dense pool has no scales: the operator passes its flags (`*u8`)
        # in the scales' place, which the program then never reads.
        scales if scaled else flags,
        flags,
        source,
        units,
        offset,
        count,
        values.shape[1],
        ROW_SIZE=row,
        ROW_POW2=row_pow2,
        PLANE_ELEMENTS=values.shape[2],
        COLUMNS=columns,
        BF16=int(element == "bf16"),
        SCALED=int(scaled),
        BLOCK=row_pow2,
        PDL=pdl,
        launch_pdl=pdl,
    )


def span_of(count):
    """The span the conversion step decodes, layer-major ``[layers, count,
    row]`` like a host page: values a BF16 and an E4M3 round differently,
    every layer's distinct.
    """  # noqa: D205
    span = torch.arange(LAYERS * count * ROW, dtype=torch.float32).reshape(
        LAYERS, count, ROW
    )
    return (span - 37.0) / 7.0


def layer_of(column, slot):
    """The page layer a (column, unit slot) cell holds."""
    return slot * COLUMNS + column


def pool(element, resident_scale=None):
    """A pool of `element` planes: empty, or with every cell of the page's
    units already written with values a reader decoded at `resident_scale`.

    Returns the values, the scales, the flags and each written cell's plane.
    """  # noqa: D205
    values = torch.zeros((COLUMNS, UNITS_TOTAL, PLANE), dtype=STORAGE[element])
    scales = torch.zeros((COLUMNS, UNITS_TOTAL), dtype=torch.float32)
    flags = torch.zeros((COLUMNS, UNITS_TOTAL), dtype=torch.uint8)
    resident = {}
    if resident_scale is None:
        return values, scales, flags, resident
    decoded = ((torch.arange(PLANE, dtype=torch.float32) % 97.0) - 48.0) / 512.0
    for column in range(COLUMNS):
        for unit in UNITS:
            flags[column, unit] = 1
            scales[column, unit] = resident_scale
            if element == "f8e4m3":
                stored = (decoded / resident_scale).clamp(-FP8_MAX, FP8_MAX)
                values[column, unit] = stored.to(torch.float8_e4m3fn).view(
                    torch.uint8
                )
            else:
                values[column, unit] = decoded.to(STORAGE[element])
            resident[(column, unit)] = values[column, unit].clone()
    return values, scales, flags, resident


def launched(
    kernels, device, values, scales, flags, source, offset, count, element
):
    """Launch over device copies of a pool and a span; the pool as written."""
    on_device = [
        tensor.clone().to(device) for tensor in (values, scales, flags)
    ]
    launch(
        kernels,
        *on_device,
        source.to(device),
        torch.tensor(UNITS, dtype=torch.int32, device=device),
        offset,
        count,
        element=element,
    )
    return [tensor.cpu() for tensor in on_device]


@pytest.mark.parametrize(
    ("element", "offset", "count"),
    [
        pytest.param("bf16", 0, BLOCK_TOKENS, id="bf16, a whole page"),
        pytest.param("f16", 2, 3, id="f16, at an offset"),
    ],
)
def test_a_dense_span_is_the_cast_of_its_layers(
    element, offset, count, torch_device, kernel_source
):
    source = span_of(count)
    values, scales, flags, _ = pool(element)
    values, _, flags = launched(
        kernel_source("cache_copy_region"),
        torch_device,
        values,
        scales,
        flags,
        source,
        offset,
        count,
        element,
    )

    stored = STORAGE[element]
    problems = []
    for column in range(COLUMNS):
        for slot, unit in enumerate(UNITS):
            expected = torch.zeros((PLANE,), dtype=stored)
            for token in range(count):
                rows = slice((offset + token) * ROW, (offset + token + 1) * ROW)
                expected[rows] = source[layer_of(column, slot), token].to(
                    stored
                )
            got = values[column, unit]
            if not torch.equal(got, expected):
                first = int((got != expected).nonzero()[0])
                problems.append(
                    f"column {column} slot {slot} (unit {unit}) element "
                    f"{first}: {got[first].item()!r} where the cast gives "
                    f"{expected[first].item()!r}"
                )
    touched = {(column, unit) for column in range(COLUMNS) for unit in UNITS}
    for column in range(COLUMNS):
        for unit in range(UNITS_TOTAL):
            raised = int(flags[column, unit].item())
            if raised != int((column, unit) in touched):
                problems.append(
                    f"column {column} unit {unit}: flag {raised}, the span "
                    f"{'touches' if (column, unit) in touched else 'misses'} "
                    "it"
                )
    assert not problems, (
        f"{element} span of {count} tokens at {offset}: "
        f"{len(problems)} cells differ: " + "; ".join(problems)
    )


@pytest.mark.parametrize(
    ("fresh", "offset", "count"),
    [
        pytest.param(True, 2, 3, id="a fresh unit"),
        pytest.param(False, 4, 2, id="a unit already holding values"),
    ],
)
def test_an_fp8_span_takes_its_scale_and_rescales_the_residents(
    fresh, offset, count, torch_device, kernel_source
):
    previous = 1.0
    source = span_of(count)
    values, scales, flags, resident = pool(
        "f8e4m3", None if fresh else previous
    )
    values, scales, flags = launched(
        kernel_source("cache_copy_region"),
        torch_device,
        values,
        scales,
        flags,
        source,
        offset,
        count,
        "f8e4m3",
    )

    problems = []
    for column in range(COLUMNS):
        for slot, unit in enumerate(UNITS):
            span = source[layer_of(column, slot)]
            # The reference's expression in the store's precision: the
            # program divides FP32 by FP32, and an FP64 quotient would differ
            # in the last bit.
            proposed = float(
                span.abs().max().clamp_min(SCALE_FLOOR).div(FP8_MAX).item()
            )
            expected_scale = proposed if fresh else max(previous, proposed)
            got_scale = float(scales[column, unit].item())
            if got_scale != expected_scale:
                problems.append(
                    f"column {column} unit {unit}: scale {got_scale!r}, the "
                    f"reference gives {expected_scale!r}"
                )
            expected = torch.zeros((PLANE,), dtype=torch.float8_e4m3fn)
            if not fresh:
                before = (
                    resident[(column, unit)].view(torch.float8_e4m3fn).float()
                )
                rescaled = before * (previous / expected_scale)
                expected[:] = rescaled.clamp(-FP8_MAX, FP8_MAX).to(
                    torch.float8_e4m3fn
                )
            for token in range(count):
                rows = slice((offset + token) * ROW, (offset + token + 1) * ROW)
                expected[rows] = (
                    (span[token] / expected_scale)
                    .clamp(-FP8_MAX, FP8_MAX)
                    .to(torch.float8_e4m3fn)
                )
            got = values[column, unit].view(torch.float8_e4m3fn).float()
            if not torch.equal(got, expected.float()):
                first = int((got != expected.float()).nonzero()[0])
                problems.append(
                    f"column {column} unit {unit} element {first}: "
                    f"{got[first].item()!r} where the reference gives "
                    f"{expected[first].float().item()!r}"
                )
            if int(flags[column, unit].item()) != 1:
                problems.append(f"column {column} unit {unit}: flag not raised")
    assert not problems, (
        f"fp8 span of {count} tokens at {offset} into "
        f"{'a fresh unit' if fresh else 'a unit holding values'}: "
        f"{len(problems)} differences: " + "; ".join(problems)
    )


#: Values on the format's own edges: just under a power of two, exactly on a
#: subnormal quantum, the largest finite value and one past it.
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


def test_the_fp8_encoder_meets_the_format_edges(torch_device, kernel_source):
    """The E4M3 encoder's edges, against torch's cast of the same quotients.

    A program that converts FP8 in software is where a boundary value is most
    likely converted wrongly (Triton's interpreter's own cast maps
    ``7.943263`` to ``4.0``), so the values are the ones a plausible encoder
    gets wrong rather than the ones a span happens to hold. The span's
    largest value is one past the format's ceiling, so its scale is
    ``449 / 448`` and every comparison is a quotient by it, as a span's is.
    """
    count = 1
    source = torch.zeros((LAYERS, count, ROW), dtype=torch.float32)
    source[0, 0, : len(BOUNDARY)] = torch.tensor(BOUNDARY, dtype=torch.float32)
    source[0, 0, len(BOUNDARY)] = 448.0
    values, scales, flags, _ = pool("f8e4m3")
    values, scales, _ = launched(
        kernel_source("cache_copy_region"),
        torch_device,
        values,
        scales,
        flags,
        source,
        0,
        count,
        "f8e4m3",
    )

    # Layer 0 is column 0 of the page's first unit slot, `UNITS[0]`; the span
    # is one token's row and the rest of the plane stays zero.
    scale = float(scales[0, UNITS[0]].item())
    got = values[0, UNITS[0]].view(torch.float8_e4m3fn).float()
    expected = torch.zeros_like(got)
    expected[:ROW] = (
        (source[0, 0] / scale)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    differing = (got != expected).nonzero().flatten().tolist()
    assert not differing, (
        f"{len(differing)} edge values encoded differently at scale "
        f"{scale!r}: "
        + "; ".join(
            f"{source[0, 0, index].item()!r} as {got[index].item()!r}, torch "
            f"gives {expected[index].item()!r}"
            for index in differing[:6]
        )
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("cache_copy_region")
    # A page of an FP8 pool: `columns` layer planes per unit, each
    # `block_tokens` token rows of `heads * head_dim`; the source is the
    # page's layers `[units_per_page * columns]`, layer-major.
    columns, units_total, heads, head_dim, block_tokens = 4, 6, 8, 16, 16
    units = [3, 1]
    row = heads * head_dim
    plane = block_tokens * row

    def launched_dependently(inputs, outputs, pdl):
        launch(
            kernels,
            outputs["values"],
            outputs["scales"],
            outputs["flags"],
            inputs["source"],
            inputs["units"],
            0,
            block_tokens,
            element="f8e4m3",
            columns=columns,
            row=row,
            pdl=pdl,
        )
        return [outputs["values"], outputs["scales"], outputs["flags"]]

    case = chain.Case(
        "cache_copy_region",
        {
            "source": torch.randn(len(units) * columns, block_tokens, row),
            "units": torch.tensor(units, dtype=torch.int32),
        },
        {
            "values": torch.zeros(
                columns, units_total, plane, dtype=torch.uint8
            ),
            "scales": torch.zeros(columns, units_total),
            "flags": torch.zeros(columns, units_total, dtype=torch.uint8),
        },
        launched_dependently,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"cache_copy_region launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a "
        "synchronization"
    )
