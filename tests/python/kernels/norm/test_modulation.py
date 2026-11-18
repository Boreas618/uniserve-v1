"""The diffusion modulation pair against FP64, every element.

One module covers modulated_rms_norm and gated_residual because they are one
indexing scheme and one rounding rule: a timestep embedding's row
``i = row_indices[row]`` is selected from tables rows ``stride`` elements
apart, the row is normalized and modulated, and a gated update is folded into
the residual stream. ``gated_residual`` is the second half alone
(``u = h + gate[i] u``); ``modulated_rms_norm`` is the block's whole
expression::

    x = value + gate[i] * update                    (with an update)
    n = x * rsqrt(mean(x^2) + eps) * weight
    out = n * (1 + scale[i]) + shift[i]

**The operator's constraint, and what the check is for.** Every intermediate
rounds to the activation dtype, so the packages compile with
``enable_fp_fusion = False``: without it ``value + gate * update`` fuses into
one multiply-add and the gated product never rounds. The reference takes each
rounding explicitly, in order, through the activation dtype, so a module that
fused any of them is a difference the check sees rather than one a bound
absorbs.

Cases (``modulated_rms_norm``): the deployed geometry, tables read through
padded row strides (twice the width, the padding NaN) so a wrong index is a
wrong address, ``update`` off with ``retain`` on, a table whose one row serves
every activation row (the index is the only selector), and an E4M3 output,
whose per-row dequantization scale the check also recomputes. Every case runs
with each index type a descriptor declares: ``u32`` as the model library
passes it (``nn::modules::adaln``), ``i32`` and ``i64``. ``gated_residual``
bakes its width and masks its last block, so it is checked with an element
count that is not a multiple of the block and with one that is.

Bounds. ``modulated_rms_norm``'s is ``gamma_n + c u``: the reduction over the
row in FP32 plus ``c`` = 6 (or 8, with an update) roundings of the activation
type, ``atol`` zero, within the declared ``2 gamma_n + (c + 1) u``
(`ModulatedRmsNorm::tolerance`). An E4M3 output quantizes what that tolerance
does not describe, so the dequantized row is compared with a reference
quantized the module's way, and E4M3's unit roundoff is the floor a failure
reports. ``gated_residual``'s is ``gamma_n + 2 u``: one rounding of the gated
product and one of the sum. In FP32 a row whose shift cancels the modulated
value leaves an element with nothing for a relative bound to divide by, so
FP32 is decided with the floor of the sum's terms; the other types by the
relative bound alone. A failure states both counts.

A BF16 store on Triton's interpreter **truncates** where hardware rounds, and
an F16 store there rounds as hardware does; the reference models each
(``interpreted``). The interpreter stores E4M3 wrong for values a device
stores correctly (-126.4941, -31.6235 and 100.0 become -64, -16 and 104 where
``torch.float8_e4m3fn`` gives -128, -32 and 96: neither a truncation nor a
tie-break), so the E4M3 case runs on a device.

Hand-computed cases hold each form to literals, which a mistake the kernel and
the reference share (a gate row selected by the wrong index, a rounding fused
away) cannot pass.
"""

import pytest
import torch
from uniserve_testing.kernels.norm.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    gamma,
    largest_relative,
    outside,
    store,
)

pytestmark = pytest.mark.unit

#: The index types a descriptor declares, as torch spells them.
INDICES = {"u32": torch.uint32, "i32": torch.int32, "i64": torch.int64}

#: E4M3's unit roundoff, and its largest finite magnitude.
E4M3_ROUNDOFF = 2.0**-4
E4M3_MAX = 448.0

#: The relative bound a recomputed dequantization scale is held to.
SCALE_RTOL = 2.0**-20

#: The block `gated_residual` walks, the `BLOCK` its launch table binds.
GATED_BLOCK = 1024

#: Whether the floor of the terms decides an element type: an FP32 row whose
#: shift cancels the modulated value leaves the relative bound nothing.
FLOORED = {"bf16": False, "f16": False, "f32": True}

#: The deployed geometry: the row width and the lane block, the epsilon and
#: the rows of the parameter tables.
WIDTH, BLOCK, EPS, PARAMETERS = 5376, 8192, 1e-5, 4

#: `modulated_rms_norm`'s cases: the rows, the row width and block, whether
#: the block folds an update in and retains its source, the tables' row
#: stride, the output type (the activation's unless named), whether one table
#: row serves every activation row, and the seed.
MODULATED = {
    "the deployed geometry": {"rows": 3, "update": True, "seed": 1},
    "padded tables, no update, the source retained": {
        "rows": 2,
        "retain": True,
        "stride": 2 * WIDTH,
        "seed": 2,
    },
    "one parameter row for every activation row": {
        "rows": 4,
        "update": True,
        "retain": True,
        "reused_table": True,
        "seed": 3,
    },
    "an E4M3 output with its per-row scale": {
        "rows": 2,
        "width": 1024,
        "block": 1024,
        "output": "f8e4m3",
        "update": True,
        "seed": 4,
    },
}

#: `gated_residual`'s cases: the rows, the row width and the seed.
GATED = {
    "an element count that is not a multiple of the block": (3, WIDTH, 1),
    "an element count that is a multiple of the block": (2, 3072, 2),
}

#: Two activation rows of four with fixed operands and the values of each form
#: computed by hand: `modulated_rms_norm`'s mean squares are 9.063125 and
#: 2.421875; `gated_residual`'s values are exact in binary except where the
#: gate product itself rounds.
HAND = {
    "value": [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]],
    "update": [[0.1, 0.2, 0.3, 0.4], [-0.5, 0.25, 0.5, -0.25]],
    "gate": [[1.0, 2.0, 0.5, 1.0], [0.5, 1.0, 2.0, 1.0]],
    "scale": [[0.0, 0.5, 0.0, 0.25], [0.25, 0.0, 0.5, 0.0]],
    "shift": [[0.5, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.5]],
    "weight": [1.0, 0.5, -1.0, 2.0],
    "indices": [0, 1],
    "eps": 1e-6,
    "modulated": [
        [
            0.8653874918463837,
            0.5979068048395371,
            -1.0463369084691898,
            3.6538749184638375,
        ],
        [
            0.20080479076915603,
            0.00903425107701275,
            -2.891588987075847,
            0.17871233476935033,
        ],
    ],
    "gated": [[1.1, 2.4, 3.15, 4.4], [0.25, -0.75, 3.0, -0.25]],
}

#: The lane block the hand-computed rows are normalized in.
HAND_BLOCK = 1024


def launch_modulated(kernels, operands, strides, width, block, eps, flags):
    """One `modulated_rms_norm` launch, one program per row.

    `operands` are the program's ten pointers in order; an operand whose flag
    is off carries another view's address, which the program never
    dereferences. `strides` are the gate, shift and scale row strides.
    """
    rows = operands[0].shape[0]
    kernels._modulated_rms_kernel[(rows,)](
        *operands,
        *strides,
        WIDTH=width,
        BLOCK=block,
        EPS=eps,
        HAS_UPDATE=flags["update"],
        FP8_OUTPUT=flags["fp8"],
        RETAIN=flags["retain"],
        # The launch table's warps and options: no rounding fuses away.
        num_warps=8 if width >= 2048 else 4,
        enable_fp_fusion=False,
    )


def launch_gated(kernels, hidden, update, gate, indices, stride, width):
    """One `gated_residual` launch over `rows x width` elements.

    The program takes the row count and multiplies it by the width it was
    compiled for; `update` receives the written residual.
    """
    rows = indices.numel()
    blocks = (rows * width + GATED_BLOCK - 1) // GATED_BLOCK
    kernels._gated_residual_kernel[(blocks,)](
        hidden,
        update,
        gate,
        indices,
        stride,
        rows,
        WIDTH=width,
        BLOCK=GATED_BLOCK,
        # The launch table's options: the gated product rounds before the sum.
        enable_fp_fusion=False,
    )


def table(generator, rows, width, stride, element):
    """A modulation table of `rows` rows of `width` elements, `stride` apart,
    with the padding between rows NaN: a wrong index lands on it.
    """  # noqa: D205
    storage = torch.randn(rows * stride, generator=generator)
    laid = torch.full((rows, stride), float("nan"))
    for row in range(rows):
        laid[row, :width] = storage[row * stride : row * stride + width]
    return laid.to(element)


def selected_rows(generator, rows, reused, index_type):
    """The parameter row each activation row reads, drawn or all the last."""
    indices = torch.randint(
        PARAMETERS, (rows,), generator=generator, dtype=torch.int64
    )
    if reused:
        indices = torch.full((rows,), PARAMETERS - 1, dtype=torch.int64)
    return indices.to(INDICES[index_type])


def compare(name, got, expected, relative, floor, floored):
    """The message of an output outside its deciding bound, or None."""
    difference = (got - expected).abs()
    declared = outside(difference, relative * expected.abs())
    with_floor = outside(difference, relative * expected.abs() + floor)
    deciding = with_floor if floored else declared
    if deciding == 0:
        return None
    return (
        f"{name}: {declared} of {expected.numel()} outside the declared "
        f"bound, {with_floor} with the operands' floor ("
        f"{'deciding' if floored else 'reported'}), max absolute "
        f"{difference.nan_to_num(nan=float('inf')).max().item():.3e}, max "
        f"relative {largest_relative(difference, expected):.3e}, rtol "
        f"{relative:.3e}"
    )


def test_the_hand_computed_modulation_matches_its_literals(
    torch_device, kernel_source
):
    rows, width = 2, 4
    operand = {
        name: torch.tensor(HAND[name], device=torch_device)
        for name in ("value", "update", "gate", "weight", "shift", "scale")
    }
    indices = torch.tensor(HAND["indices"], device=torch_device)
    got = torch.full((rows, width), float("nan"), device=torch_device)

    launch_modulated(
        kernel_source("modulated_rms_norm"),
        [
            operand["value"],
            operand["update"],
            operand["gate"],
            operand["weight"],
            operand["shift"],
            operand["scale"],
            indices,
            got,
            got,
            got,
        ],
        (width, width, width),
        width,
        HAND_BLOCK,
        HAND["eps"],
        {"update": True, "fp8": False, "retain": False},
    )

    expect = torch.tensor(HAND["modulated"], dtype=torch.float64)
    difference = (got.cpu().double() - expect).abs()
    # The row reduction in FP32 plus the chain's eight roundings, relative to
    # the result, and a floor of one FP32 rounding of each operand: the
    # literal at [1, 1] is 0.009 while the operands producing it are of order
    # one, which a relative term alone cannot cover.
    relative = gamma(width) + 8.0 * UNIT_ROUNDOFF["f32"]
    floor = UNIT_ROUNDOFF["f32"] * sum(
        torch.tensor(HAND[name], dtype=torch.float64).abs()
        for name in ("value", "update", "weight", "shift", "scale", "gate")
    )
    missed = outside(difference, relative * expect.abs() + floor)
    assert missed == 0, (
        f"hand-computed 2 x 4, f32: {missed} outside the bound, max absolute "
        f"{difference.max().item():.3e}, rtol {relative:.3e} with the "
        "operands' floor"
    )


def test_the_hand_computed_gated_residual_matches_its_literals(
    torch_device, kernel_source
):
    rows, width = 2, 4
    hidden = torch.tensor(HAND["value"], device=torch_device).reshape(-1)
    update = torch.tensor(HAND["update"], device=torch_device).reshape(-1)
    gate = torch.tensor(HAND["gate"], device=torch_device)
    indices = torch.tensor(HAND["indices"], device=torch_device)

    launch_gated(
        kernel_source("gated_residual"),
        hidden,
        update,
        gate,
        indices,
        width,
        width,
    )

    expect = torch.tensor(HAND["gated"], dtype=torch.float64)
    got = update.cpu().double().reshape(rows, width)
    difference = (got - expect).abs()
    relative = gamma(width) + 2.0 * UNIT_ROUNDOFF["f32"]
    floor = UNIT_ROUNDOFF["f32"] * (
        torch.tensor(HAND["value"], dtype=torch.float64).abs()
        + torch.tensor(HAND["update"], dtype=torch.float64).abs()
    )
    missed = outside(difference, relative * expect.abs() + floor)
    assert missed == 0, (
        f"hand-computed 2 x 4, f32: {missed} outside the bound, max absolute "
        f"{difference.max().item():.3e}, rtol {relative:.3e} with the "
        "operands' floor"
    )


@pytest.mark.parametrize("indices", INDICES)
@pytest.mark.parametrize("dtype", ["bf16", "f16", "f32"])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            name,
            marks=[pytest.mark.gpu] if spec.get("output") == "f8e4m3" else [],
        )
        for name, spec in MODULATED.items()
    ],
)
def test_a_modulated_row_matches_its_chain_of_roundings(
    case, dtype, indices, torch_device, interpreted, kernel_source
):
    spec = MODULATED[case]
    rows = spec["rows"]
    width, block = spec.get("width", WIDTH), spec.get("block", BLOCK)
    stride = spec.get("stride", width)
    update_on, retain = spec.get("update", False), spec.get("retain", False)
    output = spec.get("output", dtype)
    fp8 = output == "f8e4m3"
    element = TORCH_TYPES[dtype]
    generator = torch.Generator().manual_seed(spec["seed"])
    gate = table(generator, PARAMETERS, width, stride, element)
    shift = table(generator, PARAMETERS, width, stride, element)
    scale = table(generator, PARAMETERS, width, stride, element)
    selected = selected_rows(
        generator, rows, spec.get("reused_table", False), indices
    )
    hidden = torch.randn(rows, width, generator=generator).to(element)
    update = torch.randn(rows, width, generator=generator).to(element)
    weight = torch.randn(width, generator=generator).to(element)
    hidden_before, update_before = hidden.double(), update.double()

    on_device = {
        name: tensor.to(torch_device)
        for name, tensor in (
            ("hidden", hidden),
            ("update", update),
            ("gate", gate),
            ("weight", weight),
            ("shift", shift),
            ("scale", scale),
            ("indices", selected),
        )
    }
    out = torch.full(
        (rows, width),
        float("nan"),
        dtype=torch.float8_e4m3fn if fp8 else element,
        device=torch_device,
    )
    retained = torch.full_like(on_device["hidden"], float("nan"))
    dequantization = torch.full((rows,), float("nan"), device=torch_device)

    launch_modulated(
        kernel_source("modulated_rms_norm"),
        [
            on_device["hidden"],
            on_device["update"],
            on_device["gate"] if update_on else on_device["hidden"],
            on_device["weight"],
            on_device["shift"],
            on_device["scale"],
            on_device["indices"],
            out,
            dequantization if fp8 else out,
            retained if retain else out,
        ],
        (stride, stride, stride),
        width,
        block,
        EPS,
        {"update": update_on, "fp8": fp8, "retain": retain},
    )

    # The reference, one rounding at a time, in the order the operator
    # states: each FP32 step rounds to FP32 and is stored into the
    # activation type.
    def stored(values):
        return store(values.float().double(), dtype, interpreted)

    rows_of = selected.long()
    x = hidden_before
    if update_on:
        gated = stored(gate[rows_of, :width].double() * update_before)
        x = stored(x + gated)
    mean_square = (x * x).sum(dim=-1, keepdim=True) / width
    normalized = stored(x * torch.rsqrt(mean_square + EPS) * weight.double())
    factor = stored(1.0 + scale[rows_of, :width].double())
    product = stored(normalized * factor)
    expected = stored(product + shift[rows_of, :width].double())

    chain = 8.0 if update_on else 6.0
    relative = gamma(width) + chain * UNIT_ROUNDOFF[dtype]
    floored = FLOORED[dtype]
    where = f"{case}, {rows} x {width} of block {block}, {dtype} -> {output}"
    where += f", {indices} indices"
    failures = []
    if update_on:
        failures.append(
            compare(
                f"{where}, the update",
                on_device["update"].cpu().double(),
                x,
                relative,
                0.0,
                floored,
            )
        )
    if retain:
        failures.append(
            compare(
                f"{where}, the retained source",
                retained.cpu().double(),
                x,
                relative,
                0.0,
                floored,
            )
        )

    got = out.cpu().double()
    if fp8:
        # Dequantized with the per-row scale the module wrote, against the
        # reference quantized the way the module quantizes it: by the row's
        # own scale, inside the format's range, one E4M3 rounding.
        written = dequantization.cpu().double()
        row_scale = (
            expected.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
            / E4M3_MAX
        )
        wanted = (expected / row_scale).clamp(-E4M3_MAX, E4M3_MAX).to(
            torch.float8_e4m3fn
        ).double() * row_scale
        failures.append(
            compare(
                f"{where}, the dequantized output",
                got * written.unsqueeze(-1),
                wanted,
                relative,
                E4M3_ROUNDOFF * wanted.abs(),
                False,
            )
        )
        failures.append(
            compare(
                f"{where}, the dequantization scale",
                written,
                row_scale.squeeze(-1),
                SCALE_RTOL,
                0.0,
                False,
            )
        )
    else:
        # The shifted sum's roundings are of its terms: a row whose shift
        # cancels the modulated value leaves nothing for a relative bound.
        floor = UNIT_ROUNDOFF[dtype] * (
            product.abs() + shift[rows_of, :width].double().abs()
        )
        failures.append(
            compare(
                f"{where}, the output", got, expected, relative, floor, floored
            )
        )
    failures = [failure for failure in failures if failure is not None]
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("indices", INDICES)
@pytest.mark.parametrize("dtype", ["bf16", "f16", "f32"])
@pytest.mark.parametrize("case", GATED)
def test_a_gated_residual_matches_its_two_roundings(
    case, dtype, indices, torch_device, interpreted, kernel_source
):
    rows, width, seed = GATED[case]
    element = TORCH_TYPES[dtype]
    generator = torch.Generator().manual_seed(seed)
    gate = table(generator, PARAMETERS, width, width, element)
    selected = selected_rows(generator, rows, False, indices)
    elements = rows * width
    hidden = torch.randn(elements, generator=generator).to(element)
    update = torch.randn(elements, generator=generator).to(element)
    hidden_before, update_before = hidden.double(), update.double()
    written = update.to(torch_device)

    launch_gated(
        kernel_source("gated_residual"),
        hidden.to(torch_device),
        written,
        gate.to(torch_device),
        selected.to(torch_device),
        width,
        width,
    )

    reads = gate.reshape(-1)[
        selected.long().repeat_interleave(width) * width
        + torch.arange(elements) % width
    ]
    gated = store(
        (reads.double() * update_before).float().double(), dtype, interpreted
    )
    expected = store(
        (hidden_before + gated).float().double(), dtype, interpreted
    )
    relative = gamma(width) + 2.0 * UNIT_ROUNDOFF[dtype]
    floor = UNIT_ROUNDOFF[dtype] * (hidden_before.abs() + gated.abs())
    failure = compare(
        f"{case}, {rows} x {width} of {elements} elements in blocks of "
        f"{GATED_BLOCK}, {dtype}, {indices} indices, the update",
        written.cpu().double(),
        expected,
        relative,
        floor,
        FLOORED[dtype],
    )
    assert failure is None, failure
