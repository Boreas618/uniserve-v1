#!/usr/bin/env python3
"""Run the diffusion modulation pair and check every element.

    python modulation_check.py --op modulated_rms_norm
    python modulation_check.py --op gated_residual --interpreter
    python modulation_check.py --output f8e4m3 --retain true

One script covers both because they are one indexing scheme and one rounding rule: a timestep
embedding's row `i = row_indices[row]` is selected from tables whose leading stride the
descriptor bakes, the row is normalized and modulated, and a gated update is folded into the
residual stream. `gated_residual` is the second half alone (`u = h + gate[i] · u`);
`modulated_rms_norm` is the block's whole expression:

```text
x = value + gate[i] · update                    (with an update)
n = x · rsqrt(mean(x²) + eps) · weight
out = n · (1 + scale[i]) + shift[i]
```

**The contract's constraint, and what the check is really for.** Every intermediate rounds to
the activation dtype (`specs/worker.md` 9), so the packages compile with
`enable_fp_fusion = False` — without it `value + gate · update` contracts into one fused
multiply-add and the gated product never rounds. The reference here takes each rounding
explicitly, in order, through the activation dtype, so a module that fused any of them is a
difference the check sees rather than a difference a bound absorbs.

Cases (`modulated_rms_norm`): the deployed geometry of the fixture's own key, the same table
read through padded strides so a wrong index is a wrong address, `update` off with `retain`
on, and an E4M3 output — whose per-row dequantization scale the check also recomputes. Two
further cases reach paths the deployed one does not: a table whose rows are reused by many
activation rows (the index is the only thing that selects a row), and an `i64` index table.

`gated_residual` bakes its element count and masks its last tile, so it is checked with a
count that is not a multiple of the block, next to one that is.

Bounds. `modulated_rms_norm`'s declared tolerance is the reduction over the row in FP32 plus
6 (or 8, with an update) roundings of the activation type, and `atol` is zero. An E4M3 output
quantizes what the operator's own tolerance does not describe, so the dequantized row is
compared with E4M3's unit roundoff added and both counts are printed; `--bound` decides.
`gated_residual` declares one rounding of the gated product and one of the sum, so its bound
is two roundings and its check is exact wherever the interpreter and the device agree.

In `f32` a row whose shift cancels the modulated value leaves an element with nothing for a
relative bound to divide by — `atol` is zero — so `f32` runs are decided with
`--bound scale`, which is the same bound with the sum's terms as the floor. The number is
printed either way.

A `bf16` store through Triton's interpreter **truncates** where hardware rounds, and an `f16`
store there rounds as hardware does. The reference models both: the difference between the two
is builder02's measurement (`14336 of 14336` truncated in `bf16`, `199 of 384` rounded in
`f16`), and a reference that assumed one for the other would be wrong on half the tree.
"""

import argparse
import pathlib
import struct
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The pair's other package, so the harness finds its sibling wherever it is run from.
SIBLINGS = ("modulated_rms_norm", "gated_residual")

for _sibling in SIBLINGS:
    _source = (
        pathlib.Path(__file__).resolve().parents[3]
        / _sibling
        / "triton"
        / "src"
    )
    if _source.is_dir() and str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

#: The activation types' unit roundoff, as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

#: E4M3's unit roundoff: three mantissa bits, and the format's largest finite magnitude.
E4M3_ROUNDOFF = 2.0**-4
E4M3_MAX = 448.0

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f8e4m3": torch.float8_e4m3fn,
}

#: The operators this script runs.
OPERATORS = ("modulated_rms_norm", "gated_residual")

#: The tile `gated_residual` walks, the operator's `GATED_BLOCK`.
GATED_BLOCK = 1024


def default_op():
    """The operator this copy belongs to, from the package it is kept in."""
    package = pathlib.Path(__file__).resolve().parents[2].name
    return package if package in OPERATORS else OPERATORS[0]


def gamma(reduction):
    """`γₙ = n·u/(1 − n·u)` for `n` FP32 roundings, as the operator crate computes it."""
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def as_bits(value):
    """The f32 a descriptor's `epsilon` prints as, 1e-06 -> 0x358637bd."""
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    return f"0x{raw:08x}"


def store(values, dtype, truncated):
    """A value as the module stores it: cast to the activation type, one way or the other.

    Every intermediate of both operators rounds to the activation dtype, and `truncated` says
    which model the run's store follows: Triton's interpreter truncates a `bf16` store and
    rounds an `f16` one, and hardware rounds both.
    """
    if dtype == "f32":
        return values
    if dtype == "bf16" and truncated:
        # One ulp low: the low 45 mantissa bits of the FP64 form, which is a bf16 store's
        # truncation of the FP32 value it holds.
        bits = values.view(torch.int64)
        return (bits & ~((1 << 45) - 1)).view(torch.float64)
    return values.to(TORCH_TYPES[dtype]).double()


def compare(name, got, expected, relative, floor, deciding_bound):
    """Count the elements outside each bound and give the line that reports them."""
    difference = (got - expected).abs()
    declared_outside = (~(difference <= relative * expected.abs())).sum().item()
    scaled_outside = (
        (~(difference <= relative * expected.abs() + floor)).sum().item()
    )
    largest = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    deciding = (
        declared_outside if deciding_bound == "declared" else scaled_outside
    )
    line = (
        f"  {name}: max absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, "
        f"max relative {largest:.3e}, declared {declared_outside} of {expected.numel()} "
        f"outside, with the operands' floor {scaled_outside} outside"
    )
    return deciding, line


def tables(case, generator, dtype):
    """The modulation table pair, laid out at the case's strides.

    A row of the table is `width` unit-strided elements `stride` apart from the next, so the
    padding between rows is filled with values a wrong index would land on.
    """

    def table(stride):
        rows = case["parameters"]
        storage = torch.randn(rows * stride, generator=generator)
        table = torch.empty(rows, stride)
        for row in range(rows):
            table[row, : case["width"]] = storage[
                row * stride : row * stride + case["width"]
            ]
        # The padding holds a value nothing should read: a wrong index lands on it.
        table[:, case["width"] :] = float("nan")
        return table.to(dtype)

    return table


def check(operator, case):
    """Run one case of one operator and compare every buffer it writes."""
    if case["output"] == "f8e4m3" and case["interpreter"]:
        # Measured, not assumed: the interpreter's E4M3 store is wrong for values a device
        # stores correctly. Storing -126.4941, -31.6235 and 100.0 gives -64, -16 and 104 where
        # `torch.float8_e4m3fn` gives -128, -32 and 96 — not a truncation and not a tie-break,
        # so no reference can model it. The case runs on a device.
        print(
            "modulated_rms_norm's E4M3 output needs a device: Triton's interpreter stores "
            "E4M3 wrong (-126.4941 -> -64, -31.6235 -> -16, 100.0 -> 104)",
            file=sys.stderr,
        )
        return 0
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = TORCH_TYPES[case["dtype"]]
    truncated = case["truncate"]
    width, rows = case["width"], case["rows"]
    output = (
        torch.float32
        if case["output"] == "f32"
        else TORCH_TYPES[case["output"]]
    )

    table = tables(case, generator, dtype)
    gate = table(case["gate_stride"])
    shift = table(case["shift_stride"])
    scale = table(case["scale_stride"])
    indices = torch.randint(
        case["parameters"], (rows,), generator=generator, dtype=torch.int64
    )
    if case.get("reused_table"):
        # One parameter row serves every activation row: the index is the only selector.
        indices = torch.full((rows,), case["parameters"] - 1, dtype=torch.int64)
    indices = indices.to(TORCH_TYPES_INDICES[case["indices"]])

    if operator == "gated_residual":
        return check_gated(case, generator, dtype, width, rows, gate, indices)

    hidden = torch.randn(rows, width, generator=generator).to(dtype)
    update = torch.randn(rows, width, generator=generator).to(dtype)
    weight = torch.randn(width, generator=generator).to(dtype)
    written = torch.full((rows, width), float("nan"), dtype=dtype)
    retained = (
        torch.full((rows, width), float("nan"), dtype=dtype)
        if case["retain"]
        else None
    )
    dequantization = (
        torch.full((rows,), float("nan"), dtype=torch.float32)
        if case["output"] == "f8e4m3"
        else None
    )
    out = torch.full((rows, width), float("nan"), dtype=output)
    eps = case["eps"]

    hidden_before = hidden.cpu().double()
    update_before = update.cpu().double()

    import modulated_rms_norm as kernels

    kernels._modulated_rms_kernel[(rows,)](
        hidden,
        update,
        gate if case["update"] else hidden,
        weight,
        shift,
        scale,
        indices,
        out,
        dequantization if dequantization is not None else out,
        retained if retained is not None else out,
        case["gate_stride"],
        case["shift_stride"],
        case["scale_stride"],
        WIDTH=width,
        BLOCK=case["block"],
        EPS=eps,
        HAS_UPDATE=case["update"],
        FP8_OUTPUT=case["output"] == "f8e4m3",
        RETAIN=case["retain"],
    )

    # The reference, one rounding at a time, in the order the contract states.
    selected = indices.cpu().long()
    x = hidden_before
    if case["update"]:
        gated = store(
            (gate[selected, :width].double() * update_before).float().double(),
            case["dtype"],
            truncated,
        )
        x = store((x + gated).float().double(), case["dtype"], truncated)
    mean_square = (x * x).sum(dim=-1, keepdim=True) / width
    normalized = store(
        (x * torch.rsqrt(mean_square + eps) * weight.cpu().double())
        .float()
        .double(),
        case["dtype"],
        truncated,
    )
    factor = store(
        (1.0 + scale[selected, :width].double()).float().double(),
        case["dtype"],
        truncated,
    )
    expected = store(
        (normalized * factor).float().double(), case["dtype"], truncated
    )
    expected = store(
        (expected + shift[selected, :width].double()).float().double(),
        case["dtype"],
        truncated,
    )

    # The tolerance: the row reduction in FP32 plus the chain's roundings in the activation type.
    chain = 8.0 if case["update"] else 6.0
    relative = gamma(width) + chain * UNIT_ROUNDOFF[case["dtype"]]
    lines = []
    outside = 0

    if case["update"]:
        counted, line = compare(
            "update", update.cpu().double(), x, relative, 0.0, case["bound"]
        )
        outside += counted
        lines.append(line)
    if case["retain"]:
        counted, line = compare(
            "retain",
            retained.cpu().double(),
            store(
                x,
                case["output"] if case["output"] != "f8e4m3" else case["dtype"],
                truncated,
            ),
            relative,
            0.0,
            case["bound"],
        )
        outside += counted
        lines.append(line)

    got = out.cpu().double()
    if case["output"] == "f8e4m3":
        # Dequantize with the per-row scale the module wrote, and let E4M3's own quantization
        # into the floor: the operator's declared tolerance does not describe the format.
        written_scale = dequantization.cpu().double().unsqueeze(-1)
        got = got * written_scale
        # The reference row quantized the way the module quantizes it: by the row's own scale,
        # inside the format's range, one E4M3 rounding.
        row_scale = (
            expected.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
            / E4M3_MAX
        )
        wanted = (expected / row_scale).clamp(-E4M3_MAX, E4M3_MAX).to(
            torch.float8_e4m3fn
        ).float().double() * row_scale
        floor = E4M3_ROUNDOFF * wanted.abs()
        counted, line = compare(
            "out (dequantized)", got, wanted, relative, floor, case["bound"]
        )
        outside += counted
        lines.append(line)
        row_max = (
            expected.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
            / E4M3_MAX
        )
        counted, line = compare(
            "dequantization scale",
            dequantization.cpu().double(),
            row_max.squeeze(-1),
            2.0**-20,
            0.0,
            "declared",
        )
        outside += counted
        lines.append(line)
    else:
        # The shifted sum's own roundings are of its terms: a row whose shift cancels the
        # modulated value leaves nothing for a relative bound to divide by, which `atol = 0`
        # does not cover.
        floor = UNIT_ROUNDOFF[case["dtype"]] * (
            (normalized * factor).abs() + shift[selected, :width].double().abs()
        )
        counted, line = compare(
            "out", got, expected, relative, floor, case["bound"]
        )
        outside += counted
        lines.append(line)

    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {operator}, {case['label']} — "
        f"{rows} x {width} of block {case['block']}, {case['dtype']} -> {case['output']}, "
        f"{case['indices']} indices, eps {eps:g} ({as_bits(eps)}), "
        f"update {str(case['update']).lower()}, retain {str(case['retain']).lower()}"
        f"{', interpreter store truncated' if truncated else ''}"
    )
    for line in lines:
        print(line)
    return outside


def check_gated(case, generator, dtype, width, rows, gate, indices):
    """Add one gated update into the residual and compare the written residual."""
    truncated = case["truncate"]
    elements = rows * width
    hidden = torch.randn(elements, generator=generator).to(dtype)
    update = torch.randn(elements, generator=generator).to(dtype)
    hidden_before = hidden.cpu().double()
    # `update` receives the residual, so the operand is kept before the launch.
    update_before = update.cpu().double()

    import gated_residual as kernels

    grids = (elements + GATED_BLOCK - 1) // GATED_BLOCK
    kernels._gated_residual_kernel[(grids,)](
        hidden,
        update,
        gate,
        indices,
        case["gate_stride"],
        # The kernel takes the ROW count and multiplies it by the width it was compiled for; the
        # element count here masked every lane live past the activation and past `indices`, which
        # the interpreter answered by reading out of bounds.
        rows,
        WIDTH=width,
        BLOCK=GATED_BLOCK,
    )

    selected = indices.cpu().long().repeat_interleave(width)
    reads = gate.reshape(-1)[
        selected * case["gate_stride"] + torch.arange(elements) % width
    ]
    gated = store(
        (reads.double() * update_before).float().double(),
        case["dtype"],
        truncated,
    )
    expected = store(
        (hidden_before + gated).float().double(), case["dtype"], truncated
    )
    relative = gamma(width) + 2.0 * UNIT_ROUNDOFF[case["dtype"]]
    floor = UNIT_ROUNDOFF[case["dtype"]] * (hidden_before.abs() + gated.abs())
    counted, line = compare(
        "update",
        update.cpu().double(),
        expected,
        relative,
        floor,
        case["bound"],
    )
    print(
        f"{'exact' if counted == 0 else 'FAIL'}: gated_residual, {case['label']} — "
        f"{rows} x {width} of {elements} elements in blocks of {GATED_BLOCK}, "
        f"{case['dtype']}, {case['indices']} indices"
        f"{', interpreter store truncated' if truncated else ''}"
    )
    print(line)
    print(
        "  hidden: not written by this operator, and unchanged in the shape check"
    )
    return counted


#: A hand-computed case per form: two activation rows of four with fixed operands and the
#: expected values written out as decimals. The randomized cases compare a launch against a
#: reference this same file computes, so a mistake both share — a row read as a column, a
#: dropped `eps`, a gate row selected by the wrong index, a rounding fused away — is
#: invisible to them. These numbers were computed once by hand and are literals here for that
#: reason. `modulated_rms_norm`'s mean squares are `9.063125` and `2.421875`; `gated_residual`'s
#: expected values are exact in binary except where the gate product itself rounds.
HAND = {
    "width": 4,
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
    "gated": [
        [1.1, 2.4, 3.15, 4.4],
        [0.25, -0.75, 3.0, -0.25],
    ],
}


def check_hand(form):
    """Run the hand-computed case of one form and compare every element with its literal.

    `form` is `modulated` or `gated` — the same two words `HAND` is keyed by and the same derivation
    the descriptor's key uses. It selects **both** the literal table and the kernel, which is the
    point: branching on the operator's name here instead would run one form's kernel against the
    other form's literals whenever the caller passed a form, and the mismatch would read as an
    arithmetic defect in a kernel that is correct.
    """
    width = HAND["width"]
    rows = len(HAND["value"])
    gate = torch.tensor(HAND["gate"], dtype=torch.float32)
    indices = torch.tensor(HAND["indices"], dtype=torch.int64)
    expect = torch.tensor(HAND[form], dtype=torch.float64)

    if form == "gated":
        import gated_residual as kernels

        elements = rows * width
        hidden = torch.tensor(HAND["value"], dtype=torch.float32).reshape(elements)
        # `update` receives the written residual, so the operand is read before the launch.
        update = torch.tensor(HAND["update"], dtype=torch.float32).reshape(elements)
        grids = (elements + GATED_BLOCK - 1) // GATED_BLOCK
        kernels._gated_residual_kernel[(grids,)](
            hidden,
            update,
            gate,
            indices,
            width,
            rows,
            WIDTH=width,
            BLOCK=GATED_BLOCK,
        )
        got = update.reshape(rows, width)
        relative = gamma(width) + 2.0 * UNIT_ROUNDOFF["f32"]
    else:
        import modulated_rms_norm as kernels

        hidden = torch.tensor(HAND["value"], dtype=torch.float32)
        update = torch.tensor(HAND["update"], dtype=torch.float32)
        weight = torch.tensor(HAND["weight"], dtype=torch.float32)
        shift = torch.tensor(HAND["shift"], dtype=torch.float32)
        scale = torch.tensor(HAND["scale"], dtype=torch.float32)
        got = torch.full((rows, width), float("nan"), dtype=torch.float32)
        kernels._modulated_rms_kernel[(rows,)](
            hidden,
            update,
            gate,
            weight,
            shift,
            scale,
            indices,
            got,
            got,
            got,
            width,
            width,
            width,
            WIDTH=width,
            BLOCK=GATED_BLOCK,
            EPS=HAND["eps"],
            HAS_UPDATE=True,
            FP8_OUTPUT=False,
            RETAIN=False,
        )
        # The row reduction in FP32 plus the chain's eight roundings, as the operator declares.
        relative = gamma(width) + 8.0 * UNIT_ROUNDOFF["f32"]

    difference = (got.double() - expect).abs()
    # The standard rounding bound, as every operator in this crate declares it: a term relative to
    # the result plus a floor set by the operands' own magnitudes. The relative term alone cannot
    # cover an element that cancels — the modulated form's literal at [1,1] is 0.009 while the
    # operands that produce it are of order one — so a relative-only bound there asks the f32 chain
    # to be exact where no chain can be, and reports a correct program as failing by three bounds.
    floor = UNIT_ROUNDOFF["f32"] * torch.tensor(
        [HAND["value"], HAND["update"]], dtype=torch.float64
    ).abs().sum(0)
    if form == "modulated":
        # The chain's remaining operands, each of which contributes its own rounding.
        for name in ("weight", "shift", "scale", "gate"):
            floor = floor + UNIT_ROUNDOFF["f32"] * torch.tensor(
                HAND[name], dtype=torch.float64
            ).abs()
    outside = (~(difference <= relative * expect.abs() + floor)).sum().item()
    print(
        f"{'exact' if outside == 0 else 'FAIL'}: hand-computed {rows} x {width}, f32 — "
        f"max absolute {difference.max().item():.3e}, bound {relative:.3e} "
        f"({outside} outside, with the operands' floor "
        f"{(~(difference <= floor)).sum().item()} outside)"
    )
    return outside


def cases(operator, flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {
        "dtype": flags.dtype,
        "output": flags.output,
        "indices": flags.indices,
        "width": flags.width,
        "block": flags.block,
        "eps": flags.eps,
        "gate_stride": flags.gate_stride or flags.width,
        "shift_stride": flags.shift_stride or flags.width,
        "scale_stride": flags.scale_stride or flags.width,
        "parameters": flags.parameters,
    }
    if operator == "gated_residual":
        return [
            {
                **common,
                "label": "a count that is not a multiple of the block",
                "rows": 3,
                "seed": 1,
            },
            {**common, "label": "a count that is", "rows": 2, "seed": 2},
        ]
    return [
        {
            **common,
            "label": "the deployed geometry",
            "rows": flags.rows,
            "block": max(flags.block, flags.width),
            "update": True,
            "retain": False,
            "reused_table": False,
            "seed": 1,
        },
        {
            **common,
            # Padded tables, no update, and the retained source: the case whose strides are not
            # the width, so an index·width address reads the wrong channel.
            "label": "padded tables, no update, the source retained",
            "rows": 2,
            "block": max(flags.block, flags.width),
            "update": False,
            "retain": True,
            "reused_table": False,
            "seed": 2,
        },
        {
            **common,
            "label": "one parameter row for every activation row",
            "rows": 4,
            "block": max(flags.block, flags.width),
            "update": True,
            "retain": True,
            "reused_table": True,
            "seed": 3,
        },
        {
            **common,
            "label": "an E4M3 output with its per-row scale",
            "rows": 2,
            "columns": min(flags.width, 1024),
            "width": min(flags.width, 1024),
            "block": max(min(flags.block, 1024), min(flags.width, 1024)),
            "output": "f8e4m3",
            "update": True,
            "retain": False,
            "reused_table": False,
            "seed": 4,
        },
    ]


#: The index types a descriptor may declare, as torch spells them.
TORCH_TYPES_INDICES = {"i32": torch.int32, "i64": torch.int64}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPERATORS, default=default_op())
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument(
        "--output", default=None, help="the output type; `dtype` by default"
    )
    parser.add_argument(
        "--indices", choices=sorted(TORCH_TYPES_INDICES), default="i32"
    )
    parser.add_argument("--width", type=int, default=5376)
    parser.add_argument("--block", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--parameters", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--gate-stride", type=int, default=None)
    parser.add_argument("--shift-stride", type=int, default=None)
    parser.add_argument("--scale-stride", type=int, default=None)
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides the run: the operator's own (atol 0), or the same with "
        "the floor the operands' magnitudes imply",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    arguments.output = arguments.output or arguments.dtype
    if arguments.output not in TORCH_TYPES:
        parser.error(f"no output type {arguments.output}")
    if arguments.block < arguments.width:
        parser.error(
            f"a block of {arguments.block} lanes cannot cover a row of {arguments.width}"
        )
    if not arguments.interpreter and not torch.cuda.is_available():
        print(
            "no GPU on this host: running through Triton's interpreter instead",
            file=sys.stderr,
        )
        arguments.interpreter = True
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"

    gate_stride = arguments.gate_stride or arguments.width
    shift_stride = arguments.shift_stride or arguments.width
    scale_stride = arguments.scale_stride or arguments.width
    if arguments.op == "gated_residual":
        key = (
            f"dtype_{arguments.dtype}_indices_{arguments.indices}"
            f"_width_{arguments.width}"
            f"_elements_{arguments.rows * arguments.width}"
            f"_gate_stride_{gate_stride}"
        )
    else:
        key = (
            f"dtype_{arguments.dtype}_output_{arguments.output}"
            f"_indices_{arguments.indices}_width_{arguments.width}"
            f"_block_{arguments.block}_epsilon_{as_bits(arguments.eps)}"
            f"_gate_stride_{gate_stride}_shift_stride_{shift_stride}"
            f"_scale_stride_{scale_stride}_update_{str(arguments.op == 'modulated_rms_norm').lower()}"
            f"_retain_false"
        )
    print(f"descriptor: {key} ({arguments.op})")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    # The hand case is keyed by the form the operator computes, not by its name: `HAND` holds
    # `modulated` and `gated`, and this is the same test the descriptor's key uses to say which
    # form a cell is — one derivation, two readers.
    form = "modulated" if arguments.op == "modulated_rms_norm" else "gated"
    failures = check_hand(form)
    for case in cases(arguments.op, arguments):
        # Triton's interpreter truncates a bf16 store and rounds an f16 one; hardware rounds.
        case["truncate"] = arguments.interpreter and case["dtype"] == "bf16"
        case["bound"] = arguments.bound
        case["interpreter"] = arguments.interpreter
        failures += check(arguments.op, case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
