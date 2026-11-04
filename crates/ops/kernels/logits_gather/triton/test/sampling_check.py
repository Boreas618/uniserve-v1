#!/usr/bin/env python3
"""Run one operator of the sampling chain and check its output against torch.

    python sampling_check.py --op logits_gather                     # the fixture's 2048 rows
    python sampling_check.py --op vocab_unpad --interpreter          # the same on the CPU
    python sampling_check.py --op logits_shape --vocabulary 151936 --rows 2

One script covers the chain because the chain is one path — the logits row is moved, padded,
penalized, filtered, drawn from and reported — and because a defect in any step builds
cleanly, produces fluent nonsense, and gets attributed to the model. The same file is kept
in each operator's package, byte for byte, and a test in
`crates/program-server/tests/it/packages.rs` holds the copies together.

Operators, in the order the data meets them:

- `logits_gather` — copy the rows `lm_head` computes (each request's last position) out of
  the batch's hidden state. Values move, so the comparison is **exact**.
- `vocab_unpad` — drop the padding a sharded vocabulary carries. Values move, exact.
- `logits_shape` — apply the repetition, frequency and presence penalties and the
  temperature to every column. Arithmetic, compared at the operator's bound
  (`rounding(4, F32, F32)`).

Every case is chosen so a wrong stride, a wrong vocabulary offset or a missing rule shows:
gathered rows that repeat and come out of order, an unpadded width below the gathered width,
a penalty on a token whose logit is negative, a temperature of zero, a `-inf` masked logit
that a count would otherwise penalize, and a vocabulary the block does not divide. Each
operator also has its gate: the run's numbers say what a broken version reports.

A `bf16` run under Triton's interpreter truncates its reference the way the interpreter
truncates narrow stores: the value's own mantissa is truncated with no narrower type in
between, because rounding to `f32` first can carry a value across a `bf16` boundary.

**`logits_shape` is device-only.** Its kernel imports the CUDA libdevice and rounds with
`libdevice.div_rn`, `mul_rn` and `sub_rn`; the interpreter does not load the CUDA module,
and with the portable one those three return `None` — the same class of gap as `rint` and
`erf`. So the two moving operators are checked on the CPU here and the penalizing one needs
a device; the script says so instead of reporting a pass it did not observe.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The output types' unit roundoff, as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24, "i32": 0.0}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The operators this script runs.
OPERATORS = ("logits_gather", "vocab_unpad", "logits_shape")


def gamma(reduction):
    """`γₙ = n·u/(1 − n·u)` for `n` FP32 roundings, as the operator crate computes it."""
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def truncate(values):
    """The reference as Triton's interpreter stores it: one ulp low, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own mantissa.
    """
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def compare_exact(name, got, expected):
    """Compare two integer or copied tensors bit for bit."""
    difference = (got != expected).sum().item()
    line = f"  {name}: {difference} of {expected.numel()} differ from the reference"
    return difference, line


def compare_relative(name, got, expected, relative, floor, deciding_bound):
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
        f"  {name}: max absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, max relative "
        f"{largest:.3e}, declared {declared_outside} of {expected.numel()} outside, "
        f"with the operands' floor {scaled_outside} outside"
    )
    return deciding, line


def run_logits_gather(case):
    """Copy the selected rows of a hidden state and compare them exactly."""
    import logits_gather as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    tokens, columns = case["tokens"], case["columns"]
    hidden = torch.randn(tokens, columns, generator=generator).to(dtype)
    if case["orders_rows"]:
        # Rows out of order, and one request's row repeated: a copy that assumed identity
        # order, or that read `count` rows from the top, is caught here.
        rows = torch.tensor(
            [tokens - 1, 7 % tokens, 0, 7 % tokens], dtype=torch.int32
        )
    else:
        rows = torch.arange(min(tokens, 4), dtype=torch.int32)
    out = torch.full((rows.numel(), columns), float("nan"), dtype=dtype)

    kernels._logits_gather_kernel[
        (math_ceil(rows.numel() * columns, case["block"]),)
    ](hidden, rows, out, rows.numel(), columns, BLOCK=case["block"])

    expected = hidden.cpu()[rows.long()]
    failures, line = compare_exact("out", out.cpu(), expected)
    print(
        f"{'exact' if failures == 0 else 'FAIL'}: logits_gather, {case['label']} — "
        f"{rows.numel()} of {tokens} rows, {columns} columns, block {case['block']}, "
        f"{case['dtype']}"
    )
    print(line)
    return failures


def run_vocab_unpad(case):
    """Drop the padding of a gathered vocabulary and compare exactly."""
    import vocab_unpad as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, vocabulary, padded = case["rows"], case["vocabulary"], case["padded"]
    gathered = torch.randn(rows, padded, generator=generator).to(dtype)
    if case["marks_padding"]:
        # The padding columns are marked, so a copy that took the wrong offset shows at once.
        gathered[:, vocabulary:] = -7.0
    out = torch.full((rows, vocabulary), float("nan"), dtype=dtype)

    kernels._vocab_unpad_kernel[(math_ceil(rows * vocabulary, case["block"]),)](
        gathered,
        out,
        rows,
        padded,
        vocabulary,
        BLOCK=case["block"],
    )

    expected = gathered.cpu()[:, :vocabulary]
    failures, line = compare_exact("out", out.cpu(), expected)
    print(
        f"{'exact' if failures == 0 else 'FAIL'}: vocab_unpad, {case['label']} — "
        f"{rows} x {vocabulary} of {padded} gathered columns, block {case['block']}, "
        f"{case['dtype']}"
    )
    print(line)
    return failures


def run_logits_shape(case):
    """Penalize and scale every row's logits, and compare at the operator's bound."""
    import logits_shape as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, vocabulary, block = case["rows"], case["vocabulary"], case["block"]
    logits = torch.randn(rows, vocabulary, generator=generator).to(dtype)
    counts = torch.zeros(rows, vocabulary, dtype=torch.int32)
    slots = torch.arange(rows, dtype=torch.int32)
    for row in range(rows):
        # A handful of tokens already seen, one of them twice, one of them only in a row
        # whose logit is `-inf` (a masked token the penalties must leave alone).
        counts[row, 1] = 3
        counts[row, 5] = 1
        counts[row, 7] = 2
    if case["masked"]:
        logits[0, 7] = float("-inf")
    parameters = torch.zeros(rows, 8, dtype=torch.float32)
    parameters[:, 0] = case["temperature"]
    parameters[:, 4] = 1.3
    parameters[:, 5] = 0.2
    parameters[:, 6] = 0.1
    extras = None
    if case["extras"]:
        extras = torch.full((rows, case["extras"]), -1, dtype=torch.int32)
        extras[:, 0] = 9
        if case["extras"] > 1:
            extras[:, 1] = -1
    out = torch.full((rows, vocabulary), float("nan"), dtype=torch.float32)

    kernels._logits_shape_kernel[(rows, math_ceil(vocabulary, block))](
        logits,
        counts,
        slots,
        parameters,
        extras if extras is not None else slots,
        out,
        rows,
        vocabulary,
        1 if case["divide"] else 0,
        BLOCK=block,
        STEP=min(block, 4096),
        EXTRAS=case["extras"],
        DIVIDE=case["divide"],
        PDL=False,
    )

    exact_logits = logits.cpu().double()
    exact_counts = counts.double()
    if extras is not None:
        for index in range(case["extras"]):
            token = extras[:, index].cpu().long()
            for row in range(rows):
                if token[row] >= 0:
                    exact_counts[row, token[row]] += 1.0
    temperature = parameters[:, 0].cpu().double().unsqueeze(-1)
    repetition = parameters[:, 4].cpu().double().unsqueeze(-1)
    frequency = parameters[:, 5].cpu().double().unsqueeze(-1)
    presence = parameters[:, 6].cpu().double().unsqueeze(-1)

    repeated = torch.where(
        exact_logits > 0.0, exact_logits / repetition, exact_logits * repetition
    )
    adjusted = repeated - frequency * exact_counts - presence
    apply = (exact_counts > 0.0) & (exact_logits != float("-inf"))
    expected = torch.where(apply, adjusted, exact_logits)
    if case["divide"]:
        divisor = torch.where(
            temperature > 0.0, temperature, torch.ones_like(temperature)
        )
        expected = expected / divisor
    if case["truncate"]:
        expected = truncate(expected)

    relative = gamma(4) + UNIT_ROUNDOFF["f32"]
    # The penalties are subtracted from a value their own magnitudes set, not the result's.
    floor = UNIT_ROUNDOFF["f32"] * (
        repeated.abs() + (frequency * exact_counts).abs() + presence.abs()
    )
    failures, line = compare_relative(
        "out", out.cpu().double(), expected, relative, floor, case["bound"]
    )
    print(
        f"{'exact' if failures == 0 else 'FAIL'}: logits_shape, {case['label']} — "
        f"{rows} x {vocabulary}, block {block}, extras {case['extras']}, "
        f"divide {str(case['divide']).lower()}, temperature {case['temperature']:g}, "
        f"{case['dtype']} in"
        f"{', interpreter cast truncated' if case['truncate'] else ''}"
    )
    print(line)
    return failures


RUNNERS = {
    "logits_gather": run_logits_gather,
    "vocab_unpad": run_vocab_unpad,
    "logits_shape": run_logits_shape,
}


def math_ceil(total, block):
    """`ceil(total / block)` without importing math for one use."""
    return -(-total // block)


def cases(operator, flags):
    """The scenarios one run checks for one operator, cheapest and most diverse first."""
    if operator == "logits_gather":
        return [
            {
                "label": "the fixture's shape, rows out of order and one repeated",
                "dtype": flags.dtype,
                "tokens": flags.tokens,
                "columns": flags.columns,
                "block": flags.block,
                "orders_rows": True,
                "seed": 1,
            },
            {
                "label": "a copy the block does not divide, so the last program is a tail",
                "dtype": flags.dtype,
                "tokens": 3,
                "columns": flags.columns,
                "block": flags.columns * 2,
                "orders_rows": False,
                "seed": 2,
            },
        ]
    if operator == "vocab_unpad":
        return [
            {
                "label": "the fixture's width, gathered and vocabulary equal",
                "dtype": flags.dtype,
                "rows": flags.rows,
                "vocabulary": flags.vocabulary,
                "padded": flags.padded,
                "block": flags.block,
                "marks_padding": False,
                "seed": 1,
            },
            {
                "label": "a gathered width above the vocabulary, the padding marked",
                "dtype": flags.dtype,
                "rows": 3,
                "vocabulary": 1000,
                "padded": 1024,
                "block": 512,
                "marks_padding": True,
                "seed": 2,
            },
        ]
    return [
        {
            "label": "the fixture's shape, penalties and a temperature",
            "dtype": flags.dtype,
            "rows": flags.rows,
            "vocabulary": flags.vocabulary,
            "block": flags.block,
            "extras": flags.extras,
            "divide": flags.divide == "true",
            "temperature": 0.7,
            "masked": True,
            "seed": 1,
        },
        {
            "label": "a zero temperature, which divides by one",
            "dtype": flags.dtype,
            "rows": 2,
            "vocabulary": 512,
            "block": 512,
            "extras": 0,
            "divide": True,
            "temperature": 0.0,
            "masked": False,
            "seed": 2,
        },
        {
            "label": "two extra tokens, one of them counting nothing",
            "dtype": flags.dtype,
            "rows": 2,
            "vocabulary": 500,
            "block": 256,
            "extras": 2,
            "divide": False,
            "temperature": 0.7,
            "masked": True,
            "seed": 3,
        },
    ]


def descriptor_key(operator, flags):
    """The key the flags name, as each descriptor's `Display` prints it."""
    pdl = flags.pdl
    if operator == "logits_gather":
        return f"dtype_{flags.dtype}_columns_{flags.columns}_block_{flags.block}_pdl_{pdl}"
    if operator == "vocab_unpad":
        return (
            f"dtype_{flags.dtype}_vocabulary_{flags.vocabulary}"
            f"_padded_{flags.padded}_block_{flags.block}_pdl_{pdl}"
        )
    return (
        f"dtype_{flags.dtype}_vocabulary_{flags.vocabulary}_block_{flags.block}"
        f"_extras_{flags.extras}_divide_{flags.divide}_pdl_{pdl}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=OPERATORS, default="logits_gather")
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--vocabulary", type=int, default=151936)
    parser.add_argument("--padded", type=int, default=151936)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--extras", type=int, default=0)
    parser.add_argument("--divide", choices=["true", "false"], default="true")
    parser.add_argument("--pdl", default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides an arithmetic operator's run: its own (atol 0), or the "
        "same with the floor the operands' magnitudes imply",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"
    elif not torch.cuda.is_available():
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

    if arguments.op == "logits_shape" and arguments.interpreter:
        print(
            "logits_shape needs a device: its kernel rounds with libdevice's div_rn, "
            "mul_rn and sub_rn, which the interpreter either cannot load (the CUDA "
            "module) or returns None from (the portable one)",
            file=sys.stderr,
        )
        return 1

    key = descriptor_key(arguments.op, arguments)
    print(f"descriptor: {key} ({arguments.op})")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments.op, arguments):
        # The interpreter's narrow store truncates; simulate it only where it is known.
        case["truncate"] = arguments.interpreter and arguments.dtype == "bf16"
        case["bound"] = arguments.bound
        failures += RUNNERS[arguments.op](case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
