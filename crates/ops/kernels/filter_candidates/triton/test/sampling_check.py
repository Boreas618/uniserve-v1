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
- `filter_candidates` — the top-k width, the nucleus and the min-previews thresholds, which
  drop candidates by writing `-inf` over them, plus the row-validity flag. Decisions, so the
  comparison is exact; the thresholds in every case sit more than `1e-4` away from any
  candidate's cumulative boundary, because a fp32 prefix sum and a fp64 one may otherwise
  legitimately disagree about a tie.
- `inverse_cdf` — the draw: order the candidates by token id, take the inclusive cumulative
  sum, pick the lowest id whose share the draw reaches. A token id, so exact.
- `argmax` — the greedy token: the largest value's index, the lowest index among ties.
- `logprob_gather` — one token's log-probability per row, read out of the FP32 row
  `log_softmax` produced. Values move, so the comparison is exact.

Every case is chosen so a wrong stride, a wrong vocabulary offset or a missing rule shows:
gathered rows that repeat and come out of order, an unpadded width below the gathered width,
a penalty on a token whose logit is negative, a temperature of zero, a `-inf` masked logit
that a count would otherwise penalize, and a vocabulary the block does not divide. Each
operator also has its gate: the run's numbers say what a broken version reports.

A `bf16` run under Triton's interpreter truncates its reference the way the interpreter
truncates narrow stores: the value's own mantissa is truncated with no narrower type in
between, because rounding to `f32` first can carry a value across a `bf16` boundary.

**Two of the six are device-only, for two different tool gaps, and the script says so instead
of reporting a pass it did not observe.**

- `logits_shape` rounds with `libdevice.div_rn`, `mul_rn` and `sub_rn`. The interpreter does
  not load the CUDA libdevice module at all, and with the portable import the same three
  return `None` — the family of `rint` and `erf`.
- `filter_candidates` combines a *scalar* comparison with an array comparison using `|`
  (`(min_p <= 0) | (kept >= threshold)`). The interpreter evaluates the scalar side as a
  Python bool and then hands `numpy.bool_ | ndarray` to a bitwise ufunc, which refuses:
  reproduced with a five-line kernel that does nothing else, so the gap is the interpreter's
  and not this operator's.
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
OPERATORS = (
    "logits_gather",
    "vocab_unpad",
    "logits_shape",
    "filter_candidates",
    "inverse_cdf",
    "argmax",
    "logprob_gather",
)


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


def run_filter_candidates(case):
    """Filter a row of candidates in place and compare the mask and the flag exactly."""
    import filter_candidates as kernels

    generator = torch.Generator().manual_seed(case["seed"])
    rows, width = case["rows"], case["width"]
    values = torch.randn(rows, width, generator=generator).to(torch.float32)
    if case["tail_dropped"]:
        # The previous stage leaves dropped candidates as `-inf`; the thresholds must keep
        # them dropped and must measure the nucleus over what is left.
        values[:, width // 2 :] = float("-inf")
    if case["tie"]:
        values[:, 0] = 5.0
        values[:, 1] = 5.0
    ids = torch.arange(1000, 1000 + width, dtype=torch.int32).repeat(rows, 1)
    parameters = torch.zeros(rows, 8, dtype=torch.float32)
    parameters[:, 1] = case["top_k"]
    parameters[:, 2] = case["top_p"]
    parameters[:, 3] = case["min_p"]
    valid = torch.full((rows,), -1, dtype=torch.int32)
    filtered = values.clone()

    kernels._filter_candidates_kernel[(rows,)](
        filtered, ids, parameters, valid, rows, WIDTH=width
    )

    exact = values.double()
    limit = torch.where(
        torch.tensor(case["top_k"]) > 0.0,
        torch.tensor(min(case["top_k"], width)),
        torch.tensor(width),
    ).to(torch.int64)
    columns = torch.arange(width)
    kept = torch.where(
        columns < limit, exact, torch.tensor(float("-inf")).double()
    )
    maximum = kept.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(kept - maximum)
    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
    cumulative = torch.cumsum(probabilities, dim=-1)
    dropped = (cumulative - probabilities) > case["top_p"]
    kept = torch.where(dropped, torch.tensor(float("-inf")).double(), kept)
    threshold = kept.max(dim=-1, keepdim=True).values + math_log(case["min_p"])
    if case["min_p"] > 0.0:
        kept = torch.where(
            kept >= threshold, kept, torch.tensor(float("-inf")).double()
        )
    usable = ((kept != kept) | (kept == float("inf"))).sum(dim=-1) == 0
    usable = usable & ((kept != float("-inf")).sum(dim=-1) > 0)

    value_failures, value_line = compare_exact(
        "values", filtered.cpu().double(), kept
    )
    flag_failures, flag_line = compare_exact(
        "valid", valid.cpu(), usable.to(torch.int32)
    )
    print(
        f"{'exact' if value_failures + flag_failures == 0 else 'FAIL'}: filter_candidates, "
        f"{case['label']} — {rows} x {width}, top_k {case['top_k']:g}, top_p "
        f"{case['top_p']:g}, min_p {case['min_p']:g}"
    )
    print(value_line)
    print(flag_line)
    return value_failures + flag_failures


def run_inverse_cdf(case):
    """Draw one token per row and compare it exactly."""
    import inverse_cdf as kernels

    generator = torch.Generator().manual_seed(case["seed"])
    rows, width = case["rows"], case["width"]
    probabilities = torch.rand(
        rows, width, generator=generator, dtype=torch.float32
    )
    if case["zero_mass"]:
        # The *lowest* token id of every row holds no mass — id 0 on column 0 — so a draw of
        # zero has to skip it. This is the case the boundary is about: with an inclusive
        # comparison the draw lands on the filtered-out candidate.
        probabilities[:, 0] = 0.0
    if case["one_survivor"]:
        probabilities[:, 1:] = 0.0
    # Distinct ids per row, out of order, so "the lowest id the draw reaches" is well
    # defined and a row does not arrive already sorted.
    vocabulary = max(4 * width, 1024)
    ids = torch.zeros(rows, width, dtype=torch.int32)
    for row in range(rows):
        if case["zero_mass"]:
            # Ids above zero everywhere but column 0, which holds the lowest id and no mass.
            ids[row] = (
                torch.randperm(vocabulary - 1, generator=generator)[:width] + 1
            ).to(torch.int32)
            ids[row, 0] = 0
        else:
            ids[row] = torch.randperm(vocabulary, generator=generator)[
                :width
            ].to(torch.int32)
    if case["last_holds_mass"]:
        # 90 % of the mass on the candidate with the highest token id, the rest spread thin:
        # a draw near the top of the unit interval lands only on it, so a cumulative sum the
        # draw walks past — or a strict comparison at the last element — makes that token
        # unsamplable, which is the boundary this case exists for.
        probabilities[:, :] = 0.001
        for row in range(rows):
            highest = int(torch.argmax(ids[row].long()).item())
            probabilities[row, highest] = 0.9
    draws = torch.tensor(case["draws"][:rows], dtype=torch.float32)
    tokens = torch.full((rows,), -1, dtype=torch.int32)

    kernels._inverse_cdf_kernel[(rows,)](
        probabilities, ids, draws, tokens, rows, WIDTH=width
    )

    failures = 0
    wanted = []
    for row in range(rows):
        row_probabilities = probabilities[row].double()
        row_ids = ids[row].long()
        order = torch.argsort(row_ids)
        ordered_ids = row_ids[order]
        ordered = row_probabilities[order]
        total = ordered.sum()
        cumulative = torch.cumsum(ordered, dim=0)
        # Strict, as the operator's rule reads: a zero-mass candidate is never drawn, and
        # a draw of zero lands on the first candidate that has mass.
        crossed = cumulative > draws[row].double() * total
        token = ordered_ids[crossed].min().item() if crossed.any() else None
        if token is None:
            positive = ordered[ordered > 0.0]
            token = (
                ordered_ids[ordered > 0.0].max().item()
                if positive.numel()
                else 0
            )
        wanted.append(token)
    wanted = torch.tensor(wanted, dtype=torch.int32)
    failures, line = compare_exact("tokens", tokens.cpu(), wanted)
    print(
        f"{'exact' if failures == 0 else 'FAIL'}: inverse_cdf, {case['label']} — "
        f"{rows} x {width}, draws {case['draws'][:rows]}"
    )
    print(line)
    return failures


def run_argmax(case):
    """Take each row's largest value's index and compare it exactly."""
    import argmax as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    rows, width = case["rows"], case["width"]
    values = torch.randn(rows, width, generator=generator).to(dtype)
    if case["ties"]:
        # A repeated maximum, and one at the last column: the tie rule is the lowest index.
        values[0, :] = 1.0
        values[1, width - 1] = 100.0
    out = torch.full((rows,), -1, dtype=torch.int32)

    kernels._argmax_kernel[(rows,)](values, out, rows, WIDTH=width)

    exact = values.cpu()
    wanted = (
        (exact == exact.max(dim=-1, keepdim=True).values).int().argmax(dim=-1)
    )
    failures, line = compare_exact("out", out.cpu(), wanted.to(torch.int32))
    print(
        f"{'exact' if failures == 0 else 'FAIL'}: argmax, {case['label']} — "
        f"{rows} x {width}, {case['dtype']}"
    )
    print(line)
    return failures


def columns_mid(flags):
    """A column near the middle of the vocabulary, for a scattered token list."""
    return max(flags.columns // 2, 0)


def math_log(value):
    """`log` of a scalar, in torch, so the reference stays in one library."""
    return torch.log(torch.tensor(float(value))).double()


def run_logprob_gather(case):
    """Report one token's log-probability per row and compare it exactly."""
    import logprob_gather as kernels

    generator = torch.Generator().manual_seed(case["seed"])
    rows, columns = case["rows"], case["columns"]
    logprobabilities = -torch.rand(rows, columns, generator=generator).float()
    if case["masked"]:
        # A dropped candidate's log-probability is `-inf`; the gather must copy it rather
        # than repair it or turn it into a zero.
        logprobabilities[0, columns - 1] = float("-inf")
    tokens = torch.tensor(list(case["tokens"])[:rows], dtype=torch.int32)
    if case["edges"]:
        # The first and the last column, and the same token twice in different rows.
        tokens[0] = 0
        tokens[-1] = columns - 1
        if rows > 2:
            tokens[1] = columns - 1
    out = torch.full((rows,), float("nan"), dtype=torch.float32)

    kernels._logprob_gather_kernel[(rows,)](
        logprobabilities, tokens, out, rows, columns
    )

    expected = logprobabilities.cpu()[torch.arange(rows), tokens.long()]
    failures, line = compare_exact("out", out.cpu(), expected)
    print(
        f"{'exact' if failures == 0 else 'FAIL'}: logprob_gather, {case['label']} — "
        f"{rows} rows of {columns}, tokens {tokens.tolist()[:4]}"
    )
    print(line)
    return failures


RUNNERS = {
    "logits_gather": run_logits_gather,
    "vocab_unpad": run_vocab_unpad,
    "logits_shape": run_logits_shape,
    "filter_candidates": run_filter_candidates,
    "inverse_cdf": run_inverse_cdf,
    "argmax": run_argmax,
    "logprob_gather": run_logprob_gather,
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
    if operator == "filter_candidates":
        return [
            {
                "label": "a top-k width of four, a nucleus of 0.9 and a min-p of 0.1",
                "rows": 3,
                "width": flags.width,
                "top_k": 4.0,
                "top_p": 0.9,
                "min_p": 0.1,
                "tail_dropped": False,
                "tie": False,
                "seed": 1,
            },
            {
                "label": "a dropped tail, which the thresholds must leave dropped",
                "rows": 2,
                "width": flags.width,
                "top_k": 0.0,
                "top_p": 0.95,
                "min_p": 0.0,
                "tail_dropped": True,
                "tie": False,
                "seed": 2,
            },
            {
                "label": "no width and no min-p, so only the nucleus removes anything",
                "rows": 2,
                "width": flags.width,
                "top_k": 0.0,
                "top_p": 0.5,
                "min_p": -1.0,
                "tail_dropped": False,
                "tie": True,
                "seed": 3,
            },
        ]
    if operator == "inverse_cdf":
        return [
            {
                "label": "the fixture's width, four draws across the mass",
                "rows": 4,
                "width": flags.width,
                "draws": (0.0, 0.25, 0.75, 0.9999999),
                "zero_mass": False,
                "one_survivor": False,
                "last_holds_mass": False,
                "seed": 1,
            },
            {
                "label": "a zero-mass candidate at the lowest id: a draw of zero must skip it",
                "rows": 2,
                "width": flags.width,
                "draws": (0.0, 0.5),
                "zero_mass": True,
                "one_survivor": False,
                "last_holds_mass": False,
                "seed": 2,
            },
            {
                "label": "the mass only the last candidate by id holds is reachable",
                "rows": 2,
                "width": flags.width,
                "draws": (0.99, 0.9999999),
                "zero_mass": False,
                "one_survivor": False,
                "last_holds_mass": True,
                "seed": 4,
            },
            {
                "label": "one survivor, so every draw ends on it",
                "rows": 2,
                "width": flags.width,
                "draws": (0.0, 0.9999999),
                "zero_mass": False,
                "one_survivor": True,
                "last_holds_mass": False,
                "seed": 3,
            },
        ]
    if operator == "logprob_gather":
        return [
            {
                "label": "the vocabulary row, tokens spread across it",
                "rows": 4,
                "columns": flags.columns,
                "tokens": (0, columns_mid(flags), 1, 3),
                "edges": True,
                "masked": False,
                "seed": 1,
            },
            {
                "label": "a dropped candidate's -inf, gathered as it stands",
                "rows": 2,
                "columns": 64,
                "tokens": (63, 5),
                "edges": False,
                "masked": True,
                "seed": 2,
            },
        ]
    if operator == "argmax":
        return [
            {
                "label": "the fixture's width",
                "dtype": flags.dtype,
                "rows": flags.rows,
                "width": flags.width,
                "ties": False,
                "seed": 1,
            },
            {
                "label": "ties, and a maximum at the last column",
                "dtype": flags.dtype,
                "rows": 2,
                "width": flags.width,
                "ties": True,
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
    if operator == "filter_candidates":
        return f"width_{flags.width}_ids_i32_pdl_{pdl}"
    if operator == "inverse_cdf":
        return f"width_{flags.width}_ids_i32_tokens_i32_pdl_{pdl}"
    if operator == "argmax":
        return f"dtype_{flags.dtype}_width_{flags.width}_ids_i32_pdl_{pdl}"
    if operator == "logprob_gather":
        return f"columns_{flags.columns}_ids_i32_pdl_{pdl}"
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
    parser.add_argument(
        "--width",
        type=int,
        default=2048,
        help="the candidates one row of a sampling operator carries",
    )
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

    if arguments.interpreter and arguments.op == "logits_shape":
        print(
            "logits_shape needs a device: its kernel rounds with libdevice's div_rn, "
            "mul_rn and sub_rn, which the interpreter either cannot load (the CUDA "
            "module) or returns None from (the portable one)",
            file=sys.stderr,
        )
        return 1
    if arguments.interpreter and arguments.op == "filter_candidates":
        print(
            "filter_candidates needs a device: its kernel combines a scalar comparison "
            "with an array comparison using `|`, which the interpreter hands to a bitwise "
            "ufunc as `numpy.bool_ | ndarray` and which then refuses",
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
