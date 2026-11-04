#!/usr/bin/env python3
"""Run the embedding program and check the copied rows bit for bit.

    python embedding_check.py                               # a small table, ids inside the shard
    python embedding_check.py --first 100 --rows 32          # a shard's offset, and ids outside it
    python embedding_check.py --ids i64 --interpreter

`embedding` copies the rows an id table selects out of a weight table: `out[token] =
weight[id - first]` over `[tokens, hidden]`, where the weight table holds this rank's `rows`
rows starting at the vocabulary offset `first`. **An id outside `[first, first + rows)` writes
zeros** — a token whose embedding lives on another rank contributes nothing, and a row of
another rank's values would be a silent aliasing rather than an absent embedding. Its
tolerance is `Tolerance::exact()`: a lookup copies values and rounds nothing.

Cases: every id inside the shard; a shard offset with ids on both sides of it, so the zeros are
exercised next to real rows; `i64` ids, whose arithmetic is wider than the table; and a
`hidden` that the inner tile does not divide, so the column tail is masked. The shard's `rows`
is small on purpose — the operator is the same at 151936 rows, and a check that needs a
gigabyte of weights to run is a check that stops being run.

**The ids are not all distinct and are not all inside**: a wrong `first`, a wrong row stride or
a missing zero-fill each changes a known element, and the print says which.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

ID_TYPES = {"i32": torch.int32, "i64": torch.int64}


def check(case):
    """Copy one case's rows and compare them exactly."""
    import embedding as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    tokens, hidden = case["tokens"], case["hidden"]
    rows, first = case["rows"], case["first"]

    weight = torch.randn(rows, hidden, generator=generator).to(dtype)
    ids = torch.full((tokens,), first, dtype=ID_TYPES[case["ids"]])
    if case["outside"]:
        # Two ids below the shard and two above it: every one of them must write zeros.
        ids[0] = first - 1 if first > 0 else 0
        ids[-1] = first + rows
        ids[len(ids) // 2] = first + rows + 5
    if case["duplicate"]:
        # One row serves several tokens; a kernel that walks its rows instead of its ids fails.
        ids[1] = ids[0]
    out = torch.full((tokens, hidden), float("nan"), dtype=dtype)
    programs = -(-tokens // case["block"])

    kernels._embedding_kernel[(programs,)](
        ids,
        weight,
        out,
        tokens,
        hidden,
        first,
        rows,
        BLOCK=case["block"],
        TILE=case["tile"],
        PDL=case["pdl"],
    )

    local = ids.cpu().long() - first
    inside = (local >= 0) & (local < rows)
    expected = torch.zeros(tokens, hidden, dtype=torch.float64)
    expected[inside] = weight.cpu().double()[local[inside]]
    if case["dtype"] != "f32":
        expected = expected.to(dtype).double()
    got = out.cpu().double()
    difference = (got != expected).sum().item()
    print(
        f"{'exact' if difference == 0 else 'FAIL'}: {case['label']} — "
        f"{tokens} tokens of hidden {hidden} from {rows} rows at first {first}, "
        f"block {case['block']} of tile {case['tile']}, {case['dtype']} values and "
        f"{case['ids']} ids, pdl {str(case['pdl']).lower()} — "
        f"{difference} of {expected.numel()} elements differ"
    )
    if difference:
        wrong = (got != expected).nonzero()[:3].tolist()
        for token, column in wrong:
            print(
                f"  [{token}, {column}]: got {got[token, column].item():.9e}, the table says "
                f"{expected[token, column].item():.9e} (id {ids[token].item()})"
            )
    return difference


def cases(flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {
        "dtype": flags.dtype,
        "hidden": flags.hidden,
        "block": flags.block,
        "tile": flags.tile,
        "pdl": flags.pdl == "true",
        "ids": flags.ids,
    }
    return [
        {
            **common,
            "label": "every id inside the shard",
            "tokens": flags.tokens,
            "rows": flags.rows,
            "first": 0,
            "outside": False,
            "duplicate": False,
            "seed": 1,
        },
        {
            **common,
            # The shard's offset, with ids on both sides of it: the zeros are as much a part of
            # the contract as the copied rows.
            "label": "a shard's offset, and ids outside it",
            "tokens": flags.tokens,
            "rows": flags.rows,
            "first": flags.first,
            "outside": True,
            "duplicate": False,
            "seed": 2,
        },
        {
            **common,
            "label": "one row serving several tokens",
            "tokens": max(flags.tokens, 4),
            "rows": flags.rows,
            "first": flags.first,
            "outside": False,
            "duplicate": True,
            "seed": 3,
        },
        {
            **common,
            "label": "a column tail the tile does not cover",
            "tokens": flags.tokens,
            "rows": flags.rows,
            "first": 0,
            "hidden": flags.tile + 100,
            "outside": False,
            "duplicate": False,
            "seed": 4,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--ids", choices=sorted(ID_TYPES), default="i32")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--first", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--pdl", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if not arguments.interpreter and not torch.cuda.is_available():
        print(
            "no GPU on this host: running through Triton's interpreter instead",
            file=sys.stderr,
        )
        arguments.interpreter = True
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"

    key = (
        f"dtype_{arguments.dtype}_ids_{arguments.ids}_hidden_{arguments.hidden}"
        f"_first_{arguments.first}_rows_{arguments.rows}"
        f"_block_{arguments.block}_pdl_{arguments.pdl}"
    )
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments):
        failures += check(case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
