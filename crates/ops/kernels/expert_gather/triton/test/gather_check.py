#!/usr/bin/env python3
"""Run the expert_gather program and check the mapping exactly.

    python gather_check.py                      # the fixture's own vocabulary, 128 experts
    python gather_check.py --experts 64 --top 2
    python gather_check.py --tokens 20 --block 32

`expert_gather` maps the router's logical expert ids to this rank's physical ones:
`out[token, slot] = map[ids[token, slot]]`, over `[tokens, top]` int32 and an `[experts]` int32
permutation. Its tolerance is `Tolerance::exact()` — an index arithmetic rounds nothing — so the
reference is the map read at the ids, and a difference of one is a wrong expert rather than a
rounding.

Cases: the fixture's vocabulary; a token count whose picks span several blocks, so the grid is
exercised past its first program; a count that fits one block with slots to spare, so the masked
tail is exercised; and an identity map, where a kernel that ignores the map is right by
accident — which is why every other case uses a permutation whose every entry is distinct from
its index, and none of its values repeats.

The launch's block comes from the module's own metadata in production (`covering(entries,
threads)`); here it is a flag, and the run states it, so the grid arithmetic is visible.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def perms(experts):
    """A permutation whose every entry differs from its index, or a rotation if none exists."""
    order = torch.randperm(experts)
    if experts > 1 and bool((order == torch.arange(experts)).any()):
        order = torch.roll(torch.arange(experts), 1)
    return order.to(torch.int32)


def check(case):
    """Map one case's ids and compare every entry exactly."""
    import expert_gather as kernels

    experts, top, tokens, block = (
        case["experts"],
        case["top"],
        case["tokens"],
        case["block"],
    )
    generator = torch.Generator().manual_seed(case["seed"])
    ids = torch.randint(0, experts, (tokens, top), generator=generator).to(
        torch.int32
    )
    if case["identity"]:
        table = torch.arange(experts).to(torch.int32)
    else:
        table = perms(experts)
    out = torch.full((tokens, top), -7, dtype=torch.int32)
    entries = tokens * top
    programs = -(-entries // block)

    kernels._expert_gather_kernel[(programs,)](
        ids,
        table,
        out,
        entries,
        BLOCK=block,
    )

    expected = table.cpu().long()[ids.cpu().long()]
    difference = (out != expected).sum().item()
    print(
        f"{'exact' if difference == 0 else 'FAIL'}: {case['label']} — "
        f"{tokens} x {top} of {experts} experts, block {block}, {programs} programs "
        f"({entries} entries) — {difference} of {entries} entries differ"
    )
    if difference:
        wrong = (out != expected).nonzero()[:3].tolist()
        for token, slot in wrong:
            print(
                f"  [{token}, {slot}]: id {ids[token, slot].item()} mapped to "
                f"{out[token, slot].item()}, the map says "
                f"{expected[token, slot].item()}"
            )
    return difference


def cases(flags):
    """The cases one run checks, cheapest and most diverse first."""
    return [
        {
            "label": "the fixture's vocabulary",
            "experts": flags.experts,
            "top": flags.top,
            "tokens": 4,
            "block": flags.block,
            "identity": False,
            "seed": 1,
        },
        {
            "label": "picks spanning several blocks",
            "experts": flags.experts,
            "top": flags.top,
            "tokens": 20,
            "block": 32,
            "identity": False,
            "seed": 2,
        },
        {
            "label": "one block with slots to spare",
            "experts": flags.experts,
            "top": flags.top,
            "tokens": 3,
            "block": 64,
            "identity": False,
            "seed": 3,
        },
        {
            "label": "an identity map",
            "experts": flags.experts,
            "top": flags.top,
            "tokens": 4,
            "block": flags.block,
            "identity": True,
            "seed": 4,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--block", type=int, default=128)
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

    key = f"experts_{arguments.experts}_top_{arguments.top}"
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments):
        failures += check(case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
