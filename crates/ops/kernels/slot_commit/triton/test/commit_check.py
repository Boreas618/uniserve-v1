#!/usr/bin/env python3
"""Run the slot_commit program and check the state it leaves, slot by slot.

    python commit_check.py                    # a decode, a verification and a prefill step
    python commit_check.py --interpreter      # the same on the CPU, through Triton's interpreter
    python commit_check.py --stride 256 --future-stride 16 --block 64

One launch folds a step's rows into the slots' state, so the check is a state check: a
pool of slots is filled with distinct known values, a handful of rows commit into a few
of them, and every slot of every state tensor is compared — the ones the step touched
against the operator's formulas, and the ones it did not against what they held.

**Every state tensor is given a guard element in front**, and the tensors are passed as
the view after it: a row whose slot is the padding sentinel `-1` addresses `-1`, which
lands in the guard. That is not a contrived failure — a warm batch is exactly a step of
padding rows, and a kernel that wrote the lengths for those rows would write outside every
state tensor while producing perfectly plausible tokens. The check therefore fails if any
guard element moved, and it stages a padding row in every case.

The operator's formulas, from `crates/ops/src/commit.rs`:

    decode    verified += 1          logical += 1                sampling += 1
    verify    verified  = visible + points                      logical = host_logical + points
    explicit  verified  = host cache length, and the positions from the host columns
    cache     the lengths are left as they are (a row with no sample)
    sampled   future[slot, 0] = token, predicates[slot] = continuation,
              penalties[slot, token] += valid · active

Everything is an integer: the operator's tolerance is `exact()`, so the comparison is
element for element, with no tolerance to choose.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The commit table's rows, as the operator numbers them.
SLOT, MODE, TASK, CACHE_LENGTH, LOGICAL, SAMPLING, VISIBLE, PENALTIES = range(8)

#: The modes, as the operator numbers them.
DECODE, EXPLICIT, VERIFY, CACHE = 0, 1, 2, 3

#: A token relay carries the continuation in bit 31 and the token below it.
TOKEN_VALUE_MASK = (1 << 31) - 1

#: The state tensors, in the order the check reports them.
STATE = ("verified", "logical", "sampling", "future", "predicates", "penalties")


def guarded(shape, dtype, fill):
    """A tensor with one guard element in front, and the view after it."""
    elements = 1
    for extent in shape:
        elements *= extent
    base = torch.full((elements + 1,), fill, dtype=dtype)
    return base, base[1:].view(shape)


def as_strided_rows(rows, columns, stride, fill):
    """A `[rows, columns]` view whose rows are `stride` elements apart."""
    base = torch.full((rows * stride,), fill, dtype=torch.int64)
    view = torch.as_strided(base, (rows, columns), (stride, 1))
    return base, view


def pool(slots, vocab, width, future_stride, seed, device):
    """A pool of distinct slot states, each with a guard element in front."""
    generator = torch.Generator().manual_seed(seed)
    state = {}
    guards = {}
    for name, dtype, shape, high in (
        ("verified", torch.int32, (slots,), 900),
        ("logical", torch.int64, (slots,), 5000),
        ("sampling", torch.int64, (slots,), 5000),
        ("predicates", torch.uint8, (slots,), 2),
    ):
        base, view = guarded(shape, dtype, 7)
        view[:] = torch.randint(0, high, shape, generator=generator).to(dtype)
        state[name] = view.to(device)
        guards[name] = base
    # The next-input matrix is wider than the columns a call writes: the module is
    # compiled with the stride between two slots' rows, so the view is built with it.
    base = torch.full((1 + slots * future_stride,), 7, dtype=torch.int64)
    view = base[1:].as_strided((slots, width), (future_stride, 1))
    view[:] = torch.randint(0, 100000, (slots, width), generator=generator)
    state["future"] = view.to(device)
    guards["future"] = base
    base, view = guarded((slots, vocab), torch.int32, 7)
    view[:] = torch.randint(0, 5, (slots, vocab), generator=generator).to(
        torch.int32
    )
    state["penalties"] = view.to(device)
    guards["penalties"] = base
    return state, guards


def reference(state, rows, tokens, valid, active, continuation, points, vocab):
    """The state the operator's formulas say a step leaves."""
    expected = {name: value.clone() for name, value in state.items()}
    for row in rows:
        slot, mode, task = row[SLOT], row[MODE], row[TASK]
        if slot < 0:
            continue
        sampled = task >= 0
        verified = expected["verified"]
        if mode == DECODE:
            expected["verified"][slot] = verified[slot] + 1
            if sampled:
                expected["logical"][slot] += 1
                expected["sampling"][slot] += 1
        elif mode == VERIFY:
            expected["verified"][slot] = row[VISIBLE] + (
                points[task] if sampled else 0
            )
            if sampled:
                expected["logical"][slot] = row[LOGICAL] + points[task]
                expected["sampling"][slot] = row[SAMPLING] + points[task]
        else:
            expected["verified"][slot] = row[CACHE_LENGTH]
            if sampled:
                expected["logical"][slot] = row[LOGICAL]
                expected["sampling"][slot] = row[SAMPLING]
        if not sampled:
            continue
        token = int(tokens[task]) & TOKEN_VALUE_MASK
        # An id at or above the vocabulary is not a token: the operator installs no next input column
        # and adds no penalty for it, so neither moves here either.
        if token >= vocab:
            continue
        expected["future"][slot, 0] = token
        expected["predicates"][slot] = continuation[task]
        if row[PENALTIES] != 0 and valid[task] != 0 and active[task] != 0:
            expected["penalties"][slot, token] += 1
    return expected


def run(scenario, device):
    """Stage one step, launch the kernel, and compare the whole pool."""
    import triton.language as tl

    import slot_commit as kernels

    slots = scenario["slots"]
    vocab = scenario["vocab"]
    state, guards = pool(
        slots,
        vocab,
        scenario["width"],
        scenario["future_stride"],
        scenario["seed"],
        device,
    )
    tokens = torch.tensor(scenario["tokens"], dtype=torch.int64, device=device)
    # The flags are bytes in the module's signature: a flag is a boolean, not a length.
    valid = torch.tensor(scenario["valid"], dtype=torch.uint8, device=device)
    active = torch.tensor(scenario["active"], dtype=torch.uint8, device=device)
    continuation = torch.tensor(
        scenario["continuation"], dtype=torch.uint8, device=device
    )
    points = torch.tensor(scenario["points"], dtype=torch.int64, device=device)

    rows = scenario["rows"]
    count = len(rows)
    _, columns = as_strided_rows(8, count, scenario["stride"], 0)
    for at, row in enumerate(rows):
        columns[:, at] = torch.tensor(row, dtype=torch.int64)

    expected = reference(
        state,
        rows,
        scenario["tokens"],
        scenario["valid"],
        scenario["active"],
        scenario["continuation"],
        scenario["points"],
        scenario["vocab"],
    )
    # Only the element in front of each view is a guard; the rest of its buffer is the
    # state the step is meant to write.
    guards_before = {
        name: base.flatten()[0].item() for name, base in guards.items()
    }

    kernels._slot_commit_kernel[(1,)](
        columns,
        tokens,
        valid,
        active,
        continuation,
        points,
        state["future"],
        state["predicates"],
        state["logical"],
        state["sampling"],
        state["verified"],
        state["penalties"],
        count,
        INDEX=tl.int64,
        VERIFIED=tl.int32,
        PENALTY_DTYPE=tl.int32,
        STRIDE=scenario["stride"],
        FUTURE_STRIDE=scenario["future_stride"],
        VOCAB=vocab,
        BLOCK=scenario["block"],
    )

    failures = []
    for name in STATE:
        got = state[name].cpu()
        want = expected[name].cpu()
        if not torch.equal(got, want):
            where = (got != want).nonzero()[0].tolist()
            failures.append(
                f"{name} at {where}: got {got[tuple(where)].item()}, "
                f"expected {want[tuple(where)].item()}"
            )
        # A state tensor's guard element is where a padding row would land.
        if guards[name].flatten()[0].item() != guards_before[name]:
            failures.append(
                f"{name}'s guard element was written: a padding row wrote state"
            )
    return failures


def scenarios(flags):
    """The steps one run checks, each for a row shape that decides its own state."""
    stride = flags.stride
    return [
        {
            "label": "a decode step, a verification and two prefills over four slots",
            "slots": 4,
            "vocab": 16,
            "width": 4,
            "stride": stride,
            "future_stride": flags.future_stride,
            "block": flags.block,
            "seed": 1,
            "tokens": [12, 3, 7],
            "valid": [1, 0, 1],
            "active": [1, 1, 0],
            "continuation": [0, 1, 1],
            "points": [2, 1, 3],
            "rows": [
                [1, DECODE, 0, 0, 0, 0, 0, 1],
                [2, VERIFY, 1, 0, 100, 200, 40, 0],
                [3, EXPLICIT, 2, 77, 300, 400, 0, 1],
                [0, CACHE, -1, 55, 0, 0, 0, 0],
                [-1, -1, -1, 0, 0, 0, 0, 0],
            ],
        },
        {
            "label": "a step whose only rows are padding: nothing moves at all",
            "slots": 2,
            "vocab": 8,
            "width": 2,
            "stride": stride,
            "future_stride": flags.future_stride,
            "block": flags.block,
            "seed": 2,
            "tokens": [5],
            "valid": [1],
            "active": [1],
            "continuation": [1],
            "points": [1],
            "rows": [
                [-1, -1, -1, 0, 0, 0, 0, 0],
                [-1, -1, -1, 0, 0, 0, 0, 0],
            ],
        },
        {
            "label": "two slots drawing the same token: one counts a penalty, one may not",
            "slots": 2,
            "vocab": 8,
            "width": 2,
            "stride": stride,
            "future_stride": flags.future_stride,
            "block": flags.block,
            "seed": 3,
            "tokens": [4, 4],
            # A task whose sample is invalid, or inactive, adds no penalty weight —
            # speculation counts only its final emitted token.
            "valid": [1, 0],
            "active": [1, 1],
            "continuation": [0, 0],
            "points": [1, 1],
            "rows": [
                [0, DECODE, 0, 0, 0, 0, 0, 1],
                [1, DECODE, 1, 0, 0, 0, 0, 1],
            ],
        },
        {
            # The sampler can answer one past its vocabulary — an `argmax` over a row with no
            # numeric value answers the row's width. Such a value is not a token, and the operator
            # installs nothing for it: the next input column and the penalty counts keep what they
            # held. The impossible draw is in the *first* slot, so an unguarded penalty add would
            # land in the second slot's row of the count plane where the comparison reads it.
            "label": "a draw that answered the vocabulary installs no token and no penalty",
            "slots": 2,
            "vocab": 8,
            "width": 2,
            "stride": stride,
            "future_stride": flags.future_stride,
            "block": flags.block,
            "seed": 4,
            "tokens": [8, 3],
            "valid": [1, 1],
            "active": [1, 1],
            "continuation": [0, 1],
            "points": [1, 1],
            "rows": [
                [0, DECODE, 0, 0, 0, 0, 0, 1],
                [1, DECODE, 1, 0, 0, 0, 0, 1],
            ],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--future-stride", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=0)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if not arguments.interpreter and not torch.cuda.is_available():
        # A host without a device takes the interpreter: a check must run where it lands.
        print(
            "no GPU on this host: running through Triton's interpreter instead",
            file=sys.stderr,
        )
        arguments.interpreter = True
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"

    device = "cpu" if arguments.interpreter else "cuda"
    failures = 0
    for scenario in scenarios(arguments):
        if arguments.vocab:
            scenario["vocab"] = arguments.vocab
        found = run(scenario, device)
        print(f"{'exact' if not found else 'DIFFERS'}: {scenario['label']}")
        for failure in found:
            print(f"  {failure}", file=sys.stderr)
        failures += len(found)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
