"""The slot_commit program leaves every slot's state as the formulas say.

One launch folds a step's rows into the slots' continuation state, so the
check is a state check: a pool of slots holds distinct known values, a few
rows commit into some of them, and every element of every state tensor is
compared, the slots the step touched against the operator's formulas and the
rest against what they held.

**Every state tensor has a guard element in front**, and the program is handed
the view after it: a row whose slot is the padding sentinel ``-1`` addresses
``-1``, the guard. A warm batch is exactly a step of padding rows, and a
program that wrote the lengths for them would write outside every state
tensor while producing plausible tokens, so a moved guard fails, and every
case but one stages a padding row.

The operator's formulas (``crates/ops/src/commit.rs``)::

    decode    verified += 1          logical += 1          sampling += 1
    verify    verified  = visible + points     logical = host_logical + points
    explicit  verified  = host cache length, positions from the host columns
    cache     the lengths as the host columns state them (a row with no sample)
    count     penalties[slot, token] += valid * active, nothing else
    sampled   future[slot, 0] = token, predicates[slot] = continuation,
              penalties[slot, token] += valid * active

A token at or above the vocabulary is no token: it installs no next input and
adds no penalty. Everything is an integer and the operator's tolerance is
exact, so the comparison is element for element. The tables are the package's
descriptor (``index``, ``verified`` and ``penalties`` all ``i32``); its
compiled strides and blocks are parameters, at the values the check states.
"""

import pytest
import torch
import triton.language as tl

pytestmark = pytest.mark.unit

#: The commit table's rows, as the operator numbers them.
SLOT, MODE, TASK, CACHE_LENGTH, LOGICAL, SAMPLING, VISIBLE, PENALTIES = range(8)

#: The modes, as the operator numbers them.
DECODE, EXPLICIT, VERIFY, CACHE, COUNT = 0, 1, 2, 3, 4

#: A token relay carries the continuation in bit 31 and the token below it.
TOKEN_VALUE_MASK = (1 << 31) - 1

#: The table's, the lengths' and the counts' element type: the package's
#: descriptor declares `i32` for all three.
INDEX = torch.int32

#: The state tensors, in the order a failure reports them, with their element
#: types and the range of their starting values.
STATE = {
    "verified": (torch.int32, 900),
    "logical": (INDEX, 5000),
    "sampling": (INDEX, 5000),
    "predicates": (torch.uint8, 2),
    "future": (INDEX, 100000),
    "penalties": (torch.int32, 5),
}

#: The compiled table capacity, next input stride and block the cases run at.
OPTIONS = {
    "stride 32, future stride 8, block 8": (32, 8, 8),
    "stride 256, future stride 16, block 64": (256, 16, 64),
}

#: The steps, each a row shape that decides its own state: the slots, the
#: vocabulary, the next input's width, the seed, per task its token, valid,
#: active and continuation flags and accepted points, and the rows.
SCENARIOS = {
    "a decode step, a verification and two prefills over four slots": {
        "slots": 4,
        "vocab": 16,
        "width": 4,
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
    # A verification that accepted two drafts: its verify row installs the
    # token after them, and one count row per draft adds the draft to the
    # slot's penalties, including a draft that is the installed token.
    "a verification's accepted drafts count penalties and move nothing else": {
        "slots": 3,
        "vocab": 16,
        "width": 2,
        "seed": 5,
        "tokens": [9, 4, 9, 2],
        "valid": [1, 1, 1, 1],
        "active": [1, 1, 1, 1],
        "continuation": [1, 0, 0, 1],
        "points": [3, 0, 0, 1],
        "rows": [
            [1, VERIFY, 0, 0, 50, 60, 48, 1],
            [1, COUNT, 1, 0, 0, 0, 0, 1],
            [1, COUNT, 2, 0, 0, 0, 0, 1],
            [2, DECODE, 3, 0, 0, 0, 0, 0],
        ],
    },
    "a step whose only rows are padding: nothing moves at all": {
        "slots": 2,
        "vocab": 8,
        "width": 2,
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
    # A task whose sample is invalid, or inactive, adds no penalty weight:
    # speculation counts only its final emitted token.
    "two slots drawing the same token: one counts a penalty, one may not": {
        "slots": 2,
        "vocab": 8,
        "width": 2,
        "seed": 3,
        "tokens": [4, 4],
        "valid": [1, 0],
        "active": [1, 1],
        "continuation": [0, 0],
        "points": [1, 1],
        "rows": [
            [0, DECODE, 0, 0, 0, 0, 0, 1],
            [1, DECODE, 1, 0, 0, 0, 0, 1],
        ],
    },
    # The sampler can answer one past its vocabulary (an `argmax` over a row
    # with no numeric value answers its width). The impossible draw is in the
    # first slot, so an unguarded penalty add would land in the second slot's
    # row of the count plane, where the comparison reads it.
    "a draw that answered the vocabulary installs no token and no penalty": {
        "slots": 2,
        "vocab": 8,
        "width": 2,
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
}


def launch(kernels, columns, tasks, state, count, stride, future_stride, block):
    """One launch: one program applying every row of the step.

    ``tasks`` are the tokens, the valid, active and continuation flags and
    the accepted points; ``state`` the views after each tensor's guard.
    """
    kernels._slot_commit_kernel[(1,)](
        columns,
        *tasks,
        state["future"],
        state["predicates"],
        state["logical"],
        state["sampling"],
        state["verified"],
        state["penalties"],
        count,
        INDEX=tl.int32,
        VERIFIED=tl.int32,
        PENALTY_DTYPE=tl.int32,
        STRIDE=stride,
        FUTURE_STRIDE=future_stride,
        VOCAB=state["penalties"].shape[1],
        BLOCK=block,
    )


def pool(scenario, future_stride, device):
    """A pool of distinct slot states, each tensor a view after a guard
    element holding 7; returns the views and their whole buffers.
    """  # noqa: D205
    generator = torch.Generator().manual_seed(scenario["seed"])
    slots, vocab, width = (
        scenario["slots"],
        scenario["vocab"],
        scenario["width"],
    )
    shapes = {
        "verified": (slots,),
        "logical": (slots,),
        "sampling": (slots,),
        "predicates": (slots,),
        "future": (slots, width),
        "penalties": (slots, vocab),
    }
    state, buffers = {}, {}
    for name in ("verified", "logical", "sampling", "predicates"):
        dtype, high = STATE[name]
        buffers[name] = torch.full((1 + slots,), 7, dtype=dtype)
        state[name] = buffers[name][1:]
        state[name][:] = torch.randint(0, high, (slots,), generator=generator)
    # The next input matrix is wider than the columns a call writes: the
    # module is compiled with the stride between two slots' rows.
    buffers["future"] = torch.full((1 + slots * future_stride,), 7, dtype=INDEX)
    state["future"] = buffers["future"][1:].as_strided(
        shapes["future"], (future_stride, 1)
    )
    state["future"][:] = torch.randint(
        0, STATE["future"][1], shapes["future"], generator=generator
    )
    buffers["penalties"] = torch.full(
        (1 + slots * vocab,), 7, dtype=torch.int32
    )
    state["penalties"] = buffers["penalties"][1:].view(shapes["penalties"])
    state["penalties"][:] = torch.randint(
        0, STATE["penalties"][1], shapes["penalties"], generator=generator
    )

    # The whole buffers move, so each device view keeps its guard in front.
    moved = {name: buffer.to(device) for name, buffer in buffers.items()}
    views = {
        name: torch.as_strided(
            moved[name], view.shape, view.stride(), storage_offset=1
        )
        for name, view in state.items()
    }
    return views, moved


def reference(state, scenario):
    """The state the operator's formulas say a step leaves."""
    expected = {name: view.cpu().clone() for name, view in state.items()}
    tokens, valid, active = (
        scenario["tokens"],
        scenario["valid"],
        scenario["active"],
    )
    for row in scenario["rows"]:
        slot, mode, task = row[SLOT], row[MODE], row[TASK]
        if slot < 0:
            continue
        sampled = task >= 0
        token = int(tokens[task]) & TOKEN_VALUE_MASK if sampled else None
        counts = (
            sampled
            and token < scenario["vocab"]
            and row[PENALTIES] != 0
            and valid[task] != 0
            and active[task] != 0
        )
        if mode == COUNT:
            if counts:
                expected["penalties"][slot, token] += 1
            continue
        if mode == DECODE:
            expected["verified"][slot] += 1
            if sampled:
                expected["logical"][slot] += 1
                expected["sampling"][slot] += 1
        elif mode == VERIFY:
            points = scenario["points"][task] if sampled else 0
            expected["verified"][slot] = row[VISIBLE] + points
            if sampled:
                expected["logical"][slot] = row[LOGICAL] + points
                expected["sampling"][slot] = row[SAMPLING] + points
        else:
            expected["verified"][slot] = row[CACHE_LENGTH]
            if sampled:
                expected["logical"][slot] = row[LOGICAL]
                expected["sampling"][slot] = row[SAMPLING]
        # An id at or above the vocabulary is no token: no next input and no
        # penalty.
        if not sampled or token >= scenario["vocab"]:
            continue
        expected["future"][slot, 0] = token
        expected["predicates"][slot] = scenario["continuation"][task]
        if counts:
            expected["penalties"][slot, token] += 1
    return expected


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_slot_holds_the_state_its_rows_leave(
    scenario, option, torch_device, kernel_source
):
    name, scenario = scenario, SCENARIOS[scenario]
    stride, future_stride, block = OPTIONS[option]
    state, buffers = pool(scenario, future_stride, torch_device)
    expected = reference(state, scenario)

    # The step's rows as the table's columns, `stride` elements apart.
    rows = scenario["rows"]
    table = torch.zeros(8 * stride, dtype=INDEX)
    columns = torch.as_strided(table, (8, len(rows)), (stride, 1))
    columns[:] = torch.tensor(rows, dtype=INDEX).t()
    # The flags are bytes in the module's signature: a flag is a boolean.
    tasks = [
        torch.tensor(scenario[field], dtype=dtype, device=torch_device)
        for field, dtype in (
            ("tokens", INDEX),
            ("valid", torch.uint8),
            ("active", torch.uint8),
            ("continuation", torch.uint8),
            ("points", INDEX),
        )
    ]

    launch(
        kernel_source("slot_commit"),
        torch.as_strided(table.to(torch_device), (8, len(rows)), (stride, 1)),
        tasks,
        state,
        len(rows),
        stride,
        future_stride,
        block,
    )

    faults = []
    for tensor in STATE:
        got, want = state[tensor].cpu(), expected[tensor]
        if not torch.equal(got, want):
            where = tuple((got != want).nonzero()[0].tolist())
            faults.append(
                f"{tensor} at {where}: got {got[where].item()}, expected "
                f"{want[where].item()}"
            )
        # The guard is where a padding row would land.
        if buffers[tensor][0].item() != 7:
            faults.append(f"{tensor}'s guard was written: a padding row wrote")
    assert not faults, f"{name}, {option}: " + "; ".join(faults)
