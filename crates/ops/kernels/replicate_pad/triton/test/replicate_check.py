#!/usr/bin/env python3
"""Execute the replicate_pad kernel's map on the CPU and check it against ATen.

The kernel moves elements and repeats edges, so it can be run without a GPU:
Triton's interpreter executes the kernel body with numpy-backed tensors, one
program at a time, and the result is compared with ``torch.nn.functional.pad``
in ``replicate`` mode, which is the operator's definition.

Run it where Triton is installed (the Program Server's own environment):

    python test/replicate_check.py [--axes 3]

It checks one, two and three padded axes, an uneven pad on each (a pad that
differs before and after), a negative pad on both sides — which crops — a pad
larger than the extent, and an interior axis that is not padded. The
interpreter is slow, so the geometries are small; it is the map under test:
the index arithmetic, the clamped edges, the crop and the unused trailing
slots the operator always passes.
"""

import argparse
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import replicate_pad as kernels  # noqa: E402


def launch(x, out, leading, slots, block, axes):
    """Run the kernel over its whole grid, one interpreter program at a time."""
    total = out.numel()
    grid = ((total + block - 1) // block, 1, 1)
    flat = [
        leading,
        slots[0][0], slots[0][1], slots[0][2], slots[0][3],
        slots[1][0], slots[1][1], slots[1][2], slots[1][3],
        slots[2][0], slots[2][1], slots[2][2], slots[2][3],
        0,
    ]
    kernels._replicate_pad_kernel[grid](
        x.reshape(-1),
        out.reshape(-1),
        *flat,
        BLOCK=block,
        AXES=axes,
        PDL=False,
    )


def check(shape, pads, block):
    """One case: the operator's own pad against the kernel's map."""
    # `torch.nn.functional.pad` takes the pads from the last axis backwards,
    # two numbers per padded axis; the case states them from the first padded
    # axis instead, which is how the operator's arguments are numbered.
    torch_pads = []
    for left, right in reversed(pads):
        torch_pads.extend([left, right])
    x = torch.arange(
        torch.tensor(shape).prod().item(), dtype=torch.float32
    ).reshape(shape)
    expected = torch.nn.functional.pad(x, tuple(torch_pads), mode="replicate")
    out = torch.zeros_like(expected)

    # The operator numbers its slots from the innermost padded axis outward,
    # while a shape and its pads read outermost first.
    axes = len(pads)
    padded = shape[len(shape) - axes :]
    leading = 1
    for extent in shape[: len(shape) - axes]:
        leading *= extent
    slots = []
    for slot in range(3):
        if slot < axes:
            outermost_first = axes - 1 - slot
            slots.append(
                (
                    padded[outermost_first],
                    expected.shape[len(expected.shape) - axes + outermost_first],
                    pads[outermost_first][0],
                    pads[outermost_first][1],
                )
            )
        else:
            slots.append((0, 0, 0, 0))

    launch(x, out, leading, slots, block, axes)
    if not torch.equal(out, expected):
        difference = (out != expected).nonzero()
        first = difference[0].tolist() if len(difference) else None
        raise SystemExit(
            f"{shape} pads {pads}: differs at {first}: "
            f"{out[tuple(first)].item()} is not {expected[tuple(first)].item()}"
        )
    print(f"ok  {shape} pads {pads} block {block}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, default=64)
    arguments = parser.parse_args()
    block = arguments.block

    # One padded axis, uneven pads.
    check((2, 5), [(1, 3)], block)
    # Two padded axes, the first with a pad larger than its extent.
    check((2, 3, 4), [(5, 0), (1, 1)], block)
    # Three padded axes with an uneven pad on each.
    check((1, 3, 4, 5), [(1, 2), (0, 3), (2, 0)], block)
    # A negative pad on both sides of the last axis: a crop.
    check((2, 3, 6), [(0, 0), (-2, -1)], block)
    # A crop and a pad in the same case, from one axis to the next.
    check((2, 4, 5), [(-1, 1), (2, -1)], block)
    # A block wider than the whole output, so one program covers it.
    check((2, 3), [(2, 2)], 4096)
    print("all cases match torch's replicate pad")


if __name__ == "__main__":
    main()
