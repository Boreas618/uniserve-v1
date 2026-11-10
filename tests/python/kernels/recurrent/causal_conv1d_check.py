#!/usr/bin/env python3
"""Run the causal_conv1d program over a batch of rows and check it against FP64.

    python causal_conv1d_check.py                    # 8192 channels of 4 taps, on a GPU
    python causal_conv1d_check.py --interpreter      # 48 channels on Triton's interpreter

The batch is the operator's: packed rows of different lengths, rows starting at position zero and
rows continuing from the inputs their slot keeps (including rows shorter than the state, whose new
state is part old state and part new input), a padding row of slot −1, rows that keep a checkpoint
inside their tokens, at their end and within the state's reach of their start, and decode rows of
one token. The inputs are a column range of a wider projection, so the program reads its row
stride rather than the view's width.

The reference is `torch.nn.functional.conv1d` in FP64 over each row's inputs preceded by its
slot's state (or by zeros), followed by the SiLU, as transformers' `causal_conv1d_fn` and
`causal_conv1d_update` compute it. The bound follows the arithmetic: `K` products of rounded
inputs and taps summed in FP32, the SiLU's few FP32 ulps, and one rounding into the output type,
measured against the sum of the products' magnitudes (the sum may cancel); the states and the
checkpoints move inputs unchanged, so they must agree exactly, and a padding row's output is
exactly zero with its slot untouched.
"""

import argparse
import os
import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
#: The package this check covers.
PACKAGE = KERNELS / "causal_conv1d" / "triton"
sys.path.insert(0, str(PACKAGE / "src"))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpreter", action="store_true", help="run on Triton's interpreter")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


FLAGS = parse()
# The family suite's interpreter run states it in the environment, with no device visible.
FLAGS.interpreter = FLAGS.interpreter or os.environ.get("TRITON_INTERPRET") == "1"
if FLAGS.interpreter:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402

import causal_conv1d as kernels  # noqa: E402

TORCH_TYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
UNIT_ROUNDOFF = {"bf16": 2.0**-8, "f32": 2.0**-24}


def run(case, device):
    """Launch one batch and count the values outside their bounds."""
    torch.manual_seed(case["seed"])
    dtype = TORCH_TYPES[case["dtype"]]
    channels, width, block = case["channels"], case["width"], case["block"]
    lengths, starts, slots, checkpoints = (case[name] for name in ("lengths", "starts", "slots",
                                                                    "checkpoints"))
    rows, tokens = len(lengths), sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    stride = channels + 64
    wide = (torch.randn(tokens, stride) * 2.0).to(dtype)
    x = wide[:, 16 : 16 + channels]
    weight = (torch.randn(channels, width) * 0.5).to(dtype)
    slot_count = max(slots) + 2
    initial = torch.randn(slot_count, width - 1, channels).to(dtype)

    expected = torch.zeros(tokens, channels, dtype=torch.float64)
    terms = torch.zeros(tokens, channels, dtype=torch.float64)
    finals, kept = {}, {}
    for r in range(rows):
        if slots[r] < 0:
            continue
        lo, hi = offsets[r], offsets[r + 1]
        history = (initial[slots[r]] if starts[r] > 0 else torch.zeros(width - 1, channels))
        extended = torch.cat([history.double(), x[lo:hi].double()], dim=0)  # [K-1+n, C]
        signal = extended.t()[None]  # [1, C, K-1+n]
        taps = weight.double()[:, None, :]
        summed = torch.nn.functional.conv1d(signal, taps, groups=channels)[0].t()
        magnitude = torch.nn.functional.conv1d(signal.abs(), taps.abs(), groups=channels)[0].t()
        expected[lo:hi] = summed * torch.sigmoid(summed)
        terms[lo:hi] = magnitude
        finals[slots[r]] = extended[-(width - 1):]
        point = checkpoints[r] - starts[r]  # the checkpoint's place among the row's new tokens
        if 0 < point <= lengths[r]:
            kept[slots[r]] = extended[point : point + width - 1]

    def column(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    seq_lens = [start + length for start, length in zip(starts, lengths)]
    state = initial.clone().to(device)
    checkpoint = torch.full_like(state, float("nan"))
    out = torch.full((tokens, channels), float("nan"), dtype=dtype, device=device)
    wide_d = wide.to(device)
    kernels._causal_conv1d_kernel[((channels + block - 1) // block, rows)](
        wide_d[:, 16 : 16 + channels], weight.to(device), out, column(offsets), column(seq_lens),
        column(slots), column(checkpoints), state, checkpoint, stride,
        CHANNELS=channels, WIDTH=width, KEPT=1 << (width - 2).bit_length(), BLOCK=block,
        TOKENS=16, num_warps=4,
    )

    outside = 0
    live = torch.tensor([slots[r] >= 0 for r in range(rows) for _ in range(lengths[r])])
    got = out.cpu().double()
    # The FP32 sum of K products and the SiLU, then one rounding: of the products' magnitudes.
    allowed = (UNIT_ROUNDOFF[case["dtype"]] + (width + 4) * 2.0**-24) * (
        expected.abs() + terms
    )
    error = (got - expected).abs()
    bad = int((~(error[live] <= allowed[live])).sum())
    outside += bad
    relative = float((error[live] / terms[live].clamp(min=1e-30)).max())
    print(f"{case['label']}, {case['dtype']}: rows {lengths} from {starts}, slots {slots}, "
          f"checkpoints {checkpoints}")
    print(f"  output: max abs {float(error[live].max()):.3e}, max relative to its terms "
          f"{relative:.3e}, {bad} of {int(live.sum()) * channels} outside")
    zeros = int((got[~live] != 0).sum())
    outside += zeros
    for slot, final in finals.items():
        differ = int((state[slot].cpu().double() != final).sum())
        outside += differ
        print(f"  state of slot {slot}: {differ} elements differ")
    for slot, expected_kept in kept.items():
        differ = int((checkpoint[slot].cpu().double() != expected_kept).sum())
        outside += differ
        print(f"  checkpoint of slot {slot}: {differ} elements differ")
    for slot in range(slot_count):
        if slot not in finals:
            outside += int((state[slot].cpu() != initial[slot]).sum())
    print(f"  padding rows: {zeros} nonzero outputs; untouched slots checked")
    return outside, float(error[live].max()), relative


def main():
    device = "cpu" if FLAGS.interpreter else "cuda"
    if FLAGS.interpreter:
        shape = {"channels": 48, "width": 4, "block": 32}
    else:
        shape = {"channels": 8192, "width": 4, "block": 256}
    outside = 0
    for case in (
        {"label": "prompt rows", "lengths": [1, 2, 7, 40, 130, 33], "starts": [0, 5, 0, 37, 64, 2],
         "slots": [3, 0, -1, 1, 2, 4], "checkpoints": [0, 6, 0, 64, 192, 3]},
        {"label": "decode rows", "lengths": [1] * 9, "starts": [3 * r + 1 for r in range(9)],
         "slots": [r if r % 4 else -1 for r in range(9)],
         "checkpoints": [3 * r + 2 if r % 2 else 0 for r in range(9)]},
    ):
        bad, _, _ = run({**case, **shape, "dtype": FLAGS.dtype, "seed": FLAGS.seed}, device)
        outside += bad
    print(f"{'ok' if outside == 0 else 'FAIL'}: {outside} values outside their bounds")
    return 0 if outside == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
