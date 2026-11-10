#!/usr/bin/env python3
"""Run the sigmoid_gated_add program and check every row against FP64.

    python sigmoid_gated_add_check.py                    # rows of 2048 in blocks of 1024, on a GPU
    python sigmoid_gated_add_check.py --interpreter      # the same on Triton's interpreter

The operator joins a shared expert to the routed ones: `target[r] += sigmoid(gate[r]) · update[r]`
with one logit per row. The cases are a row width the blocks divide, one they do not (a masked
tail), and logits spanning the sigmoid's saturation, so a row whose gate is exactly zero or one in
FP32 is in the batch.

The reference is FP64 and the bound the operator's: the sigmoid's few FP32 ulps, one product and
one sum, rounded once into the target type — `2 · u_out + 4 · u_f32` (the device exponential is
an approximation of a few FP32 ulps) relative to the sum of the two terms' magnitudes, because
the sum may cancel.
"""

import argparse
import os
import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
#: The package this check covers.
PACKAGE = KERNELS / "sigmoid_gated_add" / "triton"
sys.path.insert(0, str(PACKAGE / "src"))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpreter", action="store_true", help="run on Triton's interpreter")
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    return parser.parse_args()


FLAGS = parse()
# The family suite's interpreter run states it in the environment, with no device visible.
FLAGS.interpreter = FLAGS.interpreter or os.environ.get("TRITON_INTERPRET") == "1"
if FLAGS.interpreter:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402

import sigmoid_gated_add as kernels  # noqa: E402

TORCH_TYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
UNIT_ROUNDOFF = {"bf16": 2.0**-8, "f32": 2.0**-24}


def check(label, rows, columns, block, device, seed):
    torch.manual_seed(seed)
    dtype = TORCH_TYPES[FLAGS.dtype]
    target = torch.randn(rows, columns).to(dtype)
    update = torch.randn(rows, columns).to(dtype)
    gate = (torch.randn(rows, 1) * 8.0).to(dtype)
    gate[0, 0] = 40.0
    gate[-1, 0] = -40.0
    moved = target.clone().to(device)
    kernels._sigmoid_gated_add_kernel[((columns + block - 1) // block, rows)](
        moved, update.to(device), gate.to(device), columns, BLOCK=block, num_warps=4
    )
    weight = torch.sigmoid(gate.double())
    expected = target.double() + weight * update.double()
    terms = target.double().abs() + (weight * update.double()).abs()
    error = (moved.cpu().double() - expected).abs()
    relative = 2.0 * UNIT_ROUNDOFF[FLAGS.dtype] + 4.0 * UNIT_ROUNDOFF["f32"]
    outside = int((~(error <= relative * terms)).sum())
    print(f"{'exact' if outside == 0 else 'FAIL'}: {label}, {rows} x {columns} in blocks of "
          f"{block}, {FLAGS.dtype} — max abs {float(error.max()):.3e}, {outside} outside")
    return outside


def main():
    device = "cpu" if FLAGS.interpreter else "cuda"
    outside = 0
    outside += check("the model's hidden rows", 17, 2048, 1024, device, 1)
    outside += check("a masked tail", 4, 1000, 256, device, 2)
    print(f"{'ok' if outside == 0 else 'FAIL'}: {outside} values outside the bound")
    return 0 if outside == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
