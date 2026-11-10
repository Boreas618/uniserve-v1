#!/usr/bin/env python3
"""Run the gated_rms_norm program and check every row against FP64.

    python gated_rms_norm_check.py                    # rows of 128 under a tile of 128, on a GPU
    python gated_rms_norm_check.py --interpreter      # the same on Triton's interpreter

The operator is a gated delta network's output norm: `x · rsqrt(mean(x²) + eps) · w · silu(z)`,
normalized before it is gated (FLA's `RMSNormGated`, `norm_before_gate=True`). The cases are what
a masked row reduction and a gate can get wrong: the head width the model serves, a row far below
its tile, a row of zeros (whose output is exactly zero), and gates spanning the SiLU's saturation.

The reference is FP64 and the bound the operator's: the sum of squares of `columns` terms in
FP32 (`2γ_n`), the scale, the SiLU's and the products' few FP32 ulps, and one rounding into the
output type, relative to each element. A narrow store under Triton's interpreter truncates rather
than rounds, so an interpreter run allows the one ulp that costs.
"""

import argparse
import os
import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "gated_rms_norm" / "triton"
sys.path.insert(0, str(PACKAGE / "src"))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpreter", action="store_true", help="run on Triton's interpreter"
    )
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    return parser.parse_args()


FLAGS = parse()
# The family suite's interpreter run states it in the environment, with no device visible.
FLAGS.interpreter = (
    FLAGS.interpreter or os.environ.get("TRITON_INTERPRET") == "1"
)
if FLAGS.interpreter:
    os.environ["TRITON_INTERPRET"] = "1"

import gated_rms_norm as kernels  # noqa: E402
import torch  # noqa: E402

TORCH_TYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
UNIT_ROUNDOFF = {"bf16": 2.0**-8, "f32": 2.0**-24}


def bound(columns, dtype):
    """`rounding(columns, F32, out)` plus three FP32 ulps, as `GatedRmsNorm::tolerance` states."""
    unit = 2.0**-24
    gamma = columns * unit / (1 - columns * unit)
    # Triton's interpreter truncates a narrow store (one ulp) where a device rounds (half of one).
    store = UNIT_ROUNDOFF[dtype] * (
        2 if FLAGS.interpreter and dtype != "f32" else 1
    )
    return 2 * gamma + store + 3 * unit


def check(label, rows, columns, tile, device, seed):
    torch.manual_seed(seed)
    dtype = TORCH_TYPES[FLAGS.dtype]
    x = torch.randn(rows, columns).to(dtype)
    x[0] = 0.0
    gate = (torch.randn(rows, columns) * 6.0).to(dtype)
    weight = (torch.rand(columns) + 0.5).to(dtype)
    eps = 1e-6
    out = torch.full((rows, columns), float("nan"), dtype=dtype, device=device)
    kernels._gated_rms_norm_kernel[(rows,)](
        x.to(device),
        gate.to(device),
        weight.to(device),
        out,
        columns,
        eps,
        BLOCK=tile,
        num_warps=4,
    )
    exact = x.double()
    normalized = exact * torch.rsqrt(
        (exact * exact).mean(-1, keepdim=True) + eps
    )
    z = gate.double()
    expected = normalized * weight.double() * z * torch.sigmoid(z)
    error = (out.cpu().double() - expected).abs()
    relative = bound(columns, FLAGS.dtype)
    outside = int((~(error <= relative * expected.abs())).sum())
    print(
        f"{'exact' if outside == 0 else 'FAIL'}: {label}, {rows} x {columns} of tile {tile}, "
        f"{FLAGS.dtype} — max abs {float(error.max()):.3e}, bound {relative:.3e} relative, "
        f"{outside} outside"
    )
    return outside


def main():
    device = "cpu" if FLAGS.interpreter else "cuda"
    outside = 0
    outside += check("the model's head rows", 32 * 7, 128, 128, device, 1)
    outside += check("a row far below its tile", 5, 100, 256, device, 2)
    outside += check("a wider row", 3, 2048, 2048, device, 3)
    print(
        f"{'ok' if outside == 0 else 'FAIL'}: {outside} values outside the bound"
    )
    return 0 if outside == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
