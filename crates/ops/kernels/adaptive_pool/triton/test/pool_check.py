#!/usr/bin/env python3
"""Run one adaptive_pool module on a GPU and check it against the two reductions.

    python pool_check.py python pool_check.py --dtype bf16 --heads 16 --width
    256 --output 64 python pool_check.py --heads 8 --width 8 --output 16

The package's Triton source is launched through Triton's own launcher, which is
the code the store's module is compiled from. The head count, the width, the
pooled width and the block are all constexprs of the launched kernel, so a run
can choose a geometry no descriptor declares — which the third example above
does: with more outputs than inputs, `floor(i·width/output)` and
`ceil((i+1)·width/output)` give an empty window at every position and the pool
must take one input position per output, repeating it. That is the degenerate
case, and it is checked on purpose.

The reference is the model's own two steps: the mean over heads, then
`torch.nn.functional.adaptive_avg_pool1d` over the width. The bound is the
operator's: `rounding(heads · ceil(width/output), F32, dtype)` — the
statistical bound for one FP32 accumulation rounded once — with its absolute
envelope `64 · u_out`, which is what an output element near zero needs when a
window's terms cancel.
"""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as functional

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: `DType::unit_roundoff` for each element type, as `ops` states it.
UNIT_ROUNDOFF = {
    "f16": 4.8828125e-04,
    "bf16": 3.90625e-03,
    "f32": 5.960464477539063e-08,
}


def bound(heads, width, output, dtype):
    """The operator's tolerance for one descriptor."""
    window = max(-(-width // output), 1)
    terms = heads * window
    gamma = (terms * UNIT_ROUNDOFF["f32"]) / (
        1.0 - terms * UNIT_ROUNDOFF["f32"]
    )
    relative = 2.0 * gamma + UNIT_ROUNDOFF[dtype]
    return relative, 64.0 * UNIT_ROUNDOFF[dtype]


def launch(dtype, heads, width, output, block, rows, pdl, seed=0):
    """Launch the package's kernel; answer what it wrote and what it read."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    import adaptive_pool as kernels

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, heads, width, generator=generator).to(
        device=device, dtype=TORCH_TYPES[dtype]
    )
    # The attended values of a projection are bounded in practice; a plain
    # normal keeps the comparison about the accumulation rather than about a
    # designed distribution.
    out = torch.empty((rows, output), dtype=TORCH_TYPES[dtype], device=device)
    total = rows * output
    window = max(-(-width // output), 1)
    kernels._adaptive_pool_kernel[((total + block - 1) // block,)](
        x,
        out,
        rows,
        heads,
        width,
        output,
        1 if pdl else 0,
        HEADS=heads,
        WIDTH=width,
        OUTPUT=output,
        WINDOW=window,
        BLOCK=block,
        PDL=pdl,
        num_warps=max(1, min(8, block // 256)),
        launch_pdl=pdl,
    )
    torch.cuda.synchronize()
    return x, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="f32")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output", type=int, default=32)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--pdl", action="store_true")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

    cases = [
        (arguments.heads, arguments.width, arguments.output),
        # The degenerate pool: more outputs than inputs, every window one
        # position.
        (arguments.heads, 8, 16),
        # A width the pooled width divides evenly, so every window is the same
        # length.
        (arguments.heads, 256, 64),
    ]
    failures = 0
    for heads, width, output in cases:
        x, out = launch(
            arguments.dtype,
            heads,
            width,
            output,
            arguments.block,
            arguments.rows,
            arguments.pdl,
        )
        # The model's two steps: the mean over heads, then the pool along the
        # width. `adaptive_avg_pool1d` reads a two-dimensional input as
        # (channels, length), which is exactly the row and width here.
        averaged = x.double().mean(dim=1)
        expected = functional.adaptive_avg_pool1d(averaged, output)
        relative, absolute = bound(heads, width, output, arguments.dtype)
        got = out.double()
        allowed = absolute + relative * expected.abs()
        difference = (got - expected).abs()
        wrong = difference > allowed
        window = max(-(-width // output), 1)
        print(
            f"{arguments.dtype}, {heads} heads of {width} pooled to {output} "
            f"(window {window}), {arguments.rows} rows: max |out - exact| = "
            f"{difference.max().item():.6g}, allowed {allowed.max().item():.6g}"
            f"{', ' if not wrong.any() else ''}"
            f"{'' if not wrong.any() else f'{int(wrong.sum())} element(s) outside the bound'}"
        )
        if wrong.any():
            failures += 1
            row, position = (int(value) for value in torch.argwhere(wrong)[0])
            print(
                f"  at [row {row}, output {position}]: got {got[row, position].item():.9g}, "
                f"exact {expected[row, position].item():.9g}",
                file=sys.stderr,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
