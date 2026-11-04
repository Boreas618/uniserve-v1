#!/usr/bin/env python3
"""Run one FP32-to-i16 PCM interleave on a GPU and check every sample.

The operator's tolerance is *exact* (`ops::pcm::PcmInterleave::tolerance`): the
output is an integer the formula fixes, so the check is an equality, not a
tolerance. The formula is

    quantize(x) = round_half_to_even(clamp(x, -1, 1) · 32767)     as i16

and what it can get wrong is the rounding of ties — the reference's
`numpy.rint`/`f32::round_ties_even` takes them to the even neighbour, and a module
that truncated or rounded them away from zero would differ on exactly the half
samples. The samples below therefore include values that land on a tie under the
scale, and both ends of the range and past them.

    python pcm_check.py [--samples 65536] [--block 1024] [--pdl]

It runs the package's Triton source on the GPU through Triton's own launcher,
which is the code the store's module is compiled from. Run it with the Program
Server's environment (`crates/program-server/python/.venv/bin/python`).
"""

import argparse
import pathlib
import sys

import numpy as np
import triton

FULL_SCALE = np.float32(32767.0)


def quantize(x):
    """The reference: `round_half_to_even(clamp(x, -1, 1) · 32767)` in FP32."""
    clamped = np.clip(x, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
    return np.rint(clamped * FULL_SCALE).astype(np.int16)


def samples(count, seed):
    """One channel of decoder-like floats with the cases that catch a wrong rounding."""
    generator = np.random.default_rng(seed)
    channel = generator.normal(0.0, 0.3, count).astype(np.float32)
    # Scaled, these are half-integers: half of them round up and half down, which is
    # what tells ties-to-even from ties-away-from-zero and from truncation.
    halves = np.arange(count // 8, dtype=np.float32) + np.float32(0.5)
    ties = (halves / FULL_SCALE).astype(np.float32)
    channel[: ties.size] = ties
    # Both ends and past them: the clamp, not a wrapping overflow.
    edges = np.array(
        [-2.0, -1.0, -0.9999999, 0.0, 0.9999999, 1.0, 2.0], dtype=np.float32
    )
    channel[-edges.size :] = edges
    return channel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument(
        "--pdl", action="store_true", help="compile for dependent launch"
    )
    arguments = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print(
            "skipped: needs a device — this check runs the kernel itself and has no "
            "interpreter path",
            file=sys.stderr,
        )
        return 0
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    import pcm_interleave as kernels

    device = torch.device("cuda")
    # `GPUTarget.arch` is the compute capability as one number (100 for Blackwell),
    # not a name like "sm100": an int here, and `--pdl` is a Blackwell capability.
    architecture = triton.runtime.driver.active.get_current_target().arch
    pdl = arguments.pdl and int(architecture) == 100
    left = samples(arguments.samples, seed=0)
    right = samples(arguments.samples, seed=1)
    expected = np.stack([quantize(left), quantize(right)], axis=1)

    out = torch.empty(arguments.samples * 2, dtype=torch.int16, device=device)
    kernels._pcm_interleave_kernel[
        (triton.cdiv(arguments.samples, arguments.block),)
    ](
        torch.from_numpy(left).to(device),
        torch.from_numpy(right).to(device),
        out,
        arguments.samples,
        arguments.block,
        1 if pdl else 0,
        BLOCK=arguments.block,
        PDL=pdl,
        num_warps=max(1, min(8, arguments.block // 512)),
        launch_pdl=pdl,
    )
    got = out.view(-1, 2).cpu().numpy()

    print(
        f"{architecture}, {arguments.samples} samples, block {arguments.block}, "
        f"pdl {pdl}"
    )
    wrong = np.argwhere(got != expected)
    print(
        f"max |out - exact| = {int(np.abs(got.astype(np.int32) - expected).max())}"
    )
    if wrong.size:
        row, side = (int(one) for one in wrong[0])
        print(
            f"{wrong.shape[0]} sample(s) differ; the first at ({row}, {side}): "
            f"got {int(got[row, side])}, exact {int(expected[row, side])} "
            f"from {float(left[row] if side == 0 else right[row]):.9g}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
