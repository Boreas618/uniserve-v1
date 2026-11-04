#!/usr/bin/env python3
"""Run posterior's kernel under Triton's interpreter and compare it with the reference's own bits.

Two comparisons, because the operator has two claims to defend:

1. **The chain.** The kernel's output is compared with the reference's expression evaluated in
   torch, bit for bit. The tolerance is exact -- the arithmetic *is* the specification -- so a
   difference of one ulp is a failure and the message says which element.

2. **The fixture.** h3 recorded ground truth for four elements of two channels, after the affine a
   conditioner applies (`notes/h3/posterior-epilogue.md`): the *normalised* latents, as IEEE-754
   bits. This script applies that affine in torch to the kernel's output and compares with the
   recorded bits, which is what pins the two steps a fused kernel drops first: `0.25000014901161194`
   must become exactly `0.25` through the round trip, and `-44054.4296875` must become `-44064.0`.

The moments arrive as one `[positions, 2 · channels]` tensor cut on the channel axis -- the shape h3's
`VideoEncoder::posterior` passes -- so the kernel is exercised with a row stride twice the sample's,
which is the one thing about its addressing that could silently be wrong.

    TRITON_INTERPRET=1 python posterior_check.py [--package .]
"""

import argparse
import importlib.util
import pathlib
import struct

import torch

#: The reference's two per-channel tables (`latents_mean`, `latents_std`), channel-major.
LATENTS_MEAN = [0.858090341091156, -0.9606591463088989]
LATENTS_STD = [1.2223774194717407, 1.2767263650894165]

#: h3's fixture: two positions of two channels, and the normalised sample as recorded bits.
MEAN = [[0.25, -1.5], [3.0, 0.125]]
LOG_VARIANCE = [[-40.0, 25.0], [-0.5, 0.0]]
NOISE = [[0.5, -2.0], [1.0, 0.75]]
GROUND_TRUTH = ["befeb3c5", "c706d084", "4018f211", "3fb8095e"]


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    del package  # the check's own location is the package's, wherever it is run from
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "posterior.py"
    spec = importlib.util.spec_from_file_location("posterior_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._posterior_kernel


def reference(mean, log_variance, noise):
    """The reference's chain in torch, in its own order, including the FP16 round trip."""
    log_variance = log_variance.clamp(-30.0, 20.0)
    sample = mean + torch.exp(0.5 * log_variance) * noise
    return sample.to(torch.float16).to(torch.float32)


def bits(value):
    """The IEEE-754 single bits of a float, as eight hex digits, most significant first.

    Big-endian because h3 recorded the fixture's bits that way: `0.25`'s pattern reads `3e800000`, not
    `0000803e`, and a record of a bit pattern is read by people before it is read by a machine.
    """
    return struct.pack(">f", float(value)).hex()


def run(kernel, positions, channels, channels_last):
    """Launch the kernel on the fixture in one of the two physical layouts.

    ``channels_last`` describes the caller, not the kernel: the fixture's four elements are addressed
    through (row stride, channel stride) pairs, and the kernel takes whichever pair it is given. A
    channel-major caller -- the VAE's own rasters -- passes ``(1, positions)``; a channels-last one
    passes ``(channels, 1)``. Both launches must produce the same bits.
    """
    mean = torch.tensor(MEAN, dtype=torch.float32)
    log_variance = torch.tensor(LOG_VARIANCE, dtype=torch.float32)
    noise = torch.tensor(NOISE, dtype=torch.float32)
    if channels_last:
        # One row per position, the channel innermost -- the moments joined on that axis, so the two
        # halves are views with a row stride of twice the channels.
        moments = torch.cat([mean, log_variance], dim=1)
        mean_half, variance_half = moments.chunk(2, dim=1)
        layout = [
            (mean_half, mean_half.stride(0), mean_half.stride(1)),
            (variance_half, variance_half.stride(0), variance_half.stride(1)),
            (noise, noise.stride(0), noise.stride(1)),
        ]
        out = torch.zeros_like(noise)
    else:
        # One column per position, the position innermost -- the encoder's NCTHW raster, with the
        # moments as one 48-channel buffer whose two halves differ by an offset of 24 positions.
        moments = torch.cat([mean, log_variance], dim=0)
        mean_half, variance_half = moments.chunk(2, dim=0)
        layout = [
            (mean_half, mean_half.stride(0), mean_half.stride(1)),
            (variance_half, variance_half.stride(0), variance_half.stride(1)),
            (noise.t().contiguous().t(), 1, positions),
        ]
        out = torch.zeros_like(noise)
    total = positions * channels
    block = 1024
    kernel[((total + block - 1) // block,)](
        layout[0][0],
        layout[1][0],
        layout[2][0],
        out,
        channels,
        layout[0][1],
        layout[0][2],
        layout[1][1],
        layout[1][2],
        layout[2][1],
        layout[2][2],
        out.stride(0),
        out.stride(1),
        total,
        0,
        BLOCK=block,
        PDL=False,
    )
    return mean, log_variance, noise, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=".")
    arguments = parser.parse_args()
    kernel = load_kernel(arguments.package)
    positions, channels = len(MEAN), len(MEAN[0])
    failures = 0
    latents_mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32)
    latents_std = torch.tensor(LATENTS_STD, dtype=torch.float32)
    for name, channels_last in [("channels-last", True), ("channel-major", False)]:
        mean, log_variance, noise, out = run(kernel, positions, channels, channels_last)
        expected = reference(mean, log_variance, noise)
        for index in range(positions * channels):
            row, column = divmod(index, channels)
            got = out[row, column].item()
            wanted = expected[row, column].item()
            if bits(got) != bits(wanted):
                failures += 1
                print(
                    f"FAIL {name} chain element {index}: {got!r} ({bits(got)}) but the reference "
                    f"gives {wanted!r} ({bits(wanted)})"
                )
        else:
            print(f"ok   {name} chain: {positions * channels} elements, bit for bit")

        # The affine a conditioner applies, in the reference's own order, and h3's recorded bits.
        normalised = (out - latents_mean) / latents_std
        matched = 0
        for index, recorded in enumerate(GROUND_TRUTH):
            row, column = divmod(index, channels)
            got = bits(normalised[row, column].item())
            if got != recorded:
                failures += 1
                print(
                    f"FAIL {name} fixture element {index}: the normalised sample is {got}, h3 "
                    f"recorded {recorded}"
                )
            else:
                matched += 1
        if matched == len(GROUND_TRUTH) == positions * channels:
            print(f"ok   {name} fixture: {len(GROUND_TRUTH)} elements match h3's recorded bits")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
