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
    path = f"{package}/src/posterior.py"
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


def run(kernel, positions, channels):
    """Launch the kernel on the fixture, with the moments as one wide tensor cut in two."""
    mean = torch.tensor(MEAN, dtype=torch.float32)
    log_variance = torch.tensor(LOG_VARIANCE, dtype=torch.float32)
    noise = torch.tensor(NOISE, dtype=torch.float32)
    # The encoder's own layout: `[positions, 2 * channels]`, split on the channel axis. The two halves
    # are views with a row stride of two channels, which is what the operator is built for.
    moments = torch.cat([mean, log_variance], dim=1)
    mean_half, variance_half = moments.chunk(2, dim=1)
    out = torch.zeros_like(noise)
    total = positions * channels
    block = 1024
    kernel[((total + block - 1) // block,)](
        mean_half,
        variance_half,
        noise,
        out,
        channels,
        mean_half.stride(0),
        variance_half.stride(0),
        noise.stride(0),
        out.stride(0),
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
    mean, log_variance, noise, out = run(kernel, positions, channels)

    failures = 0
    expected = reference(mean, log_variance, noise)
    for index in range(positions * channels):
        row, column = divmod(index, channels)
        got = out[row, column].item()
        wanted = expected[row, column].item()
        if bits(got) != bits(wanted):
            failures += 1
            print(
                f"FAIL chain element {index}: {got!r} ({bits(got)}) but the reference gives "
                f"{wanted!r} ({bits(wanted)})"
            )
    if not failures:
        print(f"ok   chain: {positions * channels} elements, bit for bit")

    # The affine a conditioner applies, in the reference's own order, and h3's recorded bits.
    latents_mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32)
    latents_std = torch.tensor(LATENTS_STD, dtype=torch.float32)
    normalised = (out - latents_mean) / latents_std
    for index, recorded in enumerate(GROUND_TRUTH):
        row, column = divmod(index, channels)
        got = bits(normalised[row, column].item())
        if got != recorded:
            failures += 1
            print(
                f"FAIL fixture element {index}: the normalised sample is {got}, h3 recorded "
                f"{recorded}"
            )
    if len(GROUND_TRUTH) == positions * channels and all(
        bits((normalised[row, column]).item()) == recorded
        for index, recorded in enumerate(GROUND_TRUTH)
        for row, column in [divmod(index, channels)]
    ):
        print(f"ok   fixture: {len(GROUND_TRUTH)} elements match h3's recorded bits")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
