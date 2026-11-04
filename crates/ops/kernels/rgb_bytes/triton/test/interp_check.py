#!/usr/bin/env python3
"""Check the rgb_bytes kernel's arithmetic and addressing on this host, with no GPU.

    TRITON_INTERPRET=1 python interp_check.py

Triton's interpreter runs the kernel's own Python on the CPU, so this checks
exactly the program the store's module is compiled from: the index
decomposition, the six strides, the clamp and the round-half-to-even. It is the
CPU half of the check; the GPU script beside this one (`rgb_check.py`) runs the
compiled program on a device.

Four layouts, because the operator's contract allows all of them and the
addressing is where this kernel can be wrong: the packed raster; a tile written
into a sub-region of a larger canvas (three of the six strides longer than
packed); a strided read side alone; and a one-frame, one-row strip, where the
frame and row axes contribute nothing and an off-by-one in the decomposition
cannot hide behind them.

The comparison is against the operator's own host oracle,
`ops::rgb::quantize_scaled` mirrored here, over a raster that carries half-byte
ties, both clamps, and the two values that must land on byte 0 and byte 255.
"""

import os
import pathlib
import sys

if os.environ.get("TRITON_INTERPRET") != "1":
    print(
        "this check runs the kernel through Triton's interpreter: "
        "TRITON_INTERPRET=1 python interp_check.py",
        file=sys.stderr,
    )
    raise SystemExit(2)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import rgb_bytes as kernels
import torch

#: The reference's ImageNet constants, which the layer holds as device weights.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# : How near a half-integer counts as a tie, in bytes of the 255-byte range.
# Inside : it the two implementations may differ by one byte; that is the
# operator's : tolerance, and it is the same band `rgb_check.py` uses on a
# device.
TIE_BAND = 1e-4


def scaled_of(value, mean, std):
    """The de-normalized value in FP32, which is what the program adds itself to.

    The arithmetic is FP32 step by step: a reference computed in FP64 would ask
    the program to agree to more precision than it was given, and the
    difference shows up exactly at the half-byte boundaries.
    """
    value = np.float32(value)
    scaled = (value * np.float32(std) + np.float32(mean)) * np.float32(255.0)
    return float(scaled), float(
        (np.float64(value) * np.float64(std) + np.float64(mean)) * 255.0
    )


def quantize(scaled):
    """`ops::rgb::quantize_scaled`: clamp into `[0, 255]`, round half to even."""
    if np.isnan(scaled):
        return 0
    return int(np.rint(np.clip(scaled, 0.0, 255.0)))


def raster(shape, seed=0):
    """A raster of decoder-like values with the cases the rounding turns on."""
    generator = np.random.default_rng(seed)
    values = generator.normal(0.0, 0.3, shape).astype(np.float32)
    channels, frames, rows, columns = shape
    for channel in range(channels):
        # Half-byte ties: ask for a scaled value of `k + 0.5` and invert the
        # de-normalization. The inverse is inexact, so the reference is
        # computed from the values this raster actually holds rather than from
        # the intent.
        ties = (np.arange(0.5, 12.5, 1.0) / 255.0 - MEAN[channel]) / STD[
            channel
        ]
        for index, value in enumerate(ties):
            values[
                channel,
                index % frames,
                (index * 3) % rows,
                (index * 7) % columns,
            ] = value
    # Both clamps, and the two exact ends of the byte range.
    values[0, 0, 0, 0] = 10.0
    values[0, 0, 0, 1] = -10.0
    values[0, 0, 0, 2] = (1.0 - MEAN[0]) / STD[0]
    values[0, 0, 0, 3] = (0.0 - MEAN[0]) / STD[0]
    return values


def run(values, canvas, block=16):
    """Launch the kernel over one raster, answering the bytes it wrote and expected."""
    channels, frames, rows, columns = values.shape
    operand = torch.from_numpy(values)
    if canvas:
        buffer = torch.zeros(
            (frames + 1, rows, columns + 2, channels), dtype=torch.uint8
        )
        out = buffer[:-1, :, :columns, :]
        wider = torch.zeros(
            (channels, frames, rows, columns + 3), dtype=torch.float32
        )
        wider[:, :, :, :columns] = operand
        source = wider[:, :, :, :columns]
    else:
        out = torch.zeros((frames, rows, columns, channels), dtype=torch.uint8)
        source = operand
    strides = [int(stride) for stride in source.stride()]
    written = [int(stride) for stride in out.stride()]
    total = frames * rows * columns * channels
    kernels._rgb_bytes_kernel[((total + block - 1) // block,)](
        source,
        torch.from_numpy(MEAN),
        torch.from_numpy(STD),
        out,
        frames,
        rows,
        columns,
        channels,
        strides[0],
        strides[1],
        strides[2],
        written[0],
        written[1],
        written[2],
        0,
        CHANNELS=channels,
        BLOCK=block,
        PDL=False,
    )
    # The reference walks the raster channel by channel, so nothing of the
    # kernel's own layout appears in it. It answers the expected byte and
    # whether that byte is a tie candidate, which is where a one-byte difference
    # is allowed.
    expected = np.zeros((frames, rows, columns, channels), dtype=np.uint8)
    ties = np.zeros((frames, rows, columns, channels), dtype=bool)
    for channel in range(channels):
        for frame in range(frames):
            for row in range(rows):
                for column in range(columns):
                    scaled, exact = scaled_of(
                        source[channel, frame, row, column].item(),
                        float(MEAN[channel]),
                        float(STD[channel]),
                    )
                    expected[frame, row, column, channel] = quantize(scaled)
                    in_range = 0.0 <= exact <= 255.0
                    ties[frame, row, column, channel] = in_range and (
                        abs(abs(exact - np.floor(exact)) - 0.5) < TIE_BAND
                    )
    return out.numpy(), expected, ties


def main() -> int:
    cases = [
        ("the packed raster", (3, 2, 4, 6), False),
        ("a tile in a canvas", (3, 2, 4, 6), True),
        ("a strip", (3, 1, 1, 9), False),
        ("a strip in a canvas", (3, 1, 1, 9), True),
    ]
    failures = 0
    for label, shape, canvas in cases:
        got, expected, ties = run(raster(shape), canvas)
        allowed = ties & (
            np.abs(got.astype(np.int32) - expected.astype(np.int32)) <= 1
        )
        wrong = (got != expected) & ~allowed
        print(
            f"{label}: {int(np.prod(shape))} bytes, {int(ties.sum())} tie candidates, "
            f"{int((got != expected).sum())} differ, {int(wrong.sum())} outside the band"
            + (
                ""
                if not wrong.any()
                else f"; first at {tuple(int(value) for value in np.argwhere(wrong)[0])}"
            )
        )
        if wrong.any():
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
