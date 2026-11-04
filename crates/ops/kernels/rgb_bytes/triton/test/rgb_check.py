#!/usr/bin/env python3
"""Run one rgb_bytes module on a GPU and check the bytes it writes.

    python rgb_check.py
    python rgb_check.py --dtype bf16 --pdl

The package's Triton source is launched through Triton's own launcher, which is the
code the store's module is compiled from. Everything geometric is a launch value —
the raster, the block, the dependent launch — so one run covers a geometry the
descriptor does not name.

**The comparison is in bytes, and exact except at the ties.** The operator's
arithmetic (`(x · std + mean) · 255`, clamp, round half to even) is a real value
rounded once, so two implementations can differ by one byte only where the real value
sits within floating-point error of a half-byte boundary. The samples contain many
such values on purpose and the check separates them: an element farther than a part
in ten thousand from a half-integer must match **exactly**, and one inside that band
may differ by **one byte**, the operator's own `Tolerance::new(1.0, 0.0)`. The band
and its count are printed, so a failure says which side it came from.

Two layouts are checked, because the operator's contract allows both: the packed
raster, and a tile written into a sub-region of a larger, strided canvas with a
strided read side, which is what the decoder does with each decoded tile.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

#: The reference's ImageNet constants, which the layer holds as device weights.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: How near a half-integer counts as a tie, in bytes of the 255-byte range.
TIE_BAND = 1e-4

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


def samples(shape, seed):
    """A raster of decoder-like values in FP32, with half-byte ties placed on purpose."""
    generator = np.random.default_rng(seed)
    raster = generator.normal(0.0, 0.3, shape).astype(np.float32)
    channels, frames, rows, columns = shape
    # Ties: ask for a scaled value of `k + 0.5` and invert the de-normalization. The
    # inverse is inexact, which is why a band exists rather than an equality.
    for channel in range(channels):
        values = (np.arange(0.5, 12.5, 1.0) / 255.0 - MEAN[channel]) / STD[
            channel
        ]
        for index, value in enumerate(values):
            raster[
                channel,
                index % frames,
                (index * 3) % rows,
                (index * 7) % columns,
            ] = value
    # Both ends and past them: the clamp, not a wrapping cast. A saturated zero and a
    # saturated 255 are the two bytes a de-normalization must reach exactly.
    raster[0, 0, 0, 0] = 10.0
    raster[0, 0, 0, 1] = -10.0
    raster[0, 0, 0, 2] = (1.0 - MEAN[0]) / STD[0]
    raster[0, 0, 0, 3] = (0.0 - MEAN[0]) / STD[0]
    return raster


def reference(raster, mean, std):
    """The bytes the operator's arithmetic gives, and which of them are tie candidates.

    The arithmetic is FP64 from the input's own (already rounded) values, which is
    what "the real value" means for a comparison whose output is an integer.
    """
    scaled = (
        raster.astype(np.float64) * std.astype(np.float64)[:, None, None, None]
        + mean.astype(np.float64)[:, None, None, None]
    ) * 255.0
    byte = np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.int32)
    near_half = np.abs(np.abs(scaled - np.floor(scaled)) - 0.5) < TIE_BAND
    return byte, near_half


def launch(dtype, block, pdl, raster, canvas):
    """Launch the package's kernel over one raster; answer the bytes it wrote."""
    kernel_source = pathlib.Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(kernel_source))
    import rgb_bytes as kernels

    device = torch.device("cuda")
    channels, frames, rows, columns = raster.shape
    operand = torch.from_numpy(raster).to(
        device=device, dtype=TORCH_TYPES[dtype]
    )
    # What the program actually reads: the operand after its own type's rounding,
    # which is what the reference has to be compared against.
    rounded = operand.to(torch.float32).cpu().numpy()
    mean = torch.from_numpy(MEAN).to(device)
    std = torch.from_numpy(STD).to(device)

    if canvas:
        # The write goes into a sub-region of a larger canvas and the read side is a
        # slice of a wider raster, so three of the six strides are not the packed ones.
        buffer = torch.zeros(
            (frames + 1, rows, columns + 2, channels),
            dtype=torch.uint8,
            device=device,
        )
        out = buffer[:-1, :, :columns, :]
        wider = torch.zeros(
            (channels, frames, rows, columns + 3),
            dtype=operand.dtype,
            device=device,
        )
        wider[:, :, :, :columns] = operand
        source = wider[:, :, :, :columns]
    else:
        buffer = None
        out = torch.empty(
            (frames, rows, columns, channels), dtype=torch.uint8, device=device
        )
        source = operand

    strides = [int(stride) for stride in source.stride()]
    written = [int(stride) for stride in out.stride()]
    total = frames * rows * columns * channels
    kernels._rgb_bytes_kernel[((total + block - 1) // block,)](
        source,
        mean,
        std,
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
        1 if pdl else 0,
        CHANNELS=channels,
        BLOCK=block,
        PDL=pdl,
        num_warps=max(1, min(8, block // 512)),
        launch_pdl=pdl,
    )
    torch.cuda.synchronize()
    return rounded, out.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="f16")
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--pdl", action="store_true")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        print("no GPU on this host: this check needs one", file=sys.stderr)
        return 1

    failures = 0
    for canvas in (False, True):
        raster = samples((3, 2, 4, 6), seed=0)
        rounded, got = launch(
            arguments.dtype, arguments.block, arguments.pdl, raster, canvas
        )
        expected, near_half = reference(rounded, MEAN, STD)
        difference = np.abs(got.astype(np.int32) - expected)
        allowed = near_half & (difference <= 1)
        wrong = (difference > 0) & ~allowed
        label = "a tile in a canvas" if canvas else "the packed raster"
        print(
            f"{arguments.dtype}, block {arguments.block}, pdl {arguments.pdl}, {label}: "
            f"{raster.size} bytes, {int(near_half.sum())} tie candidates, "
            f"{int((difference > 0).sum())} bytes differ, {int(wrong.sum())} outside the band"
        )
        if wrong.any():
            failures += 1
            channel, frame, row, column = (
                int(value) for value in np.argwhere(wrong)[0]
            )
            print(
                f"  at [channel {channel}, frame {frame}, row {row}, column {column}]: "
                f"got {int(got[frame, row, column, channel])}, "
                f"exact {int(expected[channel, frame, row, column])} from "
                f"{float(rounded[channel, frame, row, column]):.9g}",
                file=sys.stderr,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
