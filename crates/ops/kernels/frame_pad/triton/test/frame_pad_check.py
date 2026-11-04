#!/usr/bin/env python3
"""Run `frame_pad`'s kernel and check the causal padding of a video frame batch, on the CPU.

    crates/program-server/python/.venv/bin/python frame_pad_check.py [--package DIR]

The source is `[batch, channels, frames, height, width]` with its width contiguous; the output is the
same batch and channels over more frames, rows and columns, and equals the two-step `F.pad` result
element for element:

- the `front` output frames **before** the source are zeros — the pad is causal, so nothing may be
  copied into them;
- every other output pixel copies one source pixel, its row and column reflected or replicated.

The domain is PyTorch's: a pad is narrower than the dimension it pads, which is what makes one fold
of `2·dim − 2 − i` reach every padded index. The case with the widest pad a 3-wide axis allows is
here for the off-by-one in that fold.

The operator's tolerance is `exact()`: this is a copy or a zero, so two implementations agree bit for
bit. Every case is therefore asserted with equality, not a bound, and the counts are printed.

The cases, each named for the mistake it catches:

- **the fixture's cell** — `f32`, reflect, 512-pixel and 4-channel blocks, on a shape whose output
  width and channel count both leave a masked tail in the last block;
- **replicate against reflect** on the same shape: the two differ at every padded pixel, and a kernel
  that clamped where it should fold (or folded where it should clamp) cannot agree with both;
- **a pad of `dim - 1`**, the widest reflection a single fold reaches, where an off-by-one in
  `2·dim − 2 − i` shows;
- **zero `front`**, where no frame is causal-padded, and **`front` equal to the whole output**, where
  every copied frame is a zero frame;
- **a channels-last output**, whose strides are the other legal layout, and **a strided source**, whose
  batch and channel strides the kernel must read rather than assume.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

# This is an interpreter check: the arithmetic is compared against a CPU reference, so the kernel runs
# on Triton's interpreter even where a device exists. The flag is set before any kernel module is
# imported, because Triton reads it when it launches.
if os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    print("running on Triton's interpreter (no device needed)")

import torch
import torch.nn.functional as F

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "frame_pad.py"
    spec = importlib.util.spec_from_file_location("frame_pad_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values, *, top, left, front, out_frames, out_height, out_width, reflect):
    """The two-step pad: zero frames at the front, then the spatial pad of every plane."""
    batch, channels, frames, height, width = values.shape
    pads = [torch.zeros(batch, channels, front, height, width, dtype=values.dtype), values]
    padded = torch.cat(pads, dim=2)
    planes = padded.reshape(batch * channels * out_frames, 1, height, width)
    mode = "reflect" if reflect else "replicate"
    planes = F.pad(planes, (left, out_width - width - left, top, out_height - height - top), mode=mode)
    return planes.reshape(batch, channels, out_frames, out_height, out_width)


def check(module, what, *, dtype, reflect, block_pixels, block_channels,
          batch, channels, frames, height, width, top, left, front, channels_last=False,
          strided_source=False, seed=29):
    """One launch, asserted element for element against the two-step pad."""
    element = DTYPES[dtype]
    torch.manual_seed(seed)
    # Distinct in every element, so a pixel copied from the wrong place cannot pass.
    base = torch.randn(batch, channels, frames, height, width, dtype=torch.float32).to(element)
    if strided_source:
        # A view of a wider tensor: its batch, channel **and frame** strides are each twice the
        # packed ones, while the width stays contiguous as the operator requires. A kernel that
        # assumed packed strides reads another sample's pixels here and agrees everywhere else.
        wide = torch.randn(batch, channels * 2, frames * 2, height, width, dtype=torch.float32)
        values = wide.to(element)[:, :channels, ::2]
    else:
        values = base

    out_frames = frames + front
    out_height = height + top + 1
    out_width = width + left + 2

    # The two layouts the operator accepts. Channels-last here is its own definition
    # (`frame.rs`): the channel strides closest to the width, i.e. the shape `(B, F, H, C, W)`
    # viewed as `(B, C, F, H, W)`, which is what a channel-first reader calls channels-last.
    if channels_last:
        out = torch.full((batch, out_frames, out_height, channels, out_width), float("nan"),
                         dtype=element).permute(0, 3, 1, 2, 4)
        if out.stride() != (channels * out_frames * out_height * out_width, out_width,
                            channels * out_height * out_width, channels * out_width, 1):
            raise SystemExit(f"the channels-last view has strides {out.stride()}, not the "
                             f"operator's")
    else:
        out = torch.full((batch, channels, out_frames, out_height, out_width), float("nan"),
                         dtype=element)

    columns = (out_width + block_pixels - 1) // block_pixels
    channel_blocks = (channels + block_channels - 1) // block_channels
    module._frame_pad_kernel[(batch * out_frames * out_height, columns, channel_blocks)](
        values, out,
        *values.stride(), *out.stride(),
        height, width, out_frames, out_height, out_width, channels,
        top, left, front,
        reflect=reflect, block_pixels=block_pixels, block_channels=block_channels,
    )

    expected = reference(values, top=top, left=left, front=front, out_frames=out_frames,
                         out_height=out_height, out_width=out_width, reflect=reflect)
    where = (f"{what}: {dtype} {'reflect' if reflect else 'replicate'}, "
             f"{batch}x{channels}x{frames} of {height}x{width} -> "
             f"{out_frames}x{out_height}x{out_width}, pads top {top} left {left} front {front}"
             f"{', channels-last output' if channels_last else ''}"
             f"{', strided source' if strided_source else ''}")
    differing = (out != expected)
    count = int(differing.sum())
    if count:
        position = int(differing.reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{where}: {count} of {out.numel()} elements differ; first at {position}: "
            f"{float(out.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}"
        )
    zeros = int((expected == 0).sum())
    return out.numel(), zeros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/frame_pad.py",
    )
    options = parser.parse_args()
    module = load_kernel(options.package)

    checked = zeros = 0
    # The fixture's cell: reflect, 512-pixel and 4-channel blocks over a small shape, so both blocks
    # are mostly masked.
    for count, zero in [
        check(module, "the fixture's cell", dtype="f32", reflect=True, block_pixels=512,
              block_channels=4, batch=1, channels=3, frames=2, height=4, width=3,
              top=2, left=2, front=3),
        check(module, "the fixture's cell in bf16", dtype="bf16", reflect=True, block_pixels=512,
              block_channels=4, batch=1, channels=3, frames=2, height=4, width=3,
              top=1, left=1, front=1),
        # Both modes on one shape: the two differ at every padded pixel.
        # Two samples, two channels and two frames, so the batch, channel and frame strides are
        # three different numbers: a kernel that reads one where another belongs reads another
        # sample's pixels, which a single-sample case cannot show.
        check(module, "reflect", dtype="f32", reflect=True, block_pixels=4, block_channels=1,
              batch=2, channels=2, frames=2, height=4, width=5, top=2, left=2, front=1),
        check(module, "replicate", dtype="f32", reflect=False, block_pixels=4, block_channels=1,
              batch=2, channels=2, frames=2, height=4, width=5, top=2, left=2, front=1),
        # The widest single reflection, where an off-by-one in the fold shows.
        check(module, "a three-wide reflection", dtype="f32", reflect=True, block_pixels=2,
              block_channels=2, batch=1, channels=2, frames=1, height=3, width=3,
              top=2, left=2, front=2),
        # No front padding at all, then every copied frame behind a zero front.
        check(module, "no front padding", dtype="f16", reflect=True, block_pixels=8,
              block_channels=8, batch=1, channels=2, frames=3, height=4, width=4,
              top=1, left=1, front=0),
        check(module, "every frame behind the front", dtype="f32", reflect=False,
              block_pixels=8, block_channels=4, batch=1, channels=2, frames=2, height=4,
              width=4, top=1, left=1, front=3),
        # The other legal output layout, and a source whose strides are not the packed ones.
        check(module, "a channels-last output", dtype="f32", reflect=True, block_pixels=4,
              block_channels=2, batch=1, channels=2, frames=1, height=3, width=3,
              top=1, left=2, front=1, channels_last=True),
        check(module, "a strided source", dtype="f32", reflect=True, block_pixels=4,
              block_channels=2, batch=2, channels=2, frames=2, height=3, width=3,
              top=2, left=1, front=1, strided_source=True),
    ]:
        checked += count
        zeros += zero

    print(f"ok: {checked} padded elements over nine cases, bit for bit against the two-step pad, "
          f"{zeros} of them zeros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
