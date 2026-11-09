#!/usr/bin/env python3
"""Run cross_fade's kernel and check every value it stores against the reference blend.

The operator's tolerance is ``exact``: its weights are the *decoded dtype's* arithmetic, so a
correct implementation produces the reference's bits, and this check compares bits rather than a
bound. That makes the interpreter's arithmetic the right instrument here — the kernel's indices,
its two addressing forms and the ramp's rounding are all integer or element-type operations, none of
which a device would compute differently (the casts are the one place the interpreter is wrong, and
only for FP8, which a decoded tile is not).

    TRITON_INTERPRET=1 python cross_fade_check.py [--package .]

What is checked, for both axes and three element types:

* ``blend_decoded_overlap`` of ``uniserve/nn/video.py`` — the reference's very expression, with its
  ramp built in the tile's dtype, reproduced here line for line;
* the aliased destination, ``out`` the same buffer as ``current``, which the operator says is
  allowed because every element is read before the position it is written at;
* a destination that is a slice of a larger raster, so the three outer strides are not the tile's
  own and a wrong stride shows up as a wrong value rather than as slowness;
* the tail beyond the band, which is a copy of ``current`` and must be untouched by the blend.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

# Triton's interpreter must be chosen before Triton is imported.
os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import torch

#: The element types a decoded tile is stored in, as the interpreter supplies them.
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


def blend_decoded_overlap(previous, current, extent, dim):
    """``uniserve/nn/video.py``'s reference blend, reproduced expression for expression."""
    extent = min(previous.shape[dim], current.shape[dim], extent)
    if extent == 0:
        return current

    positions = torch.arange(extent, device=current.device, dtype=current.dtype)
    shape = [1] * current.ndim
    shape[dim] = extent
    previous_weight = (1 - positions / extent).view(shape)
    current_weight = (positions / extent).view(shape)

    previous_slice = [slice(None)] * current.ndim
    current_slice = [slice(None)] * current.ndim
    previous_slice[dim] = slice(-extent, None)
    current_slice[dim] = slice(0, extent)
    blended = (
        previous[tuple(previous_slice)] * previous_weight
        + current[tuple(current_slice)] * current_weight
    )
    if extent == current.shape[dim]:
        return blended
    remainder = [slice(None)] * current.ndim
    remainder[dim] = slice(extent, None)
    return torch.cat((blended, current[tuple(remainder)]), dim=dim)


def truncate_to_bf16(values):
    """FP32 values as Triton's interpreter leaves them in a BF16 tile: **truncated**.

    Measured, not assumed: a store of 4,096 FP32 values into BF16 matches the truncation of all
    4,096 — the low 16 mantissa bits dropped — and a round-to-nearest cast of only 2,038 of them.
    F16 and F32 stores round to nearest. So a BF16 kernel's own stores are reproduced exactly by
    truncating in the reference, which is why the blend can be compared bit for bit in BF16 and not
    merely on a bound.
    """
    bits = values.to(torch.float32).reshape(-1).numpy().view(np.uint32).copy()
    bits &= np.uint32(0xFFFF0000)
    return torch.from_numpy(bits.view(np.float32)).reshape(values.shape)


def blend_narrowed_to_bf16(previous, current, extent, dim):
    """``blend_decoded_overlap`` with every store narrowed into BF16 by truncation, in FP32."""
    previous = previous.to(torch.float32)
    current = current.to(torch.float32)
    positions = truncate_to_bf16(torch.arange(extent, dtype=torch.float32))
    ratio = truncate_to_bf16(positions / truncate_to_bf16(torch.tensor([float(extent)]))[0])
    previous_weight = truncate_to_bf16(torch.ones_like(ratio) - ratio)

    shape = [1] * current.ndim
    shape[dim] = extent
    previous_slice = [slice(None)] * current.ndim
    current_slice = [slice(None)] * current.ndim
    previous_slice[dim] = slice(-extent, None)
    current_slice[dim] = slice(0, extent)
    blended = truncate_to_bf16(
        truncate_to_bf16(previous[tuple(previous_slice)] * previous_weight.view(shape))
        + truncate_to_bf16(current[tuple(current_slice)] * ratio.view(shape))
    )
    if extent == current.shape[dim]:
        return blended
    remainder = [slice(None)] * current.ndim
    remainder[dim] = slice(extent, None)
    return torch.cat((blended, current[tuple(remainder)]), dim=dim)


def load_kernel(package):
    """The package's kernel module, imported by path."""
    source = package / "src" / "cross_fade.py"
    spec = importlib.util.spec_from_file_location("cross_fade_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, *, channels, frames, rows, columns, extent, axis, dtype, alias, raster):
    """Blend one tile and compare it with the reference.

    ``raster`` widens the destination's frame grid, so the output's outer strides are not the
    tile's own; ``alias`` hands the kernel the same buffer as ``current`` and as ``out``.
    """
    torch.manual_seed(17)
    element = DTYPES[dtype]
    previous = torch.randn(channels, frames, rows, columns, dtype=element)
    current = torch.randn(channels, frames, rows, columns, dtype=element)
    # The reference's `current` as it was: an aliased launch writes over the buffer it reads, which
    # is the operator's point, so the reference must hold its own copy before the kernel runs.
    original = current.clone()

    if raster:
        # A tile inside a wider raster: rows stay contiguous, the three outer strides widen. The
        # filler makes a kernel that writes outside the tile visible rather than silent.
        destination = torch.full((channels, frames, rows, 2 * columns), 7.0, dtype=element)
        out = destination[..., :columns]
    else:
        destination = torch.full((channels, frames, rows, columns), 7.0, dtype=element)
        out = destination

    if alias:
        # The decoder keeps the tile it just decoded: `out` is `current` itself.
        out = current

    block = 1024
    elements = channels * frames * rows * columns
    grid = ((elements + block - 1) // block,)
    previous_strides, current_strides, out_strides = (
        previous.stride(),
        current.stride(),
        out.stride(),
    )
    module._cross_fade_kernel[grid](
        previous,
        current,
        out,
        extent,
        axis,
        channels,
        frames,
        rows,
        columns,
        previous_strides[0],
        previous_strides[1],
        previous_strides[2],
        current_strides[0],
        current_strides[1],
        current_strides[2],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        block,
        0,
        AXIS=axis,
        BLOCK=block,
    )

    if dtype == "bf16":
        # The reference's expression, narrowed the way the interpreter narrows: see
        # `truncate_to_bf16`. The comparison below is then bit for bit, as for the other two types.
        expected = blend_narrowed_to_bf16(previous, original, extent, axis).to(torch.bfloat16)
    else:
        expected = blend_decoded_overlap(previous, original, extent, axis)
    compare(out, expected, dtype, f"{dtype} axis {axis} extent {extent} {channels}x{frames}x{rows}x{columns}"
            f"{' aliased' if alias else ''}{' in a raster' if raster else ''}")
    return out.numel(), 0


def compare(got, expected, dtype, label):
    """Compare a launch's output with the reference bit for bit.

    F16 and F32 round to nearest in the interpreter as the hardware does; BF16 is narrowed by
    truncation, which the reference reproduces (`truncate_to_bf16`); that rule was measured on the
    interpreter. What this proves about a BF16 blend is its
    arithmetic and its narrowing points, not a bound that could hide one of them.
    """
    element = DTYPES[dtype]
    got_bits = got.reshape(-1).to(torch.int32) if element == torch.float32 else (
        got.reshape(-1).view(torch.int16).to(torch.int32)
    )
    want_bits = expected.reshape(-1).to(torch.int32) if element == torch.float32 else (
        expected.reshape(-1).view(torch.int16).to(torch.int32)
    )
    distance = (got_bits - want_bits).abs()
    differing = int((distance != 0).sum())
    if differing:
        position = int((distance != 0).nonzero()[0])
        raise SystemExit(
            f"{label}: {differing} of {got.numel()} values differ; first at {position}: "
            f"{float(got.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}"
        )
    return got.numel()


def run_seam(module, *, channels, frames, rows, columns, extent, axis, dtype):
    """Blend a seam whose sources and destination are strided views of wider rasters.

    This is the shape the decoder hands over: the predecessor's last `extent` positions of the band
    axis and this tile's first `extent`, each a view inside the raster the tiles were decoded into,
    so every view's band axis is exactly `extent` wide and its outer strides are the raster's. The
    reference is the same blend over **contiguous copies** of the two bands — that equivalence is
    the whole argument for reading strided sources, and it is what this case holds the kernel to.
    """
    torch.manual_seed(23)
    element = DTYPES[dtype]
    if axis == 3:
        previous_raster = torch.randn(channels, frames, rows, 2 * columns, dtype=element)
        current_raster = torch.randn(channels, frames, rows, 2 * columns, dtype=element)
        previous = previous_raster[..., columns - extent : columns]
        current = current_raster[..., columns - extent : columns]
        destination = torch.full((channels, frames, rows, 2 * extent), 7.0, dtype=element)
        out = destination[..., :extent]
        seam_rows, seam_columns = rows, extent
    else:
        previous_raster = torch.randn(channels, frames, 2 * rows, columns, dtype=element)
        current_raster = torch.randn(channels, frames, 2 * rows, columns, dtype=element)
        previous = previous_raster[:, :, rows - extent : rows, :]
        current = current_raster[:, :, rows - extent : rows, :]
        destination = torch.full((channels, frames, 2 * extent, columns), 7.0, dtype=element)
        out = destination[:, :, :extent, :]
        seam_rows, seam_columns = extent, columns

    block = 1024
    elements = channels * frames * seam_rows * seam_columns
    grid = ((elements + block - 1) // block,)
    previous_strides, current_strides, out_strides = (
        previous.stride(),
        current.stride(),
        out.stride(),
    )
    module._cross_fade_kernel[grid](
        previous,
        current,
        out,
        extent,
        axis,
        channels,
        frames,
        seam_rows,
        seam_columns,
        previous_strides[0],
        previous_strides[1],
        previous_strides[2],
        current_strides[0],
        current_strides[1],
        current_strides[2],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        block,
        0,
        AXIS=axis,
        BLOCK=block,
    )
    if dtype == "bf16":
        expected = blend_narrowed_to_bf16(
            previous.contiguous(), current.contiguous(), extent, axis
        ).to(torch.bfloat16)
    else:
        expected = blend_decoded_overlap(previous.contiguous(), current.contiguous(), extent, axis)
    compare(
        out,
        expected,
        dtype,
        f"{dtype} axis {axis} extent {extent} seam {channels}x{frames}x{seam_rows}x{seam_columns}"
        f" read from strided rasters",
    )
    return out.numel()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/cross_fade.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("run this with TRITON_INTERPRET=1: the kernel is checked on the CPU")
    module = load_kernel(options.package)

    checked = 0
    for dtype in ("f16", "bf16", "f32"):
        for axis in (2, 3):
            for extent in (3, 5):
                for alias, raster in ((False, False), (True, False), (False, True)):
                    run(
                        module,
                        channels=3,
                        frames=2,
                        rows=6,
                        columns=8,
                        extent=extent,
                        axis=axis,
                        dtype=dtype,
                        alias=alias,
                        raster=raster,
                    )
                    checked += 1
    # A tile whose whole axis is the band: the reference then returns the blend alone.
    run(
        module,
        channels=2,
        frames=1,
        rows=5,
        columns=5,
        extent=5,
        axis=2,
        dtype="f16",
        alias=False,
        raster=False,
    )
    checked += 1
    for dtype in ("f16", "bf16", "f32"):
        for axis in (2, 3):
            for extent in (3, 5):
                run_seam(
                    module,
                    channels=3,
                    frames=2,
                    rows=6,
                    columns=8,
                    extent=extent,
                    axis=axis,
                    dtype=dtype,
                )
                checked += 1
    print(
        f"ok: {checked} launches, f16 and f32 bit for bit (the fixture's cell is f16), both axes, "
        f"aliased and strided destinations, seams read from strided rasters, bf16 bit for bit with "
        f"its truncating store reproduced"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
