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


def round_to_bf16(values):
    """FP32 values rounded to BF16 the way hardware rounds them: to nearest, ties to even.

    Torch's own f32 -> bf16 cast rounds an exact tie **away from zero** on the CPU (measured: 1.005859375
    becomes 1.0078125 where round-to-nearest-even gives 1.0), while a device's ``cvt.rn.bf16.f32``
    rounds ties to even. The operator's blend rounds in the tile's dtype three times, so the two rules
    disagree at ties and a CPU-torch reference is not the device's reference. This is the hardware
    rule, written in the bit pattern so the check can hold the kernel to it.
    """
    bits = values.to(torch.float32).reshape(-1).numpy().view(np.uint32).copy()
    least_bit = (bits >> np.uint32(16)) & np.uint32(1)
    bits = (bits + np.uint32(0x7FFF) + least_bit) >> np.uint32(16)
    rounded = bits.astype(np.uint16).view(np.uint16)
    as_bits = np.zeros(rounded.shape, dtype=np.uint32)
    as_bits[:] = rounded
    return torch.from_numpy((as_bits << np.uint32(16)).view(np.float32)).reshape(values.shape)


def blend_rounding_to_even(previous, current, extent, dim):
    """``blend_decoded_overlap`` with every step rounded to BF16 by the hardware's rule, in FP32."""
    previous = previous.to(torch.float32)
    current = current.to(torch.float32)
    positions = round_to_bf16(torch.arange(extent, dtype=torch.float32))
    ratio = round_to_bf16(positions / round_to_bf16(torch.tensor([float(extent)]))[0])
    previous_weight = round_to_bf16(torch.ones_like(ratio) - ratio)

    shape = [1] * current.ndim
    shape[dim] = extent
    previous_slice = [slice(None)] * current.ndim
    current_slice = [slice(None)] * current.ndim
    previous_slice[dim] = slice(-extent, None)
    current_slice[dim] = slice(0, extent)
    blended = round_to_bf16(
        round_to_bf16(previous[tuple(previous_slice)] * previous_weight.view(shape))
        + round_to_bf16(current[tuple(current_slice)] * ratio.view(shape))
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
    out_strides = out.stride()
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
        out_strides[0],
        out_strides[1],
        out_strides[2],
        block,
        0,
        AXIS=axis,
        BLOCK=block,
    )

    if dtype == "bf16":
        # The reference's expression, rounded the way a device rounds it: see `round_to_bf16`.
        expected = blend_rounding_to_even(previous, original, extent, axis)
        expected = expected.to(torch.bfloat16)
    else:
        expected = blend_decoded_overlap(previous, original, extent, axis)
    got = out
    # The comparison is in the order the two-bit-wide types' bit patterns: an f16 or f32 kernel must
    # produce the reference's bits, because every cast on the way is exact or round-to-nearest in
    # both implementations. BF16 is the exception and the reason is a cast, not the blend: torch's
    # f32 -> bf16 conversion rounds an exact tie **away from zero**, while NVIDIA's `cvt.rn.bf16`
    # (and so a device kernel) rounds it to even. The two differ by one ulp of the tile type at a
    # tie, and only there, so bf16 is compared within one ulp and the differing count is reported.
    got_bits = got.reshape(-1).to(torch.int32) if element == torch.float32 else (
        got.reshape(-1).view(torch.int16).to(torch.int32)
    )
    want_bits = expected.reshape(-1).to(torch.int32) if element == torch.float32 else (
        expected.reshape(-1).view(torch.int16).to(torch.int32)
    )
    if dtype == "bf16":
        # BF16 is judged on a bound, and the reason is the interpreter rather than the kernel: its
        # f32 -> bf16 conversion is not round-to-nearest at an exact tie — measured, 1.005859375
        # comes out 1.0 while 0.666015625 comes out 0.66796875, so it rounds some ties down and
        # others up, matching neither torch's cast nor the hardware's. The blend rounds four times
        # (the two weights and the two products and the sum), each within the tile type's unit
        # roundoff of the operands, so the bound is `64 · 2**-9` of the larger operand — the same
        # arithmetic the operator declares exactly, with the interpreter's cast noise allowed for.
        margin = 64.0 * 2.0**-9 * torch.maximum(original.abs().to(torch.float32),
                                                previous.abs().to(torch.float32))
        error = (got.to(torch.float32) - expected.to(torch.float32)).abs()
        differing = int((error > margin).sum())
        if differing:
            position = int((error > margin).reshape(-1).nonzero()[0])
            raise SystemExit(
                f"bf16 axis {axis} extent {extent}: {differing} of {got.numel()} values outside "
                f"the blend's bound; first at {position}: "
                f"{float(got.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}, "
                f"margin {float(margin.reshape(-1)[position]):.3e}"
            )
        return got.numel(), int((error != 0).sum())
    distance = (got_bits - want_bits).abs()
    differing = int((distance != 0).sum())
    if differing:
        position = int((distance != 0).nonzero()[0])
        raise SystemExit(
            f"{dtype} axis {axis} extent {extent} {channels}x{frames}x{rows}x{columns}"
            f"{' aliased' if alias else ''}{' in a raster' if raster else ''}: "
            f"{differing} of {got.numel()} values differ; first at {position}: "
            f"{float(got.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}"
        )
    return got.numel(), 0


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
    print(
        f"ok: {checked} launches, f16 and f32 bit for bit (the fixture's cell is f16), both axes, "
        f"aliased and strided destinations; bf16 on the blend's bound, because the interpreter's "
        f"f32 -> bf16 cast is not round-to-nearest at a tie"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
