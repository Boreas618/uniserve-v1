#!/usr/bin/env python3
"""Run frame_moments through the built module and check both statistics.

This is the GPU check a build host cannot do: it loads the module the store holds
(``module.so``), calls the operator's entry with the call structure of ``crates/ops/src/frame.rs``
(608 bytes: ``values`` 0, ``values_batch`` 8, ``values_channel`` 16, ``values_frame`` 24,
``values_pixel`` 32, ``frames`` 40, ``pixels`` 48, ``channels_per_group`` 56, ``groups`` 64,
``epsilon`` 72, ``mean`` 80, ``rstd`` 88, ``error`` 96), and compares both statistics with
PyTorch's own
group-norm moments computed here — ``var_mean(unbiased=False)`` over the same group of channels
and pixels, then ``rsqrt(var + eps)``.

    python moments_check.py <module_dir> [--channels 32] [--groups 4] [--frames 3]
                            [--height 9] [--width 7] [--epsilon 1e-6] [--cpu-only]

``<module_dir>`` is what ``resolve`` hands a runner: the directory with ``module.so``. Run it on
the GPU host with the Program Server's own environment
(``crates/program-server/python/.venv/bin/python``), which has torch. Without a device,
``--cpu-only`` checks the call structure and the entry's refusal on the host.

**The strides are the model's layout.** The check builds `[1, channels, frames, height, width]`
contiguous, which is `[B, C, T, H, W]` as the model holds it: one frame of one channel is `height ·
width` elements apart and a channel of the sequence is `frames · height · width`. A module that
required a frame to hold a whole group — frame-major storage — would refuse this, and that refusal
is a module defect rather than a check defect, because no model produces that layout.

**Both layouts are checked.** The same values are also handed over stored channels-last (the same
logical shape, `values_pixel` the channel count and the pixels of a row `channels` apart), because a
channels-last span feeds the operator that way and a kernel that walked the pixels at a step of one
would reduce the wrong values there while agreeing everywhere else.

**One sample per call.** The call structure carries no sample count and the entry's grid is one
sample's rows, so the operator's caller runs one video sample per call; this check does the same
and says so rather than pretending to check a batch.

**The bound** is the operator's: ``Tolerance::rounding(1 << 16, f32, f32)``, the reduction's own
rounding at the widest frame a launch can carry. The module's documentation claims its Welford
tree is ATen's, i.e. bit for bit PyTorch's, so this check reports how many elements are bit-identical
as well as the deviation — the claim is stronger than the bound and worth measuring.
"""

import argparse
import ctypes
import pathlib
import sys

import torch

#: The largest message an entry writes (``frame.rs``'s ``MESSAGE``).
MESSAGE = 512


class Call(ctypes.Structure):
    """``FrameMomentsCall``, field for field."""

    _fields_ = [
        ("values", ctypes.c_uint64),
        ("values_batch", ctypes.c_int64),
        ("values_channel", ctypes.c_int64),
        ("values_frame", ctypes.c_int64),
        ("values_pixel", ctypes.c_int64),
        ("frames", ctypes.c_int64),
        ("pixels", ctypes.c_int64),
        ("channels_per_group", ctypes.c_int64),
        ("groups", ctypes.c_int64),
        ("epsilon", ctypes.c_float),
        ("mean", ctypes.c_uint64),
        ("rstd", ctypes.c_uint64),
        ("error", ctypes.c_char * MESSAGE),
    ]


def gamma(reduction, unit):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def declared_bound():
    """``Tolerance::rounding(1 << 16, f32, f32)``'s ``rtol``, as ``frame.rs`` declares it."""
    return 2.0 * gamma(1 << 16, 2.0**-24) + 2.0**-24


def check_call_structure():
    """The call structure this script packs is the one ``include/uniserve_frame_moments.h`` asserts."""
    if ctypes.sizeof(Call) != 88 + MESSAGE:
        raise SystemExit(f"Call is {ctypes.sizeof(Call)} bytes, the header asserts {88 + MESSAGE}")
    for name, offset in (
        ("values", 0),
        ("epsilon", 64),
        ("mean", 72),
        ("rstd", 80),
        ("error", 88),
    ):
        if getattr(Call, name).offset != offset:
            raise SystemExit(f"Call.{name} is at {getattr(Call, name).offset}, expected {offset}")
    print(f"call structure: {88 + MESSAGE} bytes, epsilon at 64, error at 88 — as the header asserts")


def load(module_dir):
    """The module's entry, with its C signature."""
    library = ctypes.CDLL(str(module_dir / "module.so"))
    entry = library.uniserve_frame_moments
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    return entry


def check_refusal(module_dir):
    """The entry refuses the two calls it must, on the host, before it touches a device."""
    entry = load(module_dir)
    call = Call(values=0, mean=0x1000, rstd=0x1000, pixels=64, channels_per_group=4, groups=2)
    if entry(ctypes.byref(call), None) == 0:
        raise SystemExit("the entry accepted a call with no values buffer")
    message = call.error.decode(errors="replace")
    if "needs the values" not in message:
        raise SystemExit(f"the entry refused with an unexpected message: {message!r}")
    print(f"entry: loads and refuses a call with no values — {message}")

    # A frame shorter than one row of pixels would read past the tensor, and that is the only
    # layout the entry may refuse: a frame holds one channel's pixels of the *sequence* in the
    # layout H3 feeds ([B, C, T, H, W], where one frame of one channel is exactly `pixels`), so a
    # guard stricter than this one refuses the operator's own calls.
    short = Call(
        values=0x1000,
        mean=0x1000,
        rstd=0x1000,
        pixels=1024,
        channels_per_group=8,
        groups=4,
        frames=2,
        values_frame=64,
        values_pixel=1,
    )
    if entry(ctypes.byref(short), None) == 0:
        raise SystemExit("the entry accepted a frame shorter than one row of pixels")
    message = short.error.decode(errors="replace")
    if "at least one row of pixels" not in message:
        raise SystemExit(f"the entry refused a short frame with an unexpected message: {message!r}")
    print(f"entry: refuses a frame shorter than a row — {message}")


def reference(values, groups, epsilon):
    """PyTorch's group-norm moments per frame: ``var_mean(unbiased=False)`` then ``rsqrt``."""
    batch, channels, frames, height, width = values.shape
    per_group = channels // groups
    folded = (
        values.reshape(batch, groups, per_group, frames, height, width)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(batch * frames, groups, per_group * height * width)
    )
    variance, mean = torch.var_mean(folded, dim=2, unbiased=False)
    return mean, torch.rsqrt(variance + epsilon)


def run(module_dir, channels, groups, frames, height, width, epsilon, channels_last=False):
    """Call the entry and compare both statistics with the reference."""
    entry = load(module_dir)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.manual_seed(3)

    base = torch.randn(1, channels, frames, height, width, dtype=torch.float32, device=device)
    # The same values in the two layouts the operator reads: channels-last stores the channel
    # innermost and hands the operator a view of the logical axes over it.
    values = (
        base.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3) if channels_last else base
    )
    if channels_last and not (values.stride(1) == 1 and values.stride(4) == channels):
        raise SystemExit(f"the channels-last view has strides {values.stride()}")
    mean = torch.zeros(frames, groups, dtype=torch.float32, device=device)
    rstd = torch.zeros(frames, groups, dtype=torch.float32, device=device)
    call = Call(
        values=values.data_ptr(),
        values_batch=values.stride(0),
        values_channel=values.stride(1),
        values_frame=values.stride(2),
        values_pixel=values.stride(4),
        frames=frames,
        pixels=height * width,
        channels_per_group=channels // groups,
        groups=groups,
        epsilon=epsilon,
        mean=mean.data_ptr(),
        rstd=rstd.data_ptr(),
    )
    status = entry(ctypes.byref(call), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if status != 0:
        raise SystemExit(f"the module refused: {call.error.decode(errors='replace')}")

    expected_mean, expected_rstd = reference(values, groups, epsilon)
    bound = declared_bound()
    failures = 0
    for name, candidate, expected in (
        ("mean", mean, expected_mean),
        ("rstd", rstd, expected_rstd),
    ):
        error = (candidate - expected).abs()
        allowed = bound * expected.abs()
        violations = int((~(error <= allowed)).sum())
        identical = int(
            (candidate.reshape(-1).view(torch.int32) == expected.reshape(-1).view(torch.int32)).sum()
        )
        print(
            f"{name}: max |error| {torch.nan_to_num(error).max().item():.3e}, rtol {bound:.3e}, "
            f"{violations} of {error.numel()} outside the bound, "
            f"{identical} of {error.numel()} bit-identical"
        )
        failures += violations
    if failures:
        print(f"FAIL: {failures} statistics outside the operator's declared bound")
        return 1
    print("ok: both statistics within the operator's declared bound")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", nargs="?", type=pathlib.Path, help="the directory with module.so")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--cpu-only", action="store_true", help="check the host-side parts only")
    options = parser.parse_args()

    check_call_structure()
    if options.module is None or not (options.module / "module.so").is_file():
        if options.cpu_only:
            print("cpu-only: the call structure is consistent")
            return 0
        parser.error("a module directory holding module.so is needed")
    check_refusal(options.module)
    if options.cpu_only:
        print("cpu-only: the host-side parts of this script are consistent")
        return 0
    # Both layouts, over the same values: the two runs' bit-identical counts are then comparable,
    # and a kernel that walked the pixels at the wrong step could not agree in both.
    for channels_last in (False, True):
        print(f"layout: {'channels-last' if channels_last else 'channel-first'}")
        status = run(
            options.module,
            options.channels,
            options.groups,
            options.frames,
            options.height,
            options.width,
            options.epsilon,
            channels_last,
        )
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
