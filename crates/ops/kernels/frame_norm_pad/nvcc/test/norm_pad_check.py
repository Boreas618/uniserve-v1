#!/usr/bin/env python3
"""Run frame_norm_pad through the built module and check the whole composition.

This is the GPU check a build host cannot do: it loads the module the store holds
(``module.so``), calls the operator's entry with the call structure of ``crates/ops/src/frame.rs``
(728 bytes: ``values`` 0, ``values_batch`` 8, ``values_channel`` 16, ``values_frame`` 24,
``mean`` 32, ``rstd`` 40, ``weight`` 48, ``bias`` 56, ``out`` 64, the five output strides 72..104,
``channels`` 112, ``frames`` 120, ``height`` 128, ``width`` 136, ``out_frames`` 144,
``out_rows`` 152, ``out_columns`` 160, ``channels_per_group`` 168, ``groups`` 176, ``top`` 184,
``left`` 192, ``front`` 200, ``reflect`` 208, ``normalization`` 212, ``error`` 216), and
compares the output with the eager composition the operator replaces:

    out = pad(silu(group_norm(values)))      # ``F.group_norm``, ``F.silu``, ``F.pad``

    python norm_pad_check.py <module_dir> [--channels 32] [--groups 4] [--frames 3]
                             [--height 9] [--width 7] [--top 4] [--left 3]
                             [--front 2] [--reflect] [--centered] [--channels-last]
                             [--single-pixel] [--epsilon 1e-6] [--cpu-only]

``<module_dir>`` is what ``resolve`` hands a runner: the directory with ``module.so``. Run it on the
GPU host with the Program Server's own environment
(``crates/program-server/python/.venv/bin/python``), which has torch. Without a device,
``--cpu-only`` checks the call structure and the entry's refusal on the host.

**What is checked, and what is not.** The moments the call carries are computed here by torch and
passed in, so this check tests the normalization, the SiLU and the padded store — the parts of the
entry that are not ``frame_moments``, whose own check proves the moments bit for bit. The three
normalization branches are the descriptor's: ``--centered`` normalizes without affine parameters
(``normalization = 1``), its absence normalizes with them (``0``), and ``--single-pixel`` takes
``GroupNorm1dForward``'s expressions (``2``) for frames of one pixel.

**The layout is the call's, not the descriptor's.** The entry's two template branches write the
same addresses, ``channel · out_channel + column · out_column``; the flag only decides which loop
coalesces. ``--channels-last`` passes an output whose channel stride is 1, so a wrong stride
handling shows up as a wrong result rather than as slowness.

**The bound** is the operator's: ``Tolerance::rounding(1 << 16, f32, f32)``. The module claims to
compose PyTorch's own kernels' arithmetic, so this check reports the bit-identical count beside the
deviation — the claim is stronger than the bound and worth measuring.

**One sample per call**, because the call structure carries no sample count; the entry's grid is one
sample's rows. A call whose ``out_frames`` exceeds ``front + frames`` would read past the source,
so this check requires them equal and says which.
"""

import argparse
import ctypes
import pathlib
import sys

import torch
import torch.nn.functional as F

#: The largest message an entry writes (``frame.rs``'s ``MESSAGE``).
MESSAGE = 512
#: The normalization branches ``frame.cu`` implements.
FUSED, CENTERED, SINGLE_PIXEL = 0, 1, 2


class Call(ctypes.Structure):
    """``FrameNormPadCall``, field for field."""

    _fields_ = [
        ("values", ctypes.c_uint64),
        ("values_batch", ctypes.c_int64),
        ("values_channel", ctypes.c_int64),
        ("values_frame", ctypes.c_int64),
        ("mean", ctypes.c_uint64),
        ("rstd", ctypes.c_uint64),
        ("weight", ctypes.c_uint64),
        ("bias", ctypes.c_uint64),
        ("out", ctypes.c_uint64),
        ("out_batch", ctypes.c_int64),
        ("out_channel", ctypes.c_int64),
        ("out_frame", ctypes.c_int64),
        ("out_height", ctypes.c_int64),
        ("out_width", ctypes.c_int64),
        ("channels", ctypes.c_int64),
        ("frames", ctypes.c_int64),
        ("height", ctypes.c_int64),
        ("width", ctypes.c_int64),
        ("out_frames", ctypes.c_int64),
        ("out_rows", ctypes.c_int64),
        ("out_columns", ctypes.c_int64),
        ("channels_per_group", ctypes.c_int64),
        ("groups", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("left", ctypes.c_int64),
        ("front", ctypes.c_int64),
        ("reflect", ctypes.c_int32),
        ("normalization", ctypes.c_int32),
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
    """The call structure this script packs is the one ``include/uniserve_frame_norm_pad.h`` asserts."""
    if ctypes.sizeof(Call) != 216 + MESSAGE:
        raise SystemExit(f"Call is {ctypes.sizeof(Call)} bytes, the header asserts {216 + MESSAGE}")
    for name, offset in (
        ("values", 0),
        ("weight", 48),
        ("out", 64),
        ("channels", 112),
        ("reflect", 208),
        ("normalization", 212),
        ("error", 216),
    ):
        field = getattr(Call, name)
        if field.offset != offset:
            raise SystemExit(f"Call.{name} is at {field.offset}, expected {offset}")
    print(f"call structure: {216 + MESSAGE} bytes, reflect at 208, error at 216 — as the header asserts")


def load(module_dir):
    """The module's entry, with its C signature."""
    library = ctypes.CDLL(str(module_dir / "module.so"))
    entry = library.uniserve_frame_norm_pad
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    return entry


def check_refusal(module_dir):
    """The entry refuses a call with no output on the host, before it touches a device."""
    call = Call(values=0x1000, mean=0x1000, rstd=0x1000, out=0, channels=8, height=4, width=4)
    if load(module_dir)(ctypes.byref(call), None) == 0:
        raise SystemExit("the entry accepted a call with no output buffer")
    message = call.error.decode(errors="replace")
    if "needs the values, both statistics and the output" not in message:
        raise SystemExit(f"the entry refused with an unexpected message: {message!r}")
    print(f"entry: loads and refuses a call with no output — {message}")


def moments(values, groups, epsilon):
    """The group statistics the call carries, from torch: ``var_mean(unbiased=False)``, ``rsqrt``."""
    _, channels, frames, height, width = values.shape
    per_group = channels // groups
    folded = (
        values.reshape(1, groups, per_group, frames, height, width)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(frames, groups, per_group * height * width)
    )
    variance, mean = torch.var_mean(folded, dim=2, unbiased=False)
    return mean, torch.rsqrt(variance + epsilon)


def reference(values, mean, rstd, weight, bias, groups, branch, top, left, front, out_shape, mode):
    """``pad(silu(group_norm(values)))`` as the eager composition, in torch."""
    _, channels, frames, height, width = values.shape
    per_group = channels // groups
    shape = values.shape

    normalized = torch.empty_like(values)
    for group in range(groups):
        first = group * per_group
        slice_ = slice(first, first + per_group)
        group_mean = mean[:, group].reshape(1, 1, frames, 1, 1)
        group_rstd = rstd[:, group].reshape(1, 1, frames, 1, 1)
        if branch == FUSED:
            scale = group_rstd * weight[slice_].reshape(1, -1, 1, 1, 1)
            shift = -scale * group_mean + bias[slice_].reshape(1, -1, 1, 1, 1)
            normalized[:, slice_] = scale * values[:, slice_] + shift
        else:
            normalized[:, slice_] = (values[:, slice_] - group_mean) * group_rstd
    activated = F.silu(normalized)

    out_frames, out_rows, out_columns = out_shape
    padded = F.pad(
        activated,
        (
            left,
            out_columns - left - width,
            top,
            out_rows - top - height,
            0,
            0,
        ),
        mode=mode,
    )
    out = torch.zeros(1, channels, out_frames, out_rows, out_columns, dtype=values.dtype, device=values.device)
    out[:, :, front : front + frames] = padded
    return out


def source_index(index, extent, reflect):
    """The source position a padded index reads: ``frame.cu``'s mirror or clamp."""
    if reflect:
        index = -index if index < 0 else index
        return 2 * extent - 2 - index if index >= extent else index
    return 0 if index < 0 else (extent - 1 if index >= extent else index)


def check_reference_by_hand():
    """The reference agrees with an element-by-element composition written here.

    Torch's ``F.pad``, ``F.silu`` and the group scaling are convenient but opaque; the operator's
    mathematical definition is not. This walks the output element by element, maps its padded
    position to a source pixel with the same mirror-or-clamp rule the kernel uses, and compares.
    Both flavours of the pad are checked, because the rule is where a pad goes wrong.
    """
    device = torch.device("cpu")
    torch.manual_seed(13)
    channels, groups, frames, height, width = 4, 2, 2, 3, 4
    top, left, bottom, right, front = 1, 2, 1, 1, 1
    epsilon = 1e-6
    values = torch.randn(1, channels, frames, height, width, dtype=torch.float32, device=device)
    mean, rstd = moments(values, groups, epsilon)
    weight = torch.randn(channels, dtype=torch.float32, device=device)
    bias = torch.randn(channels, dtype=torch.float32, device=device)

    for reflect in (False, True):
        mode = "reflect" if reflect else "replicate"
        expected = reference(
            values,
            mean,
            rstd,
            weight,
            bias,
            groups,
            FUSED,
            top,
            left,
            front,
            (front + frames, height + top + bottom, width + left + right),
            mode,
        )
        per_group = channels // groups
        for channel in range(channels):
            for out_frame in range(front + frames):
                for out_row in range(height + top + bottom):
                    for out_column in range(width + left + right):
                        if out_frame < front:
                            want = 0.0
                        else:
                            frame = out_frame - front
                            row = source_index(out_row - top, height, reflect)
                            column = source_index(out_column - left, width, reflect)
                            group = channel // per_group
                            scale = float(rstd[frame, group]) * float(weight[channel])
                            shift = -scale * float(mean[frame, group]) + float(bias[channel])
                            value = scale * float(values[0, channel, frame, row, column]) + shift
                            want = value / (1.0 + 2.718281828459045 ** -value)
                        got = float(expected[0, channel, out_frame, out_row, out_column])
                        if abs(want - got) > 1e-5 * max(1.0, abs(want)):
                            raise SystemExit(
                                f"reference disagrees with the definition at "
                                f"[{channel}, {out_frame}, {out_row}, {out_column}] "
                                f"({mode}): {got} vs {want}"
                            )
    # The leading frames are zeros, not a padding of the first frame.
    if float(expected[0, :, 0].abs().max()) != 0.0:
        raise SystemExit("the leading zero frames are not zeros")
    print("reference: matches a hand-written pad, normalize and SiLU, both pad modes")


def run(options):
    """Call the entry and compare the output with the eager composition."""
    entry = load(options.module)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.manual_seed(5)

    channels, groups, frames = options.channels, options.groups, options.frames
    height, width = options.height, options.width
    per_group = channels // groups
    out_rows = height + options.top + options.bottom
    out_columns = width + options.left + options.right
    out_frames = options.front + frames
    if options.reflect:
        if options.top + options.bottom >= height or options.left + options.right >= width:
            raise SystemExit("a reflect padding must be smaller than the extent it mirrors")
        mode = "reflect"
    else:
        mode = "replicate"

    values = torch.randn(1, channels, frames, height, width, dtype=torch.float32, device=device)
    mean, rstd = moments(values, groups, options.epsilon)
    mean = mean.contiguous()
    rstd = rstd.contiguous()

    branch = SINGLE_PIXEL if options.single_pixel else (CENTERED if options.centered else FUSED)
    weight = bias = None
    if branch == FUSED:
        weight = torch.randn(channels, dtype=torch.float32, device=device)
        bias = torch.randn(channels, dtype=torch.float32, device=device)

    out = torch.zeros(
        1, channels, out_frames, out_rows, out_columns, dtype=torch.float32, device=device
    )
    if options.channels_last:
        out = out.to(memory_format=torch.channels_last_3d)

    call = Call(
        values=values.data_ptr(),
        values_batch=values.stride(0),
        values_channel=values.stride(1),
        values_frame=values.stride(2),
        mean=mean.data_ptr(),
        rstd=rstd.data_ptr(),
        weight=weight.data_ptr() if weight is not None else 0,
        bias=bias.data_ptr() if bias is not None else 0,
        out=out.data_ptr(),
        out_batch=out.stride(0),
        out_channel=out.stride(1),
        out_frame=out.stride(2),
        out_height=out.stride(3),
        out_width=out.stride(4),
        channels=channels,
        frames=frames,
        height=height,
        width=width,
        out_frames=out_frames,
        out_rows=out_rows,
        out_columns=out_columns,
        channels_per_group=per_group,
        groups=groups,
        top=options.top,
        left=options.left,
        front=options.front,
        reflect=1 if options.reflect else 0,
        normalization=branch,
    )
    status = entry(ctypes.byref(call), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if status != 0:
        raise SystemExit(f"the module refused: {call.error.decode(errors='replace')}")

    expected = reference(
        values,
        mean,
        rstd,
        weight,
        bias,
        groups,
        branch,
        options.top,
        options.left,
        options.front,
        (out_frames, out_rows, out_columns),
        mode,
    )
    error = (out - expected).abs()
    allowed = declared_bound() * expected.abs()
    violations = int((~(error <= allowed)).sum())
    identical = int((out.reshape(-1).view(torch.int32) == expected.reshape(-1).view(torch.int32)).sum())
    print(
        f"channels {channels} groups {groups} frames {frames} {height}x{width} -> "
        f"{out_frames}x{out_rows}x{out_columns}, branch {branch}, "
        f"{'reflect' if options.reflect else 'replicate'}"
        f"{', channels last' if options.channels_last else ''}: "
        f"max |error| {torch.nan_to_num(error).max().item():.3e}, rtol {declared_bound():.3e}, "
        f"{identical} of {error.numel()} bit-identical"
    )
    if violations:
        print(f"FAIL: {violations} of {error.numel()} elements outside the operator's declared bound")
        return 1
    print("ok: the whole composition within the operator's declared bound")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", nargs="?", type=pathlib.Path, help="the directory with module.so")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--bottom", type=int, default=2)
    parser.add_argument("--left", type=int, default=3)
    parser.add_argument("--right", type=int, default=3)
    parser.add_argument("--front", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--reflect", action="store_true", help="mirror the pads instead of replicating")
    parser.add_argument("--centered", action="store_true", help="normalize without affine parameters")
    parser.add_argument("--single-pixel", action="store_true", help="one pixel per frame")
    parser.add_argument("--channels-last", action="store_true", help="an output whose channel stride is 1")
    parser.add_argument("--cpu-only", action="store_true", help="check the host-side parts only")
    options = parser.parse_args()

    check_call_structure()
    if options.single_pixel:
        options.height = options.width = 1
        options.top = options.left = options.bottom = options.right = 0
    if options.module is None or not (options.module / "module.so").is_file():
        if options.cpu_only:
            print("cpu-only: the call structure is consistent")
            return 0
        parser.error("a module directory holding module.so is needed")
    check_refusal(options.module)
    if options.cpu_only:
        check_reference_by_hand()
        print("cpu-only: the host-side parts of this script are consistent")
        return 0
    return run(options)


if __name__ == "__main__":
    sys.exit(main())
