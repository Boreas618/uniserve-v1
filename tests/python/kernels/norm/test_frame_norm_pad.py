"""The frame_norm_pad module against the eager composition it replaces.

A test loads the built module of the package's descriptor (``dtype_f32``,
resolved at the session's Build Server) as a runner loads it, calls the
operator's entry with the call structure of ``crates/ops/src/frame.rs``
(``FrameNormPadCall``, the generated header's
``uniserve_frame_norm_pad_call``), and compares the output with::

    out = pad(silu(group_norm(values)))      # F.group_norm, F.silu, F.pad

**What is checked.** The moments the call carries are computed by torch and
passed in, so the tests check the normalization, the SiLU and the padded
store: the parts of the entry that are not ``frame_moments``, whose own check
holds the moments. The three normalization branches are the call's
(``uniserve_normalization``): with affine parameters (``fused``), without
them (``centered``), and ``GroupNorm1dForward``'s expressions for frames of
one pixel (``singlepixel``); each pads by replicating or by reflecting its
border.

**The layout is the call's.** The entry's two template branches write the
same addresses, ``channel * out_channel + column * out_column``; the layout
only decides which loop coalesces. A channels-last output has a channel
stride of 1, so a wrong stride handling shows as a wrong result rather than
as slowness.

**The bound** is the operator's, ``Tolerance::rounding(1 << 16, f32, f32)``.
The module states it composes PyTorch's own kernels' arithmetic, so a failure
also states how many elements are bit-identical.

**One sample per call**, because the call structure carries no sample count.
The output frames are ``front + frames``: more would read past the source.

The call structure is a ctypes mirror of the generated ABI header
``uniserve_frame_norm_pad.h``, checked against the layout the header
asserts; the reference is checked against an element-by-element composition
on the host; every other test loads a module built for the session's device.
"""

import ctypes
import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from uniserve_testing.kernels.norm.rounding import gamma, outside
from uniserve_testing.kernels.packages import (
    abi_enumerator,
    assert_abi_layout,
)

pytestmark = pytest.mark.unit

#: The largest message an entry writes (`ops::abi::MESSAGE`).
MESSAGE = 512

#: The descriptor key of the package's one module.
KEY = "dtype_f32"

#: The normalization branches `frame.cu` implements.
FUSED = abi_enumerator("UNISERVE_NORMALIZATION_FUSED")
CENTERED = abi_enumerator("UNISERVE_NORMALIZATION_CENTERED")
SINGLE_PIXEL = abi_enumerator("UNISERVE_NORMALIZATION_SINGLEPIXEL")

#: The extents of the values and of the padding: channels, groups, frames,
#: height and width; the rows above and below, the columns left and right,
#: and the zero frames in front.
CHANNELS, GROUPS, FRAMES, HEIGHT, WIDTH = 32, 4, 3, 9, 7
TOP, BOTTOM, LEFT, RIGHT, FRONT = 4, 2, 3, 3, 2

EPSILON = 1e-6

#: The seed of a module test's values, and of the host's reference check.
SEED, REFERENCE_SEED = 5, 13

#: An address the entry refuses before it reads it, standing for a buffer.
ADDRESS = 0x1000

#: The branches a module test calls: each pads by replicating and by
#: reflecting, except a single pixel, which has no border to pad.
BRANCHES = [
    pytest.param(FUSED, mode, id=f"affine, {mode}")
    for mode in ("replicate", "reflect")
] + [
    pytest.param(CENTERED, mode, id=f"centered, {mode}")
    for mode in ("replicate", "reflect")
]
BRANCHES.append(pytest.param(SINGLE_PIXEL, "replicate", id="single pixel"))


class Call(ctypes.Structure):
    """`FrameNormPadCall`, field for field."""

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


def declared_bound():
    """`Tolerance::rounding(1 << 16, f32, f32)`'s relative bound."""
    return 2.0 * gamma(1 << 16) + 2.0**-24


def entry(kernel_module):
    """The module's entry, with its C signature."""
    library = ctypes.CDLL(
        str(kernel_module("frame_norm_pad", "nvcc", KEY) / "module.so")
    )
    function = library.uniserve_frame_norm_pad
    function.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


def moments(values, groups, epsilon):
    """The group statistics a call carries, from torch:
    `var_mean(unbiased=False)`, then `rsqrt(var + eps)`.
    """  # noqa: D205
    _, channels, frames, height, width = values.shape
    folded = (
        values.reshape(1, groups, channels // groups, frames, height, width)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(frames, groups, -1)
    )
    variance, mean = torch.var_mean(folded, dim=2, unbiased=False)
    return mean.contiguous(), torch.rsqrt(variance + epsilon).contiguous()


def reference(values, mean, rstd, weight, bias, branch, pads, out_shape, mode):
    """`pad(silu(group_norm(values)))` as the eager composition, in torch.

    `pads` are the rows above, the columns left and the frames in front;
    `out_shape` the output's frames, rows and columns.
    """
    _, channels, frames, height, width = values.shape
    groups = mean.shape[1]
    per_group = channels // groups
    top, left, front = pads
    normalized = torch.empty_like(values)
    for group in range(groups):
        channels_of = slice(group * per_group, (group + 1) * per_group)
        group_mean = mean[:, group].reshape(1, 1, frames, 1, 1)
        group_rstd = rstd[:, group].reshape(1, 1, frames, 1, 1)
        if branch == FUSED:
            scale = group_rstd * weight[channels_of].reshape(1, -1, 1, 1, 1)
            shift = -scale * group_mean + bias[channels_of].reshape(
                1, -1, 1, 1, 1
            )
            normalized[:, channels_of] = scale * values[:, channels_of] + shift
        else:
            normalized[:, channels_of] = (
                values[:, channels_of] - group_mean
            ) * group_rstd

    out_frames, out_rows, out_columns = out_shape
    padded = F.pad(
        F.silu(normalized),
        (left, out_columns - left - width, top, out_rows - top - height, 0, 0),
        mode=mode,
    )
    out = torch.zeros(
        1,
        channels,
        out_frames,
        out_rows,
        out_columns,
        dtype=values.dtype,
        device=values.device,
    )
    out[:, :, front : front + frames] = padded
    return out


def source_index(index, extent, reflect):
    """The source position a padded index reads: `frame.cu`'s mirror or
    clamp.
    """  # noqa: D205
    if reflect:
        index = -index if index < 0 else index
        return 2 * extent - 2 - index if index >= extent else index
    return min(max(index, 0), extent - 1)


def test_the_call_structure_is_the_headers():
    """The generated header asserts the size and every offset it holds."""
    assert_abi_layout(Call, "uniserve_frame_norm_pad_call")


@pytest.mark.parametrize("mode", ["replicate", "reflect"])
def test_the_reference_is_the_operators_definition(mode):
    """Torch's pad, SiLU and group scaling, element by element.

    The output is walked element by element, each padded position mapped to a
    source pixel with the kernel's mirror-or-clamp rule, and the definition
    evaluated there; the leading frames are zeros, not a padding of the
    first frame.
    """
    reflect = mode == "reflect"
    torch.manual_seed(REFERENCE_SEED)
    channels, groups, frames, height, width = 4, 2, 2, 3, 4
    top, left, bottom, right, front = 1, 2, 1, 1, 1
    values = torch.randn(1, channels, frames, height, width)
    mean, rstd = moments(values, groups, EPSILON)
    weight = torch.randn(channels)
    bias = torch.randn(channels)
    out_shape = (front + frames, height + top + bottom, width + left + right)

    expected = reference(
        values,
        mean,
        rstd,
        weight,
        bias,
        FUSED,
        (top, left, front),
        out_shape,
        mode,
    )

    per_group = channels // groups
    for channel in range(channels):
        group = channel // per_group
        for out_frame in range(out_shape[0]):
            for out_row in range(out_shape[1]):
                for out_column in range(out_shape[2]):
                    want = 0.0
                    if out_frame >= front:
                        frame = out_frame - front
                        row = source_index(out_row - top, height, reflect)
                        column = source_index(out_column - left, width, reflect)
                        scale = float(rstd[frame, group]) * float(
                            weight[channel]
                        )
                        shift = -scale * float(mean[frame, group]) + float(
                            bias[channel]
                        )
                        value = (
                            scale
                            * float(values[0, channel, frame, row, column])
                            + shift
                        )
                        want = value / (1.0 + math.exp(-value))
                    got = float(
                        expected[0, channel, out_frame, out_row, out_column]
                    )
                    assert abs(want - got) <= 1e-5 * max(1.0, abs(want)), (
                        f"the reference disagrees with the definition at "
                        f"[{channel}, {out_frame}, {out_row}, {out_column}] "
                        f"({mode}): {got} vs {want}"
                    )
    leading = float(expected[0, :, :front].abs().max())
    assert leading == 0.0, f"the leading frames hold {leading}, not zeros"


@pytest.mark.gpu
def test_the_entry_refuses_a_call_with_no_output_on_the_host(kernel_module):
    call = Call(
        values=ADDRESS,
        mean=ADDRESS,
        rstd=ADDRESS,
        out=0,
        channels=8,
        height=4,
        width=4,
    )

    status = entry(kernel_module)(ctypes.byref(call), None)

    message = call.error.decode(errors="replace")
    assert status != 0, "the entry accepted a call with no output buffer"
    assert "needs the values, both statistics and the output" in message, (
        f"the entry refused with {message!r}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "channels_last", [False, True], ids=["channels first", "channels last"]
)
@pytest.mark.parametrize(("branch", "mode"), BRANCHES)
def test_the_composition_matches_torch_within_the_operators_bound(
    branch, mode, channels_last, torch_device, kernel_module
):
    function = entry(kernel_module)
    torch.manual_seed(SEED)
    height, width = (1, 1) if branch == SINGLE_PIXEL else (HEIGHT, WIDTH)
    top, bottom, left, right = (
        (0, 0, 0, 0) if branch == SINGLE_PIXEL else (TOP, BOTTOM, LEFT, RIGHT)
    )
    out_shape = (FRONT + FRAMES, height + top + bottom, width + left + right)
    values = torch.randn(
        1, CHANNELS, FRAMES, height, width, device=torch_device
    )
    mean, rstd = moments(values, GROUPS, EPSILON)
    weight = bias = None
    if branch == FUSED:
        weight = torch.randn(CHANNELS, device=torch_device)
        bias = torch.randn(CHANNELS, device=torch_device)
    out = torch.zeros(1, CHANNELS, *out_shape, device=torch_device)
    if channels_last:
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
        channels=CHANNELS,
        frames=FRAMES,
        height=height,
        width=width,
        out_frames=out_shape[0],
        out_rows=out_shape[1],
        out_columns=out_shape[2],
        channels_per_group=CHANNELS // GROUPS,
        groups=GROUPS,
        top=top,
        left=left,
        front=FRONT,
        reflect=1 if mode == "reflect" else 0,
        normalization=branch,
    )

    status = function(
        ctypes.byref(call), torch.cuda.current_stream(torch_device).cuda_stream
    )
    torch.cuda.synchronize(torch_device)

    assert status == 0, (
        f"the module refused: {call.error.decode(errors='replace')}"
    )
    expected = reference(
        values,
        mean,
        rstd,
        weight,
        bias,
        branch,
        (top, left, FRONT),
        out_shape,
        mode,
    )
    error = (out - expected).abs()
    bound = declared_bound()
    missed = outside(error, bound * expected.abs())
    identical = int(
        (
            out.reshape(-1).view(torch.int32)
            == expected.reshape(-1).view(torch.int32)
        ).sum()
    )
    assert missed == 0, (
        f"{missed} of {error.numel()} elements outside the bound, max |error| "
        f"{torch.nan_to_num(error).max().item():.3e}, rtol {bound:.3e}, "
        f"{identical} of {error.numel()} bit-identical"
    )
