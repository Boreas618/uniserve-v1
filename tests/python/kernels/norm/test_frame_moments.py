"""The frame_moments module against PyTorch's group-norm moments.

A test loads the built module of the package's descriptor (``dtype_f32``,
resolved at the session's Build Server) as a runner loads it, calls the
operator's entry with the call structure of ``crates/ops/src/frame.rs``
(``FrameMomentsCall``, 608 bytes: ``values`` 0, ``values_batch`` 8,
``values_channel`` 16, ``values_frame`` 24, ``values_pixel`` 32, ``frames``
40, ``pixels`` 48, ``channels_per_group`` 56, ``groups`` 64, ``epsilon`` 72,
``mean`` 80, ``rstd`` 88, ``error`` 96), and compares both statistics with
PyTorch's own: ``var_mean(unbiased=False)`` over the same group of channels
and pixels of a frame, then ``rsqrt(var + eps)``.

**The strides are the model's layout.** The values are
``[1, channels, frames, height, width]`` contiguous, ``[B, C, T, H, W]`` as
the model holds them: one frame of one channel is ``height * width`` elements
apart and a channel of the sequence ``frames * height * width``. A module that
required a frame to hold a whole group (frame-major storage) would refuse
this, and that refusal is a module defect, because no model produces that
layout. The same values are also handed over stored channels-last
(``values_pixel`` the channel count), because a channels-last span feeds the
operator that way and a kernel that walked the pixels at a step of one would
reduce the wrong values there while agreeing everywhere else.

**One sample per call.** The call structure carries no sample count and the
entry's grid is one sample's rows, so the operator's caller runs one video
sample per call, and so does the check.

**The bound** is the operator's, ``Tolerance::rounding(1 << 16, f32, f32)``,
the reduction's own rounding at the widest frame a launch carries. The module
states its Welford tree is ATen's, bit for bit PyTorch's, so a failure also
states how many statistics are bit-identical.

The call structure is a ctypes mirror of the generated ABI header
``uniserve_frame_moments.h``, checked against the layout the header
asserts; every other test loads a module built for the session's device.
"""

import ctypes

import pytest
import torch

from tests.python.kernels.norm.rounding import gamma, outside
from tests.python.kernels.packages import assert_abi_layout

pytestmark = pytest.mark.unit

#: The largest message an entry writes (`ops::abi::MESSAGE`).
MESSAGE = 512

#: The descriptor key of the package's one module.
KEY = "dtype_f32"

#: The extents of the values: channels, groups, frames, height and width.
CHANNELS, GROUPS, FRAMES, HEIGHT, WIDTH = 32, 4, 3, 9, 7

EPSILON = 1e-6

#: The seed of the values.
SEED = 3

#: An address the entry refuses before it reads it, standing for a buffer.
ADDRESS = 0x1000


class Call(ctypes.Structure):
    """`FrameMomentsCall`, field for field."""

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


def declared_bound():
    """`Tolerance::rounding(1 << 16, f32, f32)`'s relative bound."""
    return 2.0 * gamma(1 << 16) + 2.0**-24


def entry(kernel_module):
    """The module's entry, with its C signature."""
    library = ctypes.CDLL(
        str(kernel_module("frame_moments", "nvcc", KEY) / "module.so")
    )
    function = library.uniserve_frame_moments
    function.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


def reference(values, groups, epsilon):
    """PyTorch's group-norm moments per frame: `var_mean(unbiased=False)`,
    then `rsqrt(var + eps)`.
    """  # noqa: D205
    batch, channels, frames, height, width = values.shape
    folded = (
        values.reshape(batch, groups, channels // groups, frames, height, width)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(batch * frames, groups, -1)
    )
    variance, mean = torch.var_mean(folded, dim=2, unbiased=False)
    return mean, torch.rsqrt(variance + epsilon)


def test_the_call_structure_is_the_headers():
    """The generated header asserts the size and every offset it holds."""
    assert_abi_layout(Call, "uniserve_frame_moments_call")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("call", "phrase"),
    [
        pytest.param(
            {
                "values": 0,
                "mean": ADDRESS,
                "rstd": ADDRESS,
                "pixels": 64,
                "channels_per_group": 4,
                "groups": 2,
            },
            "needs the values",
            id="no values buffer",
        ),
        # A frame shorter than one row of pixels would read past the tensor,
        # and that is the only layout the entry may refuse: in the layout a
        # model feeds ([B, C, T, H, W]) one frame of one channel is exactly
        # `pixels`, so a stricter guard refuses the operator's own calls.
        pytest.param(
            {
                "values": ADDRESS,
                "mean": ADDRESS,
                "rstd": ADDRESS,
                "pixels": 1024,
                "channels_per_group": 8,
                "groups": 4,
                "frames": 2,
                "values_frame": 64,
                "values_pixel": 1,
            },
            "at least one row of pixels",
            id="a frame shorter than a row of pixels",
        ),
    ],
)
def test_the_entry_refuses_on_the_host(call, phrase, kernel_module):
    refused = Call(**call)

    status = entry(kernel_module)(ctypes.byref(refused), None)

    message = refused.error.decode(errors="replace")
    assert status != 0, f"the entry accepted {call}"
    assert phrase in message, f"the entry refused with {message!r}"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "channels_last", [False, True], ids=["channels first", "channels last"]
)
def test_both_statistics_match_torch_within_the_operators_bound(
    channels_last, torch_device, kernel_module
):
    function = entry(kernel_module)
    torch.manual_seed(SEED)
    base = torch.randn(1, CHANNELS, FRAMES, HEIGHT, WIDTH, device=torch_device)
    # Channels-last stores the channel innermost and hands the operator a view
    # of the logical axes over it.
    values = (
        base.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)
        if channels_last
        else base
    )
    if channels_last:
        assert values.stride(1) == 1 and values.stride(4) == CHANNELS, (
            f"the channels-last view has strides {values.stride()}"
        )
    mean = torch.zeros(FRAMES, GROUPS, device=torch_device)
    rstd = torch.zeros(FRAMES, GROUPS, device=torch_device)
    call = Call(
        values=values.data_ptr(),
        values_batch=values.stride(0),
        values_channel=values.stride(1),
        values_frame=values.stride(2),
        values_pixel=values.stride(4),
        frames=FRAMES,
        pixels=HEIGHT * WIDTH,
        channels_per_group=CHANNELS // GROUPS,
        groups=GROUPS,
        epsilon=EPSILON,
        mean=mean.data_ptr(),
        rstd=rstd.data_ptr(),
    )

    status = function(
        ctypes.byref(call), torch.cuda.current_stream(torch_device).cuda_stream
    )
    torch.cuda.synchronize(torch_device)

    assert status == 0, (
        f"the module refused: {call.error.decode(errors='replace')}"
    )
    expected_mean, expected_rstd = reference(values, GROUPS, EPSILON)
    bound = declared_bound()
    for name, got, expected in (
        ("mean", mean, expected_mean),
        ("rstd", rstd, expected_rstd),
    ):
        error = (got - expected).abs()
        missed = outside(error, bound * expected.abs())
        identical = int(
            (
                got.reshape(-1).view(torch.int32)
                == expected.reshape(-1).view(torch.int32)
            ).sum()
        )
        assert missed == 0, (
            f"{name}: {missed} of {error.numel()} outside the bound, max "
            f"|error| {torch.nan_to_num(error).max().item():.3e}, rtol "
            f"{bound:.3e}, {identical} of {error.numel()} bit-identical"
        )
