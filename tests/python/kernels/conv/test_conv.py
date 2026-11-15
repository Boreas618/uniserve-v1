"""The cuDNN convolution modules against torch's own convolution in FP64.

The three convolution operators share one source and one check. Each test
loads the built module of its descriptor (``input_<t>_weight_<t>_output_<t>``,
resolved at the session's Build Server) as a runner loads it, calls its entry
with the call structure of ``crates/ops/src/conv.rs``, and compares the output
with ``torch.nn.functional``'s convolution in FP64.

The tolerance is the operator's: ``conv.rs::tolerance`` declares
``Tolerance::dot(taps, F32, output)``, whose relative bound is
``2 gamma_n sqrt(n) + u_out`` over all ``n`` products, the statistical bound
for a sum of mixed-sign products, held against the output's largest
magnitude. A ``bf16`` convolution keeps the same bound with ``u_out`` of its
own element type. ``tf32`` is a permission for an FP32 convolution only, so
these calls withhold it: a convolution cuDNN may run on TF32 tensor cores
does not meet the FP32 bound, which the operator's tolerance does not
account for.

Channels-last stores the input and the output with the channel axis fastest
and leaves the filter packed, which is the order ``conv.rs`` declares; a bias
is on, so that run also reads the bias against the output's order.

The operator's rule that ``tf32`` is a permission for FP32 only is checked by
a refusal: a half-precision module given ``tf32`` refuses the call with a
message containing "cannot use it", before it examines any address.

Every test loads a module built for the session's device, so every test
needs one.
"""

import ctypes
import math

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

#: The convolution operators, each with its C entry and torch's reference.
OPERATORS = {
    "conv1d": ("uniserve_conv1d", torch.nn.functional.conv1d),
    "conv3d": ("uniserve_conv3d", torch.nn.functional.conv3d),
    "conv_transpose1d": (
        "uniserve_conv_transpose1d",
        torch.nn.functional.conv_transpose1d,
    ),
}

#: The unit round-off of each element type, as `ops::tolerance::unit_roundoff`
#: reports it.
UNIT_ROUNDOFF = {"bf16": 3.90625e-03, "f32": 5.960464477539063e-08}

TORCH_TYPES = {"bf16": torch.bfloat16, "f32": torch.float32}

#: The extents every case convolves: the batch, the input and output channels
#: and the spatial extent per axis.
BATCH, CHANNELS, OUT_CHANNELS, LENGTH = 2, 8, 12, 17

#: The geometries a case takes, each value the first axis of the three-axis
#: triplet a call carries: the filter, the stride, the padding and the
#: dilation. Every other axis is 1 (0 for a padding: `conv.rs`'s 1-d geometry
#: pads only its length).
GEOMETRIES = {
    "unit stride": {"kernel": 3, "stride": 1, "padding": 1, "dilation": 1},
    "stride 2": {"kernel": 3, "stride": 2, "padding": 1, "dilation": 1},
}

#: The phrase a half-precision module's refusal of `tf32` contains.
TF32_REFUSAL = "cannot use it"


class Call(ctypes.Structure):
    """`ops::conv::Call`, field for field."""

    _fields_ = [
        ("out", ctypes.c_void_p),
        ("input", ctypes.c_void_p),
        ("weight", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("input_shape", ctypes.c_int64 * 5),
        ("weight_shape", ctypes.c_int64 * 5),
        ("out_shape", ctypes.c_int64 * 5),
        ("kernel", ctypes.c_int32 * 3),
        ("stride", ctypes.c_int32 * 3),
        ("padding", ctypes.c_int32 * 3),
        ("dilation", ctypes.c_int32 * 3),
        ("groups", ctypes.c_int32),
        ("transposed", ctypes.c_int32),
        ("tf32", ctypes.c_int32),
        ("channels_last", ctypes.c_int32),
        ("error", ctypes.c_char * 512),
    ]


def key(dtype):
    """The descriptor key of a convolution module of element type `dtype`."""
    return f"input_{dtype}_weight_{dtype}_output_{dtype}"


def entry(kernel_module, op, dtype):
    """The C entry of `op`'s built module of element type `dtype`."""
    name, _ = OPERATORS[op]
    library = ctypes.CDLL(
        str(kernel_module(op, "cudnn", key(dtype)) / "module.so")
    )
    function = getattr(library, name)
    function.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


def bound(taps, output):
    """The operator's relative bound over `taps` products into `output`."""
    terms = taps * UNIT_ROUNDOFF["f32"]
    gamma = terms / (1.0 - terms)
    return 2.0 * gamma * math.sqrt(taps) + UNIT_ROUNDOFF[output]


def triplet(value, unit=1):
    """A geometry value as the three axes a call carries."""
    return [value] + [unit] * 2


def padded(shape):
    """Five extents: the leading axes, the spatial ones, then ones."""
    return list(shape) + [1] * (5 - len(shape))


def stored(tensor, channels_last):
    """The tensor in the storage the call's order names; its logical shape
    does not move.
    """  # noqa: D205
    if not channels_last:
        return tensor
    last = tensor.dim() - 1
    return (
        tensor.permute(0, *range(2, tensor.dim()), 1)
        .contiguous()
        .permute(0, last, *range(1, last))
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize(
    "channels_last", [False, True], ids=["channels first", "channels last"]
)
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("op", OPERATORS)
def test_a_convolution_matches_torch_within_the_operators_bound(
    op, dtype, channels_last, geometry, torch_device, kernel_module
):
    function = entry(kernel_module, op, dtype)
    shape = GEOMETRIES[geometry]
    torch.manual_seed(0)
    rank = 3 if op == "conv3d" else 1
    element = TORCH_TYPES[dtype]
    # The geometry along the operator's own axes: a value names the first axis,
    # and the others keep their unit.
    kernel = triplet(shape["kernel"])[:rank]
    stride = triplet(shape["stride"])[:rank]
    padding = triplet(shape["padding"], 0)[:rank]
    dilation = triplet(shape["dilation"])[:rank]

    x = stored(
        torch.randn(
            BATCH,
            CHANNELS,
            *([LENGTH] * rank),
            dtype=element,
            device=torch_device,
        ),
        channels_last,
    )
    if op == "conv_transpose1d":
        weight = torch.randn(
            CHANNELS, OUT_CHANNELS, *kernel, dtype=element, device=torch_device
        )
    else:
        weight = torch.randn(
            OUT_CHANNELS, CHANNELS, *kernel, dtype=element, device=torch_device
        )
    bias = torch.randn(OUT_CHANNELS, dtype=element, device=torch_device)

    _, reference = OPERATORS[op]
    with torch.no_grad():
        expected = reference(
            x.double(),
            weight.double(),
            bias.double(),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    # The output's own extents, from the reference: the call carries them, and
    # a module that disagreed with them would refuse the launch.
    out = stored(
        torch.empty(expected.shape, dtype=element, device=torch_device),
        channels_last,
    )

    call = Call()
    call.out = ctypes.c_void_p(out.data_ptr())
    call.input = ctypes.c_void_p(x.data_ptr())
    call.weight = ctypes.c_void_p(weight.data_ptr())
    call.bias = ctypes.c_void_p(bias.data_ptr())
    call.input_shape = (ctypes.c_int64 * 5)(*padded(x.shape))
    call.weight_shape = (ctypes.c_int64 * 5)(*padded(weight.shape))
    call.out_shape = (ctypes.c_int64 * 5)(*padded(expected.shape))
    call.kernel = (ctypes.c_int32 * 3)(*triplet(shape["kernel"]))
    call.stride = (ctypes.c_int32 * 3)(*triplet(shape["stride"]))
    call.padding = (ctypes.c_int32 * 3)(*triplet(shape["padding"], 0))
    call.dilation = (ctypes.c_int32 * 3)(*triplet(shape["dilation"]))
    call.groups = 1
    call.transposed = int(op == "conv_transpose1d")
    call.tf32 = 0
    call.channels_last = int(channels_last)

    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = function(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the entry refused the call: {call.error.decode(errors='replace')}"
    )

    taps = math.prod(kernel) * CHANNELS
    difference = (out.double() - expected).abs()
    # A NaN anywhere makes a maximum NaN, which hides every other error and
    # passes a `>` comparison: it counts as an infinite error.
    worst = difference.nan_to_num(nan=float("inf")).max().item()
    relative = bound(taps, dtype)
    largest = expected.abs().max().item()
    assert worst <= relative * largest, (
        f"{op} {dtype}, in {tuple(x.shape)} filter {tuple(weight.shape)} out "
        f"{tuple(expected.shape)}, {taps} taps: max |out - exact| = "
        f"{worst:.6g} exceeds the operator's bound {relative:.3g} of the "
        f"largest value {largest:.6g}"
    )


@pytest.mark.parametrize("op", OPERATORS)
def test_a_half_precision_convolution_refuses_tf32(op, kernel_module):
    """`tf32` is a permission only an FP32 convolution can use.

    The answer belongs to the module, not to the buffers, so the refusal comes
    before any address is examined: the call is empty but for the flag, and is
    made on no stream.
    """
    function = entry(kernel_module, op, "bf16")
    call = Call()
    call.tf32 = 1

    status = function(ctypes.byref(call), ctypes.c_void_p(0))

    refusal = call.error.decode(errors="replace")
    assert status != 0, "the bf16 module accepted a call that sets tf32"
    assert TF32_REFUSAL in refusal, f"refused for another reason: {refusal}"
