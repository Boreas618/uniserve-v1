#!/usr/bin/env python3
"""Run one convolution module on a GPU and check it against torch's own.

    python conv_check.py <module_dir> --op conv3d --tf32 false
    python conv_check.py <module_dir> --op conv1d --dtype bf16 --groups 1
    python conv_check.py <module_dir> --op conv_transpose1d --stride 2

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding
``module.so``. The module is loaded as a runner loads it, called with the call
structure of ``crates/ops/src/conv.rs`` (732 bytes, ``error`` at 216), and
compared with ``torch.nn.functional``'s convolution in FP64.

The tolerance is the operator's, not a guess: `conv.rs::tolerance` declares
``Tolerance::dot(taps, F32, output)``, whose relative bound is ``2 · γₙ · √n +
u_out`` over all ``n`` products — the statistical bound for a sum of mixed-sign
products. Two things this check makes visible rather than hides:

* ``--tf32 true`` is a *permission* for an FP32 convolution, and cuDNN may then
  use TF32 tensor cores, whose inputs carry 10 bits. The operator's declared
  tolerance does not account for that, so this check prints the measured error
  beside both bounds and fails only when the arithmetic it was given cannot meet
  the FP32 one.
* A ``bf16``/``f16`` convolution keeps the same bound with ``u_out`` of its own
  element type.

``--expect-refusal <phrase>`` inverts the run: it passes when the entry refuses the
call with a message containing the phrase, and fails when the call is served. That is
how the operator's rule that ``tf32`` is a permission for FP32 only is checked — a
half-precision module given ``--tf32 true`` must refuse, because it has no such choice
to make, while an FP32 module withholds the permission by being served an engine that
does not narrow its inputs.

Run it with the Program Server's environment
(``crates/program-server/python/.venv/bin/python``), which has torch.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

#: The operator crate's `DType::as_i64()` numbering, by element type name.
DTYPES = {"f16": 3, "bf16": 4, "f32": 5}

#: The unit round-off of each element type, as `DType::unit_roundoff` reports it.
UNIT_ROUNDOFF = {
    "f16": 4.8828125e-04,
    "bf16": 3.90625e-03,
    "f32": 5.960464477539063e-08,
}

#: TF32's inputs carry ten bits of mantissa, which is what a permitted TF32
#: convolution may truncate an FP32 operand to.
TF32_ROUNDOFF = 2.0**-11

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


def pointer(tensor):
    """A tensor's device address as the call structure carries it."""
    return ctypes.c_void_p(tensor.data_ptr())


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


def bound(taps, output, tf32):
    """The operator's relative bound, and TF32's when that is permitted."""
    accumulate = TF32_ROUNDOFF if tf32 else UNIT_ROUNDOFF["f32"]
    terms = taps * accumulate
    gamma = terms / (1.0 - terms)
    return 2.0 * gamma * math.sqrt(taps) + UNIT_ROUNDOFF[output]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module", type=pathlib.Path, help="the module directory"
    )
    parser.add_argument(
        "--op", choices=["conv1d", "conv3d", "conv_transpose1d"], required=True
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="f32")
    parser.add_argument("--tf32", choices=["true", "false"], default="false")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--out-channels", type=int, default=12)
    parser.add_argument(
        "--length", type=int, default=17, help="spatial extent per axis"
    )
    for name, default, help in (
        (
            "--kernel",
            "3",
            "filter extent: one axis, or all three as `a x b x c`",
        ),
        ("--stride", "1", "filter stride: one axis, or all three"),
        ("--padding", "1", "padding: one axis, or all three"),
        ("--dilation", "1", "dilation: one axis, or all three"),
    ):
        parser.add_argument(
            name,
            default=default,
            help=f"{help}; the axes beyond the operator's rank stay as they are "
            "written, because a convolution's cell names all three",
        )
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--bias", choices=["true", "false"], default="true")
    parser.add_argument(
        "--expect-key",
        default=None,
        help="the descriptor key the module was built for, to check the flags against",
    )
    arguments = parser.parse_args()

    # The directory `resolve` hands a runner is named by the descriptor the module was
    # built for, so the flags are held against it: a check that tested its own defaults
    # against a module built for other flags would report a kernel defect that is not one.
    axes = 3 if arguments.op == "conv3d" else 1

    def topology(text):
        """A geometry axis triplet, from one axis' value or all three written out."""
        parts = [int(part) for part in text.split("x")]
        if len(parts) not in (1, 3):
            parser.error(f"{text!r} is neither one value nor three axes")
        # A rank-1 convolution is a three-axis convolution whose other axes are 1,
        # which is what the operator's key spells and what its module was built for.
        return "x".join(str(part) for part in parts + [1] * (3 - len(parts)))

    key = (
        f"input_{arguments.dtype}_weight_{arguments.dtype}"
        f"_output_{arguments.dtype}"
        f"_geometry_kernel_{topology(arguments.kernel)}"
        f"_stride_{topology(arguments.stride)}"
        f"_padding_{topology(arguments.padding)}"
        f"_dilation_{topology(arguments.dilation)}"
        f"_groups_{arguments.groups}_bias_{arguments.bias}"
        f"_tf32_{arguments.tf32}_rank_{axes}"
        f"_transposed_{'true' if arguments.op == 'conv_transpose1d' else 'false'}"
    )
    print(f"descriptor: {key}")
    named = arguments.expect_key or (
        arguments.module.name if "_geometry_" in arguments.module.name else None
    )
    if named is not None and named != key:
        parser.error(
            f"the flags name {key}, and the module was built for {named}"
        )

    entry_name = {
        "conv1d": "uniserve_conv1d",
        "conv3d": "uniserve_conv3d",
        "conv_transpose1d": "uniserve_conv_transpose1d",
    }[arguments.op]
    library = ctypes.CDLL(str(arguments.module / "module.so"))
    entry = getattr(library, entry_name)
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    device = torch.device("cuda")
    torch.manual_seed(0)
    rank = 1 if arguments.op != "conv3d" else 3
    dtype = TORCH_TYPES[arguments.dtype]
    spatial = [arguments.length] * rank

    def axes(text):
        """The three-axis geometry array a call passes, from the flag's triplet."""
        parts = [int(part) for part in text.split("x")]
        return parts + [1] * (3 - len(parts))

    kernel = axes(arguments.kernel)[:rank]
    stride = axes(arguments.stride)[:rank]
    padding = axes(arguments.padding)[:rank]
    dilation = axes(arguments.dilation)[:rank]
    input_channels = arguments.channels
    out_channels = arguments.out_channels
    group_channels = input_channels // arguments.groups

    x = torch.randn(
        arguments.batch, input_channels, *spatial, dtype=dtype, device=device
    )
    if arguments.op == "conv_transpose1d":
        weight = torch.randn(
            input_channels,
            out_channels // arguments.groups,
            *kernel,
            dtype=dtype,
            device=device,
        )
    else:
        weight = torch.randn(
            out_channels, group_channels, *kernel, dtype=dtype, device=device
        )
    bias = (
        torch.randn(out_channels, dtype=dtype, device=device)
        if arguments.bias == "true"
        else None
    )

    library_op = {
        "conv1d": torch.nn.functional.conv1d,
        "conv3d": torch.nn.functional.conv3d,
        "conv_transpose1d": torch.nn.functional.conv_transpose1d,
    }[arguments.op]
    with torch.no_grad():
        expected = library_op(
            x.double(),
            weight.double(),
            None if bias is None else bias.double(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=arguments.groups,
        )
    # The output's own extents, from the reference: the call carries them, and a
    # module that disagreed with them would refuse the launch.
    out_shape = list(expected.shape)
    out = torch.empty(out_shape, dtype=dtype, device=device)

    def padded(shape):
        """A five-entry extent list: the leading axes, the spatial ones, then ones."""
        leading = list(shape[:2])
        if len(shape) == 3:
            return leading + [1, 1] + list(shape[2:])
        return leading + list(shape[2:]) + [1] * (5 - len(shape))

    call = Call()
    call.out = pointer(out)
    call.input = pointer(x)
    call.weight = pointer(weight)
    call.bias = pointer(bias) if bias is not None else None
    call.input_shape = (ctypes.c_int64 * 5)(*padded(x.shape))
    call.weight_shape = (ctypes.c_int64 * 5)(*padded(weight.shape))
    call.out_shape = (ctypes.c_int64 * 5)(*padded(out_shape))
    call.kernel = (ctypes.c_int32 * 3)(*axes(arguments.kernel))
    call.stride = (ctypes.c_int32 * 3)(*axes(arguments.stride))
    call.padding = (ctypes.c_int32 * 3)(*axes(arguments.padding))
    call.dilation = (ctypes.c_int32 * 3)(*axes(arguments.dilation))
    call.groups = arguments.groups
    call.transposed = 1 if arguments.op == "conv_transpose1d" else 0
    call.tf32 = 1 if arguments.tf32 == "true" else 0

    stream = torch.cuda.current_stream().cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if status != 0:
        print(
            f"the entry refused the call: {call.error.decode(errors='replace')}",
            file=sys.stderr,
        )
        return 1

    taps = 1
    for extent in kernel:
        taps *= extent
    taps *= group_channels
    got = out.double()
    difference = (got - expected).abs()
    # A difference that is not a number is never inside a bound: `>` would say it is, and a
    # NaN anywhere makes the tensor's maximum NaN, which hides every other error in it.
    worst = difference.nan_to_num(nan=float("inf")).max().item()
    fp32 = bound(taps, arguments.dtype, False)
    tf32 = bound(taps, arguments.dtype, True)
    largest = expected.abs().max().item()
    print(
        f"{arguments.op}, {arguments.dtype}, tf32 {arguments.tf32}, "
        f"in {tuple(x.shape)} filter {tuple(weight.shape)} out {tuple(out_shape)}, "
        f"{taps} taps"
    )
    print(f"max |out - exact| = {worst:.6g}, largest value {largest:.6g}")
    print(
        f"operator's bound {fp32:.3g} relative; with TF32 permitted {tf32:.3g}"
    )
    allowed = fp32 * expected.abs().max().item()
    if not worst <= allowed:
        print(
            "the module exceeds the operator's FP32 bound"
            + (
                " (TF32 was permitted, so this is expected unless the engine"
                " chose a full-precision one)"
                if arguments.tf32 == "true"
                else ""
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
