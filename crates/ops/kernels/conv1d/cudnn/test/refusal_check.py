#!/usr/bin/env python3
"""Check that a convolution module refuses the calls the operator says it must.

    python refusal_check.py <module_dir> --expect "cannot use it"

The module's rule is that `tf32` is a permission only an FP32 convolution can use, so a
module whose inputs are half precision must refuse a call that sets it. That answer
belongs to the module, not to the buffers, so it is given before any address is
examined — which is what lets this check run **without a device**: the module is loaded,
the descriptor's flag is set, and the refusal is read out of `call->error`.

`--expect` is the phrase the refusal must contain; the run fails when the call is
accepted, when the refusal says something else, and when the module is the wrong one for
the question (the directory's descriptor names its element types). A module built for an
FP32 input serves the call instead of refusing it, and that is reported as what it is.
"""

import argparse
import ctypes
import pathlib
import sys

#: The operator crate's `DType::as_i64()` numbering, by element type name.
DTYPES = {"f16": 3, "bf16": 4, "f32": 5}


#: The call structure of `crates/ops/src/conv.rs`, as its header mirrors it.
class Call(ctypes.Structure):
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


ENTRIES = {
    "conv1d": "uniserve_conv1d",
    "conv3d": "uniserve_conv3d",
    "conv_transpose1d": "uniserve_conv_transpose1d",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module", type=pathlib.Path, help="the module directory"
    )
    parser.add_argument("--op", choices=sorted(ENTRIES), required=True)
    parser.add_argument(
        "--tf32",
        default="true",
        help="the descriptor's flag, as the call carries it: 1 or 0",
    )
    parser.add_argument(
        "--expect",
        default="cannot use it",
        help="the phrase the refusal must contain",
    )
    arguments = parser.parse_args()

    library = ctypes.CDLL(str(arguments.module / "module.so"))
    entry = getattr(library, ENTRIES[arguments.op])
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    call = Call()
    call.tf32 = 1 if arguments.tf32 in ("true", "1") else 0
    names = arguments.module.name
    if "_input_" in names:
        element = names.split("_input_", 1)[1].split("_", 1)[0]
        print(
            f"module: {element} input, tf32 {arguments.tf32} "
            f"({'the call sets it' if call.tf32 else 'the call clears it'})"
        )

    status = entry(ctypes.byref(call), ctypes.c_void_p(0))
    refusal = call.error.decode(errors="replace")
    if status == 0:
        print(
            "FAIL: the module accepted the call: an FP32 module serves either "
            "permission, so this run needed a half-precision module",
            file=sys.stderr,
        )
        return 1
    if arguments.expect not in refusal:
        print(f"FAIL: refused for another reason: {refusal}", file=sys.stderr)
        return 1
    print(f"PASS: the module refused the call as it must: {refusal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
