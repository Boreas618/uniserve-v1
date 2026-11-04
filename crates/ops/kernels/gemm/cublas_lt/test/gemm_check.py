#!/usr/bin/env python3
"""Run one cuBLASLt gemm module on a GPU and check it against an exact product.

    python gemm_check.py <module_dir>
    python gemm_check.py <module_dir> --input bf16 --weight bf16 --accumulate f32 \
        --output bf16 --order nxk --epilogue bias --m 128 --n 4096 --k 2048
    python gemm_check.py <module_dir> --alpha 0.5 --input-row-stride 2080

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding
``module.so``. The flags name the descriptor the module was built for, so this
script prints the key it corresponds to — pass ``--expect-key`` to fail when the
module served is not the one you thought.

The tolerance is ``ops::gemm``'s own: ``Tolerance::dot(k, accumulate, output)``,
the statistical bound for a sum of ``k`` mixed-sign products in the declared
accumulator. ``--accumulate bf16`` therefore allows what a bf16 accumulation
really loses; ``--accumulate f32`` is the tight case, and a module that split the
contraction into pieces accumulated in bf16 fails it, which is the point of
declaring the accumulator in the descriptor at all.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

#: The operator crate's `DType::as_i64()` numbering, by element type name.
DTYPES = {"f16": 3, "bf16": 4, "f32": 5}

#: `DType::unit_roundoff`, per element type.
UNIT_ROUNDOFF = {
    "f16": 4.8828125e-04,
    "bf16": 3.90625e-03,
    "f32": 5.960464477539063e-08,
}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


class Call(ctypes.Structure):
    """`ops::gemm::Call`, field for field: 600 bytes, `error` at 84."""

    _fields_ = [
        ("out", ctypes.c_void_p),
        ("input", ctypes.c_void_p),
        ("weight", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("m", ctypes.c_int64),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
        ("input_row_stride", ctypes.c_int64),
        ("weight_row_stride", ctypes.c_int64),
        ("out_row_stride", ctypes.c_int64),
        ("alpha", ctypes.c_float),
        ("error", ctypes.c_char * 512),
    ]


def bound(reduction, accumulate, output):
    """The operator's relative bound over the contraction."""
    terms = reduction * UNIT_ROUNDOFF[accumulate]
    gamma = terms / (1.0 - terms)
    return 2.0 * gamma * math.sqrt(reduction) + UNIT_ROUNDOFF[output]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module", type=pathlib.Path, help="the module directory"
    )
    parser.add_argument("--input", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--weight", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--accumulate", choices=sorted(DTYPES), default="f32")
    parser.add_argument("--output", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--order", choices=["kxn", "nxk"], default="nxk")
    parser.add_argument("--epilogue", choices=["none", "bias"], default="none")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--input-row-stride",
        type=int,
        default=0,
        help="elements between input rows; 0 is the packed stride",
    )
    parser.add_argument(
        "--expect-key",
        default=None,
        help="the descriptor key the module was built for, to check the flags against",
    )
    arguments = parser.parse_args()

    key = (
        f"input_{arguments.input}_weight_{arguments.weight}"
        f"_accumulate_{arguments.accumulate}_output_{arguments.output}"
        f"_order_{arguments.order}_epilogue_{arguments.epilogue}_k_{arguments.k}"
    )
    print(f"descriptor: {key}")
    # The directory `resolve` hands a runner is named by the descriptor the module was
    # built for, so the flags are held against it: a check that tested its own defaults
    # against a module built for other flags would report a kernel defect that is not one.
    named = arguments.expect_key or (
        arguments.module.name if "_order_" in arguments.module.name else None
    )
    if named is not None and named != key:
        parser.error(
            f"the flags name {key}, and the module was built for {named}"
        )

    library = ctypes.CDLL(str(arguments.module / "module.so"))
    entry = library.uniserve_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    device = torch.device("cuda")
    torch.manual_seed(0)
    m, n, k = arguments.m, arguments.n, arguments.k
    row_stride = arguments.input_row_stride or k
    if row_stride < k:
        parser.error(
            "the input's row stride cannot be shorter than its contraction"
        )

    # The input is a view with `row_stride` elements between rows, which is what a
    # fused activation leaves behind; the store's own GEMMs see packed rows.
    storage = torch.randn(
        m * row_stride, dtype=TORCH_TYPES[arguments.input], device=device
    )
    x = storage[: m * row_stride].view(m, row_stride)[:, :k]
    weight = torch.randn(
        (k, n) if arguments.order == "kxn" else (n, k),
        dtype=TORCH_TYPES[arguments.weight],
        device=device,
    )
    bias = (
        torch.randn(n, dtype=TORCH_TYPES[arguments.output], device=device)
        if arguments.epilogue == "bias"
        else None
    )
    out = torch.empty(
        (m, n), dtype=TORCH_TYPES[arguments.output], device=device
    )

    with torch.no_grad():
        # `kxn` stores the weight `[k, n]` and contracts its rows; `nxk` stores it
        # `[n, k]`, so the reference takes the transposed product.
        stored = (
            weight.double() if arguments.order == "kxn" else weight.double().T
        )
        expected = x.double() @ stored
        expected *= arguments.alpha
        if bias is not None:
            expected += bias.double()

    call = Call()
    call.out = ctypes.c_void_p(out.data_ptr())
    call.input = ctypes.c_void_p(x.data_ptr())
    call.weight = ctypes.c_void_p(weight.data_ptr())
    call.bias = ctypes.c_void_p(bias.data_ptr()) if bias is not None else None
    call.m, call.n, call.k = m, n, k
    call.input_row_stride = row_stride
    # The stored weight is `[k, n]` for `kxn` and `[n, k]` for `nxk`, so the
    # elements between its rows are its row length: `n` and `k`.
    call.weight_row_stride = n if arguments.order == "kxn" else k
    call.out_row_stride = n
    call.alpha = arguments.alpha

    stream = torch.cuda.current_stream().cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if status != 0:
        print(
            f"the entry refused the call: {call.error.decode(errors='replace')}",
            file=sys.stderr,
        )
        return 1

    difference = (out.double() - expected).abs().max().item()
    largest = expected.abs().max().item()
    allowed = bound(k, arguments.accumulate, arguments.output)
    print(
        f"[{m}, {n}] over {k}, alpha {arguments.alpha}, "
        f"bias {arguments.epilogue}, input row stride {row_stride}"
    )
    print(f"max |out - exact| = {difference:.6g}, largest value {largest:.6g}")
    print(
        f"the operator's bound {allowed:.3g} relative = {allowed * largest:.6g}"
    )
    if difference > allowed * largest:
        print("the module exceeds the operator's bound", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
