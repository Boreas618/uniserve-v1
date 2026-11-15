"""The cuBLASLt `gemm` module against the exact product.

Each test loads the built module of its descriptor
(``input_<t>_weight_<t>_accumulate_<t>_output_<t>_order_<o>_epilogue_<e>``,
resolved at the session's Build Server for the session's device) as a runner
loads it (``module.so``), calls its entry with the call structure of
``crates/ops/src/gemm.rs`` and compares the output with the product in FP64.

The tolerance is ``ops::gemm``'s own: ``Tolerance::dot(k, accumulate,
output)``, the statistical bound for a sum of ``k`` mixed-sign products in the
declared accumulator, held against the output's largest magnitude. An ``f32``
accumulator is the tight case, and a module that split the contraction into
pieces accumulated in BF16 fails it, which is the point of declaring the
accumulator in the descriptor.

The cases are the module's served form and the call's two runtime words: a
bias epilogue, and a factor `alpha` over an input whose rows are wider than
the contraction, as a fused activation leaves them.
"""

import ctypes
import math

import pytest
import torch
from uniserve_testing.kernels.packages import assert_abi_layout

pytestmark = pytest.mark.unit

#: `ops::tolerance::unit_roundoff`, per element type.
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
    """`ops::gemm::Call`, field for field."""

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


def test_the_call_mirror_is_the_headers_gemm_call():
    """The generated header asserts the size and offsets `Call` has."""
    assert_abi_layout(Call, "UniserveGemmCall")


def bound(reduction, accumulate, output):
    """The operator's relative bound over the contraction."""
    terms = reduction * UNIT_ROUNDOFF[accumulate]
    gamma = terms / (1.0 - terms)
    return 2.0 * gamma * math.sqrt(reduction) + UNIT_ROUNDOFF[output]


#: The product's extents.
M, N, K = 128, 4096, 2048

#: The cases: the epilogue, `alpha` and the elements between input rows (`K`,
#: packed, when `None`), over BF16 operands accumulated in FP32 with the weight
#: stored `[n, k]`.
CASES = [
    pytest.param("none", 1.0, None, id="the served form"),
    pytest.param("bias", 1.0, None, id="a bias"),
    pytest.param("none", 0.5, 2080, id="alpha over strided rows"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("epilogue", "alpha", "row_stride"), CASES)
def test_the_product_is_within_the_operators_bound(
    epilogue, alpha, row_stride, torch_device, kernel_module
):
    dtypes = {"input": "bf16", "weight": "bf16", "accumulate": "f32"}
    output, order = "bf16", "nxk"
    key = (
        f"input_{dtypes['input']}_weight_{dtypes['weight']}_accumulate_"
        f"{dtypes['accumulate']}_output_{output}_order_{order}_epilogue_"
        f"{epilogue}"
    )
    library = ctypes.CDLL(
        str(kernel_module("gemm", "cublas_lt", key) / "module.so")
    )
    entry = library.uniserve_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    torch.manual_seed(0)
    row_stride = row_stride or K
    # The input is a view with `row_stride` elements between rows, which is
    # what a fused activation leaves behind.
    storage = torch.randn(
        M * row_stride, dtype=TORCH_TYPES[dtypes["input"]], device=torch_device
    )
    x = storage.view(M, row_stride)[:, :K]
    # `nxk` stores the weight `[n, k]`, so the reference takes the transposed
    # product.
    weight = torch.randn(
        N, K, dtype=TORCH_TYPES[dtypes["weight"]], device=torch_device
    )
    bias = (
        torch.randn(N, dtype=TORCH_TYPES[output], device=torch_device)
        if epilogue == "bias"
        else None
    )
    out = torch.empty((M, N), dtype=TORCH_TYPES[output], device=torch_device)
    with torch.no_grad():
        expected = (x.double() @ weight.double().T) * alpha
        if bias is not None:
            expected += bias.double()

    call = Call()
    call.out = ctypes.c_void_p(out.data_ptr())
    call.input = ctypes.c_void_p(x.data_ptr())
    call.weight = ctypes.c_void_p(weight.data_ptr())
    call.bias = None if bias is None else ctypes.c_void_p(bias.data_ptr())
    call.m, call.n, call.k = M, N, K
    call.input_row_stride = row_stride
    # The stored `[n, k]` weight's rows are `k` elements apart.
    call.weight_row_stride = K
    call.out_row_stride = N
    call.alpha = alpha
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the entry refused the call: {call.error.decode(errors='replace')}"
    )

    difference = (out.double() - expected).abs().max().item()
    largest = expected.abs().max().item()
    allowed = bound(K, dtypes["accumulate"], output)
    # A difference that is not a number is never inside a bound: `>` would say
    # it is.
    assert difference <= allowed * largest, (
        f"{key}, [{M}, {N}] over {K}, alpha {alpha}, input row stride "
        f"{row_stride}: max |out - exact| = {difference:.6g} exceeds the "
        f"operator's bound {allowed:.3g} of the largest value {largest:.6g}"
    )
