"""The `mxfp8_gemm` module's product against the operator's definition.

The module (``module.so``, resolved at the session's Build Server for the
session's device) is called with the operator's call structure
(``crates/ops/src/mx.rs``, :class:`uniserve_testing.kernels.gemm.mx.
GemmCall`), and its product is compared with the operator's definition
computed in FP64 from the codes and the scale bytes the call passes:

    out[m, n] = Σ_k a[m, k] · 2^(sa[m, k/32] − 127)
              · w[n, k] · 2^(sw[n, k/32] − 127)

so the check tests the module's reading of the packed MN-major scale words as
well as its contraction. The operands are encoded by the operator's formula
(:func:`uniserve_testing.kernels.gemm.mx.encode_mxfp8`) from activations and a
block-scaled weight of a checkpoint's magnitudes, each block's scale repeated
over its 32 rows as the loader repeats it.

The widths are DeepSeek-V4.1-Flash's projections: the fused query and
key/value down projection (5120 → 1792), the query's up projection
(1280 → 32768), the indexer's query projection (1280 → 4096) and the output
projection (8192 → 5120), each at a decode step's rows and a prompt's, the
module of each the row class the operator states for them; and the grouped
output projection ``wo_a``, eight groups of 4096 → 1024 in one product, whose
group ``g`` reads columns ``4096 g ..`` of the activations and rows
``1024 g ..`` of the weight, with the weight's scales laid out as those of the
``[1024, 8 · 4096]`` matrix of its groups side by side:

    out[m, 1024 g + j] = Σ_i a[m, 4096 g + i] · w[1024 g + j, i]

**The tolerance** is the operator's declared one, ``Tolerance::dot(k, f32,
bf16)``: ``2γ_k(u_f32)·√k + u_bf16`` of the output's largest magnitude
(:func:`uniserve_testing.kernels.gemm.mx.dot_tolerance`).
"""

import ctypes

import pytest
import torch
from uniserve_testing.kernels.gemm import mx
from uniserve_testing.kernels.packages import assert_abi_layout

pytestmark = pytest.mark.unit

#: The row classes the operator states (`ops::mx::ROW_CLASSES`).
ROW_CLASSES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def row_class(rows):
    """`ops::mx::row_class`: the smallest class that holds the rows."""
    return next((one for one in ROW_CLASSES if one >= rows), ROW_CLASSES[-1])


def test_the_call_mirror_is_the_headers_call_structure():
    """The generated header asserts the size and every offset it holds."""
    assert_abi_layout(mx.GemmCall, "uniserve_mxfp8_gemm_call")


def test_packed_words_hold_each_rows_four_scales_in_block_order():
    """Word ``(g, row)`` holds the row's blocks ``4g .. 4g + 4``, the first in
    its low byte, and the padding rows are zero.
    """  # noqa: D205
    scales = torch.arange(5 * 8, dtype=torch.uint8).reshape(5, 8) + 1
    words = mx.packed_scales(scales)
    assert words.shape == (2, 8)
    word = int(words[1, 3]) & 0xFFFFFFFF
    assert [(word >> (8 * byte)) & 0xFF for byte in range(4)] == scales[
        3, 4:8
    ].tolist()
    assert torch.all(words[:, 5:] == 0)
    assert torch.equal(mx.unpacked_scales(words, 5), scales)


def weight(n, k, seed):
    """An ``[n, k]`` block-scaled FP8 weight: E4M3 codes and one UE8M0 scale
    per 32 × 32 block, the scales repeated over each block's rows.
    """  # noqa: D205
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randint(
        110, 122, (n // 32, k // 32), generator=generator, dtype=torch.int32
    ).to(torch.uint8)
    codes = (torch.randn(n, k, generator=generator) * 64).clamp(-448, 448)
    return codes.to(torch.float8_e4m3fn), blocks.repeat_interleave(32, dim=0)


#: The cases: the activation rows, the output channels and the contraction.
CASES = [
    pytest.param(1, 1792, 5120, id="wq_a+wkv, one decode row"),
    pytest.param(300, 1792, 5120, id="wq_a+wkv, a prompt"),
    pytest.param(6, 32768, 1280, id="wq_b, a decode batch"),
    pytest.param(1000, 32768, 1280, id="wq_b, a prompt"),
    pytest.param(32, 4096, 1280, id="indexer wq_b"),
    pytest.param(3, 5120, 8192, id="wo_b, a decode batch"),
    pytest.param(2048, 5120, 8192, id="wo_b, a long prompt"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("m", "n", "k"), CASES)
def test_the_product_is_within_the_operators_declared_bound(
    m, n, k, kernel_module, torch_device
):
    key = f"output_bf16_n_{n}_k_{k}_groups_1_rows_{row_class(m)}"
    library = ctypes.CDLL(
        str(kernel_module("mxfp8_gemm", "deepgemm", key) / "module.so")
    )
    entry = library.uniserve_mxfp8_gemm
    entry.argtypes = [ctypes.POINTER(mx.GemmCall), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    generator = torch.Generator().manual_seed(m * 7 + n)
    x = torch.randn(m, k, generator=generator).to(torch.bfloat16)
    input_codes, input_scales = mx.encode_mxfp8(x)
    weight_codes, weight_scales = weight(n, k, n + k)

    a = input_codes.to(torch_device)
    w = weight_codes.to(torch_device)
    sa = mx.packed_scales(input_scales).to(torch_device)
    sw = mx.packed_scales(weight_scales).to(torch_device)
    out = torch.full(
        (m, n), float("nan"), dtype=torch.bfloat16, device=torch_device
    )
    multiprocessors = torch.cuda.get_device_properties(
        torch_device
    ).multi_processor_count

    call = mx.GemmCall(
        input=a.data_ptr(),
        input_scales=sa.data_ptr(),
        weight=w.data_ptr(),
        weight_scales=sw.data_ptr(),
        out=out.data_ptr(),
        m=m,
        n=n,
        k=k,
        groups=1,
        input_scale_stride=sa.stride(0),
        weight_scale_stride=sw.stride(0),
        multiprocessors=multiprocessors,
    )
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the module refused: {call.error.decode(errors='replace')}"
    )

    # The reference reads the bytes the call passed, decoded in FP64.
    reference = (
        mx.decode_mxfp8(input_codes, input_scales)
        @ mx.decode_mxfp8(weight_codes, weight_scales).T
    )
    error = (out.double().cpu() - reference).abs()
    atol, rtol = mx.dot_tolerance(k, "bf16")
    peak = float(reference.abs().max())
    # `~(error <= bound)`: a NaN the module left unwritten fails.
    outside = ~(error <= atol + rtol * peak)
    worst = float(torch.nan_to_num(error, nan=float("inf")).max())
    assert not bool(outside.any()), (
        f"[{m}, {n}, {k}]: {int(outside.sum())} of {m * n} elements outside "
        f"atol {atol:.3e} + rtol {rtol:.3e} of the largest |reference| "
        f"{peak:.4g}; max |out - reference| {worst:.4g}"
    )


#: The grouped cases: the activation rows, each group's output channels and
#: contraction, and the groups (`wo_a`).
GROUPED_CASES = [
    pytest.param(1, 1024, 4096, 8, id="wo_a, one decode row"),
    pytest.param(37, 1024, 4096, 8, id="wo_a, a decode batch"),
    pytest.param(700, 1024, 4096, 8, id="wo_a, a prompt"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("m", "n", "k", "groups"), GROUPED_CASES)
def test_each_group_is_within_the_operators_declared_bound(
    m, n, k, groups, kernel_module, torch_device
):
    key = f"output_bf16_n_{n}_k_{k}_groups_{groups}_rows_{row_class(m)}"
    library = ctypes.CDLL(
        str(kernel_module("mxfp8_gemm", "deepgemm", key) / "module.so")
    )
    entry = library.uniserve_mxfp8_gemm
    entry.argtypes = [ctypes.POINTER(mx.GemmCall), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    generator = torch.Generator().manual_seed(m * 7 + groups)
    x = torch.randn(m, groups * k, generator=generator).to(torch.bfloat16)
    input_codes, input_scales = mx.encode_mxfp8(x)
    weight_codes, weight_scales = weight(groups * n, k, n + k)
    # The weight's scales as those of its groups side by side along the
    # contraction: row `j` of the `[n, groups · k]` table holds every group's
    # row `j`.
    beside = (
        weight_scales.reshape(groups, n, k // 32)
        .permute(1, 0, 2)
        .reshape(n, groups * k // 32)
    )

    a = input_codes.to(torch_device)
    w = weight_codes.to(torch_device)
    sa = mx.packed_scales(input_scales).to(torch_device)
    sw = mx.packed_scales(beside).to(torch_device)
    out = torch.full(
        (m, groups * n), float("nan"), dtype=torch.bfloat16, device=torch_device
    )
    multiprocessors = torch.cuda.get_device_properties(
        torch_device
    ).multi_processor_count

    call = mx.GemmCall(
        input=a.data_ptr(),
        input_scales=sa.data_ptr(),
        weight=w.data_ptr(),
        weight_scales=sw.data_ptr(),
        out=out.data_ptr(),
        m=m,
        n=n,
        k=k,
        groups=groups,
        input_scale_stride=sa.stride(0),
        weight_scale_stride=sw.stride(0),
        multiprocessors=multiprocessors,
    )
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the module refused: {call.error.decode(errors='replace')}"
    )

    # Each group's product in FP64 over the bytes the call passed, the groups'
    # outputs side by side.
    activations = mx.decode_mxfp8(input_codes, input_scales)
    matrix = mx.decode_mxfp8(weight_codes, weight_scales)
    reference = torch.cat(
        [
            activations[:, g * k : (g + 1) * k] @ matrix[g * n : (g + 1) * n].T
            for g in range(groups)
        ],
        dim=1,
    )
    error = (out.double().cpu() - reference).abs()
    atol, rtol = mx.dot_tolerance(k, "bf16")
    peak = float(reference.abs().max())
    outside = ~(error <= atol + rtol * peak)
    worst = float(torch.nan_to_num(error, nan=float("inf")).max())
    assert not bool(outside.any()), (
        f"{groups} [{m}, {n}, {k}]: {int(outside.sum())} of "
        f"{m * groups * n} elements outside atol {atol:.3e} + rtol "
        f"{rtol:.3e} of the largest |reference| {peak:.4g}; max "
        f"|out - reference| {worst:.4g}"
    )


@pytest.mark.gpu
def test_the_entry_refuses_another_weight_before_the_device(kernel_module):
    """A call naming another weight's extents is refused on the host, by
    name, with the extents the module is compiled for.
    """  # noqa: D205
    library = ctypes.CDLL(
        str(
            kernel_module(
                "mxfp8_gemm",
                "deepgemm",
                "output_bf16_n_1792_k_5120_groups_1_rows_16",
            )
            / "module.so"
        )
    )
    entry = library.uniserve_mxfp8_gemm
    entry.argtypes = [ctypes.POINTER(mx.GemmCall), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    address = 0x1000
    call = mx.GemmCall(
        input=address,
        input_scales=address,
        weight=address,
        weight_scales=address,
        out=address,
        m=4,
        n=1024,
        k=5120,
        groups=1,
        input_scale_stride=4,
        weight_scale_stride=1024,
        multiprocessors=148,
    )

    status = entry(ctypes.byref(call), None)

    message = call.error.decode(errors="replace")
    assert status != 0 and "[1792, 5120]" in message, message
