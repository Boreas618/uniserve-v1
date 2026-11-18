"""The `nvfp4_gemm` module's product against the operator's definition.

The module (``module.so``, resolved at the session's Build Server for the
session's device) is called with the call structure of ``crates/ops/src/fp4.rs``
(592 bytes: ``input`` 0, ``weight`` 8, ``input_scales`` 16, ``weight_scales``
24, ``alpha`` 32, ``bias`` 40, ``out`` 48, ``m`` 56, ``n`` 64, ``k`` 72,
``error`` 80), and its product is compared with the operator's definition
computed in FP64 from the packed bytes and scale bytes the call passes, not
from the floats they were encoded from, so the check tests the module's
decoding as well as its contraction. The descriptor's ``layout`` names the
activation scales' layout; the weight scales are swizzled whatever it says.

Without a device the check proves what the comparison rests on: the call
structure's size and offsets, that the 128x4 swizzle is the atom layout
``cute`` builds (``BlockScaledBasicChunk``: shape ``((32, 4), (16, 4))``,
strides ``((16, 4), (0, 1))``, tiled over the matrix), so a module reading its
scale tensors with ``tile_atom_to_shape_SF`` and the reference agree about
where a scale byte lives, and the reference against a direct sum.

**The tolerance** is the operator's declared one,
``Tolerance::rounding(k, f32, output)``: ``2γ_k(u_f32) + u_output`` over the
``k``-term FP32 contraction, with the output's subnormal unit as the absolute
part (:func:`uniserve_testing.kernels.gemm.nvfp4.gemm_tolerance`).
"""

import ctypes

import pytest
import torch
from uniserve_testing.kernels.gemm import nvfp4
from uniserve_testing.kernels.packages import assert_abi_layout

pytestmark = pytest.mark.unit

#: E2M1's values by code, the sign in bit 3, in FP64 (they are exact there).
E2M1_VALUES = torch.tensor(
    [*nvfp4.E2M1, *(-value for value in nvfp4.E2M1)], dtype=torch.float64
)

#: The torch element type of each output type the check states.
TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The product's factor.
ALPHA = 0.6875


def e2m1_values(codes):
    """A matrix of E2M1 codes as their values, in FP64."""
    return E2M1_VALUES[codes.long()]


def pack_e2m1(codes):
    """A ``[rows, columns]`` E2M1 code matrix as bytes, the lower column in
    the low nibble.
    """  # noqa: D205
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def decode_e4m3(bytes_):
    """E4M3 (FN) bytes as FP32."""
    return bytes_.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)


def encode_e4m3(values):
    """FP32 values as E4M3 bytes; torch's cast rounds to nearest even, as FN
    does.
    """  # noqa: D205
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def dequantized(codes, scale_buffer, offsets):
    """The matrix a packed operand and its scale buffer stand for, in FP64.

    ``scale_buffer`` is the flat scale buffer the call passes and ``offsets``
    the flat offset of every ``[row, scale]`` in it, so each block reads the
    byte its layout puts there.
    """
    values = e2m1_values(codes)
    per_block = decode_e4m3(scale_buffer[offsets]).double()
    return values * per_block.repeat_interleave(nvfp4.BLOCK_ELEMENTS, dim=1)


def scale_buffer(scale_bytes, offsets, shape):
    """The flat ``shape`` buffer that holds the logical ``[rows, scales]``
    bytes ``scale_bytes`` at ``offsets``; the padding is zero.
    """  # noqa: D205
    buffer = torch.zeros(shape[0] * shape[1], dtype=torch.uint8)
    buffer[offsets.reshape(-1)] = scale_bytes.reshape(-1)
    return buffer


def test_the_call_mirror_is_the_entrys_call_structure():
    """The generated header asserts the size and every offset it holds."""
    assert_abi_layout(nvfp4.GemmCall, "uniserve_nvfp4_gemm_call")


@pytest.mark.parametrize(
    ("rows", "columns"), [(128, 2048), (256, 2048), (129, 64), (3, 4096)]
)
def test_the_swizzle_is_cutes_scale_factor_atom(rows, columns):
    """Tiled over an ``[rows, k]`` matrix (``tile_to_shape`` order ``(2, 1)``)
    the atom is ``(row % 32)·16 + ((row // 32) % 4)·4 + (scale % 4) +
    (scale // 4)·512 + (row // 128)·512·(scales // 4)``.
    """  # noqa: D205
    scales = columns // nvfp4.BLOCK_ELEMENTS
    padded_rows, padded_scales = nvfp4.padded_shape(rows, scales)
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    scale = torch.arange(scales, dtype=torch.int64)[None, :]
    block = nvfp4.SWIZZLE_ROWS * nvfp4.SWIZZLE_COLUMNS
    atom = (
        (row % 32) * 16
        + ((row // 32) % 4) * 4
        + (scale % 4)
        + (scale // 4) * block
        + (row // nvfp4.SWIZZLE_ROWS)
        * (block * padded_scales // nvfp4.SWIZZLE_COLUMNS)
    )
    mine = nvfp4.swizzled_offsets(rows, scales)
    flat = mine.reshape(-1)

    assert torch.equal(atom, mine), (
        f"the swizzle disagrees with the atom layout at [{rows}, {scales}]"
    )
    assert flat.unique().numel() == flat.numel(), (
        f"the swizzle at [{rows}, {scales}] maps two scales to one byte"
    )
    assert (
        0 <= int(flat.min()) and int(flat.max()) < padded_rows * padded_scales
    ), f"the swizzle at [{rows}, {scales}] leaves its padded extent"
    if rows % nvfp4.SWIZZLE_ROWS == 0:
        assert flat.unique().numel() == padded_rows * padded_scales, (
            f"the swizzle at [{rows}, {scales}] does not fill its padded extent"
        )


def test_the_reference_is_a_direct_sum_over_the_packed_codes():
    torch.manual_seed(11)
    m, n, k = 4, 8, 32
    scales = k // nvfp4.BLOCK_ELEMENTS
    input_codes = torch.randint(0, 16, (m, k), dtype=torch.uint8)
    weight_codes = torch.randint(0, 16, (n, k), dtype=torch.uint8)
    # E4M3 0x40 is 2.0 and 0x38 is 1.0.
    input_scale_bytes = torch.full((m, scales), 0x40, dtype=torch.uint8)
    weight_scale_bytes = torch.full((n, scales), 0x38, dtype=torch.uint8)

    input_offsets = nvfp4.linear_offsets(m, scales)
    weight_offsets = nvfp4.swizzled_offsets(n, scales)
    a = dequantized(
        input_codes,
        scale_buffer(input_scale_bytes, input_offsets, (m, scales)),
        input_offsets,
    )
    b = dequantized(
        weight_codes,
        scale_buffer(
            weight_scale_bytes,
            weight_offsets,
            nvfp4.padded_shape(n, scales),
        ),
        weight_offsets,
    )
    product = ALPHA * (a @ b.T)

    for row in range(m):
        for column in range(n):
            direct = ALPHA * sum(
                float(a[row, element]) * float(b[column, element])
                for element in range(k)
            )
            assert abs(direct - float(product[row, column])) <= 1e-9 * max(
                1.0, abs(direct)
            ), (
                f"the reference disagrees with a direct sum at [{row}, "
                f"{column}]: {float(product[row, column])} vs {direct}"
            )
    # A scale byte is not the matrix's value: the 2.0 must reach the sum.
    assert float(a.max()) == 12.0, (
        f"the dequantization did not apply the block scale: max "
        f"{float(a.max())}"
    )


#: The module the device tests load: the extents, the output type and the
#: activation scales' layout of its descriptor, the one the package declares.
M, N, K, OUTPUT, LAYOUT = 128, 256, 2048, "bf16", "swizzled_128x4"


@pytest.fixture
def entry(kernel_module):
    """The module's C entry."""
    key = (
        f"packed_u8_scale_f8e4m3_accumulate_f32_output_{OUTPUT}_layout_{LAYOUT}"
        f"_epilogue_none_m_{M}_n_{N}_k_{K}"
    )
    library = ctypes.CDLL(
        str(kernel_module("nvfp4_gemm", "cute_dsl", key) / "module.so")
    )
    function = library.uniserve_nvfp4_gemm
    function.argtypes = [ctypes.POINTER(nvfp4.GemmCall), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


@pytest.mark.gpu
def test_the_entry_refuses_another_problem_size_before_the_device(entry):
    """The entry checks the pointers and then the call's three extents against
    the module's compiled problem before it touches a device, so a call naming
    other extents is answered on the host: the structure packed here is the
    one the entry reads, and its error buffer is where the generated header
    says.
    """  # noqa: D205
    address = 0x1000
    call = nvfp4.GemmCall(
        input=address,
        weight=address,
        input_scales=address,
        weight_scales=address,
        alpha=address,
        bias=address,
        out=address,
        m=0,
        n=0,
        k=0,
    )

    status = entry(ctypes.byref(call), None)

    message = call.error.decode(errors="replace")
    assert status != 0, "the entry accepted a call for another problem size"
    assert "another problem size" in message, (
        f"the entry refused with an unexpected message: {message!r}"
    )


@pytest.mark.gpu
def test_the_product_is_within_the_operators_declared_bound(
    entry, torch_device
):
    torch.manual_seed(7)
    scales = K // nvfp4.BLOCK_ELEMENTS
    input_codes = torch.randint(
        0, 16, (M, K), dtype=torch.uint8, device=torch_device
    )
    weight_codes = torch.randint(
        0, 16, (N, K), dtype=torch.uint8, device=torch_device
    )
    # Scale magnitudes within a factor of four of each other: a calibrated
    # checkpoint's blocks are, and a bound that only holds for one scale
    # magnitude would not be the operator's.
    input_scale = (1.0 + torch.rand(M, scales)).float()
    weight_scale = (0.5 + torch.rand(N, scales)).float()
    input_scale_bytes = encode_e4m3(input_scale)
    weight_scale_bytes = encode_e4m3(weight_scale)
    alpha = torch.tensor([ALPHA], dtype=torch.float32, device=torch_device)

    # The scale matrices are kept at the padded shape the operator's
    # `ScaleLayout::shape` says the call passes a swizzled one.
    input_shape = nvfp4.padded_shape(M, scales)
    input_offsets = nvfp4.swizzled_offsets(M, scales)
    weight_shape = nvfp4.padded_shape(N, scales)
    weight_offsets = nvfp4.swizzled_offsets(N, scales)
    input_scale_matrix = scale_buffer(
        input_scale_bytes, input_offsets, input_shape
    )
    weight_scale_matrix = scale_buffer(
        weight_scale_bytes, weight_offsets, weight_shape
    )
    packed_input = pack_e2m1(input_codes)
    packed_weight = pack_e2m1(weight_codes)
    input_scales = input_scale_matrix.to(torch_device)
    weight_scales = weight_scale_matrix.to(torch_device)
    out = torch.zeros(M, N, dtype=TORCH_TYPES[OUTPUT], device=torch_device)

    call = nvfp4.GemmCall(
        input=packed_input.data_ptr(),
        weight=packed_weight.data_ptr(),
        input_scales=input_scales.data_ptr(),
        weight_scales=weight_scales.data_ptr(),
        alpha=alpha.data_ptr(),
        bias=None,
        out=out.data_ptr(),
        m=M,
        n=N,
        k=K,
    )
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the module refused: {call.error.decode(errors='replace')}"
    )

    # The reference reads the bytes the call passed, on the host, in FP64.
    a = dequantized(input_codes.cpu(), input_scale_matrix, input_offsets)
    b = dequantized(weight_codes.cpu(), weight_scale_matrix, weight_offsets)
    reference = ALPHA * (a @ b.T)
    error = (out.double().cpu() - reference).abs()
    atol, rtol = nvfp4.gemm_tolerance(K, OUTPUT)
    # `~(error <= bound)`: the comparison must fail on a NaN.
    violations = int((~(error <= atol + rtol * reference.abs())).sum())
    worst = torch.nan_to_num(error).max().item()
    assert violations == 0, (
        f"{M}x{N} over k={K}, {OUTPUT}, scales {LAYOUT}: {violations} of "
        f"{M * N} elements outside the operator's declared rtol {rtol:.3e}; "
        f"max |error| {worst:.3e}"
    )
