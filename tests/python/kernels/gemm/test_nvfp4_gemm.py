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
``Tolerance::rounding(2, f32, output)``: ``2γ₂(u_f32) + u_output``, the bound
for a reduction whose result is a scale of its inputs. A matrix product is not
one: its terms cancel, and ``fp4.rs``'s own documentation says such a
reduction declares ``Tolerance::dot``, ``2γ_n√n + u_out`` over the ``n``-term
contraction. A failure states how many elements the contraction's own bound
would also refuse, so a deviation that only the declared bound refuses reads
as a question for the operator's declaration rather than a verdict on the
module.
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

#: The unit roundoff of the output types the check states.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-9, "f32": 2.0**-24}
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


def dequantized(codes, scale_bytes, offsets, scale_shape):
    """The matrix a packed operand and its scale bytes stand for, in FP64.

    ``scale_bytes`` are the ``[rows, scales]`` logical scales, which the
    layout puts at ``offsets`` of a ``scale_shape`` buffer.
    """
    values = e2m1_values(codes)
    scales = torch.zeros(scale_shape[0] * scale_shape[1], dtype=torch.float64)
    scales[offsets.reshape(-1)] = decode_e4m3(scale_bytes).reshape(-1).double()
    matrix = scales.reshape(scale_shape)
    per_block = matrix[
        : values.shape[0], : values.shape[1] // nvfp4.BLOCK_ELEMENTS
    ]
    return values * per_block.repeat_interleave(nvfp4.BLOCK_ELEMENTS, dim=1)


def gamma(reduction, unit):
    """``γₙ = n·u / (1 − n·u)``, saturating where the expression stops being
    a bound.
    """  # noqa: D205
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


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

    a = dequantized(
        input_codes,
        input_scale_bytes,
        nvfp4.linear_offsets(m, scales),
        (m, scales),
    )
    b = dequantized(
        weight_codes,
        weight_scale_bytes,
        nvfp4.swizzled_offsets(n, scales),
        nvfp4.padded_shape(n, scales),
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
    input_scale_matrix = torch.zeros(
        input_shape[0] * input_shape[1], dtype=torch.uint8
    )
    weight_scale_matrix = torch.zeros(
        weight_shape[0] * weight_shape[1], dtype=torch.uint8
    )
    input_scale_matrix[input_offsets.reshape(-1)] = input_scale_bytes.reshape(
        -1
    )
    weight_scale_matrix[weight_offsets.reshape(-1)] = (
        weight_scale_bytes.reshape(-1)
    )
    packed_input = pack_e2m1(input_codes)
    packed_weight = pack_e2m1(weight_codes)
    input_scale_matrix = input_scale_matrix.to(torch_device)
    weight_scale_matrix = weight_scale_matrix.to(torch_device)
    out = torch.zeros(M, N, dtype=TORCH_TYPES[OUTPUT], device=torch_device)

    call = nvfp4.GemmCall(
        input=packed_input.data_ptr(),
        weight=packed_weight.data_ptr(),
        input_scales=input_scale_matrix.data_ptr(),
        weight_scales=weight_scale_matrix.data_ptr(),
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
    a = dequantized(
        input_codes.cpu(), input_scale_bytes, input_offsets, input_shape
    )
    b = dequantized(
        weight_codes.cpu(), weight_scale_bytes, weight_offsets, weight_shape
    )
    reference = ALPHA * (a @ b.T)
    error = (out.double().cpu() - reference).abs()
    magnitude = reference.abs()
    declared = 2.0 * gamma(2, UNIT_ROUNDOFF["f32"]) + UNIT_ROUNDOFF[OUTPUT]
    contraction = 2.0 * gamma(K, UNIT_ROUNDOFF["f32"]) * K**0.5
    contraction += UNIT_ROUNDOFF[OUTPUT]
    violations = int((~(error <= declared * magnitude)).sum())
    outside_contraction = int((~(error <= contraction * magnitude)).sum())
    worst = torch.nan_to_num(error).max().item()
    assert violations == 0, (
        f"{M}x{N} over k={K}, {OUTPUT}, scales {LAYOUT}: {violations} of "
        f"{M * N} elements outside the operator's declared rtol "
        f"{declared:.3e}, {outside_contraction} of them also outside the "
        f"contraction's own rtol {contraction:.3e}; max |error| {worst:.3e}"
    )
