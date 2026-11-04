#!/usr/bin/env python3
"""Run nvfp4_gemm through the built module and check the product it writes.

This is the GPU check the build host cannot do: it loads the module the store holds
(``module.so``), calls the operator's entry with the call structure of ``crates/ops/src/fp4.rs``
(592 bytes: ``input`` 0, ``weight`` 8, ``input_scales`` 16, ``weight_scales`` 24, ``alpha`` 32,
``bias`` 40, ``out`` 48, ``m`` 56, ``n`` 64, ``k`` 72, ``error`` 80), and compares the product
with the operator's own definition computed here in FP64 from the *packed bytes and scale bytes*
the call passes — not from the floats they were encoded from, so the check tests the module's
decoding as well as its contraction.

    python gemm_check.py <module_dir> [--m 128] [--n 256] [--k 2048]
                         [--output bf16] [--layout linear] [--seed 7] [--cpu-only]

``<module_dir>`` is what ``resolve`` hands a runner: the directory with ``module.so``. Run it on
the GPU host with the Program Server's own environment
(``crates/program-server/python/.venv/bin/python``), which has torch. Without a GPU,
``--cpu-only`` checks everything about this script that does not need a device: the call
structure's size and field offsets, the two scale layouts, and the reference arithmetic against
an independent element-by-element computation.

**The two scale layouts**, as ``ScaleLayout`` in ``fp4.rs`` defines them for a matrix of
``rows`` rows and ``columns`` elements (``scales = columns / 16``):

* ``linear`` — ``[rows, scales]`` row major: offset ``row · scales + scale``. The descriptor's
  ``layout`` names the *activation* scales' layout and says which product is meant.
* ``swizzled_128x4`` — rows padded to 128 and scale columns to 4, in the atom layout the
  Blackwell block-scaled MMA reads: the padded matrix, cut into blocks of 128 rows by 4 scale
  columns, the 128 rows split into four groups of 32, and within a block
  ``lane · 16 + group · 4 + column``. The weight scales are this layout always,
  whatever the descriptor's ``layout`` says. ``test``'s ``--cpu-only`` proves this formula is
  the atom layout ``cute`` builds (``BlockScaledBasicChunk``: shape ``((32, 4), (16, 4))``,
  strides ``((16, 4), (0, 1))``, tiled over the matrix), so a module reading its SF tensors with
  ``tile_atom_to_shape_SF`` and this script agree about where a scale byte lives.

**The tolerance** is the operator's: ``Tolerance::rounding(2, f32, output)`` — ``2γ₂(u_f32) +
u_output``, the bound for a reduction whose result is a *scale of its inputs*. A matrix product is
not one: its terms cancel, and ``fp4.rs``'s own documentation says such a reduction declares
``Tolerance::dot``, ``2γ_n√n + u_out`` over the ``n``-term contraction. This script judges
against the operator's declared bound and prints the ``dot`` bound beside it, so a deviation that
the declared bound refuses but the contraction's own bound allows is reported as what it is — a
question for the operator's declaration, not a verdict on the module.
"""

import argparse
import ctypes
import pathlib
import sys

import torch

#: E2M1's eight magnitudes, by the code's low three bits; the high bit is the sign.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)

#: Elements one E4M3 block scale covers (``fp4.rs``'s ``BLOCK_ELEMENTS``).
BLOCK_ELEMENTS = 16
#: The swizzled layout's row block, its four groups, and its scale columns.
SWIZZLE_ROWS = 128
SWIZZLE_GROUPS = 4
SWIZZLE_ROWS_PER_GROUP = 32
SWIZZLE_COLUMNS = 4

#: Unit roundoff of the dtypes a call can name (``DType::unit_roundoff``).
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-9, "f32": 2.0**-24}
TORCH_TYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


class Call(ctypes.Structure):
    """``uniserve_nvfp4_gemm_call``, field for field (``src/entry.h``)."""

    _fields_ = [
        ("input", ctypes.c_void_p),
        ("weight", ctypes.c_void_p),
        ("input_scales", ctypes.c_void_p),
        ("weight_scales", ctypes.c_void_p),
        ("alpha", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("m", ctypes.c_int64),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def linear_offsets(rows, scales):
    """Flat offsets of the ``linear`` layout: ``[rows, scales]`` row major."""
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    scale = torch.arange(scales, dtype=torch.int64)[None, :]
    return row * scales + scale


def swizzled_offsets(rows, scales):
    """Flat offsets of the 128x4 swizzle, for a ``[rows, scales]`` scale matrix.

    Rows are padded to a multiple of 128 and scale columns to a multiple of 4; a block of 128
    rows by 4 scale columns is ``lane · 16 + group · 4 + column`` within itself.
    """
    padded_rows = -(-rows // SWIZZLE_ROWS) * SWIZZLE_ROWS
    padded_scales = -(-scales // SWIZZLE_COLUMNS) * SWIZZLE_COLUMNS
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    scale = torch.arange(scales, dtype=torch.int64)[None, :]
    block_row = row // SWIZZLE_ROWS
    within = row % SWIZZLE_ROWS
    group = within // SWIZZLE_ROWS_PER_GROUP
    lane = within % SWIZZLE_ROWS_PER_GROUP
    return (
        block_row * (padded_scales * SWIZZLE_ROWS)
        + (scale // SWIZZLE_COLUMNS) * (SWIZZLE_ROWS * SWIZZLE_COLUMNS)
        + lane * (SWIZZLE_GROUPS * SWIZZLE_COLUMNS)
        + group * SWIZZLE_COLUMNS
        + (scale % SWIZZLE_COLUMNS)
    )


def padded_shape(rows, scales):
    """The shape a swizzled scale matrix occupies: rows to 128, scale columns to 4."""
    return (
        -(-rows // SWIZZLE_ROWS) * SWIZZLE_ROWS,
        -(-scales // SWIZZLE_COLUMNS) * SWIZZLE_COLUMNS,
    )


def e2m1_values(codes):
    """Decode a matrix of E2M1 codes into their values, in FP64 (they are exact there)."""
    magnitude = E2M1[codes.long() & 0x7]
    sign = torch.where((codes.long() & 0x8) != 0, -1.0, 1.0).to(torch.float64)
    return magnitude * sign


def pack_e2m1(codes):
    """Pack a ``[rows, columns]`` E2M1 code matrix into bytes, the lower column in the low nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def dequantized(codes, scale_bytes, scales_expanded_offsets, scale_shape):
    """The matrix a packed operand and its scale bytes stand for, in FP64."""
    values = e2m1_values(codes)
    scales = torch.zeros(scale_shape[0] * scale_shape[1], dtype=torch.float64)
    scales[scales_expanded_offsets.reshape(-1)] = decode_e4m3(scale_bytes).reshape(-1).double()
    matrix = scales.reshape(scale_shape[0], scale_shape[1])
    per_block = matrix[: values.shape[0], : values.shape[1] // BLOCK_ELEMENTS]
    return values * per_block.repeat_interleave(BLOCK_ELEMENTS, dim=1)


def decode_e4m3(bytes_):
    """Decode a tensor of E4M3 (FN) bytes into FP32."""
    return bytes_.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)


def encode_e4m3(values):
    """Encode FP32 values as E4M3 bytes (torch's cast is round-to-nearest-even, as FN is)."""
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def gamma(reduction, unit):
    """``γₙ = n·u / (1 − n·u)``, saturating where the expression stops being a bound."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def declared_bound(m, n, k, output):
    """``Tolerance::rounding(2, f32, output)``'s ``rtol``, as ``fp4.rs`` declares it."""
    return 2.0 * gamma(2, UNIT_ROUNDOFF["f32"]) + UNIT_ROUNDOFF[output]


def dot_bound(m, n, k, output):
    """``Tolerance::dot(k, f32, output)``'s ``rtol``, the bound a contraction's own doc states."""
    return 2.0 * gamma(k, UNIT_ROUNDOFF["f32"]) * k**0.5 + UNIT_ROUNDOFF[output]


def check_call_structure():
    """The call structure this script packs is the one ``src/entry.h`` static-asserts."""
    if ctypes.sizeof(Call) != 592:
        raise SystemExit(f"Call is {ctypes.sizeof(Call)} bytes, entry.h asserts 592")
    if Call.error.offset != 80:
        raise SystemExit(f"Call.error is at {Call.error.offset}, entry.h asserts 80")
    for name, offset in (("out", 48), ("m", 56), ("k", 72)):
        field = getattr(Call, name)
        if field.offset != offset:
            raise SystemExit(f"Call.{name} is at {field.offset}, expected {offset}")
    print("call structure: 592 bytes, error at 80 — as entry.h asserts")


def check_swizzle_is_the_atom_layout():
    """The swizzle this script writes is the atom layout ``cute`` builds for the scale factors.

    ``BlockScaledBasicChunk.layout`` is ``((32, 4), (16, 4))`` with strides ``((16, 4), (0, 1))``
    — the MN axis as 32 lanes by 4 groups, the K axis as 16 elements sharing one scale (stride 0)
    by 4 scale columns. Tiled over an ``[rows, k]`` matrix (``tile_to_shape`` order ``(2, 1)``)
    it is: ``(row % 32)·16 + ((row // 32) % 4)·4 + (scale % 4) + (scale // 4)·512 +
    (row // 128)·512·(scales // 4)``.
    """
    for rows, columns in ((128, 2048), (256, 2048), (129, 64), (3, 4096)):
        scales = columns // BLOCK_ELEMENTS
        padded_rows, padded_scales = padded_shape(rows, scales)
        row = torch.arange(rows, dtype=torch.int64)[:, None]
        scale = torch.arange(scales, dtype=torch.int64)[None, :]
        atom = (
            (row % 32) * 16
            + ((row // 32) % 4) * 4
            + (scale % 4)
            + (scale // 4) * (SWIZZLE_ROWS * SWIZZLE_COLUMNS)
            + (row // SWIZZLE_ROWS) * (SWIZZLE_ROWS * SWIZZLE_COLUMNS * padded_scales // SWIZZLE_COLUMNS)
        )
        mine = swizzled_offsets(rows, scales)
        if not torch.equal(atom, mine):
            raise SystemExit(f"swizzle disagrees with the atom layout at [{rows}, {scales}]")
        flat = mine.reshape(-1)
        if flat.unique().numel() != flat.numel():
            raise SystemExit(f"the swizzle at [{rows}, {scales}] maps two scales to one byte")
        if int(flat.max()) >= padded_rows * padded_scales or int(flat.min()) < 0:
            raise SystemExit(f"the swizzle at [{rows}, {scales}] leaves its padded extent")
        if rows % SWIZZLE_ROWS == 0 and flat.unique().numel() != padded_rows * padded_scales:
            raise SystemExit(f"the swizzle at [{rows}, {scales}] does not fill its padded extent")
    print("scale layouts: the 128x4 swizzle is cute's BlockScaledBasicChunk atom, 4 shapes")


def check_reference_by_hand():
    """The reference agrees with a direct sum written here for a case small enough to read."""
    torch.manual_seed(11)
    m, n, k = 4, 8, 32
    scales = k // BLOCK_ELEMENTS
    input_codes = torch.randint(0, 16, (m, k), dtype=torch.uint8)
    weight_codes = torch.randint(0, 16, (n, k), dtype=torch.uint8)
    input_scale_bytes = torch.full((m, scales), 64, dtype=torch.uint8)  # E4M3 0x40 = 2.0
    weight_scale_bytes = torch.full((n, scales), 56, dtype=torch.uint8)  # E4M3 0x38 = 1.0
    alpha = 0.6875

    input_offsets = linear_offsets(m, scales)
    weight_offsets = swizzled_offsets(n, scales)
    a = dequantized(input_codes, input_scale_bytes, input_offsets, (m, scales))
    b = dequantized(weight_codes, weight_scale_bytes, weight_offsets, padded_shape(n, scales))

    product = alpha * (a @ b.T)
    for row in range(m):
        for column in range(n):
            direct = 0.0
            for element in range(k):
                direct += float(a[row, element]) * float(b[column, element])
            direct *= alpha
            if abs(direct - float(product[row, column])) > 1e-9 * max(1.0, abs(direct)):
                raise SystemExit(
                    f"reference disagrees with a direct sum at [{row}, {column}]: "
                    f"{float(product[row, column])} vs {direct}"
                )
    # A scale byte is not the matrix's value: 2.0 and 1.0 above must reach the sum.
    if float(a.max()) != 12.0:
        raise SystemExit(f"dequantization did not apply the block scale: max {float(a.max())}")
    print("reference: matches a direct sum over the packed codes, block scales applied")


def check_entry_without_a_device(module_dir):
    """The module loads and the entry refuses a call for another problem size.

    ``src/entry.cpp`` checks the pointers and then the call's three extents against the module's
    compiled problem **before** it touches a device, so a call naming the wrong extents is
    answered on the host. That is an ABI test with no GPU: it proves the struct this script packs
    is the struct the entry reads, and that its error buffer is where ``entry.h`` says.
    """
    library = ctypes.CDLL(str(module_dir / "module.so"))
    entry = library.uniserve_nvfp4_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    call = Call(
        input=0x1000,
        weight=0x1000,
        input_scales=0x1000,
        weight_scales=0x1000,
        alpha=0x1000,
        bias=0x1000,
        out=0x1000,
        m=0,
        n=0,
        k=0,
    )
    status = entry(ctypes.byref(call), None)
    message = call.error.decode(errors="replace")
    if status == 0:
        raise SystemExit("the entry accepted a call for another problem size")
    if "another problem size" not in message:
        raise SystemExit(f"the entry refused with an unexpected message: {message!r}")
    print(f"entry: loads and refuses another problem size — {message}")


def run(module_dir, m, n, k, output, layout, seed, bias):
    """Call the module's entry and compare its product with the reference."""
    library = ctypes.CDLL(str(module_dir / "module.so"))
    entry = library.uniserve_nvfp4_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    torch.manual_seed(seed)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    scales = k // BLOCK_ELEMENTS

    input_codes = torch.randint(0, 16, (m, k), dtype=torch.uint8, device=device)
    weight_codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=device)
    # Scale magnitudes within a factor of four of each other: a calibrated checkpoint's blocks
    # are, and a bound that only holds for one scale magnitude would not be the operator's.
    input_scale = (1.0 + torch.rand(m, scales, device=device)).to("cpu").float()
    weight_scale = (0.5 + torch.rand(n, scales, device=device)).to("cpu").float()
    input_scale_bytes = encode_e4m3(input_scale).to(device)
    weight_scale_bytes = encode_e4m3(weight_scale).to(device)
    alpha = torch.tensor([0.6875], dtype=torch.float32, device=device)

    packed_input = pack_e2m1(input_codes)
    packed_weight = pack_e2m1(weight_codes)
    # The swizzled scale matrices are kept at their padded shape, as the operator's
    # `ScaleLayout::shape` says the call passes them.
    input_layout_shape = (m, scales) if layout == "linear" else padded_shape(m, scales)
    weight_layout_shape = padded_shape(n, scales)
    input_scale_matrix = torch.zeros(
        input_layout_shape[0] * input_layout_shape[1], dtype=torch.uint8, device=device
    )
    weight_scale_matrix = torch.zeros(
        weight_layout_shape[0] * weight_layout_shape[1], dtype=torch.uint8, device=device
    )
    input_offsets = (
        linear_offsets(m, scales).to(device)
        if layout == "linear"
        else swizzled_offsets(m, scales).to(device)
    )
    weight_offsets = swizzled_offsets(n, scales).to(device)
    input_scale_matrix[input_offsets.reshape(-1)] = input_scale_bytes.reshape(-1)
    weight_scale_matrix[weight_offsets.reshape(-1)] = weight_scale_bytes.reshape(-1)

    out = torch.zeros(m, n, dtype=TORCH_TYPES[output], device=device)
    bias_tensor = None
    if bias:
        bias_tensor = (torch.rand(n, device=device) - 0.5).to(device)

    call = Call(
        input=packed_input.data_ptr(),
        weight=packed_weight.data_ptr(),
        input_scales=input_scale_matrix.data_ptr(),
        weight_scales=weight_scale_matrix.data_ptr(),
        alpha=alpha.data_ptr(),
        bias=bias_tensor.data_ptr() if bias_tensor is not None else None,
        out=out.data_ptr(),
        m=m,
        n=n,
        k=k,
    )
    status = entry(ctypes.byref(call), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if status != 0:
        raise SystemExit(f"the module refused: {call.error.decode(errors='replace')}")

    # The reference reads the *bytes the call passed*, on the host, in FP64.
    a = dequantized(
        input_codes.to("cpu"),
        input_scale_bytes.to("cpu"),
        input_offsets.to("cpu"),
        input_layout_shape,
    )
    b = dequantized(
        weight_codes.to("cpu"),
        weight_scale_bytes.to("cpu"),
        weight_offsets.to("cpu"),
        weight_layout_shape,
    )
    reference = float(alpha[0].item()) * (a @ b.T)
    if bias_tensor is not None:
        reference = reference + bias_tensor.double().unsqueeze(0).expand(m, n)
    candidate = out.double().to("cpu")

    error = (candidate - reference).abs()
    magnitude = reference.abs()
    declared = declared_bound(m, n, k, output)
    contraction = dot_bound(m, n, k, output)
    allowed = declared * magnitude
    violations = int((~(error <= allowed)).sum())
    worst = torch.nan_to_num(error).max().item()
    worst_at = int(error.reshape(-1).argmax())
    cancellation = (a.abs() @ b.T.abs()).max().item() / max(magnitude.max().item(), 1e-30)

    print(
        f"{m}x{n} over k={k}, {output}, scales {layout}"
        f"{', bias' if bias else ''}: max |error| {worst:.3e} "
        f"(reference {magnitude.reshape(-1)[worst_at].item():.3e}), "
        f"declared rtol {declared:.3e}, dot rtol {contraction:.3e}"
    )
    print(f"  worst cancellation Σ|ab| / |out|: {cancellation:.1f}")
    if violations == 0:
        print(f"ok: every element within the operator's declared bound (0 of {m * n})")
        return 0
    outside_contraction = int((error > contraction * magnitude).sum())
    print(
        f"FAIL: {violations} of {m * n} elements outside the operator's declared bound; "
        f"{outside_contraction} of them also outside the contraction's own bound"
    )
    if outside_contraction == 0:
        print(
            "  The deviation is within Tolerance::dot's bound, so the module's accumulation "
            "order may be right while fp4.rs declares the wrong bound for a matrix product "
            "(its own documentation says a mixed-sign reduction declares dot)."
        )
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", nargs="?", type=pathlib.Path, help="the directory with module.so")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--output", default="bf16", choices=sorted(TORCH_TYPES))
    parser.add_argument("--layout", default="linear", choices=("linear", "swizzled_128x4"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bias", action="store_true", help="an epilogue that adds a bias")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="check the call structure, the scale layouts and the reference, without a device",
    )
    options = parser.parse_args()

    check_call_structure()
    check_swizzle_is_the_atom_layout()
    check_reference_by_hand()
    if options.cpu_only:
        if options.module is not None and (options.module / "module.so").is_file():
            check_entry_without_a_device(options.module)
        print("cpu-only: the parts of this script that do not need a device are consistent")
        return 0
    if options.module is None:
        parser.error("a module directory is needed unless --cpu-only is given")
    if not (options.module / "module.so").is_file():
        parser.error(f"{options.module} holds no module.so")
    return run(
        options.module,
        options.m,
        options.n,
        options.k,
        options.output,
        options.layout,
        options.seed,
        options.bias,
    )


if __name__ == "__main__":
    sys.exit(main())
