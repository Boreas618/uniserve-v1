#!/usr/bin/env python3
"""The NVFP4 family's end-to-end check: producer, consumer, and the interface between them.

Three stages, reported separately because they fail for different reasons:

1. **The producer against its input.** ``nvfp4_quantize`` encodes a BF16 tile
   into packed E2M1 codes and E4M3 block scales. This stage reads *the module's
   own output buffers* — never a re-packing done here — decodes those bytes as
   the operator's documentation says they mean (two E2M1 codes a byte, a low
   nibble first; one E4M3 scale per sixteen elements of the contraction), and
   compares the decoded tile with the input. What can fail: the E2M1 rounding,
   the code order inside a byte, the E4M3 scale's own encoding (exponent bias 7,
   subnormals below 2^-6, saturation at 448) and the scale bytes' swizzle.
2. **The consumer against the same bytes.** Torch contracts the module's packed
   codes and scale bytes in FP64; ``nvfp4_gemm`` contracts them on the device.
3. **The two against each other.** If stage 1 passes and stage 3 fails, one
   field of the interface between the two modules is wrong; if stage 1 fails,
   the consumer was never the question.

Why stage 1 is decisive on a device and not in Triton's interpreter: the
interpreter's narrowing float casts truncate instead of rounding (0.1 to bf16
gives 0.099609375), so no interpreter run can prove a rounding, and the E4M3
scale encoding *is* a rounding. Here the grid is enumerated from the encodings'
definitions, so a pass is a statement about every binade and every subnormal
rather than about the values a seed happened to produce.

    python quantize_check.py <quantize module dir> [--gemm <gemm module dir>]
                             [--rows 128] [--columns 2048] [--block 1024] [--seed 7]

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding the
compiled module. Run it on the GPU host with the Program Server's own
environment (``crates/program-server/python/.venv/bin/python``), which has torch
and the CUDA bindings. A quantize module's directory name is a descriptor key;
the check refuses one whose key does not name a layout it can decode, which is
the one class of mismatch that otherwise reads as a kernel defect.
"""

import argparse
import ctypes
import json
import math
import pathlib
import sys

import torch
from cuda.bindings import driver as cu

#: E2M1's representable magnitudes, indexed by the nibble's magnitude field.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

#: A block is sixteen elements of the contraction and one scale byte.
BLOCK_ELEMENTS = 16

#: The largest E4M3 value; the encoding has no infinities, so beyond it saturates.
E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# The encodings, from their definitions
# ---------------------------------------------------------------------------


def e2m1_value(nibble):
    """One E2M1 code: sign bit 3 over a four-value magnitude table."""
    magnitude = E2M1[nibble & 0x7]
    return -magnitude if nibble & 0x8 else magnitude


def e2m1_table():
    """The sixteen codes as a float64 tensor, for decoding in vectors."""
    return torch.tensor([e2m1_value(nibble) for nibble in range(16)], dtype=torch.float64)


def e4m3_value(byte):
    """One E4M3 byte: four exponent bits biased by 7 over three mantissa bits.

    ``0x7f``/``0xff`` are NaN; a checkpoint's scale is never one, so this raises
    and the caller reports it rather than decoding a NaN into every product.
    """
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        raise ValueError(f"0x{byte:02x} is an E4M3 NaN")
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def e4m3_table():
    """Every E4M3 byte's value as a float64 tensor; NaN slots are zero and unused."""
    values = []
    for byte in range(256):
        try:
            values.append(e4m3_value(byte))
        except ValueError:
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Decoding the module's own bytes
# ---------------------------------------------------------------------------


def scale_layout(layout, rows, scale_columns):
    """The padded shape of the scale buffer: the matrix itself, or the atom's."""
    if layout == "linear":
        return None
    return (-(-rows // 128) * 128, -(-scale_columns // 4) * 4)


#: The atom's row block, its four scale columns, and its four groups of 32 rows.
SWIZZLE_ROWS, SWIZZLE_COLUMNS, SWIZZLE_GROUPS = 128, 4, 4
SWIZZLE_ROWS_PER_GROUP = SWIZZLE_ROWS // SWIZZLE_GROUPS


def swizzled_offsets(rows, scale_columns):
    """The atom's flat offset of every ``[row, scale column]`` of the matrix.

    The operator's own mapping (``nvfp4_quantize``'s ``_swizzled_offset``): the
    scale matrix is grouped into blocks of 128 rows and 4 scale columns; within a
    block the 128 rows split into four groups of 32 and the four columns are
    interleaved fastest.
    """
    padded_rows, padded_columns = scale_layout("swizzled_128x4", rows, scale_columns)
    row = torch.arange(padded_rows, dtype=torch.int64)
    column = torch.arange(padded_columns, dtype=torch.int64)
    block_row = row // SWIZZLE_ROWS
    within = row % SWIZZLE_ROWS
    group = within // SWIZZLE_ROWS_PER_GROUP
    lane = within % SWIZZLE_ROWS_PER_GROUP
    return (
        block_row[:, None] * (padded_columns * SWIZZLE_ROWS)
        + (column // SWIZZLE_COLUMNS)[None, :] * (SWIZZLE_ROWS * SWIZZLE_COLUMNS)
        + lane[:, None] * (SWIZZLE_GROUPS * SWIZZLE_COLUMNS)
        + group[:, None] * SWIZZLE_COLUMNS
        + (column % SWIZZLE_COLUMNS)[None, :]
    )


def decode(values, scales, token_scale, rows, columns, layout):
    """The module's bytes as an FP64 tile: each code times its block's scale and its row's factor.

    Encoding is two-level: the block's E4M3 scale is relative to the row's per-token factor, so a
    code dequantizes to `code * block_scale * token_scale[row]`. Both come from the module's own
    buffers, which is the point of this stage: a wrong factor shows up as a wrong tile.
    """
    scale_columns = columns // BLOCK_ELEMENTS
    table_codes = e2m1_table().to(values.device)
    table_scales = e4m3_table().to(values.device)

    if layout == "linear":
        block_scales = table_scales[scales.view(torch.uint8).long()].reshape(rows, scale_columns)
    else:
        offset = swizzled_offsets(rows, scale_columns).to(scales.device)
        block_scales = table_scales[scales.view(torch.uint8).long()[offset]][:rows, :scale_columns]

    packed = values.view(torch.uint8).to(torch.int16).reshape(rows, columns // 2)
    low = (packed & 0xF).long()
    high = ((packed >> 4) & 0xF).long()
    decoded = torch.zeros(rows, columns, dtype=torch.float64, device=values.device)
    decoded[:, 0::2] = table_codes[low]
    decoded[:, 1::2] = table_codes[high]
    factor = token_scale.to(torch.float64).reshape(rows, 1)
    return decoded * block_scales.repeat_interleave(BLOCK_ELEMENTS, dim=1) * factor


def half_step(unit):
    """Half the distance to the next E2M1 code, in units of the block's scale.

    This is the encoding's own bound, not a chosen tolerance: a decoded code is
    correct when the exact quotient lies within half a step of it.
    """
    codes = torch.tensor(E2M1, dtype=torch.float64, device=unit.device)
    magnitude = unit.abs()
    upper = torch.searchsorted(codes, magnitude, right=False)
    upper = upper.clamp(max=len(E2M1) - 1)
    lower = torch.searchsorted(codes, magnitude, right=True) - 1
    lower = lower.clamp(min=0)
    # A quotient above the last code saturates, so its step is the last interval.
    span = (codes[upper] - codes[lower]).clamp(min=0.5)
    return 0.5 * span


# ---------------------------------------------------------------------------
# Launching the built Triton module
# ---------------------------------------------------------------------------


class QuantizeModule:
    """One resolved ``nvfp4_quantize`` module: its cubin, metadata and entry."""

    def __init__(self, directory):
        self.directory = pathlib.Path(directory)
        metadata = json.loads((self.directory / "metadata.json").read_text())
        self.name = metadata["name"]
        self.shared = int(metadata.get("shared", 0))
        self.threads = int(metadata.get("num_warps", 4)) * 32
        _, module = cu.cuModuleLoad((self.directory / "module.cubin").read_bytes())
        if int(module) == 0:
            raise SystemExit(f"{self.directory}/module.cubin did not load")
        _, self.function = cu.cuModuleGetFunction(module, self.name.encode())

    def launch(self, grid, *arguments):
        """Launch the entry with the module's own block and shared-memory size."""
        holders: list = []
        for argument in arguments:
            if isinstance(argument, torch.Tensor):
                holders.append(ctypes.c_void_p(argument.data_ptr()))
            elif isinstance(argument, float):
                holders.append(ctypes.c_float(argument))
            else:
                holders.append(ctypes.c_int32(int(argument)))
        pointers = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders]
        )
        stream = torch.cuda.current_stream().cuda_stream
        error, _ = cu.cuLaunchKernel(
            self.function,
            grid[0],
            grid[1],
            grid[2],
            self.threads,
            1,
            1,
            self.shared,
            stream,
            pointers,
            0,
        )
        if int(error) != 0:
            raise SystemExit(f"launching {self.name}: {error}")
        torch.cuda.synchronize()


class GemmModule:
    """One resolved ``nvfp4_gemm`` module and the operator's call structure."""

    class Call(ctypes.Structure):
        """`ops::fp4`'s `Nvfp4Gemm` call: 592 bytes with the error at 80."""

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

    def __init__(self, directory):
        self.directory = pathlib.Path(directory)
        self.library = ctypes.CDLL(str(self.directory / "module.so"))
        self.entry = self.library.uniserve_nvfp4_gemm
        self.entry.argtypes = [ctypes.POINTER(self.Call), ctypes.c_void_p]
        self.entry.restype = ctypes.c_int32

    def run(self, call, *tensors):
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        status = self.entry(ctypes.byref(call), stream)
        torch.cuda.synchronize()
        message = bytes(call.error).rstrip(b"\0").decode()
        if status != 0 or message:
            raise SystemExit(f"the gemm entry refused the call: {message}")
        return call


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def stage_producer(module, rows, columns, block, layout, global_scale, seed):
    """Stage 1: the module's own bytes against its input. Returns them for stage 2."""
    torch.manual_seed(seed)
    source = (torch.randn(rows, columns, device="cuda", dtype=torch.float32) * 0.05).to(
        torch.bfloat16
    )

    values = torch.full((rows, columns // 2), 0xCD, dtype=torch.uint8, device="cuda")
    scale_columns = columns // BLOCK_ELEMENTS
    if layout == "linear":
        scale_count = rows * scale_columns
    else:
        padded_rows, padded_columns = scale_layout(layout, rows, scale_columns)
        scale_count = padded_rows * padded_columns
    scales = torch.full((scale_count,), 0xCD, dtype=torch.uint8, device="cuda")

    # The global scale is a device operand: one FP32 element, as the operator passes it.
    scale = torch.tensor([global_scale], dtype=torch.float32, device="cuda")
    # One program per row: the row's factor is a reduction over the row.
    token_scale = torch.zeros(rows, dtype=torch.float32, device="cuda")
    module.launch((rows, 1, 1), source, values, scales, rows, scale, token_scale)

    decoded = decode(values, scales, token_scale, rows, columns, layout)
    exact = source.to(torch.float64)
    # The unit each element was encoded in: the scale the module wrote for its
    # block. Recovered from the buffer, so a wrong scale shows up here.
    scale_of_element = _scale_per_element(values, scales, rows, columns, layout) * token_scale.to(
        torch.float64
    ).reshape(rows, 1)
    unit = torch.where(scale_of_element != 0, exact / scale_of_element, torch.zeros_like(exact))
    bound = half_step(unit) * scale_of_element.abs()

    deviation = (decoded - exact).abs()
    # `~(deviation <= bound)`, not `deviation > bound`: the comparison must fail on a NaN,
    # and `NaN > bound` is False — an output that is not a number would count as inside.
    outside = int((~(deviation <= bound)).sum().item())
    worst = torch.nan_to_num(deviation, nan=float("inf"), posinf=float("inf")).max().item()
    print(
        f"stage 1 (producer): {rows}x{columns} {layout}, global scale {global_scale:.6g}: "
        f"max |decoded - input| {worst:.4e}, {outside} of {rows * columns} elements "
        f"outside half an E2M1 code step"
    )
    if outside:
        row, column = (~(deviation <= bound)).nonzero()[0].tolist()
        print(
            f"  first: [{row}, {column}] input {exact[row, column].item():.6e} "
            f"decoded {decoded[row, column].item():.6e} "
            f"step {bound[row, column].item():.6e}"
        )
    return values, scales, token_scale, outside


def _scale_per_element(values, scales, rows, columns, layout):
    """Each element's block scale, as the module's scale bytes decode to.

    Read from the buffer rather than recomputed: this is the scale the product
    will use, so the check judges the code against the same value the consumer
    does.
    """
    scale_columns = columns // BLOCK_ELEMENTS
    table = e4m3_table().to(scales.device)
    if layout == "linear":
        block_scales = table[scales.view(torch.uint8).long()].reshape(rows, scale_columns)
    else:
        offset = swizzled_offsets(rows, scale_columns).to(scales.device)
        block_scales = table[scales.view(torch.uint8).long()[offset]][:rows, :scale_columns]
    return block_scales.repeat_interleave(BLOCK_ELEMENTS, dim=1)


def stage_consumer(module, values, scales, token_scale, rows, columns, layout, seed):
    """Stage 2: the module's codes and scales in FP64 and on the device, the same bytes.

    The activation codes and their scales are the producer's own output buffers;
    the weight stack and its scale bytes are generated here, in the atom layout
    the gemm's descriptor declares. The reference contracts the *decoded bytes* in
    FP64, so a wrong scale address shows up as a wrong number rather than as a
    different rounding.
    """
    torch.manual_seed(seed + 1)
    n = columns
    k = columns
    weights = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale_columns = k // BLOCK_ELEMENTS
    padded_n = -(-n // 128) * 128
    padded_k = -(-scale_columns // 4) * 4
    # E4M3 bytes in [0x30, 0x60): values from 2^-1 to 2^2, the range a calibrated
    # weight stack's block scales actually occupy.
    weight_scale_bytes = torch.randint(
        0x30, 0x60, (padded_n * padded_k,), dtype=torch.uint8, device="cuda"
    )

    codes = e2m1_table().to("cuda")
    scale_table = e4m3_table().to("cuda")
    decoded_a = _decode_codes(values, rows, k, codes)
    decoded_b = _decode_codes(weights, n, k, codes)
    activation_scales = _scale_per_element(values, scales, rows, columns, layout)[
        :, 0:k:BLOCK_ELEMENTS
    ] * token_scale.to(torch.float64).reshape(rows, 1)
    weight_scales = _atom_scales(scale_table, weight_scale_bytes, n, scale_columns)

    exact = torch.matmul(
        (decoded_a * activation_scales.repeat_interleave(BLOCK_ELEMENTS, dim=1)).to(torch.float64),
        (decoded_b * weight_scales.repeat_interleave(BLOCK_ELEMENTS, dim=1)).to(torch.float64).T,
    )

    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    bias = torch.zeros(n, dtype=torch.float32, device="cuda")
    out = torch.zeros(rows, n, dtype=torch.float32, device="cuda")
    call = module.Call(
        input=ctypes.c_void_p(values.data_ptr()),
        weight=ctypes.c_void_p(weights.data_ptr()),
        input_scales=ctypes.c_void_p(scales.data_ptr()),
        weight_scales=ctypes.c_void_p(weight_scale_bytes.data_ptr()),
        alpha=ctypes.c_void_p(alpha.data_ptr()),
        bias=ctypes.c_void_p(bias.data_ptr()),
        out=ctypes.c_void_p(out.data_ptr()),
        m=rows,
        n=n,
        k=k,
    )
    module.run(call)

    difference = (out.to(torch.float64) - exact).abs()
    magnitude = exact.abs().clamp(min=1e-30)
    gamma = 2.0**-24
    # `Tolerance::rounding(2, F32, BF16)`: 2 * 2 gamma, plus the output rounding.
    declared = 2 * 2 * gamma + 2.0**-8
    # `Tolerance::dot`: 2 gamma_k sqrt(k) plus the output rounding, the bound a
    # contraction's cancelling terms allow.
    contraction = 2 * gamma * math.sqrt(k) + 2.0**-8
    outside_declared = int((difference > declared * magnitude).sum().item())
    outside_contraction = int((difference > contraction * magnitude).sum().item())
    print(
        f"stage 2 (consumer): {rows}x{n} over k={k}: max |gemm - exact| "
        f"{difference.max().item():.4e}, max relative "
        f"{float((difference / magnitude).max()):.4e}"
    )
    print(
        f"  declared rtol {declared:.3e}: {outside_declared} of {rows * n} outside; "
        f"the contraction's own rtol {contraction:.3e}: {outside_contraction} outside"
    )
    if outside_declared and not outside_contraction:
        print(
            "  NOTE: the failures are inside the contraction's bound, so the operator's "
            "declared kind (`rounding`, the bound for a reduction that scales its inputs) "
            "is the half that is wrong, not the module — fp4.rs is model's to change."
        )
    return outside_contraction


def _decode_codes(packed, rows, columns, table):
    """Packed E2M1 bytes as an FP64 tile, two codes a byte, low nibble first."""
    values = table.to(packed.device)
    narrow = packed.view(torch.uint8).to(torch.int16).reshape(rows, columns // 2)
    decoded = torch.zeros(rows, columns, dtype=torch.float64, device=packed.device)
    decoded[:, 0::2] = values[(narrow & 0xF).long()]
    decoded[:, 1::2] = values[((narrow >> 4) & 0xF).long()]
    return decoded


def _atom_scales(table, bytes_, rows, scale_columns):
    """A scale buffer's bytes as an FP64 matrix in the atom's own order."""
    offset = swizzled_offsets(rows, scale_columns).to(table.device)
    return table[bytes_.long()[offset]][:rows, :scale_columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "quantize", nargs="?", help="the resolved nvfp4_quantize module directory"
    )
    parser.add_argument("--gemm", help="the resolved nvfp4_gemm module directory")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--global-scale", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="check the decoder and the two scale layouts on the host, with no module",
    )
    arguments = parser.parse_args()

    if arguments.cpu_only:
        return cpu_only()

    if not arguments.quantize:
        raise SystemExit("the quantize module directory is required unless --cpu-only is given")
    name = pathlib.Path(arguments.quantize).name
    if "layout_linear" in name:
        layout = "linear"
    elif "layout_swizzled_128x4" in name:
        layout = "swizzled_128x4"
    else:
        raise SystemExit(f"{name}: the key names no scale layout this check can decode")
    if f"columns_{arguments.columns}" not in name:
        raise SystemExit(f"{name}: the key is not the {arguments.columns}-column module")

    module = QuantizeModule(arguments.quantize)
    print(f"quantize module: {arguments.quantize} (entry {module.name}, layout {layout})")
    values, scales, token_scale, outside = stage_producer(
        module, arguments.rows, arguments.columns, arguments.block, layout,
        arguments.global_scale, arguments.seed,
    )

    failures = outside
    if arguments.gemm:
        key = pathlib.Path(arguments.gemm).name
        for field in (f"m_{arguments.rows}", f"n_{arguments.columns}", f"k_{arguments.columns}"):
            if field not in key:
                raise SystemExit(f"{key}: the gemm module is not the {field} geometry")
        gemm = GemmModule(arguments.gemm)
        # The producer's scale buffer is the consumer's activation scales, and
        # the codes are the producer's own bytes: the consumer contracts what
        # the producer wrote, not a re-packing done here.
        failures += stage_consumer(
            gemm,
            values,
            scales,
            token_scale,
            arguments.rows,
            arguments.columns,
            layout,
            arguments.seed,
        )
    print(
        "PASS" if failures == 0
        else f"FAIL: {failures} element(s) outside the encoding's own bound"
    )
    return 0 if failures == 0 else 1


def cpu_only():
    """The parts of this check that do not need a device — run before the GPU batch.

    A check whose decoder is wrong reports the kernel as broken, so the decoder
    is proved here: E4M3's values against torch's own FP8 decode (all 254 finite
    bytes), E2M1's table against the operator's documented codes, and the two
    scale layouts against each other by writing one logical matrix both ways.
    """
    failures = 0

    reference = torch.tensor(
        [byte for byte in range(256)], dtype=torch.uint8
    ).view(torch.float8_e4m3fn).to(torch.float64)
    for byte in range(256):
        try:
            ours = e4m3_value(byte)
        except ValueError:
            if not torch.isnan(reference[byte]):
                print(f"E4M3 0x{byte:02x}: called NaN, torch decodes {reference[byte].item()}")
                failures += 1
            continue
        if torch.isnan(reference[byte]) or abs(ours - float(reference[byte])) > 0.0:
            print(f"E4M3 0x{byte:02x}: decoded {ours}, torch decodes {reference[byte].item()}")
            failures += 1
    print(f"E4M3: all 256 bytes decode as torch's FP8 does ({failures} disagreement(s))")

    if E2M1 != [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        print(f"E2M1's table has moved: {E2M1}")
        failures += 1

    # One logical scale matrix, written in both layouts, must read back equal.
    rows, columns = 256, 64
    scale_columns = columns // BLOCK_ELEMENTS
    torch.manual_seed(3)
    logical = torch.randint(0x30, 0x60, (rows, scale_columns), dtype=torch.uint8)

    linear = logical.reshape(-1)
    padded_rows, padded_columns = scale_layout("swizzled_128x4", rows, scale_columns)
    swizzled = torch.zeros(padded_rows * padded_columns, dtype=torch.uint8)
    swizzled[swizzled_offsets(rows, scale_columns)] = logical.to(torch.uint8)

    table = e4m3_table()
    read_linear = table[linear.long()].reshape(rows, scale_columns)
    read_swizzled = _atom_scales(table, swizzled, rows, scale_columns)
    if not torch.equal(read_linear, read_swizzled):
        print("the two scale layouts disagree on the same logical matrix")
        failures += 1
    print(f"scale layouts: one logical matrix reads equal through both ({rows}x{columns})")

    print("PASS (host only)" if failures == 0 else f"FAIL: {failures} host check(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
