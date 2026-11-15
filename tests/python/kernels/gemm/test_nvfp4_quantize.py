"""The NVFP4 family end to end: producer, consumer, the interface between.

Three stages, each a test, because they fail for different reasons:

1. **The producer against its input.** The built ``nvfp4_quantize`` module
   encodes a BF16 tile into packed E2M1 codes, E4M3 block scales and a factor
   per row. The test reads the module's own output buffers, never a re-packing
   done here, decodes those bytes as the operator's documentation says they
   mean (two E2M1 codes a byte, the low nibble first; one E4M3 scale per
   sixteen elements of the contraction, relative to the row's factor) and
   compares the decoded tile with the input, each element within half a step
   of E2M1 at its block's scale. What can fail: the E2M1 rounding, the code
   order inside a byte, the E4M3 scale's own encoding (exponent bias 7,
   subnormals below 2^-6, saturation at 448) and the scale bytes' layout. A
   row of zeros must encode to zeros with a finite factor.
2. **The consumer against the same bytes.** Torch multiplies the producer's
   codes and scale bytes in FP64 and the built ``nvfp4_gemm`` module multiplies
   them on the device.
3. **The two against each other.** If the producer passes and the consumer
   fails, one field of the interface between the two modules is wrong; if the
   producer fails, the consumer was never the question.

The producer is decisive on a device and not on Triton's interpreter, whose
narrowing float casts truncate instead of rounding, and the E4M3 scale
encoding is a rounding: the module stages need a device. Without one, the
check proves its decoder (every E4M3 byte against torch's FP8 decode) and the
two scale layouts against each other, since a wrong decoder would report the
kernel as broken.

The operator launched as a programmatic dependent reads only what the grid
before it wrote (:mod:`uniserve_testing.kernels.chain`), which needs a device
too.
"""

import ctypes
import json
import math

import pytest
import torch
from uniserve_testing.kernels import chain
from uniserve_testing.kernels.gemm import nvfp4

pytestmark = pytest.mark.unit

#: The largest E4M3 value; the encoding has no infinities, so beyond it
#: saturates.
E4M3_MAX = 448.0

#: The value an output buffer starts as: neither a plausible code nor a
#: plausible E4M3 factor, so an element the module never writes reads as a
#: mismatch rather than as a zero that agrees with a zero.
UNWRITTEN = 0xCD

#: The producer's problem: rows, columns, the encode block, the global scale
#: and the seed.
ROWS, COLUMNS, BLOCK, GLOBAL_SCALE, SEED = 128, 2048, 1024, 50.0, 7


def e2m1_value(nibble):
    """One E2M1 code: sign bit 3 over a four-value magnitude table."""
    magnitude = nvfp4.E2M1[nibble & 0x7]
    return -magnitude if nibble & 0x8 else magnitude


def e2m1_table():
    """The sixteen codes as a float64 tensor, for decoding in vectors."""
    return torch.tensor(
        [e2m1_value(nibble) for nibble in range(16)], dtype=torch.float64
    )


def e4m3_value(byte):
    """One E4M3 byte: four exponent bits biased by 7 over three mantissa
    bits.

    ``0x7f``/``0xff`` are NaN; a checkpoint's scale is never one, so this
    raises rather than decoding a NaN into every product.
    """  # noqa: D205
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        raise ValueError(f"0x{byte:02x} is an E4M3 NaN")
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def e4m3_table():
    """Every E4M3 byte's value as a float64 tensor; the NaN slots are zero and
    unused.
    """  # noqa: D205
    values = []
    for byte in range(256):
        try:
            values.append(e4m3_value(byte))
        except ValueError:
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float64)


def scale_offsets(layout, rows, scale_columns):
    """The flat offset of every ``[row, scale column]`` in `layout`."""
    if layout == "linear":
        return nvfp4.linear_offsets(rows, scale_columns)
    return nvfp4.swizzled_offsets(rows, scale_columns)


def block_scales(scales, rows, columns, layout):
    """Each element's block scale, as the module's scale bytes decode to.

    Read from the buffer rather than recomputed: this is the scale the product
    will use, so the code is judged against the same value the consumer uses.
    """
    scale_columns = columns // nvfp4.BLOCK_ELEMENTS
    offset = scale_offsets(layout, rows, scale_columns).to(scales.device)
    table = e4m3_table().to(scales.device)
    matrix = table[scales.view(torch.uint8).long()[offset]]
    return matrix.repeat_interleave(nvfp4.BLOCK_ELEMENTS, dim=1)


def decode(values, scales, token_scale, rows, columns, layout):
    """The module's bytes as an FP64 tile: each code times its block's scale
    and its row's factor.
    """  # noqa: D205
    codes = e2m1_table().to(values.device)[
        nvfp4.unpacked(values, rows, columns)
    ]
    factor = token_scale.to(torch.float64).reshape(rows, 1)
    return codes * block_scales(scales, rows, columns, layout) * factor


def half_step(unit):
    """Half the distance to the next E2M1 code, in units of the block's scale.

    This is the encoding's own bound, not a chosen tolerance: a decoded code
    is correct when the exact quotient lies within half a step of it.
    """
    codes = torch.tensor(nvfp4.E2M1, dtype=torch.float64, device=unit.device)
    magnitude = unit.abs()
    upper = torch.searchsorted(codes, magnitude, right=False)
    upper = upper.clamp(max=len(nvfp4.E2M1) - 1)
    lower = torch.searchsorted(codes, magnitude, right=True) - 1
    lower = lower.clamp(min=0)
    # A quotient above the last code saturates, so its step is the last
    # interval.
    span = (codes[upper] - codes[lower]).clamp(min=0.5)
    return 0.5 * span


def test_the_decoders_are_the_formats_definitions():
    """E4M3's values against torch's own FP8 decode (all 254 finite bytes and
    the two NaNs), and E2M1's table against the operator's documented codes.
    """  # noqa: D205
    reference = (
        torch.arange(256, dtype=torch.int32)
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float64)
    )
    disagreements = []
    for byte in range(256):
        try:
            ours = e4m3_value(byte)
        except ValueError:
            if not torch.isnan(reference[byte]):
                disagreements.append(
                    f"0x{byte:02x}: called NaN, torch decodes "
                    f"{reference[byte].item()}"
                )
            continue
        if torch.isnan(reference[byte]) or ours != float(reference[byte]):
            disagreements.append(
                f"0x{byte:02x}: decoded {ours}, torch decodes "
                f"{reference[byte].item()}"
            )

    assert not disagreements, (
        f"{len(disagreements)} E4M3 bytes decode unlike torch's FP8: "
        f"{disagreements}"
    )
    assert nvfp4.E2M1 == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), (
        f"E2M1's table has moved: {nvfp4.E2M1}"
    )


def test_both_scale_layouts_read_one_logical_matrix_alike():
    rows, columns = 256, 64
    scale_columns = columns // nvfp4.BLOCK_ELEMENTS
    torch.manual_seed(3)
    logical = torch.randint(
        0x30, 0x60, (rows, scale_columns), dtype=torch.uint8
    )
    padded_rows, padded_columns = nvfp4.padded_shape(rows, scale_columns)
    swizzled = torch.zeros(padded_rows * padded_columns, dtype=torch.uint8)
    swizzled[nvfp4.swizzled_offsets(rows, scale_columns)] = logical

    table = e4m3_table()
    read_linear = table[logical.reshape(-1).long()].reshape(rows, scale_columns)
    offsets = nvfp4.swizzled_offsets(rows, scale_columns)
    read_swizzled = table[swizzled.long()[offsets]]

    assert torch.equal(read_linear, read_swizzled), (
        "the two scale layouts disagree on the same logical matrix"
    )


class Module:
    """A built Triton module's cubin, launched through the CUDA driver with
    the parameters its metadata records, as the operator launches it.
    """  # noqa: D205

    def __init__(self, directory, device):
        """Load ``module.cubin`` of `directory` into the device's context."""
        from cuda.bindings import driver

        self._driver = driver
        self._device = device
        metadata = json.loads((directory / "metadata.json").read_text())
        self.name = metadata["name"]
        self.signature = metadata["signature"]
        self.threads = metadata["num_warps"] * metadata["warp_size"]
        self.shared = metadata["shared"]
        # The compiler appends a global and a profile scratch address to every
        # program's parameters; the launch passes both null, which a module
        # that asks for no scratch never reads.
        scratch = (
            metadata["global_scratch_size"],
            metadata["profile_scratch_size"],
        )
        assert scratch == (0, 0), (
            f"{self.name} asks for {scratch} bytes of global and profile "
            "scratch, which this launch does not provide"
        )
        # The driver loads into the current context, torch's primary one.
        torch.zeros(1, device=device)
        error, module = driver.cuModuleLoadData(
            (directory / "module.cubin").read_bytes()
        )
        assert int(error) == 0, (
            f"{directory}/module.cubin did not load: {error}"
        )
        error, self._function = driver.cuModuleGetFunction(
            module, self.name.encode()
        )
        assert int(error) == 0, f"{self.name} is not in the cubin: {error}"

    def launch(self, grid, *arguments):
        """Launch the program over `grid` with one argument per parameter of
        its signature, each passed as the type the signature records.
        """  # noqa: D205
        assert len(arguments) == len(self.signature), (
            f"{self.name} takes {list(self.signature)}, the launch passes "
            f"{len(arguments)} arguments"
        )
        holders = []
        for argument, kind in zip(arguments, self.signature.values()):
            if kind.startswith("*"):
                holders.append(ctypes.c_void_p(argument.data_ptr()))
            elif kind == "i64":
                holders.append(ctypes.c_int64(argument))
            elif kind == "i32":
                holders.append(ctypes.c_int32(argument))
            else:
                assert kind == "fp32", f"{self.name}: a {kind} parameter"
                holders.append(ctypes.c_float(argument))
        holders += [ctypes.c_void_p(0), ctypes.c_void_p(0)]
        pointers = (ctypes.c_void_p * len(holders))(
            *(ctypes.addressof(holder) for holder in holders)
        )
        stream = torch.cuda.current_stream(self._device).cuda_stream
        (error,) = self._driver.cuLaunchKernel(
            self._function,
            *grid,
            self.threads,
            1,
            1,
            self.shared,
            stream,
            ctypes.addressof(pointers),
            0,
        )
        assert int(error) == 0, f"launching {self.name}: {error}"
        torch.cuda.synchronize(self._device)


def quantize_module(kernel_module, device, layout):
    """The built quantize module of the producer's problem in `layout`."""
    key = (
        f"input_bf16_columns_{COLUMNS}_layout_{layout}_block_{BLOCK}_pdl_false"
    )
    return Module(kernel_module("nvfp4_quantize", "triton", key), device)


def encoded(module, source, layout):
    """One launch of the producer over `source`, one program per row, into
    buffers that start as `UNWRITTEN`; answers the values, the scales and the
    rows' factors.
    """  # noqa: D205
    rows, columns = source.shape
    device = source.device
    values = torch.full(
        (rows, columns // 2), UNWRITTEN, dtype=torch.uint8, device=device
    )
    scale_columns = columns // nvfp4.BLOCK_ELEMENTS
    if layout == "linear":
        scale_count = rows * scale_columns
    else:
        padded_rows, padded_columns = nvfp4.padded_shape(rows, scale_columns)
        scale_count = padded_rows * padded_columns
    scales = torch.full(
        (scale_count,), UNWRITTEN, dtype=torch.uint8, device=device
    )
    # The global scale is a device operand: one FP32 element, as the operator
    # passes it.
    scale = torch.tensor([GLOBAL_SCALE], dtype=torch.float32, device=device)
    token_scale = torch.zeros(rows, dtype=torch.float32, device=device)
    module.launch(
        (rows, 1, 1), source, values, scales, rows, scale, token_scale
    )
    return values, scales, token_scale


def producer_source(device):
    """The producer's BF16 tile."""
    torch.manual_seed(SEED)
    return (
        torch.randn(ROWS, COLUMNS, device=device, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["linear", "swizzled_128x4"])
def test_the_producers_bytes_decode_to_its_input(
    layout, torch_device, kernel_module
):
    module = quantize_module(kernel_module, torch_device, layout)
    source = producer_source(torch_device)

    values, scales, token_scale = encoded(module, source, layout)

    decoded = decode(values, scales, token_scale, ROWS, COLUMNS, layout)
    exact = source.to(torch.float64)
    # The unit each element was encoded in: the scale the module wrote for its
    # block, recovered from the buffer, so a wrong scale shows up here.
    scale_of_element = block_scales(
        scales, ROWS, COLUMNS, layout
    ) * token_scale.to(torch.float64).reshape(ROWS, 1)
    unit = torch.where(
        scale_of_element != 0, exact / scale_of_element, torch.zeros_like(exact)
    )
    bound = half_step(unit) * scale_of_element.abs()
    deviation = (decoded - exact).abs()
    # `~(deviation <= bound)`: the comparison must fail on a NaN.
    outside = ~(deviation <= bound)
    worst = torch.nan_to_num(deviation, nan=math.inf).max().item()
    row, column = outside.nonzero()[0].tolist() if outside.any() else (0, 0)
    assert not bool(outside.any()), (
        f"{ROWS}x{COLUMNS} {layout}, global scale {GLOBAL_SCALE}: "
        f"{int(outside.sum())} of {ROWS * COLUMNS} elements outside half an "
        f"E2M1 code step, max |decoded - input| {worst:.4e}; first [{row}, "
        f"{column}]: input {exact[row, column].item():.6e} decoded "
        f"{decoded[row, column].item():.6e} step "
        f"{bound[row, column].item():.6e}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["linear", "swizzled_128x4"])
def test_a_row_of_zeros_encodes_to_zeros_with_a_finite_factor(
    layout, torch_device, kernel_module
):
    """A zero row's blocks have an amax of 0, so their E4M3 scale is 0 and a
    reciprocal of it is infinite, which would make the codes NaN. A deployment
    reaches it wherever a bucket carries padded rows, and a NaN row in the
    cache poisons every later attention step.
    """  # noqa: D205
    module = quantize_module(kernel_module, torch_device, layout)
    source = producer_source(torch_device)
    source[-1] = 0.0

    values, scales, token_scale = encoded(module, source, layout)

    decoded = decode(values, scales, token_scale, ROWS, COLUMNS, layout)
    # The factor is what the reciprocal divides: an infinite or NaN one is the
    # defect itself, whatever the codes came out as.
    bad_factors = int((~torch.isfinite(token_scale)).sum())
    bad_decoded = int((~torch.isfinite(decoded)).sum())
    assert bad_factors == 0 and bad_decoded == 0, (
        f"row {ROWS - 1} of {ROWS}x{COLUMNS} {layout} is zero: {bad_factors} "
        f"of {ROWS} factors and {bad_decoded} of {ROWS * COLUMNS} decoded "
        "elements are not finite"
    )
    zero_row_worst = decoded[-1].abs().max().item()
    assert zero_row_worst == 0.0, (
        f"the zero row decoded to values as large as {zero_row_worst:.3e}: a "
        "zero input is zero, not a small value"
    )


#: The consumer's descriptor: the producer's rows by a square weight of its
#: columns, the activation scales in the layout the gemm package declares, a
#: BF16 output.
GEMM_KEY = (
    "packed_u8_scale_f8e4m3_accumulate_f32_output_bf16_layout_swizzled_128x4_"
    f"epilogue_none_m_{ROWS}_n_{COLUMNS}_k_{COLUMNS}"
)

#: The unit roundoffs of the consumer's accumulation and of its BF16 output.
F32_ROUNDOFF = 2.0**-24
BF16_ROUNDOFF = 2.0**-8


@pytest.mark.gpu
def test_the_consumer_multiplies_the_producers_bytes(
    torch_device, kernel_module
):
    """The activation codes and their scales are the producer's own output
    buffers; the weight stack and its scale bytes are generated here, in the
    atom layout the gemm's descriptor declares. The reference multiplies the
    decoded bytes in FP64, so a wrong scale address shows up as a wrong number
    rather than as a different rounding.

    The comparison is the contraction's own bound, `Tolerance::dot`:
    `2 γ_k sqrt(k)` plus the output's rounding. A failure also counts the
    elements the operator's declared `Tolerance::rounding(2, F32, BF16)`
    refuses.
    """  # noqa: D205
    layout = "swizzled_128x4"
    producer = quantize_module(kernel_module, torch_device, layout)
    values, scales, token_scale = encoded(
        producer, producer_source(torch_device), layout
    )
    library = ctypes.CDLL(
        str(kernel_module("nvfp4_gemm", "cute_dsl", GEMM_KEY) / "module.so")
    )
    entry = library.uniserve_nvfp4_gemm
    entry.argtypes = [ctypes.POINTER(nvfp4.GemmCall), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    torch.manual_seed(SEED + 1)
    n = k = COLUMNS
    weights = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device=torch_device
    )
    scale_columns = k // nvfp4.BLOCK_ELEMENTS
    padded_n, padded_k = nvfp4.padded_shape(n, scale_columns)
    # E4M3 bytes in [0x30, 0x60): values from 2^-1 to 2^2, the range a
    # calibrated weight stack's block scales occupy.
    weight_scale_bytes = torch.randint(
        0x30,
        0x60,
        (padded_n * padded_k,),
        dtype=torch.uint8,
        device=torch_device,
    )

    codes = e2m1_table().to(torch_device)
    decoded_a = codes[nvfp4.unpacked(values, ROWS, k)]
    decoded_b = codes[nvfp4.unpacked(weights, n, k)]
    activation_scales = block_scales(scales, ROWS, k, layout) * token_scale.to(
        torch.float64
    ).reshape(ROWS, 1)
    weight_scales = block_scales(weight_scale_bytes, n, k, layout)
    exact = (decoded_a * activation_scales) @ (decoded_b * weight_scales).T

    alpha = torch.ones(1, dtype=torch.float32, device=torch_device)
    out = torch.zeros(ROWS, n, dtype=torch.bfloat16, device=torch_device)
    call = nvfp4.GemmCall(
        input=values.data_ptr(),
        weight=weights.data_ptr(),
        input_scales=scales.data_ptr(),
        weight_scales=weight_scale_bytes.data_ptr(),
        alpha=alpha.data_ptr(),
        bias=None,
        out=out.data_ptr(),
        m=ROWS,
        n=n,
        k=k,
    )
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    message = bytes(call.error).rstrip(b"\0").decode()
    assert status == 0 and not message, (
        f"the gemm entry refused the call: {message}"
    )

    difference = (out.to(torch.float64) - exact).abs()
    magnitude = exact.abs().clamp(min=1e-30)
    declared = 2 * 2 * F32_ROUNDOFF + BF16_ROUNDOFF
    contraction = 2 * F32_ROUNDOFF * math.sqrt(k) + BF16_ROUNDOFF
    outside_declared = int((~(difference <= declared * magnitude)).sum())
    outside_contraction = int((~(difference <= contraction * magnitude)).sum())
    assert outside_contraction == 0, (
        f"{ROWS}x{n} over k={k}: {outside_contraction} of {ROWS * n} outside "
        f"the contraction's rtol {contraction:.3e} ({outside_declared} outside "
        f"the declared rtol {declared:.3e}); max |gemm - exact| "
        f"{difference.max().item():.4e}, max relative "
        f"{float((difference / magnitude).max()):.4e}"
    )


def launch(kernels, inputs, outputs, columns, block, pdl):
    """The operator's launch over `inputs["input"]` in the linear layout, one
    program per row, compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    rows = inputs["input"].shape[0]
    scale_columns = columns // nvfp4.BLOCK_ELEMENTS
    kernels._nvfp4_quantize_kernel[(rows, 1, 1)](
        inputs["input"],
        outputs["values"],
        outputs["scales"],
        rows,
        inputs["global_scale"],
        outputs["token_scale"],
        COLUMNS=columns,
        BLOCK=block,
        SCALE_COLUMNS=scale_columns,
        SWIZZLED=False,
        PADDED_COLUMNS=scale_columns,
        PADDED_TILE=scale_columns,
        PDL=pdl,
        launch_pdl=pdl,
        num_warps=2,
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("nvfp4_quantize")
    rows, columns, block = 16, 2048, 64
    scale_columns = columns // nvfp4.BLOCK_ELEMENTS

    def launched(inputs, outputs, pdl):
        launch(kernels, inputs, outputs, columns, block, pdl)
        return [outputs["values"], outputs["scales"], outputs["token_scale"]]

    case = chain.Case(
        "nvfp4_quantize",
        {
            "input": (torch.randn(rows, columns) * 0.05).to(torch.bfloat16),
            "global_scale": torch.tensor([1.0]),
        },
        {
            "values": torch.zeros(rows, columns // 2, dtype=torch.uint8),
            "scales": torch.zeros(rows, scale_columns, dtype=torch.uint8),
            "token_scale": torch.zeros(rows),
        },
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"nvfp4_quantize launched as a dependent: {failures} of "
        f"{chain.ROUNDS} rounds differ from its launch after a synchronization"
    )
