"""The `mxfp8_quantize` program against the operator's encoding formula.

The operator (``crates/ops/src/mx.rs``) encodes a row in blocks of 32 elements:
``amax = max(max |x|, 1e-4)``, the UE8M0 scale the least power of two at or
above ``fl(amax · fl(1/448))``, and each code ``e4m3(clamp(x / scale, ±448))``
rounded to nearest even. Every step is exact or one rounding to nearest, so the
operator's tolerance is exact and the check compares the codes and the scale
bytes it writes with the formula's (:func:`uniserve_testing.kernels.gemm.mx.
encode_mxfp8`) byte for byte, in both layouts the encoder writes: the row's
own byte table, and DeepGEMM's MN-major packed words.

The widths are DeepSeek-V4.1-Flash's projections' inputs (hidden 5120, the
query's low rank 1280, the shared expert's and the routed experts' 2304, the
grouped output's 4096 and 8192), in a decode step's rows and a prompt's. The
interpreter does not round to E4M3 as a device does, so every case needs one.
"""

import pytest
import torch
import triton
from uniserve_testing.kernels.gemm import mx

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

#: The encoder's program width the operator states for a row (`ops::mx::
#: QuantizeDescriptor::new`): the operators' block, halved until it divides
#: the row.
OPERATORS_BLOCK = 1024


def program_block(columns):
    """The elements one program encodes, as the operator's descriptor states."""
    block = OPERATORS_BLOCK
    while block > 128 and columns % block != 0:
        block //= 2
    return block


def encode(kernels, x, packed, row_stride=None):
    """Launch the encoder over ``x`` ``[rows, columns]`` as the operator does:
    one grid row a row, one program a block across. Answers the codes and the
    scale table.
    """  # noqa: D205
    rows, columns = x.shape
    block = program_block(columns)
    codes = torch.empty(
        rows, columns, dtype=torch.float8_e4m3fn, device=x.device
    )
    if packed:
        scales = torch.zeros(
            mx.packed_shape(rows, columns), dtype=torch.int32, device=x.device
        )
    else:
        scales = torch.zeros(
            rows, columns // mx.BLOCK, dtype=torch.uint8, device=x.device
        )
    kernels._mxfp8_quantize_kernel[(rows, triton.cdiv(columns, block))](
        x,
        codes,
        scales,
        rows,
        row_stride or x.stride(0),
        COLUMNS=columns,
        BLOCK=block,
        PACKED=packed,
        PDL=False,
        num_warps=4,
    )
    return codes, scales


#: The cases: rows, columns, the seed, the activation type.
CASES = [
    pytest.param(1, 5120, 1, torch.bfloat16, id="hidden, one decode row"),
    pytest.param(7, 1280, 2, torch.bfloat16, id="q_lora, a decode batch"),
    pytest.param(300, 2304, 3, torch.bfloat16, id="expert intermediate"),
    pytest.param(129, 4096, 4, torch.bfloat16, id="a grouped output"),
    pytest.param(64, 8192, 5, torch.bfloat16, id="wo_b input"),
    pytest.param(5, 640, 6, torch.float16, id="f16"),
]


def activations(rows, columns, seed, dtype):
    """Rows with the magnitudes an activation spans: most blocks unit scale,
    some tiny, some large, one block of zeros and a saturating outlier.
    """  # noqa: D205
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, columns, generator=generator)
    magnitude = torch.exp2(
        torch.randint(-12, 9, (rows, columns // mx.BLOCK), generator=generator)
    ).float()
    x *= magnitude.repeat_interleave(mx.BLOCK, dim=1)
    x[0, : mx.BLOCK] = 0.0
    x[-1, -1] = 3.0e4
    return x.to(dtype)


@pytest.mark.parametrize("packed", [False, True], ids=["linear", "packed"])
@pytest.mark.parametrize(("rows", "columns", "seed", "dtype"), CASES)
def test_the_codes_and_scales_are_the_formulas(
    rows, columns, seed, dtype, packed, torch_device, kernel_source
):
    x = activations(rows, columns, seed, dtype)

    codes, scales = encode(
        kernel_source("mxfp8_quantize"), x.to(torch_device), packed
    )

    wanted_codes, wanted_scales = mx.encode_mxfp8(x)
    got_scales = mx.unpacked_scales(scales, rows) if packed else scales.cpu()
    mismatched = (got_scales != wanted_scales).nonzero()
    assert mismatched.numel() == 0, (
        f"{len(mismatched)} of {wanted_scales.numel()} scale bytes differ; "
        f"first at {mismatched[0].tolist()}: "
        f"{int(got_scales[tuple(mismatched[0])])} vs "
        f"{int(wanted_scales[tuple(mismatched[0])])}"
    )
    got = codes.cpu().view(torch.uint8)
    wanted = wanted_codes.view(torch.uint8)
    differing = (got != wanted).nonzero()
    assert differing.numel() == 0, (
        f"{len(differing)} of {got.numel()} codes differ; first at "
        f"{differing[0].tolist()}: {int(got[tuple(differing[0])])} vs "
        f"{int(wanted[tuple(differing[0])])}"
    )


def test_rows_any_stride_apart_encode_as_contiguous_ones(
    torch_device, kernel_source
):
    """The activation may be a view whose rows lie further apart than its
    width, as a slice of a wider projection's output does.
    """  # noqa: D205
    wide = activations(9, 2048, 7, torch.bfloat16)
    x = wide[:, :1280]

    codes, scales = encode(
        kernel_source("mxfp8_quantize"), wide.to(torch_device)[:, :1280], False
    )

    wanted_codes, wanted_scales = mx.encode_mxfp8(x)
    assert torch.equal(scales.cpu(), wanted_scales)
    assert torch.equal(
        codes.cpu().view(torch.uint8), wanted_codes.view(torch.uint8)
    )
