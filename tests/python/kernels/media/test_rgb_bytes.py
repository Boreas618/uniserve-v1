"""The rgb_bytes program against the operator's quantization, byte for byte.

The operator de-normalizes a channel-major raster with the layer's ImageNet
constants and packs it into bytes: ``(x · std + mean) · 255``, clamped into
``[0, 255]`` and rounded half to even (``ops::rgb::quantize_scaled``). That
is a real value rounded once, so two implementations can differ by one byte
only where the real value sits within floating-point error of a half-byte
boundary. The rasters carry many such values on purpose, both clamps, and the
two values that must land on byte 0 and byte 255; an element farther than a
part in ten thousand from a half-integer must match exactly, and one inside
that band may differ by one byte, the operator's own
``Tolerance::new(1.0, 0.0)``.

Two references, each with its cases:

- the host oracle, ``quantize_scaled`` mirrored step by step in FP32 (what the
  program adds itself to), over FP32 rasters in four layouts: the packed
  raster; a tile written into a sub-region of a larger canvas with a strided
  read side (three of the six strides longer than packed); and a one-frame,
  one-row strip of each, where the frame and row axes contribute nothing and
  an off-by-one in the index decomposition cannot hide behind them;
- the arithmetic in FP64 from the operand's own rounded values, the real value
  an integer output is compared with, over every element type the operator
  reads, packed and in a canvas, at the module's block of 1024.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`tests.python.kernels.chain`); that launch needs a device.
"""

import numpy as np
import pytest
import torch

from tests.python.kernels import chain

pytestmark = pytest.mark.unit

#: The reference's ImageNet constants, which the layer holds as device
#: weights.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: How near a half-integer counts as a tie, in bytes of the 255-byte range.
TIE_BAND = 1e-4

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The host oracle's layouts: (raster shape, canvas).
LAYOUTS = {
    "the packed raster": ((3, 2, 4, 6), False),
    "a tile in a canvas": ((3, 2, 4, 6), True),
    "a strip": ((3, 1, 1, 9), False),
    "a strip in a canvas": ((3, 1, 1, 9), True),
}


def launch(kernels, source, mean, std, out, block, pdl=False):
    """One launch packing every pixel of `source` into `out`, both addressed by
    their three outer strides; compiled for a dependent launch when `pdl` is.
    """  # noqa: D205
    channels, frames, rows, columns = source.shape
    total = frames * rows * columns * channels
    kernels._rgb_bytes_kernel[(-(-total // block),)](
        source,
        mean,
        std,
        out,
        frames,
        rows,
        columns,
        channels,
        *source.stride()[:3],
        *out.stride()[:3],
        int(pdl),
        CHANNELS=channels,
        BLOCK=block,
        PDL=pdl,
        num_warps=max(1, min(8, block // 512)),
        launch_pdl=pdl,
    )


def raster(shape, seed=0):
    """A raster of decoder-like values in FP32 with half-byte ties, both
    clamps, and the two exact ends of the byte range.
    """  # noqa: D205
    generator = np.random.default_rng(seed)
    values = generator.normal(0.0, 0.3, shape).astype(np.float32)
    channels, frames, rows, columns = shape
    for channel in range(channels):
        # Half-byte ties: ask for a scaled value of `k + 0.5` and invert the
        # de-normalization. The inverse is inexact, so the references work
        # from the values the raster holds rather than from the intent.
        ties = (np.arange(0.5, 12.5, 1.0) / 255.0 - MEAN[channel]) / STD[
            channel
        ]
        for index, value in enumerate(ties):
            values[
                channel,
                index % frames,
                (index * 3) % rows,
                (index * 7) % columns,
            ] = value
    values[0, 0, 0, 0] = 10.0
    values[0, 0, 0, 1] = -10.0
    values[0, 0, 0, 2] = (1.0 - MEAN[0]) / STD[0]
    values[0, 0, 0, 3] = (0.0 - MEAN[0]) / STD[0]
    return values


def operands(values, element, canvas, device):
    """The read and write views of one launch on `device`.

    In a canvas the write goes into a sub-region of a larger buffer and the
    read side is a slice of a wider raster, so three of the six strides are
    not the packed ones.
    """
    channels, frames, rows, columns = values.shape
    operand = torch.from_numpy(values).to(element)
    if not canvas:
        out = torch.zeros(
            (frames, rows, columns, channels), dtype=torch.uint8, device=device
        )
        return operand.to(device), out
    buffer = torch.zeros(
        (frames + 1, rows, columns + 2, channels),
        dtype=torch.uint8,
        device=device,
    )
    wider = torch.zeros((channels, frames, rows, columns + 3), dtype=element)
    wider[:, :, :, :columns] = operand
    return wider.to(device)[:, :, :, :columns], buffer[:-1, :, :columns, :]


def constants(device):
    """The layer's mean and deviation as the program reads them."""
    return torch.from_numpy(MEAN).to(device), torch.from_numpy(STD).to(device)


def assert_within_the_band(label, got, expected, ties):
    """Exact bytes, or one byte apart inside the tie band, on
    ``[frames, rows, columns, channels]`` arrays.
    """  # noqa: D205
    difference = np.abs(got.astype(np.int32) - expected.astype(np.int32))
    wrong = (difference > 0) & ~(ties & (difference <= 1))
    assert not wrong.any(), (
        f"{label}: {got.size} bytes, {int(ties.sum())} tie candidates, "
        f"{int((difference > 0).sum())} differ, {int(wrong.sum())} outside the "
        f"band; first at [frame, row, column, channel] "
        f"{tuple(int(value) for value in np.argwhere(wrong)[0])}"
    )


def host_oracle(source):
    """``ops::rgb::quantize_scaled`` in FP32 step by step over a channel-major
    raster: the bytes, packed, and which of them are tie candidates (decided
    on the FP64 value).
    """  # noqa: D205
    channels, frames, rows, columns = source.shape
    expected = np.zeros((frames, rows, columns, channels), dtype=np.uint8)
    ties = np.zeros((frames, rows, columns, channels), dtype=bool)
    for channel in range(channels):
        mean, std = MEAN[channel], STD[channel]
        for frame in range(frames):
            for row in range(rows):
                for column in range(columns):
                    value = np.float32(source[channel, frame, row, column])
                    scaled = (value * std + mean) * np.float32(255.0)
                    exact = (
                        np.float64(value) * np.float64(std) + np.float64(mean)
                    ) * 255.0
                    expected[frame, row, column, channel] = (
                        0
                        if np.isnan(scaled)
                        else int(np.rint(np.clip(scaled, 0.0, 255.0)))
                    )
                    ties[frame, row, column, channel] = (
                        0.0 <= exact <= 255.0
                        and abs(abs(exact - np.floor(exact)) - 0.5) < TIE_BAND
                    )
    return expected, ties


def exact_bytes(rounded):
    """The bytes the operator's arithmetic gives in FP64 from the operand's
    own rounded values, packed, and which of them are tie candidates.
    """  # noqa: D205
    scaled = (
        rounded.astype(np.float64) * STD.astype(np.float64)[:, None, None, None]
        + MEAN.astype(np.float64)[:, None, None, None]
    ) * 255.0
    byte = np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.int32)
    near_half = np.abs(np.abs(scaled - np.floor(scaled)) - 0.5) < TIE_BAND
    # Channel-major `[channels, frames, rows, columns]` to packed pixels.
    return byte.transpose(1, 2, 3, 0), near_half.transpose(1, 2, 3, 0)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_layout_matches_the_host_oracle(
    layout, torch_device, kernel_source
):
    shape, canvas = LAYOUTS[layout]
    values = raster(shape)
    source, out = operands(values, torch.float32, canvas, torch_device)

    launch(
        kernel_source("rgb_bytes"), source, *constants(torch_device), out, 16
    )

    expected, ties = host_oracle(source.cpu().numpy())
    assert_within_the_band(layout, out.cpu().numpy(), expected, ties)


@pytest.mark.parametrize("canvas", [False, True], ids=["packed", "canvas"])
@pytest.mark.parametrize("dtype", list(TORCH_TYPES))
def test_every_element_type_matches_the_exact_bytes(
    dtype, canvas, torch_device, kernel_source
):
    values = raster((3, 2, 4, 6))
    source, out = operands(values, TORCH_TYPES[dtype], canvas, torch_device)

    launch(
        kernel_source("rgb_bytes"), source, *constants(torch_device), out, 1024
    )

    # What the program reads: the operand after its own type's rounding.
    rounded = source.to(torch.float32).cpu().numpy()
    expected, ties = exact_bytes(rounded)
    assert_within_the_band(
        f"{dtype}, block 1024, {'a tile in a canvas' if canvas else 'packed'}",
        out.cpu().numpy(),
        expected,
        ties,
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    if not chain.supported():
        pytest.skip("this device compiles no dependent launch")
    torch.manual_seed(0)
    kernels = kernel_source("rgb_bytes")
    channels, frames, rows, columns, block = 3, 2, 16, 32, 256

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["raster"],
            inputs["mean"],
            inputs["std"],
            outputs["out"],
            block,
            pdl,
        )
        return [outputs["out"]]

    case = chain.Case(
        "rgb_bytes",
        {
            "raster": torch.rand(channels, frames, rows, columns).to(
                torch.float16
            ),
            "mean": torch.full((channels,), 0.5),
            "std": torch.full((channels,), 0.5),
        },
        {
            "out": torch.zeros(
                frames, rows, columns, channels, dtype=torch.uint8
            )
        },
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"rgb_bytes launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
