"""The cross_fade program against the decoded-overlap blend, bit for bit.

The operator's tolerance is ``exact``: its weights are the decoded type's own
arithmetic (the ramp ``arange(extent, dtype) / extent``, ``1 - that``, the two
products and their sum, each rounded into the tile's type), so a correct
implementation produces the reference's bits, and every case compares bits
rather than a bound.

What is checked, for both axes and three element types:

- the decoded-overlap blend of two whole tiles, the tail beyond the band a
  copy of ``current`` the blend leaves untouched;
- the aliased destination, ``out`` the same buffer as ``current``, which the
  operator allows because every element is read before the position it is
  written at;
- a destination that is a slice of a larger raster, so the three outer strides
  are not the tile's own and a wrong stride shows as a wrong value, with a
  filler that shows a write outside the tile;
- a tile whose whole axis is the band, where the blend alone is the result;
- a seam whose sources and destination are strided views of wider rasters,
  each band exactly ``extent`` wide: the shape the decoder hands over, held to
  the same blend over contiguous copies of the two bands.

A BF16 tile's arithmetic rounds each step to nearest on a device; Triton's
interpreter narrows into BF16 by truncation instead (its F16 and F32 narrowings
round to nearest), so on the interpreter the BF16 reference narrows every step
by truncation, and the comparison stays bit for bit.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import numpy as np
import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

#: The element types a decoded tile is stored in.
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}

#: The block of every launch: 1024 voxels, one program per block.
BLOCK = 1024

#: The destinations of a whole-tile blend: (alias, raster).
DESTINATIONS = {
    "own buffer": (False, False),
    "aliased": (True, False),
    "in a raster": (False, True),
}

#: The whole-tile blends: every element type, axis, band and destination over
#: a 3x2x6x8 tile, and a tile whose whole axis is the band.
TILES = [
    pytest.param(dtype, axis, extent, destination, (3, 2, 6, 8), id=identity)
    for dtype in DTYPES
    for axis in (2, 3)
    for extent in (3, 5)
    for destination in DESTINATIONS
    for identity in [f"{dtype}-axis{axis}-extent{extent}-{destination}"]
] + [
    pytest.param(
        "f16", 2, 5, "own buffer", (2, 1, 5, 5), id="f16-the-whole-axis-a-band"
    )
]


def launch(kernels, previous, current, out, extent, axis, block, pdl=False):
    """One launch over every voxel of `out`, each view addressed by its three
    outer strides; compiled for a dependent launch when `pdl` is, and with the
    package's warps and options.
    """  # noqa: D205
    channels, frames, rows, columns = out.shape
    elements = channels * frames * rows * columns
    kernels._cross_fade_kernel[(-(-elements // block),)](
        previous,
        current,
        out,
        extent,
        axis,
        channels,
        frames,
        rows,
        columns,
        *previous.stride()[:3],
        *current.stride()[:3],
        *out.stride()[:3],
        block,
        AXIS=axis,
        BLOCK=block,
        PDL=pdl,
        num_warps=8 if block >= 1024 else 4,
        launch_pdl=pdl,
        # The launch table's: each product rounds before the sum.
        enable_fp_fusion=False,
    )


def truncated(values):
    """FP32 values as Triton's interpreter narrows them into BF16: the low
    sixteen bits dropped, returned in FP32.
    """  # noqa: D205
    bits = values.to(torch.float32).reshape(-1).numpy().view(np.uint32).copy()
    bits &= np.uint32(0xFFFF0000)
    return torch.from_numpy(bits.view(np.float32)).reshape(values.shape)


def band(tensor, axis, part):
    """The band slice `part` of `tensor` along `axis`."""
    index = [slice(None)] * tensor.ndim
    index[axis] = part
    return tensor[tuple(index)]


def blend(previous, current, extent, axis, interpreted):
    """The decoded-overlap blend, every step in the tile's type, narrowed as
    the session's stores narrow: by truncation for BF16 on the interpreter, to
    nearest otherwise.
    """  # noqa: D205
    element = current.dtype
    extent = min(previous.shape[axis], current.shape[axis], extent)
    shape = [1] * current.ndim
    shape[axis] = extent
    if interpreted and element == torch.bfloat16:
        positions = truncated(torch.arange(extent, dtype=torch.float32))
        divisor = truncated(torch.tensor([float(extent)]))[0]
        ratio = truncated(positions / divisor)
        previous_weight = truncated(torch.ones_like(ratio) - ratio)
        leaving = band(previous, axis, slice(-extent, None)).to(torch.float32)
        arriving = band(current, axis, slice(0, extent)).to(torch.float32)
        blended = truncated(
            truncated(leaving * previous_weight.view(shape))
            + truncated(arriving * ratio.view(shape))
        )
    else:
        positions = torch.arange(extent, dtype=element)
        previous_weight = (1 - positions / extent).view(shape)
        current_weight = (positions / extent).view(shape)
        blended = (
            band(previous, axis, slice(-extent, None)) * previous_weight
            + band(current, axis, slice(0, extent)) * current_weight
        )
    blended = blended.to(element)
    if extent == current.shape[axis]:
        return blended
    return torch.cat((blended, band(current, axis, slice(extent, None))), axis)


def assert_same_bits(label, got, expected):
    """Hold `got` to the bits of `expected`, element for element."""
    width = {4: torch.int32, 2: torch.int16}[got.element_size()]
    got_bits = got.contiguous().reshape(-1).view(width)
    expected_bits = expected.contiguous().reshape(-1).view(width)
    differing = (got_bits != expected_bits).nonzero()
    if differing.numel():
        position = int(differing[0])
        raise AssertionError(
            f"{label}: {differing.numel()} of {got.numel()} values differ; "
            f"first at {position}: {float(got.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}"
        )


@pytest.mark.parametrize(
    ("dtype", "axis", "extent", "destination", "shape"), TILES
)
def test_a_tile_blends_bit_for_bit(
    dtype,
    axis,
    extent,
    destination,
    shape,
    torch_device,
    interpreted,
    kernel_source,
):
    alias, raster = DESTINATIONS[destination]
    element = DTYPES[dtype]
    torch.manual_seed(17)
    previous = torch.randn(shape, dtype=element)
    current = torch.randn(shape, dtype=element)
    # The reference's `current` as it was: an aliased launch writes over the
    # buffer it reads, which is the operator's point.
    original = current.clone()
    previous = previous.to(torch_device)
    current = current.to(torch_device)

    if raster:
        # A tile inside a wider raster: rows stay contiguous, the three outer
        # strides widen. The filler shows a write outside the tile.
        channels, frames, rows, columns = shape
        destination_buffer = torch.full(
            (channels, frames, rows, 2 * columns), 7.0, dtype=element
        ).to(torch_device)
        out = destination_buffer[..., :columns]
    else:
        destination_buffer = torch.full(shape, 7.0, dtype=element).to(
            torch_device
        )
        out = destination_buffer
    if alias:
        # The decoder keeps the tile it just decoded: `out` is `current`.
        out = current

    launch(
        kernel_source("cross_fade"), previous, current, out, extent, axis, BLOCK
    )

    expected = blend(previous.cpu(), original, extent, axis, interpreted)
    assert_same_bits(
        f"{dtype} axis {axis} extent {extent} {shape} {destination}",
        out.cpu(),
        expected,
    )
    if raster:
        outside = destination_buffer[..., shape[3] :].cpu()
        assert bool((outside == 7.0).all()), (
            f"{dtype} axis {axis} extent {extent}: "
            f"{int((outside != 7.0).sum())} raster elements outside the tile "
            "were written"
        )


@pytest.mark.parametrize("extent", [3, 5])
@pytest.mark.parametrize("axis", [2, 3])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_a_seam_read_from_strided_rasters_blends_bit_for_bit(
    dtype, axis, extent, torch_device, interpreted, kernel_source
):
    """The predecessor's last `extent` positions of the band axis and this
    tile's first `extent`, each a view inside the raster the tiles were
    decoded into, against the blend of contiguous copies of the two bands.
    """  # noqa: D205
    tile = [3, 2, 6, 8]
    element = DTYPES[dtype]
    torch.manual_seed(23)
    wide = list(tile)
    wide[axis] *= 2
    previous_raster = torch.randn(wide, dtype=element).to(torch_device)
    current_raster = torch.randn(wide, dtype=element).to(torch_device)
    along = tile[axis]
    previous = band(previous_raster, axis, slice(along - extent, along))
    current = band(current_raster, axis, slice(along - extent, along))
    seam = list(tile)
    seam[axis] = 2 * extent
    destination = torch.full(seam, 7.0, dtype=element).to(torch_device)
    out = band(destination, axis, slice(0, extent))

    launch(
        kernel_source("cross_fade"), previous, current, out, extent, axis, BLOCK
    )

    expected = blend(
        previous.cpu().contiguous(),
        current.cpu().contiguous(),
        extent,
        axis,
        interpreted,
    )
    assert_same_bits(
        f"{dtype} axis {axis} extent {extent} seam {tuple(out.shape)} read "
        "from strided rasters",
        out.cpu(),
        expected,
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("cross_fade")
    shape, extent, axis, block = (3, 4, 16, 32), 8, 2, 256

    def launched(inputs, outputs, pdl):
        launch(
            kernels,
            inputs["previous"],
            inputs["current"],
            outputs["out"],
            extent,
            axis,
            block,
            pdl,
        )
        return [outputs["out"]]

    case = chain.Case(
        "cross_fade",
        {
            "previous": torch.randn(shape).to(torch.bfloat16),
            "current": torch.randn(shape).to(torch.bfloat16),
        },
        {"out": torch.zeros(shape, dtype=torch.bfloat16)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"cross_fade launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
