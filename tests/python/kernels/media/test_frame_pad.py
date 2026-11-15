"""The frame_pad program against the two-step pad of a video frame batch.

The source is ``[batch, channels, frames, height, width]``; the output is the
same batch and channels over more frames, rows and columns, and equals the
two-step ``F.pad`` result element for element:

- the ``front`` output frames before the source are zeros: the pad is causal,
  so nothing may be copied into them;
- every other output pixel copies one source pixel, its row and column
  reflected or replicated.

The domain is PyTorch's: a pad is narrower than the dimension it pads, which
is what makes one fold of ``2·dim - 2 - i`` reach every padded index. The
operator's tolerance is ``exact()``: the program copies or stores a zero, so
every case is an equality.

The cases, each named for the mistake it catches:

- **the fixture's cell**: ``f32``, reflect, 512-pixel and 4-channel blocks on
  a shape whose output width and channel count both leave a masked tail;
- **replicate against reflect** on one shape: the two differ at every padded
  pixel, so a kernel that clamped where it should fold (or the other way)
  cannot agree with both. Two samples, two channels and two frames make the
  batch, channel and frame strides three different numbers;
- **a pad of ``dim - 1``**, the widest reflection one fold reaches, where an
  off-by-one in ``2·dim - 2 - i`` shows;
- **zero ``front``**, where no frame is causal-padded, and **``front`` past
  the source's frames**, where every copied frame follows zero frames;
- **a channels-last output**, the other legal layout; **a strided source**,
  whose batch, channel and frame strides the kernel must read rather than
  assume; and **a channels-last source**, whose width stride is the channel
  count, the read a kernel that took the columns as contiguous gets wrong.
"""

import dataclasses

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytestmark = pytest.mark.unit

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


@dataclasses.dataclass(frozen=True)
class Pad:
    """One case: the element type, the mode, the blocks one program covers,
    the source's shape, the three pads and the two views' layouts.
    """  # noqa: D205

    dtype: str
    reflect: bool
    blocks: tuple[int, int]
    shape: tuple[int, int, int, int, int]
    top: int
    left: int
    front: int
    channels_last: bool = False
    strided_source: bool = False
    channels_last_source: bool = False


CASES = {
    "the fixture's cell": Pad("f32", True, (512, 4), (1, 3, 2, 4, 3), 2, 2, 3),
    "the fixture's cell in bf16": Pad(
        "bf16", True, (512, 4), (1, 3, 2, 4, 3), 1, 1, 1
    ),
    "reflect": Pad("f32", True, (4, 1), (2, 2, 2, 4, 5), 2, 2, 1),
    "replicate": Pad("f32", False, (4, 1), (2, 2, 2, 4, 5), 2, 2, 1),
    "a three-wide reflection": Pad(
        "f32", True, (2, 2), (1, 2, 1, 3, 3), 2, 2, 2
    ),
    "no front padding": Pad("f16", True, (8, 8), (1, 2, 3, 4, 4), 1, 1, 0),
    "every frame behind the front": Pad(
        "f32", False, (8, 4), (1, 2, 2, 4, 4), 1, 1, 3
    ),
    "a channels-last output": Pad(
        "f32", True, (4, 2), (1, 2, 1, 3, 3), 1, 2, 1, channels_last=True
    ),
    "a strided source": Pad(
        "f32", True, (4, 2), (2, 2, 2, 3, 3), 2, 1, 1, strided_source=True
    ),
    "a channels-last source": Pad(
        "f32", True, (4, 2), (1, 3, 2, 3, 4), 1, 2, 1, channels_last_source=True
    ),
    "a channels-last source and output": Pad(
        "bf16",
        False,
        (2, 2),
        (1, 2, 2, 3, 3),
        2,
        1,
        1,
        channels_last=True,
        channels_last_source=True,
    ),
}

#: The seed every case draws its pixels from.
SEED = 29


def launch(kernels, values, out, top, left, front, reflect, blocks):
    """One launch writing every pixel of `out` through both views' strides;
    `blocks` are the pixels and the channels one program covers.
    """  # noqa: D205
    batch, channels, out_frames, out_height, out_width = out.shape
    height, width = values.shape[3:]
    block_pixels, block_channels = blocks
    kernels._frame_pad_kernel[
        (
            batch * out_frames * out_height,
            -(-out_width // block_pixels),
            -(-channels // block_channels),
        )
    ](
        values,
        out,
        *values.stride(),
        *out.stride(),
        height,
        width,
        out_frames,
        out_height,
        out_width,
        channels,
        top,
        left,
        front,
        reflect=reflect,
        block_pixels=block_pixels,
        block_channels=block_channels,
    )


def reference(values, *, top, left, front, out_shape, reflect):
    """The two-step pad: zero frames at the front, then the spatial pad of
    every plane.
    """  # noqa: D205
    batch, channels, frames, height, width = values.shape
    out_frames, out_height, out_width = out_shape
    padded = torch.cat(
        [
            torch.zeros(
                batch, channels, front, height, width, dtype=values.dtype
            ),
            values,
        ],
        dim=2,
    )
    planes = padded.reshape(batch * channels * out_frames, 1, height, width)
    planes = F.pad(
        planes,
        (left, out_width - width - left, top, out_height - height - top),
        mode="reflect" if reflect else "replicate",
    )
    return planes.reshape(batch, channels, out_frames, out_height, out_width)


def source(case, element, device):
    """The case's source on `device` in the layout it names, distinct in every
    pixel so a pixel copied from the wrong place cannot pass.
    """  # noqa: D205
    batch, channels, frames, height, width = case.shape
    torch.manual_seed(SEED)
    base = torch.randn(
        batch, channels, frames, height, width, dtype=torch.float32
    ).to(element)
    if case.strided_source:
        # A view of a wider tensor: its batch, channel and frame strides are
        # each twice the packed ones, while the width stays contiguous.
        wide = torch.randn(
            batch, channels * 2, frames * 2, height, width, dtype=torch.float32
        )
        return wide.to(element).to(device)[:, :channels, ::2]
    if case.channels_last_source:
        # Stored channels-last and viewed in the operator's logical shape: a
        # row of pixels is `channels` elements apart.
        packed = base.permute(0, 2, 3, 4, 1).contiguous().to(device)
        values = packed.permute(0, 4, 1, 2, 3)
        assert values.stride(4) == channels and values.stride(1) == 1, (
            f"the channels-last source has strides {values.stride()}"
        )
        return values
    return base.to(device)


def destination(case, element, out_shape, device):
    """A NaN-filled output on `device` in the layout the case names.

    Channels-last is the operator's own definition (``frame.rs``): the channel
    strides closest to the width, the shape ``(B, F, H, C, W)`` viewed as
    ``(B, C, F, H, W)``.
    """
    batch, channels = case.shape[:2]
    out_frames, out_height, out_width = out_shape
    if not case.channels_last:
        return torch.full(
            (batch, channels, *out_shape),
            float("nan"),
            dtype=element,
            device=device,
        )
    out = torch.full(
        (batch, out_frames, out_height, channels, out_width),
        float("nan"),
        dtype=element,
        device=device,
    ).permute(0, 3, 1, 2, 4)
    assert out.stride() == (
        channels * out_frames * out_height * out_width,
        out_width,
        channels * out_height * out_width,
        channels * out_width,
        1,
    ), f"the channels-last view has strides {out.stride()}"
    return out


@pytest.mark.parametrize("name", CASES)
def test_the_pad_matches_the_two_step_pad_bit_for_bit(
    name, torch_device, kernel_source
):
    case = CASES[name]
    element = DTYPES[case.dtype]
    _, _, frames, height, width = case.shape
    top, left, front = case.top, case.left, case.front
    out_shape = (frames + front, height + top + 1, width + left + 2)
    values = source(case, element, torch_device)
    out = destination(case, element, out_shape, torch_device)

    launch(
        kernel_source("frame_pad"),
        values,
        out,
        top,
        left,
        front,
        case.reflect,
        case.blocks,
    )

    expected = reference(
        values.cpu(),
        top=top,
        left=left,
        front=front,
        out_shape=out_shape,
        reflect=case.reflect,
    )
    got = out.cpu()
    differing = got != expected
    if differing.any():
        position = int(differing.reshape(-1).nonzero()[0])
        pytest.fail(
            f"{name}: {case.dtype} "
            f"{'reflect' if case.reflect else 'replicate'}, "
            f"{case.shape} -> {out_shape}, pads top {top} left {left} "
            f"front {front}: {int(differing.sum())} of {got.numel()} elements "
            f"differ; first at {position}: {float(got.reshape(-1)[position])} "
            f"vs {float(expected.reshape(-1)[position])}"
        )
