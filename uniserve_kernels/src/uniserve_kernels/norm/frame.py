"""Causal convolution padding of video frames, alone or normalized.

A causal 3D convolution over ``[batch, channels, frames, height, width]``
reads its input padded on both sides of height and width (reflected or
replicated) and preceded by zero frames. :func:`pad` writes that padded input
in one pass into channel-first or channels-last storage, reading the source
through its own strides, so channel-first, channels-last and permuted views
all work. The output equals the two-step ``F.pad`` result bit for bit: every
element is a copy of one source element or zero.

A causal video convolution after a pre-activation reads
``pad(silu(group_norm(frame)))`` for every frame. :func:`normalize_pad`
computes the moments of every frame's channel groups straight from the
values and writes the padded, normalized and activated input in one further
pass, into channel-first or channels-last storage. The result equals
PyTorch's composition (a frame-folding copy, ``F.group_norm``, ``F.silu``
and the causal padding) bit for bit: the native extension built from
``csrc/frame.cu`` reproduces PyTorch's CUDA moments reduction, normalization
branches and SiLU expression.

The kernels back ``uniserve.nn.functional.frame_pad`` and
``frame_norm_pad``, which validate the padding, allocate the output in its
storage order and raise on CUDA whenever :func:`unsupported` or
:func:`unsupported_normalize` reports a reason; the launchers do not
revalidate.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from uniserve_kernels.triton import (
    records_autograd,
    tl,
    triton,
    unsupported_operands,
)

# Output pixels along the width and channels each program writes.
_PAD_PIXELS = 32
_PAD_CHANNELS = 64


if triton is not None:  # pragma: no cover - depends on the accelerator stack.

    @triton.jit
    def _frame_pad_kernel(
        values,
        out,
        stride_batch,
        stride_channel,
        stride_frame,
        stride_height,
        stride_width,
        out_batch,
        out_channel,
        out_frame_stride,
        out_height_stride,
        out_width_stride,
        height,
        width,
        out_frames,
        out_height,
        out_width,
        channels,
        pad_top,
        pad_left,
        pad_front,
        reflect: tl.constexpr,
        block_pixels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Write one row segment of the padded output through its strides.

        Grid: (samples * out_frames * out_height, width blocks, channel
        blocks). Output frames before ``pad_front`` are zero; every other
        output pixel copies the reflected or replicated source pixel.
        """
        row_id = tl.program_id(0)
        out_row = row_id % out_height
        out_frame = (row_id // out_height) % out_frames
        sample = row_id // (out_height * out_frames)
        columns = tl.program_id(1) * block_pixels + tl.arange(0, block_pixels)
        channel = tl.program_id(2) * block_channels + tl.arange(
            0, block_channels
        )
        valid = (columns < out_width)[:, None] & (channel < channels)[None, :]

        # The block's columns are adjacent in channel-first storage and its
        # channels in channels-last storage, so either layout's stores
        # coalesce.
        destination = (
            out
            + sample.to(tl.int64) * out_batch
            + out_frame.to(tl.int64) * out_frame_stride
            + out_row.to(tl.int64) * out_height_stride
            + columns[:, None].to(tl.int64) * out_width_stride
            + channel[None, :].to(tl.int64) * out_channel
        )

        frame = out_frame - pad_front
        if frame < 0:
            tl.store(
                destination,
                tl.zeros((block_pixels, block_channels), dtype=tl.float32).to(
                    out.dtype.element_ty
                ),
                mask=valid,
            )
        else:
            source_row = out_row - pad_top
            source_columns = columns - pad_left
            if reflect:
                source_row = tl.where(source_row < 0, -source_row, source_row)
                source_row = tl.where(
                    source_row >= height,
                    2 * height - 2 - source_row,
                    source_row,
                )
                source_columns = tl.where(
                    source_columns < 0, -source_columns, source_columns
                )
                source_columns = tl.where(
                    source_columns >= width,
                    2 * width - 2 - source_columns,
                    source_columns,
                )
            else:
                source_row = tl.minimum(tl.maximum(source_row, 0), height - 1)
                source_columns = tl.minimum(
                    tl.maximum(source_columns, 0), width - 1
                )
            copied = tl.load(
                values
                + sample.to(tl.int64) * stride_batch
                + frame.to(tl.int64) * stride_frame
                + source_row.to(tl.int64) * stride_height
                + source_columns[:, None].to(tl.int64) * stride_width
                + channel[None, :].to(tl.int64) * stride_channel,
                mask=valid,
            )
            tl.store(destination, copied, mask=valid)


def unsupported(values: torch.Tensor, out: torch.Tensor) -> str | None:
    """Return why the kernel cannot pad ``values`` into ``out``, or ``None``.

    The input may use any strides. The output must be contiguous or
    channels-last storage of the input's dtype, batch and channels; the
    caller sizes its frames, rows and columns from the padding. The launch
    records no autograd graph.
    """
    reason = unsupported_operands(values, out)
    if reason is not None:
        return reason
    if values.ndim != 5 or out.ndim != 5:
        return "values and output are not [batch, channels, frames, h, w]"
    if out.dtype != values.dtype:
        return f"output dtype {out.dtype} differs from {values.dtype}"
    if not (
        out.is_contiguous()
        or out.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        return "the output is neither contiguous nor channels-last"
    if out.shape[:2] != values.shape[:2]:
        return "the output's batch and channels differ from the input's"
    return None


def pad(
    values: torch.Tensor,
    out: torch.Tensor,
    *,
    padding: tuple[int, int, int],
    reflect: bool,
) -> None:
    """Write ``values`` padded into ``out``.

    ``padding`` is ``(top, left, front)``: the rows above, columns left of
    and zero frames before the source within ``out``; the remaining rows and
    columns of ``out`` pad the bottom and right.
    """
    batch, channels, _, height, width = (int(size) for size in values.shape)
    _, _, out_frames, out_height, out_width = (int(size) for size in out.shape)
    _frame_pad_kernel[
        (
            batch * out_frames * out_height,
            triton.cdiv(out_width, _PAD_PIXELS),
            triton.cdiv(channels, _PAD_CHANNELS),
        )
    ](
        values,
        out,
        *(int(stride) for stride in values.stride()),
        *(int(stride) for stride in out.stride()),
        height,
        width,
        out_frames,
        out_height,
        out_width,
        channels,
        *padding,
        reflect,
        _PAD_PIXELS,
        _PAD_CHANNELS,
        num_warps=4,
    )


@lru_cache(maxsize=1)
def _extension():
    from uniserve_kernels import jit

    # The sources compile without fast math, as PyTorch's normalization and
    # SiLU kernels do, whose arithmetic they reproduce bit for bit.
    return jit.load(
        "uniserve_frame_norm",
        [Path(__file__).parent / "csrc" / "frame.cu"],
        cuda_flags=(
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            jit.device_architecture(),
        ),
    )


def load() -> None:
    """Compile or load the normalization extension for the current device."""
    _extension()


def unsupported_normalize(
    values: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> str | None:
    """Return why the kernels cannot normalize ``values`` into ``out``.

    The values are CUDA float32 whose frames store each channel's
    ``height * width`` pixels contiguously, such as channel-first or
    frame-major storage; the output is contiguous or channels-last float32
    storage of the values' batch and channels; ``weight`` and ``bias``, when
    present, are contiguous float32 per-channel vectors. Returns ``None``
    when every condition holds.
    """
    operands = (values, out, weight, bias)
    present = tuple(tensor for tensor in operands if tensor is not None)
    if any(tensor.device != values.device for tensor in present):
        return "operands reside on different devices"
    if values.device.type != "cuda":
        return f"operands reside on {values.device}, not a CUDA device"
    if records_autograd(*present):
        return (
            "autograd is recording operands that require grad, and the "
            "kernel defines no backward; call under torch.inference_mode()"
        )
    if any(tensor.dtype != torch.float32 for tensor in present):
        return "the kernels normalize float32 values"
    if values.ndim != 5 or out.ndim != 5:
        return "values and output are not [batch, channels, frames, h, w]"
    if values.stride(4) != 1 or values.stride(3) != values.shape[4]:
        return "a frame's pixels of one channel are not contiguous"
    if not (
        out.is_contiguous()
        or out.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        return "the output is neither contiguous nor channels-last"
    if out.shape[:2] != values.shape[:2]:
        return "the output's batch and channels differ from the input's"
    for parameter in (weight, bias):
        if parameter is not None and (
            parameter.shape != (values.shape[1],)
            or not parameter.is_contiguous()
        ):
            return "normalization parameters are not per-channel vectors"
    return None


def normalize_pad(
    values: torch.Tensor,
    out: torch.Tensor,
    *,
    groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    padding: tuple[int, int, int],
    reflect: bool,
) -> None:
    """Write ``values`` normalized, activated and padded into ``out``.

    ``padding`` is ``(top, left, front)``: the rows above, columns left of
    and zero frames before the source within ``out``; the remaining rows and
    columns of ``out`` pad the bottom and right, reflected or replicated.
    The moments occupy a ``[batch * frames, groups]`` FP32 intermediate.
    """
    extension = _extension()
    mean, rstd = extension.frame_moments(values, groups, eps)
    top, left, front = padding
    extension.frame_norm_pad(
        values, mean, rstd, weight, bias, out, groups, top, left, front, reflect
    )
