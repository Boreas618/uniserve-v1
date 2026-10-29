"""Numerical conversion of retained fields using caller-owned storage."""

from __future__ import annotations

import json
from collections.abc import Sequence
from math import prod

import torch
import triton
import triton.language as tl

from uniserve.prefix_cache.units import stream_scope

from .retained import from_canonical, restore_tensor, to_canonical


@triton.jit
def _copy_field(
    stored,
    canonical,
    count: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    order: tl.constexpr,
    layout: tl.constexpr,
    to_linear: tl.constexpr,
    block: tl.constexpr,
):
    # Each lane owns one canonical byte. Only rows and block-scale tiles
    # move: packed FP4 pairs and FP8 bit patterns are never decoded.
    index = tl.program_id(0) * block + tl.arange(0, block)
    row, column = index // width, index % width
    matrix, local = row // height, row % height
    if order == "interleaved-64":
        local = (
            local % (height // 2) // 64 * 128
            + local // (height // 2) * 64
            + local % 64
        )
    elif order == "interleaved-16" or order == "interleaved-8":
        row_block: tl.constexpr = 16 if order == "interleaved-16" else 8
        local = (
            local % (height // 2) // row_block * (2 * row_block)
            + (1 - local // (height // 2)) * row_block
            + local % row_block
        )
    else:
        if order == "interleaved-shuffled-128":
            local = local % (height // 2) * 2 + local // (height // 2)
        if order == "shuffled-128" or order == "interleaved-shuffled-128":
            local = local // 32 * 32 + local % 4 * 8 + local % 32 // 4
    row = matrix * height + local
    if layout == "linear":
        offset = row * width + column
    else:
        if layout == "128x4-mn":
            tile = (matrix * tl.cdiv(width, 4) + column // 4) * (
                height // 128
            ) + local // 128
        else:
            tile = row // 128 * tl.cdiv(width, 4) + column // 4
        offset = tile * 512 + row % 32 * 16 + row % 128 // 32 * 4 + column % 4
    if to_linear:
        value = tl.load(stored + offset, index < count, other=0)
        tl.store(canonical + index, value, index < count)
    else:
        value = tl.load(canonical + index, index < count, other=0)
        tl.store(stored + offset, value, index < count)


def _copy_cuda(parameters, stored, canonical, to_linear):
    """Reorder borrowed buffers without allocating device scratch."""
    shape = parameters["shape"]
    for name, value in stored.items():
        linear = canonical[name]
        if name == "tensor_scale":
            destination, source = (
                (linear, value) if to_linear else (value, linear)
            )
            destination.copy_(source.reshape_as(destination))
            continue
        scales = name != "values"
        layout = parameters["scale_layout"] if scales else "linear"
        if not to_linear and scales and layout != "linear":
            # Canonical fields omit scale padding. Direct loading initializes
            # the stored padding to zero, so conversion does the same.
            value.zero_()
        width = linear.numel() // prod(shape[:-1])
        _copy_field[(triton.cdiv(linear.numel(), 1024),)](
            value.view(torch.uint8),
            linear.view(torch.uint8),
            linear.numel(),
            shape[-2],
            width,
            parameters["row_order"],
            layout,
            to_linear,
            1024,
        )


def convert(
    description: str,
    fields: Sequence[torch.Tensor],
    staging: torch.Tensor,
    to_linear: bool,
    stream: int,
) -> None:
    """Copy a declared encoded group to or from its canonical byte buffer.

    ``description`` contains the group's manifest fields, including the
    numerical parameters and canonical offsets. ``fields`` borrow its
    stored tensors in the same order. The caller owns both buffers and the
    CUDA stream, and waits for that stream before publishing the result.
    No dequantization occurs: the encoded values and scales are reordered.
    """
    described = json.loads(description)
    parameters = described[0]["encoding"]["canonical"]["parameters"]
    linear = dict(parameters, scale_layout="linear", row_order="linear")
    stored, canonical = {}, {}
    for field, value in zip(described, fields, strict=True):
        form = field["encoding"]["canonical"]
        if form["parameters"] != parameters:
            raise ValueError(
                "encoded fields disagree on their numerical tensor"
            )
        name = field["tensor"].rsplit("/", 1)[1]
        stored[name] = value
        dtype = getattr(torch, field["dtype"])
        canonical[name] = (
            staging[form["offset"] :]
            .view(dtype)
            .as_strided(form["shape"], form["strides"])
        )
    with stream_scope(stream, staging.device)():
        if staging.is_cuda:
            _copy_cuda(parameters, stored, canonical, to_linear)
            return
        if to_linear:
            result = to_canonical(restore_tensor(parameters, stored))
            outputs = canonical
        else:
            scale = (
                "block_scale" if parameters["format"] == "nvfp4" else "scale"
            )
            canonical[scale] = canonical[scale].reshape(
                -1, canonical[scale].shape[-1]
            )
            result = from_canonical(
                restore_tensor(linear, canonical), parameters
            )
            outputs = stored
        for name, value in result.buffers().items():
            outputs[name].copy_(value.reshape_as(outputs[name]))
