"""Image decoder bindings: latents to pixels, quantized for image outputs.

The worker's ``DecoderRunner`` (``crates/worker/src/runner/decoder.rs``)
serves the ``ImageDecoder`` at ``path`` within a bound component.
``decoder.bind(prepared, loaded, path, buffers)`` binds the worker's
buffers once:

- ``latents``: the latent pool's step workspace ``[units, width]`` on the
  binding's device, into which the worker gathers each completed
  trajectory's pages before decoding (a decoder placed on a generation
  tower computes there and returns its results to these buffers);
- ``staging``: ``uint8`` bytes receiving each image's quantized HWC pixels,
  which the worker copies to the host and encodes;
- ``products``: the device's product arena (image products are regions of
  it, see ``uniserve_models.interface.regions``).

Step functions:

- ``decode(rows, images) -> counts``: ``rows`` holds one ``[offset, units,
  height, width, staging]`` row per image: its latent rows
  ``latents[offset:offset + units]``, its size and the staging offset of
  its pixels. Decodes every image (one numerical call per size), writes
  image ``i`` to the product region ``images[i]`` when given (the decoded
  values in the product's dtype), quantizes it into the staging bytes and
  returns each image's byte count, ``height * width * 3``.
- ``quantize(image, low, high, staging) -> count``: quantize a resident
  image product (a region holding values in ``[low, high]``) into the
  staging bytes at ``staging``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from uniserve_models.media import image
from uniserve_models.media.pixels import quantize_image_hwc
from uniserve_models.model import ImageDecoder

from . import calls, regions
from .errors import translate


def functions() -> dict[str, Callable[..., Any]]:
    return {"decoder.bind": bind}


def bind(
    prepared, loaded, path: str, buffers: Mapping[str, Any]
) -> dict[str, Callable[..., Any]]:
    """Bind the image decoder at ``path`` to the worker's buffers.

    Raises:
        UnsupportedCall: The module is not an ``ImageDecoder``.
    """
    decoder = calls.module(prepared, path)
    if not isinstance(decoder, ImageDecoder):
        raise NotImplementedError(
            f"{type(decoder).__qualname__} has no image decoder binding"
        )
    context = prepared.context
    if context is None:
        raise RuntimeError("decoder binding requires a bound component")
    tensors = calls.tensors(buffers)
    latents, staging = tensors["latents"], tensors["staging"]
    products = tensors["products"]
    value_range = tuple(float(value) for value in decoder.decoder.value_range)

    def stage(value: torch.Tensor, low: float, high: float, offset: int):
        """Quantize one image into the staging bytes at ``offset``."""
        pixels = quantize_image_hwc(value, value_range=(low, high))
        count = pixels.numel()
        if offset < 0 or offset + count > staging.numel():
            raise ValueError("image pixels exceed the decoder staging")
        staging[offset : offset + count].copy_(
            calls.local(pixels.reshape(-1), staging.device)
        )
        return count

    def decode(rows: Sequence[Sequence[int]], images: Sequence) -> list[int]:
        rows = [tuple(int(value) for value in row) for row in rows]
        if len(rows) != len(images) or any(len(row) != 5 for row in rows):
            raise ValueError("decoder rows and image products must align")
        with calls.scope(context):
            # A decoder placed on a generation tower reads its latents there.
            values = decoder.decode(
                tuple(
                    latents[offset : offset + units].to(
                        context.device, non_blocking=True
                    )
                    for offset, units, *_ in rows
                ),
                sizes=tuple(
                    image.Config(height, width)
                    for _, _, height, width, _ in rows
                ),
            )
            counts = []
            for value, row, region in zip(values, rows, images, strict=True):
                if region is not None:
                    calls.write(regions.view(products, region), value)
                counts.append(stage(value, *value_range, row[4]))
            return counts

    def quantize(region, low: float, high: float, offset: int) -> int:
        with calls.scope(context):
            return stage(
                regions.view(products, region),
                float(low),
                float(high),
                int(offset),
            )

    return {
        name: translate(step)
        for name, step in {"decode": decode, "quantize": quantize}.items()
    }
