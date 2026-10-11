"""Encoder bindings of image models: vision patch and latent encoders.

The worker's ``EncoderRunner`` (``crates/worker/src/runner/encoder.rs``)
serves one encoder call of a bound component: the module at ``path``
within the binding's root (``""`` for the root itself). Two table entries
serve it:

- ``encoder.describe(loaded, path, dtype)`` reports, before the encoder's
  component is bound, the facts the worker sizes the binding, the
  encoder's buffers and its calls with (``path`` is the module's path in
  the model). It also describes a text encoder, whose forward is
  ``uniserve_models.runner.encoder.EncoderRunner``.
- ``encoder.bind(prepared, loaded, path, buffers)`` binds the worker's
  buffers once and returns the encoder's step functions.

``prepared`` is the bound component (``component.bind``), whose execution
context runs every call on the worker's stream for the binding; ``loaded``
is the loaded model, whose image policy prepares resident images.

Every binding receives ``products``: the ``uint8`` product arena of the
binding's device (see ``uniserve_models.interface.regions``). Products are
named by regions of it. A call writes each result into its product region
in the region's dtype, and returns the result's shape, which the worker
publishes. Kind-specific buffers are listed with each kind below.

Images reach an encoder as opaque handles of ``PreparedImage`` values:
host-prepared inline images (``image.prepare_input``, in page-locked
memory) or device-prepared resident images (``prepare_resident``). The
encoder copies host pixels to its device on its stream; the caching host
allocator keeps their page-locked blocks until those copies complete.

Vision (``PatchEncoder``) functions:

- ``encode(images, out) -> shapes``: encode unpacked images, one product
  region per image.
- ``prepare_resident(image, tower, signed_unit) -> prepared``: prepare a
  resident image product (a region) for encoding.
- ``warm(side)``: encode a black ``side`` x ``side`` image through the
  model's input transform and discard the result (startup warmup of
  unpacked encoders).
- Packed encoders (``max_patches`` set) bind ``pixels`` ``[slots *
  max_patches, row]``, ``grids`` ``[2 * slots, 2]`` int64 and
  ``features`` ``[slots * tokens, output_size]``, where ``tokens`` is
  ``max_patches // downsample**2``. The worker writes the slot grids
  (``PatchEncoder.packed_grids``) into ``grids`` before each call.
  ``encode_packed(images, slots, graph, out) -> shapes`` copies image
  ``i``'s patch rows to the front of slot ``i``, encodes ``slots`` slots
  (replaying ``graph`` when given) and writes each image's features to its
  region. ``capture_packed(slots, pool) -> graph`` captures that encode
  for ``slots`` slots; ``release(graph)`` drops it.

Latent (``PatchAutoencoder``) functions: ``encode(images, out) -> shapes``
encodes images of one pixel shape into latent patch rows; ``warm`` and
``prepare_resident`` as above. A latent encoder that samples its posterior
draws from the device's default generator, as every encode does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch

from uniserve_models.media.pixels import (
    PreparedImage,
    Tower,
    prepare_tensor_image,
)
from uniserve_models.model import PatchEncoder, TextEncoder, VisionInput
from uniserve_models.nn.vae import PatchAutoencoder
from uniserve_models.processing import PatchTransform

from . import calls, graphs, regions
from .errors import binding
from .image import image_facts


def functions() -> dict[str, Callable[..., Any]]:
    return {"encoder.describe": describe, "encoder.bind": bind}


def _kind(module) -> str:
    if isinstance(module, PatchEncoder):
        return "vision"
    if isinstance(module, PatchAutoencoder):
        return "latent"
    if isinstance(module, TextEncoder):
        return "text"
    raise NotImplementedError(
        f"{type(module).__qualname__} has no worker encoder binding"
    )


def _defines(module, method: str, base: type) -> bool:
    """Whether ``module`` overrides ``base``'s optional ``method``."""
    return getattr(type(module), method) is not getattr(base, method)


def _max_patches(processor, patch_size: int) -> int:
    """Patch rows of the largest image the processor's vit tower admits."""
    transform = None if processor is None else processor.vit
    if transform is None:
        return 0
    pixels = (
        transform.pixel_bound()
        if isinstance(transform, PatchTransform)
        else min(transform.resize.max_pixels, transform.resize.max_size**2)
    )
    return pixels // patch_size**2


def describe(loaded, path: str, dtype: str) -> dict[str, Any]:
    """Facts of the encoder at module ``path`` of the loaded model.

    ``dtype`` is the deployment's model dtype, which a text encoder's hidden
    states carry. Planned before the encoder's component is bound.

    Returns:
        ``kind`` (``"vision"``, ``"latent"`` or ``"text"``) and:

        - vision: ``patch_size``, ``downsample`` (input patches per feature
          on each axis), ``output_size``, ``output_dtype``, ``max_patches``
          (patch rows of one packed slot, 0 for an unpacked encoder),
          ``image_patches`` (patch rows of the largest image the image
          policy admits), ``row_width`` (values per packed patch row) and
          ``pixel_dtype`` (the image policy's output dtype).
        - latent: ``downsample`` (pixels per latent patch on each axis),
          ``width`` (values per latent patch row) and ``output_dtype``.
        - text: ``hidden_size``, ``output_dtype`` and ``positions``
          (whether the encoder places vision tokens with rotary
          coordinates).

    Raises:
        UnsupportedCall: The module has no worker encoder binding.
    """
    path = str(path)
    module = loaded.module.get_submodule(path) if path else loaded.module
    kind = _kind(module)
    if kind == "vision":
        encoder = cast(PatchEncoder, module)
        processor = getattr(loaded, "image_processor", None)
        pixel_dtype = (
            None if processor is None else processor.output_dtype
        ) or torch.float32
        layout = encoder.features_layout(1)["features"]
        return {
            "kind": kind,
            "patch_size": encoder.patch_size,
            "downsample": encoder.downsample,
            "output_size": int(layout.shape[1]),
            "output_dtype": str(layout.dtype).removeprefix("torch."),
            "max_patches": encoder.max_patches or 0,
            "image_patches": _max_patches(processor, encoder.patch_size),
            "row_width": 3 * encoder.patch_size**2,
            "pixel_dtype": str(pixel_dtype).removeprefix("torch."),
        }
    if kind == "latent":
        autoencoder = cast(PatchAutoencoder, module)
        return {
            "kind": kind,
            "downsample": autoencoder.downsample,
            "width": autoencoder.latent_channels * autoencoder.patch_size**2,
            "output_dtype": str(autoencoder.latent_dtype).removeprefix(
                "torch."
            ),
        }
    text_encoder = cast(TextEncoder, module)
    return {
        "kind": kind,
        "hidden_size": int(text_encoder.network.hidden_size),
        "output_dtype": str(dtype).removeprefix("torch."),
        "positions": _defines(text_encoder, "positions", TextEncoder),
    }


def bind(
    prepared, loaded, path: str, buffers: Mapping[str, Any]
) -> dict[str, Callable[..., Any]]:
    """Bind the encoder at ``path`` to the worker's buffers.

    Returns:
        The kind's step functions (see the module documentation), each
        translating its failures into interface error kinds.
    """
    module = calls.module(prepared, path)
    kind = _kind(module)
    tensors = calls.tensors(buffers)
    context = prepared.context
    if context is None:
        raise RuntimeError("encoder binding requires a bound component")
    if kind == "vision":
        steps = _vision(cast(PatchEncoder, module), loaded, context, tensors)
    elif kind == "latent":
        steps = _latent(
            cast(PatchAutoencoder, module), loaded, context, tensors
        )
    else:
        raise NotImplementedError(
            "a text encoder computes through its runner, not a binding"
        )
    return binding(steps)


def _device_pixels(image: PreparedImage, device: torch.device) -> torch.Tensor:
    # Host pixels are page-locked; the copy is asynchronous on the stream.
    return image.pixels.to(device, non_blocking=True)


def _device_grid(
    shape: tuple[int, ...] | None, device: torch.device
) -> torch.Tensor | None:
    """The ``[1, axes]`` int64 patch grid of a packed sample on ``device``.

    Built in page-locked memory first, so the copy does not synchronize the
    host with the stream.
    """
    if shape is None:
        return None
    grid = torch.tensor([shape], dtype=torch.long)
    if device.type == "cuda":
        grid = grid.pin_memory()
    return grid.to(device, non_blocking=True)


def _warm(loaded, context, tower: Tower, run):
    """The ``warm`` step: ``run`` one black image prepared for ``tower``."""

    def warm(side: int) -> None:
        processor = getattr(loaded, "image_processor", None)
        if processor is None:
            raise ValueError("the model declares no image processing")
        side = int(side)
        with calls.scope(context):
            prepared = prepare_tensor_image(
                processor,
                tower,
                torch.zeros((3, side, side)),
                device=context.device,
                signed_unit=False,
            )
            run(prepared)

    return warm


def _resident(loaded, context, products):
    """The ``prepare_resident`` step of an image encoder."""

    def prepare_resident(image, tower: str, signed_unit: int) -> dict:
        processor = getattr(loaded, "image_processor", None)
        if processor is None:
            raise ValueError("the model declares no image processing")
        with calls.scope(context):
            prepared = prepare_tensor_image(
                processor,
                # The worker sends a plain string; the selection rejects
                # unknown towers.
                cast(Tower, str(tower)),
                regions.view(products, image),
                device=context.device,
                signed_unit=bool(signed_unit),
            )
        return image_facts(prepared)

    return prepare_resident


def _vision(encoder: PatchEncoder, loaded, context, tensors):
    products = tensors["products"]
    device = context.device

    def run(images: Sequence[PreparedImage]):
        return encoder.encode(
            VisionInput(
                tuple(_device_pixels(image, device) for image in images),
                tuple(
                    _device_grid(image.grid_shape, device) for image in images
                ),
                tuple(image.grid_shape for image in images),
            )
        )

    def encode(images: Sequence[PreparedImage], out: Sequence) -> list:
        if len(images) != len(out):
            raise ValueError("vision images and products must align")
        with calls.scope(context):
            features = run(images)
            return [
                calls.write(regions.view(products, region), value)
                for value, region in zip(features, out, strict=True)
            ]

    steps: dict[str, Callable[..., Any]] = {
        "encode": encode,
        "prepare_resident": _resident(loaded, context, products),
        "warm": _warm(loaded, context, "vit", lambda image: run((image,))),
    }
    if encoder.max_patches is None:
        return steps

    capacity = encoder.max_patches
    tokens = capacity // encoder.downsample**2
    pixels, grids, features = (
        tensors["pixels"],
        tensors["grids"],
        tensors["features"],
    )

    def run_packed(slots: int) -> None:
        features[: slots * tokens].copy_(
            encoder.encode_packed(
                pixels[: slots * capacity], grids[: 2 * slots]
            )
        )

    def encode_packed(
        images: Sequence[PreparedImage], slots: int, graph, out: Sequence
    ) -> list:
        slots = int(slots)
        if len(images) != len(out) or len(images) > slots:
            raise ValueError("packed images must fit their slots")
        with calls.scope(context):
            # Image i's rows fill the front of slot i; rows past an image
            # keep earlier finite values, which only padding reads.
            for index, image in enumerate(images):
                rows = image.pixels.shape[0]
                if image.grid_shape is None or rows > capacity:
                    raise ValueError(
                        "packed vision images require patch rows within "
                        "their slot"
                    )
                pixels[index * capacity : index * capacity + rows].copy_(
                    image.pixels, non_blocking=True
                )
            if graph is None:
                run_packed(slots)
            else:
                graph.replay()
            shapes = []
            for index, (image, region) in enumerate(
                zip(images, out, strict=True)
            ):
                # The copy loop above rejected images without a patch grid.
                assert image.grid_shape is not None
                height, width = image.grid_shape
                count = height * width // encoder.downsample**2
                start = index * tokens
                shapes.append(
                    calls.write(
                        regions.view(products, region),
                        features[start : start + count],
                    )
                )
            return shapes

    def capture_packed(slots: int, pool) -> list[Any]:
        slots = int(slots)
        with calls.scope(context):
            # Warm eagerly so lazy initialization stays out of the capture.
            run_packed(slots)
        return graphs.export(
            graphs.capture(
                lambda: run_packed(slots),
                stream=context.stream or torch.cuda.current_stream(device),
                pool=pool,
            )
        )

    steps.update(
        encode_packed=encode_packed,
        capture_packed=capture_packed,
        release=lambda graph: graph.release(),
    )
    return steps


def _latent(autoencoder: PatchAutoencoder, loaded, context, tensors):
    products = tensors["products"]
    device = context.device

    def run(images: Sequence[PreparedImage]):
        pixels = torch.stack(
            tuple(_device_pixels(image, device) for image in images)
        )
        return autoencoder.encode(pixels).unbind(0)

    def encode(images: Sequence[PreparedImage], out: Sequence) -> list:
        if len(images) != len(out):
            raise ValueError("latent images and products must align")
        with calls.scope(context):
            latents = run(images)
            return [
                calls.write(regions.view(products, region), value)
                for value, region in zip(latents, out, strict=True)
            ]

    return {
        "encode": encode,
        "prepare_resident": _resident(loaded, context, products),
        "warm": _warm(loaded, context, "vae", lambda image: run((image,))),
    }
