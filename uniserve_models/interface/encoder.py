"""Encoder bindings: vision patch encoders, latent encoders, text encoders.

The worker's ``EncoderRunner`` (``crates/worker/src/runner/encoder.rs``)
serves one encoder call of a bound component: the module at ``path``
within the binding's root (``""`` for the root itself, ``"vision"`` for a
text encoder's vision tower bound with it). Two table entries serve it:

- ``encoder.describe(loaded, path, dtype)`` reports, before the encoder's
  component is bound, the facts the worker sizes the binding, the
  encoder's buffers and its calls with (``path`` is the module's path in
  the model).
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
- ``encode_patches(rows, grids, out) -> shapes``: encode vision blocks
  whose patch rows are already packed (a product region holding every
  block's rows back to back), one ``[time, height, width]`` patch grid and
  one product region per block.
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

Text (``TextEncoder``) functions bind ``tokens`` ``[capacity]`` int64,
``features`` ``[capacity, hidden]`` and, for an encoder that places vision
tokens, ``positions`` ``[3, capacity]`` int64; ``capacity`` is the largest
text capacity. The worker writes a prompt's token ids, zero-padded to the
call's capacity, into ``tokens``.

- ``text_positions(capacity, token_ids, image_grids, video_grids)`` writes
  the prompt's rotary coordinates into ``positions``.
- ``encode_text(capacity, count, graph, positions, visual, out) -> shape``
  encodes ``tokens[:capacity]`` (replaying ``graph`` when given), with the
  bound positions when ``positions`` is nonzero and the vision rows of the
  ``visual`` region when given, and writes the first ``count`` feature rows
  to ``out``. Causal attention keeps the padding out of those rows. On a
  pipeline stage without features the call writes nothing and returns an
  empty shape.
- ``capture_text(capacity, pool) -> graph`` captures the text-only encode
  of one capacity; ``release(graph)`` drops it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
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
        steps = _text(cast(TextEncoder, module), context, tensors)
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


def _grid(values: Sequence[int]) -> tuple[int, int, int]:
    """A vision block's patch grid as integers."""
    # The worker sends three extents per block; the encoder reads them.
    return cast("tuple[int, int, int]", tuple(int(value) for value in values))


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

    def encode_patches(rows, grids: Sequence, out: Sequence) -> list:
        shapes = [tuple(int(extent) for extent in grid) for grid in grids]
        if len(shapes) != len(out):
            raise ValueError("vision blocks and products must align")
        with calls.scope(context):
            # Each block is one packed sample: its patch rows and its grid,
            # which the encoder also takes as a [1, 3] device tensor.
            samples = regions.view(products, rows).split(
                [prod(shape) for shape in shapes]
            )
            features = encoder.encode(
                VisionInput(
                    tuple(samples),
                    tuple(_device_grid(shape, device) for shape in shapes),
                    tuple(shapes),
                )
            )
            return [
                calls.write(regions.view(products, region), value)
                for value, region in zip(features, out, strict=True)
            ]

    steps: dict[str, Callable[..., Any]] = {
        "encode": encode,
        "encode_patches": encode_patches,
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

    def capture_packed(slots: int, pool) -> graphs.Graph:
        slots = int(slots)
        with calls.scope(context):
            # Warm eagerly so lazy initialization stays out of the capture.
            run_packed(slots)
        return graphs.capture(
            lambda: run_packed(slots),
            stream=context.stream or torch.cuda.current_stream(device),
            pool=pool,
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


def _text(encoder: TextEncoder, context, tensors):
    products = tensors["products"]
    tokens, features = tensors["tokens"], tensors["features"]
    positions = tensors.get("positions")
    # Whether this pipeline stage computes the final features; captured
    # replays of a stage without them write nothing.
    final = {"stage": True}

    def compute(capacity: int, place: bool, visual) -> bool:
        options: dict[str, Any] = {}
        if place:
            if positions is None:
                raise ValueError("this text encoder places no vision tokens")
            options["positions"] = (positions[:, :capacity],)
        if visual is not None:
            options["visual"] = (visual,)
        values = encoder.encode((tokens[:capacity],), **options)
        final["stage"] = values is not None
        if values is None:
            # A pipeline stage without the final layers has no features.
            return False
        features[:capacity].copy_(values[0])
        return True

    def text_positions(
        capacity: int,
        token_ids: Sequence[int],
        image_grids: Sequence[Sequence[int]],
        video_grids: Sequence[Sequence[int]],
    ) -> None:
        if positions is None:
            raise ValueError("this text encoder places no vision tokens")
        capacity = int(capacity)
        values = encoder.positions(
            [int(token) for token in token_ids],
            image_grids=[_grid(grid) for grid in image_grids],
            video_grids=[_grid(grid) for grid in video_grids],
        )
        if values.shape != (3, capacity):
            raise ValueError("rotary coordinates must cover the capacity")
        with calls.scope(context):
            positions[:, :capacity].copy_(values, non_blocking=True)

    def encode_text(
        capacity: int, count: int, graph, place: int, visual, out
    ) -> list:
        capacity, count = int(capacity), int(count)
        if not 0 < count <= capacity <= tokens.numel():
            raise ValueError("text encoder input exceeds its token capacity")
        with calls.scope(context):
            if graph is not None:
                graph.replay()
                if not final["stage"]:
                    return []
            elif not compute(
                capacity,
                bool(place),
                None if visual is None else regions.view(products, visual),
            ):
                return []
            return calls.write(regions.view(products, out), features[:count])

    def capture_text(capacity: int, pool) -> graphs.Graph:
        capacity = int(capacity)
        with calls.scope(context):
            compute(capacity, False, None)
        return graphs.capture(
            lambda: compute(capacity, False, None),
            stream=context.stream or torch.cuda.current_stream(context.device),
            pool=pool,
        )

    return {
        "text_positions": text_positions,
        "encode_text": encode_text,
        "capture_text": capture_text,
        "release": lambda graph: graph.release(),
    }
