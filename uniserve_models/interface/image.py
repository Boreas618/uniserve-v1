"""A model's image policy, as the worker plans image calls with it.

- ``image.describe`` reports the image facts of a loaded model: its image
  processing (towers, feature injection), its image denoiser's latent
  layout and its image decoder's value range.
- ``image.prepare_input`` decodes one inline request image and applies the
  model's image policy on the host (``uniserve_models.media.pixels``). It
  touches no device and keeps no state, so the worker calls it from a host
  lane thread while other batches execute. The prepared image crosses back
  as an opaque handle; the worker passes it unchanged to the encoder
  binding that copies it to the device
  (``uniserve_models.interface.encoder``).
- ``image.patch_grid`` reports the patch grid the vision tower divides an
  image into, which places its features' positions.
- ``image.guidance_prefix`` tokenizes a guidance branch's prefix with the
  model's prompt framing and tokenizer.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from uniserve_models.media.pixels import (
    PreparedImage,
    Tower,
    prepare_host_image,
)
from uniserve_models.model import ImageDecoder, ImageDenoiser
from uniserve_models.processing import (
    FeatureInjection,
    FlowPrompt,
    PatchTransform,
    load_tokenizer,
)

from .loading import capability


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "image.describe": describe_model,
        "image.prepare_input": prepare_input,
        "image.patch_grid": patch_grid,
        "image.guidance_prefix": guidance_prefix,
    }


def _dtype(value) -> str:
    return str(value).removeprefix("torch.")


def _injection(injection: FeatureInjection | None) -> dict[str, Any] | None:
    if injection is None:
        return None
    return {
        "layout": injection.layout.value,
        "positions": injection.positions.value,
        "start_token": injection.start_token_id,
        "end_token": injection.end_token_id,
    }


def describe_model(loaded) -> dict[str, Any]:
    """The image facts of a loaded model.

    Returns:
        ``{"processor", "denoiser", "decoder_range"}``:

        - ``processor``: None without image processing, else ``{"vit",
          "vae"}`` (whether each tower is declared), ``patch_tower``
          (whether the vit tower packs patches), ``pixel_dtype`` and
          ``injection`` (``{"layout", "positions", "start_token",
          "end_token"}`` or None).
        - ``denoiser``: None without an image denoiser, else its latent
          layout: ``patch_size``, ``latent_channels``, ``downsample``
          (pixels per latent patch on each axis), ``framing_tokens``,
          ``image_unconditional`` (the branch source of its
          image-unconditional guidance), ``max_sequence_tokens`` (latent
          patch rows, framing excluded) and ``prediction_dtype``.
        - ``decoder_range``: the image decoder's ``[low, high]`` pixel
          range, or None without an image decoder.
    """
    model = loaded.module
    processor = loaded.image_processor
    # mypy rejects an abstract class as a ``type`` argument; the lookup
    # only tests membership.
    denoiser = capability(model, ImageDenoiser)  # type: ignore[type-abstract]
    decoder = capability(model, ImageDecoder)
    return {
        "processor": None
        if processor is None
        else {
            "vit": processor.vit is not None,
            "vae": processor.vae is not None,
            "patch_tower": isinstance(processor.vit, PatchTransform),
            "pixel_dtype": _dtype(processor.output_dtype or "float32"),
            "injection": _injection(processor.feature_injection),
        },
        "denoiser": None
        if denoiser is None
        else {
            "patch_size": int(denoiser.patch_size),
            "latent_channels": int(denoiser.latent_channels),
            "downsample": int(denoiser.downsample),
            "framing_tokens": int(denoiser.framing_tokens),
            "image_unconditional": denoiser.image_unconditional.value,
            "max_sequence_tokens": int(denoiser.max_sequence_tokens),
            "prediction_dtype": _dtype(denoiser.prediction_dtype),
        },
        "decoder_range": None
        if decoder is None
        else [float(value) for value in decoder.decoder.value_range],
    }


def patch_grid(loaded, height: int, width: int, input_images: int) -> list:
    """The vit tower's ``[rows, columns]`` patch grid of an image.

    ``input_images`` is the request's input image count for a request input
    image, or 0 for a generated image, which keeps the single-image pixel
    bound.

    Raises:
        ValueError: The model has no patch tower, or its resize policy
            rejects the dimensions.
    """
    processor = loaded.image_processor
    transform = None if processor is None else processor.vit
    if not isinstance(transform, PatchTransform):
        raise ValueError(
            "temporal-spatial feature injection requires a patch image "
            "transform"
        )
    count = int(input_images)
    return list(transform.grid_shape(int(height), int(width), count or None))


@functools.cache
def _tokenizer(path: Path):
    return load_tokenizer(path)


def resolve_prefix(
    prompt: FlowPrompt | None,
    tokenizer,
    source: str,
    *,
    image_prompt: str,
    negative_prompt: str,
) -> tuple[tuple[int, ...], bool]:
    """Tokenize the prefix a guidance branch of ``source`` conditions on.

    ``conditioning`` frames the image prompt, and a blank one reuses the
    request's own context; ``negative_or_start`` frames the negative
    prompt; ``start`` frames empty text. Text is stripped as Python strips
    it. Without prompt framing a model reuses its context for a blank
    conditioning prompt, refuses a prompt override and leaves its other
    prefixes empty. ``tokenizer`` returns the model's tokenizer; it is called
    only when ``prompt`` frames text.

    Returns:
        The prefix tokens and whether the branch reuses the request's own
        context.

    Raises:
        ValueError: A prompt override for a model without prompt framing.
    """
    source = str(source)
    conditioned = source == "conditioning"
    text = {
        "conditioning": str(image_prompt),
        "negative_or_start": str(negative_prompt),
    }.get(source, "")
    text = text.strip()
    if conditioned and not text:
        return (), True
    if prompt is None:
        if conditioned:
            raise ValueError(
                "this model does not accept a generation prompt override"
            )
        return (), False
    return prompt.encode(tokenizer(), text=text, conditioned=conditioned), False


def guidance_prefix(
    loaded, source: str, image_prompt: str, negative_prompt: str
) -> list:
    """The prefix of a guidance branch (see :func:`resolve_prefix`).

    Returns:
        ``[tokens, reuses_conditioning]``.
    """

    def tokenizer():
        return (
            None if loaded.tokenizer is None else _tokenizer(loaded.tokenizer)
        )

    tokens, reuses = resolve_prefix(
        loaded.flow_prompt,
        tokenizer,
        source,
        image_prompt=image_prompt,
        negative_prompt=negative_prompt,
    )
    return [list(tokens), reuses]


def image_facts(image: PreparedImage) -> dict[str, Any]:
    """The host facts the worker plans an encode with.

    Returns:
        ``{"image", "shape", "grid", "height", "width"}``: the prepared
        image (opaque), its pixel shape (``[patches, row]`` for a patch
        tower, ``[channels, height, width]`` otherwise), its ``[rows,
        columns]`` patch grid or None, and the canvas dimensions the
        encoder's features describe.
    """
    return {
        "image": image,
        "shape": list(image.pixels.shape),
        "grid": None if image.grid_shape is None else list(image.grid_shape),
        "height": image.height,
        "width": image.width,
    }


def prepare_input(
    loaded, tower: str, payload: str, input_images: int, pin: int
) -> dict[str, Any]:
    """Decode and transform one inline image of a request on the host.

    Args:
        loaded: The loaded model; its ``image_processor`` is the policy.
        tower: ``"vit"`` to prepare understanding features (vision
            encoding), ``"vae"`` to prepare latent encoding.
        payload: The image's base64 payload.
        input_images: The request's input image count, which a shared
            pixel budget divides.
        pin: Nonzero to place the pixels in page-locked memory for an
            asynchronous copy to a CUDA device.

    Returns:
        See :func:`image_facts`.

    Raises:
        ValueError: The model declares no image policy, or the payload is
            not an image the policy accepts.
    """
    processor = loaded.image_processor
    if processor is None:
        raise ValueError("the model declares no image processing")
    if int(input_images) < 1:
        raise ValueError("request admission declares no input images")
    image = prepare_host_image(
        processor,
        # The worker sends a plain string; the selection rejects unknown towers.
        cast(Tower, str(tower)),
        str(payload),
        input_images=int(input_images),
        pin=bool(pin),
    )
    return image_facts(image)
