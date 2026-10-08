"""Image input preparation the worker runs on its host lanes.

``image.prepare_input`` decodes one inline request image and applies the
model's image policy on the host (``uniserve_models.media.pixels``). It
touches no device and keeps no state, so the worker calls it from a host
lane thread while other batches execute. The prepared image crosses back as
an opaque handle; the worker passes it unchanged to the encoder binding
that copies it to the device (``uniserve_models.interface.encoder``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from uniserve_models.media.pixels import PreparedImage, prepare_host_image


def functions() -> dict[str, Callable[..., Any]]:
    return {"image.prepare_input": prepare_input}


def describe(image: PreparedImage) -> dict[str, Any]:
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
        See :func:`describe`.

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
        str(tower),
        str(payload),
        input_images=int(input_images),
        pin=bool(pin),
    )
    return describe(image)
