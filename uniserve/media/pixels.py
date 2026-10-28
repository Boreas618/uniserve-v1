"""Pixel preparation for vision towers and pixel quantization for outputs.

A model declares its image policy as a ``uniserve.processing``
``ImageProcessor``; this module applies it. An inline base64 payload
(``prepare_host_image``) or an already decoded RGB tensor
(``prepare_tensor_image``) is resized to the model's canvas (a patch tower
keeps the source size) and then to the selected tower's input size with the
declared resampling, normalized, and packed into patches in the declared
layout for a ``PatchTransform`` tower.

The tower is named by the transform it uses: ``"vit"`` (the processor's
``vit`` transform, used to encode understanding features) or ``"vae"``
(its ``vae`` transform, used by latent encoding).

An inline payload's host work (``prepare_host_image``: decoding, resizing,
normalization, patch packing) touches no device, so it runs on a host
thread; its pixels are copied to the device later by the encoder that
consumes them. ``quantize_image_hwc`` maps a decoded image to the packed
8-bit RGB layout image containers store.

Preprocessing is bit-exact with the reference image processors: Python
``round`` (ties to even) in stride rounding, PIL versus torchvision filter
choice, the float32 ``1 / 255`` rescale factor, host-side dtype rounding and
the alpha policy are all part of the contract.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision
from torchvision.transforms.v2 import functional as tensor_vision

from uniserve.nn.functional import patchify
from uniserve.processing import (
    ImageProcessor,
    PatchTransform,
    StrideResize,
    TowerTransform,
)
from uniserve.tensors import staged

Tower = Literal["vit", "vae"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# The Transformers image processors' rescale factor, 0.00392156862745098;
# ``"unit"`` normalization multiplies 8-bit values by it in float32.
_UNIT_SCALE = 1 / 255


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Normalized tower input pixels and the model canvas dimensions.

    For a tower that takes patch rows (a ``PatchTransform``, or a
    ``TowerTransform`` with a ``patch_layout``), ``pixels`` is ``[patches,
    channels * patch * patch]`` with patches in raster order and each row
    in the transform's ``patch_layout``, and ``grid_shape`` holds the
    (rows, columns) patch grid. For another ``TowerTransform`` tower,
    ``pixels`` is ``[channels, height, width]`` and ``grid_shape`` is None.
    The pixels are in the processor's output dtype. ``height`` and
    ``width`` are the canvas dimensions, not the tower's input size.

    A host-prepared image keeps its pixels in page-locked storage when it is
    prepared for a CUDA device, so the consumer's copy to the device is
    asynchronous; the caching host allocator keeps the block from reuse
    until that copy completes.
    """

    pixels: torch.Tensor
    grid_shape: tuple[int, int] | None
    height: int
    width: int


def _image_plan(
    processor: ImageProcessor,
    tower: Tower,
    height: int,
    width: int,
    input_images: int | None,
) -> tuple[PatchTransform | TowerTransform, tuple[int, int], tuple[int, int]]:
    """Resolve canvas and tower dimensions independently of pixel storage.

    A patch tower keeps the source size as its canvas, and its input size
    is ``PatchTransform.resized_size`` for ``input_images``. For a
    ``TowerTransform``, whether ``vit`` or ``vae``, the canvas is the source
    resized by the ``vae`` stride policy when the processor declares a
    ``vae`` transform, and the tower input applies the selected transform's
    stride policy to that canvas.

    Returns:
        The selected transform, the canvas (height, width) and the tower's
        input (height, width).

    Raises:
        ValueError: The model declares no transform for ``tower``, or, for a
            patch tower, its resize policy rejects the dimensions.
    """
    if tower not in ("vit", "vae"):
        raise ValueError(f"unknown image tower {tower!r}")
    transform = processor.vit if tower == "vit" else processor.vae
    if transform is None:
        raise ValueError(f"model declares no {tower} image transform")
    if isinstance(transform, PatchTransform):
        return (
            transform,
            (height, width),
            transform.resized_size(height, width, input_images),
        )

    canvas = (
        (height, width)
        if processor.vae is None
        else _stride_image_shape(height, width, processor.vae.resize)
    )
    return transform, canvas, _stride_image_shape(*canvas, transform.resize)


def _patches(
    transform: PatchTransform | TowerTransform,
) -> tuple[int, str] | None:
    """The patch side and row layout of a tower that takes patch rows.

    A ``PatchTransform`` cuts ``patch_size`` squares; a ``TowerTransform``
    with a ``patch_layout`` cuts squares of its resize stride. None for a
    tower that takes ``[C, H, W]`` pixels.
    """
    if isinstance(transform, PatchTransform):
        return transform.patch_size, transform.patch_layout
    if transform.patch_layout is not None:
        return transform.resize.stride, transform.patch_layout
    return None


def _packed_pixels(
    transform: PatchTransform | TowerTransform, pixels: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """Pack a patch tower's [C, H, W] input into patch rows.

    Returns the rows and the (rows, columns) patch grid for a tower that
    takes patch rows (:func:`_patches`), and the unchanged pixels with no
    grid otherwise. The packing is a pure layout change on the pixels' own
    device.
    """
    packing = _patches(transform)
    if packing is None:
        return pixels, None

    # [C, H, W] -> one flattened row per patch, in raster order over the
    # patch grid; the declared layout orders the values within a row.
    patch, layout = packing
    channels, height, width = pixels.shape
    grid_shape = (height // patch, width // patch)
    if layout == "channels_last":
        return patchify(pixels, patch_size=patch), grid_shape

    # [C, H, W] -> [gh, gw, C, patch, patch] -> rows.
    rows = (
        pixels.reshape(channels, grid_shape[0], patch, grid_shape[1], patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_shape[0] * grid_shape[1], channels * patch * patch)
    )
    return rows, grid_shape


def prepare_host_image(
    processor: ImageProcessor,
    tower: Tower,
    encoded: str | bytes,
    *,
    input_images: int,
    pin: bool,
) -> PreparedImage:
    """Decode a request input image and apply the model's transforms on host.

    ``input_images`` is the number of input images in the image's request;
    a patch tower whose images share a pixel budget bounds each by its share
    (see ``PatchTransform.pixel_bound``). With ``pin`` the pixels are copied
    into page-locked storage, so a consumer can copy them to a CUDA device
    without synchronizing the host with the device.

    Touches no device and holds no state, so any host thread may run it;
    decoding, torchvision resizing and NumPy normalization release the GIL
    for most of their work. The output dtype conversion runs here, on the
    host, as a dtype-changing host-to-device copy would perform it, so the
    rounding matches.

    Raises:
        ValueError: The payload is not a decodable image, or the model's
            transforms reject it.
    """
    image = _decode_rgb(encoded, processor.alpha)
    transform, canvas, size = _image_plan(
        processor, tower, image.height, image.width, input_images
    )
    if (
        isinstance(transform, PatchTransform)
        and transform.resampling == "torchvision"
    ):
        pixels = _normalize(
            _resample_bytes(image, size).numpy(), transform.normalization
        )
    else:
        if not isinstance(transform, PatchTransform):
            image = vision.resize(
                image,
                canvas,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        image = vision.resize(
            image,
            size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        pixels = _normalize(
            np.asarray(image).transpose(2, 0, 1), transform.normalization
        )

    pixels, grid_shape = _packed_pixels(transform, pixels)
    dtype = _output_dtype(processor)
    if dtype is not None:
        pixels = pixels.to(dtype)
    pixels = pixels.contiguous()
    if pin:
        pixels = pixels.pin_memory()
    return PreparedImage(pixels, grid_shape, *canvas)


def _resample_bytes(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Resize an RGB image as a ``uint8`` CHW tensor with torchvision.

    Returns the ``[3, height, width]`` ``uint8`` image at ``size``, following
    the Transformers torchvision image-processor path call for call:
    ``pil_to_tensor`` yields a channels-last view of the decoded bytes, and
    the antialiased bicubic filter runs on it only when the size changes,
    rounding back to 8 bits. torchvision picks its interpolation kernel by
    host architecture and storage format, so the same calls on the host
    reproduce the reference pixels independently of the destination device.
    """
    pixels = tensor_vision.pil_to_tensor(image)
    if tuple(pixels.shape[1:]) == size:
        return pixels
    return tensor_vision.resize(
        pixels,
        list(size),
        interpolation=tensor_vision.InterpolationMode.BICUBIC,
        antialias=True,
    )


def prepare_tensor_image(
    processor: ImageProcessor,
    tower: Tower,
    image: torch.Tensor,
    *,
    device: torch.device,
    signed_unit: bool,
) -> PreparedImage:
    """Apply the same canvas and tower policy to an already decoded RGB view.

    ``image`` is [3, height, width] or [1, 3, height, width] with values in
    [0, 1], or in [-1, 1] when ``signed_unit``; values are clamped to [0, 1]
    before the transforms. The view is a generated image rather than a
    request input, so a patch tower applies its single-image pixel bound.
    Continuous values resize with torch's antialiased bicubic filter
    whatever resampling the transform declares for 8-bit images. The result
    is on ``device`` in the processor's output dtype, enqueued on the
    current stream.

    Raises:
        ValueError: ``image`` is not one RGB image, or the model's transforms
            reject it.
    """
    value = image.detach().to(dtype=torch.float32)
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError("generated image preparation accepts one image")
        value = value[0]
    if value.ndim != 3 or int(value.shape[0]) != 3:
        raise ValueError(
            "generated image tensor must have shape [3, height, width]"
        )
    if signed_unit:
        value = (value + 1.0) * 0.5
    value = value.clamp(0.0, 1.0)
    transform, canvas, size = _image_plan(
        processor, tower, int(value.shape[1]), int(value.shape[2]), None
    )
    if not isinstance(transform, PatchTransform):
        value = _resize_tensor(value, *canvas)
    value = _resize_tensor(value, *size)
    pixels, grid_shape = _packed_pixels(
        transform, _normalize_tensor(value, transform.normalization)
    )
    pixels = pixels.to(
        device=device, dtype=_output_dtype(processor), non_blocking=True
    )
    return PreparedImage(pixels, grid_shape, *canvas)


def image_size(encoded: str | bytes) -> tuple[int, int]:
    """Return the ``(height, width)`` of a base64 image payload.

    Read from the image's header without decoding its pixels; decoding
    (:func:`prepare_host_image`) yields an image of these dimensions.

    Raises:
        ValueError: The payload is not a valid encoded image.
    """
    try:
        with Image.open(io.BytesIO(_payload_bytes(encoded))) as image:
            width, height = image.size
    except (OSError, ValueError) as error:
        raise ValueError(
            "inline image payload is not a valid encoded image"
        ) from error
    return int(height), int(width)


def prepared_shape(
    processor: ImageProcessor,
    tower: Tower,
    height: int,
    width: int,
    input_images: int | None,
) -> tuple[tuple[int, int], tuple[int, int] | None, tuple[int, ...]]:
    """Return what preparing an image of ``height`` x ``width`` produces.

    The canvas (height, width), the (rows, columns) patch grid of a tower
    that takes patch rows (None otherwise) and the shape of
    ``PreparedImage.pixels``, as :func:`prepare_host_image` (with
    ``input_images``) and :func:`prepare_tensor_image` (with None) prepare
    the image.

    Raises:
        ValueError: The model declares no transform for ``tower``, or its
            resize policy rejects the dimensions.
    """
    transform, canvas, size = _image_plan(
        processor, tower, height, width, input_images
    )
    packing = _patches(transform)
    if packing is not None:
        patch = packing[0]
        grid = (size[0] // patch, size[1] // patch)
        return canvas, grid, (grid[0] * grid[1], 3 * patch * patch)
    return canvas, None, (3, *size)


def _payload_bytes(encoded: str | bytes) -> bytes:
    """The encoded image bytes of a base64 payload."""
    if not isinstance(encoded, (str, bytes)) or not encoded:
        raise ValueError("inline image payload must be non-empty base64")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(
            "inline image payload is not a valid encoded image"
        ) from error


def _decode_rgb(encoded: str | bytes, alpha: str) -> Image.Image:
    """Decode a base64 image payload and convert it to RGB.

    ``alpha`` is the processor's ``ImageProcessor.alpha`` policy for images
    with transparency.
    """
    raw = _payload_bytes(encoded)
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (OSError, ValueError) as error:
        raise ValueError(
            "inline image payload is not a valid encoded image"
        ) from error

    transparent = (
        image.mode == "RGBA" or image.info.get("transparency") is not None
    )
    if alpha == "white" and transparent:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        return rgb
    return image.convert("RGB")


def patch_grid_shape(
    transform: PatchTransform,
    source_height: int,
    source_width: int,
    input_images: int | None,
) -> tuple[int, int]:
    """Return the host-known patch grid for the declared image dimensions.

    ``input_images`` is the number of input images in the request of a
    request input image, or None for a generated image.

    Raises:
        ValueError: The transform's resize policy rejects the dimensions.
    """
    return transform.grid_shape(source_height, source_width, input_images)


def _stride_image_shape(
    height: int, width: int, processor: StrideResize
) -> tuple[int, int]:
    """Resolve both image dimensions at the configured spatial stride.

    Returns (height, width), while ``_stride_shape`` takes and returns
    (width, height).
    """
    scale = min(int(processor.max_size) / max(width, height), 1.0)
    scale = max(scale, int(processor.min_size) / min(width, height))
    new_width, new_height = _stride_shape(
        width, height, scale, int(processor.stride)
    )
    if new_width * new_height > int(processor.max_pixels):
        scale = int(processor.max_pixels) / (new_width * new_height)
        new_width, new_height = _stride_shape(
            new_width, new_height, scale, int(processor.stride)
        )
    if max(new_width, new_height) > int(processor.max_size):
        scale = int(processor.max_size) / max(new_width, new_height)
        new_width, new_height = _stride_shape(
            new_width, new_height, scale, int(processor.stride)
        )
    return new_height, new_width


def _stride_shape(
    width: int, height: int, scale: float, stride: int
) -> tuple[int, int]:
    """Round scaled dimensions to positive multiples of the required stride."""

    def align(value: float) -> int:
        """Round a scaled edge to the nearest positive model stride."""
        return max(stride, round(value / stride) * stride)

    return align(width * scale), align(height * scale)


def _normalize(values: np.ndarray, name: str) -> torch.Tensor:
    """Normalize ``uint8`` CHW RGB bytes into contiguous FP32 model inputs."""
    # PIL decoding and resizing produce host bytes. Keep the pointwise FP32
    # transform in one owned CHW array instead of dispatching each pass
    # through the process-wide tensor thread pool.
    pixels = values.astype(np.float32, order="C")
    if name == "unit":
        # One float32 multiplication by the float32 factor, bit for bit the
        # Transformers ``rescale`` of a uint8 tensor.
        pixels *= np.float32(_UNIT_SCALE)
        return torch.from_numpy(pixels)

    pixels /= np.float32(255.0)
    if name == "signed_unit":
        pixels -= np.float32(0.5)
        pixels /= np.float32(0.5)
    elif name == "imagenet":
        pixels -= np.asarray(_IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        pixels /= np.asarray(_IMAGENET_STD, dtype=np.float32)[:, None, None]
    else:
        raise ValueError(f"unknown image normalization {name!r}")
    return torch.from_numpy(pixels)


def _normalize_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Apply the named normalization policy to an image tensor in [0, 1]."""
    if name == "unit":
        return tensor
    if name == "signed_unit":
        return (tensor - 0.5) / 0.5
    if name == "imagenet":
        # The constants round to the tensor's dtype on the host, as
        # ``tensor.new_tensor`` would, and reach its device in stream order.
        mean, std = (
            staged(torch.tensor(values, dtype=tensor.dtype), tensor.device)
            .to(tensor.device, non_blocking=True)
            .view(3, 1, 1)
            for values in (_IMAGENET_MEAN, _IMAGENET_STD)
        )
        return (tensor - mean) / std
    raise ValueError(f"unknown image normalization {name!r}")


def _resize_tensor(
    value: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Resize a batched tensor image with bicubic interpolation."""
    return F.interpolate(
        value.unsqueeze(0),
        size=(int(height), int(width)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0]


def _output_dtype(processor: ImageProcessor) -> torch.dtype | None:
    """Return the declared output dtype, or None to keep FP32.

    Raises:
        ValueError: The declaration is not a ``torch.dtype``.
    """
    dtype = processor.output_dtype
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown image output dtype {dtype!r}")
    return dtype


def quantize_image_hwc(
    tensor: torch.Tensor,
    *,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Quantize one CHW/NCHW image to contiguous HWC uint8 on its device.

    ``value_range`` declares the source interval mapped onto ``[0, 1]``
    before clamping and conversion to ``[0, 255]``. The default ``(-1, 1)``
    matches diffusion decoder output; callers with normalized pixels pass
    ``(0, 1)``. A degenerate range skips the affine map and only clamps.

    Raises:
        ValueError: A 4-D input's batch is not one, or the image is not
            3-channel CHW.
    """
    image = tensor.detach()
    if image.ndim == 4:
        if int(image.shape[0]) != 1:
            raise ValueError("image quantization requires a single-item batch")
        image = image[0]
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("image quantization requires RGB CHW pixels")

    image = image.float()
    lo, hi = float(value_range[0]), float(value_range[1])
    span = hi - lo
    if span != 0:
        image = (image - lo) / span
    image = image.clamp(0, 1)
    # [3, H, W] -> [H, W, 3] bytes, the packed layout of 8-bit RGB images.
    return (
        (image.permute(1, 2, 0) * 255.0)
        .round()
        .to(dtype=torch.uint8)
        .contiguous()
    )


__all__ = [
    "PreparedImage",
    "Tower",
    "patch_grid_shape",
    "prepare_host_image",
    "prepare_tensor_image",
    "quantize_image_hwc",
]
