"""The forwards of the encoders: :class:`EncoderRunner`.

One runner serves one encoder module on one stream at one size. The tensors
of a forward are parameters of :meth:`EncoderRunner.encode`; the parameters
depend on the encoder's capability class and are documented there. A text
encoder's forward of a prompt without vision tokens has shapes fixed by its
tensors and no host work, so it is captured into one CUDA graph per prompt
capacity; a text conditioner's forward likewise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, cast

import torch

from uniserve_models.bindings.context import ExecutionContext
from uniserve_models.interface import calls, regions
from uniserve_models.interface.errors import SetupError
from uniserve_models.interface.loading import capability
from uniserve_models.interface.workspace import execution_context
from uniserve_models.media.pixels import (
    PreparedImage,
    Tower,
    prepare_host_image,
    prepare_tensor_image,
)
from uniserve_models.model import (
    AudioEncoder,
    PatchEncoder,
    TextConditioner,
    TextEncoder,
    VideoEncoder,
    VisionInput,
)
from uniserve_models.nn.vae import PatchAutoencoder
from uniserve_models.processing import PatchTransform

__all__ = ["EncoderRunner"]


def _image_facts(prepared: PreparedImage) -> dict[str, Any]:
    """The host facts a caller plans an encode with."""
    return {
        "image": prepared,
        "shape": list(prepared.pixels.shape),
        "grid": None
        if prepared.grid_shape is None
        else list(prepared.grid_shape),
        "height": prepared.height,
        "width": prepared.width,
    }


def _chunks(values: Sequence[int], width: int) -> list[tuple[int, ...]]:
    """A flat integer list as consecutive tuples of ``width``."""
    flat = [int(value) for value in values]
    return [
        tuple(flat[start : start + width])
        for start in range(0, len(flat), width)
    ]


def _device_grid(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """The ``[1, axes]`` int64 patch grid of a packed sample on ``device``.

    Built in page-locked memory first, so the copy does not synchronize the
    host with the stream.
    """
    grid = torch.tensor([shape], dtype=torch.long)
    if device.type == "cuda":
        grid = grid.pin_memory()
    return grid.to(device, non_blocking=True)


class EncoderRunner:
    """The forward of one encoder on one stream at one size.

    Args:
        loaded: The loaded model.
        path: The encoder module's path in the model.
        stream: The CUDA stream handle every forward enqueues on.
        attention: The attention backend the model loaded with.
        allocate: The caller's allocation function for workspace torch
            cannot provide.
        size: The shapes of the encoder's inputs, flattened: ``[rows,
            width]`` of a text conditioner's features, ``[frames, height,
            width, 3]`` of each visual unit of a video encoder's call,
            ``[samples, 2]`` of each track of an audio encoder's call.
    """

    # The encoder, of one of the capability classes :meth:`encode` lists.
    _module: Any
    # The loaded model and the product arena of an image model's encoder.
    _loaded: Any = None
    _products: Any = None

    def __init__(
        self,
        loaded,
        path: str,
        stream: int,
        attention: str,
        allocate,
        size: Sequence[int],
    ) -> None:
        module = loaded.module.get_submodule(str(path))
        if isinstance(module, TextConditioner):
            rank, name = 2, "conditioner"
        elif isinstance(module, VideoEncoder):
            rank, name = 4, "encoder.video"
        elif isinstance(module, AudioEncoder):
            rank, name = 2, "encoder.audio"
        else:
            raise SetupError(
                f"{type(module).__qualname__} is not constructed with a size"
            )
        shapes = tuple(_chunks(size, rank))
        context = execution_context(
            module, int(stream), str(attention), allocate, f"{name}.{shapes}"
        )
        # The context is keyed by its inputs' shapes, as one numerical
        # signature of the encoder.
        context.prepare(shapes)
        self._bind(module, context)

    @classmethod
    def bound(
        cls, prepared, path: str, loaded=None, products=None
    ) -> EncoderRunner:
        """A runner of the encoder at ``path`` of a bound component.

        ``path`` is relative to the component's root (``""`` for the root
        itself, ``"vision"`` for a text encoder's vision tower); the runner
        computes with the component's execution context. An image model's
        vision or latent encoder also takes ``loaded``, whose image policy
        prepares resident images, and ``products``, the device's uint8
        product arena in which its products are regions.
        """
        # FIXME: a text encoder and its vision tower are still prepared by
        # the component binding, which sizes their operators and declares
        # their workspace before the caller allocates it. They take the
        # constructor above once that binding is replaced by the caller's
        # allocation function.
        context = prepared.context
        if context is None:
            raise ValueError("a runner requires a bound component")
        path = str(path)
        root = prepared.module
        module = root.get_submodule(path) if path else root
        if not isinstance(
            module, (TextEncoder, PatchEncoder, PatchAutoencoder)
        ):
            raise SetupError(
                f"{type(module).__qualname__} is not a text, patch or "
                "latent encoder"
            )
        runner = cls.__new__(cls)
        runner._loaded = loaded
        runner._products = products
        runner._bind(module, context)
        if products is not None:
            # An image model's encoder takes prepared images.
            runner._encode = (
                runner._encode_packed
                if isinstance(module, PatchEncoder)
                and module.max_patches is not None
                else runner._encode_images
            )
        return runner

    def _bind(self, module, context: ExecutionContext) -> None:
        self._module = module
        self._context = context
        self._device = context.device
        self._encode: Callable[..., Any]
        if isinstance(module, TextEncoder):
            self._encode = self._encode_text
        elif isinstance(module, PatchEncoder):
            self._encode = self._encode_patches
        elif isinstance(module, TextConditioner):
            self._encode = self._encode_condition
        elif isinstance(module, VideoEncoder):
            self._encode = self._encode_video
        else:
            self._encode = self._encode_audio

    # Facts.

    @staticmethod
    def unit_frames(loaded, frames: int) -> list[int]:
        """Frames of each visual unit of a condition of ``frames`` frames.

        Answered from the video encoder's configuration alone.
        """
        encoder = capability(loaded.module, VideoEncoder)
        if encoder is None:
            raise SetupError("the model has no VideoEncoder")
        return [
            window.stop - window.start
            for window in encoder.frame_slices(int(frames))
        ]

    @staticmethod
    def sample_rate(loaded) -> int:
        """The sample rate condition audio is resampled to for the encoder."""
        encoder = capability(loaded.module, AudioEncoder)
        if encoder is None:
            raise SetupError("the model has no AudioEncoder")
        return int(encoder.sample_rate)

    @staticmethod
    def pack_pixels(
        loaded,
        frames: torch.Tensor,
        frame_indices: Sequence[int],
        grid: Sequence[int],
        out: torch.Tensor,
    ) -> None:
        """Pack condition frames into the patch rows a patch encoder takes.

        ``frames`` are ``[frames, height, width, 3]`` uint8 frames, of which
        the frames at ``frame_indices`` (every frame when it is empty) are
        packed on the ``[time, height, width]`` patch ``grid`` into
        ``out``. Packing reads no parameter, so a rank without the
        encoder's weights packs too.
        """
        encoder = capability(loaded.module, PatchEncoder)
        if encoder is None:
            raise SetupError("the model has no PatchEncoder")
        indices = [int(index) for index in frame_indices]
        sampled = frames[indices] if indices else frames
        time, height, width = (int(extent) for extent in grid)
        rows = encoder.pack_pixels(sampled, (time, height, width))
        if out.shape != rows.shape or out.dtype != rows.dtype:
            raise ValueError("a condition's patch rows disagree with its plan")
        out.copy_(rows)

    def final_stage(self) -> bool:
        """Whether this rank's pipeline stage of a text encoder emits features.

        Every stage runs the forward; only the last one writes its output.
        """
        pipeline = self._module.network._pipeline
        return bool(pipeline.rank == pipeline.size - 1)

    # Host planning.

    def position_ids(
        self,
        token_ids: Sequence[int],
        image_grids: Sequence[int],
        video_grids: Sequence[int],
        out: torch.Tensor,
    ) -> None:
        """Write the rotary coordinates of a prompt holding vision blocks.

        ``token_ids`` is the prompt padded to its capacity; ``image_grids``
        and ``video_grids`` are the ``[time, height, width]`` patch grids of
        its image and video blocks in prompt order, flattened; ``out`` is
        the ``[3, columns]`` int64 device tensor :meth:`encode` takes, whose
        leading ``capacity`` columns receive the coordinates.
        """
        values = self._module.positions(
            [int(token) for token in token_ids],
            image_grids=_chunks(image_grids, 3),
            video_grids=_chunks(video_grids, 3),
        )
        if values.shape != (3, len(token_ids)):
            raise ValueError("rotary coordinates must cover the capacity")
        with torch.inference_mode(), self._context.activate():
            out[:, : len(token_ids)].copy_(values, non_blocking=True)

    # Forward.

    def encode(self, *arguments: Any) -> Any:
        """Encode into the caller's output tensor.

        A text encoder takes ``(tokens, out)`` or ``(tokens, out,
        positions, visual)``: ``tokens`` is the ``[capacity]`` int64
        prompt, zero-padded (causal attention keeps the padding out of the
        prompt's rows); ``out`` the ``[capacity, hidden]`` features, which
        only the final pipeline stage writes; ``positions`` holds the
        prompt's rotary coordinates in its leading ``capacity`` columns
        (:meth:`position_ids`) and ``visual`` the vision rows spliced in at
        its placeholders.

        A patch encoder takes ``(rows, out, grids)``: ``rows`` holds every
        vision block's packed patch rows back to back, ``grids`` each
        block's ``[time, height, width]`` patch grid, flattened; the
        blocks' features fill ``out``'s rows in block order. Each block is
        one sample of the call, so its features do not depend on the other
        blocks.

        A text conditioner takes ``(features, lengths, out)``: the
        ``[capacity, width]`` prompt features, zero past the prompt, the
        ``[1]`` int32 prompt length, and the ``[capacity, width]`` refined
        rows.

        A video encoder takes ``(pixels, out, units)``: ``pixels`` is the
        request's ``[pixels, 3]`` uint8 condition pixels and each unit is
        ``[pixel_offset, frames, width, height, unit]``, flattened: its
        condition's first pixel row, frame count and raster, and the
        unit's index among the condition's units. The units' rows fill
        ``out`` in order.

        An audio encoder takes ``(samples, out, lengths)``: the request's
        ``[samples, 2]`` FP32 PCM, the tracks' sample counts ``lengths``;
        the tracks' rows fill ``out`` in order.

        An image model's unpacked vision encoder and its latent encoder
        take ``(images, out)``: prepared images of one pixel shape
        (:meth:`prepare_image`, :meth:`prepare_resident`) and one product
        region per image, which receives the image's features in the
        product's dtype. They return each image's feature shape. A latent
        encoder that samples its posterior draws from the device's default
        generator.

        An image model's packed vision encoder (``max_patches`` set) takes
        ``(pixels, grids, features)``: ``pixels`` ``[slots * max_patches,
        row]`` holds image ``i``'s patch rows at the front of slot ``i``
        (:meth:`pack_images`), ``grids`` ``[2 * slots, 2]`` int64 the slot
        grids (``PatchEncoder.packed_grids``) and ``features`` ``[slots *
        tokens, output_size]`` receives every slot's features, ``tokens``
        being ``max_patches // downsample**2``. Its shapes are fixed by its
        tensors, so it is captured into one CUDA graph per slot count.
        """
        with torch.inference_mode(), self._context.activate():
            return self._encode(*arguments)

    def _encode_text(
        self,
        tokens: torch.Tensor,
        out: torch.Tensor,
        positions: torch.Tensor | None = None,
        visual: torch.Tensor | None = None,
    ) -> None:
        options: dict[str, Any] = {}
        if positions is not None:
            options["positions"] = (positions[:, : tokens.shape[0]],)
        if visual is not None:
            options["visual"] = (visual,)
        values = self._module.encode((tokens,), **options)
        if values is not None:
            # A pipeline stage without the final layers has no features.
            out.copy_(values[0])

    def _encode_patches(
        self, rows: torch.Tensor, out: torch.Tensor, grids: Sequence[int]
    ) -> None:
        shapes = _chunks(grids, 3)
        # Each block is one packed sample: its patch rows and its grid,
        # which the encoder also takes as a [1, 3] device tensor.
        samples = rows.split([prod(shape) for shape in shapes])
        features = self._module.encode(
            VisionInput(
                tuple(samples),
                tuple(_device_grid(shape, self._device) for shape in shapes),
                tuple(shapes),
            )
        )
        start = 0
        for value in features:
            stop = start + value.shape[0]
            out[start:stop].copy_(value, non_blocking=True)
            start = stop
        if start != out.shape[0]:
            raise ValueError("vision features disagree with their output rows")

    def _encode_condition(
        self, features: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor
    ) -> None:
        out.copy_(self._module.encode((features,), lengths=lengths)[0])

    def _encode_video(
        self, pixels: torch.Tensor, out: torch.Tensor, units: Sequence[int]
    ) -> None:
        encoder = self._module
        values, windows, counts = [], [], []
        for offset, frames, width, height, unit in _chunks(units, 5):
            window = encoder.frame_slices(frames)[unit]
            area = width * height
            values.append(
                pixels[
                    offset + window.start * area : offset + window.stop * area
                ].view(window.stop - window.start, height, width, 3)
            )
            windows.append(window)
            counts.append(frames)
        encoded = encoder.encode(
            tuple(values), frames=tuple(windows), num_frames=tuple(counts)
        )
        if (
            sum(int(output.tensor.shape[0]) for output in encoded)
            != (out.shape[0])
        ):
            raise ValueError("encoded condition units disagree with their rows")
        torch.cat([output.tensor for output in encoded], out=out)

    def _encode_audio(
        self, samples: torch.Tensor, out: torch.Tensor, lengths: Sequence[int]
    ) -> None:
        tracks = samples.split([int(length) for length in lengths])
        encoded = self._module.encode(tracks)
        if sum(int(value.shape[0]) for value in encoded) != out.shape[0]:
            raise ValueError("encoded audio tracks disagree with their rows")
        torch.cat(encoded, out=out)

    # Image models.

    @staticmethod
    def describe(loaded, path: str, dtype: str) -> dict[str, Any]:
        """Facts of the encoder at module ``path`` of the loaded model.

        ``dtype`` is the deployment's model dtype, which a text encoder's
        hidden states carry.

        Returns:
            ``kind`` (``"vision"``, ``"latent"`` or ``"text"``) and:

            - vision: ``patch_size``, ``downsample`` (input patches per
              feature on each axis), ``output_size``, ``output_dtype``,
              ``max_patches`` (patch rows of one packed slot, 0 for an
              unpacked encoder), ``image_patches`` (patch rows of the
              largest image the image policy admits), ``row_width``
              (values per packed patch row) and ``pixel_dtype`` (the image
              policy's output dtype).
            - latent: ``downsample`` (pixels per latent patch on each
              axis), ``width`` (values per latent patch row) and
              ``output_dtype``.
            - text: ``hidden_size``, ``output_dtype`` and ``positions``
              (whether the encoder places vision tokens with rotary
              coordinates).

        Raises:
            NotImplementedError: The module is no encoder a runner serves.
        """
        # FIXME: these facts still cross as one record.
        path = str(path)
        module = loaded.module.get_submodule(path) if path else loaded.module
        if isinstance(module, PatchEncoder):
            processor = getattr(loaded, "image_processor", None)
            pixel_dtype = (
                None if processor is None else processor.output_dtype
            ) or torch.float32
            layout = module.features_layout(1)["features"]
            transform = None if processor is None else processor.vit
            if transform is None:
                image_patches = 0
            else:
                bound = (
                    transform.pixel_bound()
                    if isinstance(transform, PatchTransform)
                    else min(
                        transform.resize.max_pixels,
                        transform.resize.max_size**2,
                    )
                )
                image_patches = bound // module.patch_size**2
            return {
                "kind": "vision",
                "patch_size": module.patch_size,
                "downsample": module.downsample,
                "output_size": int(layout.shape[1]),
                "output_dtype": str(layout.dtype).removeprefix("torch."),
                "max_patches": module.max_patches or 0,
                "image_patches": image_patches,
                "row_width": 3 * module.patch_size**2,
                "pixel_dtype": str(pixel_dtype).removeprefix("torch."),
            }
        if isinstance(module, PatchAutoencoder):
            return {
                "kind": "latent",
                "downsample": module.downsample,
                "width": module.latent_channels * module.patch_size**2,
                "output_dtype": str(module.latent_dtype).removeprefix("torch."),
            }
        if isinstance(module, TextEncoder):
            return {
                "kind": "text",
                "hidden_size": int(module.network.hidden_size),
                "output_dtype": str(dtype).removeprefix("torch."),
                "positions": type(module).positions
                is not TextEncoder.positions,
            }
        raise NotImplementedError(
            f"{type(module).__qualname__} is no encoder a runner serves"
        )

    @staticmethod
    def prepare_image(
        loaded, tower: str, payload: str, input_images: int, pin: int
    ) -> dict[str, Any]:
        """Decode and transform one inline image of a request on the host.

        Touches no device and keeps no state, so the caller runs it beside
        device work.

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
            ``{"image", "shape", "grid", "height", "width"}``: the prepared
            image (opaque), its pixel shape (``[patches, row]`` for a patch
            tower, ``[channels, height, width]`` otherwise), its ``[rows,
            columns]`` patch grid or None, and the canvas dimensions the
            encoder's features describe.

        Raises:
            ValueError: The model declares no image policy, or the payload
                is not an image the policy accepts.
        """
        # FIXME: a prepared image crosses as an opaque handle with a record
        # of its facts; its pixels become a tensor of the caller's
        # page-locked storage.
        processor = loaded.image_processor
        if processor is None:
            raise ValueError("the model declares no image processing")
        if int(input_images) < 1:
            raise ValueError("request admission declares no input images")
        return _image_facts(
            prepare_host_image(
                processor,
                # The caller sends a plain string; the selection rejects
                # unknown towers.
                cast(Tower, str(tower)),
                str(payload),
                input_images=int(input_images),
                pin=bool(pin),
            )
        )

    @staticmethod
    def patch_grid(
        loaded, height: int, width: int, input_images: int
    ) -> list[int]:
        """The vit tower's ``[rows, columns]`` patch grid of an image.

        ``input_images`` is the request's input image count for a request
        input image, or 0 for a generated image, which keeps the
        single-image pixel bound.

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
        return list(
            transform.grid_shape(int(height), int(width), count or None)
        )

    def _processor(self):
        processor = getattr(self._loaded, "image_processor", None)
        if processor is None:
            raise ValueError("the model declares no image processing")
        return processor

    def _tower(self) -> Tower:
        return "vit" if isinstance(self._module, PatchEncoder) else "vae"

    def prepare_resident(
        self, image, tower: str, signed_unit: int
    ) -> dict[str, Any]:
        """Prepare a resident image product (a region) for encoding.

        ``signed_unit`` declares pixel values in ``[-1, 1]`` rather than
        ``[0, 1]``. Returns the prepared image's facts (see
        :meth:`prepare_image`).
        """
        with torch.inference_mode(), self._context.activate():
            prepared = prepare_tensor_image(
                self._processor(),
                cast(Tower, str(tower)),
                regions.view(self._products, image),
                device=self._device,
                signed_unit=bool(signed_unit),
            )
        return _image_facts(prepared)

    def warm(self, side: int) -> None:
        """Encode a black ``side`` x ``side`` image and discard the result.

        The image passes through the model's input transform; an unpacked
        encoder's kernels compile here, before serving.
        """
        side = int(side)
        with torch.inference_mode(), self._context.activate():
            prepared = prepare_tensor_image(
                self._processor(),
                self._tower(),
                torch.zeros((3, side, side)),
                device=self._device,
                signed_unit=False,
            )
            self._run_images((prepared,))

    def pack_images(self, pixels: torch.Tensor, *images: PreparedImage) -> None:
        """Copy image ``i``'s patch rows to the front of slot ``i``.

        ``pixels`` is the packed encoder's ``[slots * max_patches, row]``
        input. Rows past an image keep earlier finite values, which only
        padding reads.
        """
        capacity = self._module.max_patches
        with torch.inference_mode(), self._context.activate():
            for index, prepared in enumerate(images):
                rows = prepared.pixels.shape[0]
                if prepared.grid_shape is None or rows > capacity:
                    raise ValueError(
                        "packed vision images require patch rows within "
                        "their slot"
                    )
                pixels[index * capacity : index * capacity + rows].copy_(
                    prepared.pixels, non_blocking=True
                )

    def _run_images(self, images: Sequence[PreparedImage]):
        module, device = self._module, self._device
        pixels = tuple(
            # Host pixels are page-locked; the copy is asynchronous on the
            # stream.
            prepared.pixels.to(device, non_blocking=True)
            for prepared in images
        )
        if isinstance(module, PatchAutoencoder):
            # FIXME: the posterior draws from the device's default
            # generator; its noise becomes an input the caller supplies.
            return module.encode(torch.stack(pixels)).unbind(0)
        return module.encode(
            VisionInput(
                pixels,
                tuple(
                    None
                    if prepared.grid_shape is None
                    else _device_grid(prepared.grid_shape, device)
                    for prepared in images
                ),
                tuple(prepared.grid_shape for prepared in images),
            )
        )

    def _encode_images(
        self, images: Sequence[PreparedImage], out: Sequence
    ) -> list[list[int]]:
        if len(images) != len(out):
            raise ValueError("images and products must align")
        return [
            calls.write(regions.view(self._products, region), value)
            for value, region in zip(self._run_images(images), out, strict=True)
        ]

    def _encode_packed(
        self, pixels: torch.Tensor, grids: torch.Tensor, features: torch.Tensor
    ) -> None:
        features.copy_(self._module.encode_packed(pixels, grids))

    def kernels(self) -> list[dict]:
        """The kernels the runner's operators selected."""
        return list(self._context.kernels())

    def close(self, aborted: bool) -> None:
        """Release the runner's operators and workspace.

        An aborted close keeps everything device work may still address.
        """
        self._context.close(aborted=bool(aborted))
