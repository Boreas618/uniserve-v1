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

from uniserve.bindings.context import ExecutionContext
from uniserve.errors import SetupError
from uniserve.loading.rank import capability
from uniserve.media.pixels import (
    PreparedImage,
    Tower,
    prepare_host_image,
    prepare_tensor_image,
)
from uniserve.models import (
    AudioEncoder,
    PatchEncoder,
    TextConditioner,
    TextEncoder,
    TextSize,
    VideoEncoder,
    VisionInput,
)
from uniserve.nn.vae import PatchAutoencoder
from uniserve.processing import PatchTransform
from uniserve.runner.context import (
    execution_context,
    held_module,
    kernel_records,
    write,
)

__all__ = ["EncoderRunner"]


def _module(loaded, path: str):
    """The module at ``path`` of the loaded model (``""`` for its root)."""
    path = str(path)
    return loaded.module.get_submodule(path) if path else loaded.module


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
        scratch: The stream's shared work areas
            (``uniserve.bindings.context.Scratch``).
        size: The size the operators are prepared for, flattened:
            ``[tokens, rows]`` of a text encoder's largest call (its own
            and its vision tower's, see :meth:`tower`), ``[rows, width]``
            of a text conditioner's features, ``[frames, height, width,
            3]`` of each visual unit of a video encoder's call, ``[samples,
            2]`` of each track of an audio encoder's call.
    """

    # The encoder, of one of the capability classes :meth:`encode` lists.
    _module: Any
    # The loaded model of an image model's encoder.
    _loaded: Any = None

    def __init__(
        self,
        loaded,
        path: str,
        stream: int,
        attention: str,
        scratch,
        size: Sequence[int],
    ) -> None:
        path, attention = str(path), str(attention)
        module = loaded.module.get_submodule(path) if path else loaded.module
        prepared: Any
        if isinstance(module, TextEncoder):
            tokens, rows = (int(value) for value in size)
            prepared = TextSize(tokens, rows)
            if attention == "auto":
                attention = loaded.attention
        elif isinstance(module, TextConditioner):
            prepared = tuple(_chunks(size, 2))
        elif isinstance(module, VideoEncoder):
            prepared = tuple(_chunks(size, 4))
        elif isinstance(module, AudioEncoder):
            prepared = tuple(_chunks(size, 2))
        else:
            raise SetupError(
                f"{type(module).__qualname__} is not constructed with a size"
            )
        context = execution_context(module, int(stream), attention, scratch)
        # A condition encoder's context is keyed by its inputs' shapes, as
        # one numerical signature of the encoder.
        context.prepare(prepared)
        self._bind(module, context)

    def tower(self, path: str) -> EncoderRunner:
        """A runner of the encoder at ``path`` below this runner's module.

        The tower (a text encoder's ``"vision"`` patch encoder) computes
        with this runner's execution context, whose operators were prepared
        for the larger of the two encoders' calls; this runner releases it.
        """
        module = self._module.get_submodule(str(path))
        if not isinstance(module, PatchEncoder):
            raise SetupError(
                f"{type(module).__qualname__} is not a patch encoder"
            )
        runner = type(self).__new__(type(self))
        runner._bind(module, self._context)
        return runner

    @classmethod
    def image(
        cls,
        loaded,
        path: str,
        stream: int,
        device: int,
        scratch,
        size: Sequence[int],
    ) -> EncoderRunner:
        """A runner of an image model's vision or latent encoder.

        Args:
            loaded: The loaded model, whose image policy prepares resident
                images.
            path: The encoder module's path in the model.
            stream: The CUDA stream handle every forward enqueues on (0 on
                the host).
            device: The CUDA ordinal ``stream`` belongs to, -1 for the
                host. An encoder placed on a generation tower computes
                there between the stream's steps.
            scratch: The stream's shared work areas.
            size: ``[tokens, rows]``, the most one call passes through the
                encoder's projections; empty for a latent encoder, whose
                convolutions need none.
        """
        module = _module(loaded, path)
        if not isinstance(module, (PatchEncoder, PatchAutoencoder)):
            raise SetupError(
                f"{type(module).__qualname__} is not a patch or latent encoder"
            )
        context = execution_context(
            module,
            int(stream),
            loaded.attention,
            scratch,
            device=int(device),
            derive_host_lengths=True,
        )
        values = [int(value) for value in size]
        context.prepare(TextSize(*values) if values else None)
        runner = cls.__new__(cls)
        runner._loaded = loaded
        runner._bind(module, context)
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

    # Facts of an encoder, answered from the loaded model before a runner
    # of it exists. ``path`` is the encoder's module path in the model.

    @staticmethod
    def holds(loaded, path: str) -> bool:
        """Whether this rank holds the encoder at ``path``.

        A pipeline stage that does not run an encoder's call holds no
        module for it, and constructs no runner of it.
        """
        return held_module(loaded.module, str(path)) is not None

    @staticmethod
    def kind(loaded, path: str) -> str:
        """``"vision"``, ``"latent"`` or ``"text"``.

        Raises:
            NotImplementedError: The module is no encoder a runner serves.
        """
        module = _module(loaded, path)
        if isinstance(module, PatchEncoder):
            return "vision"
        if isinstance(module, PatchAutoencoder):
            return "latent"
        if isinstance(module, TextEncoder):
            return "text"
        raise NotImplementedError(
            f"{type(module).__qualname__} is no encoder a runner serves"
        )

    @staticmethod
    def patch_size(loaded, path: str) -> int:
        """Pixels per input patch of a vision encoder, on each axis."""
        return int(_module(loaded, path).patch_size)

    @staticmethod
    def downsample(loaded, path: str) -> int:
        """Input units per output row on each axis.

        A vision encoder's input patches per feature; a latent encoder's
        pixels per latent patch.
        """
        return int(_module(loaded, path).downsample)

    @staticmethod
    def output_width(loaded, path: str) -> int:
        """Values per output row.

        A vision encoder's feature width, a latent encoder's patch row
        width, a text encoder's hidden size.
        """
        module = _module(loaded, path)
        if isinstance(module, PatchEncoder):
            return int(module.features_layout(1)["features"].shape[1])
        if isinstance(module, PatchAutoencoder):
            return int(module.latent_channels * module.patch_size**2)
        return int(module.network.hidden_size)

    @staticmethod
    def output_dtype(loaded, path: str) -> str:
        """The dtype of a vision or latent encoder's output rows.

        A text encoder's hidden states carry the deployment's model dtype.
        """
        module = _module(loaded, path)
        dtype = (
            module.features_layout(1)["features"].dtype
            if isinstance(module, PatchEncoder)
            else module.latent_dtype
        )
        return str(dtype).removeprefix("torch.")

    @staticmethod
    def max_patches(loaded, path: str) -> int:
        """Patch rows of one packed slot of a vision encoder.

        0 for an unpacked encoder.
        """
        return int(_module(loaded, path).max_patches or 0)

    @staticmethod
    def image_patches(loaded, path: str) -> int:
        """Patch rows of the largest image the image policy admits.

        0 for a model without an understanding transform.
        """
        module = _module(loaded, path)
        processor = getattr(loaded, "image_processor", None)
        transform = None if processor is None else processor.vit
        if transform is None:
            return 0
        bound = (
            transform.pixel_bound()
            if isinstance(transform, PatchTransform)
            else min(transform.resize.max_pixels, transform.resize.max_size**2)
        )
        return int(bound // module.patch_size**2)

    @staticmethod
    def processes_images(loaded) -> bool:
        """Whether the model declares an image policy."""
        return getattr(loaded, "image_processor", None) is not None

    @staticmethod
    def image_towers(loaded) -> list[str]:
        """The image transforms the policy declares.

        ``"vit"`` prepares understanding features (vision encoding),
        ``"vae"`` latent encoding.
        """
        processor = loaded.image_processor
        return [
            tower
            for tower in ("vit", "vae")
            if getattr(processor, tower) is not None
        ]

    @staticmethod
    def packs_patches(loaded) -> bool:
        """Whether the understanding transform packs patch rows."""
        return isinstance(loaded.image_processor.vit, PatchTransform)

    @staticmethod
    def feature_injection(loaded) -> list[str]:
        """How image features enter a prompt; empty without injection.

        ``[layout, positions]``: the feature layout (``"direct"`` or
        ``"framed"``) and the position layout (``"temporal"``,
        ``"temporal_spatial"`` or ``"sequential"``).
        """
        injection = loaded.image_processor.feature_injection
        if injection is None:
            return []
        return [injection.layout.value, injection.positions.value]

    @staticmethod
    def injection_tokens(loaded) -> list[int]:
        """The ``[start, end]`` token ids framing injected features.

        -1 stands for a token the injection does not place.
        """
        injection = loaded.image_processor.feature_injection
        return [
            -1 if token is None else int(token)
            for token in (injection.start_token_id, injection.end_token_id)
        ]

    @staticmethod
    def pixel_dtype(loaded) -> str:
        """The dtype of the pixels the image policy prepares."""
        processor = getattr(loaded, "image_processor", None)
        dtype = (
            None if processor is None else processor.output_dtype
        ) or torch.float32
        return str(dtype).removeprefix("torch.")

    @staticmethod
    def places_vision(loaded, path: str) -> bool:
        """Whether a text encoder gives vision tokens rotary coordinates."""
        return (
            type(_module(loaded, path)).positions is not TextEncoder.positions
        )

    @staticmethod
    def prepare_image(
        loaded, tower: str, payload: str, input_images: int, pin: int
    ) -> PreparedImage:
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
            The prepared image, opaque to the caller, which reads what it
            plans an encode with through :meth:`image_facts`.

        Raises:
            ValueError: The model declares no image policy, or the payload
                is not an image the policy accepts.
        """
        # FIXME: a prepared image crosses as an opaque handle; its pixels
        # become a tensor of the caller's page-locked storage.
        processor = loaded.image_processor
        if processor is None:
            raise ValueError("the model declares no image processing")
        if int(input_images) < 1:
            raise ValueError("request admission declares no input images")
        return prepare_host_image(
            processor,
            # The caller sends a plain string; the selection rejects
            # unknown towers.
            cast(Tower, str(tower)),
            str(payload),
            input_images=int(input_images),
            pin=bool(pin),
        )

    @staticmethod
    def image_facts(prepared: PreparedImage) -> list[int]:
        """What a caller plans an encode of a prepared image with.

        ``[height, width, grid_rows, grid_columns, *pixel_shape]``: the
        canvas the encoder's features describe, the patch grid of a patch
        tower (``0, 0`` otherwise) and the pixel shape (``[patches, row]``
        for a patch tower, ``[channels, height, width]`` otherwise).
        """
        rows, columns = prepared.grid_shape or (0, 0)
        return [
            int(prepared.height),
            int(prepared.width),
            int(rows),
            int(columns),
            *(int(extent) for extent in prepared.pixels.shape),
        ]

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
    ) -> PreparedImage:
        """Prepare a resident image product (a tensor) for encoding.

        ``signed_unit`` declares pixel values in ``[-1, 1]`` rather than
        ``[0, 1]``. Returns the prepared image (see :meth:`prepare_image`).
        """
        with torch.inference_mode(), self._context.activate():
            prepared = prepare_tensor_image(
                self._processor(),
                cast(Tower, str(tower)),
                image,
                device=self._device,
                signed_unit=bool(signed_unit),
            )
        return prepared

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
        self, images: Sequence[PreparedImage], *out: torch.Tensor
    ) -> list[list[int]]:
        if len(images) != len(out):
            raise ValueError("images and products must align")
        return [
            write(product, value)
            for value, product in zip(
                self._run_images(images), out, strict=True
            )
        ]

    def _encode_packed(
        self, pixels: torch.Tensor, grids: torch.Tensor, features: torch.Tensor
    ) -> None:
        features.copy_(self._module.encode_packed(pixels, grids))

    def kernels(self) -> list[str]:
        """The kernels the runner's operators selected, as JSON objects."""
        return kernel_records(self._context.kernels())

    def close(self, aborted: bool) -> None:
        """Release the runner's operators and workspace.

        An aborted close keeps everything device work may still address.
        """
        self._context.close(aborted=bool(aborted))
