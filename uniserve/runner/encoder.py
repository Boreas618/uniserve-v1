"""The forwards of the encoders: :class:`EncoderRunner`.

One runner serves one encoder module on one stream at one size. The tensors
of a forward are parameters of :meth:`EncoderRunner.encode`; the parameters
depend on the encoder's capability class and are documented there. A text
encoder's forward of a prompt without vision tokens has shapes fixed by its
tensors and no host work, so it is captured into one CUDA graph per prompt
capacity; a text conditioner's forward likewise.

A video encoder's still frames encode as runs of tiles of one shape, each
many small kernels, so their eager encoding is bound by the host launching
them. The rank captures one tile's encoding at startup
(:meth:`EncoderRunner.tile`, :meth:`EncoderRunner.encode_tile`), and every
video encoder runner constructed afterwards replays it for each still-frame
tile of that shape (``_TileReplay``) inside its eager forward; a video
clip's tiles stay eager.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, cast

import torch

from uniserve.bindings.context import ExecutionContext
from uniserve.errors import SetupError
from uniserve.loading.rank import capability
from uniserve.media import image, video
from uniserve.media.pixels import (
    Tower,
    image_size,
    prepare_host_image,
    prepared_shape,
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
from uniserve.nn.vae import PatchAutoencoder, TiledEncoder
from uniserve.processing import PatchTransform
from uniserve.runner.context import (
    execution_context,
    held_module,
    kernel_records,
)

__all__ = ["EncoderRunner"]


class _TileReplay:
    """The replay of a captured still-frame tile encoding, as a tile operator.

    Called with one tile's native input, it returns a new tensor equal to the
    tiled encoder's encoding of that tile: a tile of the captured input's
    shape, dtype and device is copied into that input, the graph launched by
    the worker on the stream every runner of a rank's video encoder computes
    on (``launch``, which takes no arguments) and its output copied out; any
    other tile is encoded eagerly.
    """

    def __init__(self, runner: EncoderRunner, launch) -> None:
        self._encoder = runner._tile_encoder
        self._input = runner._tile_input
        self._output = runner._tile_output
        self._launch = launch

    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        static, output = self._input, self._output
        if (
            static is None
            or output is None
            or tile.shape != static.shape
            or tile.dtype != static.dtype
            or tile.device != static.device
        ):
            return self._encoder(tile)
        static.copy_(tile)
        self._launch()
        # The next replay overwrites the output.
        return output.clone()


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


def _fill(out: torch.Tensor, pixels: torch.Tensor) -> None:
    """Write prepared ``pixels`` into the worker storage ``out``."""
    if tuple(out.shape) != tuple(pixels.shape) or out.dtype != pixels.dtype:
        raise ValueError(
            "prepared pixels disagree with the shape and dtype of their storage"
        )
    out.copy_(pixels, non_blocking=True)


class EncoderRunner:
    """The forward of one encoder on one stream at one size.

    Args:
        loaded: The loaded model.
        path: The encoder module's path in the model.
        stream: The CUDA stream handle every forward enqueues on.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas
            (``uniserve.bindings.workspace.Scratch``).
        size: The size the operators are prepared for, flattened:
            ``[tokens, rows]`` of a text encoder's largest call (its own
            and its vision tower's, see :meth:`tower`), ``[rows, width]``
            of a text conditioner's features, ``[frames, height, width,
            3]`` of each visual unit of a video encoder's call, ``[samples,
            2]`` of each track of an audio encoder's call.
        tile_runner: A video encoder's tile runner (:meth:`tile`), whose
            captured still-frame tile encoding the worker's ``tile_launch``
            launches for every tile of that shape this runner encodes; both
            borrowed, both ``None`` to encode every tile eagerly.
    """

    # The encoder, of one of the capability classes :meth:`encode` lists.
    _module: Any
    # A tile runner's tiled encoder, the one-tile input its captured
    # encoding reads and the output it writes (:meth:`tile`).
    _tile_encoder: Any = None
    _tile_input: torch.Tensor | None = None
    _tile_output: torch.Tensor | None = None

    def __init__(
        self,
        loaded,
        path: str,
        stream: int,
        attention: str,
        scratch,
        size: Sequence[int],
        tile_runner: EncoderRunner | None = None,
        tile_launch: Any = None,
    ) -> None:
        attention = str(attention)
        module = _module(loaded, path)
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
        tiles = None
        if tile_runner is not None and tile_launch is not None:
            replay = _TileReplay(tile_runner, tile_launch)
            tiles = {tile_runner._tile_encoder: replay}
        context = execution_context(
            module, int(stream), attention, scratch, tiles=tiles
        )
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

    @classmethod
    def tile(
        cls,
        loaded,
        path: str,
        stream: int,
        attention: str,
        scratch,
    ) -> EncoderRunner:
        """A runner of the video encoder at ``path`` for one still-frame tile.

        Its context is prepared for a tile-sized frame. It holds the tile's
        native input, zero, and the output :meth:`encode_tile` writes the
        input's encoding into: the forward the rank captures. Construction
        encodes the zero tile once, on the runner's stream, to size that
        output. Construct it only for an encoder whose latent encoder tiles
        (:meth:`still_tile` is not empty).
        """
        module = _module(loaded, path)
        frame = cls.still_tile(loaded, path)
        if not isinstance(module, VideoEncoder) or not frame:
            raise SetupError(f"{path!r} encodes no still-frame tiles")
        runner = cls(loaded, path, stream, attention, scratch, frame)
        tile = module.still_tile(runner._device)
        if tile is None:
            raise SetupError(f"{path!r} encodes no still-frame tiles")
        runner._tile_encoder, runner._tile_input = tile
        with torch.inference_mode(), runner._context.activate():
            runner._tile_output = torch.empty_like(
                runner._tile_encoder(runner._tile_input)
            )
        return runner

    def encode_tile(self) -> None:
        """Encode the runner's one-tile input into its tile output.

        The tiled encoder's encoding of one tile (``TiledEncoder.forward``)
        of the input :meth:`tile` holds: the forward a still-frame tile's
        graph captures; its replays read the input and write the output.
        """
        if self._tile_input is None or self._tile_output is None:
            raise SetupError("only a tile runner encodes a still-frame tile")
        with torch.inference_mode(), self._context.activate():
            self._tile_output.copy_(self._tile_encoder(self._tile_input))

    # Facts.

    @staticmethod
    def still_tile(loaded, path: str) -> list[int]:
        """``[1, height, width, 3]`` of a still frame one tile in size.

        Empty for a video encoder whose latent encoder does not tile still
        frames, and for any other module.
        """
        module = _module(loaded, path)
        if not isinstance(module, VideoEncoder):
            return []
        tiled = module.encoder.encoder
        if not isinstance(tiled, TiledEncoder):
            return []
        return [1, int(tiled.tile_height), int(tiled.tile_width), 3]

    @staticmethod
    def unit_frames(loaded, frames: int) -> list[int]:
        """Frames of each visual unit of a condition of ``frames`` frames.

        Answered from the video encoder's configuration alone.
        """
        encoder = capability(loaded.module, VideoEncoder)
        if encoder is None:
            raise SetupError("the model has no VideoEncoder")
        return [clip.stop - clip.start for clip in encoder.clips(int(frames))]

    @staticmethod
    def load_condition_kernels(loaded) -> None:
        """Load the video encoder's natively built kernels, before serving.

        Condition encoders are constructed for each input shape while
        serving; loading their kernels once beforehand keeps compilation
        out of every encoding.
        """
        encoder = capability(loaded.module, VideoEncoder)
        if encoder is not None:
            encoder.load_kernels()

    @staticmethod
    def posterior_elements(loaded, frames: int, width: int, height: int) -> int:
        """FP32 elements of a condition size's complete posterior draw.

        0 when the video encoder takes no draw.
        """
        encoder = capability(loaded.module, VideoEncoder)
        if encoder is None:
            raise SetupError("the model has no VideoEncoder")
        shape = encoder.posterior_shape(
            video.Config(int(frames), image.Config(int(height), int(width)))
        )
        return 0 if shape is None else prod(shape)

    @staticmethod
    def draw_posterior(
        loaded, frames: int, width: int, height: int, noise: torch.Tensor
    ) -> None:
        """Fill a condition size's complete posterior draw.

        ``noise`` is a flat FP32 tensor of :meth:`posterior_elements`
        elements, the worker's storage of the draw, which every
        encoding of a condition of this size then reads (:meth:`encode`).
        """
        encoder = capability(loaded.module, VideoEncoder)
        if encoder is None:
            raise SetupError("the model has no VideoEncoder")
        size = video.Config(int(frames), image.Config(int(height), int(width)))
        shape = encoder.posterior_shape(size)
        if shape is None:
            raise SetupError("the video encoder takes no posterior draw")
        with torch.inference_mode():
            encoder.draw_noise(size, noise.view(shape))

    @staticmethod
    def vision_processor(loaded) -> list[str]:
        """The JSON texts of the model's image and video processor configs.

        The server plans condition media's vision blocks by them; empty for
        a model that keeps none.
        """
        config = getattr(loaded.root, "config", None)
        texts = getattr(config, "vision_processor", None)
        return [] if texts is None else [str(text) for text in texts]

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

        A patch encoder takes ``(rows, out, grids, *device_grids)``:
        ``rows`` holds every vision block's packed patch rows back to back,
        ``grids`` each block's ``[time, height, width]`` patch grid,
        flattened, and ``device_grids``, when given, each block's grid as
        the ``[1, 3]`` int64 device tensor the caller wrote it into; the
        blocks' features fill ``out``'s rows in block order. Each block is
        one sample of the call, so its features do not depend on the other
        blocks, and one block's encode at one grid is captured into one CUDA
        graph.

        A text conditioner takes ``(features, lengths, out)``: the
        ``[capacity, width]`` prompt features, zero past the prompt, the
        ``[1]`` int32 prompt length, and the ``[capacity, width]`` refined
        rows.

        A video encoder takes ``(pixels, out, units, *noise)``: ``pixels``
        is the request's ``[pixels, 3]`` uint8 condition pixels and each
        unit is ``[pixel_offset, frames, width, height, unit, first_row,
        stop_row]``, flattened: its condition's first pixel row, frame count
        and raster, the unit's clip among the condition's clips, and the run
        of the frames' latent row groups it encodes: a still frame's band
        (``VideoEncoder.encode(rows=...)``), every row group of a whole frame
        or of a video's clip. An encoder that samples its posterior takes one
        ``noise`` tensor per unit, the complete draw of the unit's condition
        size (:meth:`draw_posterior`). The units' rows fill ``out`` in
        order.

        An audio encoder takes ``(samples, out, lengths)``: the request's
        ``[samples, 2]`` FP32 PCM, the tracks' sample counts ``lengths``;
        the tracks' rows fill ``out`` in order.
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
        self,
        rows: torch.Tensor,
        out: torch.Tensor,
        grids: Sequence[int],
        *device_grids: torch.Tensor,
    ) -> None:
        shapes = _chunks(grids, 3)
        # Each block is one packed sample: its patch rows and its grid,
        # which the encoder also takes as a [1, 3] device tensor: the
        # caller's, written before the call (a captured block encode reads
        # the same tensor on every replay), or a copy made here.
        if device_grids and len(device_grids) != len(shapes):
            raise ValueError("device grids must align with the blocks")
        samples = rows.split([prod(shape) for shape in shapes])
        features = self._module.encode(
            VisionInput(
                tuple(samples),
                device_grids
                or tuple(_device_grid(shape, self._device) for shape in shapes),
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
        self,
        pixels: torch.Tensor,
        out: torch.Tensor,
        units: Sequence[int],
        *noise: torch.Tensor,
    ) -> None:
        encoder = self._module
        values, clips, counts, draws, bands = [], [], [], [], []
        for index, (
            offset,
            frames,
            width,
            height,
            unit,
            first,
            stop,
        ) in enumerate(_chunks(units, 7)):
            # Media unit ``unit`` encodes the condition's clip of that index.
            clip = encoder.clips(frames)[unit]
            area = width * height
            values.append(
                pixels[
                    offset + clip.start * area : offset + clip.stop * area
                ].view(clip.stop - clip.start, height, width, 3)
            )
            clips.append(clip)
            counts.append(frames)
            if noise:
                shape = encoder.posterior_shape(
                    video.Config(frames, image.Config(height, width))
                )
                draws.append(noise[index].view(shape))
            # A still frame's run of row groups is a band of its latent rows
            # (all of them for a whole frame); a video's clip encodes whole.
            bands.append(
                slice(first * encoder.row_group, stop * encoder.row_group)
                if frames == 1
                else None
            )
        encoded = encoder.encode(
            tuple(values),
            clips=tuple(clips),
            num_frames=tuple(counts),
            noise=tuple(draws) if noise else None,
            rows=tuple(bands),
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
    def image_patches(loaded, path: str) -> int:
        """Patch rows of the largest image the image policy admits.

        0 for a model without an understanding transform.
        """
        module = _module(loaded, path)
        processor = loaded.image_processor
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
    def pixel_dtype(loaded) -> str:
        """The dtype of the pixels the image policy prepares.

        The policy's ``output_dtype``, float32 when it states none or the
        model keeps no policy. The worker's host blocks of prepared pixels
        and the image executor's pixel buffers both take it.
        """
        processor = loaded.image_processor
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
    def image_size(payload: str) -> list[int]:
        """The ``[height, width]`` of an inline image, read from its header.

        Raises:
            ValueError: The payload is not a valid encoded image.
        """
        return list(image_size(str(payload)))

    @staticmethod
    def image_facts(
        loaded, tower: str, height: int, width: int, input_images: int
    ) -> list[int]:
        """What preparing an image of ``height`` x ``width`` pixels produces.

        ``[canvas_height, canvas_width, grid_rows, grid_columns,
        *pixel_shape]``: the canvas the encoder's features describe, the
        patch grid of a patch tower (``0, 0`` otherwise) and the shape of
        the prepared pixels (``[patches, row]`` for a patch tower,
        ``[channels, height, width]`` otherwise), which
        :meth:`prepare_image` (and ``prepare_tensor_image``) write in
        :meth:`pixel_dtype`. ``input_images`` is the request's input image
        count for a request input image, which a shared pixel budget
        divides, or 0 for a generated image.

        Raises:
            ValueError: The model declares no transform for ``tower``, or
                its resize policy rejects the dimensions.
        """
        processor = loaded.image_processor
        if processor is None:
            raise ValueError("the model declares no image processing")
        canvas, grid, shape = prepared_shape(
            processor,
            # The caller sends a plain string; the plan rejects unknown
            # towers.
            cast(Tower, str(tower)),
            int(height),
            int(width),
            int(input_images) or None,
        )
        rows, columns = grid or (0, 0)
        return [*canvas, rows, columns, *shape]

    @staticmethod
    def prepare_image(
        loaded, tower: str, payload: str, input_images: int, out
    ) -> None:
        """Decode and transform one inline request image into ``out``.

        Touches no device and keeps no state, so the caller runs it beside
        device work. ``out`` is host memory of the pixel shape
        :meth:`image_facts` reports for the image and of
        :meth:`pixel_dtype`.

        Args:
            loaded: The loaded model; its ``image_processor`` is the policy.
            tower: ``"vit"`` to prepare understanding features (vision
                encoding), ``"vae"`` to prepare latent encoding.
            payload: The image's base64 payload.
            input_images: The request's input image count, which a shared
                pixel budget divides.
            out: The prepared pixels' storage.

        Raises:
            ValueError: The model declares no image policy, the payload is
                not an image the policy accepts, or ``out`` is not its
                prepared pixels' shape and dtype.
        """
        processor = loaded.image_processor
        if processor is None:
            raise ValueError("the model declares no image processing")
        if int(input_images) < 1:
            raise ValueError("request admission declares no input images")
        prepared = prepare_host_image(
            processor,
            cast(Tower, str(tower)),
            str(payload),
            input_images=int(input_images),
            pin=False,
        )
        _fill(out, prepared.pixels)

    def kernels(self) -> list[str]:
        """The kernels the runner's operators selected, as JSON objects."""
        return kernel_records(self._context.kernels())

    def close(self, aborted: bool) -> None:
        """Release the runner's operators and workspace.

        An aborted close keeps everything device work may still address.
        """
        self._context.close(aborted=bool(aborted))
