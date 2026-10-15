"""The forwards of the media decoders: :class:`DecoderRunner`.

One runner serves one decoder module on one stream: a video decoder at one
segment (the native frames and raster a media unit decodes to), an audio
decoder at one track length, an image decoder of a bound component at
every image size (:meth:`DecoderRunner.bound`). The tensors of a forward are
parameters of :meth:`DecoderRunner.decode`; the parameters depend on the
decoder's capability class and are documented there. A video decoder's
forward has shapes fixed by its scalar arguments and takes every changing
value as the contents of a tensor, so it is captured into one CUDA graph
per kind of unit and round size.

The static methods answer a deployment's sizing questions from the model's
configuration alone, so a rank without the decoders' weights asks them too.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import torch

from uniserve.bindings.context import ExecutionContext, TensorBuffers
from uniserve.interface import calls
from uniserve.interface.errors import SetupError
from uniserve.interface.loading import capability
from uniserve.media import image, video
from uniserve.media.pixels import quantize_image_hwc
from uniserve.models import (
    AudioDecoder,
    ImageDecoder,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.runner.context import execution_context, kernel_records
from uniserve.tensors import BufferConfig

__all__ = ["DecoderRunner"]


class _Buffered(Protocol):
    """The constants a post-processor declares for a canvas.

    The capability class does not declare these methods, so the runner
    names the signatures it calls.
    """

    def constant_buffers(self, size: Any) -> Mapping[str, BufferConfig]: ...

    def prepare_constants(
        self, size: Any, *, out: Mapping[str, torch.Tensor]
    ) -> None: ...


def _require(loaded, kind: type):
    module = capability(loaded.module, kind)
    if module is None:
        raise SetupError(f"the model has no {kind.__name__}")
    return module


def _video(frames: int, width: int, height: int) -> video.Config:
    return video.Config(int(frames), image.Config(int(height), int(width)))


def _elements(shape: Sequence[int]) -> int:
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _first_frames(decoder: VideoDecoder, frames: int = 1) -> int:
    """The shortest video of at least ``frames`` the decoder reconstructs.

    The overlap state and RGB constants of the shortest of all are every
    video's of its canvas.
    """
    while True:
        try:
            decoder.frame_slices(frames)
        except ValueError:
            frames += 1
            continue
        return frames


class DecoderRunner:
    """The forward of one decoder on one stream at one size.

    Args:
        loaded: The loaded model.
        path: The decoder module's path in the model.
        stream: The CUDA stream handle every forward enqueues on.
        attention: The attention backend the model loaded with.
        scratch: The stream's shared work areas
            (``uniserve.bindings.context.Scratch``).
        size: ``[frames, width, height]`` of a video decoder's segment, or
            ``[latent_frames]`` of an audio decoder's track.
    """

    # An image decoder and its caller's tensors (see :meth:`bound`).
    _image: ImageDecoder
    _latents: torch.Tensor
    _staging: torch.Tensor

    def __init__(
        self,
        loaded,
        path: str,
        stream: int,
        attention: str,
        scratch,
        size: Sequence[int],
    ) -> None:
        module = loaded.module.get_submodule(str(path))
        self._device = next(module.parameters()).device
        extents = [int(extent) for extent in size]
        self._decode: Callable[..., Any]
        if isinstance(module, VideoDecoder):
            frames, width, height = extents
            self._decoder = module
            self._segment = _video(frames, width, height)
            self._context: ExecutionContext = execution_context(
                module, int(stream), str(attention), scratch
            )
            self._context.prepare(self._segment)
            self._postprocessor = _require(loaded, VideoPostprocessor)

            # The canvas's RGB reconstruction constants and the shape of
            # the overlap a unit carries are those of its shortest video.
            first = _video(_first_frames(module), width, height)
            buffered = cast(_Buffered, self._postprocessor)
            requirements = buffered.constant_buffers(first)
            self._constants = TensorBuffers.allocate(
                requirements, device=self._device
            ).view(requirements)
            with torch.inference_mode():
                buffered.prepare_constants(first, out=self._constants)
            overlap = self._postprocessor.state_buffers(first)["video_overlap"]
            self._overlap_shape = tuple(overlap.shape)

            # The decoder input and the overlap the unit ring delivers are
            # read and written by the forward alone.
            window = module.window_input(self._segment)
            self._window = torch.empty(
                window.shape, dtype=window.dtype, device=self._device
            )
            self._exchange = torch.empty(
                self._overlap_shape, dtype=overlap.dtype, device=self._device
            )

            # A unit's reconstruction depends on its video only through
            # whether the unit is the video's first and whether it is its
            # last. The shortest videos holding a unit of each kind stand
            # for every video of the raster.
            self._units: dict[tuple[bool, bool], tuple[Any, slice]] = {}
            length, count = 0, 0
            while len(self._units) < 4 and count < 3:
                length = _first_frames(module, length + 1)
                shortest = video.Config(length, self._segment.frame)
                slices = module.frame_slices(length)
                count = len(slices)
                for index, unit in enumerate(slices):
                    kind = (index == 0, index == count - 1)
                    self._units.setdefault(kind, (shortest, unit))
            self._decode = self._decode_video
        elif isinstance(module, AudioDecoder):
            (frames,) = extents
            self._audio = module
            self._context = execution_context(
                module, int(stream), str(attention), scratch
            )
            self._context.prepare(frames)
            self._decode = self._decode_audio
        else:
            raise SetupError(
                f"{type(module).__qualname__} is not a video or audio decoder"
            )

    @classmethod
    def image(
        cls,
        loaded,
        path: str,
        stream: int,
        device: int,
        scratch,
        latents: torch.Tensor,
        staging: torch.Tensor,
    ) -> DecoderRunner:
        """A runner of an image model's decoder.

        Args:
            loaded: The loaded model.
            path: The decoder module's path in the model.
            stream: The CUDA stream handle every forward enqueues on (0 on
                the host).
            device: The CUDA ordinal ``stream`` belongs to, -1 for the
                host. A decoder placed on a generation tower computes
                there and returns its results to the caller's tensors.
            scratch: The stream's shared work areas.
            latents: The latent pool's step workspace ``[units, width]``,
                into which the caller gathers each completed trajectory's
                pages before decoding.
            staging: uint8 bytes receiving each image's quantized HWC
                pixels, which the caller copies to the host and encodes.
        """
        path = str(path)
        decoder = loaded.module.get_submodule(path) if path else loaded.module
        if not isinstance(decoder, ImageDecoder):
            raise SetupError(
                f"{type(decoder).__qualname__} is not an image decoder"
            )
        context = execution_context(
            decoder,
            int(stream),
            loaded.attention,
            scratch,
            device=int(device),
            derive_host_lengths=True,
        )
        context.prepare(None)
        runner = cls.__new__(cls)
        runner._image = decoder
        runner._context = context
        runner._device = context.device
        runner._latents = latents
        runner._staging = staging
        runner._decode = runner._decode_image
        return runner

    # Sizing, answered from configuration.

    @staticmethod
    def value_range(loaded) -> list[float]:
        """The image decoder's ``[low, high]`` pixel range.

        Empty without an image decoder.
        """
        decoder = capability(loaded.module, ImageDecoder)
        if decoder is None:
            return []
        return [float(value) for value in decoder.decoder.value_range]

    @staticmethod
    def frame_rate(loaded) -> int:
        """Output video frames per second."""
        return int(_require(loaded, VideoPostprocessor).frame_rate)

    @staticmethod
    def sample_rate(loaded) -> int:
        """Output audio samples per second."""
        return int(_require(loaded, AudioDecoder).sample_rate)

    @staticmethod
    def unit_frames(loaded, frames: int) -> list[int]:
        """Frames of each media unit of a video of ``frames`` frames."""
        decoder = _require(loaded, VideoDecoder)
        return [
            window.stop - window.start
            for window in decoder.frame_slices(int(frames))
        ]

    @staticmethod
    def segment(
        loaded, frames: int, width: int, height: int, unit: int
    ) -> list[int]:
        """The ``[frames, width, height]`` segment one unit decodes to."""
        decoder = _require(loaded, VideoDecoder)
        size = _video(frames, width, height)
        window = decoder.frame_slices(size.num_frames)[int(unit)]
        segment = decoder.segment(size, window)
        return [
            int(segment.num_frames),
            int(segment.frame.width),
            int(segment.frame.height),
        ]

    @staticmethod
    def window_rows(
        loaded, frames: int, width: int, height: int, unit: int
    ) -> list[int]:
        """The rows of a video's packed latent that one unit decodes.

        The video has ``frames`` frames at the raster; the result lists, in
        the order the decoder reads them, the rows of its complete packed
        latent that the window of media unit ``unit`` holds.
        """
        decoder = _require(loaded, VideoDecoder)
        size = _video(frames, width, height)
        window = decoder.frame_slices(size.num_frames)[int(unit)]
        return [int(row) for row in decoder.window_rows(window, size)]

    @staticmethod
    def overlap_elements(loaded, width: int, height: int) -> int:
        """Elements of the overlap a canvas's unit carries to the next."""
        decoder = _require(loaded, VideoDecoder)
        size = _video(_first_frames(decoder), width, height)
        state = _require(loaded, VideoPostprocessor).state_buffers(size)
        return _elements(state["video_overlap"].shape)

    @staticmethod
    def overlap_dtype(loaded, width: int, height: int) -> str:
        """The dtype of that overlap."""
        decoder = _require(loaded, VideoDecoder)
        size = _video(_first_frames(decoder), width, height)
        state = _require(loaded, VideoPostprocessor).state_buffers(size)
        return str(state["video_overlap"].dtype).removeprefix("torch.")

    @staticmethod
    def track_samples(loaded, frames: int) -> int:
        """Samples of the audio track generated with ``frames`` frames."""
        clock = _require(loaded, VideoPostprocessor)
        audio = _require(loaded, AudioDecoder)
        return int(audio.track_samples(int(frames), clock.frame_rate))

    @staticmethod
    def audio_units(loaded, samples: int, units: int) -> list[int]:
        """``[start, stop]`` samples of each of a track's ``units`` units."""
        spans = _require(loaded, AudioDecoder).unit_samples(
            int(samples), int(units)
        )
        return [
            int(value) for span in spans for value in (span.start, span.stop)
        ]

    @staticmethod
    def audio_latent_frames(loaded, samples: int) -> int:
        """Latent frames of a track of ``samples`` samples."""
        return int(_require(loaded, AudioDecoder).latent_frames(int(samples)))

    # Forward.

    def decode(self, *arguments: Any) -> Any:
        """Decode into the caller's output tensor.

        A video decoder takes ``(latent, rows, overlap, out, first, last,
        unit_count)`` and decodes and reconstructs one media unit at the
        runner's raster. ``latent`` holds the request's complete packed
        video latent in its leading rows; ``rows`` (int64) names the rows
        of it that the unit's window reads (:meth:`window_rows`);
        ``overlap`` is the request's retained overlap (flat, the decoded
        tail its previous round left), which the round's first unit blends
        with and replaces; ``out`` the unit's ``[longest, height, width,
        3]`` uint8 frames, whose leading frames receive the unit's RGB.
        ``first`` and ``last`` tell whether the unit is its video's first
        and last. The post-processor exchanges overlaps with the
        ``unit_count`` ring members holding a unit of the round, so every
        one of them runs the forward.

        An audio decoder takes ``(latent, out, samples, first, count,
        total)`` and decodes audio units ``[first, first + count)`` of
        ``total`` of a track of ``samples`` samples. ``latent`` is the
        request's complete audio latent timeline; the units' samples, which
        abut, land in ``out`` (``[span, 2]`` int16) in order.

        An image decoder takes ``(rows, *images)``: ``rows`` holds one
        ``[offset, units, height, width, staging]`` row per image,
        flattened: its latent rows ``latents[offset:offset + units]``, its
        size and the staging offset of its pixels. It decodes every image
        (one numerical call per size), writes image ``i`` to the product
        tensor ``images[i]`` unless that is None (the decoded values in the
        product's dtype), quantizes it into the staging bytes and returns
        each image's byte count, ``height * width * 3``.
        """
        return self._decode(*arguments)

    def quantize(
        self, image: torch.Tensor, low: float, high: float, offset: int
    ) -> int:
        """Quantize a resident image product into the staging bytes.

        ``image`` holds values in ``[low, high]``; its HWC pixels land at
        ``offset`` of the staging bytes. Returns their byte count.
        """
        with calls.scope(self._context):
            return self._stage(
                image,
                float(low),
                float(high),
                int(offset),
            )

    def _stage(
        self, value: torch.Tensor, low: float, high: float, offset: int
    ) -> int:
        """Quantize one image into the staging bytes at ``offset``."""
        staging = self._staging
        pixels = quantize_image_hwc(value, value_range=(low, high))
        count = pixels.numel()
        if offset < 0 or offset + count > staging.numel():
            raise ValueError("image pixels exceed the decoder staging")
        staging[offset : offset + count].copy_(
            calls.local(pixels.reshape(-1), staging.device)
        )
        return count

    def _decode_image(
        self, rows: Sequence[int], *images: torch.Tensor | None
    ) -> list[int]:
        decoder, context = self._image, self._context
        flat = [int(value) for value in rows]
        described = [
            tuple(flat[start : start + 5]) for start in range(0, len(flat), 5)
        ]
        if len(described) != len(images):
            raise ValueError("decoder rows and image products must align")
        low, high = (float(value) for value in decoder.decoder.value_range)
        with calls.scope(context):
            # A decoder placed on a generation tower reads its latents there.
            values = decoder.decode(
                tuple(
                    self._latents[offset : offset + units].to(
                        context.device, non_blocking=True
                    )
                    for offset, units, *_ in described
                ),
                sizes=tuple(
                    image.Config(height, width)
                    for _, _, height, width, _ in described
                ),
            )
            counts = []
            for value, row, product in zip(
                values, described, images, strict=True
            ):
                if product is not None:
                    calls.write(product, value)
                counts.append(self._stage(value, low, high, row[4]))
            return counts

    def _decode_video(
        self,
        latent: torch.Tensor,
        rows: torch.Tensor,
        overlap: torch.Tensor,
        out: torch.Tensor,
        first: int,
        last: int,
        unit_count: int,
    ) -> None:
        decoder, postprocessor = self._decoder, self._postprocessor
        size, frames_slice = self._units[bool(first), bool(last)]
        retained = overlap.reshape(-1)[: self._exchange.numel()].view(
            self._overlap_shape
        )
        with torch.inference_mode(), self._context.activate():
            decoder.unpack_window(
                latent.index_select(0, rows), self._segment, out=self._window
            )
            decoded = decoder.decode(
                (self._window,), segments=(self._segment,)
            )[0]
            postprocessor(
                (decoder.place(decoded, frames_slice, size),),
                frames=(frames_slice,),
                sizes=(size,),
                state={"video_overlap": retained},
                constants=self._constants,
                workspace={
                    "rgb_frames": out,
                    "overlap_exchange": self._exchange,
                },
                unit_count=int(unit_count),
            )

    def _decode_audio(
        self,
        latent: torch.Tensor,
        out: torch.Tensor,
        samples: int,
        first: int,
        count: int,
        total: int,
    ) -> None:
        audio = self._audio
        samples = int(samples)
        windows = audio.unit_frames(samples, int(total))[
            int(first) : int(first) + int(count)
        ]
        with torch.inference_mode(), self._context.activate():
            outputs = audio.decode(
                (latent,) * len(windows),
                frames=tuple(windows),
                num_samples=(samples,) * len(windows),
                workspace=self._context.workspace_views,
            )
            written = sum(int(output.shape[0]) for output in outputs)
            if tuple(out.shape) != (written, 2):
                raise ValueError("audio units disagree with their output rows")
            torch.cat(outputs, out=out)

    def kernels(self) -> list[str]:
        """The kernels the runner's operators selected, as JSON objects."""
        return kernel_records(self._context.kernels())

    def close(self, aborted: bool) -> None:
        """Release the runner's operators and workspace.

        An aborted close keeps everything device work may still address.
        """
        self._context.close(aborted=bool(aborted))
