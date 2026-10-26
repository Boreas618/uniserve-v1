"""The forwards of the media decoders: :class:`DecoderRunner`.

One runner serves one decoder module on one stream: a video decoder at one
clip output (the native frames and raster a media unit's clip decodes to),
or an audio decoder at one track length; an image model's decoder is the
image executor's (``uniserve.executor.image``). The tensors of a forward are
parameters of :meth:`DecoderRunner.decode`; the parameters depend on the
decoder's capability class and are documented there. A video decoder's
forward has shapes fixed by its scalar arguments and takes every changing
value as the contents of a tensor, so it is captured into one CUDA graph
per kind of unit and round size.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import torch

from uniserve.bindings.context import ExecutionContext
from uniserve.bindings.workspace import TensorBuffers
from uniserve.errors import SetupError
from uniserve.loading.rank import capability
from uniserve.media import image, video
from uniserve.models import (
    AudioDecoder,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.runner.context import (
    execution_context,
    kernel_records,
)
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


def _first_frames(decoder: VideoDecoder, frames: int = 1) -> int:
    """The shortest video of at least ``frames`` the decoder reconstructs.

    The overlap state and RGB constants of the shortest of all are every
    video's of its canvas.
    """
    while True:
        try:
            decoder.clips(frames)
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
            (``uniserve.bindings.workspace.Scratch``).
        size: ``[frames, width, height]`` of a video decoder's clip output,
            or ``[latent_frames]`` of an audio decoder's track.
    """

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
            self._output = _video(frames, width, height)
            self._context: ExecutionContext = execution_context(
                module, int(stream), str(attention), scratch
            )
            self._context.prepare(self._output)
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
            clip_input = module.clip_input(self._output)
            self._input = torch.empty(
                clip_input.shape, dtype=clip_input.dtype, device=self._device
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
                shortest = video.Config(length, self._output.frame)
                clips = module.clips(length)
                count = len(clips)
                for index, unit in enumerate(clips):
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

    # Forward.

    def decode(self, *arguments: Any) -> Any:
        """Decode into the caller's output tensor.

        A video decoder takes ``(latent, rows, overlap, out, first, last,
        unit_count)`` and decodes and reconstructs one media unit at the
        runner's raster. ``latent`` holds the request's complete packed
        video latent in its leading rows; ``rows`` (int64) names the rows
        of it that the unit's clip input reads
        (``uniserve.executor.video.decoder.clip``);
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
        """
        return self._decode(*arguments)

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
        size, clip = self._units[bool(first), bool(last)]
        retained = overlap.reshape(-1)[: self._exchange.numel()].view(
            self._overlap_shape
        )
        with torch.inference_mode(), self._context.activate():
            decoder.unpack_rows(
                latent.index_select(0, rows), self._output, out=self._input
            )
            decoded = decoder.decode(
                (self._input,), clip_outputs=(self._output,)
            )[0]
            postprocessor(
                (decoder.place(decoded, clip, size),),
                clips=(clip,),
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
