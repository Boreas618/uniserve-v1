"""Host media functions: condition decoding, codecs and MP4 assembly.

Host ranks read a video request's condition media and encode and assemble
its output. The worker owns every piece of state around these calls: which
request a call serves, the shared-memory objects and products it reads and
writes, the FFmpeg process that decodes reference videos, the order and
decode times of the units an MP4 receives, and when the MP4 is complete.
These functions decode and encode the values they are handed:

- ``media.require_codecs()``: fail startup when PyAV lacks a serving codec.
- ``media.read_image(data, fit, out)`` and ``media.read_audio(data, clip,
  rate, out)``: decode one condition's bytes into its product rows.
- ``media.reader(loaded) -> {"pack_vision", "sample_rate"}``: the vision
  patch packing of the model's conditioner and the audio encoder's rate,
  which a host rank answers from the model's description.
- ``media.encode_video_unit(rgb, frame_rate) -> bytes`` and
  ``media.encode_audio_track(pcm, samples, rate) -> bytes``.
- ``media.mux_open(first_unit, frame_rate, audio_rate) -> container``,
  ``media.mux_append(container, unit, offset) -> end``,
  ``media.mux_finish(container, audio) -> bytes`` and
  ``media.mux_close(container)``: one MP4's container, an opaque handle the
  worker passes back.

Tensors are CPU views of worker-owned storage; every call runs on the
calling host thread.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from uniserve_models.media import codec, reader
from uniserve_models.model import AudioEncoder, PatchEncoder

from .errors import SetupError, binding


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "media.require_codecs": codec.require_codecs,
        "media.read_image": read_image,
        "media.read_audio": read_audio,
        "media.reader": bind_reader,
        "media.encode_video_unit": encode_video_unit,
        "media.encode_audio_track": encode_audio_track,
        "media.mux_open": mux_open,
        "media.mux_append": mux_append,
        "media.mux_finish": mux_finish,
        "media.mux_close": mux_close,
    }


def _host(tensor) -> torch.Tensor:
    value = torch.from_dlpack(tensor)
    if value.device.type != "cpu":
        raise ValueError("host media functions take CPU tensors")
    return value


def read_image(data, fit: Sequence[int], out) -> None:
    """Decode one image condition into ``out``.

    ``data`` is the condition's encoded bytes as a uint8 tensor; ``fit`` is
    ``[resized_width, resized_height, left, top, width, height]``; ``out``
    receives ``[1, height, width, 3]`` uint8 RGB.
    """
    width, height, left, top, kept_width, kept_height = (int(v) for v in fit)
    pixels = reader.read_image(
        memoryview(_host(data).numpy()),
        resized=(width, height),
        window=(left, top, kept_width, kept_height),
    )
    target = _host(out)
    if tuple(target.shape) != pixels.shape:
        raise ValueError("a condition image's rows disagree with its plan")
    target.copy_(torch.from_numpy(pixels))


def read_audio(data, clip: Sequence[int], rate: int, out) -> None:
    """Decode one condition's audio track into ``out``.

    ``clip`` is ``[native_rate, start, source_samples, samples]`` and
    ``out`` receives ``[samples, 2]`` FP32 PCM at ``rate``.
    """
    native, start, kept, samples = (int(value) for value in clip)
    waveform = reader.read_audio(
        memoryview(_host(data).numpy()),
        native_rate=native,
        start=start,
        source_samples=kept,
        samples=samples,
        rate=int(rate),
    )
    target = _host(out)
    if target.shape != waveform.shape:
        raise ValueError("a condition track's rows disagree with its plan")
    target.copy_(waveform)


def _unique(model, kind: type):
    found = [module for module in model.modules() if isinstance(module, kind)]
    unique = list({id(module): module for module in found}.values())
    if len(unique) != 1:
        raise SetupError(
            f"media reading requires one {kind.__name__} capability"
        )
    return unique[0]


def bind_reader(loaded) -> dict[str, Any]:
    """Return the condition reader's model functions.

    ``pack_vision(frames, frame_indices, grid, out)`` packs the
    conditioner's patch rows of ``[frames, height, width, 3]`` uint8
    frames (the frames at ``frame_indices``, or every frame when it is
    empty) on a ``[time, height, width]`` patch grid into ``out``.
    ``sample_rate`` is the audio encoder's rate. Neither reads a parameter,
    so a host rank holding the model's description serves them.
    """
    vision = _unique(loaded.module, PatchEncoder)
    audio = _unique(loaded.module, AudioEncoder)

    def pack_vision(frames, frame_indices, grid, out) -> None:
        values = _host(frames)
        indices = [int(index) for index in frame_indices]
        sampled = values[indices] if indices else values
        rows = vision.pack_pixels(sampled, tuple(int(v) for v in grid))
        target = _host(out)
        if target.shape != rows.shape or target.dtype != rows.dtype:
            raise ValueError("a condition's patch rows disagree with its plan")
        target.copy_(rows)

    return binding(
        {
            "pack_vision": pack_vision,
            "sample_rate": int(audio.sample_rate),
        }
    )


def encode_video_unit(rgb, frame_rate: int) -> bytes:
    """Encode one media unit's ``[frames, height, width, 3]`` RGB24 frames."""
    return codec.encode_video_unit(
        _host(rgb).numpy(), frame_rate=int(frame_rate)
    )


def encode_audio_track(pcm, samples: int, rate: int) -> bytes:
    """Encode ``[samples, 2]`` int16 PCM truncated or padded to ``samples``."""
    return codec.encode_audio_track(
        _host(pcm).numpy(), samples=int(samples), rate=int(rate)
    )


def mux_open(first_unit: bytes, frame_rate: int, audio_rate: int):
    """Open one MP4 whose streams follow the first encoded unit and AAC."""
    return codec.Container.open(
        bytes(first_unit),
        frame_rate=int(frame_rate),
        audio_rate=int(audio_rate),
    )


def mux_append(container: codec.Container, unit: bytes, offset: int) -> int:
    """Mux one unit at decode time ``offset``; returns its end."""
    return int(container.append(bytes(unit), int(offset)))


def mux_finish(container: codec.Container, audio: bytes) -> bytes:
    """Mux the audio track and return the complete MP4."""
    return container.finish(bytes(audio))


def mux_close(container: codec.Container) -> None:
    """Discard an unfinished MP4."""
    container.close()
