"""Decode a video request's condition media exactly as its plan states.

The server fetches every condition's bytes and plans how each is prepared;
the serving worker reads the bytes and calls these decoders with the plan's
values. They apply the reference conditioning's operations in its order, so
their results equal the reference's own:

- :func:`read_image`: PIL decoding with the EXIF orientation applied and
  RGB conversion (diffusers' ``load_image``), then a LANCZOS resize and the
  planned crop: a stretched keyframe, a cover-cropped one, or an image
  reference at its planned raster.
- :func:`read_audio`: PyAV decoding to planar float at the track's native
  rate and channel layout, the planned native samples kept, a mono track
  duplicated to stereo, and one torchaudio resample to the model's rate.

Reference videos are decoded by one FFmpeg pass the worker runs itself.
Every function checks that it produced exactly the planned extent and raises
``ValueError`` otherwise, so media that decodes differently from what the
server probed fails the request instead of misaligning its rows.
"""

from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np
import torch

__all__ = ["read_audio", "read_image"]


def read_image(
    data: bytes | memoryview,
    *,
    resized: tuple[int, int],
    window: tuple[int, int, int, int],
) -> np.ndarray:
    """Decode an image and put it on its planned raster.

    ``resized`` is the ``(width, height)`` the decoded image is resized to,
    and ``window`` the kept ``(left, top, width, height)`` of the resized
    image.

    Returns:
        ``[1, height, width, 3]`` uint8 RGB of the kept window.

    Raises:
        ValueError: The image decodes to another raster than planned.
    """
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(io.BytesIO(bytes(data))))
    image = image.convert("RGB")
    # Pillow returns a copy for a resize to the image's own size, so an
    # image already at its raster passes through unchanged.
    if image.size != resized:
        image = image.resize(resized, Image.Resampling.LANCZOS)
    left, top, width, height = window
    if (left, top, width, height) != (0, 0, *resized):
        image = image.crop((left, top, left + width, top + height))
    # A writable copy, which tensors may view.
    pixels = np.array(image)
    if pixels.shape != (height, width, 3):
        raise ValueError("a condition image decoded to another raster")
    return pixels[None]


def read_audio(
    source: bytes | memoryview | BinaryIO,
    *,
    native_rate: int,
    start: int,
    source_samples: int,
    samples: int,
    rate: int,
) -> torch.Tensor:
    """Decode an audio track's kept samples at the model's rate.

    ``source`` holds the container's encoded bytes or an open binary file;
    its first audio stream is read at ``native_rate``. The ``source_samples``
    native samples from ``start`` are kept, a mono track is duplicated to
    stereo, and one resample brings the track to ``rate``.

    Returns:
        ``[samples, 2]`` FP32 sample-major PCM.

    Raises:
        ValueError: The track has no audio stream, another rate, too few
            samples, or resamples to another count.
    """
    import av

    container = av.open(
        io.BytesIO(bytes(source))
        if isinstance(source, (bytes, memoryview))
        else source,
        mode="r",
    )
    with container:
        if not container.streams.audio:
            raise ValueError("a condition carries no audio stream")
        stream = container.streams.audio[0]
        native = int(stream.codec_context.sample_rate)
        if native != native_rate:
            raise ValueError(
                f"an audio track decodes at {native} Hz, "
                f"{native_rate} Hz were planned"
            )
        # Planar float is a format conversion only: rate and layout stay
        # the stream's own.
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout=stream.layout, rate=native
        )
        chunks: list[torch.Tensor] = []
        for frame in container.decode(stream):
            chunks.extend(
                torch.from_numpy(part.to_ndarray())
                for part in resampler.resample(frame)
            )
        chunks.extend(
            torch.from_numpy(part.to_ndarray())
            for part in resampler.resample(None)
        )
    # [channels, samples] at the native rate.
    waveform = torch.cat(chunks, dim=-1).to(torch.float32)
    stop = start + source_samples
    if waveform.shape[-1] < stop or waveform.shape[0] not in (1, 2):
        raise ValueError("an audio track holds fewer samples than planned")
    waveform = waveform[:, start:stop]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if native != rate:
        import torchaudio

        waveform = torchaudio.transforms.Resample(native, rate)(waveform)
    if waveform.shape[-1] != samples:
        raise ValueError(
            f"an audio track resampled to {waveform.shape[-1]} samples, "
            f"{samples} were planned"
        )
    return waveform.transpose(0, 1).contiguous()
