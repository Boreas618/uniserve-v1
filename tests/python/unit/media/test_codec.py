"""Independently encoded media units assemble into one playable MP4.

The serving worker encodes each media unit on its own codec rank and
remuxes the units, in order and at the decode-time offsets it tracks, with
the request's soundtrack into the request's MP4. These tests drive the same
codec boundary (``uniserve.media.codec``) and decode the result.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from uniserve.media import codec

av = pytest.importorskip("av")

pytestmark = pytest.mark.unit

FRAME_RATE = 24
AUDIO_RATE = 32_000


def _unit(frames: int, shade: int) -> np.ndarray:
    """``frames`` 64x48 frames of one flat gray level."""
    return np.full((frames, 48, 64, 3), shade, dtype=np.uint8)


def _assemble(units: list[np.ndarray], audio_samples: int) -> bytes:
    encoded = [
        codec.encode_video_unit(unit, frame_rate=FRAME_RATE) for unit in units
    ]
    container = codec.Container.open(
        encoded[0], frame_rate=FRAME_RATE, audio_rate=AUDIO_RATE
    )
    offset = 0
    for payload in encoded:
        offset = container.append(payload, offset)
    track = codec.encode_audio_track(
        np.zeros((audio_samples, 2), dtype=np.int16),
        samples=audio_samples,
        rate=AUDIO_RATE,
    )
    return container.finish(track)


def test_units_play_back_in_order_on_one_timeline():
    units = [_unit(17, 40), _unit(17, 120), _unit(5, 200)]
    # 39 frames at 24 fps last 1.625 s, 52 000 samples at 32 kHz.
    payload = _assemble(units, 52_000)

    with av.open(io.BytesIO(payload)) as container:
        video = container.streams.video[0]
        assert (video.codec_context.name, video.width, video.height) == (
            "h264",
            64,
            48,
        )
        frames = [
            (frame.time, frame.to_ndarray(format="rgb24"))
            for frame in container.decode(video=0)
        ]
    times = [time for time, _ in frames]
    assert len(frames) == 39
    assert times == sorted(times)
    assert times[-1] == pytest.approx(38 / FRAME_RATE)
    # Each unit's frames keep their content, in unit order; the encoder
    # is lossy, so compare mean levels.
    levels = [int(round(pixels.mean())) for _, pixels in frames]
    for start, stop, shade in ((0, 17, 40), (17, 34, 120), (34, 39, 200)):
        assert all(abs(level - shade) <= 3 for level in levels[start:stop])

    with av.open(io.BytesIO(payload)) as container:
        audio = container.streams.audio[0]
        assert (audio.codec_context.name, audio.channels) == ("aac", 2)
        samples = sum(frame.samples for frame in container.decode(audio=0))
    # AAC frames hold 1024 samples; the track covers the video's duration.
    assert 52_000 <= samples < 52_000 + 2 * codec.AUDIO_FRAME_SAMPLES


def test_audio_tracks_are_cut_or_padded_to_the_video_duration():
    long = codec.encode_audio_track(
        np.ones((5000, 2), dtype=np.int16), samples=3000, rate=AUDIO_RATE
    )
    short = codec.encode_audio_track(
        np.ones((100, 2), dtype=np.int16), samples=3000, rate=AUDIO_RATE
    )
    for payload in (long, short):
        with av.open(io.BytesIO(payload)) as container:
            decoded = sum(frame.samples for frame in container.decode(audio=0))
        assert 3000 <= decoded < 3000 + 2 * codec.AUDIO_FRAME_SAMPLES


def test_malformed_units_are_refused():
    with pytest.raises(ValueError, match="RGB24"):
        codec.encode_video_unit(
            np.zeros((1, 8, 8, 4), np.uint8), frame_rate=FRAME_RATE
        )
    with pytest.raises(ValueError, match="frame rate"):
        codec.encode_video_unit(_unit(1, 0), frame_rate=0)
    with pytest.raises(ValueError, match="stereo"):
        codec.encode_audio_track(
            np.zeros((4, 1), np.int16), samples=4, rate=AUDIO_RATE
        )
    track = codec.encode_audio_track(
        np.zeros((4, 2), np.int16), samples=4, rate=AUDIO_RATE
    )
    with pytest.raises(ValueError, match="no video"):
        codec.Container.open(
            track, frame_rate=FRAME_RATE, audio_rate=AUDIO_RATE
        )


def test_a_callers_tensor_encodes_like_its_array():
    # The serving worker passes views of its own storage positionally.
    unit = _unit(5, 90)
    assert codec.encode_video_unit(
        torch.from_numpy(unit), FRAME_RATE
    ) == codec.encode_video_unit(unit, frame_rate=FRAME_RATE)

    pcm = np.arange(4096, dtype=np.int16).reshape(2048, 2)
    assert codec.encode_audio_track(
        torch.from_numpy(pcm), 2048, AUDIO_RATE
    ) == codec.encode_audio_track(pcm, samples=2048, rate=AUDIO_RATE)
