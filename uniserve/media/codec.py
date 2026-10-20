"""Video and audio encoders and MP4 container operations through PyAV.

A video request's output is assembled from independently encoded media
units: each unit is a self-contained MP4 of one H.264 stream starting at a
keyframe (:func:`encode_video_unit`), the soundtrack is one AAC MP4
(:func:`encode_audio_track`), and a :class:`Container` remuxes their packets
into the request's MP4 without decoding them. The serving worker owns the
assembly's state (which units arrived, in which order, at which decode
time); these functions hold the codec work only.

A CPU worker runs these calls on its host lane's thread. Closing an x264
encoder joins its thread pool while PyAV holds the interpreter lock, so a
codec call stalls other Python calls of the process for its duration; host
ranks have no device work to launch beside it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    # PyAV is imported where a codec runs; its names here only annotate.
    from av.container import OutputContainer

__all__ = [
    "AUDIO_CODEC",
    "VIDEO_CODEC",
    "Container",
    "encode_audio_track",
    "encode_video_unit",
    "require_codecs",
]

# The encoders every served container uses.
VIDEO_CODEC: Final = "libx264"
AUDIO_CODEC: Final = "aac"

# Each media unit is its own encoder session, and several ranks encode at once
# on one host, so a session must not size its thread pool to the machine.
# Measured at this preset and raster, eight threads encode a unit as fast as an
# unbounded pool does; one thread takes four to five times as long. The
# worker's bound on an encoded unit's bytes budgets one slice header per
# thread per frame, so it follows this value.
ENCODER_THREADS = 8

# Samples of one planar AAC frame.
AUDIO_FRAME_SAMPLES = 1024


def require_codecs() -> None:
    """Verify that the serving encoders are available through PyAV.

    Both codec names must be listed by PyAV and open as an encoder context.

    Raises:
        RuntimeError: PyAV is not installed or does not list a codec. PyAV
            raises its own error when a listed codec cannot be created as an
            encoder.
    """
    try:
        import av
    except ImportError as error:
        raise RuntimeError("media output requires the PyAV runtime") from error
    missing = [
        name
        for name in (VIDEO_CODEC, AUDIO_CODEC)
        if name not in av.codecs_available
    ]
    if missing:
        raise RuntimeError(
            "media output requires the codecs "
            + ", ".join(missing)
            + " which this PyAV runtime does not provide"
        )
    for name in (VIDEO_CODEC, AUDIO_CODEC):
        av.CodecContext.create(name, "w")


def encode_video_unit(rgb24, frame_rate: int) -> bytes:
    """Encode one media unit as a self-contained MP4 starting at a keyframe.

    ``rgb24`` (an array or a CPU tensor) holds the unit's uint8 frames as
    ``[frames, height, width, 3]``. Frame timestamps start at zero in every
    unit; the assembly shifts them onto the request's timeline.

    Raises:
        ValueError: ``rgb24`` holds no frame or is not RGB24, or the frame
            rate is not positive. PyAV raises its own errors for codec and
            encoding failures.
    """
    import av

    rgb24 = np.asarray(rgb24)
    if (
        rgb24.ndim != 4
        or rgb24.shape[0] < 1
        or rgb24.shape[3] != 3
        or rgb24.dtype != np.uint8
    ):
        raise ValueError("video capture has invalid RGB24 dimensions")
    if type(frame_rate) is not int or frame_rate < 1:
        raise ValueError("video frame rate must be a positive integer")
    height, width = int(rgb24.shape[1]), int(rgb24.shape[2])
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        stream = container.add_stream(VIDEO_CODEC, rate=frame_rate)
        stream.width, stream.height = width, height
        # The worker's bound on an encoded unit's bytes derives from these
        # settings (baseline CAVLC at this preset); a change it does not
        # account for can overflow the unit's reserved product row, which
        # the worker then rejects by name.
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
        stream.codec_context.thread_count = ENCODER_THREADS
        for index, pixels in enumerate(rgb24):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, frame_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return buffer.getvalue()


def encode_audio_track(pcm, samples: int, rate: int) -> bytes:
    """Encode stereo PCM as a self-contained AAC MP4.

    ``pcm`` (an array or a CPU tensor) holds interleaved int16 samples as
    ``[samples, 2]``. The track
    is truncated or zero-padded to ``samples`` (the video's duration at
    ``rate``) and encoded in planar frames of 1024 samples; the last frame
    is zero-padded to that size when the track does not fill it.

    Raises:
        ValueError: ``pcm`` is not a ``[samples, 2]`` int16 array, or the
            target length or rate is not positive. PyAV raises its own errors
            for codec and encoding failures.
    """
    import av

    pcm = np.asarray(pcm)
    if pcm.ndim != 2 or pcm.shape[1] != 2 or pcm.dtype != np.int16:
        raise ValueError("audio capture has invalid stereo dimensions")
    if type(samples) is not int or samples < 1 or type(rate) is not int:
        raise ValueError("audio track length and rate must be positive")
    if rate < 1:
        raise ValueError("audio track length and rate must be positive")
    source = pcm[:samples]
    if source.shape[0] < samples:
        source = np.pad(source, ((0, samples - source.shape[0]), (0, 0)))

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        stream = container.add_stream(AUDIO_CODEC, rate=rate)
        stream.layout = "stereo"
        stream.sample_rate = rate
        stream.bit_rate = 144_000
        stream.options = {"aac_coder": "fast"}
        stream.codec_context.thread_count = ENCODER_THREADS
        pts = 0
        for start in range(0, samples, AUDIO_FRAME_SAMPLES):
            stop = min(start + AUDIO_FRAME_SAMPLES, samples)
            planar = np.zeros((2, AUDIO_FRAME_SAMPLES), dtype=np.int16)
            planar[:, : stop - start] = source[start:stop].T
            frame = av.AudioFrame.from_ndarray(
                planar, format="s16p", layout="stereo"
            )
            frame.sample_rate = rate
            frame.pts = pts
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
            pts += AUDIO_FRAME_SAMPLES
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return buffer.getvalue()


def _read(payload: bytes):
    import av

    return av.open(io.BytesIO(bytes(payload)), mode="r")


@dataclass
class Container:
    """An MP4 being assembled from encoded units and one audio track.

    The container's video stream is templated from the first unit and its
    audio stream from a one-sample silent AAC encoding, because both streams
    must exist before the MP4 header is written. The caller orders the
    units and owns their decode-time offsets: :meth:`append` shifts one
    unit's packets by the offset it is given and returns where the unit
    ends, the offset of the unit that follows.
    """

    buffer: io.BytesIO
    output: OutputContainer
    video: object
    audio: object

    @classmethod
    def open(
        cls, first_unit: bytes, frame_rate: int, audio_rate: int
    ) -> Container:
        """Open an empty container whose streams match the request's codecs.

        The audio template encodes one video frame's duration of silence.

        Raises:
            ValueError: ``first_unit`` has no video stream.
        """
        import av

        buffer = io.BytesIO()
        output = av.open(buffer, mode="w", format="mp4")
        try:
            with _read(first_unit) as source:
                if not source.streams.video:
                    raise ValueError("an encoded media unit has no video")
                video = output.add_stream_from_template(source.streams.video[0])
            silence = encode_audio_track(
                np.zeros((1, 2), dtype=np.int16),
                samples=round(audio_rate / frame_rate),
                rate=audio_rate,
            )
            with _read(silence) as track:
                audio = output.add_stream_from_template(track.streams.audio[0])
        except BaseException:
            output.close()
            raise
        return cls(buffer, output, video, audio)

    def append(self, unit: bytes, offset: int) -> int:
        """Mux one unit's packets at decode-time ``offset``.

        Returns the decode time after the unit's last packet: its largest
        ``dts + duration`` (a zero duration counts one tick).

        Raises:
            ValueError: ``unit`` has no video stream.
        """
        last = offset
        with _read(unit) as source:
            if not source.streams.video:
                raise ValueError("an encoded media unit has no video")
            for packet in source.demux(source.streams.video[0]):
                # Demux ends with an empty flush packet without DTS.
                if packet.dts is None:
                    continue
                pts = 0 if packet.pts is None else packet.pts
                packet.stream = self.video
                packet.pts = pts + offset
                packet.dts = packet.dts + offset
                self.output.mux(packet)
                duration = packet.duration or 1
                last = max(last, packet.dts + duration)
        return last

    def finish(self, audio: bytes) -> bytes:
        """Mux the encoded audio track and close the container.

        Returns the complete MP4.

        Raises:
            RuntimeError: The container came out empty.
        """
        with _read(audio) as track:
            for packet in track.demux(track.streams.audio[0]):
                if packet.dts is None:
                    continue
                packet.stream = self.audio
                self.output.mux(packet)
        self.output.close()
        value = self.buffer.getvalue()
        if not value:
            raise RuntimeError("media mux produced an empty container")
        return value

    def close(self) -> None:
        """Discard an unfinished container."""
        self.output.close()
