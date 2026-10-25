"""Host media of a video model.

It covers condition reading, unit and soundtrack encoding, and MP4
assembly.

Host media has no model. Its work runs on one host thread of the executor,
in submission order, beside the device work of the rank's streams; the
codecs release the GIL while they encode. Every finished task reports to
the worker as one message on the pipe the worker passes in
(:class:`Reports`): the batch and call it belongs to, a row, a status, a
value and a payload. The worker reads the pipe on a thread of its own, so a
result's memory is ordered before the worker reads it by the pipe's write
and read.

**Condition reading.** The server publishes every condition's fetched bytes
to POSIX shared memory and plans how each is prepared. One media reading
call per conditioned request decodes them, in request order, into the
request's condition products: ``condition_pixels`` (each visual
condition's RGB24 frames, ``[pixels, 3]`` uint8, frame-major),
``condition_samples`` (each audio track's model-rate stereo PCM,
``[samples, 2]`` FP32) and ``vision_pixels`` (the conditioner's packed patch
rows of every condition it reads). Reference videos are decoded by one
FFmpeg process each, on threads of their own; images, tracks and patch
packing are the library's. A CUDA rank decodes into page-locked staging and
copies it to its device products on a stream of its own.

**Encoding.** Each media unit becomes a self-contained MP4 that starts at a
keyframe, written into its row of the round's ``encoded_units`` product
behind a native-endian ``u64`` length; the muxer appends the rows to the
request's container in unit order, encodes the soundtrack once and
finalizes the container when every unit has arrived.
"""

from __future__ import annotations

import os
import queue
import struct
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from uniserve.media import codec
from uniserve.media.reader import read_audio, read_image
from uniserve.runner.encoder import EncoderRunner

__all__ = [
    "Conditions",
    "HostLane",
    "MuxSession",
    "Reports",
    "encoded_unit_bytes",
    "frame_unit",
    "read_unit",
]

# Bytes of an encoded unit's length prefix.
LENGTH_BYTES = 8

# Threads one serving encoder session uses; the encoded-size bound charges
# one slice header per thread per frame (``uniserve.media.codec``).
ENCODER_THREADS = 8

# A report's header: batch, call, row, status, value and payload bytes.
_HEADER = struct.Struct("<qqqqqq")

# Report statuses.
DONE = 1
FAILED = 2


def encoded_video_bytes(frames: int, height: int, width: int) -> int:
    """Bound the MP4 bytes of one unit from the serving encoder (libx264,
    yuv420p, ``preset=ultrafast``, ``tune=zerolatency``).

    This preset uses baseline CAVLC. A padded 16x16 macroblock has at most
    27 residual blocks (including the separate DC blocks), each with up to
    16 coefficients. Baseline level codes occupy at most 28 bits; x264
    re-encodes CAVLC overflows at a higher QP. Per-block token, zero and run
    codes need at most 16 + 9 + 15 * 11 bits. A further 1536 bits covers all
    macroblock headers and 16 pairs of motion-vector differences. NAL
    escaping adds at most one byte per two source bytes. Each encoder thread
    writes one slice header per frame of at most 1024 bytes; MP4 sample
    tables need at most 64 bytes per packet, and parameter sets, encoder SEI
    and fixed container boxes fit in 64 KiB. These are syntax bounds,
    independent of image entropy (x264 ``encoder/cavlc.c``,
    ``encoder/encoder.c`` and FFmpeg ``libavformat/movenc.c``).
    """  # noqa: D205
    blocks = -(-height // 16) * -(-width // 16)
    residual_bits = 27 * (16 + 16 * 28 + 9 + 15 * 11)
    macroblock_bytes = -(-(residual_bits + 1536) // 8)
    escaped_bytes = -(-(3 * macroblock_bytes) // 2)
    return (1 << 16) + frames * (
        blocks * escaped_bytes + ENCODER_THREADS * 1024 + 64
    )


def encoded_unit_bytes(frames: int, height: int, width: int) -> int:
    """Bytes of one ``encoded_units`` row: the length prefix and the largest
    encoding of a unit of ``frames`` frames at the raster.
    """  # noqa: D205
    return LENGTH_BYTES + encoded_video_bytes(frames, height, width)


def frame_unit(payload: bytes, row: torch.Tensor) -> int:
    """Write ``payload`` behind its native-endian ``u64`` length into
    ``row`` (uint8). Returns the framed length; bytes past it stay
    untouched, so only that prefix of the row is published.

    Raises:
        ValueError: The payload exceeds the row.
    """  # noqa: D205
    capacity = row.numel() - LENGTH_BYTES
    if row.numel() < LENGTH_BYTES or len(payload) > capacity:
        raise ValueError(
            f"encoded media unit of {len(payload)} bytes exceeds the "
            f"{capacity} bytes reserved for it"
        )
    framed = LENGTH_BYTES + len(payload)
    header = len(payload).to_bytes(LENGTH_BYTES, sys.byteorder)
    row[:framed].copy_(
        torch.frombuffer(bytearray(header + payload), dtype=torch.uint8)
    )
    return framed


def read_unit(row: bytes | memoryview) -> bytes:
    """The payload of one framed unit row.

    Raises:
        ValueError: The row has no length prefix or names an invalid length.
    """
    if len(row) < LENGTH_BYTES:
        raise ValueError("encoded media unit has no length prefix")
    length = int.from_bytes(bytes(row[:LENGTH_BYTES]), sys.byteorder)
    body = bytes(row[LENGTH_BYTES:])
    if length > len(body):
        raise ValueError("encoded media unit names an invalid length")
    return body[:length]


class Reports:
    """The pipe a host task's results reach the worker through.

    A report is a header (batch, call, row, status, value, payload bytes)
    and its payload, written under one lock so reports never interleave.
    """

    def __init__(self, descriptor: int) -> None:
        self._descriptor = int(descriptor)
        self._lock = threading.Lock()

    def send(
        self,
        batch: int,
        call: int,
        row: int,
        status: int,
        value: int = 0,
        payload: bytes = b"",
    ) -> None:
        message = memoryview(
            _HEADER.pack(batch, call, row, status, value, len(payload))
            + payload
        )
        with self._lock:
            while message:
                written = os.write(self._descriptor, message)
                message = message[written:]


class HostLane:
    """One host thread running media tasks in submission order."""

    def __init__(self, name: str) -> None:
        self._tasks: queue.SimpleQueue[Callable[[], None] | None] = (
            queue.SimpleQueue()
        )
        self._thread = threading.Thread(
            target=self._serve, name=name, daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            task()

    def submit(self, task: Callable[[], None]) -> None:
        self._tasks.put(task)

    def close(self) -> None:
        """Finish the queued tasks and stop the thread."""
        self._tasks.put(None)
        self._thread.join()


def _shared_bytes(source: Mapping[str, Any]) -> bytes:
    """The published bytes of a condition."""
    name, length = str(source["name"]), int(source["bytes"])
    try:
        with open(f"/dev/shm/{name}", "rb") as file:
            data = file.read(length)
    except OSError as error:
        raise ValueError(
            f"condition media {name} cannot be opened: {error}"
        ) from error
    if len(data) != length:
        raise ValueError("condition media is shorter than its locator")
    return data


@dataclass
class _Rows:
    """Where one condition's decoded media lands: its rows of each product."""

    pixels: torch.Tensor | None = None
    patches: torch.Tensor | None = None
    samples: torch.Tensor | None = None


@dataclass
class Conditions:
    """Decodes one request's conditions into its products.

    ``products`` maps ``condition_pixels``, ``condition_samples`` and
    ``vision_pixels`` to the host views each condition's rows are taken
    from, in request order; together the conditions fill each exactly.
    """

    loaded: Any
    ffmpeg: str
    # The frame rate the model conditions on, which videos resample to.
    frame_rate: int
    # The audio encoder's sample rate, which tracks resample to.
    sample_rate: int = field(init=False)

    def __post_init__(self) -> None:
        self.sample_rate = EncoderRunner.sample_rate(self.loaded)

    @staticmethod
    def rows(
        conditions: Sequence[Mapping[str, Any]],
        products: Mapping[str, torch.Tensor | None],
    ) -> list[_Rows]:
        """Each condition's rows of each product, taken in request order,
        which must fill every product exactly.

        Raises:
            ValueError: The conditions exceed a product or leave it unfilled.
        """  # noqa: D205
        cursor = {"pixels": 0, "samples": 0, "patches": 0}
        names = {
            "pixels": "condition_pixels",
            "samples": "condition_samples",
            "patches": "vision_pixels",
        }

        def take(kind: str, rows: int) -> torch.Tensor:
            product = products.get(names[kind])
            if product is None:
                raise ValueError(f"a condition's {kind} have no product")
            start = cursor[kind]
            if start + rows > product.shape[0]:
                raise ValueError(
                    f"the request's conditions exceed their {kind}"
                )
            cursor[kind] = start + rows
            return product[start : start + rows]

        taken = []
        for condition in conditions:
            rows = _Rows()
            media = condition["media"]
            if "image" in media:
                size = media["image"]["size"]
                height, width = int(size["height"]), int(size["width"])
                rows.pixels = take("pixels", height * width).view(
                    1, height, width, 3
                )
            elif "video" in media:
                clip = media["video"]["clip"]
                canvas = clip["canvas"]
                area = int(canvas["height"]) * int(canvas["width"])
                rows.pixels = take("pixels", int(clip["vae_frames"]) * area)
            vision = condition.get("vision")
            if vision is not None:
                grid = vision["grid"]
                rows.patches = take(
                    "patches", int(grid["t"]) * int(grid["h"]) * int(grid["w"])
                )
            track = _track(media)
            if track is not None:
                rows.samples = take("samples", int(track["samples"]))
            taken.append(rows)
        for kind, name in names.items():
            product = products.get(name)
            if product is not None and product.shape[0] != cursor[kind]:
                raise ValueError(
                    f"the request's conditions leave {kind} unfilled"
                )
        return taken

    def read(
        self,
        conditions: Sequence[Mapping[str, Any]],
        products: Mapping[str, torch.Tensor | None],
    ) -> None:
        """Decode every condition into its rows of the products.

        Every reference video's FFmpeg decode runs on a thread of its own
        from the start, beside the library's decoders on this thread: each
        image and audio track in request order, then each video's patches
        once its frames arrived.
        """
        rows = self.rows(conditions, products)
        videos: list[tuple[threading.Thread, dict[str, Any]] | None] = []
        for condition in conditions:
            media = condition["media"]
            if "video" not in media:
                videos.append(None)
                continue
            result: dict[str, Any] = {}
            thread = threading.Thread(
                target=self._read_video_into,
                args=(condition["source"], media["video"]["clip"], result),
                daemon=True,
            )
            thread.start()
            videos.append((thread, result))
        try:
            for condition, taken in zip(conditions, rows, strict=True):
                media = condition["media"]
                if "image" in media:
                    fit = media["image"]
                    resized = fit["resized"]
                    size = fit["size"]
                    assert taken.pixels is not None
                    read_image(
                        torch.frombuffer(
                            bytearray(_shared_bytes(condition["source"])),
                            dtype=torch.uint8,
                        ),
                        (int(resized["width"]), int(resized["height"])),
                        (
                            int(fit["left"]),
                            int(fit["top"]),
                            int(size["width"]),
                            int(size["height"]),
                        ),
                        taken.pixels,
                    )
                    self._pack(condition, taken.pixels, taken)
                track = _track(media)
                if track is not None:
                    assert taken.samples is not None
                    read_audio(
                        torch.frombuffer(
                            bytearray(_shared_bytes(condition["source"])),
                            dtype=torch.uint8,
                        ),
                        int(track["sample_rate"]),
                        int(track["start_sample"]),
                        int(track["source_samples"]),
                        int(track["samples"]),
                        self.sample_rate,
                        taken.samples,
                    )
        finally:
            for entry in videos:
                if entry is not None:
                    entry[0].join()
        for condition, taken, entry in zip(
            conditions, rows, videos, strict=True
        ):
            if entry is None:
                continue
            _, result = entry
            if "error" in result:
                raise result["error"]
            frames: torch.Tensor = result["frames"]
            clip = condition["media"]["video"]["clip"]
            # The video encoder encodes the leading kept frames.
            encoded = frames[: int(clip["vae_frames"])].reshape(-1)
            assert taken.pixels is not None
            target = taken.pixels.reshape(-1)
            if target.numel() != encoded.numel():
                raise ValueError(
                    "condition pixels disagree with their product rows"
                )
            target.copy_(encoded)
            self._pack(condition, frames, taken)

    def _pack(
        self, condition: Mapping[str, Any], frames: torch.Tensor, rows: _Rows
    ) -> None:
        """Pack the conditioner's patch rows of a condition it reads."""
        vision = condition.get("vision")
        if vision is None:
            return
        assert rows.patches is not None
        grid = vision["grid"]
        EncoderRunner.pack_pixels(
            self.loaded,
            frames,
            [int(index) for index in vision.get("frame_indices", [])],
            [int(grid["t"]), int(grid["h"]), int(grid["w"])],
            rows.patches,
        )

    def _read_video_into(
        self,
        source: Mapping[str, Any],
        clip: Mapping[str, Any],
        result: dict[str, Any],
    ) -> None:
        try:
            result["frames"] = self._read_video(source, clip)
        except BaseException as error:  # noqa: BLE001 - the reader raises it
            result["error"] = error

    def _read_video(
        self, source: Mapping[str, Any], clip: Mapping[str, Any]
    ) -> torch.Tensor:
        """Decode a reference video's kept frames on its planned canvas with
        one FFmpeg pass: resampled to the model's frame rate, the start
        frames skipped, then scaled per frame with LANCZOS (the scale
        follows the trim, so it never sees a skipped frame). Returns host
        ``[frames, height, width, 3]`` uint8.
        """  # noqa: D205
        canvas = clip["canvas"]
        width, height = int(canvas["width"]), int(canvas["height"])
        frames = int(clip["frames"])
        filters = (
            f"fps={self.frame_rate},trim=start_frame={int(clip['start_frame'])},"
            f"scale={width}:{height}:flags=lanczos,setsar=1"
        )
        name = str(source["name"])
        try:
            # The shared-memory object is a seekable file FFmpeg reads
            # directly.
            input_file = open(f"/dev/shm/{name}", "rb")  # noqa: SIM115
        except OSError as error:
            raise ValueError(
                f"condition media {name} cannot be opened: {error}"
            ) from error
        with input_file:
            process = subprocess.run(  # noqa: S603 - the deployment's FFmpeg
                [
                    self.ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    "fd:",
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    filters,
                    "-frames:v",
                    str(frames),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                stdin=input_file,
                capture_output=True,
                check=False,
            )
        if process.returncode != 0:
            message = process.stderr.decode(errors="replace").strip()
            raise ValueError(f"decoding a reference video failed: {message}")
        frame_bytes = height * width * 3
        if len(process.stdout) != frame_bytes * frames:
            raise ValueError(
                "a reference video decoded to "
                f"{len(process.stdout) / frame_bytes} frames, {frames} were "
                "planned"
            )
        return torch.frombuffer(
            bytearray(process.stdout), dtype=torch.uint8
        ).view(frames, height, width, 3)


def _track(media: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The audio track a condition carries, if any."""
    if "video" in media:
        return media["video"].get("soundtrack")
    return media.get("audio")


def audio_samples(frames: int, frame_rate: int, audio_rate: int) -> int:
    """Samples of the soundtrack at the audio rate: the video's duration,
    ``frames * audio_rate / frame_rate`` rounded half to even.
    """  # noqa: D205
    quotient, remainder = divmod(frames * audio_rate, frame_rate)
    twice = 2 * remainder
    if twice > frame_rate or (twice == frame_rate and quotient % 2 == 1):
        return quotient + 1
    return quotient


class MuxSession:
    """One request's MP4 under assembly.

    The engine hands a request's units, audio and finalization to one muxer
    rank one call at a time, and the host lane runs them in order.
    """

    def __init__(
        self,
        width: int,
        height: int,
        frame_rate: int,
        audio_rate: int,
        unit_frames: Sequence[int],
    ) -> None:
        if width <= 0 or height <= 0 or frame_rate <= 0 or audio_rate <= 0:
            raise ValueError("media mux dimensions and rates must be positive")
        if not unit_frames or 0 in unit_frames:
            raise ValueError("media mux units must cover the output frames")
        self.frame_rate = frame_rate
        self.audio_rate = audio_rate
        self.unit_frames = tuple(unit_frames)
        self.container: codec.Container | None = None
        self.audio: bytes | None = None
        # Decode time, in the video stream's time base, where the next unit
        # starts.
        self.offset = 0
        self.units = 0
        self.closed = False

    @property
    def samples(self) -> int:
        return audio_samples(
            sum(self.unit_frames), self.frame_rate, self.audio_rate
        )

    def append(self, units: Sequence[bytes]) -> None:
        """Append encoded units in request order without decoding their
        frames. Excess units are rejected before any packet is written; a
        failed append closes the session.
        """  # noqa: D205
        if self.closed:
            raise ValueError("artifact assembly is closed")
        if self.units + len(units) > len(self.unit_frames):
            raise ValueError(
                "artifact assembly received more media units than the request"
            )
        try:
            for unit in units:
                if self.container is None:
                    self.container = codec.Container.open(
                        unit, self.frame_rate, self.audio_rate
                    )
                self.offset = self.container.append(unit, self.offset)
                self.units += 1
        except BaseException:
            self.closed = True
            raise

    def set_audio(self, audio: bytes) -> None:
        """Keep the encoded audio track until finalization."""
        if self.closed:
            raise ValueError("artifact assembly is closed")
        if self.audio is not None:
            raise ValueError("artifact assembly received its audio twice")
        self.audio = audio

    def finish(self) -> bytes:
        """Add the audio track and finalize a complete request: every unit
        and the audio must have arrived. Returns the MP4 bytes.
        """  # noqa: D205
        if self.closed:
            raise ValueError("artifact assembly is closed")
        if self.units != len(self.unit_frames):
            raise ValueError(
                "artifact assembly requires every media unit of the request"
            )
        if self.audio is None:
            raise ValueError("artifact assembly has no encoded audio")
        if self.container is None:
            raise AssertionError(
                "a complete artifact assembly has no container"
            )
        container, self.container = self.container, None
        self.closed = True
        data = container.finish(self.audio)
        if not data:
            raise ValueError("media mux produced an empty container")
        return data

    def close(self) -> None:
        """Discard an unfinished container."""
        container, self.container = self.container, None
        if container is not None:
            container.close()


def encode_unit(
    frames: torch.Tensor, frame_rate: int, row: torch.Tensor
) -> int:
    """Encode one media unit's ``[frames, height, width, 3]`` uint8 host
    frames into its encoded row; returns the framed length.
    """  # noqa: D205
    return frame_unit(codec.encode_video_unit(frames, frame_rate), row)


def encode_audio(pcm: torch.Tensor, samples: int, rate: int) -> bytes:
    """Encode a request's ``[samples, 2]`` int16 soundtrack."""
    return codec.encode_audio_track(pcm, samples, rate)


@dataclass
class Staging:
    """A CUDA media reader's way to its device products: page-locked
    staging it decodes into and the stream it copies the staging on.
    """  # noqa: D205

    device: torch.device
    stream: torch.cuda.Stream = field(init=False)

    def __post_init__(self) -> None:
        self.stream = torch.cuda.Stream(device=self.device)

    def host(
        self, products: Mapping[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        """Page-locked staging of each device product's shape."""
        return {
            name: None
            if product is None
            else torch.empty(
                product.shape, dtype=product.dtype, pin_memory=True
            )
            for name, product in products.items()
        }

    def upload(
        self,
        staging: Mapping[str, torch.Tensor | None],
        products: Mapping[str, torch.Tensor | None],
    ) -> None:
        """Copy every staging view into its product and wait for the copies,
        so the products are complete when the task finishes.
        """  # noqa: D205
        with torch.cuda.stream(self.stream):
            for name, product in products.items():
                source = staging[name]
                if product is not None and source is not None:
                    product.copy_(source, non_blocking=True)
        self.stream.synchronize()


def run_reported(
    reports: Reports,
    batch: int,
    call: int,
    row: int,
    work: Callable[[], tuple[int, bytes]],
) -> Callable[[], None]:
    """A host task that runs ``work`` and reports its ``(value, payload)``,
    or its failure, for ``(batch, call, row)``.
    """  # noqa: D205

    def task() -> None:
        try:
            value, payload = work()
        except BaseException as error:  # noqa: BLE001 - reported to the worker
            import logging

            logging.getLogger(__name__).error(
                "host media task of batch %s call %s failed: %s",
                batch,
                call,
                error,
            )
            reports.send(batch, call, row, FAILED)
            return
        reports.send(batch, call, row, DONE, value, payload)

    return task
