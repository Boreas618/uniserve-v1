"""Generates the condition-reading fixtures from the reference implementations.

This is a code generator, not a test: it writes small condition media (PNG,
JPEG with an EXIF orientation, WAV tracks, one tiny H.264 reference video) and
the reference outputs the readers must produce, computed by the `.venv`
implementations the readers replace (`uniserve/media/reader.py` and
`Conditions.read`) and by the environment's FFmpeg for the reference video.

The reference versions are Pillow 12.3.0, PyAV with libavcodec 8.1.2, torch
2.13.0, torchvision 0.28.0 and torchaudio 2.11.0 (the read-only `.venv` this
port replaces), and the provisioned FFmpeg 8.1.2 with libx264 for the reference
video and its expected frames.

Usage: `.venv/bin/python crates/media/tests/fixtures/read/generate.py`

Manifests (one case per line, one `<name>.out` per case beside the media):

- `images/cases.txt`: `name file resized_w resized_h left top width height tolerance`
  (the expected bytes are RGB24 rows; `tolerance` is the per-channel bound of a
  lossy decoder, zero for a lossless input).
- `audio/cases.txt`: `name file native_rate start source_samples samples rate`
  (the expected bytes are `[samples, 2]` float32, sample-major).
- `video/cases.txt`: `name file frame_rate canvas_w canvas_h frames start_frame`
  (the expected bytes are `[frames, canvas_h, canvas_w, 3]` RGB24).
"""

from __future__ import annotations

import os
import subprocess
import wave

import numpy as np
import torch
import torchaudio
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(ROOT, "..", "..", "..", "..", ".."))
def _ffmpeg() -> str:
    """The provisioned 8.1.2 build, or the image's own 8.1.2 build."""
    named = os.environ.get("UNISERVE_FFMPEG")
    if named:
        return named
    directory = os.environ.get("UNISERVE_FFMPEG_DIR")
    candidates = []
    if directory:
        candidates.append(os.path.join(directory, "bin", "ffmpeg"))
    candidates.append(
        os.path.join(WORKSPACE, "artifacts", "native", "prefix", "bin", "ffmpeg")
    )
    candidates.append("/opt/uv/ffmpeg-8.1.2/bin/ffmpeg")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return "ffmpeg"


FFMPEG = _ffmpeg()
# The provisioned FFmpeg links the shared runtime beside it.
ENVIRONMENT = dict(os.environ)
_LIBRARY = os.path.join(WORKSPACE, "artifacts", "native", "prefix", "lib")
_PREVIOUS = ENVIRONMENT.get("LD_LIBRARY_PATH")
ENVIRONMENT["LD_LIBRARY_PATH"] = _LIBRARY + (":" + _PREVIOUS if _PREVIOUS else "")


def pattern(channels: int, height: int, width: int) -> np.ndarray:
    pixels = np.zeros((height, width, channels), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                pixels[y, x, c] = (x * 9 + y * 17 + c * 41 + (x * y) % 7 * 13) % 256
    return pixels


def clean() -> None:
    for kind in ("images", "audio", "video"):
        directory = os.path.join(ROOT, kind)
        os.makedirs(directory, exist_ok=True)
        for entry in os.listdir(directory):
            os.remove(os.path.join(directory, entry))


def write(kind: str, name: str, case: str, expected: bytes) -> None:
    directory = os.path.join(ROOT, kind)
    with open(os.path.join(directory, f"{name}.out"), "wb") as file:
        file.write(expected)
    with open(os.path.join(directory, "cases.txt"), "a", encoding="ascii") as file:
        file.write(case + "\n")


def reference_image(path: str, resized: tuple[int, int], window: tuple[int, int, int, int]) -> bytes:
    """`uniserve.media.reader.read_image`: decode, orient, LANCZOS, crop."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if image.size != resized:
        image = image.resize(resized, Image.Resampling.LANCZOS)
    left, top, width, height = window
    if (left, top, width, height) != (0, 0, *resized):
        image = image.crop((left, top, left + width, top + height))
    return np.array(image).tobytes()


def image_cases() -> None:
    directory = os.path.join(ROOT, "images")
    cases = [
        # (name, format, source size, resized, window, lossy)
        ("pattern", "PNG", (7, 5), (13, 11), (0, 0, 13, 11), False),
        ("crop", "PNG", (9, 7), (12, 4), (1, 1, 9, 2), False),
        ("same", "PNG", (6, 4), (6, 4), (1, 1, 4, 2), False),
        ("rotated", "JPEG", (9, 7), (7, 9), (0, 0, 7, 9), True),
        ("upside", "JPEG", (9, 7), (9, 7), (1, 1, 7, 5), True),
        ("mirrored", "JPEG", (9, 7), (4, 6), (0, 0, 4, 6), True),
    ]
    orientation = {"rotated": 6, "upside": 3, "mirrored": 2}
    for name, form, (width, height), resized, window, lossy in cases:
        pixels = pattern(3, height, width)
        image = Image.frombytes("RGB", (width, height), pixels.tobytes())
        path = os.path.join(directory, f"{name}.{form.lower()}")
        if form == "JPEG":
            exif = Image.Exif()
            exif[274] = orientation[name]
            image.save(path, quality=95, exif=exif)
        else:
            image.save(path)
        expected = reference_image(path, resized, window)
        tolerance = 8 if lossy else 0
        write(
            "images",
            name,
            f"{name} {os.path.basename(path)} {resized[0]} {resized[1]} "
            f"{window[0]} {window[1]} {window[2]} {window[3]} {tolerance}",
            expected,
        )


def reference_audio(
    path: str, native: int, start: int, source_samples: int, samples: int, rate: int
) -> bytes:
    """`uniserve.media.reader.read_audio`, call for call."""
    import av

    container = av.open(path, mode="r")
    with container:
        stream = container.streams.audio[0]
        decoded_rate = int(stream.codec_context.sample_rate)
        assert decoded_rate == native, (decoded_rate, native)
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout=stream.layout, rate=native
        )
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(
                torch.from_numpy(part.to_ndarray()) for part in resampler.resample(frame)
            )
        chunks.extend(
            torch.from_numpy(part.to_ndarray()) for part in resampler.resample(None)
        )
    waveform = torch.cat(chunks, dim=-1).to(torch.float32)
    waveform = waveform[:, start : start + source_samples]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if native != rate:
        waveform = torchaudio.transforms.Resample(native, rate)(waveform)
    assert waveform.shape[-1] == samples, (waveform.shape[-1], samples)
    return waveform.transpose(0, 1).contiguous().numpy().tobytes()


def write_wav(path: str, rate: int, channels: int, samples: int) -> None:
    times = np.arange(samples, dtype=np.float64) / rate
    signal = np.zeros((samples, channels), dtype=np.int16)
    for channel in range(channels):
        frequency = 440.0 + 220.0 * channel
        values = np.sin(2.0 * np.pi * frequency * times)
        values += 0.5 * np.sin(2.0 * np.pi * (frequency * 2.5) * times)
        signal[:, channel] = np.round(values * 0.4 * 32767).astype(np.int16)
    with wave.open(path, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(signal.tobytes())


def audio_cases() -> None:
    directory = os.path.join(ROOT, "audio")
    cases = [
        # (name, rate, channels, length, native, start, source_samples, rate_out)
        ("stereo", 44100, 2, 1000, 44100, 100, 800, 16000),
        ("mono", 22050, 1, 500, 22050, 0, 500, 44100),
        ("noop", 48000, 2, 400, 48000, 50, 300, 48000),
    ]
    for name, rate, channels, length, native, start, source_samples, rate_out in cases:
        path = os.path.join(directory, f"{name}.wav")
        write_wav(path, rate, channels, length)
        samples = int(np.ceil(rate_out * source_samples / native))
        expected = reference_audio(path, native, start, source_samples, samples, rate_out)
        write(
            "audio",
            name,
            f"{name} {os.path.basename(path)} {native} {start} {source_samples} "
            f"{samples} {rate_out}",
            expected,
        )


def video_cases() -> None:
    directory = os.path.join(ROOT, "video")
    path = os.path.join(directory, "clip.mp4")
    # The provisioned FFmpeg carries libx264; the environment's static build
    # has no H.264 encoder.
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x24:rate=10:duration=1.5",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            path,
        ],
        check=True,
        env=ENVIRONMENT,
    )
    # A reference video that carries a soundtrack: its audio is decoded from
    # the same file the frames come from.
    soundtrack = os.path.join(directory, "soundtrack.mp4")
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x24:rate=10:duration=1.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1.5",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            "-c:a",
            "aac",
            "-b:a",
            "144k",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-shortest",
            soundtrack,
        ],
        check=True,
        env=ENVIRONMENT,
    )
    # Five frames at the model's 10 fps hold half a second, 22050 native
    # samples, which resample to 8000 at the model's 16 kHz.
    audio_plan = (44100, 0, 22050, 8000, 16000)
    audio = reference_audio(soundtrack, *audio_plan)
    with open(os.path.join(directory, "soundtrack.audio"), "wb") as file:
        file.write(audio)
    with open(os.path.join(directory, "soundtrack.audio.txt"), "w", encoding="ascii") as file:
        file.write(" ".join(str(value) for value in audio_plan) + "\n")

    cases = [
        # (name, source, frame_rate, canvas, frames, start_frame)
        ("scaled", path, 8, (16, 12), 3, 2),
        ("native", path, 10, (32, 24), 5, 0),
        ("soundtrack", soundtrack, 10, (32, 24), 5, 0),
    ]
    for name, source, frame_rate, (width, height), frames, start_frame in cases:
        filters = (
            f"fps={frame_rate},trim=start_frame={start_frame},"
            f"scale={width}:{height}:flags=lanczos,setsar=1"
        )
        expected = subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-i",
                source,
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
            check=True,
            capture_output=True,
            env=ENVIRONMENT,
        ).stdout
        assert len(expected) == frames * width * height * 3, len(expected)
        write(
            "video",
            name,
            f"{name} {os.path.basename(source)} {frame_rate} {width} {height} {frames} {start_frame}",
            expected,
        )


def main() -> None:
    clean()
    image_cases()
    audio_cases()
    video_cases()
    total = 0
    for kind in ("images", "audio", "video"):
        directory = os.path.join(ROOT, kind)
        total += sum(
            os.path.getsize(os.path.join(directory, entry))
            for entry in os.listdir(directory)
        )
    print(f"wrote reading fixtures under {ROOT} ({total} bytes)")


if __name__ == "__main__":
    main()
