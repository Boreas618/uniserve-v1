"""Generates the resampler reference fixtures from the `.venv` implementations.

This is a code generator, not a test: it runs the reference implementations
(Pillow 12.3, torchaudio 2.11, torchvision 0.28 / torch 2.13) once on small
fixed inputs and writes their outputs beside the inputs, for the Rust tests to
compare against.

Usage: `.venv/bin/python crates/media/tests/fixtures/resample/generate.py`

Each kind has a directory with a `cases.txt` (one case per line) and, per case,
`<name>.in` and `<name>.out` raw files:

- `lanczos/cases.txt`:   `name channels in_h in_w out_h out_w`, RGB24 rows.
- `bicubic/cases.txt`:   `name channels in_h in_w out_h out_w`, planar CHW u8.
- `bicubic_f32/cases.txt`: the same extents, planar CHW float32.
- `sinc/cases.txt`:      `name channels orig_freq new_freq length`, planar f32.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image
from torchvision.transforms import v2 as vision

ROOT = os.path.dirname(os.path.abspath(__file__))


def pattern(channels: int, height: int, width: int) -> np.ndarray:
    """A deterministic, varied image: every pixel and channel differs."""
    pixels = np.zeros((channels, height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                pixels[c, y, x] = (x * 7 + y * 13 + c * 29 + (x * y) % 5 * 11) % 256
    return pixels


def write_case(kind: str, name: str, case: str, source: bytes, expected: bytes) -> None:
    directory = os.path.join(ROOT, kind)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{name}.in"), "wb") as file:
        file.write(source)
    with open(os.path.join(directory, f"{name}.out"), "wb") as file:
        file.write(expected)
    with open(os.path.join(directory, "cases.txt"), "a", encoding="ascii") as file:
        file.write(case + "\n")


def clean() -> None:
    for kind in ("lanczos", "bicubic", "bicubic_f32", "sinc"):
        directory = os.path.join(ROOT, kind)
        os.makedirs(directory, exist_ok=True)
        for entry in os.listdir(directory):
            os.remove(os.path.join(directory, entry))


def lanczos_cases() -> None:
    """Pillow's `Image.resize(size, Image.Resampling.LANCZOS)` on RGB images."""
    cases = {
        # upscale on both axes
        "upscale": (3, 5, 7, 11, 13),
        # downscale on both axes
        "downscale": (3, 16, 9, 5, 3),
        # one axis up, one down
        "aspect": (3, 9, 7, 12, 4),
        # a strong downscale, where the filter widens
        "wide": (3, 24, 17, 6, 4),
    }
    for name, (channels, height, width, out_height, out_width) in cases.items():
        pixels = pattern(channels, height, width)
        image = Image.frombytes("RGB", (width, height), pixels.transpose(1, 2, 0).tobytes())
        expected = image.resize((out_width, out_height), Image.Resampling.LANCZOS)
        write_case(
            "lanczos",
            name,
            f"{name} {channels} {height} {width} {out_height} {out_width}",
            bytes(pixels.tobytes()),
            expected.tobytes(),
        )


def bicubic_cases() -> None:
    """torchvision's antialiased bicubic resize of a uint8 CHW tensor."""
    cases = {
        "upscale": (3, 5, 7, 11, 13),
        "downscale": (3, 16, 9, 5, 3),
        "aspect": (3, 9, 7, 12, 4),
        "wide": (3, 24, 17, 6, 4),
        "single": (1, 5, 7, 11, 13),
    }
    for name, (channels, height, width, out_height, out_width) in cases.items():
        pixels = pattern(channels, height, width)
        source = torch.from_numpy(pixels.copy())
        expected = vision.functional.resize(
            source,
            [out_height, out_width],
            interpolation=vision.InterpolationMode.BICUBIC,
            antialias=True,
        )
        write_case(
            "bicubic",
            name,
            f"{name} {channels} {height} {width} {out_height} {out_width}",
            pixels.tobytes(),
            expected.contiguous().numpy().tobytes(),
        )


def bicubic_f32_cases() -> None:
    """torch's antialiased bicubic resize of a float32 CHW tensor."""
    cases = {
        "upscale": (3, 5, 7, 11, 13),
        "downscale": (3, 16, 9, 5, 3),
        "aspect": (3, 9, 7, 12, 4),
        "wide": (3, 24, 17, 6, 4),
        "single": (1, 5, 7, 11, 13),
    }
    for name, (channels, height, width, out_height, out_width) in cases.items():
        pixels = pattern(channels, height, width).astype(np.float32) / 255.0
        source = torch.from_numpy(pixels.copy())
        expected = torch.nn.functional.interpolate(
            source[None],
            size=[out_height, out_width],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0]
        write_case(
            "bicubic_f32",
            name,
            f"{name} {channels} {height} {width} {out_height} {out_width}",
            pixels.tobytes(),
            expected.contiguous().numpy().tobytes(),
        )


def sinc_cases() -> None:
    """torchaudio's bandlimited sinc resample of planar float32 audio."""
    cases = {
        # mono, speech-rate down to the audio model's rate
        "mono_down": (1, 44100, 16000, 100),
        # stereo, upsampling
        "stereo_up": (2, 16000, 44100, 100),
        # an odd length at an unchanged rate
        "same": (2, 8000, 8000, 97),
        # a rational, non-integer ratio
        "ratio": (2, 24000, 16000, 50),
    }
    for name, (channels, orig, new, length) in cases.items():
        plan = []
        for channel in range(channels):
            frequency = 440.0 + 310.0 * channel
            times = np.arange(length, dtype=np.float64) / orig
            signal = np.sin(2.0 * np.pi * frequency * times)
            signal += 0.5 * np.sin(2.0 * np.pi * (frequency * 3.0) * times)
            plan.append(signal.astype(np.float32))
        source = torch.from_numpy(np.stack(plan))
        expected = torchaudio.transforms.Resample(orig, new)(source)
        write_case(
            "sinc",
            name,
            f"{name} {channels} {orig} {new} {length}",
            np.ascontiguousarray(source.numpy()).tobytes(),
            np.ascontiguousarray(expected.numpy()).tobytes(),
        )


def main() -> None:
    clean()
    lanczos_cases()
    bicubic_cases()
    bicubic_f32_cases()
    sinc_cases()
    total = 0
    for kind in ("lanczos", "bicubic", "bicubic_f32", "sinc"):
        directory = os.path.join(ROOT, kind)
        total += sum(
            os.path.getsize(os.path.join(directory, entry))
            for entry in os.listdir(directory)
        )
    print(f"wrote reference fixtures under {ROOT} ({total} bytes)")


if __name__ == "__main__":
    main()
