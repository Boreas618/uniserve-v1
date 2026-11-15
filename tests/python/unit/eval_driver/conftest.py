"""Fixtures shared by the evaluator driver tests."""

from __future__ import annotations

import io
import os
import select
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from fractions import Fraction
from importlib.resources import as_file, files
from pathlib import Path

import av
import numpy as np
import pytest

from uniserve_eval.cli import main


@pytest.fixture(scope="session")
def shipped_profile() -> Iterator[Callable[[str], Path]]:
    """Locate a profile the evaluator ships, by its file name.

    The profiles are package data of ``uniserve_eval``, read through
    ``importlib.resources``. The returned function gives each as a file for
    the rest of the session, since a profile states its ``root`` relative to
    its own file.
    """
    with ExitStack() as on_disk:

        def locate(name: str) -> Path:
            resource = files("uniserve_eval") / name
            return on_disk.enter_context(as_file(resource))

        yield locate


@pytest.fixture(scope="session")
def evaluate() -> Callable[[Sequence[str]], int]:
    """Run one ``uniserve-eval`` command line in this process.

    The returned function takes the arguments after the program name and
    returns the command's exit status, as the console script would exit with
    it: 0 when the command returns, else the status of its ``SystemExit``.
    """

    def run(argv: Sequence[str]) -> int:
        try:
            main(argv)
        except SystemExit as exited:
            return int(exited.code or 0)
        return 0

    return run


@pytest.fixture(scope="session")
def wait_exit() -> Callable[[int, float], bool]:
    """Wait for a process to exit, by its process descriptor.

    The returned function takes a process id and a timeout in seconds and
    returns whether the process has exited by then. Any process can be
    waited on, not only a child of the test: its descriptor
    (``pidfd_open``) becomes readable when it exits, and a process that no
    longer exists at all has exited.
    """

    def wait(pid: int, timeout_s: float) -> bool:
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            return True
        try:
            readable, _, _ = select.select([descriptor], [], [], timeout_s)
        finally:
            os.close(descriptor)
        return bool(readable)

    return wait


@pytest.fixture(scope="session")
def media():
    """Encoded fixture with the real H3 shape, audio and frame count."""
    result = io.BytesIO()
    frames = 107  # 17 * 6 + 5, a frame count H3 generates.
    with av.open(result, "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = 1344, 768, "yuv420p"
        video.options = {"preset": "ultrafast", "crf": "35"}
        audio = container.add_stream("aac", rate=32000)
        audio.layout = "stereo"
        pixels = np.zeros((768, 1344, 3), dtype=np.uint8)
        pixels[:, :672] = 180
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = round(frames / 24 * 32000)
        signal = (
            0.1 * np.sin(np.arange(samples) * 2 * np.pi * 440 / 32000)
        ).astype("float32")
        frame = av.AudioFrame.from_ndarray(
            np.stack([signal, signal]), format="fltp", layout="stereo"
        )
        frame.sample_rate = 32000
        frame.pts = 0
        frame.time_base = Fraction(1, 32000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return result.getvalue()
