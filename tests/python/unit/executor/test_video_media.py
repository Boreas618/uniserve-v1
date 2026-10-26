"""A video executor's host media (``uniserve.executor.video.media``).

Encoded unit rows, the MP4 under assembly, condition rows and the reports
host tasks send the worker.
"""

from __future__ import annotations

import os
import struct
import sys

import pytest
import torch

from uniserve.executor.video import media
from uniserve.executor.video.media import (
    Conditions,
    MuxSession,
    Reports,
    audio_samples,
    encoded_unit_bytes,
    encoded_video_bytes,
    frame_unit,
    read_unit,
)

pytestmark = pytest.mark.unit


def test_units_are_framed_with_their_length_and_read_back():
    row = torch.full((20,), 0xAA, dtype=torch.uint8)
    assert frame_unit(b"payload", row) == 15
    assert read_unit(row.numpy().tobytes()) == b"payload"
    # Bytes past the framed prefix are untouched.
    assert row[15:].tolist() == [0xAA] * 5
    with pytest.raises(
        ValueError,
        match="encoded media unit of 13 bytes exceeds the 12 bytes reserved",
    ):
        frame_unit(bytes(13), row)
    with pytest.raises(ValueError):
        read_unit(row.numpy().tobytes()[:4])
    short = bytearray(row.numpy().tobytes())
    short[:8] = (100).to_bytes(8, sys.byteorder)
    with pytest.raises(ValueError):
        read_unit(bytes(short))


def test_the_encoded_bound_covers_every_macroblock_and_slice_header():
    # 1344x768 is 84x48 macroblocks; one frame adds its macroblocks' escaped
    # residual bound, eight slice headers and a sample entry.
    one = encoded_video_bytes(1, 768, 1344)
    assert one - (1 << 16) == 84 * 48 * 3519 + 8 * 1024 + 64
    assert encoded_video_bytes(22, 768, 1344) - (1 << 16) == 22 * (
        one - (1 << 16)
    )
    # Partial macroblocks are padded.
    assert encoded_video_bytes(1, 17, 17) == encoded_video_bytes(1, 32, 32)
    assert encoded_unit_bytes(1, 16, 16) == 8 + encoded_video_bytes(1, 16, 16)


def test_the_soundtrack_spans_the_video_duration_rounded_half_to_even():
    # 124 frames at 24 fps and 32 kHz.
    session = MuxSession(64, 64, 24, 32_000, [17, 17, 17, 17, 17, 17, 22])
    assert session.samples == 165_333
    # 1 * 36 / 24 = 1.5 rounds to 2; 1 * 60 / 24 = 2.5 rounds to 2.
    assert audio_samples(1, 24, 36) == 2
    assert audio_samples(1, 24, 60) == 2
    with pytest.raises(ValueError):
        MuxSession(64, 64, 24, 32_000, [])


class _Recorder:
    """A container recording what the session hands it.

    A unit ends at its offset plus its length.
    """

    log: list[str] = []

    @classmethod
    def open(cls, first_unit: bytes, frame_rate: int, audio_rate: int):
        cls.log.append("open")
        return cls()

    def append(self, unit: bytes, offset: int) -> int:
        self.log.append(f"append {len(unit)} at {offset}")
        return offset + len(unit)

    def finish(self, audio: bytes) -> bytes:
        self.log.append(f"finish {len(audio)}")
        return b"mp4 bytes"

    def close(self) -> None:
        self.log.append("close")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    monkeypatch.setattr(_Recorder, "log", log)
    monkeypatch.setattr(media.codec, "Container", _Recorder)
    return log


def test_a_session_offsets_units_in_order_and_finishes_once_complete(
    recorder: list[str],
):
    session = MuxSession(64, 64, 24, 32_000, [17, 17, 22])
    session.append([b"aaaa", b"bb"])
    # Excess units are refused before any is written.
    with pytest.raises(ValueError):
        session.append([b"c", b"d"])
    with pytest.raises(ValueError):
        session.finish()
    session.set_audio(b"aac")
    with pytest.raises(ValueError):
        session.finish()
    session.append([b"ccc"])
    with pytest.raises(ValueError):
        session.set_audio(b"x")
    assert session.finish() == b"mp4 bytes"
    with pytest.raises(ValueError):
        session.finish()
    session.close()
    assert recorder == [
        "open",
        "append 4 at 0",
        "append 2 at 4",
        "append 3 at 6",
        "finish 3",
    ]


def test_an_unfinished_session_closes_its_container(recorder: list[str]):
    session = MuxSession(64, 64, 24, 32_000, [17, 22])
    session.append([b"a"])
    session.close()
    assert recorder == ["open", "append 1 at 0", "close"]


def _condition(media_json: dict, patches: int | None) -> dict:
    return {
        "role": "reference",
        "source": {"name": "uniserve-media-0", "bytes": 1},
        "media": media_json,
        "vision": None
        if patches is None
        else {"grid": {"t": 1, "h": 1, "w": patches}, "tokens": patches},
    }


def test_conditions_take_their_product_rows_in_request_order():
    # Each condition's rows of each product follow the conditions before it
    # in request order and together fill every product; a product left
    # short is refused.
    canvas = {"width": 2, "height": 2}

    def track(samples: int) -> dict:
        return {
            "sample_rate": 32_000,
            "start_sample": 0,
            "source_samples": samples,
            "samples": samples,
        }

    conditions = [
        _condition(
            {
                "video": {
                    "clip": {
                        "canvas": canvas,
                        "start_frame": 0,
                        "frames": 7,
                        "vae_frames": 5,
                    },
                    "soundtrack": track(10),
                }
            },
            6,
        ),
        _condition(
            {
                "image": {
                    "resized": canvas,
                    "left": 0,
                    "top": 0,
                    "size": canvas,
                }
            },
            2,
        ),
        _condition({"audio": track(3)}, None),
    ]
    products = {
        "condition_pixels": torch.zeros(5 * 4 + 4, 3, dtype=torch.uint8),
        "condition_samples": torch.zeros(13, 2),
        "vision_pixels": torch.zeros(8, 4),
    }
    rows = Conditions.rows(conditions, products)

    def start(view: torch.Tensor | None, name: str) -> int | None:
        if view is None:
            return None
        product = products[name]
        offset = view.data_ptr() - product.data_ptr()
        return offset // (product.stride(0) * product.element_size())

    def at(index: int) -> tuple[int | None, ...]:
        return (
            start(rows[index].pixels, "condition_pixels"),
            start(rows[index].samples, "condition_samples"),
            start(rows[index].patches, "vision_pixels"),
        )

    assert at(0) == (0, 0, 0)
    assert at(1) == (20, None, 6)
    assert at(2) == (None, 10, None)
    with pytest.raises(ValueError):
        Conditions.rows(
            conditions, {**products, "condition_samples": torch.zeros(14, 2)}
        )


def test_reports_reach_the_worker_whole_in_the_order_sent():
    # The worker reads each report as six little-endian i64 (batch, call,
    # row, status, value, payload bytes) and its payload.
    read, write = os.pipe()
    try:
        reports = Reports(write)
        reports.send(7, 2, 0, media.DONE, 15, b"mp4 bytes")
        reports.send(7, 3, 1, media.FAILED)
        header = struct.Struct("<qqqqqq")
        data = b""
        expected = 2 * header.size + len(b"mp4 bytes")
        while len(data) < expected:
            data += os.read(read, expected - len(data))
    finally:
        os.close(read)
        os.close(write)
    first = header.unpack_from(data, 0)
    assert first == (7, 2, 0, media.DONE, 15, 9)
    payload = data[header.size : header.size + 9]
    assert payload == b"mp4 bytes"
    second = header.unpack_from(data, header.size + 9)
    assert second == (7, 3, 1, media.FAILED, 0, 0)
