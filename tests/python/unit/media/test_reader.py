"""Condition images decode onto their planned raster.

A caller holds a condition's encoded bytes as bytes or as a uint8 tensor of
its own storage, and takes the decoded pixels as a returned array or in a
tensor it supplies (``uniserve_models.media.reader``).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from uniserve_models.media.reader import read_image

Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.unit


def _png(width: int, height: int) -> tuple[bytes, np.ndarray]:
    """A PNG of a horizontal gradient and its ``[height, width, 3]`` pixels."""
    row = np.linspace(0, 255, width, dtype=np.uint8)
    pixels = np.repeat(row[None, :, None], height, axis=0).repeat(3, axis=2)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue(), pixels


def test_an_image_on_its_raster_decodes_to_its_kept_window():
    data, pixels = _png(16, 8)
    decoded = read_image(data, (16, 8), (4, 2, 8, 4))
    assert decoded.shape == (1, 4, 8, 3)
    assert np.array_equal(decoded[0], pixels[2:6, 4:12])


def test_tensor_bytes_decode_into_a_supplied_tensor():
    data, pixels = _png(16, 8)
    source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    out = torch.zeros((1, 8, 16, 3), dtype=torch.uint8)
    assert read_image(source, [16, 8], [0, 0, 16, 8], out) is None
    assert np.array_equal(out[0].numpy(), pixels)

    # A tensor planned for another raster is refused.
    with pytest.raises(ValueError):
        read_image(data, (16, 8), (0, 0, 16, 8), torch.zeros((1, 8, 8, 3)))
