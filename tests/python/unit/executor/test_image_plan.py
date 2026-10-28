"""Each image call's plan names what it reads and writes."""

import pytest

from uniserve.executor.image.plan import (
    NOTHING,
    Field,
    PlanEntry,
    decoding_entry,
    latent_entry,
    range_name,
    vision_entry,
)

pytestmark = pytest.mark.unit


def test_a_vision_encoding_writes_one_feature_per_merged_square():
    entry = vision_entry(
        grid=(4, 6),
        downsample=2,
        output_size=32,
        height=56,
        width=84,
        resident=False,
    )
    assert entry.to_json() == {
        "reads": [],
        "writes": [],
        "publication": None,
        "image": {
            "read": False,
            "pixels": True,
            "writes": [
                {
                    "field": "encoder_output",
                    "shape": [6, 32],
                    "metadata": {"feature": {"height": 56, "width": 84}},
                }
            ],
            "bytes": 0,
            "latent": False,
        },
    }
    # Its prepared pixels, then its features.
    assert entry.tensors == 2
    # A resident image is read from its product instead.
    resident = vision_entry(
        grid=(4, 6),
        downsample=2,
        output_size=32,
        height=56,
        width=84,
        resident=True,
    )
    assert (resident.read, resident.pixels) == (True, False)
    with pytest.raises(ValueError, match="downsampling"):
        vision_entry(
            grid=(3, 6),
            downsample=2,
            output_size=32,
            height=42,
            width=84,
            resident=False,
        )


def test_a_latent_encoding_writes_the_images_latent_rows():
    entry = latent_entry(
        rows=6, width=64, height=32, canvas_width=48, resident=True
    )
    (write,) = entry.writes
    assert (write.field, write.shape) == (Field.ENCODER_OUTPUT, (6, 64))
    assert write.metadata == {"feature": {"height": 32, "width": 48}}
    assert entry.to_json()["image"]["read"] is True


def test_an_image_decoding_stages_its_bytes_and_reads_its_latent():
    entry = decoding_entry(
        height=32,
        width=48,
        value_range="signed_unit",
        publishes=True,
        trajectory=True,
    )
    planned = entry.to_json()["image"]
    # The RGB bytes, then the published product's BF16 values.
    assert planned["bytes"] == 3 * 32 * 48 + 2 * 3 * 32 * 48
    assert planned["latent"] is True
    assert planned["writes"] == [
        {
            "field": "image_output",
            "shape": [3, 32, 48],
            "metadata": {
                "image": {
                    "height": 32,
                    "width": 48,
                    "value_range": "signed_unit",
                }
            },
        }
    ]
    # Its pixel product, then its staged bytes.
    assert entry.tensors == 2
    # Without a pixel product only the RGB bytes leave the call.
    unpublished = decoding_entry(
        height=32,
        width=48,
        value_range="unit",
        publishes=False,
        trajectory=True,
    )
    assert (unpublished.writes, unpublished.staging) == ((), 3 * 32 * 48)
    # A resident image is read and quantized, not decoded.
    resident = decoding_entry(
        height=32,
        width=48,
        value_range="unit",
        publishes=True,
        trajectory=False,
    )
    assert (resident.read, resident.latent, resident.writes) == (
        True,
        False,
        (),
    )


def test_entries_round_trip_through_the_preparation():
    entry = PlanEntry(latent=True)
    assert PlanEntry.from_json(entry.to_json()) == entry
    assert PlanEntry.from_json(dict(NOTHING)) is None
    assert entry.tensors == 0


def test_pixel_ranges_name_their_product_value_range():
    assert range_name((0.0, 1.0)) == "unit"
    assert range_name((-1.0, 1.0)) == "signed_unit"
    assert range_name((0.0, 255.0)) is None
