"""The bounded outputs a rank reads column by column."""

import pytest

from uniserve.runner.denoiser import Outputs

pytestmark = pytest.mark.unit


def test_columns_describe_each_output_in_declared_order():
    outputs = Outputs.of(
        {
            "denoiser": [
                {
                    "name": "video_latents",
                    "dtype": "bfloat16",
                    "shape": [4096, 64],
                    "variable_axes": [0],
                    "raster_axes": None,
                }
            ],
            "decoder": [
                {
                    "name": "video_units",
                    "dtype": "uint8",
                    "shape": [7, 24, 1152, 2048, 3],
                    "variable_axes": [0],
                    "raster_axes": [2, 3],
                },
                {
                    "name": "frame_count",
                    "dtype": "int32",
                    "shape": [1],
                    "variable_axes": [],
                    "raster_axes": None,
                },
            ],
        }
    )

    assert outputs.owners() == ["denoiser", "decoder", "decoder"]
    assert outputs.names() == ["video_latents", "video_units", "frame_count"]
    assert outputs.dtypes() == ["bfloat16", "uint8", "int32"]
    # Each output's extents follow one another in `extents`.
    assert outputs.ranks() == [2, 5, 1]
    assert outputs.extents() == [4096, 64, 7, 24, 1152, 2048, 3, 1]
    # -1 marks a fixed shape and an output without a raster.
    assert outputs.variable_axes() == [0, 0, -1]
    assert outputs.raster_axes() == [-1, -1, 2, 3, -1, -1]


def test_an_output_varying_along_two_axes_is_refused():
    with pytest.raises(ValueError, match="more than one axis"):
        Outputs.of(
            {
                "decoder": [
                    {
                        "name": "video_units",
                        "dtype": "uint8",
                        "shape": [7, 24],
                        "variable_axes": [0, 1],
                        "raster_axes": None,
                    }
                ]
            }
        )
