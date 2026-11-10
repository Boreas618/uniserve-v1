#!/usr/bin/env python3
"""The media layout operators launched as programmatic dependents read their written inputs.

    python pdl_chain_check.py --op <operator>

Each case launches its operator's Triton source on the device as the operator's module is compiled
for a dependent launch, behind a producer still writing its inputs (`../chain.py`).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chain  # noqa: E402


def adaptive_pool():
    module = chain.source("adaptive_pool")
    rows, heads, width, output, block = 16, 4, 64, 8, 64
    total = rows * output

    def launch(i, o, pdl):
        module._adaptive_pool_kernel[(-(-total // block),)](
            i["x"],
            o["out"],
            rows,
            heads,
            width,
            output,
            int(pdl),
            HEADS=heads,
            WIDTH=width,
            OUTPUT=output,
            WINDOW=-(-width // output) + 1,
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=1,
        )
        return [o["out"]]

    return chain.Case(
        "adaptive_pool",
        {"x": torch.randn(rows, heads, width).to(torch.bfloat16)},
        {"out": torch.zeros(rows, output, dtype=torch.bfloat16)},
        launch,
    )


def cross_fade():
    module = chain.source("cross_fade")
    channels, frames, rows, columns, extent, axis, block = (
        3,
        4,
        16,
        32,
        8,
        2,
        256,
    )
    shape = (channels, frames, rows, columns)
    strides = (frames * rows * columns, rows * columns, columns)

    def launch(i, o, pdl):
        module._cross_fade_kernel[
            (-(-channels * frames * rows * columns // block),)
        ](
            i["previous"],
            i["current"],
            o["out"],
            extent,
            axis,
            channels,
            frames,
            rows,
            columns,
            *strides,
            *strides,
            *strides,
            block,
            int(pdl),
            AXIS=axis,
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"]]

    return chain.Case(
        "cross_fade",
        {
            "previous": torch.randn(shape).to(torch.bfloat16),
            "current": torch.randn(shape).to(torch.bfloat16),
        },
        {"out": torch.zeros(shape, dtype=torch.bfloat16)},
        launch,
    )


def rgb_bytes():
    module = chain.source("rgb_bytes")
    channels, frames, rows, columns, block = 3, 2, 16, 32, 256
    total = frames * rows * columns * channels
    # The raster is `[channels, frames, rows, columns]`, the bytes `[frames, rows, columns, channels]`.
    read = (frames * rows * columns, rows * columns, columns)
    written = (rows * columns * channels, columns * channels, channels)

    def launch(i, o, pdl):
        module._rgb_bytes_kernel[(-(-total // block),)](
            i["raster"],
            i["mean"],
            i["std"],
            o["out"],
            frames,
            rows,
            columns,
            channels,
            *read,
            *written,
            int(pdl),
            CHANNELS=channels,
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=1,
        )
        return [o["out"]]

    return chain.Case(
        "rgb_bytes",
        {
            "raster": torch.rand(channels, frames, rows, columns).to(
                torch.float16
            ),
            "mean": torch.full((channels,), 0.5),
            "std": torch.full((channels,), 0.5),
        },
        {
            "out": torch.zeros(
                frames, rows, columns, channels, dtype=torch.uint8
            )
        },
        launch,
    )


CASES = {case.__name__: case for case in (adaptive_pool, cross_fade, rgb_bytes)}

if __name__ == "__main__":
    sys.exit(chain.main(CASES))
