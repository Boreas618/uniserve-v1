#!/usr/bin/env python3
"""The position operators launched as programmatic dependents read their written inputs.

    python pdl_chain_check.py --op <operator>

Each case launches its operator's Triton source on the device as the operator's module is compiled
for a dependent launch, behind a producer still writing its inputs (`../chain.py`).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chain  # noqa: E402


def position_interpolate():
    module = chain.source("position_interpolate")
    side, points, channels, block = 16, 64, 64, 64

    def launch(i, o, pdl):
        module._position_interpolate_kernel[(points, channels // block, 1)](
            i["table"],
            i["indices"],
            i["weights"],
            o["out"],
            points,
            channels,
            block,
            int(pdl),
            CHANNELS=channels,
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=2,
        )
        return [o["out"]]

    return chain.Case(
        "position_interpolate",
        {
            "table": torch.randn(side * side, channels).to(torch.bfloat16),
            "indices": torch.randint(
                0, side * side, (points, 4), dtype=torch.int32
            ),
            "weights": torch.rand(points, 4),
        },
        {"out": torch.zeros(points, channels, dtype=torch.bfloat16)},
        launch,
    )


CASES = {case.__name__: case for case in (position_interpolate,)}

if __name__ == "__main__":
    sys.exit(chain.main(CASES))
