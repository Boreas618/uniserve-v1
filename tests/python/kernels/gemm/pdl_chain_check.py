#!/usr/bin/env python3
"""The GEMM operand operators launched as programmatic dependents read their written inputs.

    python pdl_chain_check.py --op <operator>

Each case launches its operator's Triton source on the device as the operator's module is compiled
for a dependent launch, behind a producer still writing its inputs (`../chain.py`).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chain  # noqa: E402


def nvfp4_quantize():
    module = chain.source("nvfp4_quantize")
    rows, columns, block = 16, 2048, 64
    scale_columns = columns // 16

    def launch(i, o, pdl):
        # The linear scale layout: one E4M3 scale per sixteen values, rows unpadded.
        module._nvfp4_quantize_kernel[(rows, 1, 1)](
            i["input"],
            o["values"],
            o["scales"],
            rows,
            i["global_scale"],
            o["token_scale"],
            COLUMNS=columns,
            BLOCK=block,
            SCALE_COLUMNS=scale_columns,
            SWIZZLED=False,
            PADDED_COLUMNS=scale_columns,
            PADDED_TILE=scale_columns,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=2,
        )
        return [o["values"], o["scales"], o["token_scale"]]

    return chain.Case(
        "nvfp4_quantize",
        {
            "input": (torch.randn(rows, columns) * 0.05).to(torch.bfloat16),
            "global_scale": torch.tensor([1.0]),
        },
        {
            "values": torch.zeros(rows, columns // 2, dtype=torch.uint8),
            "scales": torch.zeros(rows, scale_columns, dtype=torch.uint8),
            "token_scale": torch.zeros(rows),
        },
        launch,
    )


CASES = {case.__name__: case for case in (nvfp4_quantize,)}

if __name__ == "__main__":
    sys.exit(chain.main(CASES))
