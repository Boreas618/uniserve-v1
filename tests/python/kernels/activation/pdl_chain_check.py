#!/usr/bin/env python3
"""The activation operators launched as programmatic dependents read their written inputs.

    python pdl_chain_check.py --op <operator>

Each case launches its operator's Triton source on the device as the operator's module is compiled
for a dependent launch, behind a producer still writing its inputs (`../chain.py`).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chain  # noqa: E402


def silu():
    module = chain.source("silu")
    rows, columns, block = 8, 1024, 1024

    def launch(i, o, pdl):
        module._silu_kernel[(rows * columns // block,)](
            i["x"],
            o["out"],
            rows * columns,
            columns,
            int(pdl),
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"]]

    return chain.Case(
        "silu",
        {"x": torch.randn(rows, columns).to(torch.bfloat16)},
        {"out": torch.zeros(rows, columns, dtype=torch.bfloat16)},
        launch,
    )


CASES = {case.__name__: case for case in (silu,)}

if __name__ == "__main__":
    sys.exit(chain.main(CASES))
