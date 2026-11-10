#!/usr/bin/env python3
"""The movement operators launched as programmatic dependents read their written inputs.

    python pdl_chain_check.py --op <operator>

Each case launches its operator's Triton source on the device as the operator's module is compiled
for a dependent launch, behind a producer still writing its inputs (`../chain.py`).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chain  # noqa: E402


def rows():
    module = chain.source("rows")
    count, columns, block = 8, 1024, 1024

    def launch(i, o, pdl):
        module._rows_kernel[(count, columns // block, 1)](
            i["source"],
            o["destination"],
            i["indices"],
            count,
            columns,
            block,
            int(pdl),
            COLUMNS=columns,
            REPLACE=True,
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["destination"]]

    return chain.Case(
        "rows",
        {
            "source": torch.randn(count, columns).to(torch.bfloat16),
            "indices": torch.randperm(32)[:count].to(torch.int32),
        },
        {"destination": torch.zeros(32, columns, dtype=torch.bfloat16)},
        launch,
    )


def cache_copy_region():
    module = chain.source("cache_copy_region")
    # A page of an FP8 pool: `columns` layer planes per unit, each `block_tokens` token rows of
    # `heads · head_dim`; the source is the page's layers `[units_per_page · columns]`, layer-major.
    columns, units_per_page, units_total, heads, head_dim, block_tokens = (
        4,
        2,
        6,
        8,
        16,
        16,
    )
    row = heads * head_dim
    plane = block_tokens * row
    layers = units_per_page * columns

    def launch(i, o, pdl):
        module._cache_copy_region_kernel[(columns, units_per_page, 1)](
            o["values"],
            o["scales"],
            o["flags"],
            i["source"],
            i["units"],
            0,
            block_tokens,
            units_total,
            ROW_SIZE=row,
            ROW_POW2=row,
            PLANE_ELEMENTS=plane,
            COLUMNS=columns,
            BF16=0,
            SCALED=1,
            BLOCK=row,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["values"], o["scales"], o["flags"]]

    return chain.Case(
        "cache_copy_region",
        {
            "source": torch.randn(layers, block_tokens, row),
            "units": torch.tensor([3, 1], dtype=torch.int32),
        },
        {
            "values": torch.zeros(
                columns, units_total, plane, dtype=torch.uint8
            ),
            "scales": torch.zeros(columns, units_total),
            "flags": torch.zeros(columns, units_total, dtype=torch.uint8),
        },
        launch,
    )


CASES = {case.__name__: case for case in (rows, cache_copy_region)}

if __name__ == "__main__":
    sys.exit(chain.main(CASES))
