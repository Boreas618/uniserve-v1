#!/usr/bin/env python3
"""The sparse-attention tile operators launched as programmatic dependents read their written inputs.

    python pdl_chain_check.py --op <operator>

Each case launches its operator's Triton source on the device as the operator's module is compiled
for a dependent launch, behind a producer still writing its inputs (`../chain.py`).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chain  # noqa: E402


def vsa_scores():
    module = chain.source("vsa_scores")
    heads, width, block, query_tiles, key_tiles = 4, 64, 64, 32, 48

    def launch(i, o, pdl):
        module._vsa_scores_kernel[(-(-query_tiles // 16), heads, 1)](
            i["query"],
            i["key"],
            o["out"],
            query_tiles,
            key_tiles,
            int(pdl),
            WIDTH=width,
            BLOCK=block,
            KEY_TILES=32,
            QUERIES=16,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"]]

    return chain.Case(
        "vsa_scores",
        {
            "query": torch.randn(query_tiles, heads, width),
            "key": torch.randn(key_tiles, heads, width),
        },
        {"out": torch.zeros(heads, query_tiles, key_tiles)},
        launch,
    )


def vsa_compress_pooled():
    module = chain.source("vsa_compress_pooled")
    heads, width, block, query_tiles, tiles = 4, 64, 64, 16, 80

    def launch(i, o, pdl):
        module._vsa_compress_pooled_kernel[(1, heads, 1)](
            i["scores"],
            i["values"],
            i["sizes"],
            o["out"],
            query_tiles,
            tiles,
            int(pdl),
            HEADS=heads,
            WIDTH=width,
            BLOCK=block,
            QUERIES=16,
            KEYS=64,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"]]

    return chain.Case(
        "vsa_compress_pooled",
        {
            "scores": torch.randn(heads, query_tiles, tiles),
            "values": torch.randn(tiles, heads, width),
            "sizes": torch.randint(1, 64, (tiles,), dtype=torch.int32),
        },
        {"out": torch.zeros(heads, query_tiles, width, dtype=torch.bfloat16)},
        launch,
    )


CASES = {case.__name__: case for case in (vsa_scores, vsa_compress_pooled)}

if __name__ == "__main__":
    sys.exit(chain.main(CASES))
