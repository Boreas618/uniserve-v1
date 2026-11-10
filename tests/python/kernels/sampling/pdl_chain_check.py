#!/usr/bin/env python3
"""The sampling operators launched as programmatic dependents read their written inputs.

    python pdl_chain_check.py --op <operator>

Each case launches its operator's Triton source on the device as the operator's module is compiled
for a dependent launch, behind a producer still writing its inputs (`../chain.py`).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chain  # noqa: E402


def logits_gather():
    module = chain.source("logits_gather")
    count, columns, block = 8, 1024, 1024

    def launch(i, o, pdl):
        module._logits_gather_kernel[(count * columns // block,)](
            i["hidden"],
            i["rows"],
            o["out"],
            count,
            columns,
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"]]

    return chain.Case(
        "logits_gather",
        {
            "hidden": torch.randn(64, columns).to(torch.bfloat16),
            "rows": torch.randperm(64)[:count].to(torch.int32),
        },
        {"out": torch.zeros(count, columns, dtype=torch.bfloat16)},
        launch,
    )


def vocab_unpad():
    module = chain.source("vocab_unpad")
    rows, padded, vocabulary, ranks, block = 4, 1024, 1000, 2, 1024

    def launch(i, o, pdl):
        module._vocab_unpad_kernel[(-(-rows * vocabulary // block),)](
            i["gathered"],
            o["out"],
            rows,
            padded,
            vocabulary,
            ranks,
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"]]

    return chain.Case(
        "vocab_unpad",
        {
            "gathered": torch.randn(ranks * rows * padded // ranks).to(
                torch.bfloat16
            )
        },
        {"out": torch.zeros(rows, vocabulary)},
        launch,
    )


def filter_candidates():
    module = chain.source("filter_candidates")
    rows, width = 4, 1024
    parameters = torch.zeros(rows, 8)
    parameters[:, 1], parameters[:, 2], parameters[:, 3] = 64.0, 0.9, 0.01

    def launch(i, o, pdl):
        # The operator filters its values in place: they are both an input and its result.
        module._filter_candidates_kernel[(rows,)](
            i["values"],
            i["ids"],
            i["parameters"],
            o["valid"],
            rows,
            width,
            BLOCK=width,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [i["values"], o["valid"]]

    return chain.Case(
        "filter_candidates",
        {
            "values": torch.randn(rows, width)
            .sort(dim=-1, descending=True)
            .values
            * 3.0,
            "ids": torch.randperm(width).to(torch.int32).repeat(rows, 1),
            "parameters": parameters,
        },
        {"valid": torch.full((rows,), -1, dtype=torch.int32)},
        launch,
    )


def inverse_cdf():
    module = chain.source("inverse_cdf")
    rows, width, tile = 4, 16384, 8192

    def launch(i, o, pdl):
        module._inverse_cdf_kernel[(rows,)](
            i["probabilities"],
            i["ids"],
            i["draws"],
            o["tokens"],
            rows,
            width,
            BLOCK=tile,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=8,
        )
        return [o["tokens"]]

    return chain.Case(
        "inverse_cdf",
        {
            "probabilities": torch.softmax(
                torch.randn(rows, width) * 3.0, dim=-1
            ),
            "ids": torch.arange(width, dtype=torch.int32).repeat(rows, 1),
            "draws": torch.rand(rows),
        },
        {"tokens": torch.full((rows,), -7, dtype=torch.int32)},
        launch,
    )


def argmax():
    module = chain.source("argmax")
    rows, width = 4, 4096

    def launch(i, o, pdl):
        module._argmax_kernel[(rows,)](
            i["values"],
            o["out"],
            rows,
            WIDTH=width,
            BLOCK=1024,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=8,
        )
        return [o["out"]]

    return chain.Case(
        "argmax",
        {"values": torch.randn(rows, width)},
        {"out": torch.full((rows,), -7, dtype=torch.int32)},
        launch,
    )


def logprob_gather():
    module = chain.source("logprob_gather")
    rows, columns, step = 8, 1024, 1024

    def launch(i, o, pdl):
        module._logprob_gather_kernel[(rows,)](
            i["logprobabilities"],
            i["tokens"],
            o["out"],
            o["ranks"],
            rows,
            columns,
            STEP=step,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"], o["ranks"]]

    return chain.Case(
        "logprob_gather",
        {
            "logprobabilities": torch.log_softmax(
                torch.randn(rows, columns), dim=-1
            ),
            "tokens": torch.randint(1, columns, (rows,), dtype=torch.int64),
        },
        {
            "out": torch.full((rows,), float("nan")),
            "ranks": torch.full((rows,), -1, dtype=torch.int32),
        },
        launch,
    )


def top_logprobs():
    module = chain.source("top_logprobs")
    rows, width, tile, candidates = 4, 4096, 1024, 8
    tiles = width // tile

    def launch(i, o, pdl):
        module._top_logprobs_kernel[(rows, tiles)](
            i["values"],
            i["values"],
            o["values"],
            o["ids"],
            width,
            tiles,
            CANDIDATES=candidates,
            TILE=tile,
            INDIRECT=False,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["values"], o["ids"]]

    return chain.Case(
        "top_logprobs",
        {"values": torch.randn(rows, width)},
        {
            "values": torch.full((rows, tiles * candidates), float("nan")),
            "ids": torch.full(
                (rows, tiles * candidates), -1, dtype=torch.int32
            ),
        },
        launch,
    )


def top_k():
    module = chain.source("top_k")
    rows, vocabulary, width, bins = 4, 8192, 64, 16

    def launch(i, o, pdl):
        module._top_k_kernel[(rows,)](
            i["logits"],
            o["values"],
            o["ids"],
            o["workspace"],
            rows,
            vocabulary,
            width,
            TILE=4096,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=8,
        )
        return [o["values"], o["ids"]]

    return chain.Case(
        "top_k",
        {"logits": torch.randn(rows, vocabulary)},
        {
            "values": torch.full((rows, width), float("nan")),
            "ids": torch.full((rows, width), -1, dtype=torch.int32),
            # Two key planes, two id planes and the two nibble tables per row.
            "workspace": torch.zeros(
                rows, 4 * width + 2 * bins, dtype=torch.int32
            ),
        },
        launch,
    )


def candidate_order():
    module = chain.source("candidate_order")
    rows, width = 4, 300

    def launch(i, o, pdl):
        module._candidate_order_kernel[(rows,)](
            i["values"],
            i["ids"],
            o["values"],
            o["ids"],
            o["counts"],
            rows,
            width,
            width,
            BLOCK=128,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["values"], o["ids"]]

    return chain.Case(
        "candidate_order",
        {
            "values": torch.rand(rows, width),
            "ids": torch.randperm(width).to(torch.int32).repeat(rows, 1),
        },
        {
            "values": torch.full((rows, width), float("nan")),
            "ids": torch.full((rows, width), -1, dtype=torch.int32),
            "counts": torch.zeros(rows, width, dtype=torch.int32),
        },
        launch,
    )


def posterior():
    module = chain.source("posterior")
    positions, channels, block = 64, 16, 256
    total = positions * channels

    # Every operand is a `[positions, channels]` view with a channel stride of 2: a stride of 1
    # would be specialized into the launch here, where the module takes it as a word.
    row, step = 2 * channels, 2

    def launch(i, o, pdl):
        module._posterior_kernel[(-(-total // block),)](
            i["mean"],
            i["log_variance"],
            i["noise"],
            o["out"],
            channels,
            row,
            step,
            row,
            step,
            row,
            step,
            row,
            step,
            total,
            int(pdl),
            BLOCK=block,
            PDL=pdl,
            launch_pdl=pdl,
            num_warps=4,
        )
        return [o["out"]]

    return chain.Case(
        "posterior",
        {
            "mean": torch.randn(positions, row),
            "log_variance": torch.randn(positions, row),
            "noise": torch.randn(positions, row),
        },
        {"out": torch.zeros(positions, row)},
        launch,
    )


CASES = {
    case.__name__: case
    for case in (
        logits_gather,
        vocab_unpad,
        filter_candidates,
        inverse_cdf,
        argmax,
        logprob_gather,
        top_logprobs,
        top_k,
        candidate_order,
        posterior,
    )
}

if __name__ == "__main__":
    sys.exit(chain.main(CASES))
