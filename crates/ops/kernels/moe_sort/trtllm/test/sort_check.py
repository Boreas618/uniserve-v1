#!/usr/bin/env python3
"""Run moe_sort through the built module and check every output it writes.

This is the GPU check a build host cannot do: it loads the module the store
holds (``module.so``), calls the operator's entry with the call structure of
``crates/ops/src/moe/sort.rs`` (632 bytes: ``ids`` 0, ``scales`` 8, the six
outputs 16..64, the extents 64..120, ``error`` 120), and compares all six
outputs with TensorRT-LLM's routing order computed here in integer arithmetic.

    python sort_check.py <module_dir> [--tokens 64] [--top 8] [--experts 128]
                         [--local 128] [--offset 0] [--tile 128] [--scales f32]

``<module_dir>`` is what ``resolve`` hands a runner: the directory with
``module.so``. Run it on the GPU host with the Program Server's own environment
(``crates/program-server/python/.venv/bin/python``), which has torch.

The reference is the order the design states and TensorRT-LLM implements
(``cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingKernel.cuh``): routes
grouped by local expert, each expert's permuted rows in ascending expanded-route
order **and whole tiles** — an expert's rows pad to the next ``tile`` boundary —
``tile_to_expert`` the local expert of each tile, ``tile_to_limit`` the
exclusive end row of a tile's live rows counted over the whole routing
(TensorRT-LLM's ``mnLimit``), ``total_padded_tokens`` and
``num_non_exiting_tiles`` the scalars the grouped GEMMs read, and ``-1`` for
every route this rank does not hold and every padding row of a last tile.

The buffers beyond ``total_padded_tokens`` are the caller's unused tail (the
GEMMs read only the live prefix), so they are reported, not compared.
"""

import argparse
import ctypes
import pathlib
import sys

import torch

# The operator's `DType::as_i64()` numbering, for `scales_dtype`.
ELEMENT_TYPES = {
    "u8": 0,
    "i32": 1,
    "i64": 2,
    "f16": 3,
    "bf16": 4,
    "f32": 5,
    "f8e4m3": 6,
    "f4e2m1": 7,
}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


class Call(ctypes.Structure):
    """`ops::moe::sort::Call`, field for field."""

    _fields_ = [
        ("ids", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("tile_to_expert", ctypes.c_void_p),
        ("tile_to_limit", ctypes.c_void_p),
        ("expanded_to_permuted", ctypes.c_void_p),
        ("permuted_to_expanded", ctypes.c_void_p),
        ("total_padded_tokens", ctypes.c_void_p),
        ("num_non_exiting_tiles", ctypes.c_void_p),
        ("tokens", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("local_experts", ctypes.c_int64),
        ("local_offset", ctypes.c_int64),
        ("tile_tokens", ctypes.c_int64),
        ("scales_dtype", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def max_tiles(tokens, top, local_experts, tile):
    """The tiles a sort's buffers hold, as `ops::moe::max_tiles` computes."""
    expanded = tokens * top
    if expanded <= local_experts or tile == 0:
        return expanded
    return (expanded + (tile - 1) * local_experts) // tile


def reference(ids, tokens, top, experts, local, offset, tile):
    """The six outputs TensorRT-LLM's routing order gives for `ids`.

    `ids` is `[tokens, top]` on the host, as Python lists of expert ids.
    """
    counts = [0] * local
    for token in range(tokens):
        for route in range(top):
            expert = ids[token][route] - offset
            if 0 <= expert < local:
                counts[expert] += 1

    # Every expert's rows are whole tiles, and the tiles follow each other in
    # expert order: the grouped GEMM reads one expert per tile group.
    tiles_of = [(count + tile - 1) // tile for count in counts]
    tile_offsets = []
    running = 0
    for tiles in tiles_of:
        tile_offsets.append(running)
        running += tiles
    total_tiles = running
    total_padded = total_tiles * tile

    tile_to_expert = [-1] * max_tiles(tokens, top, local, tile)
    tile_to_limit = [-1] * len(tile_to_expert)
    for expert, count in enumerate(counts):
        for index in range(tiles_of[expert]):
            at = tile_offsets[expert] + index
            tile_to_expert[at] = expert
            # The live rows end at the next tile boundary, or at the expert's
            # last row, whichever comes first — counted from the first permuted
            # row of the whole routing.
            boundary = (tile_offsets[expert] + index + 1) * tile
            end = tile_offsets[expert] * tile + count
            tile_to_limit[at] = min(boundary, end)

    expanded_to_permuted = [-1] * (tokens * top)
    permuted_to_expanded = [-1] * (len(tile_to_expert) * tile)
    filled = [0] * local
    for token in range(tokens):
        for route in range(top):
            expanded = token * top + route
            expert = ids[token][route] - offset
            if 0 <= expert < local:
                row = tile_offsets[expert] * tile + filled[expert]
                filled[expert] += 1
                expanded_to_permuted[expanded] = row
                permuted_to_expanded[row] = expanded

    return {
        "tile_to_expert": tile_to_expert,
        "tile_to_limit": tile_to_limit,
        "expanded_to_permuted": expanded_to_permuted,
        "permuted_to_expanded": permuted_to_expanded,
        "total_padded_tokens": total_padded,
        "num_non_exiting_tiles": total_tiles,
    }


def compare(name, expected, actual, length=None):
    """The first place two integer arrays differ, or `None`."""
    count = len(expected) if length is None else length
    for at in range(count):
        if int(actual[at]) != expected[at]:
            return (
                f"{name}[{at}] = {int(actual[at])}, the order says "
                f"{expected[at]}"
            )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module", type=pathlib.Path, help="the directory with module.so"
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--local", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--scales", default="f32", choices=sorted(TORCH_TYPES))
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    module = arguments.module
    if not (module / "module.so").is_file():
        beside = pathlib.Path(__file__).resolve().parent.parent / module
        if (beside / "module.so").is_file():
            module = beside
    module_path = (module / "module.so").resolve()
    if not module_path.is_file():
        print(
            f"no module at {module_path}: build it first (the package's "
            f"src/build.py with UNISERVE_OUT pointing at that directory)",
            file=sys.stderr,
        )
        return 1
    if ctypes.sizeof(Call) != 632 or Call.error.offset != 120:
        print(
            "this script's call structure is not the operator's",
            file=sys.stderr,
        )
        return 1
    if not torch.cuda.is_available():
        print("SKIPPED: this host has no GPU", file=sys.stderr)
        return 0

    library = ctypes.CDLL(str(module_path))
    entry = library.uniserve_moe_sort
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    tokens, top, experts = arguments.tokens, arguments.top, arguments.experts
    local, offset, tile = arguments.local, arguments.offset, arguments.tile
    device = torch.device("cuda")
    torch.manual_seed(arguments.seed)

    # Routes over the rank's whole vocabulary, so both the held and the
    # unheld window are exercised when `local < experts`.
    ids = torch.randint(
        0, experts, (tokens, top), dtype=torch.int32, device=device
    )
    scales = torch.rand(
        (tokens, top), dtype=TORCH_TYPES[arguments.scales], device=device
    )
    tiles = max_tiles(tokens, top, local, tile)
    permuted = tiles * tile

    tile_to_expert = torch.full((tiles,), -1, dtype=torch.int32, device=device)
    tile_to_limit = torch.full((tiles,), -1, dtype=torch.int32, device=device)
    expanded_to_permuted = torch.full(
        (tokens, top), -1, dtype=torch.int32, device=device
    )
    permuted_to_expanded = torch.full(
        (permuted,), -1, dtype=torch.int32, device=device
    )
    total_padded_tokens = torch.zeros((1,), dtype=torch.int32, device=device)
    num_non_exiting_tiles = torch.zeros((1,), dtype=torch.int32, device=device)

    call = Call()
    call.ids = ctypes.c_void_p(ids.data_ptr())
    call.scales = ctypes.c_void_p(scales.data_ptr())
    call.tile_to_expert = ctypes.c_void_p(tile_to_expert.data_ptr())
    call.tile_to_limit = ctypes.c_void_p(tile_to_limit.data_ptr())
    call.expanded_to_permuted = ctypes.c_void_p(expanded_to_permuted.data_ptr())
    call.permuted_to_expanded = ctypes.c_void_p(permuted_to_expanded.data_ptr())
    call.total_padded_tokens = ctypes.c_void_p(total_padded_tokens.data_ptr())
    call.num_non_exiting_tiles = ctypes.c_void_p(
        num_non_exiting_tiles.data_ptr()
    )
    call.tokens = tokens
    call.top = top
    call.experts = experts
    call.local_experts = local
    call.local_offset = offset
    call.tile_tokens = tile
    call.scales_dtype = ELEMENT_TYPES[arguments.scales]

    stream = torch.cuda.current_stream().cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if status != 0:
        print(
            f"the entry refused the call with {status}: "
            f"{call.error.decode(errors='replace')}",
            file=sys.stderr,
        )
        return 1

    expected = reference(
        ids.tolist(), tokens, top, experts, local, offset, tile
    )
    padded = int(total_padded_tokens[0])
    live_tiles = int(num_non_exiting_tiles[0])
    failures = []
    if padded != expected["total_padded_tokens"]:
        failures.append(
            f"total_padded_tokens = {padded}, the order says "
            f"{expected['total_padded_tokens']}"
        )
    if live_tiles != expected["num_non_exiting_tiles"]:
        failures.append(
            f"num_non_exiting_tiles = {live_tiles}, the order says "
            f"{expected['num_non_exiting_tiles']}"
        )
    for name, actual, length in [
        (
            "tile_to_expert",
            tile_to_expert.cpu(),
            expected["num_non_exiting_tiles"],
        ),
        (
            "tile_to_limit",
            tile_to_limit.cpu(),
            expected["num_non_exiting_tiles"],
        ),
        ("expanded_to_permuted", expanded_to_permuted.flatten().cpu(), None),
        ("permuted_to_expanded", permuted_to_expanded.cpu(), padded),
    ]:
        difference = compare(name, expected[name], actual, length)
        if difference is not None:
            failures.append(difference)

    live_routes = sum(
        1 for value in expected["permuted_to_expanded"][:padded] if value >= 0
    )
    print(
        f"tokens {tokens} top {top} experts {experts} local {local} "
        f"offset {offset} tile {tile}: {live_tiles} tile(s), "
        f"{padded} padded row(s), {live_routes} live route(s)"
    )
    tail = permuted_to_expanded[padded:].cpu()
    if padded < permuted and bool((tail >= 0).any()):
        print(
            f"note: {int((tail >= 0).sum())} row(s) past "
            f"total_padded_tokens hold a route; the GEMMs read only the "
            f"live prefix"
        )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("all six outputs match TensorRT-LLM's routing order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
