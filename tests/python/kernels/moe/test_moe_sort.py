"""The `moe_sort` module's every output against TensorRT-LLM's routing order.

Each test loads the built module of its descriptor (resolved at the session's
Build Server for the session's device), calls the operator's entry with the
call structure of ``crates/ops/src/moe/sort.rs`` (648 bytes: ``ids`` 0,
``scales`` 8, the six outputs 16..64, the extents 64..120, ``error`` 120,
``live_tokens`` 632, ``permuted_to_token`` 640) and compares every output with
the order computed here in integer arithmetic.

The order is the one the design states and TensorRT-LLM implements
(``cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingKernel.cuh``): routes
grouped by local expert, each expert's permuted rows in ascending
expanded-route order and whole tiles (an expert's rows pad to the next tile
boundary), ``tile_to_expert`` the local expert of each tile,
``tile_to_limit`` the exclusive end row of a tile's live rows counted over the
whole routing (TensorRT-LLM's ``mnLimit``), ``total_padded_tokens`` and
``num_non_exiting_tiles`` the scalars the grouped GEMMs read, ``-1`` for every
route this rank does not hold and every padding row of a last tile, and
``-2`` in the route map of every row past a staged live token count. The
buffers beyond ``total_padded_tokens`` are the caller's unused tail, which the
GEMMs never read, so they are not compared.

The cases: every route held and every row live; a staged live count, below the
batch; and a rank that holds a window of the vocabulary, so routes it does not
hold meet routes it does.
"""

import ctypes

import pytest
import torch

pytestmark = pytest.mark.unit

#: `ops::abi::code`'s numbers of the scales' element types.
ELEMENT_TYPES = {"f16": 3, "bf16": 4, "f32": 5}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The `static_assert` of the package's header that pins the call's size, and
#: the offset of its error buffer.
SIZE_ASSERTION = "sizeof(uniserve_moe_sort_call) =="
ERROR_OFFSET = 120

#: The value the route map starts as: the sort must overwrite it, where a
#: pre-filled -1 would hide a kernel that never writes the absent entries.
UNWRITTEN = -7


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
        ("live_tokens", ctypes.c_void_p),
        ("permuted_to_token", ctypes.c_void_p),
    ]


def test_the_call_mirror_is_the_headers_sort_call(kernels):
    """The package's header pins the size, read rather than restated."""
    header = kernels / "moe_sort" / "trtllm" / "include" / "uniserve_moe_sort.h"
    text = header.read_text()
    assert SIZE_ASSERTION in text, f"{header}: no `{SIZE_ASSERTION}` assertion"
    declared = int(
        text.split(SIZE_ASSERTION, 1)[1].lstrip(" =").split()[0].rstrip(",")
    )
    assert ctypes.sizeof(Call) == declared, (
        f"the check's Call holds {ctypes.sizeof(Call)} bytes and the header "
        f"declares {declared}: the mirror is stale, and the module would read "
        "past it"
    )
    assert Call.error.offset == ERROR_OFFSET, (
        f"Call.error is at {Call.error.offset}, the operator's at "
        f"{ERROR_OFFSET}"
    )


def max_tiles(tokens, top, local_experts, tile):
    """The tiles a sort's buffers hold, as `ops::moe::max_tiles` computes."""
    expanded = tokens * top
    if expanded <= local_experts or tile == 0:
        return expanded
    return (expanded + (tile - 1) * local_experts) // tile


def reference(ids, tokens, top, local, offset, tile, live=None):
    """The outputs TensorRT-LLM's routing order gives for `ids`.

    `ids` is `[tokens, top]` expert ids as lists. `live` is the batch's token
    count when the caller stages one, and `None` when the call passes none: a
    row of `token >= live` is counted into no tile and its route map entry is
    `-2`, padding, a different absence from the `-1` of a route this rank
    lacks.
    """
    counts = [0] * local
    for token in range(tokens):
        if live is not None and token >= live:
            continue
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

    tile_to_expert = [-1] * max_tiles(tokens, top, local, tile)
    tile_to_limit = [-1] * len(tile_to_expert)
    for expert, count in enumerate(counts):
        for index in range(tiles_of[expert]):
            at = tile_offsets[expert] + index
            tile_to_expert[at] = expert
            # The live rows end at the next tile boundary, or at the expert's
            # last row, whichever comes first, counted from the first permuted
            # row of the whole routing.
            boundary = (at + 1) * tile
            end = tile_offsets[expert] * tile + count
            tile_to_limit[at] = min(boundary, end)

    expanded_to_permuted = [-1] * (tokens * top)
    permuted_to_expanded = [-1] * (len(tile_to_expert) * tile)
    permuted_to_token = [-1] * (len(tile_to_expert) * tile)
    filled = [0] * local
    for token in range(tokens):
        for route in range(top):
            expanded = token * top + route
            if live is not None and token >= live:
                expanded_to_permuted[expanded] = -2
                continue
            expert = ids[token][route] - offset
            if 0 <= expert < local:
                row = tile_offsets[expert] * tile + filled[expert]
                filled[expert] += 1
                expanded_to_permuted[expanded] = row
                permuted_to_expanded[row] = expanded
                permuted_to_token[row] = token

    return {
        "tile_to_expert": tile_to_expert,
        "tile_to_limit": tile_to_limit,
        "expanded_to_permuted": expanded_to_permuted,
        "permuted_to_expanded": permuted_to_expanded,
        "permuted_to_token": permuted_to_token,
        "total_padded_tokens": total_tiles * tile,
        "num_non_exiting_tiles": total_tiles,
    }


def first_difference(name, expected, actual, length):
    """The first place two integer arrays differ within `length`, or `None`."""
    for at in range(length):
        if int(actual[at]) != expected[at]:
            return (
                f"{name}[{at}] = {int(actual[at])}, the order says "
                f"{expected[at]}"
            )
    return None


#: The routing: tokens, routes per token, the vocabulary, the tile and the
#: scales' type.
TOKENS, TOP, EXPERTS, TILE, SCALES = 64, 8, 128, 128, "f32"


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("local", "offset", "live"),
    [
        pytest.param(128, 0, None, id="every route held, every row live"),
        pytest.param(128, 0, 40, id="40 live rows of 64"),
        pytest.param(64, 32, None, id="a window of 64 experts from 32"),
    ],
)
def test_every_output_is_the_routing_order(
    local, offset, live, torch_device, kernel_module
):
    key = (
        f"experts_{EXPERTS}_top_{TOP}_local_{local}_tile_{TILE}_scales_"
        f"{SCALES}_counted_true"
    )
    library = ctypes.CDLL(
        str(kernel_module("moe_sort", "trtllm", key) / "module.so")
    )
    entry = library.uniserve_moe_sort
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    torch.manual_seed(0)
    # Routes over the whole vocabulary, so a rank holding a window meets both
    # the routes it holds and the routes it does not.
    ids = torch.randint(
        0, EXPERTS, (TOKENS, TOP), dtype=torch.int32, device=torch_device
    )
    scales = torch.rand(
        (TOKENS, TOP), dtype=TORCH_TYPES[SCALES], device=torch_device
    )
    tiles = max_tiles(TOKENS, TOP, local, TILE)
    permuted = tiles * TILE

    def filled(shape, value):
        return torch.full(shape, value, dtype=torch.int32, device=torch_device)

    outputs = {
        "tile_to_expert": filled((tiles,), -1),
        "tile_to_limit": filled((tiles,), -1),
        "expanded_to_permuted": filled((TOKENS, TOP), UNWRITTEN),
        "permuted_to_expanded": filled((permuted,), -1),
        "permuted_to_token": filled((permuted,), UNWRITTEN),
        "total_padded_tokens": filled((1,), 0),
        "num_non_exiting_tiles": filled((1,), 0),
    }
    live_tokens = None if live is None else filled((1,), live)

    call = Call()
    call.ids = ids.data_ptr()
    call.scales = scales.data_ptr()
    for name, tensor in outputs.items():
        setattr(call, name, tensor.data_ptr())
    call.tokens = TOKENS
    call.top = TOP
    call.experts = EXPERTS
    call.local_experts = local
    call.local_offset = offset
    call.tile_tokens = TILE
    call.scales_dtype = ELEMENT_TYPES[SCALES]
    call.live_tokens = None if live_tokens is None else live_tokens.data_ptr()
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the entry refused the call with {status}: "
        f"{call.error.decode(errors='replace')}"
    )

    expected = reference(ids.tolist(), TOKENS, TOP, local, offset, TILE, live)
    padded = int(outputs["total_padded_tokens"][0])
    live_tiles = int(outputs["num_non_exiting_tiles"][0])
    differences = []
    for name, value in (
        ("total_padded_tokens", padded),
        ("num_non_exiting_tiles", live_tiles),
    ):
        if value != expected[name]:
            differences.append(
                f"{name} = {value}, the order says {expected[name]}"
            )
    tiles_live = expected["num_non_exiting_tiles"]
    for name, length in (
        ("tile_to_expert", tiles_live),
        ("tile_to_limit", tiles_live),
        ("expanded_to_permuted", TOKENS * TOP),
        ("permuted_to_expanded", padded),
        ("permuted_to_token", padded),
    ):
        actual = outputs[name].flatten().cpu()
        difference = first_difference(name, expected[name], actual, length)
        if difference is not None:
            differences.append(difference)
    assert not differences, (
        f"tokens {TOKENS} top {TOP} experts {EXPERTS} local {local} offset "
        f"{offset} tile {TILE} live {live}: {'; '.join(differences)}"
    )
