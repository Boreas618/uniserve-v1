#!/usr/bin/env python3
"""Run the `gather_gemm` module on a device and check the FC1 gated product it writes.

    crates/program-server/python/.venv/bin/python gather_check.py <module_dir> [--tokens N]

`<module_dir>` is what `resolve` hands a runner: the directory holding `module.so`. The module is
compiled for one descriptor of the fixture —
`hidden_2048_intermediate_1536_experts_128_top_8_activation_silu_dtype_bf16_tile_128x128_cluster_1x1`
— so every case runs those shapes and varies only the token count, which is a runtime argument.

The call is the operator's `moe::grouped::GatherCall` (`crates/ops/src/moe/grouped.rs:173`, 624
bytes, `error` at 112), and the metadata is a **valid sort built here** rather than assumed: routes
are grouped by expert, each expert's rows are cut into tiles of 128, `tile_to_expert` and
`tile_to_limit` describe them, `permuted_to_expanded` is the inverse map, and
`num_non_exiting_tiles` counts the live tiles. That is what `moe_sort` produces. Building it here is
what lets the cases include **an expert with no routes at all** and a **last tile that is only
partly filled** — the two places a gather that indexes the map without respecting `tile_to_limit`
writes outside its tile.

The reference is the operator's text: for a live tile's row `p` with map entry `r`,
`out[p] = act(gate) · up` where `up = W[e, :I] · x` and `gate = W[e, I:] · x` (each expert's rows hold
its up block then its gate block) and `x` is row `r / top` of the hidden states. The comparison uses
the operator's own bound — `Tolerance::dot(hidden, f32, bf16)` carried through the activation, whose
absolute term is `2 γ (8² · 2)` (`ACTIVATION_ENVELOPE`, `crates/ops/src/moe/grouped.rs`) — computed
here from the same expressions rather than guessed.

Rows the metadata does not name are **not** compared: they are the padding the sort added so the
tiles are whole, and the operator writes only the rows its map holds.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

BF16 = 4
#: The descriptor's M tile: the rows of one tile of a sorted expert.
TILE_ROWS = 128


class Call(ctypes.Structure):
    """`moe::grouped::GatherCall`, field for field."""

    _fields_ = [
        ("hidden", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("tile_to_expert", ctypes.c_void_p),
        ("tile_to_limit", ctypes.c_void_p),
        ("permuted_to_expanded", ctypes.c_void_p),
        ("num_non_exiting_tiles", ctypes.c_void_p),
        ("tokens", ctypes.c_int64),
        ("permuted_rows", ctypes.c_int64),
        ("width", ctypes.c_int64),
        ("hidden_size", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


#: `crates/ops/src/moe/grouped.rs:192` — `GATHER_CALL_LAYOUT`'s size. The mirror above is written by
#: hand, so it is measured rather than remembered: a field added to the operator's struct shifts
#: everything after it, and the call below would pass garbage flags instead of failing.
GATHER_CALL_SIZE = 624
assert ctypes.sizeof(Call) == GATHER_CALL_SIZE, (
    f"the check's GatherCall mirror is {ctypes.sizeof(Call)} bytes, the operator's layout says "
    f"{GATHER_CALL_SIZE} (crates/ops/src/moe/grouped.rs:192)"
)


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(hidden):
    """`GatherGemm`'s tolerance: `dot(hidden, f32, bf16)` with the activation's envelope."""
    accumulate = 2.0**-24
    bf16_roundoff = 2.0**-8
    relative = 2.0 * gamma(hidden, accumulate) * math.sqrt(hidden) + bf16_roundoff
    absolute = 2.0 * gamma(hidden, accumulate) * (8.0 * 8.0 * 2.0)
    return relative, absolute


def tiles_of(routes, experts, top):
    """The sort's view of the routes: one list of `(token, slot)` per expert, in order."""
    per_expert = [[] for _ in range(experts)]
    for token, slots in enumerate(routes.tolist()):
        for slot, expert in enumerate(slots):
            if expert >= 0:
                per_expert[expert].append((token, slot))
    return per_expert


def sort_metadata(per_expert, top, tile_rows=TILE_ROWS):
    """The three tables and the live tile count, as `moe_sort` writes them."""
    tile_to_expert, tile_to_limit, permuted_to_expanded = [], [], []
    for expert, members in enumerate(per_expert):
        for start in range(0, len(members), tile_rows):
            chunk = members[start:start + tile_rows]
            tile_to_expert.append(expert)
            tile_to_limit.append(len(chunk))
            permuted_to_expanded.extend(token * top + slot for token, slot in chunk)
            # The sort pads every tile to whole rows; the map names each of them.
            permuted_to_expanded.extend([0] * (tile_rows - len(chunk)))
    if not tile_to_expert:
        tile_to_expert, tile_to_limit = [-1], [0]
        permuted_to_expanded = [0] * tile_rows
    live = sum(1 for limit in tile_to_limit if limit > 0)
    return tile_to_expert, tile_to_limit, permuted_to_expanded, live


def reference(hidden, weights, tile_expert, tile_limit, map_rows, intermediate, top):
    """The gated product the metadata names, in FP32, tile by tile."""
    expected = torch.zeros(len(map_rows), intermediate, dtype=torch.float32, device="cuda")
    for tile, (expert, limit) in enumerate(zip(tile_expert, tile_limit)):
        if limit == 0:
            continue
        first = tile * TILE_ROWS
        expanded = torch.tensor(
            map_rows[first:first + limit], dtype=torch.int64, device="cuda"
        )
        tokens = torch.div(expanded, top, rounding_mode="floor")
        x = hidden[tokens].to(torch.float32)
        weight = weights[expert].to(torch.float32)
        up = x @ weight[:intermediate].t()
        gate = x @ weight[intermediate:].t()
        expected[first:first + limit] = torch.nn.functional.silu(gate) * up
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=pathlib.Path, help="the module directory resolve hands a runner")
    parser.add_argument("--tokens", type=int, default=512, help="tokens whose routes are sorted")
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--top", type=int, default=8)
    options = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "this module is a compiled CuTe DSL native library: it needs the device it was built for"
        )
    library = ctypes.CDLL(str(options.module / "module.so"))
    entry = library.uniserve_gather_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    device = torch.device("cuda")
    torch.manual_seed(0)
    tokens, top = options.tokens, options.top
    experts, hidden_size, intermediate = options.experts, options.hidden, options.intermediate

    # Every token takes `top` distinct experts, and **one expert is emptied by hand**: with 512
    # tokens × 8 routes over 128 experts every expert expects about 32 routes, so a dropped route
    # would leave no expert empty and the case would be vacuous. The dead expert's routes are moved
    # to a neighbour, which keeps every token's route count and leaves exactly one expert with none.
    routes = torch.stack([torch.randperm(experts)[:top] for _ in range(tokens)])
    emptied = 7 % experts
    routes[routes == emptied] = (emptied + 1) % experts
    # And one route dies, so one tile is only partly filled.
    routes[0, 0] = -1
    per_expert = tiles_of(routes, experts, top)
    if not any(not members for members in per_expert):
        raise SystemExit("the case was built with an empty expert in mind and has none")

    hidden = (torch.randn(tokens, hidden_size, dtype=torch.float32, device=device) * 0.5).to(
        torch.bfloat16
    )
    weights = (
        torch.randn(experts, 2 * intermediate, hidden_size, dtype=torch.float32, device=device)
        * 0.05
    ).to(torch.bfloat16)

    tile_expert, tile_limit, map_rows, live_tiles = sort_metadata(per_expert, top)
    # The tables stay alive in these names until after the call: a temporary's pointer would be
    # freed the moment the expression ended.
    tile_to_expert = torch.tensor(tile_expert, dtype=torch.int32, device=device)
    tile_to_limit = torch.tensor(tile_limit, dtype=torch.int32, device=device)
    permuted_to_expanded = torch.tensor(map_rows, dtype=torch.int32, device=device)
    live_table = torch.tensor([live_tiles], dtype=torch.int32, device=device)
    permuted_rows = len(map_rows)
    out = torch.full(
        (permuted_rows, intermediate), float("nan"), dtype=torch.bfloat16, device=device
    )

    call = Call(
        hidden=ctypes.c_void_p(hidden.data_ptr()),
        weights=ctypes.c_void_p(weights.data_ptr()),
        out=ctypes.c_void_p(out.data_ptr()),
        tile_to_expert=ctypes.c_void_p(tile_to_expert.data_ptr()),
        tile_to_limit=ctypes.c_void_p(tile_to_limit.data_ptr()),
        permuted_to_expanded=ctypes.c_void_p(permuted_to_expanded.data_ptr()),
        num_non_exiting_tiles=ctypes.c_void_p(live_table.data_ptr()),
        tokens=tokens,
        permuted_rows=permuted_rows,
        width=2 * intermediate,
        hidden_size=hidden_size,
        experts=experts,
        top=top,
        dtype=BF16,
        error=b"",
    )
    status = entry(ctypes.byref(call), stream)
    if status != 0:
        message = call.error.decode("utf-8", "replace").rstrip("\x00")
        raise SystemExit(f"the module refused the call with status {status}: {message}")

    # The rows the metadata names are a tile's **first `limit` rows**; the rest of each tile is the
    # sort's padding, and a kernel that walks a tile's rows without respecting `tile_to_limit` writes
    # into it. The NaN fill is only evidence if something looks at it, and the live rows are not a
    # prefix of the table: every tile pads at its own end.
    live_mask = torch.zeros(permuted_rows, dtype=torch.bool, device=device)
    for tile, limit in enumerate(tile_limit):
        live_mask[tile * TILE_ROWS:tile * TILE_ROWS + limit] = True
    if not bool(live_mask.any()):
        raise SystemExit("the metadata names no rows, so the case measures nothing")
    if not any(0 < limit < TILE_ROWS for limit in tile_limit):
        raise SystemExit(
            "no tile is partly filled, so the case does not exercise a tile's `tile_to_limit`"
        )
    if not bool(torch.isnan(out[~live_mask]).all()):
        written = int((~torch.isnan(out[~live_mask])).sum())
        raise SystemExit(
            f"a padding row was written: {written} of {out[~live_mask].numel()} values inside the "
            f"tiles past their `tile_to_limit`"
        )

    expected = reference(
        hidden, weights, tile_expert, tile_limit, map_rows, intermediate, top
    )
    got = out[live_mask].to(torch.float32)
    want = expected[live_mask]

    relative, absolute = bound(hidden_size)
    allowed = absolute + relative * want.abs()
    error = (got - want).abs()
    # `~(error <= allowed)`, not `error > allowed`: `out` starts as NaN precisely so that an
    # unwritten row shows up, and `NaN > allowed` is False — the sentinel would be invisible.
    violations = int((~(error <= allowed)).sum())
    print(
        f"tokens {tokens} top {top} experts {experts} hidden {hidden_size} intermediate "
        f"{intermediate}: {live_rows} live rows of {permuted_rows} in {live_tiles} tiles, "
        f"max |out - reference| = {float(torch.nan_to_num(error).max()):.6g}, bound rtol {relative:.3e} "
        f"atol {absolute:.3e}, experts with no routes {sum(1 for m in per_expert if not m)}, "
        f"partly filled tiles {sum(1 for limit in tile_limit if 0 < limit < TILE_ROWS)}"
    )
    if violations:
        position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
        print(
            f"{violations} of {error.numel()} gated products are outside the bound; first at "
            f"{position}: {float(got.reshape(-1)[position])} vs "
            f"{float(want.reshape(-1)[position])}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
