#!/usr/bin/env python3
"""Run the `route_gemm` module on a device and check the FC2 product it writes.

    crates/program-server/python/.venv/bin/python route_check.py <module_dir> [--tokens N]

`<module_dir>` is what `resolve` hands a runner: the directory holding `module.so`. The module is
compiled for one descriptor of the fixture —
`hidden_2048_intermediate_768_experts_128_top_8_dtype_bf16_tile_128x128_cluster_1x1` — so every case
runs those shapes and varies only the token count.

The call is the operator's `RouteCall` (`crates/ops/src/moe/grouped.rs`, `error` at 104), and the
metadata is a **valid sort built here**, as in `gather_gemm`'s check: routes grouped by expert, each
expert's rows cut into tiles of 128, `permuted_to_expanded` the inverse map. That is what makes an
**expert with no routes** and a **partly filled last tile** possible — the two cases a gather that
ignores `tile_to_limit` fails and a full, regular one passes.

The operator's product is a scatter as well as a multiply, and the check is written around that:

```text
out[t · top + k] = W[ids[t, k]] · rows[p]        for the permuted row p whose map entry
                                                 names the route t · top + k
```

so every expanded row must appear **exactly once** — a route whose permuted row was lost would leave
its output row untouched, which the NaN fill catches — and no row may be written twice. Rows are
compared with `Tolerance::dot(intermediate, f32, bf16)`, the reduction length being the FC2
contraction; there is no activation here, so no envelope term.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

BF16 = 4
TILE_ROWS = 128


class Call(ctypes.Structure):
    """`moe::grouped::RouteCall`, field for field."""

    _fields_ = [
        ("rows", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("tile_to_expert", ctypes.c_void_p),
        ("tile_to_limit", ctypes.c_void_p),
        ("permuted_to_expanded", ctypes.c_void_p),
        ("num_non_exiting_tiles", ctypes.c_void_p),
        ("permuted_rows", ctypes.c_int64),
        ("width", ctypes.c_int64),
        ("intermediate", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("expanded_rows", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


#: `crates/ops/src/moe/grouped.rs` — `ROUTE_CALL_LAYOUT`'s size, which `entry.h` also pins. Measured,
#: not remembered: a field added to the operator's struct shifts every field after it.
ROUTE_CALL_SIZE = 616
assert ctypes.sizeof(Call) == ROUTE_CALL_SIZE, (
    f"the check's RouteCall mirror is {ctypes.sizeof(Call)} bytes, the operator's layout says "
    f"{ROUTE_CALL_SIZE}"
)


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(intermediate):
    """`Tolerance::dot(intermediate, f32, bf16)`: FC2 applies no activation."""
    return 2.0 * gamma(intermediate, 2.0**-24) * math.sqrt(intermediate) + 2.0**-8


def tiles_of(routes, experts, top):
    """The sort's view of the routes: one list of `(token, slot)` per expert."""
    per_expert = [[] for _ in range(experts)]
    for token, slots in enumerate(routes.tolist()):
        for slot, expert in enumerate(slots):
            if expert >= 0:
                per_expert[expert].append((token, slot))
    return per_expert


def sort_metadata(per_expert, top, tile_rows=TILE_ROWS):
    """The tables `moe_sort` writes, and the live tile count."""
    tile_to_expert, tile_to_limit, permuted_to_expanded = [], [], []
    for expert, members in enumerate(per_expert):
        for start in range(0, len(members), tile_rows):
            chunk = members[start:start + tile_rows]
            tile_to_expert.append(expert)
            tile_to_limit.append(len(chunk))
            permuted_to_expanded.extend(token * top + slot for token, slot in chunk)
            permuted_to_expanded.extend([0] * (tile_rows - len(chunk)))
    if not tile_to_expert:
        tile_to_expert, tile_to_limit = [-1], [0]
        permuted_to_expanded = [0] * tile_rows
    return tile_to_expert, tile_to_limit, permuted_to_expanded, sum(
        1 for limit in tile_to_limit if limit > 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=pathlib.Path, help="the module directory resolve hands a runner")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--top", type=int, default=8)
    options = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "this module is a compiled CuTe DSL native library: it needs the device it was built for"
        )
    library = ctypes.CDLL(str(options.module / "module.so"))
    entry = library.uniserve_route_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    device = torch.device("cuda")
    torch.manual_seed(0)
    tokens, top = options.tokens, options.top
    experts, hidden_size, intermediate = options.experts, options.hidden, options.intermediate

    # As in `gather_gemm`'s check: one expert is emptied by hand (a dropped route would not empty
    # one at this size) and one route dies, so a tile is only partly filled.
    routes = torch.stack([torch.randperm(experts)[:top] for _ in range(tokens)])
    emptied = 11 % experts
    routes[routes == emptied] = (emptied + 1) % experts
    routes[0, 0] = -1
    per_expert = tiles_of(routes, experts, top)
    if not any(not members for members in per_expert):
        raise SystemExit("the case was built with an empty expert in mind and has none")
    tile_expert, tile_limit, map_rows, live_tiles = sort_metadata(per_expert, top)

    permuted_rows = len(map_rows)
    rows = (
        torch.randn(permuted_rows, intermediate, dtype=torch.float32, device=device) * 0.5
    ).to(torch.bfloat16)
    weights = (
        torch.randn(experts, hidden_size, intermediate, dtype=torch.float32, device=device) * 0.05
    ).to(torch.bfloat16)
    expanded = tokens * top
    out = torch.full((expanded, hidden_size), float("nan"), dtype=torch.bfloat16, device=device)
    rows_before, weights_before = rows.clone(), weights.clone()

    tile_to_expert = torch.tensor(tile_expert, dtype=torch.int32, device=device)
    tile_to_limit = torch.tensor(tile_limit, dtype=torch.int32, device=device)
    permuted_to_expanded = torch.tensor(map_rows, dtype=torch.int32, device=device)
    live_table = torch.tensor([live_tiles], dtype=torch.int32, device=device)

    call = Call(
        rows=ctypes.c_void_p(rows.data_ptr()),
        weights=ctypes.c_void_p(weights.data_ptr()),
        out=ctypes.c_void_p(out.data_ptr()),
        tile_to_expert=ctypes.c_void_p(tile_to_expert.data_ptr()),
        tile_to_limit=ctypes.c_void_p(tile_to_limit.data_ptr()),
        permuted_to_expanded=ctypes.c_void_p(permuted_to_expanded.data_ptr()),
        num_non_exiting_tiles=ctypes.c_void_p(live_table.data_ptr()),
        permuted_rows=permuted_rows,
        width=hidden_size,
        intermediate=intermediate,
        experts=experts,
        expanded_rows=expanded,
        dtype=BF16,
        error=b"",
    )
    status = entry(ctypes.byref(call), stream)
    if status != 0:
        message = call.error.decode("utf-8", "replace").rstrip("\x00")
        raise SystemExit(f"the module refused the call with status {status}: {message}")

    if not torch.equal(rows, rows_before) or not torch.equal(weights, weights_before):
        raise SystemExit("the module wrote into its inputs; FC2 scatters only into `out`")

    # The reference, tile by tile: each live tile's rows contracted with its expert's weights.
    expected = torch.full(
        (expanded, hidden_size), float("nan"), dtype=torch.float32, device=device
    )
    covered = torch.zeros(expanded, dtype=torch.bool, device=device)
    for tile, (expert, limit) in enumerate(zip(tile_expert, tile_limit)):
        if limit == 0:
            continue
        first = tile * TILE_ROWS
        entries = torch.tensor(map_rows[first:first + limit], dtype=torch.int64, device=device)
        # `rows [M, I] · weights [E, H, I] -> out [T · top, H]`: the expert's block is
        # [hidden, intermediate], so the contraction runs over its transposed second axis.
        product = rows[first:first + limit].to(torch.float32) @ weights[expert].t().to(
            torch.float32
        )
        expected[entries] = product
        if bool(covered[entries].any()):
            raise SystemExit(
                f"tile {tile} writes an expanded row another tile already wrote, so the metadata "
                f"built here is not a sort and the case would not measure the module"
            )
        covered[entries] = True

    if not bool(covered.all()):
        raise SystemExit(
            f"{int((~covered).sum())} expanded rows are named by no permuted row, so the metadata "
            f"built here is not a sort"
        )

    got = out.to(torch.float32)
    relative = bound(intermediate)
    allowed = relative * expected.abs()
    error = (got - expected).abs()
    # `~(error <= allowed)`: a NaN difference (a route no tile wrote) is not `> allowed` either,
    # so the plain comparison would pass it.
    violations = int((~(error <= allowed)).sum())
    print(
        f"tokens {tokens} top {top} experts {experts} hidden {hidden_size} intermediate "
        f"{intermediate}: {permuted_rows} permuted rows in {live_tiles} tiles, "
        f"max |out - reference| = {float(torch.nan_to_num(error).max()):.6g}, bound rtol {relative:.3e}, "
        f"experts with no routes {sum(1 for m in per_expert if not m)}"
    )
    if violations:
        position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
        print(
            f"{violations} of {error.numel()} products are outside the bound; first at {position}: "
            f"{float(got.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
