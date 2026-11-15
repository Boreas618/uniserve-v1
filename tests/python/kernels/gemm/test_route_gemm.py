"""The `route_gemm` module's FC2 product, scattered to the routes it sorts.

The module (resolved at the session's Build Server for the session's device)
is compiled for one descriptor, below, so every case runs its shapes. It is
called with the operator's `RouteCall` (``crates/ops/src/moe/grouped.rs``)
over a valid sort built on the host
(:mod:`uniserve_testing.kernels.gemm.routes`) that holds an expert with no
routes and a partly filled last tile.

The operator's product is a scatter as well as a multiply::

    out[t · top + k] = W[ids[t, k]] · rows[p]
        for the permuted row p whose map entry names route t · top + k

so every live route's row must be written exactly once (the sort built here
names each once, and a route whose permuted row was lost would leave its row
at the NaN the output starts as), the dead route's row is written by no tile,
and the module writes nothing into its inputs. Rows are compared with
`Tolerance::dot(intermediate, f32, bf16)`, the reduction being the FC2
contraction; there is no activation, so no envelope term.
"""

import ctypes
import math

import pytest
import torch
from uniserve_testing.kernels.gemm import routes
from uniserve_testing.kernels.packages import (
    abi_enumerator,
    assert_abi_layout,
)

pytestmark = pytest.mark.unit

#: `ops::abi::code`'s number of BF16.
BF16 = abi_enumerator("UNISERVE_DTYPE_BF16")

#: The module's descriptor: hidden and intermediate widths, experts, routes
#: per token; and the tokens the case sorts.
HIDDEN, INTERMEDIATE, EXPERTS, TOP = 2048, 768, 128, 8
TOKENS = 512
KEY = (
    f"hidden_{HIDDEN}_intermediate_{INTERMEDIATE}_experts_{EXPERTS}_top_{TOP}"
    "_dtype_bf16_tile_128x128_cluster_1x1"
)

#: The expert the case leaves without routes.
EMPTIED = 11


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
        ("dtype", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def test_the_call_mirror_is_the_headers_route_call():
    """The generated header asserts the size and offsets `RouteCall` has."""
    assert_abi_layout(Call, "uniserve_route_call")


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(intermediate):
    """`Tolerance::dot(intermediate, f32, bf16)`: FC2 applies no activation."""
    return (
        2.0 * gamma(intermediate, 2.0**-24) * math.sqrt(intermediate) + 2.0**-8
    )


@pytest.mark.gpu
def test_every_route_receives_its_product_once_within_the_operators_bound(
    torch_device, kernel_module
):
    library = ctypes.CDLL(
        str(kernel_module("route_gemm", "cute_dsl", KEY) / "module.so")
    )
    entry = library.uniserve_route_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    torch.manual_seed(0)
    routed = routes.routed(TOKENS, EXPERTS, TOP, EMPTIED)
    per_expert = routes.tiles_of(routed, EXPERTS)
    assert any(not members for members in per_expert), (
        "the case was built with an empty expert in mind and has none"
    )
    tile_expert, tile_limit, map_rows, live_tiles = routes.sort_metadata(
        per_expert, TOP
    )
    permuted_rows = len(map_rows)
    rows = (
        torch.randn(
            permuted_rows,
            INTERMEDIATE,
            dtype=torch.float32,
            device=torch_device,
        )
        * 0.5
    ).to(torch.bfloat16)
    weights = (
        torch.randn(
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            dtype=torch.float32,
            device=torch_device,
        )
        * 0.05
    ).to(torch.bfloat16)
    expanded = TOKENS * TOP
    out = torch.full(
        (expanded, HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device=torch_device,
    )
    rows_before, weights_before = rows.clone(), weights.clone()

    def table(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    # The tables stay alive in these names until after the call.
    tile_to_expert = table(tile_expert)
    tile_to_limit = table(tile_limit)
    permuted_to_expanded = table(map_rows)
    live_table = table([live_tiles])
    call = Call(
        rows=rows.data_ptr(),
        weights=weights.data_ptr(),
        out=out.data_ptr(),
        tile_to_expert=tile_to_expert.data_ptr(),
        tile_to_limit=tile_to_limit.data_ptr(),
        permuted_to_expanded=permuted_to_expanded.data_ptr(),
        num_non_exiting_tiles=live_table.data_ptr(),
        permuted_rows=permuted_rows,
        width=HIDDEN,
        intermediate=INTERMEDIATE,
        experts=EXPERTS,
        dtype=BF16,
        error=b"",
    )
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    message = call.error.decode("utf-8", "replace").rstrip("\x00")
    assert status == 0, f"the module refused the call ({status}): {message}"
    assert torch.equal(rows, rows_before) and torch.equal(
        weights, weights_before
    ), "the module wrote into its inputs; FC2 scatters only into `out`"

    # The reference, tile by tile: each live tile's rows multiplied by its
    # expert's `[hidden, intermediate]` block, scattered to the routes its map
    # entries name.
    expected = torch.full(
        (expanded, HIDDEN),
        float("nan"),
        dtype=torch.float32,
        device=torch_device,
    )
    covered = torch.zeros(expanded, dtype=torch.bool, device=torch_device)
    for tile, (expert, limit) in enumerate(zip(tile_expert, tile_limit)):
        if limit == 0:
            continue
        first = tile * routes.TILE_ROWS
        entries = torch.tensor(
            map_rows[first : first + limit], dtype=torch.int64
        ).to(torch_device)
        assert not bool(covered[entries].any()), (
            f"tile {tile} names a route another tile already names, so the "
            "metadata built here is not a sort"
        )
        expected[entries] = (
            rows[first : first + limit].float() @ weights[expert].t().float()
        )
        covered[entries] = True
    live_routes = (routed.reshape(-1) >= 0).to(torch_device)
    assert torch.equal(covered, live_routes), (
        f"{int((covered != live_routes).sum())} routes are named by the "
        "permuted rows other than once each live route, so the metadata built "
        "here is not a sort"
    )

    unwritten = out[~live_routes]
    written = int((~torch.isnan(unwritten)).sum())
    assert written == 0, (
        f"the dead route's row was written: {written} of "
        f"{unwritten.numel()} values"
    )
    got = out[live_routes].float()
    want = expected[live_routes]
    relative = bound(INTERMEDIATE)
    error = (got - want).abs()
    # `~(error <= allowed)`: a NaN difference (a route no tile wrote) is not
    # `> allowed` either.
    outside = ~(error <= relative * want.abs())
    position = int(outside.reshape(-1).nonzero()[0]) if outside.any() else 0
    assert not bool(outside.any()), (
        f"{int(outside.sum())} of {error.numel()} products over "
        f"{permuted_rows} permuted rows in {live_tiles} tiles are outside rtol "
        f"{relative:.3e}; max |out - reference| = "
        f"{float(torch.nan_to_num(error).max()):.6g}; first at {position}: "
        f"{float(got.reshape(-1)[position])} vs "
        f"{float(want.reshape(-1)[position])}"
    )
