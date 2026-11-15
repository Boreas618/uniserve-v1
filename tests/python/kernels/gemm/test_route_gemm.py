"""The `route_gemm` module's FC2 product, stored at the permuted rows it reads.

The module (resolved at the session's Build Server for the session's device)
is compiled for one descriptor, below, so every case runs its shapes. It is
called with the operator's `RouteCall` (``crates/ops/src/moe/grouped.rs``),
which states the multiprocessors of the session's device as the operator
does, over a valid sort built on the host
(:mod:`uniserve_testing.kernels.gemm.routes`) that holds an expert with no
routes and partly filled tiles.

The contract (``RouteGemmArgs::out``, ``nn::functional::route_gemm``): `out`
is `[permuted rows, hidden]`, row `p` of it the product of the sort's
permuted row `p`, unweighted, so the combine (`moe_unpermute`, which reads
the permuted layout) finds route `r = t · top + k` of expert `e` at
`p = expanded_to_permuted[r]`::

    out[p] = W[e] · rows[p]

Those rows are the defined ones: a tile's rows past its `tile_to_limit` are
the sort's padding and a route no expert holds has no row, so no reader
reads either and the check does not. The reference is that formula over the
routes themselves, in FP32, independent of the tile tables, and the module
writes nothing into its inputs.

The comparison is the operator's bound (``RouteGemm::tolerance``):
`Tolerance::dot(intermediate, f32, bf16)`, the reduction being the FC2
contraction, whose relative term is held against the output's largest
magnitude because a dot product's element can cancel below its own terms,
with the absolute term `2 γ (8² · 2)` of the activation envelope
(``tolerance::VALUE_ENVELOPE``) that its input, a gated product, is held to.
"""

import ctypes
import math

import pytest
import torch
from uniserve_testing.kernels import capture
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
        ("multiprocessors", ctypes.c_int64),
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
    """`RouteGemm`'s tolerance: `dot(intermediate, f32, bf16)` with the
    activation's envelope; answers the relative term, which multiplies the
    output's largest magnitude, and the absolute term.
    """  # noqa: D205
    accumulate = 2.0**-24
    bf16_roundoff = 2.0**-8
    relative = (
        2.0 * gamma(intermediate, accumulate) * math.sqrt(intermediate)
        + bf16_roundoff
    )
    absolute = 2.0 * gamma(intermediate, accumulate) * (8.0 * 8.0 * 2.0)
    return relative, absolute


def reference(rows, weights, routed, expanded_to_permuted):
    """Every route's product in FP32, `[tokens · top, hidden]` in route order
    (`t · top + k`), from the routes alone: route `r` of expert `e` multiplies
    `W[e]` by the row the sort placed it at, `rows[expanded_to_permuted[r]]`;
    zero for a route no expert holds.
    """  # noqa: D205
    experts = routed.reshape(-1).to(rows.device)
    expected = torch.zeros(
        experts.numel(), HIDDEN, dtype=torch.float32, device=rows.device
    )
    for expert in range(EXPERTS):
        route = (experts == expert).nonzero().squeeze(1)
        if route.numel() == 0:
            continue
        x = rows[expanded_to_permuted[route]].float()
        expected[route] = x @ weights[expert].float().t()
    return expected


@pytest.mark.gpu
@pytest.mark.parametrize("captured", capture.MODES)
def test_each_route_s_row_holds_its_product_within_the_operators_bound(
    torch_device, kernel_module, captured
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
    sort = routes.sort_metadata(per_expert, TOKENS, TOP)
    assert any(
        0 < limit - tile * routes.TILE_ROWS < routes.TILE_ROWS
        for tile, limit in enumerate(sort.tile_to_limit)
    ), (
        "no tile is partly filled, so the case does not exercise a tile's "
        "`tile_to_limit`"
    )
    permuted_rows = len(sort.permuted_to_expanded)
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
    # The operator's output is the permuted height: one row per row of `rows`.
    out = torch.full(
        (permuted_rows, HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device=torch_device,
    )
    rows_before, weights_before = rows.clone(), weights.clone()

    def table(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    # The tables stay alive in these names until after the call.
    tile_to_expert = table(sort.tile_to_expert)
    tile_to_limit = table(sort.tile_to_limit)
    permuted_to_expanded = table(sort.permuted_to_expanded)
    live_table = table([sort.live_tiles])
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
        multiprocessors=torch.cuda.get_device_properties(
            torch_device
        ).multi_processor_count,
        error=b"",
    )
    capture.call(entry, call, torch_device, captured)
    assert torch.equal(rows, rows_before) and torch.equal(
        weights, weights_before
    ), "the module wrote into its inputs; FC2 stores only into `out`"

    # The defined rows, read as the combine reads them: each held route's
    # product at its permuted row. Every other row of `out` is padding or no
    # route's, and no reader reads it.
    held = (routed.reshape(-1) >= 0).to(torch_device)
    assert bool(held.any()), "the routes name no expert: the case is empty"
    expanded_to_permuted = table(sort.expanded_to_permuted).long()
    got = out[expanded_to_permuted[held]].float()
    want = reference(rows, weights, routed, expanded_to_permuted)[held]

    relative, absolute = bound(INTERMEDIATE)
    peak = float(want.abs().max())
    allowed = absolute + relative * peak
    error = (got - want).abs()
    # `~(error <= allowed)`: `out` starts as NaN so that a route's row the
    # module left unwritten shows, and `NaN > allowed` is false.
    outside = ~(error <= allowed)
    position = int(outside.reshape(-1).nonzero()[0]) if outside.any() else 0
    assert not bool(outside.any()), (
        f"{int(outside.sum())} of {error.numel()} products of "
        f"{int(held.sum())} routes over {permuted_rows} permuted rows in "
        f"{sort.live_tiles} tiles are outside atol {absolute:.3e} + rtol "
        f"{relative:.3e} of the largest |reference| {peak:.6g}; max "
        f"|out - reference| = "
        f"{float(torch.nan_to_num(error, nan=math.inf).max()):.6g}; first at "
        f"{position}: {float(got.reshape(-1)[position])} vs "
        f"{float(want.reshape(-1)[position])}"
    )
