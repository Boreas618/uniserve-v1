"""The `gather_gemm` module's FC1 gated product over a sort built here.

The module (resolved at the session's Build Server for the session's device)
is compiled for one descriptor, below, so every case runs its shapes and the
token count, a runtime argument, is the check's. It is called with the
operator's `moe::grouped::GatherCall` (``crates/ops/src/moe/grouped.rs``, the
generated header's ``uniserve_gather_call``), which states the multiprocessors
of the session's device as the operator does, over a valid sort built on the
host (:mod:`uniserve_testing.kernels.gemm.routes`) that holds an expert with
no routes and partly filled tiles.

The contract (``GatherGemmArgs::out``, ``nn::functional::gather_gemm``): `out`
is `[permuted rows, I]` in the sort's permuted order, and the row of route
`r = t · top + k` of expert `e`, the permuted row `expanded_to_permuted[r]`
where the combine reads it, is `act(gate) · up` with `up = W[e, :I] · x` and
`gate = W[e, I:] · x` (each expert's rows hold its up block then its gate
block), `x` row `t` of the hidden states. Those rows are the defined ones: a
tile's rows past its `tile_to_limit` are the sort's padding, which the
operator leaves unspecified (its package stores whole tiles), and a route no
expert holds has no row, so the check reads neither. The reference is that
formula over the routes themselves, in FP32, independent of the tile tables.

The comparison is the operator's bound (``GatherGemm::tolerance``):
`Tolerance::dot(hidden, f32, bf16)`, whose relative term is held against the
output's largest magnitude because a dot product's element can cancel below
its own terms, with the activation envelope's absolute term `2 γ (8² · 2)`
(``tolerance::VALUE_ENVELOPE``).
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
HIDDEN, INTERMEDIATE, EXPERTS, TOP = 2048, 1536, 128, 8
TOKENS = 512
KEY = (
    f"hidden_{HIDDEN}_intermediate_{INTERMEDIATE}_experts_{EXPERTS}_top_{TOP}"
    "_activation_silu_dtype_bf16_tile_128x128_cluster_1x1"
)

#: The expert the case leaves without routes.
EMPTIED = 7


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
        ("multiprocessors", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def test_the_call_mirror_is_the_headers_gather_call():
    """The generated header asserts the size and offsets `GatherCall` has."""
    assert_abi_layout(Call, "uniserve_gather_call")


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(hidden):
    """`GatherGemm`'s tolerance: `dot(hidden, f32, bf16)` with the
    activation's envelope; answers the relative term, which multiplies the
    output's largest magnitude, and the absolute term.
    """  # noqa: D205
    accumulate = 2.0**-24
    bf16_roundoff = 2.0**-8
    relative = (
        2.0 * gamma(hidden, accumulate) * math.sqrt(hidden) + bf16_roundoff
    )
    absolute = 2.0 * gamma(hidden, accumulate) * (8.0 * 8.0 * 2.0)
    return relative, absolute


def reference(hidden, weights, routed):
    """Every route's gated product in FP32, `[tokens · top, intermediate]` in
    route order (`t · top + k`), from the routes alone: zero for a route no
    expert holds.
    """  # noqa: D205
    experts = routed.reshape(-1).to(hidden.device)
    expected = torch.zeros(
        experts.numel(),
        INTERMEDIATE,
        dtype=torch.float32,
        device=hidden.device,
    )
    for expert in range(EXPERTS):
        route = (experts == expert).nonzero().squeeze(1)
        if route.numel() == 0:
            continue
        x = hidden[torch.div(route, TOP, rounding_mode="floor")].float()
        weight = weights[expert].float()
        up = x @ weight[:INTERMEDIATE].t()
        gate = x @ weight[INTERMEDIATE:].t()
        expected[route] = torch.nn.functional.silu(gate) * up
    return expected


@pytest.mark.gpu
@pytest.mark.parametrize("captured", capture.MODES)
def test_the_gated_product_writes_the_live_rows_within_the_operators_bound(
    torch_device, kernel_module, captured
):
    library = ctypes.CDLL(
        str(kernel_module("gather_gemm", "cute_dsl", KEY) / "module.so")
    )
    entry = library.uniserve_gather_gemm
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    torch.manual_seed(0)
    routed = routes.routed(TOKENS, EXPERTS, TOP, EMPTIED)
    per_expert = routes.tiles_of(routed, EXPERTS)
    assert any(not members for members in per_expert), (
        "the case was built with an empty expert in mind and has none"
    )
    hidden = (
        torch.randn(TOKENS, HIDDEN, dtype=torch.float32, device=torch_device)
        * 0.5
    ).to(torch.bfloat16)
    weights = (
        torch.randn(
            EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            dtype=torch.float32,
            device=torch_device,
        )
        * 0.05
    ).to(torch.bfloat16)
    sort = routes.sort_metadata(per_expert, TOKENS, TOP)
    assert any(
        0 < limit - tile * routes.TILE_ROWS < routes.TILE_ROWS
        for tile, limit in enumerate(sort.tile_to_limit)
    ), (
        "no tile is partly filled, so the case does not exercise a tile's "
        "`tile_to_limit`"
    )

    def table(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    # The tables stay alive in these names until after the call.
    tile_to_expert = table(sort.tile_to_expert)
    tile_to_limit = table(sort.tile_to_limit)
    permuted_to_expanded = table(sort.permuted_to_expanded)
    live_table = table([sort.live_tiles])
    permuted_rows = len(sort.permuted_to_expanded)
    out = torch.full(
        (permuted_rows, INTERMEDIATE),
        float("nan"),
        dtype=torch.bfloat16,
        device=torch_device,
    )
    call = Call(
        hidden=hidden.data_ptr(),
        weights=weights.data_ptr(),
        out=out.data_ptr(),
        tile_to_expert=tile_to_expert.data_ptr(),
        tile_to_limit=tile_to_limit.data_ptr(),
        permuted_to_expanded=permuted_to_expanded.data_ptr(),
        num_non_exiting_tiles=live_table.data_ptr(),
        tokens=TOKENS,
        permuted_rows=permuted_rows,
        width=2 * INTERMEDIATE,
        hidden_size=HIDDEN,
        experts=EXPERTS,
        top=TOP,
        dtype=BF16,
        multiprocessors=torch.cuda.get_device_properties(
            torch_device
        ).multi_processor_count,
        error=b"",
    )
    capture.call(entry, call, torch_device, captured)

    # The defined rows, read as the combine reads them: each held route's
    # product at its permuted row. Every other row of `out` is a tile's
    # padding, which the operator leaves unspecified, or no route's.
    held = (routed.reshape(-1) >= 0).to(torch_device)
    assert bool(held.any()), "the routes name no expert: the case is empty"
    route_rows = table(sort.expanded_to_permuted).long()[held]
    got = out[route_rows].float()
    want = reference(hidden, weights, routed)[held]

    relative, absolute = bound(HIDDEN)
    peak = float(want.abs().max())
    allowed = absolute + relative * peak
    error = (got - want).abs()
    # `~(error <= allowed)`: `out` starts as NaN so that a route's row the
    # module left unwritten shows, and `NaN > allowed` is false.
    outside = ~(error <= allowed)
    position = int(outside.reshape(-1).nonzero()[0]) if outside.any() else 0
    assert not bool(outside.any()), (
        f"{int(outside.sum())} of {error.numel()} gated products of "
        f"{int(held.sum())} routes in {sort.live_tiles} tiles are outside "
        f"atol {absolute:.3e} + rtol {relative:.3e} of the largest "
        f"|reference| {peak:.6g}; max |out - reference| = "
        f"{float(torch.nan_to_num(error, nan=math.inf).max()):.6g}; first at "
        f"{position}: {float(got.reshape(-1)[position])} vs "
        f"{float(want.reshape(-1)[position])}"
    )
