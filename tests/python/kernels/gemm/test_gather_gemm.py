"""The `gather_gemm` module's FC1 gated product over a sort built here.

The module (resolved at the session's Build Server for the session's device)
is compiled for one descriptor, below, so every case runs its shapes and the
token count, a runtime argument, is the check's. It is called with the
operator's `moe::grouped::GatherCall` (``crates/ops/src/moe/grouped.rs``, the
generated header's ``uniserve_gather_call``), which states the multiprocessors
of the session's device as the operator does, over a valid sort built on the
host (:mod:`uniserve_testing.kernels.gemm.routes`) that holds an expert with
no routes and a partly filled last tile.

The reference is the operator's text: for a live tile's row `p` with map entry
`r`, `out[p] = act(gate) · up` where `up = W[e, :I] · x` and
`gate = W[e, I:] · x` (each expert's rows hold its up block then its gate
block) and `x` is row `r / top` of the hidden states. The comparison is the
operator's bound, `Tolerance::dot(hidden, f32, bf16)` carried through the
activation, whose absolute term is `2 γ (8² · 2)` (`ACTIVATION_ENVELOPE`).

Rows the metadata does not name are the padding the sort added so the tiles
are whole, and the operator writes only the rows its map holds: they must keep
the NaN the output starts as.
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
    activation's envelope; answers the relative and the absolute term.
    """  # noqa: D205
    accumulate = 2.0**-24
    bf16_roundoff = 2.0**-8
    relative = (
        2.0 * gamma(hidden, accumulate) * math.sqrt(hidden) + bf16_roundoff
    )
    absolute = 2.0 * gamma(hidden, accumulate) * (8.0 * 8.0 * 2.0)
    return relative, absolute


def reference(hidden, weights, tile_expert, tile_limit, map_rows):
    """The gated product the metadata names, in FP32, tile by tile."""
    expected = torch.zeros(
        len(map_rows), INTERMEDIATE, dtype=torch.float32, device=hidden.device
    )
    for tile, (expert, limit) in enumerate(zip(tile_expert, tile_limit)):
        if limit == 0:
            continue
        first = tile * routes.TILE_ROWS
        expanded = torch.tensor(
            map_rows[first : first + limit], dtype=torch.int64
        ).to(hidden.device)
        x = hidden[torch.div(expanded, TOP, rounding_mode="floor")].float()
        weight = weights[expert].float()
        up = x @ weight[:INTERMEDIATE].t()
        gate = x @ weight[INTERMEDIATE:].t()
        expected[first : first + limit] = torch.nn.functional.silu(gate) * up
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
    per_expert = routes.tiles_of(
        routes.routed(TOKENS, EXPERTS, TOP, EMPTIED), EXPERTS
    )
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
    tile_expert, tile_limit, map_rows, live_tiles = routes.sort_metadata(
        per_expert, TOP
    )
    assert any(0 < limit < routes.TILE_ROWS for limit in tile_limit), (
        "no tile is partly filled, so the case does not exercise a tile's "
        "`tile_to_limit`"
    )

    def table(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    # The tables stay alive in these names until after the call.
    tile_to_expert = table(tile_expert)
    tile_to_limit = table(tile_limit)
    permuted_to_expanded = table(map_rows)
    live_table = table([live_tiles])
    permuted_rows = len(map_rows)
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

    # The rows the metadata names are a tile's first `limit` rows; the rest of
    # each tile is the sort's padding, which every tile carries at its own end.
    live = torch.zeros(permuted_rows, dtype=torch.bool, device=torch_device)
    for tile, limit in enumerate(tile_limit):
        start = tile * routes.TILE_ROWS
        live[start : start + limit] = True
    assert bool(live.any()), "the metadata names no rows: the case is empty"
    padding = out[~live]
    written = int((~torch.isnan(padding)).sum())
    assert written == 0, (
        f"a padding row was written: {written} of {padding.numel()} values "
        "inside the tiles past their `tile_to_limit`"
    )

    expected = reference(hidden, weights, tile_expert, tile_limit, map_rows)
    got = out[live].float()
    want = expected[live]
    relative, absolute = bound(HIDDEN)
    error = (got - want).abs()
    # `~(error <= allowed)`: `out` starts as NaN so that an unwritten row
    # shows, and `NaN > allowed` is false.
    outside = ~(error <= absolute + relative * want.abs())
    position = int(outside.reshape(-1).nonzero()[0]) if outside.any() else 0
    assert not bool(outside.any()), (
        f"{int(outside.sum())} of {error.numel()} gated products of "
        f"{int(live.sum())} live rows in {live_tiles} tiles are outside rtol "
        f"{relative:.3e} atol {absolute:.3e}; max |out - reference| = "
        f"{float(torch.nan_to_num(error).max()):.6g}; first at {position}: "
        f"{float(got.reshape(-1)[position])} vs "
        f"{float(want.reshape(-1)[position])}"
    )
