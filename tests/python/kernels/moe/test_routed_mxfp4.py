"""The MXFP4 routed products against the operators' definitions.

`gather_gemm_mxfp4` and `route_gemm_mxfp4` are checked at
DeepSeek-V4.1-Flash's expert block.

The modules (resolved at the session's Build Server for the session's device)
are called with the operators' call structures (``crates/ops/src/moe/
mxfp4.rs``) over a valid sort built on the host
(:mod:`uniserve_testing.kernels.gemm.routes`) with an expert of no routes and
partly filled tiles, and weights in the arrangement the operators declare:
each matrix's packed E2M1 rows (the gated stack's halves interleaved, up
first) shuffled inside 32-row blocks, its UE8M0 scales moved with their rows
and swizzled into the 128x4 atom.

The first product's definition (``nn::functional::gather_gemm_mxfp4``)::

    x̂[t]   = mxfp8(hidden[t])
    g[p]   = silu(min(W_gate,e · x̂[t], L)) · clamp(W_up,e · x̂[t], ±L)
    out[p] = mxfp8_ocp(g[p])    E4M3 codes, scales in the 128x4 atom

is checked by decoding the module's codes and scales and holding them to the
exact gated product computed in FP64 from the decoded operands. The gated rows
are encoded by the OCP rule (``ops::mx::ScaleRule::Ocp``): each block's scale
is ``2^(⌊log₂ amax⌋ − 8)`` of its exact largest magnitude, but where the
contraction's FP32 bound may carry that magnitude across a power of two, and
its quotients above 448 saturate; under that scale a code errs by half a step
(2⁻⁴ of a normal value, half a subnormal step of the scale below), and the
contraction by the operator's FP32 bound. The second's (``route_gemm_mxfp4``)::

    out[p] = W_down,e · ĝ[p]

reads the first module's output, so it is held to the FP64 product of those
very codes and scales under ``Tolerance::dot(intermediate, f32, bf16)`` of
the output's largest magnitude, with the gated envelope's absolute term.
"""

import ctypes
import math
import os

import pytest
import torch
from uniserve_testing.kernels.gemm import mx, routes
from uniserve_testing.kernels.packages import assert_abi_layout

pytestmark = pytest.mark.unit

#: DeepSeek-V4.1-Flash's routed block: hidden, expert width, experts, routes
#: per token; and its SwiGLU clamp.
HIDDEN, INTERMEDIATE, EXPERTS, TOP = 5120, 2304, 384, 6
LIMIT = 10.0
WIDTHS = (
    f"hidden_{HIDDEN}_intermediate_{INTERMEDIATE}_experts_{EXPERTS}_top_{TOP}"
    "_tile_128x128_cluster_2x1"
)

#: The expert the cases leave without routes.
EMPTIED = 11

#: The shuffle's row block: 32 rows for a 128-row epilogue tile, in four
#: groups of eight.
SHUFFLE_ROWS, SHUFFLE_GROUPS = 32, 4


class Load(ctypes.Structure):
    """The load entry's call (`uniserve_trtllm/cubins.h`, `ops::LoadCall`)."""

    _fields_ = [
        ("directory", ctypes.c_char_p),
        ("error", ctypes.c_char * 512),
    ]


class GatherCall(ctypes.Structure):
    """`moe::mxfp4::GatherMxfp4Call`, field for field."""

    _fields_ = [
        ("hidden", ctypes.c_void_p),
        ("hidden_scale", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("limit", ctypes.c_void_p),
        ("tile_to_expert", ctypes.c_void_p),
        ("tile_to_limit", ctypes.c_void_p),
        ("permuted_to_token", ctypes.c_void_p),
        ("num_non_exiting_tiles", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("out_scale", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_int64),
        ("tokens", ctypes.c_int64),
        ("routes", ctypes.c_int64),
        ("hidden_size", ctypes.c_int64),
        ("intermediate", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("tile_m", ctypes.c_int64),
        ("tile_n", ctypes.c_int64),
        ("cluster_m", ctypes.c_int64),
        ("cluster_n", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


class RouteCall(ctypes.Structure):
    """`moe::mxfp4::RouteMxfp4Call`, field for field."""

    _fields_ = [
        ("hidden", ctypes.c_void_p),
        ("hidden_scale", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("tile_to_expert", ctypes.c_void_p),
        ("tile_to_limit", ctypes.c_void_p),
        ("permuted_to_expanded", ctypes.c_void_p),
        ("num_non_exiting_tiles", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_int64),
        ("routes", ctypes.c_int64),
        ("hidden_size", ctypes.c_int64),
        ("intermediate", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("tile_m", ctypes.c_int64),
        ("tile_n", ctypes.c_int64),
        ("cluster_m", ctypes.c_int64),
        ("cluster_n", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


@pytest.mark.parametrize(
    ("mirror", "structure"),
    [
        (GatherCall, "uniserve_gather_gemm_mxfp4_call"),
        (RouteCall, "uniserve_route_gemm_mxfp4_call"),
    ],
)
def test_the_call_mirrors_are_the_headers_structures(mirror, structure):
    """The generated headers assert the size and offsets each call has."""
    assert_abi_layout(mirror, structure)


def shuffled(rows, interleaved):
    """``ValueLayout::destination`` of every row of a matrix of ``rows``: the
    two halves interleaved first when ``interleaved``, then each 32-row block
    shuffled, row ``r`` of a block to ``(r % 4) · 8 + r / 4``.
    """  # noqa: D205
    row = torch.arange(rows)
    if interleaved:
        half = rows // 2
        row = torch.where(row < half, 2 * row, 2 * (row - half) + 1)
    within = row % SHUFFLE_ROWS
    lanes = SHUFFLE_ROWS // 8
    return (
        (row // SHUFFLE_ROWS) * SHUFFLE_ROWS
        + (within % lanes) * 8
        + within // lanes
    )


def arranged(packed, scales, interleaved):
    """A stack's packed rows and linear scales ``[experts, rows, ...]`` in the
    products' arrangement: each expert's rows moved to their destinations, its
    scales with them and swizzled into the 128x4 atom.
    """  # noqa: D205
    experts, rows, _ = packed.shape
    destination = shuffled(rows, interleaved).to(packed.device)
    moved = torch.empty_like(packed)
    moved[:, destination] = packed
    moved_scales = torch.empty_like(scales)
    moved_scales[:, destination] = scales
    blocks = scales.shape[2]
    padded = (-(-rows // 128) * 128, -(-blocks // 4) * 4)
    offsets = mx.swizzled_offsets(rows, blocks).reshape(-1).to(packed.device)
    table = torch.zeros(
        experts, padded[0] * padded[1], dtype=torch.uint8, device=packed.device
    )
    table[:, offsets] = moved_scales.reshape(experts, -1)
    return moved, table.reshape(experts, *padded)


def stack(rows, columns, seed, device):
    """``EXPERTS`` MXFP4 matrices of ``rows`` by ``columns`` on ``device``: the
    packed codes and linear scales a checkpoint holds, with scales a factor of
    16 apart. They are drawn on the device: the stacks are gigabytes.
    """  # noqa: D205
    generator = torch.Generator(device=device).manual_seed(seed)
    packed = torch.randint(
        0,
        256,
        (EXPERTS, rows, columns // 2),
        generator=generator,
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.randint(
        118,
        122,
        (EXPERTS, rows, columns // 32),
        generator=generator,
        dtype=torch.uint8,
        device=device,
    )
    return packed, scales


def opened(path):
    """The shared object at ``path``, opened as the engine opens a module.

    ``platform::NativeLibrary`` opens it ``RTLD_LAZY | RTLD_LOCAL``.

    The runner's translation unit names FlashInfer's routing, activation and
    finalize launches, which the entry never calls and the module does not
    link; a lazy open leaves them unresolved, as the engine's does, where
    ``ctypes.CDLL``'s own ``dlopen`` binds every symbol at once.
    """
    libc = ctypes.CDLL(None)
    libc.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.dlopen.restype = ctypes.c_void_p
    libc.dlerror.restype = ctypes.c_char_p
    handle = libc.dlopen(str(path).encode(), os.RTLD_LAZY | os.RTLD_LOCAL)
    assert handle, libc.dlerror().decode(errors="replace")
    return ctypes.CDLL(str(path), handle=handle)


def module(kernel_module, operator, entry, call):
    """A module's C entry, after its load entry read its cubins."""
    directory = kernel_module(operator, "trtllm", WIDTHS)
    library = opened(directory / "module.so")
    load = library.uniserve_trtllm_load
    load.argtypes = [ctypes.POINTER(Load), ctypes.c_void_p]
    load.restype = ctypes.c_int32
    loaded = Load(str(directory).encode(), b"")
    assert load(ctypes.byref(loaded), None) == 0, loaded.error.decode(
        errors="replace"
    )
    function = getattr(library, entry)
    function.argtypes = [ctypes.POINTER(call), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


def workspace_bytes(routes_count, width):
    """``moe::grouped::packed_workspace`` at tile 128x128, cluster 2x1."""
    kibibytes = lambda bytes_: -(-bytes_ // 1024) * 1024  # noqa: E731
    token_tile = 128 * 2
    padded = -(-max(routes_count, 1) // token_tile) * token_tile
    row_maxima = width * padded // 128 * 4
    bars = (padded // token_tile) * -(-width // 128) // 2 * 4
    return 16 + kibibytes(row_maxima) + kibibytes(bars) + 4


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [7, 300], ids=["a decode batch", "a prompt"])
def test_the_routed_products_are_the_operators_definitions(
    tokens, kernel_module, torch_device
):
    gather = module(
        kernel_module,
        "gather_gemm_mxfp4",
        "uniserve_gather_gemm_mxfp4",
        GatherCall,
    )
    route = module(
        kernel_module,
        "route_gemm_mxfp4",
        "uniserve_route_gemm_mxfp4",
        RouteCall,
    )

    torch.manual_seed(tokens)
    routed = routes.routed(tokens, EXPERTS, TOP, EMPTIED)
    sort = routes.sort_metadata(routes.tiles_of(routed, EXPERTS), tokens, TOP)
    permuted = len(sort.permuted_to_expanded)
    to_token = [
        route // TOP if route >= 0 else -1
        for route in sort.permuted_to_expanded
    ]

    # The operands: the hidden rows' encoding, the stacks as a checkpoint
    # holds them and in the products' arrangement.
    hidden = (torch.randn(tokens, HIDDEN) * 2.0).to(torch.bfloat16)
    hidden_codes, hidden_scales = mx.encode_mxfp8(hidden)
    up_gate_codes, up_gate_scales = stack(
        2 * INTERMEDIATE, HIDDEN, 1, torch_device
    )
    down_codes, down_scales = stack(HIDDEN, INTERMEDIATE, 2, torch_device)
    up_gate = arranged(up_gate_codes, up_gate_scales, True)
    down = arranged(down_codes, down_scales, False)

    def device(value, dtype=None):
        return (
            (value if dtype is None else value.to(dtype))
            .contiguous()
            .to(torch_device)
        )

    def table(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    tile_to_expert = table(sort.tile_to_expert)
    tile_to_limit = table(sort.tile_to_limit)
    permuted_to_token = table(to_token)
    permuted_to_expanded = table(sort.permuted_to_expanded)
    live = table([sort.live_tiles])
    gated_codes = torch.zeros(
        permuted, INTERMEDIATE, dtype=torch.uint8, device=torch_device
    )
    gated_shape = (-(-permuted // 128) * 128, -(-(INTERMEDIATE // 32) // 4) * 4)
    gated_scales = torch.zeros(
        gated_shape, dtype=torch.uint8, device=torch_device
    )
    out = torch.full(
        (permuted, HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device=torch_device,
    )
    first_scratch = torch.zeros(
        workspace_bytes(permuted, 2 * INTERMEDIATE),
        dtype=torch.uint8,
        device=torch_device,
    )
    second_scratch = torch.zeros(
        workspace_bytes(permuted, HIDDEN),
        dtype=torch.uint8,
        device=torch_device,
    )
    operands = {
        "hidden": device(hidden_codes),
        "hidden_scale": device(hidden_scales),
        "up_gate": up_gate[0],
        "up_gate_scales": up_gate[1],
        "down": down[0],
        "down_scales": down[1],
        "limit": torch.full(
            (EXPERTS,), LIMIT, dtype=torch.float32, device=torch_device
        ),
    }
    shape = {
        "hidden_size": HIDDEN,
        "intermediate": INTERMEDIATE,
        "experts": EXPERTS,
        "top": TOP,
        "tile_m": 128,
        "tile_n": 128,
        "cluster_m": 2,
        "cluster_n": 1,
    }
    stream = ctypes.c_void_p(
        torch.cuda.current_stream(torch_device).cuda_stream
    )

    first = GatherCall(
        hidden=operands["hidden"].data_ptr(),
        hidden_scale=operands["hidden_scale"].data_ptr(),
        weights=operands["up_gate"].data_ptr(),
        scales=operands["up_gate_scales"].data_ptr(),
        limit=operands["limit"].data_ptr(),
        tile_to_expert=tile_to_expert.data_ptr(),
        tile_to_limit=tile_to_limit.data_ptr(),
        permuted_to_token=permuted_to_token.data_ptr(),
        num_non_exiting_tiles=live.data_ptr(),
        out=gated_codes.data_ptr(),
        out_scale=gated_scales.data_ptr(),
        workspace=first_scratch.data_ptr(),
        workspace_bytes=first_scratch.numel(),
        tokens=tokens,
        routes=permuted,
        **shape,
    )
    assert gather(ctypes.byref(first), stream) == 0, first.error.decode(
        errors="replace"
    )
    second = RouteCall(
        hidden=gated_codes.data_ptr(),
        hidden_scale=gated_scales.data_ptr(),
        weights=operands["down"].data_ptr(),
        scales=operands["down_scales"].data_ptr(),
        tile_to_expert=tile_to_expert.data_ptr(),
        tile_to_limit=tile_to_limit.data_ptr(),
        permuted_to_expanded=permuted_to_expanded.data_ptr(),
        num_non_exiting_tiles=live.data_ptr(),
        out=out.data_ptr(),
        workspace=second_scratch.data_ptr(),
        workspace_bytes=second_scratch.numel(),
        routes=permuted,
        **shape,
    )
    assert route(ctypes.byref(second), stream) == 0, second.error.decode(
        errors="replace"
    )
    torch.cuda.synchronize(torch_device)

    # The live permuted rows, grouped by expert: each a route of a token.
    by_expert = {}
    for tile, expert in enumerate(sort.tile_to_expert):
        for p in range(
            tile * 128, min((tile + 1) * 128, sort.tile_to_limit[tile])
        ):
            by_expert.setdefault(expert, []).append(p)
    x = mx.decode_mxfp8(hidden_codes, hidden_scales).to(torch_device)
    offsets = mx.swizzled_offsets(permuted, INTERMEDIATE // 32).reshape(-1)
    gated_linear = (
        gated_scales.cpu()
        .reshape(-1)[offsets]
        .reshape(permuted, INTERMEDIATE // 32)
    )
    decoded = mx.decode_mxfp8(
        gated_codes.cpu().view(torch.float8_e4m3fn), gated_linear
    ).to(torch_device)
    units = (
        mx.scale_values(gated_linear)
        .double()
        .repeat_interleave(32, dim=1)
        .to(torch_device)
    )
    gamma = HIDDEN * 2.0**-24 / (1 - HIDDEN * 2.0**-24)

    # 1. The gated rows, decoded, against the exact gated product of each
    # expert's routes, in FP64 on the device one expert at a time.
    outside, worst, rescaled = 0, 0.0, 0
    for expert, members in by_expert.items():
        weight = mx.decode_mxfp4(
            up_gate_codes[expert], up_gate_scales[expert], HIDDEN
        )
        tokens_of = torch.tensor(
            [to_token[p] for p in members], device=torch_device
        )
        inputs = x[tokens_of]
        projected = inputs @ weight.T
        terms = inputs.abs() @ weight.abs().T
        up, gate = projected[:, :INTERMEDIATE], projected[:, INTERMEDIATE:]
        clamped_gate = gate.clamp(max=LIMIT)
        activated = clamped_gate * torch.sigmoid(clamped_gate)
        clamped_up = up.clamp(-LIMIT, LIMIT)
        exact = activated * clamped_up
        # The FP32 association of each half's contraction, carried through the
        # SwiGLU (silu' stays below 1.1).
        rows = torch.tensor(members, device=torch_device)
        block_scale = units[rows]
        accumulation = (
            2
            * gamma
            * (
                activated.abs() * terms[:, :INTERMEDIATE]
                + 1.1 * clamped_up.abs() * terms[:, INTERMEDIATE:]
            )
        )
        # The module's scales are the OCP rule's of the exact rows, but where
        # the accumulation may move a block's largest magnitude across a power
        # of two.
        blocks = exact.reshape(len(members), -1, 32).abs()
        amax, at = blocks.max(dim=-1)
        slack = accumulation.reshape(len(members), -1, 32).gather(
            2, at.unsqueeze(-1)
        )[..., 0]
        crossing = torch.exp2(torch.log2(amax).round())
        wanted_scale = mx.scale_values(mx.ocp_exponents(exact.float())).double()
        written = block_scale[:, ::32]
        rescaled += int(
            (
                (written != wanted_scale) & ((amax - crossing).abs() > slack)
            ).sum()
        )
        # Under the written scale the quotients above 448 saturate, and a code
        # is within half a step of its quotient: 2⁻⁴ of a normal value, 2⁻¹⁰ of
        # the scale for a subnormal one.
        saturated = torch.maximum(
            torch.minimum(exact, 448.0 * block_scale), -448.0 * block_scale
        )
        bound = (
            2.0**-4 * saturated.abs() + 2.0**-10 * block_scale + accumulation
        )
        error = (decoded[rows] - saturated).abs()
        outside += int((~(error <= bound)).sum())
        worst = max(
            worst, float(torch.nan_to_num(error / bound.clamp(min=1e-30)).max())
        )
    assert rescaled == 0, (
        f"{tokens} tokens: {rescaled} gated blocks under another scale than "
        "the OCP rule's"
    )
    assert outside == 0, (
        f"{tokens} tokens: {outside} gated elements outside the encoding's "
        f"bound; worst at {worst:.3f} of it"
    )

    # 2. The second product of those very rows.
    got = out.double()
    gamma = INTERMEDIATE * 2.0**-24 / (1 - INTERMEDIATE * 2.0**-24)
    rtol = 2 * gamma * math.sqrt(INTERMEDIATE) + 2.0**-8
    atol = 2 * gamma * (8.0 * 8.0 * 2.0)
    wanted, held = [], []
    for expert, members in by_expert.items():
        weight = mx.decode_mxfp4(
            down_codes[expert], down_scales[expert], INTERMEDIATE
        )
        rows = torch.tensor(members, device=torch_device)
        wanted.append(decoded[rows] @ weight.T)
        held.append(got[rows])
    want, got = torch.cat(wanted), torch.cat(held)
    peak = float(want.abs().max())
    error = (got - want).abs()
    outside = ~(error <= atol + rtol * peak)
    assert not bool(outside.any()), (
        f"{tokens} tokens: {int(outside.sum())} of {error.numel()} products "
        f"outside atol {atol:.3e} + rtol {rtol:.3e} of the largest "
        f"|reference| {peak:.4g}; max error "
        f"{float(torch.nan_to_num(error, nan=math.inf).max()):.4g}"
    )
