"""The fused MXFP4 expert exchange (the ``deepgemm`` package) over four devices.

The module of the descriptor below (resolved at the session's Build Server for
the session's device) is driven by four ranks, one per device, each on a thread
of the check's process (:mod:`uniserve_testing.kernels.moe.exchange`), at
DeepSeek-V4.1-Flash's expert block: 384 MXFP4 experts of width 2304 over rows
of 5120, 96 a rank, six routes a token, one MXFP8 shared expert, the SwiGLU
clamped at 10. The weights are random codes under scales that clamp about a
fifth of the gate values, in the arrangement the operator declares
(``ops::moe::exchange::Descriptor::MXFP4_UP_GATE_VALUES``,
``MXFP4_DOWN_VALUES``, ``MXFP4_SCALES``): each gated stack's gate and up rows
interleaved by eight, gate first, and every weight's UE8M0 scales as packed
MN-major words with each 128-row group's rows in the order of the tensor-core
scale copy.

Each rank's rows are held to the operator's formula (``ops::moe::exchange``'s
module documentation), computed here from the decoded operands in FP64 with
the roundings the formula states::

    x̂     = mxfp8(x)                       mxfp8_quantize's covering rule
    g, u  = bf16(W_gate,e x̂), bf16(W_up,e x̂)
    a     = silu(min(g, L)) · clamp(u, ±L) · w     FP32; w = 1 for shared
    â     = mxfp8_exact(a)                 least 2^s, amax ≤ 448 · 2^s
    o_e   = bf16(W_down,e â)
    y     = bf16(Σ_routes o_e + Σ_shared o_s)

within ``ExpertExchange::tolerance`` of the fused mode, relative to each row's
largest magnitude (a row is a sum that cancels): its products' contractions,
the gated rows' codes one E4M3 step apart, and the sum's rounding. The trials
meet what served steps meet: one row a rank, a live count below the extent
(whose rows must come back zero) beside a rank with no rows, and steps whose
capacity picks each of DeepGEMM's tile heights up to the transport's 4096 rows,
the larger ones checked on a sample of rows.

It needs four CUDA devices of at least 16 GiB free each. The call structures'
mirrors are held to the operator crate's generated header, which needs none.
"""

import ctypes
import math
import struct

import pytest
import torch
from uniserve_testing.kernels.gemm import mx
from uniserve_testing.kernels.moe import exchange
from uniserve_testing.kernels.packages import (
    abi_enumerator,
    assert_abi_layout,
)
from uniserve_testing.kernels.tensor.operands import UNIT, gamma

pytestmark = pytest.mark.unit

#: The element type and the launch direction, as the generated headers number
#: them.
BF16 = abi_enumerator("UNISERVE_DTYPE_BF16")
FUSED = abi_enumerator("UNISERVE_EXCHANGE_FUSED")
MXFP4 = abi_enumerator("UNISERVE_EXCHANGE_WEIGHTS_MXFP4")

#: DeepSeek-V4.1-Flash's expert block over four ranks, and the transport's
#: rows a rank.
RANKS, HIDDEN, INTERMEDIATE, EXPERTS, TOP, SHARED = 4, 5120, 2304, 384, 6, 1
LIMIT = 10.0
LOCAL = EXPERTS // RANKS
CAPACITY = 4096

#: The gate and up rows a block of the interleaved stack holds of each.
INTERLEAVE = 8

#: The rows of a tensor-core scale copy, whose lanes are stored side by side.
UTCCP_ROWS = 128

#: E4M3's unit roundoff: half its step of three mantissa bits.
E4M3_UNIT = 2.0**-4

#: The trials: the rows each rank sends, the live count among them and the
#: step's capacity, which picks the tile height (16, 16, 32, 128 and 192 rows),
#: and how many rows of each rank are checked (every row when ``None``). The
#: second trial's last rank sends nothing.
TRIALS = (
    (1, 1, 1, None),
    (8, 5, 8, None),
    (256, 256, 256, None),
    (1024, 1000, 1024, 96),
    (CAPACITY, CAPACITY, CAPACITY, 96),
)

#: The scales of the random weights, as UE8M0 bytes: the routed experts'
#: first product spans the clamp, their second and the shared expert's stay
#: within BF16's comfortable range.
UP_GATE_EXPONENTS, DOWN_EXPONENTS, SHARED_EXPONENTS = (
    (120, 124),
    (118, 122),
    (118, 122),
)


def limit_bits():
    """The clamp's FP32 bit pattern, as a key prints it."""
    return struct.unpack("<I", struct.pack("<f", LIMIT))[0]


def key(multiprocessors):
    """The descriptor of the module, for a device of ``multiprocessors``."""
    return (
        f"mode_fused_hidden_{HIDDEN}_intermediate_{INTERMEDIATE}_top_{TOP}_"
        f"experts_{EXPERTS}_max_tokens_{CAPACITY}_ranks_{RANKS}_attention_0_"
        f"multiprocessors_{multiprocessors}_dtype_bf16_weights_mxfp4_"
        f"shared_{SHARED}_limit_{limit_bits():#x}"
    )


def fused_tolerance():
    """``ExpertExchange::tolerance`` of a fused launch: ``(atol, rtol)``.

    Two implementations differ by the first product's contraction rounded to
    BF16, a gated row's code one E4M3 step apart (``2 · 2⁻⁴`` of it), the
    second product's contraction rounded to BF16, and the FP32 sum of a
    token's routes and shared experts rounded once; each contraction under
    ``Tolerance::dot``'s reading, relative to the output's largest magnitude.
    """

    def dot(terms):
        return 2.0 * gamma(terms, UNIT["f32"]) * math.sqrt(terms) + UNIT["bf16"]

    rtol = dot(HIDDEN) + 2.0 * E4M3_UNIT + dot(INTERMEDIATE) + dot(TOP + SHARED)
    return mx.UNDERFLOW["bf16"], rtol


@pytest.mark.parametrize("structure", sorted(exchange.MIRRORS))
def test_a_call_mirror_is_its_generated_header_structure(structure):
    """The generated header asserts the size and offsets the mirror has."""
    assert_abi_layout(exchange.MIRRORS[structure], structure)


# ---- The weights -------------------------------------------------------------


def random_mxfp4(generator, experts, rows, columns, exponents, device):
    """Random packed E2M1 codes ``[experts, rows, columns / 2]`` and their
    UE8M0 scale bytes ``[experts, rows, columns / 32]``.
    """  # noqa: D205
    codes = torch.randint(
        0,
        256,
        (experts, rows, columns // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    scales = torch.randint(
        exponents[0],
        exponents[1],
        (experts, rows, columns // mx.BLOCK),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    return codes, scales


def random_mxfp8(generator, rows, columns, exponents, device):
    """Random E4M3 codes ``[rows, columns]`` (finite, below 64 in magnitude)
    and their UE8M0 scale bytes ``[rows, columns / 32]``.
    """  # noqa: D205
    values = (
        torch.randn((rows, columns), device=device, generator=generator) * 16.0
    )
    codes = values.clamp(-64.0, 64.0).to(torch.float8_e4m3fn)
    scales = torch.randint(
        exponents[0],
        exponents[1],
        (rows, columns // mx.BLOCK),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    return codes, scales


def interleaved(gate, up):
    """``[..., rows, c]`` gate and up halves as the gated stack's rows: each
    block of sixteen rows the gate's eight, then the same eight of up.
    """  # noqa: D205
    *batch, rows, columns = gate.shape
    shape = (*batch, rows // INTERLEAVE, INTERLEAVE, columns)
    stacked = torch.stack([gate.reshape(shape), up.reshape(shape)], dim=-3)
    return stacked.reshape(*batch, 2 * rows, columns).contiguous()


def utccp_scales(scales):
    """UE8M0 bytes ``[..., rows, blocks]`` as the packed MN-major words
    ``[..., blocks / 4, rows]``, each 128-row group's row ``32i + j`` stored at
    ``4j + i`` (``ops::mx::ScaleLayout::PackedMnMajorUtccp``).
    """  # noqa: D205
    *batch, rows, blocks = scales.shape
    words = scales.contiguous().view(torch.int32)  # [..., rows, blocks / 4]
    position = torch.arange(rows, device=scales.device)
    within = position % UTCCP_ROWS
    stored_from = position - within + (within % 4) * 32 + within // 4
    return words[..., stored_from, :].transpose(-1, -2).contiguous()


class Weights:
    """One rank's experts, as the reference decodes them and as the module
    reads them.
    """  # noqa: D205

    def __init__(self, rank, device):
        """Rank ``rank``'s random experts on ``device``."""
        generator = torch.Generator(device=device).manual_seed(7000 + rank)
        self.gate, self.gate_scales = random_mxfp4(
            generator, LOCAL, INTERMEDIATE, HIDDEN, UP_GATE_EXPONENTS, device
        )
        self.up, self.up_scales = random_mxfp4(
            generator, LOCAL, INTERMEDIATE, HIDDEN, UP_GATE_EXPONENTS, device
        )
        self.down, self.down_scales = random_mxfp4(
            generator, LOCAL, HIDDEN, INTERMEDIATE, DOWN_EXPONENTS, device
        )
        shared = SHARED * INTERMEDIATE
        self.shared_gate, self.shared_gate_scales = random_mxfp8(
            generator, shared, HIDDEN, SHARED_EXPONENTS, device
        )
        self.shared_up, self.shared_up_scales = random_mxfp8(
            generator, shared, HIDDEN, SHARED_EXPONENTS, device
        )
        self.shared_down, self.shared_down_scales = random_mxfp8(
            generator, HIDDEN, shared, SHARED_EXPONENTS, device
        )

        # The arrangement the module reads.
        self.arranged = [
            interleaved(self.gate, self.up),
            utccp_scales(interleaved(self.gate_scales, self.up_scales)),
            self.down,
            utccp_scales(self.down_scales),
            interleaved(
                self.shared_gate.view(torch.uint8),
                self.shared_up.view(torch.uint8),
            ),
            utccp_scales(
                interleaved(self.shared_gate_scales, self.shared_up_scales)
            ),
            self.shared_down.view(torch.uint8).contiguous(),
            utccp_scales(self.shared_down_scales),
        ]

    def layer(self):
        """The module's ``uniserve_fused_layer`` over the arranged weights."""
        up_gate, up_gate_scales, down, down_scales, sg, sgs, sd, sds = (
            self.arranged
        )
        return exchange.FusedLayer(
            up_gate=up_gate.data_ptr(),
            up_gate_scales=up_gate_scales.data_ptr(),
            down=down.data_ptr(),
            down_scales=down_scales.data_ptr(),
            up_gate_scale_bytes=up_gate_scales[0].numel() * 4,
            down_scale_bytes=down_scales[0].numel() * 4,
            shared_up_gate=sg.data_ptr(),
            shared_up_gate_scales=sgs.data_ptr(),
            shared_down=sd.data_ptr(),
            shared_down_scales=sds.data_ptr(),
        )


# ---- The formula -------------------------------------------------------------


def exact_mxfp8(values):
    """``values`` ``[rows, columns]`` FP32 encoded as the fused launch encodes
    a gated row and decoded, in FP64: per 32 elements the least power of two
    ``2^s`` with ``amax ≤ 448 · 2^s`` (at least ``2⁻²²``), the codes
    ``e4m3(x / 2^s)`` round to nearest even.
    """  # noqa: D205
    rows, columns = values.shape
    blocks = values.reshape(rows, columns // mx.BLOCK, mx.BLOCK)
    amax = blocks.abs().amax(dim=-1).contiguous()
    bits = amax.view(torch.int32).to(torch.int64)
    rounded = (bits + 0x7FFFFF - 0x600000) >> 23
    exponent = rounded.clamp_min(105 + 8) - 8
    inverse = torch.ldexp(
        torch.ones_like(amax), (127 - exponent).to(torch.int32)
    )
    codes = (
        (blocks * inverse[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    scale = torch.ldexp(
        torch.ones_like(amax, dtype=torch.float64),
        (exponent - 127).to(torch.int32),
    )
    return (codes.double() * scale[..., None]).reshape(rows, columns)


def gated(xhat, gate, up, weights):
    """The gated rows ``a`` of rows ``xhat`` (FP64) through decoded ``gate``
    and ``up`` (FP64 ``[width, hidden]``), each row weighed by its ``weights``
    (FP32 ``[rows]``), in FP32 as the formula states.
    """  # noqa: D205
    g = (xhat @ gate.T).float().to(torch.bfloat16).float()
    u = (xhat @ up.T).float().to(torch.bfloat16).float()
    g = torch.minimum(g, torch.tensor(LIMIT, device=g.device))
    u = u.clamp(-LIMIT, LIMIT)
    silu = g / (1.0 + torch.exp(-g))
    return silu * u * weights[:, None]


def references(rank, inputs, routes, route_weights, weights_of):
    """The formula's ``[rows, hidden]`` output for rank ``rank``'s checked
    ``inputs`` (BF16) and their ``[rows, top]`` routes, every expert's weights
    read from its owner (``weights_of(owner)``).
    """  # noqa: D205
    device = inputs.device
    codes, scales = mx.encode_mxfp8(inputs.float())
    xhat = mx.decode_mxfp8(codes, scales)
    total = torch.zeros(inputs.shape, dtype=torch.float64, device=device)

    # The routes, by expert, each expert's weights decoded once.
    for expert in torch.unique(routes).tolist():
        rows, slots = torch.nonzero(routes == expert, as_tuple=True)
        owner, local = divmod(expert, LOCAL)
        held = weights_of(owner)
        decode = lambda codes, scales, columns: mx.decode_mxfp4(  # noqa: E731
            codes[local].to(device), scales[local].to(device), columns
        )
        a = gated(
            xhat[rows],
            decode(held.gate, held.gate_scales, HIDDEN),
            decode(held.up, held.up_scales, HIDDEN),
            route_weights[rows, slots].float(),
        )
        down = decode(held.down, held.down_scales, INTERMEDIATE)
        out = (exact_mxfp8(a) @ down.T).float().to(torch.bfloat16).double()
        total.index_add_(0, rows, out)

    # The rank's own shared expert, over every row.
    held = weights_of(rank)
    a = gated(
        xhat,
        mx.decode_mxfp8(held.shared_gate, held.shared_gate_scales),
        mx.decode_mxfp8(held.shared_up, held.shared_up_scales),
        torch.ones(inputs.shape[0], device=device),
    )
    shared_down = mx.decode_mxfp8(held.shared_down, held.shared_down_scales)
    total += (
        (exact_mxfp8(a) @ shared_down.T).float().to(torch.bfloat16).double()
    )
    return total.float().to(torch.bfloat16)


# ---- The ranks ---------------------------------------------------------------


def trial_inputs(trial, rank, device):
    """Rank ``rank``'s rows, routes and route weights of trial ``trial``."""
    tokens, live, capacity, _ = TRIALS[trial]
    silent = trial == 1 and rank == RANKS - 1
    sent = 0 if silent else tokens
    generator = torch.Generator(device=device).manual_seed(100 * trial + rank)
    rows = max(sent, 1)
    states = torch.randn((rows, HIDDEN), device=device, generator=generator).to(
        torch.bfloat16
    )
    scores = torch.rand((rows, EXPERTS), device=device, generator=generator)
    routes = torch.topk(scores, TOP, dim=1).indices.to(torch.int32)
    route_weights = torch.rand((rows, TOP), device=device, generator=generator)
    count = 0 if silent else live
    return sent, count, capacity, states, routes, route_weights


def run_rank(library, channel, rank, held, results):
    """Rank ``rank``'s lifecycle: create, the trials' launches, release. Its
    outputs and inputs land in ``results[rank]``.
    """  # noqa: D205
    device = rank
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    group, callbacks = exchange.group_of(channel, RANKS, rank, device)
    life = exchange.Lifecycle(
        group=ctypes.cast(ctypes.pointer(group), ctypes.c_void_p),
        ranks=RANKS,
        rank=rank,
        experts=EXPERTS,
        hidden=HIDDEN,
        max_tokens=CAPACITY,
        top=TOP,
        dtype=BF16,
        device=device,
        intermediate=INTERMEDIATE,
        attention_ranks=0,
        weights=MXFP4,
        shared=SHARED,
        limit=LIMIT,
    )
    exchange.entered(library, "create", life)
    weights = held[rank]
    layer = weights.layer()
    results[rank] = []
    for trial in range(len(TRIALS)):
        sent, count, capacity, states, routes, route_weights = trial_inputs(
            trial, rank, device
        )
        live = torch.tensor([count], dtype=torch.int32, device=device)
        out = torch.full(
            (max(sent, 1), HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        call = exchange.Call(
            transport=life.transport,
            states=states.data_ptr(),
            ids=routes.data_ptr(),
            weights=route_weights.data_ptr(),
            live=live.data_ptr(),
            out=out.data_ptr(),
            layer=ctypes.cast(ctypes.pointer(layer), ctypes.c_void_p),
            mode=FUSED,
            tokens=sent,
            capacity=capacity,
            top=TOP,
            hidden=HIDDEN,
            experts=EXPERTS,
            max_tokens=CAPACITY,
            ranks=RANKS,
            dtype=BF16,
        )
        with torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.current_stream(device))
            exchange.entered(library, "launch", call, stream.cuda_stream)
        stream.synchronize()
        results[rank].append((sent, count, states, routes, route_weights, out))
    channel.barrier()
    exchange.entered(library, "release", life)
    del callbacks


def check_rank(rank, results, held):
    """Holds rank ``rank``'s outputs of every trial to the formula; answers
    the largest error relative to its row's largest magnitude.
    """  # noqa: D205
    device = rank
    torch.cuda.set_device(device)
    atol, rtol = fused_tolerance()
    worst = 0.0
    for trial, (sent, count, states, routes, route_weights, out) in enumerate(
        results[rank]
    ):
        if sent == 0:
            continue
        assert bool((out[count:sent] == 0).all()), (
            f"trial {trial}: a row past the live count {count} of {sent} "
            "is not zero"
        )
        sample = TRIALS[trial][3]
        rows = torch.arange(count, device=device)
        if sample is not None and sample < count:
            generator = torch.Generator(device=device).manual_seed(trial)
            rows = torch.randperm(count, device=device, generator=generator)[
                :sample
            ]
        expected = references(
            rank,
            states[rows],
            routes[rows].long(),
            route_weights[rows],
            lambda owner: held[owner],
        ).float()
        got = out[rows].float()
        peak = expected.abs().amax(dim=1, keepdim=True)
        error = (got - expected).abs()
        bound = atol + rtol * peak
        relative = (error / peak.clamp_min(1e-30)).amax().item()
        worst = max(worst, relative)
        assert bool((error <= bound).all()), (
            f"trial {trial} (rows {sent}, live {count}): an output differs by "
            f"{relative:.4g} of its row's largest magnitude, bound {rtol:.4g}"
        )
    return worst


@pytest.mark.gpu
def test_four_ranks_run_the_fused_mxfp4_layer(kernel_module):
    if torch.cuda.device_count() < RANKS:
        pytest.skip(
            f"needs {RANKS} CUDA devices, this host shows "
            f"{torch.cuda.device_count()}"
        )
    multiprocessors = torch.cuda.get_device_properties(0).multi_processor_count
    library = exchange.load(
        kernel_module("expert_exchange", "deepgemm", key(multiprocessors))
        / "module.so"
    )
    run_and_check(library)


def run_and_check(library):
    """The whole check over the module ``library``: answers each rank's
    largest relative error.
    """  # noqa: D205
    held = [Weights(rank, rank) for rank in range(RANKS)]
    results = [None] * RANKS
    channel = exchange.Channel(RANKS)
    failures = exchange.run_ranks(
        channel,
        RANKS,
        lambda rank: run_rank(library, channel, rank, held, results),
    )
    assert not failures, "; ".join(failures)
    worst = [None] * RANKS

    def check(rank):
        worst[rank] = check_rank(rank, results, held)

    failures = exchange.run_ranks(exchange.Channel(RANKS), RANKS, check)
    assert not failures, "; ".join(failures)
    return worst
