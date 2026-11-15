"""The dense attention modules against exact attention, and the f32 guards.

Each test loads a built module of the `dense_attention` operator (resolved at
the session's Build Server for the session's device) as a runner loads it
(``module.so``), calls its entry with the call structure of
``crates/ops/src/attention/dense.rs`` and compares the output with exact
attention computed in torch.

**The f32 module.** Its mask (`UNISERVE_MASK`) and its sequence layout
(`UNISERVE_SEQUENCES`: packed sequences with offsets, or one sequence whose
key carries ``cu_seq_lens_false``) are compiled in, so each form is a module
of its own, compared with exact attention in FP64; a history window is not
compiled in, and a causal launch exercises it. The tolerance is the
operator's (`dense.rs::tolerance`): with `n` the key extent, `u = 2^-24` and
`γₙ = n·u/(1 − n·u)`, two implementations differ by at most
`(2γₙ + u)·V + (2γₙ + u)·|out|` over the value envelope `V = 8`, which is
`Tolerance::rounding(n, F32, F32).with_atol((2γₙ + u)·V)`. The module
normalizes the weights before its second product, so each weight carries one
rounding of its own, the `u·V` term of the same bound.

**The f32 module's guards.** A module built causal and called dense would
mask the future away and return a plausible answer, and a runner reusing one
module across descriptors would not notice; the entry's guards refuse such a
call before its first CUDA call, reading only the call's scalars and the
null-ness of its pointers. The cases are the mismatches the descriptor can
express: a causal module called dense and a dense module called causal, a
module that attends one sequence called with offsets or with two sequences, a
null buffer and element types that are not FP32, each refusal naming its
guard; and a call that matches the module, which no guard refuses.

**The fa4 module.** A BF16 module compiled for the representative problem its
manifest's ``[parameters]`` state (batch 8, 512 queries per launch); the check
stays at that shape and varies only the sequence lengths, because a
configuration chosen for another shape is a different compilation. Its causal
reference aligns the queries with the last of their sequence's keys.

Every test but the call structure's size loads a module built for the
session's device.
"""

import ctypes
import math

import pytest
import torch

pytestmark = pytest.mark.unit

#: `ops::abi::code`'s numbers of the element types these modules serve.
F16 = 3
BF16 = 4
F32 = 5

#: The operator's value envelope, `ops::tolerance::VALUE_ENVELOPE`.
VALUE_ENVELOPE = 8.0

#: `DType::F32.unit_roundoff()`.
UNIT_ROUNDOFF = 5.960464477539063e-08

#: The size `DENSE_ATTENTION_CALL_LAYOUT` states for the call structure.
DENSE_CALL_SIZE = 720


class Call(ctypes.Structure):
    """`ops::attention::dense::DenseCall`, field for field."""

    _fields_ = [
        ("out", ctypes.c_void_p),
        ("query", ctypes.c_void_p),
        ("key", ctypes.c_void_p),
        ("value", ctypes.c_void_p),
        ("cu_seq_lens_q", ctypes.c_void_p),
        ("cu_seq_lens_k", ctypes.c_void_p),
        ("batch", ctypes.c_int64),
        ("q_tokens", ctypes.c_int64),
        ("kv_tokens", ctypes.c_int64),
        ("max_q_len", ctypes.c_int64),
        ("max_kv_len", ctypes.c_int64),
        ("q_heads", ctypes.c_int64),
        ("kv_heads", ctypes.c_int64),
        ("head_dim", ctypes.c_int64),
        ("q_stride_tokens", ctypes.c_int64),
        ("q_stride_heads", ctypes.c_int64),
        ("kv_stride_tokens", ctypes.c_int64),
        ("kv_stride_heads", ctypes.c_int64),
        ("o_stride_tokens", ctypes.c_int64),
        ("o_stride_heads", ctypes.c_int64),
        ("scale", ctypes.c_double),
        ("window_left", ctypes.c_int64),
        ("causal", ctypes.c_int64),
        ("dtype_query", ctypes.c_int64),
        ("dtype_key_value", ctypes.c_int64),
        ("dtype_output", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def test_the_call_mirror_is_the_operators_dense_call():
    assert ctypes.sizeof(Call) == DENSE_CALL_SIZE, (
        f"the check's Call mirror is {ctypes.sizeof(Call)} bytes; the "
        f"operator's DENSE_ATTENTION_CALL_LAYOUT says {DENSE_CALL_SIZE} "
        "(`crates/ops/src/attention/dense.rs`)"
    )


def pointer(tensor):
    """A tensor's device address as the call structure carries it."""
    return ctypes.c_void_p(tensor.data_ptr())


def entry(kernel_module, backend, key):
    """The C entry of the `dense_attention` module of `backend` and `key`."""
    library = ctypes.CDLL(
        str(kernel_module("dense_attention", backend, key) / "module.so")
    )
    function = library.uniserve_dense_attention
    function.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


def f32_key(mask, head_dim, q_heads, kv_heads, window=False, offsets=True):
    """The descriptor key of an f32 module: the one-sequence form's key
    carries ``cu_seq_lens_false``.
    """  # noqa: D205
    key = (
        f"dtype_f32_kv_f32_out_f32_head_dim_{head_dim}_q_heads_{q_heads}_"
        f"kv_heads_{kv_heads}_mask_{mask}_window_{str(window).lower()}"
    )
    return key if offsets else key + "_cu_seq_lens_false"


def call_of(
    out, query, key, value, offsets, lengths, max_q_len, scale, causal, dtype
):
    """A call over packed `[tokens, heads, head_dim]` rows; `offsets` are the
    two `[batch + 1]` tables, or `None` for a call that carries none.
    """  # noqa: D205
    q_tokens, q_heads, head_dim = query.shape
    kv_tokens, kv_heads, _ = key.shape
    call = Call()
    call.out = pointer(out)
    call.query = pointer(query)
    call.key = pointer(key)
    call.value = pointer(value)
    call.cu_seq_lens_q = None if offsets is None else pointer(offsets[0])
    call.cu_seq_lens_k = None if offsets is None else pointer(offsets[1])
    call.batch = lengths.numel()
    call.q_tokens = q_tokens
    call.kv_tokens = kv_tokens
    call.max_q_len = max_q_len
    call.max_kv_len = int(lengths.max())
    call.q_heads = q_heads
    call.kv_heads = kv_heads
    call.head_dim = head_dim
    call.q_stride_tokens = q_heads * head_dim
    call.q_stride_heads = head_dim
    call.kv_stride_tokens = kv_heads * head_dim
    call.kv_stride_heads = head_dim
    call.o_stride_tokens = q_heads * head_dim
    call.o_stride_heads = head_dim
    call.scale = scale
    call.window_left = -1
    call.causal = int(causal)
    call.dtype_query = dtype
    call.dtype_key_value = dtype
    call.dtype_output = dtype
    return call


def launched(function, call, device):
    """Call `function` on the session's stream and wait for it."""
    stream = torch.cuda.current_stream(device).cuda_stream
    status = function(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(device)
    assert status == 0, (
        f"the entry refused the call: {call.error.decode(errors='replace')}"
    )


def f32_reference(
    query, key, value, cu_seqlens_q, cu_seqlens_k, scale, causal, window
):
    """Exact attention over the same packed sequences, in FP64.

    The mask is the operator's: key `j` of the sequence is visible to query
    row `i` when `j <= i` (causal), and no further left than `window` keys
    when a history window is launched. A one-sequence launch names no tables,
    but it is the same arithmetic as one sequence whose keys run from the
    first to the last.
    """
    out = torch.empty(query.shape, dtype=torch.float64, device=query.device)
    for index in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        k_start, k_end = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        rows = query[q_start:q_end].double().transpose(0, 1)
        keys = key[k_start:k_end].double().transpose(0, 1)
        values = value[k_start:k_end].double().transpose(0, 1)
        grouped = rows.shape[0] // keys.shape[0]
        keys = keys.repeat_interleave(grouped, dim=0)
        values = values.repeat_interleave(grouped, dim=0)
        logits = torch.einsum("htd,hkd->htk", rows, keys) * scale
        queries = torch.arange(q_end - q_start, device=query.device)
        keyed = torch.arange(k_end - k_start, device=query.device)
        hidden = torch.zeros(
            queries.numel(),
            keyed.numel(),
            dtype=torch.bool,
            device=query.device,
        )
        if causal:
            hidden |= keyed[None, :] > queries[:, None]
            if window >= 0:
                hidden |= keyed[None, :] < queries[:, None] - window
        weights = torch.softmax(
            logits.masked_fill(hidden[None, :, :], float("-inf")), dim=-1
        )
        out[q_start:q_end] = torch.einsum(
            "htk,hkd->htd", weights, values
        ).transpose(0, 1)
    return out


def f32_tolerance(max_kv_len):
    """The operator's absolute and relative tolerance over a key extent of
    `max_kv_len`.
    """  # noqa: D205
    terms = max_kv_len * UNIT_ROUNDOFF
    gamma = terms / (1.0 - terms)
    both = 2.0 * gamma + UNIT_ROUNDOFF
    return both * VALUE_ENVELOPE, both


#: The f32 module's extents: query and key-value heads, the head width, the
#: sequences of a packed call, its query rows per sequence and the keys of its
#: shortest sequence.
F32_HEADS, F32_KV_HEADS, F32_HEAD_DIM = 8, 8, 64
F32_BATCH, F32_QUERY_LEN = 4, 200


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("mask", "offsets", "keys", "window"),
    [
        pytest.param("causal", True, 900, -1, id="causal, packed"),
        pytest.param("dense", True, 900, -1, id="dense, packed"),
        pytest.param("dense", False, 900, -1, id="dense, one sequence"),
        pytest.param(
            "causal", False, 1500, -1, id="causal, one sequence of 1500"
        ),
        pytest.param(
            "causal", True, 900, 128, id="causal, packed, a window of 128"
        ),
    ],
)
def test_the_f32_module_matches_exact_attention_within_the_operators_bound(
    mask, offsets, keys, window, torch_device, kernel_module
):
    causal = mask == "causal"
    function = entry(
        kernel_module,
        "f32",
        f32_key(
            mask,
            F32_HEAD_DIM,
            F32_HEADS,
            F32_KV_HEADS,
            window=window >= 0,
            offsets=offsets,
        ),
    )
    torch.manual_seed(0)
    batch = F32_BATCH if offsets else 1
    lengths = torch.tensor(
        [keys + index * 37 for index in range(batch)],
        dtype=torch.int32,
        device=torch_device,
    )
    cu_seqlens_q = torch.zeros(
        batch + 1, dtype=torch.int32, device=torch_device
    )
    cu_seqlens_k = torch.zeros(
        batch + 1, dtype=torch.int32, device=torch_device
    )
    cu_seqlens_q[1:] = torch.cumsum(
        torch.full((batch,), F32_QUERY_LEN, dtype=torch.int32), dim=0
    ).to(torch_device)
    cu_seqlens_k[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    q_tokens = int(cu_seqlens_q[-1])
    kv_tokens = int(cu_seqlens_k[-1])

    def rows(tokens, heads):
        return (
            torch.randn(
                tokens,
                heads,
                F32_HEAD_DIM,
                dtype=torch.float32,
                device=torch_device,
            )
            * 0.1
        )

    query = rows(q_tokens, F32_HEADS)
    key = rows(kv_tokens, F32_KV_HEADS)
    value = torch.randn_like(key) * 0.1
    out = torch.zeros_like(query)

    # The one-sequence form's program has no ports for the tables, so the call
    # carries none: this is what `dense.rs::sequence_layout` hands a module
    # built for it.
    call = call_of(
        out,
        query,
        key,
        value,
        (cu_seqlens_q, cu_seqlens_k) if offsets else None,
        lengths,
        q_tokens,
        1.0 / math.sqrt(F32_HEAD_DIM),
        causal,
        F32,
    )
    call.window_left = window
    launched(function, call, torch_device)

    expected = f32_reference(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        call.scale,
        causal,
        window,
    )
    absolute, relative = f32_tolerance(int(lengths.max()))
    difference = (out.double() - expected).abs()
    # `~(error <= bound)` rather than `error > bound`: a NaN difference (an
    # unwritten row, or `-inf - -inf`) is neither, and the plain comparison
    # would count it as inside the bound.
    exceeded = ~(difference <= absolute + relative * expected.abs())
    crossing = exceeded.nonzero()
    assert not bool(exceeded.any()), (
        f"{batch} sequence(s), {q_tokens} query rows over {kv_tokens} keys: "
        f"the FP32 pass exceeds the operator's tolerance {absolute:.6g} + "
        f"{relative:.3g}·|out| at {len(crossing)} element(s), the first "
        f"{tuple(crossing[0].tolist())}; max |out - exact| = "
        f"{difference.nan_to_num().max().item():.6g}, largest value "
        f"{expected.abs().max().item():.6g}"
    )


#: The extents of the guard calls: the tokens, heads and head width a call
#: states, which the matching call's buffers hold.
GUARD_TOKENS, GUARD_HEADS, GUARD_KV_HEADS, GUARD_HEAD_DIM = 64, 32, 8, 128


@pytest.fixture
def guard_buffers(torch_device):
    """The buffers of a guard call, sized for the call that matches the
    module: the guards read only their addresses.
    """  # noqa: D205

    def rows(heads):
        return torch.zeros(
            GUARD_TOKENS,
            heads,
            GUARD_HEAD_DIM,
            dtype=torch.float32,
            device=torch_device,
        )

    offsets = torch.tensor([0, GUARD_TOKENS], dtype=torch.int32)
    return {
        "out": rows(GUARD_HEADS),
        "query": rows(GUARD_HEADS),
        "key": rows(GUARD_KV_HEADS),
        "value": rows(GUARD_KV_HEADS),
        "offsets": (offsets.to(torch_device), offsets.to(torch_device)),
    }


def guard_call(buffers, *, causal, offsets, batch, dtype, nulls=()):
    """A call of the guards' extents; `nulls` names buffer fields left null,
    a call that omits one of its tensors.
    """  # noqa: D205
    call = call_of(
        buffers["out"],
        buffers["query"],
        buffers["key"],
        buffers["value"],
        buffers["offsets"] if offsets else None,
        torch.full((batch,), 1024),
        GUARD_TOKENS,
        GUARD_HEAD_DIM**-0.5,
        causal,
        dtype,
    )
    for name in nulls:
        setattr(call, name, None)
    return call


def guard_key(mask):
    """The one-sequence f32 module of `mask`, at the guard calls' extents."""
    return f32_key(
        mask, GUARD_HEAD_DIM, GUARD_HEADS, GUARD_KV_HEADS, offsets=False
    )


#: The calls a one-sequence module refuses: the module's mask, the call, and
#: the words its refusal states.
REFUSALS = [
    pytest.param(
        "causal",
        {"causal": False, "offsets": False, "batch": 1, "dtype": F32},
        ("compiles the causal mask", "causal = 0"),
        id="a causal module launched dense",
    ),
    pytest.param(
        "dense",
        {"causal": True, "offsets": False, "batch": 1, "dtype": F32},
        ("compiles the dense mask", "causal = 1"),
        id="a dense module launched causal",
    ),
    *(
        pytest.param(
            mask,
            {
                "causal": mask == "causal",
                "offsets": True,
                "batch": 2,
                "dtype": F32,
            },
            ("attends one sequence", "offsets"),
            id=f"a {mask} module given sequence offsets",
        )
        for mask in ("causal", "dense")
    ),
    *(
        pytest.param(
            mask,
            {
                "causal": mask == "causal",
                "offsets": False,
                "batch": 2,
                "dtype": F32,
            },
            ("attends one sequence", "names 2"),
            id=f"a {mask} module given two sequences",
        )
        for mask in ("causal", "dense")
    ),
    pytest.param(
        "causal",
        {"causal": True, "offsets": False, "batch": 1, "dtype": F16},
        ("serves FP32 activations only",),
        id="FP16 rows",
    ),
    # The field is left null rather than pointed at a stand-in, so the entry
    # has to look at it.
    pytest.param(
        "causal",
        {
            "causal": True,
            "offsets": False,
            "batch": 1,
            "dtype": F32,
            "nulls": ("value",),
        },
        ("leaves one of its buffers null",),
        id="a null buffer",
    ),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("mask", "call", "words"), REFUSALS)
def test_the_f32_module_refuses_a_call_it_was_not_built_for(
    mask, call, words, guard_buffers, kernel_module
):
    function = entry(kernel_module, "f32", guard_key(mask))
    refused = guard_call(guard_buffers, **call)

    status = function(ctypes.byref(refused), None)

    message = refused.error.decode("utf-8", "replace").rstrip("\x00")
    assert status != 0, "the module accepted a call it was not built for"
    for phrase in words:
        assert phrase in message, (
            f"the refusal does not say {phrase!r}: {message!r}"
        )


@pytest.mark.gpu
def test_the_f32_module_passes_a_call_that_matches_it_through_every_guard(
    torch_device, guard_buffers, kernel_module
):
    """A guard that silently widened would refuse the module's own call."""
    function = entry(kernel_module, "f32", guard_key("causal"))
    call = guard_call(
        guard_buffers, causal=True, offsets=False, batch=1, dtype=F32
    )

    status = function(ctypes.byref(call), None)
    torch.cuda.synchronize(torch_device)

    message = call.error.decode("utf-8", "replace").rstrip("\x00")
    guards = (
        "compiles",
        "attends one sequence",
        "FP32 activations",
        "buffers null",
    )
    assert status == 0 or not any(phrase in message for phrase in guards), (
        f"a call that matches the module was refused by a guard: {message!r}"
    )


def fa4_reference(query, key, value, cu_seqlens_q, cu_seqlens_k, scale, causal):
    """Exact attention over the same packed sequences, in FP32; a causal
    sequence's queries are the last of its keys.
    """  # noqa: D205
    out = torch.empty_like(query, dtype=torch.float32)
    for index in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        k_start, k_end = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        rows = query[q_start:q_end].float().transpose(0, 1)
        keys = key[k_start:k_end].float().transpose(0, 1)
        values = value[k_start:k_end].float().transpose(0, 1)
        grouped = rows.shape[0] // keys.shape[0]
        keys = keys.repeat_interleave(grouped, dim=0)
        values = values.repeat_interleave(grouped, dim=0)
        logits = torch.einsum("thd,hkd->thk", rows, keys) * scale
        if causal:
            queries = torch.arange(q_end - q_start, device=query.device)
            keyed = torch.arange(k_end - k_start, device=query.device)
            hidden = keyed[None, :] > queries[:, None] + (k_end - k_start) - (
                q_end - q_start
            )
            logits = logits.masked_fill(hidden[None, :, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        out[q_start:q_end] = torch.einsum(
            "thk,hkd->thd", weights, values
        ).transpose(0, 1)
    return out


#: The fa4 module's problem, its manifest's `[parameters]`: the sequences, the
#: queries per sequence, and the heads of its descriptor; and the keys of the
#: shortest sequence.
FA4_BATCH, FA4_QUERY_LEN = 8, 512
FA4_HEADS, FA4_KV_HEADS, FA4_HEAD_DIM = 32, 4, 128
FA4_KEYS = 3000

#: The fa4 check's allowance: 5% of the largest value (at least 1).
FA4_RELATIVE = 0.05


@pytest.mark.gpu
@pytest.mark.parametrize("mask", ["causal", "dense"])
def test_the_fa4_module_matches_exact_attention(
    mask, torch_device, kernel_module
):
    causal = mask == "causal"
    function = entry(
        kernel_module,
        "fa4",
        f"dtype_bf16_kv_bf16_out_bf16_head_dim_{FA4_HEAD_DIM}_q_heads_"
        f"{FA4_HEADS}_kv_heads_{FA4_KV_HEADS}_mask_{mask}_window_false",
    )
    torch.manual_seed(0)
    lengths = torch.tensor(
        [FA4_KEYS + index * 137 for index in range(FA4_BATCH)],
        dtype=torch.int32,
        device=torch_device,
    )
    cu_seqlens_q = torch.arange(
        0,
        FA4_QUERY_LEN * (FA4_BATCH + 1),
        FA4_QUERY_LEN,
        dtype=torch.int32,
        device=torch_device,
    )
    cu_seqlens_k = torch.zeros(
        FA4_BATCH + 1, dtype=torch.int32, device=torch_device
    )
    cu_seqlens_k[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    q_tokens = int(cu_seqlens_q[-1])
    kv_tokens = int(cu_seqlens_k[-1])

    def rows(tokens, heads):
        return (
            torch.randn(
                tokens,
                heads,
                FA4_HEAD_DIM,
                dtype=torch.bfloat16,
                device=torch_device,
            )
            * 0.1
        )

    query = rows(q_tokens, FA4_HEADS)
    key = rows(kv_tokens, FA4_KV_HEADS)
    value = torch.randn_like(key) * 0.1
    out = torch.empty_like(query)

    call = call_of(
        out,
        query,
        key,
        value,
        (cu_seqlens_q, cu_seqlens_k),
        lengths,
        FA4_QUERY_LEN,
        1.0 / math.sqrt(FA4_HEAD_DIM),
        causal,
        BF16,
    )
    launched(function, call, torch_device)

    expected = fa4_reference(
        query, key, value, cu_seqlens_q, cu_seqlens_k, call.scale, causal
    ).to(torch.bfloat16)
    error = (out.float() - expected.float()).abs().max().item()
    largest = expected.float().abs().max().item()
    allowed = FA4_RELATIVE * max(largest, 1.0)
    assert error <= allowed, (
        f"{mask} mask, {FA4_HEAD_DIM}-element heads, "
        f"{FA4_HEADS}/{FA4_KV_HEADS} query/key heads, longest keys "
        f"{int(lengths.max())}: max |out - exact attention| = {error:.6g} "
        f"exceeds {allowed:.6g} (largest value {largest:.6g})"
    )
