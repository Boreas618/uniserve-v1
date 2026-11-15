"""The trtllm paged decode against exact attention, and the module's guards.

Each test loads the built module of the descriptor below (resolved at the
session's Build Server for the session's device) as a runner loads it
(``module.so`` with the ``cubins/`` beside it, the directory handed to the
module's load entry ``uniserve_trtllm_load`` as ``ops`` hands it) and calls the
operator's entry with the call structure of ``crates/ops/src/attention/mod.rs``.

The decode is compared with exact attention computed in torch, and states
which kernels served it as the profiler records them (``torch.profiler``, the
kernels' own names on the device, a cubin's kernel named as the library's table
names it): the test asserts that at least one kernel ran.

A call the module cannot serve is refused by one of the guards its entry
states before it reaches the device, and the refusal must say which of its
own conditions the call fails: a head width that is not a whole number of
8-element groups or is past 256, a page that is not a power of two, and an
attention window, which this module is compiled without. Each refused call is
built exactly as the served one is, so the refusal is the guard's and not a
side effect of a malformed call.

Every test needs a device.
"""

import ctypes
import math

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

#: `ops::abi::code`'s number of BF16, and the operator's decode phase.
BF16 = 4
DECODE = 1

#: The decode: requests, the keys of the shortest one, query and key-value
#: heads, the head width, the page and the workspace in MiB.
BATCH, LENGTH, Q_HEADS, KV_HEADS, HEAD_DIM, PAGE = 8, 3000, 32, 4, 128, 64
WORKSPACE_MIB = 512

#: The module: the decode's descriptor, whose `len` is the longest key extent
#: a call may state and changes no compiled output.
KEY = (
    f"phase_decode_dtype_bf16_kv_bf16_out_bf16_head_dim_{HEAD_DIM}_page_{PAGE}"
    f"_q_heads_{Q_HEADS}_kv_heads_{KV_HEADS}_mask_causal_window_false_len_40960"
)

#: The check's allowance: 5% of the largest value (at least 1).
RELATIVE = 0.05

#: The guards: the field a call perturbs, the value it sets and the words the
#: entry answers with, from `src/entry.cpp`.
HEAD_WIDTH = (
    "paged_attention serves head widths in whole 8-element groups up to 256"
)
PAGE_SIZE = "paged_attention needs a power-of-two page"
WINDOW = "the trtllm module is compiled without an attention window"
REFUSALS = [
    pytest.param("head_dim", 100, HEAD_WIDTH, id="head width 100"),
    pytest.param("head_dim", 264, HEAD_WIDTH, id="head width 264"),
    pytest.param("page_size", 48, PAGE_SIZE, id="page 48"),
    pytest.param("page_size", 96, PAGE_SIZE, id="page 96"),
    pytest.param("window_left", 0, WINDOW, id="window 0"),
    pytest.param("window_left", 128, WINDOW, id="window 128"),
]


class Call(ctypes.Structure):
    """`ops::attention::paged::Call`, field for field."""

    _fields_ = [
        ("out", ctypes.c_void_p),
        ("query", ctypes.c_void_p),
        ("key", ctypes.c_void_p),
        ("value", ctypes.c_void_p),
        ("key_scale", ctypes.c_void_p),
        ("value_scale", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p),
        ("page_table", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("cum_seq_lens_q", ctypes.c_void_p),
        ("batch", ctypes.c_int64),
        ("tokens", ctypes.c_int64),
        ("max_q_len", ctypes.c_int64),
        ("max_kv_len", ctypes.c_int64),
        ("num_pages", ctypes.c_int64),
        ("page_size", ctypes.c_int64),
        ("q_heads", ctypes.c_int64),
        ("kv_heads", ctypes.c_int64),
        ("head_dim", ctypes.c_int64),
        ("q_stride_tokens", ctypes.c_int64),
        ("q_stride_heads", ctypes.c_int64),
        ("kv_stride_page", ctypes.c_int64),
        ("kv_stride_tokens", ctypes.c_int64),
        ("kv_stride_heads", ctypes.c_int64),
        ("o_stride_tokens", ctypes.c_int64),
        ("o_stride_heads", ctypes.c_int64),
        ("scale", ctypes.c_double),
        ("window_left", ctypes.c_int64),
        ("causal", ctypes.c_int64),
        ("workspace_bytes", ctypes.c_int64),
        ("dtype_query", ctypes.c_int64),
        ("dtype_key_value", ctypes.c_int64),
        ("dtype_output", ctypes.c_int64),
        ("phase", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


class Load(ctypes.Structure):
    """The load entry's call (`uniserve_trtllm/cubins.h`, `ops::LoadCall`)."""

    _fields_ = [
        ("directory", ctypes.c_char_p),
        ("error", ctypes.c_char * 512),
    ]


def pointer(tensor):
    """A tensor's device address as the call carries it."""
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.fixture
def entry(kernel_module):
    """The module's C entry, after its load entry read its cubins."""
    directory = kernel_module("paged_attention", "trtllm", KEY)
    library = ctypes.CDLL(str(directory / "module.so"))
    load = library.uniserve_trtllm_load
    load.argtypes = [ctypes.POINTER(Load), ctypes.c_void_p]
    load.restype = ctypes.c_int32
    loaded = Load(str(directory).encode(), b"")
    assert load(ctypes.byref(loaded), None) == 0, (
        "the load entry refused the module's directory: "
        f"{loaded.error.decode(errors='replace')}"
    )
    function = library.uniserve_paged_attention
    function.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


def decode(device, head_dim=HEAD_DIM, page=PAGE, window_left=-1):
    """One decode call and the tensors it points at.

    The geometry is a parameter so that a refused call can state the value
    the entry must refuse and keep every stride and shape consistent with it:
    a call whose head width is wrong in two places at once would be a
    malformed call, and the guard it reaches first would not be the one under
    test.
    """
    torch.manual_seed(0)
    # Each request reads a different length, so the page table's width is the
    # longest one's and the shorter requests leave the rest unused.
    seq_lens = torch.tensor(
        [LENGTH + index * 137 for index in range(BATCH)],
        dtype=torch.int32,
        device=device,
    )
    max_pages = (int(seq_lens.max()) + page - 1) // page
    pool_pages = max_pages * BATCH

    def rows(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device) * 0.1

    key = rows(pool_pages, page, KV_HEADS, head_dim)
    value = rows(pool_pages, page, KV_HEADS, head_dim)
    query = rows(BATCH, Q_HEADS, head_dim)
    # Request b owns the pages [b * max_pages, (b + 1) * max_pages).
    page_table = torch.arange(pool_pages, dtype=torch.int32, device=device)
    page_table = page_table.view(BATCH, max_pages)
    out = torch.empty(
        BATCH, Q_HEADS, head_dim, dtype=torch.bfloat16, device=device
    )
    workspace = torch.empty(
        WORKSPACE_MIB * 1024 * 1024, dtype=torch.uint8, device=device
    )

    call = Call()
    call.out = pointer(out)
    call.query = pointer(query)
    call.key = pointer(key)
    call.value = pointer(value)
    call.workspace = pointer(workspace)
    call.page_table = pointer(page_table)
    call.seq_lens = pointer(seq_lens)
    call.batch = BATCH
    call.tokens = BATCH
    call.max_q_len = 1
    call.max_kv_len = max_pages * page
    call.num_pages = pool_pages
    call.page_size = page
    call.q_heads = Q_HEADS
    call.kv_heads = KV_HEADS
    call.head_dim = head_dim
    call.q_stride_tokens = Q_HEADS * head_dim
    call.q_stride_heads = head_dim
    call.kv_stride_page = page * KV_HEADS * head_dim
    call.kv_stride_tokens = KV_HEADS * head_dim
    call.kv_stride_heads = head_dim
    call.o_stride_tokens = Q_HEADS * head_dim
    call.o_stride_heads = head_dim
    call.scale = 1.0 / math.sqrt(head_dim)
    call.window_left = window_left
    call.causal = 1
    call.workspace_bytes = workspace.numel()
    call.dtype_query = BF16
    call.dtype_key_value = BF16
    call.dtype_output = BF16
    call.phase = DECODE
    tensors = (out, query, key, value, page_table, seq_lens, workspace)
    return call, tensors


def reference(query, key_pages, value_pages, page_table, seq_lens, scale):
    """Exact attention over each request's pages, in float32.

    `query` is `[batch, q_heads, head_dim]`, the pages are
    `[pages, page_size, kv_heads, head_dim]`; request `b` reads the pages
    `page_table[b, 0]..` in order, cut at its length.
    """
    batch, q_heads, head_dim = query.shape
    page, kv_heads = key_pages.shape[1], key_pages.shape[2]
    grouped = q_heads // kv_heads
    out = torch.empty(
        batch, q_heads, head_dim, dtype=torch.float32, device=query.device
    )
    for b in range(batch):
        length = int(seq_lens[b])
        pages = page_table[b, : (length + page - 1) // page].long()
        k = key_pages[pages].reshape(-1, kv_heads, head_dim)[:length]
        v = value_pages[pages].reshape(-1, kv_heads, head_dim)[:length]
        k = k.repeat_interleave(grouped, dim=1).transpose(0, 1).float()
        v = v.repeat_interleave(grouped, dim=1).transpose(0, 1).float()
        logits = torch.einsum("hd,hkd->hk", query[b].float(), k) * scale
        out[b] = torch.einsum("hk,hkd->hd", torch.softmax(logits, dim=-1), v)
    return out


def test_a_decode_matches_exact_attention_and_names_its_kernels(
    entry, torch_device
):
    call, (out, query, key, value, page_table, seq_lens, _) = decode(
        torch_device
    )

    stream = torch.cuda.current_stream(torch_device).cuda_stream
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiled:
        status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
        torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the entry refused the call: {call.error.decode(errors='replace')}"
    )

    expected = reference(query, key, value, page_table, seq_lens, call.scale)
    expected = expected.to(torch.bfloat16).float()
    error = (out.float() - expected).abs().max().item()
    largest = expected.abs().max().item()
    allowed = RELATIVE * max(largest, 1.0)
    assert error <= allowed, (
        f"max |out - exact attention| = {error:.6g} exceeds {allowed:.6g} "
        f"(largest value {largest:.6g})"
    )
    kernels = {
        event.name
        for event in profiled.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    }
    assert kernels, (
        "the profiler recorded no kernel of the decode: it does not state "
        "which kernel served it"
    )


@pytest.mark.parametrize(("field", "value", "words"), REFUSALS)
def test_a_call_the_module_cannot_serve_is_refused_by_its_guard(
    field, value, words, entry, torch_device
):
    # The geometry the guard reads comes from the case; the rest is the served
    # call's.
    call, _ = decode(
        torch_device,
        head_dim=value if field == "head_dim" else HEAD_DIM,
        page=value if field == "page_size" else PAGE,
        window_left=value if field == "window_left" else -1,
    )

    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)

    answered = call.error.decode(errors="replace").rstrip("\x00")
    assert status != 0, f"{field} {value}: the entry served the call"
    assert words in answered, (
        f"{field} {value}: refused with {answered!r}, which does not state "
        f"{words!r}"
    )
