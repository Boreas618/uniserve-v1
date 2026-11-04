#!/usr/bin/env python3
"""Run one BF16 paged prefill through the built fa4 module and check it.

This is the GPU check a build host cannot do: it loads the module the store holds
exactly as a runner does (``module.so``), calls the operator's entry with the
call structure of ``crates/ops/src/attention/mod.rs`` (784 bytes, ``error`` at
272), and compares the module's output with exact attention computed in torch.

    python prefill_check.py <module_dir> [--page 64] [--batch 8] [--len 3000]

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding
``module.so``. Run it on the GB200 with the Program Server's own environment
(``crates/program-server/python/.venv/bin/python``), which has torch.

The module is compiled for the representative problem its manifest's
``[parameters]`` state (batch 8, 512 queries per request); this check stays at
that shape and varies only the lengths, because a configuration chosen for
another shape is a different compilation.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

# The operator's `DType::as_i64()` numbering.
BF16 = 4
# The operator's phase numbering.
PREFILL = 0


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


def pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    """A tensor's device address as the call carries it."""
    return ctypes.c_void_p(tensor.data_ptr())


def reference(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Exact causal attention over each request's pages, in float32.

    `query` is `[tokens, q_heads, head_dim]` with request `b` holding the rows
    `cu_seqlens_q[b]..cu_seqlens_q[b+1]`, the cache is
    `[pages, page_size, kv_heads, head_dim]`, and request `b` reads the pages
    `page_table[b, 0]..page_table[b, len - 1]` in order.
    """
    q_heads, head_dim = query.shape[1], query.shape[2]
    kv_heads, page = key_pages.shape[2], key_pages.shape[1]
    grouped = q_heads // kv_heads
    out = torch.empty_like(query, dtype=torch.float32)
    for b in range(seq_lens.numel()):
        start, end = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        length = int(seq_lens[b])
        pages = page_table[b, : (length + page - 1) // page].long()
        k = key_pages[pages].reshape(-1, kv_heads, head_dim)[:length]
        v = value_pages[pages].reshape(-1, kv_heads, head_dim)[:length]
        k = k.repeat_interleave(grouped, dim=1).transpose(0, 1).float()
        v = v.repeat_interleave(grouped, dim=1).transpose(0, 1).float()
        rows = query[start:end].float()
        logits = torch.einsum("thd,hkd->thk", rows, k) * scale
        # The whole-query pass is causal: row i of a request reads keys 0..i.
        positions = torch.arange(end - start, device=query.device)
        keys = torch.arange(length, device=query.device)
        logits = logits.masked_fill(keys[None, None, :] > positions[:, None, None], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        out[start:end] = torch.einsum("thk,hkd->thd", weights, v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=pathlib.Path, help="the module directory resolve hands a runner")
    parser.add_argument("--page", type=int, default=64, help="tokens per page of the cache")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--len", type=int, default=3000, help="key length of the shortest request")
    parser.add_argument("--query-len", type=int, default=64, help="queries per request")
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    arguments = parser.parse_args()

    library = ctypes.CDLL(str(arguments.module / "module.so"))
    entry = library.uniserve_paged_attention
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, page = arguments.batch, arguments.page
    head_dim, q_heads, kv_heads = arguments.head_dim, arguments.q_heads, arguments.kv_heads
    query_len = arguments.query_len
    tokens = batch * query_len
    # Each request reads a different length, so the page table's width is the
    # longest one's and the shorter requests leave the rest unused.
    seq_lens = torch.tensor(
        [arguments.len + index * 137 for index in range(batch)], dtype=torch.int32, device=device
    )
    max_pages = (int(seq_lens.max()) + page - 1) // page
    pool_pages = max_pages * batch
    key = torch.randn(pool_pages, page, kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    value = torch.randn_like(key) * 0.1
    query = torch.randn(tokens, q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    page_table = torch.arange(pool_pages, dtype=torch.int32, device=device).view(batch, max_pages)
    cu_seqlens_q = torch.arange(
        0, tokens + 1, query_len, dtype=torch.int32, device=device
    )
    out = torch.empty(tokens, q_heads, head_dim, dtype=torch.bfloat16, device=device)

    call = Call()
    call.out = pointer(out)
    call.query = pointer(query)
    call.key = pointer(key)
    call.value = pointer(value)
    call.page_table = pointer(page_table)
    call.seq_lens = pointer(seq_lens)
    call.cum_seq_lens_q = pointer(cu_seqlens_q)
    call.batch = batch
    call.tokens = tokens
    call.max_q_len = query_len
    call.max_kv_len = max_pages * page
    call.num_pages = pool_pages
    call.page_size = page
    call.q_heads = q_heads
    call.kv_heads = kv_heads
    call.head_dim = head_dim
    call.q_stride_tokens = q_heads * head_dim
    call.q_stride_heads = head_dim
    call.kv_stride_page = page * kv_heads * head_dim
    call.kv_stride_tokens = kv_heads * head_dim
    call.kv_stride_heads = head_dim
    call.o_stride_tokens = q_heads * head_dim
    call.o_stride_heads = head_dim
    call.scale = 1.0 / math.sqrt(head_dim)
    call.window_left = -1  # no window
    call.causal = 1
    call.dtype_query = BF16
    call.dtype_key_value = BF16
    call.dtype_output = BF16
    call.phase = PREFILL

    stream = torch.cuda.current_stream().cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if status != 0:
        print(f"the entry refused the call: {call.error.decode(errors='replace')}", file=sys.stderr)
        return 1

    expected = reference(query, key, value, page_table, seq_lens, cu_seqlens_q, call.scale).to(
        torch.bfloat16
    )
    error = (out.float() - expected.float()).abs().max().item()
    scale = expected.float().abs().max().item()
    print(f"max |out - exact attention| = {error:.6g} (largest value {scale:.6g})")
    if error > 0.05 * max(scale, 1.0):
        print("the prefill does not match exact attention", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
