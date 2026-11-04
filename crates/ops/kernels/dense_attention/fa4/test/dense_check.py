#!/usr/bin/env python3
"""Run one BF16 dense attention launch through the built fa4 module and check it.

This is the GPU check a build host cannot do: it loads the module the store holds
exactly as a runner does (``module.so``), calls the operator's entry with the call
structure of ``crates/ops/src/attention/dense.rs`` (720 bytes, ``error`` at 208), and
compares the module's output with exact attention computed in torch.

    python dense_check.py <module_dir> [--batch 8] [--len 3000] [--query-len 512]

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding
``module.so``. Run it on the GB200 with the Program Server's own environment
(``crates/program-server/python/.venv/bin/python``), which has torch.

The module is compiled for the representative problem its manifest's ``[parameters]``
state (batch 8, 512 queries per launch); this check stays at that shape and varies only
the sequence lengths, because a configuration chosen for another shape is a different
compilation.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

# The operator's `DType::as_i64()` numbering.
BF16 = 4


def pointer(tensor):
    """A tensor's device address as the call structure carries it."""
    return ctypes.c_void_p(tensor.data_ptr())


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


def reference(query, key, value, cu_seqlens_q, cu_seqlens_k, scale, causal):
    """Exact attention over the same packed sequences, in FP32."""
    head_dim = query.shape[2]
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
            visible = keyed[None, :] > queries[:, None] + (k_end - k_start) - (q_end - q_start)
            logits = logits.masked_fill(visible[None, :, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        out[q_start:q_end] = torch.einsum("thk,hkd->thd", weights, values).transpose(0, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=pathlib.Path, help="the module directory resolve hands a runner")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--len", type=int, default=3000, help="keys of the shortest sequence")
    parser.add_argument("--query-len", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    arguments = parser.parse_args()

    library = ctypes.CDLL(str(arguments.module / "module.so"))
    entry = library.uniserve_dense_attention
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch = arguments.batch
    head_dim, q_heads, kv_heads = arguments.head_dim, arguments.q_heads, arguments.kv_heads
    lengths = torch.tensor(
        [arguments.len + index * 137 for index in range(batch)], dtype=torch.int32, device=device
    )
    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.arange(
        arguments.query_len, arguments.query_len * (batch + 1), arguments.query_len, device=device
    ).to(torch.int32)
    cu_seqlens_k[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)

    q_tokens = int(cu_seqlens_q[-1])
    kv_tokens = int(cu_seqlens_k[-1])
    query = torch.randn(q_tokens, q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    key = torch.randn(kv_tokens, kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    value = torch.randn_like(key) * 0.1
    out = torch.empty(q_tokens, q_heads, head_dim, dtype=torch.bfloat16, device=device)

    call = Call()
    call.out = pointer(out)
    call.query = pointer(query)
    call.key = pointer(key)
    call.value = pointer(value)
    call.cu_seq_lens_q = pointer(cu_seqlens_q)
    call.cu_seq_lens_k = pointer(cu_seqlens_k)
    call.batch = batch
    call.q_tokens = q_tokens
    call.kv_tokens = kv_tokens
    call.max_q_len = arguments.query_len
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
    call.scale = 1.0 / math.sqrt(head_dim)
    call.window_left = -1  # no window
    call.causal = 1
    call.dtype_query = BF16
    call.dtype_key_value = BF16
    call.dtype_output = BF16

    stream = torch.cuda.current_stream().cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if status != 0:
        print(f"the entry refused the call: {call.error.decode(errors='replace')}", file=sys.stderr)
        return 1

    expected = reference(query, key, value, cu_seqlens_q, cu_seqlens_k, call.scale, True).to(
        torch.bfloat16
    )
    error = (out.float() - expected.float()).abs().max().item()
    largest = expected.float().abs().max().item()
    print(f"max |out - exact attention| = {error:.6g} (largest value {largest:.6g})")
    if error > 0.05 * max(largest, 1.0):
        print("the dense pass does not match exact attention", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
