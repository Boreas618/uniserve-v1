#!/usr/bin/env python3
"""Run one FP32 dense attention launch through the f32 module and check it.

This is the GPU check a build host cannot do: it loads the module the store
holds exactly as a runner does (``module.so``), calls the operator's entry with
the call structure of ``crates/ops/src/attention/dense.rs`` (720 bytes,
``error`` at 208), and compares the module's output with exact attention
computed in torch.

python f32_check.py <module_dir> --mask causal --sequences offsets python
f32_check.py <module_dir> --mask dense  --sequences one python f32_check.py
<module_dir> --mask causal --sequences one --len 1500

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding
``module.so``. Run it on a GPU host with the Program Server's own environment
(``crates/program-server/python/.venv/bin/python``), which has torch.

Both compiled forms are checked separately because they are different programs
of one operator. ``--mask`` must name the mask the module was built for (a
dense module refuses a causal call, and the check reports that rather than
comparing outputs of a call the operator would not have made); ``--sequences``
must name the form, and the no-offsets form is a module whose key carries
``cu_seq_lens_false``. A history window is not compiled in, and ``--window``
exercises it.

The tolerance is not a guess. The operator derives one (`dense.rs::tolerance`):
with `n` the descriptor's cap on the key extent, unit round-off `u = 2^-24` and
`γₙ = n·u/(1 − n·u)`, two implementations differ by at most `(2γₙ + u)·V + (2γₙ
+ u)·|out|` over the declared value envelope `V = 8`, which is what
`Tolerance::rounding(n, F32, F32).with_atol((2γₙ + u)·V)` asserts. This module
normalizes the weights before its second product, so each weight carries one
rounding of its own; that is the `u·V` term of the same bound (`Σ|p_j| ≤ 1`),
which is why the check asserts the operator's tolerance and not a looser one.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

# The operator's `DType::as_i64()` numbering.
F32 = 5

# The operator's bound: `dense.rs`'s `VALUE_ENVELOPE`.
VALUE_ENVELOPE = 8.0

# `DType::F32.unit_roundoff()`.
UNIT_ROUNDOFF = 5.960464477539063e-08


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


def reference(
    query, key, value, cu_seqlens_q, cu_seqlens_k, scale, causal, window
):
    """Exact attention over the same packed sequences, in FP64.

    The mask is the operator's: key `j` of the sequence is visible to query
    row `i` when `j <= i` (causal), and no further left than `window` keys
    when a history window is launched. The two tables are the reference's
    own: a no-offsets launch names no table, but it is the same arithmetic
    as one sequence whose keys run from the first to the last.
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
        logits = torch.einsum("thd,hkd->thk", rows, keys) * scale
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
            "thk,hkd->thd", weights, values
        ).transpose(0, 1)
    return out


def tolerance(max_kv_len):
    """The operator's tolerance over a key extent of `max_kv_len`."""
    terms = max_kv_len * UNIT_ROUNDOFF
    gamma = terms / (1.0 - terms)
    both = 2.0 * gamma + UNIT_ROUNDOFF
    return both * VALUE_ENVELOPE, both


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module",
        type=pathlib.Path,
        help="the module directory resolve hands a runner",
    )
    parser.add_argument("--mask", choices=["causal", "dense"], required=True)
    parser.add_argument(
        "--sequences",
        choices=["offsets", "one"],
        default="offsets",
        help="the form the module was built for",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--len", type=int, default=900, help="keys of the shortest sequence"
    )
    parser.add_argument("--query-len", type=int, default=200)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--window", type=int, default=-1, help="history window in keys"
    )
    arguments = parser.parse_args()
    if arguments.window >= 0 and arguments.mask != "causal":
        parser.error(
            "a history window is expressible only with the causal mask"
        )
    one = arguments.sequences == "one"
    if one and arguments.batch != 1:
        parser.error("the form without offsets attends one sequence: --batch 1")
    if one and arguments.query_len > arguments.len * 8:
        parser.error(
            "a one-sequence check keeps the query rows and keys comparable"
        )

    library = ctypes.CDLL(str(arguments.module / "module.so"))
    entry = library.uniserve_dense_attention
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch = 1 if one else arguments.batch
    head_dim = arguments.head_dim
    q_heads, kv_heads = arguments.q_heads, arguments.kv_heads
    lengths = torch.tensor(
        [arguments.len + index * 37 for index in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    query_rows = [arguments.query_len] if one else [arguments.query_len] * batch
    cu_seqlens_q[1:] = torch.cumsum(
        torch.tensor(query_rows, dtype=torch.int32, device=device), dim=0
    )
    cu_seqlens_k[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)

    q_tokens = int(cu_seqlens_q[-1])
    kv_tokens = int(cu_seqlens_k[-1])
    query = (
        torch.randn(
            q_tokens, q_heads, head_dim, dtype=torch.float32, device=device
        )
        * 0.1
    )
    key = (
        torch.randn(
            kv_tokens, kv_heads, head_dim, dtype=torch.float32, device=device
        )
        * 0.1
    )
    value = torch.randn_like(key) * 0.1
    out = torch.zeros(
        q_tokens, q_heads, head_dim, dtype=torch.float32, device=device
    )

    call = Call()
    call.out = pointer(out)
    call.query = pointer(query)
    call.key = pointer(key)
    call.value = pointer(value)
    # The no-offsets form's program has no ports for the tables, so the call
    # carries none: this is what `dense.rs::sequence_layout` hands a module
    # built for it.
    call.cu_seq_lens_q = None if one else pointer(cu_seqlens_q)
    call.cu_seq_lens_k = None if one else pointer(cu_seqlens_k)
    call.batch = batch
    call.q_tokens = q_tokens
    call.kv_tokens = kv_tokens
    call.max_q_len = q_tokens
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
    call.window_left = arguments.window
    call.causal = 1 if arguments.mask == "causal" else 0
    call.dtype_query = F32
    call.dtype_key_value = F32
    call.dtype_output = F32

    stream = torch.cuda.current_stream().cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if status != 0:
        print(
            f"the entry refused the call: "
            f"{call.error.decode(errors='replace')}",
            file=sys.stderr,
        )
        return 1

    expected = reference(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        call.scale,
        arguments.mask == "causal",
        arguments.window,
    )
    absolute, relative = tolerance(int(lengths.max()))
    difference = (out.double() - expected).abs()
    allowed = absolute + relative * expected.abs()
    exceeded = int((difference > allowed).sum().item())
    print(
        f"mask {arguments.mask}, {arguments.sequences} sequence form, "
        f"window {arguments.window}: {batch} sequence(s), "
        f"{q_tokens} query rows over {kv_tokens} keys"
    )
    print(
        f"max |out - exact| = {difference.max().item():.6g}, "
        f"tolerance = {absolute:.6g} + {relative:.3g}·|out|, "
        f"largest value {expected.abs().max().item():.6g}"
    )
    if exceeded:
        crossing = (difference > allowed).nonzero()
        print(
            f"the FP32 pass exceeds the operator's tolerance at "
            f"{exceeded} element(s), "
            f"the first {tuple(crossing[0].tolist())}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
