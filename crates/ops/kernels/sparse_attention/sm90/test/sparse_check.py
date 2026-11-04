#!/usr/bin/env python3
"""Run one BF16 block-sparse attention launch through the built sm90 module and check it.

This is the GPU check a build host cannot do: it loads the module the store holds exactly
as a runner does (``module.so``), calls the operator's entry with the call structure of
``crates/ops/src/attention/sparse.rs`` (704 bytes, ``error`` at 188), and compares the
module's output with the operator's own definition computed in torch.

    python sparse_check.py <module_dir> [--block 128] [--keys 512] [--heads 8]

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding ``module.so``.
The block size must be the one the module was compiled for: this backend's kernel tile is
128 rows in both directions. The normalizer is not requested, because this backend's kernel
writes none. Run it on the GB200 with the
Program Server's own environment (``crates/program-server/python/.venv/bin/python``),
which has torch.
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
    """`ops::attention::sparse::SparseCall`, field for field."""

    _fields_ = [
        ("query", ctypes.c_void_p),
        ("key", ctypes.c_void_p),
        ("value", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("lse", ctypes.c_void_p),
        ("indices", ctypes.c_void_p),
        ("counts", ctypes.c_void_p),
        ("valid", ctypes.c_void_p),
        ("heads", ctypes.c_int64),
        ("query_rows", ctypes.c_int64),
        ("key_rows", ctypes.c_int64),
        ("q_row_stride", ctypes.c_int64),
        ("q_head_stride", ctypes.c_int64),
        ("o_row_stride", ctypes.c_int64),
        ("o_head_stride", ctypes.c_int64),
        ("k_row_stride", ctypes.c_int64),
        ("k_head_stride", ctypes.c_int64),
        ("v_row_stride", ctypes.c_int64),
        ("v_head_stride", ctypes.c_int64),
        ("selected", ctypes.c_int64),
        ("head_dim", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("output_dtype", ctypes.c_int64),
        ("scale", ctypes.c_float),
        ("error", ctypes.c_char * 512),
    ]


def reference(query, key, value, indices, counts, valid, scale, block):
    """The operator's definition, in FP32 over the whole device: out, and log2's base change."""
    heads, query_rows, head_dim = query.shape
    query_tiles = query_rows // block
    out = torch.zeros(query_rows, heads, head_dim, dtype=torch.float32, device=query.device)
    lse = torch.full((heads, query_rows), float("-inf"), dtype=torch.float32, device=query.device)
    for head in range(heads):
        rows = query[:, head].float()
        keys = key[:, head].float()
        values = value[:, head].float()
        for tile in range(query_tiles):
            blocks = indices[head, tile, : int(counts[head, tile])].long()
            # The visible key rows of the selection: whole blocks, cut to each block's valid
            # count, which is what the kernel's masking reads.
            visible = torch.cat(
                [
                    block_id * block + torch.arange(int(valid[block_id]), device=query.device)
                    for block_id in blocks
                ]
            ) if blocks.numel() else torch.empty(0, dtype=torch.long, device=query.device)
            start = tile * block
            stop = start + block
            logits = rows[start:stop] @ keys[visible].transpose(0, 1) * scale
            weights = torch.softmax(logits, dim=-1) if visible.numel() else None
            if weights is not None:
                out[start:stop, head] = weights @ values[visible]
                top = logits.max(dim=-1).values
                lse[head, start:stop] = top + torch.log(torch.exp(logits - top[:, None]).sum(-1))
    return out, lse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=pathlib.Path, help="the module directory resolve hands a runner")
    parser.add_argument("--block", type=int, default=64, help="the module's compiled sparse block")
    parser.add_argument("--keys", type=int, default=512, help="key rows, a whole number of blocks")
    parser.add_argument("--tiles", type=int, default=4, help="query tiles")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--selected", type=int, default=3, help="the widest selection")
    parser.add_argument("--head-dim", type=int, default=128)
    arguments = parser.parse_args()

    library = ctypes.CDLL(str(arguments.module / "module.so"))
    entry = library.uniserve_sparse_attention
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    device = torch.device("cuda")
    torch.manual_seed(0)
    block, heads, head_dim = arguments.block, arguments.heads, arguments.head_dim
    query_rows = arguments.tiles * block
    key_rows = arguments.keys
    key_blocks = key_rows // block
    selected = arguments.selected

    query = torch.randn(query_rows, heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    key = torch.randn(key_rows, heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    value = torch.randn(key_rows, heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    out = torch.zeros(query_rows, heads, head_dim, dtype=torch.bfloat16, device=device)

    # A selection that keeps every block id in range, and a validity array with a partial
    # trailing block so the masking is exercised.
    indices = torch.randint(0, key_blocks, (heads, arguments.tiles, selected), device=device)
    indices = indices.to(torch.int32)
    counts = torch.full((heads, arguments.tiles), selected, dtype=torch.int32, device=device)
    counts[:, 0, 0] = 1  # one tile reads a single block
    valid = torch.full((key_blocks,), block, dtype=torch.int32, device=device)
    valid[-1] = block // 2

    call = Call()
    call.query = pointer(query)
    call.key = pointer(key)
    call.value = pointer(value)
    call.out = pointer(out)
    call.lse = 0  # this backend's kernel writes no normalizer
    call.indices = pointer(indices)
    call.counts = pointer(counts)
    call.valid = pointer(valid)
    call.heads = heads
    call.query_rows = query_rows
    call.key_rows = key_rows
    call.q_row_stride = heads * head_dim
    call.q_head_stride = head_dim
    call.o_row_stride = heads * head_dim
    call.o_head_stride = head_dim
    call.k_row_stride = heads * head_dim
    call.k_head_stride = head_dim
    call.v_row_stride = heads * head_dim
    call.v_head_stride = head_dim
    call.selected = selected
    call.head_dim = head_dim
    call.dtype = BF16
    call.output_dtype = BF16
    call.scale = 1.0 / math.sqrt(head_dim)

    stream = torch.cuda.current_stream().cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if status != 0:
        print(f"the entry refused the call: {call.error.decode(errors='replace')}", file=sys.stderr)
        return 1

    expected, _ = reference(query, key, value, indices, counts, valid, call.scale, block)
    error = (out.float() - expected).abs().max().item()
    largest = expected.abs().max().item()
    print(f"max |out - exact| = {error:.6g} (largest value {largest:.6g})")
    if error > 0.05 * max(largest, 1.0):
        print("the block-sparse pass does not match the operator's definition", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
