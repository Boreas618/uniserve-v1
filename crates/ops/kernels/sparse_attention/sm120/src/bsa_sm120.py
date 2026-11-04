"""FlashInfer's SM120 block-sparse attention: what this package compiles and exports.

The kernel is FlashInfer's `BlockSparseAttnForwardSm120Blk64` — a CuTe DSL kernel for SM120/SM121,
64-row tiles in both the query and the key domain, grouped queries against their keys, GQA, a
128-wide head. It is not a kernel source this package owns: FlashInfer ships the kernel and a
torch-shaped wrapper (`bsa_attn_sm120_blk64_fwd`) that validates tensors, permutes them into the
kernel's layouts and compiles it on first use. A Program Server module cannot take a torch caller,
so this module drives the kernel once at build time and keeps what it produced:

1. describe the call with fake tensors *on the CPU* — the kernel only reads shapes, strides and
   layouts at compile time, so no device is opened and no kernel runs;
2. compile it the plain CuTe DSL way and export it to C.

The layouts are the kernel's own, and the operator's call already has them:

```text
mQ, mK, mO   (rows, 128, heads, 1)        head_dim contiguous
mV           (128, key_rows, heads, 1)    head_dim contiguous
mLSE         (rows, heads, 1)             the row contiguous
q2k indices  (selected, query_tiles, heads, 1)   the selected block id contiguous
q2k counts   (query_tiles, heads, 1)      the query tile contiguous
valid        (key_blocks)                 one entry per key block
```

Our `SparseCall` gives rows, heads and the key rows as packed `[rows, heads, 128]` views with
explicit strides, so each of those tensors is the kernel's layout with batch 1 and no copy.
"""

import torch

import cutlass
import cutlass.cute as cute
from torch._subclasses.fake_tensor import FakeTensorMode

TORCH_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}

#: The kernel's tile in both the query and the key domain.
BLOCK = 64

#: The module's name: the exported C wrapper and the module handle carry it.
NAME = "flash_attn_sparse_sm120"


def cute_tensor(fake, assumed_align, leading_dim):
    """A cute tensor over a fake tensor's layout, with the leading dimension fixed.

    The vendor's own conversion, without its tvm-ffi layer: the compiled function then takes a
    pointer, the extents a module may vary and the strides that are not fixed.
    """
    from flashinfer.cute_dsl.sparse.bsa_utils.cute_tensor_utils import to_cute_tensor

    return to_cute_tensor(
        fake, assumed_align=assumed_align, leading_dim=leading_dim, enable_tvm_ffi=False
    )


class BsaCompiler:
    """One compilation of FlashInfer's SM120 block-sparse kernel."""

    def __init__(self, fields, parameters, target):
        if fields["block"] != str(BLOCK):
            raise SystemExit(
                f"the SM120 kernel's tile is {BLOCK} rows in both domains; "
                f"this descriptor names {fields['block']}"
            )
        if fields["head_dim"] != "128":
            raise SystemExit("the SM120 kernel is compiled for 128 elements per head")
        if fields["lse"] != "true":
            raise SystemExit(
                "the SM120 kernel always writes the softmax normalizer, so it needs the "
                "launch's normalizer buffer"
            )
        self.dtype = TORCH_DTYPES[fields["dtype"]]
        self.heads = int(parameters.get("heads", 8))
        self.kv_heads = int(parameters.get("kv_heads", 8))
        self.query_tiles = int(parameters.get("query_tiles", 4))
        self.key_blocks = int(parameters.get("key_blocks", 8))
        self.selected = int(parameters.get("selected", 3))
        self.target = target

    def compile(self):
        """Compile the kernel for one representative call, to C."""
        heads, kv_heads = self.heads, self.kv_heads
        query_rows = self.query_tiles * BLOCK
        key_rows = self.key_blocks * BLOCK
        with FakeTensorMode():
            # A CPU device: only the layouts matter at compile time, and a CUDA fake tensor would
            # ask torch for a device that no build host has.
            device = "cpu"
            query = torch.empty(query_rows, heads, 128, dtype=self.dtype, device=device)
            key = torch.empty(key_rows, kv_heads, 128, dtype=self.dtype, device=device)
            value = torch.empty(key_rows, kv_heads, 128, dtype=self.dtype, device=device)
            out = torch.empty(query_rows, heads, 128, dtype=self.dtype, device=device)
            lse = torch.empty(heads, query_rows, dtype=torch.float32, device=device)
            indices = torch.zeros(
                heads, self.query_tiles, self.selected, dtype=torch.int32, device=device
            )
            counts = torch.full(
                (heads, self.query_tiles), self.selected, dtype=torch.int32, device=device
            )
            valid = torch.full((self.key_blocks,), BLOCK, dtype=torch.int32, device=device)

            # The tensors are described in the operator's own layouts — a packed
            # `[rows, heads, 128]`, the selection `[heads, query_tiles, selected]`, the normalizer
            # `[heads, rows]` — and viewed as the kernel wants them: batch 1 makes each of those a
            # permutation of the kernel's layout, so no call copies anything.
            to_qk = lambda tensor: tensor.unsqueeze(-1).permute(0, 2, 1, 3)  # noqa: E731
            to_v = lambda tensor: tensor.unsqueeze(-1).permute(2, 0, 1, 3)  # noqa: E731
            to_lse = lambda tensor: tensor.unsqueeze(-1).permute(1, 0, 2)  # noqa: E731
            to_indices = lambda tensor: tensor.permute(2, 1, 0).unsqueeze(-1)  # noqa: E731
            to_counts = lambda tensor: tensor.permute(1, 0).unsqueeze(-1)  # noqa: E731

            args = (
                cute_tensor(to_qk(query), 128, 1),
                cute_tensor(to_qk(key), 128, 1),
                cute_tensor(to_v(value), 128, 0),
                cute_tensor(to_qk(out), 128, 1),
                cute_tensor(to_lse(lse), 4, 0),
                cute_tensor(to_indices(indices), None, 0),
                cute_tensor(to_counts(counts), None, 0),
                cutlass.Int32(self.selected),
                cute_tensor(valid, None, 0),
                cutlass.Float32(0.08838834764831845),
                cute.runtime.make_fake_stream(),
            )

        from flashinfer.cute_dsl.sparse.sm120_blk64.flash_fwd_sm120 import (
            BlockSparseAttnForwardSm120Blk64,
        )

        kernel = BlockSparseAttnForwardSm120Blk64(
            gqa_ratio=heads // kv_heads,
            head_dim=128,
            value_dim=128,
            blocksparse_blocksize_q=BLOCK,
            blocksparse_blocksize_k=BLOCK,
            dtype=cutlass.BFloat16 if self.dtype == torch.bfloat16 else cutlass.Float16,
            acc_dtype=cutlass.Float32,
            # The operator's call carries the valid rows of every key block, so the kernel masks a
            # partial block's padded rows; the counts are the selection's per-tile widths.
            has_block_sizes=True,
            has_block_nums=True,
            block_sizes_mode=1,
        )
        return cute.compile(kernel, *args)
