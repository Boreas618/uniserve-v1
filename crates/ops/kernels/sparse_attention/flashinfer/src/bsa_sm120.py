"""FlashInfer's SM120 block-sparse attention: the kernel this package compiles.

The kernel is FlashInfer's ``BlockSparseAttnForwardSm120Blk64``
(``flashinfer.cute_dsl.sparse.sm120_blk64.flash_fwd_sm120`` of the locked
``flashinfer-python`` distribution), a CuTe DSL kernel for SM120 and SM121:
64-row tiles in both the query and the key domain, grouped queries against
their selected key blocks, a 128-wide head. FlashInfer also ships a torch-shaped
wrapper that permutes tensors into the kernel's layouts and compiles on first
use; a module cannot take a torch caller, so the package constructs the kernel
class itself and compiles it for fake tensors in the kernel's own layouts,
which the operator's call already has:

```text
mQ, mK, mO   (rows, 128, heads, 1)        head_dim contiguous
mV           (128, key_rows, heads, 1)    head_dim contiguous
mLSE         (rows, heads, 1)             the row contiguous
q2k indices  (selected, query_tiles, heads, 1)  the selected block contiguous
q2k counts   (query_tiles, heads, 1)      the query tile contiguous
valid        (key_blocks)                 one entry per key block
```

The operator's ``SparseCall`` gives rows, heads and the key rows as packed
``[rows, heads, 128]`` views with explicit strides, so each of those tensors is
the kernel's layout with batch 1 and no copy.
"""

import cutlass
from flashinfer.cute_dsl.sparse.sm120_blk64.flash_fwd_sm120 import (
    BlockSparseAttnForwardSm120Blk64,
)

#: The module's name: the exported C wrapper and the module handle carry it.
NAME = "flash_attn_sparse_sm120"

#: The element types of the queries, keys, values and output, by the
#: operator's name.
DTYPES = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16}


def sparse_kernel(fields):
    """FlashInfer's SM120 kernel for one descriptor."""
    block = fields["block"]
    head_dim = fields["head_dim"]
    return BlockSparseAttnForwardSm120Blk64(
        # The operator's keys and values carry the queries' heads
        # (`ops::attention::sparse`): one key head per query head.
        gqa_ratio=1,
        head_dim=head_dim,
        value_dim=head_dim,
        blocksparse_blocksize_q=block,
        blocksparse_blocksize_k=block,
        dtype=DTYPES[fields["dtype"]],
        acc_dtype=cutlass.Float32,
        # The operator's call carries the valid rows of every key block, so the
        # kernel masks a partial block's padded rows; the counts are the
        # selection's per-tile widths.
        has_block_sizes=True,
        has_block_nums=True,
        block_sizes_mode=1,
    )
