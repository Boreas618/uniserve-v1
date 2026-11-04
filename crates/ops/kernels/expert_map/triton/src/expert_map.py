"""Expert id mapping: the Triton source of the ``expert_map`` operator.

One program maps a token's top-k logical expert ids to their physical experts:
``out[t, j] = logical_to_physical[ids[t, j]]``. UniServe evaluated this with
two ATen ops (``uniserve/nn/moe.py:287-288``); the mapping is one launch here,
so a captured graph replays it without host work.
"""

import triton
import triton.language as tl


@triton.jit
def _expert_map_kernel(ids_ptr, map_ptr, out_ptr, top: tl.constexpr, block: tl.constexpr):
    """Map one token's picks through the expert permutation.

    ``ids`` and ``out`` are ``[tokens, top]`` int32 and ``map`` is the
    ``[experts]`` int32 permutation of logical to physical experts. Grid:
    (tokens,). A pick past ``top`` is not read.
    """
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, block)
    mask = lanes < top
    logical = tl.load(ids_ptr + row * top + lanes, mask=mask, other=0)
    physical = tl.load(map_ptr + logical.to(tl.int64), mask=mask, other=0)
    tl.store(out_ptr + row * top + lanes, physical, mask=mask)
