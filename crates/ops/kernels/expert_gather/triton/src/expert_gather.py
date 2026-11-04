"""Expert id mapping: the Triton source of the ``expert_gather`` operator.

One program maps a block of a token's top-k logical expert ids to their
physical experts: ``out[i] = map[ids[i]]``. UniServe evaluated this with two
ATen ops (``uniserve/nn/moe.py:287-288``); the mapping is one launch here, so a
captured graph replays it without host work.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["entries"])
def _expert_gather_kernel(
    ids_ptr,
    map_ptr,
    out_ptr,
    entries,
    BLOCK: tl.constexpr,
):
    """Map one block of expert picks through the expert permutation.

    ``ids`` and ``out`` are ``[tokens, top]`` int32 flattened over
    ``entries = tokens · top``, ``map`` is the ``[experts]`` int32 permutation of
    logical to physical experts. Grid: ``ceil(entries / BLOCK)``.
    """
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < entries
    logical = tl.load(ids_ptr + offsets, mask=mask, other=0).to(tl.int64)
    physical = tl.load(map_ptr + logical, mask=mask, other=-1)
    tl.store(out_ptr + offsets, physical, mask=mask)
