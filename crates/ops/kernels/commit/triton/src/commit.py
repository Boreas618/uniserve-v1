"""Commit sampled tokens: the Triton source of the ``commit`` operator.

One program applies a batch's commit rows to per-slot continuation state: the
next input token, the continuation flag, the logical and sampling positions,
the verified cache length and the penalty counts. Slot 0 is padding
everywhere. Each row names its slot, a mode and, for sampled rows, the task
whose selection it installs.

The kernel body is the one UniServe committed with
(``uniserve/sampling/state.py``), unchanged except that it imports nothing but
Triton: the row table's column order, the selection columns and the slot state
stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count"])
def _commit_kernel(
    columns_ptr,
    tokens_ptr,
    valid_ptr,
    active_ptr,
    continuation_ptr,
    points_ptr,
    future_ptr,
    predicates_ptr,
    logical_ptr,
    sampling_ptr,
    cache_ptr,
    penalties_ptr,
    count,
    stride: tl.constexpr,
    future_stride: tl.constexpr,
    vocab: tl.constexpr,
    block: tl.constexpr,
):
    """Apply ``count`` commit rows in one program.

    Slots are unique, so each row's loads and stores of its slot need no
    synchronization. Penalty counts accumulate with an atomic add, as an
    indexed accumulation does.
    """
    offsets = tl.arange(0, block)
    mask = offsets < count
    slot = tl.load(columns_ptr + offsets, mask=mask, other=0)
    mode = tl.load(columns_ptr + stride + offsets, mask=mask, other=3)
    task = tl.load(columns_ptr + 2 * stride + offsets, mask=mask, other=-1)
    host_cache = tl.load(columns_ptr + 3 * stride + offsets, mask=mask)
    host_logical = tl.load(columns_ptr + 4 * stride + offsets, mask=mask)
    host_sampling = tl.load(columns_ptr + 5 * stride + offsets, mask=mask)
    visible = tl.load(columns_ptr + 6 * stride + offsets, mask=mask)
    penalties = tl.load(columns_ptr + 7 * stride + offsets, mask=mask)

    sampled = mask & (task >= 0)
    selected = tl.where(sampled, task, 0)
    token = tl.load(tokens_ptr + selected, mask=sampled, other=0)
    token = token & 0x7FFFFFFF
    continuation = tl.load(continuation_ptr + selected, mask=sampled)
    points = tl.load(points_ptr + selected, mask=sampled, other=0)
    weight = tl.load(valid_ptr + selected, mask=sampled, other=0) * tl.load(
        active_ptr + selected, mask=sampled, other=0
    )

    decode = mode == 0
    verify = mode == 2
    cache = tl.load(cache_ptr + slot, mask=mask, other=0)
    logical = tl.load(logical_ptr + slot, mask=mask, other=0)
    sampling = tl.load(sampling_ptr + slot, mask=mask, other=0)
    cache = tl.where(
        decode, cache + 1, tl.where(verify, visible + points, host_cache)
    )
    logical = tl.where(
        decode,
        logical + 1,
        tl.where(verify, host_logical + points, host_logical),
    )
    sampling = tl.where(
        decode,
        sampling + 1,
        tl.where(verify, host_sampling + points, host_sampling),
    )
    tl.store(cache_ptr + slot, cache.to(tl.int32), mask=mask)
    tl.store(future_ptr + slot * future_stride, token, mask=sampled)
    tl.store(predicates_ptr + slot, continuation, mask=sampled)
    tl.store(logical_ptr + slot, logical, mask=sampled)
    tl.store(sampling_ptr + slot, sampling, mask=sampled)
    tl.atomic_add(
        penalties_ptr + slot.to(tl.int64) * vocab + token,
        weight.to(tl.int32),
        mask=sampled & (penalties != 0),
    )
