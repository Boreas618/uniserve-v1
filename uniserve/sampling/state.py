"""Commit sampled tokens into per-slot request continuation state.

A request slot's continuation state is what the next call of the request
reads on the device without a host round trip: its next input token, its
continuation flag, its logical and sampling positions, its verified cache
length and its penalty counts. Slot 0 is padding everywhere.

One commit applies a batch's rows in one call. Each row names its slot, a
mode and, for sampled rows, the task whose selection it installs:

- ``DECODE``: advance by one sampled token on the device (position, sampling
  position and cache length plus one);
- ``EXPLICIT``: install a prefill's sampled token at host coordinates;
- ``VERIFY``: install a verification's emitted token, advancing host base
  coordinates by its device-selected emitted-token count;
- ``CACHE``: set only the cache length (a prefill without a sample).

Penalty rows count the installed token once, weighted by the selection's
validity and activity. Speculative verification therefore counts only its
final emitted token, as the request's penalty history always has.
"""

from __future__ import annotations

import torch
from uniserve_kernels.triton import launchable

from .select import TOKEN_VALUE_MASK

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

DECODE, EXPLICIT, VERIFY, CACHE = 0, 1, 2, 3

# Rows of the int64 commit column table: slot, mode, task (-1 without a
# sample), host cache length, host logical position, host sampling position,
# verification base visible length, penalties (0 or 1).
SLOT, MODE, TASK, CACHE_LENGTH, LOGICAL, SAMPLING, VISIBLE, PENALTIES = range(8)
COMMIT_FIELDS = 8

if triton is not None:

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


def commit(
    columns: torch.Tensor,
    count: int,
    *,
    tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    continuation: torch.Tensor,
    points: torch.Tensor,
    future_input_tokens: torch.Tensor,
    predicates: torch.Tensor,
    logical_lengths: torch.Tensor,
    sampling_positions: torch.Tensor,
    verified_lengths: torch.Tensor,
    penalty_counts: torch.Tensor,
) -> None:
    """Apply ``count`` commit rows of ``columns`` (``[8, capacity]`` int64).

    ``tokens``, ``valid``, ``active``, ``continuation`` and ``points`` are
    the batch's per-task selection columns, indexed by each row's task.
    The state tensors are indexed by slot: ``future_input_tokens`` ``[S,
    width]`` int64 (column 0 is the next input), ``predicates`` ``[S]`` uint8,
    ``logical_lengths`` and ``sampling_positions`` ``[S]`` int64,
    ``verified_lengths`` ``[S]`` int32 and ``penalty_counts`` ``[S, vocab]``
    int32. The caller guarantees unique slots.
    """
    count = int(count)
    if count == 0:
        return
    device = columns.device
    if triton is not None and device.type == "cuda" and launchable(device):
        _commit_kernel[(1,)](
            columns,
            tokens,
            valid,
            active,
            continuation,
            points,
            future_input_tokens,
            predicates,
            logical_lengths,
            sampling_positions,
            verified_lengths,
            penalty_counts,
            count,
            stride=int(columns.stride(0)),
            future_stride=int(future_input_tokens.stride(0)),
            vocab=int(penalty_counts.shape[1]),
            block=triton.next_power_of_2(count),
        )
        return

    rows = columns[:, :count]
    slot, mode, task = rows[SLOT], rows[MODE], rows[TASK]
    sampled = task >= 0
    selected = task.clamp_min(0)
    token = tokens.index_select(0, selected).to(torch.int64) & TOKEN_VALUE_MASK
    points = points.index_select(0, selected)
    decode, verify = mode == DECODE, mode == VERIFY
    cache = torch.where(
        decode,
        verified_lengths.index_select(0, slot).to(torch.int64) + 1,
        torch.where(verify, rows[VISIBLE] + points, rows[CACHE_LENGTH]),
    )
    logical = torch.where(
        decode,
        logical_lengths.index_select(0, slot) + 1,
        torch.where(verify, rows[LOGICAL] + points, rows[LOGICAL]),
    )
    sampling = torch.where(
        decode,
        sampling_positions.index_select(0, slot) + 1,
        torch.where(verify, rows[SAMPLING] + points, rows[SAMPLING]),
    )
    verified_lengths.index_copy_(0, slot, cache.to(verified_lengths.dtype))

    # Rows without a sample keep their token state.
    if bool(sampled.any()):
        slots = slot[sampled]
        installed = token[sampled]
        future_input_tokens[:, 0].index_copy_(0, slots, installed)
        flags = continuation.index_select(0, selected)[sampled]
        predicates.index_copy_(0, slots, flags.to(predicates.dtype))
        logical_lengths.index_copy_(0, slots, logical[sampled])
        sampling_positions.index_copy_(0, slots, sampling[sampled])
        counted = sampled & (rows[PENALTIES] != 0)
        if bool(counted.any()):
            weight = (
                valid.index_select(0, selected).bool()
                & active.index_select(0, selected).bool()
            ).to(penalty_counts.dtype)
            penalty_counts.index_put_(
                (slot[counted], token[counted]),
                weight[counted],
                accumulate=True,
            )


__all__ = [
    "CACHE",
    "COMMIT_FIELDS",
    "DECODE",
    "EXPLICIT",
    "VERIFY",
    "commit",
]
