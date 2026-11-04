"""Committing a sampling step's rows into per-slot request continuation state.

A request slot's continuation state is what the next call of the request reads
on the device, without a host round trip: its next input token, its
continuation predicate, its logical and sampling positions, its verified cache
length and its penalty counts. This program is that commit, one program for the
whole step, and it is the reference kernel's
(``uniserve/sampling/state.py::_commit_kernel``) with the descriptor naming the
state's element types.

Each row names its slot, a mode and, when it sampled, the task whose selection
it installs:

- ``DECODE``: advance by the sampled token — cache length, logical position and
  sampling position plus one;
- ``EXPLICIT``: install a prefill's sampled token at the host's coordinates;
- ``VERIFY``: install a verification's emitted token, taking the three lengths
  to the host base plus the device-selected emitted-token count;
- ``CACHE``: set only the cache length (a prefill without a sample).

Only rows that sampled write the next input token, the predicate and the
positions; every row writes the cache length. Penalty counts accumulate through
an atomic add, weighted by whether the selection is valid and active, so
speculation counts only its final emitted token. A row's token relay carries the
continuation in bit 31 and the token in the low 31 bits.

A token's slot is unique within a step, so a slot's state needs no
synchronization between rows.
"""

import triton
import triton.language as tl

# The commit table's rows and the modes' numbers, as the kernel reads them. They
# are constexprs because a kernel sees no other kind of module global; the plain
# names below are what a reader and the reference's module both use.
SLOT = tl.constexpr(0)
MODE = tl.constexpr(1)
TASK = tl.constexpr(2)
CACHE_LENGTH = tl.constexpr(3)
LOGICAL = tl.constexpr(4)
SAMPLING = tl.constexpr(5)
VISIBLE = tl.constexpr(6)
PENALTIES = tl.constexpr(7)

DECODE = tl.constexpr(0)
VERIFY = tl.constexpr(2)
CACHE = tl.constexpr(3)

# A token relay carries the continuation in bit 31 and the token below it.
TOKEN_VALUE_MASK = tl.constexpr((1 << 31) - 1)


@triton.jit(do_not_specialize=["count"])
def _slot_commit_kernel(
    columns,  # *index: [8, capacity] the step's rows, `STRIDE` apart
    tokens,  # *index: the sampled token of each task
    valid,  # *flags: whether a task's sample is valid
    active,  # *flags: whether a task is active in this step
    continuation,  # *u8: whether a task continues
    points,  # *index: the accepted span's length per task
    future,  # *index: [slots, width] the next input token goes to column 0
    predicates,  # *u8: receives each slot's continuation predicate
    logical,  # *index: each slot's logical position
    sampling,  # *index: each slot's sampling position
    verified,  # *verified: each slot's verified cache length
    penalties,  # *penalties: [slots, VOCAB] accepted tokens' penalty weights
    count,  # the step's rows
    INDEX: tl.constexpr,  # the column table's and the lengths' element type
    VERIFIED: tl.constexpr,  # the verified length's element type
    PENALTY_DTYPE: tl.constexpr,  # the penalty counts' element type
    STRIDE: tl.constexpr,  # elements between two columns of the table
    FUTURE_STRIDE: tl.constexpr,  # elements between two slots of the next input
    VOCAB: tl.constexpr,  # the penalty counts' width
    BLOCK: tl.constexpr,  # the lanes one program covers
):
    lanes = tl.arange(0, BLOCK)
    mask = lanes < count

    slot = tl.load(columns + SLOT * STRIDE + lanes, mask=mask, other=0)
    mode = tl.load(columns + MODE * STRIDE + lanes, mask=mask, other=CACHE)
    task = tl.load(columns + TASK * STRIDE + lanes, mask=mask, other=-1)
    host_cache = tl.load(columns + CACHE_LENGTH * STRIDE + lanes, mask=mask)
    host_logical = tl.load(columns + LOGICAL * STRIDE + lanes, mask=mask)
    host_sampling = tl.load(columns + SAMPLING * STRIDE + lanes, mask=mask)
    visible = tl.load(columns + VISIBLE * STRIDE + lanes, mask=mask)
    counts_penalty = tl.load(columns + PENALTIES * STRIDE + lanes, mask=mask)

    # A row without a sampled task reads task 0's relay and writes no token
    # state, so `selected` only has to stay inside the per-task vectors.
    sampled = mask & (task >= 0)
    selected = tl.where(sampled, task, 0)
    token = tl.load(tokens + selected, mask=sampled, other=0) & TOKEN_VALUE_MASK
    carries = tl.load(continuation + selected, mask=sampled)
    accepted = tl.load(points + selected, mask=sampled, other=0)
    # A selection counts when its task is both valid and active: the flags'
    # magnitudes do not matter, only whether they are set.
    selected_valid = tl.load(valid + selected, mask=sampled, other=0) != 0
    selected_active = tl.load(active + selected, mask=sampled, other=0) != 0
    weight = (selected_valid & selected_active).to(tl.int32)

    decode = mode == DECODE
    verify = mode == VERIFY
    cache = tl.load(verified + slot, mask=mask, other=0)
    position = tl.load(logical + slot, mask=mask, other=0)
    sampling_position = tl.load(sampling + slot, mask=mask, other=0)
    cache = tl.where(
        decode,
        cache + 1,
        tl.where(verify, (visible + accepted).to(VERIFIED), host_cache.to(VERIFIED)),
    )
    position = tl.where(
        decode, position + 1, tl.where(verify, host_logical + accepted, host_logical)
    )
    sampling_position = tl.where(
        decode,
        sampling_position + 1,
        tl.where(verify, host_sampling + accepted, host_sampling),
    )

    tl.store(verified + slot, cache.to(VERIFIED), mask=mask)
    tl.store(future + slot * FUTURE_STRIDE, token, mask=sampled)
    tl.store(predicates + slot, carries, mask=sampled)
    tl.store(logical + slot, position, mask=sampled)
    tl.store(sampling + slot, sampling_position, mask=sampled)
    # The penalty counts are the one field two rows could share: the add is
    # atomic so a step that installs two tokens of one slot to the same word
    # still sums both weights.
    tl.atomic_add(
        penalties + slot.to(tl.int64) * VOCAB + token,
        weight,
        mask=sampled & (counts_penalty != 0),
    )
