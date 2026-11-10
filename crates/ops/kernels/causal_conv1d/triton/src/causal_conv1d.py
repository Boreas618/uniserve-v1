"""Short causal convolution with state: the Triton source of the ``causal_conv1d`` operator.

A gated delta network mixes each channel of its projected queries, keys and
values through ``K`` taps of a depthwise causal convolution and a SiLU:

    out[t, c] = silu(Σ_j w[c, j] · x[t − (K − 1) + j, c])        j < K

over a batch of rows that continue from their request slots' last inputs. Row
``r``'s new tokens are ``x[offsets[r] .. offsets[r + 1]]`` at absolute positions
``start .. end`` with ``end = seq_lens[r]``; an input before ``start`` is the
slot's state (``[slots, K − 1, channels]``, oldest first) when ``start > 0`` and
zero otherwise. After the row the slot's state holds its last ``K − 1`` inputs,
and a row whose ``checkpoints[r]`` falls in ``(start, end]`` also writes the
``K − 1`` inputs before that position into its slot of the checkpoint bank. A row
of a negative slot is padding: its output rows are zero and no state is touched.

When ``trailing`` is set the row also keeps its **trail**: the ``K − 1`` inputs before each of its
positions ``start + 1 .. end`` go into row ``t`` of ``trail`` (``[tokens, K − 1, channels]``) for
the row's token ``t``, the state the slot would hold had the row ended at that token, so a caller
that learns afterwards how many of the row's tokens stand (a speculative verification) can take the
slot's state after any of them.

One program owns ``BLOCK`` channels of one row and walks its tokens in tiles of
``TOKENS``; each output reads its ``K`` inputs straight from the row's inputs or
the slot's state, so a program reads every state value it needs before it
writes the row's new state. The taps and the sum are FP32 and the activated
value rounds once.

The kernel body imports nothing but Triton: the operand checks and the launch
stay with the operator's Rust side.
"""

import triton
import triton.language as tl


@triton.jit
def _inputs(
    x_ptr,
    state_ptr,
    positions,
    valid,
    channels,
    live,
    first,
    start,
    slot,
    x_row_stride,
    CHANNELS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
):
    """The FP32 inputs at absolute ``positions`` (``[P]``) of the program's ``channels``.

    ``first`` is the row's first token in ``x``; a position at or after ``start`` is the
    row's own input, an earlier one the slot's state, and before a row with no history
    (``start == 0``) zero. ``valid`` masks positions that hold no input at all.
    """
    own = valid & (positions >= start)
    rows = (first + positions - start).to(tl.int64)
    taken = tl.load(
        x_ptr + rows[:, None] * x_row_stride + channels[None, :],
        mask=own[:, None] & live[None, :],
        other=0.0,
    ).to(tl.float32)
    kept = valid & (positions < start) & (start > 0)
    # The state's entry `j` is the input at `start − (K − 1) + j`.
    entry = (positions - start + (WIDTH - 1)).to(tl.int64)
    remembered = tl.load(
        state_ptr
        + slot.to(tl.int64) * ((WIDTH - 1) * CHANNELS)
        + entry[:, None] * CHANNELS
        + channels[None, :],
        mask=kept[:, None] & live[None, :],
        other=0.0,
    ).to(tl.float32)
    return taken + remembered


@triton.jit(do_not_specialize=["x_row_stride", "trailing"])
def _causal_conv1d_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    offsets_ptr,
    seq_lens_ptr,
    slots_ptr,
    checkpoints_ptr,
    state_ptr,
    checkpoint_ptr,
    trail_ptr,
    x_row_stride,
    trailing,
    CHANNELS: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    KEPT: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    TOKENS: tl.constexpr,  # noqa: N803
):
    """Convolve ``BLOCK`` channels of one row.

    Grid: ``(ceil(CHANNELS / BLOCK), rows)``. ``KEPT`` is ``K − 1`` rounded up to a
    power of two, the lanes of one state tile.
    """
    block = tl.program_id(0)
    row = tl.program_id(1)
    channels = block * BLOCK + tl.arange(0, BLOCK)
    live = channels < CHANNELS

    first = tl.load(offsets_ptr + row)
    count = tl.load(offsets_ptr + row + 1) - first
    end = tl.load(seq_lens_ptr + row)
    start = end - count
    slot = tl.load(slots_ptr + row)
    checkpoint = tl.load(checkpoints_ptr + row)

    tile = tl.arange(0, TOKENS)
    if slot < 0:
        # A padding row: finite zeros in its output, and no state read or written.
        for base in range(0, count, TOKENS):
            token = base + tile
            rows = (first + token).to(tl.int64)
            tl.store(
                out_ptr + rows[:, None] * CHANNELS + channels[None, :],
                tl.zeros((TOKENS, BLOCK), dtype=tl.float32),
                mask=(token < count)[:, None] & live[None, :],
            )
    else:
        for base in range(0, count, TOKENS):
            token = base + tile
            valid = token < count
            position = start + token
            total = tl.zeros((TOKENS, BLOCK), dtype=tl.float32)
            for tap in tl.static_range(WIDTH):
                weight = tl.load(
                    weight_ptr + channels * WIDTH + tap, mask=live, other=0.0
                ).to(tl.float32)
                inputs = _inputs(
                    x_ptr,
                    state_ptr,
                    position - (WIDTH - 1) + tap,
                    valid,
                    channels,
                    live,
                    first,
                    start,
                    slot,
                    x_row_stride,
                    CHANNELS,
                    WIDTH,
                )
                total += inputs * weight[None, :]
            activated = total / (1.0 + tl.exp(-total))
            rows = (first + token).to(tl.int64)
            tl.store(
                out_ptr + rows[:, None] * CHANNELS + channels[None, :],
                activated,
                mask=valid[:, None] & live[None, :],
            )

        entry = tl.arange(0, KEPT)
        held = entry < WIDTH - 1
        bank = slot.to(tl.int64) * ((WIDTH - 1) * CHANNELS)
        stored = bank + entry[:, None].to(tl.int64) * CHANNELS + channels[None, :]
        # The checkpoint and the new state may both read the slot's old state, and the thread that
        # stores an entry is not the one that read it: every read lands before the barrier and
        # every store after it.
        keeps = (checkpoint > start) & (checkpoint <= end)
        kept = _inputs(
            x_ptr,
            state_ptr,
            checkpoint - (WIDTH - 1) + entry,
            held & keeps,
            channels,
            live,
            first,
            start,
            slot,
            x_row_stride,
            CHANNELS,
            WIDTH,
        )
        last = _inputs(
            x_ptr,
            state_ptr,
            end - (WIDTH - 1) + entry,
            held,
            channels,
            live,
            first,
            start,
            slot,
            x_row_stride,
            CHANNELS,
            WIDTH,
        )
        # The trail reads the slot's old state too, so it lands before the barrier as well: the
        # window after token `t` is the `K − 1` inputs ending at its position.
        if trailing != 0:
            for index in range(0, count):
                window = _inputs(
                    x_ptr,
                    state_ptr,
                    start + index + 1 - (WIDTH - 1) + entry,
                    held,
                    channels,
                    live,
                    first,
                    start,
                    slot,
                    x_row_stride,
                    CHANNELS,
                    WIDTH,
                )
                trailed = (
                    (first + index).to(tl.int64) * ((WIDTH - 1) * CHANNELS)
                    + entry[:, None].to(tl.int64) * CHANNELS
                    + channels[None, :]
                )
                tl.store(trail_ptr + trailed, window, mask=held[:, None] & live[None, :])
        tl.debug_barrier()
        tl.store(checkpoint_ptr + stored, kept, mask=(held & keeps)[:, None] & live[None, :])
        tl.store(state_ptr + stored, last, mask=held[:, None] & live[None, :])
