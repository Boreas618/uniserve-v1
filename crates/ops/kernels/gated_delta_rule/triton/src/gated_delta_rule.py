"""The gated delta rule over rows that keep a recurrent state: the Triton source of the
``gated_delta_rule`` operator.

Per value head ``h`` (reading key head ``h // (value_heads / key_heads)``) and token ``t``:

    g = −exp(A_log[h]) · softplus(a[t, h] + dt_bias[h]),   β = sigmoid(b[t, h])
    q̂ = q · rsqrt(Σq² + 1e-6) · scale,   k̂ = k · rsqrt(Σk² + 1e-6)
    S ← exp(g) · S;   S ← S + k̂ ⊗ β (v − k̂ᵀS);   o = q̂ᵀS

with ``S`` the head's ``[key_dim, value_dim]`` state. Row ``r``'s new tokens are
``offsets[r] .. offsets[r + 1]`` at absolute positions ``start .. end`` with
``end = seq_lens[r]``; a row starting at position zero begins from a zero state,
any other from its slot's, and the slot keeps the state after the row's last
token. When ``start < checkpoints[r] ≤ end`` names a position the walk stops at,
the state after the token at ``checkpoints[r] − 1`` also goes into the slot of the
checkpoint bank. A negative slot is a padding row: zero output, no state.

A recurrent walk may also keep its **trail**: when ``trailing`` is set, the state after every
token ``t`` of the batch goes into row ``t`` of ``trail`` (``[tokens, value_heads, key_dim,
value_dim]``), so a caller that learns afterwards how many of a row's tokens stand (a speculative
verification) can take the slot's state after any of them. The chunked form keeps no trail.

One program owns ``BLOCK`` value columns of one value head of one row, so the
state it carries is ``key_dim × BLOCK`` FP32 and it never shares a state column
with another program. Two forms:

- ``CHUNK == 1`` walks the tokens one at a time (the decode step);
- ``CHUNK > 1`` walks chunks aligned to absolute multiples of ``CHUNK`` (the
  first one partial) and solves each in its WY representation, as FLA's
  ``chunk_gated_delta_rule`` and transformers' ``torch_chunk_gated_delta_rule``
  do: with ``Γ`` the chunk's cumulative log decay,

      A = strictly_lower(β k̂ k̂ᵀ ⊙ exp(Γᵢ − Γⱼ)),   T = (I + A)⁻¹   (block doubling)
      V' = T (β ⊙ V − (β exp(Γ) ⊙ K̂) S)            (the WY form's U − W S)
      O = (exp(Γ) ⊙ Q̂) S + lower(Q̂ K̂ᵀ ⊙ exp(Γᵢ − Γⱼ)) V'
      S ← exp(Γ_last) S + (exp(Γ_last − Γ) ⊙ K̂)ᵀ V'

  so a checkpoint is taken at a chunk's end: a multiple of ``CHUNK`` or the row's
  end, and no other position. A chunk of ``CHUNK`` tokens over ``BLOCK`` value
  columns holds ``2·CHUNK + 4·BLOCK`` tensor-memory columns on sm_100, which
  ``build.py`` checks against the 512 a multiprocessor has.

Every intermediate is FP32 and the products run at FP32 accuracy on the tensor
cores (``_product``: three TF32 passes into one accumulator); the output rounds
once.

The kernel body imports nothing but Triton: the operand checks and the launch
stay with the operator's Rust side.
"""

import triton
import triton.language as tl

#: The l2 normalization's floor under the square root, as transformers and FLA state it.
_L2_EPSILON = tl.constexpr(1e-6)

#: Above this argument the softplus is the identity, as `torch.nn.functional.softplus` has it.
_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)

#: The bits a TF32 operand keeps of an FP32 one: the sign, the exponent and ten mantissa bits.
_TF32_BITS = tl.constexpr(0xFFFFE000)


@triton.jit
def _product(left, right):
    """``left @ right`` of FP32 tiles at FP32 accuracy, on the tensor cores.

    Each operand is split into its TF32 head and the rest, and the three products that matter
    (head·head, rest·head, head·rest) accumulate into one sum: the dropped rest·rest term is
    below FP32's resolution. The three passes run through one ``tl.dot`` site, because every
    site holds its own tensor-memory accumulator on sm_100 and a chunk's products would
    otherwise outgrow it. (Triton's own ``ieee`` and ``tf32x3`` either fall back to one TF32
    pass for operands widened from a narrower load or need three accumulators.)
    """
    left_head = (left.to(tl.uint32, bitcast=True) & _TF32_BITS).to(tl.float32, bitcast=True)
    right_head = (right.to(tl.uint32, bitcast=True) & _TF32_BITS).to(tl.float32, bitcast=True)
    total = tl.zeros((left.shape[0], right.shape[1]), dtype=tl.float32)
    for part in range(3):
        a = tl.where(part == 1, left - left_head, left_head)
        b = tl.where(part == 2, right - right_head, right_head)
        total = tl.dot(a, b, total, input_precision="tf32")
    return total


@triton.jit
def _decay_and_step(a, b, a_log, dt_bias):
    """``(g, β)`` of FP32 projections ``a`` and ``b`` for one head's FP32 ``A_log``, ``dt_bias``."""
    x = a + dt_bias
    # `log(1 + exp(x))` without overflow; above the threshold torch answers `x` itself.
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    softplus = tl.where(x > _SOFTPLUS_THRESHOLD, x, softplus)
    g = -tl.exp(a_log) * softplus
    beta = 1.0 / (1.0 + tl.exp(-b))
    return g, beta


@triton.jit
def _normalized(rows, axis: tl.constexpr):  # noqa: N803
    """``rows`` divided by their l2 norms along ``axis``, with the floor under the root."""
    return rows * tl.rsqrt(tl.sum(rows * rows, axis=axis, keep_dims=True) + _L2_EPSILON)


@triton.jit
def _recurrent(
    query_ptr,
    key_ptr,
    value_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    checkpoint_ptr,
    trail_ptr,
    trailing,
    state,
    first,
    count,
    start,
    checkpoint,
    head,
    key_head,
    columns,
    bank,
    a_log,
    dt_bias,
    row_stride,
    gate_stride,
    scale,
    VALUE_HEADS: tl.constexpr,  # noqa: N803
    KEY_DIM: tl.constexpr,  # noqa: N803
    VALUE_DIM: tl.constexpr,  # noqa: N803
):
    """Walk ``count`` tokens one at a time from ``state``; answer the state after the last."""
    keys = tl.arange(0, KEY_DIM)
    # This program's columns of one token's `[key_dim, value_dim]` state of this head in the trail.
    within = keys[:, None] * VALUE_DIM + columns[None, :]
    for index in range(0, count):
        token = (first + index).to(tl.int64)
        q = tl.load(query_ptr + token * row_stride + key_head * KEY_DIM + keys).to(tl.float32)
        k = tl.load(key_ptr + token * row_stride + key_head * KEY_DIM + keys).to(tl.float32)
        v = tl.load(value_ptr + token * row_stride + head * VALUE_DIM + columns).to(tl.float32)
        a = tl.load(a_ptr + token * gate_stride + head).to(tl.float32)
        b = tl.load(b_ptr + token * gate_stride + head).to(tl.float32)
        g, beta = _decay_and_step(a, b, a_log, dt_bias)

        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + _L2_EPSILON) * scale
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + _L2_EPSILON)
        state = state * tl.exp(g)
        recalled = tl.sum(k[:, None] * state, axis=0)
        delta = (v - recalled) * beta
        state = state + k[:, None] * delta[None, :]
        out = tl.sum(q[:, None] * state, axis=0)
        tl.store(out_ptr + (token * VALUE_HEADS + head) * VALUE_DIM + columns, out)

        if checkpoint == start + index + 1:
            tl.store(checkpoint_ptr + bank, state)
        if trailing != 0:
            tl.store(trail_ptr + (token * VALUE_HEADS + head) * (KEY_DIM * VALUE_DIM) + within, state)
    return state


@triton.jit
def _chunked(
    query_ptr,
    key_ptr,
    value_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    checkpoint_ptr,
    state,
    first,
    start,
    end,
    checkpoint,
    head,
    key_head,
    columns,
    bank,
    a_log,
    dt_bias,
    row_stride,
    gate_stride,
    scale,
    VALUE_HEADS: tl.constexpr,  # noqa: N803
    KEY_DIM: tl.constexpr,  # noqa: N803
    VALUE_DIM: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    CHUNK: tl.constexpr,  # noqa: N803
    LEVELS: tl.constexpr,  # noqa: N803
):
    """Walk the row's chunks from ``state``; answer the state after the last token."""
    keys = tl.arange(0, KEY_DIM)
    lanes = tl.arange(0, CHUNK)
    lower = lanes[:, None] >= lanes[None, :]
    strictly_lower = lanes[:, None] > lanes[None, :]
    identity = tl.where(lanes[:, None] == lanes[None, :], 1.0, 0.0)

    aligned = start // CHUNK * CHUNK
    for base in range(aligned, end, CHUNK):
        position = base + lanes
        valid = (position >= start) & (position < end)
        token = (first + position - start).to(tl.int64)

        q = tl.load(
            query_ptr + token[:, None] * row_stride + key_head * KEY_DIM + keys[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            key_ptr + token[:, None] * row_stride + key_head * KEY_DIM + keys[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            value_ptr + token[:, None] * row_stride + head * VALUE_DIM + columns[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        a = tl.load(a_ptr + token * gate_stride + head, mask=valid, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + token * gate_stride + head, mask=valid, other=0.0).to(tl.float32)
        g, beta = _decay_and_step(a, b, a_log, dt_bias)
        # A position outside the row neither decays nor writes: it is the identity step.
        g = tl.where(valid, g, 0.0)
        beta = tl.where(valid, beta, 0.0)

        q = _normalized(q, 1) * scale
        k = _normalized(k, 1)
        gamma = tl.cumsum(g, axis=0)
        last = tl.sum(tl.where(lanes == CHUNK - 1, gamma, 0.0), axis=0)
        # exp(Γᵢ − Γⱼ) on and below the diagonal, where Γᵢ ≤ Γⱼ; zero above it, never overflowing.
        gap = tl.where(lower, gamma[:, None] - gamma[None, :], 0.0)
        decay = tl.where(lower, tl.exp(gap), 0.0)

        # Every `tl.dot` site holds its own tensor-memory accumulator on sm_100, so products of one
        # shape share a site through a loop of two: `K̂K̂ᵀ` and `Q̂K̂ᵀ`; `(βe^Γ K̂) S` and `(e^Γ Q̂) S`.
        keys_t = tl.trans(k)
        kk = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        scores = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        for which in range(2):
            rows = tl.where(which == 0, k, q)
            product = _product(rows, keys_t)
            kk = tl.where(which == 0, product, kk)
            scores = tl.where(which == 1, product, scores)
        scores = scores * decay
        recalled = tl.zeros((CHUNK, BLOCK), dtype=tl.float32)
        out = tl.zeros((CHUNK, BLOCK), dtype=tl.float32)
        for which in range(2):
            rows = tl.where(which == 0, k * (beta * tl.exp(gamma))[:, None], q * tl.exp(gamma)[:, None])
            product = _product(rows, state)
            recalled = tl.where(which == 0, product, recalled)
            out = tl.where(which == 1, product, out)

        # T = (I + A)⁻¹ by doubling: with D the inverse of the diagonal blocks of width w and O the
        # coupling of each pair of them in A, the inverse of the blocks of width 2w is D − D O D,
        # exactly, since O D O vanishes. Six levels from the identity reach the whole chunk.
        coupling = tl.where(strictly_lower, beta[:, None] * kk * decay, 0.0)
        inverse = identity
        partial = identity
        for step in range(2 * LEVELS):
            width = 1 << (step // 2)
            within = (lanes[:, None] // (2 * width)) == (lanes[None, :] // (2 * width))
            pair = within & ((lanes[:, None] // width) % 2 == 1) & ((lanes[None, :] // width) % 2 == 0)
            even = step % 2 == 0
            left = tl.where(even, inverse, partial)
            right = tl.where(even, tl.where(pair, coupling, 0.0), inverse)
            product = _product(left, right)
            partial = tl.where(even, product, partial)
            inverse = tl.where(even, inverse, inverse - product)

        # V' = T (βV − (βe^Γ K̂) S): the new values with what the carried state already predicts
        # taken out, which is `U − W S` of the WY form with `T` applied once.
        fresh = _product(inverse, v * beta[:, None] - recalled)
        out += _product(scores, fresh)
        tl.store(
            out_ptr + (token[:, None] * VALUE_HEADS + head) * VALUE_DIM + columns[None, :],
            out,
            mask=valid[:, None],
        )

        carried = k * tl.exp(last - gamma)[:, None]
        state = state * tl.exp(last) + _product(tl.trans(carried), fresh)
        if checkpoint == tl.minimum(base + CHUNK, end):
            tl.store(checkpoint_ptr + bank, state)
    return state


@triton.jit(do_not_specialize=["row_stride", "gate_stride", "trailing"])
def _gated_delta_rule_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    offsets_ptr,
    seq_lens_ptr,
    slots_ptr,
    checkpoints_ptr,
    state_ptr,
    checkpoint_ptr,
    trail_ptr,
    out_ptr,
    row_stride,
    gate_stride,
    scale,
    trailing,
    KEY_HEADS: tl.constexpr,  # noqa: N803
    VALUE_HEADS: tl.constexpr,  # noqa: N803
    KEY_DIM: tl.constexpr,  # noqa: N803
    VALUE_DIM: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
    CHUNK: tl.constexpr,  # noqa: N803
    LEVELS: tl.constexpr,  # noqa: N803
):
    """Advance ``BLOCK`` value columns of one value head of one row.

    Grid: ``(VALUE_DIM / BLOCK, VALUE_HEADS, rows)``.
    """
    block = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.program_id(2)
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    keys = tl.arange(0, KEY_DIM)

    first = tl.load(offsets_ptr + row)
    count = tl.load(offsets_ptr + row + 1) - first
    end = tl.load(seq_lens_ptr + row)
    start = end - count
    slot = tl.load(slots_ptr + row)
    checkpoint = tl.load(checkpoints_ptr + row)

    if slot < 0:
        # A padding row: finite zeros in its output, and no state read or written.
        for index in range(0, count):
            token = (first + index).to(tl.int64)
            tl.store(
                out_ptr + (token * VALUE_HEADS + head) * VALUE_DIM + columns,
                tl.zeros((BLOCK,), dtype=tl.float32),
            )
    else:
        # This program's columns of the slot's `[key_dim, value_dim]` state of this head.
        bank = (
            (slot.to(tl.int64) * VALUE_HEADS + head) * (KEY_DIM * VALUE_DIM)
            + keys[:, None] * VALUE_DIM
            + columns[None, :]
        )
        if start > 0:
            state = tl.load(state_ptr + bank)
        else:
            state = tl.zeros((KEY_DIM, BLOCK), dtype=tl.float32)
        a_log = tl.load(a_log_ptr + head).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)

        if CHUNK == 1:
            state = _recurrent(
                query_ptr,
                key_ptr,
                value_ptr,
                a_ptr,
                b_ptr,
                out_ptr,
                checkpoint_ptr,
                trail_ptr,
                trailing,
                state,
                first,
                count,
                start,
                checkpoint,
                head,
                key_head,
                columns,
                bank,
                a_log,
                dt_bias,
                row_stride,
                gate_stride,
                scale,
                VALUE_HEADS,
                KEY_DIM,
                VALUE_DIM,
            )
        else:
            state = _chunked(
                query_ptr,
                key_ptr,
                value_ptr,
                a_ptr,
                b_ptr,
                out_ptr,
                checkpoint_ptr,
                state,
                first,
                start,
                end,
                checkpoint,
                head,
                key_head,
                columns,
                bank,
                a_log,
                dt_bias,
                row_stride,
                gate_stride,
                scale,
                VALUE_HEADS,
                KEY_DIM,
                VALUE_DIM,
                BLOCK,
                CHUNK,
                LEVELS,
            )
        # The thread that stores a state element need not be the one that loaded it: every load of
        # the slot's state lands before any store of the new one.
        tl.debug_barrier()
        tl.store(state_ptr + bank, state)
