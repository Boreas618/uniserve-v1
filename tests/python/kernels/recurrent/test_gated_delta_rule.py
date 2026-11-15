"""The gated_delta_rule program over a batch of rows against FP64.

A batch is what the operator's rows are: packed rows of different lengths,
some starting at position zero and some continuing from a state their slot
already holds (at starts that are not multiples of the chunk), a padding row
of slot -1, rows that keep a checkpoint inside their tokens and at their end,
and a decode batch of one-token rows. The queries, keys and values are column
ranges of one merged projection and the two gate inputs of another, so the
row strides the program reads are not the views' widths. The interpreter runs
small widths; a device runs the model's (16 key heads and 32 value heads of
128).

The reference is the recurrence itself in FP64, token by token, from the same
rounded inputs (transformers' ``torch_recurrent_gated_delta_rule``, with the
gating and the l2 norm of ``Qwen3_5MoeGatedDeltaNet.forward``). Each output is
compared against the largest magnitude of the reference output in its value
head, because a head's outputs are dot products that cancel. The bound follows
the arithmetic: the program's sums are FP32, a state element is a decayed sum
of rank-one updates whose rounding no later step amplifies (``I - beta k k^T``
has norm at most one and the decay at most one), and the output rounds once::

    |o - o'| <= 2 (2 gamma_(dk+e) sqrt(dk) + u_out) max|o|

(``ops::recurrent::GatedDeltaRule``'s bound). The recurrent walk's products
are FP32 ones (``e = 0``); the chunked form's are three TF32 passes of
truncated splits on the tensor cores, each within ``3 * 2^-20`` of exact, and
its passes of rests are summed first, which costs
``e = 48 + ceil(3 dk / 512)`` roundings more (``chunked_roundings``).

The states, the checkpoints, the trail and the zero rows of the padding row
are checked too: a state against the largest magnitude of its head's reference
state at the same FP32 bound, a padding row's output and its slot's untouched
state exactly.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit

TORCH_TYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
UNIT_ROUNDOFF = {"bf16": 2.0**-8, "f32": 2.0**-24}

#: The seed of the first batch; each later batch draws from the next one.
SEED = 7

#: The batches, in the order of their seeds.
BATCHES = ("prompt rows", "decode rows", "verification rows")

#: The forms of the walk: the recurrent one, one token at a time, and the
#: chunked one (its chunk the width's).
FORMS = ("recurrent", "chunked")


def widths(interpreted):
    """The heads, their dimensions, the value block and the chunk of a run:
    small on the interpreter, the model's on a device.
    """  # noqa: D205
    if interpreted:
        return {
            "key_heads": 2,
            "value_heads": 4,
            "key_dim": 16,
            "value_dim": 32,
            "block": 16,
            "chunk": 16,
        }
    return {
        "key_heads": 16,
        "value_heads": 32,
        "key_dim": 128,
        "value_dim": 128,
        "block": 64,
        "chunk": 64,
    }


def batch(name, interpreted):
    """The rows of batch `name`: their lengths, the positions they start at,
    their slots (-1 pads), the position each keeps a checkpoint at, and
    whether every token's state is kept in the trail.
    """  # noqa: D205
    if name == "prompt rows":
        if interpreted:
            return {
                "lengths": [1, 5, 20, 37],
                "starts": [0, 9, 0, 13],
                "slots": [2, -1, 0, 1],
                "checkpoints": [0, 0, 16, 48],
            }
        return {
            "lengths": [1, 7, 130, 300, 64, 129],
            "starts": [0, 0, 37, 0, 64, 1000],
            "slots": [3, -1, 0, 2, 1, 4],
            "checkpoints": [0, 0, 128, 256, 128, 1129],
        }
    if name == "decode rows":
        decode = 6 if interpreted else 48
        return {
            "lengths": [1] * decode,
            "starts": [5 + 17 * r for r in range(decode)],
            "slots": [r if r % 5 else -1 for r in range(decode)],
            "checkpoints": [
                5 + 17 * r + 1 if r % 3 == 0 else 0 for r in range(decode)
            ],
        }
    # A verification's rows: a few tokens each from their slots' states, the
    # state after every token kept in the trail.
    return {
        "lengths": [4, 1, 3, 4],
        "starts": [9, 30, 61, 2],
        "slots": [1, 3, -1, 0],
        "checkpoints": [0, 0, 0, 4],
        "trailing": True,
    }


#: The cases: every batch in every form, but the trail, which only the
#: recurrent walk keeps.
CASES = [
    (name, form)
    for name in BATCHES
    for form in FORMS
    if form == "recurrent" or name != "verification rows"
]


def bound(key_dim, dtype, chunk):
    """The operator's relative bound against a head's largest output, for a
    form's chunk.
    """  # noqa: D205
    extra = 0 if chunk == 1 else 48 + -(-3 * key_dim // 512)
    terms = (key_dim + extra) * 2.0**-24
    gamma = terms / (1 - terms)
    return 2.0 * (2.0 * gamma * math.sqrt(key_dim) + UNIT_ROUNDOFF[dtype])


def reference_row(
    q, k, v, a, b, a_log, dt_bias, state, scale, start, checkpoint
):
    """The recurrence over one row in FP64.

    Answers the outputs, the final state, the checkpoint state (or None) and
    the state after every token.
    """
    tokens, value_heads, _ = v.shape
    group = value_heads // q.shape[1]
    q = q.double().repeat_interleave(group, dim=1)
    k = k.double().repeat_interleave(group, dim=1)
    v = v.double()
    x = a.double() + dt_bias.double()
    softplus = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    g = -torch.exp(a_log.double()) * softplus
    beta = torch.sigmoid(b.double())
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    s = state.double().clone()
    out = torch.zeros_like(v)
    kept = None
    trail = []
    for t in range(tokens):
        s = s * torch.exp(g[t])[:, None, None]
        recalled = (s * k[t][:, :, None]).sum(1)
        delta = (v[t] - recalled) * beta[t][:, None]
        s = s + k[t][:, :, None] * delta[:, None, :]
        out[t] = (s * q[t][:, :, None]).sum(1)
        if start + t + 1 == checkpoint:
            kept = s.clone()
        trail.append(s.clone())
    return out, s, kept, trail


def assert_within(name, got, expected, relative):
    """Hold `got` to `relative * max|expected|` per head, the first axis after
    the rows of an output or the first of a state.
    """  # noqa: D205
    got = got.double().cpu()
    expected = expected.double().cpu()
    if expected.dim() == 3:  # [tokens, heads, dim]
        peak = expected.abs().amax(dim=(0, 2))[None, :, None]
    else:  # [heads, key_dim, value_dim]
        peak = expected.abs().amax(dim=(1, 2))[:, None, None]
    error = (got - expected).abs()
    outside = int((~(error <= relative * peak)).sum())
    worst = float((error / peak.clamp(min=1e-30)).max())
    assert outside == 0, (
        f"{name}: {outside} of {got.numel()} outside the bound {relative:.3e}; "
        f"max abs {float(error.max()):.3e}, max relative to the peak "
        f"{worst:.3e}"
    )


def launch(kernels, views, banks, rows, scalars, trailing, **shape):
    """One launch of the program: `views` are the query, key, value, the two
    gate inputs, `A_log`, the time-step bias and the int32 offsets, sequence
    lengths, slots and checkpoints; `banks` the state, checkpoint and trail
    banks and the output; `scalars` the two row strides and the scale.
    """  # noqa: D205
    chunk = shape["chunk"]
    kernels._gated_delta_rule_kernel[
        (shape["value_dim"] // shape["block"], shape["value_heads"], rows)
    ](
        *views,
        *banks,
        *scalars,
        int(trailing),
        KEY_HEADS=shape["key_heads"],
        VALUE_HEADS=shape["value_heads"],
        KEY_DIM=shape["key_dim"],
        VALUE_DIM=shape["value_dim"],
        BLOCK=shape["block"],
        CHUNK=chunk,
        LEVELS=chunk.bit_length() - 1,
        num_warps=8 if chunk > 1 else 4,
    )


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize(("name", "form"), CASES)
def test_a_batch_matches_the_fp64_recurrence(
    name, form, dtype, torch_device, interpreted, kernel_source
):
    shape = widths(interpreted)
    rows_of = batch(name, interpreted)
    torch.manual_seed(SEED + BATCHES.index(name))
    kh, vh = shape["key_heads"], shape["value_heads"]
    dk, dv = shape["key_dim"], shape["value_dim"]
    chunk = 1 if form == "recurrent" else shape["chunk"]
    element = TORCH_TYPES[dtype]
    lengths, starts, slots, checkpoints = (
        rows_of[field]
        for field in ("lengths", "starts", "slots", "checkpoints")
    )
    rows = len(lengths)
    tokens = sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    width = 2 * kh * dk + vh * dv
    # A wider row than the three views.
    merged = torch.randn(tokens, width + 16).to(element)

    def views_of(source):
        """The query, key and value views of the merged projection `source`."""
        return (
            source.as_strided((tokens, kh, dk), (width + 16, dk, 1), 0),
            source.as_strided((tokens, kh, dk), (width + 16, dk, 1), kh * dk),
            source.as_strided(
                (tokens, vh, dv), (width + 16, dv, 1), 2 * kh * dk
            ),
        )

    query, key, value = views_of(merged)
    gates = (torch.randn(tokens, 2 * vh) * 2.0).to(element)
    b = gates[:, :vh]
    a = gates[:, vh:]
    a_log = torch.log(torch.empty(vh).uniform_(1.0, 16.0)).to(element)
    dt_bias = torch.randn(vh).to(element)
    slot_count = max(max(slots) + 1, 1) + 1
    initial = torch.randn(slot_count, vh, dk, dv, dtype=torch.float32) * 0.3
    scale = dk**-0.5

    expected = torch.zeros(tokens, vh, dv, dtype=torch.float64)
    finals, kept, trails = {}, {}, {}
    for r in range(rows):
        if slots[r] < 0:
            continue
        lo, hi = offsets[r], offsets[r + 1]
        state = initial[slots[r]] if starts[r] > 0 else torch.zeros(vh, dk, dv)
        out, final, checkpoint, trail = reference_row(
            query[lo:hi],
            key[lo:hi],
            value[lo:hi],
            a[lo:hi],
            b[lo:hi],
            a_log,
            dt_bias,
            state,
            scale,
            starts[r],
            checkpoints[r],
        )
        expected[lo:hi] = out
        finals[slots[r]] = final
        for index, after in enumerate(trail):
            trails[lo + index] = after
        if checkpoint is not None:
            kept[slots[r]] = checkpoint

    def column(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    seq_lens = [start + length for start, length in zip(starts, lengths)]
    relative = bound(dk, dtype, chunk)
    state_relative = bound(dk, "f32", chunk)
    trailing = rows_of.get("trailing", False)
    state = initial.clone().to(torch_device)
    checkpoint_bank = torch.full_like(state, float("nan"))
    trail_bank = torch.full(
        (tokens, vh, dk, dv),
        float("nan"),
        dtype=torch.float32,
        device=torch_device,
    )
    out = torch.full(
        (tokens, vh, dv), float("nan"), dtype=element, device=torch_device
    )
    merged_on_device, gates_on_device = (
        merged.to(torch_device),
        gates.to(torch_device),
    )
    launch(
        kernel_source("gated_delta_rule"),
        (
            *views_of(merged_on_device),
            gates_on_device[:, vh:],
            gates_on_device[:, :vh],
            a_log.to(torch_device),
            dt_bias.to(torch_device),
            *map(column, (offsets, seq_lens, slots, checkpoints)),
        ),
        (state, checkpoint_bank, trail_bank, out),
        rows,
        (width + 16, 2 * vh, scale),
        trailing,
        **{**shape, "chunk": chunk},
    )

    label = f"{name}, {form} (chunk {chunk}), {dtype}"
    live = torch.tensor(
        [s >= 0 for r, s in enumerate(slots) for _ in range(lengths[r])],
        dtype=torch.bool,
    )
    assert_within(f"{label}: output", out.cpu()[live], expected[live], relative)
    padding = ~live
    if padding.any():
        zeros = int((out.cpu()[padding] != 0).sum())
        assert zeros == 0, f"{label}: padding rows: {zeros} nonzero outputs"
    for slot, final in finals.items():
        assert_within(
            f"{label}: state of slot {slot}",
            state[slot].cpu(),
            final,
            state_relative,
        )
    for slot, checkpoint in kept.items():
        assert_within(
            f"{label}: checkpoint of slot {slot}",
            checkpoint_bank[slot].cpu(),
            checkpoint,
            state_relative,
        )
    if trailing:
        for token, after in trails.items():
            assert_within(
                f"{label}: trail of token {token}",
                trail_bank[token].cpu(),
                after,
                state_relative,
            )
    for slot in range(slot_count):
        if slot not in finals:
            changed = int((state[slot].cpu() != initial[slot]).sum())
            assert changed == 0, (
                f"{label}: slot {slot}, no row's, changed in {changed} elements"
            )
