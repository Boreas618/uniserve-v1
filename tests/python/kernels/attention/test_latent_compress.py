"""The `latent_compress` program against the pooling it computes.

A token at position `p` completes its group of `ratio` positions when
`(p + 1) mod ratio = 0`, and then pools the group's projections
(`ops::compress::LatentCompress`)::

    pooled[c] = Σ_m softmax_m(score[m, c]) · kv[m, c]        FP32
    latent    = rms_norm(bf16(pooled), gain, eps)            rounded to BF16

reading a predecessor from the batch when its state row holds it in this batch
and from the request's ring otherwise. Every token answers the entry it
completes (`p / ratio` or `-1`) and the ring row it is kept in (`slot · ring +
p mod ring` for its state row's last `ring` tokens, or `-1`).

What a wrong kernel gets wrong, and the case that catches each:

- **the member's source**: a chunk that starts inside a group, whose first
  completing token reads its predecessor from the ring, a single-token row
  that reads every predecessor from it, and a ratio of four whose group is
  split between the two;
- **the state row**: rows of one to six tokens with an empty row among them,
  a padding row and padding tokens past the last row, each answering `-1`
  twice and writing no latent;
- **the kept rows**: the last `ring` tokens of every row and no other;
- **the rounding points**: the pooled row rounds into BF16 before it is
  normalized, and the latent once more.

The answers are exact. A latent is held to an FP64 reference at the bound of
its roundings: the store's (one unit roundoff, or one step on the
interpreter, which truncates a BF16 cast and models it), one step of the
pooled row's rounding and the normalization's statistics over it (two unit
roundoffs each, the worst case of a rounding the FP32 pooling moves across a
midpoint), and the statistics' FP32 sum (`2γ_width`). Rows that complete no
group must keep the NaN the latent is pre-filled with.
"""

import pytest
import torch
from uniserve_testing.kernels.rope.rounding import UNIT_ROUNDOFF, gamma

pytestmark = pytest.mark.unit

#: A padding row's slot, `NO_SLOT` read as I32.
NO_SLOT = -1

#: What the answer columns hold before the launch, which no answer is.
UNANSWERED = 12345

#: The epsilon: DeepSeek-V4.1's.
EPS = 1e-20

#: The slots of the ring buffers every case allocates.
SLOTS = 4

#: The cases: the ratio, the latent's width, the ring's rows, and the state
#: rows as `(slot, first position, tokens)`; `padding` tokens follow the last
#: row's.
CASES = {
    "a prefill chunk from position zero": (2, 512, 8, [(2, 0, 9)], 0),
    "a chunk starting inside a group, its first member in the ring": (
        2,
        512,
        8,
        [(0, 5, 6)],
        0,
    ),
    "verification rows of one to six tokens, an empty and a padding row": (
        2,
        512,
        8,
        [(1, 13, 1), (3, 20, 3), (0, 40, 0), (NO_SLOT, 7, 2), (2, 30, 6)],
        2,
    ),
    "a ratio of four, a group split between the ring and the batch": (
        4,
        128,
        8,
        [(0, 6, 5), (3, 0, 8)],
        1,
    ),
}


def batch(case, seed):
    """The case's columns and operands: positions, slots, query offsets,
    sequence lengths, projections, the ring and the gain.
    """  # noqa: D205
    ratio, width, ring, state_rows, padding = CASES[case]
    generator = torch.Generator().manual_seed(seed)
    positions, offsets, seen, slots = [], [0], [], []
    for slot, first, tokens in state_rows:
        positions.extend(range(first, first + tokens))
        offsets.append(offsets[-1] + tokens)
        seen.append(first + tokens)
        slots.append(slot)
    positions.extend([0] * padding)
    tokens = len(positions)

    # Values of a few units and gates spread enough that a group's softmax is
    # not uniform.
    projected = torch.randn(tokens, 2 * width, generator=generator)
    projected[:, width:] *= 3.0
    states = torch.randn(SLOTS, ring, 2 * width, generator=generator)
    states[..., width:] *= 3.0
    gain = (1.0 + 0.25 * torch.randn(width, generator=generator)).to(
        torch.bfloat16
    )
    return {
        "positions": torch.tensor(positions, dtype=torch.int32),
        "slots": torch.tensor(slots, dtype=torch.int32),
        "query_offsets": torch.tensor(offsets, dtype=torch.int32),
        "seq_lens": torch.tensor(seen, dtype=torch.int32),
        "projected": projected,
        "ring": states,
        "gain": gain,
    }


def key(ratio, width, ring):
    """`LatentCompressDescriptor`'s key."""
    return f"ratio_{ratio}_width_{width}_ring_{ring}_gain_bf16"


def launch(stated, operands, device):
    """One launch over every token, as the package states it: the latent,
    the entries and the ring rows it answers.
    """  # noqa: D205
    tokens, double = operands["projected"].shape
    on = {name: value.to(device) for name, value in operands.items()}
    latent = torch.full(
        (tokens, double // 2), float("nan"), dtype=torch.bfloat16, device=device
    )
    entries = torch.full(
        (tokens,), UNANSWERED, dtype=torch.int32, device=device
    )
    ring_rows = torch.full_like(entries, UNANSWERED)
    stated.kernel[(tokens,)](
        on["projected"],
        on["gain"],
        on["ring"],
        on["slots"],
        on["query_offsets"],
        on["seq_lens"],
        on["positions"],
        latent,
        entries,
        ring_rows,
        tokens,
        on["slots"].numel(),
        on["projected"].stride(0),
        EPS,
        **stated.constexprs,
        **stated.options,
    )
    return latent.cpu(), entries.cpu(), ring_rows.cpu()


def rounded(values, interpreted):
    """FP64 `values` rounded into BF16 as the program rounds them: to nearest
    on a device, truncated on the interpreter.
    """  # noqa: D205
    if not interpreted:
        return values.to(torch.bfloat16).to(torch.float64)
    bits = values.to(torch.float32).contiguous().view(torch.int32)
    return (bits & ~0xFFFF).view(torch.float32).to(torch.float64)


def reference(case, operands, interpreted):
    """Each token's answers and, for a completing one, its FP64 latent."""
    ratio, width, ring, _, _ = CASES[case]
    offsets = operands["query_offsets"].tolist()
    seen = operands["seq_lens"].tolist()
    slots = operands["slots"].tolist()
    positions = operands["positions"].tolist()
    projected = operands["projected"].to(torch.float64)
    states = operands["ring"].to(torch.float64)
    gain = operands["gain"].to(torch.float64)

    entries, kept, latents = [], [], {}
    for token, position in enumerate(positions):
        rows = [
            row
            for row in range(len(slots))
            if offsets[row] <= token < offsets[row + 1]
        ]
        if not rows or slots[rows[0]] == NO_SLOT:
            entries.append(-1)
            kept.append(-1)
            continue
        row = rows[0]
        slot = slots[row]
        first = seen[row] - (offsets[row + 1] - offsets[row])
        kept.append(
            slot * ring + position % ring
            if position >= seen[row] - ring
            else -1
        )
        if (position + 1) % ratio:
            entries.append(-1)
            continue
        entries.append(position // ratio)

        members = []
        for q in range(position - ratio + 1, position + 1):
            if q >= first:
                members.append(projected[token - (position - q)])
            else:
                members.append(states[slot, q % ring])
        members = torch.stack(members)
        weights = torch.softmax(members[:, width:], dim=0)
        pooled = rounded((weights * members[:, :width]).sum(dim=0), interpreted)
        scale = torch.rsqrt(pooled.square().mean() + EPS)
        latents[token] = pooled * scale * gain
    return entries, kept, latents


@pytest.mark.parametrize("case", CASES)
def test_every_token_answers_its_group_and_its_ring_row(
    case, torch_device, interpreted, stated_launch, launch_target
):
    ratio, width, ring, _, _ = CASES[case]
    operands = batch(case, seed=len(case))

    stated = stated_launch(
        "latent_compress", key(ratio, width, ring), launch_target
    )
    latent, entries, ring_rows = launch(stated, operands, torch_device)
    expected_entries, expected_kept, latents = reference(
        case, operands, interpreted
    )

    assert entries.tolist() == expected_entries, f"{case}: the entries"
    assert ring_rows.tolist() == expected_kept, f"{case}: the ring rows"

    bf16 = UNIT_ROUNDOFF[torch.bfloat16]
    store = (2.0 if interpreted else 1.0) * bf16
    relative = (
        store
        + 2.0 * (2.0 * bf16)
        + 2.0 * gamma(width)
        + 8.0 * UNIT_ROUNDOFF[torch.float32]
    )
    for token in range(latent.shape[0]):
        got = latent[token].to(torch.float64)
        if token not in latents:
            assert got.isnan().all(), (
                f"{case}: token {token} completes no group and wrote its row"
            )
            continue
        expected = latents[token]
        difference = (got - expected).abs()
        allowed = relative * expected.abs() + 2.0**-133
        outside = ~(difference <= allowed)
        if outside.any():
            channel = int(outside.nonzero()[0])
            pytest.fail(
                f"{case}: token {token}: {int(outside.sum())} of {width} "
                f"channels outside the bound; first {channel}: "
                f"{float(got[channel])} vs {float(expected[channel])}, "
                f"allowed {float(allowed[channel]):.3e}"
            )
