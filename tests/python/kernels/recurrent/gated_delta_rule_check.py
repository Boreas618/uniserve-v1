#!/usr/bin/env python3
"""Run the gated_delta_rule program over a batch of rows and check it against FP64.

    python gated_delta_rule_check.py                    # the Qwen3.5 widths, both forms, on a GPU
    python gated_delta_rule_check.py --interpreter      # small widths on Triton's interpreter

A batch here is what the operator's rows are: packed rows of different lengths, some starting at
position zero and some continuing from a state their slot already holds (at starts that are not
multiples of the chunk), a padding row of slot −1, rows that keep a checkpoint inside their tokens
and at their end, and a decode batch of one-token rows. The queries, keys and values are column
ranges of one merged projection and the two gate inputs of another, so the row strides the
program reads are not the views' widths.

The reference is the recurrence itself in FP64, token by token, from the same rounded inputs
(transformers' `torch_recurrent_gated_delta_rule`, with the gating and the l2 norm of
`Qwen3_5MoeGatedDeltaNet.forward`). Each output is compared against the largest magnitude of the
reference output in its value head, because a head's outputs are dot products that cancel. The
bound follows the arithmetic: the program's sums are FP32, a state element is a decayed sum of
rank-one updates whose rounding no later step amplifies (`I − β k̂k̂ᵀ` has norm at most one and
the decay at most one), and the output rounds once:

    |o − ô| ≤ 2 · (2 γ_(dk+e) √dk + u_out) · max|o|      (`ops::recurrent::GatedDeltaRule`'s bound)

The recurrent walk's products are FP32 ones (`e = 0`); the chunked form's are three TF32 passes
of truncated splits on the tensor cores, each within `3·2⁻²⁰` of exact, and its passes of rests
are summed first, which costs `e = 48 + ⌈3·dk/512⌉` roundings more (`chunked_roundings`).

The states, the checkpoints and the zero rows of the padding row are checked too: a state against
the largest magnitude of its head's reference state at the same FP32 bound, a padding row's output
and its slot's untouched state exactly.
"""

import argparse
import math
import os
import pathlib
import sys

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "gated_delta_rule" / "triton"
sys.path.insert(0, str(PACKAGE / "src"))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpreter", action="store_true", help="run on Triton's interpreter"
    )
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


FLAGS = parse()
# The family suite's interpreter run states it in the environment, with no device visible.
FLAGS.interpreter = (
    FLAGS.interpreter or os.environ.get("TRITON_INTERPRET") == "1"
)
if FLAGS.interpreter:
    os.environ["TRITON_INTERPRET"] = "1"

import gated_delta_rule as kernels  # noqa: E402
import torch  # noqa: E402

TORCH_TYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
UNIT_ROUNDOFF = {"bf16": 2.0**-8, "f32": 2.0**-24}


def bound(key_dim, dtype, chunk):
    """The operator's relative bound against a head's largest output, for a form's chunk."""
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


def compare(name, got, expected, relative, failures):
    """Count elements outside `relative · max|expected|` per head (the first axis after rows)."""
    got = got.double().cpu()
    expected = expected.double().cpu()
    if expected.dim() == 3:  # [tokens, heads, dim]
        peak = expected.abs().amax(dim=(0, 2))[None, :, None]
    else:  # [heads, key_dim, value_dim]
        peak = expected.abs().amax(dim=(1, 2))[:, None, None]
    error = (got - expected).abs()
    outside = int((~(error <= relative * peak)).sum())
    worst = float((error / peak.clamp(min=1e-30)).max())
    absolute = float(error.max())
    print(
        f"  {name}: max abs {absolute:.3e}, max rel-to-peak {worst:.3e}, "
        f"bound {relative:.3e}, {outside} of {got.numel()} outside"
    )
    failures.append(outside)
    return worst, absolute


def run(case, device, failures, report):
    """Launch both requested forms over one batch and compare everything they write."""
    torch.manual_seed(case["seed"])
    kh, vh, dk, dv = (
        case["key_heads"],
        case["value_heads"],
        case["key_dim"],
        case["value_dim"],
    )
    dtype = TORCH_TYPES[case["dtype"]]
    lengths, starts, slots, checkpoints = (
        case[name] for name in ("lengths", "starts", "slots", "checkpoints")
    )
    rows = len(lengths)
    tokens = sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    width = 2 * kh * dk + vh * dv
    merged = torch.randn(tokens, width + 16).to(
        dtype
    )  # a wider row than the three views
    query = merged.as_strided((tokens, kh, dk), (width + 16, dk, 1), 0)
    key = merged.as_strided((tokens, kh, dk), (width + 16, dk, 1), kh * dk)
    value = merged.as_strided(
        (tokens, vh, dv), (width + 16, dv, 1), 2 * kh * dk
    )
    gates = (torch.randn(tokens, 2 * vh) * 2.0).to(dtype)
    b = gates[:, :vh]
    a = gates[:, vh:]
    a_log = torch.log(torch.empty(vh).uniform_(1.0, 16.0)).to(dtype)
    dt_bias = torch.randn(vh).to(dtype)
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
        return torch.tensor(values, dtype=torch.int32, device=device)

    seq_lens = [start + length for start, length in zip(starts, lengths)]
    for form, chunk in case["forms"]:
        relative = bound(dk, case["dtype"], chunk)
        state_relative = bound(dk, "f32", chunk)
        # Only the recurrent walk keeps a trail.
        trailing = case.get("trailing", False) and chunk == 1
        state = initial.clone().to(device)
        checkpoint_bank = torch.full_like(state, float("nan"))
        trail_bank = torch.full(
            (tokens, vh, dk, dv),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        out = torch.full(
            (tokens, vh, dv), float("nan"), dtype=dtype, device=device
        )
        on = [t.to(device) for t in (merged, gates)]
        merged_d, gates_d = on
        query_d = merged_d.as_strided((tokens, kh, dk), (width + 16, dk, 1), 0)
        key_d = merged_d.as_strided(
            (tokens, kh, dk), (width + 16, dk, 1), kh * dk
        )
        value_d = merged_d.as_strided(
            (tokens, vh, dv), (width + 16, dv, 1), 2 * kh * dk
        )
        kernels._gated_delta_rule_kernel[(dv // case["block"], vh, rows)](
            query_d,
            key_d,
            value_d,
            gates_d[:, vh:],
            gates_d[:, :vh],
            a_log.to(device),
            dt_bias.to(device),
            column(offsets),
            column(seq_lens),
            column(slots),
            column(checkpoints),
            state,
            checkpoint_bank,
            trail_bank,
            out,
            width + 16,
            2 * vh,
            scale,
            int(trailing),
            KEY_HEADS=kh,
            VALUE_HEADS=vh,
            KEY_DIM=dk,
            VALUE_DIM=dv,
            BLOCK=case["block"],
            CHUNK=chunk,
            LEVELS=chunk.bit_length() - 1,
            num_warps=8 if chunk > 1 else 4,
        )
        print(
            f"{case['label']}, {form} (chunk {chunk}), {case['dtype']}: rows {lengths} from "
            f"{starts}, slots {slots}, checkpoints {checkpoints}"
        )
        live = torch.tensor(
            [s >= 0 for r, s in enumerate(slots) for _ in range(lengths[r])],
            dtype=torch.bool,
        )
        worst, absolute = compare(
            "output", out.cpu()[live], expected[live], relative, failures
        )
        report.append(
            (f"{case['label']} {form}", "output", absolute, worst, relative)
        )
        padding = ~live
        if padding.any():
            zeros = int((out.cpu()[padding] != 0).sum())
            print(f"  padding rows: {zeros} nonzero outputs")
            failures.append(zeros)
        for slot, final in finals.items():
            worst, absolute = compare(
                f"state of slot {slot}",
                state[slot].cpu(),
                final,
                state_relative,
                failures,
            )
            report.append(
                (
                    f"{case['label']} {form}",
                    f"state {slot}",
                    absolute,
                    worst,
                    state_relative,
                )
            )
        for slot, checkpoint in kept.items():
            worst, absolute = compare(
                f"checkpoint of slot {slot}",
                checkpoint_bank[slot].cpu(),
                checkpoint,
                state_relative,
                failures,
            )
            report.append(
                (
                    f"{case['label']} {form}",
                    f"checkpoint {slot}",
                    absolute,
                    worst,
                    state_relative,
                )
            )
        if trailing:
            for token, after in trails.items():
                worst, absolute = compare(
                    f"trail of token {token}",
                    trail_bank[token].cpu(),
                    after,
                    state_relative,
                    failures,
                )
                report.append(
                    (
                        f"{case['label']} {form}",
                        f"trail {token}",
                        absolute,
                        worst,
                        state_relative,
                    )
                )
        untouched = [s for s in range(slot_count) if s not in finals]
        for slot in untouched:
            changed = int((state[slot].cpu() != initial[slot]).sum())
            if changed:
                print(f"  slot {slot}, no row's, changed in {changed} elements")
            failures.append(changed)


def main():
    device = "cpu" if FLAGS.interpreter else "cuda"
    if FLAGS.interpreter:
        widths = {
            "key_heads": 2,
            "value_heads": 4,
            "key_dim": 16,
            "value_dim": 32,
            "block": 16,
        }
        chunk = 16
        lengths, starts = [1, 5, 20, 37], [0, 9, 0, 13]
        slots, checkpoints = [2, -1, 0, 1], [0, 0, 16, 48]
    else:
        widths = {
            "key_heads": 16,
            "value_heads": 32,
            "key_dim": 128,
            "value_dim": 128,
            "block": 64,
        }
        chunk = 64
        lengths, starts = [1, 7, 130, 300, 64, 129], [0, 0, 37, 0, 64, 1000]
        slots, checkpoints = [3, -1, 0, 2, 1, 4], [0, 0, 128, 256, 128, 1129]
    forms = [("recurrent", 1), ("chunked", chunk)]
    failures, report = [], []
    run(
        {
            "label": "prompt rows",
            "dtype": FLAGS.dtype,
            "seed": FLAGS.seed,
            **widths,
            "lengths": lengths,
            "starts": starts,
            "slots": slots,
            "checkpoints": checkpoints,
            "forms": forms,
        },
        device,
        failures,
        report,
    )
    decode = 6 if FLAGS.interpreter else 48
    run(
        {
            "label": "decode rows",
            "dtype": FLAGS.dtype,
            "seed": FLAGS.seed + 1,
            **widths,
            "lengths": [1] * decode,
            "starts": [5 + 17 * r for r in range(decode)],
            "slots": [r if r % 5 else -1 for r in range(decode)],
            "checkpoints": [
                5 + 17 * r + 1 if r % 3 == 0 else 0 for r in range(decode)
            ],
            "forms": forms,
        },
        device,
        failures,
        report,
    )
    # A verification's rows: a few tokens each from their slots' states, the state after every
    # token kept in the trail.
    run(
        {
            "label": "verification rows",
            "dtype": FLAGS.dtype,
            "seed": FLAGS.seed + 2,
            **widths,
            "lengths": [4, 1, 3, 4],
            "starts": [9, 30, 61, 2],
            "slots": [1, 3, -1, 0],
            "checkpoints": [0, 0, 0, 4],
            "forms": [("recurrent", 1)],
            "trailing": True,
        },
        device,
        failures,
        report,
    )
    outside = sum(failures)
    print(
        f"{'ok' if outside == 0 else 'FAIL'}: {outside} values outside their bounds"
    )
    return 0 if outside == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
