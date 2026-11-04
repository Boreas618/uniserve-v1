#!/usr/bin/env python3
"""Compare a Qwen3-MoE hidden-state dump against the reference, layer by layer.

The slice compares in the run (`--reference DIR`, naming the first stage past 5%); this is the
device-free view of two directories, for a dump kept from an earlier run and for the per-row worst
that the run's one-line verdict does not carry. It needs no GPU and no store.

One run of the model's slice with `--dump DIR` writes `hidden_00.npy` (the embedding's output),
`hidden_NN.npy` for `NN = 01..48` (layer `NN - 1`'s output) and `logits.npy` (the last row), all
float32, row-major, `[rows, hidden]`. The reference writes the same names from the same prompt, so a
defect is located by the first name whose numbers leave the reference's own rounding floor.

    python tools/diagnosis/hidden_layers.py \
        --reference notes/reference/qwen3-hidden --dump artifacts/qwen3-hidden-dump

The reference holds `hidden_00` to `hidden_48` and a `meta.json` with the prompt's ids; it holds no
`logits.npy`, so the logits are not compared here - the run prints its top-10 and the served
reference states what they should be.

Two measures, because they answer different questions:

- `rel` is `max|a - b| / max|b|` over the whole file: the worst single element, in units of the
  layer's own magnitude. One bad channel or one bad row shows up here and nowhere else.
- `rms` is `rms(a - b) / rms(b)`: the error of the bulk, which is what a rounding difference looks
  like. A structural fault raises both; a bf16 rounding difference raises `rms` to about `2^-8` and
  leaves `rel` near it.

The dump holds **bf16** values widened to float32, so a perfectly correct run still differs from a
float32 reference by that rounding: `--floor` defaults to `3e-3` per value and the table prints the
first layer that leaves it. Rows are reported too: the last row is the one that becomes the logits,
and a divergence that starts at row 0 while the later rows agree is a causal-mask fault, not a
numerical one.
"""

import argparse
import pathlib
import sys

import numpy as np


def load(path):
    if not path.exists():
        return None
    return np.load(path).astype(np.float64)


def measure(actual, expected):
    """The worst element relative to the layer's own magnitude, and the bulk's relative rms."""
    scale = np.max(np.abs(expected))
    peak = np.max(np.abs(actual - expected))
    rel = float(peak / scale) if scale > 0 else float(peak)
    expected_rms = float(np.sqrt(np.mean(expected**2)))
    error_rms = float(np.sqrt(np.mean((actual - expected) ** 2)))
    rms = error_rms / expected_rms if expected_rms > 0 else float("inf")
    return rel, rms


def per_row(actual, expected):
    """Each row's own `max|a - b| / max|b|`, so a fault confined to some positions is visible."""
    scale = np.max(np.abs(expected), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return np.max(np.abs(actual - expected) / scale, axis=1)


def names(reference):
    """Every file the reference holds, in the order a run writes them: hidden, then logits."""
    hidden = sorted(reference.glob("hidden_*.npy"))
    logits = sorted(reference.glob("logits*.npy"))
    return hidden + logits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--dump", required=True, type=pathlib.Path)
    parser.add_argument(
        "--floor",
        type=float,
        default=3e-3,
        help="the relative error a correct run may still show (bf16 rounding), default 3e-3",
    )
    arguments = parser.parse_args()

    files = names(arguments.reference)
    if not files:
        print(f"{arguments.reference}: holds no hidden_*.npy", file=sys.stderr)
        return 1

    first = None
    print(f"{'file':<16} {'shape':<12} {'rel':>10} {'rms':>10}  worst row")
    for path in files:
        expected = load(path)
        actual = load(arguments.dump / path.name)
        if actual is None:
            print(f"{path.name:<16} {'-':<12} {'missing from the dump':>22}")
            if first is None:
                first = path.name
            continue
        if actual.shape != expected.shape:
            print(f"{path.name:<16} {str(actual.shape):<12} shape differs from {expected.shape}")
            if first is None:
                first = path.name
            continue
        rel, rms = measure(actual, expected)
        rows = per_row(actual, expected)
        worst = int(np.argmax(rows))
        flag = ""
        if first is None and (rel > arguments.floor or rms > arguments.floor):
            first = path.name
            flag = "  <- first outside the floor"
        print(
            f"{path.name:<16} {str(expected.shape):<12} {rel:10.3e} {rms:10.3e}"
            f"  {worst} ({rows[worst]:.3e}){flag}"
        )

    print()
    if first is None:
        print(f"every file is inside the {arguments.floor:g} floor: no layer diverges")
        return 0
    print(f"first divergence: {first}")

    # The logits are compared as log-softmaxes as well, because that is what a token is chosen from.
    expected = load(arguments.reference / "logits.npy")
    actual = load(arguments.dump / "logits.npy")
    if expected is not None and actual is not None and expected.shape == actual.shape:
        expected_log = expected[0] - np.log(np.sum(np.exp(expected[0] - expected[0].max()))) - expected[0].max()
        actual_log = actual[0] - np.log(np.sum(np.exp(actual[0] - actual[0].max()))) - actual[0].max()
        for name, values in (("reference", expected_log), ("dump", actual_log)):
            best = np.argsort(-values)[:10]
            print(f"  {name}: " + ", ".join(f"{int(t)} {values[t]:.3f}" for t in best))
        scale = np.max(np.abs(expected))
        print(f"  logits: max|a - b| / max|b| = {np.max(np.abs(actual - expected)) / scale:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
