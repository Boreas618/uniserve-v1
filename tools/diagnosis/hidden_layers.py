#!/usr/bin/env python3
"""Compare a Qwen3-MoE hidden-state dump against the reference, layer by layer.

The slice compares in the run (`--reference DIR`, naming the first stage past 5%); this is the
device-free view of two directories, for a dump kept from an earlier run and for the per-row worst
that the run's one-line verdict does not carry. It needs no GPU and no store.

One run of the model's slice with `--dump DIR` writes `hidden_00.npy` (the embedding's output),
`hidden_NN.npy` for `NN = 01..47` (layer `NN - 1`'s output), `hidden_48.npy` (the whole run's output
after the final norm) and `logits.npy` (the last row), all float32, row-major, `[rows, hidden]`. The
reference writes the same names from the same prompt, so a defect is located by the first name whose
numbers leave the reference's own rounding floor. `hidden_48` is the run's own last stage and not
layer 47's output: the reference does not hold that one, so the last layer is read through its cuts.

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
float32 reference by that rounding - about `2e-3` per value, and it does not grow with depth, because
each hidden state is rounded on its own rather than accumulated. The verdict uses the same 5% the
slice uses (`--floor`), so the two agree on which stage is the first to diverge; a number below the
floor is rounding and a number above it is structure. Rows are reported too: the last row is the one that becomes the logits,
and a divergence that starts at row 0 while the later rows agree is a causal-mask fault, not a
numerical one.
"""

import argparse
import pathlib
import sys

import numpy as np

#: The three cuts a layer is walked through, in the order the layer runs them. A reference that sets
#: `hidden_NN_normalized.npy` and the two after it lets one layer be read operator by operator.
CUTS = ["normalized", "attended", "mlp_out"]

#: The reference's own module at which each cut's tensor is read, so a cut's file name states which
#: site wrote it and a comparison never rests on two hands agreeing what a word means.
CUT_SITES = [
    ("normalized", "input_layernorm's output (its residual add is already in it)"),
    ("attended", "self_attn's output, element [0], before the residual add"),
    ("mlp_out", "the layer's state after the block's residual add, which the next boundary holds"),
]


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


def cut_prefix(path, measured):
    """The error of the boundary a cut's layer starts from, or `None` for a boundary file.

    A cut's file is `hidden_NN_<cut>.npy`, layer `NN - 1`'s; the boundary it inherits from is
    `hidden_{NN-1}.npy`, which is layer `NN - 2`'s output and therefore already in the table.
    """
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    return measured.get(f"hidden_{int(parts[1]) - 1:02}")


def names(reference):
    """Every file the reference holds, in the order a run reaches them.

    The stage files are the reference's own set: `hidden_00.npy` is the embedding, `hidden_NN.npy`
    for `NN` from 1 to the last is layer `NN - 1`'s output, and the last file is the **whole run's**
    output after the final norm - not the last layer's, which has no file of its own because the
    reference does not hold it. A layer cut open inside holds `hidden_NN_<cut>.npy` for the cuts the
    run asked the model to stop at (`normalized`, `attended`, `mlp_out`, in that order), so the cut files
    of layer `NN - 1` sort between `hidden_{NN-1}.npy` and `hidden_NN.npy` and the table reads as the
    walk that produced it. The last layer is read through its cuts for that reason.
    """
    def order(path):
        return order_of(path)

    return sorted(reference.glob("hidden_*.npy"), key=order) + sorted(
        reference.glob("logits*.npy")
    )


def order_of(path):
    """Where `path` sits in the walk, so a file the reference lacks can still be placed.

    A cut's name may carry an underscore of its own (`mlp_out`), so the cut is everything after the
    stage number and not one word.
    """
    parts = path.stem.split("_")
    index = int(parts[1])
    if len(parts) < 3:
        return (index, len(CUTS))
    cut = "_".join(parts[2:])
    return (index, CUTS.index(cut) if cut in CUTS else len(CUTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--dump", required=True, type=pathlib.Path)
    parser.add_argument(
        "--floor",
        type=float,
        default=0.05,
        help="the relative error that counts as divergence, matching the slice's 5%, default 0.05",
    )
    arguments = parser.parse_args()

    files = names(arguments.reference)
    # A run asked for a cut the reference does not hold: the file is in the dump and there is nothing
    # to compare it to. It is listed rather than dropped, because a silent absence reads as agreement.
    known = {path.name for path in files}
    missing = [
        path for path in arguments.dump.glob("hidden_*.npy") if path.name not in known
    ]
    if not files:
        print(f"{arguments.reference}: holds no hidden_*.npy", file=sys.stderr)
        return 1

    first = None
    # A defect is named by where the error *enters*, not by where a threshold is first crossed: a
    # fault in layer 1 keeps every later stage above the floor, so the crossing names layer 1 for a
    # fault that is in layer 1 and for one that is in layer 40 alike. The increment does not - it is
    # zero where the error was inherited and largest at the stage that produced it.
    largest = (None, float("-inf"))
    previous = None
    measured = {}
    lines = []
    print(f"{'file':<16} {'shape':<12} {'rel':>10} {'rms':>10} {'increment':>10}  worst row")
    # One order for both: a cut the dump holds and the reference lacks still belongs where the walk
    # reaches it, or the table's "reads as the walk" promise breaks.
    for path in sorted(files + missing, key=order_of):
        expected = load(path)
        if expected is None:
            print(
                f"{path.name:<16} {'-':<12} the dump holds it and the reference does not: nothing "
                "to compare without a reference file of that cut"
            )
            continue
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
        # The prefix is the enclosing boundary's error, printed beside a cut so an error a cut
        # inherited is not read as an error the cut made.
        prefix = cut_prefix(path, measured)
        increment = float("nan") if previous is None else rel - previous
        measured[path.stem] = rel
        previous = rel
        flag = ""
        if first is None and (rel > arguments.floor or rms > arguments.floor):
            first = path.name
            flag = "  <- first outside the floor"
        if increment > largest[1]:
            largest = (path.name, increment)
        prefix_text = "" if prefix is None else f"  (prefix {prefix:.3e})"
        lines.append(
            f"{path.name:<16} {str(expected.shape):<12} {rel:10.3e} {rms:10.3e}"
            f" {increment:10.3e}  {worst} ({rows[worst]:.3e}){flag}{prefix_text}"
        )

    # The marker goes on the one row whose increment is the largest: every later row of a persistent
    # fault also sits above the floor, and marking them all would name every stage at once.
    if largest[0] is not None and largest[1] > 0.0:
        for index, line in enumerate(lines):
            if line.startswith(largest[0]):
                lines[index] = line + "  <- the increment enters here"
    print("\n".join(lines))

    print()
    print("cuts, each named after the reference tensor that answers it: " + ", ".join(
        f"{cut} <- {site}" for cut, site in CUT_SITES
    ))
    if first is None:
        print(f"every file is inside the {arguments.floor:g} floor: no layer diverges")
        return 0
    print(f"first divergence: {first}")
    print(
        f"largest increment: {largest[0]}, +{largest[1]:.3e} over the stage before it — the stage "
        "the error enters at"
    )

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
