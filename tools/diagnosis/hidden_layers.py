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

import math

import numpy as np

#: The three cuts a layer is walked through, in the order the layer runs them. A reference that sets
#: `hidden_NN_normalized.npy` and the two after it lets one layer be read operator by operator.
CUTS = ["normalized", "attended", "router_ids", "router_weights", "mlp_out"]

#: The reference's own module at which each cut's tensor is read, so a cut's file name states which
#: site wrote it and a comparison never rests on two hands agreeing what a word means.
CUT_SITES = [
    ("normalized", "input_layernorm's output (its residual add is already in it)"),
    ("attended", "self_attn's output, element [0], before the residual add"),
    ("router_ids", "the router's selected experts, as ids - compared exactly, never by rms"),
    ("router_weights", "the router's renormalized weights, in the reference's own dtype (bf16)"),
    ("mlp_out", "mlp's output, before the residual add"),
]


def load(path):
    if not path.exists():
        return None
    # The dtype is kept: a router's ids are indices and their comparison is exact, which a cast to
    # float would silently turn into a relative one.
    return np.load(path)


def measure(actual, expected):
    """The worst element relative to the layer's own magnitude, and the bulk's relative rms."""
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    scale = np.max(np.abs(expected))
    peak = np.max(np.abs(actual - expected))
    rel = float(peak / scale) if scale > 0 else float(peak)
    expected_rms = float(np.sqrt(np.mean(expected**2)))
    error_rms = float(np.sqrt(np.mean((actual - expected) ** 2)))
    rms = error_rms / expected_rms if expected_rms > 0 else float("inf")
    return rel, rms


def per_row(actual, expected):
    """Each row's own `max|a - b| / max|b|`, so a fault confined to some positions is visible."""
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    scale = np.max(np.abs(expected), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return np.max(np.abs(actual - expected) / scale, axis=1)


def compare_delta(arguments, order, first):
    """Split a stage's deficit into the residual stream and the block's update, per row.

    A stage's state is its input plus the block's update, so the reference answers its own update
    (`ref_n - ref_{n-1}`) and the dump answers ours. Their difference `D` is what the model failed to
    add, and the direction of `D` says which of the two halves is short: a `D` along `ref_{n-1}` is a
    residual that arrived short (the carry between layers), a `D` along the update is the block itself.
    Layer `first` is reprinted as the calibration: its decomposition is the floor this measurement
    carries, so a later stage's cosines are read against it and not against 1.
    """
    reference = arguments.reference
    dump = arguments.dump
    stems = [stem for stem in order if stem.startswith("hidden_")]
    stages = []
    for index, stem in enumerate(stems):
        if index == 0 or "_" in stem[7:]:
            continue
        stages.append(stem)
    if not stages:
        return
    target = first[:-4] if first and first.endswith(".npy") else first
    chosen = [stem for stem in (stages[0], target) if stem in stages]
    for stem in dict.fromkeys(chosen):
        index = stages.index(stem)
        previous = stages[index - 1]
        before_ref = load(reference / f"{previous}.npy")
        now_ref = load(reference / f"{stem}.npy")
        before_our = load(dump / f"{previous}.npy")
        now_our = load(dump / f"{stem}.npy")
        if any(value is None for value in (before_ref, now_ref, before_our, now_our)):
            continue
        delta_ref = now_ref - before_ref
        delta_our = now_our - before_our
        deficit = delta_ref - delta_our
        print()
        print(f"delta {stem}: the update the model failed to add, per row "
              f"(calibration is layer {stems[1] if len(stems) > 1 else stem})")
        print(f"{'row':>4} {'|Δour|/|Δref|':>14} {'cos(Δour,Δref)':>15} "
              f"{'|D|/|Δref|':>11} {'cos(D,state)':>13} {'cos(D,Δref)':>12}  reads as")
        for row in range(now_ref.shape[0]):
            reference_update, our_update = delta_ref[row], delta_our[row]
            gap = deficit[row]
            reference_norm = np.linalg.norm(reference_update)
            our_norm = np.linalg.norm(our_update)
            gap_norm = np.linalg.norm(gap)
            if reference_norm == 0 or gap_norm == 0:
                continue
            # A row whose update is a rounding of its own state cannot be read this way: the ratio
            # below is a quotient of two small numbers and its direction is noise.
            if reference_norm < 0.01 * np.linalg.norm(before_ref[row]):
                print(f"{row:>4} {'-':>14} {'-':>15} {'-':>11} {'-':>13} {'-':>12}  update below 1% of the state")
                continue
            cosine_update = float(np.dot(our_update, reference_update) / (our_norm * reference_norm)) if our_norm else 0.0
            cosine_state = float(np.dot(gap, before_ref[row]) / (gap_norm * np.linalg.norm(before_ref[row])))
            cosine_reference = float(np.dot(gap, reference_update) / (gap_norm * reference_norm))
            # The unambiguous part is the update itself: a block that is short keeps the update's
            # direction and loses length; a block that routed differently turns it. The deficit's
            # own direction is printed as context and given no verdict, because a piece of state
            # missing shows up in both terms of this decomposition and the tie is not information.
            ratio = our_norm / reference_norm
            if cosine_update < 0.9:
                reading = "the update turned (routing or attention)"
            elif ratio < 0.95:
                reading = "the update is short (the block)"
            elif ratio > 1.05:
                reading = "the update is long (added twice?)"
            else:
                reading = "the update matches"
            print(f"{row:>4} {our_norm / reference_norm:14.3f} {cosine_update:15.4f} "
                  f"{gap_norm / reference_norm:11.3f} {cosine_state:13.4f} {cosine_reference:12.4f}  {reading}")


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


def renormalization_note(actual, expected):
    """The per-row sum of a router's weights, which is what says whether they were renormalized.

    Renormalized weights divide by the top-k's own sum, so every row of them sums to 1. Raw softmax
    probabilities sum to the row's share in its top few instead, and that share is a *different*
    number for each row -- so a disabled renormalization multiplies a whole row's weights by one
    factor and keeps the update's direction, which is exactly what a short block update with a
    preserved direction looks like.
    """
    raw = np.asarray(actual, dtype=np.float64)
    if raw.ndim == 2 and raw.shape[1] > 1:
        # A flat row is its own signature, and it is not a small-scale one: eight equal weights are
        # what a score vector whose top entries differ by less than a thousandth gives, and no
        # selection can survive that spread without the ranking deciding the set.
        spread_of_values = np.nanmax(raw, axis=1) / np.where(raw.min(axis=1) > 0.0, raw.min(axis=1), np.nan)
        flat = int(np.sum(spread_of_values < 1.01))
        if flat == raw.shape[0]:
            return (
                f"; every one of {raw.shape[0]} rows is flat to within 1% - eight equal weights. A"
                " score vector merely multiplied by a small factor keeps its ranking and its set; a"
                " set that also differs is a different score vector, not a smaller one"
            )
    ours = np.asarray(actual, dtype=np.float64).sum(axis=1)
    reference = np.asarray(expected, dtype=np.float64).sum(axis=1)
    ratio = ours / np.where(reference == 0.0, np.nan, reference)
    spread = float(np.nanmax(ratio) - np.nanmin(ratio))
    note = (
        f"  weights sum to {ours[0]:.6f} ours against {reference[0]:.6f} the reference's at row 0;"
        f" every row's ratio {np.nanmin(ratio):.4f} to {np.nanmax(ratio):.4f}"
    )
    if spread < 0.02 and abs(float(np.nanmean(ratio)) - 1.0) > 0.02:
        note += (
            " - one factor on every row, the shape a weight vector has when it was not normalized"
            " over the top k"
        )
    return note + logit_scale_note(actual, expected)


def logit_scale_note(actual, expected):
    """The logit scale and the constant factor, fitted from the weights alone.

    A weight is a normalized exponential of a score, so if our scores were the reference's times `a`,
    our weights are the reference's raised to `a` and renormalized: `log(ours) = a * log(ref) + c`
    holds to the rounding, and a least-squares fit over a row's entries recovers both. `a` different
    from 1 is a score that was scaled on the way in; `a` equal to 1 with `exp(c)` away from 1 is a
    factor applied after the exponential. Neither needs the logits, which our side does not write.
    """
    ours = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    if ours.shape != reference.shape or ours.ndim != 2:
        return ""
    slopes, offsets = [], []
    for row in range(ours.shape[0]):
        positive = (ours[row] > 0.0) & (reference[row] > 0.0)
        if positive.sum() < 3:
            continue
        x = np.log(reference[row][positive])
        y = np.log(ours[row][positive])
        slope, intercept = np.polyfit(x, y, 1)
        slopes.append(slope)
        offsets.append(intercept)
    if not slopes:
        return ""
    slopes = np.asarray(slopes)
    offsets = np.asarray(offsets)
    # The widest pair is the most stable estimate, because adjacent weights differ by little and
    # their ratio carries the rounding: take each row's largest and smallest positive weight.
    pairs = []
    for row in range(ours.shape[0]):
        positive = (ours[row] > 0.0) & (reference[row] > 0.0)
        if positive.sum() < 2:
            continue
        mine = ours[row][positive]
        theirs = reference[row][positive]
        high, low = int(np.argmax(theirs)), int(np.argmin(theirs))
        if theirs[high] / theirs[low] < 10.0:
            continue
        pairs.append(np.log(mine[high] / mine[low]) / np.log(theirs[high] / theirs[low]))
    note = (
        f"; fitted log(ours) = a*log(ref) + c over every positive entry gives a = "
        f"{np.median(slopes):.4f} ({slopes.min():.4f} to {slopes.max():.4f}) and exp(c) = "
        f"{np.exp(np.median(offsets)):.4f}"
    )
    if pairs:
        note += (
            f"; from the widest pair of each row a = {np.median(pairs):.4f} "
            f"({min(pairs):.4f} to {max(pairs):.4f})"
        )
    return note + (
        " - a = 1 with exp(c) != 1 is a factor applied after the exponential, a != 1 is a score that"
        " was scaled on the way in, and a row sum of 1.0 alone does not clear the scores:"
        " renormalization makes the sum 1 whatever the scale is"
    )


def compare_ids(actual, expected, scores):
    """How two expert selections differ, in the classes that decide what a difference means.

    A row whose *set* of experts agrees and whose order does not is not a difference at all: the
    weights travel with their ids, so the block's sum is the same. A row whose set differs is only a
    defect if the two sides were not free to differ: with a bf16 gate the probability of the 8th and
    9th expert is *exactly* equal often enough to matter (measured: about six rows in a hundred), and
    then either selection is right. `scores` is the reference's `router_logits`, when it was written
    beside the ids, and it is what tells those two apart.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if expected.ndim != 2:
        return None
    sets_agree = [
        set(row_a.tolist()) == set(row_e.tolist())
        for row_a, row_e in zip(actual, expected)
    ]
    order_only = sum(
        1
        for agrees, row_a, row_e in zip(sets_agree, actual, expected)
        if agrees and not np.array_equal(row_a, row_e)
    )
    differing = [index for index, agrees in enumerate(sets_agree) if not agrees]
    if not differing:
        if order_only:
            return (
                f"every row's set of experts agrees; the order differs on {order_only} of "
                f"{len(sets_agree)} rows, which the sum does not see"
            )
        return "every index agrees"
    if scores is None:
        return (
            f"{len(differing)} of {len(sets_agree)} rows select a different set; first at row "
            f"{differing[0]}: {actual[differing[0]].tolist()} against the reference's "
            f"{expected[differing[0]].tolist()} — no router logits beside the ids, so a boundary tie "
            "cannot be told from a defect"
        )
    top = expected.shape[1]
    tied, unexplained = [], []
    for row in differing:
        row_scores = np.asarray(scores[row], dtype=np.float64)
        probabilities = np.exp(row_scores - row_scores.max())
        probabilities /= probabilities.sum()
        ordered = np.sort(probabilities)[::-1]
        (tied if ordered[top - 1] == ordered[top] else unexplained).append(row)
    parts = []
    if tied:
        parts.append(f"{len(tied)} on a tied boundary (the 8th and 9th probabilities are equal in the "
                     "reference's own logits, so either selection is right)")
    if unexplained:
        row = unexplained[0]
        # Where our picks sit in the reference's own ordering, and what the boundary gap is: a set
        # that is the reference's top six plus ranks nine and eleven is a different score vector,
        # while ranks one to eight in a different order is the same one.
        rank_of = {int(expert): place for place, expert in enumerate(np.argsort(-np.asarray(scores[row]), kind="stable"))}
        ranks = [rank_of.get(int(expert), -1) for expert in actual[row]]
        ordered = np.asarray(scores[row], dtype=np.float64)[np.argsort(-np.asarray(scores[row]), kind="stable")]
        parts.append(
            f"{len(unexplained)} unexplained; first at row {row}: {actual[row].tolist()} against the "
            f"reference's {expected[row].tolist()}; our picks are the reference's ranks {ranks}, and "
            f"its {top}/{top + 1} gap is {float(ordered[top - 1] - ordered[top]):.4f}"
        )
    return f"{len(differing)} of {len(sets_agree)} rows select a different set: " + "; ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--dump", required=True, type=pathlib.Path)
    parser.add_argument(
        "--delta",
        action="store_true",
        help="decompose the first divergent stage's deficit into residual and update, per row",
    )
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
    uncorrelated = []
    measured_order = []
    print(
        f"{'file':<16} {'shape':<12} {'rel':>10} {'rms':>10} {'max|diff|':>10} {'increment':>10}"
        "  worst row"
    )
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
        # A file that *says* it holds ids is compared exactly whatever its dtype: a writer that
        # stored them as floats would otherwise be handed back to the relative metric, which is the
        # defect this rule exists to prevent.
        named_exact = path.stem.endswith("_ids")
        if expected.dtype.kind in "iu" or named_exact:
            stored = expected.dtype if expected.dtype.kind not in "iu" else actual.dtype
            if named_exact and stored.kind not in "iu":
                print(
                    f"note: {path.name} is written as {stored}, not an integer; ids are compared "
                    "exactly anyway"
                )
            # An index is not a quantity: nothing about `17` is "3% away" from `18`. It is equal or
            # it is not, and the classes below say what a difference costs.
            # The scores that decide a tie live beside the ids under a name one step away from it:
            # `hidden_02_router_ids` -> `hidden_02_router_logits`, and for a writer that names the
            # file only for its ids, `_ids` -> `_logits`.
            candidates = [
                path.name.replace("_router_ids", "_router_logits"),
                path.name.replace("_ids", "_logits"),
            ]
            logits_path = next(
                (beside for beside in (path.with_name(name) for name in candidates) if beside.exists()),
                None,
            )
            scores = np.load(logits_path) if logits_path is not None else None
            reading = compare_ids(actual, expected, scores) or "every index agrees"
            differ = np.argwhere(actual != expected)
            lines.append(
                f"{path.name:<16} {str(expected.shape):<12} {'exact':>10} {'-':>10} {'-':>10}"
                f" {'-':>10}  {reading}"
            )
            if differ.size and first is None:
                first = path.name
            continue
        rel, rms = measure(actual, expected)
        rows = per_row(actual, expected)
        worst = int(np.argmax(rows))
        # The prefix is the enclosing boundary's error, printed beside a cut so an error a cut
        # inherited is not read as an error the cut made.
        prefix = cut_prefix(path, measured)
        # The increment is measured on the rms ratio and not on the peak: a single element's worst
        # case jumps by a factor of two between stages on rounding alone, while the rms ratio moves
        # continuously and is the quantity a multiplicative fault scales. The peak stays in the table
        # as the diagnostic that finds one bad channel, and it gates as well.
        increment = float("nan") if previous is None else rms - previous
        growth = 0.0 if math.isnan(increment) else increment
        measured[path.stem] = rms
        measured_order.append(path.stem)
        previous = rms
        flag = ""
        if first is None and (rel > arguments.floor or rms > arguments.floor):
            first = path.name
            which = (
                "both"
                if rel > arguments.floor and rms > arguments.floor
                else ("peak" if rel > arguments.floor else "rms")
            )
            flag = f"  <- first outside the floor ({which})"
        # The embedding is the one stage whose answer the harness can prove without the model: the
        # gather of the prompt's ids. Anything above the floor there is the comparison's own
        # misalignment - two sides that are not the same tensor, or not the same rows - and every
        # row after it would be believed. So it is refused before any other row is read.
        if path.stem == "hidden_00" and rel > arguments.floor:
            print(
                f"{path.name:<16} {str(expected.shape):<12} {rel:10.3e} {rms:10.3e}"
                f" {'-':>10}  {int(np.argmax(rows))} ({rows[int(np.argmax(rows))]:.3e})"
                "  <- above the floor at an identity stage"
            )
            print(
                f"the comparison is misaligned: hidden_00 is the embedding's own gather and has no "
                f"model in it, so {rel:.3e} there says the two sides are not the same tensor (or not "
                "the same rows), not that the model is wrong at its first stage"
            )
            return 2
        uncorrelated.append(rms)
        if growth > largest[1]:
            largest = (path.name, growth)
        prefix_text = "" if prefix is None else f"  (prefix {prefix:.3e})"
        # The absolute peak is the number a small operand needs: the router's weights sit near 1/8,
        # so a relative figure says nothing about whether a difference is a rounding or a defect.
        absolute = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        sums = ""
        if path.stem.endswith("_router_weights") and expected.ndim == 2:
            sums = renormalization_note(actual, expected)
        lines.append(
            f"{path.name:<16} {str(expected.shape):<12} {rel:10.3e} {rms:10.3e} {absolute:10.3e}"
            f" {increment:10.3e}  {worst} ({rows[worst]:.3e}){flag}{prefix_text}{sums}"
        )

    # The marker goes on the one row whose increment is the largest: every later row of a persistent
    # fault also sits above the floor, and marking them all would name every stage at once.
    if largest[0] is not None and largest[1] > 0.0:
        for index, line in enumerate(lines):
            if line.startswith(largest[0]):
                lines[index] = line + "  <- the increment enters here"
    print("\n".join(lines))

    # Two unrelated vectors of one magnitude differ by about sqrt(2) times that magnitude, so a
    # median near or above 1 is not "the model is uniformly bad" - it is two sides that are not the
    # same quantity. Saying so in words is what stops the table being read as thirty defects.
    median = float(np.median(uncorrelated)) if uncorrelated else 0.0
    if 1.2 <= median <= 1.7:
        print(
            f"every stage is uncorrelated (median rms(diff)/rms(reference) {median:.2f}, and two "
            "unrelated vectors of one magnitude sit at sqrt(2) = 1.41): check that the two sides "
            "hold the same quantity before reading any row as a model defect"
        )
    elif median > 1.7:
        print(
            f"every stage differs by more than decorrelation can explain (median "
            f"rms(diff)/rms(reference) {median:.2f}): the two sides carry different magnitudes - a "
            "scale, or a tensor read from the wrong place - not a defect repeated thirty times"
        )

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

    if arguments.delta:
        compare_delta(arguments, measured_order, first)

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
