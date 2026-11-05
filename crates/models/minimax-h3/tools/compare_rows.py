#!/usr/bin/env python3
"""Compare the rows a recorder observed against the rows a file declares.

Two instruments in this lane compare a declaration file to a *walk*, and both
ask one question: does the walk state what the file states. That question cannot
find a row the file has *wrong*, because the walk shares the file's assumptions.
This comparison takes the file's other side - the model's own forward, which a
recorder observes - and looks for the shape a disagreement takes when the field
that differs is neither the operator nor a width but one axis:

    observed  act_and_mul  bf16 ... value_stride_28672  wide_false
    declared  act_and_mul  bf16 ... value_stride_28672  wide_true

Counted as rows, those are two differences of zero. Read one field at a time
they are one field apart, which is also the shape of a `dtype` defect (`silu` at
BF16 where the forward launches FP32). Both were real stops for a served run.

The comparison is one-sidedly informative and is deliberately not a gate on
coverage: a recorder observes what one component launched at one set of axes and
a declaration file states what a deployment can launch, so each side holds rows
the other does not, and that is expected. What is not expected is a near miss -
two rows with the same operator and the same value for every field but one,
whose remaining field has no value in common. A near miss means one of the two
is wrong about a key, and it is what this command fails on.

Exit status: 0 when the observed rows raise no near miss, 1 when they do.
"""

import argparse
import pathlib
import sys
from collections import defaultdict


def rows(path):
    """The `name<TAB>axes` rows of one file, with `#` headers dropped."""
    try:
        text = pathlib.Path(path).read_text()
    except OSError as error:
        raise SystemExit(f"cannot read {path}: {error}")

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise SystemExit(
                f"{path}: a row is `name<TAB>axes`, and this one is {line!r}"
            )
        out.append((parts[0], parts[1]))
    return out


def near_misses(observed, declared):
    """Rows agreeing on the operator and every field but one, disjoint in it.

    A field is one `_`-separated token of the axis string. Two rows are compared
    only when their axis strings have the same number of tokens, so a value that
    itself holds an underscore (`activation_gelu_tanh`) shifts every position
    after it out of the other row's reach instead of inventing a match. That
    direction is safe: this can miss a disagreement, never report one that is
    not there.

    Args:
        observed: the `(name, axes)` rows a recorder produced.
        declared: the `(name, axes)` rows a declaration file states.

    Returns:
        One tuple per near miss: operator, field position, the two value sets,
        and the two rows that disagree.
    """
    by_shape = defaultdict(lambda: (defaultdict(set), defaultdict(set)))
    for side, source in ((0, observed), (1, declared)):
        for name, axes in source:
            tokens = axes.split("_")
            for position in range(len(tokens)):
                rest = tokens[:position] + tokens[position + 1 :]
                shape = (name, len(tokens), position, tuple(rest))
                by_shape[shape][side][tokens[position]].add(axes)

    found = []
    for shape, (observed_values, declared_values) in by_shape.items():
        name, _, position, _ = shape
        if not observed_values or not declared_values:
            continue
        if set(observed_values) & set(declared_values):
            continue
        found.append(
            (
                name,
                position,
                sorted(observed_values),
                sorted(declared_values),
                sorted(next(iter(observed_values.values())))[0],
                sorted(next(iter(declared_values.values())))[0],
            )
        )
    return sorted(found)


def self_test():
    """Prove the comparison on a pair whose answer is known.

    Three cases, each one this lane met: a swapped `dtype` (the `silu`
    defect), a swapped flag inside an otherwise identical row (the `wide`
    defect), and a pair that differs in a value both sides hold, which is not
    a near miss.

    Returns:
        0, or raises on a case that disagrees with the written answer.
    """
    observed = {
        ("silu", "dtype_f32_columns_5376_block_1024_pdl_false"),
        ("act_and_mul", "dtype_bf16_width_14336_wide_false_pdl_false"),
        ("scaled_residual", "dtype_bf16_columns_5120_block_1024_bias_false"),
    }
    declared = {
        ("silu", "dtype_bf16_columns_5376_block_1024_pdl_false"),
        ("act_and_mul", "dtype_bf16_width_14336_wide_true_pdl_false"),
        ("scaled_residual", "dtype_bf16_columns_1152_block_1024_bias_false"),
        ("scaled_residual", "dtype_bf16_columns_5120_block_1024_bias_false"),
    }
    found = near_misses(observed, declared)
    by_name = {
        name: (taken[0], stated[0]) for name, _, taken, stated, _, _ in found
    }

    assert set(by_name) == {"silu", "act_and_mul"}, sorted(by_name)
    assert by_name["silu"] == ("f32", "bf16"), by_name["silu"]
    assert by_name["act_and_mul"] == ("false", "true"), by_name["act_and_mul"]
    print("self-test: 3 cases pass")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--observed", help="the recorder's rows")
    parser.add_argument("--declared", help="a deployment's declared rows")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the comparison itself on a written pair and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.observed or not args.declared:
        parser.error("--observed and --declared are needed to compare anything")

    observed, declared = rows(args.observed), rows(args.declared)
    observed_set, declared_set = set(observed), set(declared)

    print(f"observed {len(observed_set)} rows, declared {len(declared_set)}")
    print(f"observed only: {len(observed_set - declared_set)}")
    print(f"declared only: {len(declared_set - observed_set)}")

    found = near_misses(observed_set, declared_set)
    print(f"near misses (one field apart, nothing in common): {len(found)}")
    for name, position, taken, stated, observed_row, declared_row in found:
        print(f"  {name} position {position}:")
        print(f"    observed {'/'.join(taken)}  declared {'/'.join(stated)}")
        print(f"    observed {observed_row}")
        print(f"    declared {declared_row}")

    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
